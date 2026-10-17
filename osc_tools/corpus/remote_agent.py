"""Удалённая модель Responses API поверх инструментов одной папки.

Модель получает словарь, инструкцию, выбранный текст и изображения. Архивы и
осциллограммы не загружаются. Ключ берётся из окружения и не записывается в журнал.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import urllib.error
import urllib.request

from .agent_runtime import CaseRuntime, now, tool_definitions, write_json


class ResponsesTransport:
    """Выполнить один HTTPS-запрос без установки SDK и без автоматических повторов."""
    def __init__(self, key_env='OPENAI_API_KEY', timeout=180):
        self.key = os.environ.get(key_env)
        if not self.key:
            raise ValueError(f'Задайте ключ в переменной окружения {key_env}; не помещайте его в ноутбук')
        self.timeout = timeout

    def __call__(self, payload):
        request = urllib.request.Request('https://api.openai.com/v1/responses',
            data=json.dumps(payload, ensure_ascii=False).encode('utf-8'),
            headers={'Authorization': 'Bearer ' + self.key, 'Content-Type': 'application/json'}, method='POST')
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            # Не возвращаем заголовки запроса или исходный ключ в диагностике.
            try:
                error = json.loads(exc.read()).get('error', {})
                code = error.get('code') or error.get('type') or 'unknown'
            except Exception:
                code = 'unknown'
            raise RuntimeError(f'Responses API HTTP {exc.code}, code={code}') from None
        except (urllib.error.URLError, TimeoutError):
            raise RuntimeError('Нет ответа API; запрос не повторяется автоматически, возможен расход на предыдущий запрос') from None


def _run_remote(workspace, model, transport=None, max_requests=80, max_request_bytes=16 * 1024 * 1024,
               max_output_tokens=12000, resume=False):
    """Выполнить ограниченный цикл; остановка сохраняет причину и историю обмена."""
    if not model.strip() or max_requests < 1 or max_request_bytes < 1024:
        raise ValueError('Укажите модель и положительные ограничения')
    runtime = CaseRuntime(workspace)
    checkpoint_path = runtime.root / 'remote_checkpoint.json'
    if checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text(encoding='utf-8'))
        if checkpoint['status'] == 'report_submitted':
            return {'status': 'already_completed', 'case_id': runtime.manifest['case_id'], 'model': checkpoint['model']}
        if not resume:
            raise ValueError('Уже есть незавершённый прогон; укажите resume=True или создайте новый запуск')
        if checkpoint['model'] != model:
            raise ValueError('При продолжении модель менять нельзя')
        if checkpoint.get('executing_tool'):
            raise ValueError('Остановка во время исполнения инструмента: проверьте state.json; нужен новый контекст для безопасного повтора')
    else:
        checkpoint = {'model': model, 'case_id': runtime.manifest['case_id'], 'status': 'prepared', 'requests': 0,
            'history': [{'role': 'developer', 'content': (runtime.root / 'bootstrap.md').read_text(encoding='utf-8')},
                        {'role': 'user', 'content': 'Составь отчёт по этому комплекту, используя доступные инструменты.'}],
            'usage': [], 'pending_calls': [], 'transmissions': [], 'created_at': now()}
    transport = transport or ResponsesTransport()
    write_json(checkpoint_path, checkpoint)
    try:
        while True:
            while checkpoint['pending_calls']:
                call = checkpoint['pending_calls'][0]
                checkpoint['executing_tool'] = call['call_id']
                write_json(checkpoint_path, checkpoint)
                try:
                    result = runtime.call(call['name'], json.loads(call['arguments']))
                except (ValueError, KeyError, TypeError) as exc:
                    result = {'error': type(exc).__name__, 'message': str(exc)}
                image_data = result.pop('image_base64', None) if isinstance(result, dict) else None
                checkpoint['history'].append({'type': 'function_call_output', 'call_id': call['call_id'],
                                               'output': json.dumps(result, ensure_ascii=False)})
                if image_data:
                    mime = result.get('mime_type', result.get('mime', 'image/png'))
                    checkpoint['history'].append({'role': 'user', 'content': [
                        {'type': 'input_text', 'text': 'Изображение из последнего get_image; источник: ' + str(result.get('image_id', result.get('file_id', '')))},
                        {'type': 'input_image', 'image_url': f'data:{mime};base64,{image_data}', 'detail': 'high'}]})
                checkpoint['pending_calls'].pop(0)
                checkpoint.pop('executing_tool', None)
                if call['name'] == 'submit_report' and not (isinstance(result, dict) and 'error' in result):
                    checkpoint['status'] = 'report_submitted'
                    write_json(checkpoint_path, checkpoint)
                    return {'status': 'report_submitted', 'case_id': checkpoint['case_id'], 'model': model,
                            'requests': checkpoint['requests'], 'result': result}
                write_json(checkpoint_path, checkpoint)
            if checkpoint['requests'] >= max_requests:
                checkpoint['status'] = 'request_limit'
                break
            payload = {'model': model, 'store': False, 'input': checkpoint['history'], 'tools': tool_definitions(),
                       'parallel_tool_calls': False, 'max_output_tokens': max_output_tokens,
                       'include': ['reasoning.encrypted_content']}
            encoded = json.dumps(payload, ensure_ascii=False).encode('utf-8')
            if len(encoded) > max_request_bytes:
                checkpoint['status'] = 'request_size_limit'
                break
            checkpoint['transmissions'].append({'request_number': checkpoint['requests'] + 1,
                'bytes': len(encoded), 'payload_sha256': hashlib.sha256(encoded).hexdigest(), 'at': now()})
            checkpoint['status'] = 'calling_api'
            checkpoint['requests'] += 1
            write_json(checkpoint_path, checkpoint)
            response = transport(payload)
            checkpoint['usage'].append(response.get('usage', {}))
            checkpoint['history'].extend(response.get('output', []))
            checkpoint['pending_calls'] = [i for i in response.get('output', []) if i.get('type') == 'function_call']
            checkpoint['status'] = 'tools_pending'
            if response.get('status', 'completed') != 'completed':
                checkpoint.update(status='api_incomplete', incomplete_details=response.get('incomplete_details'))
                break
            if not checkpoint['pending_calls']:
                checkpoint['status'] = 'ended_without_submit'
                break
            write_json(checkpoint_path, checkpoint)
    except Exception as exc:
        checkpoint.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        write_json(checkpoint_path, checkpoint)
        raise
    write_json(checkpoint_path, checkpoint)
    return {'status': checkpoint['status'], 'requests': checkpoint['requests'], 'case_id': checkpoint['case_id']}


def run_remote(workspace, model, transport=None, **kwargs):
    """Дать одному процессу владеть контекстом; при аварии оставить явный lock."""
    lock = Path(workspace).resolve(strict=True) / 'remote.lock'
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise RuntimeError('Контекст занят либо остался lock после аварии; проверьте процесс перед удалением remote.lock') from None
    os.close(descriptor)
    try:
        return _run_remote(workspace, model, transport, **kwargs)
    finally:
        lock.unlink()


def run_batch(workspaces, model, transport=None, resume=False, **kwargs):
    """Последовательно пройти контексты от новых годов к старым, пропуская готовые."""
    def key(path):
        manifest = json.loads((Path(path) / 'manifest.json').read_text(encoding='utf-8'))
        # Подготовленный source_root содержит case-ID; год берётся из внешнего provenance.
        provenance = Path(manifest['source_root']).parent / 'provenance.json'
        folder = json.loads(provenance.read_text(encoding='utf-8'))['source_folder'] if provenance.exists() else manifest['source_root']
        import re
        years = re.findall(r'(?<!\d)(20\d{2})(?!\d)', folder)
        return (years[0] if years else '', manifest['case_id'])
    results = []
    for workspace in sorted(workspaces, key=key, reverse=True):
        try:
            results.append(run_remote(workspace, model, transport, resume=resume, **kwargs))
        except Exception as exc:
            results.append({'workspace': str(workspace), 'status': 'failed', 'error': f'{type(exc).__name__}: {exc}'})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--max-requests', type=int, default=80)
    args = parser.parse_args()
    print(json.dumps(run_remote(args.workspace, args.model, max_requests=args.max_requests, resume=args.resume), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
