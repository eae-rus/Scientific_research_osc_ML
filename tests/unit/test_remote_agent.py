"""Проверки протокола удалённого цикла без обращений к платному API."""
import json
import base64
from pathlib import Path
import tempfile
import unittest

from osc_tools.corpus.agent_runtime import prepare
from osc_tools.corpus.remote_agent import run_remote


class RemoteAgentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        source = self.root / 'corpus' / '2025' / 'event'
        source.mkdir(parents=True)
        (source / 'a.txt').write_text('Первичный факт', encoding='utf-8')
        (source / 'image.png').write_bytes(base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aL1cAAAAASUVORK5CYII='))
        inventory = self.root / 'inventory.json'
        inventory.write_text(json.dumps({'source_root': str(source.parent.parent), 'cases': [
            {'case_id': 'case_000001', 'source_folder': '2025/event'}]}), encoding='utf-8')
        instruction = self.root / 'instruction.md'
        vocabulary = self.root / 'vocabulary.md'
        instruction.write_text('Инструкция', encoding='utf-8')
        vocabulary.write_text('СЛОВАРЬ', encoding='utf-8')
        self.workspace = self.root / 'agent'
        prepare(inventory, 'case_000001', self.workspace, instruction, vocabulary)

    def tearDown(self):
        self.temp.cleanup()

    def test_tool_roundtrip_preserves_reasoning_and_dictionary(self):
        requests = []
        def fake(payload):
            requests.append(json.loads(json.dumps(payload)))
            if len(requests) == 1:
                return {'status': 'completed', 'output': [
                    {'type': 'reasoning', 'encrypted_content': 'opaque'},
                    {'type': 'function_call', 'name': 'get_manifest', 'arguments': '{}', 'call_id': 'call1'}]}
            return {'status': 'completed', 'output': [{'type': 'message', 'role': 'assistant', 'content': []}]}
        result = run_remote(self.workspace, 'test-model', transport=fake)
        self.assertEqual(result['status'], 'ended_without_submit')
        self.assertIn('СЛОВАРЬ', requests[0]['input'][0]['content'])
        self.assertFalse(requests[0]['store'])
        self.assertTrue(any(i.get('encrypted_content') == 'opaque' for i in requests[1]['input']))
        self.assertTrue(any(i.get('type') == 'function_call_output' for i in requests[1]['input']))

    def test_failure_saved_without_automatic_retry(self):
        calls = []
        def failed(payload):
            calls.append(payload)
            raise RuntimeError('offline')
        with self.assertRaises(RuntimeError):
            run_remote(self.workspace, 'test-model', transport=failed)
        self.assertEqual(len(calls), 1)
        checkpoint = json.loads((self.workspace / 'remote_checkpoint.json').read_text(encoding='utf-8'))
        self.assertEqual(checkpoint['status'], 'failed')

    def test_size_limit_stops_before_transmission(self):
        calls = []
        result = run_remote(self.workspace, 'test-model', transport=lambda p: calls.append(p), max_request_bytes=1024)
        self.assertEqual(result['status'], 'request_size_limit')
        self.assertEqual(calls, [])

    def test_image_bytes_are_input_image_not_only_path(self):
        requests = []
        manifest = json.loads((self.workspace / 'manifest.json').read_text(encoding='utf-8'))
        fid = next(f['file_id'] for f in manifest['files'] if f['path'] == 'image.png')
        def fake(payload):
            requests.append(json.loads(json.dumps(payload)))
            if len(requests) == 1:
                return {'output': [{'type':'function_call','call_id':'image1','name':'get_image',
                    'arguments':json.dumps({'image_id':fid})}]}
            return {'output': []}
        run_remote(self.workspace, 'test-model', transport=fake)
        image_message = requests[1]['input'][-1]
        self.assertEqual(image_message['content'][1]['type'], 'input_image')
        self.assertTrue(image_message['content'][1]['image_url'].startswith('data:image/png;base64,'))
        self.assertNotIn('image_base64', requests[1]['input'][-2]['output'])

    def test_lock_blocks_overlapping_worker(self):
        (self.workspace / 'remote.lock').write_text('occupied', encoding='utf-8')
        with self.assertRaisesRegex(RuntimeError, 'Контекст занят'):
            run_remote(self.workspace, 'test-model', transport=lambda p: {})

    def test_submit_finishes_and_next_run_skips_api(self):
        manifest = json.loads((self.workspace / 'manifest.json').read_text(encoding='utf-8'))
        text_id = next(f['file_id'] for f in manifest['files'] if f['path'] == 'a.txt')
        image_id = next(f['file_id'] for f in manifest['files'] if f['path'] == 'image.png')
        steps = [('read_document', {'file_id':text_id}),
            ('add_claim', {'text':'Первичный факт','category':'факт','modality':'сообщено','episode':'E1',
                'file_id':text_id,'locator':'line/1','quote':'Первичный факт'}),
            ('audit', {}),
            ('submit_report', {'sections':['Факт (C1)'] + ['—'] * 11,'claim_ids':['C1'],
                'limitations':[{'file_id':image_id,'reason':'Изображение не прочитано'}]})]
        requests = []
        def fake(payload):
            name, args = steps[len(requests)]
            requests.append(payload)
            return {'output':[{'type':'function_call','name':name,'arguments':json.dumps(args),'call_id':str(len(requests))}]}
        result = run_remote(self.workspace, 'test-model', transport=fake)
        self.assertEqual(result['status'], 'report_submitted')
        self.assertEqual(result['result']['status'], 'draft')
        repeated = run_remote(self.workspace, 'test-model', transport=fake)
        self.assertEqual(repeated['status'], 'already_completed')
        self.assertEqual(len(requests), 4)


if __name__ == '__main__':
    unittest.main()
