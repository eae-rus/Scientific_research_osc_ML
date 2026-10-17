"""Проверки происхождения цитат, охвата чтения и сохранности черновиков."""
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from osc_tools.corpus.agent_runtime import CaseRuntime, prepare, tool_definitions


class AgentRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.corpus = self.root / 'corpus'
        self.source = self.corpus / '2024' / 'case'
        self.source.mkdir(parents=True)
        (self.source / 'note.txt').write_text('Причина предположительно дефект кабеля.\nРемонт только предложен.', encoding='utf-8')
        (self.source / 'record.bb').write_bytes(b'not-a-real-record')
        self.inventory = self.root / 'inventory.json'
        self.inventory.write_text(json.dumps({'source_root': str(self.corpus), 'cases': [
            {'case_id': 'case_000099', 'source_folder': '2024/case'}]}), encoding='utf-8')
        self.instruction = self.root / 'instruction.md'
        self.instruction.write_text('## 8. Контроль\nПравила\n## 9. Случаи\nСТАРЫЙ ОТЧЕТ\n## 10. Словарь\nПолный\n', encoding='utf-8')
        self.vocabulary = self.root / 'vocabulary.md'
        self.vocabulary.write_text('Предположительно — сохранять модальность.', encoding='utf-8')
        self.workspace = self.root / 'work'
        prepare(self.inventory, 'case_000099', self.workspace, self.instruction, self.vocabulary)
        self.runtime = CaseRuntime(self.workspace)
        self.file_id = next(k for k, v in self.runtime.files.items() if v['path'] == 'note.txt')

    def tearDown(self):
        self.temporary.cleanup()

    def read_all(self):
        return self.runtime.call('read_document', {'file_id': self.file_id})

    def claim(self, quote='Ремонт только предложен.'):
        return self.runtime.call('add_claim', {'text': 'Замена предложена', 'category': 'мероприятие',
            'modality': 'предложение', 'episode': 'E1', 'file_id': self.file_id,
            'locator': 'line/2', 'quote': quote})

    def test_preparation_does_not_send_prior_examples_or_claim_reading(self):
        bootstrap = (self.workspace / 'bootstrap.md').read_text(encoding='utf-8')
        self.assertNotIn('СТАРЫЙ ОТЧЕТ', bootstrap)
        self.assertIn('Предположительно — сохранять модальность.', bootstrap)
        self.assertEqual(self.runtime.call('audit', {})['text_coverage'][0]['status'], 'partial_or_unread')

    def test_rejects_quote_from_extracted_but_not_delivered_block(self):
        self.runtime.call('read_document', {'file_id': self.file_id, 'max_chars': 8})
        with self.assertRaisesRegex(ValueError, 'ранее выданном'):
            self.claim()
        audit = self.runtime.call('audit', {})
        self.assertEqual(audit['text_coverage'][0]['modes'][0]['incomplete_blocks'], 2)

    def test_cursor_resumes_without_losing_characters_and_claim_is_checked(self):
        result = self.runtime.call('read_document', {'file_id': self.file_id, 'max_chars': 7})
        parts = [b['text'] for b in result['blocks']]
        while result['next_cursor']:
            result = self.runtime.call('read_document', {'file_id': self.file_id, **result['next_cursor'], 'max_chars': 7})
            parts.extend(b['text'] for b in result['blocks'])
        self.assertEqual(''.join(parts), 'Причина предположительно дефект кабеля.Ремонт только предложен.')
        self.assertEqual(self.runtime.call('audit', {})['text_coverage'][0]['status'], 'all_text_delivered')
        self.assertEqual(self.claim()['status'], 'quote_verified_semantics_pending')
        with self.assertRaises(ValueError):
            self.claim('Ремонт выполнен.')

    def test_source_change_prevents_cached_reading(self):
        self.read_all()
        (self.source / 'note.txt').write_text('Новая редакция', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'изменён'):
            self.read_all()

    def test_search_reports_pagination_and_does_not_mark_entire_source_read(self):
        (self.source / 'note.txt').write_text(''.join(f'Тест {i} причина\n' for i in range(30)), encoding='utf-8')
        workspace = self.root / 'new_search'
        prepare(self.inventory, 'case_000099', workspace, self.instruction, self.vocabulary)
        runtime = CaseRuntime(workspace)
        found = runtime.call('search_document', {'file_id': self.file_id, 'query': 'причина', 'max_hits': 2})
        self.assertEqual(found['matches_total_blocks'], 30)
        self.assertEqual(found['next_start'], 2)
        self.assertEqual(runtime.call('audit', {})['text_coverage'][0]['status'], 'partial_or_unread')

    def test_image_path_is_not_a_view_and_observation_requires_delivery(self):
        import base64
        png = base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+a7x8AAAAASUVORK5CYII=')
        (self.source / 'photo.png').write_bytes(png)
        workspace = self.root / 'images'
        prepare(self.inventory, 'case_000099', workspace, self.instruction, self.vocabulary)
        runtime = CaseRuntime(workspace)
        image_id = next(k for k, f in runtime.files.items() if f['role'] == 'image')
        with self.assertRaises(KeyError):
            runtime.call('record_image_observation', {'image_id': image_id, 'area': 'вся', 'observation': 'пиксель', 'limitation': 'тест'})
        delivered = runtime.call('get_image', {'image_id': image_id})
        self.assertEqual(base64.b64decode(delivered['image_base64']), png)
        self.assertEqual(len(runtime.state['observations']), 0)
        observation = runtime.call('record_image_observation', {'image_id': image_id, 'area': 'вся', 'observation': 'пиксель', 'limitation': 'смысл не проверен'})
        self.assertEqual(observation['status'], 'viewer_reported_not_independently_verified')

    def test_schema_covers_only_bound_tools_and_has_closed_arguments(self):
        definitions = tool_definitions()
        for tool in definitions:
            self.assertFalse(tool['parameters']['additionalProperties'])
            self.assertEqual(set(tool['parameters']['required']), set(tool['parameters']['properties']))
        self.assertNotIn('exec_command', {t['name'] for t in definitions})

    def test_repeated_docx_image_is_not_overwritten_or_counted_as_viewed(self):
        document = '''<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><w:body><w:p><w:r><w:t>Начало</w:t><a:blip r:embed="r1"/></w:r></w:p><w:p><w:r><w:t>Повтор</w:t><a:blip r:embed="r1"/></w:r></w:p></w:body></w:document>'''
        with zipfile.ZipFile(self.source / 'illustrated.docx', 'w') as archive:
            archive.writestr('word/document.xml', document)
            archive.writestr('word/_rels/document.xml.rels', '<Relationships><Relationship Id="r1" Target="media/image1.png"/></Relationships>')
            archive.writestr('word/media/image1.png', b'image-data')
        workspace = self.root / 'repeated_image'
        prepare(self.inventory, 'case_000099', workspace, self.instruction, self.vocabulary)
        runtime = CaseRuntime(workspace)
        file_id = next(k for k, f in runtime.files.items() if f['path'] == 'illustrated.docx')
        result = runtime.call('read_document', {'file_id': file_id})
        self.assertEqual(len(result['images']), 2)
        self.assertTrue(all(i['status'] == 'not_delivered' for i in runtime.state['images'].values()))
        self.assertEqual(len(list((workspace / 'extractions' / (file_id + '__media')).glob('*'))), 1)
        runtime.call('read_document', {'file_id': file_id})

        # Кэш мог быть создан перед сохранением состояния; связи должны вернуться.
        runtime.state['images'] = {}
        result = runtime.call('read_document', {'file_id': file_id})
        self.assertEqual(len(result['images']), 2)


    def test_unknown_id_and_shell_tool_are_rejected(self):
        with self.assertRaises(KeyError):
            self.runtime.call('read_document', {'file_id': '../note.txt'})
        with self.assertRaises(ValueError):
            self.runtime.call('exec_command', {'cmd': 'anything'})

    def test_draft_revisions_do_not_replace_previous_or_claim_human_review(self):
        self.read_all()
        claim = self.claim()
        sections = ['Мероприятие лишь предложено (C1).'] + ['—'] * 11
        first = self.runtime.call('submit_report', {'sections': sections, 'claim_ids': [claim['claim_id']], 'limitations': []})
        previous = Path(first['report']).read_bytes()
        second = self.runtime.call('submit_report', {'sections': sections, 'claim_ids': ['C1'], 'limitations': []})
        self.assertNotEqual(first['report'], second['report'])
        self.assertEqual(Path(first['report']).read_bytes(), previous)
        self.assertEqual(second['status'], 'draft')
        self.assertEqual(second['semantic_review'], 'not_performed')

    def test_unread_source_is_partial_until_limitation_is_explicit(self):
        sections = ['Ограничение чтения.'] + ['—'] * 11
        result = self.runtime.call('submit_report', {'sections': sections, 'claim_ids': [], 'limitations': []})
        self.assertEqual(result['status'], 'partial')
        declared = self.runtime.call('submit_report', {'sections': sections, 'claim_ids': [],
            'limitations': [{'file_id': self.file_id, 'reason': 'Чтение не выполнено'}]})
        self.assertEqual(declared['status'], 'draft')
        quality = json.loads(Path(declared['quality']).read_text(encoding='utf-8'))
        self.assertEqual(quality['audit']['text_coverage'][0]['status'], 'partial_or_unread')

    def test_resume_preserves_evidence_and_enforces_call_limit(self):
        self.read_all()
        reloaded = CaseRuntime(self.workspace)
        self.assertEqual(reloaded.call('audit', {})['text_coverage'][0]['status'], 'all_text_delivered')
        reloaded.state['max_calls'] = reloaded.state['calls']
        with self.assertRaisesRegex(RuntimeError, 'лимит'):
            reloaded.call('audit', {})

    def test_prepare_refuses_source_output_and_existing_workspace(self):
        with self.assertRaises(ValueError):
            prepare(self.inventory, 'case_000099', self.source / 'work', self.instruction, self.vocabulary)
        with self.assertRaises(FileExistsError):
            prepare(self.inventory, 'case_000099', self.workspace, self.instruction, self.vocabulary)


if __name__ == '__main__':
    unittest.main()
