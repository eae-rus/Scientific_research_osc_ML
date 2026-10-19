"""Проверки подготовки на месте, отметок готовности и ограниченного входа агента."""
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from osc_tools.corpus.preparation import CorpusPreparer, PreparationConfig, prepare_agents
from osc_tools.corpus.agent_runtime import CaseRuntime
from osc_tools.corpus.report_export import export_reports
from osc_tools.corpus.record_identity import register_derivative, match_flat


class InPlaceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)/'working_copy'
        self.case = self.root/'2025'/'true folder'
        self.case.mkdir(parents=True)
        (self.case/'note.txt').write_text('первичный текст',encoding='utf-8')
        (self.case/'empty.doc').write_bytes(b'')
        with zipfile.ZipFile(self.case/'data.zip','w') as archive:
            archive.writestr('included.txt',b'primary in archive')

    def tearDown(self):
        self.temporary.cleanup()

    def run_prepare(self, **kwargs):
        return CorpusPreparer(PreparationConfig(str(self.root),**kwargs)).run()

    def test_repeat_skips_and_does_not_copy_originals(self):
        first = self.run_prepare()
        second = self.run_prepare()
        self.assertEqual(first['cases_processed'],1)
        self.assertEqual(second['cases_skipped'],1)
        self.assertEqual(second['extracted_bytes'],0)
        self.assertFalse((self.root/'_phase6/blobs').exists())
        self.assertEqual(len(list(self.root.rglob('note.txt'))),1)
        self.assertTrue((self.case/'empty.doc').exists())

    def test_changed_or_missing_extraction_is_reprocessed(self):
        self.run_prepare()
        next((self.case/'_phase6_extracted').rglob('included.txt')).unlink()
        self.assertEqual(self.run_prepare()['cases_processed'],1)
        (self.case/'note.txt').write_text('новый первичный текст',encoding='utf-8')
        self.assertEqual(self.run_prepare()['cases_processed'],1)

    def test_agent_sees_only_selected_sources_and_results_are_outside_case(self):
        self.run_prepare()
        instruction, vocabulary = self.root/'instruction.md', self.root/'vocabulary.md'
        instruction.write_text('Правила',encoding='utf-8')
        vocabulary.write_text('Словарь',encoding='utf-8')
        contexts = prepare_agents(self.root/'_phase6',instruction,vocabulary,run_name='test')
        runtime = CaseRuntime(contexts[0]['workspace'])
        paths = [f['path'] for f in runtime.files.values()]
        self.assertIn('note.txt',paths)
        self.assertTrue(any(p.endswith('included.txt') for p in paths))
        self.assertNotIn('empty.doc',paths)
        self.assertFalse(any(p.startswith('_phase6/') for p in paths))
        self.assertFalse(Path(contexts[0]['workspace']).is_relative_to(self.case))
        repeated = prepare_agents(self.root/'_phase6',instruction,vocabulary,run_name='test',reuse_existing=True)
        self.assertEqual(repeated[0]['status'],'existing_context')

    def test_new_year_added_without_reprocessing_previous(self):
        self.run_prepare()
        folder = self.root/'2024'/'new'
        folder.mkdir(parents=True)
        (folder/'a.txt').write_text('другой комплект',encoding='utf-8')
        summary = self.run_prepare()
        self.assertEqual(summary['cases_processed'],1)
        self.assertEqual(summary['cases_skipped'],1)
        self.assertEqual(summary['cases'],2)

    def test_export_keeps_user_edits_outside_work_context(self):
        self.run_prepare()
        instruction, vocabulary = self.root/'instruction.md', self.root/'vocabulary.md'
        instruction.write_text('Правила',encoding='utf-8')
        vocabulary.write_text('Словарь',encoding='utf-8')
        prepared = self.root/'_phase6'
        contexts = prepare_agents(prepared,instruction,vocabulary,run_name='test')
        runtime = CaseRuntime(contexts[0]['workspace'])
        runtime.call('submit_report',{'sections':['—']*12,'claim_ids':[],
            'limitations':[{'file_id':fid,'reason':'Не прочитан в тесте'} for fid in runtime.files]})
        first = export_reports(prepared,contexts,'test')
        target = Path(first[0]['report'])
        self.assertIn('true folder',str(target))
        target.write_text(target.read_text(encoding='utf-8')+'\nПримечание автора: проверить.\n',encoding='utf-8')
        second = export_reports(prepared,contexts,'test')
        self.assertNotEqual(first[0]['report'],second[0]['report'])
        self.assertIn('Примечание автора',target.read_text(encoding='utf-8'))

    def test_repeat_preserves_derivative_lineage_and_old_record_snapshot(self):
        (self.case/'rec.cfg').write_bytes(b'config')
        (self.case/'rec.dat').write_bytes(b'original data')
        self.run_prepare()
        index = self.root/'_phase6/record_index.json'
        record = json.loads(index.read_text(encoding='utf-8'))['records'][0]
        flat = self.root.parent/'flat'
        flat.mkdir()
        converted = flat/'anonymous.cfg'
        converted.write_bytes(b'derived configuration')
        register_derivative(index,record['record_id'],[converted],'test_conversion')
        self.run_prepare()
        self.assertEqual(match_flat(index,flat)['files'][0]['status'],'matched')
        (self.case/'rec.dat').write_bytes(b'changed primary data')
        self.run_prepare()
        records = json.loads(index.read_text(encoding='utf-8'))['records']
        self.assertEqual(len(records),2)
        old = next(r for r in records if r['record_id'] == record['record_id'])
        self.assertFalse(old['current_input'])
        self.assertEqual(len(old['derivatives']),1)


if __name__ == '__main__':
    unittest.main()
