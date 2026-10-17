"""Проверки архивных границ, постоянных ID и происхождения преобразованных записей."""
import json
from pathlib import Path
import tempfile
import unittest
import zipfile
import io

from osc_tools.corpus.preparation import CorpusPreparer, PreparationConfig
from osc_tools.corpus.record_identity import match_flat, register_derivative, restore_flat


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / 'source'
        self.case = self.source / '2024' / 'case'
        self.case.mkdir(parents=True)

    def tearDown(self):
        self.temp.cleanup()

    def run_prepare(self, name='prepared', **kwargs):
        output = self.root / name
        summary = CorpusPreparer(PreparationConfig(str(self.source), str(output), **kwargs)).run()
        return output, summary

    def test_archive_unsafe_empty_and_dedup(self):
        (self.case / 'empty.hdr').write_bytes(b'')
        (self.case / 'note.txt').write_bytes(b'good')
        with zipfile.ZipFile(self.case / 'archive.zip', 'w') as archive:
            archive.writestr('note.txt', b'good')
            archive.writestr('../outside.txt', b'bad')
            archive.writestr('big.txt', b'a' * 20)
        output, summary = self.run_prepare(max_member_bytes=10)
        self.assertEqual(summary['unique_blobs'], 1)
        self.assertEqual(summary['archive_member_issues'], 2)
        self.assertFalse((self.root / 'outside.txt').exists())
        self.assertTrue((self.case / 'empty.hdr').exists())
        files = json.loads((output / 'provenance_index.json').read_text(encoding='utf-8'))['cases'][0]['files']
        self.assertEqual(next(f for f in files if f['logical_path'] == 'empty.hdr')['preparation_status'], 'zero_byte_not_copied')

    def test_loose_reserved_id_survives_next_run(self):
        doc = self.source / '2024' / 'loose .doc'
        doc.write_bytes(b'original')
        seed = self.root / 'seed.json'
        seed.write_text(json.dumps({'case_registry': {'__prepared__/reserved': 'case_000012'},
            'prepared_cases': [{'case_id': 'case_000012', 'primary_document': str(doc)}]}), encoding='utf-8')
        output, _ = self.run_prepare(seed_inventory=str(seed))
        repeated, _ = self.run_prepare('repeated', seed_inventory=str(output / 'original_inventory.json'))
        for location in (output, repeated):
            cases = json.loads((location / 'provenance_index.json').read_text(encoding='utf-8'))['cases']
            self.assertIn('case_000012', [c['case_id'] for c in cases])

    def test_comtrade_legacy_dat_and_derivative_return(self):
        (self.case / 'rec.cfg').write_bytes(b'configuration')
        (self.case / 'rec.dat').write_bytes(b'samples')
        output, _ = self.run_prepare()
        index = output / 'record_index.json'
        record = json.loads(index.read_text(encoding='utf-8'))['records'][0]
        self.assertEqual(record['legacy_hash_basis'], 'DAT')
        flat = self.root / 'flat'
        flat.mkdir()
        derivative = flat / 'anonymous.cfg'
        derivative.write_bytes(b'converted config')
        self.assertEqual(match_flat(index, flat)['files'][0]['status'], 'unknown')
        register_derivative(index, record['record_id'], [derivative], 'conversion', {'crop': False})
        plan = match_flat(index, flat)
        self.assertEqual(plan['files'][0]['status'], 'matched')
        targets = restore_flat(plan, self.root / 'restored')
        self.assertEqual(Path(targets[0]).read_bytes(), derivative.read_bytes())
        self.assertIn('derived', targets[0])

    def test_same_dat_different_cfg_requires_review(self):
        for name in ('a', 'b'):
            (self.case / (name + '.cfg')).write_bytes(name.encode())
            (self.case / (name + '.dat')).write_bytes(b'shared')
        output, _ = self.run_prepare()
        flat = self.root / 'flat'
        flat.mkdir()
        (flat / 'anon.dat').write_bytes(b'shared')
        plan = match_flat(output / 'record_index.json', flat)
        self.assertEqual(plan['files'][0]['status'], 'multiple_matches_review')
        self.assertEqual(restore_flat(plan, self.root / 'restored'), [])

    def test_reject_output_inside_source(self):
        with self.assertRaises(ValueError):
            CorpusPreparer(PreparationConfig(str(self.source), str(self.source / 'output')))

    def test_nested_archive_respects_depth(self):
        nested = io.BytesIO()
        with zipfile.ZipFile(nested, 'w') as archive:
            archive.writestr('note.txt', b'nested primary')
        with zipfile.ZipFile(self.case / 'outer.zip', 'w') as archive:
            archive.writestr('inner.zip', nested.getvalue())
        output, summary = self.run_prepare()
        self.assertEqual(summary['archives'], 2)
        files = json.loads((output / 'provenance_index.json').read_text(encoding='utf-8'))['cases'][0]['files']
        self.assertEqual(len(files[0]['origin']['archive_chain']), 2)
        _, skipped = self.run_prepare('limited', max_archive_depth=0)
        self.assertEqual(skipped['unique_blobs'], 0)
        self.assertEqual(skipped['archive_member_issues'], 1)

    def test_shared_blob_corruption_is_detected(self):
        (self.case / 'note.txt').write_bytes(b'primary')
        shared = self.root / 'blobs'
        output, _ = self.run_prepare(blob_root=str(shared))
        second, _ = self.run_prepare('second', blob_root=str(shared))
        blob = next(p for p in shared.rglob('*') if p.is_file())
        blob.write_bytes(b'changed working copy')
        self.assertEqual((self.case / 'note.txt').read_bytes(), b'primary')
        with self.assertRaisesRegex(ValueError, 'blob изменён'):
            self.run_prepare('third', blob_root=str(shared))

    def test_source_indexes_repeat_without_changing_primary(self):
        primary = self.case / '_phase6_author_note.txt'
        primary.write_text('важный первичный текст', encoding='utf-8')
        (self.case / 'blank.txt').write_text(' \r\n\t', encoding='utf-16')
        output, _ = self.run_prepare(write_source_indexes=True)
        repeated, summary = self.run_prepare('repeated', seed_inventory=str(output / 'original_inventory.json'), write_source_indexes=True)
        files = json.loads((repeated / 'provenance_index.json').read_text(encoding='utf-8'))['cases'][0]['files']
        self.assertTrue(next(f for f in files if f['logical_path'] == '_phase6_author_note.txt').get('prepared_path'))
        self.assertTrue(next(f for f in files if f['logical_path'] == 'blank.txt')['text_empty'])
        self.assertTrue(summary['source_files_unchanged'])
        self.assertEqual(primary.read_text(encoding='utf-8'), 'важный первичный текст')

    def test_ndr_identity_keeps_fragment_positions(self):
        first = self.case / 'DR1F0.001'
        second = self.case / 'DR1F1.001'
        first.write_bytes(b'first fragment')
        second.write_bytes(b'second fragment')
        (self.case / 'WConfig.001').write_bytes(b'header')
        output, _ = self.run_prepare()
        records = json.loads((output / 'record_index.json').read_text(encoding='utf-8'))['records']
        self.assertEqual(len(records), 1)
        self.assertEqual(len(records[0]['components']), 3)
        first.write_bytes(b'second fragment')
        second.write_bytes(b'first fragment')
        repeated, _ = self.run_prepare('swapped')
        after = json.loads((repeated / 'record_index.json').read_text(encoding='utf-8'))['records']
        self.assertNotEqual(records[0]['record_id'], after[0]['record_id'])


if __name__ == '__main__':
    unittest.main()
