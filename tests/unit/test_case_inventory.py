"""Standard-library tests: folder preparation must not mislabel or lose files."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from osc_tools.corpus.case_inventory import (
    apply_normalization_plan, classify, folder_inventory, inventory_corpus,
    normalization_plan, sha256,
)
from osc_tools.io.format_detection import detect_format, UnsupportedFormatError


class CaseInventoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def write(self, relative, data=b'example'):
        p = self.root / relative
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        return p

    def test_windows_trailing_space_needs_review(self):
        self.write('2010/report .doc')
        plan = normalization_plan(self.root)
        self.assertEqual(plan['groups'][0]['status'], 'needs_review')
        apply_normalization_plan(self.root, plan)
        self.assertTrue((self.root/'2010/report .doc').is_file())
        forged = copy.deepcopy(plan)
        forged['groups'][0]['status'] = 'ready'
        with self.assertRaises(ValueError):
            apply_normalization_plan(self.root, forged)

    def test_common_detector_contract(self):
        for name, kind in [('a.CFG','comtrade'),('a.DAT','comtrade'),('a.cff','comtrade'),
            ('a.do','parma'),('a.D02','parma'),('a.dfr','ekra-dfr'),('a.brs','bresler'),
            ('a.bb','blackbox'),('a.sg2','res3'),('a.os5','neva'),('DR177F1.046','ekra-ndr')]:
            self.assertEqual(detect_format(name),kind)
        for name in ['a.to','a.t01','a.os5.xml','WConfig.046','arbitrary.046']:
            with self.assertRaises(UnsupportedFormatError):
                detect_format(name)

    def test_components_are_not_independent_candidates(self):
        for name in ['a.CFG','a.dat','b.DAT','p.DO','p.d01','orphan.d02','x.bb','x.to','photo.jpg']:
            self.write('case/'+name)
        m = folder_inventory(self.root/'case')
        self.assertEqual(m['physical_file_counts']['recording_candidate'],3)
        self.assertEqual(m['physical_file_counts']['recording_component'],4)
        files = {x['path']:x for x in m['files']}
        self.assertEqual(files['a.CFG']['companion_status'],'found_by_name')
        self.assertEqual(files['b.DAT']['companion_status'],'missing_by_name')
        self.assertEqual(files['orphan.d02']['main_status'],'missing_by_name')
        self.assertFalse(m['recording_content_read'])
        self.assertFalse(m['images_viewed'])

    def test_archive_is_listed_without_extracting(self):
        p=self.root/'case/archive.zip';p.parent.mkdir()
        with zipfile.ZipFile(p,'w') as z:
            z.writestr('inside/r.bb',b'not parsed')
            z.writestr('../escape.docx',b'no extraction')
        m=folder_inventory(p.parent)
        self.assertEqual(m['recording_presence'],'candidates_found')
        listing=m['files'][0]['listing']
        self.assertFalse(listing['content_read'])
        self.assertFalse(listing['members'][1]['safe_path'])
        self.assertFalse((self.root/'escape.docx').exists())
        self.assertEqual(list(p.parent.iterdir()),[p])

    def test_broken_archive_keeps_presence_unknown(self):
        self.write('case/broken.zip',b'not a zip')
        self.assertEqual(folder_inventory(self.root/'case')['recording_presence'],'unknown')

    def test_exact_duplicates_and_archive_limits(self):
        self.write('case/a.txt',b'same');self.write('case/b.txt',b'same')
        p=self.root/'case/x.zip'
        with zipfile.ZipFile(p,'w') as z:
            z.writestr('a.txt','1');z.writestr('b.txt','2')
        m=folder_inventory(p.parent,max_archive_members=1)
        self.assertEqual(m['byte_duplicate_groups'],[['a.txt','b.txt']])
        self.assertEqual(m['files'][2]['listing']['status'],'partially_listed')
        self.assertEqual(m['recording_presence'],'unknown')

    def test_ids_survive_insertion_and_loose_files_do_not_become_agent_cases(self):
        self.write('2021/Z/a.docx');self.write('2010/loose.doc')
        first=inventory_corpus(self.root)
        self.write('2021/A/b.pdf')
        second=inventory_corpus(self.root,first['case_registry'])
        self.assertEqual(second['case_registry']['2021/Z'],first['case_registry']['2021/Z'])
        self.assertNotEqual(second['case_registry']['2021/A'],first['case_registry']['2021/Z'])
        self.assertEqual(len(second['unprepared_year_files']),1)
        self.assertEqual(len(second['cases']),2)

    def test_plan_is_dry_and_explicit_apply_preserves_bytes(self):
        a=self.write('2010/event.doc',b'word');b=self.write('2010/event.pdf',b'pdf')
        before={a.name:sha256(a),b.name:sha256(b)}
        plan=normalization_plan(self.root)
        self.assertTrue(a.exists() and b.exists())
        self.assertEqual(len(plan['groups']),1)
        apply_normalization_plan(self.root,plan)
        for name,digest in before.items():
            self.assertEqual(sha256(self.root/'2010/event'/name),digest)
        self.assertEqual(normalization_plan(self.root)['groups'],[])

    def test_conflict_and_modified_source_do_not_move_any_files(self):
        a=self.write('2010/a.doc');self.write('2010/existing/doc.docx')
        b=self.write('2010/existing.doc')
        plan=normalization_plan(self.root)
        self.assertIn('target_exists',[x['status'] for x in plan['groups']])
        a.write_bytes(b'changed')
        with self.assertRaises(ValueError):apply_normalization_plan(self.root,plan)
        self.assertTrue(a.exists() and b.exists())

    def test_out_of_scope_plan_is_rejected(self):
        self.write('2010/a.doc')
        plan=normalization_plan(self.root)
        forged=copy.deepcopy(plan);forged['groups'][0]['target']='../outside'
        with self.assertRaises(ValueError):apply_normalization_plan(self.root,forged)
        self.assertTrue((self.root/'2010/a.doc').exists())

    def test_failure_rolls_back_already_moved_document(self):
        a=self.write('2010/a.doc');b=self.write('2010/b.doc')
        plan=normalization_plan(self.root);original=Path.rename
        def fail_second(path,destination):
            if path.resolve()==b.resolve():raise OSError('simulated move failure')
            return original(path,destination)
        with patch.object(Path,'rename',fail_second):
            with self.assertRaises(OSError):apply_normalization_plan(self.root,plan)
        self.assertTrue(a.exists() and b.exists())
        self.assertFalse((self.root/'2010/a').exists())


if __name__=='__main__':
    unittest.main()
