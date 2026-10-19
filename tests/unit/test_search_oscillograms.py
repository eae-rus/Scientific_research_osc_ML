"""Проверки реального поисковика ноутбука 001 без чтения сигналов."""
import json
from pathlib import Path
import tempfile
import subprocess
import unittest
import zipfile
from unittest.mock import patch

from osc_tools.data_management.search import SearchOscillograms
from osc_tools.core.constants import TYPE_OSC


class SearchTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source = self.root/'source'
        self.source.mkdir()
        self.destination = self.root/'output'

    def tearDown(self):
        self.temporary.cleanup()

    def test_zip_does_not_extract_exe_and_unknown_files(self):
        with zipfile.ZipFile(self.source/'data.zip','w') as archive:
            archive.writestr('setup.exe',b'MZ-not-executed')
            archive.writestr('name.do.exe',b'MZ-not-parma')
            archive.writestr('foo.os5.exe',b'MZ-not-neva')
            archive.writestr('note.txt',b'not a recording')
            archive.writestr('RECORD.DO',b'main')
            archive.writestr('RECORD.D01',b'continuation')
        original = zipfile.ZipFile.open
        opened = []
        def traced(instance, name, *args, **kwargs):
            opened.append(name)
            return original(instance,name,*args,**kwargs)
        with patch.object(zipfile.ZipFile,'open',traced):
            count = SearchOscillograms().copy_new_oscillograms(self.source,self.destination)
        self.assertEqual(count,1)
        self.assertEqual(set(opened),{'RECORD.DO','RECORD.D01'})
        self.assertFalse(list(self.destination.rglob('*.exe')))
        self.assertEqual(len(list(self.destination.rglob('*.D01'))),1)

    def test_nested_zip_reached_and_paths_are_bounded(self):
        nested = self.root/'nested.zip'
        with zipfile.ZipFile(nested,'w') as archive:
            archive.writestr('PAIR.CFG',b'configuration')
            archive.writestr('PAIR.DAT',b'data')
            archive.writestr('../outside.bb',b'bad path')
        with zipfile.ZipFile(self.source/'outer.zip','w') as archive:
            archive.write(nested,'nested.zip')
        search = SearchOscillograms()
        self.assertEqual(search.copy_new_oscillograms(self.source,self.destination),1)
        self.assertEqual(len(list(self.destination.rglob('*.CFG'))),1)
        self.assertFalse((self.root/'outside.bb').exists())
        self.assertTrue(any(e['status'] == 'member_skipped' for e in search.last_result['events']))
        self.assertTrue(all('_search_temp' not in r['origins'][0] for r in search.last_result['records']))

    def test_ndr_dex_and_exact_extension(self):
        for name,data in {'DR1F0.000':b'header','DR1F1.000':b'samples','WConfig.000':b'configuration',
                          'data.dex':b'opaque','signal.os12':b'neva','setup.do.exe':b'no'}.items():
            (self.source/name).write_bytes(data)
        search = SearchOscillograms()
        self.assertEqual(search.copy_new_oscillograms(self.source,self.destination),3)
        kinds = {r['type'] for r in search.last_result['records']}
        self.assertEqual(kinds,{'EKRA_NDR','DEX','NEVA'})
        self.assertTrue(list(self.destination.rglob('WConfig.000')))

    def test_sfx_zip_is_not_given_to_unpacker(self):
        (self.source/'pretend.zip').write_bytes(b'MZ executable disguised as zip')
        search = SearchOscillograms()
        with patch('subprocess.Popen',side_effect=AssertionError('No executable should run')):
            self.assertEqual(search.copy_new_oscillograms(self.source,self.destination),0)
        self.assertEqual(search.last_result['events'][0]['status'],'archive_failed')

    def test_hash_false_counts_and_repeat_repairs_missing_component(self):
        (self.source/'rec.cfg').write_bytes(b'config')
        (self.source/'rec.dat').write_bytes(b'data')
        search = SearchOscillograms()
        self.assertEqual(search.copy_new_oscillograms(self.source,self.destination,use_hashes=False),1)
        self.assertEqual(search.copy_new_oscillograms(self.source,self.destination),0)
        self.assertTrue(json.loads((self.destination/'_hash_table.json').read_text(encoding='utf-8')))
        next(self.destination.rglob('rec.dat')).unlink()
        self.assertEqual(search.copy_new_oscillograms(self.source,self.destination),1)

    def test_output_inside_source_is_not_searched(self):
        (self.source/'one.bb').write_bytes(b'raw')
        destination = self.source/'Output'
        search = SearchOscillograms()
        self.assertEqual(search.copy_new_oscillograms(self.source,destination),1)
        self.assertEqual(search.copy_new_oscillograms(self.source,destination),0)

    def test_filter_does_not_erase_previous_hashes_and_md5_points_to_main(self):
        (self.source/'REC.DO').write_bytes(b'main')
        (self.source/'REC.D01').write_bytes(b'continuation')
        (self.source/'other.bb').write_bytes(b'blackbox')
        search = SearchOscillograms()
        self.assertEqual(search.copy_new_oscillograms(self.source,self.destination),2)
        table = json.loads((self.destination/'_hash_table.json').read_text(encoding='utf-8'))
        parma = next(v for v in table.values() if v[0] == 'REC.DO')
        self.assertTrue(parma[1].endswith('REC.DO'))
        search.copy_new_oscillograms(self.source,self.destination,types_to_copy=[TYPE_OSC.BLACK_BOX])
        after = json.loads((self.destination/'_hash_table.json').read_text(encoding='utf-8'))
        self.assertEqual(table,after)

    @unittest.skipUnless(Path('C:/Program Files/7-Zip/7z.exe').is_file(),'7-Zip не установлен')
    def test_real_7z_extracts_pair_but_not_exe(self):
        seven = 'C:/Program Files/7-Zip/7z.exe'
        files = self.root/'files'
        files.mkdir()
        for name,data in {'pair.cfg':b'config','pair.dat':b'data','installer.exe':b'MZ'}.items():
            (files/name).write_bytes(data)
        subprocess.run([seven,'a',str(self.source/'data.7z'),'pair.cfg','pair.dat','installer.exe'],cwd=files,
                       check=True,capture_output=True,timeout=30)
        search = SearchOscillograms(seven_zip=seven)
        self.assertEqual(search.copy_new_oscillograms(self.source,self.destination),1)
        self.assertFalse(list(self.destination.rglob('*.exe')))
        self.assertEqual(next(self.destination.rglob('pair.dat')).read_bytes(),b'data')


if __name__ == '__main__':
    unittest.main()
