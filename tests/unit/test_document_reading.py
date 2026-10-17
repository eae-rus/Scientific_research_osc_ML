"""Проверки границ извлечения DOC, порядка DOCX и происхождения изображений."""
from pathlib import Path
import struct
import tempfile
import unittest
import zipfile

from osc_tools.corpus.document_reading import doc_piece_text, read_docx, read_text


def mixed_piece_fixture():
    """Собрать два независимых фрагмента DOC: ANSI и Unicode с хвостом вне тела."""
    word = bytearray(1200)
    struct.pack_into('<HH', word, 0, 0xA5EC, 0x00C1)
    struct.pack_into('<H', word, 32, 0)  # csw
    struct.pack_into('<H', word, 34, 4)  # cslw
    struct.pack_into('<I', word, 48, 5)  # ccpText: AB + Рус
    struct.pack_into('<H', word, 52, 34)  # cbRgFcLcb
    plc = struct.pack('<III', 0, 2, 7)
    plc += struct.pack('<HIH', 0, 0x40000000 | (800 * 2), 0)
    plc += struct.pack('<HIH', 0, 900, 0)
    clx = b'\x01\x02\x00xy\x02' + struct.pack('<I', len(plc)) + plc
    table = b'prefix' + clx
    struct.pack_into('<II', word, 54 + 33 * 8, 6, len(clx))
    word[800:802] = b'AB'
    word[900:910] = 'РусХХ'.encode('utf-16-le')
    return bytes(word), table


class DocumentReadingTests(unittest.TestCase):
    def test_doc_reorders_pieces_and_excludes_non_main_text(self):
        word, table = mixed_piece_fixture()
        self.assertEqual(doc_piece_text(word, table), 'ABРус')

    def test_doc_rejects_encryption_and_invalid_piece_ranges(self):
        word, table = mixed_piece_fixture()
        encrypted = bytearray(word)
        struct.pack_into('<H', encrypted, 10, 0x100)
        with self.assertRaisesRegex(ValueError, 'Зашифрованный'):
            doc_piece_text(bytes(encrypted), table)
        with self.assertRaises(ValueError):
            doc_piece_text(word, table[:-1])
        with self.assertRaises(ValueError):
            doc_piece_text(word[:850], table)

    def test_docx_keeps_table_order_marks_and_image_relationship(self):
        document = '''<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><w:body><w:p><w:r><w:t>Начало</w:t></w:r></w:p><w:tbl><w:tr><w:tc><w:p><w:r><w:rPr><w:b/><w:u w:val="single"/></w:rPr><w:t>Завершенная</w:t><w:drawing><a:blip r:embed="rId1"/></w:drawing></w:r></w:p></w:tc></w:tr></w:tbl><w:p><w:r><w:t>Конец</w:t></w:r></w:p></w:body></w:document>'''
        rels = '''<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Target="media/image1.png"/></Relationships>'''
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / 'source.docx'
            with zipfile.ZipFile(source, 'w') as z:
                z.writestr('word/document.xml', document)
                z.writestr('word/_rels/document.xml.rels', rels)
                z.writestr('word/media/image1.png', b'image-data')
            result = read_docx(source, root / 'media')
            self.assertEqual([b['locator'] for b in result['blocks']], ['body/p1', 'body/tbl2', 'body/p3'])
            self.assertEqual(result['table_rows'][0]['cells'], ['Завершенная'])
            self.assertEqual(result['marked_runs'][0]['properties'], {'b': 'true', 'u': 'single'})
            image = result['images'][0]
            self.assertEqual(image['locator'], 'body/tbl2')
            self.assertEqual(image['status'], 'not_viewed')
            self.assertEqual(Path(image['extracted_path']).read_bytes(), b'image-data')

    def test_docx_does_not_follow_external_or_escape_relationships(self):
        document = '''<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><w:body><w:p><a:blip r:embed="r1"/><a:blip r:embed="r2"/></w:p></w:body></w:document>'''
        rels = '''<Relationships><Relationship Id="r1" Target="https://example.com/a.png" TargetMode="External"/><Relationship Id="r2" Target="..\\..\\escape.png"/></Relationships>'''
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / 'source.docx'
            with zipfile.ZipFile(source, 'w') as z:
                z.writestr('word/document.xml', document)
                z.writestr('word/_rels/document.xml.rels', rels)
            result = read_docx(source, root / 'media')
            self.assertTrue(all('extracted_path' not in i for i in result['images']))
            self.assertFalse((root / 'media').exists())

    def test_cp866_requires_explicit_choice_and_utf16_bom_is_read(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'source.txt'
            path.write_bytes('Авария не обнаружена'.encode('cp866'))
            initial = read_text(path)
            self.assertEqual(initial['alternative_cp866'], 'Авария не обнаружена')
            selected = read_text(path, 'cp866')
            self.assertEqual(selected['blocks'][0]['text'], 'Авария не обнаружена')
            path.write_bytes('Проверка'.encode('utf-16'))
            self.assertEqual(read_text(path)['blocks'][0]['text'], 'Проверка')


if __name__ == '__main__':
    unittest.main()
