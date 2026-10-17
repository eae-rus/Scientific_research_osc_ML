"""Проверка полного потока и отказа от некорректных отсчётов."""
import json
from pathlib import Path
import struct
import tempfile
import unittest

from osc_tools.corpus.signal_export import validate_payload


class SignalExportTests(unittest.TestCase):
    def make_payload(self, folder, time=(0.0, 0.1, 0.2), analog=(1.0, 2.0, 3.0), digital=b'\x00\x01\x00'):
        metadata = {'protocol': 1, 'total_samples': 3, 'format': 'parma',
                    'analog_channels': [{'name': 'IA', 'uu': 'A'}], 'status_channels': [{'name': 'trip'}]}
        (folder / 'metadata.json').write_text(json.dumps(metadata), encoding='utf-8')
        (folder / 'samples.bin').write_bytes(struct.pack('<3d', *time) + struct.pack('<3d', *analog) + digital)

    def test_all_samples_and_full_time_extent_are_retained(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            self.make_payload(folder)
            original = (folder / 'samples.bin').read_bytes()
            result = validate_payload(folder)
            self.assertEqual(result['total_samples'], 3)
            self.assertEqual(result['duration_s'], 0.2)
            self.assertEqual(result['cropping'], 'none')
            self.assertFalse(result['recovery_requested'])
            self.assertEqual((folder / 'samples.bin').read_bytes(), original)

    def test_truncation_is_rejected_instead_of_recovered(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            self.make_payload(folder, digital=b'\x00\x01')
            with self.assertRaisesRegex(ValueError, 'Размер'):
                validate_payload(folder)

    def test_invalid_time_analog_and_status_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            for options in ({'time': (0.0, 0.0, 0.1)}, {'analog': (1.0, float('nan'), 3.0)}, {'digital': b'\x00\x02\x00'}):
                self.make_payload(folder, **options)
                with self.assertRaises(ValueError):
                    validate_payload(folder)


if __name__ == '__main__':
    unittest.main()
