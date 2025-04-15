'''Файл для локальной проверки, можно удалить позже'''

import hashlib, os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)


def calculate_file_hash(file_path, algorithm='sha256'):
    hash_obj = hashlib.new(algorithm)

    with open(file_path, 'rb') as f:
        # Считываем файл блоками по 4096 байт для эффективности с большими файлами
        for chunk in iter(lambda: f.read(4096), b''):
            hash_obj.update(chunk)

    return hash_obj.hexdigest()


def compare_files(file1, file2, algorithm='sha256'):
    hash1 = calculate_file_hash(file1, algorithm)
    hash2 = calculate_file_hash(file2, algorithm)

    print(f"File {file1} hash: {hash1}")
    print(f"File {file2} hash: {hash2}")

    return hash1 == hash2

original_csv_path = os.path.join(project_root, 'tests', 'test_data', 'test_csv', 'original.csv')
test_csv_path = os.path.join(project_root, 'tests', 'test_data' ,'test_csv', 'test_output.csv')

if compare_files(original_csv_path, test_csv_path):
    print("Файлы идентичны!")
else:
    print("Файлы отличаются!")
