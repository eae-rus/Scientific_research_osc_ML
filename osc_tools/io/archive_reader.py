"""Выборочное извлечение обычных архивов без запуска EXE и самораспаковщиков."""
from pathlib import Path, PurePosixPath
import os
import re
import shutil
import stat
import subprocess
import threading
import zipfile

ARCHIVE_SUFFIXES = {'.zip', '.7z', '.rar'}


def safe_archive_member(name):
    """Проверить относительное имя с учётом путей и специальных имён Windows."""
    name = name.replace('\\', '/')
    parts = PurePosixPath(name).parts
    return bool(parts) and not name.startswith('/') and all(
        p not in {'.', '..'} and not p.endswith((' ', '.')) and not any(c in p for c in ':*?"<>|\x00') and
        not re.match(r'^(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\.|$)', p, re.I) for p in parts)


class ArchiveReader:
    """Перечислять архив и извлекать один проверенный элемент с общими лимитами."""
    def __init__(self, seven_zip=None, max_member_bytes=128*1024**2, max_total_bytes=512*1024**2, max_members=5000):
        self.seven_zip = seven_zip or os.environ.get('PHASE6_7ZIP') or shutil.which('7zz') or shutil.which('7z')
        if not self.seven_zip and Path('C:/Program Files/7-Zip/7z.exe').is_file():
            self.seven_zip = 'C:/Program Files/7-Zip/7z.exe'
        self.max_member_bytes, self.max_total_bytes, self.max_members = max_member_bytes, max_total_bytes, max_members
        if min(max_member_bytes, max_total_bytes, max_members) <= 0:
            raise ValueError('Лимиты архива должны быть положительными')
        self.bytes_read = 0

    def list(self, path):
        """Проверить расширение/сигнатуру; EXE/SFX никогда не передавать распаковщику."""
        path = Path(path)
        suffix = path.suffix.lower()
        with path.open('rb') as stream:
            signature = stream.read(8)
        signatures = {'.zip':(b'PK\x03\x04',b'PK\x05\x06',b'PK\x07\x08'),
                      '.7z':(b'7z\xbc\xaf\x27\x1c',), '.rar':(b'Rar!\x1a\x07',)}
        if suffix not in signatures or not signature.startswith(signatures[suffix]):
            raise ValueError('Не обычный ZIP/7z/RAR; исполняемый файл или SFX не обрабатывается')
        if suffix == '.zip':
            with zipfile.ZipFile(path) as archive:
                if len(archive.infolist()) > self.max_members:
                    raise ValueError('Превышено число элементов архива')
                return [{'name':m.filename,'size':m.file_size,'directory':m.is_dir(),
                         'encrypted':bool(m.flag_bits & 1), 'link':stat.S_ISLNK(m.external_attr >> 16)} for m in archive.infolist()]
        if not self.seven_zip:
            raise RuntimeError('Нужен 7z/7zz в PATH или путь seven_zip')
        result = subprocess.run([self.seven_zip,'l','-slt','-ba','-sccUTF-8','--',str(path)],
                                capture_output=True,timeout=60,check=True)
        members = []
        for block in re.split(r'\r?\n\s*\r?\n', result.stdout.decode('utf-8-sig',errors='replace')):
            data = dict(line.split(' = ',1) for line in block.splitlines() if ' = ' in line)
            if 'Path' in data:
                members.append({'name':data['Path'],'size':int(data.get('Size','0')),
                    'directory':data.get('Folder') == '+' or data.get('Attributes','').startswith('D'),
                    'encrypted':data.get('Encrypted') == '+', 'link':'Symbolic Link' in data or 'Hard Link' in data})
                if len(members) > self.max_members:
                    raise ValueError('Превышено число элементов архива')
        return members

    def extract(self, archive_path, member, target_root):
        """Извлечь выбранный элемент, не используя extractall и имена как команды."""
        name = member['name']
        if (not safe_archive_member(name) or member.get('link') or member.get('encrypted') or
            member.get('directory') or Path(name).suffix.lower() in {'.exe','.dll','.msi','.bat','.cmd','.ps1','.com','.scr'}):
            raise ValueError('Элемент не разрешён для извлечения')
        if member['size'] > self.max_member_bytes or self.bytes_read + member['size'] > self.max_total_bytes:
            raise ValueError('Превышен лимит размера извлечения')
        root = Path(target_root).resolve(strict=True)
        target = root / name.replace('\\','/')
        if not target.resolve().is_relative_to(root):
            raise ValueError('Путь выходит из временного каталога')
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():
            raise FileExistsError(target)
        process, timer, archive = None, None, None
        try:
            if Path(archive_path).suffix.lower() == '.zip':
                archive = zipfile.ZipFile(archive_path)
                stream = archive.open(name)
            else:
                process = subprocess.Popen([self.seven_zip,'e','-so','-bd','-y','-spd','--',str(archive_path),name],
                    stdout=subprocess.PIPE,stderr=subprocess.DEVNULL)
                timer = threading.Timer(60,process.kill)
                timer.start()
                stream = process.stdout
            total = 0
            with stream, target.open('xb') as output:
                while chunk := stream.read(128*1024):
                    total += len(chunk)
                    self.bytes_read += len(chunk)
                    if total > self.max_member_bytes or self.bytes_read > self.max_total_bytes:
                        raise ValueError('Превышен лимит фактически извлечённых байтов')
                    output.write(chunk)
            if process and process.wait(timeout=60) != 0:
                raise RuntimeError('7-Zip не смог извлечь элемент')
            if total != member['size']:
                raise ValueError('Размер элемента не совпал со списком')
            return target
        except Exception:
            target.unlink(missing_ok=True)
            raise
        finally:
            if timer:
                timer.cancel()
            if process and process.poll() is None:
                process.kill()
                process.wait()
            if archive:
                archive.close()
