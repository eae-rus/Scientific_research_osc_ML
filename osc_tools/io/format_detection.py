"""Name-based format detection without loading array libraries or parsers."""
from pathlib import Path
import re


class UnsupportedFormatError(ValueError):
    """Unsupported recording; daily trends are handled separately."""


def detect_format(path: str | Path) -> str:
    """Identify a candidate by name; this does not validate its contents."""
    path = Path(path)
    ext = path.suffix.lower()
    formats = {'.cfg': 'comtrade', '.dat': 'comtrade', '.cff': 'comtrade',
               '.do': 'parma', '.dfr': 'ekra-dfr', '.brs': 'bresler',
               '.bb': 'blackbox', '.sg2': 'res3'}
    if ext in formats:
        return formats[ext]
    if re.fullmatch(r'\.d0\d+', ext):
        return 'parma'
    if re.fullmatch(r'\.os\d*', ext):
        return 'neva'
    if re.fullmatch(r'DR\d+F[0-3]\.\d+', path.name, re.IGNORECASE):
        return 'ekra-ndr'
    raise UnsupportedFormatError(f'Unsupported oscillogram: {path.name}')
