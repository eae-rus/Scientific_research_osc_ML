"""Inventory the manually annotated OZZ sections before dataset conversion."""

import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from osc_tools.core.comtrade_custom import Comtrade

INPUT = Path(r"D:\DataSet\_spef_dataset_v1.2\osc_comtrade_confirmed_ozz\Размеченные")
OUTPUT = ROOT / "reports/phase5/real_ozz_label_audit.json"
LABEL = re.compile(r"MLsignal_(\d+)_2_1(?:_[123])?", re.IGNORECASE)


def audit(directory):
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    records, errors = [], []
    for cfg in sorted(directory.glob("*.cfg")):
        try:
            record = Comtrade()
            record.load(str(cfg))
            frame = record.to_dataframe()
            sections = {}
            for column in frame.columns:
                match = LABEL.fullmatch(column.strip())
                if match:
                    mask = frame[column].fill_null(0).to_numpy() > 0
                    bus = match[1]
                    sections[bus] = sections[bus] | mask if bus in sections else mask
            records.append(dict(
                id=cfg.stem,
                samples=frame.height,
                positive_samples_by_section={bus: int(mask.sum()) for bus, mask in sections.items()},
            ))
        except (OSError, ValueError, IndexError, KeyError, UnicodeError) as exc:
            errors.append(dict(id=cfg.stem, error=str(exc)))
    return dict(records=records, errors=errors)


if __name__ == "__main__":
    result = audit(INPUT)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    counts = [sum(n > 0 for n in row["positive_samples_by_section"].values()) for row in result["records"]]
    print(f"Read: {len(counts)}; positive files: {sum(n > 0 for n in counts)}; "
          f"positive sections: {sum(counts)}; errors: {len(result['errors'])}")
