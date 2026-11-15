"""Evaluate the held-out real OZZ sections and archive interval predictions."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from osc_tools.ml.phase5_ozz_inference import load_hierarchical, predict_record
from osc_tools.ml.phase5_ozz_study_data import load_section
from osc_tools.ml.phase5_ozz_evaluation import runs

VARIANT = "full"
CHECKPOINT = ROOT / "experiments/phase5/ozz_study" / VARIANT / "latest_checkpoint.pt"
SPLIT = "validation"  # choose threshold here, then change to "real_test"
THRESHOLD = .5


def sustained_mask(scores, valid, threshold, tokens):
    result = np.zeros_like(valid)
    for start, end in runs(valid & (scores >= threshold)):
        if end - start >= tokens:
            result[start+tokens-1:end] = True
    return result


def summarize(records, threshold, tokens):
    files = {}
    for row, score, valid, truth in records:
        prediction = sustained_mask(score, valid, threshold, tokens)
        confirmed, triggered, localized = files.get(row["id"], (False, False, False))
        files[row["id"]] = (confirmed or row["group"] == 5, triggered or bool(prediction.any()),
                            localized or bool((prediction & truth).any()))
    return dict(
        positive_files=sum(y for y, _, _ in files.values()),
        negative_files=sum(not y for y, _, _ in files.values()),
        detected_positive_files=sum(y and detected for y, detected, _ in files.values()),
        localized_positive_files=sum(y and hit for y, _, hit in files.values()),
        false_alarm_files=sum(not y and detected for y, detected, _ in files.values()),
    )


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model, config = load_hierarchical(state, CHECKPOINT, device)
    manifest = json.loads(CHECKPOINT.with_name("split.json").read_text(encoding="utf-8"))
    digest = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    if digest != config["split_sha256"]:
        raise ValueError("Split differs from checkpoint")
    rows = [r for r in manifest[SPLIT] if r["kind"] == "prepared_ozz"]
    if not rows:
        raise ValueError(f"No prepared sections in {SPLIT}")
    output = ROOT / "reports/phase5/ozz_study" / VARIANT / f"{SPLIT}_{state['step']}"
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for i, row in enumerate(rows, 1):
        raw, y = load_section(row)
        pred = predict_record(model, raw, config, device, row["fs"], row["f_network"])
        truth = np.maximum.reduceat(y, pred["starts"]).astype(bool)
        np.savez_compressed(output / f"{row['id']}_bus{row['bus']}.npz", **pred,
                            target=truth, fs=row["fs"], f_network=row["f_network"])
        records.append((row, pred["presence_probability"], pred["valid"], truth))
        if i % 25 == 0 or i == len(rows):
            print(f"{i}/{len(rows)}")
    report = {str(periods): summarize(records, THRESHOLD, tokens)
              for periods, tokens in ((0, 1), (.5, 4), (2, 16))}
    (output / "metrics.json").write_text(json.dumps(dict(threshold=THRESHOLD, results=report), indent=2),
                                         encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
