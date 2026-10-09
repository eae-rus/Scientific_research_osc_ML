"""Recompute OZZ scalar statistics and draw the dissertation's two vector plots.

No COMTRADE processing or new physical labels. The CSV is an immutable snapshot
of an existing manually reviewed report. Run with the bundled Python runtime.
"""
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from reportlab.graphics.shapes import Drawing, Line, Rect, String, PolyLine
from reportlab.graphics import renderPDF
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

ROOT = Path(__file__).resolve().parents[3]
DISS = ROOT / "docs/Dissertation"
PRIMARY = ROOT / "data/real_OZZ/overvoltage_report_T1_with_com_v1.7.csv"
INPUT_SHA = "eb9f193dcf4e5ec8e37cd3269e90279ba9e5291da138009b64f0f34031806c5b"
SNAPSHOT = DISS / "planning/sources/STAT-OZZ" / INPUT_SHA / PRIMARY.name
FONT = "OzzArial"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def chart(ymax, ylabel, yticks):
    from reportlab.lib.colors import HexColor
    d = Drawing(540, 300)
    left, bottom, width, height = 62, 48, 460, 230
    x = lambda v: left + (v - .8) / 2.1 * width
    y = lambda v: bottom + v / ymax * height
    for tick in yticks:
        d.add(Line(left, y(tick), left + width, y(tick), strokeColor=HexColor("#dddddd"), strokeWidth=.5))
        d.add(String(left - 9, y(tick) - 3, str(tick).replace(".", ","), fontName=FONT, fontSize=10, textAnchor="end"))
    for tick in np.arange(.8, 2.91, .2):
        d.add(Line(x(tick), bottom, x(tick), bottom - 4, strokeWidth=.6))
        d.add(String(x(tick), bottom - 17, f"{tick:.1f}".replace(".", ","), fontName=FONT, fontSize=10, textAnchor="middle"))
    d.add(Line(left, bottom, left + width, bottom, strokeWidth=.7))
    d.add(Line(left, bottom, left, bottom + height, strokeWidth=.7))
    d.add(String(left, 288, ylabel, fontName=FONT, fontSize=11))
    d.add(String(left + width / 2, 12, "Максимальная кратность перенапряжения", fontName=FONT, fontSize=11, textAnchor="middle"))
    return d, x, y


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=SNAPSHOT if SNAPSHOT.exists() else PRIMARY)
    args = parser.parse_args()
    sha = digest(args.input)
    assert sha == INPUT_SHA, "Input version differs from the audited D03-002 scalar snapshot"
    snapshot = DISS / "planning/sources/STAT-OZZ" / sha / args.input.name
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    if snapshot.exists():
        assert digest(snapshot) == sha, "Snapshot hash mismatch"
    else:
        snapshot.write_bytes(args.input.read_bytes())
    with snapshot.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter=";"))
    flags = Counter(r["Проверка"] for r in rows)
    assert set(flags) == {"+", "-"}, flags
    accepted = [r for r in rows if r["Проверка"] == "+"]
    assert len({r["filename"] for r in accepted}) == len(accepted) == 830
    values = np.array([float(r["overvoltage"]) for r in accepted])
    assert np.isfinite(values).all()
    names = ["n", "mean", "median", "sample_sd", "sample_variance", "min", "q1", "q3", "max"]
    stats = dict(zip(names, [len(values), float(values.mean()), float(np.median(values)),
        float(values.std(ddof=1)), float(values.var(ddof=1)), float(values.min()),
        float(np.quantile(values, .25, method="linear")),
        float(np.quantile(values, .75, method="linear")), float(values.max())]))
    published = dict(zip(names, [830, 1.960, 1.926, .274, .075, .862, 1.823, 2.099, 2.843]))
    assert all(abs(stats[k] - published[k]) <= .0005 + 1e-12 for k in names)
    expected_bins = {"< 1.2": 10, "1.2 - 1.71": 83, "1.71 - 1.75": 17,
        "1.75 - 2.0": 403, "2.0 - 2.5": 286, "2.5 - 3.0": 31}
    observed_bins = dict(Counter(r["overvoltage_group"] for r in accepted))
    assert observed_bins == expected_bins
    conditions = {
        "le_1.65": values <= 1.65,
        "le_2.35": values <= 2.35,
        "le_2.45": values <= 2.45,
        "interval_1.65_2.35_inclusive": (values >= 1.65) & (values <= 2.35),
    }
    counts = {k: {"count": int(mask.sum()), "denominator": len(values),
        "percent": float(mask.mean() * 100)} for k, mask in conditions.items()}
    assert counts["interval_1.65_2.35_inclusive"]["count"] == 678
    pdfmetrics.registerFont(TTFont(FONT, "C:/Windows/Fonts/arial.ttf"))
    from reportlab.lib.colors import HexColor
    figures = DISS / "manuscript/figures"
    figures.mkdir(exist_ok=True)
    # A fresh histogram with declared edges; no attempt to reproduce unknown KDE.
    edges = np.linspace(.8, 2.9, 22)
    hist, _ = np.histogram(values, bins=edges)
    assert int(hist.sum()) == len(values)
    d, x, y = chart(240, "Число записей", [0, 50, 100, 150, 200])
    for a, b, n in zip(edges[:-1], edges[1:], hist):
        d.add(Rect(x(a), y(0), x(b) - x(a), y(int(n)) - y(0),
            fillColor=HexColor("#477fa9"), strokeColor=HexColor("#ffffff"), strokeWidth=.5))
    hist_path = figures / "ozz_overvoltage_histogram.pdf"
    renderPDF.drawToFile(d, str(hist_path))
    d, x, y = chart(100, "Доля записей, %", [0, 20, 40, 60, 80, 100])
    unique, frequency = np.unique(values, return_counts=True)
    points = [x(.8), y(0)]
    cumulative = 0
    for value, number in zip(unique, frequency):
        points.extend([x(value), y(cumulative / len(values) * 100)])
        cumulative += int(number)
        points.extend([x(value), y(cumulative / len(values) * 100)])
    points.extend([x(2.9), y(100)])
    d.add(PolyLine(points, strokeColor=HexColor("#205a89"), strokeWidth=1.3))
    for threshold, key, color, label_y in [(1.65, "le_1.65", "#656565", 66),
            (2.35, "le_2.35", "#a45d25", 116), (2.45, "le_2.45", "#327354", 90)]:
        pct = counts[key]["percent"]
        d.add(Line(x(threshold), y(0), x(threshold), y(pct),
            strokeColor=HexColor(color), strokeWidth=.8, strokeDashArray=[3, 3]))
        label = f"{threshold:.2f} — {pct:.2f}%".replace(".", ",")
        d.add(String(x(threshold) - 8, label_y, label,
            fontName=FONT, fontSize=10, fillColor=HexColor(color), textAnchor="end"))
    cdf_path = figures / "ozz_overvoltage_cdf.pdf"
    renderPDF.drawToFile(d, str(cdf_path))
    proof = {"task_id": "D03-002", "date": "2026-10-08",
        "source_path": str(PRIMARY.relative_to(ROOT)), "source_sha256": sha,
        "snapshot_path": str(snapshot.relative_to(DISS)), "delimiter": ";",
        "source_rows": len(rows), "flags": dict(flags), "selection": 'Проверка == "+"',
        "unique_retained_filenames": len(accepted), "numpy_version": np.__version__,
        "statistics": stats, "published_rounded": published, "all_nine_statistics_match": True,
        "quantile_method": "linear", "variance_ddof": 1, "retained_source_bins": observed_bins,
        "counts": counts, "boundary_equal_counts": {str(v): int((values == v).sum()) for v in [1.65, 2.35, 2.45]},
        "histogram_edges": edges.tolist(), "histogram_counts": hist.tolist(),
        "histogram_convention": "left-closed right-open, last bin right-closed",
        "cdf_method": "empirical right-continuous step; denominator 830; no smoothing",
        "figures": {str(p.relative_to(DISS)): digest(p) for p in [hist_path, cdf_path]},
        "limits": ["Original 1865-candidate article snapshot is not identified with this 1855-row CSV.",
            "830 unique filenames do not establish 830 independent physical events.",
            "Scalar audit does not verify waveform maxima or the original normalization implementation.",
            "Unknown original KDE bandwidth; its curve is not reproduced."]}
    output = DISS / "review/checks/D03-002-statistics.json"
    output.write_text(json.dumps(proof, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"sha256": sha, "rows": len(rows), "stats": stats, "counts": counts}, ensure_ascii=False))


if __name__ == "__main__":
    main()
