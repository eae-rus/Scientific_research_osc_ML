"""Независимая RTDS-проверка: две секции, COMTRADE и предварительный эксперт.

Ручной запуск — run_manual() внизу. Исходные файлы никогда не изменяются.
--prepare-only сохраняет только сигналы и экспертную заготовку (быстрый этап).
Без этого флага рассчитываются 9 формульных органов и 12 вариантов ИИ на секцию.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
from datetime import datetime
import json
from pathlib import Path
import re
import sys
import time

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from osc_tools.io.comtrade_ascii import AnalogChannel, DigitalChannel, ExportRecord, write_comtrade_ascii
from osc_tools.ml.phase5_contracts import TimebaseContract
from osc_tools.pdr.expert_labels import _read_text, read_comtrade_1999_ascii
from osc_tools.pdr.labeler import precompute_causal_h1_phasors
from scripts.phase5_experiments.run_pdr_dataset_study import (
    DEFAULT_ALGORITHMS, _atomic_write_json, create_algorithms,
)
from scripts.phase5_experiments.evaluate_pdr_expert_holdout import _sha256

# РУЧНОЙ ЗАПУСК: F5/Run Python File запускает весь набор; сначала MAX_RECORDS=1.
# Использовать окружение с CUDA PyTorch. Исходные CFG/DAT скрипт не изменяет.
INPUT_ROOT = PROJECT_ROOT / "data/phase5/_FABT_RTDS"  # Исходные опыты, не ручная разметка.
OUTPUT_ROOT = PROJECT_ROOT / "data/phase5/pdr_rtds_review"  # Новые COMTRADE; здесь сначала ждать завершения расчёта.
VERIFIED_ROOT = PROJECT_ROOT / "data/phase5/pdr_rtds_labels"  # Принятые автором пары, включая VALID=0; можно по подпапкам.
AUTHOR_RECHECK_PENDING = True  # Черновой отчёт: автор ещё проверит валидность напряжений; файлы пока не исключать.
CURRENT_NOMINAL_A = 3000.0  # Действующий номинальный фазный ток, А; резерв 20 применяется ниже один раз.
VOLTAGE_NOMINAL_V = 6000.0  # Линейный номинал сети; фазные отсчёты делятся на него без повторного sqrt(3).
INFERENCE_STRIDE = 1  # Разметка каждого исходного отсчёта. Другие значения для итогового COMTRADE запрещены.
BATCH_SIZE = 512  # Окна ИИ в GPU-пакете; автоматически уменьшается при нехватке VRAM.
DEVICE = "cuda"  # Требовать CUDA, не переходить незаметно на медленный CPU.
PHASOR_BACKEND = "fft"  # Точный прежний вход моделей; prefix быстрее, но меняет углы почти нулевых компонент.
STARTUP_POLICY = "early"  # early: с первого периода; full_context: прежнее ожидание 20T.
PREPARE_ONLY = False  # True: только физические каналы и эксперт; False: также 9 РНМ и 12 ИИ.
MAX_RECORDS = None  # None: все 100; 1: короткая проверка полного маршрута, без обучения.
MODES = ("snapshot_2", "snapshot_5", "sequence_1_8")  # Для каждого берутся weak/expert и last/best.
INTERNAL_FAULT_RECORDS = {"2.25", "2.26"}  # Исключение только для секции 1 по таблице Г.5.


def read_rtds(path: Path) -> tuple[ExportRecord, np.ndarray, np.ndarray, TimebaseContract]:
    """Строго проверить карту каналов, единицы и близость дискретов к 0/1."""
    rec = read_comtrade_1999_ascii(path)
    rows = list(csv.reader(_read_text(path).splitlines()))
    if rows[1] != ["25", "25A", "0D"]:
        raise ValueError(f"Изменилась карта 25A/0D: {path}")
    info = rows[2:27]
    expected = ["Ток через ВВ1", "Напряжение ввода 1", "Напряжение 1 СШ",
                "Ток через ВВ2", "Напряжение ввода 2", "Напряжение 2 СШ"]
    values, channels = [], []
    phase_alias = {"А": "A", "В": "B", "С": "C", "A": "A", "B": "B", "C": "C"}
    for index, row in enumerate(info[:18]):
        name = row[1].strip()
        phase = phase_alias.get(name[-1])
        if not name.startswith(expected[index // 3]) or phase != "ABC"[index % 3]:
            raise ValueError(f"Неожиданный канал {index + 1}: {name}")
        unit = row[4].strip().lower().replace("к", "k").replace("а", "a").replace("в", "v")
        target = "a" if index // 3 in (0, 3) else "v"
        if unit not in (target, "k" + target):
            raise ValueError(f"Неожиданные единицы {name}: {row[4]}")
        physical = (rec.analog[name] * float(row[5]) + float(row[6])) * (1000 if unit.startswith("k") else 1)
        values.append(physical)
        channels.append(AnalogChannel(name, target.upper(), physical, phase=phase))
    binary, digital = [], []
    expected_digital = ("Положение ВВ1", "Положение ВВ2", "Положение СВ", "Наличие КЗ в сети",
                        "Сигнал отключения ВВ1 от БАВР", "Сигнал отключения ВВ2 от БАВР", "Сигнал включения СВ от БАВР")
    for row, expected_name in zip(info[18:], expected_digital):
        if row[1].strip() != expected_name:
            raise ValueError(f"Изменилась карта дискретов: {row[1]}")
        physical = rec.analog[expected_name] * float(row[5]) + float(row[6])
        rounded = np.rint(physical)
        if not np.isfinite(physical).all() or not np.isin(rounded, (0, 1)).all() or np.max(np.abs(rounded - physical)) > 1e-4:
            raise ValueError(f"Не двоичный канал {expected_name}: {path}")
        binary.append(rounded.astype(np.uint8))
        digital.append(DigitalChannel(expected_name, binary[-1]))
    tb = TimebaseContract.create(rec.sample_rate_hz, float(rows[27][0]))
    expected_time = np.arange(rec.n_samples) * 1e6 / rec.sample_rate_hz
    if not np.allclose(rec.timestamps_us, expected_time, atol=1, rtol=0):
        raise ValueError(f"Неравномерная временная сетка: {path}")
    def date(row: list[str]) -> datetime:
        text = ",".join(row)
        for fmt in ("%d/%m/%Y,%H:%M:%S.%f", "%m/%d/%y,%H:%M:%S.%f"):
            try:
                return datetime.strptime(text, fmt)
            except ValueError:
                pass
        raise ValueError(f"Нераспознанная дата: {text}")
    out = ExportRecord(rows[0][0], rows[0][1], rec.sample_rate_hz, tb.network_frequency_hz,
                       date(rows[30]), date(rows[31]), tuple(channels), tuple(digital), "cp1251")
    return out, np.asarray(values), np.asarray(binary), tb


def section_signals(physical: np.ndarray, section: int) -> tuple[np.ndarray, np.ndarray]:
    """I ввода и U сборных шин; номинальные резервы применяются ровно один раз."""
    if section not in (1, 2):
        raise ValueError("Секция должна быть 1 или 2")
    offset = 9 * (section - 1)
    signals = np.full((8, physical.shape[1]), np.nan, dtype=np.float32)
    signals[:3] = physical[offset:offset + 3] / (CURRENT_NOMINAL_A * 20)
    signals[4:7] = physical[offset + 6:offset + 9] / (VOLTAGE_NOMINAL_V * 3)
    return signals, np.array([1, 1, 1, 0, 1, 1, 1, 0], dtype=np.uint8)


def expert_seed(current_rms: np.ndarray, fault: np.ndarray, breaker: np.ndarray,
                *, internal_fault: bool) -> tuple[np.ndarray, np.ndarray]:
    """Ретроспективный черновик, не причинный орган и не подтверждённый эталон.

    Причина: 0 — базовое 1; 1 — все I1<30А; 2 — внешнее КЗ с отключением;
    3 — внутреннее КЗ; 4 — ещё нет полного Фурье (оставлена базовая единица).
    Приоритет отсутствия тока выше требования внутреннего КЗ по указанию автора.
    """
    result = np.ones(len(fault), dtype=np.uint8)
    reason = np.zeros(len(fault), dtype=np.uint8)
    known = np.isfinite(current_rms).all(axis=0)
    reason[~known] = 4
    edges = np.diff(np.r_[0, fault, 0].astype(np.int16))
    falling = np.r_[False, np.diff(breaker.astype(np.int16)) == -1]
    for start, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if internal_fault:
            reason[start:stop] = 3
        elif falling[start:stop].any():
            result[start:stop], reason[start:stop] = 0, 2
    absent = known & (current_rms < .01 * CURRENT_NOMINAL_A).all(axis=0)
    result[absent], reason[absent] = 0, 1
    return result, reason


def load_models(*, h123_only: bool = False, model_runs=None, checkpoint_kinds=None) -> dict:
    """Все weak/expert × три представления × latest/best; без подмены весов."""
    from scripts.phase5_experiments.run_phase5_pdr_training import _imports, _build_model
    from scripts.phase5_experiments.evaluate_pdr_expert_holdout import _config_from_json
    torch, *_ = _imports()
    torch.set_num_threads(2)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if DEVICE == "auto" else DEVICE
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("GPU недоступен: используйте PyTorch с CUDA или явно DEVICE='cpu'.")
    print(f"[Устройство] {device}; torch={torch.__version__}; batch={BATCH_SIZE}", flush=True)
    result = {}
    if model_runs is not None:
        filenames = {"latest": "latest_checkpoint.pt", "best": "best_model.pt"}
        for name, folder_name in model_runs.items():
            if not name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in name):
                raise ValueError("Недопустимое имя модели")
            folder = PROJECT_ROOT / folder_name
            for kind in (checkpoint_kinds or ("latest", "best")):
                path = folder / filenames[kind]
                cfg = _config_from_json(folder / "config.json")
                model, head, _ = _build_model(cfg, None, path)
                key = f"nn_{name}_{'last' if kind == 'latest' else kind}"
                result[key] = (cfg, model.to(device).eval(), head.to(device).eval(), device,
                               {"path": str(path), "sha256": _sha256(path), "config_sha256": _sha256(folder / "config.json")})
        if not result:
            raise ValueError("Список моделей пуст")
        return result
    for stage in ("weak", "expert"):
        for mode in (("snapshot_5",) if h123_only else MODES):
            folder = PROJECT_ROOT / f"experiments/phase5/pdr_{stage}_{mode}_stride5"
            if h123_only:
                folder = folder.with_name(folder.name + "_h123_deep24")
            for kind, filename in (("last", "latest_checkpoint.pt"), ("best", "best_model.pt")):
                path = folder / filename
                cfg = _config_from_json(folder / "config.json")
                model, head, _ = _build_model(cfg, None, path)
                key = f"nn_{stage}_{mode}{'_h123_deep24' if h123_only else ''}_{kind}"
                result[key] = (cfg, model.to(device).eval(), head.to(device).eval(), device,
                               {"path": str(path), "sha256": _sha256(path), "config_sha256": _sha256(folder / "config.json")})
    return result


def augment_h123(input_root: Path, output_root: Path, *, max_records: int | None = None,
                 skip_existing: bool = True, model_runs=None, checkpoint_kinds=None) -> None:
    """Дополнить копии ручных COMTRADE, не пересчитывая прежние органы."""
    from dataclasses import replace
    from scripts.visualization.generate_pdr_article_figures import _record_spectral_cache, _predict_cached
    from scripts.phase5_experiments.progress import ProgressReporter
    input_root, output_root = input_root.resolve(), output_root.resolve()
    if input_root == output_root or input_root in output_root.parents or output_root in input_root.parents:
        raise ValueError("Вход и выход должны быть отдельными непересекающимися папками")
    if INFERENCE_STRIDE != 1:
        raise ValueError("COMTRADE требует INFERENCE_STRIDE=1")
    paths = sorted(input_root.rglob("*.cfg"))
    if max_records is not None:
        paths = paths[:max_records]
    if not paths:
        raise FileNotFoundError(input_root)
    if len({p.stem for p in paths}) != len(paths):
        raise ValueError("Повтор номера опыта во входной папке")
    models = (load_models(h123_only=True) if model_runs is None else
              load_models(model_runs=model_runs, checkpoint_kinds=checkpoint_kinds))
    code = {str(p.relative_to(PROJECT_ROOT)): _sha256(p) for folder in ("osc_tools/ml", "osc_tools/pdr")
            for p in (PROJECT_ROOT / folder).rglob("*.py")}
    code["runner"] = _sha256(Path(__file__))
    code["inference"] = _sha256(PROJECT_ROOT / "scripts/visualization/generate_pdr_article_figures.py")
    reporter = ProgressReporter("RTDS: добавление моделей", len(paths), unit="опыт")
    output_root.mkdir(parents=True, exist_ok=True)
    completed = 0
    try:
        for path in paths:
            out = output_root / path.relative_to(input_root)
            sidecar = path.with_suffix(".json")
            if not sidecar.exists():
                sidecar = OUTPUT_ROOT / (path.stem + ".json")
            metadata = json.loads(sidecar.read_text(encoding="utf-8"))
            fingerprint = {"cfg": _sha256(path), "dat": _sha256(path.with_suffix(".dat")),
                "metadata": _sha256(sidecar), "models": {k: v[4] for k, v in models.items()},
                "code": code, "startup": STARTUP_POLICY, "phasor_backend": PHASOR_BACKEND,
                "current_nominal_a": CURRENT_NOMINAL_A, "voltage_nominal_v": VOLTAGE_NOMINAL_V}
            old = json.loads(out.with_suffix(".json").read_text(encoding="utf-8")) if out.with_suffix(".json").exists() else {}
            for suffix in (".cfg", ".dat"):
                existing = out.with_suffix(suffix)
                if existing.exists() and _sha256(existing) != old.get("output_hashes", {}).get(suffix):
                    raise ValueError(f"Выход изменён или не подтверждён; перезапись запрещена: {existing}")
            ready = all(out.with_suffix(s).exists() for s in (".cfg", ".dat"))
            if skip_existing and ready and old.get("augmentation_fingerprint") == fingerprint:
                print(f"[Готово ранее] {path.stem}", flush=True)
            else:
                record, physical, _, tb = read_rtds(INPUT_ROOT / path.name)
                edited = read_comtrade_1999_ascii(path)
                if edited.n_samples != physical.shape[1] or edited.sample_rate_hz != record.sample_rate_hz:
                    raise ValueError(f"Изменены длина/частота: {path}")
                if not np.allclose(edited.timestamps_us, np.arange(edited.n_samples)*1e6/edited.sample_rate_hz, atol=1, rtol=0):
                    raise ValueError(f"Изменена временная сетка: {path}")
                expected = metadata.get("automatic_digital_hashes", {})
                if not expected:
                    raise ValueError(f"Нет хешей исходных автоматических ответов: {sidecar}")
                for name, digest in expected.items():
                    if name not in edited.digital or hashlib.sha256(edited.digital[name].tobytes()).hexdigest() != digest:
                        raise ValueError(f"Изменён автоматический дискрет: {name}, {path}")
                cfg_rows = [[v.strip() for v in r] for r in csv.reader(_read_text(path).splitlines())]
                analog_count = int(cfg_rows[1][1].rstrip("Aa"))
                analog = tuple(AnalogChannel(r[1], r[4], edited.analog[r[1]]*float(r[5])+float(r[6]), r[2], r[3])
                               for r in cfg_rows[2:2+analog_count])
                by_name = {a.name: a for a in analog}
                for a in record.analog:
                    b = by_name[a.name]
                    if a.unit != b.unit or not np.allclose(a.values, b.values, rtol=1e-7, atol=1e-6):
                        raise ValueError(f"Изменён исходный аналоговый сигнал: {a.name}")
                digital = [DigitalChannel(name, values) for name, values in edited.digital.items()]
                new_names = {f"S{s}__{name}__{field}" for s in (1, 2) for name in models for field in ("FWD", "VALID")}
                if new_names.intersection(edited.digital):
                    raise ValueError("Во входе уже есть выбранные модели: исключите их из списка дополнения")
                for section in (1, 2):
                    for field in ("FWD", "VALID"):
                        if f"S{section}__expert__{field}" not in edited.digital:
                            raise ValueError("Нет экспертных каналов")
                    signals, prov = section_signals(physical, section)
                    shared = {}
                    for name, (cfg, model, head, device, _) in models.items():
                        from osc_tools.ml.spectral_features import SpectralFeatureConfig
                        history = max(SpectralFeatureConfig(cfg.feature_version).low_periods, default=1)
                        first = tb.spp-1 if STARTUP_POLICY == "early" else (10+history)*tb.spp-1
                        ends = np.arange(first, edited.n_samples)
                        print(f"[Модель] {completed+1}/{len(paths)} {path.stem}, S{section}, {name}", flush=True)
                        f, p, lookup = _record_spectral_cache(signals, prov, "phase", tb, cfg.temporal_mode,
                            cfg.feature_version, ends, startup_policy=STARTUP_POLICY, shared_cache=shared, phasor_backend=PHASOR_BACKEND)
                        direction, probability = _predict_cached(model, head, f, p, lookup, device, BATCH_SIZE, shared)
                        state = np.full(edited.n_samples, -999, dtype=np.int16)
                        state[ends] = np.where(probability >= .5, direction, -999)
                        digital.extend(_state_channels(f"S{section}__{name}", state))
                    shared.clear()
                write_comtrade_ascii(replace(record, analog=analog, digital=tuple(digital)), out, out.with_suffix(".dat"))
                metadata.update({"augmentation_fingerprint": fingerprint, "expert_status": "copied_from_author_labels",
                    "expert_reference_path": str(path), "additional_algorithm_ids": list(dict.fromkeys([*metadata.get("additional_algorithm_ids", []), *models])),
                    "automatic_digital_hashes": {ch.name: hashlib.sha256(ch.values.astype(np.uint8).tobytes()).hexdigest()
                        for ch in digital if "__expert__" not in ch.name},
                    "output_hashes": {s: _sha256(out.with_suffix(s)) for s in (".cfg", ".dat")}})
                _atomic_write_json(out.with_suffix(".json"), metadata)
            completed += 1
            reporter.update(completed)
            _atomic_write_json(output_root / "progress.json", {"status": "running", "completed": completed, "total": len(paths)})
    except Exception as exc:
        _atomic_write_json(output_root / "progress.json", {"status": "failed", "completed": completed, "total": len(paths), "error": str(exc)})
        raise
    reporter.finish()
    _atomic_write_json(output_root / "progress.json", {"status": "complete", "completed": completed, "total": len(paths)})


def run(*, prepare_only: bool = False, max_records: int | None = None) -> None:
    """Последовательно обрабатывать записи; выход возобновляемый по хешам."""
    from dataclasses import replace
    from osc_tools.pdr.study import label_record_multi
    if INFERENCE_STRIDE != 1:
        raise ValueError("Для итоговых COMTRADE нужен шаг 1: прореживание дискретов запрещено")
    models = {} if prepare_only else load_models()
    algorithms = [] if prepare_only else create_algorithms(DEFAULT_ALGORITHMS)
    algorithm_params = json.loads(json.dumps({a.resolved_algorithm_id: a.params for a in algorithms}, default=str))
    verified = {p.stem for p in VERIFIED_ROOT.rglob("*.cfg") if p.with_suffix(".dat").exists()} if VERIFIED_ROOT.exists() else set()
    paths = sorted(INPUT_ROOT.glob("*.cfg"), key=lambda p: tuple(map(int, re.findall(r"\d+", p.stem))))
    if max_records is not None:
        paths = paths[:max_records]
    if not paths:
        raise FileNotFoundError(INPUT_ROOT)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    started, completed = time.monotonic(), 0
    _atomic_write_json(OUTPUT_ROOT / "progress.json", {"status": "running", "completed": 0,
        "total": len(paths), "prepare_only": prepare_only})
    try:
        for path in paths:
            if path.stem in verified:
                completed += 1
                print(f"[Уже перенесено в проверенные] {path.stem}", flush=True)
                continue
            fingerprint = {"schema": 2, "source_cfg": _sha256(path), "source_dat": _sha256(path.with_suffix(".dat")),
                           "models": {k: v[4] for k, v in models.items()}, "prepare_only": prepare_only,
                           "algorithm_params": algorithm_params,
                           "stride": INFERENCE_STRIDE, "startup": STARTUP_POLICY, "phasor_backend": PHASOR_BACKEND,
                           "current_nominal_a": CURRENT_NOMINAL_A, "voltage_nominal_v": VOLTAGE_NOMINAL_V}
            sidecar = OUTPUT_ROOT / (path.stem + ".json")
            fingerprint = json.loads(json.dumps(fingerprint, default=str))
            existing = json.loads(sidecar.read_text(encoding="utf-8")) if sidecar.exists() else {}
            out_cfg = OUTPUT_ROOT / path.name
            for suffix in (".cfg", ".dat"):
                output = out_cfg.with_suffix(suffix)
                recorded_hash = existing.get("output_hashes", {}).get(suffix)
                if output.exists() and recorded_hash and _sha256(output) != recorded_hash:
                    raise RuntimeError(f"Выход изменён вручную, перезапись запрещена: {output}. Выберите новый OUTPUT_ROOT.")
            if existing.get("fingerprint") == fingerprint and out_cfg.exists() and out_cfg.with_suffix(".dat").exists():
                print(f"[Готово ранее] {path.stem}", flush=True)
            else:
                print(f"[RTDS] {completed + 1}/{len(paths)} {path.stem}", flush=True)
                record, physical, binary, tb = read_rtds(path)
                digital, summaries = list(record.digital), {}
                experiment = path.stem.removeprefix("Oscilogramma")
                for section in (1, 2):
                    signals, prov = section_signals(physical, section)
                    n = signals.shape[1]
                    first = tb.spp - 1 if STARTUP_POLICY == "early" else 20 * tb.spp - 1
                    ends = np.arange(first, n, INFERENCE_STRIDE, dtype=np.int64)
                    rms = np.full((3, n), np.nan)
                    h1 = precompute_causal_h1_phasors(signals, list(range(tb.spp - 1, n)), tb.spp)
                    for sample, phasors in h1.items():
                        rms[:, sample] = [abs(phasors[c]) * CURRENT_NOMINAL_A * 20 / np.sqrt(2) for c in ("IA", "IB", "IC")]
                    del h1
                    seed, reason = expert_seed(rms, binary[3], binary[section - 1],
                                               internal_fault=experiment in INTERNAL_FAULT_RECORDS and section == 1)
                    digital.extend((DigitalChannel(f"S{section}__expert__FWD", seed),
                                    DigitalChannel(f"S{section}__expert__VALID", np.ones(n, dtype=np.uint8))))
                    summaries[f"S{section}"] = {"seed_reason_counts": {str(k): int((reason == k).sum()) for k in range(5)}}
                    if algorithms:
                        print(f"  Секция {section}: формульные органы", flush=True)
                        result = label_record_multi(signals, prov, tb, "phase", algorithms, sample_step=INFERENCE_STRIDE)
                        for index, name in enumerate(result.algorithm_ids):
                            state = np.full(n, -999, dtype=np.int16)
                            state[result.sample_indices] = result.directions[index]
                            digital.extend(_state_channels(f"S{section}__{name}", state))
                    shared = {}
                    for name, (cfg, model, head, device, _) in models.items():
                        import torch
                        from osc_tools.pdr.pdr_trainer import extract_backbone_features
                        from scripts.visualization.generate_pdr_article_figures import _record_spectral_cache, _predict_cached
                        print(f"  Секция {section}: {name}, {len(ends)} точек", flush=True)
                        features, provenance, lookup = _record_spectral_cache(signals, prov, "phase", tb,
                            cfg.temporal_mode, cfg.feature_version, ends, startup_policy=STARTUP_POLICY, shared_cache=shared, phasor_backend=PHASOR_BACKEND)
                        state = np.full(n, -999, dtype=np.int16)
                        direction, probability = _predict_cached(model, head, features, provenance, lookup, device, BATCH_SIZE, shared)
                        state[ends] = np.where(probability >= .5, direction, -999)
                        digital.extend(_state_channels(f"S{section}__{name}", state))
                    shared.clear()
                write_comtrade_ascii(replace(record, digital=tuple(digital)), out_cfg, out_cfg.with_suffix(".dat"))
                _atomic_write_json(sidecar, {"fingerprint": fingerprint, "expert_status": "provisional_requires_review",
                                             "sections": summaries, "source": str(path),
                                             "signal_peak_absolute": {ch.name: float(np.max(np.abs(ch.values))) for ch in record.analog},
                                             "automatic_digital_hashes": {ch.name: hashlib.sha256(np.asarray(ch.values, dtype=np.uint8).tobytes()).hexdigest()
                                                 for ch in digital if "__expert__" not in ch.name},
                                             "output_hashes": {suffix: _sha256(out_cfg.with_suffix(suffix)) for suffix in (".cfg", ".dat")}})
            completed += 1
            elapsed = time.monotonic() - started
            _atomic_write_json(OUTPUT_ROOT / "progress.json", {"status": "running", "completed": completed,
                "total": len(paths), "elapsed_seconds": elapsed, "eta_seconds": elapsed * (len(paths) - completed) / completed})
    except Exception as exc:
        _atomic_write_json(OUTPUT_ROOT / "progress.json", {"status": "failed", "completed": completed, "total": len(paths), "error": str(exc)})
        raise
    _atomic_write_json(OUTPUT_ROOT / "progress.json", {"status": "complete", "completed": completed, "total": len(paths), "prepare_only": prepare_only})


def _state_channels(prefix: str, state: np.ndarray) -> tuple[DigitalChannel, DigitalChannel]:
    return (DigitalChannel(prefix + "__FWD", (state == 1).astype(np.uint8)),
            DigitalChannel(prefix + "__VALID", (state != -999).astype(np.uint8)))


def _comparison_metrics(target: np.ndarray, applicable: np.ndarray, direction: np.ndarray,
                        valid: np.ndarray, sample_rate: float) -> dict:
    """Три состояния для оценки (не классы обучения); отказ не повышает покрытие.

    Строки матрицы — эксперт, столбцы — орган, порядок: INVALID, REV, FWD.
    Дополнительная оценка исключает переход экспертного состояния и 5 мс после
    него. Основные показатели по-прежнему включают все отсчёты, в том числе старт.
    """
    applicable, valid = applicable.astype(bool), valid.astype(bool)
    truth = np.where(applicable, target + 1, 0).astype(np.int64)
    pred = np.where(valid, direction + 1, 0).astype(np.int64)
    both = applicable & valid
    correct = truth == pred
    transitions = np.flatnonzero(np.diff(truth) != 0) + 1
    stable = np.ones(len(target), dtype=bool)
    for start in transitions:
        stable[start:start + int(np.floor(.005 * sample_rate)) + 1] = False
    error_edges = np.diff(np.r_[False, ~correct, False].astype(np.int8))
    lengths = np.flatnonzero(error_edges == -1) - np.flatnonzero(error_edges == 1)
    return {
        "samples": len(target), "direction_samples": int(both.sum()),
        "direction_accuracy": float((direction[both] == target[both]).mean()) if both.any() else None,
        "direction_coverage": float(both.sum() / applicable.sum()) if applicable.any() else None,
        "validity_accuracy": float((valid == applicable).mean()),
        "state_accuracy": float(correct.mean()),
        "stable_state_accuracy": float(correct[stable].mean()) if stable.any() else None,
        "state_confusion": np.bincount(truth * 3 + pred, minlength=9).reshape(3, 3).tolist(),
        "expert_transitions": int(np.count_nonzero((np.diff(target.astype(int)) != 0) & applicable[1:] & applicable[:-1])),
        "algorithm_transitions": int(np.count_nonzero((np.diff(direction.astype(int)) != 0) & valid[1:] & valid[:-1])),
        "expert_state_transitions": len(transitions),
        "algorithm_state_transitions": int(np.count_nonzero(np.diff(pred))),
        "error_episodes_over_20ms": int(np.count_nonzero(lengths / sample_rate > .020)),
        "max_error_episode_ms": float(lengths.max() * 1000 / sample_rate) if len(lengths) else 0.,
    }


def evaluate_verified(*, preliminary: bool = False, root: Path | None = None, output: Path | None = None) -> dict:
    """Прочитать принятые экспертом пары; лишние сигналы игнорируются.

    Сверка аналоговых отсчётов защищает от случайной подмены опыта. Автоматические
    ответы берутся из неизменённых цифровых каналов экспортированной пары;
    это первичная статистика, не оценка сертификационной надёжности БАВР.
    """
    rows, inventory, references = [], [], []
    started = time.monotonic()
    evaluation_root = root if root is not None else (OUTPUT_ROOT if preliminary else VERIFIED_ROOT)
    files = sorted(evaluation_root.rglob("*.cfg")) if evaluation_root.exists() else []
    if not files:
        raise FileNotFoundError(f"Нет CFG в {evaluation_root}")
    seen = set()
    for file_index, cfg_path in enumerate(files, 1):
        if cfg_path.stem in seen:
            raise ValueError(f"Повтор одного опыта: {cfg_path.stem}")
        seen.add(cfg_path.stem)
        original, _, _, _ = read_rtds(INPUT_ROOT / cfg_path.name)
        edited = read_comtrade_1999_ascii(cfg_path)
        sidecar = cfg_path.with_suffix(".json")
        if not sidecar.exists():
            sidecar = OUTPUT_ROOT / (cfg_path.stem + ".json")
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
        references.append({"record": cfg_path.stem, "cfg_sha256": _sha256(cfg_path),
                           "dat_sha256": _sha256(cfg_path.with_suffix(".dat")),
                           "calculation_fingerprint": metadata.get("fingerprint"),
                           "augmentation_fingerprint": metadata.get("augmentation_fingerprint")})
        expected_hashes = metadata.get("automatic_digital_hashes")
        if not expected_hashes:
            raise ValueError(f"Нет контрольных хешей автоматических сигналов: {sidecar}")
        for name, expected_hash in expected_hashes.items():
            if name not in edited.digital or hashlib.sha256(edited.digital[name].tobytes()).hexdigest() != expected_hash:
                raise ValueError(f"Изменён неэкспертный дискрет {name}: {cfg_path}")
        if edited.sample_rate_hz != original.sample_rate_hz:
            raise ValueError(f"Изменена частота: {cfg_path}")
        # Редактор COMTRADE добавляет пробелы после запятых: это не смена каналов.
        cfg_rows = [[field.strip() for field in row] for row in csv.reader(_read_text(cfg_path).splitlines())]
        count = int(cfg_rows[1][1].rstrip("Aa"))
        analog_info = {r[1]: r for r in cfg_rows[2:2 + count]}
        for channel in original.analog:
            info = analog_info[channel.name]
            measured = edited.analog[channel.name] * float(info[5]) + float(info[6])
            if info[4] != channel.unit or measured.shape != channel.values.shape or not np.allclose(measured, channel.values, rtol=1e-7, atol=1e-6):
                raise ValueError(f"Изменён исходный сигнал: {cfg_path}, {channel.name}")
        if not np.allclose(edited.timestamps_us, np.arange(edited.n_samples) * 1e6 / edited.sample_rate_hz, rtol=0, atol=1):
            raise ValueError(f"Изменена временная сетка: {cfg_path}")
        for section in (1, 2):
            prefix = f"S{section}__"
            target = edited.digital[prefix + "expert__FWD"]
            applicable = edited.digital[prefix + "expert__VALID"].astype(bool)
            voltage_channels = [ch for ch in original.analog if ch.name.startswith(f"Напряжение {section} СШ")]
            peak = max((float(np.max(np.abs(ch.values))) for ch in voltage_channels), default=None)
            inventory.append({"record": cfg_path.stem, "section": section,
                "samples": edited.n_samples, "sample_rate_hz": edited.sample_rate_hz,
                "duration_seconds": edited.n_samples / edited.sample_rate_hz,
                "expert_invalid_samples": int((~applicable).sum()),
                "expert_reverse_samples": int((applicable & (target == 0)).sum()),
                "expert_forward_samples": int((applicable & (target == 1)).sum()),
                "voltage_absolute_peak_v": peak,
                "voltage_peak_to_nominal_phase_peak": peak / (VOLTAGE_NOMINAL_V * np.sqrt(2/3)) if peak is not None else None})
            names = list(DEFAULT_ALGORITHMS) + [f"nn_{stage}_{mode}_{kind}" for stage in ("weak", "expert") for mode in MODES for kind in ("last", "best")]
            names += metadata.get("additional_algorithm_ids", [])
            for name in names:
                fwd_key, valid_key = prefix + name + "__FWD", prefix + name + "__VALID"
                if fwd_key not in edited.digital or valid_key not in edited.digital:
                    raise ValueError(f"Нет расчёта {name} в {cfg_path}; сначала нужен полный экспорт")
                direction, valid = edited.digital[fwd_key], edited.digital[valid_key].astype(bool)
                rows.append({"record": cfg_path.stem, "section": section, "algorithm": name,
                    **_comparison_metrics(target, applicable, direction, valid, edited.sample_rate_hz)})
        elapsed = time.monotonic() - started
        print(f"[Оценка RTDS] {file_index}/{len(files)}; {elapsed:.1f} с; {cfg_path.stem}", flush=True)
    aggregate = {}
    for name in sorted({r["algorithm"] for r in rows}):
        group = [r for r in rows if r["algorithm"] == name]
        aggregate[name] = {key: float(np.mean([r[key] for r in group if r[key] is not None]))
                           for key in ("direction_accuracy", "direction_coverage", "validity_accuracy", "state_accuracy", "stable_state_accuracy")
                           if any(r[key] is not None for r in group)}
        aggregate[name]["state_confusion_sum"] = np.sum([r["state_confusion"] for r in group], axis=0).tolist()
        aggregate[name]["record_sections"] = len(group)
        aggregate[name]["direction_evaluable_record_sections"] = sum(r["direction_accuracy"] is not None for r in group)
        aggregate[name]["record_sections_with_error_over_20ms"] = sum(r["error_episodes_over_20ms"] > 0 for r in group)
    result = {"n_verified_records": 0 if preliminary else len(files), "n_records": len(files),
              "reference_status": "automatic_provisional" if preliminary else "expert_verified", "record_section_metrics": rows,
              "author_recheck_pending": AUTHOR_RECHECK_PENDING if not preliminary else True,
              "reference_root": str(evaluation_root), "references": references, "record_section_inventory": inventory,
              "state_order": ["INVALID", "REV", "FWD"],
              "macro_by_record_section": aggregate,
              "warning": "Первичная оценка. Отдельные задержки, ранний контекст и типы КЗ анализируются перед выводами."}
    filename = "pdr_rtds_evaluation_preliminary.json" if preliminary else "pdr_rtds_evaluation.json"
    _atomic_write_json(output or OUTPUT_ROOT.parent / filename, result)
    return result


def run_manual() -> None:
    # F5: дополнить копии и сразу собрать статистику; CLI сохраняет прежние команды.
    from scripts.phase5_experiments.evaluate_pdr_expert_holdout import MANUAL_MODEL_RUNS, MANUAL_CHECKPOINT_KINDS
    MODEL_RUNS = MANUAL_MODEL_RUNS  # Общий редактируемый список папок моделей в evaluate_pdr_expert_holdout.py.
    CHECKPOINT_KINDS = MANUAL_CHECKPOINT_KINDS  # latest/best; не подменяются автоматически.
    ACTION = "augment_models"  # augment_models | evaluate_models | original_export | evaluate_original
    LABELS_ROOT = VERIFIED_ROOT  # Авторские файлы: читаются, не изменяются.
    AUGMENTED_ROOT = PROJECT_ROOT / "data/phase5/pdr_rtds_review_h123"  # Отдельные копии со всеми прежними ответами.
    SKIP_EXISTING = True  # Совместимые законченные пары не рассчитываются заново.
    MAX_FILES = None  # None: все 100; 1: проба, не итоговая оценка.
    if ACTION == "augment_models":
        augment_h123(LABELS_ROOT, AUGMENTED_ROOT, max_records=MAX_FILES, skip_existing=SKIP_EXISTING,
                     model_runs=MODEL_RUNS, checkpoint_kinds=CHECKPOINT_KINDS)
    if ACTION in ("augment_models", "evaluate_models"):
        evaluate_verified(root=AUGMENTED_ROOT, output=PROJECT_ROOT / "data/phase5/pdr_rtds_evaluation_h123.json")
    elif ACTION == "original_export":
        run(prepare_only=PREPARE_ONLY, max_records=MAX_RECORDS)
    elif ACTION == "evaluate_original":
        evaluate_verified()
    else:
        raise ValueError("Неизвестный ACTION")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        run_manual()
        raise SystemExit(0)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true", default=PREPARE_ONLY)
    parser.add_argument("--max-records", type=int, default=MAX_RECORDS)
    parser.add_argument("--evaluate-verified", action="store_true", help="Первичная статистика только папки проверенных")
    parser.add_argument("--evaluate-preliminary", action="store_true", help="Согласие с автоматическими экспертными каналами, НЕ ручная проверка")
    args = parser.parse_args()
    if args.evaluate_verified or args.evaluate_preliminary:
        evaluate_verified(preliminary=args.evaluate_preliminary)
    else:
        run(prepare_only=args.prepare_only, max_records=args.max_records)
