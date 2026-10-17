"""Инженерная оценка плотных решений РНМ; не изменяет выход органа.

Состояния: -1 — неприменимо, 0 — REVERSE, 1 — FORWARD.
Каждый отсчёт представляет интервал длиной 1/fs. Неполученные расчёты
не разрешается подменять неприменимостью. Все длительности — миллисекунды.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class TemporalConfig:
    transition_ms: float = 5.0
    thresholds_ms: tuple[float, ...] = (5., 20., 50., 100.)
    match_windows_ms: tuple[float, ...] = (100., 50.)
    hold_ms: tuple[float, ...] = (0., 5., 20.)
    sliding_ms: float = 100.0

    def __post_init__(self) -> None:
        values = (self.transition_ms, self.sliding_ms, *self.thresholds_ms,
                  *self.match_windows_ms, *self.hold_ms)
        if not all(np.isfinite(v) and v >= 0 for v in values) or self.sliding_ms <= 0:
            raise ValueError("Некорректные временные параметры")
        if not self.match_windows_ms or min(self.match_windows_ms) <= 0:
            raise ValueError("Нужно положительное окно сопоставления")
        if not self.hold_ms or not self.thresholds_ms:
            raise ValueError("Нужны длительности удержания и диагностические пороги")


def _runs(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    edges = np.diff(np.r_[False, mask, False].astype(np.int8))
    return np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)


def _rate(numerator: float, denominator: float) -> float | None:
    return float(numerator / denominator) if denominator else None


def _errors(truth: np.ndarray, pred: np.ndarray, region: np.ndarray, dt: float,
            cfg: TemporalConfig) -> dict:
    masks = {
        "all": truth != pred,
        "false_forward": (truth == 0) & (pred == 1),
        "false_reverse": (truth == 1) & (pred == 0),
        "unnecessary_refusal": (truth >= 0) & (pred == -1),
        "refusal_on_reverse": (truth == 0) & (pred == -1),
        "refusal_on_forward": (truth == 1) & (pred == -1),
        "missed_invalid": (truth == -1) & (pred >= 0),
    }
    supports = {"all": region, "false_forward": region & (truth == 0),
                "false_reverse": region & (truth == 1),
                "unnecessary_refusal": region & (truth >= 0),
                "refusal_on_reverse": region & (truth == 0),
                "refusal_on_forward": region & (truth == 1),
                "missed_invalid": region & (truth == -1)}
    result = {}
    for name, mask in masks.items():
        mask = mask & region
        starts, stops = _runs(mask)
        lengths = (stops-starts)*dt
        error, support = float(mask.sum()*dt), float(supports[name].sum()*dt)
        result[name] = {"error_ms": error, "support_ms": support,
            "fraction": _rate(error, support), "max_episode_ms": float(lengths.max()) if len(lengths) else 0.,
            "episodes": len(lengths), "thresholds": {
                str(t): {"count": int((lengths > t+1e-9).sum()),
                         "total_ms": float(lengths[lengths > t+1e-9].sum())}
                for t in cfg.thresholds_ms}}
    return result


def _events(truth: np.ndarray, pred: np.ndarray, dt: float, cfg: TemporalConfig,
            startup_samples: int) -> list[dict]:
    # Одна предсказанная граница не может объяснять несколько экспертных.
    # Штраф несопоставления обеспечивает сначала максимум числа пар,
    # затем минимум суммы |сдвигов|; одинаковый тип имеет монотонное решение.
    te = np.flatnonzero(np.diff(truth)) + 1
    pe = np.flatnonzero(np.diff(pred)) + 1
    pe = pe[pe != startup_samples]  # Готовность Фурье — не восстановление VALID.
    tstop = np.r_[te[1:], len(truth)]
    all_pe = np.flatnonzero(np.diff(pred)) + 1
    pstop = np.searchsorted(all_pe, pe, side="right")
    pstop = np.r_[all_pe, len(pred)][pstop]
    output = []
    for window in cfg.match_windows_ms:
        for hold in cfg.hold_ms:
            for a, b in ((0, 1), (1, 0), (0, -1), (1, -1), (-1, 0), (-1, 1)):
                ti = np.flatnonzero((truth[te-1] == a) & (truth[te] == b))
                pi = np.flatnonzero((pred[pe-1] == a) & (pred[pe] == b))
                if not len(ti) and not len(pi):
                    continue
                t, p = te[ti], pe[pi]
                delay = (p[None, :] - t[:, None])*dt
                needed = max(1, int(np.ceil(hold/dt - 1e-10)))
                target_long = (tstop[ti] - t) >= needed
                # При опережении правильный ответ обязан дожить до границы
                # эксперта и удержаться после неё; короткий ранний импульс не успех.
                valid = ((np.abs(delay) <= window+1e-9) & target_long[:, None]
                         & (pstop[pi][None, :] >= np.maximum(p[None, :], t[:, None])+needed)
                         & (np.maximum(p[None, :], t[:, None])+needed <= tstop[ti, None]))
                penalty = (len(t)+1)*(window+1)
                costs = np.full((len(t), len(p)+len(t)), penalty)
                costs[:, :len(p)] = np.where(valid, np.abs(delay), penalty*3)
                rr, cc = linear_sum_assignment(costs)
                matched = {int(i): int(j) for i,j in zip(rr,cc) if j < len(p) and valid[i,j]}
                rows = []
                for i, pos in enumerate(t):
                    observed = (tstop[ti[i]]-pos)*dt
                    j = matched.get(i)
                    if j is not None:
                        status, lag = "matched", float(delay[i,j])
                    elif not target_long[i]:
                        status, lag = "short_reference", None
                    elif observed + 1e-9 < window+hold:
                        status = "next_reference_event" if tstop[ti[i]] < len(truth) else "censored"
                        lag = None
                    else:
                        status, lag = "missed", None
                    rows.append({"reference_ms": float(pos*dt), "status": status,
                        "delay_ms": lag, "confirmation_ms": max(lag, 0.)+hold if lag is not None else None,
                        "observable_after_ms": float(observed)})
                output.append({"from": a, "to": b, "window_ms": window, "hold_ms": hold,
                    "reference_events": len(t), "predicted_events": len(p),
                    "unmatched_predicted_events": len(p)-len(matched), "events": rows,
                    "by_deadline": {str(limit): {
                        "eligible": sum(x["observable_after_ms"]+1e-9 >= limit+hold for x in rows),
                        "responded": sum(x["observable_after_ms"]+1e-9 >= limit+hold and
                            x["status"] == "matched" and x["delay_ms"] <= limit+1e-9 for x in rows)}
                        for limit in cfg.thresholds_ms if limit <= window}})
    return output


def temporal_metrics(truth: np.ndarray, pred: np.ndarray, sample_rate: float, *,
                     config: TemporalConfig | None = None, startup_samples: int = 0) -> dict:
    """Плотная общая сетка. Маски режут интервалы, но не склеивают их концы."""
    cfg = config or TemporalConfig()
    truth, pred = np.asarray(truth), np.asarray(pred)
    if (truth.ndim != 1 or truth.shape != pred.shape or not len(truth)
            or not np.isin(truth, (-1, 0, 1)).all() or not np.isin(pred, (-1, 0, 1)).all()
            or not np.isfinite(sample_rate) or sample_rate <= 0):
        raise ValueError("Нужны одинаковые плотные массивы состояний -1/0/1 и положительная частота")
    if not 0 <= startup_samples <= len(truth):
        raise ValueError("Неверная длительность начального накопления")
    truth, pred = truth.astype(np.int8), pred.astype(np.int8)
    dt = 1000/sample_rate
    te = np.flatnonzero(np.diff(truth)) + 1
    transition = np.zeros(len(truth), dtype=bool)
    for i in te:
        transition[i:min(len(truth), i+int(np.floor(cfg.transition_ms/dt+1e-9))+1)] = True
    all_points = np.ones(len(truth), dtype=bool)
    startup = np.arange(len(truth)) < startup_samples
    regions = {"all": all_points, "transition": transition,
               "outside_transition": ~transition, "startup": startup}
    parts = {name: _errors(truth, pred, mask, dt, cfg) for name, mask in regions.items()}
    stable = ~transition & ~startup
    switches = np.flatnonzero(np.diff(pred)) + 1
    stable_switches = switches[stable[switches] & stable[switches-1]]
    edges = np.r_[0, switches, len(pred)]
    inner_durations = np.diff(edges)[1:-1]*dt  # Крайние обрезанные состояния исключены.
    width = int(np.ceil(cfg.sliding_ms/dt-1e-10))
    sums = np.r_[0, np.cumsum(truth != pred)]
    sliding = float((sums[width:] - sums[:-width]).max()*dt) if width <= len(truth) else None
    confusion = np.bincount((truth+1)*3+pred+1, minlength=9).reshape(3,3)*dt
    return {"schema": 1, "config": asdict(cfg), "duration_ms": len(truth)*dt,
        "sample_rate_hz": sample_rate, "startup_ms": startup_samples*dt,
        "confusion_ms": confusion.tolist(), "regions": parts,
        "max_error_in_sliding_window_ms": sliding,
        "stability": {"stable_support_ms": float(stable.sum()*dt),
            "state_switches": len(switches), "stable_switches": len(stable_switches),
            "stable_switches_per_second": _rate(len(stable_switches), stable.sum()*dt/1000),
            "stable_direction_switches": int(((pred[stable_switches]>=0)&(pred[stable_switches-1]>=0)).sum()),
            "stable_validity_switches": int(((pred[stable_switches]<0)|(pred[stable_switches-1]<0)).sum()),
            "short_states": {str(t): int((inner_durations < t-1e-9).sum()) for t in (5.,10.)}},
        "event_matching": _events(truth, pred, dt, cfg, startup_samples)}
