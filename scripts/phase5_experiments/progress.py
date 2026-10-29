"""Минимальный progress reporter без внешних зависимостей."""

from __future__ import annotations

import sys
import time
import json
from pathlib import Path


class ProgressReporter:
    """Показывать детерминированный прогресс длительного локального сценария."""

    def __init__(
        self,
        label: str,
        total: int,
        unit: str = "элем.",
        initial_completed: int = 0,
    ) -> None:
        if total <= 0:
            raise ValueError("Общий объём progress должен быть положительным")
        self.label = label
        self.total = total
        self.unit = unit
        self._initial_completed = min(max(initial_completed, 0), total)
        self._last_percent = -1
        self._started = time.monotonic()

    def update(self, completed: int) -> None:
        """Обновить индикатор, не печатая лишние строки."""

        bounded = min(max(completed, 0), self.total)
        percent = int(100 * bounded / self.total)
        if percent == self._last_percent:
            return
        elapsed = max(time.monotonic() - self._started, 1e-9)
        completed_this_run = max(0, bounded - self._initial_completed)
        rate = completed_this_run / elapsed
        remaining = (self.total - bounded) / rate if rate else 0.0
        print(
            f"\r{self.label}: {percent:3d}% ({bounded:,}/{self.total:,}; "
            f"{rate:.2f} {self.unit}/с; прошло {elapsed:.0f} с; ETA {remaining:.0f} с)",
            end="", file=sys.stderr, flush=True,
        )
        self._last_percent = percent

    def finish(self) -> None:
        """Завершить строку индикатора."""

        self.update(self.total)
        print(file=sys.stderr, flush=True)

    def snapshot(self, completed: int) -> dict[str, float | int | str]:
        """Вернуть машинно-читаемые rate/ETA для progress.json."""

        bounded = min(max(completed, 0), self.total)
        elapsed = max(time.monotonic() - self._started, 1e-9)
        completed_this_run = max(0, bounded - self._initial_completed)
        rate = completed_this_run / elapsed
        return {
            "label": self.label,
            "completed": bounded,
            "total": self.total,
            "percent": 100.0 * bounded / self.total,
            "elapsed_seconds": elapsed,
            "records_per_second": rate,
            "eta_seconds": (self.total - bounded) / rate if rate else 0.0,
        }


class TrainingProgress:
    """Compact training display; detailed records remain in JSONL.

    TTY: overwrite one line. IDE output panes/files: one line per 30 seconds.
    No dependency on notebooks or a particular terminal library.
    """
    def __init__(self, output: Path, interval=2.0, stream=None):
        self.output=Path(output)
        self.stream=stream if stream is not None else sys.stderr
        self.tty=bool(getattr(self.stream,"isatty",lambda:False)())
        self.interval=max(interval,0.1) if self.tty else max(interval,30.)
        self.last=0.; self.ema=None; self.width=0

    def begin(self,label,total,initial=0):
        self.label,self.total,self.initial=label,total,initial
        self.started=time.monotonic(); self.last=0.; self.ema=None

    def message(self,text):
        if self.width:
            print(file=self.stream); self.width=0
        print(text,file=self.stream,flush=True)

    def update(self,completed,metrics=None,force=False,status="running"):
        metrics=metrics or {}
        if "loss" in metrics:
            self.ema=metrics["loss"] if self.ema is None else .95*self.ema+.05*metrics["loss"]
        now=time.monotonic()
        if not force and now-self.last<self.interval: return
        elapsed=max(now-self.started,1e-9)
        rate=max(0,completed-self.initial)/elapsed
        eta=(self.total-completed)/rate if rate else None
        pct=100*completed/max(1,self.total)
        filled=min(20,int(pct/5))
        text=f"{self.label} [{'#'*filled}{'-'*(20-filled)}] {pct:5.1f}% {completed:,}/{self.total:,}"
        if self.ema is not None: text+=f" | loss~ {self.ema:.6f}"
        if "lr" in metrics: text+=f" | lr {metrics['lr']:.2e}"
        if "grad_norm" in metrics: text+=f" | grad {metrics['grad_norm']:.3g}"
        if "step" in metrics: text+=f" | step {metrics['step']:,}"
        text+=f" | {rate:.1f} окон/с | ETA {eta/60:.1f} мин" if eta is not None else " | ETA —"
        if self.tty:
            print('\r'+text.ljust(self.width),end='',file=self.stream,flush=True)
            self.width=len(text)
        else:
            print(text,file=self.stream,flush=True)
        self.last=now
        payload={"status":status,"label":self.label,"completed":completed,"total":self.total,
                 "percent":pct,"elapsed_seconds":elapsed,"eta_seconds":eta,"loss_ema":self.ema,**metrics}
        temporary=self.output/'progress.tmp'
        temporary.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
        temporary.replace(self.output/'progress.json')


def export_hierarchical_curves(output: Path):
    """Deduplicate restarted steps; write small summary and human-readable PNG."""
    output=Path(output)
    steps,validations={},{}
    for line in (output/'training_log.jsonl').read_text(encoding='utf-8').splitlines():
        if not line.strip(): continue
        try: row=json.loads(line)
        except json.JSONDecodeError: continue  # interrupted final log line
        if row['kind']=='train': steps[row['step']]=row
        elif row['kind']=='validation': validations[row['epoch']]=row
    rows=sorted(steps.values(),key=lambda r:r['step'])
    vals=sorted(validations.values(),key=lambda r:r['epoch'])
    summary={"validation":vals,"epochs":[]}
    for epoch in sorted({r['epoch'] for r in rows}):
        subset=[r for r in rows if r['epoch']==epoch]
        targets=sum(r['targets'] for r in subset)
        summary['epochs'].append({"epoch":epoch,"train_weighted":sum(r['loss']*r['targets'] for r in subset)/max(1,targets),
                                  "steps":len(subset),"cursor":max(r['cursor'] for r in subset)})
    (output/'training_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    try:
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
    except ImportError: return
    fig=Figure(figsize=(12,8)); FigureCanvasAgg(fig)
    ax=fig.subplots(2,2)
    smooth=[]; ema=None
    for row in rows:
        ema=row['loss'] if ema is None else .99*ema+.01*row['loss']; smooth.append(ema)
    ax[0,0].plot([r['step'] for r in rows],smooth,label='Train EMA (0.99)')
    ax[0,0].set(xlabel='Optimizer step',ylabel='Combined vector loss',yscale='log')
    for key,label in [('all','All'),('open_ee','Open_EE'),('french_rte','French/RTE')]:
        selected=[r for r in vals if key in r['metrics']]
        ax[0,1].plot([r['epoch']+1 for r in selected],[r['metrics'][key] for r in selected],marker='o',label=label)
    ax[0,1].set(xlabel='Completed epoch',ylabel='Fixed validation loss',yscale='log')
    ax[1,0].plot([r['step'] for r in rows],[r['lr'] for r in rows],label='Learning rate')
    ax[1,0].set(xlabel='Optimizer step',ylabel='LR')
    ax[1,1].plot([r['step'] for r in rows],[r['grad_norm'] for r in rows],lw=.5,label='Gradient norm before clipping')
    ax[1,1].set(xlabel='Optimizer step',ylabel='Gradient norm',yscale='log')
    for axis in ax.flat: axis.grid(alpha=.25); axis.legend()
    fig.suptitle('PhasorHierarchicalKAN: training and fixed validation')
    fig.tight_layout()
    fig.savefig(output/'training_curves.png',dpi=140)
