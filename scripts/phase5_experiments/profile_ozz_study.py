"""Profile OZZ encoders on the same feature tensors and context lengths."""

import gc
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from osc_tools.ml.models.ozz_study import build_encoder

VARIANTS = ("full", "mlp", "no_direction", "no_physics", "joint", "transformer", "patchtst")
LENGTHS = (80, 800, 12000)
REPEATS = 3
# Optional installed adapters, keyed by "timesnet" or "patchtst".
EXTERNAL_FACTORIES = {}


def profile(variant, length, device):
    model = build_encoder(variant, external_factory=EXTERNAL_FACTORIES.get(variant)).to(device).train()
    z = torch.randn(1, length, 13, 6, 2, device=device)
    state = torch.ones(z.shape[:-1], dtype=torch.long, device=device)
    times = torch.arange(length, device=device)[None] / 8
    valid = torch.ones(1, length, dtype=torch.bool, device=device)
    cuda = device.type == "cuda"
    durations = []
    for run in range(REPEATS + 1):
        model.zero_grad(set_to_none=True)
        if cuda:
            torch.cuda.synchronize()
            if run == 1:
                torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=cuda and torch.cuda.is_bf16_supported()):
            result = model(z, state, times, valid)
            loss = result["token_features"].float().square().mean()
        loss.backward()
        if cuda:
            torch.cuda.synchronize()
        if run:
            durations.append(time.perf_counter() - start)
    parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    peak_mib = torch.cuda.max_memory_allocated() / 2**20 if cuda else None
    return parameters, sorted(durations)[len(durations)//2], peak_mib


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for variant in dict.fromkeys((*VARIANTS, "timesnet", *EXTERNAL_FACTORIES)):
        for length in LENGTHS:
            try:
                count, seconds, peak_mib = profile(variant, length, device)
                print(f"{variant:16} {length:6} tokens  {count:10} params  {seconds:7.3f} s  peak MiB={peak_mib}")
            except ImportError as exc:
                print(exc)
                break
            except torch.OutOfMemoryError:
                print(f"{variant:16} {length:6} tokens  OOM")
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
