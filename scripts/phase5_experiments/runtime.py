"""Execution-only optimizations; model parameters and data order stay unchanged."""
from contextlib import contextmanager
from dataclasses import replace
import torch
import time


def timed_batches(batches):
    """Host time waiting for the next prepared batch, not GPU idle-time profiling."""
    iterator = iter(batches)
    while True:
        started = time.perf_counter()
        try:
            batch = next(iterator)
        except StopIteration:
            return
        yield batch, time.perf_counter()-started


@contextmanager
def activation_policy(model, tokens, no_checkpoint_max_tokens=2000):
    """Keep short-window activations; recompute long ones to bound GPU memory.

    Restore the immutable model config before checkpoint serialization. Disabling
    recomputation does not change forward operations or dropout RNG consumption.
    Zero retains the original always-checkpoint execution for comparisons.
    """
    original = model.cfg
    if original.checkpoint_blocks and tokens <= no_checkpoint_max_tokens:
        model.cfg = replace(original, checkpoint_blocks=False)
    try:
        yield
    finally:
        model.cfg = original


def scale_gradients(model, factor):
    """Same FP32 multiplication per gradient with batched CUDA kernel dispatch."""
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    if grads:
        torch._foreach_mul_(grads, factor)


def effective_batches(windows, target_tokens):
    """Collect exactly the old accumulation boundary without reordering windows."""
    pending, tokens = [], 0
    for window in windows:
        pending.append(window)
        # Input is CPU data; no CUDA scalar synchronization.
        tokens += int(window['valid'].sum())
        if tokens >= target_tokens:
            yield pending
            pending, tokens = [], 0
    if pending:
        yield pending


def window_packs(windows, max_windows=8, max_padded_tokens=4000):
    """Contiguous microbatches within one effective batch; never mix examples in time."""
    if max_windows < 1 or max_padded_tokens < 1:
        raise ValueError('Packing limits must be positive')
    pending, longest = [], 0
    for window in windows:
        length = window['z'].shape[1]
        next_longest = max(longest, length)
        if pending and (len(pending) >= max_windows or next_longest*(len(pending)+1) > max_padded_tokens):
            yield pending
            pending, longest = [], 0
        pending.append(window)
        longest = max(longest, length)
    if pending:
        yield pending


def pack_windows(windows):
    """Pad independent examples along time; selected/valid padding is always false."""
    if len(windows) == 1:
        return windows[0]
    length = max(w['z'].shape[1] for w in windows)
    packed = {}
    for key, value in windows[0].items():
        if isinstance(value, torch.Tensor):
            result = torch.zeros((len(windows), length)+value.shape[2:], dtype=value.dtype,
                                 pin_memory=value.is_pinned())
            for i, window in enumerate(windows):
                result[i, :window[key].shape[1]] = window[key][0]
            packed[key] = result
        else:
            packed[key] = [item for window in windows for item in window[key]]
    return packed


class TrainingMemoryError(RuntimeError):
    """Even a single checkpointed window cannot fit; no optimizer update occurred."""


def fit_execution_to_memory(max_windows=8, max_padded_tokens=4000,
                            no_checkpoint_max_tokens=4000, reserve_mib=1024, device=None):
    """Conservative calibrated bound; actual OOM still has a transactional fallback."""
    policy = dict(max_windows=max_windows, max_padded_tokens=max_padded_tokens,
                  no_checkpoint_max_tokens=no_checkpoint_max_tokens)
    if min(max_windows, max_padded_tokens) < 1 or min(no_checkpoint_max_tokens, reserve_mib) < 0:
        raise ValueError('Invalid execution/memory limits')
    if device is not None and device.type == 'cuda':
        free, _ = torch.cuda.mem_get_info(device)
        reusable = torch.cuda.memory_reserved(device)-torch.cuda.memory_allocated(device)
        # Measured ~0.65 MiB/padded token on this architecture; allow 0.75 plus reserve.
        available_mib = max(0, (free+reusable)/2**20-reserve_mib)
        capacity = int(available_mib/.75/80)*80
        policy['max_padded_tokens'] = max(80, min(max_padded_tokens, capacity))
        policy['no_checkpoint_max_tokens'] = max(0, min(no_checkpoint_max_tokens, capacity))
    return policy


def _backward_packs(model, windows, device, amp, policy, loss_fn):
    import math
    values, pack_count, padded_tokens = [], 0, 0
    for group in window_packs(windows, policy['max_windows'], policy['max_padded_tokens']):
        cpu_batch = pack_windows(group)
        batch = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                 for k, v in cpu_batch.items()}
        volume = batch['z'].shape[0]*batch['z'].shape[1]
        with activation_policy(model, volume, policy['no_checkpoint_max_tokens']):
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                total, _ = loss_fn(model, batch)
            (total/4096).backward()
        values.append(total.detach())
        pack_count += 1; padded_tokens += volume
    scalars = torch.stack(values).cpu().tolist()
    if not all(math.isfinite(v) for v in scalars):
        raise FloatingPointError('Non-finite loss; optimizer step was not executed')
    return dict(loss_sum=sum(scalars), packs=pack_count, padded_tokens=padded_tokens)


def backward_with_oom_retry(model, windows, device, amp, policy, loss_fn,
                            capture_rng, restore_rng, message=lambda text: None):
    """Retry the whole effective batch, never retain partially accumulated gradients.

    CPU windows and the pre-batch RNG survive retries. Parameters/optimizer are
    untouched here. Final failure restores RNG, allowing checkpoint of the exact
    pre-batch cursor. The caller must handle optimizer OOM separately: AdamW may
    already have modified parameters when it fails, so that state must NOT be saved.
    """
    import gc
    before = capture_rng()
    retries = 0
    while True:
        model.zero_grad(set_to_none=True)
        restore_rng(before)
        try:
            result = _backward_packs(model, windows, device, amp, policy, loss_fn)
            return dict(result, oom_retries=retries)
        except torch.OutOfMemoryError:
            pass  # Leave except before collecting: release traceback/failed graph.
        model.zero_grad(set_to_none=True)
        gc.collect()
        if device.type == 'cuda': torch.cuda.empty_cache()
        restore_rng(before)
        if policy['no_checkpoint_max_tokens'] > 0:
            policy['no_checkpoint_max_tokens'] = 0
        elif policy['max_windows'] > 1:
            policy['max_windows'] = max(1, policy['max_windows']//2)
        else:
            raise TrainingMemoryError('One window does not fit even with activation checkpointing')
        retries += 1
        message(f'Недостаточно VRAM: повтор того же шага; окон за проход={policy["max_windows"]}, '
                f'порог без пересчёта={policy["no_checkpoint_max_tokens"]}.')
