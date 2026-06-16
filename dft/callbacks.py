from __future__ import annotations

import time

from transformers import TrainerCallback


def compute_throughput(num_samples: int, seq_len: int, elapsed_s: float):
    """返回 (samples/s, tokens/s)；elapsed<=0 时返回 (0,0) 避免除零。"""
    if elapsed_s <= 0:
        return 0.0, 0.0
    return num_samples / elapsed_s, num_samples * seq_len / elapsed_s


class ThroughputMemoryCallback(TrainerCallback):
    """每次 on_log 记录全局吞吐 + 峰值显存到 logs（W&B 自动接住）。"""

    def __init__(self, seq_len: int):
        self.seq_len = seq_len
        self._t0 = None

    def on_step_begin(self, args, state, control, **kwargs):
        if self._t0 is None:
            self._t0 = time.perf_counter()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None or self._t0 is None:
            return
        import torch

        now = time.perf_counter()
        elapsed = now - self._t0
        world = max(1, getattr(args, "world_size", 1))
        num_samples = (args.per_device_train_batch_size
                       * args.gradient_accumulation_steps
                       * world
                       * max(1, args.logging_steps))
        sps, tps = compute_throughput(num_samples, self.seq_len, elapsed)
        logs["throughput/samples_per_s"] = round(sps, 3)
        logs["throughput/tokens_per_s"] = round(tps, 1)
        if torch.cuda.is_available():
            logs["gpu/mem_peak_alloc_gb"] = round(
                torch.cuda.max_memory_allocated() / 1e9, 3)
            logs["gpu/mem_peak_reserved_gb"] = round(
                torch.cuda.max_memory_reserved() / 1e9, 3)
            torch.cuda.reset_peak_memory_stats()
        self._t0 = now
