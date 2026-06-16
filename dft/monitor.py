from __future__ import annotations

GPU_FIELDS = [
    "timestamp", "index", "utilization_gpu",
    "memory_used", "memory_total", "temperature_gpu", "power_draw",
]


def parse_nvidia_smi_line(line: str) -> dict:
    parts = [p.strip() for p in line.split(",")]
    if len(parts) != len(GPU_FIELDS):
        raise ValueError(f"expected {len(GPU_FIELDS)} cols, got {len(parts)}: {line!r}")
    return dict(zip(GPU_FIELDS, parts))
