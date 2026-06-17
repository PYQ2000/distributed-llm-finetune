from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class TrainConfig:
    # 模型 / 数据
    model_name: str = "Qwen/Qwen2.5-0.5B"
    dataset_name: str = "llamafactory/alpaca_gpt4_zh"
    dataset_split: str = "train"
    max_samples: int = 3000
    max_seq_len: int = 1024
    # 训练
    output_dir: str = "outputs/run"
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2.0e-5
    num_train_epochs: float = 1.0
    max_steps: int = -1
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    logging_steps: int = 5
    save_strategy: str = "no"   # 默认不存(省盘)；E7 续传演示再 --save_strategy steps
    save_steps: int = 50
    save_total_limit: int = 2
    seed: int = 42
    fp16: bool = True
    bf16: bool = False
    gradient_checkpointing: bool = False
    # 追踪
    run_name: str = "run"
    wandb_project: str = "distributed-llm-finetune"
    report_to: str = "wandb"

    @classmethod
    def from_yaml(cls, path, overrides: Optional[dict] = None) -> "TrainConfig":
        data: dict = {}
        if path is not None:
            with open(Path(path), "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        if overrides:
            data.update({k: v for k, v in overrides.items() if v is not None})
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        return cls(**data)
