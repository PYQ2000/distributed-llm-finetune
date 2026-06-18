from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # 仓库根可导入 dft

# 减少显存碎片化导致的 OOM（必须在 import torch 之前设置才生效）
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
    set_seed,
)

from dft.config import TrainConfig
from dft.data import load_sft_dataset
from dft.callbacks import ThroughputMemoryCallback


def build_tiny_model(tokenizer):
    """CPU 冒烟用的极小 Qwen2，几秒可跑，验证全链路 wiring。"""
    from transformers import Qwen2Config, Qwen2ForCausalLM

    cfg = Qwen2Config(
        vocab_size=max(len(tokenizer), 1000),
        hidden_size=128, intermediate_size=256,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=2048,
    )
    return Qwen2ForCausalLM(cfg)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--smoke", action="store_true", help="CPU tiny-model 冒烟")
    ap.add_argument("--resume_from_checkpoint", default=None)
    ap.add_argument("--deepspeed", default=None, help="DeepSpeed json 路径")
    ap.add_argument("--fsdp", default=None, help='如 "full_shard auto_wrap"')
    ap.add_argument("--fsdp_config", default=None, help="FSDP json 路径")
    for k in ["model_name", "output_dir", "run_name", "save_strategy"]:
        ap.add_argument(f"--{k}", default=None)
    ap.add_argument("--max_steps", type=int, default=None)
    ap.add_argument("--max_samples", type=int, default=None)
    return ap.parse_args()


def main():
    args = parse_args()
    overrides = {k: getattr(args, k) for k in
                 ["model_name", "output_dir", "run_name", "save_strategy",
                  "max_steps", "max_samples"]}
    cfg = TrainConfig.from_yaml(args.config, overrides)
    set_seed(cfg.seed)

    if args.smoke:
        cfg.report_to = "none"
        cfg.max_samples = 16
        cfg.max_steps = 2
        cfg.save_strategy = "steps"   # 冒烟时验证存盘路径
        cfg.save_steps = 2
        cfg.fp16 = False
        cfg.bf16 = False

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 先构建 TrainingArguments，再 load 模型：这样 DeepSpeed ZeRO-3 的 zero.Init
    # 才能在 from_pretrained 期间生效，避免每个 rank 先在内存里实体化整个模型。
    extra = {}
    if args.deepspeed:
        extra["deepspeed"] = args.deepspeed
    if args.fsdp:
        extra["fsdp"] = args.fsdp
    if args.fsdp_config:
        extra["fsdp_config"] = args.fsdp_config
    if cfg.gradient_checkpointing:
        # use_reentrant=False 更稳；交给 Trainer 统一 enable，避免重复开启
        extra["gradient_checkpointing_kwargs"] = {"use_reentrant": False}

    if cfg.report_to == "wandb":
        os.environ.setdefault("WANDB_PROJECT", cfg.wandb_project)

    targs = TrainingArguments(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.num_train_epochs,
        max_steps=cfg.max_steps,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        optim=cfg.optim,
        logging_steps=cfg.logging_steps,
        logging_first_step=True,
        save_strategy=cfg.save_strategy,
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        fp16=cfg.fp16,
        bf16=cfg.bf16,
        gradient_checkpointing=cfg.gradient_checkpointing,
        report_to=cfg.report_to,
        run_name=cfg.run_name,
        seed=cfg.seed,
        ddp_find_unused_parameters=False,
        **extra,
    )

    if args.smoke:
        model = build_tiny_model(tokenizer)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name, trust_remote_code=True)
    # 梯度检查点由 TrainingArguments(gradient_checkpointing=...) 统一开启，
    # Trainer 会自动处理 use_cache，无需手动 enable（避免重复开启）。

    train_ds = load_sft_dataset(cfg, tokenizer)
    collator = DataCollatorForSeq2Seq(
        tokenizer, padding=True, label_pad_token_id=-100)

    trainer = Trainer(
        model=model, args=targs, train_dataset=train_ds,
        data_collator=collator,
        callbacks=[ThroughputMemoryCallback(cfg.max_seq_len)],
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    # 仅在显式开启保存时才存最终模型（吞吐/对比实验不存，省 Kaggle 磁盘）。
    if cfg.save_strategy != "no":
        trainer.save_model(cfg.output_dir)
        tokenizer.save_pretrained(cfg.output_dir)


if __name__ == "__main__":
    main()
