from __future__ import annotations


def build_messages(example: dict) -> list[dict]:
    instruction = (example.get("instruction") or "").strip()
    inp = (example.get("input") or "").strip()
    user = instruction if not inp else f"{instruction}\n\n{inp}"
    return [
        {"role": "user", "content": user},
        {"role": "assistant", "content": (example.get("output") or "").strip()},
    ]


def tokenize_example(example: dict, tokenizer, max_seq_len: int) -> dict:
    messages = build_messages(example)
    prompt_ids = tokenizer.apply_chat_template(
        messages[:-1], add_generation_prompt=True, tokenize=True)
    full_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=False, tokenize=True)
    full_ids = list(full_ids)[:max_seq_len]
    labels = list(full_ids)
    prompt_len = min(len(prompt_ids), len(full_ids))
    for i in range(prompt_len):
        labels[i] = -100
    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
    }


def load_sft_dataset(cfg, tokenizer):
    """加载 alpaca_gpt4_zh 子集并 tokenize。需要联网（下载数据集）。"""
    from datasets import load_dataset

    ds = load_dataset(cfg.dataset_name, split=cfg.dataset_split)
    if cfg.max_samples and cfg.max_samples > 0:
        n = min(cfg.max_samples, len(ds))
        ds = ds.shuffle(seed=cfg.seed).select(range(n))
    ds = ds.map(
        lambda ex: tokenize_example(ex, tokenizer, cfg.max_seq_len),
        remove_columns=ds.column_names,
        desc="tokenizing",
    )
    return ds
