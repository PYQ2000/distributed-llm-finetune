# 在 Kaggle T4×2 上复现实验

## 0. 准备
1. Kaggle → New Notebook → Settings → Accelerator 选 **GPU T4 ×2**；Internet 打开。
2. 拿到 W&B API key（wandb.ai/authorize），存进 Kaggle Secrets，名 `WANDB_API_KEY`。

## 1. 环境
```bash
!git clone https://github.com/PYQ2000/distributed-llm-finetune.git
%cd distributed-llm-finetune
!pip install -q -r requirements.txt
import os
from kaggle_secrets import UserSecretsClient
os.environ["WANDB_API_KEY"] = UserSecretsClient().get_secret("WANDB_API_KEY")
```

## 2. 实验命令（每条跑完去 W&B 看曲线）
```bash
# E1 单卡基线 0.5B
!CUDA_VISIBLE_DEVICES=0 python train/sft.py --config configs/base.yaml \
    --run_name e1-single-0.5b --output_dir outputs/e1

# E2 双卡 DDP 0.5B
!bash scripts/run.sh --config configs/base.yaml \
    --run_name e2-ddp-0.5b --output_dir outputs/e2

# E3 ZeRO-2 0.5B
!bash scripts/run.sh --config configs/base.yaml --deepspeed configs/ds_zero2.json \
    --run_name e3-zero2-0.5b --output_dir outputs/e3

# E4 FSDP 0.5B
!bash scripts/run.sh --config configs/base.yaml \
    --fsdp "full_shard auto_wrap" --fsdp_config configs/fsdp_config.json \
    --run_name e4-fsdp-0.5b --output_dir outputs/e4

# E5 OOM 故事：1.5B 单卡全量微调 —— 预期 CUDA OOM，截图报错
!CUDA_VISIBLE_DEVICES=0 python train/sft.py --config configs/base_1.5b.yaml \
    --run_name e5-oom-1.5b --output_dir outputs/e5

# E6 救场：1.5B 双卡 ZeRO-3 + CPU offload —— 预期能跑起来
!bash scripts/run.sh --config configs/base_1.5b.yaml --deepspeed configs/ds_zero3.json \
    --run_name e6-zero3-1.5b --output_dir outputs/e6
```

## 3. 断点续传（E7）
```bash
# 后台监控显存（另起一个 cell 或 &）
!python scripts/monitor_gpu.py --out results/gpu_log.csv --interval 2 &
# 跑 E2 配置，中途手动 Interrupt/kill；再续训：
!bash scripts/resume_demo.sh outputs/e7-resume/checkpoint-50
# 画 GPU 曲线
!python scripts/plot_gpu_log.py --csv results/gpu_log.csv --out results/figures/gpu_util_mem.png
```

## 4. 要收集的产出
- 每个实验的 W&B run 链接（把 project 设为 **public**，复制分享链接）。
- E5 的 OOM 报错截图。
- E6 的 1.5B 训练 loss 曲线截图。
- E7 续训前后 loss 衔接截图。
- `results/figures/gpu_util_mem.png`。
- 把吞吐/显存数字填进 `results/throughput.md`。

> 红线：以上全部为真实跑出的数据/截图，禁止编造。
