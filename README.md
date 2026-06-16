# distributed-llm-finetune · 双卡分布式 LLM 微调实践

> 在双卡 **T4 (16G)** 上，用 🤗 Transformers `Trainer` + Accelerate，**一份训练脚本通过配置切换
> DeepSpeed ZeRO 与 PyTorch FSDP**，对 Qwen2.5 做指令微调（SFT）；对比单/双卡吞吐与显存占用，
> 演示「大模型单卡 OOM → 多卡分片放得下」的显存优化，并实现 checkpoint 断点续传。
> 全程用 **Weights & Biases** 公开追踪。

> 🚧 **状态：进行中（Phase 0 · 仓库脚手架）**。下方路线图实时反映进度。
> **所有指标/曲线/截图均为真实跑出后回填，仓库内不放占位假数据。**

---

## 为什么做这个

用一个**小而完整、可复现**的项目，把分布式训练里几件核心的事亲手跑一遍并讲清楚：

- **数据并行 (DDP)** 与 **ZeRO 显存切分**、**FSDP 全分片** 的原理与工程差异；
- 「单卡放不下 → 多卡分片放得下」这条**显存优化主线**（用 1.5B 全量微调制造真实的单卡 OOM）；
- 训练的**容错**：checkpoint 断点续传，进程中断后无缝续训；
- 训练全链路的**实验管理与可观测性**（W&B；GPU 显存/利用率监控）。

工程上刻意站在工业标准之上：训练用 `Trainer + Accelerate`，**后端（DeepSpeed / FSDP）完全由配置驱动，训练脚本本身不变**。

## 核心实验设计

| 实验 | 模型 | 卡 | 后端 | 看什么 |
|---|---|---|---|---|
| E1 单卡基线 | Qwen2.5-0.5B | 1×T4 | — | loss 下降、吞吐基线 |
| E2 双卡 DDP | 0.5B | 2×T4 | DDP | 吞吐提升、扩展效率（PCIe 无 NVLink 的通信开销） |
| E3 ZeRO-2 | 0.5B | 2×T4 | DeepSpeed | 显存 ↓ vs 吞吐 |
| E4 FSDP | 0.5B | 2×T4 | PyTorch FSDP | 与 ZeRO 对照（分片策略差异） |
| **E5 OOM** | **1.5B** | 1×T4 | full FT | **单卡显存溢出（复现 + 算账）** |
| **E6 分片救场** | **1.5B** | 2×T4 | ZeRO-3 + CPU offload | **放得下、能训（核心）** |
| E7 断点续传 | 0.5B | 2×T4 | 任一 | kill → 续训，loss 无缝衔接 |

> 全程**全量微调**（非 LoRA），才能让 ZeRO/FSDP 的显存切分产生真实意义。

## 技术栈

PyTorch · 🤗 Transformers / Accelerate · DeepSpeed (ZeRO-2/3) · PyTorch FSDP ·
Weights & Biases · 数据集 `alpaca-zh` · *(二期)* Prometheus + DCGM-exporter + Grafana

## 路线图

- [x] **Phase 0** 仓库脚手架
- [x] **Phase 1** 训练 harness 代码就绪（单卡 / DDP / ZeRO / FSDP 一份脚本切换）；Kaggle 实验与 W&B 曲线回填中
- [ ] **Phase 2** ZeRO vs FSDP 吞吐 / 显存对比
- [ ] **Phase 3** 1.5B 单卡 OOM → 双卡 ZeRO-3 救场 *(核心卖点)*
- [ ] **Phase 4** 断点续传容错演示
- [ ] **Phase 5** 训练期 GPU 显存 / 利用率监控
- [ ] **Phase 6** *(autodl)* Prometheus + DCGM-exporter + Grafana 监控看板
- [ ] **Phase 7** *(可选)* K8s + Volcano 提交训练 Job

> Kaggle T4×2 复现步骤见 [`notebooks/kaggle_run_guide.md`](notebooks/kaggle_run_guide.md)；W&B 公开看板链接待实验跑出后回填。

## 参考与致谢

配置与范式借鉴自以下优秀开源项目（具体处会在对应文件注明出处）：

- [huggingface/alignment-handbook](https://github.com/huggingface/alignment-handbook) — ZeRO / FSDP recipe 配置
- [microsoft/DeepSpeedExamples](https://github.com/microsoft/DeepSpeedExamples) — checkpoint 保存 / 续训
- [meta-llama/llama-cookbook](https://github.com/meta-llama/llama-cookbook) — FSDP `state_dict_type`
- [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) — 训练循环可读性参考
- [NVIDIA/dcgm-exporter](https://github.com/NVIDIA/dcgm-exporter) — GPU 监控栈
- [MiniMind](https://github.com/jingyaogong/minimind) — 小模型全流程与文档风格

## 许可

[MIT](LICENSE)
