# 分布式训练原理速览（本项目用到的部分）

## 并行维度 DP / MP / PP
- **数据并行 (DP)**：每卡一份完整模型，喂不同数据，反向后 all-reduce 梯度求平均。
  通信量 ∝ 参数量。最简单，本项目 E2(DDP) 即此。
- **张量并行 (TP/MP)**：把单层权重矩阵按行/列切到多卡，前向需 all-gather/reduce。
  适合单层放不下的超大模型。本项目未用（小模型不需要）。
- **流水线并行 (PP)**：把层按段分到不同卡，micro-batch 流水以减少气泡。
  本项目未用。

## ZeRO（DeepSpeed）三阶段——在 DP 基础上消除显存冗余
DP 里每卡都存了完整的「优化器状态 + 梯度 + 参数」，三份冗余。ZeRO 把它们切片：
- **Stage 1**：切**优化器状态**（Adam 的 m,v，fp32）。
- **Stage 2**：再切**梯度**。（本项目 E3）
- **Stage 3**：再切**参数本身**——前向/反向时按需 all-gather 该层参数，用完释放。（本项目 E6）
- **CPU offload**：把优化器状态/参数进一步卸到 CPU 内存，拿时间换显存。E6 用它让 1.5B 在 16G 卡上放得下。

## FSDP（PyTorch 原生）
理念与 ZeRO-3 类似：`FULL_SHARD` 把参数/梯度/优化器状态全分片，按 transformer 层包裹
（`auto_wrap` + `Qwen2DecoderLayer`），前向时 all-gather 本层、算完释放。
与 ZeRO-3 的差异主要在实现与 checkpoint 形态（`state_dict_type` 决定存全量还是分片）。

## 为什么 1.5B 单卡会 OOM（E5 算账）
全量微调、Adam、fp32 主权重：
参数 1.5B × 4B(fp32) ≈ 6G；梯度 ≈ 6G；Adam m+v ≈ 12G；合计 ≈ 24G > 16G(T4)，
还没算激活值与 fp16 副本 → 必 OOM。ZeRO-3 把这 24G 切到 2 卡 + CPU，单卡占用骤降 → 放得下。

## T4 的两个细节
- Turing 架构**无 bf16**，只能 fp16 + loss scaling。
- 双卡走 **PCIe 无 NVLink**，通信带宽有限 → DDP 扩展效率达不到理想 2×，这正是 E2 要量化的点。
