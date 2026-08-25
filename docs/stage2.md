# Stage2: Categorical Entropy Model Training

本文记录 MiniMax-H3 4096×24 tokenizer 对应的 Stage2 entropy model，包括原始
fixed-order baseline、当前 clean dynamic-rank 实验、progressive coding 评估，以及
围绕 attention context 和 teacher/inference gap 得到的设计结论。Stage2 只训练
categorical entropy Transformer；Stage1 tokenizer 全程冻结。

文中状态标签：

- **当前实现**：已经进入训练或 eval pipeline 的代码路径。
- **实验结论**：已有结构化 benchmark 支持的判断。
- **候选设计**：尚未训练验证，不能视为当前模型能力。

主要代码：

- 唯一正式训练入口：`scripts/stage2.py`
- 默认配置：`configs/stage2.yaml`
- entropy model：`entropy_model/model_categorical.py`
- block layout：`entropy_model/layout.py`
- Transformer：`entropy_model/transformer.py`

当前 clean dynamic-rank 平行实现：

- 训练入口：`scripts/stage2_cosine_dynamic_rank_clean.py`
- entropy model：`entropy_model/model_categorical_dynamic_rank_reliable.py`
- dynamic layout：`entropy_model/layout_dynamic_rank.py`
- full-rate eval：`scripts/evaluate_stage2_dynamic_rank_clean_distributed.py`
- progressive eval：`scripts/run_stage2_dynamic_rank_clean_progressive.py`
- progressive completion：
  `training/evaluation_progressive_groups_dynamic_rank_clean.py`

## 1. 输入与 Stage1 依赖

Stage2 必须接收一个完整 Stage1 checkpoint。当前实验链使用：

```text
results/minimax_h3_vq4096x24_s1_55k_s2_55k_20260822/
  stage1/checkpoints/step_0055000.pt
```

训练时 Stage1 model 被设为 `eval()` 和 `requires_grad_(false)`。每个视频仍使用与
Stage1 相同的 107×256×256 输入和 fast OpenCV data pipeline：

```text
video
→ frozen MiniMax-H3 encoder
→ frozen pre_quant
→ frozen raw-L2 EMA-VQ nearest assignment
→ quantized vectors [B,24,Tz,16,16]
  + hard indices [B,Tz,16,16]
```

这一段在 `no_grad` 和可选 BF16 autocast 下运行。Stage2 不训练 decoder，也不以
重建 loss 更新 tokenizer。

## 2. Four-group block layout

本节描述原始 fixed-order baseline；当前 dynamic-rank 只替换 group-to-rank mapping，
其余 block 结构保持一致。

entropy model 将每个 latent frame 的 16×16 index grid 划成 8×8 windows。每个
window 再按 `(y parity, x parity)` 分为 4 个颜色组：

```text
group 0: even y, even x
group 1: even y, odd x
group 2: odd y,  even x
group 3: odd y,  odd x
```

每组有 8×8 / 4 = 16 个位置。序列顺序是 time-major：

```text
(t0,g0) → (t0,g1) → (t0,g2) → (t0,g3)
→ (t1,g0) → ... → (t_last,g3)
```

不同 8×8 spatial window 被折叠到 batch 维并行处理。同一 group 的 16 个 index
并行预测；当前 group 可以注意到自身 group 的输入 token 以及所有更早 group，
但输入整体右移一组，因此不会看到当前目标 index。

## 3. Entropy Transformer 结构

| 组件 | setting |
| --- | ---: |
| codebook size | 4096 |
| code vector dimension | 24 |
| window size | 8 |
| groups / latent frame | 4 |
| sites / group | 16 |
| d_model | 384 |
| Transformer layers | 8 |
| attention heads | 6 |
| head dimension | 64 |
| FFN | SwiGLU, hidden dim 1024 |
| normalization | pre-RMSNorm + output RMSNorm |
| attention | block-causal SDPA, dropout 0 |
| maximum latent frames | 32 |
| output classes / site | 4096 |

输入由以下部分相加：

```text
Linear(previous-group code vector, 24 → 384) 或首组 BOS
+ target time/group/local-y/local-x embedding
+ source time/group/local-y/local-x embedding
```

经过 8 层 Transformer 和 output RMSNorm 后，linear head 输出每个位置的
4096-way logits。output head 以全零权重初始化，因此初始预测为均匀分布。

## 4. Forward、模型输出与 loss

```text
quantized vectors + hard indices
→ 8×8 windows
→ four-color groups in time-major order
→ shifted code projection + position embeddings
→ 8-layer block-causal Transformer
→ 4096-way logits
→ cross entropy in bits
```

每个 index 的 bit cost：

```text
bits_i = cross_entropy(logits_i, hard_index_i) / ln(2)
```

训练 loss 是所有 index 的平均 bit cost：

```text
L_stage2 = bits_per_index = Σ bits_i / index_count
```

结构化训练指标中的 estimated BPP 为：

```text
bpp = Σ estimated_bits / (B × T × H × W)
```

分母是原始视频的像素位置数，不包含 channel 数。

## 5. 当前代码默认训练 setting

`configs/stage2.yaml`：

| 项目 | 值 |
| --- | ---: |
| optimizer | AdamW |
| initial learning rate | 5e-5 |
| eta_min | 1e-6 |
| scheduler | cosine annealing，step 1 到 60000 |
| betas | (0.9, 0.97) |
| weight decay | 0.05 |
| gradient clip norm | 1.0 |
| batch size / GPU | 2 |
| max steps | 60000 |
| BF16 | true |
| seed | 2026 |
| log interval | 20 |
| checkpoint/eval interval | 5000 |
| checkpoint selection | UVG best estimated BPP |

启动模板：

```bash
torchrun --standalone --nproc_per_node=<gpu_count> scripts/stage2.py \
  -c configs/stage2.yaml \
  --dataset-root <absolute_dataset_root> \
  --metadata <absolute_metadata_dir> \
  --uvg-root <absolute_uvg_root> \
  --mcl-jcv-root <absolute_mcl_jcv_root> \
  --hevc-b-root <absolute_hevc_b_root> \
  --minimax-h3-model-dir <absolute_minimax_h3_dir> \
  --tokenizer-checkpoint results/<run>/stage1/checkpoints/<checkpoint>.pt \
  --output-dir outputs/<stage2_run>
```

GPU 数由启动命令决定。global batch size = batch size per GPU × world size。

## 6. 当前归档模型的实际 setting

当前 Stage3 使用的 entropy checkpoint 为：

```text
results/minimax_h3_vq4096x24_s1_55k_s2_55k_20260822/
  stage2/checkpoints/step_0055000.pt
```

该 checkpoint 自带的实际 metadata：

| 项目 | 实际值 |
| --- | ---: |
| checkpoint step | 55000 |
| world size | 8 |
| batch size / GPU | 2 |
| global batch size | 16 |
| max steps | 60000 |
| initial LR / eta_min | 5e-5 / 1e-6 |
| scheduler | cosine annealing |
| betas | (0.9, 0.97) |
| weight decay | 0.05 |
| BF16 | true |
| checkpoint policy | UVG best estimated BPP |

当前 canonical 默认配置与归档模型一致，均为每卡 batch size 2；8 卡时 global
batch size 为 16。

## 7. Resume、输出与周期 eval

Stage2 checkpoint version 2 保存：model、optimizer、scheduler、当前 step、epoch、
batch position、world size、global batch size，以及每个 rank 的 RNG state。
`--resume` 要求 Stage1 checkpoint、配置、world size 和 global batch size 与原 run
一致，然后从精确数据位置继续。跨机器移动同一份 Stage1 checkpoint 时必须显式增加
`--allow-checkpoint-relocation`，否则 source checkpoint 绝对路径不一致会直接失败。

rank 0 输出 `run_manifest.json`、`train_metrics.jsonl`、`training.log`、checkpoint
和每 5000 step 的 eval。训练指标包含 `bpp`、`bits_per_index`、`latency_ms`、
gradient norm 和 learning rate。

周期 eval 使用全部训练 rank 对 UVG、MCL-JCV、HEVC-B 做 Gaussian distributed
eval，输出结构化的 `psnr`、`bpp`、`latency_ms`、LPIPS-VGG、LPIPS-Alex 和
DISTS。Stage2 不保存 recon。全部 shard 成功后才按 UVG estimated BPP 更新 best
checkpoint 并恢复训练。


## 8. 设计目标与不可破坏的约束

Stage2 的首要任务仍然是为完整 bitstream 提供准确的 categorical probability；
progressive completion 是在同一个 entropy model 上测试的附加能力。任何改进必须同时满足：

1. K=4/full-rate 的 estimated BPP 不劣于 fixed-order baseline，最好进一步降低。
2. K=1/2/3 截断后，未传输 ranks 的 zero-shot completion 质量提高。
3. attention dependency 必须对应合法的 encoder/decoder 解码拓扑，不能读取尚未解码的
   target index。
4. 同一帧的 spatial windows 不能串行化；每帧仍然只允许 4 个 rank steps。
5. 时间因果性不能破坏：frame t 只能读取 frame ≤ t 的合法 context。
6. Stage1 tokenizer、codebook、encoder 和 decoder 保持不变。
7. 完整码率和 progressive benchmark 必须使用同一 tokenizer、数据协议和 metric 实现。

当前研究只修改 entropy model 的 layout、context scheduler、attention mask 或训练 context。
不以 tokenizer 改动掩盖 entropy model 的变化。

## 9. 当前真实 attention 拓扑

### 9.1 Window 是独立时序样本

**当前实现。** `layout.to_blocks` 将 `[B,Tz,Hz,Wz,C]` 变为：

```text
[B × windows_y × windows_x, Tz × 4, sites_per_rank, C]
```

因此每个 8×8 latent window 是一条独立 Transformer 序列；不同 windows 被折叠到
batch 维，彼此没有 attention。对于第 32 个 latent frame 中的 window `w_i`：

```text
可以读取：frame 1...31 中 w_i 的全部 ranks/groups
不能读取：frame 1...31 中其他 window w_j 的任何 rank/group
```

也就是说，当前模型不是“frame t 访问此前所有 frames 的所有 windows”，而是
“frame t 的每个 window 只访问同一 window 的历史”。dynamic rank 不改变这个边界。

### 9.2 Block-causal mask

每条 window 序列的 block id 为：

```text
block_id = 4 × time + rank
```

mask 为：

```text
allow(query, key) ⇔ key_block ≤ query_block
```

同一 block 内的 sites 共享一个 coding step。输入整体右移一个 block，因此 query 虽然
可以读取当前 block 的全部 input positions，但这些 positions 存放的是前一个 block 的
code vectors，而不是当前 targets。由此得到：

- 每帧 4 个串行 rank steps。
- 每个 rank 的 16 个 targets 并行预测。
- 所有 spatial windows 并行预测。
- 当前 window 可读取自身全部历史 blocks。
- 当前实现没有跨 window entropy context。

## 10. Progressive coding 的严格定义

**当前实现。** 对每个 latent frame、每个 window，按 coding rank 保留前 K 个 ranks：

```text
K=1: 传输 rank 0，生成 ranks 1,2,3
K=2: 传输 ranks 0,1，生成 ranks 2,3
K=3: 传输 ranks 0,1,2，生成 rank 3
K=4: 传输全部 ranks，不生成
```

缺失 ranks 使用 deterministic argmax，生成 index 查询 frozen Stage1 codebook 后进入
后续 block context。当前 clean dynamic-rank 模型从未训练 `source_status=1`，因此有效
progressive evaluator 对保留和生成 context 都使用 `source_status=0`，严格匹配 clean
teacher-forcing 训练。把生成 context 标成未训练的 status 1 会引入额外 distribution
shift，相关输出已明确标记为 invalid，不能用于结论。

progressive BPP 为保留 indices 的 estimated cross-entropy rate：

```text
retained_bits = Σ -log₂ p(true retained index | current context)
bpp = retained_bits / source_video_pixel_count
```

它不是 arithmetic coder 实际写出的 bitstream 大小。K=4 必须同时满足：

1. completed indices 与原始 indices 完全一致。
2. retained bits 与 full-rate teacher-forcing estimated bits 对齐。
3. reconstruction metrics 与同一个 frozen tokenizer 的正常重建一致。

## 11. Dynamic-rank group scheduler

### 11.1 动机

原始固定顺序 `g0→g1→g2→g3` 让 K=2 永远保留相同的两个 physical parity groups，
形成固定方向的空间采样。该方向性会让某些纹理、边缘和运动方向在 prefix 中长期缺失。

当前 dynamic-rank 方案不改变 4 个 physical parity groups，也不改变 8×8 window。它只
根据 window 坐标和时间，为 physical group 分配确定性的 rank，从而让不同 window、
不同 frame 的 prefix 覆盖不同 parity pattern。

### 11.2 确定性映射

定义 physical parity：

```text
b = (b_y, b_x) = (y mod 2, x mod 2)
```

对 window `(w_y,w_x)` 和 latent frame `t`：

```text
p = ((w_y + t) mod 2, (w_x + floor(t / 2)) mod 2)
u = b XOR p
a = (w_y + w_x) mod 2
rank = 2 × u_a + u_(1-a)
```

encoder 和 decoder 由同一公式计算 `rank_to_group`，不传输额外 side information。
该映射具有以下性质：

- 每个 frame/window 的四个 physical groups 仍恰好映射到 ranks 0...3。
- 每帧仍只有 4 个串行 steps。
- 同一帧所有 windows 仍并行，不引入 window raster order。
- time-major temporal causality 保持不变。
- K=1/2/3 prefix 的 physical parity 随 time/window phase rotation。

### 11.3 训练设置、context 边界与 checkpoint

dynamic-rank run：

```text
outputs/stage2_dynamic_rank_clean_cosine_s1_55k_20260824
```

实际 setting：

| item | value |
| --- | ---: |
| initialization | Stage2 step 0 |
| frozen tokenizer | Stage1 step 55000 |
| GPUs | 4 |
| batch / GPU | 4 |
| global batch | 16 |
| objective | clean teacher-forcing categorical CE |
| group order | coordinate/time dynamic rank |
| KNN corruption | disabled |
| generic corruption | disabled |
| self-forcing training | disabled |
| scheduler | cosine annealing |
| maximum steps | 60000 |

训练后来在 step 31580 附近被中断；本文所有 30k 结果统一使用
`checkpoint_best.pt`，其结构化 eval 标记的 checkpoint step 为 30000。不能把 run
最后出现的训练 step 与本文评估 checkpoint 混用。

layout 将 `batch × windows_y × windows_x` 合并到 batch 维。每个 8×8 window 是一条
独立的 time-major 序列：

```text
t0:r0 → t0:r1 → t0:r2 → t0:r3 → t1:r0 → ...
```

因此，frame `t` 的某个 window 可以读取该 window 的历史 frames 和当前 frame 的更早
ranks，但不能读取其他 window 的历史。dynamic rank 只改变 physical group 到 rank 的
映射，不增加跨 window context。

类名中保留了 `Reliable`，但本次 clean forward 使用普通 dense block-causal mask；
`_reliability_attention_mask` 不在本轮 clean training 或有效 clean progressive eval 路径中。
不能仅根据类名声称当前模型已经启用 reliability mask。

## 12. Dynamic-rank 训练结果与结论

### 12.1 评估协议与归因边界

固定 group-prefix 对每个 frame/window 保留前 K 个 dynamic ranks：

```text
K=1: 传输 rank 0，argmax 生成 ranks 1,2,3
K=2: 传输 ranks 0,1，argmax 生成 ranks 2,3
K=3: 传输 ranks 0,1,2，argmax 生成 rank 3
K=4: 全部传输，不生成
```

生成 token 会进入后续 rank 和后续 frame 的 context，形成 closed-loop rollout。当前模型
只用 clean teacher forcing 训练，从未对这种 generated context 优化。K=4 是 full-rate
sanity check，必须与同一 frozen tokenizer 的正常重建完全一致。

有效结构化输出：

```text
results/entropy_model_prefix_coding/dynamic_rank_clean_step_0005000/summary.json
results/entropy_model_prefix_coding/dynamic_rank_clean_step_0030000/summary.json
```

这里没有同 tokenizer、同训练步数、同数据顺序的 fixed-order control。因此以下实验可以
说明 dynamic-rank 模型随训练变强，以及该模型能否支持 prefix rollout；不能单独证明
dynamic-rank scheduler 优于 fixed-order scheduler。

### 12.2 Step 30000 group-prefix 结果

UVG：

| K | bpp | PSNR | LPIPS-VGG | LPIPS-Alex | DISTS | generated top-1 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.004180 | 12.930 | 0.59567 | 0.57293 | 0.34776 | 1.30% |
| 2 | 0.006847 | 17.022 | 0.40095 | 0.34148 | 0.19459 | 3.96% |
| 3 | 0.009003 | 24.997 | 0.24119 | 0.16142 | 0.07732 | 8.14% |
| 4 | 0.011345 | 28.840 | 0.18704 | 0.11135 | 0.04739 | N/A |

MCL-JCV：

| K | bpp | PSNR | LPIPS-VGG | LPIPS-Alex | DISTS | generated top-1 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.004043 | 12.881 | 0.55938 | 0.54875 | 0.32831 | 4.18% |
| 2 | 0.006758 | 16.031 | 0.42132 | 0.36862 | 0.21327 | 6.21% |
| 3 | 0.009074 | 22.732 | 0.26976 | 0.18356 | 0.09940 | 9.99% |
| 4 | 0.011500 | 26.675 | 0.20333 | 0.11802 | 0.05817 | N/A |

HEVC-B：

| K | bpp | PSNR | LPIPS-VGG | LPIPS-Alex | DISTS | generated top-1 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.004283 | 12.418 | 0.64129 | 0.65573 | 0.32835 | 1.28% |
| 2 | 0.007137 | 15.609 | 0.47524 | 0.43721 | 0.19722 | 3.48% |
| 3 | 0.009654 | 22.726 | 0.28977 | 0.21240 | 0.08263 | 6.31% |
| 4 | 0.012275 | 26.230 | 0.21203 | 0.13509 | 0.04507 | N/A |

从 5k 到 30k，full-rate BPP 的变化为：

| dataset | 5k BPP | 30k BPP | relative change |
| --- | ---: | ---: | ---: |
| UVG | 0.012519 | 0.011345 | -9.38% |
| MCL-JCV | 0.012561 | 0.011500 | -8.45% |
| HEVC-B | 0.013445 | 0.012275 | -8.70% |

K=3 是训练收益最稳定的 prefix level：UVG 从 23.223 dB / 6.30% top-1 提升到
24.997 dB / 8.14%，MCL-JCV 从 21.822 dB / 7.59% 提升到 22.732 dB /
9.99%，HEVC-B 从 21.543 dB / 4.23% 提升到 22.726 dB / 6.31%。但 K=1/2
不是单调改善，例如 UVG K=1 和 HEVC-B K=1/2 的 PSNR 下降。继续 clean training
改善了 full-rate entropy coding 和较浅的 suffix generation，但没有解决长 rollout。

### 12.3 Temporal dynamic-window matched-rate eval

该实验没有训练新模型，而是在同一个 dynamic-rank 30k checkpoint 上改变 transmission
policy。第一帧全部传输；后续每帧计算当前真实 token 与上一 completed frame 的 codebook
distance，完整传输变化最大的 K 个 8×8 windows，其余 windows 的四个 ranks 全部生成。
固定 cardinality window mask 的 side bits 已计入 BPP。

结构化输出：

```text
results/entropy_model_prefix_coding/
  temporal_window_matched_rates_dynamic_rank_step_0030000/summary.json
results/entropy_model_prefix_coding/
  temporal_window_matched_rates_dynamic_rank_step_0030000/comparison_group_vs_temporal_window.json
```

相对同 checkpoint 的固定 group-prefix，`Δ` 表示 temporal-window 减去 group-prefix；
PSNR 越大越好，其余 distortion metrics 越小越好：

| retained | dataset | ΔBPP | ΔPSNR | ΔLPIPS-VGG | ΔLPIPS-Alex | ΔDISTS |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 25% | UVG | -0.000652 | +3.049 | -0.08603 | -0.08572 | -0.11538 |
| 25% | MCL-JCV | -0.000338 | +1.382 | -0.02926 | -0.03398 | -0.08312 |
| 25% | HEVC-B | -0.000427 | +1.497 | -0.04540 | -0.04544 | -0.10275 |
| 50% | UVG | -0.000543 | +2.106 | -0.02477 | -0.01781 | -0.05860 |
| 50% | MCL-JCV | -0.000046 | +1.347 | -0.02402 | -0.02410 | -0.07025 |
| 50% | HEVC-B | -0.000159 | +1.529 | -0.01672 | -0.00209 | -0.06102 |
| 75% | UVG | -0.000092 | -1.877 | +0.02220 | +0.03216 | +0.00041 |
| 75% | MCL-JCV | +0.000367 | -0.981 | +0.00556 | +0.01408 | -0.01781 |
| 75% | HEVC-B | +0.000200 | -1.616 | +0.02493 | +0.04328 | -0.00625 |

25% 和 50% 在三个数据集上都同时改善 BPP、PSNR、LPIPS 和 DISTS，说明“完整保存
高变化 windows、生成稳定 windows”的空间资源分配在低码率区间有效。75% 时固定
group-prefix 只需要在每个 window 生成最后一个 group；temporal-window 则让约 25% 的
windows 生成全部四组，错误集中成完整 8×8 区域，因此 PSNR 和 LPIPS 明显退化。

### 12.4 优点、缺点与是否 work

**优点：**

- rank rotation 是确定性的，不需要 side information。
- 不改变四个 rank steps，同 frame windows 仍然并行。
- K=1/2/3 不再永久缺失固定 parity pattern，减少固定方向的空间采样偏置。
- 不修改 tokenizer、Transformer 宽度或训练 objective，工程和计算成本低。
- 30k full-rate BPP 和 K=3 rollout 均显示模型已经学到有效的概率模型。

**缺点：**

- rank permutation 是手工坐标/时间相位函数，不依据内容、运动或模型置信度。
- physical group 的 rank 随 window/time 变化，模型要额外学习变化的条件分布；它可能
  为 prefix coverage 牺牲 full-rate entropy efficiency。
- 每个 window 仍然只能访问自身历史，没有利用其他 windows 的 temporal correlation。
- clean teacher forcing 与 closed-loop argmax 之间的 gap 没有被训练目标覆盖；生成
  top-1 只有约 1%...10%，长 suffix rollout 仍然很弱。
- temporal-window 的明显增益来自“传输哪些 windows”，不能归因于 rank scheduler。
- 缺少同 step fixed-order control，无法证明 dynamic rank 本身优于原始顺序。

**结论：部分 work。** dynamic rank 作为低成本、无 side-bit、保持并行的 prefix
scheduler 是合理且实现正确的；现有结果证明它能够训练、能够熵编码，也能支持
progressive rollout。但当前证据不足以把它声明为更强的 entropy model。明确 work 的是
同一模型上的 temporal dynamic-window policy 在 25%/50% 区间的 RD 改善；该 policy 在
75% 区间不 work。证明 dynamic rank 本身有效仍需相同 tokenizer、初始化、数据顺序和
30k 训练预算的 fixed-order ablation，并同时比较 full-rate BPP 与 K=1/2/3 RD。

## 13. KNN corruption 实验与失败原因

### 13.1 实际训练机制

已停止的 KNN 方案使用：

```text
KNN k = 8
batch size / GPU = 4
每个 batch 恰好 1 个 corrupted sample
retained K 随机取 1,2,3
corrupt loss weight = 0.25
```

对被截断 suffix 中的每个 ground-truth index，训练独立、均匀地从其 codebook KNN₈
中选择替换 index。target 始终保持 ground truth，模型学习在局部扰动 context 下预测
真实 categorical index。

### 13.2 为什么理论直觉没有转化为 accuracy

KNN corruption 只有在训练扰动分布覆盖实际 inference error distribution 时才应提高
closed-loop accuracy。当前实现至少存在四个 mismatch：

1. **错误中心不一致。** KNN 始终以 ground-truth token 为中心；实际错误来自模型
   `argmax(logits)` 的 confusion distribution，可能不在 ground-truth KNN₈ 内。
2. **距离定义不一致。** tokenizer codebook 的 L2 邻近表示 latent embedding 接近，
   不等于 entropy model 在给定 context 下容易混淆。
3. **独立扰动与闭环误差不一致。** KNN 对每个 site 独立采样；实际错误会改变后续
   group/frame context，具有空间、rank 和时间相关性，并产生 drift。
4. **oracle-centered suffix。** 训练中的每个 corrupted token 都重新靠近对应真值；
   inference 的后续 prediction 则以此前错误 prediction 为条件，不会自动回到真值附近。

此外，corrupt sample 的 loss 对全部 positions 求平均，包含尚未受 corrupted context
影响的 targets；再乘 0.25 后，真正针对错误 context 的有效梯度进一步稀释。
`source_status` 只能告诉模型“这个 context 是生成的”，不能补偿错误内容分布本身不匹配。

因此实验结论不是“所有 corruption training 都无效”，而是：

```text
independent ground-truth-centered codebook KNN corruption
≠ entropy-model closed-loop rollout errors
```

KNN 已从当前训练移除，不应继续仅通过增大 k 或 corruption probability 调参。

### 13.3 重新考虑 corruption 前必须先测量

第一项诊断应为：

```text
KNN₈ recall =
  实际错误 prediction 落入 KNN₈(ground truth) 的次数
  / 实际错误 prediction 总次数
```

同时保存：

- 实际错误相对 ground truth 的 codebook neighbor rank。
- KNN sample 与 rollout error 的 embedding-distance CDF。
- 每个 retained K、rank、temporal depth 的 confusion distribution。
- error mask 的空间和时间相关性。
- teacher-forced argmax error 与 closed-loop rollout error 的差异。

如果 KNN₈ recall 很低，当前 KNN 假设被直接证伪。如果 recall 较高，仍需验证独立
扰动是否能覆盖闭环相关性。

## 14. Teacher-forcing 与 inference gap 的当前立场

完整编码训练使用真实 previous groups；progressive eval 把 argmax indices 闭环送回
context。两者确实存在 teacher/self-forcing gap，但当前项目明确不采用 self-forcing
训练，因为逐 block rollout 会显著增加 Stage2 训练成本。

当前策略是：

1. 主训练始终保持 clean teacher-forcing CE，保护 full-rate probability calibration。
2. 先通过更好的 deterministic rank scheduler 降低 prefix 的初始预测难度。
3. 在任何新 corruption 设计前，先测量真实 rollout error distribution。
4. 若以后恢复 corruption，优先考虑离线 rollout error bank 或固定 confusion table，
   而不是训练时执行完整 self-rollout。
5. 任何 corrupted-context auxiliary loss 都必须与 clean CE 分开报告，并以 K=4 BPP
   非劣作为 checkpoint 接受条件。

离线 error-bank 也只是候选设计。它必须记录生成它的 checkpoint、dataset、K、rank、
temporal depth 和采样规则，避免把 stale error distribution 当成当前模型误差。

## 15. Attention mask 的分析与候选方向

### 15.1 Full-rate 路径

**当前结论。** 在 intervention 数据支持稀疏化前，full-rate 训练继续使用 dense
block-causal attention：

```text
allow key ⇔ key block ≤ query block
```

原因不是 dense mask 一定最优，而是 full-rate entropy 能力是首要约束。仅凭 attention
weight 小不能删除 dependency；softmax attention weight 不是条件互信息，也不直接等价于
删除该 key 后的 BPP 变化。

### 15.2 已完成 temporal mask pilot

旧 fixed-order 25k checkpoint 的单样本 intervention：

| mask | logical density | Δbpp vs dense |
| --- | ---: | ---: |
| dense causal | 100% | 0 |
| temporal radius 4 | 26.7% | +0.002484 |
| temporal radius 8 | 46.5% | +0.000099 |
| temporal radius 16 | 76.7% | +0.000016 |

attention peak 出现在若干 temporal lags，包括 `Δt=1,2,3,5,10,15...`。这个 pilot
说明 R=4 明显不足，R=16 在该样本上接近 dense；但它来自旧 fixed-order 单样本，不能
直接决定 dynamic-rank mask。

下一轮必须在 dynamic-rank checkpoint 上、多个 held-out clips 上比较：

```text
A. dense causal
B. R=16
C. R=8 + anchors at Δt=5,10,15,...
D. layerwise mask:
   layer 0 dense
   middle layers R=8/16 + anchors
   final layer dense
```

选择指标：`bits/index`、`Δbpp`、top-1 agreement、logical density 和 `latency_ms`。
普通 dense SDPA 加 bool mask 不会自动获得 sparse-kernel 加速，所以 logical density 与
真实 latency 必须分别报告。

### 15.3 Progressive reliability mask

**候选设计，当前 clean run 未启用。** progressive decoding 可区分 transmitted 和
generated context：

```text
允许：BOS
允许：当前 frame 中更早 ranks，包括同帧 generated ranks
允许：更早 frames 中 transmitted ranks
阻止：更早 frames 中 generated ranks
```

这样保留同一帧完成剩余 groups 所需的局部闭环 context，同时避免 generated error 跨帧
长期传播。encoder/decoder 可由 transmitted/generated 状态确定同一个 mask，不需要额外
side information。

该 mask 只能作为 progressive path 候选。full-rate path 没有 generated context，继续
使用 dense causal history。是否共享参数以及是否损害 K=4 BPP，必须实验验证。

### 15.4 跨 window 历史 context

当前最明显的 context 缺口是不同 windows 完全独立。一个不串行化同帧 windows 的候选
拓扑是：

```text
queries:
  frame t 的所有 windows，在同一个 rank step 并行

keys/values:
  frame 1...t-1 的所有 windows、所有已解码 ranks
  + frame t 当前 window 内更早 ranks

禁止：
  frame t 其他 windows 的当前或未解码 targets
```

该设计允许当前 window 读取历史帧其他 windows 的运动和纹理信息，但不会建立同一帧
window raster order，因此不会增加每帧串行 window steps。它不是当前实现；当前 layout
把 window 放在 batch 维，若要实现必须重新组织 KV 和记忆位置编码。

优先考虑局部历史 halo 或 motion-aligned neighboring windows，而不是立即使用全局历史，
因为全局 attention 的计算和显存增长很快。

## 16. Window size 的理论与工程权衡

当前 window size 8 指 latent grid 上的 8×8，不是输入像素的 8×8。由于当前 windows
完全独立，在模型容量无限、训练充分的理想条件下，更大 window 提供更多合法 context：

```text
H(index | larger context) ≤ H(index | smaller context)
```

因此理论最优 entropy 不会因 context 增大而变差。但固定容量 Transformer 的实际结果
不保证单调：更长 sequence 会增加优化难度、attention dilution、显存和延迟。

对当前实现，单个 window 的 sequence length 约为：

```text
sequence length = Tz × window_size²
```

全图 attention 计算量近似：

```text
cost ∝ Hz × Wz × Tz² × window_size²
```

相对 8×8：

| window | approximate attention cost | context |
| --- | ---: | --- |
| 4×4 | 0.25× | 更碎，预计 entropy 较弱 |
| 8×8 | 1× | 当前设置 |
| 16×16 | 4× | 更完整，训练和显存更贵 |

所以目前不应直接把 window 缩小来追求更强 entropy model。更合理的验证顺序是先保持
8×8 targets，增加历史 halo 或跨 window history；只有 context intervention 证明更大
范围确实降低 BPP 后，再训练 16×16 完整模型。

## 17. 实验接受标准与下一步顺序

每个新 Stage2 方案至少保存以下结构化指标：

| path | required metrics |
| --- | --- |
| full-rate | bpp, bits/index, latency_ms |
| progressive K=1/2/3/4 | bpp, PSNR, LPIPS-VGG, LPIPS-Alex, DISTS |
| prediction diagnostic | per-rank top-1, CE, generated depth |
| mask intervention | Δbpp, top-1 agreement, logical density, latency_ms |

接受一个改进需要同时满足：

1. 与 baseline 使用相同 Stage1 tokenizer 和同 step Stage2 checkpoint。
2. K=4/full-rate BPP 非劣；tolerance 必须在实验前声明。
3. K=1/2/3 至少两个 levels 在相近 BPP 下稳定改善，而不是只比较相同 K。
4. 改善必须在 UVG 和 MCL-JCV 上复现，不能只依赖单数据集。
5. K=4 progressive bits 与 full-rate teacher-forcing bits 对齐。
6. 所有 shard 完成后才生成 summary；不使用缺视频的 partial aggregate。

当前实验顺序固定为：

1. 以 dynamic-rank 30k 为已完成实验，不再用不同 step 的旧模型充当 fixed-order control。
2. 若需要确认 scheduler 贡献，训练同 tokenizer、初始化、数据顺序和 30k budget 的
   fixed-order baseline，同时比较 full-rate BPP 与 K=1/2/3 RD。
3. 将 temporal dynamic-window 记录为独立 transmission policy，不把其收益归因于
   dynamic-rank 训练。
4. 在 dynamic-rank 30k checkpoint 上重新统计 attention 和 mask intervention。
5. 比较 dense、R=16、R=8+anchors 和 layerwise mask。
6. 若历史跨 window context 有明确 Δbpp 收益，再实现 8×8 target + local history halo。
7. 在没有真实 rollout error 统计前，不恢复 KNN 或其他 corruption training。

更广的研究候选和文献记录保存在 `docs/entropy_model.md`；本文件中的当前状态和已验证
结论优先于其中尚未实施的 proposal。
