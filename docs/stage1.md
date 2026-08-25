# Stage1: VQ Tokenizer Training

本文记录当前 MiniMax-H3 实验链的 Stage1 训练设定。唯一训练入口以
`scripts/stage1.py` 为准；当前下游 Stage2/Stage3 实际使用的 Stage1
模型是归档的 55k checkpoint。两者的学习率策略不同，见“当前归档模型”一节。

## 1. 训练目标与依赖

Stage1 将预训练视频 VAE 改造成离散 VQ tokenizer。它联合训练 VAE backbone、
量化前后 projection，并通过 EMA 更新 codebook。其输出是重建视频和每个 latent
位置的 4096-way 离散 index。

主要代码：

- 训练入口：`scripts/stage1.py`
- 默认配置：`configs/stage1.yaml`
- tokenizer 与 EMA-VQ：`tokenizer/stage1_vq.py`
- MiniMax-H3 wrapper：`tokenizer/stage_backbones.py`
- 数据入口：`training/stage_data_fast.py`
- 高吞吐 DataLoader：`training/distributed_runtime_fast.py`

## 2. 输入与数据管线

当前 MiniMax-H3 setting 的单个样本为 `[C,T,H,W] = [3,107,256,256]`：

1. 从 metadata 指向的 MP4 中随机选择连续 107 帧。
2. OpenCV seek 后逐帧解码，将短边用 bicubic resize 到 512。
3. 对整段视频使用同一个随机位置裁出 256×256。
4. BGR 转 RGB，并将 uint8 映射到 `[-1,1]`。
5. worker 直接生成裁剪后帧，避免保留完整分辨率视频序列。

数据路径必须按机器在训练命令中显式传入；完整绝对路径表见
`docs/train_eval_workflow.md`。数据加载协议为：

| 项目 | 值 |
| --- | --- |
| dataset root | `--dataset-root` 显式指定 |
| metadata | `--metadata` 显式指定 |
| dataset format | `trainable_jsons` |
| workers / rank | 4 |
| prefetch factor | 2 |
| pinned memory | true |
| persistent workers | true |

采样使用 `DistributedSampler(drop_last=true, shuffle=true)` 和可恢复的 batch
sampler。坏视频只对显式的 `VideoDecodeError` 执行确定性替换，最多尝试 16 个
候选，并写入 `<output-dir>/rejected_samples.jsonl`；其他数据或 shape 错误直接失败。

## 3. 模型结构

### 3.1 MiniMax-H3 VAE backbone

- 使用 `pretrained_ckpts/MiniMax-H3/FL2VA/video_vae`。
- backbone 参与训练，并启用内部 gradient checkpointing。
- 输入首先从 `[-1,1]` 转成 `[0,1]`，再经过 MiniMax processor 标准化。
- encoder 使用 posterior moments 的 mean，不采样 posterior variance。
- 空间压缩率为 16，时间压缩率为 4，latent channel 为 24。
- 对 256×256 输入，latent 空间网格为 16×16。
- decoder 固定按 MiniMax-H3 的 107 帧协议解码，再还原到 `[-1,1]`。

### 3.2 Projection 与 EMA-VQ

```text
VAE latent [B,24,Tz,16,16]
→ pre_quant: Conv3d(24 → 24, kernel=1)
→ raw-L2 nearest-neighbor EMA-VQ, codebook 4096×24
→ post_quant: Conv3d(24 → 24, kernel=1)
→ VAE decoder
```

两个 1×1×1 projection 均以 channel identity 初始化。量化距离在 FP32 下计算：

```text
index = argmin over k of ||projected - codebook[k]||²
quantized = projected + stop_gradient(selected - projected)
```

codebook 不是普通 optimizer parameter，而是全局 online k-means EMA buffer：

| 项目 | 值 |
| --- | ---: |
| codebook size | 4096 |
| code dimension | 24 |
| distance | raw L2 |
| EMA decay | 0.99 |
| initialization | 首个 global latent batch 上运行 10 次 Lloyd iteration |
| dead-code threshold | cluster size < 1.0 |
| revival interval | 每次 VQ update |
| revival samples / rank | 512 |
| reset cluster size | 1.0 |

DDP 下初始化中心、assignment count、embedding sum 和 dead-code replacement 都在
rank 间同步。

## 4. Forward 与 loss

完整 forward：

```text
video
→ MiniMax-H3 encode → posterior mean
→ pre_quant
→ nearest code index + straight-through quantized latent
→ post_quant
→ MiniMax-H3 decode
→ reconstruction, indices, commitment loss, codebook error
```

训练目标为：

```text
L_stage1 = 1.0 × MSE(reconstruction, video)
         + 1.0 × LPIPS-VGG(reconstruction, video)
         + 0.25 × MSE(projected, stop_gradient(selected))
```

LPIPS 按视频帧计算，每次送入 16 帧，并启用 activation checkpointing。
`codebook_error = MSE(selected, stop_gradient(projected))` 仅作为诊断指标；codebook
由 EMA 更新，不由该项反向传播。

训练 forward 使用 BF16 autocast；VQ assignment、MSE、PSNR 和主要 loss 聚合使用
FP32。训练日志中的固定长度近似 BPP 为：

```text
bpp = latent index 数 × 12 / 原视频像素数
```

其中 12 = log₂(4096)。这不是 Stage2 entropy model 的实际 estimated BPP。

## 5. 当前代码默认训练 setting

`configs/stage1.yaml`：

| 项目 | 值 |
| --- | ---: |
| optimizer | AdamW |
| initial learning rate | 1e-4 |
| eta_min | 5e-6 |
| scheduler | cosine annealing，step 1 到 60000 |
| betas | (0.9, 0.95) |
| weight decay | PyTorch AdamW 默认值 0.01 |
| gradient clip norm | 1.0 |
| batch size / GPU | 1 |
| max steps | 60000 |
| seed | 2026 |
| log interval | 20 |
| checkpoint/eval interval | 5000 |
| checkpoint selection | UVG best LPIPS-VGG |

GPU 数不写死，由 `torchrun --nproc_per_node=<gpu_count>` 决定；global batch size
等于 `batch size / GPU × world size`。

启动模板：

```bash
torchrun --standalone --nproc_per_node=<gpu_count> scripts/stage1.py \
  --tokenizer minimax_h3 \
  -c configs/stage1.yaml \
  --dataset-root <absolute_dataset_root> \
  --metadata <absolute_metadata_dir> \
  --uvg-root <absolute_uvg_root> \
  --mcl-jcv-root <absolute_mcl_jcv_root> \
  --hevc-b-root <absolute_hevc_b_root> \
  --minimax-h3-model-dir <absolute_minimax_h3_dir> \
  --output-dir outputs/<stage1_run>
```

恢复时增加 `--resume <checkpoint>`。version-3 checkpoint 精确恢复 model、optimizer、
数据 epoch/batch position 和各 rank RNG state，并校验 tokenizer、latent geometry、
world size、global batch size 与 scientific config。历史 version-2 checkpoint 仍可读，
但其数据位置只能由 step 重建。

## 6. 当前归档模型的实际 setting

当前实验链使用：

```text
results/minimax_h3_vq4096x24_s1_55k_s2_55k_20260822/
  stage1/checkpoints/step_0055000.pt
```

该 checkpoint 是历史 multistep Stage1 run，而不是当前 cosine 默认配置：

| 项目 | 实际值 |
| --- | ---: |
| step | 55000 |
| world size | 8（来自归档 run manifest） |
| batch size / GPU | 1 |
| global batch size | 8 |
| LR at step 0 | 1e-4 |
| LR at step 15000 | 5e-5 |
| LR at step 30000 | 1e-5 |
| LR at step 45000 | 5e-6 |
| checkpoint policy | best LPIPS-VGG |

模型结构、数据 shape、EMA-VQ 和 loss 与本文前述 setting 一致。新 Stage1 run
使用 cosine；复现当前 55k 归档权重时则必须保留这里的 multistep provenance。

## 7. 输出、checkpoint 与周期 eval

rank 0 写入：

- `run_manifest.json`：完整结构、数据、精度和训练配置；
- `train_metrics.jsonl`：`psnr`、`bpp`、`lpips`、`latency_ms`、loss、codebook
  active count/perplexity 和 gradient norm；
- `training.log`：按 `log_interval` 写入可读日志；
- `checkpoints/` 与 `evaluations/step_<step>/`。

每 5000 step 先保存 candidate checkpoint，再释放训练 GPU，使用全部训练 rank
执行 Gaussian distributed eval。聚合指标至少包含 `psnr`、`bpp` 和
`latency_ms`，并包含 LPIPS-VGG、LPIPS-Alex 与 DISTS。Stage1 不保留 recon；eval
全部成功后按 UVG LPIPS-VGG 更新 `checkpoint_best.pt` 并恢复训练。
