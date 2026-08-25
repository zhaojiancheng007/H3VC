# Stage3: Joint Rate-Distortion Training

Stage3 对 Stage1 VQ tokenizer 与 Stage2 categorical entropy model 做联合
rate-distortion 微调。唯一正式入口为 `scripts/stage3.py`；默认配置为
`configs/stage3.yaml`。

## 1. 初始化约束

Stage3 必须显式提供：

1. `--tokenizer-checkpoint`：Stage1 checkpoint；
2. `--entropy-checkpoint`：Stage2 checkpoint；
3. `-c/--config`：本 RD point 的 Stage3 config；
4. 数据、metadata、三个 eval 数据集和 MiniMax-H3 的绝对路径。

Stage2 必须由同一份 Stage1 checkpoint 训练，且 Stage2/Stage3 entropy model
architecture config 必须完全一致。checkpoint 内记录的 source path 与当前路径不同时
默认失败；只有确认是跨机器搬迁的同一份文件时才传入
`--allow-checkpoint-relocation`。

Stage1 tokenizer 和 Stage2 entropy model 的参数参与优化，4096×24 EMA codebook
保持冻结。

## 2. Forward

```text
video [B,3,107,256,256]
→ trainable MiniMax-H3 encoder
→ pre_quant → projected [B,24,Tz,16,16]
→ 对固定 4096×24 codebook 计算 FP32 squared L2 distance
→ hard nearest indices + selected code vectors
→ straight-through quantized latent
→ post_quant → trainable MiniMax-H3 decoder → reconstruction
→ entropy model forward_soft
```

hard path：

```text
index = argmin over k of distance²_k
quantized = projected + stop_gradient(selected - projected)
```

soft target：

```text
p_k = softmax(-distance²_k / τ)
```

`soft_estimated_bits` 参与训练；hard nearest index 对应的
`hard_estimated_bits` 用于结构化 BPP 指标。

## 3. Loss 与 RD point

```text
L_stage3 = mse_weight × MSE
         + lpips_weight × LPIPS-VGG
         + commitment_weight × commitment
         + λ × soft_bpp

commitment = MSE(projected, stop_gradient(selected))
soft_bpp = Σ soft_estimated_bits / (B × T × H × W)
hard_bpp = Σ hard_estimated_bits / (B × T × H × W)
```

λ 不是固定协议常量。每个 RD point 在独立 config 中设置
`loss.rate_weight`，并写入独立 output directory。τ 由 `loss.temperature` 指定。
不同 λ 的实验不得复用同一个 output directory。

Stage3 统一不训练 discriminator：canonical 入口不构建 discriminator、不创建其
optimizer、不执行 forward/backward，也不在 version-3 checkpoint 中保存判别器状态。
canonical config 不包含 GAN 字段。为读取历史 checkpoint，loader 可消费完整的遗留
GAN 字段组，但只接受 `gan_weight=0`；历史 version-2 checkpoint 中的
discriminator state 会被明确忽略。

## 4. Optimizer 与 scheduler

默认 `configs/stage3.yaml`：

| 项目 | 值 |
| --- | ---: |
| tokenizer optimizer | Adam parameter group 0 |
| tokenizer initial LR / eta_min | 1e-5 / 5e-7 |
| entropy initial LR / eta_min | 5e-5 / 1e-6 |
| scheduler | 两组独立 cosine annealing，step 1 到 60000 |
| betas | (0.9, 0.95) |
| gradient clip norm | 1.0 |
| batch size / GPU | 1 |
| max steps | 60000 |
| BF16 | true |
| log interval | 10 |
| checkpoint/eval interval | 5000 |

默认 config 中的 λ 和 τ 只是一个 RD point 的值。创建新 RD point 时只修改所需的
`loss.rate_weight`/`loss.temperature`，其他 scientific setting 保持一致。

## 5. 启动和恢复

当前 root 训练容器模板：

```bash
torchrun --standalone --nproc_per_node=<gpu_count> scripts/stage3.py \
  -c configs/stage3.yaml \
  --dataset-root /ddn/team/shared/gemini \
  --metadata /zjc/ARCodec/data/trainable_jsons_v1_filtered_decode_failures_step0 \
  --uvg-root /luojx/data/UVG \
  --mcl-jcv-root /luojx/data/MCL-JCV/figure4_bt709_first96 \
  --hevc-b-root /luojx/data/HEVC-B/figure4_bt709_first96 \
  --minimax-h3-model-dir /data/heli/model/MiniMax-H3/FL2VA/video_vae \
  --tokenizer-checkpoint results/<run>/stage1/checkpoints/<checkpoint>.pt \
  --entropy-checkpoint results/<run>/stage2/checkpoints/<checkpoint>.pt \
  --output-dir results/<run>/stage3/lambda_<value>
```

两种 checkpoint 语义：

| 参数 | 加载内容 | 起始 step |
| --- | --- | ---: |
| `--init-checkpoint` | joint model | 0 |
| `--resume` | joint model、optimizer、data position、各 rank RNG state | checkpoint step |

两者不能同时使用。`--resume` 要求 scientific config、world size 和 global batch
size 不变；cosine scheduler 从 checkpoint step 对应位置继续。

## 6. 数据与坏视频

Stage3 使用统一 fast OpenCV loader：4 workers/rank、prefetch factor 2、pinned
memory、persistent workers。worker 逐帧解码、resize、crop，不保存整段全分辨率视频。

明确的 `VideoDecodeError` 会被确定性替换为另一个样本，最多尝试 16 次，并写入
`<output-dir>/rejected_samples.jsonl`。shape、metadata 或模型异常不会被吞掉，训练会
直接失败并暴露原因。

## 7. 输出与 eval

每 step 的 `train_metrics.jsonl` 至少包含 `psnr`、hard `bpp`、`soft_bpp`、
`lpips`、`latency_ms`、λ、τ 和两个 learning rate。进度与 `training.log` 每 10 step
输出一次。

每 5000 step 保存 version-3 checkpoint，再使用全部训练 rank 对 UVG、MCL-JCV、
HEVC-B 做 Gaussian distributed eval。结构化结果包含 `psnr`、`bpp`、
`latency_ms`、LPIPS-VGG、LPIPS-Alex 和 DISTS。全部 shard 成功后才合并指标、替换
该 λ 的 canonical recon 并恢复训练。
