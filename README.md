# ARCodec

ARCodec trains one video VQ tokenizer, one categorical entropy model, and then
jointly optimizes both for rate-distortion performance.

## Clone

Wan2.1 and Cosmos-Tokenizer are pinned Git submodules. Clone the complete
source tree with:

```bash
git clone --recurse-submodules https://github.com/zhaojiancheng007/ARCodec-.git
```

For an existing clone, initialize the dependencies with:

```bash
git submodule update --init --recursive
```

Datasets, pretrained weights, checkpoints, logs, reconstructions, and generated
plots are local artifacts and are not stored in Git. Small structured loader
benchmarks under `outputs/benchmarks/` are the only versioned output artifacts.

The current train-eval pipeline has one entry per stage:

| Stage | Cosine entry | Integrated behavior |
| --- | --- | --- |
| 1 | `scripts/stage1_cosine.py` | cosine LR, high-throughput loader, video-distributed Gaussian eval |
| 2 | `scripts/stage2_cosine.py` | cosine LR, high-throughput loader, video-distributed Gaussian eval |
| 3 | `scripts/stage3_cosine.py` | cosine LR, high-throughput loader, video-distributed Gaussian eval, direct `--resume` |

These entries use the canonical `configs/stage*_cosine.yaml` files. GPU count is
selected at launch through `torchrun --nproc_per_node`; it is not fixed by the
repository. A checkpoint must resume with the same `world_size` and global
batch size. The original `scripts/stage1.py`, `scripts/stage2.py`, and
`scripts/stage3.py` entries remain as baseline references.

The complete checkpoint, distributed eval, resume, metric, and reconstruction
contract is fixed in [docs/train_eval_workflow.md](docs/train_eval_workflow.md).

## Shared representation

The checked-in configs target MiniMax-H3 and its 24-channel latent grid:

```text
video [B,3,107,H,W]
→ MiniMax-H3 encoder
→ pre_quant 24→24
→ one 4096×24 EMA-VQ codebook
→ one index per latent position [B,T′,H′,W′]
→ post_quant 24→24
→ matching decoder
```

Each index is in `[0,4095]` and requires 12 raw bits before entropy coding.
There is no RVQ and no extra index dimension.

## Stage 1

Stage1 jointly trains the encoder, projections, and decoder. The codebook is
initialized with ten Lloyd iterations and then updated from global DDP EMA
assignment statistics.

```text
loss = 1.0 × MSE + 1.0 × LPIPS-VGG + 0.25 × commitment
```

```bash
torchrun --standalone --nproc_per_node=<gpu_count> scripts/stage1_cosine.py \
  --tokenizer minimax_h3 \
  -c configs/stage1_cosine.yaml \
  --output-dir outputs/stage1_h3
```

## Stage 2

Stage2 freezes the complete Stage1 tokenizer. The selected 24-dimensional code
vectors are the context and the target is one complete 4096-way index:

```text
video
→ frozen Stage1 tokenizer
→ quantized features + indices
→ four-group, fixed 8×8-window categorical entropy model
→ 4096 logits per latent position

loss = total negative log₂ probability / number of indices
```

```bash
torchrun --standalone --nproc_per_node=<gpu_count> scripts/stage2_cosine.py \
  -c configs/stage2_cosine.yaml \
  --tokenizer-checkpoint outputs/stage1_h3/checkpoints/checkpoint_best.pt \
  --output-dir outputs/stage2_h3
```

The tokenizer type is read from the Stage1 checkpoint, so the same command also
supports WanVAE checkpoints.

## Stage 3

Stage3 loads the exact Stage1 checkpoint used by Stage2. One 4096-way distance
matrix produces both the hard index and the differentiable soft target.
The EMA codebook remains frozen; tokenizer projections, encoder/decoder, and
entropy model are trainable.

```text
loss = 1.0 × MSE
     + 1.0 × LPIPS-VGG
     + 0.25 × commitment
     + λ × soft_bpp
     + gan_weight × GAN loss
```

The current λ=18 config uses cosine schedules for tokenizer and entropy-model
learning rates. `gan_weight=0`, so GAN training is disabled.

```bash
torchrun --standalone --nproc_per_node=<gpu_count> scripts/stage3_cosine.py \
  -c configs/stage3_lambda18_cosine.yaml \
  --tokenizer-checkpoint outputs/stage1_h3/checkpoints/checkpoint_best.pt \
  --entropy-checkpoint outputs/stage2_h3/checkpoints/checkpoint_best.pt \
  --output-dir outputs/stage3_h3
```

## Outputs

Every 5000 steps, all three stages distribute the 42-video UVG, MCL-JCV, and
HEVC-B benchmark across all training ranks and use Gaussian spatial-tile
blending. Stage1 retains the lowest-LPIPS checkpoint, Stage2 retains the
lowest-BPP checkpoint, and Stage3 retains every 5000-step checkpoint. Benchmark
JSON contains `psnr`, `bpp`, `latency_ms`, `lpips_vgg`, `lpips_alex`, and
`dists`. Only Stage3 saves reconstructions, with one canonical recon archive
per λ.

Training metrics are written to `train_metrics.jsonl`, compact logs to
`training.log`, and the exact run configuration to `run_manifest.json`.
