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

Datasets, pretrained weights, checkpoints, logs, and generated plots are local
artifacts and are not stored in Git.

The training surface contains exactly three scripts:

| Stage | Script | Trainable modules |
| --- | --- | --- |
| 1 | `scripts/stage1.py` | tokenizer backbone, projections, EMA-VQ |
| 2 | `scripts/stage2.py` | categorical entropy model |
| 3 | `scripts/stage3.py` | tokenizer, entropy model, optional GAN |

The active scripts do not import one another. Shared model, data, loss,
checkpoint, and evaluation code lives under `tokenizer/`, `entropy_model/`, and
`training/`.

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
torchrun --standalone --nproc_per_node=4 scripts/stage1.py \
  --tokenizer minimax_h3 \
  -c configs/stage1.yaml \
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
torchrun --standalone --nproc_per_node=4 scripts/stage2.py \
  -c configs/stage2.yaml \
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

The canonical config fixes `λ=7.2` and `τ=0.01` for 60000 steps. Tokenizer and
entropy-model learning rates change at steps 15000, 30000, and 45000.
`gan_weight=0`, so GAN training is disabled. There is no gradient accumulation.

```bash
torchrun --standalone --nproc_per_node=4 scripts/stage3.py \
  -c configs/stage3.yaml \
  --tokenizer-checkpoint outputs/stage1_h3/checkpoints/checkpoint_best.pt \
  --entropy-checkpoint outputs/stage2_h3/checkpoints/checkpoint_best.pt \
  --output-dir outputs/stage3_h3
```

## Outputs

Every 5000 steps, all three stages run the same seven-sequence, 96-frame UVG
protocol and write structured JSON under `evaluations/`. Stage1 retains the
lowest-LPIPS checkpoint, Stage2 retains the lowest-BPP checkpoint, and Stage3
retains every 5000-step checkpoint. Benchmark output contains at least `psnr`,
`bpp`, and `latency_ms`; LPIPS and DISTS are also reported.

Training metrics are written to `train_metrics.jsonl`, compact logs to
`training.log`, and the exact run configuration to `run_manifest.json`.
