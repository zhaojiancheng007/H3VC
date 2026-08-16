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
| 1 | `scripts/stage1_tokenizertraining.py` | Cosmos CV4×8×8 or WanVAE, projections, EMA-VQ |
| 2 | `scripts/stage2_entropytraining.py` | categorical entropy model |
| 3 | `scripts/stage3_jointtraining.py` | tokenizer, entropy model, optional GAN |

The active scripts do not import one another. Shared model, data, loss,
checkpoint, and evaluation code lives under `tokenizer/`, `entropy_model/`, and
`training/`.

## Shared representation

Both supported tokenizer backbones produce a 16-channel latent grid:

```text
video [B,3,25,H,W]
→ Cosmos CV4×8×8 or WanVAE encoder
→ pre_quant 16→16
→ one 4096×16 EMA-VQ codebook
→ one index per latent position [B,T′,H′,W′]
→ post_quant 16→16
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
torchrun --standalone --nproc_per_node=4 scripts/stage1_tokenizertraining.py \
  --tokenizer cosmos_cv488 \
  --config configs/stage1_tokenizer_cosmos_cv488_b4.yaml \
  --dataset-root /luojix/data/OpenVidHD/openvid_mp4 \
  --metadata /luojix/data/OpenVidHD/openvid_mp4/full_multilength_scenes/metadata.jsonl \
  --output-dir outputs/stage1_cosmos_cv488_vq
```

Use `--tokenizer wanvae` and `configs/stage1_tokenizer.yaml` for WanVAE.

## Stage 2

Stage2 freezes the complete Stage1 tokenizer. The selected 16-dimensional code
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
torchrun --standalone --nproc_per_node=4 scripts/stage2_entropytraining.py \
  --config configs/stage2_entropy.yaml \
  --dataset-root /luojix/data/OpenVidHD/openvid_mp4 \
  --metadata /luojix/data/OpenVidHD/openvid_mp4/full_multilength_scenes/metadata.jsonl \
  --tokenizer-checkpoint outputs/stage1_cosmos_cv488_vq/checkpoints/step_0020000.pt \
  --output-dir outputs/stage2_cosmos_cv488_entropy
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

The canonical config linearly increases `λ` from 0.8 to 4.8 over steps 0–2000,
then holds `λ=4.8` through step 20000; both phases use `τ=0.01`. From steps
20000–22000, `λ` linearly increases from 4.8 to 12 with `τ=0.1`, then remains
at 12 through step 30000. `gan_weight=0`, so GAN training is disabled. There is
no gradient accumulation.

```bash
torchrun --standalone --nproc_per_node=4 scripts/stage3_jointtraining.py \
  --config configs/stage3_joint_vq.yaml \
  --dataset-root /luojix/data/OpenVidHD/openvid_mp4 \
  --metadata /luojix/data/OpenVidHD/openvid_mp4/full_multilength_scenes/metadata.jsonl \
  --tokenizer-checkpoint outputs/stage1_cosmos_cv488_vq/checkpoints/step_0020000.pt \
  --entropy-checkpoint outputs/stage2_cosmos_cv488_entropy/checkpoints/step_0060000.pt \
  --output-dir outputs/stage3_cosmos_cv488_joint
```

## Outputs

Every 5000 steps, each stage saves `checkpoints/step_N.pt`. Stage1, Stage2, and
Stage3 run the same seven-sequence, 96-frame UVG protocol and write structured
JSON under `evaluations/`. Benchmark output contains at least `psnr`, `bpp`, and
`latency_ms`; LPIPS and DISTS are also reported.

Training metrics are written to `train_metrics.jsonl`, compact logs to
`training.log`, and the exact run configuration to `run_manifest.json`.
