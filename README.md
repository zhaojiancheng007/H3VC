# H3VC

Three-stage video codec training with a MiniMax-H3 VAE, a single 4096 × 24
EMA vector-quantization codebook, and either a **local-only** or
**global-local FIFO-memory** entropy model.

This release provides training, reconstruction, and **estimated entropy bpp**.
It does not produce or decode an arithmetic-coded bitstream. Model weights,
datasets, experiment logs, and reconstructed videos are not included.

## Installation

Tested with Python 3.11, PyTorch 2.5.1 / CUDA 12.1 and H100 GPUs.
Run commands from the repository root in a dedicated environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install pytest
```

Download the external VAE code, configuration, and weights from the
[official MiniMax-H3 repository](https://huggingface.co/MiniMaxAI/MiniMax-H3/tree/main/FL2VA/video_vae):

```bash
hf download MiniMaxAI/MiniMax-H3 --include 'FL2VA/video_vae/*' --local-dir pretrained_ckpts/MiniMax-H3
export VAE_DIR="$PWD/pretrained_ckpts/MiniMax-H3/FL2VA/video_vae"
```

The model directory must contain `minimax_h3_video_vae.py`, `config.json`,
and `source/` with the pretrained model. Its Python code is imported at runtime;
use the trusted upstream source. See [third-party notices](THIRD_PARTY_NOTICES.md).
LPIPS/DISTS may download their pretrained metric weights on first use.

## Repository layout

- `scripts/stage1.py`: tokenizer training with EMA VQ.
- `scripts/stage2.py`: frozen-tokenizer entropy training; configuration selects the model.
- `scripts/stage3.py`: joint rate-distortion training with a soft marginal entropy floor.
- `scripts/inference.py`: one PNG sequence → reconstructions and `metrics.json`.
- `scripts/evaluate.py`: UVG / MCL-JCV / HEVC-B benchmark shard.
- `entropy_model/`: only local-only and global-local model implementations.
- `tokenizer/`, `training/`, `datasets/`, `metrics/`: their required runtime code.

## Data

Training uses MP4 videos and JSON metadata; paths are relative to `DATA_ROOT`.
See [data format](docs/data.md) and `examples/valid_all_chunk_0.json`.
Each training sample is 107 RGB frames. The shorter spatial side is resized to
512 with OpenCV cubic interpolation, followed by a seeded 256 × 256 crop.

Benchmark input is RGB PNG frames, numbered `000001.png` through `000096.png`
(or the original benchmark frame names supported by the loader), at 1920 × 1080.
The exact dataset directory names are documented in [data format](docs/data.md).

```bash
export DATA_ROOT=/path/to/training/videos
export META_DIR=/path/to/metadata
export UVG_ROOT=/path/to/UVG
export MCL_ROOT=/path/to/MCL-JCV/frames
export HEVC_ROOT=/path/to/HEVC-B/frames
```

## Training

All three stages use `torchrun`, cosine schedules, deterministic sampler/RNG
checkpointing, and subprocess benchmark evaluation. Stage1 and Stage2 select the
best checkpoint by UVG LPIPS and bpp, respectively. Stage3 keeps every 5k checkpoint.
The published Stage3 recipe requires **8 GPUs, batch size 1/GPU**. Stage2 uses
batch size 2/GPU (the global-local experiment used 4 GPUs). GPU memory requirements
depend on the VAE; CPU unit tests are not a substitute for a full training run.

### Stage 1 — tokenizer

```bash
torchrun --standalone --nproc_per_node=8 scripts/stage1.py \
  --tokenizer minimax_h3 -c configs/stage1.yaml \
  --dataset-root "$DATA_ROOT" --metadata "$META_DIR" \
  --output-dir results/stage1 --minimax-h3-model-dir "$VAE_DIR" \
  --uvg-root "$UVG_ROOT" --mcl-jcv-root "$MCL_ROOT" --hevc-b-root "$HEVC_ROOT"
```

Set `S1` to the resulting tokenizer checkpoint. The experiments used step 55k
from a 60k cosine schedule; select the checkpoint deliberately.

### Stage 2 — entropy model

```bash
export S1=/path/to/stage1/checkpoint.pt
torchrun --standalone --nproc_per_node=4 scripts/stage2.py \
  -c configs/stage2_local_global_memory.yaml --k 1 \
  --dataset-root "$DATA_ROOT" --metadata "$META_DIR" \
  --tokenizer-checkpoint "$S1" --output-dir results/stage2 \
  --minimax-h3-model-dir "$VAE_DIR" \
  --uvg-root "$UVG_ROOT" --mcl-jcv-root "$MCL_ROOT" --hevc-b-root "$HEVC_ROOT"
```

Use `configs/stage2_local.yaml` for local-only training. Both released training
recipes use fixed K=1. Global-local uses 8 × 8 spatial windows, 512 FIFO memory
tokens, 32 summary tokens/frame, and 32 readout tokens with two readout layers.

### Stage 3 — rate-distortion tuning

```bash
export S2=/path/to/stage2/checkpoint_best.pt
torchrun --standalone --nproc_per_node=8 scripts/stage3.py \
  -c configs/stage3.yaml \
  --dataset-root "$DATA_ROOT" --metadata "$META_DIR" \
  --tokenizer-checkpoint "$S1" --entropy-checkpoint "$S2" \
  --output-dir results/stage3 --minimax-h3-model-dir "$VAE_DIR" \
  --marginal-entropy-floor-bits 10.965784284662087 --entropy-floor-multiplier 1 \
  --uvg-root "$UVG_ROOT" --mcl-jcv-root "$MCL_ROOT" --hevc-b-root "$HEVC_ROOT"
```

The default is λ=7.2, τ=0.1, P=2000, 40k steps; H_min = log₂(P).
Edit `loss.rate_weight` and `loss.temperature` in your config. An existing Stage3
checkpoint can initialize a new RD point with `--init-checkpoint PATH`: **weights
only**, fresh optimizer/scheduler, step 0. `--resume PATH` instead restores full
training state and requires the same configuration and world size. Do not combine
them. Original research checkpoints can initialize new runs; the old GAN-placeholder
configuration is not a resume configuration for this cleaned release.

Use `--allow-checkpoint-relocation` only when explicitly supplied Stage1/Stage2
paths are relocated copies of the original files. Model state loading remains strict.
`--stop-after-steps N` limits additional steps, without changing the cosine period.

See [training and evaluation](docs/training.md) for losses, floors, and validation.

## Reconstruction / inference

Supply a Stage3 checkpoint and its Stage1/Stage2 source checkpoints. Paths are
explicit, so inference does not rely on absolute paths embedded in checkpoints.

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/inference.py \
  --checkpoint /path/to/stage3/checkpoint.pt \
  --tokenizer-checkpoint "$S1" --entropy-checkpoint "$S2" \
  --minimax-h3-model-dir "$VAE_DIR" \
  --frames-dir /path/to/sequence --output-dir outputs/example
```

The output directory must be new. Outputs are 96 reconstructed PNGs under
`recon/input/<sequence>/` and `metrics.json`, including PSNR, bpp, LPIPS-VGG,
DISTS, latency_ms, perplexity, active codes, marginal entropy, and coding cost/token.
No training loss is computed in inference.

## Benchmark evaluation

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate.py \
  --checkpoint /path/to/stage3/checkpoint.pt \
  --tokenizer-checkpoint "$S1" --entropy-checkpoint "$S2" \
  --minimax-h3-model-dir "$VAE_DIR" \
  --uvg-root "$UVG_ROOT" --mcl-jcv-root "$MCL_ROOT" --hevc-b-root "$HEVC_ROOT" \
  --rank 0 --world-size 1 --k 1 \
  --output outputs/eval.json --recon-dir outputs/recon
```

This runs all 42 sequences sequentially on one GPU. For sharding, launch one
independent process per GPU with distinct rank/output, and the same world size.
Stage3's periodic evaluator does this automatically and writes per-dataset JSON,
`summary.json`, and `summary.csv`. The standard protocol scores 96 frames at
1080p, feeds 107 frames (last-frame repetition), uses spatial padding/cropping
and Gaussian decoder tile blending (variance 0.01). Estimated bpp is normalized
by the original 96-frame pixel count. Latency includes encode, VQ, entropy
estimation, and decode, but excludes metrics and PNG writes; it is **not**
bitstream encoder/decoder latency.

## Tests

```bash
OMP_NUM_THREADS=1 python -m pytest -q
```

Tests cover both entropy models, causal context, soft-rate gradients, frozen
codebook behavior, floor gradients, configuration loading, and independent imports.
No dataset or VAE weights are needed for CPU tests. Full GPU training/evaluation
requires the external VAE and datasets.

## License

H3VC code is MIT-licensed. External VAE code, pretrained weights, datasets,
and metric packages retain their own licenses. No third-party weights are
redistributed here.
