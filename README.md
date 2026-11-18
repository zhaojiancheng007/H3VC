# H3VC — inference

Minimal video reconstruction inference with the MiniMax-H3 tokenizer and either
**Local-only** or **Global-local** entropy model.

This repository outputs reconstructed RGB frames and **estimated bpp**. It does
not include an arithmetic-coded bitstream encoder or decoder. Checkpoints and
the external MiniMax-H3 VAE are required separately; weights are not bundled.

## Install

Use Python 3.11 and a CUDA GPU with BF16 support. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Download the VAE Python code, configuration and pretrained weights from the
[official MiniMax-H3 model repository](https://huggingface.co/MiniMaxAI/MiniMax-H3/tree/main/FL2VA/video_vae):

```bash
hf download MiniMaxAI/MiniMax-H3 --include 'FL2VA/video_vae/*' --local-dir pretrained_ckpts/MiniMax-H3
```

The resulting `FL2VA/video_vae` directory must contain
`minimax_h3_video_vae.py`, `config.json` and the weights referenced by that config.
Only use trusted VAE code and checkpoints: inference imports the VAE Python code
and loads PyTorch checkpoints with `weights_only=False`.

## Run

Provide a Stage3 checkpoint and its matching Stage2 source checkpoint.
Stage2 supplies the entropy architecture configuration; **all final tokenizer and
entropy weights are loaded strictly from Stage3**. Paths are explicit and do not
need to match the original machine's paths. Local-only and Global-local are
selected from the Stage2 checkpoint metadata.

Input: one directory of consecutive RGB PNGs, `000001.png` through `000096.png`,
all at the same resolution. If more frames exist, only the first 96 are used.
The standard protocol uses 1920 × 1080; the input is not resized.

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/inference.py \
  --checkpoint /path/to/stage3/checkpoint.pt \
  --entropy-checkpoint /path/to/stage2/checkpoint.pt \
  --minimax-h3-model-dir pretrained_ckpts/MiniMax-H3/FL2VA/video_vae \
  --frames-dir /path/to/rgb_frames \
  --output-dir outputs/example
```

The output directory must not already exist:

- `recon/000001.png` … `recon/000096.png`: reconstructed frames.
- `metrics.json`: PSNR, estimated bpp, latency in ms/frame, codebook perplexity,
  active codes, marginal entropy and estimated bits/token, plus the protocol.

Inference repeats the last frame to form 107 model-input frames, pads spatial
dimensions to multiples of 16, and crops the reconstruction back to the original
96 frames and resolution. It uses hard nearest-code assignment, K=1 entropy
prediction, BF16 autocast, and Gaussian decoder tile blending with variance
**0.04**. No training loss or parameter updates are computed.

Estimated bpp = categorical cross-entropy bits / (96 × original height × original width).
The reported latency includes encoder, quantization, entropy estimation and
decoder, but excludes loading weights, reading/writing PNGs and computing metrics.
It is a single-run diagnostic, not a warmed-up latency benchmark or real
bitstream coding latency. PNG or playback-video size does not represent H3VC bpp.

## Code

- `scripts/inference.py`: the only command-line entrypoint.
- `inference/`: checkpoint loading, frame IO, padding and reconstruction.
- `tokenizer/`: tokenizer implementation and tile blending.
- `entropy_model/`: Local-only and Global-local implementations.

The inference runtime has no dependency on training scripts or the research
repository. The original parameter namespaces are retained for checkpoint
compatibility. CPU regression tests require `pytest`:

```bash
pip install pytest
OMP_NUM_THREADS=1 python -m pytest -q
```

## License

Original H3VC code is MIT-licensed. External code and weights retain their own
licenses; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
