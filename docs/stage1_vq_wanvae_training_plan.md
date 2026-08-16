# Stage 1: VQ-WanVAE Codebook Adaptation Plan

## Goal and validation gate

Adapt the official pretrained Wan2.1 causal VAE into a discrete tokenizer with
one learned `4096×32` codebook while preserving the original temporal factor 4
and spatial factor `8×8`.

The continuous FP32 WanVAE upper-bound evaluation passed on the first 96 frames
of all seven UVG sequences, using GOP lengths `[25, 25, 25, 21]`:

| Metric | WanVAE FP32 | Best GNVC-VD Figure 4 | Gate |
|---|---:|---:|---|
| PSNR | 35.148913 dB | not reported | report only |
| LPIPS-VGG | 0.116270 | 0.182037 | pass |
| DISTS | 0.023590 | 0.053244 | pass |
| Codec latency | 287.985 ms/frame | not reported | report only |

LPIPS-VGG and DISTS are lower-is-better. Continuous WanVAE has no bitstream, so
its `bpp` is undefined rather than zero.

## Model path

Keep the official Wan encoder and decoder intact and add a parallel adaptation
module around the normalized deterministic encoder mean:

```text
RGB video [B,3,T,H,W]
  → official Wan encoder mean [B,16,T',H/8,W/8]
  → pointwise projection 16→32
  → nearest entry in one learned codebook [4096,32]
  → straight-through quantized tensor [B,32,T',H/8,W/8]
  → pointwise projection 32→16
  → official Wan decoder
  → reconstructed RGB video [B,3,T,H,W]
```

For every latent position:

```text
zₑ = Penc(z)
k = argmin over j of ||zₑ - eⱼ||²
zq = zₑ + stopgrad(eₖ - zₑ)
ẑ = Pdec(zq)
x̂ = WanDecoder(ẑ)
```

The model outputs the reconstruction, integer indices in `[0,4095]`, the
pre-quantization tensor, and the quantized tensor. The integer index grid is the
only Stage 2 input.

Initialize `Penc` as `[I₁₆; 0]` and `Pdec` as `[I₁₆, 0]`, with zero biases. The
unquantized adapter therefore starts as an exact identity. Initialize the 4096
entries with k-means over projected Wan latents sampled from the training set;
do not initialize them randomly.

## Data and protocol

- Do not train on UVG. UVG remains the final held-out comparison set.
- Follow the repository's existing OpenVidHD video loader and sampling path in
  a new parallel Stage 1 script.
- Train on causal clips of 25 frames (`4k+1`) so the temporal interface matches
  evaluation and later entropy coding.
- Begin with random `256×256` crops; after the frozen-backbone phase, use a
  `256/384/512` crop mixture to expose the decoder to larger spatial contexts.
- Use the same RGB conversion and `[-1,1]` normalization as official WanVAE.
- Keep a fixed held-out OpenVidHD validation split and never reshuffle it into
  training.

For a 96-frame 1080p video, `[25,25,25,21]` produces 27 latent frames and
`27×135×240 = 874800` indices. Raw fixed-width storage is
`874800×12 / (96×1080×1920) = 0.052734375 bpp`; this is not the final compressed
rate. Stage 2 must entropy-code the nonuniform indices.

## Loss

Use no entropy, bitrate, GAN, or index-uniformity loss in Stage 1.

```text
Llatent = mean((ẑ - stopgrad(z))²)
Lpixel = mean(|x̂ - x|)
Lperceptual = LPIPS-VGG(x̂, x)
Lcodebook = mean((stopgrad(zₑ) - eₖ)²)
Lcommit = mean((zₑ - stopgrad(eₖ))²)

L = 1.0×Llatent
  + 1.0×Lpixel
  + 0.1×Lperceptual
  + 1.0×Lcodebook
  + 0.25×Lcommit
```

DISTS is an evaluation metric, not a training loss. Keeping it out makes the
training objective shorter and avoids doubling the VGG feature cost.

## Training phases

### Phase A: deterministic initialization

- Run the frozen official encoder over training clips and sample at least ten
  million latent positions.
- Apply the identity-initialized `16→32` projection.
- Fit exactly 4096 k-means centers and load them into the codebook.
- Verify the no-quantization adapter path reproduces the official continuous
  baseline before starting optimization.

### Phase B: frozen Wan backbone, 30000 steps

- Train only `Penc`, the codebook, and `Pdec`.
- AdamW; projection learning rate `2×10⁻⁴`, codebook learning rate `1×10⁻⁴`,
  1000-step warmup followed by cosine decay.
- No weight decay on codebook vectors; `1×10⁻⁴` weight decay on projections.
- Target effective batch: 32 clips across four GPUs using gradient
  accumulation when required by memory.
- Save a structured checkpoint and evaluate the fixed validation split every
  5000 steps.

### Phase C: decoder adaptation, at most 20000 steps

Enter this phase only if Phase B fails either UVG perceptual threshold.

- Keep the Wan encoder frozen.
- Unfreeze the Wan latent input convolution and decoder.
- Use `1×10⁻⁵` for newly unfrozen Wan parameters and `1×10⁻⁴` for the adapter
  and codebook.
- Keep the same loss and evaluation cadence.

### Phase D: encoder-tail adaptation, at most 10000 steps

Enter this phase only if Phase C still fails.

- Unfreeze only the final encoder block and latent output convolution.
- Use `2×10⁻⁶` for Wan encoder parameters.
- Stop if the continuous path itself degrades, rather than masking the failure
  with a stronger perceptual or GAN loss.

## Measurements and acceptance

Every checkpoint report must be structured JSON and contain at least `psnr`,
`bpp`, and `latency_ms`, plus `lpips_vgg`, `dists`, codebook active entries,
perplexity, top-1 index frequency, and index histogram.

The final Stage 1 checkpoint passes only when all conditions hold:

1. Decode from saved integer indices exactly reproduces decode from the
   quantized tensor.
2. UVG LPIPS-VGG is below `0.18203738`.
3. UVG DISTS is below `0.05324428`.
4. All seven UVG sequences and all 672 frames are evaluated.
5. At least 1024 entries are active over one million fixed held-out training
   tokens. This is a collapse alarm, not a uniform-usage objective.
6. No UVG sample was used for optimization or k-means initialization.

If condition 5 fails, stop and inspect the assignment distribution. Do not
silently recycle entries or add an entropy bonus, because either action would
change the scientific question being tested.

## Implementation boundary

Create parallel files next to the existing pipeline, for example:

- `tokenizer/wan_vq/model.py`: self-contained adapter and quantizer.
- `scripts/train_stage1_wan_vq.py`: minor adaptation of the current training
  data and distributed setup.
- `scripts/evaluate_stage1_wan_vq_uvg.py`: index round-trip and UVG metrics.

Import the official Wan2.1 package and existing shared dataset/metric helpers.
Do not import `src/v1`, `src/legacy`, `src/deprecated`, or an older tokenizer
implementation at runtime.
