# Training and evaluation contract

Input: `[B, 3, 107, 256, 256]`, normalized to [-1,1]. The VAE produces posterior
means, followed by pre-VQ projection, raw-L2 nearest-neighbor quantization with
one 4096 × 24 codebook, post-VQ projection, and reconstruction.

Stage1 trains the VAE/projections and updates the codebook with distributed EMA,
including the original dead-code reset. Objective: MSE + LPIPS-VGG + commitment
(weight 0.25). Stage2 freezes the tokenizer and minimizes estimated entropy bpp.
Stage3 freezes the codebook and jointly trains tokenizer and entropy model.

The retained core Stage3 path is:

```python
output = model(video, temperature=config.loss.temperature)
soft_bpp = output.entropy.soft_estimated_bits.sum() / video_pixels
floor_bpp = multiplier * latent_density * relu(H_min - H_soft)
loss = mse_weight * mse + lpips_weight * lpips_vgg
loss += commitment_weight * commitment
loss += rate_weight * (soft_bpp + floor_bpp)
loss.backward()
```

Hard VQ supplies forward reconstruction; the straight-through path sends gradients
to the encoder. Soft rate and marginal entropy use softmax(-squared_distance / τ).
H_soft pools over all training ranks. H_min = log₂(P). This soft penalty does not
guarantee hard-assignment eval perplexity ≥ P. Eval pools hard histograms across
sequences in each dataset. No loss is computed during inference.

Defaults: MSE=1, LPIPS-VGG=1, commitment=0.25, no GAN, no DISTS training loss.
Stage3: Adam betas=(0.9,0.95), weight decay=0, clip norm=1, BF16, tokenizer LR
1e-5 → 5e-7, entropy LR 5e-5 → 1e-6, 40k cosine steps. DISTS is eval-only.

## Current RD chain

| λ | τ | P | H_min (bits) | Initialization |
|---|---|---|---|---|
| 7.2 | 0.1 | 2000 | 10.9657842847 | Stage1 + global-local Stage2 |
| 12 | 0.1 | 1000 | 9.9657842847 | λ7.2 / 40k |
| 12 | 0.1 | 800 | 9.6438561898 | λ12 P1000 / 15k |
| 24 | 0.1 | 400 | 8.6438561898 | λ12 P800 / 25k |
| 48 | 0.1 | 200 | 7.6438561898 | λ24 / 20k |
| 72 | 0.1 | 100 | 6.6438561898 | λ48 / 25k |
| 120 | 0.1 | 50 | 5.6438561898 | λ72 / 25k |

Each new point resets optimizer/scheduler. These are experiment settings, not a
claim that every run completed 40k steps. Floor multiplier is 1 throughout.

## Checkpoint / eval behavior

Stage3 saves every 5k steps, offloads state to CPU, launches one independent eval
subprocess per rank, merges all shards, restores RNG/state, and resumes training.
Failures propagate. The first step writes `gradient_check_step_0000001.json`.

Supported entropy checkpoints: global-local v1, local-only v13, and original
local-only Stage2 v2 (explicitly recognized as K=1). Model loads are strict.
Both published training recipes use fixed K=1. Full-state resume requires this
release's configuration; historical Stage3 weights can initialize a new run
with `--init-checkpoint` even when the historical config contains unused GAN fields.

Eval: 96 source/metric frames, 107 model input frames with last-frame repetition,
spatial padding to multiples of 16 and cropping back, Gaussian tile variance 0.01.
Estimated bpp uses the original 96-frame pixel count. Latency includes codec
forward only; excludes metrics/PNG writes and does not measure bitstream coding.
