# H3VC Inference

```bash
pip install -r requirements.txt
hf download MiniMaxAI/MiniMax-H3 --include 'FL2VA/video_vae/*' --local-dir pretrained_ckpts/MiniMax-H3

CUDA_VISIBLE_DEVICES=0 python scripts/inference.py \
  --checkpoint /path/to/stage3/checkpoint.pt \
  --entropy-checkpoint /path/to/stage2/checkpoint.pt \
  --minimax-h3-model-dir pretrained_ckpts/MiniMax-H3/FL2VA/video_vae \
  --frames-dir /path/to/rgb_frames \
  --output-dir outputs/example
```

| Parameter | Value |
| --- | --- |
| `CUDA_VISIBLE_DEVICES` | GPU index |
| `--checkpoint` | H3VC Stage3 checkpoint |
| `--entropy-checkpoint` | Matching Stage2 checkpoint; selects Local-only or Global-local |
| `--minimax-h3-model-dir` | MiniMax-H3 `video_vae` directory |
| `--frames-dir` | 96 RGB PNG frames: `000001.png`–`000096.png` |
| `--output-dir` | New directory for reconstruction PNGs and `metrics.json` (PSNR, estimated bpp, latency) |
