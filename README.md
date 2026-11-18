# H3VC Inference

```bash
pip install -r requirements.txt

CUDA_VISIBLE_DEVICES=0 python scripts/inference.py \
  --checkpoint /path/to/stage3/checkpoint.pt \
  --frames-dir /path/to/rgb_frames \
  --output-dir outputs/example
```

| Parameter | Value |
| --- | --- |
| `CUDA_VISIBLE_DEVICES` | GPU index |
| `--checkpoint` | H3VC Stage3 checkpoint |
| `--frames-dir` | 96 RGB PNG frames: `000001.png`–`000096.png` |
| `--output-dir` | New directory for reconstruction PNGs and `metrics.json` (PSNR, estimated bpp, latency) |

## TODO

- [x] Release inference code
- [ ] Release checkpoint
- [ ] Release training code
