# H3VC Inference

```bash
pip install -r requirements.txt

CUDA_VISIBLE_DEVICES=0 python scripts/inference.py \
  --checkpoint /path/to/checkpoint.pt \
  --frames-dir /path/to/rgb_frames \
  --output-dir outputs/example
```

| Parameter | Value |
| --- | --- |
| `CUDA_VISIBLE_DEVICES` | GPU index |
| `--checkpoint` | H3VC inference checkpoint |
| `--frames-dir` | RGB frame directory |
| `--output-dir` | output directory for reconstruction PNGs |

## TODO

- [x] Release inference code
- [ ] Release checkpoint
- [ ] Release training code
