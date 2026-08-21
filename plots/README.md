# Perceptual RD curves

`plot_rd_curves.py` is the only maintained plotting entry point.

Inputs:

- `baseline_rd_points.json`: GNVC-VD and GLC-Video points digitized from
  Figure 4 of `papers/GNVC-VD.pdf`. These are PDF vector-path measurements,
  not the authors' raw logs.
- `gvc1d_rd_points.json`: measured GVC1D UVG points from the local 96-frame FP16 benchmark.
- `ours_rd_points.json`: protocol-matched ARCodec points. Every point records
  estimated categorical-cross-entropy BPP, PSNR, latency, LPIPS-VGG, and
  DISTS. The ARCodec rate source remains explicit in the JSON metadata.

Run:

```bash
/data/heli/miniconda3/envs/arcodec/bin/python plots/plot_rd_curves.py
```

Outputs are written to `plots/output/rd_curves.png` and
`plots/output/rd_curves.pdf`.

To add an operating point, append one object to `ours_rd_points.json`. Do not
mix estimated BPP with measured rANS points in the same method curve.
