# Data format

Training metadata directories contain `valid_all_chunk_0.json`,
`valid_all_chunk_1.json`, etc. Each contains:

```json
{"clips": [{"video_path": "videos/example.mp4", "video_length": 120, "video_height": 1080, "video_width": 1920}]}
```

Paths are relative to the dataset root; absolute paths and `..` are rejected.
Frame count and dimensions must match the MP4. Videos shorter than 107 frames
are excluded. Data acquisition and scene segmentation are not included.
The existing fast loader logs failed decodes and deterministic replacement
attempts to `rejected_samples.jsonl`; exhaustion raises an error.

Benchmark directories:

- UVG: `<root>/<Name>_1920x1080_120fps_420_8bit_YUV_frames/`, with Beauty,
  Bosphorus, HoneyBee, Jockey, ReadySteadyGo, ShakeNDry, YachtRide.
- MCL-JCV: `<root>/videoSRC01/` through `videoSRC30/`.
- HEVC-B: `<root>/BasketballDrive_1920x1080_50/`, `BQTerrace_1920x1080_60/`,
  `Cactus_1920x1080_50/`, `Kimono1_1920x1080_24/`, `ParkScene_1920x1080_24/`.

Use 96 consecutive RGB PNG frames numbered 1 through 96 (zero padding allowed).
Use the same BT.709 YUV-to-RGB conversion for all compared codecs. The standard
protocol is 1920 × 1080; 1792 × 1024 resize/crop is a separate test protocol.
Single-sequence inference accepts other spatial sizes, but their metrics must
not be compared as standard 1080p results.
