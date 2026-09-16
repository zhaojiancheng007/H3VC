# Release validation

Validation performed on 2026-09-16:

- CPU tests: both entropy models, global-local causal/online equivalence,
  differentiable rate, joint tokenizer/entropy backward, frozen codebook,
  entropy-floor gradient, configuration loading, and relative data-path checks.
- Independent imports: every exported runtime module imports without the original
  research repository or removed backbones. No `src.v1`, `src.legacy`, or
  `src.deprecated` module is loaded.
- Actual research Stage2 checkpoints: original local-only v2 and global-local v1
  load strictly into the exported entropy models and produce finite CPU forward
  outputs on a small latent grid.
- All five CLI entrypoints pass `--help`; Python sources compile.

This release reuses the experiment's mathematical forward/loss and evaluation
paths. Full GPU end-to-end VAE reconstruction, an eight-GPU train/eval/resume
cycle, and a fresh-environment dependency installation were **not rerun** for
this export. The VAE, metric weights, and datasets must be obtained separately.
CPU checks establish the extracted model/core behavior, not a new RD benchmark.
