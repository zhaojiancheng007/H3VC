# Contributing

Keep changes scoped and preserve the documented evaluation protocol. Run
`OMP_NUM_THREADS=1 python -m pytest -q` and the affected CLI's `--help` before
submitting a pull request. Include configuration and validation details for
changes to training, entropy models, or reconstruction. Do not commit weights,
datasets, credentials, logs, or reconstructed videos.
