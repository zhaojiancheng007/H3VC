from __future__ import annotations

import sys
from pathlib import Path

import torch


class CosmosDVDecoder:
    model_id = "Cosmos-0.1-Tokenizer-DV4x8x8"

    def __init__(
        self,
        decoder_checkpoint: str | Path,
        *,
        cosmos_repo: str | Path,
        device: str = "cuda",
        dtype: str = "bfloat16",
    ) -> None:
        if dtype != "bfloat16":
            raise ValueError(
                "Cosmos-0.1 DV4x8x8 decoder.jit is BF16-only; "
                f"received dtype={dtype!r}"
            )
        decoder_checkpoint = Path(decoder_checkpoint)
        cosmos_repo = Path(cosmos_repo)
        if not decoder_checkpoint.is_file():
            raise FileNotFoundError(f"Missing decoder checkpoint: {decoder_checkpoint}")
        if not cosmos_repo.is_dir():
            raise FileNotFoundError(f"Missing Cosmos repository: {cosmos_repo}")
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")

        repo_path = str(cosmos_repo.resolve())
        if repo_path not in sys.path:
            sys.path.insert(0, repo_path)
        try:
            from cosmos_tokenizer.video_lib import CausalVideoTokenizer
        except ImportError as exc:
            raise ImportError(f"Could not import cosmos_tokenizer from {cosmos_repo}") from exc

        self.device = torch.device(device)
        self.decoder = CausalVideoTokenizer(
            checkpoint_dec=str(decoder_checkpoint), device=device, dtype=dtype
        )

    @torch.no_grad()
    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        if indices.ndim != 4:
            raise ValueError(f"Expected indices [B,T,H,W], got {tuple(indices.shape)}")
        with torch.jit.optimized_execution(False):
            return self.decoder.decode(indices.to(self.device))
