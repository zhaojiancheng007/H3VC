from __future__ import annotations

import sys
from pathlib import Path

import torch


class CosmosDVEncoder:
    model_id = "Cosmos-0.1-Tokenizer-DV4x8x8"

    def __init__(
        self,
        encoder_checkpoint: str | Path,
        *,
        cosmos_repo: str | Path,
        device: str = "cuda",
        dtype: str = "bfloat16",
    ) -> None:
        if dtype != "bfloat16":
            raise ValueError(
                "Cosmos-0.1 DV4x8x8 encoder.jit is BF16-only; "
                f"received dtype={dtype!r}"
            )
        encoder_checkpoint = Path(encoder_checkpoint)
        cosmos_repo = Path(cosmos_repo)
        if not encoder_checkpoint.is_file():
            raise FileNotFoundError(f"Missing encoder checkpoint: {encoder_checkpoint}")
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
        self.dtype = torch.bfloat16
        self.encoder = CausalVideoTokenizer(
            checkpoint_enc=str(encoder_checkpoint), device=device, dtype=dtype
        )

    @torch.no_grad()
    def encode(self, video: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if video.ndim != 5 or video.shape[1] != 3:
            raise ValueError(f"Expected [B,3,T,H,W], got {tuple(video.shape)}")
        if video.min().item() < -1.0001 or video.max().item() > 1.0001:
            raise ValueError("Cosmos input must be normalized to [-1,1]")
        # PyTorch 2.5 changes BF16 FSQ decisions when the profiling executor
        # replaces this checkpoint's graph. Keep one fixed discrete mapping.
        with torch.jit.optimized_execution(False):
            output = self.encoder.encode(video.to(self.device, dtype=self.dtype))
        if not isinstance(output, tuple) or len(output) != 2:
            raise RuntimeError(f"Expected Cosmos DV encoder to return 2 tensors, got {type(output)}")
        indices, codes = output
        if indices.ndim != 4 or codes.ndim != 5 or codes.shape[1] != 6:
            raise RuntimeError(
                f"Unexpected Cosmos outputs: indices={tuple(indices.shape)}, codes={tuple(codes.shape)}"
            )
        return indices, codes
