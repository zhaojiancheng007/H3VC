from __future__ import annotations

import sys
from pathlib import Path

import torch


class CosmosDVTokenizer:
    model_id = "Cosmos-0.1-Tokenizer-DV4x8x8"

    def __init__(
        self,
        encoder_checkpoint: str | Path,
        decoder_checkpoint: str | Path,
        *,
        cosmos_repo: str | Path,
        device: str = "cuda",
        dtype: str = "bfloat16",
    ) -> None:
        if dtype != "bfloat16":
            raise ValueError(
                "Cosmos-0.1 DV4x8x8 JIT checkpoints are BF16-only; "
                f"received dtype={dtype!r}"
            )
        encoder_checkpoint = Path(encoder_checkpoint)
        decoder_checkpoint = Path(decoder_checkpoint)
        cosmos_repo = Path(cosmos_repo)
        for name, path in (
            ("encoder checkpoint", encoder_checkpoint),
            ("decoder checkpoint", decoder_checkpoint),
            ("Cosmos repository", cosmos_repo),
        ):
            if not path.exists():
                raise FileNotFoundError(f"Missing {name}: {path}")
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")

        sys.path.insert(0, str(cosmos_repo.resolve()))
        try:
            from cosmos_tokenizer.video_lib import CausalVideoTokenizer
        except ImportError as exc:
            raise ImportError(
                f"Could not import cosmos_tokenizer from {cosmos_repo}"
            ) from exc

        self.device = torch.device(device)
        self.dtype = torch.bfloat16
        self.encoder = CausalVideoTokenizer(
            checkpoint_enc=str(encoder_checkpoint), device=device, dtype=dtype
        )
        self.decoder = CausalVideoTokenizer(
            checkpoint_dec=str(decoder_checkpoint), device=device, dtype=dtype
        )

    @torch.no_grad()
    def encode(self, video: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if video.ndim != 5 or video.shape[1] != 3:
            raise ValueError(f"Expected [B,3,T,H,W], got {tuple(video.shape)}")
        if video.min().item() < -1.0001 or video.max().item() > 1.0001:
            raise ValueError("Cosmos input must be normalized to [-1, 1]")
        # Pin the TorchScript graph because optimized BF16 fusion changes FSQ
        # indices after profiling warmup on the current PyTorch/CUDA stack.
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

    @torch.no_grad()
    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        if indices.ndim != 4:
            raise ValueError(f"Expected indices [B,T,H,W], got {tuple(indices.shape)}")
        with torch.jit.optimized_execution(False):
            return self.decoder.decode(indices.to(self.device))
