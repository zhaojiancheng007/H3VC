from __future__ import annotations

import torch

from .encoder import CosmosDVEncoder


class MicrobatchedCosmosDVEncoder:
    def __init__(
        self,
        encoder: CosmosDVEncoder,
        *,
        microbatch_size: int,
        num_streams: int,
    ) -> None:
        if microbatch_size <= 0:
            raise ValueError("microbatch_size must be positive")
        if num_streams <= 0:
            raise ValueError("num_streams must be positive")
        if encoder.device.type != "cuda":
            raise ValueError("CUDA streams require a CUDA Cosmos encoder")
        self.encoder = encoder
        self.microbatch_size = microbatch_size
        self.num_streams = num_streams
        self.streams = tuple(
            torch.cuda.Stream(device=encoder.device) for _ in range(num_streams)
        )

    @torch.no_grad()
    def encode(self, video: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if video.ndim != 5 or video.shape[1] != 3:
            raise ValueError(f"Expected [B,3,T,H,W], got {tuple(video.shape)}")
        batch_size = int(video.shape[0])
        if batch_size % self.microbatch_size:
            raise ValueError(
                f"Input batch {batch_size} is not divisible by "
                f"microbatch_size={self.microbatch_size}"
            )
        chunk_count = batch_size // self.microbatch_size
        if self.num_streams > chunk_count:
            raise ValueError(
                f"num_streams={self.num_streams} exceeds chunk_count={chunk_count}"
            )

        outputs: list[tuple[torch.Tensor, torch.Tensor]] = []
        if self.num_streams == 1:
            for start in range(0, batch_size, self.microbatch_size):
                end = start + self.microbatch_size
                outputs.append(self.encoder.encode(video[start:end]))
        else:
            for chunk, start in enumerate(
                range(0, batch_size, self.microbatch_size)
            ):
                end = start + self.microbatch_size
                stream = self.streams[chunk % self.num_streams]
                with torch.cuda.stream(stream):
                    outputs.append(self.encoder.encode(video[start:end]))

            current_stream = torch.cuda.current_stream(self.encoder.device)
            for stream in self.streams:
                current_stream.wait_stream(stream)
            for output in outputs:
                for tensor in output:
                    tensor.record_stream(current_stream)

        indices = torch.cat([output[0] for output in outputs], dim=0)
        codes = torch.cat([output[1] for output in outputs], dim=0)
        if int(indices.shape[0]) != batch_size or int(codes.shape[0]) != batch_size:
            raise RuntimeError("Microbatched Cosmos output batch is incomplete")
        return indices, codes
