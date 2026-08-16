from __future__ import annotations

import copy
from collections.abc import Callable
from pathlib import Path
import sys
from typing import Mapping

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .trainable_stage3_checkpointed import (
    DV_LEVELS,
    TrainableCosmosDVOutput,
    fsq_bound,
)


class NativeCheckpointedTrainableCosmosDV(nn.Module):
    """Official native Cosmos-DV modules with block-level activation replay."""

    model_id = "Cosmos-0.1-Tokenizer-DV4x8x8"
    # At 1080p this caps each BF16 global-attention score tensor below 3.5 GiB.
    SPATIAL_ATTENTION_QUERY_CHUNK = 8192

    def __init__(
        self,
        encoder_checkpoint: str | Path,
        decoder_checkpoint: str | Path,
        *,
        cosmos_repo: str | Path,
        activation_checkpointing: bool,
    ) -> None:
        super().__init__()
        encoder_checkpoint = Path(encoder_checkpoint)
        decoder_checkpoint = Path(decoder_checkpoint)
        cosmos_repo = Path(cosmos_repo)
        if not encoder_checkpoint.is_file():
            raise FileNotFoundError(encoder_checkpoint)
        if not decoder_checkpoint.is_file():
            raise FileNotFoundError(decoder_checkpoint)
        if not cosmos_repo.is_dir():
            raise FileNotFoundError(cosmos_repo)

        repo_path = str(cosmos_repo.resolve())
        if repo_path not in sys.path:
            sys.path.insert(0, repo_path)
        from cosmos_tokenizer.networks import TokenizerConfigs
        from cosmos_tokenizer.networks.discrete_video import (
            CausalDiscreteVideoTokenizer,
        )
        from cosmos_tokenizer.modules.layers3d import CausalAttnBlock

        config = copy.deepcopy(TokenizerConfigs.DV.value)
        config.update(spatial_compression=8, temporal_compression=4)
        native = CausalDiscreteVideoTokenizer(**config).to(dtype=torch.bfloat16)
        self.encoder_jit = native.encoder_jit()
        self.decoder_jit = native.decoder_jit()
        self.activation_checkpointing = activation_checkpointing
        self._spatial_attention_type = CausalAttnBlock

        source_encoder = torch.jit.load(str(encoder_checkpoint), map_location="cpu")
        source_decoder = torch.jit.load(str(decoder_checkpoint), map_location="cpu")
        self._encoder_template = {
            name: value.detach().cpu().clone()
            for name, value in source_encoder.state_dict().items()
        }
        self._decoder_template = {
            name: value.detach().cpu().clone()
            for name, value in source_decoder.state_dict().items()
        }
        self._prepare_component(
            self.encoder_jit,
            source_encoder,
            self._encoder_template,
            component_name="encoder",
        )
        self._prepare_component(
            self.decoder_jit,
            source_decoder,
            self._decoder_template,
            component_name="decoder",
        )
        del source_encoder, source_decoder

        levels = tuple(int(value) for value in self.encoder_jit.quantizer._levels)
        if levels != DV_LEVELS:
            raise RuntimeError(f"Unexpected native Cosmos FSQ levels: {levels}")
        parameter_dtypes = {
            parameter.dtype
            for parameter in self.parameters()
            if parameter.requires_grad
        }
        if parameter_dtypes != {torch.bfloat16}:
            raise RuntimeError(
                f"Native Cosmos trainable parameters must be BF16, got "
                f"{parameter_dtypes}"
            )

    @staticmethod
    def _prepare_component(
        native: nn.Module,
        source: torch.jit.ScriptModule,
        source_state: Mapping[str, torch.Tensor],
        *,
        component_name: str,
    ) -> None:
        source_parameter_names = set(dict(source.named_parameters()))
        native_parameters = dict(native.named_parameters())
        missing_native_parameters = source_parameter_names - set(native_parameters)
        if missing_native_parameters:
            raise RuntimeError(
                f"Native {component_name} lacks released parameters: "
                f"{sorted(missing_native_parameters)}"
            )
        for name, parameter in native_parameters.items():
            if name not in source_parameter_names:
                parameter.requires_grad_(False)

        trainable_names = {
            name for name, parameter in native.named_parameters()
            if parameter.requires_grad
        }
        if trainable_names != source_parameter_names:
            raise RuntimeError(
                f"Native {component_name} trainable parameter mapping changed"
            )
        NativeCheckpointedTrainableCosmosDV._load_component_state(
            native,
            source_state,
            source_state,
            component_name=component_name,
        )

    @staticmethod
    def _load_component_state(
        native: nn.Module,
        values: Mapping[str, torch.Tensor],
        template: Mapping[str, torch.Tensor],
        *,
        component_name: str,
    ) -> None:
        if set(values) != set(template):
            raise ValueError(
                f"{component_name} JIT-compatible state keys changed"
            )
        native_state = native.state_dict()
        loadable = {
            name: value for name, value in values.items() if name in native_state
        }
        result = native.load_state_dict(loadable, strict=False)
        expected_missing = set(native_state) - set(loadable)
        if set(result.missing_keys) != expected_missing or result.unexpected_keys:
            raise RuntimeError(
                f"Native {component_name} state mapping changed: "
                f"missing={result.missing_keys}, "
                f"unexpected={result.unexpected_keys}"
            )
        for name in set(template) - set(native_state):
            if not torch.equal(values[name].cpu(), template[name].cpu()):
                raise ValueError(
                    f"Immutable {component_name} JIT state {name} changed"
                )

        current = native.state_dict()
        changed = [
            name
            for name in loadable
            if not torch.equal(current[name].cpu(), values[name].cpu())
        ]
        if changed:
            raise RuntimeError(
                f"Native {component_name} did not reproduce state: {changed[:5]}"
            )

    @staticmethod
    def _component_jit_state(
        native: nn.Module,
        template: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        native_state = native.state_dict()
        return {
            name: native_state[name] if name in native_state else template[name]
            for name in template
        }

    def jit_compatible_state_dict(self) -> dict[str, torch.Tensor]:
        state: dict[str, torch.Tensor] = {}
        for name, value in self._component_jit_state(
            self.encoder_jit, self._encoder_template
        ).items():
            state[f"encoder_jit.{name}"] = value
        for name, value in self._component_jit_state(
            self.decoder_jit, self._decoder_template
        ).items():
            state[f"decoder_jit.{name}"] = value
        return state

    def load_jit_compatible_state_dict(
        self, state: Mapping[str, torch.Tensor]
    ) -> None:
        encoder_prefix = "encoder_jit."
        decoder_prefix = "decoder_jit."
        encoder_state = {
            name.removeprefix(encoder_prefix): value
            for name, value in state.items()
            if name.startswith(encoder_prefix)
        }
        decoder_state = {
            name.removeprefix(decoder_prefix): value
            for name, value in state.items()
            if name.startswith(decoder_prefix)
        }
        consumed = {
            f"{encoder_prefix}{name}" for name in encoder_state
        } | {
            f"{decoder_prefix}{name}" for name in decoder_state
        }
        if consumed != set(state):
            raise ValueError(
                f"Unexpected native tokenizer checkpoint keys: "
                f"{sorted(set(state) - consumed)}"
            )
        self._load_component_state(
            self.encoder_jit,
            encoder_state,
            self._encoder_template,
            component_name="encoder",
        )
        self._load_component_state(
            self.decoder_jit,
            decoder_state,
            self._decoder_template,
            component_name="decoder",
        )

    def _run(
        self,
        function: Callable[[torch.Tensor], torch.Tensor],
        value: torch.Tensor,
    ) -> torch.Tensor:
        if self.activation_checkpointing and torch.is_grad_enabled():
            return checkpoint(
                function,
                value,
                use_reentrant=False,
                preserve_rng_state=False,
            )
        return function(value)

    def _run_sequence(self, modules: nn.Sequential, value: torch.Tensor) -> torch.Tensor:
        for module in modules:
            if (
                self.activation_checkpointing
                and torch.is_grad_enabled()
                and isinstance(module, self._spatial_attention_type)
            ):
                value = self._run_spatial_attention(module, value)
            else:
                value = self._run(module, value)
        return value

    def _run_spatial_attention(
        self,
        module: nn.Module,
        value: torch.Tensor,
    ) -> torch.Tensor:
        from cosmos_tokenizer.modules.utils import batch2time, time2batch

        normalized = self._run(module.norm, value)
        query = self._run(module.q, normalized)
        key = self._run(module.k, normalized)
        values = self._run(module.v, normalized)

        query, batch_size = time2batch(query)
        key, _ = time2batch(key)
        values, _ = time2batch(values)
        batch_frames, channels, height, width = query.shape
        sites = height * width
        query = query.reshape(batch_frames, channels, sites).permute(0, 2, 1)
        key = key.reshape(batch_frames, channels, sites)
        values = values.reshape(batch_frames, channels, sites)

        attended_chunks: list[torch.Tensor] = []
        scale = int(channels) ** (-0.5)

        def attend(
            query_chunk: torch.Tensor,
            full_key: torch.Tensor,
            full_values: torch.Tensor,
        ) -> torch.Tensor:
            weights = torch.bmm(query_chunk, full_key)
            weights = weights * scale
            weights = torch.softmax(weights, dim=2)
            return torch.bmm(full_values, weights.permute(0, 2, 1))

        for start in range(0, sites, self.SPATIAL_ATTENTION_QUERY_CHUNK):
            end = min(start + self.SPATIAL_ATTENTION_QUERY_CHUNK, sites)
            attended_chunks.append(
                checkpoint(
                    attend,
                    query[:, start:end],
                    key,
                    values,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            )

        attended = torch.cat(attended_chunks, dim=2).reshape(
            batch_frames, channels, height, width
        )
        attended = batch2time(attended, batch_size)
        attended = self._run(module.proj_out, attended)
        return value + attended

    def _encode_prequant(self, video: torch.Tensor) -> torch.Tensor:
        encoder = self.encoder_jit.encoder
        value = encoder.patcher3d(video)
        value = self._run_sequence(encoder.conv_in, value)
        for level in range(encoder.num_resolutions):
            for block_index in range(encoder.num_res_blocks):
                value = self._run(encoder.down[level].block[block_index], value)
                if len(encoder.down[level].attn) > 0:
                    value = self._run_sequence(
                        encoder.down[level].attn[block_index], value
                    )
            if level != encoder.num_resolutions - 1:
                value = self._run(encoder.down[level].downsample, value)
        value = self._run(encoder.mid.block_1, value)
        value = self._run_sequence(encoder.mid.attn_1, value)
        value = self._run(encoder.mid.block_2, value)
        value = self._run(encoder.norm_out, value)
        value = value * torch.sigmoid(value)
        return self._run_sequence(encoder.conv_out, value)

    def _decode_features(self, quant_features: torch.Tensor) -> torch.Tensor:
        decoder = self.decoder_jit.decoder
        value = self._run_sequence(decoder.conv_in, quant_features)
        value = self._run(decoder.mid.block_1, value)
        value = self._run_sequence(decoder.mid.attn_1, value)
        value = self._run(decoder.mid.block_2, value)
        for level in reversed(range(decoder.num_resolutions)):
            for block_index in range(decoder.num_res_blocks + 1):
                value = self._run(decoder.up[level].block[block_index], value)
                if len(decoder.up[level].attn) > 0:
                    value = self._run_sequence(
                        decoder.up[level].attn[block_index], value
                    )
            if level != 0:
                value = self._run(decoder.up[level].upsample, value)
        value = self._run(decoder.norm_out, value)
        value = value * torch.sigmoid(value)
        value = self._run_sequence(decoder.conv_out, value)
        return decoder.unpatcher3d(value)

    def forward(self, video: torch.Tensor) -> TrainableCosmosDVOutput:
        if video.ndim != 5 or video.shape[1] != 3:
            raise ValueError(f"Expected video [B,3,T,H,W], got {tuple(video.shape)}")
        prequant = self._run(
            self._encode_prequant, video.to(torch.bfloat16)
        )
        prequant = self._run(self.encoder_jit.quant_conv, prequant)
        indices, quant_features, _ = self.encoder_jit.quantizer(prequant)
        bounded = fsq_bound(prequant)
        reconstruction = self._run(
            self.decoder_jit.post_quant_conv, quant_features
        )
        reconstruction = self._run(self._decode_features, reconstruction)
        return TrainableCosmosDVOutput(
            prequant=prequant,
            bounded=bounded,
            indices=indices,
            quant_features=quant_features,
            reconstruction=reconstruction,
        )
