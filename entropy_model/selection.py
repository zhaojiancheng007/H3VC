from .config_categorical import CategoricalEntropyModelConfig
from .config_local_global_memory import LocalGlobalMemoryEntropyConfig
from .model_categorical import CategoricalEntropyModel
from .model_local_global_memory import LocalGlobalMemoryEntropyModel

LOCAL = "unified_spatiotemporal_window_diagonal_v1"
GLOBAL_LOCAL = "canonical_local_window_plus_fifo_global_v3"


def architecture(config):
    return GLOBAL_LOCAL if isinstance(config, LocalGlobalMemoryEntropyConfig) else LOCAL


def checkpoint_version(config):
    return 1 if architecture(config) == GLOBAL_LOCAL else 13


def build_model(config):
    if isinstance(config, LocalGlobalMemoryEntropyConfig):
        return LocalGlobalMemoryEntropyModel(config)
    if isinstance(config, CategoricalEntropyModelConfig):
        return CategoricalEntropyModel(config)
    raise TypeError(type(config))


def config_from_checkpoint(payload):
    if payload["version"] == 2 and payload["stage"] == 2 and "architecture" not in payload:
        return CategoricalEntropyModelConfig(**payload["config"]["model"])
    pair = (payload["architecture"], payload["version"])
    if pair == (GLOBAL_LOCAL, 1):
        return LocalGlobalMemoryEntropyConfig(**payload["config"]["model"])
    if pair == (LOCAL, 13):
        return CategoricalEntropyModelConfig(**payload["config"]["model"])
    raise ValueError(f"Unsupported entropy checkpoint architecture/version: {pair}")


def normalize_stage2_checkpoint(payload):
    # The original local-only experiment used this exact Stage2 v2 schema.
    if payload["version"] == 2 and payload["stage"] == 2 and "architecture" not in payload:
        required = {"version", "stage", "step", "tokenizer", "tokenizer_checkpoint",
                    "config", "model", "optimizer", "scheduler", "training_state", "rng_states"}
        if set(payload) != required:
            raise ValueError("Invalid local-only Stage2 v2 checkpoint fields")
        from .layout import training_k_probabilities
        return {**payload, "version": 13, "architecture": LOCAL, "k": 1,
                "evaluation_k": 1, "mixed_prefix": False,
                "training_k_probabilities": training_k_probabilities(1, False)}
    return payload
