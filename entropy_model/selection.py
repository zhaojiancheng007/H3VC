from .config_categorical import CategoricalEntropyModelConfig
from .config_local_global_memory import LocalGlobalMemoryEntropyConfig
from .model_categorical import CategoricalEntropyModel
from .model_local_global_memory import LocalGlobalMemoryEntropyModel

LOCAL = "unified_spatiotemporal_window_diagonal_v1"
GLOBAL_LOCAL = "canonical_local_window_plus_fifo_global_v3"


def architecture(config):
    return GLOBAL_LOCAL if isinstance(config, LocalGlobalMemoryEntropyConfig) else LOCAL


def build_model(config):
    if isinstance(config, LocalGlobalMemoryEntropyConfig):
        return LocalGlobalMemoryEntropyModel(config)
    if isinstance(config, CategoricalEntropyModelConfig):
        return CategoricalEntropyModel(config)
    raise TypeError(type(config))


def config_from_checkpoint(payload):
    # The global-memory parameter is present only in the Global-local architecture.
    config_type = (LocalGlobalMemoryEntropyConfig
                   if "entropy_model.initial_memory" in payload["model"]
                   else CategoricalEntropyModelConfig)
    return config_type(**payload["config"]["model"])
