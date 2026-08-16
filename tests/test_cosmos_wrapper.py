import pytest

from tokenizer.cosmos_dv.encoder import CosmosDVEncoder
from tokenizer.cosmos_dv.decoder import CosmosDVDecoder
from tokenizer.cosmos_dv.model import CosmosDVTokenizer


def test_encoder_rejects_dtype_not_supported_by_jit_checkpoint() -> None:
    with pytest.raises(ValueError, match="BF16-only"):
        CosmosDVEncoder("missing.jit", cosmos_repo="missing", dtype="float32")


def test_decoder_rejects_dtype_not_supported_by_jit_checkpoint() -> None:
    with pytest.raises(ValueError, match="BF16-only"):
        CosmosDVDecoder("missing.jit", cosmos_repo="missing", dtype="float32")


def test_tokenizer_rejects_dtype_not_supported_by_jit_checkpoints() -> None:
    with pytest.raises(ValueError, match="BF16-only"):
        CosmosDVTokenizer(
            "missing-encoder.jit",
            "missing-decoder.jit",
            cosmos_repo="missing",
            dtype="float16",
        )
