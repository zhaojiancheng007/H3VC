from codec.ar_stream_row_group import (
    RowGroupEntropyStreamHeader,
    read_row_group_entropy_stream,
    write_row_group_entropy_stream,
)


def test_row_group_stream_round_trip(tmp_path) -> None:
    path = tmp_path / "sample.rgans"
    header = RowGroupEntropyStreamHeader(
        tokenizer_model_id="Cosmos-0.1-Tokenizer-DV4x8x8",
        entropy_model_sha256="a" * 64,
        input_shape=(25, 1080, 1920),
        padded_shape=(25, 1088, 1920),
        latent_shape=(7, 136, 240),
        fsq_levels=(8, 8, 8, 5, 5, 5),
        rows_per_group=4,
    )
    streams = tuple(bytes([dimension]) * (dimension + 1) for dimension in range(6))

    stats = write_row_group_entropy_stream(path, header, streams)
    restored_streams, restored_header = read_row_group_entropy_stream(path)

    assert restored_header == header
    assert restored_streams == streams
    assert stats["payload_bytes"] == sum(map(len, streams))
    assert stats["container_bytes"] == path.stat().st_size
