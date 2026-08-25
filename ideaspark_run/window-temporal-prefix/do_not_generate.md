# Do not generate

Phase 1 stopped because the retrieved corpus is too sparse for a defensible ARCodec-specific mechanism claim. Only 4 of 12 papers directly address learned entropy coding or video compression at the requested level, and none documents ARCodec's categorical tokenizer, four parallel rank steps, or exact encoder-decoder dependency graph.

Remedial steps:

1. Add ARCodec's canonical paper, repository revision, or architecture document defining tokens, rank steps, entropy-model inputs, and decoding order.
2. Re-run retrieval with queries targeting categorical video entropy models, parallel rank or checkerboard decoding, spatially scalable or prefix video coding, and window-level causal context.
3. Supply at least five directly comparable papers or systems, annotated for tokenizer changes, same-frame serialization, self-generated-history training, and full-rate BPP behavior.
4. Specify the prefix unit and evaluation contract: window geometry and order, permissible temporal context, full-rate BPP baseline and tolerance, and latency metric.

The provisional bottleneck diagnosis is preserved in `phase1/phase1_output.json`; candidate generation should not proceed until the literature and baseline anchor are repaired.
