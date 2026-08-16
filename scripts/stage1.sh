/data/heli/miniconda3/envs/arcodec/bin/torchrun \
    --standalone --nproc_per_node=4 scripts/stage1.py \
    --tokenizer minimax_h3 \
    -c configs/stage1.yaml \
    --output-dir outputs/stage1_h3