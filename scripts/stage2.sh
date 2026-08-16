/data/heli/miniconda3/envs/arcodec/bin/torchrun \
    --standalone --nproc_per_node=4 scripts/stage2.py \
    -c configs/stage2.yaml \
    --tokenizer-checkpoint outputs/stage1_h3/checkpoints/checkpoint_best.pt \
    --output-dir outputs/stage2_h3