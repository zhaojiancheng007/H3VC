/data/heli/miniconda3/envs/arcodec/bin/torchrun \
    --standalone --nproc_per_node=4 scripts/stage3.py \
    -c configs/stage3.yaml \
    --tokenizer-checkpoint outputs/stage1_h3/checkpoints/checkpoint_best.pt \
    --entropy-checkpoint outputs/stage2_h3/checkpoints/checkpoint_best.pt \
    --output-dir outputs/stage3_h3