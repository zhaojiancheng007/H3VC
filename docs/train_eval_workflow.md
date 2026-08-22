# ARCodec Train-Eval 固定流程

本文固定当前实验流水线。后续训练、定期评测、恢复训练和结果归档均以本文为准。
基线入口 `scripts/stage1.py`、`scripts/stage2.py`、`scripts/stage3.py` 不属于本文流程。

## 1. 唯一运行入口

| Stage | 训练入口 | 配置 | 训练对象 |
| --- | --- | --- | --- |
| Stage1 | `scripts/stage1_cosine.py` | `configs/stage1_cosine.yaml` | tokenizer、projection、EMA-VQ |
| Stage2 | `scripts/stage2_cosine.py` | `configs/stage2_cosine.yaml` | categorical entropy model，tokenizer 冻结 |
| Stage3 | `scripts/stage3_cosine.py` | 对应 λ 的 cosine config | tokenizer、entropy model 与 discriminator 联合训练 |

三个入口统一使用：

- `world_size` 卡 DDP 训练，GPU 数由 `torchrun --nproc_per_node` 指定；
- high-throughput OpenCV worker loader；
- cosine annealing learning-rate scheduler；
- 每 `checkpoint_interval` step 保存 checkpoint，并使用全部训练 GPU 执行 Gaussian eval；
- eval 完整成功后才恢复训练。

当前 λ=18 的 Stage3 配置为 `configs/stage3_lambda18_cosine.yaml`。

## 2. 训练命令模板

### Stage1

~~~bash
torchrun --standalone --nproc_per_node=<gpu_count> scripts/stage1_cosine.py \
  --tokenizer minimax_h3 \
  -c configs/stage1_cosine.yaml \
  --output-dir outputs/<stage1_run>
~~~

### Stage2

~~~bash
torchrun --standalone --nproc_per_node=<gpu_count> scripts/stage2_cosine.py \
  -c configs/stage2_cosine.yaml \
  --tokenizer-checkpoint outputs/<stage1_run>/checkpoints/checkpoint_best.pt \
  --output-dir outputs/<stage2_run>
~~~

### Stage3

~~~bash
torchrun --standalone --nproc_per_node=<gpu_count> scripts/stage3_cosine.py \
  -c configs/stage3_lambda18_cosine.yaml \
  --tokenizer-checkpoint outputs/<stage1_run>/checkpoints/<stage1_checkpoint>.pt \
  --entropy-checkpoint outputs/<stage2_run>/checkpoints/<stage2_checkpoint>.pt \
  --output-dir outputs/<stage3_lambda18_run>
~~~

`<gpu_count>` 是每次启动时选择的 GPU 数，不是仓库固定设置；新实验可按机器资源设为 4、8 或其他可用数量。
同一 run 从 checkpoint 恢复时，必须保持 checkpoint 记录的 `world_size` 和 global batch size。
因此八卡机器可以启动新的八卡实验，但不能用八个 rank 直接恢复一个四卡 checkpoint。

恢复训练时仅增加：

~~~bash
--resume outputs/<run>/checkpoints/<checkpoint>.pt
~~~

Stage3 的 `--resume` 恢复模型、discriminator、两个 optimizer、数据位置和各 rank
RNG 状态；cosine scheduler 根据 checkpoint step 重建到对应位置。
`--init-checkpoint` 只加载模型与 discriminator 作为初始化，不恢复 optimizer、
数据位置或 RNG 状态。两者不能同时使用。

## 3. Checkpoint 边界状态机

默认 `checkpoint_interval=5000`。到达 checkpoint step 后执行以下固定顺序：

~~~text
完成当前 train step
→ 写入 train_metrics.jsonl
→ 保存完整 checkpoint
→ optimizer 清零
→ 当前 Stage 占用 GPU 的训练模型与 optimizer state 移到 CPU
→ Stage1/3 的 LPIPS training metric 也移到 CPU
→ 释放训练 GPU 显存
→ 每个训练 rank 分别启动一个独立 eval subprocess
→ 等待 `world_size` 个 eval shard 全部成功
→ rank 0 校验并合并结构化指标
→ Stage3 提升 canonical recon；Stage1/2 不保存 recon
→ 模型和 optimizer 移回 GPU
→ 重建 DDP wrapper
→ 从下一个 train step 继续
~~~

任何 eval subprocess 非零退出都会使 `subprocess.run(..., check=True)` 直接失败。
此时不合并部分指标、不替换 canonical recon、也不恢复训练。

## 4. 多卡 Eval 分片

共享评测集合共 42 个视频：

| Dataset | 视频数 |
| --- | ---: |
| UVG | 7 |
| MCL-JCV | 30 |
| HEVC-B | 5 |

所有视频先组成一个固定顺序的 job list，再按 `jobs[rank::world_size]` 分片。
分片单位是完整视频，不按 dataset 分卡。每个 rank 分到 ⌊42 / world_size⌋ 或
⌈42 / world_size⌉ 个视频，数量最多相差 1。例如四卡为 `11 / 11 / 10 / 10`，
八卡为 `6 / 6 / 5 / 5 / 5 / 5 / 5 / 5`。

Eval 入口：

| Stage | Eval subprocess |
| --- | --- |
| Stage1 | `scripts/evaluate_stage1_tokenizer_gaussian_distributed.py` |
| Stage2 | `scripts/evaluate_stage2_cosine_gaussian_distributed.py` |
| Stage3 | `scripts/evaluate_stage3_cosine_gaussian_distributed.py` |

每个 subprocess 只暴露一张物理 GPU，并在其中使用 `cuda:0`。共享命令构造位于
`training/periodic_benchmarks_gaussian.py`。

## 5. Eval 协议

MiniMax-H3 的固定协议为：

- 从每个视频读取 96 帧；
- 空间尺寸补齐到 16 的倍数；
- 末帧复制补齐到模型输入的 107 帧；
- decode 后裁回原始空间尺寸和前 96 帧；
- distortion 与 perceptual metrics 只在这 96 帧上计算；
- BPP 分母固定使用 96 帧像素数。

Decoder spatial tiles 使用 Gaussian blend：

- `type = gaussian`；
- `gaussian_variance = 0.01`；
- `scope = decoder_spatial_tiles`。

Stage 间 rate 口径不同：

| Stage | BPP 口径 |
| --- | --- |
| Stage1 | 每个 4096-way index 固定 12 raw bits，不使用 entropy coding |
| Stage2 | categorical entropy model 估计的 cross-entropy bits |
| Stage3 | categorical entropy model 估计的 cross-entropy bits |

每个 dataset 的 aggregate JSON 必须包含：

- `psnr`；
- `bpp`；
- `latency_ms`；
- `lpips_vgg`；
- `lpips_alex`；
- `dists`。

`latency_ms` 统计 encode、VQ assignment、rate estimation 和 decode，按输出帧平均。
它不包含 PNG 写盘，也不包含 LPIPS/DISTS 计算。

## 6. Shard 合并条件

Rank 0 只在以下条件全部成立时合并：

- `world_size` 一致；
- rank 恰好覆盖 `0..world_size-1`；
- tokenizer 一致；
- checkpoint step 一致；
- protocol 完全一致；
- 42 个预期视频全部存在；
- 没有重复或额外视频。

合并逻辑位于 `training/evaluation_categorical_distributed.py`。不允许缺视频后继续、
不允许用旧结果补缺失 shard，也不允许用默认值生成 aggregate。

## 7. Recon 归档规则

只有 Stage3 保存 recon。Stage1 和 Stage2 只保存结构化指标。

Stage3 eval 期间，所有 rank 先写入当前 checkpoint 的临时目录：

~~~text
evaluations/step_<step>/recon/<dataset>/<sequence>/000001.png
...
evaluations/step_<step>/recon/<dataset>/<sequence>/000096.png
~~~

`world_size` 个 shard 全部成功并完成指标合并后，rank 0 将整个临时 recon 目录提升为：

~~~text
evaluations/recon/<dataset>/<sequence>/*.png
~~~

一个 Stage3 output directory 对应一个 λ，因此 `evaluations/recon` 是该 λ 唯一的
canonical recon。后续 checkpoint 成功 eval 后整体覆盖上一 checkpoint 的 recon。
step 目录继续保留每个 checkpoint 的指标 JSON，但不保留重复 recon。

每个 step 的 dataset JSON 使用以下相对路径指向 canonical recon：

~~~json
{
  "protocol": {
    "reconstruction": "../recon/<dataset>"
  }
}
~~~

覆盖只发生在全部 `world_size` 个 shard 成功之后。失败 eval 的部分 recon 不得替换上一份完整结果。

## 8. 输出目录

~~~text
outputs/<run>/
├── run_manifest.json
├── resume_<step>.json
├── train_metrics.jsonl
├── training.log
├── checkpoints/
│   └── ...
└── evaluations/
    ├── recon/                         # 仅 Stage3；当前 λ 的唯一 recon
    │   ├── uvg/
    │   ├── mcl_jcv/
    │   └── hevc_b/
    └── step_<step>/
        ├── uvg.json
        ├── mcl_jcv.json
        ├── hevc_b.json
        └── shards/
            ├── rank_00.json
            ├── ...
            └── rank_<world_size-1>.json
~~~

Checkpoint 策略：

| Stage | 策略 |
| --- | --- |
| Stage1 | 根据 UVG aggregate LPIPS-VGG 更新 `checkpoint_best.pt` |
| Stage2 | 根据 UVG aggregate estimated BPP 更新 `checkpoint_best.pt` |
| Stage3 | 保留每个 `checkpoint_interval` 的 `step_<step>.pt` |

Stage1/2 的非最优 candidate checkpoint 会删除；结构化 eval JSON 仍保留。

## 9. 完整实验 Setting 归档规则

归档单位是一次完整实验 setting，不是单独的 Stage 或单独的 eval。一个完整 setting
必须包含一个确定的 Stage1、一个由该 Stage1 训练的 Stage2，以及一个或多个使用这组
Stage1/Stage2 checkpoint 的 Stage3 λ 分支。

### 9.1 命名和目录

归档根目录固定为：

~~~text
results/<setting_name>/
~~~

`<setting_name>` 必须描述实验本身，至少包含 tokenizer/backbone、codebook setting、
选定的 Stage1/Stage2 step 和实验日期。不得使用 `archive`、`eval_archive` 等不包含
实验信息的通用名称。例如：

~~~text
results/minimax_h3_vq4096x24_s1_65k_s2_25k_20260816/
~~~

固定目录结构为：

~~~text
results/<setting_name>/
├── experiment.json
├── stage1/
│   └── checkpoints/
│       └── <selected_stage1_checkpoint>.pt
├── stage2/
│   └── checkpoints/
│       └── <selected_stage2_checkpoint>.pt
└── stage3/
    ├── lambda_<value>/
    │   ├── checkpoints/
    │   │   └── <selected_stage3_checkpoint>.pt
    │   └── eval/
    │       ├── periodic/              # 可选；历史 checkpoint 的结构化指标
    │       └── step_<step>/           # 当前正式归档 eval
    │           ├── <dataset metrics>
    │           └── <dataset recon>
    └── lambda_<other_value>/
        └── ...
~~~

λ 目录使用规范化名称，例如 `lambda_12`、`lambda_18`、`lambda_24`。同一组
Stage1/Stage2 checkpoint 的不同 λ 必须放在同一个 setting 的 `stage3/` 下，不能拆成
多个平行归档根目录。

### 9.2 Checkpoint 保留

- Stage1 和 Stage2 被 Stage3 引用的 source checkpoint 是归档的一部分，必须保留，
  不得因清理 `outputs` 或 checkpoint 去重而删除；
- 每个 Stage3 λ 必须保留生成正式 eval 的 checkpoint；
- 若同一 `checkpoints/` 目录有多个 `step_<N>.pt`，默认只保留数值最大的 step，
  但被下游 Stage、正式 eval 或 `experiment.json` 引用的 checkpoint 优先受保护；
- `checkpoint_best.pt`、`last.pt` 和其他非 `step_<N>.pt` 文件不属于数值 step 去重规则，
  不得顺带删除；
- 删除任何 checkpoint 前必须先解析精确引用关系。存在引用不明确时直接停止，不允许
  猜测或通过 fallback 修复。

### 9.3 Stage3 Eval 和 Recon

每个 λ 的正式 eval 必须放在与 checkpoint step 对应的目录：

~~~text
stage3/lambda_<value>/eval/step_<step>/
~~~

该目录必须同时包含结构化 metrics 和该 λ 当前唯一的 canonical recon。历史 checkpoint
的 metrics 可以保留在 `eval/periodic` 或其他 `eval/step_<old_step>` 目录，但同一个 λ
不得保存多套 recon。新的 checkpoint 完整 eval 成功后，使用新 recon 整体替换旧 recon；
不得混合两个 checkpoint 的帧。

正式归档 metrics 必须包含：

- `psnr`；
- `bpp`；
- `latency_ms`；
- `lpips_vgg`；
- `lpips_alex`；
- `dists`。

Stage1 和 Stage2 不归档 recon，只保留 checkpoint、训练记录和结构化 eval metrics。

### 9.4 experiment.json

每个 setting 根目录必须包含可解析的 `experiment.json`，作为归档的结构化索引。它至少记录：

- setting 名称、tokenizer/backbone 和 codebook setting；
- Stage1、Stage2 的 step 和相对 checkpoint 路径；
- 每个 Stage3 λ 的 λ 值、step、相对 checkpoint 路径和正式 eval 路径；
- 每个 dataset 的 metrics、recon 相对路径和 recon 帧数；
- 固定 metrics 列表；
- 每个 λ 只保留一份 recon 的覆盖策略。

索引中的路径必须相对于 setting 根目录或对应的 Stage3 eval 根目录，不使用机器相关的
绝对路径。

### 9.5 移动和兼容路径

已完成的实验实体目录从 `outputs` 移动到 `results/<setting_name>`。同一文件系统内使用
rename，不复制大 checkpoint 或 recon。若训练脚本、manifest 或 resume 流程仍依赖旧的
`outputs/<run>` 路径，可以在 `outputs` 保留指向新实体目录的兼容符号链接；
`results/<setting_name>` 始终是唯一归档真源。

移动前必须确认没有 train/eval 进程仍在写目标目录。移动后必须验证符号链接可以解析到
真实 checkpoint，不能留下断链，也不能使用缺失路径的静默 fallback。

### 9.6 归档完成条件

一次 setting 只有在以下条件全部满足后才算归档完成：

1. `stage1`、`stage2` 和每个已归档 λ 的 checkpoint 均存在；
2. 每个正式 eval 的 checkpoint step 与目录名一致；
3. 所有 dataset metrics JSON 可解析并包含六项固定 metrics；
4. 每个 recon sequence 恰好包含 96 帧，汇总帧数与 dataset 视频数一致；
5. 同一个 λ 只存在一套正式 recon；
6. `experiment.json` 可解析，且其中所有相对路径和帧数均与文件系统一致；
7. `results` 下不存在 `archive`、`eval_archive` 等通用归档层级；
8. 所有必要的 `outputs` 兼容链接均可解析。

任何一项失败都必须停止归档并暴露具体不一致；不得移动部分结果后生成占位索引，也不得
用旧 metrics 或旧 recon 补齐缺失数据。

## 10. Resume 一致性规则

各 Stage 的恢复状态为：

| Stage | Checkpoint 中保存的恢复状态 |
| --- | --- |
| Stage1 | model、optimizer、step；学习率和数据 epoch/batch position 由 step 确定性重建 |
| Stage2 | model、optimizer、scheduler、数据 epoch/batch position、各 rank RNG state |
| Stage3 | model、discriminator、两个 optimizer、数据 epoch/batch position、各 rank RNG state；scheduler 由 step 重建 |

恢复前必须满足：

- resume checkpoint 的 scientific config 与当前 config 一致；
- tokenizer、entropy checkpoint 和 tokenizer 类型一致；
- DDP world size 与 global batch size 一致；
- `train_metrics.jsonl` 最后一个 step 等于 resume checkpoint step；
- Stage2/3 checkpoint 内的数据 epoch、batch position 和各 rank RNG state 完整。

Resume config 比较只将 `train.num_workers` 视为 runtime-only 字段；其他 scientific config
不允许静默变化。当前 fast loader 仍明确要求每个 rank 使用 4 个 worker，且
`prefetch_factor=2` 固定在实现中，因此当前流程不得修改这两个值。

如果进程曾从 checkpoint 短暂恢复并写出了新 metrics，但尚未生成对应的新 checkpoint，
这些 metrics 属于 uncheckpointed records。固定处理方法是：

1. 将 checkpoint step 之后的 JSONL 记录单独归档；
2. 让主 `train_metrics.jsonl` 严格结束于 checkpoint step；
3. 再次使用该 checkpoint 恢复。

不得直接忽略 step mismatch，也不得伪造缺失 checkpoint。

## 11. 运行确认

一次 train-eval-resume 周期只有在以下条件同时满足时才算完成：

1. checkpoint 文件存在且 step 正确；
2. `world_size` 个 shard JSON 均存在；
3. UVG、MCL-JCV、HEVC-B aggregate JSON 均可解析；
4. aggregate 包含 `psnr`、`bpp` 和 `latency_ms`；
5. Stage3 canonical recon 覆盖 42 个视频 × 96 帧；
6. `train_metrics.jsonl` 出现 checkpoint step 之后的新 step；
7. `world_size` 个训练 rank 均存活并占用对应 GPU；
8. eval 和 resume 日志中没有 traceback、OOM 或 subprocess failure。

只看到 tmux session 或 Python 进程不等于训练已经恢复，必须以新的 train metric 为准。

## 12. 代码责任边界

| 责任 | 文件 |
| --- | --- |
| Stage1 train-eval orchestration | `scripts/stage1_cosine.py` |
| Stage2 train-eval orchestration | `scripts/stage2_cosine.py` |
| Stage3 train-eval-recon orchestration | `scripts/stage3_cosine.py` |
| Gaussian distributed command | `training/periodic_benchmarks_gaussian.py` |
| 42-video job、分片、指标与合并 | `training/evaluation_categorical_distributed.py` |
| Gaussian tile decode | `tokenizer/gaussian_tile_decode.py` |
| Resume config compatibility | `training/checkpoint_config_fast.py` |
| High-throughput loader | `training/stage_data_fast.py`、`training/distributed_runtime_fast.py` |
| 回归测试 | `tests/test_distributed_gaussian_eval.py`、`tests/test_fast_stage_loading.py`、`tests/test_gaussian_tile_decode.py`、`tests/test_cosine_schedulers.py` |

修改上述流程时，必须同步更新本文和对应测试。
