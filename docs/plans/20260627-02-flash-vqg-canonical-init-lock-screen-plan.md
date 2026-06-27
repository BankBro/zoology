# 20260627-02 Flash-VQG Canonical Init-Lock Screen Plan

updated: 2026-06-27
status: planned
experiment_id: `20260627-02-flash-vqg-canonical-init-lock-screen`

## 目标

本轮是 debug hygiene, 不是方法实验, 不写 official ledger.

目标只回答:

```text
在 MQAR cache 内容相同, 初始模型 state_dict 相同后, 3090 上 cb64-r16 default s123 的 1 epoch 低分/不稳是否明显缓解.
```

## 已有前提

- `20260626-01` 已把 2080ti 的 13 个 cache 文件作为 canonical 复制到 3090.
- 3090 侧已验证 13/13 content-level cache match.
- `20260627-01` early-step hash probe 显示: cache 和 first batch 一致后, 当前跨机第一分叉点仍在模型初始化, 主要是 CUDA-side embedding / lm_head / codebook init.

## 非目标

- 不追求跨 GPU bitwise training 完全一致.
- 不把 init-lock 当训练稳定性方法.
- 不跑 4 epoch.
- 不跑 official longer-MQAR.
- 不展开 s123/s124/s125 大矩阵.
- 不改 read/write/beta/guard 机制.

## 实验材料

canonical source:

- 机器: `2080ti` 的 `Flash-VQG-tun` 容器.
- 仓库: `/home/lyj/mnt/project/zoology`.
- 分支: `flash-vqg`.
- cache: `./data/flash_vqg` 中本轮实际会加载的 13 个 canonical `data_*.pt`.

canonical init:

- 在 2080ti 上用本轮 `s123` 配置构建 `LanguageModel`.
- 立即保存未训练的完整 `model.state_dict()`.
- checkpoint payload 必须包含:
  - `model_state_dict`.
  - `model_state_sha256`.
  - `per_tensor_sha256`.
  - config summary.
  - git branch/commit.
- 复制到 3090 后, 3090 必须对 checkpoint 内 tensor hash 做相同校验.

## 预检硬门槛

两台机器启动任何 GPU 任务前, 必须在对应宿主机的 `Flash-VQG-tun` 容器内确认:

- `nvidia-smi` / NVML 可用.
- `torch.cuda.is_available()` 为 true.
- `zoology` 分支和 commit 与 canonical source 一致.
- `Flash-VQG` 分支和 commit 与 canonical source 一致.
- 本轮 cache content hash 一致.
- canonical init checkpoint tensor hash 一致.

任一项失败则停止, 不启动训练.

## Step 1: init-lock early-step probe

在 2080ti 和 3090 各跑一次 `20260627-01` early-step hash probe, 参数相同:

- `s123`.
- canonical cache.
- canonical init checkpoint.
- 默认 dtype policy.
- 不额外开启 deterministic/TF32 改动.

通过标准:

- `inputs_sha256` match.
- `targets_sha256` match.
- `after_model_to_device_before_optimizer_step/model_params_sha256` match.

若 model hash 不 match, 说明 init-lock 没生效, 停止.

forward/logits/grad/optimizer hash 若后续分叉, 记录为数值路径分叉, 但仍可进入最小 1 epoch screen, 因为本轮目标是效果 screen, 不是 bitwise proof.

## Step 2: 最小 1 epoch screen

只跑:

| 机器 | target | repeat | 目的 |
|---|---|---:|---|
| 2080ti | `default-s123-r1` | 1 | canonical high baseline |
| 3090 | `default-s123-r1` | 1 | init-lock 后问题点 repeat 1 |
| 3090 | `default-s123-r2` | 2 | init-lock 后问题点 repeat 2 |

公共配置:

- layout: `cb64-r16`.
- `data_seed=123`.
- train seed: `123`.
- `read_topk=2`.
- `num_codebook_vectors=64`.
- `gd_residual_rank=16`.
- `train_batch_order=global_shuffle`.
- `train_batch_size=64`.
- `eval_batch_size=16`.
- `gradient_accumulation_steps=4`.
- `max_epochs=1`.
- `validations_per_epoch=4`.
- `disable_early_stopping=true`.
- `cache_dir=./data/flash_vqg`.
- `init_checkpoint_path=<canonical-init-checkpoint>`.

## 判读规则

对照基线来自 `20260626-01` canonical-cache-only rerun:

```text
3090 s123: 0.865, 0.554, 0.927, 0.756
mean 0.7755
gap 0.373
```

如果 init-lock 后 3090 两个 run 均 `>=0.90`, 且 repeat gap 明显小于 `0.373`, 则说明此前 3090 s123 低/不稳有相当部分来自 init mismatch.

如果 init-lock 后 3090 仍明显低或 repeat gap 仍大, 则说明:

```text
cache/init 不是主要解释, 训练数值路径或优化过程仍足以把该配置推向不同 basin.
```

无论哪种结果, 本轮都不直接推出新的稳定训练方法. 下一步仍需回到 `20260605+` evidence ledger 中的 read-side gate 或 write/state guarded release 方向.

## 产物

脚本目录:

```text
zoology/experiments/flash_vqg/scripts/20260627-02-flash-vqg-canonical-init-lock-screen/
```

本地临时输出:

```text
zoology/experiments/flash_vqg/scripts/20260627-02-flash-vqg-canonical-init-lock-screen/outputs/
```

正式 artifact:

```text
docs/artifacts/20260627-02-flash-vqg-canonical-init-lock-screen/
```

报告:

```text
docs/20260627-02-flash-vqg-canonical-init-lock-screen-report.md
```
