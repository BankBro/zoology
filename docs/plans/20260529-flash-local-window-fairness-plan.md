# 20260529 Flash local window fairness plan

## 目标

验证 Flash-VQG 相对 GDN 的优势是否被 `block_len=32, local_num_blocks=2` 带来的 64-token exact local attention window 混杂, 并拆分 local exact attention 与 remote/VQ/GD residual 的贡献.

最终报告必须清楚回答:

- Flash-VQG 与 GDN 的当前比较是否可能混入 local exact attention 优势.
- Flash 优势是否主要集中在 `<=64` query-target distance.
- 在超过 local window 的距离桶里, remote/VQ/GD residual 是否仍有稳定独立贡献.
- 是否需要继续做 `GDN + exact local attention` 的更严格 fairness baseline.

## 固定约束

- 工作区: `/home/lyj/mnt/project/zoology`.
- 实验执行机器: 3090, 即 `mclab-3090` / `192.168.2.114` 的 `Flash-VQG-tun` 容器.
- source-of-truth 仓库: 3090 容器内 `/home/lyj/mnt/project/zoology`.
- 基础分支: `flash-vqg`.
- 工作分支: `codex/20260529-flash-local-window-fairness`.
- 后续合入目标: `flash-vqg`.
- 实验名: `20260529-flash-local-window-fairness`.
- 代码修改, 实验脚本, generated configs, artifacts, report, commit 和 push 都在 3090 的 zoology 工作分支上完成.
- 2080ti 上的文件只可作为临时草稿或参考, 不作为正式实验 source-of-truth.
- 当前阶段 0-4 都优先只修改 zoology 仓库, 不修改 `/home/lyj/mnt/project/Flash-VQG`.
- 只有需要改 Flash-VQG core kernel/API, 支持 `local_num_blocks=0`, 或修复 core bug 时, 才另行评估是否修改 Flash-VQG 仓库.
- 一个 3090 GPU 同时只跑一个正式训练, 不并跑正式训练.
- eval-only diagnostic, smoke, aborted, invalid 结果不得混入 formal summary.
- 只有完整跑到 final checkpoint 的正式训练和完整正式 longer-MQAR eval 才写 formal ledger.
- 中止的 3090 `cb128-r8` seed 不作为有效实验.

## 分支策略

只使用一个 zoology 工作分支:

```text
codex/20260529-flash-local-window-fairness
```

该分支覆盖:

- 阶段 0-3 的 Flash local window fairness 主实验.
- 阶段 4 的可选 `GDN + exact local attention` baseline.
- 所有 runner, configs, artifacts, README, metadata, report 和 ledger 适配.

阶段 4 不另开分支, 但必须与阶段 0-3 做 artifact/report 边界隔离. 阶段 4 仍不是主实验最小闭环完成条件.

推荐 git 流程:

```text
cd /home/lyj/mnt/project/zoology
git switch flash-vqg
git pull --ff-only
git switch -c codex/20260529-flash-local-window-fairness
```

如果分支已存在:

```text
git switch codex/20260529-flash-local-window-fairness
```

实验闭环完成后, 将 `codex/20260529-flash-local-window-fairness` 合入 `flash-vqg`.

## 路径组织

脚本和配置:

```text
/home/lyj/mnt/project/zoology/zoology/experiments/flash_vqg/scripts/20260529-flash-local-window-fairness/
/home/lyj/mnt/project/zoology/zoology/experiments/flash_vqg/generated/20260529-flash-local-window-fairness/
```

artifact:

```text
/home/lyj/mnt/project/zoology/docs/artifacts/longer-mqar/local-window-fairness-20260529/
/home/lyj/mnt/project/zoology/docs/artifacts/20260529-flash-local-window-fairness/
/home/lyj/mnt/project/zoology/docs/artifacts/20260529-gdn-exact-local-fairness/  # only if 阶段 4 启动
```

最终报告:

```text
/home/lyj/mnt/project/zoology/docs/20260529-flash-local-window-fairness-report.md
/home/lyj/mnt/project/zoology/docs/20260529-gdn-exact-local-fairness-report.md  # only if 阶段 4 启动
```

## 阶段 0: 代码事实确认

目的: 固化代码层面的公平性假设.

需要确认并在报告中引用:

- `zoology/mixers/flash_vqg.py` 会把 `block_len` 和 `local_num_blocks` 传给 `FlashVQGAttention`.
- 当前 `gd_residual_v1` builder 固定 `block_len=32`, `local_num_blocks=2`, 所以 exact local window 是 64 token.
- Flash-VQG FoX phase2 local path 使用 causal exact local attention.
- 当前 local path 默认使用原始 local `K/V`, 不是只用 quantized keys.
- `if_remote_enabled=False` 时 remote 输出置零, local exact path 仍保留.
- `zoology/mixers/gated_delta_net.py` 中 GDN 只有 short convolution 和 gated delta recurrent state, 没有同等 exact local attention branch.

完成条件:

- 报告中有 `Code evidence` 小节.
- 明确写出结论: 当前 Flash-VQG vs GDN 比较可能混入 64-token exact local attention 优势.

是否需要训练: 不需要.

## 阶段 1: 已有 checkpoint 的 distance-bucket eval

目的: 不训练, 先检查 Flash 相对 GDN 的优势是否集中在 local window 可覆盖的 query-target distance.

模型组:

- GDN: `gdn-h2-ev8`, `gdn-h2-ev10`, `gdn-h2-ev16`.
- Flash full: 现有 official `gd_residual_v1` checkpoint, 优先 `cb64-r16`, `cb256-r4`.
- 第一轮优先 seed `123`.
- 如果现象可能改变结论, 再补 Flash seed `124/125`.

第一轮 slices:

```text
1024x256
2048x512
4096x512
4096x1024
```

第二轮补充 slices:

```text
8190x512
8190x2047
```

distance 定义:

- 主定义: `distance = query_pos - value_pos`.
- sanity 定义: `query_pos - key_pos`, 可计算但不作为主结论口径.
- `query_pos`, `key_pos`, `value_pos` 必须来自 MQAR sample-level position metadata 或等价的生成时记录.
- 禁止只通过 token value 在 `input_ids` 中反查 target 位置, 因为 token 可能重复, 且 `random_non_queries=True` 会引入额外碰撞.
- 当前 single-pass MQAR 中, 第 `j` 个 KV pair 可按生成逻辑得到 `key_pos=2*j`, `value_pos=2*j+1`, `query_pos=context_size+2*gaps[example,j]`.
- 如果未来使用 `num_passes>1`, distance 口径必须先在 plan/report 中补充定义; 默认采用离 query 最近的 value occurrence.

distance buckets:

```text
<=32
33-64
65-128
129-256
257-512
513-1024
1025-2048
2049-4096
>4096
```

每个 bucket 记录:

- `n`
- `correct`
- `accuracy`
- `stderr`
- `ci95_low`
- `ci95_high`
- `Flash_full - GDN`
- bucket 样本数过少时只报告, 不强行下结论.

runner 质量门禁:

- `source_checkpoints.csv` 必须由 `prepare_source_checkpoints` 类脚本自动生成, 来源为 canonical ledger, existing artifact manifest 和 checkpoint metadata.
- eval runner 只接受自动生成的 `source_checkpoints.csv` 作为 checkpoint 输入, 禁止手工散填 checkpoint path.
- 如果 `checkpoint_path` 不存在, final checkpoint 状态不完整, `seed`/`git_commit`/`dtype_policy`/`status` 缺失, 或 source ledger 无法追溯, prepare 阶段必须 fail fast.
- bucket runner 在正式 ablation 前必须先跑 slice-level accuracy sanity check, 复现已有 official longer-MQAR full accuracy.
- sanity check 的 absolute difference 必须 `<=1e-4`; 任一 model/slice 超过阈值时, 该 run 标记为 `invalid`, 不进入 conclusion table, 除非 README 明确解释差异来源并重新定义 official reference.
- `stderr` 使用 binomial standard error `sqrt(p*(1-p)/n)`, `ci95_low` 和 `ci95_high` 可用 normal approximation; `n` 很小时 CI 只作为参考并在 README 说明.
- eval runner 必须支持 OOM batch-size fallback, 遇到 OOM 时按预设序列降低 `eval_batch_size` 直到 1.
- 每次 fallback 必须记录到 metadata/status, 包括原 batch size, 最终 batch size, OOM slice 和 `model_label`.
- `eval_batch_size=1` 仍失败时, 该 model/slice 标记为 `failed` 或 `oom`, 不允许静默跳过或用部分结果冒充完整结果.

判读规则:

- `delta < 0.02`: 不明显.
- `0.02 <= delta < 0.05`: 有趋势, 需要 seed 支撑.
- `delta >= 0.05`: 明显优势.
- `delta >= 0.10`: 很强优势.
- 多 seed 时, `mean_delta >= 0.05` 且 `win_rate >= 2/3` 才称稳定优势.

关键判断:

- 如果 Flash 优势主要在 `<=64`, local exact attention 嫌疑很大.
- 如果 Flash 在 `65-128`, `129-256`, `257-512`, `>512` 仍明显强于 GDN, 说明 remote/VQ/GD residual 不只是依赖 local window.
- `4096x512` 和 `4096x1024` 必须同时看, 用于区分同长度下 KV 密度变化带来的影响.

是否需要训练: 不需要, 只需要已有 checkpoint eval.

## 阶段 2: Flash eval-time local/remote ablation

目的: 用同一个 Flash checkpoint 做推理时开关和窗口大小诊断, 粗拆 local 与 remote 贡献.

配置组:

- `Flash full`: `local_num_blocks=2`, `if_remote_enabled=True`.
- `Flash local-only eval`: `local_num_blocks=2`, `if_remote_enabled=False`.
- `Flash local1 eval`: `local_num_blocks=1`, `if_remote_enabled=True`.
- `Flash local4 eval`: `local_num_blocks=4`, `if_remote_enabled=True`.

注意:

- 这是 eval-time diagnostic, 不是正式训练公平结论.
- 因为 checkpoint 是按 local2/full 训练的, eval 时改 local window 或关 remote 会有分布不匹配.
- `local_num_blocks=0` 当前不支持, 不纳入最小方案.
- 每个 eval-time override 必须保存实际 model config dump, 至少包含 `block_len`, `local_num_blocks`, `if_remote_enabled` 和 checkpoint source.

重点输出表:

```text
slice, bucket, GDN, Flash full, Flash local-only, Flash local1, Flash local4, full-GDN, full-local-only
```

判读规则:

- `Flash local-only` 在 `<=64` 接近 `Flash full`: 近距离优势主要来自 local branch.
- `Flash full - Flash local-only` 在 `>64` 很大: remote/VQ/GD residual 对远距离有贡献.
- `local1 -> local2 -> local4` 随窗口变大明显提高: local window size 是重要混杂变量.

是否需要训练: 不需要, 只需要已有 checkpoint eval.

## 阶段 3: 最小训练 ablation

目的: 消除 eval-time override 的分布不匹配, 用正式训练验证 local window 与 remote/VQ/GD residual 的贡献.

优先模型:

- 第一优先: `cb64-r16`, seed `123`.
- 第二优先: `cb256-r4`.
- 只有结果可能改变结论时, 再补 seed `124/125`.

训练配置:

- `Flash full local2`: 已有 anchor, 不重跑.
- `Flash local-only train`: `block_len=32`, `local_num_blocks=2`, `if_remote_enabled=False`.
- `Flash remote+local1 train`: `block_len=32`, `local_num_blocks=1`, `if_remote_enabled=True`.
- `Flash remote+local4 train`: `block_len=32`, `local_num_blocks=4`, `if_remote_enabled=True`.

训练要求:

- 与现有 official anchor 使用同一训练预算, dtype policy, MQAR train config 和记录规则.
- 3090 正式训练独占 GPU.
- 完整 final checkpoint 才进入 formal training summary.
- 训练失败, OOM, 用户中止只写 status/report, 不进 formal summary.

训练后 eval:

- 原 MQAR eval.
- longer-MQAR official slices.
- distance-bucket eval.

关键比较:

- `local-only train` vs `full local2`: remote/VQ/GD residual 的训练后贡献.
- `remote+local1` vs `remote+local2` vs `remote+local4`: local window size 的训练后贡献.
- `local-only train` vs GDN: local exact attention 单独是否已经解释大量 Flash 优势.
- `remote+local1` vs GDN: 即使 local window 变小, Flash 是否仍有稳定远距离优势.

是否需要训练: 需要.

## 阶段 4: 更严格 GDN fairness baseline, 可选

触发条件:

- 阶段 1-3 显示 Flash 优势大部分集中在 `<=64`.
- 或 `local-only train` 接近 `full local2`.
- 或 `local1/local2/local4` 差异很大.

可选实验:

- 新建 `GDN + exact local attention` baseline.
- 给 GDN 加同等 64-token causal exact local branch, 再与 Flash full 比较.

定位:

- 这是更严格 fairness baseline.
- 不作为第一轮最小闭环完成条件.
- 在同一个 zoology 工作分支实现, 但使用单独 artifact/report 与阶段 0-3 隔离.
- 需要改模型, 需要训练, 需要补充 plan 小节或单独阶段 4 report.

是否需要训练: 需要.

## 阶段 5: artifact 和报告归档

longer-MQAR diagnostic artifact 至少包含:

- `README.md`
- `metadata.json`
- `source_checkpoints.csv`
- `slice_summary.csv`
- `distance_bucket.csv`
- `ablation_summary.csv`
- `status.md`

formal training artifact 至少包含:

- `README.md`
- `metadata.json`
- `train_runs.csv`
- `eval_runs.csv`
- `final_summary.csv`
- `status.md`

建议 CSV schema:

```text
source_checkpoints.csv:
model_label,kind,checkpoint_path,source_run,source_ledger,seed,git_commit,flash_vqg_commit,machine,dtype_policy,status

distance_bucket.csv:
model_label,slice_seq_len,slice_num_kv_pairs,eval_seed,dataset_hash,distance_bucket,n,correct,accuracy,stderr,ci95_low,ci95_high,config_override

slice_summary.csv:
model_label,slice_seq_len,slice_num_kv_pairs,eval_seed,dataset_hash,accuracy,n,official_accuracy_ref,abs_diff_from_ref,sanity_status,eval_batch_size,config_override

train_runs.csv:
run_id,model_label,seed,config_path,checkpoint_path,machine,gpu,exclusive_gpu,start_time,end_time,resume_from,status,notes

final_summary.csv:
model_label,seed,train_status,main_mqar_acc,longer_1024x256,longer_2048x512,longer_4096x512,longer_4096x1024,longer_8190x512,longer_8190x2047
```

metadata 必须记录:

- command
- git commit
- Flash-VQG dependency commit
- dirty status
- branch
- machine
- GPU
- dtype policy
- seed
- dataset hash
- checkpoint path
- config path
- actual eval batch size
- config override dump
- status
- 是否 co-run
- 是否 formal

报告结构:

- `Code evidence`
- `Distance-bucket diagnostics`
- `Flash local/remote ablation`
- `Training ablation`
- `Conclusion and next baseline`

完成条件:

- `docs/20260529-flash-local-window-fairness-report.md` 存在.
- 报告明确区分 formal 与 diagnostic.
- 报告明确回答目标部分列出的四个问题.
- artifact 中有可复查的 CSV, metadata, source manifest 和 README.

## 推荐执行顺序

1. 写 `prepare_source_checkpoints` 和 bucket eval runner, 自动生成 checkpoint source manifest, 并先复现已有 official full accuracy 做 sanity check.
2. 跑第一轮 `1024x256`, `2048x512`, `4096x512`, `4096x1024` 的 GDN vs Flash full.
3. 加 Flash `local-only eval`, `local1 eval`, `local4 eval`.
4. 如果趋势清楚, 补 `8190x512`, `8190x2047`.
5. 再决定是否启动 `cb64-r16 seed123` 的 `local-only`, `local1`, `local4` 训练.
6. 只有训练结果可能改变结论时, 补 seed `124/125`.
7. 最后再判断是否需要阶段 4 的 `GDN + exact local attention`.

## 最小闭环定义

最小可回答闭环:

- 阶段 0 完成.
- 阶段 1 完成第一轮 slices.
- 阶段 2 完成第一轮 slices.
- 产出 diagnostic artifact 和初版报告.

最小正式 fairness ablation 闭环:

- 阶段 0-3 完成.
- 至少 `cb64-r16 seed123` 的 local-only, local1, local4 训练完整到 final checkpoint.
- 所有训练 checkpoint 跑完同一套 longer-MQAR bucket eval.
- 产出 formal artifact 和最终报告.

扩展闭环:

- 阶段 4 完成.
- 产出 `GDN + exact local attention` baseline 的独立 artifact 和报告补充.
