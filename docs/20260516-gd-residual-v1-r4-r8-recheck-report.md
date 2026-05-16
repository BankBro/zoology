# gd_residual_v1 cb256 r4/r8 seed124 复核报告

日期: 2026-05-16

## 1. 摘要

本次只复跑 `gd_residual_v1` dynamic capacity sweep 的最小复核实验:

- `cb256-r4`, `SEED_VALUES=124`, `DATA_SEED=123`
- `cb256-r8`, `SEED_VALUES=124`, `DATA_SEED=123`

目标是判断前一轮 seed123 中出现的 `cb256-r4 > cb256-r8` 非单调现象是否稳定. 本次没有改代码, 没有改 dataset, 没有扩展成新 sweep, 只跑两个 official noearly4ep 复核 run.

阶段结论:

- 两个 run 都 completed, 都完整到 8 个 validation 点, footer success, `exit_code=0`.
- 新 seed 下 `cb256-r4` 没有继续强于 `cb256-r8`. 相反, `cb256-r8` 明显优于 `cb256-r4`.
- seed124 final:
  - `cb256-r4`: `valid/loss=0.341568`, `valid/accuracy=0.949452`, `1024x256=0.687289`
  - `cb256-r8`: `valid/loss=0.0773385`, `valid/accuracy=0.992765`, `1024x256=0.965027`
- 这次复核不支持把 `cb256-r4` 作为稳定强于 `cb256-r8` 的低容量高性价比主线.
- 更严谨的当前判断是: seed123 的 `r4 > r8` 很可能包含 seed/优化动态波动. 至少在当前两个 seed 证据下, `rank=4` 和 `rank=8` 的相对优劣不稳定, 不能称为稳定 rank optimum.
- slice-level 上, seed124 的 r8 优势与 overall accuracy 一致. r8 不只是 hard case 更强, 在 `1024x256`, `512x128`, `512x64`, `num_kv_pairs=128/256`, `input_seq_len=512/1024` 等较难 slice 上都明显优于 r4.

## 2. 仓库状态与测试

Flash-VQG:

- path: `/home/lyj/mnt/project/Flash-VQG`
- branch: `20260428-gd-residual-v1-sync`
- commit: `811e1ce5f140e97d93ad6f1adae07b95b4219143`

zoology:

- path: `/home/lyj/mnt/project/zoology`
- branch: `flash-vqg`
- pre-run commit: `c9741c307709fb17c90b2e59e5736380d2f1e072`

Correctness gate:

| command | result |
|---|---:|
| `pytest tests/test_fox_gd_residual_v1.py -q` | `17 passed` |
| `pytest tests/test_attn_fox_compat.py -q` | `5 passed` |

## 3. 共同配置

两次 run 保持:

| item | value |
|---|---|
| project | `flash_vqg_gd_residual_v1_mqar` |
| max epochs | `4` |
| train batch size | `64` |
| eval batch size | `16` |
| gradient accumulation steps | `4` |
| seed / data seed | `124 / 123` |
| d_model | `128` |
| learning rate | `1e-3` |
| validations per epoch | `2` |
| early stopping | disabled |
| formula | `gd_residual_v1` |
| read_topk / write_topk | `2 / 4` |
| builder / pack | `grouped_chunk_torch_ref / semivec_ref` |
| chunk size | `64` |
| mu_min_count | `0.1` |
| codebook vectors | `256` |
| VQ score / weight / update | `codebook_dot / dense_softmax / grad` |
| VQ softmax tau | `0.25` |
| VQ topk | `4` |

只变化:

| run | rank | GPU | run_id |
|---|---:|---:|---|
| `cb256-r4-s124` | `4` | `0` | `gd-r4-wk4-mu01-t025-cb256-s124-d123-noearly4ep-r4r8-recheck` |
| `cb256-r8-s124` | `8` | `1` | `gd-r8-wk4-mu01-t025-cb256-s124-d123-noearly4ep-r4r8-recheck` |

noearly 通过临时 config-builder 实现, 只把 `early_stopping_metric=None` 和 `early_stopping_threshold=None`; tracked code 未修改.

## 4. Final 结果

| config | rank | capacity | wall-clock | valid/loss | valid/accuracy | 1024x256 acc |
|---|---:|---:|---:|---:|---:|---:|
| `cb256-r4-s124` | `4` | `131,072` | `04:54:35` | `0.341568` | `0.949452` | `0.687289` |
| `cb256-r8-s124` | `8` | `262,144` | `05:03:05` | `0.0773385` | `0.992765` | `0.965027` |

差值, `r8 - r4`:

| metric | delta |
|---|---:|
| `valid/accuracy` | `+0.043313` |
| `1024x256 acc` | `+0.277738` |
| `valid/loss` | `-0.264230` |

seed124 下, `cb256-r8` 在 overall, hard case 和 loss 上都明显优于 `cb256-r4`.

## 5. Epoch-End 曲线

| epoch | r4 loss | r4 acc | r4 1024x256 | r8 loss | r8 acc | r8 1024x256 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | `1.582149` | `0.777299` | `0.140594` | `0.720488` | `0.911697` | `0.564254` |
| 2 | `0.541403` | `0.918368` | `0.542684` | `0.154914` | `0.983642` | `0.920363` |
| 3 | `0.389365` | `0.941946` | `0.647109` | `0.0923385` | `0.990852` | `0.954508` |
| 4 | `0.341568` | `0.949452` | `0.687289` | `0.0773385` | `0.992765` | `0.965027` |

观察:

- `cb256-r8` 从 epoch1 end 开始就领先 r4, 并且优势保持到 epoch4 end.
- r4 也在持续提升, 但 hard case 提升速度明显低于 r8.
- 这说明 seed124 下不是末期波动导致 r8 胜出, 而是全程训练动态都更好.

## 6. 与 seed123 的对照

前一轮 capacity sweep 的 seed123 结果:

| config | seed/data_seed | valid/loss | valid/accuracy | 1024x256 acc |
|---|---|---:|---:|---:|
| `cb256-r4` | `123 / 123` | `0.0961` | `0.986` | `0.938` |
| `cb256-r8` | `123 / 123` | `0.280` | `0.957` | `0.737` |

本次复核 seed124 结果:

| config | seed/data_seed | valid/loss | valid/accuracy | 1024x256 acc |
|---|---|---:|---:|---:|
| `cb256-r4` | `124 / 123` | `0.341568` | `0.949452` | `0.687289` |
| `cb256-r8` | `124 / 123` | `0.0773385` | `0.992765` | `0.965027` |

解释:

- `r4 > r8` 没有在 seed124 复现.
- seed124 中, r8 的 final quality 接近高质量配置, r4 则明显弱于它在 seed123 的表现.
- 当前最合理解释是 `cb256-r4 > cb256-r8` 属于单 seed 非单调现象, 可能来自 seed 波动, 优化动态, rank 与 routing/codebook 的交互, 或 residual branch 使用强度差异.
- 不能根据 seed123 单点把 r4 称为稳定最优 rank; 也不能仅根据 seed124 单点把 r8 称为稳定最优 rank. 但这次复核已经足以否定 “r4 稳定强于 r8” 的表述.

## 7. Slice-Level 差异

### 7.1 mqar_case

| slice | r4 acc | r8 acc | r8 - r4 |
|---|---:|---:|---:|
| `64x4` | `0.999500` | `0.999750` | `+0.000250` |
| `64x8` | `0.999875` | `0.999875` | `+0.000000` |
| `64x16` | `0.999563` | `1.000000` | `+0.000437` |
| `128x32` | `0.994781` | `0.996031` | `+0.001250` |
| `256x64` | `0.991797` | `0.996766` | `+0.004969` |
| `512x64` | `0.978938` | `0.991609` | `+0.012672` |
| `512x128` | `0.943875` | `0.993063` | `+0.049188` |
| `1024x256` | `0.687289` | `0.965027` | `+0.277738` |

### 7.2 input_seq_len

| slice | r4 acc | r8 acc | r8 - r4 |
|---|---:|---:|---:|
| `64` | `0.999646` | `0.999875` | `+0.000229` |
| `128` | `0.994781` | `0.996031` | `+0.001250` |
| `256` | `0.991797` | `0.996766` | `+0.004969` |
| `512` | `0.961406` | `0.992336` | `+0.030930` |
| `1024` | `0.687289` | `0.965027` | `+0.277738` |

### 7.3 num_kv_pairs

| slice | r4 acc | r8 acc | r8 - r4 |
|---|---:|---:|---:|
| `4` | `0.999500` | `0.999750` | `+0.000250` |
| `8` | `0.999875` | `0.999875` | `+0.000000` |
| `16` | `0.999563` | `1.000000` | `+0.000437` |
| `32` | `0.994781` | `0.996031` | `+0.001250` |
| `64` | `0.985367` | `0.994188` | `+0.008820` |
| `128` | `0.943875` | `0.993063` | `+0.049188` |
| `256` | `0.687289` | `0.965027` | `+0.277738` |

Slice-level 结论:

- easy slices 基本都接近饱和, r4/r8 差距很小.
- 差距主要集中在 long-context / high-KV hard slices:
  - `1024x256`: `+0.277738`
  - `512x128`: `+0.049188`
  - `input_seq_len=512`: `+0.030930`
  - `num_kv_pairs=128`: `+0.049188`
- 因此 seed124 下 hard-case 与 overall accuracy 的方向一致: r8 的 overall 优势主要来自更难 slice 的明显提升, 不是某个 easy slice 的统计噪声.

## 8. gd residual metrics

Final validation:

| metric | r4 | r8 |
|---|---:|---:|
| `valid/attn/gd_residual_write_strength_mean` | `0.000510` | `0.012735` |
| `valid/attn/gd_residual_m_norm_mean` | `0.001370` | `0.007332` |
| `valid/attn/gd_residual_m_norm_max` | `0.104011` | `0.703070` |
| `valid/attn/gd_residual_mu_valid_ratio` | `0.323889` | `0.362198` |
| `valid/attn/gd_residual_lambda_mean` | `0.154536` | `0.067582` |
| `valid/attn/gd_residual_inject_ratio` | `0.330866` | `0.186522` |

观察:

- r8 的 residual state norm 和 write strength 明显高于 r4, 说明它在 seed124 下更强地使用 residual branch.
- r4 的 `lambda_mean` 和 `inject_ratio` 更高, 但质量更低. 这提示更强注入不等于更好质量, 需要结合 residual state 表达能力和 routing 动态理解.
- 这也支持 “r4/r8 差异可能来自 rank 与优化动态/branch 使用强度交互” 的解释.

## 9. VQ metrics

Final validation:

| metric | r4 | r8 |
|---|---:|---:|
| `valid/vq/relative_err_mean` | `0.092305` | `0.063932` |
| `valid/vq/c_entropy` | `3.441058` | `3.627937` |
| `valid/vq/c_usage_mean` | `20.337302` | `20.337302` |
| `valid/vq/write_entropy_mean` | `3.027183` | `3.317474` |
| `valid/vq/write_top1_mass_mean` | `0.257043` | `0.192074` |

观察:

- r8 的 VQ relative error 更低, code/write entropy 更高, write top1 mass 更低.
- seed124 下 r8 的 routing/write 分布更分散, 与更好的 hard-case recall 同向.
- 这不是单独证明 rank=8 稳定更优, 但说明本次 seed 中 r8 训练动态明显更健康.

## 10. 结论与下一步

本次复核回答了原问题:

1. 新 seed 下 `cb256-r4` 是否仍强于 `cb256-r8`?
   - 否. seed124 下 `cb256-r8` 明显强于 `cb256-r4`.

2. 如果 r4 仍强, 是否支持把 r4 作为低容量高性价比主线?
   - 不适用. r4 没有在新 seed 下维持优势.

3. 如果 r8 追上或超过 r4, 是否说明前一轮 r4 > r8 主要是 seed/优化波动?
   - 支持这个解释. 当前两次 seed 的方向相反, 所以不能再把 seed123 的 `r4 > r8` 当成稳定 rank 结论.

4. 当前应如何调整主线?
   - `cb256-r16` 仍是当前最强 high-capacity practical anchor.
   - `cb256-r4` 不能单独作为稳定低容量高性价比主线.
   - `cb256-r8` 在 seed124 下表现非常强, 值得与 r4 一起继续复核.

建议下一步:

1. 优先再补 `seed/data_seed=125/123` 或 `125/125` 的 `cb256-r4` 与 `cb256-r8` paired recheck, 让 r4/r8 至少有 3 个 seed 方向.
2. 如果资源有限, 先跑 `seed=125, data_seed=123`, 因为本次已经固定 data_seed=123, 可以更聚焦看 model seed/optimization 波动.
3. 在有 3 seed 前, 不启动 rank=3/4/5/6 局部搜索作为主线; 否则容易围绕一个不稳定局部现象过拟合实验设计.
4. 保留本次 slice-level artifacts, 后续把 seed123/124/125 的 slice-level 差异合并, 判断 r4/r8 分歧是否集中在 `1024x256` 与 `512x128`.

可引用结论:

> 本次 `cb256-r4` vs `cb256-r8` seed124 paired recheck 表明, 前一轮 seed123 中的 `r4 > r8` 非单调现象不稳定. 在相同 official noearly4ep 配置, 相同 `DATA_SEED=123`, 仅改变 `SEED_VALUES=124` 的复核中, `cb256-r8` final `valid/accuracy=0.992765`, `1024x256=0.965027`, 明显优于 `cb256-r4` 的 `0.949452 / 0.687289`. 因此当前不能把 `rank=4` 称为稳定最优 rank, 也不能把 `cb256-r4` 单独作为低容量高性价比主线. 更合理的结论是 `cb256` rank sweep 存在明显 seed/优化动态波动, 后续应补第三个 paired seed 后再决定是否围绕 r4/r8 或 rank=3/4/5/6 做局部搜索.

## 11. Artifacts

提交到远端 GitHub 的小体积 artifacts:

- `docs/artifacts/20260516-gd-r4-r8-recheck/r4-r8-recheck-final.csv`
- `docs/artifacts/20260516-gd-r4-r8-recheck/r4-r8-recheck-epoch-end-valid.csv`
- `docs/artifacts/20260516-gd-r4-r8-recheck/r4-r8-recheck-validation-history.csv`
- `docs/artifacts/20260516-gd-r4-r8-recheck/r4-r8-recheck-slice-level.csv`
- `docs/artifacts/20260516-gd-r4-r8-recheck/r4-r8-recheck-run-manifest.json`

未提交:

- `tmp/20260516-gd-r4-r8-recheck-logs/`
- `tmp/20260516-gd-r4-r8-recheck-status/`
- checkpoints
- SwanLab 本地完整目录
- data cache
- profiler trace
- `__pycache__`
