# gd_residual_v1 vs GDN noearly4ep 对比报告

日期: 2026-05-15

## 1. 摘要

本报告汇总 2026-05-14 到 2026-05-15 几次 `gd_residual_v1` 研究的结果, 重点补充三类信息:

1. `grouped_chunk_torch_ref` bucketed reference 优化后的 runtime/correctness 结论.
2. `gd_residual_v1 mu01` 禁用 early stopping 后的完整 official 4 epoch 质量结果.
3. 同一 SwanLab project 下补跑的 GDN 两个 baseline: `use_gate=False` 与 `use_gate=True`.

阶段结论:

- `bucketed grouped_chunk_torch_ref` 是成功的工程优化. 它保持 loop oracle recurrence 语义, correctness gate 通过, 并把 `gd_residual/grouped_chunk` CUDA total 从 `2.783s` 降到 `76.141ms`.
- `gd_residual_v1 mu01 noearly4ep` 是当前最强结果: final `valid/accuracy=0.9962060546875`, hard case `1024x256=0.9803125`, final `valid/loss=0.06721608340740204`.
- GDN 新补跑的两个同口径 baseline 中, `use_gate=False` 明显强于 `use_gate=True`: final `valid/accuracy=0.962` vs `0.884`, `1024x256=0.711` vs `0.334`.
- 在近似 trainable parameter 数量口径下, `gd_residual_v1 mu01 noearly4ep` 明显优于 GDN `use_gate=False`. 但这不是等动态容量比较, 因为 `gd_residual_v1` 的 residual matrix dynamic state capacity 约为 GDN KV state 的 `32x`.
- 当前 GDN noearly4ep 的同口径 baseline 应使用 `gated_delta_net-usegate0-s123-d123-noearly4ep`, 或 2026-05-19 的 `64x4` same-seed repeat. 不要把 legacy `gated_delta_net-default-s123-d123` 当作当前 noearly4ep 同口径 baseline.

## 2. 仓库状态

Flash-VQG:

- repo: `BankBro/Flash-VQG`
- branch: `20260428-gd-residual-v1-sync`
- commit: `811e1ce5f140e97d93ad6f1adae07b95b4219143`
- relevant files:
  - `src/flash_vqg/nn/fox/gd_residual.py`
  - `tests/test_fox_gd_residual_v1.py`
  - `tests/test_attn_fox_compat.py`

zoology:

- repo: `BankBro/zoology`
- branch: `flash-vqg`
- base commit before this report: `66b1a9b131caec6c1a4dccd217906f1ef6787d9b`
- this report commit: see final response / Git history after push
- relevant files:
  - `docs/20260514-gd-residual-v1-groupedchunk-bucketed-ref-implementation-plan.md`
  - `docs/20260514-gd-residual-v1-groupedchunk-bucketed-ref-report.md`
  - `docs/20260514-gd-residual-v1-bucketed-smoke-pilot-report.md`
  - `docs/20260514-gd-residual-v1-official-4epoch-mu01-mu015-report.md`
  - `docs/20260515-gd-residual-v1-vs-gdn-noearly4ep-comparison-report.md`
  - `docs/artifacts/20260515-gd-gdn-noearly4ep-comparison/`

## 3. 运行配置口径

`gd_residual_v1 mu01 noearly4ep` 与两个 GDN noearly4ep baseline 保持以下共同训练口径:

| item | value |
|---|---|
| project | `flash_vqg_gd_residual_v1_mqar` |
| max epochs | `4` |
| train batch size | `64` |
| eval batch size | `16` |
| gradient accumulation steps | `4` |
| seed / data seed | `123 / 123` |
| d_model | `128` |
| learning rate | `1e-3` |
| validations per epoch | `2` |
| early stopping | disabled |
| dataset | unchanged MQAR data/cache |

`gd_residual_v1` 额外保持:

| item | value |
|---|---|
| formula | `gd_residual_v1` |
| rank | `16` |
| write_topk / read_topk | `4 / 2` |
| codebook vectors | `256` |
| builder / pack | `grouped_chunk_torch_ref / semivec_ref` |
| chunk size | `64` |
| VQ score / weight / update | `codebook_dot / dense_softmax / grad` |
| VQ softmax tau | `0.25` |
| mu_min_count | `0.1` |

GDN 两个新 baseline 只改变单个注意力模块里的 output gate 设置:

| run | use_gate |
|---|---:|
| `gated_delta_net-usegate0-s123-d123-noearly4ep` | `False` |
| `gated_delta_net-usegate1-s123-d123-noearly4ep` | `True` |

## 4. Run 状态

| run_id | model | status | wall-clock approx | final valid/loss | final valid/accuracy | final 1024x256 |
|---|---|---|---:|---:|---:|---:|
| `gd-r16-wk4-mu01-t025-cb256-s123-d123-noearly4ep` | `gd_residual_v1_mu01` | completed | `05:17:00` | `0.06721608340740204` | `0.9962060546875` | `0.9803125` |
| `gated_delta_net-usegate0-s123-d123-noearly4ep` | `gdn_use_gate_false` | completed | `00:06:23` | `0.345` | `0.962` | `0.711` |
| `gated_delta_net-usegate1-s123-d123-noearly4ep` | `gdn_use_gate_true` | completed | `00:05:27` | `0.729` | `0.884` | `0.334` |

说明:

- 三个 run 都完整执行 4 epoch, 没有 early stopping.
- 两个 GDN run 都在 `flash_vqg_gd_residual_v1_mqar` project 下补跑, 便于和 `gd_residual_v1` 在同一 project 中比较.
- 这次没有重跑 dense baseline 或旧 `gated_delta_net-default-s123-d123` baseline.
- 当前可直接比较的 GDN baseline 是 `gated_delta_net-usegate0-s123-d123-noearly4ep`; 旧 `gated_delta_net-default-s123-d123` 只作为 legacy reference 保留.

## 5. Epoch-end 质量曲线

训练脚本每个 epoch 做 2 次 validation. 下表只列 epoch-end validation.

| model | epoch1 loss/acc/1024x256 | epoch2 loss/acc/1024x256 | epoch3 loss/acc/1024x256 | epoch4 loss/acc/1024x256 |
|---|---|---|---|---|
| `gd_residual_v1_mu01` | `0.231 / 0.987 / 0.931` | `0.109 / 0.994 / 0.964` | `0.0773 / 0.995 / 0.975` | `0.067216 / 0.996206 / 0.980313` |
| `GDN use_gate=False` | `0.694 / 0.931 / 0.516` | `0.424 / 0.955 / 0.660` | `0.365 / 0.960 / 0.699` | `0.345 / 0.962 / 0.711` |
| `GDN use_gate=True` | `1.400 / 0.814 / 0.190` | `0.873 / 0.869 / 0.292` | `0.766 / 0.880 / 0.322` | `0.729 / 0.884 / 0.334` |

主要观察:

- `gd_residual_v1_mu01` 在 epoch1 end 已达到 `valid/accuracy=0.987`, epoch4 end 到 `0.996206`.
- `GDN use_gate=False` 稳定提升, 但 long hard case 到 epoch4 只有 `0.711`.
- `GDN use_gate=True` 训练完成但显著较弱, hard case 到 epoch4 只有 `0.334`.

## 6. Baseline 对比

旧 baseline 没有重跑, 只引用已有已完成 run:

| run/checkpoint | valid/loss | valid/accuracy | 1024x256 acc | notes |
|---|---:|---:|---:|---|
| dense baseline epoch4 | `0.237862` | `0.961423` | `0.774844` | old project, not rerun |
| dense baseline epoch32 final | `0.084111` | `0.981071` | `0.871535` | old project, not rerun |
| old GDN default epoch4 | `0.268798` | `0.972832` | `0.788387` | legacy old project, not current noearly4ep comparison scope |
| old GDN default epoch32 final | `0.072575` | `0.986256` | `0.891031` | legacy old project, not current noearly4ep comparison scope |
| new GDN use_gate=False epoch4 | `0.345` | `0.962` | `0.711` | same new project, noearly4ep |
| new GDN use_gate=True epoch4 | `0.729` | `0.884` | `0.334` | same new project, noearly4ep |
| `gd_residual_v1_mu01` noearly epoch4 | `0.067216` | `0.996206` | `0.980313` | same new project, noearly4ep |

结论:

- 对比新补跑的同 project GDN baseline, `gd_residual_v1_mu01` 明显更强.
- 对比旧 32 epoch GDN default final, `gd_residual_v1_mu01` 在 4 epoch 内仍取得更高 `valid/accuracy` 和更高 hard case accuracy.
- 如果只看 final `valid/loss`, `gd_residual_v1_mu01` 也低于旧 GDN default final 的 `0.072575`, 达到 `0.067216`.
- 新 GDN `use_gate=False` 低于旧 `gated_delta_net-default-s123-d123` 的 epoch4 和 final32. 后续 2026-05-19 的 `64x4` same-seed repeat 已复现 `use_gate=False` noearly4ep, 没有复现 legacy default. 因此旧 default 不应作为当前 noearly4ep 同口径 baseline 使用.

## 7. 参数量和动态容量 caveat

| model | trainable params | dynamic state capacity | notes |
|---|---:|---:|---|
| `gd_residual_v1_mu01` | `1,184,966` | `524,288` | residual matrix capacity only, approx `32x` GDN |
| `GDN use_gate=False` | `1,167,878` | `16,384` | strongest new GDN baseline |
| `GDN use_gate=True` | `1,200,646` | `16,384` | output gate adds params, not state capacity |

公平性表述建议:

- 可以说这是近 trainable parameter 数量的比较.
- 不应说这是等动态容量比较.
- `gd_residual_v1` 的优势可能来自 recurrence 设计, VQ routing, 以及更大的 dynamic residual matrix capacity. 下一步若要更严格归因, 需要设计 capacity-controlled ablation.

## 8. grouped_chunk runtime 结论

本质量对比建立在 2026-05-14 的 bucketed reference 优化之后. 关键 runtime 结论如下:

| metric | old loop reference | bucketed reference |
|---|---:|---:|
| B8 `gd_residual/grouped_chunk` CUDA total | `2.783s` | `76.141ms` |
| B64 avg microbatch sec | `80.005957s` | `1.195595s` |
| B64 avg backward sec | `57.025518s` | `0.594279s` |
| B64 peak reserved GiB | `8.501953` | `8.431641` |

这支持 runtime 结论: `grouped_chunk` 已不再是 `gd_residual_v1` 的主要 runtime 瓶颈. 当前质量结论来自 full 4 epoch run, 不是 profile/smoke/pilot 的替代解释.

## 9. Artifacts

本次选择提交给远端 GitHub 的小体积数据:

- `docs/artifacts/20260515-gd-gdn-noearly4ep-comparison/epoch-end-valid.csv`
- `docs/artifacts/20260515-gd-gdn-noearly4ep-comparison/final-comparison.csv`
- `docs/artifacts/20260515-gd-gdn-noearly4ep-comparison/model-capacity-and-params.csv`
- `docs/artifacts/20260515-gd-gdn-noearly4ep-comparison/run-manifest.json`

继续引用已存在 artifacts:

- `docs/artifacts/20260514-gd-groupedchunk-bucketed-ref/`
- `docs/artifacts/20260514-gd-bucketed-smoke-pilot/`
- `docs/artifacts/20260514-gd-residual-v1-official-4epoch-mu01-mu015/`

未提交:

- `tmp/` 下完整日志.
- checkpoint: `best.pt`, `last.pt`.
- SwanLab 本地运行目录.
- data cache.
- 临时 config-builder wrapper.

## 10. 下一步建议

优先级建议:

1. 让网页版 ChatGPT 审阅代码和本报告, 判断新 GDN `use_gate=False` 为什么低于旧 GDN default baseline.
2. 对 `gd_residual_v1` 做 capacity caveat 下的正式表述: 先声明近参数量优势, 再声明非等动态容量.
3. 若要归因更严格, 做 `gd_residual_v1` capacity ablation: 降低 codebook slots/rank 或增加 GDN state size, 比较同等 dynamic state budget 下的结果.
4. 若只追求质量, 可以把 `gd_residual_v1 mu01 noearly4ep` 作为当前主结果, 后续再探索 `mu_min_count`, rank, codebook size 的质量/容量 tradeoff.

## 11. Web ChatGPT 阅读入口

推荐网页版 ChatGPT 先读:

1. `BankBro/zoology`, branch `flash-vqg`, this report and artifacts:
   - `docs/20260515-gd-residual-v1-vs-gdn-noearly4ep-comparison-report.md`
   - `docs/artifacts/20260515-gd-gdn-noearly4ep-comparison/`
2. 再读前置文档:
   - `docs/20260514-gd-residual-v1-groupedchunk-bucketed-ref-report.md`
   - `docs/20260514-gd-residual-v1-bucketed-smoke-pilot-report.md`
   - `docs/20260514-gd-residual-v1-official-4epoch-mu01-mu015-report.md`
3. 读实现代码:
   - `BankBro/Flash-VQG`, branch `20260428-gd-residual-v1-sync`, commit `811e1ce5f140e97d93ad6f1adae07b95b4219143`
   - `src/flash_vqg/nn/fox/gd_residual.py`
   - `tests/test_fox_gd_residual_v1.py`
