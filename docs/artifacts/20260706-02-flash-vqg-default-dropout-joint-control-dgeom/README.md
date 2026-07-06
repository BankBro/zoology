# 20260706-02-flash-vqg-default-dropout-joint-control-dgeom

本 artifact 收尾 default-dropout joint control paired 1ep screen 和 D-direction geometry 诊断. Formal runs 固定 `write_topk=4`, `embed_dropout=0.1`, canonical cache/init/batch order, 并关闭 read trace, hash probe, train inline event trace 和 D-geometry trace. Diagnostic D-geometry targets 单独开启 train-inline scalar trace, 不参与 formal pass/fail 判定.

核心文件:

- `run-summary.csv`: per-run final/best metrics.
- `cross-machine-comparison.csv`: 2080ti/3090 final hard gap by variant.
- `mechanism-metrics-summary.csv`: final validation residual memory/read/write metrics parsed from logs.
- `d-geometry-summary.csv`: D_pack pairwise cosine/rank/update_norm diagnostic rows.
- `cache-init-preflight-summary.csv`: cache/init hash evidence.
- `source-manifest.csv`: mirrored lightweight raw evidence.

关键结果:

| variant | read_topk | 2080ti final 1024x256 | 3090 final 1024x256 | gap | 判定 |
|---|---:|---:|---:|---:|---|
| `r16-update-softcap0p5-injwarm512-rerun` | 16 | `0.901` | `0.945` | `4.4pp` | 高分但严格 fail |
| `r8-update-softcap0p5-injwarm512` | 8 | `0.930` | `0.943` | `1.3pp` | pass |
| `r4-update-softcap0p5-injwarm512` | 4 | `0.837` | `0.955` | `11.8pp` | fail |
| `r2-update-softcap0p5-injwarm512` | 2 | `0.859` | `0.696` | `16.3pp` | fail |
| `r16-injwarm512-only` | 16 | `0.956` | `0.841` | `11.5pp` | fail |

本轮唯一严格过线的 formal 配置是 `r8-update-softcap0p5-injwarm512`. `r16` joint control 高分但 same-seed rerun gap 为 `4.4pp`, 未严格复现 `<=4pp` 屏线. `r16-injwarm512-only` 不过线, 支持 `update softcap + injection warmup` 联合控制比单独 warmup 更合理.

D-geometry 诊断显示 `D_pack=normalize((K-codebook)@addr_proj)` 存在高相关/低 effective-rank 窗口, 尤其 step `64/256` 附近常见 `pair_abs_cos_p95` 接近 `0.98-0.99`. 但该现象在好轨迹和坏轨迹中都出现, 当前只能作为结构性风险信号, 不能单独定为跨机器不稳定根因.
