# 20260706-02 Flash-VQG default dropout joint control and D-geometry report

## 摘要

本轮在 default dropout 下继续检查 Flash-VQG / `gd_residual_v1` 的跨机器稳定性. Formal 训练固定 `seed=124`, `data_seed=123`, canonical MQAR cache, canonical seed124 init, same batch order, `cb64-r16`, `write_topk=4`, `embed_dropout=0.1`, `resid_dropout=0.0`, `drop_path=0.0`, `max_epochs=1`, `704` optimizer steps. 机器为 2080ti + 3090 paired.

核心结论:

1. `r8-update-softcap0p5-injwarm512` 是本轮唯一严格过线的 formal 配置: 2080ti `0.930`, 3090 `0.943`, gap `1.3pp`.
2. 上轮唯一过线配置 `r16-update-softcap0p5-injwarm512` 本轮复跑得到 2080ti `0.901`, 3090 `0.945`, gap `4.4pp`. 两机都高分, 但按本轮 `<=4pp` 规则不算严格复现.
3. `r16-injwarm512-only` 不过线: 2080ti `0.956`, 3090 `0.841`, gap `11.5pp`. 这说明 residual injection warmup 单独不够, update softcap 仍然有价值.
4. `r4/r2` joint control 没有救回来. `r4` 是 `0.837/0.955`, gap `11.8pp`; `r2` 是 `0.859/0.696`, gap `16.3pp`. 当前方案仍依赖适度宽的 read support, 不能直接推到窄 read support.
5. D-geometry 诊断显示 `D_pack=normalize((K-codebook)@addr_proj)` 的写入方向确实经常高度相关, 在部分 step 上 `pair_abs_cos_p95` 接近 `0.98-0.99`, effective rank 接近 `1`. 但这种现象在好轨迹和坏轨迹里都存在, 当前还不能证明它单独解释 final accuracy gap.

一句话: `update softcap + injection warmup` 是当前最强的正信号, 但不是已经稳定解决问题. 本轮最值得继续复验的是 `read_topk=8` joint control, 同时需要把 D-geometry 视为后续机制设计的风险信号, 而不是已经坐实的根因.

## 代码和产物

- zoology commit: `89edb2a`.
- Flash-VQG commit: `0eba390`.
- Plan: `docs/plans/20260706-02-flash-vqg-default-dropout-joint-control-dgeom-plan.md`.
- Artifact: `docs/artifacts/20260706-02-flash-vqg-default-dropout-joint-control-dgeom/`.
- Main script: `zoology/experiments/flash_vqg/scripts/20260706-02-flash-vqg-default-dropout-joint-control-dgeom/joint_control_dgeom.py`.

3090 formal output 已镜像回 2080ti 主工作区. 镜像后 3090 formal 目录内容 hash:

```text
files: 112
bytes: 20255809
dir_sha256: 1b5e6d65f445a0b8e5280187941fa3d697b08b4a4ca4686177b9c5fafc50105a
```

## 硬门槛检查

本轮 formal collect 只使用两个 formal queue 输出目录, 不混入 smoke 输出.

| 项目 | 结果 |
|---|---|
| run count | `16` |
| invalid run count | `0` |
| 2080ti queue | `8/8 completed` |
| 3090 queue | `8/8 completed` |
| cache file count | `13` |
| cache content hash | `d9098e876a036b8cb90a7186174fd827e0f5b422482266772850069c905bd8c8` |
| init model state hash | `2a1107bf22d0804ed485ab94bdc7af8004ef7b892a60c2967f842ba0f4b4efb0` |
| batch order hash | `fb11b66aca3ad686a85f9623c9ae6769bb4a799fdaa0d952c0af518f0dfcc320` |
| Traceback / OOM / NaN | 未发现 |

Formal runs 默认关闭 heavy `read_trace`, `hash_probe`, `train_inline_event_trace`, `D-geometry trace`. D-geometry diagnostic targets 单独开启 detached scalar trace, 不参与 formal pass/fail 判定.

## Formal 训练结果

判定标准:

- 两机 final `1024x256` accuracy 都 `>=0.85`.
- paired gap `<=4pp`.
- overall valid accuracy 两机都高.
- 无 NaN, OOM, Traceback.

| variant | read_topk | 2080ti final 1024x256 | 3090 final 1024x256 | gap | 2080ti overall | 3090 overall | 判定 |
|---|---:|---:|---:|---:|---:|---:|---|
| `r16-update-softcap0p5-injwarm512-rerun` | 16 | `0.901` | `0.945` | `4.4pp` | `0.981` | `0.989` | 高分但严格 fail |
| `r8-update-softcap0p5-injwarm512` | 8 | `0.930` | `0.943` | `1.3pp` | `0.986` | `0.989` | pass |
| `r4-update-softcap0p5-injwarm512` | 4 | `0.837` | `0.955` | `11.8pp` | `0.967` | `0.991` | fail |
| `r2-update-softcap0p5-injwarm512` | 2 | `0.859` | `0.696` | `16.3pp` | `0.974` | `0.940` | fail |
| `r16-injwarm512-only` | 16 | `0.956` | `0.841` | `11.5pp` | `0.991` | `0.971` | fail |

### 对 formal 结果的判断

`r16-update-softcap0p5-injwarm512` 没有严格复现上一轮 pass. 上一轮 `20260706-01` 是 `0.901/0.923`, gap `2.2pp`; 本轮 same-seed rerun 是 `0.901/0.945`, gap `4.4pp`. 它仍然是高分配置, 但跨机器 gap 贴着阈值上下波动, 不能直接推进为稳定默认方案.

`r8-update-softcap0p5-injwarm512` 是本轮最干净的结果. 两机都高, gap 小, 并且没有出现 r2/r4 那种单机明显掉队. 这说明联合控制可能需要一个适中的 read support 宽度: 太窄时候选集合更容易被路径差异影响, 太宽时 residual branch 的实际注入和 M-state 轨迹仍可能在两机间漂移.

`r16-injwarm512-only` 不过线非常关键. 它说明上轮和本轮 joint control 的收益不能简单归因于 injection warmup. Warmup 能缓解“早期不可靠 residual 太快进入输出”的问题, 但没有 update softcap 时, M-state 写入幅度仍可能把扰动写入长期 memory. 反过来说, 单独 update softcap 在 `20260706-01` 也失败过, 因此当前证据更支持“写入幅度控制 + 输出注入延迟”联合使用, 而不是单点控制.

`r4/r2` 的失败也很有信息量. 它说明问题不是“把 residual branch 慢一点打开”就能解决. 窄 read support 下, read candidate 支持集本身仍然太敏感. 如果后续继续考虑 `read_topk=2/4`, 需要引入 read confidence guard, margin-aware read, support-aware injection, 或类似机制, 不能只靠 M-state update softcap.

## 机制指标

下表列出 formal variants 的关键机制指标. `update_softcap_scale_mean` 越接近 `1`, 表示 softcap 实际缩放越弱; `update_softcap_hit_ratio` 越高, 表示更多位置触发 softcap. `inject_ratio` 是 final validation 中 residual injection 相对输出的比例指标.

| variant | read_topk | mass 2080ti/3090 | entropy 2080ti/3090 | update p95 2080ti/3090 | update max 2080ti/3090 | softcap hit 2080ti/3090 | scale mean 2080ti/3090 | M max 2080ti/3090 | lambda 2080ti/3090 | inject 2080ti/3090 |
|---|---:|---|---|---|---|---|---|---|---|---|
| `r16-update-softcap0p5-injwarm512-rerun` | 16 | `0.410/0.413` | `0.540/0.744` | `0.043/0.993` | `0.146/7.06` | `0.000/0.0765` | `1.000/0.956` | `1.00/12.00` | `0.319/0.792` | `0.235/0.237` |
| `r8-update-softcap0p5-injwarm512` | 8 | `0.391/0.394` | `0.422/0.875` | `0.256/0.457` | `2.24/7.05` | `0.0184/0.0482` | `0.990/0.970` | `6.28/11.40` | `0.166/0.962` | `0.159/0.284` |
| `r4-update-softcap0p5-injwarm512` | 4 | `0.381/0.388` | `0.610/0.664` | `0.128/1.920` | `0.946/11.00` | `0.00216/0.0989` | `0.999/0.935` | `7.06/13.40` | `0.495/0.772` | `0.232/0.225` |
| `r2-update-softcap0p5-injwarm512` | 2 | `0.350/0.369` | `0.545/0.620` | `0.0117/0.733` | `0.123/7.60` | `0.000/0.0739` | `1.000/0.955` | `0.816/7.35` | `0.620/0.352` | `0.232/0.240` |
| `r16-injwarm512-only` | 16 | `0.424/0.427` | `0.744/0.687` | `0.161/0.577` | `2.33/6.60` | `0.000/0.000` | `1.000/1.000` | `5.28/14.60` | `0.543/0.817` | `0.270/0.211` |

这些指标不能单独决定成败, 但支持两个判断:

1. 3090 上多个失败或边界配置的 `update_norm_max` 和 `M_norm_max` 明显更大, 说明 M-state update/state scale 仍然是风险点.
2. `r8` 虽然也有较大的 3090 `update_norm_max`, 但最终过线. 因此“大 update”不是充分解释, 它需要和 read support, lambda, residual injection, routing 等路径一起看.

## D-geometry 诊断

D-geometry 只对 3 个 diagnostic target 运行, 采样 optimizer step `0,64,256,512,703`. 诊断对象是 `D_pack=normalize((K-codebook)@addr_proj)`, 也就是实际进入 `M_state` update 的 normalized residual write direction, 不是原始 `K`.

Diagnostic target 的 final hard 结果如下. 这些 target 带额外 detached scalar trace, 不参与 formal pass/fail.

| diagnostic variant | read_topk | 2080ti final 1024x256 | 3090 final 1024x256 | gap |
|---|---:|---:|---:|---:|
| `dgeom-r16-update-softcap0p5-injwarm512` | 16 | `0.895` | `0.933` | `3.8pp` |
| `dgeom-r16-injwarm512-only` | 16 | `0.957` | `0.880` | `7.7pp` |
| `dgeom-r4-update-softcap0p5-injwarm512` | 4 | `0.823` | `0.883` | `6.0pp` |

### D-geometry 聚合结果

| variant | machine | pair p95 mean | pair p95 max | effective rank mean | effective rank min | condition mean | update p95 mean | update p95 max |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `dgeom-r16-injwarm512-only` | 2080ti | `0.716` | `1.000` | `6.73` | `1.00` | `130.0` | `0.0837` | `1.260` |
| `dgeom-r16-injwarm512-only` | 3090 | `0.753` | `1.000` | `6.08` | `1.00` | `97.6` | `0.0895` | `2.943` |
| `dgeom-r16-update-softcap0p5-injwarm512` | 2080ti | `0.754` | `1.000` | `5.30` | `1.00` | `96.3` | `0.115` | `0.461` |
| `dgeom-r16-update-softcap0p5-injwarm512` | 3090 | `0.741` | `1.000` | `6.39` | `1.00` | `86.1` | `0.0878` | `2.799` |
| `dgeom-r4-update-softcap0p5-injwarm512` | 2080ti | `0.754` | `1.000` | `5.82` | `1.00` | `107.5` | `0.0937` | `1.716` |
| `dgeom-r4-update-softcap0p5-injwarm512` | 3090 | `0.751` | `1.000` | `6.12` | `1.00` | `149.3` | `0.107` | `3.200` |

按 step 看, 在 step `64/256` 附近经常出现 `pair_abs_cos_p95` 接近 `0.98-0.99`, effective rank 接近 `1` 的窗口. 这说明部分 code/head 内的 residual write direction 会阶段性塌到高度相似方向. 这类写入如果同时带有较大的 update strength, 理论上容易让 `M_state` 沿少数方向快速累积, 增加后续读写互相干扰的风险.

但是, 当前 D-geometry 还不能作为单独根因:

1. 好轨迹和坏轨迹中都能看到高相关/低 effective-rank 窗口.
2. Cross-machine D-geometry 平均差异不大. 例如 `pair_abs_cos_p95` abs diff mean 大约在 `0.036-0.041` 范围.
3. 某些低分 run 的 D-geometry 并不显著比高分 run 更差, 至少用本轮 summary 指标还不能直接区分.

因此, D-geometry 的合理定位是: `gd_residual_v1` 的 residual write direction 存在结构性相关和低秩风险, 它可能参与放大, 值得后续做 direction-aware write damping 或 code/head-aware 控制的诊断, 但本轮还不能说“跨机器不稳定就是 D 不正交导致的”.

## 对计划问题的回答

1. `r16-update-softcap0p5-injwarm512` 是否 same-seed paired 复现?

   不严格复现. 本轮 `0.901/0.945`, gap `4.4pp`, 两机都高但略超过 `4pp` 阈值. 结合上一轮 `0.901/0.923`, gap `2.2pp`, 可以说它是强候选, 但边界波动仍然存在.

2. 联合控制在 `r8/r4/r2` 是否仍有效?

   `r8` 有效并严格过线. `r4/r2` 不过线. 这说明联合控制不是对所有 read support 都通用, 它目前更适合中等 read support.

3. 如果 `r16` 复现但 `r8/r4/r2` 不过线, 是否说明联合控制仍依赖适度宽 read support?

   本轮更准确的说法是: `r8` 过线, `r16` 高分边界, `r4/r2` 失败. 这支持“联合控制需要适度 read support”的判断. 过窄的 read support 仍会暴露 read candidate/support 敏感性.

4. `r16-injwarm512-only` 是否过线?

   不过线. 2080ti `0.956`, 3090 `0.841`, gap `11.5pp`.

5. 如果 warmup-only 过线, 是否说明上一轮收益主要来自 injection warmup?

   不适用, 因为 warmup-only 没过线. 当前更支持 joint control, 即 injection warmup 需要和 update softcap 配合.

6. 如果 warmup-only 不过线而 joint 过线, 是否更支持 update softcap + warmup 联合必要?

   是, 但要谨慎. 本轮 `r8` joint 过线, `r16` joint 高分边界, `r16` warmup-only fail. 它支持联合控制的必要性, 但还缺 `r8-injwarm512-only` ablation 来排除 `r8` 本身的 read support 贡献.

7. D direction 是否明显非正交或低秩?

   是. 多个 step 和 code/head 上出现高 `pair_abs_cos_p95` 和低 effective rank, step `64/256` 尤其明显.

8. D geometry 是否能解释 bad run / high update / high M_norm / low hard accuracy?

   当前只能部分支持, 不能直接解释. D 方向高相关是普遍风险, 但本轮 summary 没有给出足够强的“坏 run 独有 D-geometry 异常”证据.

9. 是否需要下一轮设计 direction-aware write damping?

   可以作为 P2 方向, 但不应马上当主线 formal 机制. 更优先的是复验 `r8` joint, 以及补 `r8-injwarm512-only` 和 read support confidence guard. 如果后续 event-level trace 发现高相关 D hotspot 与大 update 和 loss/read divergence 同步, 再进入 direction-aware damping.

## 当前结论

本轮不是“找到最终方案”, 而是把可行方向缩窄了一步.

比较稳的结论:

1. 单独 injection warmup 不够.
2. 单独 update softcap 之前也不够.
3. `update softcap + injection warmup` 是当前最强正信号.
4. 这个 joint control 对 `read_topk=8` 表现最好, 对 `r16` 是高分边界, 对 `r4/r2` 不足.
5. `D_pack` 写入方向有高相关/低秩风险, 但目前不是已经证明的主因.

更人话地说: 现在的模型不是只需要“读多一点”或“慢一点注入”这么简单. 它需要同时避免两件事: 早期不可靠 residual 太快影响输出, 以及偏大的 residual update 太快写进长期 `M_state`. 但 read support 太窄时, 即使做了这两件事, 读哪些 code 这件事本身仍然会分叉.

## 下一步建议

短期建议按优先级:

1. same-seed paired rerun `r8-update-softcap0p5-injwarm512`.

   这是本轮唯一严格过线 formal 配置. 先验证它是否能复现, 不要立刻跑 4ep 或多 seed.

2. 补 `r8-injwarm512-only` ablation.

   本轮只有 `r16-injwarm512-only`. 因为 `r8` joint 过线, 需要确认 `r8` 的收益来自 joint control, 还是主要来自 read_topk=8 加 injection warmup.

3. 对 `r8` 做轻量 support-aware 诊断.

   关注 read margin, selected mass, top-k support churn, lambda/inject ratio. 目标是判断为什么 `r8` 比 `r16/r4/r2` 更稳.

4. 暂不推进 `r2/r4` joint control 4ep.

   当前 1ep 已经失败. 如果要救 `r2/r4`, 应转向 read confidence guard, margin-aware read, support-aware residual injection, 而不是继续只调 update cap 或 warmup.

5. D-geometry 进入机制储备线.

   后续可以设计 direction-aware write damping, code/head-aware update scale 或 hotspot guard, 但需要先证明 D hotspot 和真实训练分叉同步. 目前还不建议直接加入 formal 主线.

本轮暂不建议:

- 不直接跑 4ep.
- 不把 `r16` 或 `r8` 作为默认最终配置.
- 不继续扩大 read_topk sweep.
- 不把 D-geometry orthogonalization 直接变成训练机制.

