# gd_residual_v1 `mu_min_count` 聚焦诊断报告

日期: 2026-04-26

## 结论

本次 short-run ablation 支持判断: `fox_gd_residual_mu_min_count=1.0` 会把当前 `L_state≈0.12` 的状态全部判为无效, 因而 `mu_valid_ratio=0.0`.

`mu_min_count=0.5` 也仍然过强, 因为本批次 valid 的 `L_state_max≈0.22`, 低于 0.5. `mu_min_count=0.1` 后 `mu_valid_ratio≈0.746`, 且 loss, `m_norm` 和 NaN/Inf 状态稳定. 按预设规则, 建议下一步做 `{0.1,0.25,0.5,1.0}` 的 50 batch 对照. `0.0` 仅作为诊断下界, 不作为正式候选配置.

## 测试结果

Flash-VQG:

```bash
cd /home/lyj/mnt/project/Flash-VQG
pytest tests/test_fox_gd_residual_v1.py -q
# 11 passed in 0.23s

pytest tests/test_fox_guards.py tests/test_fox_dense_write.py tests/test_fox_clr_delta_v1.py tests/test_fox_phase2_metrics.py -q
# 66 passed in 18.78s
```

zoology:

```bash
cd /home/lyj/mnt/project/zoology
pytest tests/test_flash_vqg_wrapper.py tests/test_flash_vqg_scripts.py tests/test_flash_vqg_metrics_white_list.py tests/test_train_logging_steps.py tests/test_train_batch_order.py -q
# 79 passed, 1 warning in 12.60s
```

另有针对修改点的 4 条定向测试通过, 覆盖 short-run `mu_min_count` 参数传递和 valid `phase/step` record.

## 实验配置

- variant: `gd_r16_wk4`
- `fox_remote_read_topk=2`
- `fox_gd_residual_rank=16`
- `fox_gd_residual_write_topk=4`
- builder: `grouped_chunk_torch_ref`
- pack mode: `semivec_ref`
- chunk size: `64`
- train batch size: `8`
- eval batch size: `8`
- seeds: `123,124`
- train batches: `20`
- valid: `step0 initial_valid`, `step10 periodic_valid`, `step20 final_valid`
- diagnostics: `PROFILE_ENABLE_GD_DIAGNOSTICS=1`
- outputs: `tmp/20260425-gd-residual-v1-mu-ablation/{mu1_0,mu0_5,mu0_1,mu0_0}`

## Per Seed Final

| mu_min_count | seed | status | train records | final train loss | final valid loss | final mu_valid_ratio | final m_norm mean | final m_norm max | final L_state mean | final L_state max | inject ratio | mean step sec | peak alloc GiB | NaN/Inf |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1.0 | 123 | completed | 20 | 10.694852 | 9.031549 | 0.000000 | 0.020681 | 0.306283 | 0.119474 | 0.212881 | 0.016345 | 11.937 | 2.508 | false |
| 1.0 | 124 | completed | 20 | 10.758697 | 9.046191 | 0.000000 | 0.021633 | 0.326539 | 0.124749 | 0.226357 | 0.021321 | 11.676 | 2.507 | false |
| 0.5 | 123 | completed | 20 | 10.694852 | 9.031549 | 0.000000 | 0.020681 | 0.306283 | 0.119474 | 0.212881 | 0.016345 | 11.744 | 2.508 | false |
| 0.5 | 124 | completed | 20 | 10.758697 | 9.046191 | 0.000000 | 0.021633 | 0.326539 | 0.124749 | 0.226357 | 0.021321 | 11.526 | 2.507 | false |
| 0.1 | 123 | completed | 20 | 10.694835 | 9.031551 | 0.745672 | 0.020186 | 0.289115 | 0.119329 | 0.212677 | 0.015665 | 12.059 | 2.508 | false |
| 0.1 | 124 | completed | 20 | 10.758719 | 9.046203 | 0.745928 | 0.021180 | 0.307009 | 0.124808 | 0.226195 | 0.020602 | 11.695 | 2.507 | false |
| 0.0 | 123 | completed | 20 | 10.694835 | 9.031551 | 1.000000 | 0.020187 | 0.289115 | 0.119329 | 0.212677 | 0.015665 | 12.110 | 2.508 | false |
| 0.0 | 124 | completed | 20 | 10.758719 | 9.046203 | 1.000000 | 0.021180 | 0.307009 | 0.124808 | 0.226195 | 0.020602 | 11.904 | 2.507 | false |

## Two Seed Mean

| mu_min_count | final train loss | final valid loss | final mu_valid_ratio | final m_norm mean | final m_norm max | final L_state mean | final L_state max | inject ratio | mean step sec | valid sec | peak alloc GiB | peak reserved GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.0 | 10.726774 | 9.038870 | 0.000000 | 0.021157 | 0.316411 | 0.122112 | 0.219619 | 0.018833 | 11.807 | 29.797 | 2.508 | 3.639 |
| 0.5 | 10.726774 | 9.038870 | 0.000000 | 0.021157 | 0.316411 | 0.122112 | 0.219619 | 0.018833 | 11.635 | 30.141 | 2.508 | 3.639 |
| 0.1 | 10.726777 | 9.038877 | 0.745800 | 0.020683 | 0.298062 | 0.122069 | 0.219436 | 0.018134 | 11.877 | 29.889 | 2.508 | 3.639 |
| 0.0 | 10.726777 | 9.038877 | 1.000000 | 0.020683 | 0.298062 | 0.122069 | 0.219436 | 0.018134 | 12.007 | 29.736 | 2.508 | 3.639 |

## Valid Loss By Step

两 seed 均值:

| mu_min_count | step | valid loss | mu_valid_ratio | L_state mean |
| --- | ---: | ---: | ---: | ---: |
| 1.0 | 0 | 9.039952 | 0.000000 | 0.123917 |
| 1.0 | 10 | 9.038161 | 0.000000 | 0.123051 |
| 1.0 | 20 | 9.038870 | 0.000000 | 0.122112 |
| 0.5 | 0 | 9.039952 | 0.000000 | 0.123917 |
| 0.5 | 10 | 9.038161 | 0.000000 | 0.123051 |
| 0.5 | 20 | 9.038870 | 0.000000 | 0.122112 |
| 0.1 | 0 | 9.039951 | 0.746094 | 0.123917 |
| 0.1 | 10 | 9.038164 | 0.746094 | 0.123030 |
| 0.1 | 20 | 9.038877 | 0.745800 | 0.122069 |
| 0.0 | 0 | 9.039951 | 1.000000 | 0.123917 |
| 0.0 | 10 | 9.038164 | 1.000000 | 0.123030 |
| 0.0 | 20 | 9.038877 | 1.000000 | 0.122069 |

## `mu_valid_ratio` 随阈值变化

| mu_min_count | step0 | step10 | step20 | 诊断 |
| --- | ---: | ---: | ---: | --- |
| 1.0 | 0.000000 | 0.000000 | 0.000000 | 全部无效 |
| 0.5 | 0.000000 | 0.000000 | 0.000000 | 仍全部无效 |
| 0.1 | 0.746094 | 0.746094 | 0.745800 | 约 74.6% 有效 |
| 0.0 | 1.000000 | 1.000000 | 1.000000 | 诊断下界, 全部有效 |

## L_state 解释

`L_state` 在本实验中的 two-seed final mean 约为 `0.122`, final max 约为 `0.219`. 同时 `debug_l_state_frac_ge_0_5=0.0`, `debug_l_state_frac_ge_1_0=0.0`. 因此 `mu_min_count=1.0` 和 `0.5` 都会把当前状态全判无效, 直接导致 `mu_valid_ratio=0.0`.

当阈值降到 `0.1` 时, `mu_valid_ratio≈0.746` 与 `debug_l_state_frac_ge_0_1≈0.746` 对齐. 这说明本次现象主要来自阈值过强, 不是 diagnostics 没接上, 也不是 valid 记录缺指标.

## 稳定性

- 所有 8 个 run 均 `completed`, 每个 seed 都有 20 个 train records.
- 所有 records 的 `has_nan_or_inf=false`.
- final valid loss 在四档之间几乎不变, two-seed mean 约 `9.03887`.
- `m_norm_mean` 在 `0.1/0.0` 下约 `0.02068`, 比 `1.0/0.5` 的 `0.02116` 略低, 未见放大风险.
- peak allocated memory 约 `2.508 GiB`, peak reserved memory 约 `3.639 GiB`.
- diagnostics 开启后 mean train step time 约 `11.6s` 到 `12.0s`, final valid time 约 `29.7s` 到 `30.1s`.

## 建议

按预设 A-E 规则, 本次命中: `1.0` 为 0, `0.1` 非零, 且低阈值 loss 和 `m_norm` 稳定. 因此停止继续扩大 `1.0` 或 `0.5` 的 short-run, 下一步做 `{0.1,0.25,0.5,1.0}` 的 50 batch 对照. `0.0` 保留为诊断下界, 不进入正式候选.
