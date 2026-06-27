# 20260627-02-flash-vqg-canonical-init-lock-screen 报告

## 背景

此前 cache sync 和 early-step probe 表明, 仅同步 MQAR cache 还不能保证跨机器训练路径一致. 本轮进一步使用 2080ti 上生成的一份 canonical init checkpoint, 复制到 3090 后用 state_dict tensor hash 验证, 再配合同一批 canonical cache 做 1 epoch screen.

本轮是 debug / hygiene screen, 不写入正式 MQAR ledger.

## 实验设置

- zoology branch / commit: `flash-vqg` / `bc4a4bc`.
- Flash-VQG branch / commit: `20260428-gd-residual-v1-sync` / `eed5778`.
- cache: 本轮实际加载 13 个 `data_*.pt`, 两机 `combined_content_sha256=d9098e876a036b8cb90a7186174fd827e0f5b422482266772850069c905bd8c8`.
- init: `cb64r16-s123-init.pt`, model state hash `dd0b7bb57b8bbb1fd2bd6dbcfcaa82d609c318b9b1c9ead0e89032f17bf16edf`, 两机 verify match.
- runs: 2080ti x1, 3090 x2, 均为 `default-s123` 1 epoch screen.

## 结果

| machine | target | duration | valid/accuracy | 1024x256 acc |
|---|---:|---:|---:|---:|
| 2080ti | default-s123-r1 | 84m 37s | 0.986 | 0.933 |
| 3090 | default-s123-r1 | 44m 26s | 0.971 | 0.853 |
| 3090 | default-s123-r2 | 44m 5s | 0.966 | 0.833 |


以 2080ti r1 为参考:

| candidate | overall gap | overall <= 4pp | 1024x256 gap | 1024x256 <= 4pp |
|---|---:|---:|---:|---:|
| 3090:default-s123-r1 | 1.5 pp | True | 8.0 pp | False |
| 3090:default-s123-r2 | 2.0 pp | True | 10.0 pp | False |


结论要分开说: overall `valid/accuracy` 的跨机器差距是 1.5-2.0 percentage points, 在用户接受的 4 percentage points 内. 但是本任务重点关注 `1024x256` 时, 这个 slice 的差距是 8-10 percentage points, 已经超过 4 percentage points 容忍线. 因此本轮不能以 overall 指标为依据说关键目标误差可接受.

## 已排除和未排除

已排除:

- 本轮实际加载 cache 不一致: 13/13 内容级 hash match.
- 初始模型权重不一致: canonical init checkpoint 加载后 state_dict tensor hash match.
- batch 顺序和第一批输入/target 不一致: init-lock probe 中 batch order, first inputs, first targets 均 match.

未排除:

- 具体 GPU kernel 或算子导致的数值漂移.
- 某一层或某一个 attention / VQ / GD residual 子模块导致的局部差异.
- 2080ti 单次偏高的可能性, 因为本轮 2080ti 只有 r1, 3090 有 r1/r2.

## probe 解释

init-lock early-step probe 在相同 cache, 相同 batch order, 相同 initial model params, 相同 first inputs/targets 下, first logits 和 first grad 仍不一致, optimizer step 后参数也不一致. 因此当前最干净的表述是: 差异已经被缩小到相同输入和相同初始化之后的 GPU 数值执行路径, 但尚未定位到具体算子或层.

这也解释了为什么前面很多 1 epoch 结果只能告诉我们“效果不同”, 不能直接告诉我们“误差出在哪里”. 如果要定位根因, 下一步不应继续盲目补完整 epoch, 而应做更小的 layer/kernel 级 trace 或禁用特定 fast path 的 probe.

## 建议

如果目标是判断当前方案是否可用, 需要以 `1024x256` 为主判据. 按这个判据, 本轮 2080ti 与 3090 的差距不可接受, 但还不能说明是 3090 错, 2080ti 对, 或某个具体模块错.

最小补强有两条路径:

1. 补一条 2080ti r2, 判断 2080ti r1 的 0.933 是否单次偏高.
2. 若目标是找根因, 设计 layer/kernel 级 early-step probe, 优先比较第一步 forward 中关键模块输出, 而不是继续跑更多 1 epoch.

## 原始证据

轻量 raw evidence 位于 `zoology/experiments/flash_vqg/scripts/20260627-02-flash-vqg-canonical-init-lock-screen/outputs/`, 该目录按仓库规则 ignored, 不提交. 3090 evidence 已镜像回 2080ti 主工作区, 见 artifact 的 `source-manifest.csv`.
