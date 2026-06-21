# Flash-VQG 稳定性研究方向独立复核

日期: 2026-06-05

复核对象: `docs/20260605-flash-vqg-stability-research-direction-report.md`

复核方式: 按证据块只读复核报告主张, 本机 write/state 线, beta/BBSB 线, official/init 线, 3090 read-side 正反证据, 以及源码/入口可实施性. CSV/manifest/history 以过滤关键字段和统计为主, 未全量转储原始日志. 本报告只写入当前文件, 不修改其他文件.

## 总体结论

1. 原报告整体可以作为“研究路线图”接受. 它把 codebook size/rank 定位为容量/布局轴, 把稳定性问题拆成 write/state 放大链和 read-side basin lock-in 两类候选机制, 并明确禁止把 fixed readk4, BBSB/bounded beta, caprel0406late, init transplant 写成已验证通用方案. 这个表述边界总体合理.
2. 已有实验解释中, write/state 放大链证据最强. cb64-r16 default 三 seed hard spread 为 `0.167488`, hard04 降到 `0.018016`, 且 s124 same-seed repeat4ep final gap 为 `0.001445`. 但 hard04 对 good seed 有 ceiling tax, 所以更适合作为 trust-region 诊断/稳定基准, 不是直接正式主线.
3. read-side 证据是“局部强正 + 明确反例”. cb256-r4/cb256-r8 的 readk4 结果支持 read-side 方向继续研究, 但 cb128-r8 rerun 和 cb64-r16 反例直接反驳 fixed readk4 全局默认. 原报告对此边界处理正确, 但后续应把重点从“readk4 是否有效”改成“在哪些边界条件下有效”.
4. BBSB/bounded beta 目前只支持“可能存在 ceiling 恢复信号”, 不支持“已验证稳定方案”. 关键候选存在缺 seed, 缺 repeat, late drift, best-final gap 大和只覆盖 cb64-r16/readk2 的问题.
5. official 与 exploratory 口径必须继续分开. official longer-MQAR core 支持 Flash family 在 core long slices 上强于 GDN, 但 cb256-r10 方差大, cb256-r4/cb64-r16 core 只有 seed123, cb128 目前没有 official core 多 seed longer-MQAR 结论. init transplant 是探索性因果诊断, 不是训练方法.
6. 后续 Phase A-E 的方向基本合理, 但执行优先级应更集中: 先处理 cb128-r8 readk4 rerun 崩盘和 cb256-r4/r8 readk4 正证据正式化, 再做 write/beta 组合, 最后才进入 longer-MQAR official formalization.

## 主要问题

### 1. “两条机制”仍应写成候选机制, 不是因果定论

write/state 线有 early write strength 和 m_norm 分叉, 也有 hard04 干预降低 spread 的证据, 因果支撑较强. read-side basin lock-in 主要来自 read_topk 改变后的 final 结果和正反例分布, 还缺少 early routing margin, residual candidate coverage, uncertainty 等直接轨迹指标. 因此报告中的机制表述应保持“更像是”或“候选解释”, 不宜升级为已证明机制.

### 2. hard04 稳定性强, 但 ceiling tax 也真实

cb64-r16 hard04 把 s124 从 default `0.819797` 拉到 `0.963055`, 三 seed spread 降到 `0.018016`, repeat4ep gap 也很小. 但 s123 从 `0.968711` 降到 `0.945039`, s125 从 `0.987285` 降到 `0.952605`. 这说明 hard04 是很好的 trust-region baseline 和诊断对照, 但不能直接当性能最优或 official 主线.

### 3. caprel0406late 不能被当成 hard04 的安全低税替代

caprel0406late 三 seed hard 为 `0.949371/0.963004/0.960484`, spread `0.013633`, 相比 hard04 对 s123/s125 有小幅恢复. 但 s123 final `m_norm_max=14.487579`, 明显高于 hard04/default. 这支持“release 思路值得保留”, 不支持“0.04 -> 0.06 late release 可直接推荐”. 后续应优先测试 `0.04 -> 0.05` 和 m_norm/update guard.

### 4. beta/BBSB 证据等级偏低

BBSB t2 只有 s123/s124, s124 final/best 为 `0.914547/0.944094`, best-final gap `0.029547`; bounded beta fixed + cap0405 的 s124 final/best 为 `0.900934/0.948012`, gap `0.047078`; WQA 和 BTB/budgeted 多数结果偏低或出现强 seed split. 这些结果目前只说明 beta band 可能参与 ceiling 恢复, 不足以作为稳定候选.

### 5. cb128-r8 是最高优先级风险点

cb128-r8 readk2 s124/s125 都为 `0.956`, readk4 首次 pair 为 `0.973/0.972`, 但 s125 rerun 低到 `0.609`. 这个配置同时包含 readk4 高分和严重 rerun 崩盘, 是 fixed readk4 口径最危险的反例. 在查清之前, cb128-r8 不应被用于支持 readk4 方案.

### 6. 3090 read-side 正证据不能外推成全局默认

cb256-r4 formal readk4 三个 completed run 为 `0.943/0.958/0.944`, spread `0.015`; cb256-r8 readk4 四个 completed run为 `0.982/0.982/0.988/0.992`, spread `0.010`. 这些是强局部正证据. 但 cb64-r16 readk4 s124 两个 rerun 只有 `0.831/0.849`, cb128-r8 也有 rerun 崩盘. 因此结论应是“read-side control 值得机制化”, 不是“固定 readk4 应设为默认”.

### 7. official, preliminary, exploratory 仍有混用风险

20260528 strict official seed stability, 20260526 longer-MQAR official core, 20260529/0601/0603 探索性控制实验, 3090 read-side 诊断线属于不同证据层级. 原报告总体有区分, 但后续报告和实验表格必须继续强制标注 status, GPU, dtype policy, final checkpoint, source_scope, 是否 selected official core, 是否 exploratory.

### 8. 源码链路“可配置”不等于“机制已完整可实施”

read_topk, write cap/release, write budget/total cap, beta cap/band, init seed/codebook_init_seed 等字段已有 suite/builder/mixer/phase2 run_train 透传链路. 但旧 `20260425` run_train 未显式透传 phase2 cap/budget/total cap 等控制项; attention fallback 未见现成自动开关; read schedule/gate/margin-aware control 也不是完整现成功能. 正式实验前需要 config-to-runtime smoke 校验.

### 9. 失败模式需要分型

目前至少有三类失败模式: early write/state amplification, late best-final collapse, rerun instability. 如果只看 final hard 和 spread, 容易把不同机制混成同一类不稳定. 后续实验应在 artifact 中显式标注 failure type, 并保留 best-final gap, m_norm_max, write strength, read_topk, run repeat 等字段.

## 证据来源

### 报告主张与表述边界

- `docs/20260605-flash-vqg-stability-research-direction-report.md:8` 到 `docs/20260605-flash-vqg-stability-research-direction-report.md:21`: codebook size/rank 是容量轴, write/state 与 read-side 两条候选机制.
- `docs/20260605-flash-vqg-stability-research-direction-report.md:36` 到 `docs/20260605-flash-vqg-stability-research-direction-report.md:59`: hard04, caprel0406late, BBSB/bounded beta 的主张与 caveat.
- `docs/20260605-flash-vqg-stability-research-direction-report.md:116` 到 `docs/20260605-flash-vqg-stability-research-direction-report.md:130`: fixed readk4 的正反证据边界和 gate/schedule 不是完整现成功能.
- `docs/20260605-flash-vqg-stability-research-direction-report.md:146` 到 `docs/20260605-flash-vqg-stability-research-direction-report.md:215`: Phase A-E 实验路线和 official longer-MQAR 门槛.
- `docs/20260605-flash-vqg-stability-research-direction-report.md:311` 到 `docs/20260605-flash-vqg-stability-research-direction-report.md:328`: 后续表述禁区和建议表述.

### 本机 write/state 证据

- `docs/artifacts/20260520-flash-capacity-decomposition/flash-capacity-decomp-final.csv`: cb64-r16 s123 default `valid_mqar_case_accuracy_1024x256=0.9687109375`.
- `docs/artifacts/20260528-flash-seed-stability/flash-seed-stability-final.csv`: cb64-r16 s124/s125 default hard 为 `0.819796875/0.98728515625`.
- `tmp/20260529-seed-instability-full-cap/seed-instability-final-summary.csv`: hard04 和 caprel0406late 三 seed final hard, m_norm, write strength 等汇总.
- `zoology/analysis/flash_vqg/results/.../data/history.csv`: default s123/s124 early write_mean, m_norm_max; hard04 repeat4ep; caprel0406late final `m_norm_max=14.487579`.
- `zoology/experiments/flash_vqg/generated/.../manifest.json`: hard04 和 caprel0406late 的 `fox_remote_read_topk=2`, write cap/release 配置, `codebook_init_seed=None`.

### beta/BBSB 证据

- `zoology/analysis/flash_vqg/results/flash-vqg-20260601-bbsb-t2-s123-.../data/history.csv` 和 s124 对应 history: BBSB t2 final/best, spread, best-final gap.
- `zoology/analysis/flash_vqg/results/flash-vqg-20260529-...betactrlbounded.../data/history.csv`: bounded beta fixed + cap0405 final/best 和 best-final gap.
- `zoology/analysis/flash_vqg/results/flash-vqg-20260603-wqa0p75-.../data/history.csv` 和 `...wqa0p5.../data/history.csv`: WQA final/best 和 seed split.
- `zoology/analysis/flash_vqg/results/**/launch_analysis/run_summary.csv`: BBSB/bounded beta/WQA/BTB/budgeted 复核行均为 `num_codebook_vectors=64`, `fox_gd_residual_rank=16`, `fox_remote_read_topk=2`, `data_seed=123`.

### official/init 证据

- `docs/reference/mqar-official-recording-rules.md`: official MQAR 记录规则, dtype policy, final checkpoint, ledger 和 GPU/status 要求.
- `docs/artifacts/20260528-flash-seed-stability/flash-seed-stability-final.csv` 和 `docs/20260528-flash-seed-stability-report.md`: strict official seed stability 行.
- `docs/artifacts/longer-mqar/official-core-20260526/manifest.csv`, `longer-mqar-official-core-summary.csv`, `verification.json`, `docs/20260526-longer-mqar-official-core-report.md`: RNG-locked official longer-MQAR core 口径.
- `docs/artifacts/longer-mqar/longer-mqar-eval-summary.csv` 和 `docs/20260521-longer-mqar-canonical-full-preliminary-report.md`: preliminary/historical longer-MQAR 口径, 不应与 official core 混用.
- `docs/artifacts/20260603-gd-init-transplant/train-core-final.csv`, `init-geometry-audit.csv`, `init-geometry-probe.csv`, `early-core-final.csv`, `docs/20260603-gd-init-transplant-report.md`: init transplant 作为诊断工具的证据.

### 3090 read-side 证据

3090 访问口径: `ssh lyj@192.168.2.114 "docker exec -u lyj Flash-VQG-tun bash -lc 'cd /home/lyj/mnt/project/zoology && <cmd>'"`.

- `3090:/home/lyj/mnt/project/zoology/docs/artifacts/20260530-gd-seed-diag/gd-seed-diag-key-metrics.csv`: cb256-r4 readk2 seed split, formal readk4 completed run, runtime probe, lambda015 counterexample.
- `3090:/home/lyj/mnt/project/zoology/docs/artifacts/20260530-gd-seed-diag/gd-seed-diag-spread-summary.csv`: cb256-r4 formal readk4 spread 和 runtime readk4 spread.
- `3090:/home/lyj/mnt/project/zoology/docs/artifacts/20260530-gd-seed-diag/gd-seed-diag-cross-config-final.csv`: cb64-r16, cb128-r8, cb256-r8 readk2/readk4 final hard.
- `3090:/home/lyj/mnt/project/zoology/docs/artifacts/20260530-gd-seed-diag/gd-seed-diag-cross-config-spread-summary.csv`: cb64-r16/cb128-r8 readk4 反例 spread, cb256-r8 readk4 正证据 spread.
- `3090:/home/lyj/mnt/project/zoology/docs/artifacts/20260530-gd-seed-diag/metadata.json`: fixed phase2 read_topk=4 不是 cross-configuration default solution.
- `3090:/home/lyj/mnt/project/zoology/zoology/analysis/flash_vqg/results/.../data/history.csv`: cb64-r16 readk4 s124/s125 和 cb128-r8 readk4 s125-r2 的原始 final history 复核.

### 源码与入口证据

- `zoology/experiments/flash_vqg/run_flash_vqg_suite.py`: `read_topk`, write cap/release, beta cap/band, write budget/total cap, init seed 等 suite CLI 参数.
- `zoology/experiments/flash_vqg/scripts/20260425-gd-residual-v1-mqar/config_builder.py`: builder 对 read_topk, write_topk, write cap, beta band, budget/total cap 等字段的配置组装.
- `zoology/experiments/flash_vqg/scripts/20260425-gd-residual-v1-mqar/run_train.sh`: 旧入口主要透传基础 GD/VQ 参数, 对 phase2 控制项支持不足.
- `zoology/experiments/flash_vqg/scripts/20260526-gdn-flash-fairness-phase2/run_train.sh`: phase2 入口透传 cap, budget, total cap, beta cap/band, init rng/seed, write strength mode, 并有可选 guard.
- `zoology/mixers/flash_vqg.py`: `FlashVQGMixer` 保存并向 `FlashVQGConfig` 透传 read/write/beta/init 控制字段.
- `/home/lyj/mnt/project/Flash-VQG/src/flash_vqg/nn/attn.py`, `/home/lyj/mnt/project/Flash-VQG/src/flash_vqg/nn/fox/gd_residual.py`, `/home/lyj/mnt/project/Flash-VQG/src/flash_vqg/nn/vq_init.py`: 原报告列为后续 attention/gd_residual/vq_init 语义核验来源; 本轮源码复核确认配置链路, 但 attention 内部完整语义和自动 fallback 未完全证明.

## 后续实验优先级表

| 优先级 | 实验 | 理由 | 决策标准 |
|---|---|---|---|
| P0 | cb128-r8 readk4 rerun triage: 固定 readk4 配置, 补 s123/s124/s125 多 repeat, 保存 final/best, best-final gap, early write/state 指标 | cb128-r8 同时有 readk4 高分 pair 和 s125 rerun `0.609`, 是当前最大口径风险 | 若多 repeat 仍出现低分或 spread 大, cb128-r8 不进入 readk4 正证据; 若低分不可复现, 必须定位配置, 环境或非确定性原因 |
| P0 | cb256-r4/cb256-r8 readk2 vs readk4 matched formalization | cb256-r4/r8 是 readk4 最强正证据来源, 需要同一口径 seed/repeat 正式化 | 每个容量轴至少覆盖 s123/s124/s125 和最差 seed repeat; readk4 final spread 持续 `<=0.03`, repeat gap `<=0.01`, best-final gap `<=0.01` 才升级为局部稳定方向 |
| P1 | cb64-r16 hard04 vs cap0405 vs caprel0406late + guard | hard04 稳定但有 ceiling tax, caprel0406late 有 m_norm 过冲; 需要找到低税且不过冲的 write control | cap0405 或 guard 方案需同时满足 spread `<=0.03`, m_norm_max 不超过红线, good-seed ceiling tax 明显小于 hard04 |
| P1 | BBSB/bounded beta 补 s125 和 repeat, 与 hard04/cap0405 做 matched 对照 | 当前 beta/BBSB 缺 seed 和 repeat, 且 best-final gap 大 | 只有三 seed spread `<=0.03`, best-final gap `<=0.01`, repeat gap `<=0.01` 时才作为 ceiling 恢复候选; 否则降级为 late-drift 诊断 |
| P2 | read_topk 边界扫描: cb64-r16, cb128-r8, cb256-r4, cb256-r8 上比较 read_topk=2/3/4 | 需要从 fixed readk4 走向边界条件, 判断 read-side 机制是否随 capacity/rank 有规律 | 若 read_topk 效果随配置呈稳定规律, 支持设计 schedule/gate; 若 repeat 方差大且无规律, 暂缓机制化实现 |
| P2 | config-to-runtime smoke: read_topk, write_topk, cap/release, beta cap/band, budget/guard 的实际生效核验 | 入口透传存在, 但旧脚本和 phase2 脚本能力不同, attention 内部语义未完全复核 | 每个正式实验前都要在 manifest, launch config, runtime metric 中证明控制项实际生效; 否则先修入口或统一使用 suite CLI/phase2 脚本 |
| P3 | failure taxonomy artifact: 为每个 run 标注 early write/state amplification, late collapse, rerun instability, normal high | 现有 final spread 混合了不同失败模式, 会误导组合策略 | 后续组合实验按 failure type 选择干预; 若同一配置出现多 failure type, 先定位主导失败再组合控制 |
| P3 | official longer-MQAR 准入 | 防止 exploratory/preliminary 结果过早进入 official 结论 | 只有通过 P0/P1 门槛的候选才进入 official longer-MQAR, 并记录 final checkpoint, ledger, 时间, GPU, dtype policy, seed/data_seed, read/write/beta 控制项 |

## 建议改写口径

- 保留: “codebook size/rank 是容量/布局轴, 不是稳定控制手段.”
- 保留但加限定: “write/state 放大链有较强干预证据; read-side basin lock-in 是被 readk4 正反证据支持的候选机制.”
- 改写: “hard04 是 cb64-r16 上最强稳定基准和诊断对照, 但有 ceiling tax, 不是直接正式主线.”
- 改写: “BBSB/bounded beta 目前是局部 ceiling signal, 不是稳定候选; 需先补 seed/repeat 和 best-final gap.”
- 保留: “fixed readk4 在 cb256-r4/r8 有强局部正证据, 但不是全局默认.”
- 保留: “init transplant/codebook_init_seed 是诊断和复现工具, 不是训练方法.”
- 强化: “所有进入 Phase E 的候选必须先通过 seed spread, repeat gap, best-final gap, m_norm_max 和 official recording 门槛.”

## 后期实验规划

### 规划原则

1. 先冻结证据口径, 再扩展实验矩阵. 每个 run 必须标明 `official`, `preliminary`, `exploratory`, `debug/smoke` 之一, 并记录 GPU, dtype policy, final checkpoint, source manifest, data seed, model seed, read/write/beta 控制项和是否完成到预期终点.
2. codebook size/rank 只作为容量/布局轴. 后续实验可以比较 cb64-r16, cb128-r8, cb256-r4, cb256-r8, cb256-r10 的边界条件, 但不能把 codebook size/rank 改写成稳定控制手段.
3. 每个实验都要先声明检验对象: write/state amplification, read-side basin lock-in, late best-final collapse, rerun instability, 或 config/runtime 生效性. 不允许只凭 final hard spread 把多种 failure type 混成一个机制.
4. 控制项升级必须通过 matched seed 和 repeat. 候选至少覆盖 s123/s124/s125, 对最差 seed 做 repeat, 并同时检查 final hard, best hard, spread, repeat gap, best-final gap, ceiling delta, `m_norm_max`, write strength 和 read_topk 实际生效.
5. fixed readk4 只作为局部 read-side 候选验证. cb256-r4/r8 正证据需要正式化, cb64-r16/cb128-r8 反例必须保留在同一张边界表里; 在边界条件清楚前, 不做全局默认.
6. hard04 当前定位为 stability-accuracy tradeoff probe 和 trust-region baseline. 只有当 cap0405, guarded release 或其他 write control 同时降低 spread, 缩小 ceiling tax, 且不引入 m_norm 过冲时, 才能进入正式候选.
7. BBSB, bounded beta, WQA/BTB 和 budgeted 写法先按 mechanism probe 或 incomplete candidate 管理. 它们需要先补 seed/repeat 和 late-drift 诊断, 不能直接进入 official 验证.
8. init transplant 和 `codebook_init_seed` 只保留为诊断与复现工具. 不进入训练方法候选表, 也不作为 official 稳定性方案.

### 分阶段安排

| 阶段 | 目标 | 主要动作 | 产物 | 升级或停止条件 |
|---|---|---|---|---|
| 阶段 0: 口径冻结与功能就绪检查 | 防止后续实验把探索性证据, 旧入口行为和正式结论混用 | 建立 evidence/entrypoint matrix; 明确 suite CLI, phase2 `run_train.sh`, 旧 `20260425` 入口各自能透传的字段; 对 read_topk, write_topk, cap/release, beta cap/band, budget/guard 做 config-to-runtime smoke | `docs/artifacts/<date>-flash-vqg-stability-readiness/` 下的 manifest-backed readiness 表, smoke README, 失败原因表 | 任一关键控制项不能在 manifest, launch config 和 runtime metric 中闭环证明时, 暂停正式实验, 先修入口或统一入口 |
| 阶段 1: 主问题事实表与锚定 | 把已有 cb64-r16/readk2 主问题整理成可复查事实表 | 汇总 default, hard04, caprel0406late, beta/BBSB 相关 run 的 final/best, spread, repeat gap, best-final gap, m_norm/write 指标; 明确 default `0.968711/0.819797/0.987285`, hard04 `0.945039/0.963055/0.952605`, caprel0406late `0.949371/0.963004/0.960484` 的证据层级 | manifest-backed fact table, failure taxonomy 初版, 每行 source file/run id/status | 若事实表不能复现原有关键数值或 source/status 不清, 先做证据修复, 不扩大 sweep |
| 阶段 2: read-side family map | 从“readk4 是否有效”转向“read_topk 在哪些容量/布局边界有效” | 在 cb64-r16, cb128-r8, cb256-r4, cb256-r8 上做 read_topk=2/3/4 matched 对照; P0 先处理 cb128-r8 readk4 rerun 崩盘和 cb256-r4/r8 readk4 正证据正式化 | read_topk boundary table, cb128-r8 rerun triage report, cb256-r4/r8 formalization artifact | readk4 只有在对应容量轴满足三 seed spread `<=0.03`, repeat gap `<=0.01`, best-final gap `<=0.01`, 且无低分 rerun 时, 才升级为局部稳定方向; 否则仅作为边界反例或诊断 |
| 阶段 3: write/state 时间序列诊断 | 验证 early write/state amplification 是否是 cb64-r16 不稳定主链 | 对 default, hard04, cap0405, caprel0406late, guarded release, bounded beta/budget 做小网格; 重点看 early write strength, `m_norm_max`, state/update norm, cap hit ratio, late slope 与 final collapse 的先后关系 | write/state trajectory report, cap/release guard 对照表, m_norm 红线建议 | 候选必须在 bad seed 上修复 final, 在 good seed 上 ceiling tax 小于 hard04, spread `<=0.03`, repeat gap `<=0.01`, 且 m_norm 不超过红线; caprel0406late 若继续过冲, 降级为 release 风险案例 |
| 阶段 4: beta/BBSB completion 与 negative-control | 判断 beta band 是否真有 ceiling 恢复价值, 还是 late-drift 噪声 | BBSB t2 补 s125 和最差 seed repeat; bounded beta fixed + cap0405 补三 seed; WQA/BTB 只保留少量 negative-control, 不做大 sweep | beta/BBSB completion table, best-final gap/late drift taxonomy | 只有三 seed spread `<=0.03`, best-final gap `<=0.01`, repeat gap `<=0.01`, 且与 hard04/cap0405 matched 对照有明确增益时, 才进入候选; 否则标为 diagnostic-only 或 deprioritize |
| 阶段 5: capacity/layout transfer map | 判断局部控制项能否跨容量/布局迁移 | 对通过阶段 2-4 的候选, 在 cb64-r16, cb128-r8, cb256-r4, cb256-r8 上做最小迁移矩阵; cb256-r10 因 longer-MQAR 方差大, 单列风险复核 | capacity/layout transfer map, negative transfer cases, 配置边界说明 | 若控制项只在单一容量/布局有效, 写成局部方案; 若跨配置出现反向效果或 rerun instability, 暂停机制化实现 |
| 阶段 6: longer-MQAR official seed expansion | 只让通过前置门槛的候选进入 official longer-MQAR | 使用 official recording rules 和 20260526 official core 口径; 对 cb256-r4/cb64-r16 补多 seed, cb128 先补 official core, cb256-r10 单独复核方差 | official ledger, final checkpoint artifact, longer-MQAR official core summary 增量报告 | 缺 ledger, GPU, dtype policy, final checkpoint, source manifest 或未完成 final checkpoint 的 run, 一律不得写成 official; preliminary/historical 只作背景 |
| 阶段 7: 决策收束 | 把实验结果转成可发表/可汇报结论 | 将每个控制项分为 `validated-local`, `candidate-for-official`, `diagnostic-only`, `rejected/deprioritized`; 更新报告推荐语和禁区语 | final decision table, report-ready conclusion, next-cycle plan | 连续两个阶段无法通过 seed/repeat/gap/norm 门槛的方向停止扩展; 只保留能解释失败模式或能进入 official 的方向 |

### 推荐优先级

1. P0: 阶段 0 readiness 和阶段 1 manifest-backed fact table. 这是所有后续实验的硬前置, 能避免把旧入口, 探索性 run 和正式结论混用.
2. P0: cb128-r8 readk4 rerun triage 与 cb256-r4/r8 readk4 matched formalization. 前者是 fixed readk4 的最大风险点, 后者是 read-side 正证据的主要来源.
3. P1: cb64-r16/readk2 write/state 小网格. 优先比较 hard04, cap0405, caprel0406late, guarded release, 并把 m_norm/update guard 作为防过冲检查.
4. P1: read_topk family map. 用 readk2/3/4 单因素 paired 对照替代“固定 readk4 是否有效”的二分问题.
5. P2: BBSB/bounded beta completion 与 negative-control. 只在补齐 seed/repeat 且 best-final gap 收敛后, 才考虑进入候选池.
6. P2: capacity/layout transfer map. 只迁移已过阶段 2-4 门槛的候选, 不对 incomplete candidate 做大规模扩展.
7. P3: read schedule/gate/margin-aware pilot. 只有当 read_topk family map 呈现稳定边界规律, 且 telemetry 能记录 margin/entropy/churn 时, 才开始实现或试跑.
8. P3: longer-MQAR official seed expansion. 这是验证层, 不是探索层; 只有前置门槛通过后才进入.

### 统一准入门槛与产物要求

- 最小 seed 门槛: s123/s124/s125 全覆盖; 对最差 seed 或最不稳定配置至少做一次 repeat.
- 稳定性门槛: final hard spread `<=0.03`, repeat gap `<=0.01`, best-final gap `<=0.01`; 若目标是诊断而非候选, 可不满足门槛, 但必须明确标注 `diagnostic-only`.
- 性能门槛: bad seed 必须显著修复; good seed ceiling tax 必须小于 hard04 或有明确机制收益, 否则不能称为更优稳定方案.
- 状态门槛: `m_norm_max`, write strength, state/update norm, cap hit ratio 和 late slope 不得出现新的过冲风险; caprel/release 类实验尤其要检查 late m_norm.
- read-side 门槛: read_topk 实际生效必须由 manifest, launch config 和 runtime 指标共同证明; readk4 结果必须同时报告 cb256 正例和 cb64/cb128 反例.
- official 门槛: 正式 MQAR 或 longer-MQAR 必须记录 ledger, 时间, GPU, dtype policy, final checkpoint, status, source_scope 和 selected official core; 未满足者只能写入 exploratory/preliminary artifact.
- 产物门槛: 每个阶段至少输出 final CSV, source manifest CSV, metadata JSON 和 README; 失败/中断/smoke 也要记录状态和原因, 但不得进入正式 ledger.

### 主要风险与规避

- 单 seed 或单 repeat 的 lucky basin 风险: 用 matched seed sweep 和 worst-seed repeat 规避.
- hard04 ceiling tax 风险: 把 hard04 作为 trust-region baseline, 不作为默认主线; 后续重点寻找低税 cap/guard.
- caprel0406late norm 过冲风险: 先做 `0.04 -> 0.05` 和 guard, 不直接扩大 `0.04 -> 0.06` 推荐.
- BBSB/bounded beta late drift 风险: 强制记录 best-final gap 和 late slope, 先补齐缺 seed/repeat.
- readk4 跨配置外推风险: 将 cb128-r8 rerun 崩盘和 cb64-r16 低分 rerun 作为同等权重反例写入边界表.
- 入口与 runtime 不一致风险: 每个正式实验前做 config-to-runtime smoke; 旧 `20260425` 入口不承担 phase2 控制项正式复现.
- official/preliminary 混用风险: reporting 层强制分栏, 任何没有 official recording 证据的 run 不进入 official 结论.

