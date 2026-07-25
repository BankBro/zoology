# 20260725-01 当前基线跨GPU Longer-MQAR

本artifact记录当前Flash `baseline-r16-joint`与GDN `gdnxk-h2-ek4-ev4-usegate0`在RTX 2080 Ti和RTX 3090上的独立三seed 4ep重训及RNG-locked Longer-MQAR对照.

预注册主结果使用`last.pt`. 两机都显示Flash在`1024x256`不支持领先, 但在四个真正外推slice上平均准确率高于GDN. 2080 Ti为三个`稳健领先`和一个`混合领先`; 3090四个slice均为3/3 seeds `稳健领先`. GDN跨机器结果高度稳定, Flash存在更明显的seed×GPU数值路径敏感性, 尤其seed124.

目录结构:

- `machines/2080ti/`: 2080 Ti的training、60行逻辑结果、机器内统计、source和审计信息.
- `machines/3090/`: 3090的对应机器级artifact、76份raw evidence镜像manifest和`failure-recovery.json`.
- `combined/`: 120行跨机器逻辑结果、两机分层summary、paired delta和60行cross-machine delta.
- `figures/longer-mqar-accuracy-last.*`: last.pt跨GPU四曲线正式图及20行绘图数据.
- `figures/longer-mqar-accuracy-best.*`: best.pt跨GPU四曲线正式图及20行绘图数据.

关键审计:

- 两机共12/12正式训练到达epoch4.
- 2080 Ti完成35个唯一checkpoint-state formal事件和7个repro; 3090完成30个formal事件和6个repro.
- `combined/longer-mqar-detail.csv`为120行且主键唯一.
- 五个500-example formal dataset hash在两机完全一致, repro accuracy delta均为`0`.
- 3090机器artifact与源机逐文件SHA256一致, 76份轻量raw evidence镜像hash全部通过.
- 3090首次formal eval的非正式OOM失败、根因、修复和恢复过程见正式报告第7节及`docs/EXPERIMENT_LOG.md`.

完整解释见[正式报告](../../20260725-01-current-baselines-longer-mqar-report.md). 本轮没有改写`longer-mqar/official-core-20260526/`或旧preliminary ledger.
