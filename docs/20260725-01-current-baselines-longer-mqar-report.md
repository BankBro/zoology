# 当前 Flash-VQG 与 GDN 基线跨GPU Longer-MQAR 报告

实验 ID: `20260725-01-current-baselines-longer-mqar`.

## 1. 结论

本轮在 RTX 2080 Ti和RTX 3090上分别独立重训当前Flash-VQG `baseline-r16-joint` 与GDN `gdnxk-h2-ek4-ev4-usegate0`. 每张GPU均使用training seeds `123/124/125`, 固定data seed `123`, 相同MQAR cache和各模型自己的canonical seed124 init. 两机共12/12正式训练到达epoch4, 随后的Longer-MQAR formal、dataset hash和repro审计全部通过.

预注册主结果使用`last.pt`. 两张GPU给出一致的定性结论:

- 在训练长度端点`1024x256`, Flash的三seed均值都低于GDN, 不支持Flash领先.
- 在四个真正的长度外推slice, Flash的平均准确率都高于同active-state-capacity的GDN.
- 2080 Ti上三个外推slice达到3/3 seeds `稳健领先`, `8190x512`为2/3 seeds的`混合领先`.
- 3090上四个外推slice全部达到3/3 seeds `稳健领先`.

3090结果同时澄清了此前的波动问题. GDN在两张GPU上的同seed结果非常接近, 每个slice的三seed平均跨机器差值不超过`0.004987`. Flash的跨机器差异明显更大, 并主要由seed124驱动: 3090上的seed124外推结果显著高于2080 Ti, seed123则在3090上略低. 因而当前证据支持“Flash平均长度外推优于GDN”, 也支持“Flash对training seed和GPU数值执行路径更敏感”. 由于训练使用`TORCH_DETERMINISTIC=0`且GPU kernel路径不同, 本实验不能把差异单独归因为GPU硬件性能.

3090的6条run全部为best等于last. 2080 Ti只有Flash seed124的best来自epoch3, 因而2080 Ti的`best.pt`敏感性结果略高于last. 两张图分别展示last和best, 不混合checkpoint role.

本轮不自动替换当前综合baseline. 下一步应优先研究Flash的seed×数值路径敏感性, 而不是把3090较高均值解释为确定性的硬件增益.

## 2. 固定实验口径

- Flash: `baseline-r16-joint`, codebook `64`, rank `16`, read top-k `16`, write top-k `4`, `smooth_p4` softcap `0.5`, active state capacity `131072`, 参数量`1160390`.
- GDN: `gdnxk-h2-ek4-ev4-usegate0`, 2 heads, expand K/V `4/4`, use gate false, active state capacity `131072`, 参数量`1335942`.
- Training seeds `123/124/125`, data seed `123`; 两模型分别固定自己的canonical seed124 init.
- Train `B64`, validation `B16`, GA4, effective batch `256`, 4 epochs, 每epoch 4次validation, early stopping关闭.
- Longer-MQAR固定`eval_seed=123`, vocab `8192`, `random_non_queries=true`, `power_a=0.01`, 每slice `500` examples.
- `last.pt`是预注册主结果; `best.pt`只按4个epoch-end的整体`valid/accuracy`选择.
- 2080 Ti使用GPU1, 3090使用GPU0. 两机均为full-model FP32, `TRITON_F32_DEFAULT=ieee`, TF32 off, `GDN_KERNEL_DTYPE=float32`.
- 两机环境均为Python 3.12.11, PyTorch 2.6.0+cu118, CUDA 11.8, Triton 3.2.0, FLA 0.4.2.
- 两机13个实际MQAR cache逐文件SHA256和tensor content hash一致; 两份init文件SHA256一致; 四个epoch batch-order hash一致.
- 12份3090 resolved config归一化后与预注册2080 Ti配置hash一致. 只允许machine、输出路径和run/launch identity不同.
- Flash-VQG源码commit为`ec770f33676036432c6514acd1ac05bd2d01f3e8`. 2080 Ti正式训练commit为`0dd9572`; 3090正式训练commit为`d6616f9`, 最终eval恢复runner为`ed95ec2`.

## 3. 正式训练结果

下表为每条`last.pt`的epoch4常规validation. Wall time包含完整训练和validation, 仅用于审计, 不作为跨GPU效率排名.

| GPU | 模型 | Seed | Overall accuracy | `1024x256` | Wall time | Best epoch |
|---|---|---:|---:|---:|---:|---:|
| 2080 Ti | Flash | 123 | 0.995230 | 0.974562 | 26.45 min | 4 |
| 2080 Ti | Flash | 124 | 0.986959 | 0.912543 | 26.42 min | 3 |
| 2080 Ti | Flash | 125 | 0.994573 | 0.970098 | 26.41 min | 4 |
| 2080 Ti | GDN | 123 | 0.995790 | 0.967371 | 18.87 min | 4 |
| 2080 Ti | GDN | 124 | 0.995667 | 0.966355 | 18.78 min | 4 |
| 2080 Ti | GDN | 125 | 0.996248 | 0.971063 | 18.74 min | 4 |
| 3090 | Flash | 123 | 0.992059 | 0.953703 | 22.84 min | 4 |
| 3090 | Flash | 124 | 0.993355 | 0.961645 | 23.27 min | 4 |
| 3090 | Flash | 125 | 0.995498 | 0.976785 | 23.50 min | 4 |
| 3090 | GDN | 123 | 0.996093 | 0.969746 | 29.72 min | 4 |
| 3090 | GDN | 124 | 0.995620 | 0.965961 | 28.27 min | 4 |
| 3090 | GDN | 125 | 0.995970 | 0.968840 | 28.87 min | 4 |

2080 Ti的6条训练合计wall time为`8140.06 s`; 3090为`9387.97 s`. GDN在两机的训练端点都表现稳定. Flash seed124在2080 Ti出现epoch3->epoch4退化, 但在3090没有复现同样的checkpoint选择现象.

## 4. Longer-MQAR 主结果: `last.pt`

### 4.1. RTX 2080 Ti

| Slice | Flash mean ± population std | GDN mean ± population std | Flash-GDN paired delta | Positive seeds | 分类 |
|---|---:|---:|---:|---:|---|
| `1024x256` | 0.953406 ± 0.027658 | 0.968518 ± 0.001889 | -0.015112 | 2/3 | 不支持Flash领先 |
| `2048x512` | 0.704091 ± 0.161179 | 0.476111 ± 0.004779 | +0.227980 | 3/3 | 稳健领先 |
| `4096x1024` | 0.325602 ± 0.165438 | 0.072722 ± 0.002686 | +0.252880 | 3/3 | 稳健领先 |
| `8190x512` | 0.556421 ± 0.200708 | 0.293578 ± 0.001806 | +0.262842 | 2/3 | 混合领先 |
| `8190x2047` | 0.097774 ± 0.065491 | 0.003378 ± 0.000215 | +0.094396 | 3/3 | 稳健领先 |

2080 Ti上Flash的主要方差来源是seed124. 它在四个外推slice的准确率分别为`0.476398`, `0.092451`, `0.273195`, `0.005586`, 明显低于seeds 123/125.

### 4.2. RTX 3090

| Slice | Flash mean ± population std | GDN mean ± population std | Flash-GDN paired delta | Positive seeds | 分类 |
|---|---:|---:|---:|---:|---|
| `1024x256` | 0.964422 ± 0.009993 | 0.968737 ± 0.001359 | -0.004315 | 1/3 | 不支持Flash领先 |
| `2048x512` | 0.790643 ± 0.037995 | 0.478858 ± 0.003640 | +0.311785 | 3/3 | 稳健领先 |
| `4096x1024` | 0.420096 ± 0.050292 | 0.075021 ± 0.001106 | +0.345074 | 3/3 | 稳健领先 |
| `8190x512` | 0.659965 ± 0.043513 | 0.298565 ± 0.004394 | +0.361400 | 3/3 | 稳健领先 |
| `8190x2047` | 0.139405 ± 0.024786 | 0.003613 ± 0.000067 | +0.135792 | 3/3 | 稳健领先 |

3090逐seed准确率如下:

| 模型 | Seed | `1024x256` | `2048x512` | `4096x1024` | `8190x512` | `8190x2047` |
|---|---:|---:|---:|---:|---:|---:|
| Flash | 123 | 0.953484 | 0.751871 | 0.379656 | 0.626156 | 0.123979 |
| Flash | 124 | 0.962141 | 0.777813 | 0.389646 | 0.632340 | 0.119859 |
| Flash | 125 | 0.977641 | 0.842246 | 0.490984 | 0.721398 | 0.174377 |
| GDN | 123 | 0.970000 | 0.478105 | 0.073572 | 0.295875 | 0.003525 |
| GDN | 124 | 0.966852 | 0.474824 | 0.075236 | 0.295059 | 0.003628 |
| GDN | 125 | 0.969359 | 0.483645 | 0.076256 | 0.304762 | 0.003687 |

## 5. `best.pt` 敏感性

2080 Ti只有Flash seed124的best与last不是同一model-state, best来自epoch3. 因而12个逻辑角色去重为7个物理checkpoint. 其best结果在四个外推slice均为3/3 seeds稳健领先:

| Slice | Flash best mean | GDN best mean | Paired delta | Positive seeds | 分类 |
|---|---:|---:|---:|---:|---|
| `1024x256` | 0.956076 | 0.968518 | -0.012443 | 2/3 | 不支持Flash领先 |
| `2048x512` | 0.731868 | 0.476111 | +0.255758 | 3/3 | 稳健领先 |
| `4096x1024` | 0.342249 | 0.072722 | +0.269527 | 3/3 | 稳健领先 |
| `8190x512` | 0.578517 | 0.293578 | +0.284939 | 3/3 | 稳健领先 |
| `8190x2047` | 0.100694 | 0.003378 | +0.097316 | 3/3 | 稳健领先 |

3090的6条run均为best等于last, 12个逻辑角色去重为6个物理checkpoint. 因此3090的best表与第4.2节last表完全相同. 这不是重复评估错误, 而是checkpoint model-state hash审计后的结果.

## 6. 跨机器稳定性

下表为同模型、同seed、同role、同slice的`3090 - 2080 Ti`准确率差值在三个seeds上的均值. 这些是配对稳定性诊断, 不是额外独立seeds.

| Role | 模型 | `1024x256` | `2048x512` | `4096x1024` | `8190x512` | `8190x2047` |
|---|---|---:|---:|---:|---:|---:|
| last | Flash | +0.011016 | +0.086552 | +0.094493 | +0.103544 | +0.041632 |
| last | GDN | +0.000219 | +0.002747 | +0.002299 | +0.004987 | +0.000235 |
| best | Flash | +0.008346 | +0.058775 | +0.077846 | +0.081448 | +0.038711 |
| best | GDN | +0.000219 | +0.002747 | +0.002299 | +0.004987 | +0.000235 |

Flash last的seedwise跨机器差异并不一致. 以四个外推slice为例:

- Seed123: `-0.075293`, `-0.079410`, `-0.088105`, `-0.027610`.
- Seed124: `+0.301414`, `+0.297195`, `+0.359145`, `+0.114274`.
- Seed125: `+0.033535`, `+0.065695`, `+0.039594`, `+0.038232`.

这说明3090较高的Flash均值主要来自seed124训练轨迹没有重现2080 Ti退化, 而不是所有seed都获得同方向提升. 相比之下, GDN所有同seed跨机器差值的最大绝对值为`0.011121`. 后续若要区分GPU kernel路径、数值误差放大和seed敏感性, 需要确定性kernel或更专门的早期step state-hash实验.

## 7. 完成性、失败与恢复审计

2080 Ti自动队列从`2026-07-24T19:06:22Z`运行到`2026-07-24T22:23:25Z`. 6/6正式训练、35/35物理formal事件和7/7 repro完成. 首次preflight曾因入口文件遮蔽Python标准库失败, 当时尚未启动任何GPU run; commit `0dd9572`修复后全流程通过.

3090首次队列从`2026-07-25T03:34:47Z`运行到`2026-07-25T06:49:59Z`. 它已完成:

- 环境、13个cache、init、batch order和12份归一化config硬预检.
- 两模型shape smoke, 6/6训练smoke, 30/30 source smoke, 30/30 formal-probe和6/6 repro smoke.
- 6/6独立正式训练到epoch4及12个逻辑checkpoint角色审计.

首次formal eval在Flash s123 `8190x512`、batch 32处OOM并fail-fast. 根因是旧formal batch-search只使用32 examples, 没有覆盖500-example多batch执行中的CUDA allocator碎片化. 失败formal结果没有写入正式ledger.

恢复设计由plan addendum `fb11df0`记录, runner commit `ed95ec2`实现. Formal batch-search改为使用完整500 examples; 恢复队列重新通过preflight和全部smoke, 并仅在config、result、checkpoint文件hash和model-state hash全部匹配时跳过已完成训练. 完整负载搜索在两模型的`8190x512`和`8190x2047`上各记录一次允许的batch32 OOM, 随后batch16全部成功. 其余三个slice使用batch32.

恢复队列从`2026-07-25T07:11:25Z`运行到`2026-07-25T07:43:02Z`, 最终30/30物理formal事件、6/6 repro完成并写入`DONE.json`. Formal与repro没有OOM、NaN、Inf或未处理Traceback. 五个formal dataset hash在两机完全一致, 所有repro accuracy delta为`0`.

## 8. Artifact、图与ledger

正式artifact结构为:

```text
docs/artifacts/20260725-01-current-baselines-longer-mqar/
├── machines/2080ti/
├── machines/3090/
├── combined/
└── figures/
```

- 两个机器目录各包含6条training、60条last/best逻辑结果、summary、paired delta、checkpoint role、source manifest、batch size、repro、verification、metadata和raw evidence manifest.
- `combined/longer-mqar-detail.csv`为120条唯一`machine × model × seed × role × slice`结果.
- `combined/cross-machine-deltas.csv`为60条同模型、seed、role、slice的跨机器配对差值.
- 3090的14个机器artifact文件与源机逐文件SHA256一致; 76份轻量raw evidence镜像hash全部一致. 大型checkpoint保留在3090原路径.
- `figures/longer-mqar-accuracy-last.{pdf,png,svg}`和`longer-mqar-accuracy-best.{pdf,png,svg}`各包含4条曲线; 对应CSV各20行. PNG为`2643 × 1595`, 约300 DPI; PDF字体已嵌入; 图中mean/std与combined summary误差不超过`1e-12`.
- Flash canonical ledger新增3条3090 cross-GPU记录; GDN expanded-K ledger新增3条3090记录, 均未覆盖2080 Ti行.

当前基线重训对照没有改写`docs/artifacts/longer-mqar/official-core-20260526/`或旧preliminary ledger.

## 9. 下一步

1. 对Flash seeds 123/124/125增加跨GPU早期optimizer-step model-state hash和关键kernel输出hash, 定位数值路径首次分叉位置.
2. 若需把“Flash长度泛化更强”升级为更强统计结论, 增加新的training seeds, 不把两张GPU上的相同seed合并成`n=6`.
3. 维持当前Flash和GDN综合baseline不变. 本轮提供长度泛化与数值稳定性证据, 不单独触发baseline替换.
