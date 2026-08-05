# 项目协作规范

- 交互语言: 与仓库内的任何协作型 Agent 交互时, 以及与用户交互过程中, 请始终使用中文, 以保持沟通一致性.
- 子代理层级: 子代理只允许由根代理派生一层. 子代理禁止继续创建或派生子代理; 如需进一步拆分任务, 应返回根代理统一调度.
- 文本格式: 所有写入文件或终端的文本使用 UTF-8; 中文说明使用英文标点.
- 术语与标题: 除必要专业词汇, 代码标识符, 文件路径, 命令和指标名外, 标题与解释性文本使用中文.
- 最小适配: 允许为了完成任务进行最小化修改适配, 包括修复 bug 或增加外围开关/脚本/报告适配, 但不得改变原有语义和机制原理.
- 不要过度设计: 以当前任务和已经验证的需求为边界, 优先采用最小、直接、可维护的实现; 不为未经确认的未来需求预先增加抽象层、兼容层、配置项或复杂机制.
- MQAR 正式实验: 完整跑到预期 final checkpoint 的 MQAR 正式实验和正式 longer-MQAR eval 必须记录 ledger, 时间, GPU, dtype policy 和状态; smoke/debug/失败/中断实验不写入正式 ledger, 但要在 artifact/status/report 中记录状态和原因.
- 长任务监控: 对耗时较长的训练或评估任务, 通过启动自检后应继续监控到进入稳定运行阶段再退出会话等待. 稳定运行阶段至少包括: GPU/NVML 与 `torch.cuda.is_available()` 检查通过, 代码/配置/cache/init 等本轮硬门槛已通过, 日志显示已进入实际 train/eval loop 并持续推进, 队列/进程/日志/result 路径已明确记录. 退出前必须告知用户当前状态, 有效 run 数, 已知输出路径, 预计剩余时间和恢复检查命令; 不需要全程在线守到任务结束, 除非用户明确要求.
- MQAR 细则: canonical ledger 字段, 时间字段, dtype 默认策略, GDN kernel dtype 和 official 对比口径详见 `docs/reference/mqar-official-recording-rules.md`.

## 当前活跃开发分支

- 当前 Flash-VQG/GD residual 相关实验默认从 `flash-vqg` 分支派生, 完成后优先合入 `flash-vqg`, 不默认合入 `main`.
- 只有明确属于仓库通用基础设施, 文档规范或跨项目公共逻辑的改动, 才考虑合入 `main`.
- 若用户在任务中明确指定 base 或目标分支, 以用户指定为准; 不确定时先确认再执行 merge/rebase/PR.

## Flash-VQG/GDN 默认 Conda 环境

- 适用范围: 在 `2080ti` 和 `3090` 的 `Flash-VQG-tun` 容器内运行 Flash-VQG、GDN/Expanded-K 及其 MQAR、longer-MQAR、效率、兼容性和质量回归实验时, 默认统一使用 `/home/lyj/miniconda3/envs/flash-vqg-fla042`.
- 固定核心版本: Python 3.12.11, PyTorch 2.6.0+cu118, Triton 3.2.0, FLA 0.4.2. 环境锁和重建依据见 `zoology/experiments/flash_vqg/scripts/20260724-02-gdn-ek4-fla-compatibility/environment-lock.md`.
- 显式解释器: 自动化脚本、跨机器 SSH 命令和非交互 `docker exec` 不得依赖 shell 是否自动激活 Conda; 默认直接使用 `/home/lyj/miniconda3/envs/flash-vqg-fla042/bin/python` 或在命令中显式激活该环境.
- 启动自检: 运行实验前必须记录并核对 `sys.executable`, Python、PyTorch、CUDA、Triton 和 FLA 版本; 若实际环境不是 `flash-vqg-fla042`, 必须在启动训练或评估前停止并纠正.
- 例外边界: 仅在复现历史结果、执行 legacy FLA v0.4.0 等价性对照或明确研究其他依赖版本时, 才允许使用 `flash-vqg`、`flash-vqg-fla050` 或其他环境; 必须在命令、artifact 和报告中明确记录例外原因、环境路径和版本. `flash-vqg-fla050` 是已淘汰候选, 不得作为后续正式实验的默认环境.
- 新机器适配: `4090` 或其他尚未验证的机器不得自行退回旧环境; 应先依据上述环境锁创建或验证等价的 `flash-vqg-fla042` 环境, 完成 GPU/CUDA、依赖和最小 smoke 检查后再启动正式实验.

## 项目状态与实验记录

- 开始新的 Flash-VQG/GDN 实验任务前先阅读 `docs/STATUS.md`.
- `docs/STATUS.md` 保持精简, 固定记录当前 Flash-VQG 基线、GDN 对照基线、默认环境、当前进展和下一步. 只有正式实验确认替代方案后才更新基线, 并链接对应 report 和 artifact.
- `docs/EXPERIMENT_LOG.md` 按时间追加正式实验以及会影响后续决策的重要 smoke、失败或中断记录. 每条简要说明日期、`experiment_id`、目的、结果、输出位置和下一步; 历史记录不静默改写, 纠正时追加说明.
- 详细配置、指标和审计信息仍保存在 plan、report、artifact 和 canonical ledger 中, 不在状态页或实验日志中重复展开.

## zoology 实验文件组织管理规范

- 新实验统一先定义 `experiment_id`, 格式为 `YYYYMMDD-NN-experiment-name`. 其中 `NN` 是当天第几个实验或研究单元, 从 `01` 开始递增; 同一实验的 script, plan, artifact, report 使用同一个 `experiment_id`. 历史已有 `YYYYMMDD-experiment-name` 路径和文档不强制重命名.
- 实验入口脚本放在 `zoology/experiments/flash_vqg/scripts/<experiment_id>/`.
- 实验脚本旁的本地中间输出放在 `zoology/experiments/flash_vqg/scripts/<experiment_id>/outputs/`; 该目录用于 debug, smoke, 临时拼表等未整理产物, 默认不提交. 收尾时只把可审计的轻量 summary/metadata/README 提炼到 `docs/artifacts/<experiment_id>/`.
- 实验 plan 如需落盘, 放在 `docs/plans/<experiment_id>-plan.md`.
- 自动生成配置和 manifest 放在 `zoology/experiments/flash_vqg/generated/<launch_id>/`; 若先在临时 worktree 或其他机器生成, 收尾时补回 base repo 标准路径.
- 原始 analysis 放在 `zoology/analysis/flash_vqg/results/<launch_id>/`; 若没有生成, 在 artifact metadata 或 README 说明原因.
- 正式 artifact 放在 `docs/artifacts/<experiment_id>/`, 至少包含 final CSV, source manifest CSV, metadata JSON 和 README.
- 人读报告放在 `docs/<experiment_id>-report.md`.
- 总表归属: Flash-VQG rank/seed/capacity 写 `docs/artifacts/gd-residual-v1/`; 普通 GDN 写 `docs/artifacts/gdn/`; expanded-K 或 kernel 线写 `docs/artifacts/gdn-expanded-k/`; longer-MQAR eval 写 `docs/artifacts/longer-mqar/`; 正式结果和探索性结果不要混成同一个推荐总表.
- checkpoint, swanlog 和大型 raw 默认原位保留; 若来源 worktree/env 会删除, 先把关键 manifest, hash, command, config, log 和 source/env snapshot 归档到 artifact.

## Flash-VQG 展示稿管理

- 展示稿源码和发布产物的 canonical 位置为 `/home/lyj/mnt/project/Flash-VQG/docs/presentations/flash-vqg-weekly/`.
- zoology 只保留实验 report, artifact 和 ledger, 不再保存展示稿副本.

## 多机 IP 与容器路径

### 跨机器执行规则

- 启动自检: Agent 接手任务时, 先确认当前会话运行在哪台宿主机的 `Flash-VQG-tun` 容器内; 不得仅凭容器内 `hostname -I`, Docker 网关或路由表判断宿主机.
- 机器名默认语义: 用户说 `3090`, `2080ti`, `4090` 时, 默认指对应宿主机上的 `Flash-VQG-tun` 容器内环境, 不是宿主机裸环境. 只有用户明确说“宿主机”, 或任务确实需要 Docker 管理, GPU/进程粗查, 路径映射确认等宿主机层操作时, 才进入宿主机语境. 若上下文不清, 先按容器内环境理解并在操作前确认.
- 标准链路: 跨机器操作默认先 SSH 到目标宿主机, 再进入该宿主机的 `Flash-VQG-tun` 容器执行, 例如 `ssh lyj@<host> "docker exec -u lyj Flash-VQG-tun bash -lc '<cmd>'"`.
- 容器内优先: 项目更新, Git 状态检查, Python/CUDA 命令, 依赖检查, 实验脚本, `mihomo` 配置/启动/日志和 sudo 免密配置, 默认都在目标机器的 `Flash-VQG-tun` 容器内处理; 操作仓库时优先使用 `docker exec -u lyj`.
- 容器 GPU 可用性硬门槛: 启动任何需要 GPU 的实验前, 必须先在目标机器的 `Flash-VQG-tun` 容器内确认 `nvidia-smi` / NVML 和 `torch.cuda.is_available()` 均可用. 若容器内 NVML/CUDA 不可用, 必须立即提醒用户并暂停实验启动, 记录机器, 容器, 检查命令和错误现象; 不得自动改用宿主机直接运行, 临时 `docker run --gpus`, host-side runner 或其他绕过路径继续开启实验, 除非用户在知情后明确要求这样做.
- 跨机器 MQAR/Flash-VQG 数据一致性硬门槛: 当实验目标包含跨机器对照, 且结论需要建立在相同输入数据集或相同 MQAR cache 上时, 启动训练前必须检查本轮实际会加载的 `data_*.pt` cache 在各目标机器上是否内容一致. 检查不得只依赖文件名; 至少要比较源机器上的文件级 sha256, 若 `.pt` 序列化字节可能漂移, 还要用 `torch.load(..., map_location="cpu")` 后对 tensor 内容做稳定 hash. 若 cache 不一致, 必须暂停训练启动并提醒用户选择 canonical cache, 重新生成 cache, 或明确接受“各机器独立数据集”的实验口径; 不得在用户未确认的情况下继续跑并把结果解释为严格跨机器对照.
- MQAR cache 语义提醒: 当前 MQAR 数据生成中 `random_non_queries=True` 时包含未显式绑定到 data seed 的 PyTorch 随机填充, 因此不同机器单独生成的同名 cache 可能内容不同. 若任务只关心各机器独立随机数据下的整体趋势, 可以记录该口径后继续; 若任务关心 machine/runtime/GPU/seed path 的严格比较, 必须先使用同一份 canonical cache 或验证内容级 cache 一致.
- 多机实验代码同步: 多机实验的源码和实验脚本默认先在主工作区完成修改, 通过 git commit/push 同步到远端仓库, 再在目标机器的 `Flash-VQG-tun` 容器内 git pull 到相同分支和 commit 后启动实验. 不默认用 `scp`/`rsync` 临时覆盖源码或脚本.
- 多机实验产物回收: 目标机器运行时生成的 `zoology/experiments/flash_vqg/generated/<launch_id>/`, logs, checkpoints, swanlog 等属于该机器本地产物. 收尾时必须明确区分 source machine 与主工作区/base repo mirror. 用于 artifact/report 审计的轻量 raw evidence, 包括 generated config/manifest, stdout/stderr logs, resolved config, command, hash 和 source/env snapshot, 默认镜像回主工作区的相同相对路径或 artifact 指明的 mirror path, 并用 `sha256sum` 或等效方式校验; mirror 状态, source path, mirror path 和未回收项必须写入 `source-manifest.csv`, `metadata.json` 或 report. 镜像回来的 ignored raw 文件默认仍不提交, 只提交整理后的 summary/metadata/README/source-manifest. 大型 raw, checkpoints, swanlog backup 和 tensor dump 默认仍在 source machine 原位保留且不提交; 若未镜像轻量 raw evidence, 必须在 artifact/report 中说明原因和可追溯源路径.
- 目标机器临时改代码: 若实验必须在目标机器临时修改代码, 收尾前必须把改动正规化为 git commit, 并同步回主工作区或明确记录差异, 避免多台机器出现不可追踪的源码分叉.
- 多层 shell 写入: 跨机器 `ssh` + `docker exec` + `sudo tee` 场景中, 不用单条远程命令拼复杂多行脚本或配置. 需要写多行文件时, 优先本地生成并检查, 再通过 `scp`/`docker cp`/`install` 放入目标容器.
- 宿主机边界: 宿主机只作为 SSH 入口, Docker 管理, GPU/进程粗查和路径映射确认的外层环境. 若确需在宿主机改文件, 服务或 sudoers, 必须先说明这是宿主机操作及原因.

### 局域网机器表

| 机器名 | 局域网 IP | 别称 | 说明 |
|---|---|---|---|
| `mclab-3090` | `192.168.2.114` | `3090` | 已确认可 SSH, 3090 GPU 机器 |
| `mclab-2080ti` | `192.168.2.131` | `2080ti` | 当前 zoology 容器所在宿主机 |
| `mclab` | `192.168.2.188` | `4090` | 4090 GPU 机器 |

### Flash-VQG-tun 容器路径表

Flash-VQG 和 zoology 项目可能运行在多台机器的 `Flash-VQG-tun` 容器内. 容器内项目路径通常形如 `/home/lyj/mnt/project/...`, 但每台宿主机对应的真实宿主机路径可能不同, 不得跨机器套用.

| 宿主机 | 容器名 | 宿主机路径 | 容器路径 | 状态 |
|---|---|---|---|---|
| `mclab-3090` (`192.168.2.114`) | `Flash-VQG-tun` | `/mnt/980pro/lyj` | `/home/lyj/mnt` | 已确认 |
| `mclab-2080ti` (`192.168.2.131`) | `Flash-VQG-tun` | `/mnt/WD40EZRZ/lyj` | `/home/lyj/mnt` | 已确认 |
| `mclab` (`192.168.2.188`) | `Flash-VQG-tun` | `/home/lyj` | `/home/lyj` | 已确认 |

已确认的项目路径映射:

| 宿主机 | 容器内路径 | 宿主机路径 |
|---|---|---|
| `mclab-3090` | `/home/lyj/mnt/project` | `/mnt/980pro/lyj/project` |
| `mclab-2080ti` | `/home/lyj/mnt/project` | `/mnt/WD40EZRZ/lyj/project` |
| `mclab` | `/home/lyj/mnt/project` | `/home/lyj/mnt/project` |
| `mclab` | `/home/lyj/project` | `/home/lyj/project` |

项目子目录按项目根路径直接拼接, 例如容器内 `/home/lyj/mnt/project/zoology` 对应宿主机 `<宿主机项目根>/zoology`.

### IP 与路径确认规则

- 宿主机 IP 识别约束: 涉及多机, 容器, 宿主机, NAT, TUN 或代理时, 不得仅凭容器内 `hostname -I`, Docker 网关, `/proc/net/route`, mihomo fake-ip/TUN 地址推断宿主机局域网 IP.
- 出口 IP 反查方法: 如需确认当前容器经宿主机访问局域网目标时的出口 IP, 必须使用 `lyj` 用户 SSH 到已知可达的局域网机器反查, 例如 `ssh lyj@192.168.2.114 'printf "SSH_CLIENT=%s\n" "$SSH_CLIENT"; hostname; hostname -I'`, 以 `SSH_CLIENT` 第一字段作为对端看到的来源 IP.
- 多出口处理: 若多台机器反查来源 IP 不一致, 必须报告多网卡, 多出口或策略路由可能性, 不得武断合并为单一宿主机 IP.
- 容器路径确认: 如需确认其他机器的 `Flash-VQG-tun` 路径映射, 必须在对应宿主机执行 `docker inspect Flash-VQG-tun` 查看 `.Mounts`, 不得凭路径名称推断.
