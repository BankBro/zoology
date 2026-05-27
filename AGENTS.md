# 项目协作规范

- 交互语言: 与仓库内的任何协作型 Agent 交互时, 以及与用户交互过程中, 请始终使用中文, 以保持沟通一致性.
- 文本格式: 所有写入文件或终端的文本使用 UTF-8; 中文说明使用英文标点.
- 术语与标题: 除必要专业词汇, 代码标识符, 文件路径, 命令和指标名外, 标题与解释性文本使用中文.
- 最小适配: 允许为了完成任务进行最小化修改适配, 包括修复 bug 或增加外围开关/脚本/报告适配, 但不得改变原有语义和机制原理.
- MQAR 正式实验: 完整跑到预期 final checkpoint 的 MQAR 正式实验和正式 longer-MQAR eval 必须记录 ledger, 时间, GPU, dtype policy 和状态; smoke/debug/失败/中断实验不写入正式 ledger, 但要在 artifact/status/report 中记录状态和原因.
- MQAR 细则: canonical ledger 字段, 时间字段, dtype 默认策略, GDN kernel dtype 和 official 对比口径详见 `docs/reference/mqar-official-recording-rules.md`.

## 多机 IP 与容器路径

### 跨机器执行规则

- 启动自检: Agent 接手任务时, 先确认当前会话运行在哪台宿主机的 `Flash-VQG-tun` 容器内; 不得仅凭容器内 `hostname -I`, Docker 网关或路由表判断宿主机.
- 机器名默认语义: 用户说 `3090`, `2080ti`, `4090` 时, 默认指对应宿主机上的 `Flash-VQG-tun` 容器内环境, 不是宿主机裸环境. 只有用户明确说“宿主机”, 或任务确实需要 Docker 管理, GPU/进程粗查, 路径映射确认等宿主机层操作时, 才进入宿主机语境. 若上下文不清, 先按容器内环境理解并在操作前确认.
- 标准链路: 跨机器操作默认先 SSH 到目标宿主机, 再进入该宿主机的 `Flash-VQG-tun` 容器执行, 例如 `ssh lyj@<host> "docker exec -u lyj Flash-VQG-tun bash -lc '<cmd>'"`.
- 容器内优先: 项目更新, Git 状态检查, Python/CUDA 命令, 依赖检查, 实验脚本, `mihomo` 配置/启动/日志和 sudo 免密配置, 默认都在目标机器的 `Flash-VQG-tun` 容器内处理; 操作仓库时优先使用 `docker exec -u lyj`.
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
| `mclab` (`192.168.2.188`) | `Flash-VQG-tun` | 待确认 | 待确认 | 未记录 |

已确认的项目路径映射:

| 宿主机 | 容器内路径 | 宿主机路径 |
|---|---|---|
| `mclab-3090` | `/home/lyj/mnt/project` | `/mnt/980pro/lyj/project` |
| `mclab-2080ti` | `/home/lyj/mnt/project` | `/mnt/WD40EZRZ/lyj/project` |

项目子目录按项目根路径直接拼接, 例如容器内 `/home/lyj/mnt/project/zoology` 对应宿主机 `<宿主机项目根>/zoology`.

### IP 与路径确认规则

- 宿主机 IP 识别约束: 涉及多机, 容器, 宿主机, NAT, TUN 或代理时, 不得仅凭容器内 `hostname -I`, Docker 网关, `/proc/net/route`, mihomo fake-ip/TUN 地址推断宿主机局域网 IP.
- 出口 IP 反查方法: 如需确认当前容器经宿主机访问局域网目标时的出口 IP, 必须使用 `lyj` 用户 SSH 到已知可达的局域网机器反查, 例如 `ssh lyj@192.168.2.114 'printf "SSH_CLIENT=%s\n" "$SSH_CLIENT"; hostname; hostname -I'`, 以 `SSH_CLIENT` 第一字段作为对端看到的来源 IP.
- 多出口处理: 若多台机器反查来源 IP 不一致, 必须报告多网卡, 多出口或策略路由可能性, 不得武断合并为单一宿主机 IP.
- 容器路径确认: 如需确认其他机器的 `Flash-VQG-tun` 路径映射, 必须在对应宿主机执行 `docker inspect Flash-VQG-tun` 查看 `.Mounts`, 不得凭路径名称推断.
