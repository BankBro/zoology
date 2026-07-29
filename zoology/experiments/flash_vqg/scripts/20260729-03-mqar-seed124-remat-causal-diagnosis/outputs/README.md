# 输出说明

本目录保存 3090 上的诊断原始 JSONL、preflight、比较结果、checkpoint 和日志. 除本文件外默认不提交 Git.

大型 replay capsule 如有生成, 保留在源机器原路径; artifact 只记录 manifest、大小和 SHA256.

- `20260729-seed124-diag-01`: 失败的基础设施run. 在首个backward前因标量hash和错误工作目录停止, 误生成cache已原位保留在`failed-site/`.
- `20260729-seed124-diag-02`: 有效因果诊断. 3090 raw约1.2 GiB, 其中checkpoint约1208.5 MiB.
- 有效run的非checkpoint轻量证据已镜像到2080 Ti主工作区. Artifact的10个source文件均已逐文件验证SHA256一致.
