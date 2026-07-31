# 输出目录

3090原始输出按`outputs/3090/<run-tag>/`保存. Checkpoint和完整日志不进入Git; 收尾时镜像轻量结果并将清洗后的证据提升到`docs/artifacts/20260731-01-selected-read-warp-mqar-screen/`.

正式run tag为`20260731-selected-warp-mqar-01`. 排除18个checkpoint后的89个轻量文件已镜像到2080 Ti同相对路径, aggregate SHA256为`2181c777ab52b32f089c50db5141f7e1c4f51e2ca360df8cd3b8454ffafca2c0`. Checkpoint共189,993,688 bytes, 继续保留在3090.
