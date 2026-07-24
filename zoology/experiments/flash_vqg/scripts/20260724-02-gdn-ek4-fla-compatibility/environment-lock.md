# 候选环境锁

两机均从既有 `/home/lyj/miniconda3/envs/flash-vqg` 克隆, 不原位修改源环境.

## v0.4.2

```text
environment: /home/lyj/miniconda3/envs/flash-vqg-fla042
FLA tag: v0.4.2
FLA commit: ca910f88529565b28b6e16465258f2e239a02dc7
PyTorch: 2.6.0+cu118
Triton: 3.2.0
cffi/pycparser: 2.1.0/3.0
```

FLA 从只读 detached worktree 安装, 使用 `pip install --no-deps --no-build-isolation --force-reinstall <worktree>`.

两台源环境的非模型工具包原本不同. 为使候选环境 `pip check` 通过, 两机均补齐 `cffi==2.1.0`; 3090 侧因已有 `xprof/gcsfs`, 另将 `fsspec/gcsfs` 从 2025.10.0 对齐到 datasets 允许的 2025.9.0. 这些包不在模型、FLA 或 benchmark 导入链上. 两机实际完整 package list 仍以 environment snapshot 为准.

## v0.5.0

```text
environment: /home/lyj/miniconda3/envs/flash-vqg-fla050
FLA tag: v0.5.0
FLA commit: 3a9ce1c83a13994d824dbb3421e2989d330bb38b
PyTorch: 2.7.1+cu118
torchvision: 0.22.1+cu118
torchaudio: 2.7.1+cu118
Triton: 3.3.1
NumPy: 2.1.2
fsspec/gcsfs: 2025.9.0
protobuf: 6.32.1
```

PyTorch 三件套从 `https://download.pytorch.org/whl/cu118` 安装. 固定 NumPy、fsspec/gcsfs 和 protobuf 是为了避免 pip 强制升级带来的 datasets、W&B、SwanLab 依赖冲突. 两个候选环境在两机最终都必须通过普通和 `PYTHONNOUSERSITE=1` 两种 `pip check`; 原始 current040 环境保持不变.

## 固定运行环境变量

```text
GDN_KERNEL_DTYPE=float32
TRITON_F32_DEFAULT=ieee
NVIDIA_TF32_OVERRIDE=0
```

正式 artifact 中的 `environment-snapshot-*.json` 保存完整 `pip freeze --all`, Conda package list、GPU、源码 commit 和实际安装 kernel SHA256; 本文件只记录人工审核所需的核心锁.
