# Flash-VQG Triton 路径分派汇总

本文总结当前 Flash-VQG 在训练态 materialize 路径下, 当开启如下配置时, Triton shortcode / state build / remote path 在不同 `value_dim` 下会进入哪些默认实现.

假设配置如下:

- `vq_use_triton_shortcodes=True`
- `fox_state_build_backend="triton"`
- `fox_remote_path_backend="triton"`
- 没有额外设置 state build / remote path 的 impl override
- 关注的是训练态 forward + backward, 不是 decode cache 路径

## 1. 总结表

| d_model | H | key_dim | value_dim | shortcode kernel | state build kernel | remote kernel | backward类型 |
|---|---:|---:|---:|---|---|---|---|
| 64 | 2 | 32 | 32 | `vq_assign_kernel` | `k_suffix_gamma` + `k_fused_bc_scan` | `k_remote_reduce_online` | shortcode: 无 Triton backward. state build: PyTorch `_fox_state_build_backward_torch`. remote path: PyTorch `_fox_remote_path_backward_torch`. |
| 128 | 2 | 64 | 64 | `vq_assign_kernel` | `k_suffix_gamma` + `k_fused_bc_scan_v64` | `k_remote_reduce_online_v64_train_aux` | shortcode: 无 Triton backward. state build: Triton cached backward, 即 `k_state_build_backward_scan_cached_v64` + `k_state_build_grad_logf_from_cached_decay_v64`. remote path: Triton `k_remote_reduce_backward_rows_v64_v2`. |
| 256 | 2 | 128 | 128 | `vq_assign_kernel` | `k_suffix_gamma` + `k_fused_bc_scan` | `k_remote_reduce_online_v128` | shortcode: 无 Triton backward. state build: PyTorch `_fox_state_build_backward_torch`. remote path: PyTorch `_fox_remote_path_backward_torch`. |

## 2. 重要说明

### 2.1 shortcode 实际按 key_dim 分派, 不是按 value_dim 分派

虽然这里按 `value_dim=32/64/128` 讨论, 但 shortcode Triton 路径实际看的是输入 `vecs` 和 `codebook` 的最后一维, 也就是 `key_dim`.

在当前实验里我们恰好有:

- `d_model=64, H=2 -> key_dim=value_dim=32`
- `d_model=128, H=2 -> key_dim=value_dim=64`
- `d_model=256, H=2 -> key_dim=value_dim=128`

所以表里的 shortcode 结论可以直接套用.

### 2.2 当前最完整的默认 Triton 档是 value_dim=64

在默认实现下:

- `value_dim=32` 是 generic Triton forward + Torch backward
- `value_dim=64` 同时命中 state build v64 fast path 和 remote path v64 owner v2 路径
- `value_dim=128` 的 remote forward 有专门 v128 kernel, 但 backward 仍是 Torch

如果我们关注的是训练态 forward + backward 总体加速完整性, `value_dim=64` 是当前最强的一档.

## 3. 分模块说明

### 3.1 Triton shortcode

默认实现:

- forward: 统一走 `compute_shortcodes_triton_l2(...)`
- 其内部实际 launch 的 Triton kernel 是 `vq_assign_kernel`
- 没有 shortcode 专用的 Triton backward kernel

因此:

- `key_dim=32` -> `vq_assign_kernel`
- `key_dim=64` -> `vq_assign_kernel`
- `key_dim=128` -> `vq_assign_kernel`

backward 说明:

- shortcode 是离散 index
- 后续误差 `errs2` 是通过 gather codewords 后用普通 PyTorch 图继续计算
- 因此没有单独的 Triton shortcode backward

### 3.2 state build

默认 dispatch:

- `value_dim == 64` -> `dispatch_tag = "dim64"`
- 其它 -> `dispatch_tag = "non64"`

#### value_dim = 32

默认 impl:

- `triton_generic_fw_torch_bw`

forward:

- `k_suffix_gamma`
- `k_fused_bc_scan`

backward:

- `_fox_state_build_backward_torch`

#### value_dim = 64

默认 impl 选择逻辑:

- 先进入 `dim64`
- 在没有 override 时, 默认会自动升级成 cached v64 实现
- 只有满足更严格 shape 条件时, 才会进一步进入 `owner_constexpr`

当前实验中:

- `block_len = 8`
- 不满足 `owner_constexpr` 所要求的 `block_len in {32, 64, 128}`

所以默认实际落在:

- `triton_v64_fw_cached_bw`

forward:

- `k_suffix_gamma`
- `k_fused_bc_scan_v64`

backward:

- `k_state_build_backward_scan_cached_v64`
- `k_state_build_grad_logf_from_cached_decay_v64`

#### value_dim = 128

默认 impl:

- `triton_generic_fw_torch_bw`

forward:

- `k_suffix_gamma`
- `k_fused_bc_scan`

backward:

- `_fox_state_build_backward_torch`

### 3.3 remote path

默认 dispatch:

- `value_dim == 64` -> `dispatch_tag = "dim64"`
- `value_dim == 128` -> `dispatch_tag = "dim128"`
- 其它 -> `dispatch_tag = "generic"`

对应默认 impl:

- `dim64 -> triton_v64_fw_owner_bw_v2`
- `dim128 -> triton_v128_fw_torch_bw`
- `generic -> triton_generic_fw_torch_bw`

#### value_dim = 32

forward:

- `k_remote_reduce_online`

backward:

- `_fox_remote_path_backward_torch`

#### value_dim = 64

forward:

- `k_remote_reduce_online_v64_train_aux`

backward:

- `k_remote_reduce_backward_rows_v64_v2`

#### value_dim = 128

forward:

- `k_remote_reduce_online_v128`

backward:

- `_fox_remote_path_backward_torch`

## 4. 当前实验 sweep 的直接对应关系

当前统一实验入口 `zoology/experiments/flash_vqg/flash_vqg_suite.py` 生成的主对比配置中, Flash-VQG 采用:

- `H = 2`
- `d_model in {64, 128, 256}`

因此直接对应为:

- `d_model=64 -> value_dim=32 -> generic 档`
- `d_model=128 -> value_dim=64 -> 最完整的默认 Triton 档`
- `d_model=256 -> value_dim=128 -> v128 remote forward 档`

## 5. 代码锚点

- shortcode:
  - `Flash-VQG/src/flash_vqg/nn/vq.py`
  - `Flash-VQG/src/flash_vqg/nn/vq_triton.py`
- state build:
  - `Flash-VQG/src/flash_vqg/nn/attn_fox.py:66-110`
  - `Flash-VQG/src/flash_vqg/nn/attn_fox.py:539-619`
  - `Flash-VQG/src/flash_vqg/nn/attn_fox.py:689-770`
  - `Flash-VQG/src/flash_vqg/nn/attn_fox.py:1041-1100`
  - `Flash-VQG/src/flash_vqg/nn/attn_fox.py:1232-1325`
- remote path:
  - `Flash-VQG/src/flash_vqg/nn/attn_fox.py:1710-1904`
  - `Flash-VQG/src/flash_vqg/nn/attn_fox.py:1987-2065`
  - `Flash-VQG/src/flash_vqg/nn/attn_fox.py:2380-2555`
