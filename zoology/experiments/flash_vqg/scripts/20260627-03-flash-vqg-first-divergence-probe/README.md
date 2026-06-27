# 20260627-03 Flash-VQG First-Divergence Probe

本目录只放 debug probe 入口, 不写 official ledger.

核心命令:

```bash
python zoology/experiments/flash_vqg/scripts/20260627-03-flash-vqg-first-divergence-probe/probe_first_divergence.py cache-hash \
  --machine-name 2080ti \
  --output-json zoology/experiments/flash_vqg/scripts/20260627-03-flash-vqg-first-divergence-probe/outputs/cache-2080ti.json

python zoology/experiments/flash_vqg/scripts/20260627-03-flash-vqg-first-divergence-probe/probe_first_divergence.py verify-init \
  --machine-name 2080ti \
  --checkpoint zoology/experiments/flash_vqg/scripts/20260627-02-flash-vqg-canonical-init-lock-screen/outputs/canonical-init/cb64r16-s123-init.pt \
  --output-json zoology/experiments/flash_vqg/scripts/20260627-03-flash-vqg-first-divergence-probe/outputs/init-2080ti.json

python zoology/experiments/flash_vqg/scripts/20260627-03-flash-vqg-first-divergence-probe/probe_first_divergence.py probe \
  --machine-name 2080ti \
  --variant baseline \
  --max-optimizer-steps 1 \
  --output-json zoology/experiments/flash_vqg/scripts/20260627-03-flash-vqg-first-divergence-probe/outputs/probe-2080ti-baseline.json
```

variants:

- `baseline`: 当前 `20260627-02` 配置, 不额外改 dtype policy.
- `strict-fp32`: 禁 TF32, `float32_matmul_precision=highest`, 开 PyTorch deterministic algorithms.
- `shadow-read`: 训练输出仍使用 `read_topk=2`, 只额外记录 full dense residual read 和 top-k residual read 的差异指标.
- `ref-gd`: 保持 `grouped_chunk_torch_ref` builder, 将 event pack 应用切到慢速 `loop_ref`, 用于检查 semivec/chunk pack 数值路径是否是放大点. `token_step_ref` 当前只适合 forward parity, 完整训练 backward 会触发 in-place autograd 错误, 不作为本轮跨机训练 probe. `ref-gd` 明显慢于其他 variant, 默认作为可选补充.

正式跨机器启动前必须先跑 `cache-hash` 和 `verify-init`, 两项不 match 直接停止.
