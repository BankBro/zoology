#!/usr/bin/env bash
set -euo pipefail

: "${MQAR_LATE_DEGRADATION_RUN_TAG:?MQAR_LATE_DEGRADATION_RUN_TAG is required}"

export CUDA_VISIBLE_DEVICES=0
export TRITON_F32_DEFAULT=ieee
export NVIDIA_TF32_OVERRIDE=0
export GDN_KERNEL_DTYPE=bfloat16

exec /home/lyj/miniconda3/envs/flash-vqg-fla042/bin/python \
  zoology/experiments/flash_vqg/scripts/20260801-02-flash-late-degradation-causal-diagnosis/run_queue.py
