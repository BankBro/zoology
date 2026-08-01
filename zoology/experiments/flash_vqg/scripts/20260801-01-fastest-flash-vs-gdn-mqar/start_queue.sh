#!/usr/bin/env bash
set -euo pipefail

: "${MQAR_FASTEST_GDN_RUN_TAG:?MQAR_FASTEST_GDN_RUN_TAG must be set}"
export CUDA_VISIBLE_DEVICES=0
export TRITON_F32_DEFAULT=ieee
export NVIDIA_TF32_OVERRIDE=0
export TORCH_DETERMINISTIC=0
export GDN_KERNEL_DTYPE=bfloat16

exec /home/lyj/miniconda3/envs/flash-vqg-fla042/bin/python \
  /home/lyj/mnt/project/zoology/zoology/experiments/flash_vqg/scripts/20260801-01-fastest-flash-vs-gdn-mqar/run_queue.py
