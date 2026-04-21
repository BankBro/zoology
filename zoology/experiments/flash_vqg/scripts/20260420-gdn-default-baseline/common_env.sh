#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/lyj/mnt/project/zoology"
PYTHON_BIN="${PYTHON_BIN:-/home/lyj/miniconda3/envs/flash-vqg/bin/python}"
GPU_ID="${GPU_ID:-0}"
PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
export PYTHONPATH

BACKEND="${BACKEND:-torch}"
DMODEL="${DMODEL:-128}"
LR="${LR:-1e-3}"
MAX_EPOCHS="${MAX_EPOCHS:-32}"
CACHE_DIR="${CACHE_DIR:-./data/flash_vqg}"
PROJECT="${PROJECT:-flash_vqg_vs_gdn}"
ENTITY="${ENTITY:-scu-mclab}"
ANALYSIS_SOURCE="${ANALYSIS_SOURCE:-local}"

TRAIN_BATCH_ORDER="${TRAIN_BATCH_ORDER:-global_shuffle}"
SEED_VALUES="${SEED_VALUES:-123}"
DATA_SEED="${DATA_SEED:-123}"

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"

LAUNCH_ID_PREFIX="${LAUNCH_ID_PREFIX:-flash-vqg-20260420-gdn-default-baseline}"
