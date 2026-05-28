#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export GDN_EXPANDED_K_PAIRS="${GDN_EXPANDED_K_PAIRS:-16:1}"
export LAUNCH_ID_PREFIX="${LAUNCH_ID_PREFIX:-flash-vqg-20260528-kblocked-probe-ek16-ev1-retry-ieee-20260528T020346Z}"
export RUN_ID_OVERRIDE="${RUN_ID_OVERRIDE:-gdnxk-h2-ek16-ev1-s123-d123-b64-ga4-fp32-noearly4ep-retry-ieee-20260528T020346Z}"
export ARTIFACT_DIR="${ARTIFACT_DIR:-${ROOT_DIR:-/home/lyj/mnt/project/worktrees/fla-kblocked/zoology}/artifacts/fla-kblocked-kernel/training-probe-ek16-ev1-retry-ieee-20260528T020346Z}"

exec "${SCRIPT_DIR}/run_kblocked_training_probe.sh"
