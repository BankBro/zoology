#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export GDN_EXPANDED_K_PAIRS="${GDN_EXPANDED_K_PAIRS:-8:2}"
export LAUNCH_ID_PREFIX="${LAUNCH_ID_PREFIX:-flash-vqg-20260528-kblocked-probe-ek8-ev2-20260528T070753Z}"
export RUN_ID_OVERRIDE="${RUN_ID_OVERRIDE:-gdnxk-h2-ek8-ev2-s123-d123-b64-ga4-fp32-noearly4ep-kblocked-ieee-20260528T070753Z}"
export ARTIFACT_DIR="${ARTIFACT_DIR:-${ROOT_DIR:-/home/lyj/mnt/project/worktrees/fla-kblocked/zoology}/artifacts/fla-kblocked-kernel/training-probe-ek8-ev2-20260528T070753Z}"

exec "${SCRIPT_DIR}/run_kblocked_training_probe.sh"
