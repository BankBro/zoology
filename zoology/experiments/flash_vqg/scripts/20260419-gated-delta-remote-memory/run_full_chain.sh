#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WRITE_DIR="${SCRIPT_DIR}/write-mainline"
DIAG_DIR="${SCRIPT_DIR}/diagnostics-read"

echo "[full-chain] stage=write-mainline start_utc=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
bash "${WRITE_DIR}/run_write_train.sh"
echo "[full-chain] stage=write-mainline done_utc=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"

echo "[full-chain] stage=diagnostics-read start_utc=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
bash "${DIAG_DIR}/run_read_diagnostics.sh"
echo "[full-chain] stage=diagnostics-read done_utc=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
