#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/common_env.sh"

cd "${ROOT_DIR}"

echo "==> Running clr_v1 E1 smoke on GPU ${GPU_ID}"
"${PYTHON_BIN}" "${SCRIPT_DIR}/smoke_e1_batch_accum.py" \
  --gpu "${GPU_ID}" \
  --metrics-white-list-file "${METRICS_WHITE_LIST_FILE}"
