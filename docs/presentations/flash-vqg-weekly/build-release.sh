#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

if [[ $# -lt 1 || $# -gt 2 ]]; then
  printf '用法: %s <release-dir> [output-name]\n' "$0" >&2
  exit 2
fi

RELEASE_DIR="$1"
OUTPUT_NAME="${2:-flash-vqg-weekly.pdf}"
if [[ "${RELEASE_DIR}" != /* ]]; then
  RELEASE_DIR="${REPO_ROOT}/${RELEASE_DIR}"
fi
FINAL_DIR="${RELEASE_DIR}/final"
BUILD_DIR="$(mktemp -d)"

cleanup() {
  rm -rf "${BUILD_DIR}"
}
trap cleanup EXIT

mkdir -p "${FINAL_DIR}"
cd "${SCRIPT_DIR}"

for _ in 1 2; do
  xelatex \
    -interaction=nonstopmode \
    -halt-on-error \
    -output-directory="${BUILD_DIR}" \
    main.tex
done

if rg -n 'LaTeX Error|Fatal error|Overfull|Underfull' "${BUILD_DIR}/main.log"; then
  printf '错误: 编译日志中发现错误或版面溢出, 未发布 PDF.\n' >&2
  exit 1
fi

cp "${BUILD_DIR}/main.pdf" "${FINAL_DIR}/${OUTPUT_NAME}"
sha256sum "${FINAL_DIR}/${OUTPUT_NAME}"
