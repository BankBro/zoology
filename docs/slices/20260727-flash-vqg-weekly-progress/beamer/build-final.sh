#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FINAL_DIR="${SCRIPT_DIR}/../final"
BUILD_DIR="$(mktemp -d)"

cleanup() {
  rm -rf "${BUILD_DIR}"
}
trap cleanup EXIT

mkdir -p "${FINAL_DIR}"
cd "${SCRIPT_DIR}"

xelatex \
  -interaction=nonstopmode \
  -halt-on-error \
  -output-directory="${BUILD_DIR}" \
  weekly-progress.tex

xelatex \
  -interaction=nonstopmode \
  -halt-on-error \
  -output-directory="${BUILD_DIR}" \
  weekly-progress.tex

if rg -n "LaTeX Error|Fatal error|Overfull|Underfull" "${BUILD_DIR}/weekly-progress.log"; then
  printf '警告: 编译日志中发现需要检查的项目.\n' >&2
  exit 1
fi

cp "${BUILD_DIR}/weekly-progress.pdf" "${FINAL_DIR}/weekly-progress.pdf"
printf 'PDF: %s\n' "${FINAL_DIR}/weekly-progress.pdf"
