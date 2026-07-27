#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ $# -lt 1 || $# -gt 2 ]]; then
  printf '用法: %s <page-source> [output.pdf]\n' "$0" >&2
  exit 2
fi

PAGE_SOURCE="${1#./}"
if [[ "${PAGE_SOURCE}" == /* || ! -f "${SCRIPT_DIR}/${PAGE_SOURCE}" ]]; then
  printf '错误: page-source 必须是相对于 %s 的现有 tex 文件.\n' "${SCRIPT_DIR}" >&2
  exit 2
fi

if [[ $# -eq 2 ]]; then
  OUTPUT_PATH="$2"
else
  SOURCE_PARENT="$(dirname "${PAGE_SOURCE}")"
  case "${SOURCE_PARENT}" in
    weeks/*)
      PREVIEW_SUBDIR="${SOURCE_PARENT#weeks/}"
      ;;
    *)
      PREVIEW_SUBDIR="${SOURCE_PARENT}"
      ;;
  esac
  OUTPUT_PATH="${SCRIPT_DIR}/previews/${PREVIEW_SUBDIR}/$(basename "${PAGE_SOURCE}" .tex).pdf"
fi
mkdir -p "$(dirname "${OUTPUT_PATH}")"

BUILD_DIR="$(mktemp -d)"
cleanup() {
  rm -rf "${BUILD_DIR}"
}
trap cleanup EXIT

cd "${SCRIPT_DIR}"
xelatex \
  -interaction=nonstopmode \
  -halt-on-error \
  -jobname=page-preview \
  -output-directory="${BUILD_DIR}" \
  "\\def\\PageSource{${PAGE_SOURCE}}\\input{page-preview.tex}"

if rg -n 'LaTeX Error|Fatal error|Overfull|Underfull' "${BUILD_DIR}/page-preview.log"; then
  printf '错误: 单页编译日志中发现错误或版面溢出, 未生成预览.\n' >&2
  exit 1
fi

cp "${BUILD_DIR}/page-preview.pdf" "${OUTPUT_PATH}"
printf 'PDF: %s\n' "${OUTPUT_PATH}"
