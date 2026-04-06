#!/usr/bin/env bash
#
# Download CED ONNX models from Hugging Face into soundevents/models/.
#
# Source: https://huggingface.co/mispeech (ced-tiny, ced-mini, ced-small, ced-base)
#
# The four ONNX files are already checked into this repo, so you should not
# normally need to run this script. Use it after upstream releases new weights,
# to reproduce the on-disk files in a clean checkout, or to add a new variant.
#
# Requires:
#   huggingface_hub  (install with: pip install --user huggingface_hub)
#
# Usage:
#   ./scripts/download_models.sh                # fetch all four variants
#   ./scripts/download_models.sh tiny mini      # fetch a subset

set -euo pipefail

if ! command -v huggingface-cli >/dev/null 2>&1; then
  echo "error: huggingface-cli not found on PATH." >&2
  echo "       install with: pip install --user huggingface_hub" >&2
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${REPO_ROOT}/soundevents/models"
mkdir -p "$DEST"

if [[ $# -gt 0 ]]; then
  variants=("$@")
else
  variants=(tiny mini small base)
fi

for variant in "${variants[@]}"; do
  case "$variant" in
    tiny|mini|small|base) ;;
    *)
      echo "error: unknown variant '$variant' (expected: tiny, mini, small, base)" >&2
      exit 1
      ;;
  esac

  repo="mispeech/ced-${variant}"
  cache="${DEST}/.cache-${variant}"
  echo "→ fetching ${repo}"

  rm -rf "$cache"
  huggingface-cli download \
    "$repo" \
    --include "*.onnx" \
    --local-dir "$cache"

  # Find the downloaded .onnx file and place it as <variant>.onnx.
  onnx="$(find "$cache" -name '*.onnx' -type f | head -n1)"
  if [[ -z "$onnx" ]]; then
    echo "  error: no .onnx file found in ${repo}" >&2
    echo "         check https://huggingface.co/${repo}/tree/main for the file listing" >&2
    rm -rf "$cache"
    exit 1
  fi

  mv "$onnx" "${DEST}/${variant}.onnx"
  rm -rf "$cache"
  printf "  ok: %s (%s)\n" "${variant}.onnx" "$(du -h "${DEST}/${variant}.onnx" | cut -f1)"
done

echo
echo "done. ${DEST} now contains:"
ls -lh "${DEST}"/*.onnx
