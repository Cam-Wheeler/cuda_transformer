#!/usr/bin/env bash

set -euo pipefail
if [[ $# -lt 1 ]]; then
  echo "usage: $0 <kernel>" >&2
  echo "example: $0 matmul" >&2
  exit 1
fi

# Grab the kernel and the writing dir.
KERNEL="$1"
OUT="/data/profiling/${KERNEL}_l1"

# Make the dir if it doesnt exist.
mkdir -p "$(dirname "$OUT")"

# Lets profile! 
nsys profile \
  -o "$OUT" \
  --force-overwrite=true \
  --trace=cuda,nvtx \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  python layer_1_bench.py --kernel "$KERNEL"