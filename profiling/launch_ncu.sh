#!/usr/bin/env bash

set -euo pipefail
if [[ $# -lt 1 ]]; then
  echo "usage: $0 <kernel>" >&2
  echo "example: $0 matmul" >&2
  exit 1
fi

# Grab the kernel and the writing dir.
KERNEL="$1"
OUT="/data/profiling/${KERNEL}_l2"

# Make the dir if it doesnt exist.
mkdir -p "$(dirname "$OUT")"

case "$KERNEL" in
  matmul)        FILTER='regex:fwd_matmul$' ;;
  batch_matmul)  FILTER='regex:fwd_batched_matmul$' ;;
  addition)      FILTER='regex:fwd_add$' ;;
  multi)         FILTER='regex:fwd_multi$' ;;
  softmax)       FILTER='regex:fwd_softmax$' ;;
  rmsnorm)       FILTER='regex:fwd_rmsnorm$' ;;
  *)
    echo "unknown kernel: $KERNEL" >&2
    echo "expected: matmul|batch_matmul|addition|multi|softmax|rmsnorm" >&2
    exit 1
    ;;
esac

# Lets profile! 
ncu \
  --profile-from-start off \
  --kernel-name "$FILTER" \
  --launch-count 3 \
  --section SpeedOfLight \
  --section WarpStateStats \
  --section MemoryWorkloadAnalysis \
  --section Occupancy \
  --section LaunchStats \
  -o "$OUT" \
  --force-overwrite \
  python layer_one_or_two_bench.py --kernel "$KERNEL" --layer "2"