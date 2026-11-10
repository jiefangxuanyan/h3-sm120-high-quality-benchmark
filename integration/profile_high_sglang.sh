#!/usr/bin/env bash
# Capture a separate 50-step timeline; profiling durations are not benchmark results.
set -euo pipefail
H3_PROFILE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
mode="${1:-rdma}"
if (( $# )); then shift; fi
case "$mode" in nccl|rdma|pcie) ;; *) echo 'Expected nccl, rdma or pcie' >&2; exit 2;; esac
: "${MODEL_PATH:?Set MODEL_PATH}"
NSYS_BIN="${NSYS_BIN:-nsys}"
NSYS_REPORT_DIR="${NSYS_REPORT_DIR:-$H3_PROFILE_ROOT/profiles}"
NSYS_REPORT_NAME="${NSYS_REPORT_NAME:-h3-high-sage-ulysses-$mode-bf16}"
mkdir -p "$NSYS_REPORT_DIR"
export RUN_DIR="${RUN_DIR:-$NSYS_REPORT_DIR/run-$mode}"
# Keep Nsight I/O local; publish only the finished report.
work="$(mktemp -d "${H3_NSYS_WORK_DIR:-/tmp}/h3-nsys-${UID}-XXXXXX")"
trap 'rm -rf -- "$work"' EXIT
# Capture only request 2 after one complete request on the same workers.
export H3_FULL_REQUEST_WARMUP=1
export SGLANG_H3_NSYS_CAPTURE=1
export NSYS_NVTX_PROFILER_REGISTER_ONLY=0
export SGLANG_DIFFUSION_ULYSSES_NVTX=1
if [[ "$mode" == rdma ]]; then
    nsys_dir="$(dirname "$(readlink -f "$(command -v "$NSYS_BIN")")")"
    export FLASHINFER_ULYSSES_PCIE_NVTX_INCLUDE="$nsys_dir/nvtx/include"
    test -f "$FLASHINFER_ULYSSES_PCIE_NVTX_INCLUDE/nvtx3/nvToolsExt.h"
fi
NSYS_TMPDIR="$work" "$NSYS_BIN" profile --output="$work/$NSYS_REPORT_NAME" \
    --force-overwrite=true --trace=cuda,nvtx,osrt,nccl --nic-metrics=lf \
    --capture-range=nvtx --nvtx-capture=H3_MEASURED_REQUEST --capture-range-end=stop \
    --sample=none --cpuctxsw=none --wait=all \
    bash "$H3_PROFILE_ROOT/integration/test_high_sglang.sh" "$mode" \
    --enable-layerwise-nvtx-marker true "$@"
cp -- "$work/$NSYS_REPORT_NAME.nsys-rep" "$NSYS_REPORT_DIR/"
