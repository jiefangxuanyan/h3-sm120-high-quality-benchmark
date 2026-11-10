#!/usr/bin/env bash
H3_ALL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
export SGLANG_SRC="${SGLANG_SRC:-$H3_ALL_ROOT/sglang-all}"
export FLASHINFER_SRC="${FLASHINFER_SRC:-$H3_ALL_ROOT/flashinfer}"
export PYTHONPATH="$SGLANG_SRC/python:$FLASHINFER_SRC${PYTHONPATH:+:$PYTHONPATH}"
