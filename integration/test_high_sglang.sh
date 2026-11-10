#!/usr/bin/env bash
# High-quality Sage quantization -> A2A -> CAKE SSA.
set -euo pipefail
H3_ALL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
source "$H3_ALL_ROOT/integration/env-all.sh"
: "${MODEL_PATH:?Set MODEL_PATH to a complete MiniMax-H3 checkpoint}"
mode="${1:-all}"
if (( $# )); then shift; fi
gpu_ids=0,1,2,3,4,5,6,7
gpus_explicit=0
extra=()
while (( $# )); do
    case "$1" in
        --gpus) gpu_ids="${2:?--gpus requires a comma-separated GPU list}"; gpus_explicit=1; shift 2 ;;
        *) extra+=("$1"); shift ;;
    esac
done
if [[ ! "$gpu_ids" =~ ^[0-7](,[0-7])*$ ]]; then
    echo '--gpus requires comma-separated GPU indices 0..7' >&2; exit 2
fi
IFS=, read -r -a devices <<< "$gpu_ids"
num_gpus="${#devices[@]}"
case "$num_gpus" in 1|2|4|8) ;; *) echo 'Expected 1, 2, 4, or 8 GPUs' >&2; exit 2;; esac
declare -A seen=()
for device in "${devices[@]}"; do
    if [[ -v "seen[$device]" ]]; then echo 'Duplicate GPU index' >&2; exit 2; fi
    seen[$device]=1
done
export CUDA_DEVICE_ORDER=PCI_BUS_ID
set -- "${extra[@]}"
placement=()
selection=()
if (( gpus_explicit )); then
    placement=(--gpu-ids "$gpu_ids")
    selection=(--gpus "$gpu_ids")
fi
RUN_DIR="${RUN_DIR:-$H3_ALL_ROOT/integration/high-runs}"
mkdir -p "$RUN_DIR"
case "$mode" in
    all)
        modes=(nccl rdma pcie)
        if (( num_gpus == 1 )); then modes=(nccl); fi
        for item in "${modes[@]}"; do bash "$0" "$item" "${selection[@]}" "$@"; done
        exit 0 ;;
    nccl) transport=nccl; unset FLASHINFER_ULYSSES_PCIE_ROUTE ;;
    rdma) transport=flashinfer-pcie; export FLASHINFER_ULYSSES_PCIE_ROUTE=rdma ;;
    pcie)
        transport=flashinfer-pcie
        export FLASHINFER_ULYSSES_PCIE_ROUTE=p2p
        # Auxiliary NCCL collectives must not bypass the PCIe scenario via a NIC.
        # NCCL 2.29.7 src/graph/paths.cc: ncclTopoCheckNet.
        export NCCL_NET_DISABLE_INTRA=1 ;;
    *) echo "Expected nccl, rdma, pcie, or all" >&2; exit 2 ;;
esac
backend=subblock_sparse_attn_sm120_sage_a2a
strict=(--minimax-h3-ulysses-strict)
if (( num_gpus == 1 )); then
    if [[ "$mode" != nccl ]]; then
        echo 'Single GPU uses the nccl-labelled local baseline without A2A.' >&2; exit 2
    fi
    backend=subblock_sparse_attn_sm120_sage
    strict=()
fi
unset SGLANG_DIFFUSION_DISABLE_MINIMAX_H3_VAE_FAST_PATH
flags=(
    --model-path "$MODEL_PATH" --config "${CONFIG:-$H3_ALL_ROOT/integration/examples/t2va.json}"
    --num-gpus "$num_gpus" "${placement[@]}" --tp-size 1 --ulysses-degree "$num_gpus" --ring-degree 1
    --num-inference-steps 50 --quality high
    --performance-mode speed --enable-torch-compile false
    --quantization fp8 --minimax-h3-adaln-online true --minimax-h3-adaln-plan-width 2
    --minimax-h3-adaln-host-cache-gb 2
    --text-encoder-cpu-offload true --component-residency video_vae=resident
    --component-attention-backends "transformer=$backend,video_vae=torch_cudnn_sdpa"
    --attention-backend-config "${ATTENTION_CONFIG:-sparsity=0.75,skip_first_steps=10,skip_first_layers=0,n_k=4,n_q=4,min_seq_len=4096}"
    --minimax-h3-ulysses-transport "$transport"
    --minimax-h3-ulysses-max-seq-len "${MAX_SEQ_LEN:-131072}"
    --minimax-h3-ulysses-max-sequence-geometries "${MAX_GEOMETRIES:-4}"
    "${strict[@]}"
    --output-file-path "$RUN_DIR/$mode.mp4"
    --perf-dump-path "$RUN_DIR/$mode-perf.json" --save-output
)
if [[ -n "${MODEL_REVISION:-}" ]]; then
    flags+=(--revision "$MODEL_REVISION")
fi
printf 'H3 profile: quality=high, native Cache-DiT=(4,0.04,1), Sage Ulysses required, VAE=BF16, transport=%s route=%s\n' "$transport" "${FLASHINFER_ULYSSES_PCIE_ROUTE:-nccl}"
if [[ "${H3_FULL_REQUEST_WARMUP:-1}" == 1 ]]; then
    python "$H3_ALL_ROOT/integration/warmup_generate.py" "${flags[@]}" "$@" 2>&1 | tee "$RUN_DIR/$mode.log"
else
    sglang generate "${flags[@]}" "$@" 2>&1 | tee "$RUN_DIR/$mode.log"
fi
