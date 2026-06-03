#!/usr/bin/env bash
set -euo pipefail

ROLE="${1:-}"
MASTER_IP="${2:-}"
NODE_RANK="${3:-0}"
LAYOUT="${4:-tp8cp8}"

if [[ -z "${ROLE}" || -z "${MASTER_IP}" ]]; then
  echo "Usage: $0 <prefill|decode|router|cleanup> <master_ip> [node_rank] [layout]" >&2
  exit 2
fi

WORKDIR="${WORKDIR:-/mnt11/task/sgl}"
MODEL_PATH="${MODEL_PATH:-/model/vllm-w8a8-models/GLM-5-W8A8}"
PORT="${PORT:-30000}"
ROUTER_PORT="${ROUTER_PORT:-30020}"
PREFILL_URL="${PREFILL_URL:-http://${MASTER_IP}:${PORT}}"
DECODE_URL="${DECODE_URL:-}"
LOG_ROOT="${LOG_ROOT:-${WORKDIR}/pd_test_runs/glm5_int8_pd}"
HOST="${HOST_IP:-$(hostname -I 2>/dev/null | awk '{print $1}')}"
if [[ -z "${HOST}" ]]; then
  HOST="$(hostname -i 2>/dev/null | awk '{print $1}')"
fi

mkdir -p "${LOG_ROOT}/logs"
cd "${WORKDIR}"

export NCCL_MIN_NCHANNELS="${NCCL_MIN_NCHANNELS:-16}"
export NCCL_MAX_NCHANNELS="${NCCL_MAX_NCHANNELS:-16}"
export SGLANG_ENABLE_SPEC_V2="${SGLANG_ENABLE_SPEC_V2:-1}"
export HSA_ENABLE_COREDUMP="${HSA_ENABLE_COREDUMP:-1}"
export USE_DCU_CUSTOM_ALLREDUCE="${USE_DCU_CUSTOM_ALLREDUCE:-1}"
export ALLREDUCE_STREAM_WITH_COMPUTE="${ALLREDUCE_STREAM_WITH_COMPUTE:-1}"
export HIP_KERNEL_EVENT_SYSTENFENCE="${HIP_KERNEL_EVENT_SYSTENFENCE:-1}"
export SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD="${SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD:-0}"
export GLIBC_TUNABLES="${GLIBC_TUNABLES:-glibc.rtld.optional_static_tls=0x40000}"
export HIP_KERNEL_BATCH_CEILING="${HIP_KERNEL_BATCH_CEILING:-100}"
export GPU_FORCE_BLIT_COPY_SIZE="${GPU_FORCE_BLIT_COPY_SIZE:-16}"
export HSA_KERNARG_POOL_SIZE="${HSA_KERNARG_POOL_SIZE:-8388608}"
export ROC_AQL_QUEUE_SIZE="${ROC_AQL_QUEUE_SIZE:-131072}"
export SGLANG_USE_LIGHTOP="${SGLANG_USE_LIGHTOP:-1}"
export W8A8_SUPPORT_METHODS="${W8A8_SUPPORT_METHODS:-3}"
export SGLANG_KVALLOC_KERNEL="${SGLANG_KVALLOC_KERNEL:-1}"
export SGLANG_CREATE_EXTEND_AFTER_DECODE_SPEC_INFO="${SGLANG_CREATE_EXTEND_AFTER_DECODE_SPEC_INFO:-1}"
export SGLANG_ASSIGN_EXTEND_CACHE_LOCS="${SGLANG_ASSIGN_EXTEND_CACHE_LOCS:-1}"
export SGLANG_ASSIGN_REQ_TO_TOKEN_POOL="${SGLANG_ASSIGN_REQ_TO_TOKEN_POOL:-1}"
export SGLANG_GET_LAST_LOC="${SGLANG_GET_LAST_LOC:-1}"
export SGLANG_CREATE_FLASHMLA_KV_INDICES_TRITON="${SGLANG_CREATE_FLASHMLA_KV_INDICES_TRITON:-1}"
export SGLANG_CREATE_CHUNKED_PREFIX_CACHE_KV_INDICES="${SGLANG_CREATE_CHUNKED_PREFIX_CACHE_KV_INDICES:-1}"
export HIP_GRAPH_ACCUMULATE_DISPATCH="${HIP_GRAPH_ACCUMULATE_DISPATCH:-1}"
export HIP_GRAPH_USE_CMD_CACHE="${HIP_GRAPH_USE_CMD_CACHE:-0}"
export ROCBLAS_TENSILE_LIBPATH="${ROCBLAS_TENSILE_LIBPATH:-/mnt11/nmz/sgl/auto_select_tools/optimization_configs/new/config/library_gpu6}"
export HIPBLASLT_TUNING_OVERRIDE_FILE="${HIPBLASLT_TUNING_OVERRIDE_FILE:-hipblaslt.config}"
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT="${SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT:-1200}"
export SGLANG_HEALTH_CHECK_TIMEOUT="${SGLANG_HEALTH_CHECK_TIMEOUT:-600}"
export MC_IB_GID_INDEX="${MC_IB_GID_INDEX:-0}"
export ROCSHMEM_IB_GID_INDEX="${ROCSHMEM_IB_GID_INDEX:-0}"
export MC_ENABLE_DEST_DEVICE_AFFINITY="${MC_ENABLE_DEST_DEVICE_AFFINITY:-1}"
export SGLANG_HOST_IP="${SGLANG_HOST_IP:-${HOST}}"
export MC_ALLOWED_IBV_DEVICES="${MC_ALLOWED_IBV_DEVICES:-mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9}"
export MC_TOPO_FILE_FORCE="${MC_TOPO_FILE_FORCE:-}"
export ROCSHMEM_DISABLE_HDP_FLUSH="${ROCSHMEM_DISABLE_HDP_FLUSH:-1}"
export ROCSHMEM_GDA_NUM_QPS_DEFAULT_CTX="${ROCSHMEM_GDA_NUM_QPS_DEFAULT_CTX:-288}"
export ROCSHMEM_HEAP_SIZE="${ROCSHMEM_HEAP_SIZE:-3173741824}"
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK="${SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK:-128}"
export ROCSHMEM_ALLOWED_IBV_DEVICES="${ROCSHMEM_ALLOWED_IBV_DEVICES:-mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9}"
export ROCSHMEM_TOPO_FILE_FORCE="${ROCSHMEM_TOPO_FILE_FORCE:-/mnt11/lijian/hg_deep_topo.config}"
export MOONCAKE_TE_META_DATA_SERVER="${MOONCAKE_TE_META_DATA_SERVER:-http://${MASTER_IP}:8080/metadata}"
export MOONCAKE_GLOBAL_SEGMENT_SIZE="${MOONCAKE_GLOBAL_SEGMENT_SIZE:-64GB}"
export MOONCAKE_PROTOCOL="${MOONCAKE_PROTOCOL:-tcp}"
export MOONCAKE_DEVICE="${MOONCAKE_DEVICE:-mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9}"
export MOONCAKE_MASTER="${MOONCAKE_MASTER:-${MASTER_IP}:50051}"

COMMON_ARGS=(
  --model-path "${MODEL_PATH}"
  --trust-remote-code
  --host "${HOST}"
  --port "${PORT}"
  --dist-init-addr "${MASTER_IP}:5000"
  --kv-cache-dtype fp8_e4m3
  --dtype bfloat16
  --mem-fraction-static "${MEM_FRACTION_STATIC:-0.9}"
  --page-size 64
  --nsa-prefill-backend flashmla_auto
  --nsa-decode-backend flashmla_kv
  --context-length "${CONTEXT_LENGTH:-131072}"
  --dist-timeout "${DIST_TIMEOUT:-10000}"
  --watchdog-timeout "${WATCHDOG_TIMEOUT:-3600}"
  --chunked-prefill-size "${CHUNKED_PREFILL_SIZE:--1}"
  --quantization slimquant_marlin
  --cuda-graph-max-bs "${CUDA_GRAPH_MAX_BS:-32}"
  --max-running-requests "${MAX_RUNNING_REQUESTS:-512}"
)

if [[ "${NNODES:-1}" != "1" ]]; then
  COMMON_ARGS+=(--nnodes "${NNODES}" --node-rank "${NODE_RANK}")
fi

SPEC_ARGS=(
  --speculative-algorithm EAGLE
  --speculative-num-steps 3
  --speculative-eagle-topk 1
  --speculative-num-draft-tokens 4
)

layout_args() {
  case "${LAYOUT}" in
    tp)
      echo "--tp-size ${TP_SIZE:-8} --pp-size 1"
      ;;
    pp)
      export SGLANG_PP_LAYER_PARTITION="${SGLANG_PP_LAYER_PARTITION:-8,8,8,8,8,8,8,8}"
      echo "--tp-size 1 --pp-size ${PP_SIZE:-8}"
      ;;
    tp4pp2)
      export SGLANG_PP_LAYER_PARTITION="${SGLANG_PP_LAYER_PARTITION:-32,32}"
      echo "--tp-size 4 --pp-size 2"
      ;;
    cp|tp8cp8)
      echo "--tp-size 8 --pp-size 1 --attn-cp-size 8 --enable-nsa-prefill-context-parallel --nsa-prefill-cp-mode round-robin-split"
      ;;
    dp-tpmoe)
      echo "--tp-size ${TP_SIZE:-4} --dp-size ${DP_SIZE:-2} --moe-dense-tp-size 1 --enable-dp-attention --enable-dp-lm-head"
      ;;
    tp8ep8)
      echo "--tp-size 8 --pp-size 1 --ep-size 8 --moe-dense-tp-size 1"
      ;;
    deepep)
      echo "--tp-size ${TP_SIZE:-8} --pp-size 1 --ep-size ${EP_SIZE:-8} --moe-a2a-backend deepep --deepep-mode low_latency --moe-dense-tp-size 1 --enable-dp-lm-head"
      ;;
    *)
      echo "Unknown layout: ${LAYOUT}" >&2
      exit 2
      ;;
  esac
}

cleanup() {
  pkill -f "sglang.*${MODEL_PATH}" 2>/dev/null || true
  pkill -f "sglang_router.launch_router" 2>/dev/null || true
}

if [[ "${ROLE}" == "cleanup" ]]; then
  cleanup
  exit 0
fi

if [[ "${ROLE}" == "router" ]]; then
  if [[ -z "${DECODE_URL}" ]]; then
    echo "DECODE_URL is required for router role" >&2
    exit 2
  fi
  python3 -m sglang_router.launch_router \
    --pd-disaggregation \
    --prefill "${PREFILL_URL}" \
    --decode "${DECODE_URL}" \
    --policy round_robin \
    --port "${ROUTER_PORT}" 2>&1 | tee "${LOG_ROOT}/logs/router_$(date +%m%d-%H%M).log"
  exit "${PIPESTATUS[0]}"
fi

ROLE_ARGS=()
if [[ "${ROLE}" == "prefill" ]]; then
  ROLE_ARGS=(
    --disaggregation-mode prefill
    --enable-hierarchical-cache
    --hicache-ratio 1
    --hicache-mem-layout page_first
    --hicache-io-backend kernel
    --hicache-write-policy write_through
    --hicache-storage-backend mooncake
    --hicache-storage-prefetch-policy best_effort
  )
elif [[ "${ROLE}" == "decode" ]]; then
  ROLE_ARGS=(
    --disaggregation-mode decode
    --disaggregation-decode-enable-offload-kvcache
    --hicache-ratio 1
    --hicache-mem-layout page_first
    --hicache-io-backend kernel
    --hicache-write-policy write_through
    --hicache-storage-backend mooncake
    --hicache-storage-prefetch-policy best_effort
  )
else
  echo "Unknown role: ${ROLE}" >&2
  exit 2
fi

IB_ARGS=(
  --disaggregation-ib-device mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_8,mlx5_9,mlx5_6,mlx5_7
)

read -r -a LAYOUT_ARGS <<< "$(layout_args)"
LOG_FILE="${LOG_ROOT}/logs/${ROLE}_${LAYOUT}_rank${NODE_RANK}_$(hostname)_$(date +%m%d-%H%M).log"

sglang serve \
  "${COMMON_ARGS[@]}" \
  "${SPEC_ARGS[@]}" \
  "${LAYOUT_ARGS[@]}" \
  "${IB_ARGS[@]}" \
  "${ROLE_ARGS[@]}" 2>&1 | tee "${LOG_FILE}"
