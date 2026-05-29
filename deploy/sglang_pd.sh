#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CASE_NAME="${CASE_NAME:?CASE_NAME is required}"
MODE="${MODE:?MODE is required}"
PD_LAYOUT="${PD_LAYOUT:-}"
MODEL_PATH="${MODEL_PATH:?MODEL_PATH is required}"
HOST="${HOST:-127.0.0.1}"
BASE_PORT="${BASE_PORT:-18000}"
TP_SIZE="${TP_SIZE:-1}"
PP_SIZE="${PP_SIZE:-1}"
DP_SIZE="${DP_SIZE:-1}"
EP_SIZE="${EP_SIZE:-1}"
ATTEN_CP_SIZE="${ATTEN_CP_SIZE:-1}"
QUANTIZATION="${QUANTIZATION:-}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"

LOG_DIR="${ROOT_DIR}/logs/${CASE_NAME}"
PID_FILE="${LOG_DIR}/pids.txt"
mkdir -p "${LOG_DIR}"
: > "${PID_FILE}"

layout_count() {
  local layout="$1"
  local role="$2"
  if [[ -z "${layout}" || "${layout}" == "null" ]]; then
    echo 0
    return
  fi
  if [[ "${layout}" =~ ^([0-9]+)P([0-9]+)D$ ]]; then
    if [[ "${role}" == "prefill" ]]; then
      echo "${BASH_REMATCH[1]}"
    else
      echo "${BASH_REMATCH[2]}"
    fi
    return
  fi
  echo "Invalid PD_LAYOUT: ${layout}" >&2
  exit 2
}

next_port() {
  local offset="$1"
  echo "$((BASE_PORT + offset))"
}

visible_devices_for_rank() {
  local rank="$1"
  local width="${TP_SIZE}"
  local start=$((rank * width))
  local devices=()
  for ((i = 0; i < width; i++)); do
    devices+=("$((start + i))")
  done
  local IFS=,
  echo "${devices[*]}"
}

common_args=(
  "--model-path" "${MODEL_PATH}"
  "--host" "${HOST}"
  "--tp" "${TP_SIZE}"
  "--max-total-tokens" "${MAX_NUM_BATCHED_TOKENS}"
  "--max-running-requests" "${MAX_NUM_SEQS}"
)

if [[ -n "${QUANTIZATION}" && "${QUANTIZATION}" != "null" ]]; then
  common_args+=("--quantization" "${QUANTIZATION}")
fi

# Override these templates for the exact SGLang PD build in use.
SGLANG_NORMAL_CMD_TEMPLATE="${SGLANG_NORMAL_CMD_TEMPLATE:-python -m sglang.launch_server}"
SGLANG_PREFILL_CMD_TEMPLATE="${SGLANG_PREFILL_CMD_TEMPLATE:-python -m sglang.launch_server}"
SGLANG_DECODE_CMD_TEMPLATE="${SGLANG_DECODE_CMD_TEMPLATE:-python -m sglang.launch_server}"
SGLANG_ROUTER_CMD_TEMPLATE="${SGLANG_ROUTER_CMD_TEMPLATE:-python -m sglang.launch_server}"

start_component() {
  local name="$1"
  local rank="$2"
  local port="$3"
  shift 3
  local log_file="${LOG_DIR}/${name}_${rank}.log"
  local devices
  devices="$(visible_devices_for_rank "${rank}")"
  echo "Starting ${name}-${rank} on ${HOST}:${port}, CUDA_VISIBLE_DEVICES=${devices}" | tee -a "${LOG_DIR}/deploy.log"
  (
    export CUDA_VISIBLE_DEVICES="${devices}"
    "$@" "${common_args[@]}" "--port" "${port}"
  ) > "${log_file}" 2>&1 &
  echo "$!" >> "${PID_FILE}"
}

start_router() {
  local port="$1"
  local log_file="${LOG_DIR}/sglang_router.log"
  echo "Starting sglang_router on ${HOST}:${port}, CUDA_VISIBLE_DEVICES=${ROUTER_CUDA_VISIBLE_DEVICES:-}" | tee -a "${LOG_DIR}/deploy.log"
  (
    export CUDA_VISIBLE_DEVICES="${ROUTER_CUDA_VISIBLE_DEVICES:-}"
    ${SGLANG_ROUTER_CMD_TEMPLATE} "${common_args[@]}" "--port" "${port}"
  ) > "${log_file}" 2>&1 &
  echo "$!" >> "${PID_FILE}"
}

if [[ "${MODE}" == "normal" ]]; then
  start_component "sglang_normal" 0 "$(next_port 0)" ${SGLANG_NORMAL_CMD_TEMPLATE}
elif [[ "${MODE}" == "pd" ]]; then
  prefill_count="$(layout_count "${PD_LAYOUT}" prefill)"
  decode_count="$(layout_count "${PD_LAYOUT}" decode)"
  for ((i = 0; i < prefill_count; i++)); do
    start_component "sglang_prefill" "${i}" "$(next_port $((10 + i)))" ${SGLANG_PREFILL_CMD_TEMPLATE}
  done
  for ((i = 0; i < decode_count; i++)); do
    start_component "sglang_decode" "$((prefill_count + i))" "$(next_port $((30 + i)))" ${SGLANG_DECODE_CMD_TEMPLATE}
  done
  start_router "$(next_port 0)"
else
  echo "Unsupported MODE for SGLang: ${MODE}" >&2
  exit 2
fi

echo "${HOST}:$(next_port 0)" > "${LOG_DIR}/endpoint.txt"
echo "Deployment started. Logs: ${LOG_DIR}"
