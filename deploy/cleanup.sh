#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CASE_NAME="${CASE_NAME:-}"

if [[ -n "${CASE_NAME}" ]]; then
  pid_files=("${ROOT_DIR}/logs/${CASE_NAME}/pids.txt")
else
  mapfile -t pid_files < <(find "${ROOT_DIR}/logs" -name pids.txt -type f 2>/dev/null || true)
fi

for pid_file in "${pid_files[@]}"; do
  [[ -f "${pid_file}" ]] || continue
  while IFS= read -r pid; do
    [[ -n "${pid}" ]] || continue
    if kill -0 "${pid}" 2>/dev/null; then
      echo "Stopping PID ${pid}"
      kill "${pid}" 2>/dev/null || true
    fi
  done < "${pid_file}"
done

sleep 2

for pid_file in "${pid_files[@]}"; do
  [[ -f "${pid_file}" ]] || continue
  while IFS= read -r pid; do
    [[ -n "${pid}" ]] || continue
    if kill -0 "${pid}" 2>/dev/null; then
      echo "Force stopping PID ${pid}"
      kill -9 "${pid}" 2>/dev/null || true
    fi
  done < "${pid_file}"
  : > "${pid_file}"
done

echo "Cleanup complete"
