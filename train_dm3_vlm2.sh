#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./train_dm3_vlm.sh <carla_port> <gpu_id> [dreamerv3/train_vlm.py args...]
#
# Example:
#   ./train_dm3_vlm.sh 3000 0 --task carla_lane_following_vlm --dreamerv3.logdir logs/... --env.seed 0 ...

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <carla_port> <gpu_id> [additional_training_parameters]"
  exit 1
fi

CARLA_PORT="$1"
GPU_ID="$2"
shift 2

CARLA_SERVER_COMMAND="${CARLA_ROOT}/CarlaUE4.sh -RenderOffScreen -carla-port=${CARLA_PORT} -benchmark -fps=10"
TRAINING_SCRIPT="dreamerv3/train_vlm.py"

# JAX device selection is handled by these CLI flags (passed through by this wrapper):
#   --dreamerv3.jax.policy_devices <GPU_ID>
#   --dreamerv3.jax.train_devices  <GPU_ID>
#
# Do NOT set CUDA_VISIBLE_DEVICES for python here unless you also adjust those flags.
COMMON_PARAMS="--env.world.carla_port ${CARLA_PORT} --dreamerv3.jax.policy_devices ${GPU_ID} --dreamerv3.jax.train_devices ${GPU_ID}"

# Try to detect --dreamerv3.logdir <path> from remaining args so logs go into the run folder.
LOGDIR=""
ARGS=("$@")
for ((i=0; i<${#ARGS[@]}; i++)); do
  if [[ "${ARGS[$i]}" == "--dreamerv3.logdir" ]] && (( i+1 < ${#ARGS[@]} )); then
    LOGDIR="${ARGS[$i+1]}"
    break
  fi
done

if [[ -z "${LOGDIR}" ]]; then
  # Fallback: local logs directory
  LOGDIR="logs/_wrapper_fallback"
fi

mkdir -p "${LOGDIR}"
LOG_FILE="${LOGDIR}/wrapper_gpu${GPU_ID}_port${CARLA_PORT}.log"
: > "${LOG_FILE}"

log_ts() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${LOG_FILE}"
}

cleanup() {
  log_ts "Cleanup: stopping CARLA on port ${CARLA_PORT} (if any)."
  # Kill whatever is on that port (CARLA) — ignore errors
  fuser -k "${CARLA_PORT}/tcp" >/dev/null 2>&1 || true
}
trap cleanup EXIT SIGINT SIGTERM

launch_carla() {
  # If port is not open, start CARLA
  if ! nc -z localhost "${CARLA_PORT}" >/dev/null 2>&1; then
    log_ts "Starting CARLA on port ${CARLA_PORT} using GPU ${GPU_ID}"
    fuser -k "${CARLA_PORT}/tcp" >/dev/null 2>&1 || true

    # Pin CARLA process to the requested GPU
    CUDA_VISIBLE_DEVICES="${GPU_ID}" ${CARLA_SERVER_COMMAND} >> "${LOG_FILE}" 2>&1 &

    # Wait until the port opens
    until nc -z localhost "${CARLA_PORT}" >/dev/null 2>&1; do
      log_ts "Waiting for CARLA to open port ${CARLA_PORT}..."
      sleep 1
    done
    log_ts "CARLA is up on port ${CARLA_PORT}."
  else
    log_ts "CARLA already listening on port ${CARLA_PORT}."
  fi
}

TRAINING_COMMAND=(python -u "${TRAINING_SCRIPT}" ${COMMON_PARAMS} "$@")

while true; do
  launch_carla
  log_ts "Launching training:"
  log_ts "  ${TRAINING_COMMAND[*]}"

  set +e
  "${TRAINING_COMMAND[@]}" >> "${LOG_FILE}" 2>&1
  code=$?
  set -e

  if [[ $code -eq 0 ]]; then
    log_ts "Training finished normally (exit 0). Exiting wrapper."
    exit 0
  else
    log_ts "Training crashed (exit ${code}). Restarting after CARLA reset..."
    fuser -k "${CARLA_PORT}/tcp" >/dev/null 2>&1 || true
    sleep 2
  fi
done
