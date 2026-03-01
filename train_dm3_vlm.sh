#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <carla_port> <gpu_device> [additional_training_parameters]"
  exit 1
fi

CARLA_PORT="$1"
GPU_DEVICE="$2"
shift 2

LOG_FILE="log_${CARLA_PORT}.log"
TRAINING_SCRIPT="dreamerv3/train_vlm.py"

CARLA_CMD=(
  "$CARLA_ROOT/CarlaUE4.sh"
  -RenderOffScreen
  "-carla-port=${CARLA_PORT}"
  -benchmark
  -fps=10
)

PY_CMD=(
  python -u "$TRAINING_SCRIPT"
  --env.world.carla_port "$CARLA_PORT"
  --dreamerv3.jax.policy_devices "$GPU_DEVICE"
  --dreamerv3.jax.train_devices "$GPU_DEVICE"
  "$@"
)

MAX_RESTARTS="${MAX_RESTARTS:-20}"
restarts=0
carla_pid=""

log() {
  printf '[%(%F %T)T] %s\n' -1 "$*" | tee -a "$LOG_FILE"
}

kill_carla() {
  fuser -k "${CARLA_PORT}/tcp" >/dev/null 2>&1 || true

  if [[ -n "${carla_pid}" ]]; then
    kill -TERM "${carla_pid}" >/dev/null 2>&1 || true
    wait "${carla_pid}" >/dev/null 2>&1 || true
  fi
  carla_pid=""
}

cleanup() {
  log "Cleanup: stopping CARLA on port ${CARLA_PORT}"
  kill_carla
}
trap cleanup EXIT SIGINT SIGTERM

launch_carla() {
  if nc -z localhost "$CARLA_PORT" >/dev/null 2>&1; then
    return
  fi

  log "Starting CARLA on port ${CARLA_PORT} (GPU ${GPU_DEVICE})"
  kill_carla

  CUDA_VISIBLE_DEVICES="$GPU_DEVICE" "${CARLA_CMD[@]}" >>"$LOG_FILE" 2>&1 &
  carla_pid=$!

  until nc -z localhost "$CARLA_PORT" >/dev/null 2>&1; do
    sleep 1
  done
  log "CARLA is up (pid=${carla_pid})"
}

: > "$LOG_FILE"
log "Command: ${PY_CMD[*]}"

while true; do
  launch_carla

  log "Starting training (attempt $((restarts+1))/${MAX_RESTARTS})"
  set +e
  CUDA_VISIBLE_DEVICES="$GPU_DEVICE" "${PY_CMD[@]}" 2>&1 | tee -a "$LOG_FILE"
  code=${PIPESTATUS[0]}
  set -e

  if [[ $code -eq 0 ]]; then
    log "Training finished successfully (exit code 0). Exiting run."
    exit 0
  fi

  restarts=$((restarts + 1))
  log "Training exited with code=${code}. Restart ${restarts}/${MAX_RESTARTS}."

  if [[ $restarts -ge $MAX_RESTARTS ]]; then
    log "Max restarts reached. Exiting with code=${code}."
    exit "$code"
  fi
  kill_carla
  sleep 5
done
