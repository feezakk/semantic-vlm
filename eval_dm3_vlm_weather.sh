#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <carla_port> <gpu_device> <checkpoint_path> [additional_eval_parameters]"
  exit 1
fi

CARLA_PORT="$1"
GPU_DEVICE="$2"
CHECKPOINT_PATH="$3"
shift 3

EVAL_SCRIPT="dreamerv3/eval_vlm_town.py"

# Allow caller to override where the log goes (so you can log per-baseline)
LOG_FILE="${LOG_FILE:-eval_${CARLA_PORT}.log}"

CARLA_CMD=(
  "$CARLA_ROOT/CarlaUE4.sh"
  -RenderOffScreen
  "-carla-port=${CARLA_PORT}"
  -benchmark
  -fps=10
)

PY_CMD=(
  python -u "$EVAL_SCRIPT"
  --env.world.carla_port "$CARLA_PORT"
  --dreamerv3.jax.policy_devices "$GPU_DEVICE"
  --dreamerv3.jax.train_devices "$GPU_DEVICE"
  --dreamerv3.run.from_checkpoint "$CHECKPOINT_PATH"
  "$@"
)

MAX_RESTARTS="${MAX_RESTARTS:-5}"
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
log "Eval command: ${PY_CMD[*]}"

while true; do
  launch_carla

  log "Running eval (attempt $((restarts+1))/${MAX_RESTARTS})"
  set +e
  CUDA_VISIBLE_DEVICES="$GPU_DEVICE" "${PY_CMD[@]}" 2>&1 | tee -a "$LOG_FILE"
  code=${PIPESTATUS[0]}
  set -e

  if [[ $code -eq 0 ]]; then
    log "Eval finished successfully (exit code 0)."
    exit 0
  fi

  restarts=$((restarts + 1))
  log "Eval exited with code=${code}. Restart ${restarts}/${MAX_RESTARTS}."

  if [[ $restarts -ge $MAX_RESTARTS ]]; then
    log "Max restarts reached. Exiting with code=${code}."
    exit "$code"
  fi

  # Restart CARLA too (often the actual culprit)
  kill_carla
  sleep 3
done
