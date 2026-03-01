#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <carla_train_port> <carla_eval_port> <gpu_device> [additional_training_parameters]"
  exit 1
fi

TRAIN_PORT="$1"
EVAL_PORT="$2"
GPU_DEVICE="$3"
shift 3

if [[ "$TRAIN_PORT" -eq "$EVAL_PORT" ]]; then
  echo "ERROR: TRAIN_PORT and EVAL_PORT must be different."
  exit 1
fi

# Strongly recommended: keep a gap >=2 to avoid CARLA streaming-port collisions.
if (( ${TRAIN_PORT} + 1 == ${EVAL_PORT} )) || (( ${EVAL_PORT} + 1 == ${TRAIN_PORT} )); then
  echo "ERROR: Ports are adjacent ($TRAIN_PORT, $EVAL_PORT). Use a gap >=2 (e.g., 3000 and 3002)."
  exit 1
fi

LOG_FILE="log_train${TRAIN_PORT}_eval${EVAL_PORT}.log"
TRAINING_SCRIPT="dreamerv3/train_vlm.py"

CARLA_CMD_TRAIN=(
  "$CARLA_ROOT/CarlaUE4.sh"
  -RenderOffScreen
  "-carla-port=${TRAIN_PORT}"
  -benchmark
  -fps=10
)

CARLA_CMD_EVAL=(
  "$CARLA_ROOT/CarlaUE4.sh"
  -RenderOffScreen
  "-carla-port=${EVAL_PORT}"
  -benchmark
  -fps=10
)

# IMPORTANT:
#  - Do NOT pass --env.world.carla_port here anymore (it would override both envs to the same port).
#  - Instead pass two ports that train_vlm.py will use when constructing train/eval envs.
PY_CMD=(
  python -u "$TRAINING_SCRIPT"
  --carla_train_port "$TRAIN_PORT"
  --carla_eval_port "$EVAL_PORT"
  --dreamerv3.jax.policy_devices "$GPU_DEVICE"
  --dreamerv3.jax.train_devices "$GPU_DEVICE"
  "$@"
)

MAX_RESTARTS="${MAX_RESTARTS:-20}"
restarts=0
carla_train_pid=""
carla_eval_pid=""

log() {
  printf '[%(%F %T)T] %s\n' -1 "$*" | tee -a "$LOG_FILE"
}

kill_carla_port() {
  local port="$1"
  # Kill whatever is bound to the CARLA port (kills the CARLA process which also frees streaming port).
  fuser -k "${port}/tcp" >/dev/null 2>&1 || true
}

kill_train() {
  kill_carla_port "$TRAIN_PORT"
  if [[ -n "${carla_train_pid}" ]]; then
    kill -TERM "${carla_train_pid}" >/dev/null 2>&1 || true
    wait "${carla_train_pid}" >/dev/null 2>&1 || true
  fi
  carla_train_pid=""
}

kill_eval() {
  kill_carla_port "$EVAL_PORT"
  if [[ -n "${carla_eval_pid}" ]]; then
    kill -TERM "${carla_eval_pid}" >/dev/null 2>&1 || true
    wait "${carla_eval_pid}" >/dev/null 2>&1 || true
  fi
  carla_eval_pid=""
}

cleanup() {
  log "Cleanup: stopping CARLA on ports train=${TRAIN_PORT}, eval=${EVAL_PORT}"
  kill_train
  kill_eval
}
trap cleanup EXIT SIGINT SIGTERM

launch_carla() {
  local port="$1"
  local kind="$2"   # "train" or "eval"

  if nc -z localhost "$port" >/dev/null 2>&1; then
    log "CARLA already up on port ${port} (${kind})"
    return
  fi

  log "Starting CARLA (${kind}) on port ${port} (GPU ${GPU_DEVICE})"
  kill_carla_port "$port"

  if [[ "$kind" == "train" ]]; then
    CUDA_VISIBLE_DEVICES="$GPU_DEVICE" "${CARLA_CMD_TRAIN[@]}" >>"$LOG_FILE" 2>&1 &
    carla_train_pid=$!
  else
    CUDA_VISIBLE_DEVICES="$GPU_DEVICE" "${CARLA_CMD_EVAL[@]}" >>"$LOG_FILE" 2>&1 &
    carla_eval_pid=$!
  fi

  until nc -z localhost "$port" >/dev/null 2>&1; do
    sleep 1
  done
  log "CARLA up on port ${port} (${kind})"
}

: > "$LOG_FILE"
log "Command: ${PY_CMD[*]}"

while true; do
  launch_carla "$TRAIN_PORT" "train"
  launch_carla "$EVAL_PORT" "eval"

  log "Starting training (attempt $((restarts+1))/${MAX_RESTARTS})"
  set +e
  CUDA_VISIBLE_DEVICES="$GPU_DEVICE" "${PY_CMD[@]}" 2>&1 | tee -a "$LOG_FILE"
  code=${PIPESTATUS[0]}
  set -e

  if [[ $code -eq 0 ]]; then
    log "Training finished successfully (exit code 0). Exiting."
    exit 0
  fi

  restarts=$((restarts + 1))
  log "Training exited with code=${code}. Restart ${restarts}/${MAX_RESTARTS}."

  if [[ $restarts -ge $MAX_RESTARTS ]]; then
    log "Max restarts reached. Exiting with code=${code}."
    exit "$code"
  fi

  # Restart both CARLA servers in case failure was server-related.
  kill_train
  kill_eval
  sleep 5
done
