#!/usr/bin/env bash
set -euo pipefail

TRAIN_PORTS="${1:?Usage: $0 <train_ports> <train_towns> <eval_port> <eval_town> <gpu> [extra python flags...] }"
TRAIN_TOWNS="${2:?Usage: $0 <train_ports> <train_towns> <eval_port> <eval_town> <gpu> [extra python flags...] }"
EVAL_PORT="${3:?Usage: $0 <train_ports> <train_towns> <eval_port> <eval_town> <gpu> [extra python flags...] }"
EVAL_TOWN="${4:?Usage: $0 <train_ports> <train_towns> <eval_port> <eval_town> <gpu> [extra python flags...] }"
GPU="${5:?Usage: $0 <train_ports> <train_towns> <eval_port> <eval_town> <gpu> [extra python flags...] }"
shift 5

TRAIN_WEATHER_IDS="0"
EVAL_WEATHER_IDS="0"   # Town04 + training weathers ablation

TRAINING_SCRIPT="dreamerv3/train_vlm_town.py"

# Count towns -> dreamerv3.num_domains
IFS=',' read -r -a TOWNS_ARR <<< "$TRAIN_TOWNS"
NUM_DOMAINS="${#TOWNS_ARR[@]}"
if (( NUM_DOMAINS < 2 )); then
  echo "[ERR] Need at least 2 train towns, got: $TRAIN_TOWNS" >&2
  exit 2
fi

# Choose a log file. If caller passed --dreamerv3.logdir, write console there.
LOG_FILE="log_train_multitown.log"
for ((i=1; i<=$#; i++)); do
  arg="${!i}"
  if [[ "$arg" == "--dreamerv3.logdir" ]]; then
    j=$((i+1))
    LOGDIR="${!j}"
    LOG_FILE="${LOGDIR%/}/console.log"
    mkdir -p "$(dirname "$LOG_FILE")"
    break
  elif [[ "$arg" == --dreamerv3.logdir=* ]]; then
    LOGDIR="${arg#*=}"
    LOG_FILE="${LOGDIR%/}/console.log"
    mkdir -p "$(dirname "$LOG_FILE")"
    break
  fi
done

CARLA_CMD_BASE=(
  "$CARLA_ROOT/CarlaUE4.sh"
  -RenderOffScreen
  -benchmark
  -fps=10
)

log(){ printf '[%(%F %T)T] %s\n' -1 "$*" | tee -a "$LOG_FILE"; }
kill_port(){ fuser -k "${1}/tcp" >/dev/null 2>&1 || true; }

# Strong readiness: port open + CARLA RPC works (prevents fake-ready ports)
carla_ready(){
  python - <<PY >/dev/null 2>&1
import carla
c = carla.Client("localhost", int("$1"))
c.set_timeout(2.0)
w = c.get_world()
_ = w.get_map().name
PY
}

launch_carla(){
  local port="$1"
  local tag="$2"

  if nc -z localhost "$port" >/dev/null 2>&1; then
    if carla_ready "$port"; then
      log "CARLA already up and responsive on ${port} (${tag})"
      return
    else
      log "Port ${port} open but CARLA not responsive; restarting (${tag})"
      kill_port "$port"
    fi
  fi

  log "Starting CARLA on ${port} (${tag})"
  "${CARLA_CMD_BASE[@]}" "-carla-port=${port}" >>"$LOG_FILE" 2>&1 &

  until carla_ready "$port"; do sleep 1; done
  log "CARLA up on ${port} (${tag})"
}

: > "$LOG_FILE"

# Launch train servers
IFS=',' read -r -a PORTS_ARR <<< "$TRAIN_PORTS"
if (( ${#PORTS_ARR[@]} != ${#TOWNS_ARR[@]} )); then
  echo "[ERR] #ports != #towns: $TRAIN_PORTS vs $TRAIN_TOWNS" >&2
  exit 2
fi

for idx in "${!PORTS_ARR[@]}"; do
  p="${PORTS_ARR[$idx]}"
  t="${TOWNS_ARR[$idx]}"
  launch_carla "$p" "train_${t}"
done

# Launch eval server
launch_carla "$EVAL_PORT" "eval_${EVAL_TOWN}"

PY_CMD=(
  python -u "$TRAINING_SCRIPT"
  --carla_train_ports "$TRAIN_PORTS"
  --carla_train_towns "$TRAIN_TOWNS"
  --carla_eval_port "$EVAL_PORT"
  --carla_eval_town "$EVAL_TOWN"
  --dreamerv3.jax.policy_devices "$GPU"
  --dreamerv3.jax.train_devices "$GPU"

  # IMPORTANT: model domain head matches #towns
  --dreamerv3.num_domains "$NUM_DOMAINS"

  # Town-domain
  --env.domain_kind town
  --env.domain_towns "$TRAIN_TOWNS"

  --env.train_weather_ids "$TRAIN_WEATHER_IDS"
  --env.eval_weather_ids  "$EVAL_WEATHER_IDS"

  # CRITICAL: forward everything from the sweep script
  "$@"
)

log "Command: ${PY_CMD[*]}"
CUDA_VISIBLE_DEVICES="$GPU" "${PY_CMD[@]}" 2>&1 | tee -a "$LOG_FILE"
