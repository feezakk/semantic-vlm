#!/usr/bin/env bash
set -euo pipefail

TRAIN_PORTS="${1:?Usage: $0 <train_ports> <train_towns> <eval_port> <eval_town> <gpu> [extra python flags...] }"
TRAIN_TOWNS="${2:?Usage: $0 <train_ports> <train_towns> <eval_port> <eval_town> <gpu> [extra python flags...] }"
EVAL_PORT="${3:?Usage: $0 <train_ports> <train_towns> <eval_port> <eval_town> <gpu> [extra python flags...] }"
EVAL_TOWN="${4:?Usage: $0 <train_ports> <train_towns> <eval_port> <eval_town> <gpu> [extra python flags...] }"
GPU="${5:?Usage: $0 <train_ports> <train_towns> <eval_port> <eval_town> <gpu> [extra python flags...] }"
shift 5

TRAIN_WEATHER_IDS="0, 1, 2, 4"
EVAL_WEATHER_IDS="6, 7, 8, 10, 13"   
USER_SET_NUM_DOMAINS=0

TRAINING_SCRIPT="dreamerv3/train_vlm_weather.py"

# Count TRAIN weather IDs -> default dreamerv3.num_domains
count_csv_items(){
  local csv="$1"
  local IFS=','; read -r -a a <<< "$csv"
  local n=0
  for x in "${a[@]}"; do
    x="${x//[[:space:]]/}"
    [[ -n "$x" ]] && n=$((n+1))
  done
  echo "$n"
}

NUM_DOMAINS="$(count_csv_items "$TRAIN_WEATHER_IDS")"
if (( NUM_DOMAINS < 1 )); then
  echo "[ERR] Need at least 1 train weather id, got: '$TRAIN_WEATHER_IDS'" >&2
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
    elif [[ "$arg" == "--env.train_weather_ids" ]]; then
    j=$((i+1))
    TRAIN_WEATHER_IDS="${!j}"
  elif [[ "$arg" == --env.train_weather_ids=* ]]; then
    TRAIN_WEATHER_IDS="${arg#*=}"
  elif [[ "$arg" == "--env.eval_weather_ids" ]]; then
    j=$((i+1))
    EVAL_WEATHER_IDS="${!j}"
  elif [[ "$arg" == --env.eval_weather_ids=* ]]; then
    EVAL_WEATHER_IDS="${arg#*=}"
  elif [[ "$arg" == "--dreamerv3.num_domains" ]]; then
    USER_SET_NUM_DOMAINS=1
  elif [[ "$arg" == --dreamerv3.num_domains=* ]]; then
    USER_SET_NUM_DOMAINS=1
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

# --- One-port mode (strict): exactly one port + one town/map ---
IFS=',' read -r -a PORTS_ARR <<< "$TRAIN_PORTS"
IFS=',' read -r -a TOWNS_ARR <<< "$TRAIN_TOWNS"

if (( ${#PORTS_ARR[@]} != 1 || ${#TOWNS_ARR[@]} != 1 )); then
  echo "[ERR] Weather+one-port expects exactly one train port and one train town." >&2
  echo "      Got ports='$TRAIN_PORTS' towns='$TRAIN_TOWNS'" >&2
  exit 2
fi

BASE_PORT="${PORTS_ARR[0]//[[:space:]]/}"
BASE_TOWN="${TOWNS_ARR[0]//[[:space:]]/}"

# Force eval to reuse the same CARLA server
EVAL_PORT="$BASE_PORT"
EVAL_TOWN="$BASE_TOWN"

launch_carla "$BASE_PORT" "shared_${BASE_TOWN}"
log "Eval will reuse training CARLA on port ${EVAL_PORT} (town: ${EVAL_TOWN})."

# export CUDA_VISIBLE_DEVICES=1
PY_CMD=(
  python -u "$TRAINING_SCRIPT"
  --carla_train_ports "$TRAIN_PORTS"
  --carla_train_towns "$TRAIN_TOWNS"
  --carla_eval_port "$EVAL_PORT"
  --carla_eval_town "$EVAL_TOWN"
  --dreamerv3.jax.policy_devices "$GPU"
  --dreamerv3.jax.train_devices "$GPU"

  # IMPORTANT: model domain head matches #towns
  # --dreamerv3.num_domains "$NUM_DOMAINS"

  --env.train_weather_ids "$TRAIN_WEATHER_IDS"
  --env.eval_weather_ids  "$EVAL_WEATHER_IDS"

  # CRITICAL: forward everything from the sweep script
  "$@"
)

if (( USER_SET_NUM_DOMAINS == 0 )); then
  PY_CMD+=( --dreamerv3.num_domains "$NUM_DOMAINS" )
fi

log "Command: ${PY_CMD[*]}"
# CUDA_VISIBLE_DEVICES="$GPU" "${PY_CMD[@]}" 2>&1 | tee -a "$LOG_FILE"

"${PY_CMD[@]}" 2>&1 | tee -a "$LOG_FILE"