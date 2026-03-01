#!/usr/bin/env bash
set -euo pipefail

TASK="carla_lane_following_vlm"

# Two GPUs
GPU_IDS=(0 1)
# One CARLA port per GPU (keep fixed to avoid collisions)
GPU_PORTS=(3000 3100)

SEEDS=(0 1 2)

# -----------------------------
# EXPERIMENT DEFINITIONS
# Each entry: "name|overrides..."
# -----------------------------
EXPS=(
"ours|--dreamerv3.use_semsty True \
 --dreamerv3.actor_inputs sem \
 --dreamerv3.critic_inputs sem \
 --dreamerv3.domain_head_input sem \
 --dreamerv3.domain_grl_max 1.0 \
 --dreamerv3.domain_grl_schedule dann \
 --dreamerv3.loss_scales.domain_adv 1.0 \
 --dreamerv3.loss_scales.domain_sty 1.0 \
 --dreamerv3.loss_scales.sem_rollout 1.0 \
 --env.include_domain_id True \
 --env.include_vlm True"

# NOTE:
# If your WorldModel.loss() still unconditionally reads data["domain_id"],
# then env.include_domain_id MUST be True even for std/aug, otherwise it crashes.
"dreamer_std|--dreamerv3.use_semsty False \
 --dreamerv3.actor_inputs deter,stoch \
 --dreamerv3.critic_inputs deter,stoch \
 --dreamerv3.domain_head_input deter \
 --dreamerv3.domain_grl_max 0.0 \
 --dreamerv3.loss_scales.domain_adv 0.0 \
 --dreamerv3.loss_scales.domain_sty 0.0 \
 --dreamerv3.loss_scales.sem_rollout 0.0 \
 --dreamerv3.loss_scales.domain_probe 0.0 \
 --env.compute_vlm False \
 --env.include_domain_id False \
 --env.include_vlm False"

"dreamer_aug|--dreamerv3.use_semsty False \
 --dreamerv3.actor_inputs deter,stoch \
 --dreamerv3.critic_inputs deter,stoch \
 --dreamerv3.domain_head_input deter \
 --env.augment.enable True \
 --env.augment.brightness 0.2 \
 --env.augment.contrast 0.2 \
 --env.augment.gamma 0.2 \
 --env.augment.blur_prob 0.2 \
 --env.augment.noise_std 0.01 \
 --dreamerv3.domain_grl_max 0.0 \
 --dreamerv3.loss_scales.domain_adv 0.0 \
 --dreamerv3.loss_scales.domain_sty 0.0 \
 --dreamerv3.loss_scales.sem_rollout 0.0 \
 --dreamerv3.loss_scales.domain_probe 0.0 \
 --env.compute_vlm False \
 --env.include_domain_id False \
 --env.include_vlm False"

"dreamer_dann|--dreamerv3.use_semsty False \
 --dreamerv3.actor_inputs deter,stoch \
 --dreamerv3.critic_inputs deter,stoch \
 --dreamerv3.domain_head_input deter \
 --dreamerv3.domain_grl_max 1.0 \
 --dreamerv3.domain_grl_schedule dann \
 --dreamerv3.loss_scales.domain_adv 1.0 \
 --dreamerv3.loss_scales.domain_sty 0.0 \
 --dreamerv3.loss_scales.sem_rollout 0.0 \
 --env.compute_vlm False \
 --env.include_domain_id True \
 --env.include_vlm False"

"sem_rollout_only|--dreamerv3.use_semsty True \
 --dreamerv3.actor_inputs sem \
 --dreamerv3.critic_inputs sem \
 --dreamerv3.domain_grl_max 0.0 \
 --dreamerv3.loss_scales.domain_adv 0.0 \
 --dreamerv3.loss_scales.domain_sty 0.0 \
 --dreamerv3.loss_scales.domain_probe 0.0 \
 --dreamerv3.loss_scales.sem_rollout 1.0 \
 --env.include_domain_id False \
 --env.include_vlm True"

"sem_plus_adv|--dreamerv3.use_semsty True \
 --dreamerv3.actor_inputs sem \
 --dreamerv3.critic_inputs sem \
 --dreamerv3.domain_head_input sem \
 --dreamerv3.domain_grl_max 1.0 \
 --dreamerv3.domain_grl_schedule dann \
 --dreamerv3.loss_scales.domain_adv 1.0 \
 --dreamerv3.loss_scales.domain_sty 0.0 \
 --dreamerv3.loss_scales.sem_rollout 1.0 \
 --env.include_domain_id True \
 --env.include_vlm True"
)

# -----------------------------
# INTERNAL SCHEDULER
# -----------------------------
FREE_SLOTS=(0 1)         # slot 0 -> GPU_IDS[0], slot 1 -> GPU_IDS[1]
RUNNING_PIDS=()
RUNNING_SLOTS=()

start_job() {
  local slot="$1"
  local seed="$2"
  local name="$3"
  local overrides="$4"

  local gpu="${GPU_IDS[$slot]}"
  local port="${GPU_PORTS[$slot]}"
  local logdir="/data/feeza/vlm/logs/${TASK}/${name}/seed_${seed}"
  mkdir -p "${logdir}"

  echo "[LAUNCH] seed=${seed} name=${name} gpu=${gpu} port=${port} logdir=${logdir}"

  # Run wrapper in background (it blocks until training finishes)
  ./train_dm3_vlm2.sh "${port}" "${gpu}" \
    --task "${TASK}" \
    --dreamerv3.logdir "${logdir}" \
    --env.seed "${seed}" \
    --dreamerv3.seed "${seed}" \
    ${overrides} \
    &

  local pid=$!
  RUNNING_PIDS+=("${pid}")
  RUNNING_SLOTS+=("${slot}")
  echo "[LAUNCH] pid=${pid} (slot=${slot})"
}

reap_finished() {
  local new_pids=()
  local new_slots=()
  for i in "${!RUNNING_PIDS[@]}"; do
    local pid="${RUNNING_PIDS[$i]}"
    local slot="${RUNNING_SLOTS[$i]}"
    if kill -0 "${pid}" >/dev/null 2>&1; then
      new_pids+=("${pid}")
      new_slots+=("${slot}")
    else
      # finished -> free slot
      FREE_SLOTS+=("${slot}")
      echo "[DONE] pid=${pid} freed slot=${slot}"
    fi
  done
  RUNNING_PIDS=("${new_pids[@]}")
  RUNNING_SLOTS=("${new_slots[@]}")
}

wait_for_any() {
  # Wait until at least one running pid finishes
  if [[ ${#RUNNING_PIDS[@]} -eq 0 ]]; then
    return 0
  fi
  set +e
  wait -n
  set -e
  reap_finished
}

# Build a job list: all (seed, exp)
JOBS=()
for seed in "${SEEDS[@]}"; do
  for exp in "${EXPS[@]}"; do
    JOBS+=("${seed}|${exp}")
  done
done

# Main loop: schedule jobs with concurrency=2
for job in "${JOBS[@]}"; do
  # wait until a slot is free
  while [[ ${#FREE_SLOTS[@]} -eq 0 ]]; do
    wait_for_any
  done

  slot="${FREE_SLOTS[0]}"
  FREE_SLOTS=("${FREE_SLOTS[@]:1}")

  seed="${job%%|*}"
  rest="${job#*|}"
  name="${rest%%|*}"
  overrides="${rest#*|}"

  start_job "${slot}" "${seed}" "${name}" "${overrides}"
done

# Wait for remaining jobs
echo "[INFO] All jobs launched. Waiting for remaining ${#RUNNING_PIDS[@]} jobs..."
for pid in "${RUNNING_PIDS[@]}"; do
  wait "${pid}"
done

echo "[INFO] All experiments completed."
