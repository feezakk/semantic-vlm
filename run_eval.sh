#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=1

GPU=0
PORT=30000
TASK="carla_lane_following_vlm"

# IMPORTANT:
# Your eval.py (the one you pasted) expects the *task name it should create*.
# In your config you have: carla_lane_following_vlm_eval
EVAL_TASK="${TASK}_eval"

SEEDS=(0 1 2)

# Copy the EXACT same EXPS array you used for training (so model config matches checkpoint)
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
 --dreamerv3.run.eval_eps 64 \
 --env.include_domain_id True \
 --env.include_vlm True \
 --env.augment.enable False"

"dreamer_std|--dreamerv3.use_semsty False \
 --dreamerv3.actor_inputs deter,stoch \
 --dreamerv3.critic_inputs deter,stoch \
 --dreamerv3.domain_head_input deter \
 --dreamerv3.domain_grl_max 0.0 \
 --dreamerv3.loss_scales.domain_adv 0.0 \
 --dreamerv3.loss_scales.domain_sty 0.0 \
 --dreamerv3.loss_scales.sem_rollout 0.0 \
 --dreamerv3.loss_scales.domain_probe 0.0 \
 --dreamerv3.run.eval_eps 64 \
 --env.compute_vlm False \
 --env.include_domain_id False \
 --env.include_vlm False \
 --env.augment.enable False"

"dreamer_aug|--dreamerv3.use_semsty False \
 --dreamerv3.actor_inputs deter,stoch \
 --dreamerv3.critic_inputs deter,stoch \
 --dreamerv3.domain_head_input deter \
 --dreamerv3.run.eval_eps 64 \
 --dreamerv3.domain_grl_max 0.0 \
 --dreamerv3.loss_scales.domain_adv 0.0 \
 --dreamerv3.loss_scales.domain_sty 0.0 \
 --dreamerv3.loss_scales.sem_rollout 0.0 \
 --dreamerv3.loss_scales.domain_probe 0.0 \
 --env.compute_vlm False \
 --env.include_domain_id False \
 --env.include_vlm False
 --env.augment.enable False"

"dreamer_dann|--dreamerv3.use_semsty False \
 --dreamerv3.actor_inputs deter,stoch \
 --dreamerv3.critic_inputs deter,stoch \
 --dreamerv3.domain_head_input deter \
 --dreamerv3.run.eval_eps 64 \
 --dreamerv3.domain_grl_max 1.0 \
 --dreamerv3.domain_grl_schedule dann \
 --dreamerv3.loss_scales.domain_adv 1.0 \
 --dreamerv3.loss_scales.domain_sty 0.0 \
 --dreamerv3.loss_scales.sem_rollout 0.0 \
 --env.compute_vlm False \
 --env.include_domain_id True \
 --env.include_vlm False \
 --env.augment.enable False"

"sem_rollout_only|--dreamerv3.use_semsty True \
 --dreamerv3.actor_inputs sem \
 --dreamerv3.critic_inputs sem \
 --dreamerv3.domain_grl_max 0.0 \
 --dreamerv3.loss_scales.domain_adv 0.0 \
 --dreamerv3.loss_scales.domain_sty 0.0 \
 --dreamerv3.loss_scales.domain_probe 0.0 \
 --dreamerv3.loss_scales.sem_rollout 1.0 \
 --dreamerv3.run.eval_eps 64 \
 --env.include_domain_id False \
 --env.include_vlm True \
 --env.augment.enable False"

"sem_plus_adv|--dreamerv3.use_semsty True \
 --dreamerv3.actor_inputs sem \
 --dreamerv3.critic_inputs sem \
 --dreamerv3.domain_head_input sem \
 --dreamerv3.domain_grl_max 1.0 \
 --dreamerv3.domain_grl_schedule dann \
 --dreamerv3.loss_scales.domain_adv 1.0 \
 --dreamerv3.loss_scales.domain_sty 0.0 \
 --dreamerv3.loss_scales.sem_rollout 1.0 \
 --dreamerv3.run.eval_eps 64 \
 --env.include_domain_id True \
 --env.include_vlm True \
 --env.augment.enable False"
)

# Optional: ensure eval duration is long enough.
# Your deterministic eval cycles route×weather combos. A safe default is "large enough steps".
# Adjust this if you prefer shorter/faster.
EVAL_STEPS=40000

for SEED in "${SEEDS[@]}"; do
  for EXP in "${EXPS[@]}"; do
    NAME="${EXP%%|*}"
    OVERRIDES="${EXP#*|}"

    TRAIN_LOGDIR="logs/${TASK}/${NAME}/seed_${SEED}"
    CKPT="${TRAIN_LOGDIR}/checkpoint.ckpt"

    if [[ ! -f "$CKPT" ]]; then
      echo "[SKIP] Missing checkpoint: $CKPT"
      continue
    fi

    # Put eval outputs in a clean separate directory (avoid mixing with training TB/events)
    EVAL_LOGDIR="logs_eval/${TASK}/${NAME}/seed_${SEED}"
    mkdir -p "$EVAL_LOGDIR"

    # Per-baseline log file (so runs don't overwrite each other)
    LOG_FILE="${EVAL_LOGDIR}/eval_port${PORT}.log"

    LOG_FILE="$LOG_FILE" ./eval_dm3_vlm.sh "$PORT" "$GPU" "$CKPT" \
      --task "$EVAL_TASK" \
      --dreamerv3.logdir "$EVAL_LOGDIR" \
      --env.seed "$SEED" \
      --dreamerv3.seed "$SEED" \
      --dreamerv3.run.steps "$EVAL_STEPS" \
      $OVERRIDES
  done
done
