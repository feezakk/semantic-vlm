#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=1

GPU=0
PORT=20000
TASK="carla_lane_following_vlm_town"

# IMPORTANT:
# Your eval.py (the one you pasted) expects the *task name it should create*.
# In your config you have: carla_lane_following_vlm_eval
EVAL_TASK="${TASK}_eval"

SEEDS=(4)


# Copy the EXACT same EXPS array you used for training (so model config matches checkpoint)
EXPS=(
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
 --env.world.town Town04 \
 --env.augment.enable False"

"semsty_noloss|--dreamerv3.use_semsty True \
 --dreamerv3.actor_inputs deter,stoch \
 --dreamerv3.critic_inputs deter,stoch \
 --dreamerv3.loss_scales.domain_adv 0.0 \
 --dreamerv3.loss_scales.domain_sty 0.0 \
 --dreamerv3.loss_scales.domain_probe 0.0 \
 --dreamerv3.loss_scales.sem_rollout 0.0 \
 --dreamerv3.run.eval_eps 64 \
 --env.include_domain_id False \
 --env.include_vlm False \
 --env.compute_vlm False \
 --env.world.town Town04 \
 --env.augment.enable False"

"sem_rollout_aux|--dreamerv3.use_semsty True \
 --dreamerv3.actor_inputs deter,stoch \
 --dreamerv3.critic_inputs deter,stoch \
 --dreamerv3.domain_grl_max 0.0 \
 --dreamerv3.domain_grl_schedule none \
 --dreamerv3.loss_scales.domain_adv 0.0 \
 --dreamerv3.loss_scales.domain_sty 0.0 \
 --dreamerv3.loss_scales.domain_probe 0.0 \
 --dreamerv3.loss_scales.sem_rollout 0.3 \
 --dreamerv3.run.eval_eps 64 \
 --dreamerv3.domain_scale 0.0 \
 --env.include_domain_id False \
 --env.include_vlm True \
 --env.compute_vlm True \
 --env.world.town Town04 \
 --env.augment.enable False"

"policy_sem_noloss|--dreamerv3.use_semsty True \
 --dreamerv3.actor_inputs sem \
 --dreamerv3.critic_inputs sem \
 --dreamerv3.loss_scales.sem_rollout 0.0 \
 --dreamerv3.loss_scales.domain_adv 0.0 \
 --dreamerv3.loss_scales.domain_sty 0.0 \
 --dreamerv3.loss_scales.domain_probe 0.0 \
 --dreamerv3.run.eval_eps 64  \
 --dreamerv3.domain_grl_max 0.0 \
 --dreamerv3.domain_scale 0.0 \
 --env.compute_vlm False \
 --env.include_vlm False \
 --env.include_domain_id False \
 --env.world.town Town04 \
 --env.augment.enable False"

"domain_adv_small_grl|--dreamerv3.use_semsty True \
 --dreamerv3.actor_inputs sem \
 --dreamerv3.critic_inputs sem \
 --dreamerv3.domain_head_input sem \
 --dreamerv3.domain_grl_max 0.1 \
 --dreamerv3.domain_grl_schedule dann \
 --dreamerv3.loss_scales.domain_adv 1.0 \
 --dreamerv3.loss_scales.domain_sty 1.0 \
 --dreamerv3.loss_scales.sem_rollout 1.0 \
 --dreamerv3.loss_scales.domain_probe 1.0 \
 --dreamerv3.run.eval_eps 64 \
 --env.include_domain_id True \
 --env.include_vlm True \
 --env.world.town Town04 \
 --env.augment.enable False"

"domain_adv_large_grl|--dreamerv3.use_semsty True \
 --dreamerv3.actor_inputs sem \
 --dreamerv3.critic_inputs sem \
 --dreamerv3.domain_head_input sem \
 --dreamerv3.domain_grl_max 1.0 \
 --dreamerv3.domain_grl_schedule dann \
 --dreamerv3.loss_scales.domain_adv 1.0 \
 --dreamerv3.loss_scales.domain_sty 1.0 \
 --dreamerv3.loss_scales.sem_rollout 1.0 \
 --dreamerv3.loss_scales.domain_probe 1.0 \
 --dreamerv3.run.eval_eps 64 \
 --env.include_domain_id True \
 --env.include_vlm True \
 --env.world.town Town04 \
 --env.augment.enable False"

# "sem_plus_adv_aux|--dreamerv3.use_semsty True \
#  --dreamerv3.actor_inputs deter,stoch \
#  --dreamerv3.critic_inputs deter,stoch \
#  --dreamerv3.domain_head_input sem \
#  --dreamerv3.domain_grl_max 1.0 \
#  --dreamerv3.domain_grl_schedule dann \
#  --dreamerv3.loss_scales.domain_adv 1.0 \
#  --dreamerv3.loss_scales.domain_sty 0.0 \
#  --dreamerv3.loss_scales.sem_rollout 0.3 \
#  --dreamerv3.loss_scales.domain_probe 1.0 \
#  --dreamerv3.run.eval_eps 64 \
#  --dreamerv3.num_domains 2 \
#  --env.include_domain_id True \
#  --env.domain_kind town \
#  --env.domain_towns Town05,Town06 \
#  --env.include_vlm True \
#  --env.compute_vlm True \
#  --env.world.town Town04 \
#  --env.augment.enable False"

# "dreamer_aug|--dreamerv3.use_semsty False \
#  --dreamerv3.actor_inputs deter,stoch \
#  --dreamerv3.critic_inputs deter,stoch \
#  --dreamerv3.domain_head_input deter \
#  --env.augment.enable True \
#  --env.augment.brightness 0.2 \
#  --env.augment.contrast 0.2 \
#  --env.augment.gamma 0.2 \
#  --env.augment.blur_prob 0.2 \
#  --env.augment.noise_std 0.01 \
#  --dreamerv3.domain_grl_max 0.0 \
#  --dreamerv3.loss_scales.domain_adv 0.0 \
#  --dreamerv3.loss_scales.domain_sty 0.0 \
#  --dreamerv3.loss_scales.sem_rollout 0.0 \
#  --dreamerv3.loss_scales.domain_probe 0.0 \
#  --dreamerv3.run.eval_eps 64 \
#  --env.compute_vlm False \
#  --env.include_domain_id False \
#  --env.world.town Town04 \
#  --env.include_vlm False"

# "dreamer_dann|--dreamerv3.use_semsty False \
#  --dreamerv3.actor_inputs deter,stoch \
#  --dreamerv3.critic_inputs deter,stoch \
#  --dreamerv3.domain_head_input deter \
#  --dreamerv3.domain_grl_max 1.0 \
#  --dreamerv3.domain_grl_schedule dann \
#  --dreamerv3.loss_scales.domain_adv 1.0 \
#  --dreamerv3.loss_scales.domain_sty 0.0 \
#  --dreamerv3.loss_scales.sem_rollout 0.0 \
#  --dreamerv3.loss_scales.domain_probe 1.0 \
#  --dreamerv3.run.eval_eps 64 \
#  --dreamerv3.num_domains 2 \
#  --env.compute_vlm False \
#  --env.include_domain_id True \
#  --env.include_vlm False \
#  --env.domain_kind town \
#  --env.domain_towns Town05,Town06 \
#  --env.world.town Town04 \
#  --env.augment.enable False"

"ours_full_aux|--dreamerv3.use_semsty True \
 --dreamerv3.actor_inputs deter,stoch \
 --dreamerv3.critic_inputs deter,stoch \
 --dreamerv3.domain_head_input sem \
 --dreamerv3.domain_grl_max 1.0 \
 --dreamerv3.domain_grl_schedule dann \
 --dreamerv3.loss_scales.domain_adv 1.0 \
 --dreamerv3.loss_scales.domain_sty 1.0 \
 --dreamerv3.loss_scales.sem_rollout 0.3 \
 --dreamerv3.loss_scales.domain_probe 1.0 \
 --dreamerv3.run.eval_eps 64 \
 --dreamerv3.num_domains 2 \
 --env.include_domain_id True \
 --env.domain_kind town \
 --env.domain_towns Town05,Town06 \
 --env.include_vlm True \
 --env.compute_vlm True \
 --env.world.town Town04 \
 --env.augment.enable False"
)

EXP_IDX=0
for EXP in "${EXPS[@]}"; do
  NAME="${EXP%%|*}"
  OVERRIDES="${EXP#*|}"

  for SEED in "${SEEDS[@]}"; do
    PORT=$((PORT + EXP_IDX*10 + SEED))

    TRAIN_LOGDIR="/data/feeza/logs/${TASK}/${NAME}/seed_0"
    CKPT="${TRAIN_LOGDIR}/best.ckpt"
    if [[ ! -f "$CKPT" ]]; then
      echo "[SKIP] Missing checkpoint: $CKPT"
      continue
    fi

    EVAL_LOGDIR="/data/feeza/logs_eval_town04/${TASK}/${NAME}/seed_${SEED}"
    mkdir -p "$EVAL_LOGDIR"
    LOG_FILE="${EVAL_LOGDIR}/eval.log"

    echo "[RUN] baseline=${NAME} seed=${SEED} port=${PORT}"
    if ! LOG_FILE="$LOG_FILE" ./eval_dm3_vlm.sh "$PORT" "$GPU" "$CKPT" \
        --task "$EVAL_TASK" \
        $OVERRIDES \
        --dreamerv3.logdir "$EVAL_LOGDIR" \
        --env.seed "$SEED" \
        --dreamerv3.seed "$SEED" \
        --dreamerv3.run.eval_eps 64 \
        2>&1 | tee -a "$LOG_FILE"
    then
      echo "[FAIL] baseline=${NAME} seed=${SEED} (see $LOG_FILE)"
      continue
    fi
  done

  EXP_IDX=$((EXP_IDX + 1))
done
