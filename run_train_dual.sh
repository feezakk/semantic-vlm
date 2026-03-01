#!/bin/bash
# set -e
set -euo pipefail

GPU=0
TRAIN_PORT=3000
EVAL_PORT=4002   # keep +2 gap
TASK="carla_lane_following_vlm"

SEEDS=(0)

# Each entry: name | overrides
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
 --env.compute_vlm False \
 --env.include_domain_id False \
 --env.include_vlm False \
 --env.augment.enable False"

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
 --env.include_vlm False --env.augment.enable False"

"sem_rollout_only|--dreamerv3.use_semsty True \
 --dreamerv3.actor_inputs sem \
 --dreamerv3.critic_inputs sem \
 --dreamerv3.domain_grl_max 0.0 \
 --dreamerv3.loss_scales.domain_adv 0.0 \
 --dreamerv3.loss_scales.domain_sty 0.0 \
 --dreamerv3.loss_scales.domain_probe 0.0 \
 --dreamerv3.loss_scales.sem_rollout 1.0 \
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
 --env.include_domain_id True \
 --env.include_vlm True \
 --env.augment.enable False"

)

for SEED in "${SEEDS[@]}"; do
  for EXP in "${EXPS[@]}"; do
    NAME="${EXP%%|*}"
    OVERRIDES="${EXP#*|}"

    LOGDIR="logs/${TASK}/${NAME}/seed_${SEED}"
    mkdir -p "$LOGDIR"

    ./train_dm3_vlm_dual.sh "$TRAIN_PORT" "$EVAL_PORT" "$GPU" \
    --task "$TASK" \
    --dreamerv3.logdir "$LOGDIR" \
    --env.seed "$SEED" \
    --dreamerv3.seed "$SEED" \
    $OVERRIDES

  done
done