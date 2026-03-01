#!/bin/bash
set -euo pipefail
#!/bin/bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=1
GPU=0
TASK="carla_lane_following_vlm_weather"

TRAIN_PORTS="3000"
TRAIN_WEATHERS="carla.WeatherParameters.ClearNoon, carla.WeatherParameters.CloudyNoon, carla.WeatherParameters.WetNoon, carla.WeatherParameters.SoftRainNoon"

EVAL_PORT=3000
EVAL_WEATHER="carla.WeatherParameters.HardRainNoon, carla.WeatherParameters.ClearSunset, carla.WeatherParameters.CloudySunset, carla.WeatherParameters.WetCloudySunset, carla.WeatherParameters.HardRainSunset"
TRAIN_WEATHER_IDS="0,1,2,4"
EVAL_WEATHER_IDS="6,7,8,10,13"
USER_SET_NUM_DOMAINS=0

SEEDS=(0)

EXPS=(
# "dreamer_std|--dreamerv3.use_semsty False \
#  --dreamerv3.actor_inputs deter,stoch \
#  --dreamerv3.critic_inputs deter,stoch \
#  --dreamerv3.domain_head_input deter \
#  --dreamerv3.domain_grl_max 0.0 \
#  --dreamerv3.loss_scales.domain_adv 0.0 \
#  --dreamerv3.loss_scales.domain_sty 0.0 \
#  --dreamerv3.loss_scales.sem_rollout 0.0 \
#  --dreamerv3.loss_scales.domain_probe 0.0 \
#  --env.compute_vlm False \
#  --env.include_domain_id False \
#  --env.include_vlm False \
#  --env.augment.enable False"

# "semsty_noloss|--dreamerv3.use_semsty True \
#  --dreamerv3.actor_inputs deter,stoch \
#  --dreamerv3.critic_inputs deter,stoch \
#  --dreamerv3.loss_scales.domain_adv 0.0 \
#  --dreamerv3.loss_scales.domain_sty 0.0 \
#  --dreamerv3.loss_scales.domain_probe 0.0 \
#  --dreamerv3.loss_scales.sem_rollout 0.0 \
#  --env.include_domain_id False \
#  --env.include_vlm False \
#  --env.compute_vlm False \
#  --env.augment.enable False"

# "sem_rollout_aux|--dreamerv3.use_semsty True \
#  --dreamerv3.actor_inputs deter,stoch \
#  --dreamerv3.critic_inputs deter,stoch \
#  --dreamerv3.domain_grl_max 0.0 \
#  --dreamerv3.domain_grl_schedule none \
#  --dreamerv3.loss_scales.domain_adv 0.0 \
#  --dreamerv3.loss_scales.domain_sty 0.0 \
#  --dreamerv3.loss_scales.domain_probe 0.0 \
#  --dreamerv3.loss_scales.sem_rollout 0.3 \
#  --dreamerv3.domain_scale 0.0 \
#  --env.include_domain_id False \
#  --env.include_vlm True \
#  --env.compute_vlm True \
#  --env.augment.enable False"

# "policy_sem_noloss|--dreamerv3.use_semsty True \
#  --dreamerv3.actor_inputs sem \
#  --dreamerv3.critic_inputs sem \
#  --dreamerv3.loss_scales.sem_rollout 0.0 \
#  --dreamerv3.loss_scales.domain_adv 0.0 \
#  --dreamerv3.loss_scales.domain_sty 0.0 \
#  --dreamerv3.loss_scales.domain_probe 0.0 \
#  --dreamerv3.domain_grl_max 0.0 \
#  --dreamerv3.domain_scale 0.0 \
#  --env.compute_vlm False \
#  --env.include_vlm False \
#  --env.include_domain_id False \
#  --env.augment.enable False"

# "domain_adv_small_grl|--dreamerv3.use_semsty True \
#  --dreamerv3.actor_inputs sem \
#  --dreamerv3.critic_inputs sem \
#  --dreamerv3.domain_head_input sem \
#  --dreamerv3.domain_grl_max 0.1 \
#  --dreamerv3.domain_grl_schedule dann \
#  --dreamerv3.loss_scales.domain_adv 1.0 \
#  --dreamerv3.loss_scales.domain_sty 1.0 \
#  --dreamerv3.loss_scales.sem_rollout 1.0 \
#  --dreamerv3.loss_scales.domain_probe 1.0 \
#  --env.include_domain_id True \
#  --env.include_vlm True \
#  --env.augment.enable False"

# "domain_adv_large_grl|--dreamerv3.use_semsty True \
#  --dreamerv3.actor_inputs sem \
#  --dreamerv3.critic_inputs sem \
#  --dreamerv3.domain_head_input sem \
#  --dreamerv3.domain_grl_max 1.0 \
#  --dreamerv3.domain_grl_schedule dann \
#  --dreamerv3.loss_scales.domain_adv 1.0 \
#  --dreamerv3.loss_scales.domain_sty 1.0 \
#  --dreamerv3.loss_scales.sem_rollout 1.0 \
#  --dreamerv3.loss_scales.domain_probe 1.0 \
#  --env.include_domain_id True \
#  --env.include_vlm True \
#  --env.augment.enable False"

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
#  --dreamerv3.num_domains 2 \
#  --env.include_domain_id True \
#  --env.include_vlm True \
#  --env.compute_vlm True \
#  --env.augment.enable False"

# "ours_full_aux|--dreamerv3.use_semsty True \
#  --dreamerv3.actor_inputs deter,stoch \
#  --dreamerv3.critic_inputs deter,stoch \
#  --dreamerv3.domain_head_input sem \
#  --dreamerv3.domain_grl_max 1.0 \
#  --dreamerv3.domain_grl_schedule dann \
#  --dreamerv3.loss_scales.domain_adv 1.0 \
#  --dreamerv3.loss_scales.domain_sty 1.0 \
#  --dreamerv3.loss_scales.sem_rollout 0.3 \
#  --dreamerv3.loss_scales.domain_probe 1.0 \
#  --dreamerv3.num_domains 2 \
#  --env.include_domain_id True \
#  --env.include_vlm True \
#  --env.compute_vlm True \
#  --env.augment.enable False"

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
#  --dreamerv3.num_domains 2 \
#  --env.compute_vlm False \
#  --env.include_domain_id True \
#  --env.include_vlm False \
#  --env.augment.enable False"

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
)

for SEED in "${SEEDS[@]}"; do
  for EXP in "${EXPS[@]}"; do
    NAME="${EXP%%|*}"
    OVERRIDES="${EXP#*|}"

    LOGDIR="/data/feeza/logs_weather/${TASK}/${NAME}/seed_${SEED}"
    mkdir -p "$LOGDIR"

    TRAIN_TOWNS="Town07"   # choose your CARLA map
    EVAL_TOWN="Town07"     # can be the same; your script forces reuse anyway


    ./train_dm3_vlm_weather.sh \
      "$TRAIN_PORTS" "$TRAIN_TOWNS" "$EVAL_PORT" "$EVAL_TOWN" "$GPU" \
      --task "$TASK" \
      --dreamerv3.logdir "$LOGDIR" \
      --env.seed "$SEED" \
      --dreamerv3.seed "$SEED" \
      --dreamerv3.jax.policy_devices 0 \
      --dreamerv3.jax.train_devices 0 \
      --env.train_weather_ids "$TRAIN_WEATHER_IDS" \
      --env.eval_weather_ids  "$EVAL_WEATHER_IDS" \
      $OVERRIDES

  done
done

