from car_dreamer.carla_lane_following_vlm_env import CarlaLaneFollowingVlmEnv, TRAIN_ROUTES, EVAL_ROUTES, DEFAULT_EVAL_WEATHER_IDS, DEFAULT_TRAIN_WEATHER_IDS, _get_cfg
import gym

# ----------------------------
# TRAIN env
# ----------------------------
class CarlaLaneFollowingVlmTrainEnv(CarlaLaneFollowingVlmEnv):
    def __init__(self, config):
        train_weather_ids = _get_cfg(config, "train_weather_ids", DEFAULT_TRAIN_WEATHER_IDS)
        super().__init__(
            config=config,
            routes=TRAIN_ROUTES,
            weather_ids=train_weather_ids,
            eval_mode=False,
            deterministic_eval=False,
        )

