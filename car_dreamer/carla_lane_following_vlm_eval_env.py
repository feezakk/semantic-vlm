from car_dreamer.carla_lane_following_vlm_env import CarlaLaneFollowingVlmEnv, TRAIN_ROUTES, EVAL_ROUTES, DEFAULT_EVAL_WEATHER_IDS, DEFAULT_TRAIN_WEATHER_IDS, _get_cfg
import gym

# ----------------------------
# EVAL env
# ----------------------------
class CarlaLaneFollowingVlmEvalEnv(CarlaLaneFollowingVlmEnv):
    def __init__(self, config):
        eval_weather_ids = _get_cfg(config, "eval_weather_ids", DEFAULT_EVAL_WEATHER_IDS)
        super().__init__(
            config=config,
            routes=EVAL_ROUTES,
            weather_ids=eval_weather_ids,
            eval_mode=True,
            deterministic_eval=True,   # cycles (weather, route) combos deterministically
        )