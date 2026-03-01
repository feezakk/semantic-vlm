import gym
from car_dreamer.carla_lane_following_vlm_weather_env import (
    CarlaLaneFollowingVlmWeatherEnv,
    DEFAULT_TRAIN_WEATHER_IDS,
    TRAIN_ROUTES,
    _get_cfg,
    DEFAULT_EVAL_WEATHER_IDS,
    EVAL_ROUTES
)

class CarlaLaneFollowingVlmWeatherEvalEnv(CarlaLaneFollowingVlmWeatherEnv):
    def __init__(self, config):
        eval_weather_ids = _get_cfg(config, "eval_weather_ids", DEFAULT_EVAL_WEATHER_IDS)
        super().__init__(
            config=config,
            routes=EVAL_ROUTES,
            weather_ids=eval_weather_ids,
            eval_mode=True,
            deterministic_eval=True,   
        )