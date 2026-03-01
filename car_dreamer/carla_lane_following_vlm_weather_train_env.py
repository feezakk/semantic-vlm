import gym
from car_dreamer.carla_lane_following_vlm_weather_env import (
    CarlaLaneFollowingVlmWeatherEnv,
    DEFAULT_TRAIN_WEATHER_IDS,
    TRAIN_ROUTES,
    _get_cfg,
)

class CarlaLaneFollowingVlmWeatherTrainEnv(CarlaLaneFollowingVlmWeatherEnv):
    def __init__(self, config):
        train_weather_ids = _get_cfg(config, "train_weather_ids", DEFAULT_TRAIN_WEATHER_IDS)

        super().__init__(
            config=config,
            routes=TRAIN_ROUTES,
            weather_ids=train_weather_ids,
            eval_mode=False,
            deterministic_eval=False,
        )


