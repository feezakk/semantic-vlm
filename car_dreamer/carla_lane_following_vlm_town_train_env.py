# from car_dreamer.carla_lane_following_vlm_town_env import CarlaLaneFollowingVlmTownEnv, TRAIN_ROUTES, EVAL_ROUTES, DEFAULT_EVAL_WEATHER_IDS, DEFAULT_TRAIN_WEATHER_IDS, _get_cfg, ROUTES_BY_TOWN
import gym
from car_dreamer.carla_lane_following_vlm_town_env import (
    CarlaLaneFollowingVlmTownEnv,
    DEFAULT_TRAIN_WEATHER_IDS,
    ROUTES_BY_TOWN,
    _get_cfg,
)

class CarlaLaneFollowingVlmTownTrainEnv(CarlaLaneFollowingVlmTownEnv):
    def __init__(self, config):
        train_weather_ids = _get_cfg(config, "train_weather_ids", DEFAULT_TRAIN_WEATHER_IDS)

        # Robust town getter (supports world.town, env.world.town, town, env.town)
        town = _get_cfg(config, "world.town",
               _get_cfg(config, "env.world.town",
               _get_cfg(config, "town",
               _get_cfg(config, "env.town", "Town06"))))

        if town not in ROUTES_BY_TOWN:
            raise KeyError(f"No routes registered for town={town}. Add to ROUTES_BY_TOWN.")

        super().__init__(
            config=config,
            routes=ROUTES_BY_TOWN[town],
            weather_ids=train_weather_ids,
            eval_mode=False,
            deterministic_eval=False,
        )


