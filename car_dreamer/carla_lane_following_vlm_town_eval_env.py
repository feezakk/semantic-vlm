# from car_dreamer.carla_lane_following_vlm_town_env import CarlaLaneFollowingVlmTownEnv, TRAIN_ROUTES, EVAL_ROUTES, DEFAULT_EVAL_WEATHER_IDS, DEFAULT_TRAIN_WEATHER_IDS, _get_cfg
import gym
from car_dreamer.carla_lane_following_vlm_town_env import (
    CarlaLaneFollowingVlmTownEnv,
    DEFAULT_TRAIN_WEATHER_IDS,
    ROUTES_BY_TOWN,
    _get_cfg,
    DEFAULT_EVAL_WEATHER_IDS,
    EVAL_ROUTES
)

# ----------------------------
# EVAL env
# ----------------------------
class CarlaLaneFollowingVlmTownEvalEnv(CarlaLaneFollowingVlmTownEnv):
    def __init__(self, config):
        eval_weather_ids = _get_cfg(config, "eval_weather_ids", DEFAULT_EVAL_WEATHER_IDS)
        town = _get_cfg(config, "world.town",
               _get_cfg(config, "env.world.town",
               _get_cfg(config, "env.town",
               _get_cfg(config, "town", "Town04"))))
        routes = ROUTES_BY_TOWN.get(str(town), EVAL_ROUTES)
        super().__init__(
            config=config,
            routes=routes,
            weather_ids=eval_weather_ids,
            eval_mode=True,
            deterministic_eval=True,   # cycles (weather, route) combos deterministically
        )