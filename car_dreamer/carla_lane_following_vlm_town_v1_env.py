import gym
import numpy as np
import math
import carla
from gym import spaces
import random
import time
import cv2
import queue
import weakref
from collections import deque
from pathlib import Path
import itertools
from typing import Tuple, Optional
import pygame

from .toolkit.planner import FixedEndingPlanner

import os
import random
import time




########################################################################################
# Town05 (TRAIN routes)
########################################################################################
TOWN05_TRAIN_SPAWN =  [
    [-272.70, 47.07, 0.0, 0, 90, 0], #L
    [-190.77, 122.54, 0.06, 0, 80, 0], #L
    [-268.89, -36.69, 0.0, 0, -90, 0], #R
    [-188.17, -105.33, 0.1, 0, -90, 0], #R
    [-226.80, 99.52, 10.0, 0, -90, 0], #S
    [-4.48, -207.39, 1.0, 0, -180, 0], #S
]

TOWN05_TRAIN_END = [
    [-206.51, 91.21, 0.0, 0, 0, 0], #L
    [-146.08, 151.24, 0.06, 0, 0.84, 0], #L
    [-208.06, -88.29, 0.0, 0, 0, 0], #R
    [-145.53, -139.30, 0.1, 0, 0, 0], #R
    [-226.80, 49.49, 10.0, 0, -90, 0], #S
    [-123.26, -207.39, 10.05, 0, -180, 0], #S
]

TOWN05_TRAIN_ROUTES = list(zip(TOWN05_TRAIN_SPAWN, TOWN05_TRAIN_END))


########################################################################################
# Town06 (TRAIN routes)
########################################################################################
TOWN06_TRAIN_SPAWN = [
    [659.80, 28.50, 0.0, 0, -100, 0], #L
    [-247.46, 195.54, 0.0, 0, 90.91, 0], #L
    [-230.09, 139.45, 0.0, 0, -180, 0], #R
    [-317.50, 95.52, 0.0, 0, -90, 0], #R
    [146.45, 247.71, 0.0, 0, 0, 0], #S
    [298.46, -20.31, 0.0, 0, 180, 0], #S
]

TOWN06_TRAIN_END = [
    [580.75, -16.72, 0.0, 0, -180, 0], #L
    [-231.63, 231.24, 0.0, 0, 33.48, 0], #L
    [-317.50, 95.52, 0.0, 0, -96, 0], #R
    [-230.76, 49.86, 0.0, 0, -3.24, 0], #R
    [265.78, 247.71, 0.0, 0, 0, 0], #S
    [190.26, -20.31, 0.0, 0, 180, 0], #S
]

TOWN06_TRAIN_ROUTES = list(zip(TOWN06_TRAIN_SPAWN, TOWN06_TRAIN_END))

# ----------------------------
# Town04 (EVAL routes)
# ----------------------------
TOWN04_EVAL_SPAWN = [
    [-185.12, 432.08, 0.0, 0, 0, 0], #L
    [288.38, 37.73, 2.06, 0, 1.02, 0], #L
    [277.78, -365.03, 0.0, 0, 10.75, 0], #R
    [34.23, -299.55, 0.0, 0, -57.86, 0], #R
    [-350.19, 33.68, 0.0, 0, 0, 0], #S
    [-124.33, 9.56, 8.46, 0, 180, 0], #S
]
        
TOWN04_EVAL_END = [
    [-84.72, 398.98, 0.0, 0, -32.07, 0], #L
    [389.61, 18.68, 0.01, 0, -44.66, 0], #L
    [373.85, -288.92, 0.0, 0, 65.06, 0], #R
    [111.44, -360.14, 0.0, 0, -21.83, 0], #R
    [-240.41, 33.62, 2.71, 0, 0, 0], #S
    [-239.00, 9.12, 2.77, 0, 180, 0], #S
]


EVAL_ROUTES = list(zip(TOWN04_EVAL_SPAWN, TOWN04_EVAL_END))

ROUTES_BY_TOWN = {
    "Town05": TOWN05_TRAIN_ROUTES,
    "Town06": TOWN06_TRAIN_ROUTES,
    "Town04": EVAL_ROUTES,  # keep your eval routes
}


# # ----------------------------
# # Discrete actions
# # ----------------------------
# DISCRETE_ACC = [0.0, 0.3, 0.6]
# DISCRETE_STEER = [-0.5, -0.2, -0.1, 0.0, 0.1, 0.2, 0.5]

# ----------------------------
# Discrete actions
# ----------------------------
# Longitudinal command: <0 => brake, >0 => throttle
DISCRETE_LONG = [-0.6, -0.3, 0.0, 0.3, 0.6]  # values are in [0,1] after mapping

# Steering command
DISCRETE_STEER = [-0.5, -0.25, -0.1, 0.0, 0.1, 0.25, 0.5]


# ----------------------------
# Helper
# ----------------------------
def distance_2d(loc1, loc2) -> float:
    return math.sqrt((loc1.x - loc2.x) ** 2 + (loc1.y - loc2.y) ** 2)


def _get_cfg(obj, path: str, default=None):
    """
    Robust nested config getter. Works for dict-like or attribute-like configs.
    Example: _get_cfg(config, "world.carla_port", 3000)
    """
    cur = obj
    for key in path.split("."):
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(key, None)
        else:
            cur = getattr(cur, key, None)
    return default if cur is None else cur


# ----------------------------
# Weather table (domain_id = weather_id)
# ----------------------------
WEATHER_TABLE = [
    ("ClearNoon",       carla.WeatherParameters.ClearNoon),
    ("CloudyNoon",      carla.WeatherParameters.CloudyNoon),
    ("WetNoon",         carla.WeatherParameters.WetNoon),
    ("WetCloudyNoon",   carla.WeatherParameters.WetCloudyNoon),
    ("SoftRainNoon",    carla.WeatherParameters.SoftRainNoon),
    ("MidRainyNoon",    carla.WeatherParameters.MidRainyNoon),
    ("HardRainNoon",    carla.WeatherParameters.HardRainNoon),

    ("ClearSunset",     carla.WeatherParameters.ClearSunset),
    ("CloudySunset",    carla.WeatherParameters.CloudySunset),
    ("WetSunset",       carla.WeatherParameters.WetSunset),
    ("WetCloudySunset", carla.WeatherParameters.WetCloudySunset),
    ("SoftRainSunset",  carla.WeatherParameters.SoftRainSunset),
    ("MidRainSunset",   carla.WeatherParameters.MidRainSunset),
    ("HardRainSunset",  carla.WeatherParameters.HardRainSunset),
]

# # Default train/eval weather splits (IDs into WEATHER_TABLE)
# DEFAULT_TRAIN_WEATHER_IDS = [0, 1, 2, 4]       # clear/cloudy/wet/soft-rain noon
# DEFAULT_EVAL_WEATHER_IDS  = [6, 7, 8, 10, 13]  # hard rain + sunset variants

# # Train: include both Noon and Sunset for each coarse class
# DEFAULT_TRAIN_WEATHER_IDS = [0, 7, 1, 8, 2, 9, 4, 11]
# DEFAULT_EVAL_WEATHER_IDS = [3, 10, 5, 12, 6, 13]

# # Train: Noon variants (one per domain)
# DEFAULT_TRAIN_WEATHER_IDS = [0, 1, 2, 4]           # clearNoon, cloudyNoon, wetNoon, softRainNoon

# # Eval: Sunset / harder variants (one per domain)
# DEFAULT_EVAL_WEATHER_IDS  = [7, 8, 10, 12, 13]     # clearSunset, cloudySunset, wetCloudySunset, midRainSunset, hardRainSunset

# Train: Noon variants (one per domain)
DEFAULT_TRAIN_WEATHER_IDS = [0,0]           # clearNoon, cloudyNoon, wetNoon, softRainNoon

# Eval: Sunset / harder variants (one per domain)
DEFAULT_EVAL_WEATHER_IDS  = [0,0]     # clearSunset, cloudySunset, wetCloudySunset, midRainSunset, hardRainSunset


class CarlaLaneFollowingVlmTownEnv(gym.Env):
    """
    Base env that:
      - samples routes from a provided route list
      - samples weather from provided weather_id list
      - returns domain_id = weather_id (float32 shape (1,))
    """

    def __init__(
        self,
        config,
        routes,
        weather_ids,
        eval_mode: bool = False,
        deterministic_eval: bool = True,
        save_dir: str = "data",
    ):
        super().__init__()
        assert len(routes) > 0, "routes must be non-empty"
        # assert len(weather_ids) > 0, "weather_ids must be non-empty"

        # key = "env.eval_weather_ids" if eval_mode else "env.train_weather_ids"
        # override = _get_cfg(config, key, None)

        # Put this inside CarlaLaneFollowingVlmTownEnv (class scope)
        self.REWARD_COLORS = {
            "total":          (255, 255, 255),

            "route_progress": ( 60, 220,  60),
            "goal":           (  0, 200,   0),

            "r_speed":        (  0, 220, 220),
            "lane":           (255, 165,   0),
            "heading":        (255, 215,   0),

            "time":           (180, 180, 180),
            "low_speed":      (255, 105, 180),

            "wrong_lane":     (255,   0, 255),
            "offroad":        (255,  60,  60),

            "invasion":       (138,  43, 226),
            "collision":      (139,   0,   0),

            "waypoint":       ( 80, 160, 255),
        }

        key_plain = "eval_weather_ids" if eval_mode else "train_weather_ids"

        override = _get_cfg(config, key_plain, None)
        if override is None:
            override = _get_cfg(config, f"env.{key_plain}", None)


        if override is None:
            use_weather_ids = weather_ids
        elif isinstance(override, str):
            use_weather_ids = [int(x) for x in override.split(",") if x.strip()]
        elif isinstance(override, (list, tuple)):
            use_weather_ids = [int(x) for x in override]
        else:
            use_weather_ids = [int(override)]

        assert use_weather_ids is not None and len(use_weather_ids) > 0, (
            f"weather_ids must be non-empty (got {use_weather_ids} from {key})."
        )

        self._config = config
        self.routes = routes
        self.weather_ids = list(use_weather_ids)

        self.eval_mode = bool(eval_mode)
        self.deterministic_eval = bool(deterministic_eval)

        # ---- CARLA connection (IMPORTANT: do NOT hardcode port) ----
        host = _get_cfg(config, "world.carla_host", "localhost")
        port = int(_get_cfg(config, "world.carla_port", _get_cfg(config, "carla_port", 3000)))
        self.client = carla.Client(host, port)
        self.client.set_timeout(float(_get_cfg(config, "world.timeout_s", 300.0)))

        # w = self.client.get_world()
        # try:
        #     name = w.get_map().name
        # except RuntimeError:
        #     name = ""
        # if name != "Carla/Maps/Town07":
        #     w = self.client.load_world("Town07")

        w = self.client.get_world()
        try:
            cur_map = w.get_map().name
        except RuntimeError:
            cur_map = ""

        # target_town = _get_cfg(config, "env.town", _get_cfg(config, "town", "Town07"))

        target_town = _get_cfg(config, "world.town",
               _get_cfg(config, "env.world.town",
               _get_cfg(config, "env.town",
               _get_cfg(config, "town", "Town07"))))


        target_map = f"Carla/Maps/{target_town}"

        if cur_map != target_map:
            w = self.client.load_world(target_town)

        self.world = w
        self.map = self.world.get_map()
        print(f"[CARLA] host={host} port={port} map={self.map.name}")


        # self.world = w
        # self.map = self.world.get_map()

        # sync settings
        settings = self.world.get_settings()
        fixed_dt = float(_get_cfg(config, "world.fixed_delta_seconds", 0.05))
        if (not settings.synchronous_mode) or (settings.fixed_delta_seconds != fixed_dt):
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = fixed_dt
            self.world.apply_settings(settings)
        self._sync_enabled = True
        self._fixed_dt = fixed_dt

        print("CARLA environment initialized")
        print("Map name:", self.map.name)
        print(f"[CARLA] host={host} port={port} sync_dt={fixed_dt}")

        # blueprint library
        self.blueprint_library = self.world.get_blueprint_library()

        # actors
        self.ego = None
        self.actors = []

        # episode/time
        self._time_step = 0
        self._max_time_step = int(_get_cfg(config, "time_limit_steps", 2000))

        # collision/lane
        self.collision_detected = False
        self.collision_hist = []
        self.lane_invasion_detected = False
        self.lane_invasion_hist = []
        self.previous_lane_invasions = 0
        self.previous_collisions = 0
        self._lane_invasion_step = 0

        # rendering
        self.render_enabled = bool(_get_cfg(config, "render", True))

        self.include_vlm = bool(_get_cfg(config, "include_vlm", True))
        self.include_domain_id = bool(_get_cfg(config, "include_domain_id", True))

        # ----------------------------
        # Domain label selection
        # ----------------------------
        self.domain_kind = str(_get_cfg(config, "domain_kind", _get_cfg(config, "env.domain_kind", "weather"))).lower()

        domain_towns = _get_cfg(config, "domain_towns", _get_cfg(config, "env.domain_towns", None))
        if domain_towns is None:
            domain_towns = []
        if isinstance(domain_towns, str):
            domain_towns = [t.strip() for t in domain_towns.split(",") if t.strip()]
        self.domain_towns = list(domain_towns)

        if self.domain_kind == "town":
            if len(self.domain_towns) < 2:
                raise ValueError("domain_kind='town' requires domain_towns with at least 2 entries, e.g. Town05,Town06.")
            self._town_to_domain = {t: i for i, t in enumerate(self.domain_towns)}
            self.num_domains = len(self.domain_towns)
        else:
            # original weather-domain assumption
            self.num_domains = int(_get_cfg(config, "num_domains", 4))


        # VLM
        self.vlm_dim = int(_get_cfg(config, "vlm_dim", 512))
        self.compute_vlm = bool(_get_cfg(config, "compute_vlm", True))
        self._vlm = None
        self._vlm_preprocess = None
        self._vlm_device = str(_get_cfg(config, "vlm_device", "cpu"))
        
        # domains: fixed shared domain labels (0..3) from _weather_to_domain
        # self.num_domains = int(_get_cfg(config, "num_domains", 4))

        # self.DOMAIN_NAMES = ("clear", "cloudy", "wet", "rain")
        self.DOMAIN_NAMES = ("clear")

        # Build a mapping domain -> list of weather_ids in that domain
        # self._domain_to_weather = {d: [] for d in range(self.num_domains)}
        # for wid in self.weather_ids:
        #     d = int(self._weather_to_domain(wid))
        #     self._domain_to_weather[d].append(int(wid))

        self._weather_domain_to_weather = {d: [] for d in range(1)}  # always 4 weather buckets
        for wid in self.weather_ids:
            d = int(self._weather_to_domain(wid))  # returns 0..3
            self._weather_domain_to_weather[d].append(int(wid))


        # Sanity check: every domain should be present in training, or sampling will be biased.
        # if not self.eval_mode:
        #     missing = [d for d, ws in self._domain_to_weather.items() if len(ws) == 0]
        #     if missing:
        #         raise ValueError(f"Training weather_ids do not cover domains {missing}. "
        #                         f"Either add weathers or reduce num_domains.")

        if not self.eval_mode:
            missing = [d for d, ws in self._weather_domain_to_weather.items() if len(ws) == 0]
            if missing:
                raise ValueError(
                    f"Training weather_ids do not cover weather buckets {missing}. "
                    f"Either add weathers or adjust DEFAULT_TRAIN_WEATHER_IDS."
                )

        # expected = len(self.DOMAIN_NAMES)
        # if self.num_domains != expected:
        #     raise ValueError(
        #         f"num_domains must be {expected} for DOMAIN_NAMES={self.DOMAIN_NAMES}, "
        #         f"but got num_domains={self.num_domains}. Update your config."
        #     )
        
        if self.domain_kind == "weather":
            expected = len(self.DOMAIN_NAMES)
            if self.num_domains != expected:
                raise ValueError(
                    f"num_domains must be {expected} for DOMAIN_NAMES={self.DOMAIN_NAMES}, "
                    f"but got num_domains={self.num_domains}. Update your config."
                )

        print(f"[Domain] num_domains={self.num_domains} ({self.DOMAIN_NAMES}) | "
            f"weather_table={len(WEATHER_TABLE)} | weather_ids={self.weather_ids} eval_mode={self.eval_mode}")

        if self.compute_vlm:
            try:
                import torch
                import open_clip
                model, _, preprocess = open_clip.create_model_and_transforms(
                    "ViT-B-32", pretrained="laion2b_s34b_b79k"
                )
                model.eval()
                model.to(self._vlm_device)
                self._vlm = model
                self._vlm_preprocess = preprocess
                pdev = next(self._vlm.parameters()).device
                print(f"[VLM] enabled, device={pdev}, dim={self.vlm_dim}")
            except Exception as e:
                print("WARNING: VLM not available; will output zeros. Error:", e)
                self.compute_vlm = False

        # spaces
        self.action_space = self._setup_action_space()
        self.observation_space = self._setup_observation_space()

        # route/weather schedulers
        self._route_queue = deque()
        self._combo_queue = deque()
        self._refill_route_queue()
        if self.eval_mode and self.deterministic_eval:
            self._refill_combo_queue()

        # camera
        self.camera_image = None
        self.camera_image2 = None
        self._img_q = None
        self._last_tick_frame = None

        # planner and goal
        self.ego_planner = None
        self.waypoints = []
        self.planner_stats = {"num_completed": 0}
        self.num_completed = 0
        self.last_num_completed = 0

        # tracking
        self.speed_kmh = 0.0

        # distance/off-centre
        self.prev_ego_location = None
        self.travel_distance_m = 0.0
        self.off_center_sum = 0.0
        self.off_center_steps = 0

        # episode buffer
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.episode_id = 0
        self.timestep = 0

        # per-episode route/weather id
        self.route_id = 0
        self.weather_id = 0
        self.weather_name = "ClearNoon"
        self.domain_id = 0
        self.domain_mask = 1.0

        # start/end transforms (per episode)
        self.ego_transform = None
        self.end_point = None
        self._start_xy = None  # for past_goal logic

        # initial/last dist to goal
        self.initial_distance_to_goal = None
        self.last_distance_to_goal = None

        # pygame for rendering
        # default_render = True #if self.eval_mode else False
        # self.render_enabled = bool(getattr(config, "render", default_render))  # set True for eval
        # self.render_size = int(getattr(config, "render_size", 512))
        # self.render_fps_cap = int(getattr(config, "render_fps_cap", 20))  # cap UI updates
        # self._pg_initialized = False
        # self._pg_screen = None
        # self._pg_clock = None
        # self._pg_font = None
        # self._pg_last_tick = 0.0

        # pygame for rendering (per-process, one window)
        self.render_enabled = bool(_get_cfg(config, "render", True))
        self.render_size = int(_get_cfg(config, "render_size", 512))
        self.render_fps_cap = int(_get_cfg(config, "render_fps_cap", 20))  # cap UI updates

        town_short = self.map.name.split("/")[-1]  # "Town05", "Town06", ...
        host = _get_cfg(config, "world.carla_host", "localhost")
        port = int(_get_cfg(config, "world.carla_port", _get_cfg(config, "carla_port", 3000)))

        # Unique window title per env/process
        self.render_window_title = str(_get_cfg(
            config,
            "render_window_title",
            f"TRAIN | {town_short} | {host}:{port}"
        ))

        # Optional: window position so Town05/Town06 don't overlap
        pos = _get_cfg(config, "render_window_pos", None)
        if isinstance(pos, str):
            parts = [p.strip() for p in pos.split(",")]
            self.render_window_pos = (int(parts[0]), int(parts[1])) if len(parts) == 2 else None
        elif isinstance(pos, (list, tuple)) and len(pos) == 2:
            self.render_window_pos = (int(pos[0]), int(pos[1]))
        else:
            self.render_window_pos = None

        self._pg_initialized = False
        self._pg_screen = None
        self._pg_clock = None
        self._pg_font = None
        self._pg_last_tick = 0.0


        # ---- Episode accounting (for printing at termination) ----
        self.episode_return = 0.0
        self.episode_length = 0

        self.aug_enable = bool(_get_cfg(config, "augment.enable", False))
        self.aug_brightness = float(_get_cfg(config, "augment.brightness", 0.0))
        self.aug_contrast = float(_get_cfg(config, "augment.contrast", 0.0))
        self.aug_gamma = float(_get_cfg(config, "augment.gamma", 0.0))
        self.aug_blur_prob = float(_get_cfg(config, "augment.blur_prob", 0.0))
        self.aug_noise_std = float(_get_cfg(config, "augment.noise_std", 0.0))

        seed = int(_get_cfg(config, "seed", 0))
        self._rng = np.random.RandomState(seed)

        # self._best_dist_to_goal = float(self.initial_distance_to_goal)
        self._dist_regret_steps = 0

        # self.include_vlm = bool(_get_cfg(config, "include_vlm", True))
        # self.include_domain_id = bool(_get_cfg(config, "include_domain_id", True))

        # if not self.include_vlm:
        #     self.compute_vlm = False  # prevents CLIP loading + encoding

        # ----------------------------
        # Lane-keeping (multi-lane towns)
        # ----------------------------
        # terminate if agent stays in wrong lane / off-road for too long
        # self.max_wrong_lane_steps = int(_get_cfg(config, "max_wrong_lane_steps", 80))
        # self.max_offroad_steps = int(_get_cfg(config, "max_offroad_steps", 40))

        self.wrong_lane_steps = 0
        self.offroad_steps = 0

        # last-step flags (set every step in _compute_reward)
        self._wrong_lane = False
        self._offroad = False

        # ----------------------------
        # Lane-keeping termination tuning (robust + time-based)
        # ----------------------------
        # Grace period after reset: ignore wrong-lane/offroad counters for a short time
        self.lane_grace_s = float(_get_cfg(config, "lane_grace_s", 1.0 if not self.eval_mode else 1.0))
        self._lane_grace_steps = int(round(self.lane_grace_s / self._fixed_dt))

        # Persistent violation durations (seconds)
        self.max_wrong_lane_s = float(_get_cfg(config, "max_wrong_lane_s", 3.0 if not self.eval_mode else 3.0))
        self.max_offroad_s    = float(_get_cfg(config, "max_offroad_s", 1.0 if not self.eval_mode else 1.0))

        # Convert to steps
        self.max_wrong_lane_steps = int(round(self.max_wrong_lane_s / self._fixed_dt))
        self.max_offroad_steps    = int(round(self.max_offroad_s / self._fixed_dt))

        # Counters
        self.wrong_lane_steps = 0
        self.offroad_steps = 0

        # last-step flags (set every step in _compute_reward)
        self._wrong_lane = False
        self._offroad = False

        # Past-goal termination state
        self._ever_near_goal = False
        self._past_goal_steps = 0
        self._dist_to_goal_m = None
        self._prev_dist_to_goal_m = None

        # Route-progress potential (route-remaining distance)
        self._prev_route_remaining_m = None


    # ----------------------------
    # Domain ID
    # ----------------------------

    def _weather_to_domain(self, weather_id: int) -> int:
        name = WEATHER_TABLE[int(weather_id)][0].lower()

        if "rain" in name:
            return 3   # rain
        elif "wet" in name:
            return 2   # wet
        elif "cloudy" in name:
            return 1   # cloudy
        else:
            return 0   # clear

    
    # ----------------------------
    # RGB Augmentation
    # ----------------------------

    def _augment_rgb(self, rgb):
        # rgb uint8 HxWx3
        x = rgb.astype(np.float32) / 255.0

        # brightness/contrast
        if self.aug_brightness > 0:
            b = self._rng.uniform(-self.aug_brightness, self.aug_brightness)
            x = x + b
        if self.aug_contrast > 0:
            c = self._rng.uniform(1 - self.aug_contrast, 1 + self.aug_contrast)
            x = (x - 0.5) * c + 0.5

        # gamma
        if self.aug_gamma > 0:
            g = self._rng.uniform(1 - self.aug_gamma, 1 + self.aug_gamma)
            x = np.clip(x, 0, 1) ** g

        # blur
        if self.aug_blur_prob > 0 and self._rng.rand() < self.aug_blur_prob:
            k = int(self._rng.choice([3, 5]))
            x = cv2.GaussianBlur(x, (k, k), 0)

        # noise
        if self.aug_noise_std > 0:
            n = self._rng.normal(0, self.aug_noise_std, size=x.shape).astype(np.float32)
            x = x + n

        x = np.clip(x, 0, 1)
        return (x * 255.0).astype(np.uint8)


    # ----------------------------
    # Rendering
    # ----------------------------

    # def _pygame_init(self):
    #     if self._pg_initialized:
    #         return
    #     pygame.init()
    #     pygame.display.set_caption("CARLA Eval Viewer")
    #     self._pg_screen = pygame.display.set_mode((self.render_size, self.render_size))
    #     self._pg_clock = pygame.time.Clock()
    #     self._pg_font = pygame.font.SysFont("monospace", 18)
    #     self._pg_initialized = True

    def _pygame_init(self):
        if self._pg_initialized:
            return

        # IMPORTANT: set position BEFORE creating the window
        if getattr(self, "render_window_pos", None) is not None:
            x, y = self.render_window_pos
            os.environ["SDL_VIDEO_WINDOW_POS"] = f"{x},{y}"
            os.environ["SDL_VIDEO_CENTERED"] = "0"

        pygame.init()
        pygame.display.set_caption(getattr(self, "render_window_title", "CARLA"))
        self._pg_screen = pygame.display.set_mode((self.render_size, self.render_size))
        self._pg_clock = pygame.time.Clock()
        self._pg_font = pygame.font.SysFont("monospace", 18)
        self._pg_initialized = True

    def _pygame_render(self, rgb_uint8, lines=None):
        """
        rgb_uint8: HxWx3 in RGB uint8 (your camera_image2 is already RGB).
        lines: list[str] optional overlay text lines.
        """
        if not self.render_enabled:
            return
        if rgb_uint8 is None:
            return

        self._pygame_init()

        # Handle window events (required, or OS may think it's frozen)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                # Close cleanly; you can also set a flag and terminate episode.
                self.close()
                raise SystemExit("Pygame window closed by user.")

        # Optional FPS cap for UI updates
        if self.render_fps_cap > 0:
            self._pg_clock.tick(self.render_fps_cap)

        # Resize to render_size if needed (avoid expensive ops if already correct)
        img = rgb_uint8
        if img.shape[0] != self.render_size or img.shape[1] != self.render_size:
            img = cv2.resize(img, (self.render_size, self.render_size), interpolation=cv2.INTER_LINEAR)

        # Pygame surfarray expects (W,H,3), while numpy is (H,W,3)
        surf = pygame.surfarray.make_surface(np.transpose(img, (1, 0, 2)))
        self._pg_screen.blit(surf, (0, 0))

        # # Overlay text
        # if lines:
        #     y = 8
        #     for s in lines[:14]:
        #         text = self._pg_font.render(str(s), True, (255, 255, 255))
        #         # simple shadow for readability
        #         shadow = self._pg_font.render(str(s), True, (0, 0, 0))
        #         self._pg_screen.blit(shadow, (9, y + 1))
        #         self._pg_screen.blit(text, (8, y))
        #         y += 20

        # Overlay text (supports per-line colors)
        if lines:
            y0 = 8
            line_h = 20
            y = y0

            # Fit as many lines as possible in the window
            max_lines = int((self.render_size - 2 * y0) // line_h)

            for item in lines[:max_lines]:
                if isinstance(item, tuple) and len(item) == 2:
                    s, color = item
                else:
                    s, color = item, (255, 255, 255)

                # simple shadow for readability
                shadow = self._pg_font.render(str(s), True, (0, 0, 0))
                text   = self._pg_font.render(str(s), True, color)

                self._pg_screen.blit(shadow, (9, y + 1))
                self._pg_screen.blit(text,   (8, y))
                y += line_h


        pygame.display.flip()


    # ----------------------------
    # queues
    # ----------------------------
    def _refill_route_queue(self):
        idxs = list(range(len(self.routes)))
        if not self.eval_mode:
            random.shuffle(idxs)
        self._route_queue = deque(idxs)

    # def _refill_combo_queue(self):
    #     """
    #     Deterministic eval schedule, but DOMAIN-BALANCED.
    #     Each coarse domain contributes equally by upsampling weather_ids within the domain.
    #     """
    #     dom_to_wids = {d: [] for d in range(self.num_domains)}
    #     for wid in self.weather_ids:
    #         d = int(self._weather_to_domain(wid))
    #         if d < 0 or d >= self.num_domains:
    #             raise ValueError(f"weather_id {wid} mapped to domain {d}, num_domains={self.num_domains}")
    #         dom_to_wids[d].append(int(wid))

    #     missing = [d for d, ws in dom_to_wids.items() if len(ws) == 0]
    #     if missing:
    #         raise ValueError(
    #             f"Eval weather_ids do not cover domains {missing}. "
    #             f"Fix eval_weather_ids / DEFAULT_EVAL_WEATHER_IDS."
    #         )

    #     # Upsample each domain list to the same length
    #     m = max(len(ws) for ws in dom_to_wids.values())
    #     balanced_wids = []
    #     for d in range(self.num_domains):
    #         ws = dom_to_wids[d]
    #         reps = (m + len(ws) - 1) // len(ws)
    #         balanced_wids.extend((ws * reps)[:m])  # exactly m per domain

    #     combos = [(rid, wid) for rid in range(len(self.routes)) for wid in balanced_wids]

    #     seed = int(_get_cfg(self._config, "eval_combo_seed", 0))
    #     rnd = random.Random(seed)
    #     rnd.shuffle(combos)
    #     self._combo_queue = deque(combos)

    def _refill_combo_queue(self):
        dom_to_wids = {d: [] for d in range(1)}  # ALWAYS 4 weather buckets
        for wid in self.weather_ids:
            d = int(self._weather_to_domain(wid))  # 0..3
            dom_to_wids[d].append(int(wid))

        missing = [d for d, ws in dom_to_wids.items() if len(ws) == 0]
        if missing:
            raise ValueError(f"Eval weather_ids do not cover weather buckets {missing}.")

        m = max(len(ws) for ws in dom_to_wids.values())
        balanced_wids = []
        for d in range(1):
            ws = dom_to_wids[d]
            reps = (m + len(ws) - 1) // len(ws)
            balanced_wids.extend((ws * reps)[:m])

        combos = [(rid, wid) for rid in range(len(self.routes)) for wid in balanced_wids]
        seed = int(_get_cfg(self._config, "eval_combo_seed", 0))
        rnd = random.Random(seed)
        rnd.shuffle(combos)
        self._combo_queue = deque(combos)


    def _select_route_and_weather(self):
        if self.eval_mode and self.deterministic_eval:
            if not self._combo_queue:
                self._refill_combo_queue()
            # wid, rid = self._combo_queue.popleft()
            # return int(rid), int(wid)
            rid, wid = self._combo_queue.popleft()
            return int(rid), int(wid)

        # training / non-deterministic eval
        # training / non-deterministic eval
        if not self._route_queue:
            self._refill_route_queue()
        rid = int(self._route_queue.popleft())

        # valid_domains = [d for d, ws in self._domain_to_weather.items() if len(ws) > 0]
        # dom = int(self._rng.choice(valid_domains))
        # wid = int(self._rng.choice(self._domain_to_weather[dom]))

        valid_wdoms = [d for d, ws in self._weather_domain_to_weather.items() if len(ws) > 0]
        wdom = int(self._rng.choice(valid_wdoms))
        wid = int(self._rng.choice(self._weather_domain_to_weather[wdom]))

        return rid, wid


    # ----------------------------
    # spaces
    # ----------------------------
    # def _setup_action_space(self):
    #     self.n_steer = len(DISCRETE_STEER)
    #     self.n_acc = len(DISCRETE_ACC)
    #     return spaces.Discrete(self.n_steer * self.n_acc)

    def _setup_action_space(self):
        self.n_steer = len(DISCRETE_STEER)
        self.n_long = len(DISCRETE_LONG)

        # Optional: keep backward-compat name if other code expects self.n_acc
        self.n_acc = self.n_long

        return spaces.Discrete(self.n_steer * self.n_long)


    def _setup_observation_space(self):
        camera_space = spaces.Box(0, 255, shape=(128, 128, 3), dtype=np.uint8)
        collision_space = spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32)
        lane_invasion_space = spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32)
        vlm_space = spaces.Box(-np.inf, np.inf, shape=(self.vlm_dim,), dtype=np.float32)
        domain_space = spaces.Box(0.0, float(self.num_domains - 1), shape=(1,), dtype=np.float32)
        domain_mask_space = spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32)

        # return spaces.Dict({
        #     "image": camera_space,
        #     "collision": collision_space,
        #     "lane_invasion": lane_invasion_space,
        #     "vlm": vlm_space,
        #     "domain_id": domain_space,
        # })
    
        spaces_dict = {
            "image": camera_space,
            "collision": collision_space,
            "lane_invasion": lane_invasion_space,
        }

        if self.include_vlm:
            spaces_dict["vlm"] = vlm_space
        if self.include_domain_id:
            spaces_dict["domain_id"] = domain_space
            spaces_dict["domain_mask"] = domain_mask_space

        return spaces.Dict(spaces_dict)

    # ----------------------------
    # VLM
    # ----------------------------
    def _encode_vlm(self, rgb_uint8: np.ndarray) -> np.ndarray:
        if (not self.compute_vlm) or (self._vlm is None):
            return np.zeros((self.vlm_dim,), np.float32)

        import torch
        from PIL import Image

        img = Image.fromarray(rgb_uint8)
        x = self._vlm_preprocess(img).unsqueeze(0).to(self._vlm_device)
        with torch.no_grad():
            feat = self._vlm.encode_image(x)
            feat = feat / (feat.norm(dim=-1, keepdim=True) + 1e-8)
        return feat.squeeze(0).cpu().numpy().astype(np.float32)

    # ----------------------------
    # sensors
    # ----------------------------
    def setup_camera(self):
        cam_bp = self.blueprint_library.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", "512")
        cam_bp.set_attribute("image_size_y", "512")
        cam_bp.set_attribute("fov", "110")
        cam_bp.set_attribute("sensor_tick", f"{self._fixed_dt}")  # match sim dt

        # camera_spawn = carla.Transform(carla.Location(z=10), carla.Rotation(pitch=-90))
        camera_spawn = carla.Transform(carla.Location(x=1.5, z=1.6), carla.Rotation(pitch=-5))

        self.camera_sensor = self.world.spawn_actor(cam_bp, camera_spawn, attach_to=self.ego)
        self.actors.append(self.camera_sensor)

        self._img_q = queue.Queue(maxsize=16)
        ws = weakref.ref(self)

        def _on_image(img, q=self._img_q, ws=ws):
            s = ws()
            if s is None:
                return
            arr = np.frombuffer(img.raw_data, dtype=np.uint8).reshape((img.height, img.width, 4))
            rgb = arr[:, :, :3][:, :, ::-1].copy()  # BGRA -> RGB
            item = (int(img.frame), rgb)
            try:
                q.put_nowait(item)
            except queue.Full:
                pass  # drop

        self.camera_sensor.listen(_on_image)

    def setup_collision_sensor(self):
        bp = self.blueprint_library.find("sensor.other.collision")
        self.colsensor = self.world.spawn_actor(bp, carla.Transform(), attach_to=self.ego)
        self.actors.append(self.colsensor)
        self.colsensor.listen(lambda event: self._on_collision(event))

    def setup_lane_invasion_sensor(self):
        bp = self.blueprint_library.find("sensor.other.lane_invasion")
        self.lane_sensor = self.world.spawn_actor(bp, carla.Transform(), attach_to=self.ego)
        self.actors.append(self.lane_sensor)
        self.lane_sensor.listen(lambda event: self._on_lane_invasion(event))

    def _on_collision(self, event):
        self.collision_hist.append(event)
        self.collision_detected = True

    def _on_lane_invasion(self, event):
        self.lane_invasion_hist.append(event)
        self.lane_invasion_detected = True

    # ----------------------------
    # image pulling (frame-aligned best effort)
    # ----------------------------
    def _pull_image_for_frame(self, target_frame: Optional[int], timeout: float = 0.5) -> np.ndarray:
        """
        Best-effort: try to fetch the image matching target_frame.
        Falls back to the latest available frame within timeout.
        """
        if self._img_q is None:
            return np.zeros((512, 512, 3), dtype=np.uint8)

        end = time.time() + timeout
        best = None
        got = False

        while time.time() < end:
            try:
                f, rgb = self._img_q.get(timeout=max(0.0, end - time.time()))
                got = True
                if target_frame is None:
                    best = rgb
                else:
                    if f == target_frame:
                        best = rgb
                        break
                    # If we overshot, keep as fallback and break
                    if f > target_frame:
                        best = rgb
                        break
                    # If f < target_frame: keep looping
                    best = rgb
            except queue.Empty:
                break

        self._got_frame = bool(got and best is not None)
        if best is None:
            best = np.zeros((512, 512, 3), dtype=np.uint8)
        return best

    def _get_observation(self, frame_id: Optional[int] = None):
        rgb = self._pull_image_for_frame(frame_id, timeout=0.5)

        if self.aug_enable and (not self.eval_mode):
            rgb = self._augment_rgb(rgb)

        self.camera_image2 = rgb
        self.camera_image = cv2.resize(rgb, (128, 128), interpolation=cv2.INTER_AREA)

        vlm = self._encode_vlm(rgb)
        # return {
        #     "image": self.camera_image,
        #     "collision": np.array([1.0 if self.collision_detected else 0.0], np.float32),
        #     "lane_invasion": np.array([1.0 if self.lane_invasion_detected else 0.0], np.float32),
        #     "vlm": vlm.astype(np.float32),
        #     "domain_id": np.array([float(self.domain_id)], np.float32),
        # }
    
        obs = {
            "image": self.camera_image,
            "collision": np.array([1.0 if self.collision_detected else 0.0], np.float32),
            "lane_invasion": np.array([1.0 if self.lane_invasion_detected else 0.0], np.float32),
        }

        if self.include_vlm:
            # obs["vlm"] = self._encode_vlm(rgb).astype(np.float32)
            obs["vlm"] = vlm.astype(np.float32)

        if self.include_domain_id:
            obs["domain_id"] = np.array([float(self.domain_id)], np.float32)
            obs["domain_mask"] = np.array([float(self.domain_mask)], np.float32)

        return obs

    # ----------------------------
    # vehicle spawning
    # ----------------------------
    # def reset_vehicle(self):
    #     bp = self.blueprint_library.find("vehicle.tesla.model3")
    #     ego = self.world.try_spawn_actor(bp, self.ego_transform)
    #     if ego is None:
    #         raise RuntimeError("Unable to spawn ego vehicle at the chosen spawn point.")
    #     self.ego = ego
    #     self.actors.append(self.ego)
    #     self.ego.set_target_velocity(carla.Vector3D())
    #     self.ego.apply_control(carla.VehicleControl(
    #         manual_gear_shift=False, reverse=False, hand_brake=False,
    #         steer=0.0, throttle=0.0, brake=0.0
    #     ))

    def reset_vehicle(self):
        """
        Robust ego spawn:
        - retries multiple times
        - optionally destroys blocking actors very near the spawn
        - ticks between attempts so CARLA applies destruction/physics
        """
        bp = self.blueprint_library.find("vehicle.tesla.model3")
        if bp.has_attribute("role_name"):
            bp.set_attribute("role_name", "hero")

        # Copy spawn transform (avoid mutating self.ego_transform by accident)
        base_loc = self.ego_transform.location
        base_rot = self.ego_transform.rotation

        # Tunables (you can move these to config later if you want)
        max_tries = int(_get_cfg(self._config, "spawn_max_tries", 25))
        clear_radius = float(_get_cfg(self._config, "spawn_clear_radius_m", 3.0))
        kill_blockers = bool(_get_cfg(self._config, "spawn_kill_blockers", True))

        # CARLA sometimes fails if z is too tight or if the ground contact is weird
        z_offsets = [0.0, 0.25, 0.5, 0.75, 1.0]

        last_blockers = 0
        last_tf = None

        for i in range(max_tries):
            dz = z_offsets[min(i, len(z_offsets) - 1)]
            tf = carla.Transform(
                carla.Location(x=float(base_loc.x), y=float(base_loc.y), z=float(base_loc.z + dz)),
                carla.Rotation(pitch=float(base_rot.pitch), yaw=float(base_rot.yaw), roll=float(base_rot.roll)),
            )
            last_tf = tf

            # (Optional) Clear anything *very* close to the intended spawn point.
            # This is the most common cause: a vehicle/walker occupies the spawn.
            if kill_blockers:
                try:
                    blockers = []
                    for pat in ("vehicle.*", "walker.*"):
                        for a in self.world.get_actors().filter(pat):
                            # Only clear near the spawn (keep rest of world intact)
                            if a.get_location().distance(tf.location) < clear_radius:
                                blockers.append(a)

                    last_blockers = len(blockers)

                    if blockers:
                        cmds = [carla.command.DestroyActor(a) for a in blockers]
                        self.client.apply_batch_sync(cmds, True)
                        # Important: tick so destruction is actually applied in synchronous mode
                        self.world.tick()
                except Exception:
                    # Don't let blocker-clearing failures kill the episode
                    pass

            # Try to spawn
            ego = None
            try:
                ego = self.world.try_spawn_actor(bp, tf)
            except Exception:
                ego = None

            if ego is not None:
                self.ego = ego
                self.actors.append(self.ego)

                self.ego.set_target_velocity(carla.Vector3D())
                self.ego.apply_control(
                    carla.VehicleControl(
                        manual_gear_shift=False,
                        reverse=False,
                        hand_brake=False,
                        steer=0.0,
                        throttle=0.0,
                        brake=0.0,
                    )
                )
                return

            # Give CARLA a moment to settle and process any pending destruction/physics
            try:
                self.world.tick()
            except Exception:
                pass
            time.sleep(0.02)

        raise RuntimeError(
            f"Unable to spawn ego vehicle after {max_tries} tries. "
            f"last_tf=({last_tf.location.x:.2f},{last_tf.location.y:.2f},{last_tf.location.z:.2f}) "
            f"blockers_last_try={last_blockers}"
        )

    def _route_remaining_m(self, max_pts: int = 50) -> float:
        """
        Approx remaining distance along the current planned route:
        ego -> (planner waypoints...) -> goal
        Falls back to Euclidean ego->goal if waypoints are empty.
        """
        if self.ego is None or self.end_point is None:
            return 0.0

        ego_loc = self.ego.get_location()
        goal_loc = self.end_point.location

        # Fallback: if planner gives nothing, use straight-line distance
        if self.waypoints is None or len(self.waypoints) == 0:
            return float(ego_loc.distance(goal_loc))

        pts = [(float(ego_loc.x), float(ego_loc.y))]
        for w in self.waypoints[:max_pts]:
            pts.append((float(w[0]), float(w[1])))
        pts.append((float(goal_loc.x), float(goal_loc.y)))

        rem = 0.0
        for (x1, y1), (x2, y2) in zip(pts[:-1], pts[1:]):
            rem += float(math.hypot(x2 - x1, y2 - y1))
        return float(rem)

    # ----------------------------
    # gym API
    # ----------------------------
    def reset(self):
        self._clean_actors()

        # choose (route, weather)
        self.route_id, self.weather_id = self._select_route_and_weather()
        self.weather_name, weather_params = WEATHER_TABLE[self.weather_id]
        self.world.set_weather(weather_params)
        # self.domain_id = int(self._weather_to_domain(self.weather_id))

        # if self.domain_kind == "town":
        #     town_short = self.map.name.split("/")[-1]  # "Town05", "Town06", ...
        #     self.domain_id = int(self._town_to_domain.get(town_short, 0))
        # else:
        #     self.domain_id = int(self._weather_to_domain(self.weather_id))

        if self.domain_kind == "town":
            town_short = self.map.name.split("/")[-1]  # "Town05", "Town06", "Town04", ...
            if town_short in self._town_to_domain:
                self.domain_id = int(self._town_to_domain[town_short])
                self.domain_mask = 1.0
            else:
                # OOD town: keep domain_id in range but mask it out for domain losses/metrics.
                self.domain_id = 0
                self.domain_mask = 0.0
        else:
            self.domain_id = int(self._weather_to_domain(self.weather_id))
            self.domain_mask = 1.0

        # route transforms
        start, end = self.routes[self.route_id]
        self.ego_transform = carla.Transform(
            carla.Location(x=float(start[0]), y=float(start[1]), z=float(start[2])),
            carla.Rotation(pitch=float(start[3]), yaw=float(start[4]), roll=float(start[5])),
        )
        self.end_point = carla.Transform(
            carla.Location(x=float(end[0]), y=float(end[1]), z=float(end[2])),
            carla.Rotation(pitch=float(end[3]), yaw=float(end[4]), roll=float(end[5])),
        )
        self._start_xy = (float(start[0]), float(start[1]))

        # reset counters
        self._time_step = 0
        self.collision_detected = False
        self.collision_hist = []
        self.lane_invasion_detected = False
        self.lane_invasion_hist = []
        self.previous_lane_invasions = 0
        self.previous_collisions = 0
        self.low_speed_steps = 0
        self.low_speed_kmh_thresh = float(_get_cfg(self._config, "low_speed_kmh_thresh", 1.0))
        self.low_speed_timeout_s = float(_get_cfg(self._config, "low_speed_timeout_s", 10.0))
        self.low_speed_steps = 0
        self.low_speed_timeout_steps = int(round(self.low_speed_timeout_s / self._fixed_dt))

        self.speed_kmh = 0.0

        self.prev_ego_location = None
        self.travel_distance_m = 0.0
        self.off_center_sum = 0.0
        self.off_center_steps = 0
        self.last_num_completed = 0

        # tick once to apply weather and stabilize
        # frame = self.world.tick()
        # self._last_tick_frame = int(frame)
        snap = self.world.tick()
        frame_id = snap.frame if hasattr(snap, "frame") else int(snap)
        self._last_tick_frame = int(frame_id)

        # spawn ego + sensors
        self.reset_vehicle()
        self.setup_collision_sensor()
        self.setup_lane_invasion_sensor()
        self.setup_camera()

        # tick after sensors so we can get a first valid frame
        # frame = self.world.tick()
        # self._last_tick_frame = int(frame)
        snap = self.world.tick()
        frame_id = snap.frame if hasattr(snap, "frame") else int(snap)
        self._last_tick_frame = int(frame_id)

        # planner
        dest_location = carla.Location(
            x=float(self.end_point.location.x),
            y=float(self.end_point.location.y),
            z=float(self.end_point.location.z),
        )
        self.ego_planner = FixedEndingPlanner(self.ego, dest_location)
        self.waypoints, self.planner_stats = self.ego_planner.run_step()
        self.num_completed = int(self.planner_stats.get("num_completed", 0))

        self._prev_route_remaining_m = self._route_remaining_m()


        # distance to goal init
        self.initial_distance_to_goal = self.ego.get_location().distance(self.end_point.location)
        self.last_distance_to_goal = float(self.initial_distance_to_goal)

        # Past-goal termination state
        self._ever_near_goal = False
        self._past_goal_steps = 0
        self._dist_to_goal_m = None
        self._prev_dist_to_goal_m = None


        # ---- reset episode accounting ----
        self.episode_return = 0.0
        self.episode_length = 0

        self._best_dist_to_goal = float(self.initial_distance_to_goal)
        self._dist_regret_steps = 0

        self.wrong_lane_steps = 0
        self.offroad_steps = 0
        self._wrong_lane = False
        self._offroad = False

        # return obs
        obs = self._get_observation(self._last_tick_frame)
        return obs

    def step(self, action: int):
        # apply action
        self.apply_control(action)

        # tick
        self._time_step += 1
        # frame = self.world.tick()
        # self._last_tick_frame = int(frame)

        snap = self.world.tick()
        frame_id = snap.frame if hasattr(snap, "frame") else int(snap)
        self._last_tick_frame = int(frame_id)

        # distance accumulation
        cur_loc = self.ego.get_location()
        if self.prev_ego_location is None:
            step_dist = 0.0
        else:
            step_dist = distance_2d(cur_loc, self.prev_ego_location)
        self.travel_distance_m += step_dist
        self.prev_ego_location = cur_loc

        # planner update
        self.waypoints, self.planner_stats = self.ego_planner.run_step()
        self.num_completed = int(self.planner_stats.get("num_completed", 0))

        # speed
        vel = self.ego.get_velocity()
        self.speed_kmh = 3.6 * math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)

        # Step-based low-speed tracking (reproducible across machines)
        if self.speed_kmh < self.low_speed_kmh_thresh:
            self.low_speed_steps += 1
        else:
            self.low_speed_steps = 0

        # obs
        obs = self._get_observation(self._last_tick_frame)

        dist_to_goal = float(self.ego.get_location().distance(self.end_point.location))

        # Track distances for past-goal logic
        self._prev_dist_to_goal_m = self._dist_to_goal_m
        self._dist_to_goal_m = dist_to_goal

        near_goal_radius_m = float(_get_cfg(self._config, "near_goal_radius_m", 6.0))
        if dist_to_goal < near_goal_radius_m:
            self._ever_near_goal = True



        prev_dist = float(self.last_distance_to_goal) if self.last_distance_to_goal is not None else dist_to_goal
        progress_m = max(0.0, prev_dist - dist_to_goal)

        # # reward + info
        # reward, info_dict = self._compute_reward()

        # reward + info
        reward, reward_terms = self._compute_reward()

        # Keep a pristine copy for rendering/debugging (because info_dict gets overwritten later)
        reward_terms = dict(reward_terms)

        for k in ("collision", "offroad", "wrong_lane"):
            reward_terms[f"r_{k}"] = reward_terms.get(k, 0.0)

        # Keep existing behavior: start info_dict from reward terms
        info_dict = dict(reward_terms)


        # ----------------------------
        # Persistency counters (for termination)
        # ----------------------------
        # if self._offroad:
        #     self.offroad_steps += 1
        # else:
        #     self.offroad_steps = 0

        # if self._wrong_lane:
        #     self.wrong_lane_steps += 1
        # else:
        #     self.wrong_lane_steps = 0

        # ----------------------------
        # Persistency counters (for termination)
        # ----------------------------
        if self._time_step < getattr(self, "_lane_grace_steps", 0):
            # ignore for the first lane_grace_s seconds after reset
            self.offroad_steps = 0
            self.wrong_lane_steps = 0
        else:
            if self._offroad:
                self.offroad_steps += 1
            else:
                self.offroad_steps = 0

            if self._wrong_lane:
                self.wrong_lane_steps += 1
            else:
                self.wrong_lane_steps = 0



        info_dict.update({
            "goal_reached": 0,
            "time_exceeded": 0,
            "not_moving": 0,
            "past_goal": 0,
            "collision": 0,   # if you also want this as a flag distinct from collision_step
            "wrong_lane": int(self._wrong_lane),
            "offroad": int(self._offroad),
        })

        # ---- accumulate per-episode return/length ----
        self.episode_return += float(reward)
        self.episode_length += 1

        # acc = float(DISCRETE_ACC[action // self.n_steer])
        # steer = float(DISCRETE_STEER[action % self.n_steer])

        long_cmd = float(DISCRETE_LONG[action // self.n_steer])
        steer = float(DISCRETE_STEER[action % self.n_steer])

        throttle_cmd = long_cmd if long_cmd >= 0.0 else 0.0
        brake_cmd = (-long_cmd) if long_cmd < 0.0 else 0.0

        # eval metrics
        # lane_offset = self.get_lane_offset()
        # heading_err = self.get_angle_offset()
        lane_offset, cur_wp, route_wp, target_wp = self.get_lane_offset_to_route_lane()
        heading_err = self.get_angle_offset_to_wp(target_wp if target_wp is not None else cur_wp)

        self.off_center_sum += float(lane_offset)
        self.off_center_steps += 1

        info_dict.update({
            "route_id": int(self.route_id),
            "weather_id": int(self.weather_id),
            "weather_name": str(self.weather_name),
            "step_distance": float(step_dist),
            "off_center_m": float(lane_offset),
            "heading_error": float(heading_err),
            "speed_kmh": float(self.speed_kmh),
            "lane_invasion_step": int(getattr(self, "_lane_invasion_step", 0)),
            "got_frame": int(getattr(self, "_got_frame", False)),
            "action_idx": int(action),
            # "throttle_cmd": acc,
            "long_cmd": float(long_cmd),
            "throttle_cmd": float(throttle_cmd),
            "brake_cmd": float(brake_cmd),
            "steer_cmd": steer,
            "dist_to_goal_m": dist_to_goal,
            "progress_m": float(progress_m),
            "collision_step": int(getattr(self, "_collision_step", 0)),
            "domain_id": int(self.domain_id),
            "town": str(self.map.name.split("/")[-1]),
        })

        # Track whether agent is getting consistently farther from goal
        if dist_to_goal < self._best_dist_to_goal - 0.25:   # 25 cm improvement
            self._best_dist_to_goal = dist_to_goal
            self._dist_regret_steps = 0
        elif dist_to_goal > self._best_dist_to_goal + 2.0:  # 2 m worse than best
            self._dist_regret_steps += 1
        else:
            self._dist_regret_steps = 0

        done, terminal_info = self._check_termination()
        info = {**info_dict, **terminal_info}

        if done:
            # Determine a single termination reason (priority order)
            reason = "unknown"
            # for k in ("goal_reached", "collision", "past_goal", "not_moving", "time_exceeded"):
            for k in ("goal_reached", "collision", "offroad", "wrong_lane", "past_goal", "not_moving", "time_exceeded"):
                if terminal_info.get(k, False):
                    reason = k
                    break

            # Print episode summary
            print(
                f"[Episode {self.episode_id}] DONE | "
                f"steps={self.episode_length} | "
                f"return={self.episode_return:.2f} | "
                f"reason={reason} | "
                f"route={self.route_id} | "
                f"weather={self.weather_name}"
            )

            self.episode_id += 1


        if self.render_enabled and self.camera_image2 is not None:
            # Example overlay lines – customize to your info dict
            loc = self.ego.get_location()
            x, y, z = float(loc.x), float(loc.y), float(loc.z)

            # Vector from ego -> goal (XY)
            goal_loc = self.end_point.location
            dx = float(goal_loc.x - loc.x)
            dy = float(goal_loc.y - loc.y)
            goal_norm = math.hypot(dx, dy)

            # Ego heading (forward vector, XY)
            fwd = self.ego.get_transform().get_forward_vector()
            hx = float(fwd.x)
            hy = float(fwd.y)
            head_norm = math.hypot(hx, hy)

            # Angular distance (heading error to goal direction)
            if goal_norm < 1e-6 or head_norm < 1e-6:
                goal_ang_rad = 0.0
            else:
                gx, gy = dx / goal_norm, dy / goal_norm
                hx, hy = hx / head_norm, hy / head_norm
                dot = max(-1.0, min(1.0, gx * hx + gy * hy))  # clamp for numerical stability
                goal_ang_rad = math.acos(dot)

            goal_ang_deg = goal_ang_rad * 180.0 / math.pi

            rid = int(info.get("route_id", self.route_id))
            if rid in (0, 1):
                route_dir = "left"
            elif rid in (2, 3):
                route_dir = "right"
            elif rid in (4, 5):
                route_dir = "straight"
            else:
                route_dir = "unknown"

            # lines = [
            #     f"ego xyz: ({x:.2f}, {y:.2f}, {z:.2f})",
            #     f"dist_to_goal_m: {dist_to_goal:.2f}",
            #     f"goal_ang_err_deg: {goal_ang_deg:.1f}",
            #     f"progress_m: {progress_m:.2f}",
            #     f"route: {info.get('route_id', 'NA')}",
            #     f"route_dir: {route_dir}", 
            #     f"weather: {info.get('weather_name', info.get('weather_id', 'NA'))}",
            #     f"speed_kmh: {info.get('speed_kmh', 0.0):.1f}",
            #     f"reward: {reward:.2f}",
            #     f"off_center_m: {info.get('off_center_m', 0.0):.2f}",
            #     f"collision: {int(info.get('collision', False))}",
            #     f"lane_inv_step: {int(info.get('lane_invasion_step', 0))}",
            #     f"route_prog: {info.get('route_progress', 0.0):.2f}",
            #     f"time_pen: {info.get('time', 0.0):.2f}",
            # ]
            # self._pygame_render(self.camera_image2, lines=lines)

                        # ---- Build colored overlay lines ----
            state_lines = [
                (f"ego xyz: ({x:.2f}, {y:.2f}, {z:.2f})", (255, 255, 255)),
                (f"dist_to_goal_m: {dist_to_goal:.2f}", (255, 255, 255)),
                (f"goal_ang_err_deg: {goal_ang_deg:.1f}", (255, 255, 255)),
                (f"speed_kmh: {info.get('speed_kmh', 0.0):.1f}", (255, 255, 255)),
                (f"off_center_m: {info.get('off_center_m', 0.0):.2f}", (255, 255, 255)),
            ]

            # Order the reward terms explicitly (consistent overlay)
            reward_order = [
                "route_progress",
                "goal",
                "r_speed",
                "lane",
                "heading",
                "time",
                "low_speed",
                "wrong_lane",
                "offroad",
                "invasion",
                "collision",
                "waypoint",
            ]

            reward_lines = [
                (f"R_total: {reward:+.2f}", self.REWARD_COLORS["total"]),
            ]

            for k in reward_order:
                v = float(reward_terms.get(k, 0.0))
                col = self.REWARD_COLORS.get(k, (255, 255, 255))
                reward_lines.append((f"{k:>13s}: {v:+.2f}", col))

            lines = state_lines + reward_lines
            self._pygame_render(self.camera_image2, lines=lines)


        return obs, float(reward), bool(done), info

    def _signed_ahead_to_goal_m(self) -> float:
        """
        Positive => ego is past the goal plane (beyond goal along route final direction).
        """
        if self.ego is None or self.end_point is None or self._start_xy is None:
            return 0.0

        ego_loc = self.ego.get_location()
        goal_loc = self.end_point.location

        yaw_rad = math.radians(float(self.end_point.rotation.yaw))
        d = np.array([math.cos(yaw_rad), math.sin(yaw_rad)], dtype=np.float32)

        # Align with start->goal direction to avoid flipped yaw conventions
        start = np.array([float(self._start_xy[0]), float(self._start_xy[1])], dtype=np.float32)
        goal = np.array([float(goal_loc.x), float(goal_loc.y)], dtype=np.float32)
        sg = goal - start
        if float(np.dot(d, sg)) < 0.0:
            d = -d

        n = float(np.linalg.norm(d))
        if n < 1e-6:
            return 0.0
        d /= n

        p = np.array([float(ego_loc.x), float(ego_loc.y)], dtype=np.float32)
        pg = p - goal

        return float(np.dot(pg, d))


    def close(self):
        try:
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            self.world.apply_settings(settings)
        except Exception:
            pass
        self._sync_enabled = False
        self._clean_actors()

        if getattr(self, "_pg_initialized", False):
            try:
                pygame.display.quit()
                pygame.quit()
            except Exception:
                pass
            self._pg_initialized = False

    # ----------------------------
    # controls
    # ----------------------------
    def apply_control(self, action: int) -> None:
        control = self._get_vehicle_control(action)
        self.ego.apply_control(control)

    # def _get_vehicle_control(self, action: int) -> carla.VehicleControl:
    #     acc = DISCRETE_ACC[action // self.n_steer]
    #     steer = DISCRETE_STEER[action % self.n_steer]
    #     throttle = float(acc)
    #     brake = 0.0
    #     return carla.VehicleControl(throttle=throttle, steer=float(steer), brake=brake)

    def _get_vehicle_control(self, action: int) -> carla.VehicleControl:
        long_cmd = float(DISCRETE_LONG[action // self.n_steer])
        steer = float(DISCRETE_STEER[action % self.n_steer])

        if long_cmd >= 0.0:
            throttle = long_cmd
            brake = 0.0
        else:
            throttle = 0.0
            brake = -long_cmd  # convert to positive brake magnitude

        # Clamp for safety
        throttle = float(np.clip(throttle, 0.0, 1.0))
        brake    = float(np.clip(brake,    0.0, 1.0))
        steer    = float(np.clip(steer,   -1.0, 1.0))

        return carla.VehicleControl(throttle=throttle, steer=steer, brake=brake)


    # ----------------------------
    # lane metrics
    # ----------------------------
    def get_vehicle_pos(self, vehicle: carla.Actor) -> Tuple[float, float]:
        loc = vehicle.get_transform().location
        return float(loc.x), float(loc.y)

    def get_vehicle_velocity(self, vehicle: carla.Actor) -> Tuple[float, float]:
        v = vehicle.get_velocity()
        return float(v.x), float(v.y)

    def get_lane_offset(self) -> float:
        vehicle_location = self.ego.get_location()
        waypoint = self.world.get_map().get_waypoint(
            vehicle_location, project_to_road=True, lane_type=carla.LaneType.Driving
        )
        if waypoint is None:
            return 10.0  # defensive: huge offset if projection fails
        lane_center = waypoint.transform.location
        return math.sqrt((vehicle_location.x - lane_center.x) ** 2 + (vehicle_location.y - lane_center.y) ** 2)

    def get_angle_offset(self) -> float:
        vehicle_location = self.ego.get_location()
        waypoint = self.world.get_map().get_waypoint(
            vehicle_location, project_to_road=True, lane_type=carla.LaneType.Driving
        )
        if waypoint is None:
            return 1.0
        wv = waypoint.transform.get_forward_vector()
        ev = self.ego.get_transform().get_forward_vector()
        a = np.array([wv.x, wv.y], np.float32)
        b = np.array([ev.x, ev.y], np.float32)
        denom = (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
        angle = np.arccos(np.clip(float(np.dot(a, b) / denom), -1.0, 1.0))
        return float(angle / np.pi)
    
    # ----------------------------
    # Route-lane (planner-lane) metrics  ✅ important for multi-lane towns
    # ----------------------------

    def _route_lane_anchor_wp(self):
        """
        Waypoint on the lane the planner wants us to be in (uses the next planner waypoint).
        This encodes the *desired lane_id*.
        """
        if (self.waypoints is None) or (len(self.waypoints) == 0):
            return None

        x, y = float(self.waypoints[0][0]), float(self.waypoints[0][1])
        loc = carla.Location(x=x, y=y, z=float(self.ego.get_location().z))
        return self.map.get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Driving)

    def _find_wp_on_lane_at_same_s(self, base_wp, target_lane_id: int, max_hops: int = 8):
        """
        Starting from base_wp (at current vehicle s), walk left/right lanes until we hit target_lane_id.
        This gives a waypoint on the *desired lane* at approximately the same longitudinal position.
        """
        if base_wp is None:
            return None
        if base_wp.lane_id == target_lane_id:
            return base_wp

        visited = {base_wp.lane_id}
        frontier = [base_wp]
        for _ in range(max_hops):
            next_frontier = []
            for wp in frontier:
                for nxt in (wp.get_left_lane(), wp.get_right_lane()):
                    if nxt is None:
                        continue
                    if nxt.lane_type != carla.LaneType.Driving:
                        continue
                    if nxt.road_id != base_wp.road_id:
                        continue
                    if nxt.lane_id in visited:
                        continue
                    if nxt.lane_id == target_lane_id:
                        return nxt
                    visited.add(nxt.lane_id)
                    next_frontier.append(nxt)
            frontier = next_frontier
            if not frontier:
                break
        return None

    def get_lane_offset_to_route_lane(self):
        """
        Lateral distance from vehicle to the centerline of the *route lane* (planner lane).
        Returns:
          lane_offset_m, cur_wp, route_wp, target_wp
        """
        veh_loc = self.ego.get_location()

        # Current waypoint (nearest driving lane at vehicle position)
        cur_wp = self.map.get_waypoint(
            veh_loc, project_to_road=True, lane_type=carla.LaneType.Driving
        )

        # Desired lane anchor (from planner)
        route_wp = self._route_lane_anchor_wp()

        # Fallbacks (don’t crash training if planner is empty momentarily)
        if cur_wp is None or route_wp is None:
            return float(self.get_lane_offset()), cur_wp, route_wp, None

        # If road_id mismatches (junction / transition), avoid false lane-id comparisons.
        # Fall back to “nearest lane” offset; route progress/goal shaping still drives behavior there.
        if cur_wp.road_id != route_wp.road_id:
            return float(self.get_lane_offset()), cur_wp, route_wp, None

        # Find waypoint on desired lane at the same s (via left/right lane hops)
        target_wp = self._find_wp_on_lane_at_same_s(cur_wp, int(route_wp.lane_id), max_hops=8)
        if target_wp is None:
            return float(self.get_lane_offset()), cur_wp, route_wp, None

        c = target_wp.transform.location
        off = math.sqrt((veh_loc.x - c.x) ** 2 + (veh_loc.y - c.y) ** 2)
        return float(off), cur_wp, route_wp, target_wp

    def get_angle_offset_to_wp(self, wp):
        """
        Heading error w.r.t. a given waypoint forward direction (normalized by pi).
        """
        if wp is None:
            return 1.0
        wv = wp.transform.get_forward_vector()
        ev = self.ego.get_transform().get_forward_vector()
        a = np.array([wv.x, wv.y], np.float32)
        b = np.array([ev.x, ev.y], np.float32)
        denom = (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
        angle = np.arccos(np.clip(float(np.dot(a, b) / denom), -1.0, 1.0))
        return float(angle / np.pi)

    


    # ----------------------------
    # reward (your logic, minimally adapted)
    # ----------------------------
    def _compute_reward(self):
        reward_components = {}
        self._lane_invasion_step = 0

        # waypoint reward (NOTE: your original used cumulative num_completed; kept as-is)
        # if self.num_completed > 0:
        #     reward_components["waypoint"] = 60.0 * float(self.num_completed)
        # else:
        #     reward_components["waypoint"] = -2.0

        # delta_completed = int(self.num_completed) - int(self.last_num_completed)
        # reward_components["waypoint"] = 60.0 * float(max(delta_completed, 0))
        # self.last_num_completed = int(self.num_completed)

        # Remove planner-dependent waypoint reward (better for cross-town invariance)
        reward_components["waypoint"] = 0.0
        self.last_num_completed = int(self.num_completed)


        # speed reward based on waypoint direction
        ego_location = np.array(self.get_vehicle_pos(self.ego), np.float32)
        ego_velocity = np.array(self.get_vehicle_velocity(self.ego), np.float32)

        if len(self.waypoints) > 0:
            next_waypoint = self.waypoints[0]
            next_location = np.array([next_waypoint[0], next_waypoint[1]], np.float32)
            yaw_radius = float(next_waypoint[2]) * np.pi / 180.0
            waypoint_direction = np.array([np.cos(yaw_radius), np.sin(yaw_radius)], np.float32)

            goal_offset = next_location - ego_location
            perp = goal_offset - np.dot(goal_offset, waypoint_direction) * waypoint_direction
            pn = np.linalg.norm(perp)
            perp_direction = perp / pn if pn > 0.05 else np.array([0.0, 0.0], np.float32)

            desired_speed = 5.0  # m/s
            speed_parallel = float(np.dot(ego_velocity, waypoint_direction))
            speed_perp = float(np.dot(ego_velocity, perp_direction))
            # reward_components["r_speed"] = (desired_speed - abs(speed_parallel - desired_speed) - 2 * max(speed_perp, -0.5)) * 2.0
            # reward_components["r_speed"] = (desired_speed - abs(speed_parallel - desired_speed) - 2.0 * abs(speed_perp)) * 2.0
            reward_components["r_speed"] = - (abs(speed_parallel - desired_speed) + 2.0 * abs(speed_perp))
        else:
            reward_components["r_speed"] = 0.0

        # reward_components["r_speed"] = 0.0

        # lane/heading shaping
        # lane_d = self.get_lane_offset()
        # ang_d = self.get_angle_offset()

        lane_d, cur_wp, route_wp, target_wp = self.get_lane_offset_to_route_lane()
        ang_d = self.get_angle_offset_to_wp(target_wp if target_wp is not None else cur_wp)

        # ----------------------------
        # Wrong-lane / off-road flags (used for shaping + termination)
        # ----------------------------
        # strict_wp = self.map.get_waypoint(
        #     self.ego.get_location(), project_to_road=False, lane_type=carla.LaneType.Driving
        # )
        # offroad = (strict_wp is None)

        veh_loc = self.ego.get_location()

        # Strict membership check (can flicker near boundaries)
        strict_wp = self.map.get_waypoint(
            veh_loc, project_to_road=False, lane_type=carla.LaneType.Driving
        )

        # Always-available projection (stable), used as a "distance sanity check"
        proj_wp = self.map.get_waypoint(
            veh_loc, project_to_road=True, lane_type=carla.LaneType.Driving
        )

        if proj_wp is None:
            # Very rare; treat as offroad
            offroad = True
            proj_off = 1e9
            lane_width = 3.5
        else:
            c = proj_wp.transform.location
            proj_off = math.hypot(veh_loc.x - c.x, veh_loc.y - c.y)
            lane_width = float(proj_wp.lane_width)

        # Only call it offroad if:
        #   (a) strict says "not in a driving lane", AND
        #   (b) you're meaningfully far from the nearest driving lane centerline
        # This eliminates 1–2 frame boundary flicker.
        outside_lane_band = proj_off > (0.5 * lane_width + 0.30)  # 30 cm margin
        offroad = (strict_wp is None) and outside_lane_band



        # wrong_lane = False
        # if (not offroad) and (cur_wp is not None) and (route_wp is not None) and (cur_wp.road_id == route_wp.road_id):
        #     wrong_lane = (cur_wp.lane_id != route_wp.lane_id)

        wrong_lane = False

        # Ignore wrong-lane logic in junctions (lane ids / topology are unstable there)
        in_junction = False
        if cur_wp is not None and getattr(cur_wp, "is_junction", False):
            in_junction = True
        if route_wp is not None and getattr(route_wp, "is_junction", False):
            in_junction = True
        if target_wp is not None and getattr(target_wp, "is_junction", False):
            in_junction = True

        if (not offroad) and (not in_junction) and (cur_wp is not None) and (route_wp is not None) and (cur_wp.road_id == route_wp.road_id):
            lane_mismatch = (cur_wp.lane_id != route_wp.lane_id)

            # Use desired lane width when available
            ref_wp = target_wp if target_wp is not None else (route_wp if route_wp is not None else cur_wp)
            ref_lane_width = float(getattr(ref_wp, "lane_width", 3.5))

            # Only flag wrong-lane if you're *clearly* off the route lane centerline
            # Adjacent lane centerline is ~1 lane_width away; threshold ~0.55*lane_width filters jitter.
            far_from_route_center = (lane_d > (0.55 * ref_lane_width))

            wrong_lane = bool(lane_mismatch and far_from_route_center)


        # Save for step()/termination
        self._offroad = bool(offroad)
        self._wrong_lane = bool(wrong_lane)

        # Shaping penalties (tune later; these are sane starting values)
        reward_components["offroad"] = -20.0 * float(offroad)
        reward_components["wrong_lane"] = -5.0 * float(wrong_lane)

        lane_k = 40.0
        lane_rad = min(lane_d, 2.0)
        # reward_components["lane"] = -lane_k * (lane_rad ** 2)

        head_k = 100.0
        reward_components["heading"] = -head_k * (min(ang_d, 0.5) ** 2)

        # Step-based low-speed penalty
        if self.low_speed_steps > 0:
            if self.low_speed_steps >= self.low_speed_timeout_steps:
                reward_components["low_speed"] = -50.0
            else:
                reward_components["low_speed"] = -2.0
        else:
            reward_components["low_speed"] = 0.0


        # lane invasion penalty (new invasions only)
        new_inv = len(self.lane_invasion_hist) - self.previous_lane_invasions
        reward_components["invasion"] = -20.0 * float(new_inv)
        self.previous_lane_invasions += int(new_inv)
        self._lane_invasion_step = int(new_inv)

        # collision penalty (new collisions only)
        new_col = len(self.collision_hist) - self.previous_collisions
        reward_components["collision"] = -500.0 * float(new_col)
        self.previous_collisions += int(new_col)

        # NEW: per-step collision event count (0/1 usually)
        self._collision_step = int(new_col)

        # progress / goal
        # dist_goal = float(self.ego.get_location().distance(self.end_point.location))
        # distance_diff = float(self.last_distance_to_goal - dist_goal) if self.last_distance_to_goal is not None else 0.0

        # if dist_goal < 10.0:
        #     reward_components["goal"] = 200.0
        # else:
        #     # reward_components["goal"] = max(0.0, distance_diff) * 60.0
        #     reward_components["goal"] = float(distance_diff) * 60.0

        # reward_components["time"] = -0.05


        # self.last_distance_to_goal = dist_goal

        # Goal bonus only (no Euclidean shaping)
        dist_goal = float(self.ego.get_location().distance(self.end_point.location))
        reward_components["goal"] = 200.0 if dist_goal < 10.0 else 0.0

        # Route-aware progress (dense, less town-specific than num_completed / Euclidean diff)
        cur_rem = self._route_remaining_m()
        if self._prev_route_remaining_m is None:
            d_rem = 0.0
        else:
            d_rem = float(self._prev_route_remaining_m - cur_rem)

        # Clip for stability (planner can re-route around junctions)
        d_rem = float(np.clip(d_rem, -0.5, 0.5))

        k_route = float(_get_cfg(self._config, "k_route_progress", 50.0))
        reward_components["route_progress"] = k_route * d_rem
        self._prev_route_remaining_m = cur_rem

        # Small time penalty to encourage finishing (prevents loitering)
        time_pen = float(_get_cfg(self._config, "time_penalty", 0.05))
        reward_components["time"] = -time_pen

        # Keep this updated for your logging (progress_m in step)
        self.last_distance_to_goal = dist_goal


        total = float(sum(reward_components.values()))
        return total, reward_components

    # ----------------------------
    # termination (adapted to per-episode start point)
    # ----------------------------
    def _check_termination(self):
        collision = bool(self.collision_detected)
        reached_goal = (self.ego.get_location().distance(self.end_point.location) < 10.0)
        time_exceeded = (self._time_step >= self._max_time_step)
        offroad_done = (self.offroad_steps >= self.max_offroad_steps)
        wrong_lane_done = (self.wrong_lane_steps >= self.max_wrong_lane_steps)

        LOW_SPEED_TIMEOUT_STEPS = self.low_speed_timeout_steps
        stuck_too_long = (self.low_speed_steps >= LOW_SPEED_TIMEOUT_STEPS)

        # past_goal using current start point, NOT global arrays
        # past_goal = (self._dist_regret_steps >= 30)

        goal_radius_m = float(_get_cfg(self._config, "goal_radius_m", 10.0))  # must match reached_goal threshold
        past_goal_margin_m = float(_get_cfg(self._config, "past_goal_margin_m", 1.0))
        past_goal_timeout_s = float(_get_cfg(self._config, "past_goal_timeout_s", 1.0))
        past_goal_steps_needed = int(round(past_goal_timeout_s / float(self._fixed_dt)))

        ahead_m = self._signed_ahead_to_goal_m()

        # Are we geometrically past the goal and NOT already "success"?
        past_now = (
            bool(getattr(self, "_ever_near_goal", False)) and
            (ahead_m > past_goal_margin_m) and
            (self._dist_to_goal_m is not None) and
            (self._dist_to_goal_m > goal_radius_m)
        )

        # Stability: require that we're actually moving away for a short duration
        moving_away = False
        eps = float(_get_cfg(self._config, "past_goal_moving_away_eps_m", 0.05))
        if (self._prev_dist_to_goal_m is not None) and (self._dist_to_goal_m is not None):
            moving_away = (self._dist_to_goal_m > self._prev_dist_to_goal_m + eps)

        if past_now and moving_away:
            self._past_goal_steps += 1
        else:
            self._past_goal_steps = 0

        past_goal = (self._past_goal_steps >= past_goal_steps_needed)


        # info = {}

        info = {
            "collision": 0,
            "goal_reached": 0,
            "offroad": 0,
            "wrong_lane": 0,
            "past_goal": 0,
            "not_moving": 0,
            "time_exceeded": 0,
            "stuck_duration": 0.0,   # ✅ always present
            "elapsed_steps": 0,      # ✅ always present
        }
        done = False

        if collision:
            done = True
            info["collision"] = True
        elif reached_goal:
            done = True
            info["goal_reached"] = True
        elif offroad_done:
            done = True
            info["offroad"] = True
        elif wrong_lane_done:
            done = True
            info["wrong_lane"] = True
        elif past_goal:
            done = True
            info["past_goal"] = True
        elif stuck_too_long:
            done = True
            info["not_moving"] = True
            info["stuck_duration"] = float(self.low_speed_steps) * float(self._fixed_dt)
        elif time_exceeded:
            done = True
            info["time_exceeded"] = True
            info["elapsed_steps"] = int(self._time_step)

        # episode-level metrics
        info["distance_m"] = float(self.travel_distance_m)
        info["distance_km"] = float(self.travel_distance_m) / 1000.0
        info["lane_invasions"] = int(self.previous_lane_invasions)
        info["mean_off_center_m"] = float(self.off_center_sum / self.off_center_steps) if self.off_center_steps > 0 else 0.0

        return done, info

    # ----------------------------
    # cleanup
    # ----------------------------
    def _clean_actors(self):
        try:
            for s in ("camera_sensor", "colsensor", "lane_sensor"):
                sensor = getattr(self, s, None)
                if sensor is not None:
                    try:
                        sensor.stop()
                    except Exception:
                        pass

            batch = []
            for a in list(self.actors):
                if a is not None and a.is_alive:
                    batch.append(carla.command.DestroyActor(a))
            if batch:
                self.client.apply_batch_sync(batch, True)

            self.actors.clear()
            self.ego = None
            self.camera_sensor = None
            self.colsensor = None
            self.lane_sensor = None
            self._img_q = None

            if getattr(self, "_sync_enabled", False):
                self.world.tick()
        except Exception as e:
            print(f"[cleanup] error: {e}")



