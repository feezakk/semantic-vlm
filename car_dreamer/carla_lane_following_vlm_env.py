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


# ----------------------------
# Routes (TRAIN)
# ----------------------------
TRAIN_EGO_SPAWN_POINT = [
    [71.929916, -6.996726, 1.050581, 0.953298,  -62.641895,   0.000000],
    [59.233459, -68.268501, 5.817840, 4.703701,  -81.431274,   0.000000],
    [67.262825, -114.866020, 9.692015, 4.703701, -85.152115,   0.000000],
    [44.566227, -177.097443, 6.951100, -4.487109, -93.109505,  0.000000],
]

TRAIN_EGO_END_POINT = [
    [59.233459, -68.268501, 5.817840, 4.703701,  -81.431274,   0.000000],
    [67.262825, -114.866020, 9.692015, 4.703701, -85.152115,   0.000000],
    [44.566227, -177.097443, 6.951100, -4.487109, -93.109505,  0.000000],
    [6.476024,  -241.105026, 1.000000, 2.601381, -192.499680,  0.000000],
]

# TRAIN_ROUTES = list(zip(TRAIN_EGO_SPAWN_POINT, TRAIN_EGO_END_POINT))


# ----------------------------
# Routes (EVAL)
# ----------------------------
EVAL_EGO_SPAWN_POINT = [
    # one-turn scenarios
    [74.51,   7.51,   1.01, 0.953298,   45,   0.0],
    [68.92,  72.00,   1.01, 4.703701, -280,  0.0],
    [-90.68, 121.28,  1.01, 0.0,        0,   0.0],
    # [-203.12, 64.82,  1.01, 0.0,       90,   0.0],

    # straight scenarios
    [-15.60, -161.00, 1.00, 0.0,      180,   0.0],
    [-185.95, -159.74,1.00, 0.0,        0,   0.0],
    [-19.18,  57.56,  1.00, 0.0,      180,   0.0],
    [-86.19,  57.56,  1.00, 0.0,        0,   0.0],

    # multiple-turn scenarios
    [-23.86, -245.72, 1.00, 0.0,      200,   0.0],
    # [-27.81, -95.85,  1.00, 0.0,      270,   0.0],
]

EVAL_EGO_END_POINT = [
    # one-turn scenarios
    [75.51,  51.44,  1.01, 0.953298, 140, 0.0],
    [-90.68, 118.28, 1.01, 0.0,      180, 0.0],
    [72.92,  72.00,  1.01, 4.703701, -100,0.0],
    # [-190.41,110.60, 1.00, 0.0,        0, 0.0],

    # straight scenarios
    [-58.60, -161.00,1.00, 0.0,      180, 0.0],
    [-24.95, -158.74,1.00, 0.0,        0, 0.0],
    [-82.18,  53.56, 1.00, 0.0,      180, 0.0],
    [-22.19,  60.56, 1.00, 0.0,        0, 0.0],

    # multiple-turn scenarios
    [-169.98,-248.17,1.00, 0.0,      180, 0.0],
    # [-63.37,  -95.85,1.00, 0.0,      270, 0.0],
]

# EVAL_ROUTES = list(zip(EVAL_EGO_SPAWN_POINT, EVAL_EGO_END_POINT))

# ----------------------------
# Town06 (TRAIN routes)
# ----------------------------
TOWN06_TRAIN_SPAWN = [
    [-262.47, -14.12, 1.0, 0, 180, 0],
    [-215.71, 150.09, 1.0, 0, 160, 0],
    [548.55, 244.84, 1.0, 0, 0, 0],
    [659.80, 28.50, 1.0, 0, 250, 0],
    [-230.09, 139.45, 1.0, 0, 180, 0],
    [-317.50, 95.52, 1.0, 0, 270, 0],
    [146.45, 247.71, 1.0, 0, 0, 0],
    [298.46, -20.31, 1.0, 0, 180, 0],
]

TOWN06_TRAIN_END = [
    [-313.80, 243.80, 1.0, 0, 0, 0],
    [-216.77, 235.38, 1.0, 0, 10, 0],
    [665.86, 195.78, 1.0, 0, 270, 0],
    [580.75, -16.72, 1.0, 0, 180, 0],
    [-317.50, 95.52, 1.0, 0, 270, 0],
    [-230.76, 49.86, 1.0, 0, 0, 0],
    [335.78, 247.71, 1.0, 0, 0, 0],
    [154.26, -20.31, 1.0, 0, 180, 0],
]

TRAIN_ROUTES = list(zip(TOWN06_TRAIN_SPAWN, TOWN06_TRAIN_END))


# ----------------------------
# Town04 (EVAL routes)
# ----------------------------
TOWN04_EVAL_SPAWN = [
    [-405.62, 10.02, 1.0, 0, 180, 0],
    [-39.65, -226.78, 1.0, 0, 120, 0],
    [-220.49, 431.52, 1.0, 0, 0, 0],
    [268.52, 38.00, 15.0, 0, 0, 0],
    [255.83, -367.66, 1.0, 0, 0, 0],
    [10.94, -210.57, 1.0, 0, -90, 0],
    [-350.19, 33.68, 15.0, 0, 0, 0],
    [-124.33, 9.56, 15.0, 0, 180, 0],
]

TOWN04_EVAL_END = [
    [-338.83, 431.15, 1.0, 0, 0, 0],
    [-379.22, -19.55, 1.0, 0, 100, 0],
    [-42.42, 364.21, 1.0, 0, -45, 0],
    [395.24, 12.82, 15.0, 0, -45, 0],
    [385.96, -243.17, 1.0, 0, 90, 0],
    [111.44, -360.14, 1.0, 0, -15, 0],
    [128.22, 34.38, 15.0, 0, 0, 0],
    [-340.33, 9.56, 15.0, 0, 180, 0],
]

EVAL_ROUTES = list(zip(TOWN04_EVAL_SPAWN, TOWN04_EVAL_END))



# ----------------------------
# Discrete actions
# ----------------------------
DISCRETE_ACC = [0.0, 0.3]
DISCRETE_STEER = [-0.2, -0.1, 0.0, 0.1, 0.2]


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

# Train: Noon variants (one per domain)
DEFAULT_TRAIN_WEATHER_IDS = [0, 1, 2, 4]           # clearNoon, cloudyNoon, wetNoon, softRainNoon

# Eval: Sunset / harder variants (one per domain)
DEFAULT_EVAL_WEATHER_IDS  = [7, 8, 10, 12, 13]     # clearSunset, cloudySunset, wetCloudySunset, midRainSunset, hardRainSunset


class CarlaLaneFollowingVlmEnv(gym.Env):
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
        assert len(weather_ids) > 0, "weather_ids must be non-empty"

        self._config = config
        self.routes = routes
        self.weather_ids = list(weather_ids)

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

        target_town = _get_cfg(config, "env.town", _get_cfg(config, "town", "Town07"))
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
        self.render_enabled = bool(_get_cfg(config, "render", False))

        self.include_vlm = bool(_get_cfg(config, "include_vlm", True))
        self.include_domain_id = bool(_get_cfg(config, "include_domain_id", True))

        # VLM
        self.vlm_dim = int(_get_cfg(config, "vlm_dim", 512))
        self.compute_vlm = bool(_get_cfg(config, "compute_vlm", True))
        self._vlm = None
        self._vlm_preprocess = None
        self._vlm_device = str(_get_cfg(config, "vlm_device", "cpu"))
        
        # domains: fixed shared domain labels (0..3) from _weather_to_domain
        self.num_domains = int(_get_cfg(config, "num_domains", 4))

        self.DOMAIN_NAMES = ("clear", "cloudy", "wet", "rain")

        # Build a mapping domain -> list of weather_ids in that domain
        self._domain_to_weather = {d: [] for d in range(self.num_domains)}
        for wid in self.weather_ids:
            d = int(self._weather_to_domain(wid))
            self._domain_to_weather[d].append(int(wid))

        # Sanity check: every domain should be present in training, or sampling will be biased.
        if not self.eval_mode:
            missing = [d for d, ws in self._domain_to_weather.items() if len(ws) == 0]
            if missing:
                raise ValueError(f"Training weather_ids do not cover domains {missing}. "
                                f"Either add weathers or reduce num_domains.")



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

        # start/end transforms (per episode)
        self.ego_transform = None
        self.end_point = None
        self._start_xy = None  # for past_goal logic

        # initial/last dist to goal
        self.initial_distance_to_goal = None
        self.last_distance_to_goal = None

        # pygame for rendering
        default_render = True if self.eval_mode else False
        self.render_enabled = bool(getattr(config, "render", default_render))  # set True for eval
        self.render_size = int(getattr(config, "render_size", 512))
        self.render_fps_cap = int(getattr(config, "render_fps_cap", 20))  # cap UI updates
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
        self.max_wrong_lane_steps = int(_get_cfg(config, "max_wrong_lane_steps", 5))
        self.max_offroad_steps = int(_get_cfg(config, "max_offroad_steps", 2))

        self.wrong_lane_steps = 0
        self.offroad_steps = 0

        # last-step flags (set every step in _compute_reward)
        self._wrong_lane = False
        self._offroad = False

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

    def _pygame_init(self):
        if self._pg_initialized:
            return
        pygame.init()
        pygame.display.set_caption("CARLA Eval Viewer")
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

        # Overlay text
        if lines:
            y = 8
            for s in lines[:12]:
                text = self._pg_font.render(str(s), True, (255, 255, 255))
                # simple shadow for readability
                shadow = self._pg_font.render(str(s), True, (0, 0, 0))
                self._pg_screen.blit(shadow, (9, y + 1))
                self._pg_screen.blit(text, (8, y))
                y += 20

        pygame.display.flip()


    # ----------------------------
    # queues
    # ----------------------------
    def _refill_route_queue(self):
        idxs = list(range(len(self.routes)))
        if not self.eval_mode:
            random.shuffle(idxs)
        self._route_queue = deque(idxs)

    def _refill_combo_queue(self):
        """
        Deterministic eval schedule, but DOMAIN-BALANCED.
        Each coarse domain contributes equally by upsampling weather_ids within the domain.
        """
        dom_to_wids = {d: [] for d in range(self.num_domains)}
        for wid in self.weather_ids:
            d = int(self._weather_to_domain(wid))
            if d < 0 or d >= self.num_domains:
                raise ValueError(f"weather_id {wid} mapped to domain {d}, num_domains={self.num_domains}")
            dom_to_wids[d].append(int(wid))

        missing = [d for d, ws in dom_to_wids.items() if len(ws) == 0]
        if missing:
            raise ValueError(
                f"Eval weather_ids do not cover domains {missing}. "
                f"Fix eval_weather_ids / DEFAULT_EVAL_WEATHER_IDS."
            )

        # Upsample each domain list to the same length
        m = max(len(ws) for ws in dom_to_wids.values())
        balanced_wids = []
        for d in range(self.num_domains):
            ws = dom_to_wids[d]
            reps = (m + len(ws) - 1) // len(ws)
            balanced_wids.extend((ws * reps)[:m])  # exactly m per domain

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

        valid_domains = [d for d, ws in self._domain_to_weather.items() if len(ws) > 0]
        dom = int(self._rng.choice(valid_domains))
        wid = int(self._rng.choice(self._domain_to_weather[dom]))
        return rid, wid


    # ----------------------------
    # spaces
    # ----------------------------
    def _setup_action_space(self):
        self.n_steer = len(DISCRETE_STEER)
        self.n_acc = len(DISCRETE_ACC)
        return spaces.Discrete(self.n_steer * self.n_acc)

    def _setup_observation_space(self):
        camera_space = spaces.Box(0, 255, shape=(128, 128, 3), dtype=np.uint8)
        collision_space = spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32)
        lane_invasion_space = spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32)
        vlm_space = spaces.Box(-np.inf, np.inf, shape=(self.vlm_dim,), dtype=np.float32)
        domain_space = spaces.Box(0.0, float(self.num_domains - 1), shape=(1,), dtype=np.float32)
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
            obs["vlm"] = self._encode_vlm(rgb).astype(np.float32)

        if self.include_domain_id:
            obs["domain_id"] = np.array([float(self.domain_id)], np.float32)

        return obs

    # ----------------------------
    # vehicle spawning
    # ----------------------------
    def reset_vehicle(self):
        bp = self.blueprint_library.find("vehicle.tesla.model3")
        ego = self.world.try_spawn_actor(bp, self.ego_transform)
        if ego is None:
            raise RuntimeError("Unable to spawn ego vehicle at the chosen spawn point.")
        self.ego = ego
        self.actors.append(self.ego)
        self.ego.set_target_velocity(carla.Vector3D())
        self.ego.apply_control(carla.VehicleControl(
            manual_gear_shift=False, reverse=False, hand_brake=False,
            steer=0.0, throttle=0.0, brake=0.0
        ))

    # ----------------------------
    # gym API
    # ----------------------------
    def reset(self):
        self._clean_actors()

        # choose (route, weather)
        self.route_id, self.weather_id = self._select_route_and_weather()
        self.weather_name, weather_params = WEATHER_TABLE[self.weather_id]
        self.world.set_weather(weather_params)
        self.domain_id = int(self._weather_to_domain(self.weather_id))

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

        # distance to goal init
        self.initial_distance_to_goal = self.ego.get_location().distance(self.end_point.location)
        self.last_distance_to_goal = float(self.initial_distance_to_goal)

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
        prev_dist = float(self.last_distance_to_goal) if self.last_distance_to_goal is not None else dist_to_goal
        progress_m = max(0.0, prev_dist - dist_to_goal)

        # reward + info
        reward, info_dict = self._compute_reward()

        # ----------------------------
        # Persistency counters (for termination)
        # ----------------------------
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

        acc = float(DISCRETE_ACC[action // self.n_steer])
        steer = float(DISCRETE_STEER[action % self.n_steer])

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
            "throttle_cmd": acc,
            "steer_cmd": steer,
            "dist_to_goal_m": dist_to_goal,
            "progress_m": float(progress_m),
            "collision_step": int(getattr(self, "_collision_step", 0)),
            "domain_id": int(self.domain_id),
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
            lines = [
                f"route: {info.get('route_id', 'NA')}",
                f"weather: {info.get('weather_name', info.get('weather_id', 'NA'))}",
                f"speed_kmh: {info.get('speed_kmh', 0.0):.1f}",
                f"reward: {reward:.2f}",
                f"off_center_m: {info.get('off_center_m', 0.0):.2f}",
                f"collision: {int(info.get('collision', False))}",
                f"lane_inv_step: {int(info.get('lane_invasion_step', 0))}",
            ]
            self._pygame_render(self.camera_image2, lines=lines)

        return obs, float(reward), bool(done), info

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

    def _get_vehicle_control(self, action: int) -> carla.VehicleControl:
        acc = DISCRETE_ACC[action // self.n_steer]
        steer = DISCRETE_STEER[action % self.n_steer]
        throttle = float(acc)
        brake = 0.0
        return carla.VehicleControl(throttle=throttle, steer=float(steer), brake=brake)

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

        delta_completed = int(self.num_completed) - int(self.last_num_completed)
        reward_components["waypoint"] = 60.0 * float(max(delta_completed, 0))
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
            reward_components["r_speed"] = (desired_speed - abs(speed_parallel - desired_speed) - 2 * max(speed_perp, -0.5)) * 2.0
        else:
            reward_components["r_speed"] = 0.0

        # lane/heading shaping
        # lane_d = self.get_lane_offset()
        # ang_d = self.get_angle_offset()

        lane_d, cur_wp, route_wp, target_wp = self.get_lane_offset_to_route_lane()
        ang_d = self.get_angle_offset_to_wp(target_wp if target_wp is not None else cur_wp)

        # ----------------------------
        # Wrong-lane / off-road flags (used for shaping + termination)
        # ----------------------------
        strict_wp = self.map.get_waypoint(
            self.ego.get_location(), project_to_road=False, lane_type=carla.LaneType.Driving
        )
        offroad = (strict_wp is None)

        wrong_lane = False
        if (not offroad) and (cur_wp is not None) and (route_wp is not None) and (cur_wp.road_id == route_wp.road_id):
            wrong_lane = (cur_wp.lane_id != route_wp.lane_id)

        # Save for step()/termination
        self._offroad = bool(offroad)
        self._wrong_lane = bool(wrong_lane)

        # Shaping penalties (tune later; these are sane starting values)
        reward_components["offroad"] = -200.0 * float(offroad)
        reward_components["wrong_lane"] = -50.0 * float(wrong_lane)

        lane_k = 40.0
        lane_rad = min(lane_d, 2.0)
        reward_components["lane"] = -lane_k * (lane_rad ** 2)

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
        dist_goal = float(self.ego.get_location().distance(self.end_point.location))
        distance_diff = float(self.last_distance_to_goal - dist_goal) if self.last_distance_to_goal is not None else 0.0

        if dist_goal < 2.0:
            reward_components["goal"] = 200.0
        else:
            reward_components["goal"] = max(0.0, distance_diff) * 60.0

        self.last_distance_to_goal = dist_goal

        total = float(sum(reward_components.values()))
        return total, reward_components

    # ----------------------------
    # termination (adapted to per-episode start point)
    # ----------------------------
    def _check_termination(self):
        collision = bool(self.collision_detected)
        reached_goal = (self.ego.get_location().distance(self.end_point.location) < 2.0)
        time_exceeded = (self._time_step >= self._max_time_step)
        offroad_done = (self.offroad_steps >= self.max_offroad_steps)
        wrong_lane_done = (self.wrong_lane_steps >= self.max_wrong_lane_steps)


        LOW_SPEED_TIMEOUT_STEPS = self.low_speed_timeout_steps
        stuck_too_long = (self.low_speed_steps >= LOW_SPEED_TIMEOUT_STEPS)

        # past_goal using current start point, NOT global arrays
        past_goal = (self._dist_regret_steps >= 30)

        info = {}
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



