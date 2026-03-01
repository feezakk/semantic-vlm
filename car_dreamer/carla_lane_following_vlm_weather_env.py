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

TRAIN_ROUTES = list(zip(TRAIN_EGO_SPAWN_POINT, TRAIN_EGO_END_POINT))

EVAL_EGO_SPAWN_POINT = [
    [74.51,   7.51,   1.01, 0.953298,   45,   0.0],
    [68.92,  72.00,   1.01, 4.703701, -280,  0.0],
    [-90.68, 121.28,  1.01, 0.0,        0,   0.0],

    [-15.60, -161.00, 1.00, 0.0,      180,   0.0],
    [-185.95, -159.74,1.00, 0.0,        0,   0.0],
    [-19.18,  57.56,  1.00, 0.0,      180,   0.0],
    [-86.19,  57.56,  1.00, 0.0,        0,   0.0],

    [-23.86, -245.72, 1.00, 0.0,      200,   0.0],
]

EVAL_EGO_END_POINT = [
    [75.51,  51.44,  1.01, 0.953298, 140, 0.0],
    [-90.68, 118.28, 1.01, 0.0,      180, 0.0],
    [72.92,  72.00,  1.01, 4.703701, -100,0.0],

    [-58.60, -161.00,1.00, 0.0,      180, 0.0],
    [-24.95, -158.74,1.00, 0.0,        0, 0.0],
    [-82.18,  53.56, 1.00, 0.0,      180, 0.0],
    [-22.19,  60.56, 1.00, 0.0,        0, 0.0],

    [-169.98,-248.17,1.00, 0.0,      180, 0.0],
]

EVAL_ROUTES = list(zip(EVAL_EGO_SPAWN_POINT, EVAL_EGO_END_POINT))

DISCRETE_LONG =  [-1.0, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0,
                  0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]  
DISCRETE_STEER = [-0.5, -0.25, -0.1, 0.0, 0.1, 0.25, 0.5]

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

DEFAULT_TRAIN_WEATHER_IDS = [0, 1, 2, 4] 
DEFAULT_EVAL_WEATHER_IDS  = [6, 7, 8, 10, 13] 

class CarlaLaneFollowingVlmWeatherEnv(gym.Env):
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
            f"weather_ids must be non-empty (got {use_weather_ids} from {key_plain})."
        )

        self._config = config
        self.routes = routes
        self.weather_ids = list(use_weather_ids)

        self.eval_mode = bool(eval_mode)
        self.deterministic_eval = bool(deterministic_eval)
        host = _get_cfg(config, "world.carla_host", "localhost")
        port = int(_get_cfg(config, "world.carla_port", _get_cfg(config, "carla_port", 3000)))
        self.client = carla.Client(host, port)
        self.client.set_timeout(float(_get_cfg(config, "world.timeout_s", 300.0)))

        w = self.client.get_world()
        try:
            cur_map = w.get_map().name
        except RuntimeError:
            cur_map = ""

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

        self.blueprint_library = self.world.get_blueprint_library()

        self.ego = None
        self.actors = []

        self._time_step = 0
        self._max_time_step = int(_get_cfg(config, "time_limit_steps", 2000))

        self.collision_detected = False
        self.collision_hist = []
        self.lane_invasion_detected = False
        self.lane_invasion_hist = []
        self.previous_lane_invasions = 0
        self.previous_collisions = 0
        self._lane_invasion_step = 0

        self.render_enabled = bool(_get_cfg(config, "render", True))

        self.include_vlm = bool(_get_cfg(config, "include_vlm", True))
        self.include_domain_id = bool(_get_cfg(config, "include_domain_id", True))

        self.domain_kind = str(_get_cfg(config, "domain_kind", _get_cfg(config, "env.domain_kind", "weather"))).lower()
        if self.domain_kind != "weather":
            raise ValueError(
                f"This environment is weather-domain only. Got domain_kind='{self.domain_kind}'. "
                "Set --env.domain_kind weather and remove any town-domain flags."
            )

        self.DOMAIN_NAMES = ("clear", "cloudy", "wet", "rain")
        self.num_domains = len(self.DOMAIN_NAMES) 

        self.vlm_dim = int(_get_cfg(config, "vlm_dim", 512))
        self.compute_vlm = bool(_get_cfg(config, "compute_vlm", True))
        self._vlm = None
        self._vlm_preprocess = None
        self._vlm_device = str(_get_cfg(config, "vlm_device", "cpu"))

        print(
            f"[Domain] num_domains={self.num_domains} {self.DOMAIN_NAMES} | "
            f"weather_ids={self.weather_ids} eval_mode={self.eval_mode}"
        )

        self._weather_domain_to_weather = {d: [] for d in range(self.num_domains)}  

        for wid in self.weather_ids:
            d = int(self._weather_to_domain(wid))  
            if d < 0 or d >= self.num_domains:
                raise ValueError(f"weather_id {wid} mapped to domain {d}, num_domains={self.num_domains}")
            self._weather_domain_to_weather[d].append(int(wid))

        if not self.eval_mode:
            missing = [d for d, ws in self._weather_domain_to_weather.items() if len(ws) == 0]
            if missing:
                raise ValueError(
                    f"Training weather_ids do not cover weather buckets {missing}. "
                    f"Either add weathers or adjust DEFAULT_TRAIN_WEATHER_IDS."
                )

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

        self.action_space = self._setup_action_space()
        self.observation_space = self._setup_observation_space()

        self._route_queue = deque()
        self._combo_queue = deque()
        self._refill_route_queue()
        if self.eval_mode and self.deterministic_eval:
            self._refill_combo_queue()

        self.camera_image = None
        self.camera_image2 = None
        self._img_q = None
        self._last_tick_frame = None

        self.ego_planner = None
        self.waypoints = []
        self.planner_stats = {"num_completed": 0}
        self.num_completed = 0
        self.last_num_completed = 0

        self.speed_kmh = 0.0

        self.prev_ego_location = None
        self.travel_distance_m = 0.0
        self.off_center_sum = 0.0
        self.off_center_steps = 0

        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.episode_id = 0
        self.timestep = 0

        self.route_id = 0
        self.weather_id = 0
        self.weather_name = "ClearNoon"
        self.domain_id = 0
        self.domain_mask = 1.0

        self.ego_transform = None
        self.end_point = None
        self._start_xy = None  
        self.initial_distance_to_goal = None
        self.last_distance_to_goal = None

        self.render_enabled = bool(_get_cfg(config, "render", True))
        self.render_size = int(_get_cfg(config, "render_size", 512))
        self.render_fps_cap = int(_get_cfg(config, "render_fps_cap", 20))  

        town_short = self.map.name.split("/")[-1]  
        host = _get_cfg(config, "world.carla_host", "localhost")
        port = int(_get_cfg(config, "world.carla_port", _get_cfg(config, "carla_port", 3000)))

        self.render_window_title = str(_get_cfg(
            config,
            "render_window_title",
            f"TRAIN | {town_short} | {host}:{port}"
        ))

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

        self._dist_regret_steps = 0

        self.wrong_lane_steps = 0
        self.offroad_steps = 0

        self._wrong_lane = False
        self._offroad = False

        self.lane_grace_s = float(_get_cfg(config, "lane_grace_s", 1.0 if not self.eval_mode else 1.0))
        self._lane_grace_steps = int(round(self.lane_grace_s / self._fixed_dt))

        self.max_wrong_lane_s = float(_get_cfg(config, "max_wrong_lane_s", 3.0 if not self.eval_mode else 3.0))
        self.max_offroad_s    = float(_get_cfg(config, "max_offroad_s", 1.0 if not self.eval_mode else 1.0))

        self.max_wrong_lane_steps = int(round(self.max_wrong_lane_s / self._fixed_dt))
        self.max_offroad_steps    = int(round(self.max_offroad_s / self._fixed_dt))

        self.wrong_lane_steps = 0
        self.offroad_steps = 0

        self._wrong_lane = False
        self._offroad = False

        self._ever_near_goal = False
        self._past_goal_steps = 0
        self._dist_to_goal_m = None
        self._prev_dist_to_goal_m = None

        self._prev_route_remaining_m = None
        self.ref_step_m = float(_get_cfg(config, "ref_step_m", 0.5))          
        self.ref_max_m = float(_get_cfg(config, "ref_max_m", 1500.0))         
        self.ref_goal_radius_m = float(_get_cfg(config, "ref_goal_radius_m", 3.0))

        self._ref_xy = None               
        self._ref_seg_len = None         
        self._ref_cum_s = None            
        self._ref_total_len = 0.0
        self._ref_seg_idx = 0            
        self._ref_s = 0.0
        self._prev_ref_s = 0.0

        self._last_lane_d = 0.0
        self._last_heading_err = 0.0
        self.k_route_progress = float(_get_cfg(config, "k_route_progress", 50.0))

        self.k_lane_center = float(_get_cfg(config, "k_lane_center", 10.0))
        self.lane_center_cap_m = float(_get_cfg(config, "lane_center_cap_m", 1.5))

        self.lane_change_threshold_m = float(_get_cfg(config, "lane_change_threshold_m", 1.8))
        self.k_wrong_lane = float(_get_cfg(config, "k_wrong_lane", 20.0))

        self.k_heading = float(_get_cfg(config, "k_heading", 50.0))
        self.heading_cap = float(_get_cfg(config, "heading_cap", 0.5))

        self.desired_speed_mps = float(_get_cfg(config, "desired_speed_mps", 5.0))
        self.k_speed_par = float(_get_cfg(config, "k_speed_par", 1.0))
        self.k_speed_perp = float(_get_cfg(config, "k_speed_perp", 2.0))
        self.reverse_penalty = float(_get_cfg(config, "reverse_penalty", 2.0))

        self.k_offroad = float(_get_cfg(config, "k_offroad", 20.0))
        self.offroad_center_thresh_m = float(_get_cfg(config, "offroad_center_thresh_m", 2.2))

        self.k_invasion = float(_get_cfg(config, "k_invasion", 5.0))         
        self.k_collision = float(_get_cfg(config, "k_collision", 500.0))

        self.goal_bonus = float(_get_cfg(config, "goal_bonus", 200.0))
        self.spawn_lane_id = None
        self.spawn_road_id = None
        self.spawn_section_id = None
        self.spawn_lane_type = None

        self.cur_lane_id = None
        self.cur_road_id = None
        self.cur_section_id = None
        self.cur_is_junction = None
        self.cur_lane_type = None
        self.render_draw_route = bool(_get_cfg(config, "render_draw_route", True))
        self.render_route_max_pts = int(_get_cfg(config, "render_route_max_pts", 60))
        self.render_route_thickness = int(_get_cfg(config, "render_route_thickness", 2))

        self._camera_fov_deg = float(_get_cfg(config, "camera_fov_deg", 110.0))

    def _lane_id_at_world(self, loc: carla.Location) -> Optional[int]:
        wp = self.map.get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Driving)
        if wp is None:
            return None
        return int(wp.lane_id)

    def _put_text_shadow(self, img, xy, text, color, scale=0.45, thickness=1):
        h, w = img.shape[:2]
        x, y = int(xy[0]), int(xy[1])

        x = max(0, min(w - 1, x))
        y = max(0, min(h - 1, y))

        ox, oy = 6, -6
        tx, ty = x + ox, y + oy
        if ty < 12:
            ty = y + 16

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, text, (tx, ty), font, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
        cv2.putText(img, text, (tx, ty), font, scale, color, thickness, cv2.LINE_AA)

    def _get_lane_info(self, loc: carla.Location, project_to_road: bool):
        wp = self.map.get_waypoint(loc, project_to_road=project_to_road, lane_type=carla.LaneType.Driving)
        if wp is None:
            return None

        c = wp.transform.location
        off = float(math.hypot(float(loc.x - c.x), float(loc.y - c.y)))

        return {
            "road_id": int(wp.road_id),
            "section_id": int(wp.section_id),
            "lane_id": int(wp.lane_id),
            "is_junction": int(bool(wp.is_junction)),
            "lane_type": int(wp.lane_type),
            "center_off_m": off,
        }

    def _draw_dotted_segment(self, img, p0, p1, color, thickness=1, gap_px=14):
        x0, y0 = p0
        x1, y1 = p1
        dx = x1 - x0
        dy = y1 - y0
        dist = math.hypot(dx, dy)
        if dist < 1.0:
            return

        n = max(1, int(dist / float(gap_px)))
        for i in range(n + 1):
            t = i / float(n)
            x = int(round(x0 + t * dx))
            y = int(round(y0 + t * dy))
            cv2.circle(img, (x, y), max(1, thickness), color, -1, lineType=cv2.LINE_AA)

    def _camera_K(self, w: int, h: int, fov_deg: float) -> np.ndarray:
        fov_rad = math.radians(float(fov_deg))
        focal = w / (2.0 * math.tan(fov_rad / 2.0))
        K = np.eye(3, dtype=np.float32)
        K[0, 0] = focal
        K[1, 1] = focal
        K[0, 2] = w / 2.0
        K[1, 2] = h / 2.0
        return K

    def _project_world_to_pixel(
        self,
        loc: carla.Location,
        world_2_camera: np.ndarray,
        K: np.ndarray,
        w: int,
        h: int,
        z_min: float = 0.1,
    ):
        p_world = np.array([float(loc.x), float(loc.y), float(loc.z), 1.0], dtype=np.float32)
        p_cam = world_2_camera @ p_world

        x = float(p_cam[1])
        y = float(-p_cam[2])
        z = float(p_cam[0])

        if z <= z_min:
            return None

        u = (K[0, 0] * x / z) + K[0, 2]
        v = (K[1, 1] * y / z) + K[1, 2]

        return int(u), int(v)
 
    def _draw_routes_on_image(
        self,
        rgb: np.ndarray,
        planner_max_pts: int = 60,
        planner_thickness: int = 2,
        planner_color=(0, 255, 0),            
        ref_xy: Optional[np.ndarray] = None,  
        ref_world_pts: Optional[list] = None, 
        ref_max_pts: int = 200,
        ref_thickness: int = 1,
        ref_color=(255, 0, 255),             
        ref_gap_px: int = 14,
        draw_lane_ids: bool = True,
        lane_id_every: int = 10,        
        lane_id_on_change: bool = True, 
        lane_text_scale: float = 0.45,
    ):
        if rgb is None:
            return rgb
        if self.camera_sensor is None or self.ego is None:
            return rgb

        h, w = rgb.shape[:2]
        fov = float(getattr(self, "_camera_fov_deg", 110.0))
        K = self._camera_K(w, h, fov)

        cam_tf = self.camera_sensor.get_transform()
        world_2_camera = np.array(cam_tf.get_inverse_matrix(), dtype=np.float32)

        ego_z = float(self.ego.get_location().z)

        if self.waypoints is not None and len(self.waypoints) >= 2:
            planner_world = []
            for wp in self.waypoints[:planner_max_pts]:
                x, y = float(wp[0]), float(wp[1])
                planner_world.append(carla.Location(x=x, y=y, z=ego_z + 0.30))

            if self.end_point is not None:
                gl = self.end_point.location
                planner_world.append(carla.Location(x=float(gl.x), y=float(gl.y), z=float(gl.z) + 0.30))

            planner_pix = [self._project_world_to_pixel(p, world_2_camera, K, w, h) for p in planner_world]

            for p0, p1 in zip(planner_pix[:-1], planner_pix[1:]):
                if p0 is None or p1 is None:
                    continue
                cv2.line(rgb, p0, p1, planner_color, int(planner_thickness), lineType=cv2.LINE_AA)

            planner_lane_ids = [self._lane_id_at_world(p) for p in planner_world]

            if draw_lane_ids:
                last_lid = None
                for i, (p0, p1) in enumerate(zip(planner_pix[:-1], planner_pix[1:])):
                    if p0 is None or p1 is None:
                        continue

                    lid = planner_lane_ids[i] 
                    if lid is None:
                        continue
                    if lid == last_lid:
                        continue

                    mid = ((p0[0] + p1[0]) // 2, (p0[1] + p1[1]) // 2)
                    self._put_text_shadow(
                        rgb, mid, f"L{lid}", planner_color,
                        scale=float(lane_text_scale), thickness=1
                    )
                    last_lid = lid

            for p in planner_pix[::5]:
                if p is None:
                    continue
                cv2.circle(rgb, p, 2, (255, 255, 255), -1, lineType=cv2.LINE_AA)

            if planner_pix and planner_pix[-1] is not None:
                cv2.circle(rgb, planner_pix[-1], 6, (255, 255, 255), 2, lineType=cv2.LINE_AA)

        ref_world = None

        if ref_world_pts is not None and len(ref_world_pts) >= 2:
            ref_world = ref_world_pts[:ref_max_pts]

        elif ref_xy is not None:
            ref_xy = np.asarray(ref_xy, dtype=np.float32)
            if ref_xy.ndim == 2 and ref_xy.shape[0] >= 2 and ref_xy.shape[1] == 2:
                stride = max(1, int(ref_xy.shape[0] / max(2, ref_max_pts)))
                ref_xy_ds = ref_xy[::stride][:ref_max_pts]
                ref_world = [carla.Location(x=float(x), y=float(y), z=ego_z + 0.25) for x, y in ref_xy_ds]

        if ref_world is not None and len(ref_world) >= 2:
            ref_pix = [self._project_world_to_pixel(p, world_2_camera, K, w, h) for p in ref_world]
            ref_lane_ids = [self._lane_id_at_world(p) for p in ref_world]

            for p0, p1 in zip(ref_pix[:-1], ref_pix[1:]):
                if p0 is None or p1 is None:
                    continue
                self._draw_dotted_segment(rgb, p0, p1, ref_color, thickness=int(ref_thickness), gap_px=int(ref_gap_px))

            if ref_pix[0] is not None:
                cv2.circle(rgb, ref_pix[0], 5, ref_color, 2, lineType=cv2.LINE_AA)

        return rgb


    def _draw_planned_route_on_image(self, rgb: np.ndarray, max_pts: int = 60, thickness: int = 2):
        if rgb is None:
            return rgb
        if self.camera_sensor is None or self.ego is None:
            return rgb
        if self.waypoints is None or len(self.waypoints) < 2:
            return rgb

        h, w = rgb.shape[:2]
        fov = float(getattr(self, "_camera_fov_deg", 110.0))
        K = self._camera_K(w, h, fov)

        cam_tf = self.camera_sensor.get_transform()
        world_2_camera = np.array(cam_tf.get_inverse_matrix(), dtype=np.float32)

        ego_z = float(self.ego.get_location().z)

        world_pts = []
        for wp in self.waypoints[:max_pts]:
            x, y = float(wp[0]), float(wp[1])
            loc = carla.Location(x=x, y=y, z=ego_z + 0.3)
            world_pts.append(loc)

        if self.end_point is not None:
            gl = self.end_point.location
            world_pts.append(carla.Location(x=float(gl.x), y=float(gl.y), z=float(gl.z) + 0.3))

        pix = [self._project_world_to_pixel(p, world_2_camera, K, w, h) for p in world_pts]

        for p0, p1 in zip(pix[:-1], pix[1:]):
            if p0 is None or p1 is None:
                continue
            cv2.line(rgb, p0, p1, (0, 255, 0), int(thickness), lineType=cv2.LINE_AA)

        step = 5
        for p in pix[::step]:
            if p is None:
                continue
            cv2.circle(rgb, p, 2, (255, 255, 255), -1, lineType=cv2.LINE_AA)

        if pix and pix[-1] is not None:
            cv2.circle(rgb, pix[-1], 5, (255, 255, 255), 2, lineType=cv2.LINE_AA)

        return rgb

    def _get_lane_ids(self, loc: carla.Location):
        wp = self.map.get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Driving)
        if wp is None:
            return None, None, None, None, None
        return int(wp.road_id), int(wp.section_id), int(wp.lane_id), bool(wp.is_junction), int(wp.lane_type)

    def _build_spawn_lane_reference(self):
        if self.ego is None or self.end_point is None:
            self._ref_xy = None
            return

        start_wp = self.map.get_waypoint(
            self.ego.get_location(),
            project_to_road=True,
            lane_type=carla.LaneType.Driving
        )
        if start_wp is None:
            self._ref_xy = None
            return

        goal_loc = self.end_point.location
        step = float(self.ref_step_m)
        max_m = float(self.ref_max_m)
        goal_r = float(self.ref_goal_radius_m)

        pts = []
        wp = start_wp
        last_xy = None

        max_steps = int(max_m / max(step, 1e-3)) + 5
        for _ in range(max_steps):
            loc = wp.transform.location
            xy = (float(loc.x), float(loc.y))

            if last_xy is None or math.hypot(xy[0] - last_xy[0], xy[1] - last_xy[1]) > 1e-3:
                pts.append(xy)
                last_xy = xy

            if loc.distance(goal_loc) < goal_r:
                break

            nxt = wp.next(step)
            if not nxt:
                break

            if len(nxt) == 1:
                wp = nxt[0]
            else:
                best = None
                best_score = 1e18
                for cand in nxt:
                    cl = cand.transform.location
                    score = float(cl.distance(goal_loc))
                    if score < best_score:
                        best_score = score
                        best = cand
                wp = best if best is not None else nxt[0]

        if len(pts) < 2:
            ego_loc = self.ego.get_location()
            pts = [
                (float(ego_loc.x), float(ego_loc.y)),
                (float(goal_loc.x), float(goal_loc.y)),
            ]

        pts = np.asarray(pts, dtype=np.float32)

        seg = pts[1:] - pts[:-1]
        seg_len = np.linalg.norm(seg, axis=1).astype(np.float32)

        keep = seg_len > 1e-6
        if not np.all(keep):
            pts2 = [pts[0]]
            for i, k in enumerate(keep):
                if k:
                    pts2.append(pts[i + 1])
            pts = np.asarray(pts2, dtype=np.float32)
            seg = pts[1:] - pts[:-1]
            seg_len = np.linalg.norm(seg, axis=1).astype(np.float32)

        cum_s = np.concatenate([np.array([0.0], np.float32), np.cumsum(seg_len)], axis=0)

        self._ref_xy = pts
        self._ref_seg_len = seg_len
        self._ref_cum_s = cum_s
        self._ref_total_len = float(cum_s[-1])

        self._ref_seg_idx = 0
        self._ref_s = 0.0
        self._prev_ref_s = 0.0

    def _ref_lateral_error_and_progress(self, ego_xy: np.ndarray, search_back: int = 10, search_fwd: int = 40):
        pts = self._ref_xy
        if pts is None or len(pts) < 2:
            return 0.0, 0.0, np.array([1.0, 0.0], dtype=np.float32)

        nseg = len(pts) - 1
        i0 = max(0, self._ref_seg_idx - search_back)
        i1 = min(nseg - 1, self._ref_seg_idx + search_fwd)

        best_d2 = 1e18
        best_i = i0
        best_t = 0.0
        best_dir = np.array([1.0, 0.0], dtype=np.float32)

        for i in range(i0, i1 + 1):
            a = pts[i]
            b = pts[i + 1]
            ab = b - a
            denom = float(np.dot(ab, ab))
            if denom < 1e-9:
                continue

            t = float(np.clip(np.dot(ego_xy - a, ab) / denom, 0.0, 1.0))
            proj = a + t * ab
            d = ego_xy - proj
            d2 = float(np.dot(d, d))

            if d2 < best_d2:
                best_d2 = d2
                best_i = i
                best_t = t
                n = float(np.linalg.norm(ab))
                if n > 1e-6:
                    best_dir = (ab / n).astype(np.float32)

        if best_d2 > (15.0 ** 2):
            for i in range(nseg):
                a = pts[i]
                b = pts[i + 1]
                ab = b - a
                denom = float(np.dot(ab, ab))
                if denom < 1e-9:
                    continue
                t = float(np.clip(np.dot(ego_xy - a, ab) / denom, 0.0, 1.0))
                proj = a + t * ab
                d = ego_xy - proj
                d2 = float(np.dot(d, d))
                if d2 < best_d2:
                    best_d2 = d2
                    best_i = i
                    best_t = t
                    n = float(np.linalg.norm(ab))
                    if n > 1e-6:
                        best_dir = (ab / n).astype(np.float32)

        self._ref_seg_idx = best_i
        self._ref_s = float(self._ref_cum_s[best_i] + best_t * self._ref_seg_len[best_i])

        lane_d = float(math.sqrt(best_d2))
        return lane_d, self._ref_s, best_dir

    def _heading_error_to_ref_dir(self, ref_dir: np.ndarray) -> float:
        fwd = self.ego.get_transform().get_forward_vector()
        ev = np.array([float(fwd.x), float(fwd.y)], dtype=np.float32)
        n = float(np.linalg.norm(ev))
        if n < 1e-6:
            return 1.0
        ev /= n
        dot = float(np.clip(np.dot(ev, ref_dir), -1.0, 1.0))
        ang = math.acos(dot)
        return float(ang / math.pi)

    def _compute_offroad_flag(self) -> Tuple[bool, float]:
        veh_loc = self.ego.get_location()

        strict_wp = self.map.get_waypoint(
            veh_loc, project_to_road=False, lane_type=carla.LaneType.Driving
        )
        proj_wp = self.map.get_waypoint(
            veh_loc, project_to_road=True, lane_type=carla.LaneType.Driving
        )

        if proj_wp is None:
            return True, 1e9

        c = proj_wp.transform.location
        proj_off = float(math.hypot(veh_loc.x - c.x, veh_loc.y - c.y))

        offroad = (strict_wp is None) and (proj_off > float(self.offroad_center_thresh_m))
        return bool(offroad), proj_off

    def _weather_to_domain(self, weather_id: int) -> int:
        name = WEATHER_TABLE[int(weather_id)][0].lower()

        if "rain" in name:
            return 3  
        elif "wet" in name:
            return 2  
        elif "cloudy" in name:
            return 1   
        else:
            return 0   

    def _augment_rgb(self, rgb):
        x = rgb.astype(np.float32) / 255.0
        if self.aug_brightness > 0:
            b = self._rng.uniform(-self.aug_brightness, self.aug_brightness)
            x = x + b
        if self.aug_contrast > 0:
            c = self._rng.uniform(1 - self.aug_contrast, 1 + self.aug_contrast)
            x = (x - 0.5) * c + 0.5

        if self.aug_gamma > 0:
            g = self._rng.uniform(1 - self.aug_gamma, 1 + self.aug_gamma)
            x = np.clip(x, 0, 1) ** g

        if self.aug_blur_prob > 0 and self._rng.rand() < self.aug_blur_prob:
            k = int(self._rng.choice([3, 5]))
            x = cv2.GaussianBlur(x, (k, k), 0)

        if self.aug_noise_std > 0:
            n = self._rng.normal(0, self.aug_noise_std, size=x.shape).astype(np.float32)
            x = x + n

        x = np.clip(x, 0, 1)
        return (x * 255.0).astype(np.uint8)

    def _pygame_init(self):
        if self._pg_initialized:
            return

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
        if not self.render_enabled:
            return
        if rgb_uint8 is None:
            return

        self._pygame_init()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                raise SystemExit("Pygame window closed by user.")
        if self.render_fps_cap > 0:
            self._pg_clock.tick(self.render_fps_cap)

        img = rgb_uint8
        if img.shape[0] != self.render_size or img.shape[1] != self.render_size:
            img = cv2.resize(img, (self.render_size, self.render_size), interpolation=cv2.INTER_LINEAR)
        surf = pygame.surfarray.make_surface(np.transpose(img, (1, 0, 2)))
        self._pg_screen.blit(surf, (0, 0))

        if lines:
            y0 = 8
            line_h = 20
            y = y0

            max_lines = int((self.render_size - 2 * y0) // line_h)

            for item in lines[:max_lines]:
                if isinstance(item, tuple) and len(item) == 2:
                    s, color = item
                else:
                    s, color = item, (255, 255, 255)

                shadow = self._pg_font.render(str(s), True, (0, 0, 0))
                text   = self._pg_font.render(str(s), True, color)

                self._pg_screen.blit(shadow, (9, y + 1))
                self._pg_screen.blit(text,   (8, y))
                y += line_h

        pygame.display.flip()

    def _refill_route_queue(self):
        idxs = list(range(len(self.routes)))
        if not self.eval_mode:
            random.shuffle(idxs)
        self._route_queue = deque(idxs)

    def _refill_combo_queue(self):
        dom_to_wids = {d: [] for d in range(self.num_domains)}
        for wid in self.weather_ids:
            d = int(self._weather_to_domain(wid))
            if d < 0 or d >= self.num_domains:
                raise ValueError(f"weather_id {wid} mapped to domain {d}, num_domains={self.num_domains}")
            dom_to_wids[d].append(int(wid))

        missing = [d for d, ws in dom_to_wids.items() if len(ws) == 0]
        if missing:
            raise ValueError(f"Eval weather_ids do not cover weather domains {missing}. weather_ids={self.weather_ids}")

        m = max(len(ws) for ws in dom_to_wids.values())
        balanced_wids = []
        for d in range(self.num_domains):
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
            rid, wid = self._combo_queue.popleft()
            return int(rid), int(wid)

        if not self._route_queue:
            self._refill_route_queue()
        rid = int(self._route_queue.popleft())

        valid_wdoms = [d for d, ws in self._weather_domain_to_weather.items() if len(ws) > 0]
        wdom = int(self._rng.choice(valid_wdoms))
        wid = int(self._rng.choice(self._weather_domain_to_weather[wdom]))

        return rid, wid

    def _setup_action_space(self):
        self.n_steer = len(DISCRETE_STEER)
        self.n_long = len(DISCRETE_LONG)
        self.n_acc = self.n_long

        return spaces.Discrete(self.n_steer * self.n_long)


    def _setup_observation_space(self):
        camera_space = spaces.Box(0, 255, shape=(128, 128, 3), dtype=np.uint8)
        collision_space = spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32)
        lane_invasion_space = spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32)
        vlm_space = spaces.Box(-np.inf, np.inf, shape=(self.vlm_dim,), dtype=np.float32)
        domain_space = spaces.Box(0.0, float(self.num_domains - 1), shape=(1,), dtype=np.float32)
        domain_mask_space = spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32)

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

    def setup_camera(self):
        cam_bp = self.blueprint_library.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", "512")
        cam_bp.set_attribute("image_size_y", "512")
        cam_bp.set_attribute("fov", "110")
        cam_bp.set_attribute("sensor_tick", f"{self._fixed_dt}")  
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
            rgb = arr[:, :, :3][:, :, ::-1].copy()  
            item = (int(img.frame), rgb)
            try:
                q.put_nowait(item)
            except queue.Full:
                pass  

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

    def _pull_image_for_frame(self, target_frame: Optional[int], timeout: float = 0.5) -> np.ndarray:
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
                    if f > target_frame:
                        best = rgb
                        break
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
    
        obs = {
            "image": self.camera_image,
            "collision": np.array([float(min(1, int(getattr(self, "_collision_step", 0))))], np.float32),
            "lane_invasion": np.array([float(min(1, int(getattr(self, "_lane_invasion_step", 0))))], np.float32),

        }

        if self.include_vlm:
            obs["vlm"] = vlm.astype(np.float32)

        if self.include_domain_id:
            obs["domain_id"] = np.array([float(self.domain_id)], np.float32)
            obs["domain_mask"] = np.array([float(self.domain_mask)], np.float32)

        return obs

    def reset_vehicle(self):
        bp = self.blueprint_library.find("vehicle.tesla.model3")
        if bp.has_attribute("role_name"):
            bp.set_attribute("role_name", "hero")

        base_loc = self.ego_transform.location
        base_rot = self.ego_transform.rotation
        max_tries = int(_get_cfg(self._config, "spawn_max_tries", 25))
        clear_radius = float(_get_cfg(self._config, "spawn_clear_radius_m", 3.0))
        kill_blockers = bool(_get_cfg(self._config, "spawn_kill_blockers", True))

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

            if kill_blockers:
                try:
                    blockers = []
                    for pat in ("vehicle.*", "walker.*"):
                        for a in self.world.get_actors().filter(pat):
                            if a.get_location().distance(tf.location) < clear_radius:
                                blockers.append(a)

                    last_blockers = len(blockers)

                    if blockers:
                        cmds = [carla.command.DestroyActor(a) for a in blockers]
                        self.client.apply_batch_sync(cmds, True)
                        self.world.tick()
                except Exception:
                    pass

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
        if self.ego is None or self.end_point is None:
            return 0.0

        ego_loc = self.ego.get_location()
        goal_loc = self.end_point.location

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

    def reset(self):
        self._clean_actors()
        self.route_id, self.weather_id = self._select_route_and_weather()
        self.weather_name, weather_params = WEATHER_TABLE[self.weather_id]
        self.world.set_weather(weather_params)

        self.domain_id = int(self._weather_to_domain(self.weather_id)) 
        self.domain_mask = 1.0

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

        self._time_step = 0
        self.collision_detected = False
        self.collision_hist = []
        self.lane_invasion_detected = False
        self.lane_invasion_hist = []
        self.previous_lane_invasions = 0
        self.previous_collisions = 0
        self._lane_invasion_step = 0
        self._collision_step = 0
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

        snap = self.world.tick()
        frame_id = snap.frame if hasattr(snap, "frame") else int(snap)
        self._last_tick_frame = int(frame_id)

        self.reset_vehicle()
        
        self.ego.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, hand_brake=True))
        self.world.tick()
        self.world.tick()
        self.ego.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0, hand_brake=False))

        intended = self._get_lane_info(self.ego_transform.location, project_to_road=True)

        actual_strict = self._get_lane_info(self.ego.get_location(), project_to_road=False)
        actual_proj   = self._get_lane_info(self.ego.get_location(), project_to_road=True)

        actual = actual_strict if actual_strict is not None else actual_proj

        self.spawn_lane_intended = intended
        self.spawn_lane_actual = actual
        self.spawn_lane_actual_proj = actual_proj  
        if actual is not None:
            self.spawn_road_id = actual["road_id"]
            self.spawn_section_id = actual["section_id"]
            self.spawn_lane_id = actual["lane_id"]
            self.spawn_lane_type = actual["lane_type"]
        else:
            self.spawn_road_id = -1
            self.spawn_section_id = -1
            self.spawn_lane_id = 9999
            self.spawn_lane_type = -1

        self.cur_lane_info = actual
        self.cur_lane_info_proj = actual_proj

        
        self.setup_collision_sensor()
        self.setup_lane_invasion_sensor()
        self.setup_camera()

        snap = self.world.tick()
        frame_id = snap.frame if hasattr(snap, "frame") else int(snap)
        self._last_tick_frame = int(frame_id)

        dest_location = carla.Location(
            x=float(self.end_point.location.x),
            y=float(self.end_point.location.y),
            z=float(self.end_point.location.z),
        )
        self.ego_planner = FixedEndingPlanner(self.ego, dest_location)
        self.waypoints, self.planner_stats = self.ego_planner.run_step()
        self.num_completed = int(self.planner_stats.get("num_completed", 0))

        self._build_spawn_lane_reference()

        self._max_ref_s = 0.0
        self.route_len_m = float(getattr(self, "_ref_total_len", 0.0))

        self._prev_route_remaining_m = self._route_remaining_m()

        self.initial_distance_to_goal = self.ego.get_location().distance(self.end_point.location)
        self.last_distance_to_goal = float(self.initial_distance_to_goal)

        self._ever_near_goal = False
        self._past_goal_steps = 0
        self._dist_to_goal_m = None
        self._prev_dist_to_goal_m = None

        self.episode_return = 0.0
        self.episode_length = 0

        self._best_dist_to_goal = float(self.initial_distance_to_goal)
        self._dist_regret_steps = 0

        self.wrong_lane_steps = 0
        self.offroad_steps = 0
        self._wrong_lane = False
        self._offroad = False

        obs = self._get_observation(self._last_tick_frame)
        return obs

    def step(self, action: int):
        self.apply_control(action)
        self._time_step += 1
        snap = self.world.tick()
        frame_id = snap.frame if hasattr(snap, "frame") else int(snap)
        self._last_tick_frame = int(frame_id)
        new_inv = int(len(self.lane_invasion_hist) - self.previous_lane_invasions)
        new_inv = max(new_inv, 0)
        self.previous_lane_invasions += new_inv
        self._lane_invasion_step = int(new_inv)

        new_col = int(len(self.collision_hist) - self.previous_collisions)
        new_col = max(new_col, 0)
        self.previous_collisions += new_col
        self._collision_step = int(new_col)

        cur_loc = self.ego.get_location()

        cur_strict = self._get_lane_info(cur_loc, project_to_road=False)
        cur_proj   = self._get_lane_info(cur_loc, project_to_road=True)
        cur = cur_strict if cur_strict is not None else cur_proj

        self.cur_lane_info = cur
        self.cur_lane_info_proj = cur_proj

        if cur is not None:
            self.cur_road_id = cur["road_id"]
            self.cur_section_id = cur["section_id"]
            self.cur_lane_id = cur["lane_id"]
            self.cur_is_junction = bool(cur["is_junction"])
            self.cur_lane_type = cur["lane_type"]
        else:
            self.cur_road_id = -1
            self.cur_section_id = -1
            self.cur_lane_id = 9999
            self.cur_is_junction = False
            self.cur_lane_type = -1
        
        if self.prev_ego_location is None:
            step_dist = 0.0
        else:
            step_dist = distance_2d(cur_loc, self.prev_ego_location)
        self.travel_distance_m += step_dist
        self.prev_ego_location = cur_loc

        self.waypoints, self.planner_stats = self.ego_planner.run_step()
        self.num_completed = int(self.planner_stats.get("num_completed", 0))

        vel = self.ego.get_velocity()
        self.speed_kmh = 3.6 * math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)

        if self.speed_kmh < self.low_speed_kmh_thresh:
            self.low_speed_steps += 1
        else:
            self.low_speed_steps = 0

        obs = self._get_observation(self._last_tick_frame)

        dist_to_goal = float(self.ego.get_location().distance(self.end_point.location))

        self._prev_dist_to_goal_m = self._dist_to_goal_m
        self._dist_to_goal_m = dist_to_goal

        near_goal_radius_m = float(_get_cfg(self._config, "near_goal_radius_m", 6.0))
        if dist_to_goal < near_goal_radius_m:
            self._ever_near_goal = True

        prev_dist = float(self.last_distance_to_goal) if self.last_distance_to_goal is not None else dist_to_goal
        progress_m = max(0.0, prev_dist - dist_to_goal)

        reward, reward_terms = self._compute_reward()

        reward_terms = dict(reward_terms)

        for k in ("collision", "offroad", "wrong_lane"):
            reward_terms[f"r_{k}"] = reward_terms.get(k, 0.0)

        info_dict = dict(reward_terms)

        if self._time_step < getattr(self, "_lane_grace_steps", 0):
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
            "collision": 0,
            "wrong_lane_step": int(self._wrong_lane),
            "offroad_step": int(self._offroad),
        })


        info_dict.update({
            "spawn_road_id": int(self.spawn_road_id) if self.spawn_road_id is not None else -1,
            "spawn_section_id": int(self.spawn_section_id) if self.spawn_section_id is not None else -1,
            "spawn_lane_id": int(self.spawn_lane_id) if self.spawn_lane_id is not None else 9999,

            "cur_road_id": int(self.cur_road_id) if self.cur_road_id is not None else -1,
            "cur_section_id": int(self.cur_section_id) if self.cur_section_id is not None else -1,
            "cur_lane_id": int(self.cur_lane_id) if self.cur_lane_id is not None else 9999,
            "cur_is_junction": int(bool(self.cur_is_junction)) if self.cur_is_junction is not None else 0,
        })

        self.episode_return += float(reward)
        self.episode_length += 1

        long_cmd = float(DISCRETE_LONG[action // self.n_steer])
        steer = float(DISCRETE_STEER[action % self.n_steer])

        throttle_cmd = long_cmd if long_cmd >= 0.0 else 0.0
        brake_cmd = (-long_cmd) if long_cmd < 0.0 else 0.0

        lane_offset = float(getattr(self, "_last_lane_d", 0.0))
        heading_err = float(getattr(self, "_last_heading_err", 0.0))


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

        info_dict.update({
            "route_len_m": float(getattr(self, "route_len_m", 0.0)),
            "ref_s_m": float(getattr(self, "_ref_s", 0.0)),
            "max_ref_s_m": float(getattr(self, "_max_ref_s", 0.0)),
            "completion": float(
                np.clip(
                    float(getattr(self, "_max_ref_s", 0.0)) / max(float(getattr(self, "route_len_m", 0.0)), 1e-6),
                    0.0, 1.0
                )
            ),
        })

        if dist_to_goal < self._best_dist_to_goal - 0.25:   
            self._best_dist_to_goal = dist_to_goal
            self._dist_regret_steps = 0
        elif dist_to_goal > self._best_dist_to_goal + 2.0:  
            self._dist_regret_steps += 1
        else:
            self._dist_regret_steps = 0

        done, terminal_info = self._check_termination()
        info = {**info_dict, **terminal_info}

        if done:
            reason = "unknown"
            for k in ("goal_reached", "collision", "offroad", "wrong_lane", "past_goal", "not_moving", "time_exceeded"):
                if terminal_info.get(k, False):
                    reason = k
                    break

            print(
                f"[Episode {self.episode_id}] DONE | "
                f"steps={self.episode_length} | "
                f"return={self.episode_return:.2f} | "
                f"reason={reason} | "
                f"route={self.route_id} | "
                f"weather={self.weather_name}"
            )

            self.episode_id += 1

        def _fmt_lane(info, label):
            if info is None:
                return f"{label}: None"
            return (f"{label}: road={info['road_id']} sec={info['section_id']} lane={info['lane_id']} "
                    f"junc={info['is_junction']} off={info['center_off_m']:.2f}m")



        if self.render_enabled and self.camera_image2 is not None:
            loc = self.ego.get_location()
            x, y, z = float(loc.x), float(loc.y), float(loc.z)

            goal_loc = self.end_point.location
            dx = float(goal_loc.x - loc.x)
            dy = float(goal_loc.y - loc.y)
            goal_norm = math.hypot(dx, dy)

            fwd = self.ego.get_transform().get_forward_vector()
            hx = float(fwd.x)
            hy = float(fwd.y)
            head_norm = math.hypot(hx, hy)

            if goal_norm < 1e-6 or head_norm < 1e-6:
                goal_ang_rad = 0.0
            else:
                gx, gy = dx / goal_norm, dy / goal_norm
                hx, hy = hx / head_norm, hy / head_norm
                dot = max(-1.0, min(1.0, gx * hx + gy * hy))  
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

            spawn_lane_str = f"spawn: road={self.spawn_road_id} sec={self.spawn_section_id} lane={self.spawn_lane_id}"
            cur_lane_str   = f"cur:   road={self.cur_road_id} sec={self.cur_section_id} lane={self.cur_lane_id} junc={int(bool(self.cur_is_junction))}"

            state_lines = [
                (f"route_id: {rid} ({route_dir}) | town: {self.map.name.split('/')[-1]}", (255, 255, 255)),
                (f"weather: {info.get('weather_name', info.get('weather_id', 'NA'))}", (255, 255, 255)),

                (_fmt_lane(self.spawn_lane_intended, "spawn_intended"), (255,255,255)),
                (_fmt_lane(self.spawn_lane_actual,   "spawn_actual"),   (255,255,255)),
                (_fmt_lane(self.cur_lane_info,       "cur_best"),       (255,255,255)),
                (_fmt_lane(self.cur_lane_info_proj,  "cur_proj"),       (180,180,180)),

                (f"ego xyz: ({x:.2f}, {y:.2f}, {z:.2f})", (255, 255, 255)),
                (f"dist_to_goal_m: {dist_to_goal:.2f}", (255, 255, 255)),
                (f"goal_ang_err_deg: {goal_ang_deg:.1f}", (255, 255, 255)),
                (f"speed_kmh: {info.get('speed_kmh', 0.0):.1f}", (255, 255, 255)),
                (f"off_center_m: {info.get('off_center_m', 0.0):.2f}", (255, 255, 255)),
            ]

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

            img_vis = self.camera_image2

            if getattr(self, "render_draw_route", False):
                img_vis = img_vis.copy()

                self._draw_routes_on_image(
                    img_vis,
                    planner_max_pts=int(getattr(self, "render_route_max_pts", 60)),
                    planner_thickness=int(getattr(self, "render_route_thickness", 2)),
                    ref_xy=getattr(self, "_ref_xy", None),
                    ref_max_pts=200,
                    ref_thickness=1,
                    ref_gap_px=14,

                    draw_lane_ids=True,
                    lane_id_every=12,        
                    lane_id_on_change=True,
                    lane_text_scale=0.45,
                )


            self._pygame_render(img_vis, lines=lines)

        return obs, float(reward), bool(done), info

    def _signed_ahead_to_goal_m(self) -> float:
        if self.ego is None or self.end_point is None or self._start_xy is None:
            return 0.0

        ego_loc = self.ego.get_location()
        goal_loc = self.end_point.location

        yaw_rad = math.radians(float(self.end_point.rotation.yaw))
        d = np.array([math.cos(yaw_rad), math.sin(yaw_rad)], dtype=np.float32)

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

    def apply_control(self, action: int) -> None:
        control = self._get_vehicle_control(action)
        self.ego.apply_control(control)

    def _get_vehicle_control(self, action: int) -> carla.VehicleControl:
        long_cmd = float(DISCRETE_LONG[action // self.n_steer])
        steer = float(DISCRETE_STEER[action % self.n_steer])

        if long_cmd >= 0.0:
            throttle = long_cmd
            brake = 0.0
        else:
            throttle = 0.0
            brake = -long_cmd  

        throttle = float(np.clip(throttle, 0.0, 1.0))
        brake    = float(np.clip(brake,    0.0, 1.0))
        steer    = float(np.clip(steer,   -1.0, 1.0))

        return carla.VehicleControl(throttle=throttle, steer=steer, brake=brake)

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
            return 10.0  
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

    def _route_lane_anchor_wp(self):
        if (self.waypoints is None) or (len(self.waypoints) == 0):
            return None

        x, y = float(self.waypoints[0][0]), float(self.waypoints[0][1])
        loc = carla.Location(x=x, y=y, z=float(self.ego.get_location().z))
        return self.map.get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Driving)

    def _find_wp_on_lane_at_same_s(self, base_wp, target_lane_id: int, max_hops: int = 8):
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
        veh_loc = self.ego.get_location()
        cur_wp = self.map.get_waypoint(
            veh_loc, project_to_road=True, lane_type=carla.LaneType.Driving
        )

        route_wp = self._route_lane_anchor_wp()

        if cur_wp is None or route_wp is None:
            return float(self.get_lane_offset()), cur_wp, route_wp, None

        if cur_wp.road_id != route_wp.road_id:
            return float(self.get_lane_offset()), cur_wp, route_wp, None

        target_wp = self._find_wp_on_lane_at_same_s(cur_wp, int(route_wp.lane_id), max_hops=8)
        if target_wp is None:
            return float(self.get_lane_offset()), cur_wp, route_wp, None

        c = target_wp.transform.location
        off = math.sqrt((veh_loc.x - c.x) ** 2 + (veh_loc.y - c.y) ** 2)
        return float(off), cur_wp, route_wp, target_wp

    def get_angle_offset_to_wp(self, wp):
        if wp is None:
            return 1.0
        wv = wp.transform.get_forward_vector()
        ev = self.ego.get_transform().get_forward_vector()
        a = np.array([wv.x, wv.y], np.float32)
        b = np.array([ev.x, ev.y], np.float32)
        denom = (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
        angle = np.arccos(np.clip(float(np.dot(a, b) / denom), -1.0, 1.0))
        return float(angle / np.pi)

    def _compute_reward(self):
        rc = {}
        new_inv = int(getattr(self, "_lane_invasion_step", 0))
        new_col = int(getattr(self, "_collision_step", 0))

        offroad, _proj_off = self._compute_offroad_flag()

        if self._ref_xy is None or len(self._ref_xy) < 2:
            self._build_spawn_lane_reference()

        veh_loc = self.ego.get_location()
        ego_xy = np.array([float(veh_loc.x), float(veh_loc.y)], dtype=np.float32)

        lane_d, s, ref_dir = self._ref_lateral_error_and_progress(ego_xy)
        self._max_ref_s = max(getattr(self, "_max_ref_s", 0.0), float(s))

        heading_err = self._heading_error_to_ref_dir(ref_dir)

        self._last_lane_d = float(lane_d)
        self._last_heading_err = float(heading_err)

        delta_s = float(s - self._prev_ref_s)
        self._prev_ref_s = float(s)

        wrong_lane = (not offroad) and (lane_d > float(self.lane_change_threshold_m))

        self._offroad = bool(offroad)
        self._wrong_lane = bool(wrong_lane)

        rc["route_progress"] = float(self.k_route_progress) * float(np.clip(delta_s, -0.5, 0.5))

        lane_rad = min(float(lane_d), float(self.lane_center_cap_m))
        rc["lane"] = -float(self.k_lane_center) * (lane_rad ** 2)

        rc["wrong_lane"] = -float(self.k_wrong_lane) * float(wrong_lane)

        head_rad = min(float(heading_err), float(self.heading_cap))
        rc["heading"] = -float(self.k_heading) * (head_rad ** 2)

        vel = self.ego.get_velocity()
        v_xy = np.array([float(vel.x), float(vel.y)], dtype=np.float32)

        v_par = float(np.dot(v_xy, ref_dir))
        perp_dir = np.array([-ref_dir[1], ref_dir[0]], dtype=np.float32)
        v_perp = float(np.dot(v_xy, perp_dir))

        rc["r_speed"] = -(
            float(self.k_speed_par) * abs(v_par - float(self.desired_speed_mps))
            + float(self.k_speed_perp) * abs(v_perp)
        )

        if v_par < -0.5:
            rc["r_speed"] -= float(self.reverse_penalty)

        if self.low_speed_steps > 0:
            if self.low_speed_steps >= self.low_speed_timeout_steps:
                rc["low_speed"] = -50.0
            else:
                rc["low_speed"] = -2.0
        else:
            rc["low_speed"] = 0.0

        rc["offroad"] = -float(self.k_offroad) * float(offroad)
        rc["invasion"] = -float(self.k_invasion) * float(max(new_inv, 0))
        rc["collision"] = -float(self.k_collision) * float(max(new_col, 0))

        dist_goal = float(self.ego.get_location().distance(self.end_point.location))
        goal_r = float(_get_cfg(self._config, "goal_radius_m", 2.0))
        rc["goal"] = float(self.goal_bonus) if dist_goal < goal_r else 0.0

        time_pen = float(_get_cfg(self._config, "time_penalty", 0.05))
        rc["time"] = -time_pen

        rc["waypoint"] = 0.0

        self.last_distance_to_goal = dist_goal

        total = float(sum(rc.values()))
        return total, rc

    def _check_termination(self):
        collision = bool(self.collision_detected)
        reached_goal = (self.ego.get_location().distance(self.end_point.location) < 2.0)
        time_exceeded = (self._time_step >= self._max_time_step)
        offroad_done = (self.offroad_steps >= self.max_offroad_steps)
        wrong_lane_done = (self.wrong_lane_steps >= self.max_wrong_lane_steps)

        LOW_SPEED_TIMEOUT_STEPS = self.low_speed_timeout_steps
        stuck_too_long = (self.low_speed_steps >= LOW_SPEED_TIMEOUT_STEPS)

        goal_radius_m = float(_get_cfg(self._config, "goal_radius_m", 2.0)) 
        past_goal_margin_m = float(_get_cfg(self._config, "past_goal_margin_m", 1.0))
        past_goal_timeout_s = float(_get_cfg(self._config, "past_goal_timeout_s", 1.0))
        past_goal_steps_needed = int(round(past_goal_timeout_s / float(self._fixed_dt)))

        ahead_m = self._signed_ahead_to_goal_m()

        past_now = (
            bool(getattr(self, "_ever_near_goal", False)) and
            (ahead_m > past_goal_margin_m) and
            (self._dist_to_goal_m is not None) and
            (self._dist_to_goal_m > goal_radius_m)
        )

        moving_away = False
        eps = float(_get_cfg(self._config, "past_goal_moving_away_eps_m", 0.05))
        if (self._prev_dist_to_goal_m is not None) and (self._dist_to_goal_m is not None):
            moving_away = (self._dist_to_goal_m > self._prev_dist_to_goal_m + eps)

        if past_now and moving_away:
            self._past_goal_steps += 1
        else:
            self._past_goal_steps = 0

        past_goal = (self._past_goal_steps >= past_goal_steps_needed)

        info = {
            "collision": 0,
            "goal_reached": 0,
            "offroad": 0,
            "wrong_lane": 0,
            "past_goal": 0,
            "not_moving": 0,
            "time_exceeded": 0,
            "stuck_duration": 0.0,   
            "elapsed_steps": 0,      
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

        info["distance_m"] = float(self.travel_distance_m)
        info["distance_km"] = float(self.travel_distance_m) / 1000.0
        info["lane_invasions"] = int(self.previous_lane_invasions)
        info["mean_off_center_m"] = float(self.off_center_sum / self.off_center_steps) if self.off_center_steps > 0 else 0.0

        return done, info

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



