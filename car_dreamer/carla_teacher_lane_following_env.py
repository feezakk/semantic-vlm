import gym
import math
import time
import queue
import weakref
from collections import deque
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import carla
from gym import spaces

from .toolkit.planner import FixedEndingPlanner


# ==============================================================================
# Scenario configuration
# ==============================================================================

EGO_SPAWN_POINT = [
    [71.929916,  -6.996726,   1.050581,  0.953298,  -62.641895,   0.000000],
    [59.233459, -68.268501,   5.817840,  4.703701,  -81.431274,   0.000000],
    [67.262825, -114.866020,  9.692015,  4.703701,  -85.152115,   0.000000],
    [44.566227, -177.097443,  6.951100, -4.487109,  -93.109505,   0.000000],
]

EGO_END_POINT = [
    [59.233459, -68.268501,   5.817840,  4.703701,  -81.431274,   0.000000],
    [67.262825, -114.866020,  9.692015,  4.703701,  -85.152115,   0.000000],
    [44.566227, -177.097443,  6.951100, -4.487109,  -93.109505,   0.000000],
    [6.476024,  -241.105026,  1.000000,  2.601381, -192.499680,   0.000000],
]

# Discrete action set: action = acc_index * n_steer + steer_index
DISCRETE_ACC = [0.0, 0.3]
DISCRETE_STEER = [-0.2, -0.1, 0.0, 0.1, 0.2]


# ==============================================================================
# Helpers
# ==============================================================================

def _get_cfg(obj: Any, path: str, default: Any = None) -> Any:
    """
    Robust nested config getter. Works for dict-like or attribute-like configs.

    Example:
        _get_cfg(config, "world.carla_port", 3000)
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


def distance_2d(loc1: carla.Location, loc2: carla.Location) -> float:
    """Euclidean distance in the xy-plane (meters)."""
    dx = float(loc1.x - loc2.x)
    dy = float(loc1.y - loc2.y)
    return math.sqrt(dx * dx + dy * dy)


# ==============================================================================
# Environment
# ==============================================================================

class CarlaTeacherLaneFollowingEnv(gym.Env):
    """
    Discrete-action lane-following environment in CARLA Town07.

    Action space (Discrete):
        action index -> (throttle, steer)
        throttle from DISCRETE_ACC, steer from DISCRETE_STEER.

    Observation space (Dict):
        image: uint8 (128, 128, 3) RGB
        collision: float32 (1,) in {0,1}  (latched within episode)
        lane_invasion: float32 (1,) in {0,1} (latched within episode)

    Notes:
      - This environment uses synchronous mode with fixed delta seconds.
      - The camera callback pushes (frame_id, rgb) into a queue.
      - The main thread pulls an image matching the current world tick frame.
    """

    metadata = {"render.modes": ["human"]}

    # --------------------------------------------------------------------------
    # Construction / teardown
    # --------------------------------------------------------------------------

    def __init__(self, config: Any):
        super().__init__()
        self._config = config

        # ----------------------------
        # CARLA connection parameters
        # ----------------------------
        host = str(_get_cfg(config, "carla_host", "localhost"))
        port = int(_get_cfg(config, "carla_port", 3000))
        timeout_s = float(_get_cfg(config, "carla_timeout_s", 300.0))
        town = str(_get_cfg(config, "town", "Town07"))

        self.client = carla.Client(host, port)
        self.client.set_timeout(timeout_s)

        # ----------------------------
        # World loading and sync mode
        # ----------------------------
        world = self.client.get_world()
        try:
            map_name = world.get_map().name
        except RuntimeError:
            map_name = ""

        # More robust map check than exact string equality
        if not str(map_name).endswith(town):
            world = self.client.load_world(town)

        self.world = world
        self.map = self.world.get_map()
        self.blueprint_library = self.world.get_blueprint_library()

        # Store original settings so close() can restore them
        orig_settings = self.world.get_settings()
        self._orig_synchronous_mode = bool(orig_settings.synchronous_mode)
        self._orig_fixed_delta_seconds = orig_settings.fixed_delta_seconds

        fixed_dt = float(_get_cfg(config, "fixed_delta_seconds", 0.05))
        settings = self.world.get_settings()
        if (not settings.synchronous_mode) or (settings.fixed_delta_seconds != fixed_dt):
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = fixed_dt
            self.world.apply_settings(settings)

        self._sync_enabled = True
        self._fixed_dt = float(self.world.get_settings().fixed_delta_seconds or fixed_dt)

        # Prime the world and record the first tick id
        snap = self.world.tick()
        self._last_tick_frame = int(snap.frame if hasattr(snap, "frame") else int(snap))

        # ----------------------------
        # Episode / runtime state
        # ----------------------------
        self.ego: Optional[carla.Vehicle] = None
        self.actors: list = []  # includes ego + sensors for destruction

        self.camera_sensor: Optional[carla.Sensor] = None
        self.colsensor: Optional[carla.Sensor] = None
        self.lane_sensor: Optional[carla.Sensor] = None

        self._img_q: Optional[queue.Queue] = None
        self._got_frame: bool = False

        self.camera_image: Optional[np.ndarray] = None   # resized 128x128
        self.camera_image2: Optional[np.ndarray] = None  # full 512x512

        # Event flags (latched within the episode)
        self.collision_detected: bool = False
        self.lane_invasion_detected: bool = False

        # Event history buffers (used for per-step deltas)
        self.collision_hist: list = []
        self.lane_invasion_hist: list = []
        self.previous_collisions: int = 0
        self.previous_lane_invasions: int = 0

        # Planner state
        self.ego_planner: Optional[FixedEndingPlanner] = None
        self.waypoints = []
        self.planner_stats: Dict[str, Any] = {}
        self.num_completed: int = 0

        # Timing and limits
        self._time_step: int = 0
        self._max_time_step: int = int(_get_cfg(config, "max_time_step", 2000))

        # Distance metrics
        self.prev_ego_location: Optional[carla.Location] = None
        self.travel_distance_m: float = 0.0
        self.off_center_sum: float = 0.0
        self.off_center_steps: int = 0

        # Scenario bookkeeping (route/weather identifiers)
        self.route_id: int = 0
        self.weather_id: int = 0
        self.weather_name: str = "ClearNoon"

        # Spawning control
        self._spawn_queue = deque(np.random.permutation(len(EGO_SPAWN_POINT)))
        self.spawn_index: int = int(self._spawn_queue.popleft())
        self.ego_transform: carla.Transform = self._make_transform(EGO_SPAWN_POINT[self.spawn_index])
        self.end_point: carla.Transform = self._make_transform(EGO_END_POINT[self.spawn_index])

        # Reference path (spawn-lane reference polyline)
        self.ref_step_m = float(_get_cfg(config, "ref_step_m", 0.5))
        self.ref_max_m = float(_get_cfg(config, "ref_max_m", 1500.0))
        self.ref_goal_radius_m = float(_get_cfg(config, "ref_goal_radius_m", 3.0))

        self._ref_xy: Optional[np.ndarray] = None
        self._ref_seg_len: Optional[np.ndarray] = None
        self._ref_cum_s: Optional[np.ndarray] = None
        self._ref_total_len: float = 0.0
        self._ref_seg_idx: int = 0
        self._ref_s: float = 0.0
        self._prev_ref_s: float = 0.0
        self._max_ref_s: float = 0.0
        self.route_len_m: float = 0.0

        # Cached geometry (for logging)
        self._last_lane_d: float = 0.0
        self._last_heading_err: float = 0.0

        # Geometry-based lane change / offroad thresholds
        self.lane_change_threshold_m = float(_get_cfg(config, "lane_change_threshold_m", 1.8))
        self.offroad_center_thresh_m = float(_get_cfg(config, "offroad_center_thresh_m", 2.2))

        # Reward weights
        self.k_route_progress = float(_get_cfg(config, "k_route_progress", 50.0))
        self.k_lane_center = float(_get_cfg(config, "k_lane_center", 10.0))
        self.lane_center_cap_m = float(_get_cfg(config, "lane_center_cap_m", 1.5))
        self.k_wrong_lane = float(_get_cfg(config, "k_wrong_lane", 20.0))
        self.k_heading = float(_get_cfg(config, "k_heading", 50.0))
        self.heading_cap = float(_get_cfg(config, "heading_cap", 0.5))
        self.desired_speed_mps = float(_get_cfg(config, "desired_speed_mps", 5.0))
        self.k_speed_par = float(_get_cfg(config, "k_speed_par", 1.0))
        self.k_speed_perp = float(_get_cfg(config, "k_speed_perp", 2.0))
        self.reverse_penalty = float(_get_cfg(config, "reverse_penalty", 2.0))
        self.k_offroad = float(_get_cfg(config, "k_offroad", 20.0))
        self.k_invasion = float(_get_cfg(config, "k_invasion", 5.0))
        self.k_collision = float(_get_cfg(config, "k_collision", 500.0))
        self.goal_bonus = float(_get_cfg(config, "goal_bonus", 200.0))
        self.time_penalty = float(_get_cfg(config, "time_penalty", 0.05))

        # Low-speed termination (step-based)
        self.low_speed_kmh_thresh = float(_get_cfg(config, "low_speed_kmh_thresh", 1.0))
        self.low_speed_timeout_s = float(_get_cfg(config, "low_speed_timeout_s", 10.0))
        self.low_speed_timeout_steps = int(round(self.low_speed_timeout_s / self._fixed_dt))
        self.low_speed_steps: int = 0

        # Persistent offroad / wrong-lane termination (step-based)
        self.lane_grace_s = float(_get_cfg(config, "lane_grace_s", 1.0))
        self._lane_grace_steps = int(round(self.lane_grace_s / self._fixed_dt))

        self.max_wrong_lane_s = float(_get_cfg(config, "max_wrong_lane_s", 3.0))
        self.max_offroad_s = float(_get_cfg(config, "max_offroad_s", 1.0))
        self.max_wrong_lane_steps = int(round(self.max_wrong_lane_s / self._fixed_dt))
        self.max_offroad_steps = int(round(self.max_offroad_s / self._fixed_dt))

        self.wrong_lane_steps: int = 0
        self.offroad_steps: int = 0
        self._wrong_lane: bool = False
        self._offroad: bool = False

        # Past-goal termination state
        self._start_xy: Optional[Tuple[float, float]] = None
        self._ever_near_goal: bool = False
        self._past_goal_steps: int = 0
        self._dist_to_goal_m: Optional[float] = None
        self._prev_dist_to_goal_m: Optional[float] = None

        # Logging / episode accounting
        self.episode_return: float = 0.0
        self.episode_length: int = 0
        self.episode_id: int = 0
        self.timestep: int = 0

        # Optional display (disable by default for headless training)
        self.enable_display = bool(_get_cfg(config, "enable_display", False))

        # Gym spaces
        self.action_space = self._setup_action_space()
        self.observation_space = self._setup_observation_space()

        # Diagnostics
        print("CARLA environment initialized")
        print("Map name:", self.map.name)

    def close(self) -> None:
        """Restore CARLA settings and destroy all spawned actors."""
        try:
            settings = self.world.get_settings()
            settings.synchronous_mode = self._orig_synchronous_mode
            settings.fixed_delta_seconds = self._orig_fixed_delta_seconds
            self.world.apply_settings(settings)
            self._sync_enabled = False
        finally:
            self._clean_actors()

    # --------------------------------------------------------------------------
    # Gym API
    # --------------------------------------------------------------------------

    def reset(self, **kwargs) -> Dict[str, np.ndarray]:
        """
        Reset the environment.

        Returns
        -------
        obs : dict
            Initial observation.
        """
        self._clean_actors()

        # Sample next spawn
        self.spawn_index = int(self._next_spawn_index())
        self.ego_transform = self._make_transform(EGO_SPAWN_POINT[self.spawn_index])
        self.end_point = self._make_transform(EGO_END_POINT[self.spawn_index])

        self.route_id = int(self.spawn_index)
        self.weather_id = 0
        self.weather_name = "ClearNoon"
        self._start_xy = (float(self.ego_transform.location.x), float(self.ego_transform.location.y))

        # Reset per-episode counters/state
        self._time_step = 0
        self.collision_detected = False
        self.lane_invasion_detected = False
        self.collision_hist = []
        self.lane_invasion_hist = []
        self.previous_collisions = 0
        self.previous_lane_invasions = 0

        self.low_speed_steps = 0
        self.wrong_lane_steps = 0
        self.offroad_steps = 0
        self._wrong_lane = False
        self._offroad = False

        self._ever_near_goal = False
        self._past_goal_steps = 0
        self._dist_to_goal_m = None
        self._prev_dist_to_goal_m = None

        self.episode_return = 0.0
        self.episode_length = 0

        self.prev_ego_location = None
        self.travel_distance_m = 0.0
        self.off_center_sum = 0.0
        self.off_center_steps = 0

        # Tick to advance the world cleanly before spawning
        snap = self.world.tick()
        self._last_tick_frame = int(snap.frame if hasattr(snap, "frame") else int(snap))

        # Spawn ego
        self.reset_vehicle()
        self.actors.append(self.ego)

        # Stabilize spawn pose (briefly hold brake)
        self.ego.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, hand_brake=True))
        self.world.tick()
        self.world.tick()
        self.ego.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0, hand_brake=False))

        # Attach sensors
        self.setup_collision_sensor()
        self.setup_lane_invasion_sensor()
        self.setup_camera()

        # Tick once so the camera can produce a frame for this tick
        snap = self.world.tick()
        self._last_tick_frame = int(snap.frame if hasattr(snap, "frame") else int(snap))

        # Initialize planner
        ego_dest = EGO_END_POINT[self.spawn_index]
        dest_location = carla.Location(x=float(ego_dest[0]), y=float(ego_dest[1]), z=float(ego_dest[2]))
        self.ego_planner = FixedEndingPlanner(self.ego, dest_location)
        self.waypoints, self.planner_stats = self.ego_planner.run_step()
        self.num_completed = int(self.planner_stats.get("num_completed", 0))

        # Build reference path used by reward shaping
        self._build_spawn_lane_reference()
        self._max_ref_s = 0.0
        self.route_len_m = float(self._ref_total_len or 0.0)
        self._prev_ref_s = 0.0

        # Initialize distance tracker after spawn
        self.prev_ego_location = self.ego.get_location()

        obs = self._get_observation(self._last_tick_frame)
        return obs

    def step(self, action: int):
        """
        Execute one environment step.

        Returns
        -------
        obs : dict
        reward : float
        done : bool
        info : dict
        """
        # Validate action
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action} for action_space={self.action_space}")

        # 1) Apply control
        self.apply_control(action)
        self._time_step += 1

        # 2) Tick world
        snap = self.world.tick()
        self._last_tick_frame = int(snap.frame if hasattr(snap, "frame") else int(snap))

        # 3) Distance traveled (xy-plane)
        cur_loc = self.ego.get_location()
        step_dist = 0.0 if self.prev_ego_location is None else distance_2d(cur_loc, self.prev_ego_location)
        self.travel_distance_m += float(step_dist)
        self.prev_ego_location = cur_loc

        # 4) Planner update (optional for logging; reward uses reference path)
        self.waypoints, self.planner_stats = self.ego_planner.run_step()
        self.num_completed = int(self.planner_stats.get("num_completed", 0))

        # 5) Speed and low-speed counter
        vel = self.ego.get_velocity()
        speed_mps = math.sqrt(float(vel.x * vel.x + vel.y * vel.y + vel.z * vel.z))
        speed_kmh = 3.6 * speed_mps

        if speed_kmh < self.low_speed_kmh_thresh:
            self.low_speed_steps += 1
        else:
            self.low_speed_steps = 0

        # 6) Observation (frame-aligned)
        obs = self._get_observation(self._last_tick_frame)

        # 7) Dist-to-goal tracking (for past-goal termination)
        dist_to_goal = float(cur_loc.distance(self.end_point.location))
        self._prev_dist_to_goal_m = self._dist_to_goal_m
        self._dist_to_goal_m = dist_to_goal

        near_goal_radius_m = float(_get_cfg(self._config, "near_goal_radius_m", 6.0))
        if dist_to_goal < near_goal_radius_m:
            self._ever_near_goal = True

        # For logging: positive-only progress to goal (independent of reference progress reward)
        prev_dist = float(getattr(self, "last_distance_to_goal", dist_to_goal))
        progress_m = max(0.0, prev_dist - dist_to_goal)

        # 8) Reward (updates _offroad/_wrong_lane and cached geometry)
        reward, reward_terms = self._compute_reward()
        reward_terms = dict(reward_terms)

        lane_offset = float(self._last_lane_d)
        heading_err = float(self._last_heading_err)

        self.off_center_sum += lane_offset
        self.off_center_steps += 1

        # 9) Persistent offroad/wrong-lane counters (grace period supported)
        if self._time_step < self._lane_grace_steps:
            self.offroad_steps = 0
            self.wrong_lane_steps = 0
        else:
            self.offroad_steps = (self.offroad_steps + 1) if self._offroad else 0
            self.wrong_lane_steps = (self.wrong_lane_steps + 1) if self._wrong_lane else 0

        # 10) Termination (exactly once)
        done, terminal_info = self._check_termination()

        # 11) Episode accounting
        self.episode_return += float(reward)
        self.episode_length += 1

        # 12) Action decoding (for logs)
        acc = float(DISCRETE_ACC[action // self.n_steer])
        steer = float(DISCRETE_STEER[action % self.n_steer])

        # 13) Info dict (prefix reward and terminal keys to avoid collisions)
        info: Dict[str, Any] = {f"r_{k}": float(v) for k, v in reward_terms.items()}
        info.update({
            "route_id": int(self.route_id),
            "weather_id": int(self.weather_id),
            "weather_name": str(self.weather_name),
            "town": "Town07",

            "step_distance": float(step_dist),
            "speed_kmh": float(speed_kmh),
            "off_center_m": float(lane_offset),
            "heading_error": float(heading_err),

            "dist_to_goal_m": float(dist_to_goal),
            "progress_m": float(progress_m),

            "lane_invasion_step": int(getattr(self, "_lane_invasion_step", 0)),
            "collision_step": int(getattr(self, "_collision_step", 0)),
            "wrong_lane_step": int(self._wrong_lane),
            "offroad_step": int(self._offroad),

            "got_frame": int(self._got_frame),
            "action_idx": int(action),

            "long_cmd": float(acc),
            "throttle_cmd": float(acc),
            "brake_cmd": 0.0,
            "steer_cmd": float(steer),

            "route_len_m": float(self.route_len_m),
            "ref_s_m": float(getattr(self, "_ref_s", 0.0)),
            "max_ref_s_m": float(getattr(self, "_max_ref_s", 0.0)),
            "completion": float(
                np.clip(
                    float(getattr(self, "_max_ref_s", 0.0)) / max(float(self.route_len_m), 1e-6),
                    0.0, 1.0
                )
            ),
        })
        info.update({f"term_{k}": v for k, v in terminal_info.items()})

        # 14) Episode counter
        if done:
            self.episode_id += 1

        # 15) Optional on-screen display (useful for debugging, harmful for training speed)
        if self.enable_display and (self.camera_image2 is not None):
            resized_image = cv2.resize(self.camera_image2, (512, 512), interpolation=cv2.INTER_LINEAR)
            cv2.imshow("EgoCamera", resized_image)
            cv2.waitKey(1)

        self.timestep += 1
        return obs, float(reward), bool(done), info

    # --------------------------------------------------------------------------
    # Spaces
    # --------------------------------------------------------------------------

    def _setup_action_space(self) -> spaces.Discrete:
        self.n_steer = len(DISCRETE_STEER)
        self.n_acc = len(DISCRETE_ACC)
        return spaces.Discrete(self.n_steer * self.n_acc)

    def _setup_observation_space(self) -> spaces.Dict:
        return spaces.Dict({
            "image": spaces.Box(0, 255, shape=(128, 128, 3), dtype=np.uint8),
            "collision": spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
            "lane_invasion": spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
        })

    # --------------------------------------------------------------------------
    # Spawning
    # --------------------------------------------------------------------------

    @staticmethod
    def _make_transform(p: list) -> carla.Transform:
        """Create a CARLA Transform from [x,y,z,pitch,yaw,roll]."""
        return carla.Transform(
            carla.Location(x=float(p[0]), y=float(p[1]), z=float(p[2])),
            carla.Rotation(pitch=float(p[3]), yaw=float(p[4]), roll=float(p[5])),
        )

    def _next_spawn_index(self) -> int:
        if not self._spawn_queue:
            self._spawn_queue.extend(np.random.permutation(len(EGO_SPAWN_POINT)))
        return int(self._spawn_queue.popleft())

    def reset_vehicle(self) -> None:
        """
        Spawn the ego vehicle at the current ego transform.
        Raises RuntimeError if spawning fails.
        """
        vehicle_bp = self.blueprint_library.find("vehicle.tesla.model3")
        ego = self.world.try_spawn_actor(vehicle_bp, self.ego_transform)
        if ego is None:
            raise RuntimeError("Unable to spawn ego vehicle at the chosen spawn point.")
        self.ego = ego
        self.ego.set_target_velocity(carla.Vector3D())

    # --------------------------------------------------------------------------
    # Sensors
    # --------------------------------------------------------------------------

    def setup_camera(self) -> None:
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

        def _on_image(img: carla.Image, q=self._img_q, ws=ws):
            env = ws()
            if env is None:
                return
            arr = np.frombuffer(img.raw_data, dtype=np.uint8).reshape((img.height, img.width, 4))
            rgb = arr[:, :, :3][:, :, ::-1].copy()  # BGRA -> RGB
            item = (int(img.frame), rgb)
            try:
                q.put_nowait(item)
            except queue.Full:
                # If the consumer is too slow, drop frames rather than blocking CARLA.
                pass

        self.camera_sensor.listen(_on_image)

    def setup_collision_sensor(self) -> None:
        bp = self.blueprint_library.find("sensor.other.collision")
        self.colsensor = self.world.spawn_actor(bp, carla.Transform(), attach_to=self.ego)
        self.actors.append(self.colsensor)
        self.colsensor.listen(lambda event: self._on_collision(event))

    def setup_lane_invasion_sensor(self) -> None:
        bp = self.blueprint_library.find("sensor.other.lane_invasion")
        self.lane_sensor = self.world.spawn_actor(bp, carla.Transform(), attach_to=self.ego)
        self.actors.append(self.lane_sensor)
        self.lane_sensor.listen(lambda event: self._on_lane_invasion(event))

    def _on_collision(self, event: carla.CollisionEvent) -> None:
        self.collision_hist.append(event)
        self.collision_detected = True  # latched for the episode

    def _on_lane_invasion(self, event: carla.LaneInvasionEvent) -> None:
        self.lane_invasion_hist.append(event)
        self.lane_invasion_detected = True  # latched for the episode

    # --------------------------------------------------------------------------
    # Observation
    # --------------------------------------------------------------------------

    def _pull_image_for_frame(self, target_frame: Optional[int], timeout: float = 0.5) -> np.ndarray:
        """
        Pull the best available image from the queue, attempting to match `target_frame`.
        """
        if self._img_q is None:
            self._got_frame = False
            return np.zeros((512, 512, 3), dtype=np.uint8)

        end = time.time() + timeout
        best: Optional[np.ndarray] = None
        got_any = False

        while time.time() < end:
            try:
                f, rgb = self._img_q.get(timeout=max(0.0, end - time.time()))
                got_any = True
                if target_frame is None:
                    best = rgb
                else:
                    # Prefer exact match; otherwise accept the nearest frame >= target
                    if f == target_frame:
                        best = rgb
                        break
                    if f > target_frame:
                        best = rgb
                        break
                    best = rgb
            except queue.Empty:
                break

        self._got_frame = bool(got_any and best is not None)
        if best is None:
            best = np.zeros((512, 512, 3), dtype=np.uint8)
        return best

    def _get_observation(self, frame_id: Optional[int] = None) -> Dict[str, np.ndarray]:
        rgb = self._pull_image_for_frame(frame_id, timeout=0.5)
        self.camera_image2 = rgb
        self.camera_image = cv2.resize(rgb, (128, 128), interpolation=cv2.INTER_AREA)

        return {
            "image": self.camera_image,
            "collision": np.array([1.0 if self.collision_detected else 0.0], dtype=np.float32),
            "lane_invasion": np.array([1.0 if self.lane_invasion_detected else 0.0], dtype=np.float32),
        }

    # --------------------------------------------------------------------------
    # Control
    # --------------------------------------------------------------------------

    def apply_control(self, action: int) -> None:
        control = self._get_vehicle_control(action)
        self.ego.apply_control(control)

    def _get_vehicle_control(self, action: int) -> carla.VehicleControl:
        acc = float(DISCRETE_ACC[action // self.n_steer])
        steer = float(DISCRETE_STEER[action % self.n_steer])
        return carla.VehicleControl(throttle=acc, steer=steer, brake=0.0)

    # --------------------------------------------------------------------------
    # Map / lane helpers
    # --------------------------------------------------------------------------

    def _get_lane_info(self, loc: carla.Location, project_to_road: bool) -> Optional[Dict[str, Any]]:
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

    # --------------------------------------------------------------------------
    # Reference path construction and geometry
    # --------------------------------------------------------------------------

    def _build_spawn_lane_reference(self) -> None:
        """Build a polyline reference along the spawn lane towards the goal."""
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

            if (last_xy is None) or (math.hypot(xy[0] - last_xy[0], xy[1] - last_xy[1]) > 1e-3):
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
                # Choose successor waypoint that minimizes straight-line distance to goal
                best = min(nxt, key=lambda cand: float(cand.transform.location.distance(goal_loc)))
                wp = best

        if len(pts) < 2:
            ego_loc = self.ego.get_location()
            pts = [(float(ego_loc.x), float(ego_loc.y)), (float(goal_loc.x), float(goal_loc.y))]

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

    def _ref_lateral_error_and_progress(
        self,
        ego_xy: np.ndarray,
        search_back: int = 10,
        search_fwd: int = 40,
    ) -> Tuple[float, float, np.ndarray]:
        """
        Project ego position onto reference polyline; return:
            lane_d: lateral distance to polyline (m)
            s: arc-length progress along reference (m)
            ref_dir: local tangent direction (unit vector in xy)
        """
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

        # If the local search fails badly, do a global scan once.
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

        self._ref_seg_idx = int(best_i)
        self._ref_s = float(self._ref_cum_s[best_i] + best_t * self._ref_seg_len[best_i])

        lane_d = float(math.sqrt(best_d2))
        return lane_d, self._ref_s, best_dir

    def _heading_error_to_ref_dir(self, ref_dir: np.ndarray) -> float:
        """
        Normalized heading error in [0,1], where 0 = aligned and 1 = opposite.
        """
        fwd = self.ego.get_transform().get_forward_vector()
        ev = np.array([float(fwd.x), float(fwd.y)], dtype=np.float32)
        n = float(np.linalg.norm(ev))
        if n < 1e-6:
            return 1.0
        ev /= n
        dot = float(np.clip(np.dot(ev, ref_dir), -1.0, 1.0))
        ang = math.acos(dot)
        return float(ang / math.pi)

    # --------------------------------------------------------------------------
    # Offroad / past-goal helpers
    # --------------------------------------------------------------------------

    def _compute_offroad_flag(self) -> Tuple[bool, float]:
        """
        Offroad is defined as:
          - strict waypoint (project_to_road=False) is None AND
          - distance from projected lane center > offroad_center_thresh_m
        """
        veh_loc = self.ego.get_location()

        strict_wp = self.map.get_waypoint(veh_loc, project_to_road=False, lane_type=carla.LaneType.Driving)
        proj_wp = self.map.get_waypoint(veh_loc, project_to_road=True, lane_type=carla.LaneType.Driving)

        if proj_wp is None:
            return True, 1e9

        c = proj_wp.transform.location
        proj_off = float(math.hypot(float(veh_loc.x - c.x), float(veh_loc.y - c.y)))
        offroad = (strict_wp is None) and (proj_off > float(self.offroad_center_thresh_m))
        return bool(offroad), proj_off

    def _signed_ahead_to_goal_m(self) -> float:
        """
        Signed distance along a goal-aligned direction:
            >0 means ego is "ahead" of the goal along the route direction.
        Used for past-goal termination.
        """
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

    # --------------------------------------------------------------------------
    # Reward
    # --------------------------------------------------------------------------

    def _compute_reward(self) -> Tuple[float, Dict[str, float]]:
        """
        Reward components:
          - route_progress: progress along reference path
          - lane: quadratic penalty for lateral deviation from reference
          - wrong_lane: penalty if lateral deviation exceeds lane_change_threshold_m
          - heading: quadratic penalty for heading error w.r.t. reference direction
          - r_speed: L1 penalty in (parallel, perpendicular) velocity components
          - low_speed: penalty if below threshold for extended duration
          - offroad/invasion/collision: event penalties
          - goal: bonus if within goal radius
          - time: per-step time penalty
        """
        rc: Dict[str, float] = {}
        self._lane_invasion_step = 0
        self._collision_step = 0

        # (1) Event deltas (new events since last step)
        new_inv = int(len(self.lane_invasion_hist) - self.previous_lane_invasions)
        self.previous_lane_invasions += max(new_inv, 0)
        self._lane_invasion_step = max(new_inv, 0)

        new_col = int(len(self.collision_hist) - self.previous_collisions)
        self.previous_collisions += max(new_col, 0)
        self._collision_step = max(new_col, 0)

        # (2) Offroad and reference construction
        offroad, _proj_off = self._compute_offroad_flag()
        if self._ref_xy is None or len(self._ref_xy) < 2:
            self._build_spawn_lane_reference()

        # (3) Geometry relative to reference
        veh_loc = self.ego.get_location()
        ego_xy = np.array([float(veh_loc.x), float(veh_loc.y)], dtype=np.float32)

        lane_d, s, ref_dir = self._ref_lateral_error_and_progress(ego_xy)
        self._max_ref_s = max(float(getattr(self, "_max_ref_s", 0.0)), float(s))
        heading_err = self._heading_error_to_ref_dir(ref_dir)

        self._last_lane_d = float(lane_d)
        self._last_heading_err = float(heading_err)

        # (4) Reference progress reward
        delta_s = float(s - self._prev_ref_s)
        self._prev_ref_s = float(s)
        rc["route_progress"] = float(self.k_route_progress) * float(np.clip(delta_s, -0.5, 0.5))

        # (5) Lane centering penalty
        lane_rad = min(float(lane_d), float(self.lane_center_cap_m))
        rc["lane"] = -float(self.k_lane_center) * (lane_rad ** 2)

        # (6) Wrong-lane flag (purely geometric heuristic)
        wrong_lane = (not offroad) and (lane_d > float(self.lane_change_threshold_m))
        self._offroad = bool(offroad)
        self._wrong_lane = bool(wrong_lane)
        rc["wrong_lane"] = -float(self.k_wrong_lane) * float(wrong_lane)

        # (7) Heading penalty
        head_rad = min(float(heading_err), float(self.heading_cap))
        rc["heading"] = -float(self.k_heading) * (head_rad ** 2)

        # (8) Speed shaping (parallel/perpendicular to reference direction)
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

        # (9) Low-speed penalty (step-based)
        if self.low_speed_steps > 0:
            rc["low_speed"] = -50.0 if (self.low_speed_steps >= self.low_speed_timeout_steps) else -2.0
        else:
            rc["low_speed"] = 0.0

        # (10) Safety penalties
        rc["offroad"] = -float(self.k_offroad) * float(offroad)
        rc["invasion"] = -float(self.k_invasion) * float(max(new_inv, 0))
        rc["collision"] = -float(self.k_collision) * float(max(new_col, 0))

        # (11) Goal bonus + time penalty
        dist_goal = float(self.ego.get_location().distance(self.end_point.location))
        goal_r = float(_get_cfg(self._config, "goal_radius_m", 2.0))
        rc["goal"] = float(self.goal_bonus) if dist_goal < goal_r else 0.0
        rc["time"] = -float(self.time_penalty)

        # Track dist for progress logs in step()
        self.last_distance_to_goal = dist_goal

        total = float(sum(rc.values()))
        return total, rc

    # --------------------------------------------------------------------------
    # Termination
    # --------------------------------------------------------------------------

    def _check_termination(self) -> Tuple[bool, Dict[str, Any]]:
        collision = bool(self.collision_detected)

        goal_radius_m = float(_get_cfg(self._config, "goal_radius_m", 2.0))
        reached_goal = (self.ego.get_location().distance(self.end_point.location) < goal_radius_m)

        time_exceeded = (self._time_step >= self._max_time_step)
        offroad_done = (self.offroad_steps >= self.max_offroad_steps)
        wrong_lane_done = (self.wrong_lane_steps >= self.max_wrong_lane_steps)
        stuck_too_long = (self.low_speed_steps >= self.low_speed_timeout_steps)

        # Past-goal logic (requires having been near the goal, then moving away while "ahead")
        past_goal_margin_m = float(_get_cfg(self._config, "past_goal_margin_m", 1.0))
        past_goal_timeout_s = float(_get_cfg(self._config, "past_goal_timeout_s", 1.0))
        past_goal_steps_needed = int(round(past_goal_timeout_s / float(self._fixed_dt)))

        ahead_m = self._signed_ahead_to_goal_m()
        past_now = (
            bool(self._ever_near_goal)
            and (ahead_m > past_goal_margin_m)
            and (self._dist_to_goal_m is not None)
            and (self._dist_to_goal_m > goal_radius_m)
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

        info: Dict[str, Any] = {
            "collision": 0,
            "goal_reached": 0,
            "offroad": 0,
            "wrong_lane": 0,
            "past_goal": 0,
            "not_moving": 0,
            "time_exceeded": 0,
            "stuck_duration": 0.0,
            "elapsed_steps": 0,

            # episode-level metrics
            "distance_m": float(self.travel_distance_m),
            "distance_km": float(self.travel_distance_m) / 1000.0,
            "lane_invasions": int(self.previous_lane_invasions),
            "mean_off_center_m": float(self.off_center_sum / self.off_center_steps) if self.off_center_steps > 0 else 0.0,
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

        return bool(done), info

    # --------------------------------------------------------------------------
    # Cleanup
    # --------------------------------------------------------------------------

    def _clean_actors(self) -> None:
        """
        Stop sensors (best effort) and destroy all actors spawned by this env.
        """
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
                if a is not None and getattr(a, "is_alive", False):
                    batch.append(carla.command.DestroyActor(a))

            if batch:
                self.client.apply_batch_sync(batch, True)

            self.actors.clear()
            self.ego = None
            self.camera_sensor = None
            self.colsensor = None
            self.lane_sensor = None
            self._img_q = None
            self._got_frame = False

            # Tick once to flush destruction in synchronous mode
            if self._sync_enabled:
                snap = self.world.tick()
                self._last_tick_frame = int(snap.frame if hasattr(snap, "frame") else int(snap))

        except Exception as e:
            print(f"Cleanup error: {e}")
