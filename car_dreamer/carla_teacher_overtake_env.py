import gym
import math
import time
import queue
import weakref
from collections import deque
from typing import Any, Dict, Optional, Tuple

import numpy as np
import cv2
import carla
from gym import spaces

from .toolkit.planner import FixedEndingPlanner


# ==============================================================================
# Scenario geometry (Town04)
# ==============================================================================

EGO_SPAWN_POINT = [
    [-16.890745162963867, -211.24720764160156, 0.2819424271583557, 0.0, 89.7751235961914, 0.0],
    [-13.395880699157715, -212.56092834472656, 0.2819424271583557, 0.0, 89.7751235961914, 0.0],
    [-9.890790939331055,  -211.27468872070312, 0.2819424271583557, 0.0, 89.7751235961914, 0.0],
    [-6.395920276641846, -212.58840942382812, 0.2819424271583557, 0.0, 89.7751235961914, 0.0],
]

LEAD_SPAWN_POINT = [
    [-16.712175369262695,  -195.74740600585938, 0.2819424271583557, 0.0, 89.7751235961914, 0.0],
    [-13.221610069274902,  -198.1611328125,     0.2819424271583557, 0.0, 89.7751235961914, 0.0],
    [-9.712218284606934,   -195.77487182617188, 0.2819424271583557, 0.0, 89.7751235961914, 0.0],
    [-6.2216644287109375,  -198.18861389160156, 0.2819424271583557, 0.0, 89.7751235961914, 0.0],
]

EGO_END_POINT = [
    [-16.520898818969727, -175.01853942871094, 0.2819424271583557, 0.0, 89.77516174316406, 0.0],
    [-13.030336380004883, -175.4322738647461,  0.2819424271583557, 0.0, 89.77516174316406, 0.0],
    [-9.520939826965332,  -175.04601287841797, 0.2819424271583557, 0.0, 89.77516174316406, 0.0],
    [-6.030386447906494,  -175.45975494384766, 0.2819424271583557, 0.0, 89.77516174316406, 0.0],
]


# ==============================================================================
# Discrete action set (ego)
# ==============================================================================

DISCRETE_ACC = [0.0, 0.3]                       # throttle levels
DISCRETE_STEER = [-0.2, -0.1, 0.0, 0.1, 0.2]    # steering levels


# ==============================================================================
# Utilities
# ==============================================================================

def _get_cfg(obj: Any, path: str, default: Any = None) -> Any:
    """Nested config getter for dict-like or attribute-like configs."""
    cur = obj
    for key in path.split("."):
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(key, None)
        else:
            cur = getattr(cur, key, None)
    return default if cur is None else cur


def _make_transform(p: list) -> carla.Transform:
    """Create a CARLA Transform from [x,y,z,pitch,yaw,roll]."""
    return carla.Transform(
        carla.Location(x=float(p[0]), y=float(p[1]), z=float(p[2])),
        carla.Rotation(pitch=float(p[3]), yaw=float(p[4]), roll=float(p[5])),
    )


def distance_2d(a: carla.Location, b: carla.Location) -> float:
    """2D Euclidean distance in the xy-plane (meters)."""
    dx = float(a.x - b.x)
    dy = float(a.y - b.y)
    return math.sqrt(dx * dx + dy * dy)


def clamp(x: float, lo: float, hi: float) -> float:
    return float(min(max(x, lo), hi))


# ==============================================================================
# Environment
# ==============================================================================

class CarlaTeacherOvertakeEnv(gym.Env):
    """
    Overtaking environment (Town04).

    Setup:
      - Ego spawns behind a lead vehicle in one of four lanes.
      - Ego must pass (overtake) and typically return to the original lane.
      - A short goal point is placed ahead.

    Action space (Discrete):
      action -> (throttle, steer), with throttle in DISCRETE_ACC, steer in DISCRETE_STEER.

    Observation space (Dict):
      image: uint8 RGB (128,128,3) from a top-down semantic segmentation camera
      collision: int {0,1} (latched for episode)
      lane_invasion: int {0,1} (latched for episode)

    Reward (default design):
      - progress-to-goal shaping (distance reduction)
      - speed tracking shaping
      - sparse event rewards: passed lead, returned to original lane, overtake completed
      - penalties: collisions, offroad, lane invasions, time, low-speed (stuck)

    Notes:
      - Uses synchronous mode.
      - Camera frames are pulled by frame id for determinism.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, config: Any):
        super().__init__()
        self._config = config

        # ----------------------------
        # CARLA connection + world
        # ----------------------------
        host = str(_get_cfg(config, "carla_host", "localhost"))
        port = int(_get_cfg(config, "carla_port", 3000))
        timeout_s = float(_get_cfg(config, "carla_timeout_s", 60.0))
        town = str(_get_cfg(config, "town", "Town04"))

        self.client = carla.Client(host, port)
        self.client.set_timeout(timeout_s)

        world = self.client.get_world()
        try:
            map_name = world.get_map().name
        except RuntimeError:
            map_name = ""

        if not str(map_name).endswith(town):
            world = self.client.load_world(town)

        self.world = world
        self.map = self.world.get_map()
        self.blueprint_library = self.world.get_blueprint_library()

        # Remove parked vehicles layer if supported
        try:
            self.world.unload_map_layer(carla.MapLayer.ParkedVehicles)
        except Exception:
            pass

        # Store original settings
        orig = self.world.get_settings()
        self._orig_sync = bool(orig.synchronous_mode)
        self._orig_dt = orig.fixed_delta_seconds

        # Enable synchronous mode
        self._fixed_dt = float(_get_cfg(config, "fixed_delta_seconds", 0.1))
        s = self.world.get_settings()
        if (not s.synchronous_mode) or (s.fixed_delta_seconds != self._fixed_dt):
            s.synchronous_mode = True
            s.fixed_delta_seconds = self._fixed_dt
            self.world.apply_settings(s)

        self._sync_enabled = True

        # ----------------------------
        # Rendering / display
        # ----------------------------
        self.enable_display = bool(_get_cfg(config, "enable_display", False))

        # ----------------------------
        # Episode state
        # ----------------------------
        self.ego: Optional[carla.Vehicle] = None
        self.lead: Optional[carla.Vehicle] = None
        self.actors = []  # ego + lead + sensors for cleanup

        # Sensors
        self.camera_sensor: Optional[carla.Sensor] = None
        self.colsensor: Optional[carla.Sensor] = None
        self.lane_sensor: Optional[carla.Sensor] = None

        self._img_q: Optional[queue.Queue] = None
        self._last_tick_frame: Optional[int] = None
        self._got_frame: bool = False

        self.camera_image: Optional[np.ndarray] = None    # 128x128
        self.camera_image2: Optional[np.ndarray] = None   # 512x512

        # Latching event flags
        self.collision_detected: bool = False
        self.lane_invasion_detected: bool = False
        self.collision_hist = []
        self.lane_invasion_hist = []
        self.previous_collisions = 0
        self.previous_lane_invasions = 0
        self._collision_step = 0
        self._lane_invasion_step = 0

        # Discrete spaces
        self.action_space = self._setup_action_space()
        self.observation_space = self._setup_observation_space()

        # Planner (optional shaping / debug)
        self.ego_planner: Optional[FixedEndingPlanner] = None
        self.waypoints = []
        self.planner_stats: Dict[str, Any] = {}
        self.num_completed: int = 0

        # Spawn scheduling
        self._spawn_queue = deque(np.random.permutation(len(EGO_SPAWN_POINT)))
        self.spawn_index: int = int(self._spawn_queue.popleft())

        # Scenario transforms
        self.ego_transform = _make_transform(EGO_SPAWN_POINT[self.spawn_index])
        self.lead_transform = _make_transform(LEAD_SPAWN_POINT[self.spawn_index])
        self.end_point = _make_transform(EGO_END_POINT[self.spawn_index])

        # Control and counters
        self._time_step = 0
        self._max_time_step = int(_get_cfg(config, "max_time_step", 500))

        # Speed / stuck detection (step-based, not wall-clock)
        self.low_speed_kmh_thresh = float(_get_cfg(config, "low_speed_kmh_thresh", 1.0))
        self.low_speed_timeout_s = float(_get_cfg(config, "low_speed_timeout_s", 10.0))
        self.low_speed_timeout_steps = int(round(self.low_speed_timeout_s / self._fixed_dt))
        self.low_speed_steps = 0

        # Reward parameters
        self.desired_speed_mps = float(_get_cfg(config, "desired_speed_mps", 5.0))
        self.k_speed = float(_get_cfg(config, "k_speed", 1.0))
        self.k_progress = float(_get_cfg(config, "k_progress", 60.0))
        self.time_penalty = float(_get_cfg(config, "time_penalty", 0.05))

        self.R_pass = float(_get_cfg(config, "R_pass", 50.0))
        self.R_return = float(_get_cfg(config, "R_return", 50.0))
        self.R_overtake = float(_get_cfg(config, "R_overtake", 200.0))
        self.R_collision = float(_get_cfg(config, "R_collision", 200.0))
        self.R_offroad = float(_get_cfg(config, "R_offroad", 50.0))
        self.k_invasion = float(_get_cfg(config, "k_invasion", 5.0))

        self.goal_radius_m = float(_get_cfg(config, "goal_radius_m", 2.0))
        self.terminate_on_overtake = bool(_get_cfg(config, "terminate_on_overtake", False))

        # Offroad definition
        self.offroad_center_thresh_m = float(_get_cfg(config, "offroad_center_thresh_m", 2.2))

        # Lead vehicle control (simple longitudinal + lane-centering controller)
        self.lead_target_speed_mps = float(_get_cfg(config, "lead_target_speed_mps", 2.0))
        self.lead_speed_kp = float(_get_cfg(config, "lead_speed_kp", 0.4))
        self.lead_brake_kp = float(_get_cfg(config, "lead_brake_kp", 0.6))
        self.lead_steer_kp = float(_get_cfg(config, "lead_steer_kp", 0.3))
        self.lead_steer_kd = float(_get_cfg(config, "lead_steer_kd", 0.1))
        self._lead_prev_lat_err = 0.0

        # Overtake event tracking
        self._road_start_xy = None
        self._road_dir = None
        self._orig_lane_id = None
        self._orig_road_id = None
        self._orig_section_id = None

        self.exceeded = False
        self.left_original_lane = False
        self.returned = False
        self.overtake = False

        self.exceed_event_step = False
        self.return_event_step = False
        self.overtake_event_step = False

        # Distance metrics
        self.prev_ego_location: Optional[carla.Location] = None
        self.travel_distance_m: float = 0.0
        self.min_ego_lead_dist_m: float = float("inf")
        self.last_distance_to_goal: Optional[float] = None

        # Prime world
        snap = self.world.tick()
        self._last_tick_frame = int(snap.frame if hasattr(snap, "frame") else int(snap))

        print("CARLA environment initialized")
        print("Map name:", self.map.name)

    # --------------------------------------------------------------------------
    # Gym API
    # --------------------------------------------------------------------------

    def reset(self):
        self._clean_actors()

        # Spawn selection
        if not self._spawn_queue:
            self._spawn_queue = deque(np.random.permutation(len(EGO_SPAWN_POINT)))
        self.spawn_index = int(self._spawn_queue.popleft())

        self.ego_transform = _make_transform(EGO_SPAWN_POINT[self.spawn_index])
        self.lead_transform = _make_transform(LEAD_SPAWN_POINT[self.spawn_index])
        self.end_point = _make_transform(EGO_END_POINT[self.spawn_index])

        # Reset counters
        self._time_step = 0
        self.collision_detected = False
        self.lane_invasion_detected = False
        self.collision_hist = []
        self.lane_invasion_hist = []
        self.previous_collisions = 0
        self.previous_lane_invasions = 0
        self._collision_step = 0
        self._lane_invasion_step = 0

        self.low_speed_steps = 0

        self.exceeded = False
        self.left_original_lane = False
        self.returned = False
        self.overtake = False
        self.exceed_event_step = False
        self.return_event_step = False
        self.overtake_event_step = False

        self._lead_prev_lat_err = 0.0

        self.prev_ego_location = None
        self.travel_distance_m = 0.0
        self.min_ego_lead_dist_m = float("inf")

        # Tick before spawn (flush)
        snap = self.world.tick()
        self._last_tick_frame = int(snap.frame if hasattr(snap, "frame") else int(snap))

        # Spawn vehicles
        self._spawn_ego()
        self._spawn_lead()

        # Attach sensors
        self._setup_collision_sensor()
        self._setup_lane_invasion_sensor()
        self._setup_camera()

        # One tick so sensors produce frames
        snap = self.world.tick()
        self._last_tick_frame = int(snap.frame if hasattr(snap, "frame") else int(snap))

        # Planner (optional)
        dest_location = self.end_point.location
        self.ego_planner = FixedEndingPlanner(self.ego, dest_location)
        self.waypoints, self.planner_stats = self.ego_planner.run_step()
        self.num_completed = int(self.planner_stats.get("num_completed", 0))

        # Compute road direction for longitudinal projection (start -> goal)
        start_xy = np.array([float(self.ego_transform.location.x), float(self.ego_transform.location.y)], dtype=np.float32)
        goal_xy = np.array([float(self.end_point.location.x), float(self.end_point.location.y)], dtype=np.float32)
        d = goal_xy - start_xy
        n = float(np.linalg.norm(d))
        self._road_dir = (d / n) if n > 1e-6 else np.array([0.0, 1.0], dtype=np.float32)
        self._road_start_xy = start_xy

        # Record original lane identity (robust lane-return detection)
        wp0 = self.map.get_waypoint(self.ego_transform.location, project_to_road=True, lane_type=carla.LaneType.Driving)
        if wp0 is not None:
            self._orig_road_id = int(wp0.road_id)
            self._orig_section_id = int(wp0.section_id)
            self._orig_lane_id = int(wp0.lane_id)
        else:
            self._orig_road_id = None
            self._orig_section_id = None
            self._orig_lane_id = None

        # Initialize distance-to-goal
        self.last_distance_to_goal = float(self.ego.get_location().distance(self.end_point.location))

        # Initialize distance traveled
        self.prev_ego_location = self.ego.get_location()

        obs = self._get_observation(self._last_tick_frame)
        return obs

    def step(self, action: int):
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}")

        self.exceed_event_step = False
        self.return_event_step = False
        self.overtake_event_step = False

        # Apply control (ego + lead)
        self._apply_ego_control(action)
        self._apply_lead_control()

        self._time_step += 1

        # Tick simulation
        snap = self.world.tick()
        frame_id = int(snap.frame if hasattr(snap, "frame") else int(snap))
        self._last_tick_frame = frame_id

        # Refresh observation AFTER tick (correct temporal alignment)
        obs = self._get_observation(frame_id)

        # Per-step event deltas
        new_col = int(len(self.collision_hist) - self.previous_collisions)
        self.previous_collisions += max(new_col, 0)
        self._collision_step = max(new_col, 0)

        new_inv = int(len(self.lane_invasion_hist) - self.previous_lane_invasions)
        self.previous_lane_invasions += max(new_inv, 0)
        self._lane_invasion_step = max(new_inv, 0)

        # Speed + stuck tracking (step-based)
        v = self.ego.get_velocity()
        speed_mps = math.sqrt(float(v.x * v.x + v.y * v.y + v.z * v.z))
        speed_kmh = 3.6 * speed_mps
        if speed_kmh < self.low_speed_kmh_thresh:
            self.low_speed_steps += 1
        else:
            self.low_speed_steps = 0

        # Distance traveled metrics
        cur_loc = self.ego.get_location()
        step_dist = distance_2d(cur_loc, self.prev_ego_location) if self.prev_ego_location is not None else 0.0
        self.travel_distance_m += float(step_dist)
        self.prev_ego_location = cur_loc

        # Ego–lead distance
        if self.lead is not None and self.lead.is_alive:
            lead_loc = self.lead.get_location()
            ego_lead_dist = distance_2d(cur_loc, lead_loc)
            self.min_ego_lead_dist_m = min(self.min_ego_lead_dist_m, ego_lead_dist)
        else:
            ego_lead_dist = float("inf")

        # Planner update (optional)
        self.waypoints, self.planner_stats = self.ego_planner.run_step()
        self.num_completed = int(self.planner_stats.get("num_completed", 0))

        # Overtake event logic (robust)
        self._update_overtake_events()

        # Reward
        reward, r_terms = self._compute_reward(speed_mps=speed_mps)

        # Termination
        done, term_info = self._check_termination()

        # Optional: terminate immediately on overtake completion
        if self.terminate_on_overtake and self.overtake:
            done = True
            term_info["overtake_done"] = True

        # Visualization (optional; default off)
        if self.enable_display and self.camera_image2 is not None:
            frame = self.camera_image2.copy()
            cv2.putText(
                frame,
                f"spawn={self.spawn_index} pass={int(self.exceeded)} ret={int(self.returned)} ovt={int(self.overtake)}",
                (16, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            resized = cv2.resize(frame, (512, 512), interpolation=cv2.INTER_NEAREST)
            cv2.imshow("OvertakeCamera", resized)
            cv2.waitKey(1)

        info = {}
        info.update({f"r_{k}": float(v) for k, v in r_terms.items()})
        info.update({f"term_{k}": v for k, v in term_info.items()})

        info.update({
            "speed_kmh": float(speed_kmh),
            "step_distance_m": float(step_dist),
            "travel_distance_m": float(self.travel_distance_m),
            "ego_lead_dist_m": float(ego_lead_dist),
            "min_ego_lead_dist_m": float(self.min_ego_lead_dist_m),

            "collision_step": int(self._collision_step),
            "lane_invasion_step": int(self._lane_invasion_step),

            "passed_lead": int(self.exceeded),
            "returned_lane": int(self.returned),
            "overtake": int(self.overtake),

            "passed_event": int(self.exceed_event_step),
            "return_event": int(self.return_event_step),
            "overtake_event": int(self.overtake_event_step),
        })

        return obs, float(reward), bool(done), info

    def close(self):
        """Destroy actors and restore CARLA world settings."""
        try:
            s = self.world.get_settings()
            s.synchronous_mode = self._orig_sync
            s.fixed_delta_seconds = self._orig_dt
            self.world.apply_settings(s)
            self._sync_enabled = False
        finally:
            self._clean_actors()

    # --------------------------------------------------------------------------
    # Spaces
    # --------------------------------------------------------------------------

    def _setup_action_space(self) -> spaces.Discrete:
        self.n_steer = len(DISCRETE_STEER)
        self.n_acc = len(DISCRETE_ACC)
        return spaces.Discrete(self.n_steer * self.n_acc)

    def _setup_observation_space(self) -> spaces.Dict:
        return spaces.Dict({
            "image": spaces.Box(low=0, high=255, shape=(128, 128, 3), dtype=np.uint8),
            "collision": spaces.Discrete(2),
            "lane_invasion": spaces.Discrete(2),
        })

    # --------------------------------------------------------------------------
    # Spawning
    # --------------------------------------------------------------------------

    def _spawn_ego(self) -> None:
        bp = self.blueprint_library.find("vehicle.tesla.model3")
        ego = self.world.try_spawn_actor(bp, self.ego_transform)
        if ego is None:
            raise RuntimeError("Failed to spawn ego vehicle.")
        self.ego = ego
        self.actors.append(ego)

        # Initialize control state
        self.ego.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=0.0))

    def _spawn_lead(self) -> None:
        bp = self.blueprint_library.find("vehicle.tesla.model3")

        # Retry a few times in case of transient spawn collisions
        max_attempts = int(_get_cfg(self._config, "lead_spawn_attempts", 10))
        lead = None
        for _ in range(max_attempts):
            lead = self.world.try_spawn_actor(bp, self.lead_transform)
            if lead is not None:
                break
            self.world.tick()

        if lead is None:
            raise RuntimeError("Failed to spawn lead vehicle after multiple attempts.")

        self.lead = lead
        self.actors.append(lead)

        # Initialize lead control state
        self.lead.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=0.0))

    # --------------------------------------------------------------------------
    # Sensors
    # --------------------------------------------------------------------------

    def setup_camera(self):
        """
        Configure a front-facing RGB camera and stream frames into a queue.
        The observation will use RGB ordering (not BGR).
        """

        cam_bp = self.blueprint_library.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", "512")
        cam_bp.set_attribute("image_size_y", "512")
        cam_bp.set_attribute("fov", "110")

        # Keep camera aligned with synchronous stepping
        fixed_dt = float(getattr(self.world.get_settings(), "fixed_delta_seconds", 0.1) or 0.1)
        cam_bp.set_attribute("sensor_tick", f"{fixed_dt}")

        # Front camera: slightly ahead of the vehicle origin, above hood, slight downward pitch
        camera_spawn = carla.Transform(
            carla.Location(x=1.5, z=1.6),
            carla.Rotation(pitch=-5.0, yaw=0.0, roll=0.0),
        )

        self.camera_sensor = self.world.spawn_actor(cam_bp, camera_spawn, attach_to=self.ego)
        self.actors.append(self.camera_sensor)

        self._img_q = queue.Queue(maxsize=8)
        ws = weakref.ref(self)

        def _on_image(img, q=self._img_q, ws=ws):
            s = ws()
            if s is None:
                return

            # CARLA RGB camera raw_data is BGRA (uint8)
            arr = np.frombuffer(img.raw_data, dtype=np.uint8).reshape((img.height, img.width, 4))
            rgb = arr[:, :, :3][:, :, ::-1].copy()  # BGRA -> RGB

            item = (int(img.frame), rgb)
            try:
                q.put_nowait(item)
            except queue.Full:
                # Drop oldest and insert newest (keeps freshest frame)
                try:
                    q.get_nowait()
                except queue.Empty:
                    pass
                try:
                    q.put_nowait(item)
                except queue.Full:
                    pass

        self.camera_sensor.listen(_on_image)

    # Warm-up: ensure at least one frame arrives so reset() does not return black
    for _ in range(3):
        self.world.tick()
        try:
            self._img_q.get(timeout=0.5)
        except queue.Empty:
            pass


    def _setup_collision_sensor(self) -> None:
        bp = self.blueprint_library.find("sensor.other.collision")
        self.colsensor = self.world.spawn_actor(bp, carla.Transform(), attach_to=self.ego)
        self.actors.append(self.colsensor)
        self.colsensor.listen(lambda e: self._on_collision(e))

    def _setup_lane_invasion_sensor(self) -> None:
        bp = self.blueprint_library.find("sensor.other.lane_invasion")
        self.lane_sensor = self.world.spawn_actor(bp, carla.Transform(), attach_to=self.ego)
        self.actors.append(self.lane_sensor)
        self.lane_sensor.listen(lambda e: self._on_lane_invasion(e))

    def _on_collision(self, event) -> None:
        self.collision_hist.append(event)
        self.collision_detected = True

    def _on_lane_invasion(self, event) -> None:
        self.lane_invasion_hist.append(event)
        self.lane_invasion_detected = True

    # --------------------------------------------------------------------------
    # Observation
    # --------------------------------------------------------------------------

    def _pull_image_for_frame(self, target_frame: Optional[int], timeout: float = 0.5) -> np.ndarray:
        """Pull best image from queue, attempting to match `target_frame`."""
        if self._img_q is None:
            self._got_frame = False
            return np.zeros((512, 512, 3), dtype=np.uint8)

        end = time.time() + timeout
        best = None
        got_any = False

        while time.time() < end:
            try:
                f, rgb = self._img_q.get(timeout=max(0.0, end - time.time()))
                got_any = True
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

        self._got_frame = bool(got_any and best is not None)
        if best is None:
            best = np.zeros((512, 512, 3), dtype=np.uint8)
        return best

    def _get_observation(self, frame_id: Optional[int]) -> Dict[str, Any]:
        rgb = self._pull_image_for_frame(frame_id, timeout=0.5)
        self.camera_image2 = rgb
        self.camera_image = cv2.resize(rgb, (128, 128), interpolation=cv2.INTER_AREA)

        return {
            "image": self.camera_image,
            "collision": int(self.collision_detected),
            "lane_invasion": int(self.lane_invasion_detected),
        }

    # --------------------------------------------------------------------------
    # Control
    # --------------------------------------------------------------------------

    def _apply_ego_control(self, action: int) -> None:
        acc = float(DISCRETE_ACC[action // self.n_steer])
        steer = float(DISCRETE_STEER[action % self.n_steer])
        self.ego.apply_control(carla.VehicleControl(throttle=acc, steer=steer, brake=0.0))

    def _apply_lead_control(self) -> None:
        """
        Simple lead controller:
          - Longitudinal: proportional speed tracking to `lead_target_speed_mps`
          - Lateral: proportional + derivative control to lane center (using waypoint)
        """
        if self.lead is None or (not self.lead.is_alive):
            return

        lead_loc = self.lead.get_location()
        wp = self.map.get_waypoint(lead_loc, project_to_road=True, lane_type=carla.LaneType.Driving)
        if wp is None:
            # If lead is offroad for any reason, stop it
            self.lead.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0))
            return

        center = wp.transform.location
        right = wp.transform.get_right_vector()
        r = np.array([float(right.x), float(right.y)], dtype=np.float32)
        rn = float(np.linalg.norm(r)) if float(np.linalg.norm(r)) > 1e-6 else 1.0
        r /= rn

        # Signed lateral error (meters): right positive, left negative
        v = np.array([float(lead_loc.x - center.x), float(lead_loc.y - center.y)], dtype=np.float32)
        lat_err = float(np.dot(v, r))

        d_err = float(lat_err - self._lead_prev_lat_err)
        self._lead_prev_lat_err = float(lat_err)

        steer_cmd = - (self.lead_steer_kp * lat_err + self.lead_steer_kd * d_err)
        steer_cmd = clamp(steer_cmd, -0.3, 0.3)

        # Speed control
        vel = self.lead.get_velocity()
        speed_mps = math.sqrt(float(vel.x * vel.x + vel.y * vel.y + vel.z * vel.z))
        e = float(self.lead_target_speed_mps - speed_mps)

        throttle = clamp(self.lead_speed_kp * e, 0.0, 0.75)
        brake = clamp(self.lead_brake_kp * (-e), 0.0, 1.0) if e < 0 else 0.0

        self.lead.apply_control(carla.VehicleControl(throttle=throttle, steer=steer_cmd, brake=brake))

    # --------------------------------------------------------------------------
    # Overtake logic
    # --------------------------------------------------------------------------

    def _longitudinal_s(self, loc: carla.Location) -> float:
        """Project a location onto the road direction (meters)."""
        p = np.array([float(loc.x), float(loc.y)], dtype=np.float32)
        return float(np.dot(p - self._road_start_xy, self._road_dir))

    def _in_original_lane(self) -> bool:
        """Check if ego is currently on the original spawn lane (road/section/lane id)."""
        if self._orig_lane_id is None:
            return False
        wp = self.map.get_waypoint(self.ego.get_location(), project_to_road=True, lane_type=carla.LaneType.Driving)
        if wp is None:
            return False
        return (int(wp.road_id) == self._orig_road_id
                and int(wp.section_id) == self._orig_section_id
                and int(wp.lane_id) == self._orig_lane_id)

    def _update_overtake_events(self) -> None:
        """
        Event definitions:
          - exceeded: ego passes lead in longitudinal s
          - left_original_lane: ego ever leaves original lane (after reset)
          - returned: ego returns to original lane after exceeded and left_original_lane
          - overtake: exceeded + returned + ego is sufficiently ahead of lead
        """
        if self.lead is None or (not self.lead.is_alive):
            return

        ego_s = self._longitudinal_s(self.ego.get_location())
        lead_s = self._longitudinal_s(self.lead.get_location())

        # Passing event (add a small margin for stability)
        pass_margin_m = float(_get_cfg(self._config, "pass_margin_m", 0.5))
        if (not self.exceeded) and (ego_s > lead_s + pass_margin_m):
            self.exceeded = True
            self.exceed_event_step = True

        # Track whether ego has left original lane at least once
        if not self._in_original_lane():
            self.left_original_lane = True

        # Returned-to-lane event
        if (not self.returned) and self.exceeded and self.left_original_lane and self._in_original_lane():
            self.returned = True
            self.return_event_step = True

        # Overtake completion (ahead distance)
        overtake_ahead_m = float(_get_cfg(self._config, "overtake_ahead_m", 8.0))
        if (not self.overtake) and self.exceeded and self.returned and (ego_s > lead_s + overtake_ahead_m):
            self.overtake = True
            self.overtake_event_step = True

    # --------------------------------------------------------------------------
    # Reward and termination
    # --------------------------------------------------------------------------

    def _compute_offroad_flag(self) -> Tuple[bool, float]:
        """
        Offroad is defined as:
          - strict waypoint (project_to_road=False) is None AND
          - distance to projected lane center exceeds threshold
        """
        loc = self.ego.get_location()
        strict_wp = self.map.get_waypoint(loc, project_to_road=False, lane_type=carla.LaneType.Driving)
        proj_wp = self.map.get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Driving)

        if proj_wp is None:
            return True, 1e9

        c = proj_wp.transform.location
        proj_off = float(math.hypot(float(loc.x - c.x), float(loc.y - c.y)))

        offroad = (strict_wp is None) and (proj_off > float(self.offroad_center_thresh_m))
        return bool(offroad), proj_off

    def _compute_reward(self, speed_mps: float) -> Tuple[float, Dict[str, float]]:
        """
        A clean overtaking reward:
          + progress shaping (distance-to-goal reduction)
          + speed shaping
          + sparse events (pass, return, overtake)
          - collision, offroad, lane invasions
          - time penalty
        """
        rc: Dict[str, float] = {}

        # Progress to goal (distance reduction)
        dist_goal = float(self.ego.get_location().distance(self.end_point.location))
        prev = float(self.last_distance_to_goal) if self.last_distance_to_goal is not None else dist_goal
        progress_m = max(0.0, prev - dist_goal)
        self.last_distance_to_goal = dist_goal
        rc["progress"] = self.k_progress * progress_m

        # Speed tracking (L1)
        rc["speed"] = -self.k_speed * abs(float(speed_mps) - float(self.desired_speed_mps))

        # Sparse event rewards
        rc["pass"] = self.R_pass if self.exceed_event_step else 0.0
        rc["return"] = self.R_return if self.return_event_step else 0.0
        rc["overtake"] = self.R_overtake if self.overtake_event_step else 0.0

        # Safety penalties
        offroad, _ = self._compute_offroad_flag()
        rc["offroad"] = -self.R_offroad if offroad else 0.0
        rc["collision"] = -self.R_collision * float(max(self._collision_step, 0))
        rc["invasion"] = -self.k_invasion * float(max(self._lane_invasion_step, 0))

        # Time penalty
        rc["time"] = -float(self.time_penalty)

        # Stuck penalty (small, but termination handles hard stuck)
        if self.low_speed_steps > 0:
            rc["low_speed"] = -2.0
        else:
            rc["low_speed"] = 0.0

        total = float(sum(rc.values()))
        return total, rc

    def _check_termination(self) -> Tuple[bool, Dict[str, Any]]:
        info: Dict[str, Any] = {
            "collision": False,
            "goal_reached": False,
            "offroad": False,
            "not_moving": False,
            "time_exceeded": False,
        }

        # Collision
        if self.collision_detected:
            info["collision"] = True
            return True, info

        # Goal
        if self.ego.get_location().distance(self.end_point.location) < float(self.goal_radius_m):
            info["goal_reached"] = True
            return True, info

        # Time limit
        if self._time_step >= self._max_time_step:
            info["time_exceeded"] = True
            return True, info

        # Offroad (do not terminate for lane changes; only for leaving driving lanes)
        offroad, _ = self._compute_offroad_flag()
        if offroad:
            info["offroad"] = True
            return True, info

        # Stuck
        if self.low_speed_steps >= self.low_speed_timeout_steps:
            info["not_moving"] = True
            return True, info

        return False, info

    # --------------------------------------------------------------------------
    # Cleanup
    # --------------------------------------------------------------------------

    def _clean_actors(self) -> None:
        """Stop sensors and destroy all actors spawned by this environment."""
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
            self.lead = None
            self.camera_sensor = None
            self.colsensor = None
            self.lane_sensor = None
            self._img_q = None
            self._got_frame = False

            if getattr(self, "_sync_enabled", False):
                self.world.tick()

        except Exception as e:
            print(f"Cleanup error: {e}")
