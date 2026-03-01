import re
import warnings

import embodied
import numpy as np
import ruamel.yaml as yaml

import car_dreamer
import dreamerv3
import csv, atexit

warnings.filterwarnings("ignore", ".*truncated to dtype int32.*")


def wrap_env(env, config):
    args = config.wrapper
    env = embodied.wrappers.InfoWrapper(env)
    for name, space in env.act_space.items():
        if name == "reset":
            continue
        elif space.discrete:
            env = embodied.wrappers.OneHotAction(env, name)
        elif args.discretize:
            env = embodied.wrappers.DiscretizeAction(env, name, args.discretize)
        else:
            env = embodied.wrappers.NormalizeAction(env, name)
    env = embodied.wrappers.ExpandScalars(env)
    if args.length:
        env = embodied.wrappers.TimeLimit(env, args.length, args.reset)
    if args.checks:
        env = embodied.wrappers.CheckSpaces(env)
    for name, space in env.act_space.items():
        if not space.discrete:
            env = embodied.wrappers.ClipAction(env, name)
    return env

def eval_only(agent, env, logger, args):
    print("Start evaluation.")
    print("args:", args)
    logdir = embodied.Path(args.logdir)
    logdir.mkdirs()
    print("Logdir", logdir)
    step = logger.step
    metrics = embodied.Metrics()
    print("Observation space:", env.obs_space)
    print("Action space:", env.act_space)

    timer = embodied.Timer()
    timer.wrap("agent", agent, ["policy"])
    timer.wrap("env", env, ["step"])
    timer.wrap("logger", logger, ["write"])

    nonzeros = set()

    def _make_writer(path):
        path = embodied.Path(path)
        exists = path.exists()
        f = open(str(path), "a", newline="")
        fieldnames = [
            "episode_index","env_step","route_id","weather_id","weather_name",
            "length","return","termination",
            "success","collision","time_exceeded","not_moving","past_goal","offroad","wrong_lane",
            "distance_m","distance_km",
            "lane_invasions_total","collisions_total",
            "lane_inv_per_km","collisions_per_km",
            "mean_off_center_m","mean_heading_error",
            "mean_speed_kmh","max_speed_kmh",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            w.writeheader(); f.flush()
        return f, w

    def _make_step_writer(path):
        path = embodied.Path(path)
        exists = path.exists()
        f = open(str(path), "a", newline="")
        fieldnames = [
            "episode_index","t","env_step","route_id","weather_id","weather_name",
            "action_idx","throttle_cmd","steer_cmd",
            "reward","speed_kmh","off_center_m","heading_error",
            "dist_to_goal_m","progress_m",
            "collision_step","lane_invasion_step",
            "goal_reached","time_exceeded","not_moving","past_goal","offroad","wrong_lane",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            w.writeheader(); f.flush()
        return f, w

    eval_csv_f, eval_csv_w = _make_writer(logdir / "eval_success.csv")
    eval_step_f, eval_step_w = _make_step_writer(logdir / "eval_steps.csv")
    atexit.register(lambda: (eval_csv_f.close(), eval_step_f.close()))
    eval_ep_idx = {"v": 0}

    def _csv_log(ep, ep_info):
        length = int(len(ep["reward"]) - 1)
        ret = float(ep["reward"].astype(np.float64).sum())

        def _any(k):
            v = ep_info.get(k, [])
            return bool(np.any(np.array(v)))

        def _arr(name, default=0.0):
            v = ep_info.get(name, None)
            if v is None or len(v) == 0:
                return np.asarray([default])
            return np.asarray(v)

        def _first(name, default=-1):
            return _arr(name, default)[0]

        def _last(name, default=0.0):
            return _arr(name, default)[-1]

        def _mean(name, default=0.0):
            return float(np.mean(_arr(name, default)))

        def _max(name, default=0.0):
            return float(np.max(_arr(name, default)))

        route_id = int(_first("route_id", -1))
        weather_id = int(_first("weather_id", -1))
        weather_name = str(_first("weather_name", "NA"))

        distance_m = float(_last("distance_m", 0.0))
        distance_km = distance_m / 1000.0

        # lane_invasions_total = int(np.sum(_arr("lane_invasion_step", 0)))
        # Robust: use cumulative counter from env (not affected by reward resetting step flags)
        lane_invasions_total = int(_last("lane_invasions", -1))
        if lane_invasions_total < 0:
            # fallback for older logs
            lane_invasions_total = int(np.sum(_arr("lane_invasion_step", 0)))
        collisions_total = int(np.sum(_arr("collision_step", 0)))

        lane_inv_per_km = float(lane_invasions_total / (distance_km + 1e-6))
        collisions_per_km = float(collisions_total / (distance_km + 1e-6))

        mean_off_center = float(_mean("off_center_m", 0.0))
        mean_heading = float(_mean("heading_error", 0.0))
        mean_speed = float(_mean("speed_kmh", 0.0))
        max_speed = float(_max("speed_kmh", 0.0))

        termination = "unknown"
        if _any("goal_reached"): termination = "goal_reached"
        elif _any("collision"): termination = "collision"
        elif _any("not_moving"): termination = "not_moving"
        elif _any("past_goal"): termination = "past_goal"
        elif _any("time_exceeded"): termination = "time_exceeded"
        elif _any("offroad"): termination = "offroad"
        elif _any("wrong_lane"): termination = "wrong_lane"

        ep_idx = eval_ep_idx["v"]

        # --- episode-level row ---
        row_ep = {
            "episode_index": ep_idx,
            "env_step": int(logger.step),
            "route_id": route_id,
            "weather_id": weather_id,
            "weather_name": weather_name,
            "length": length,
            "return": ret,
            "termination": termination,
            "success": int(_any("goal_reached")),
            "collision": int(_any("collision")),
            "time_exceeded": int(_any("time_exceeded")),
            "not_moving": int(_any("not_moving")),
            "past_goal": int(_any("past_goal")),
            "offroad": int(_any("offroad")),
            "wrong_lane": int(_any("wrong_lane")),
            "distance_m": distance_m,
            "distance_km": distance_km,
            "lane_invasions_total": lane_invasions_total,
            "collisions_total": collisions_total,
            "lane_inv_per_km": lane_inv_per_km,
            "collisions_per_km": collisions_per_km,
            "mean_off_center_m": mean_off_center,
            "mean_heading_error": mean_heading,
            "mean_speed_kmh": mean_speed,
            "max_speed_kmh": max_speed,
        }
        eval_csv_w.writerow(row_ep)
        eval_csv_f.flush()

        # --- step-level rows ---
        rewards = np.asarray(ep["reward"][:length], np.float32)
        T = len(rewards)

        def _step_arr(name, default=0):
            v = ep_info.get(name, None)
            if v is None:
                return np.full((length,), default, np.int32)
            arr = np.asarray(v)
            if arr.shape[0] >= length:
                return arr[:length]
            pad = np.full((length - arr.shape[0],), default, arr.dtype)
            return np.concatenate([arr, pad], axis=0)

        collision_step    = _step_arr("collision_step", 0)
        # lane_inv_step     = _step_arr("lane_invasion_step", 0)
        # Derive per-step invasions from cumulative count (more reliable than lane_invasion_step)
        lane_inv_cum = _step_arr("lane_invasions", 0).astype(np.int64)
        lane_inv_step = np.diff(np.concatenate([[0], lane_inv_cum[:length]])).astype(np.int32)
        lane_inv_step = np.clip(lane_inv_step, 0, None)
        goal_reached_step = _step_arr("goal_reached", 0)
        time_exceeded     = _step_arr("time_exceeded", 0)
        not_moving        = _step_arr("not_moving", 0)
        past_goal         = _step_arr("past_goal", 0)
        offroad           = _step_arr("offroad", 0)
        wrong_lane        = _step_arr("wrong_lane", 0)
        action_idx        = _step_arr("action_idx", 0)

        # float series (may be shorter by 1 due to reset transition)
        def _safe_series(key):
            return np.asarray(ep_info.get(key, []))

        off_center   = _safe_series("off_center_m")
        heading_err  = _safe_series("heading_error")
        dist_goal    = _safe_series("dist_to_goal_m")
        progress_m   = _safe_series("progress_m")
        speed_kmh    = _safe_series("speed_kmh")
        throttle_cmd = _safe_series("throttle_cmd")
        steer_cmd    = _safe_series("steer_cmd")

        offset = T - len(off_center) if len(off_center) > 0 else 0
        offset = int(np.clip(offset, 0, T))

        def _at(arr, t, default=np.nan):
            if arr is None or len(arr) == 0:
                return default
            idx = t - offset
            if idx < 0 or idx >= len(arr):
                return default
            return arr[idx]

        # best-effort env_step per-step (if not present, keep t)
        if "env_step" in ep:
            env_steps = np.asarray(ep["env_step"][:length], np.int64)
        else:
            env_steps = np.arange(length, dtype=np.int64)

        for t in range(length):
            row_step = {
                "episode_index": ep_idx,
                "t": int(t),
                "env_step": int(env_steps[t]),
                "route_id": int(route_id),
                "weather_id": int(weather_id),
                "weather_name": str(weather_name),
                "action_idx": int(action_idx[t]),
                "throttle_cmd": float(_at(throttle_cmd, t)),
                "steer_cmd": float(_at(steer_cmd, t)),
                "reward": float(rewards[t]),
                "speed_kmh": float(_at(speed_kmh, t)),
                "off_center_m": float(_at(off_center, t)),
                "heading_error": float(_at(heading_err, t)),
                "dist_to_goal_m": float(_at(dist_goal, t)),
                "progress_m": float(_at(progress_m, t)),
                "collision_step": int(collision_step[t]),
                "lane_invasion_step": int(lane_inv_step[t]),
                "goal_reached": int(goal_reached_step[t]),
                "time_exceeded": int(time_exceeded[t]),
                "not_moving": int(not_moving[t]),
                "past_goal": int(past_goal[t]),
                "offroad": int(offroad[t]),
                "wrong_lane": int(wrong_lane[t]),
            }
            eval_step_w.writerow(row_step)
        eval_step_f.flush()

        eval_ep_idx["v"] += 1

    def per_episode(ep, ep_info):
        length = len(ep["reward"]) - 1
        score = float(ep["reward"].astype(np.float64).sum())

        success = float(np.any(np.array(ep_info.get("goal_reached", [0]))))
        collision = float(np.any(np.array(ep_info.get("collision", [0]))))
        past_goal = float(np.any(np.array(ep_info.get("past_goal", [0]))))
        offroad = float(np.any(np.array(ep_info.get("offroad", [0]))))
        wrong_lane = float(np.any(np.array(ep_info.get("wrong_lane", [0]))))
        not_moving = float(np.any(np.array(ep_info.get("not_moving", [0]))))
        time_exceeded = float(np.any(np.array(ep_info.get("time_exceeded", [0]))))

        sum_abs_reward = float(np.abs(ep["reward"]).astype(np.float64).sum())

        logger.add(
            {
                "length": length,
                "score": score,
                "sum_abs_reward": sum_abs_reward,
                "reward_rate": (np.abs(ep["reward"]) >= 0.5).mean(),
                "success": success,
                "collision": collision,
                "past_goal": past_goal,
                "offroad": offroad,
                "wrong_lane": wrong_lane,
                "not_moving": not_moving,
                "time_exceeded": time_exceeded,
            },
            prefix="episode",
        )

        stats = {}
        for key, value in ep.items():
            if re.match(args.log_keys_sum, key):
                stats[f"sum_{key}"] = ep[key].sum()
            if re.match(args.log_keys_mean, key):
                stats[f"mean_{key}"] = ep[key].mean()
            if re.match(args.log_keys_max, key):
                stats[f"max_{key}"] = ep[key].max(0).mean()

        metrics.add(stats, prefix="stats")

    def per_step(tran):
        step.increment()

    driver = embodied.DriverVLM(env)
    driver.on_episode(lambda ep, ep_info, worker, **kw: per_episode(ep, ep_info))
    driver.on_episode(lambda ep, ep_info, worker, **kw: _csv_log(ep, ep_info))
    driver.on_step(lambda *a, **kw: step.increment())

    checkpoint = embodied.Checkpoint()
    checkpoint.agent = agent
    if args.from_checkpoint:
        checkpoint.load(args.from_checkpoint, keys=["agent"])
    else:
        raise ValueError("No checkpoint specified.")

    print("Start evaluation loop.")
    policy = lambda *args: agent.policy(*args, mode="eval")

    target_episodes = int(getattr(args, "eval_eps", 0))
    if target_episodes <= 0:
        raise ValueError(
            "Please set --dreamerv3.run.eval_eps to a positive integer "
            "(e.g., --dreamerv3.run.eval_eps 64)."
        )

    episodes_done = {"v": 0}  
    def per_episode_counting(ep, ep_info):
        per_episode(ep, ep_info)          
        episodes_done["v"] += 1

    try:
        driver.reset()
        driver(policy, episodes=target_episodes)
    except TypeError:
        driver.reset()
        while episodes_done["v"] < target_episodes:
            driver(policy, steps=100)

    logger.write()



def main(argv=None):
    model_configs = yaml.YAML(typ="safe").load((embodied.Path(__file__).parent / "dreamerv3.yaml").read())
    config = embodied.Config({"dreamerv3": model_configs["defaults"]})

    parsed, other = embodied.Flags(task=["carla_navigation"]).parse_known(argv)
    for name in parsed.task:
        print("Using task: ", name)
        env, env_config = car_dreamer.create_task(name, argv)
        config = config.update(env_config)
    config = embodied.Flags(config).parse(other)

    logdir = embodied.Path(config.dreamerv3.logdir)
    step = embodied.Counter()
    logger = embodied.Logger(
        step,
        [
            embodied.logger.TerminalOutput(),
            embodied.logger.JSONLOutput(logdir, "metrics.jsonl"),
            embodied.logger.TensorBoardOutput(logdir),
        ],
    )

    from embodied.envs import from_gym

    dreamerv3_config = config.dreamerv3
    env = from_gym.FromGym(env)
    env = wrap_env(env, dreamerv3_config)
    env = embodied.BatchEnv([env], parallel=False)

    agent = dreamerv3.agent_vlm(env.obs_space, env.act_space, step, dreamerv3_config)
    args = embodied.Config(
        **dreamerv3_config.run,
        logdir=dreamerv3_config.logdir,
        batch_steps=dreamerv3_config.batch_size * dreamerv3_config.batch_length,
    )
    eval_only(agent, env, logger, args)


if __name__ == "__main__":
    main()
