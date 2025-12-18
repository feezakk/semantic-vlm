import re

import embodied
import jax
import numpy as np
import csv, atexit


def train(agent, env, eval_env, replay, eval_replay, logger, args):

    logdir = embodied.Path(args.logdir)
    logdir.mkdirs()
    print("Logdir", logdir)
    should_expl = embodied.when.Until(args.expl_until)
    should_train = embodied.when.Ratio(args.train_ratio / args.batch_steps)
    should_log = embodied.when.Clock(args.log_every)
    should_save = embodied.when.Clock(args.save_every)
    should_eval = embodied.when.Every(args.eval_every, args.eval_initial)
    should_sync = embodied.when.Every(args.sync_every)
    step = logger.step
    updates = embodied.Counter()
    metrics = embodied.Metrics()
    print("Observation space:", embodied.format(env.obs_space), sep="\n")
    print("Action space:", embodied.format(env.act_space), sep="\n")

    timer = embodied.Timer()
    timer.wrap("agent", agent, ["policy", "train", "report", "save"])
    timer.wrap("env", env, ["step"])
    timer.wrap("replay", replay, ["add", "save"])
    timer.wrap("logger", logger, ["write"])

    nonzeros = set()

    # ---------- CSV writers (NEW) ----------
        # ---------- CSV writers (NEW) ----------
    def _make_writer(path):
        path = embodied.Path(path)
        exists = path.exists()
        f = open(str(path), "a", newline="")
        fieldnames = [
            "episode_index","env_step","route_id","weather_id","weather_name",
            "length","return","termination",
            "success","collision","time_exceeded","not_moving","past_goal",
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
            "goal_reached","time_exceeded","not_moving","past_goal",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            w.writeheader(); f.flush()
        return f, w
    
    def _make_scalar_writer(path):
        path = embodied.Path(path)
        exists = path.exists()
        f = open(str(path), "a", newline="")
        fieldnames = [
            "env_step",
            "updates",

            # ---- train scalars ----
            "train/model_loss_mean",
            "train/dyn_loss_mean",
            "train/rep_loss_mean",
            "train/sem_rollout_loss_mean",
            "train/domain_adv_loss_mean",
            "train/domain_adv_scale",
            "train/domain_acc",
            "train/domain_entropy",
            "train/domain_chance_acc",
            "train/domain_majority_acc",
            "train/domain_acc_minus_chance",
            "train/domain_acc_minus_majority",
            "train/sty_domain_acc",
            "train/dom_majority_baseline",
            "train/dom_label_frac_0", "train/dom_label_frac_1", "train/dom_label_frac_2", "train/dom_label_frac_3",
            "train/dom_pred_frac_0", "train/dom_pred_frac_1", "train/dom_pred_frac_2", "train/dom_pred_frac_3",
            "train/dom_recall_0", "train/dom_recall_1", "train/dom_recall_2", "train/dom_recall_3",
            "train/sty_dom_recall_0", "train/sty_dom_recall_1", "train/sty_dom_recall_2", "train/sty_dom_recall_3",
            "train/domain_frac_0", "train/domain_frac_1", "train/domain_frac_2", "train/domain_frac_3",

            # ---- eval scalars (from agent.report(eval_batch)) ----
            "eval/model_loss_mean",
            "eval/dyn_loss_mean",
            "eval/rep_loss_mean",
            "eval/sem_rollout_loss_mean",
            "eval/domain_adv_loss_mean",
            "eval/domain_adv_scale",
            "eval/domain_acc",
            "eval/domain_entropy",
            "eval/domain_chance_acc",
            "eval/domain_majority_acc",
            "eval/domain_acc_minus_chance",
            "eval/domain_acc_minus_majority",
            "eval/sty_domain_acc",
            "eval/dom_majority_baseline",
            "eval/dom_label_frac_0",  "eval/dom_label_frac_1",  "eval/dom_label_frac_2",  "eval/dom_label_frac_3",
            "eval/dom_pred_frac_0",  "eval/dom_pred_frac_1",  "eval/dom_pred_frac_2",  "eval/dom_pred_frac_3",
            "eval/dom_recall_0",  "eval/dom_recall_1",  "eval/dom_recall_2",  "eval/dom_recall_3",
            "eval/sty_dom_recall_0",  "eval/sty_dom_recall_1",  "eval/sty_dom_recall_2",  "eval/sty_dom_recall_3",
            "eval/domain_frac_0",  "eval/domain_frac_1",  "eval/domain_frac_2",  "eval/domain_frac_3",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            w.writeheader(); f.flush()
        return f, w

    scalars_f, scalars_w = _make_scalar_writer(logdir / "scalars.csv")
    atexit.register(lambda: scalars_f.close())


    train_csv_f, train_csv_w = _make_writer(logdir / "train_success.csv")
    eval_csv_f,  eval_csv_w  = _make_writer(logdir / "eval_success.csv")

    train_step_f, train_step_w = _make_step_writer(logdir / "train_steps.csv")
    eval_step_f,  eval_step_w  = _make_step_writer(logdir / "eval_steps.csv")

    atexit.register(
        lambda: (
            train_csv_f.close(), eval_csv_f.close(),
            train_step_f.close(), eval_step_f.close()
        )
    )
    train_ep_idx = {"v": 0}
    eval_ep_idx  = {"v": 0}

    def _to_float(x):
        # Handles Python floats, numpy scalars, jax arrays, etc.
        try:
            return float(np.array(x))
        except Exception:
            return float("nan")


    def _csv_log(ep, ep_info, is_eval=False):
        # Episode stats
        length = int(len(ep["reward"]) - 1)  # you already do this
        ret = float(ep["reward"].astype(np.float64).sum())

        def _any(k):  # handles missing keys
            v = ep_info.get(k, [])
            return bool(np.any(np.array(v)))
        
        def _arr(name, default=0.0):
            v = ep_info.get(name, None)
            if v is None or len(v) == 0:
                return np.asarray([default])
            return np.asarray(v)

        def _first(name, default=-1):
            a = _arr(name, default)
            return a[0]

        def _last(name, default=0.0):
            a = _arr(name, default)
            return a[-1]

        def _mean(name, default=0.0):
            a = _arr(name, default)
            return float(np.mean(a))

        def _max(name, default=0.0):
            a = _arr(name, default)
            return float(np.max(a))
        
        route_id = int(_first("route_id", -1))
        weather_id = int(_first("weather_id", -1))
        weather_name = str(_first("weather_name", "NA"))

        distance_m = float(_last("distance_m", 0.0))
        distance_km = distance_m / 1000.0

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


        # Choose episode index and writers
        if is_eval:
            ep_idx = eval_ep_idx["v"]
            w_ep, f_ep = eval_csv_w, eval_csv_f
            w_step, f_step = eval_step_w, eval_step_f
        else:
            ep_idx = train_ep_idx["v"]
            w_ep, f_ep = train_csv_w, train_csv_f
            w_step, f_step = train_step_w, train_step_f

        # --------- 1) Episode-level row (unchanged logic) ----------
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

        w_ep.writerow(row_ep)
        f_ep.flush()

        # --------- 2) Step-level rows (NEW) ----------
        # Convert to arrays with safe defaults
        rewards = np.asarray(ep["reward"][:length], np.float32)

        def _step_arr(name, default=0):
            v = ep_info.get(name, None)
            if v is None:
                return np.full((length,), default, np.int32)
            arr = np.asarray(v)
            # make sure we have length entries
            if arr.shape[0] >= length:
                arr = arr[:length]
            else:
                pad = np.full((length - arr.shape[0],), default, arr.dtype)
                arr = np.concatenate([arr, pad], axis=0)
            return arr

        collision_step    = _step_arr("collision_step", 0)
        lane_inv_step     = _step_arr("lane_invasion_step", 0)
        goal_reached_step = _step_arr("goal_reached", 0)
        time_exceeded     = _step_arr("time_exceeded", 0)
        not_moving        = _step_arr("not_moving", 0)
        past_goal         = _step_arr("past_goal", 0)
        action_idx        = _step_arr("action_idx", 0)
        throttle_cmd      = ep_info.get("throttle_cmd", None)
        steer_cmd         = ep_info.get("steer_cmd", None)
        speed_kmh         = ep_info.get("speed_kmh", None)
        heading_err       = ep_info.get("heading_error", None)
        dist_goal         = ep_info.get("dist_to_goal_m", None)
        progress_m        = ep_info.get("progress_m", None)

        off_center        = ep_info.get("off_center_m", None)
        if off_center is None:
            off_center = np.zeros((length,), np.float32)
        else:
            off_center = np.asarray(off_center[:length], np.float32)

        # convert safely
        def _float_step(name, default=0.0):
            v = ep_info.get(name, None)
            if v is None: return np.full((length,), default, np.float32)
            a = np.asarray(v, np.float32)
            return a[:length] if a.shape[0] >= length else np.pad(a, (0, length-a.shape[0]), constant_values=default)

        throttle_cmd = _float_step("throttle_cmd", 0.0)
        steer_cmd    = _float_step("steer_cmd", 0.0)
        speed_kmh    = _float_step("speed_kmh", 0.0)
        heading_err  = _float_step("heading_error", 0.0)
        dist_goal    = _float_step("dist_to_goal_m", 0.0)
        progress_m   = _float_step("progress_m", 0.0)

        route_ids    = np.full((length,), route_id, np.int32)
        weather_ids  = np.full((length,), weather_id, np.int32)
        weather_names = [weather_name] * length

        # env_step per transition if available
        if "env_step" in ep:
            env_steps = np.asarray(ep["env_step"][:length], np.int64)
        else:
            # fallback: use logger.step as coarse reference, or leave as t
            env_steps = np.arange(length, dtype=np.int64)

        for t in range(length):
            row_step = {
                "episode_index": ep_idx,
                "t": int(t),
                "env_step": int(env_steps[t]),
                "route_id": int(route_ids[t]),
                "weather_id": int(weather_ids[t]),
                "weather_name": str(weather_names[t]),
                "action_idx": int(action_idx[t]),
                "throttle_cmd": float(throttle_cmd[t]),
                "steer_cmd": float(steer_cmd[t]),
                "reward": float(rewards[t]),
                "speed_kmh": float(speed_kmh[t]),
                "off_center_m": float(off_center[t]),
                "heading_error": float(heading_err[t]),
                "dist_to_goal_m": float(dist_goal[t]),
                "progress_m": float(progress_m[t]),
                "collision_step": int(collision_step[t]),
                "lane_invasion_step": int(lane_inv_step[t]),
                "goal_reached": int(goal_reached_step[t]),
                "time_exceeded": int(time_exceeded[t]),
                "not_moving": int(not_moving[t]),
                "past_goal": int(past_goal[t]),
            }
            w_step.writerow(row_step)

        f_step.flush()

        # bump episode index after logging
        if is_eval:
            eval_ep_idx["v"] += 1
        else:
            train_ep_idx["v"] += 1

    # ---------------------------------------

    def per_episode(ep, ep_info):
        length = len(ep["reward"]) - 1
        score = float(ep["reward"].astype(np.float64).sum())
        success = float(np.any(np.array(ep_info.get("goal_reached", [0]))))
        collision = float(np.any(np.array(ep_info.get("collision", [0]))))
        past_goal = float(np.any(np.array(ep_info.get("past_goal", [0]))))
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
                "not_moving": not_moving,
                "time_exceeded": time_exceeded
            },
            prefix="episode",
        )
        # print(f"Episode has {length} steps and return {score:.1f}.")
        stats = {}
        for key in args.log_keys_video:
            if key in ep:
                stats[f"policy_{key}"] = ep[key]
        for key, value in ep.items():
            if not args.log_zeros and key not in nonzeros and (value == 0).all():
                continue
            nonzeros.add(key)
            if re.match(args.log_keys_sum, key):
                stats[f"sum_{key}"] = ep[key].sum()
            if re.match(args.log_keys_mean, key):
                stats[f"mean_{key}"] = ep[key].mean()
            if re.match(args.log_keys_max, key):
                stats[f"max_{key}"] = ep[key].max(0).mean()
        metrics.add(stats, prefix="stats")

    driver = embodied.DriverVLM(env)
    driver.on_episode(lambda ep, ep_info, worker: per_episode(ep, ep_info))
    driver.on_episode(lambda ep, ep_info, worker: _csv_log(ep, ep_info, is_eval=False))
    driver.on_step(lambda _, __, ___: step.increment())
    driver.on_step(lambda tran, _, worker: replay.add(tran, worker))

    driver_eval = embodied.DriverVLM(eval_env)
    driver_eval.on_step(lambda trn, inf, i, **kw: eval_replay.add(trn, worker=i))
    driver_eval.on_episode(lambda ep, ep_info, worker: per_episode(ep, ep_info))
    driver_eval.on_episode(lambda ep, ep_info, worker: _csv_log(ep, ep_info, is_eval=True))

    print("Prefill train dataset.")
    random_agent = embodied.RandomAgent(env.act_space, args.actor_dist_disc)
    while len(replay) < max(args.batch_steps, args.train_fill):
        driver(random_agent.policy, steps=100)
    print("Prefill eval dataset.")
    prefill_eps = int(max(1, args.eval_eps))
    while len(eval_replay) < max(args.batch_steps, args.eval_fill):
        driver_eval(random_agent.policy, steps=prefill_eps)

    logger.add(metrics.result())
    logger.write()

    dataset = agent.dataset(replay.dataset)
    dataset_eval = agent.dataset(eval_replay.dataset)


    state = [None]  # To be writable from train step function below.
    batch = [None]

    def train_step(_, __, ___):
        for _ in range(should_train(step)):
            with timer.scope("dataset"):
                batch[0] = next(dataset)
            outs, state[0], mets = agent.train(batch[0], state[0])
            metrics.add(mets, prefix="train")

            # if getattr(replay, "update_visit_count", False):
            #     replay.update_visit_count(jax.device_get(batch[0]["env_step"]))

            if "key" in outs:
                replay.prioritize(outs["key"], outs["env_step"], outs["model_loss"], outs["td_error"])

            updates.increment()
        if should_sync(updates):
            agent.sync()
        if should_log(step):
            agg = metrics.result()
            report = agent.report(batch[0])
            report = {k: v for k, v in report.items() if "train/" + k not in agg}
            logger.add(agg)
            logger.add(replay.stats, prefix="report")
            with timer.scope("dataset_eval"):
                eval_batch = next(dataset_eval)
            eval_report = agent.report(eval_batch)
            logger.add(eval_report, prefix="eval")
            logger.add(agent.report(eval_batch), prefix="eval")
            logger.add(replay.stats, prefix="replay")
            logger.add(eval_replay.stats, prefix="eval_replay")
            logger.add(timer.stats(), prefix="timer")
            logger.write(fps=True)

            # ---------- NEW: Scalars CSV row ----------
            row = {"env_step": int(step), "updates": int(updates)}

            # train metrics live in `agg` with "train/..." keys
            for k in [
                "train/model_loss_mean",
                "train/dyn_loss_mean",
                "train/rep_loss_mean",
                "train/sem_rollout_loss_mean",
                "train/domain_adv_loss_mean",
                "train/domain_adv_scale",
                "train/domain_acc",
                "train/domain_entropy",
                "train/domain_acc",
                "train/domain_entropy",
                "train/domain_chance_acc",
                "train/domain_majority_acc",
                "train/domain_acc_minus_chance",
                "train/domain_acc_minus_majority",
                "train/sty_domain_acc",
                "train/dom_majority_baseline",
                "train/dom_label_frac_0","train/dom_label_frac_1","train/dom_label_frac_2","train/dom_label_frac_3",
                "train/dom_pred_frac_0","train/dom_pred_frac_1","train/dom_pred_frac_2","train/dom_pred_frac_3",
                "train/dom_recall_0", "train/dom_recall_1", "train/dom_recall_2", "train/dom_recall_3",
                "train/sty_dom_recall_0", "train/sty_dom_recall_1", "train/sty_dom_recall_2", "train/sty_dom_recall_3",
                "train/domain_frac_0","train/domain_frac_1","train/domain_frac_2","train/domain_frac_3"
            ]:
                row[k] = _to_float(agg.get(k, np.nan))

            # eval metrics come from eval_report; add "eval/" prefix manually
            for k in [
                "model_loss_mean",
                "dyn_loss_mean",
                "rep_loss_mean",
                "sem_rollout_loss_mean",
                "domain_adv_loss_mean",
                "domain_adv_scale",
                "domain_acc",
                "domain_entropy",
                "domain_acc",
                "domain_entropy",
                "domain_chance_acc",
                "domain_majority_acc",
                "domain_acc_minus_chance",
                "domain_acc_minus_majority",
                "sty_domain_acc",
                "dom_majority_baseline",
                "dom_label_frac_0","dom_label_frac_1","dom_label_frac_2","dom_label_frac_3",
                "dom_pred_frac_0","dom_pred_frac_1","dom_pred_frac_2","dom_pred_frac_3",
                "dom_recall_0", "dom_recall_1", "dom_recall_2", "dom_recall_3",
                "sty_dom_recall_0", "sty_dom_recall_1", "sty_dom_recall_2", "sty_dom_recall_3",
                "domain_frac_0","domain_frac_1","domain_frac_2","domain_frac_3"
            ]:
                row["eval/" + k] = _to_float(eval_report.get(k, np.nan))

            scalars_w.writerow(row)
            scalars_f.flush()



    driver.on_step(train_step)

    checkpoint = embodied.Checkpoint(logdir / "checkpoint.ckpt")
    timer.wrap("checkpoint", checkpoint, ["save", "load"])
    checkpoint.step = step
    checkpoint.agent = agent
    checkpoint.replay = replay
    if args.from_checkpoint:
        checkpoint.load(args.from_checkpoint)
    checkpoint.load_or_save()
    should_save(step)  # Register that we jused saved.

    print("Start training loop.")
    driver._state = None
    policy = lambda *args: agent.policy(*args, mode="explore" if should_expl(step) else "train")
    policy_eval = lambda *args: agent.policy(*args, mode="eval")
    while step < args.steps:
        if should_eval(step):
            print("Starting evaluation at step", int(step))
            driver_eval.reset()
            driver_eval(policy_eval, episodes=max(len(eval_env), args.eval_eps))
        driver(policy, steps=100)
        if should_save(step):
            checkpoint.save()
    logger.write()
