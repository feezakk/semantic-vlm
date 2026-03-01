import re
import json
import csv
import atexit

import embodied
import numpy as np


def train(agent, env, eval_env, replay, eval_replay, logger, args):
    """
    Training loop with CSV logging that captures *all* env info keys.

    Output files:
      - train_success.csv / eval_success.csv
          One row per episode summary

      - train_steps_kv.csv / eval_steps_kv.csv
          Long-format key-value CSV with *every* info key for *every* step:
          columns: split, episode_index, t, env_step, key, value

    Checkpoints:
      - checkpoint.ckpt : periodic checkpoint (agent + replay)
      - best.ckpt       : best eval checkpoint (agent only)
    """
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

    # Wrap only methods that exist (prevents crash on missing "sync"/"save"/etc.)
    agent_methods = [m for m in ["policy", "train", "report", "save", "sync"] if hasattr(agent, m)]
    timer.wrap("agent", agent, agent_methods)

    timer.wrap("env", env, ["step"])
    timer.wrap("replay", replay, ["add", "save"])
    timer.wrap("logger", logger, ["write"])

    nonzeros = set()

    # ---------------------------------------------------------------------
    # CSV writers
    # ---------------------------------------------------------------------

    def _make_episode_writer(path):
        path = embodied.Path(path)
        exists = path.exists()
        f = open(str(path), "a", newline="")
        fieldnames = [
            "episode_index",
            "env_step",
            "length",
            "return",
            "success",
            "collision",
            "time_exceeded",
            "not_moving",
            "past_goal",
            "offroad",
            "wrong_lane",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            w.writeheader()
            f.flush()
        return f, w

    def _make_kv_writer(path):
        path = embodied.Path(path)
        exists = path.exists()
        f = open(str(path), "a", newline="")
        fieldnames = ["split", "episode_index", "t", "env_step", "key", "value"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            w.writeheader()
            f.flush()
        return f, w

    train_ep_f, train_ep_w = _make_episode_writer(logdir / "train_success.csv")
    eval_ep_f, eval_ep_w = _make_episode_writer(logdir / "eval_success.csv")
    train_kv_f, train_kv_w = _make_kv_writer(logdir / "train_steps_kv.csv")
    eval_kv_f, eval_kv_w = _make_kv_writer(logdir / "eval_steps_kv.csv")

    def _close_all():
        try:
            train_ep_f.close()
        except Exception:
            pass
        try:
            eval_ep_f.close()
        except Exception:
            pass
        try:
            train_kv_f.close()
        except Exception:
            pass
        try:
            eval_kv_f.close()
        except Exception:
            pass

    atexit.register(_close_all)

    train_ep_idx = {"v": 0}
    eval_ep_idx = {"v": 0}

    # Gate CSV logging to avoid polluting logs with random-policy prefill
    csv_enabled = {"v": False}

    # ---------------------------------------------------------------------
    # Helpers: converting / aligning episode arrays
    # ---------------------------------------------------------------------

    def _to_jsonable(x):
        """Convert numpy/scalars/arrays into JSON-serializable Python types."""
        if isinstance(x, (np.generic,)):
            return x.item()
        if isinstance(x, np.ndarray):
            if x.size == 1:
                return x.reshape(-1)[0].item()
            return x.tolist()
        if isinstance(x, dict):
            return {str(k): _to_jsonable(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [_to_jsonable(v) for v in x]
        try:
            json.dumps(x)
            return x
        except TypeError:
            return str(x)

    def _value_to_string(v):
        """
        Store values in CSV as strings.
        - Scalars -> "3.14"
        - Arrays/dicts -> compact JSON
        """
        v = _to_jsonable(v)
        if isinstance(v, (dict, list)):
            return json.dumps(v, separators=(",", ":"))
        return str(v)

    def _align_series(v, T, default=0):
        """
        Align a per-step series to length T.
        Common pattern: arrays are length T+1 with a dummy at index 0.

        Handles scalars/strings by repeating them across T steps.
        """
        a = np.asarray(v, dtype=object)  # object-safe (dicts, strings, etc.)

        # scalar case: shape == ()
        if a.ndim == 0:
            a = np.repeat(a, T, axis=0)

        if a.shape[0] == T + 1:
            a = a[1:]
        elif a.shape[0] > T:
            a = a[:T]
        elif a.shape[0] < T:
            pad = np.full((T - a.shape[0],), default, dtype=object)
            a = np.concatenate([a, pad], axis=0)
        return a

    def _any_flag(ep_info, *names):
        """True if any named key in ep_info has any nonzero/True value."""
        for name in names:
            v = ep_info.get(name, None)
            if v is None:
                continue
            vv = np.asarray(v, dtype=object)
            # scalar -> treat as single flag
            if vv.ndim == 0:
                return bool(vv)
            if np.any(vv):
                return True
        return False

    # ---------------------------------------------------------------------
    # CSV logging: EVERYTHING in env info (ep_info) -> KV rows
    # ---------------------------------------------------------------------

    def _csv_log_everything(ep, ep_info, is_eval: bool):
        """
        Writes:
          1) One episode-level row
          2) Step-level KV rows for:
             - reward, done
             - actions
             - every key in ep_info
        """
        if not csv_enabled["v"]:
            return

        T = int(len(ep.get("reward", [])) - 1)
        if T <= 0:
            return

        split = "eval" if is_eval else "train"
        if is_eval:
            ep_idx = eval_ep_idx["v"]
            w_ep, f_ep = eval_ep_w, eval_ep_f
            w_kv, f_kv = eval_kv_w, eval_kv_f
        else:
            ep_idx = train_ep_idx["v"]
            w_ep, f_ep = train_ep_w, train_ep_f
            w_kv, f_kv = train_kv_w, train_kv_f

        rewards = _align_series(ep["reward"], T, default=0)
        # rewards might be object dtype; convert safely to float
        rewards_f = np.array([float(x) for x in rewards], dtype=np.float32)
        ret = float(rewards_f.astype(np.float64).sum())

        # Prefer "term_*" keys but accept unprefixed too.
        success = _any_flag(ep_info, "term_goal_reached", "goal_reached")
        collision = _any_flag(ep_info, "term_collision", "collision")
        time_exceeded = _any_flag(ep_info, "term_time_exceeded", "time_exceeded")
        not_moving = _any_flag(ep_info, "term_not_moving", "not_moving")
        past_goal = _any_flag(ep_info, "term_past_goal", "past_goal")
        offroad = _any_flag(ep_info, "term_offroad", "offroad")
        wrong_lane = _any_flag(ep_info, "term_wrong_lane", "wrong_lane")

        # Episode-level row
        w_ep.writerow({
            "episode_index": int(ep_idx),
            "env_step": int(step),
            "length": int(T),
            "return": float(ret),
            "success": int(success),
            "collision": int(collision),
            "time_exceeded": int(time_exceeded),
            "not_moving": int(not_moving),
            "past_goal": int(past_goal),
            "offroad": int(offroad),
            "wrong_lane": int(wrong_lane),
        })
        f_ep.flush()

        # env_step per step if present in ep; else fallback
        if "env_step" in ep:
            env_steps = _align_series(ep["env_step"], T, default=int(step))
            env_steps = np.array([int(x) for x in env_steps], dtype=np.int64)
        else:
            env_steps = np.arange(T, dtype=np.int64)

        # done flags if present
        done_arr = None
        for k in ("done", "is_last", "is_terminal"):
            if k in ep:
                done_arr = _align_series(ep[k], T, default=0)
                done_arr = np.array([int(bool(x)) for x in done_arr], dtype=np.int32)
                break
        if done_arr is None:
            done_arr = np.zeros((T,), np.int32)

        # actions can be array or dict-of-arrays
        actions = ep.get("action", None)
        action_is_dict = isinstance(actions, dict)

        # Align ep_info streams once
        aligned_info = {}
        for k, v in ep_info.items():
            aligned_info[str(k)] = _align_series(v, T, default=0)

        # Write KV rows
        for t in range(T):
            base = {
                "split": split,
                "episode_index": int(ep_idx),
                "t": int(t),
                "env_step": int(env_steps[t]),
            }

            # Reward and done as KV keys
            w_kv.writerow({**base, "key": "reward", "value": _value_to_string(rewards_f[t])})
            w_kv.writerow({**base, "key": "done", "value": _value_to_string(int(done_arr[t]))})

            # Action as KV keys
            if actions is not None:
                if action_is_dict:
                    for ak, av in actions.items():
                        av_t = _align_series(av, T, default=0)[t]
                        w_kv.writerow({**base, "key": f"action/{ak}", "value": _value_to_string(av_t)})
                else:
                    a_t = _align_series(actions, T, default=0)[t]
                    w_kv.writerow({**base, "key": "action", "value": _value_to_string(a_t)})

            # Every env info key
            for ik, iv in aligned_info.items():
                w_kv.writerow({**base, "key": f"info/{ik}", "value": _value_to_string(iv[t])})

        f_kv.flush()

        # Increment episode index
        if is_eval:
            eval_ep_idx["v"] += 1
        else:
            train_ep_idx["v"] += 1

    # ---------------------------------------------------------------------
    # Episode metrics (logger)
    # ---------------------------------------------------------------------

    def per_episode(ep, ep_info):
        T = int(len(ep["reward"]) - 1)
        if T <= 0:
            return

        rewards = _align_series(ep["reward"], T, default=0)
        rewards_f = np.array([float(x) for x in rewards], dtype=np.float32)

        score = float(rewards_f.astype(np.float64).sum())
        sum_abs_reward = float(np.abs(rewards_f).astype(np.float64).sum())

        success = float(_any_flag(ep_info, "term_goal_reached", "goal_reached"))
        collision = float(_any_flag(ep_info, "term_collision", "collision"))
        past_goal = float(_any_flag(ep_info, "term_past_goal", "past_goal"))
        not_moving = float(_any_flag(ep_info, "term_not_moving", "not_moving"))
        time_exceeded = float(_any_flag(ep_info, "term_time_exceeded", "time_exceeded"))

        logger.add(
            {
                "length": int(T),
                "score": float(score),
                "sum_abs_reward": float(sum_abs_reward),
                "reward_rate": float((np.abs(rewards_f) >= 0.5).mean()),
                "success": success,
                "collision": collision,
                "past_goal": past_goal,
                "not_moving": not_moving,
                "time_exceeded": time_exceeded,
            },
            prefix="episode",
        )

        # Optional aggregated stats from ep (guard against non-array values)
        stats = {}
        for key in args.log_keys_video:
            if key in ep:
                stats[f"policy_{key}"] = ep[key]

        for key, value in ep.items():
            try:
                arr = np.asarray(value)
            except Exception:
                continue
            if not args.log_zeros and key not in nonzeros and arr.size > 0 and (arr == 0).all():
                continue
            nonzeros.add(key)
            if re.match(args.log_keys_sum, key):
                stats[f"sum_{key}"] = float(arr.sum())
            if re.match(args.log_keys_mean, key):
                stats[f"mean_{key}"] = float(arr.mean())
            if re.match(args.log_keys_max, key):
                try:
                    stats[f"max_{key}"] = float(arr.max(0).mean())
                except Exception:
                    pass

        metrics.add(stats, prefix="stats")
        print(f"Episode has {T} steps and return {score:.1f}.")

    # ---------------------------------------------------------------------
    # Drivers
    # ---------------------------------------------------------------------

    driver = embodied.DriverTeacher(env)
    driver.on_episode(lambda ep, ep_info, worker: per_episode(ep, ep_info))
    driver.on_episode(lambda ep, ep_info, worker: _csv_log_everything(ep, ep_info, is_eval=False))
    driver.on_step(lambda tran, info, worker: step.increment())
    driver.on_step(lambda tran, info, worker: replay.add(tran, worker))

    driver_eval = embodied.DriverTeacher(eval_env)
    driver_eval.on_step(lambda trn, inf, i, **kw: eval_replay.add(trn, worker=i))
    driver_eval.on_episode(lambda ep, ep_info, worker: per_episode(ep, ep_info))
    driver_eval.on_episode(lambda ep, ep_info, worker: _csv_log_everything(ep, ep_info, is_eval=True))

    # ---------------------------------------------------------------------
    # Prefill buffers (CSV disabled)
    # ---------------------------------------------------------------------

    print("Prefill train dataset.")
    random_agent = embodied.RandomAgent(env.act_space, args.actor_dist_disc)
    while len(replay) < max(args.batch_steps, args.train_fill):
        driver(random_agent.policy, steps=100)

    print("Prefill eval dataset.")
    while len(eval_replay) < max(args.batch_steps, args.eval_fill):
        driver_eval(random_agent.policy, steps=100)

    # Enable CSV logging AFTER prefill
    csv_enabled["v"] = True

    logger.add(metrics.result())
    logger.write()

    # ---------------------------------------------------------------------
    # Datasets
    # ---------------------------------------------------------------------

    dataset = agent.dataset(replay.dataset)
    dataset_eval = agent.dataset(eval_replay.dataset)

    state = [None]
    batch = [None]

    def train_step(*_args, **_kwargs):
        for _ in range(should_train(step)):
            with timer.scope("dataset"):
                batch[0] = next(dataset)
            outs, state[0], mets = agent.train(batch[0], state[0])
            metrics.add(mets, prefix="train")

            if isinstance(outs, dict) and ("key" in outs):
                replay.prioritize(
                    outs["key"],
                    outs.get("env_step", None),
                    outs.get("model_loss", None),
                    outs.get("td_error", None),
                )
            updates.increment()

        if should_sync(updates) and hasattr(agent, "sync"):
            agent.sync()

        if should_log(step):
            agg = metrics.result()
            logger.add(agg)
            logger.add(replay.stats, prefix="replay")
            logger.add(eval_replay.stats, prefix="eval_replay")
            logger.add(timer.stats(), prefix="timer")

            if batch[0] is not None:
                logger.add(agent.report(batch[0]), prefix="report")

            with timer.scope("dataset_eval"):
                eval_batch = next(dataset_eval)
            logger.add(agent.report(eval_batch), prefix="eval")

            logger.write(fps=True)

    driver.on_step(train_step)

    # ---------------------------------------------------------------------
    # Checkpointing (periodic + best)
    # ---------------------------------------------------------------------

    checkpoint = embodied.Checkpoint(logdir / "checkpoint.ckpt")
    timer.wrap("checkpoint", checkpoint, ["save", "load"])
    checkpoint.step = step
    checkpoint.agent = agent
    checkpoint.replay = replay

    if getattr(args, "from_checkpoint", None):
        checkpoint.load(args.from_checkpoint)
    checkpoint.load_or_save()
    should_save(step)  # register that we just saved

    # Best checkpoint (agent only)
    best_ckpt = embodied.Checkpoint(logdir / "best.ckpt")
    best_ckpt.agent = agent
    best_ckpt.step = step

    best = {"success_rate": -1.0, "mean_return": -1e18, "step": -1}

    # Eval collectors for best selection
    _eval_run = {"returns": [], "success": []}

    def _eval_collect(ep, ep_info, worker):
        T = int(len(ep.get("reward", [])) - 1)
        if T <= 0:
            return
        rewards = _align_series(ep["reward"], T, default=0)
        rewards_f = np.array([float(x) for x in rewards], dtype=np.float32)
        ret = float(rewards_f.astype(np.float64).sum())
        success = int(_any_flag(ep_info, "term_goal_reached", "goal_reached"))
        _eval_run["returns"].append(ret)
        _eval_run["success"].append(success)

    driver_eval.on_episode(lambda ep, ep_info, worker: _eval_collect(ep, ep_info, worker))

    # ---------------------------------------------------------------------
    # Main loop
    # ---------------------------------------------------------------------

    print("Start training loop.")
    driver._state = None

    policy = lambda *a, **k: agent.policy(*a, mode="explore" if should_expl(step) else "train", **k)
    policy_eval = lambda *a, **k: agent.policy(*a, mode="eval", **k)

    # Robust eval episode count
    def _eval_episodes():
        n = int(args.eval_eps)
        try:
            n = max(n, len(eval_env))
        except Exception:
            pass
        return n

    while step < args.steps:
        if should_eval(step):
            print("Starting evaluation at step", int(step))

            _eval_run["returns"].clear()
            _eval_run["success"].clear()

            driver_eval.reset()
            driver_eval(policy_eval, episodes=_eval_episodes())

            if len(_eval_run["returns"]) > 0:
                mean_ret = float(np.mean(_eval_run["returns"]))
                success_rate = float(np.mean(_eval_run["success"]))
            else:
                mean_ret = -1e18
                success_rate = 0.0

            logger.add({
                "eval/mean_return": mean_ret,
                "eval/success_rate": success_rate,
                "best/success_rate": best["success_rate"],
                "best/mean_return": best["mean_return"],
                "best/step": best["step"],
            })
            logger.write()

            improved = (
                (success_rate > best["success_rate"]) or
                (success_rate == best["success_rate"] and mean_ret > best["mean_return"])
            )

            if improved:
                best["success_rate"] = success_rate
                best["mean_return"] = mean_ret
                best["step"] = int(step)

                best_ckpt.step = step
                best_ckpt.save()

                print(
                    f"[BEST] step={int(step)} success_rate={success_rate:.3f} "
                    f"mean_return={mean_ret:.2f} -> saved best.ckpt"
                )

        driver(policy, steps=100)

        if should_save(step):
            checkpoint.save()

    logger.write()
