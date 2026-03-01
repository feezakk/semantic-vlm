import datetime
import warnings

import embodied
import ruamel.yaml as yaml

import car_dreamer
import dreamerv3

import csv, atexit

import functools

import os
import numpy as np


warnings.filterwarnings("ignore", ".*truncated to dtype int32.*")


# def wrap_env(env, config):
#     args = config.wrapper
def wrap_env(env, wrapper_args):
    args = wrapper_args    
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

def _argv_get(argv, key, default=""):
    try:
        i = argv.index(key)
        return argv[i + 1]
    except (ValueError, IndexError):
        return default


class EpisodeStatsPrinter:
    """
    Prints one line when an episode ends.
    Designed to work with Dreamer-style obs dicts that include:
      - reward
      - is_first
      - is_last
    """

    def __init__(self, env, tag="train", argv=None, print_every=1):
        self._env = env
        self._tag = tag
        self._print_every = int(print_every)
        argv = argv or []

        # Useful identifiers when using process-parallel envs
        self._town = _argv_get(argv, "--env.world.town", "")
        self._port = _argv_get(argv, "--env.world.carla_port", "")
        self._pid = os.getpid()

        self._ep = 0
        self._ret = 0.0
        self._len = 0

    @property
    def obs_space(self):
        return self._env.obs_space

    @property
    def act_space(self):
        return self._env.act_space

    def reset(self):
        out = self._env.reset()
        self._ret = 0.0
        self._len = 0
        return out

    def step(self, action):
        obs, info = self._env.step(action)

        # In Dreamer-style envs, the "reset transition" comes back as is_first=True.
        # Do not count it as a step.
        is_first = bool(np.asarray(obs.get("is_first", False)).item())
        if is_first:
            self._ret = 0.0
            self._len = 0
            return obs, info

        r = float(np.asarray(obs.get("reward", 0.0)).reshape(-1)[0])
        self._ret += r
        self._len += 1

        is_last = bool(np.asarray(obs.get("is_last", False)).item())
        if is_last:
            self._ep += 1
            if self._print_every > 0 and (self._ep % self._print_every == 0):
                # Optional fields (only printed if present).
                reason = info.get("reason", info.get("termination_reason", ""))
                route = info.get("route", info.get("route_id", ""))
                weather = info.get("weather", info.get("weather_id", ""))

                msg = f"[{self._tag}] pid={self._pid}"
                if self._town:
                    msg += f" town={self._town}"
                if self._port:
                    msg += f" port={self._port}"
                msg += f" ep={self._ep} steps={self._len} return={self._ret:.2f}"
                if reason:
                    msg += f" reason={reason}"
                if route != "":
                    msg += f" route={route}"
                if weather != "":
                    msg += f" weather={weather}"
                print(msg, flush=True)

            # Ready for next episode (will also be reset on next is_first)
            self._ret = 0.0
            self._len = 0

        return obs, info

    def close(self):
        return self._env.close()

    def __getattr__(self, name):
        return getattr(self._env, name)


def _make_wrapped_env(task_name, argv_list, wrapper_args):
    """
    Factory used by embodied.Parallel(..., parallel='process').
    Must be top-level (picklable).
    """
    from embodied.envs import from_gym
    import car_dreamer

    env, _ = car_dreamer.create_task(task_name, argv_list)
    # env = from_gym.FromGym(env)
    # env = wrap_env(env, wrapper_args)
    # return env

    env = from_gym.FromGym(env)
    env = wrap_env(env, wrapper_args)

    tag = "eval" if task_name.endswith("_eval") else "train"
    env = EpisodeStatsPrinter(env, tag=tag, argv=argv_list, print_every=1)

    return env



import argparse
import sys

# def _parse_dual_carla(argv):
#     p = argparse.ArgumentParser(add_help=False)
#     p.add_argument("--carla_train_port", type=int, required=True)
#     p.add_argument("--carla_eval_port", type=int, required=True)
#     p.add_argument("--carla_train_town", type=str, default=None)
#     p.add_argument("--carla_eval_town", type=str, default=None)
#     args, rest = p.parse_known_args(argv)
#     return args, rest

def _parse_multi_train(argv):
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--carla_train_ports", type=str, required=True)   # e.g. "3000,3010"
    p.add_argument("--carla_train_towns", type=str, required=True)   # e.g. "Town05,Town06"
    p.add_argument("--carla_eval_port", type=int, required=True)
    p.add_argument("--carla_eval_town", type=str, default=None)
    args, rest = p.parse_known_args(argv)

    train_ports = [int(x) for x in args.carla_train_ports.split(",") if x.strip()]
    train_towns = [x.strip() for x in args.carla_train_towns.split(",") if x.strip()]
    assert len(train_ports) == len(train_towns) and len(train_ports) >= 2
    return args, rest, train_ports, train_towns


# def main(argv=None):
#     if argv is None:
#         argv = sys.argv[1:]
#     model_configs = yaml.YAML(typ="safe").load((embodied.Path(__file__).parent / "dreamerv3.yaml").read())
#     config = embodied.Config({"dreamerv3": model_configs["defaults"]})
#     # config = config.update({"dreamerv3": model_configs["small"]})

#     # dual, argv = _parse_dual_carla(argv)
#     multi, argv, train_ports, train_towns = _parse_multi_train(argv)

#     print("[DEBUG] argv seen by port parser:", argv[:20])
#     print("[DEBUG] train/eval ports:", dual.carla_train_port, dual.carla_eval_port)

def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    model_configs = yaml.YAML(typ="safe").load(
        (embodied.Path(__file__).parent / "dreamerv3.yaml").read()
    )
    config = embodied.Config({"dreamerv3": model_configs["defaults"]})

    multi, argv, train_ports, train_towns = _parse_multi_train(argv)

    print("[DEBUG] argv after multi parser:", argv[:20])
    print("[DEBUG] train:", list(zip(train_ports, train_towns)))
    print("[DEBUG] eval:", multi.carla_eval_port, multi.carla_eval_town)

    parsed, other = embodied.Flags(task=["carla_navigation"]).parse_known(argv)
    for name in parsed.task:
        print("Using task: ", name)
        eval_name = name + "_eval"
        train_name = name + "_train"
        print("Using train task: ", train_name)
        print("Using eval task: ", eval_name)

        # train_envs = []
        # env_config = None

        # for i, (port, town) in enumerate(zip(train_ports, train_towns)):
        #     train_argv_i = list(argv) + [
        #         "--env.world.carla_port", str(port),
        #         "--env.world.town", str(town),

        #         # Tell env to produce town-domain labels:
        #         "--env.domain_kind", "town",
        #         "--env.domain_towns", ",".join(train_towns),

        #         # make sure the observation actually includes it
        #         "--env.include_domain_id", "True",

        #         # optional: decorrelate randomness across workers
        #         "--env.seed", str(i),
        #     ]
        #     env_i, cfg_i = car_dreamer.create_task(train_name, train_argv_i)
        #     train_envs.append(env_i)

        #     if env_config is None:
        #         env_config = cfg_i  # keep first config for merging into dreamerv3 config

        # # --- EVAL ENV (single) ---
        # eval_argv = list(argv) + [
        #     "--env.world.carla_port", str(multi.carla_eval_port),
        #     "--env.include_domain_id", "True",
        #     "--env.domain_kind", "town",
        #     "--env.domain_towns", ",".join(train_towns),
        # ]
        # if multi.carla_eval_town is not None:
        #     eval_argv += ["--env.world.town", str(multi.carla_eval_town)]

        # eval_env, eval_env_config = car_dreamer.create_task(eval_name, eval_argv)

        # ---------------------------------------------------------------------
        # Build argv lists for each env (TRAIN towns + EVAL town).
        # We also create ONE temporary env to obtain env_config for config merging,
        # but we do NOT keep env objects in the main process.
        # ---------------------------------------------------------------------

        train_env_argvs = []
        env_config = None

        for i, (port, town) in enumerate(zip(train_ports, train_towns)):
            # Full argv for this specific training env worker process
            # train_argv_i = list(argv) + [
            #     "--env.world.carla_port", str(port),
            #     "--env.world.town", str(town),

            #     # town-domain labels
            #     "--env.domain_kind", "town",
            #     "--env.domain_towns", ",".join(train_towns),

            #     # make sure the observation includes it
            #     "--env.include_domain_id", "True",

            #     # decorrelate randomness
            #     "--env.seed", str(i),
            # ]

            train_argv_i = list(argv) + [
                "--env.world.carla_port", str(port),
                "--env.world.town", str(town),

                # force pygame render for each training town (Town05 + Town06)
                "--env.render", "True",

                # town-domain labels
                "--env.domain_kind", "town",
                "--env.domain_towns", ",".join(train_towns),

                "--env.include_domain_id", "True",
                "--env.seed", str(i),
            ]

            train_env_argvs.append(train_argv_i)

            # Only need cfg from ONE train env for config merging.
            if env_config is None:
                tmp_env, cfg_i = car_dreamer.create_task(
                    train_name,
                    train_argv_i + ["--env.render", "False"],  # avoid pygame window in main proc
                )
                env_config = cfg_i
                try:
                    tmp_env.close()
                except Exception:
                    pass

        # --- EVAL env argv ---
        eval_env_argv = list(argv) + [
            "--env.world.carla_port", str(multi.carla_eval_port),
            "--env.include_domain_id", "True",
            "--env.domain_kind", "town",
            "--env.domain_towns", ",".join(train_towns),
        ]
        if multi.carla_eval_town is not None:
            eval_env_argv += ["--env.world.town", str(multi.carla_eval_town)]

        # Need eval_env_config for merging; don't keep the env instance.
        tmp_eval_env, eval_env_config = car_dreamer.create_task(
            eval_name,
            eval_env_argv + ["--env.render", "False"],  # avoid pygame window in main proc
        )
        try:
            tmp_eval_env.close()
        except Exception:
            pass


        # Merge configs
        config = config.update(env_config)
        eval_config = config.update(eval_env_config)

    config = embodied.Flags(config).parse(other)
    eval_config = embodied.Flags(eval_config).parse(other)

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

    # Hard correctness check: domain head must be 4-way (clear/cloudy/wet/rain).
    # assert int(dreamerv3_config.num_domains) == 4, (
    #     f"[CONFIG BUG] dreamerv3.num_domains must be 4, got {dreamerv3_config.num_domains}. "
    #     "This usually means a YAML/CLI override or env_config.update() changed it."
    # )

    assert int(dreamerv3_config.num_domains) == len(train_towns), (
        f"dreamerv3.num_domains must equal #train towns ({len(train_towns)}), "
        f"got {dreamerv3_config.num_domains}."
    )

    # env = from_gym.FromGym(env)
    # env = wrap_env(env, dreamerv3_config)
    # env = embodied.BatchEnv([env], parallel=False)

    # wrapped_train = []
    # for e in train_envs:
    #     e = from_gym.FromGym(e)
    #     e = wrap_env(e, dreamerv3_config)
    #     wrapped_train.append(e)

    # parallel = getattr(dreamerv3_config.envs, "parallel", "process")  # e.g. "process"
    # env = embodied.BatchEnv(wrapped_train, parallel=parallel)

    # # env = embodied.BatchEnv(wrapped_train, parallel=False)  # start stable; later you can parallelize


    # eval_env = from_gym.FromGym(eval_env)
    # eval_env = wrap_env(eval_env, dreamerv3_config)
    # # eval_env = embodied.BatchEnv([eval_env], parallel=False)
    # eval_env = embodied.BatchEnv([eval_env], parallel=parallel)

    # ---------------------------------------------------------------------
    # Create process-parallel BatchEnvs so each env has its OWN pygame window.
    # ---------------------------------------------------------------------

    # train_env_fns = [
    #     functools.partial(_make_wrapped_env, train_name, argv_i, dreamerv3_config)
    #     for argv_i in train_env_argvs
    # ]
    # env = embodied.BatchEnv(train_env_fns, parallel="process")

    # # If you want Town04 eval to render in its own window too, keep eval enabled.
    # # If you do NOT want an eval window, append ["--env.render","False"] here.
    # eval_env_fns = [
    #     functools.partial(_make_wrapped_env, eval_name, eval_env_argv, dreamerv3_config)
    # ]
    # eval_env = embodied.BatchEnv(eval_env_fns, parallel="process")

    # ---------------------------------------------------------------------
    # Correct pattern for this embodied version:
    #   env_i = Parallel(make_env_fn, parallel='process')
    #   env   = BatchEnv([env_i...], parallel='process')
    # ---------------------------------------------------------------------

    parallel = "process"  # required for separate pygame windows per env

    # Some forks expose Parallel as embodied.Parallel; others as embodied.core.parallel.Parallel
    ParallelEnv = getattr(embodied, "Parallel", None)
    if ParallelEnv is None:
        from embodied.core.parallel import Parallel as ParallelEnv

    class _BatchEnvFutureAdapter:
        """
        Adapts envs whose step() returns a single Future (resolving to (obs, info))
        into the API expected by embodied.core.batch.BatchEnv:
        step() must return (obs_thunk, info_thunk), each callable.
        """

        def __init__(self, env):
            self._env = env

        def __len__(self):
            return len(self._env)

        @property
        def obs_space(self):
            return self._env.obs_space

        @property
        def act_space(self):
            return self._env.act_space

        def _split(self, fut_or_tuple):
            # Case A: already a 2-tuple (obs, info) or (obs_thunk, info_thunk)
            if isinstance(fut_or_tuple, tuple) and len(fut_or_tuple) == 2:
                obs, info = fut_or_tuple

                # If they are already thunks, keep them.
                if callable(obs) and callable(info):
                    return obs, info

                # Otherwise wrap immediate values into thunks (BatchEnv will call them).
                def obs_thunk(obs=obs):
                    return obs

                def info_thunk(info=info):
                    return info

                return obs_thunk, info_thunk

            # Case B: single Future-like that resolves to (obs, info).
            fut = fut_or_tuple
            cache = {}

            def _resolve_once():
                if "val" in cache:
                    return cache["val"]

                # Many embodied futures are callable; others may have .result()
                if callable(fut):
                    cache["val"] = fut()
                elif hasattr(fut, "result"):
                    cache["val"] = fut.result()
                else:
                    raise TypeError(f"Unsupported future type from env.step(): {type(fut)}")

                return cache["val"]

            def obs_thunk():
                return _resolve_once()[0]

            def info_thunk():
                return _resolve_once()[1]

            return obs_thunk, info_thunk

        def step(self, action):
            return self._split(self._env.step(action))

        def reset(self):
            # If your Parallel env reset() also returns a Future, handle it too.
            return self._split(self._env.reset())

        def close(self):
            return self._env.close()

        def __getattr__(self, name):
            return getattr(self._env, name)


    # train_envs = [
    #     ParallelEnv(
    #         functools.partial(_make_wrapped_env, train_name, argv_i, dreamerv3_config.wrapper),
    #         parallel,
    #     )
    #     for argv_i in train_env_argvs
    # ]
    # env = embodied.BatchEnv(train_envs, parallel=parallel)

    train_envs = [
        _BatchEnvFutureAdapter(
            ParallelEnv(
                functools.partial(_make_wrapped_env, train_name, argv_i, dreamerv3_config.wrapper),
                parallel,
            )
        )
        for argv_i in train_env_argvs
    ]
    env = embodied.BatchEnv(train_envs, parallel=True)


    # Eval: I recommend headless while debugging multi-window training.
    # If you WANT an eval pygame window too, remove the two args below.
    eval_argv_worker = eval_env_argv + ["--env.render", "True"]
    # eval_envs = [
    #     ParallelEnv(
    #         functools.partial(_make_wrapped_env, eval_name, eval_argv_worker, dreamerv3_config.wrapper),
    #         parallel,
    #     )
    # ]
    # eval_env = embodied.BatchEnv(eval_envs, parallel=parallel)

    eval_envs = [
        _BatchEnvFutureAdapter(
            ParallelEnv(
                functools.partial(_make_wrapped_env, eval_name, eval_argv_worker, dreamerv3_config.wrapper),
                parallel,
            )
        )
    ]
    eval_env = embodied.BatchEnv(eval_envs, parallel=True)




    print("Wrapped obs keys:", list(env.obs_space.keys()))
    print("Wrapped act keys:", list(env.act_space.keys()))
    print("Wrapped domain_id space:", env.obs_space.get("domain_id", None))
    
    import numpy as np

    def _max_high(space):
        # Works for scalar or vector highs.
        return float(np.max(np.asarray(space.high)))

    expected_hi = float(int(dreamerv3_config.num_domains) - 1)

    need_domain = (
        float(dreamerv3_config.loss_scales.domain_adv) > 0.0 or
        float(dreamerv3_config.loss_scales.domain_sty) > 0.0 or
        float(dreamerv3_config.loss_scales.domain_probe) > 0.0
    )
    if need_domain:
        assert "domain_id" in env.obs_space, "[ENV BUG] domain_id missing from train env obs_space."
        assert "domain_id" in eval_env.obs_space, "[ENV BUG] domain_id missing from eval env obs_space."

    need_vlm = float(dreamerv3_config.loss_scales.sem_rollout) > 0.0
    if need_vlm:
        assert "vlm" in env.obs_space
        assert "vlm" in eval_env.obs_space

    # train_hi = _max_high(env.obs_space["domain_id"])
    # eval_hi  = _max_high(eval_env.obs_space["domain_id"])

    # assert abs(train_hi - expected_hi) < 1e-6, (
    #     f"[ENV/CONFIG MISMATCH] Train env domain_id high={train_hi}, expected {expected_hi}. "
    #     "Env and model disagree on num_domains."
    # )
    # assert abs(eval_hi - expected_hi) < 1e-6, (
    #     f"[ENV/CONFIG MISMATCH] Eval env domain_id high={eval_hi}, expected {expected_hi}. "
    #     "Env and model disagree on num_domains."
    # )

    # print(f"[OK] num_domains={int(dreamerv3_config.num_domains)} and env domain_id range is [0, {expected_hi}].")

    if "domain_id" in env.obs_space and "domain_id" in eval_env.obs_space:
        train_hi = _max_high(env.obs_space["domain_id"])
        eval_hi  = _max_high(eval_env.obs_space["domain_id"])
        # asserts if you want
    else:
        print("[Info] domain_id not present in obs_space; skipping checks.")



    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    config_filename = f"config_{timestamp}.yaml"
    config.save(str(logdir / config_filename))
    print(f"[Train] Config saved to {logdir / config_filename}")

    agent = dreamerv3.agent_vlm(env.obs_space, env.act_space, step, dreamerv3_config)

    replay = embodied.replay.Uniform(dreamerv3_config.batch_length, dreamerv3_config.replay_size, logdir / "replay")
    eval_replay = embodied.replay.Uniform(dreamerv3_config.batch_length, dreamerv3_config.replay_size, logdir / "eval_replay")

    args = embodied.Config(
        **dreamerv3_config.run,
        logdir=dreamerv3_config.logdir,
        batch_steps=dreamerv3_config.batch_size * dreamerv3_config.batch_length,
        actor_dist_disc=dreamerv3_config.actor_dist_disc,
    )

    embodied.run.train_vlm_town(agent, env, eval_env, replay, eval_replay, logger, args)


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
    # carla_args, remaining = _parse_dual_carla(sys.argv[1:])
    # main(remaining, carla_args)
    # main()
