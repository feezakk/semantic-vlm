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
    
def _parse_int_list(csv_str, default=None):
    if csv_str is None or str(csv_str).strip() == "":
        return [] if default is None else list(default)
    return [int(x) for x in str(csv_str).split(",") if x.strip() != ""]


class EpisodeStatsPrinter:
    def __init__(self, env, tag="train", argv=None, print_every=1):
        self._env = env
        self._tag = tag
        self._print_every = int(print_every)
        argv = argv or []

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
            self._ret = 0.0
            self._len = 0

        return obs, info

    def close(self):
        return self._env.close()

    def __getattr__(self, name):
        return getattr(self._env, name)


def _make_wrapped_env(task_name, argv_list, wrapper_args):
    from embodied.envs import from_gym
    import car_dreamer

    env, _ = car_dreamer.create_task(task_name, argv_list)

    env = from_gym.FromGym(env)
    env = wrap_env(env, wrapper_args)

    tag = "eval" if task_name.endswith("_eval") else "train"
    env = EpisodeStatsPrinter(env, tag=tag, argv=argv_list, print_every=1)

    return env

import argparse
import sys


def _parse_multi_train(argv):
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--carla_train_ports", type=str, required=True)   
    p.add_argument("--carla_train_towns", type=str, required=True)   
    p.add_argument("--carla_eval_port", type=int, required=True)
    p.add_argument("--carla_eval_town", type=str, default=None)
    args, rest = p.parse_known_args(argv)

    train_ports = [int(x) for x in args.carla_train_ports.split(",") if x.strip()]
    train_towns = [x.strip() for x in args.carla_train_towns.split(",") if x.strip()]
    assert len(train_ports) == len(train_towns) and len(train_ports) >= 1
    return args, rest, train_ports, train_towns


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    model_configs = yaml.YAML(typ="safe").load(
        (embodied.Path(__file__).parent / "dreamerv3.yaml").read()
    )
    config = embodied.Config({"dreamerv3": model_configs["defaults"]})

    multi, argv, train_ports, train_towns = _parse_multi_train(argv)

    base_port = train_ports[0]
    base_town = train_towns[0]

    if len(train_ports) != 1 or len(train_towns) != 1:
        print(
            f"[WARN] Weather+single-port mode expects exactly 1 train port and 1 train town. "
            f"Ignoring extras and using port={base_port}, town={base_town}.",
            flush=True,
        )
    train_ports = [base_port]
    train_towns = [base_town]

    if multi.carla_eval_port != base_port:
        print(f"[WARN] Forcing eval port {multi.carla_eval_port} -> {base_port} (shared CARLA).", flush=True)
        multi.carla_eval_port = base_port

    if multi.carla_eval_town is not None and multi.carla_eval_town != base_town:
        print(f"[WARN] Forcing eval town {multi.carla_eval_town} -> {base_town} (shared map).", flush=True)
        multi.carla_eval_town = base_town

    train_weather_ids = _parse_int_list(_argv_get(argv, "--env.train_weather_ids", "0"))
    eval_weather_ids  = _parse_int_list(_argv_get(argv, "--env.eval_weather_ids", "0"))

    print("[DEBUG] base:", base_port, base_town, flush=True)
    print("[DEBUG] train weathers:", train_weather_ids, flush=True)
    print("[DEBUG] eval weathers:", eval_weather_ids, flush=True)

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


        train_env_argvs = []
        env_config = None

        train_argv_i = list(argv) + [
            "--env.world.carla_port", str(base_port),
            "--env.world.town", str(base_town),
            "--env.render", "True",
            "--env.domain_kind", "weather",
            "--env.domain_towns", str(base_town),
        ]
        train_env_argvs.append(train_argv_i)

        tmp_env, cfg_i = car_dreamer.create_task(
            train_name,
            train_argv_i + ["--env.render", "False"],
        )
        env_config = cfg_i
        try:
            tmp_env.close()
        except Exception:
            pass
        eval_env_argv = list(argv) + [
            "--env.world.carla_port", str(base_port),
            "--env.world.town", str(base_town),
            "--env.domain_kind", "weather",
        ]

        tmp_eval_env, eval_env_config = car_dreamer.create_task(
            eval_name,
            eval_env_argv + ["--env.render", "False"],  
        )
        try:
            tmp_eval_env.close()
        except Exception:
            pass

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

    parallel = "process"  
    ParallelEnv = getattr(embodied, "Parallel", None)
    if ParallelEnv is None:
        from embodied.core.parallel import Parallel as ParallelEnv

    class _BatchEnvFutureAdapter:
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
            if isinstance(fut_or_tuple, tuple) and len(fut_or_tuple) == 2:
                obs, info = fut_or_tuple

                if callable(obs) and callable(info):
                    return obs, info

                def obs_thunk(obs=obs):
                    return obs

                def info_thunk(info=info):
                    return info

                return obs_thunk, info_thunk

            fut = fut_or_tuple
            cache = {}

            def _resolve_once():
                if "val" in cache:
                    return cache["val"]

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
            return self._split(self._env.reset())

        def close(self):
            return self._env.close()

        def __getattr__(self, name):
            return getattr(self._env, name)


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

    eval_argv_worker = eval_env_argv + ["--env.render", "True"]

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
        return float(np.max(np.asarray(space.high)))

    expected_hi = float(int(dreamerv3_config.num_domains) - 1)

    need_domain = (
        float(dreamerv3_config.loss_scales.domain_adv) > 0.0 or
        float(dreamerv3_config.loss_scales.domain_sty) > 0.0 or
        float(dreamerv3_config.loss_scales.domain_probe) > 0.0
    )

    def _max_high(space):
        return float(np.max(np.asarray(space.high)))

    if need_domain:
        assert "domain_id" in env.obs_space, (
            "[ENV BUG] domain_id missing from TRAIN env obs_space, "
            "but domain losses are enabled."
        )
        train_hi = _max_high(env.obs_space["domain_id"])
        assert train_hi <= expected_hi + 1e-6, (
            f"[ENV BUG] TRAIN domain_id high={train_hi} exceeds num_domains-1={expected_hi}. "
            f"num_domains={dreamerv3_config.num_domains}"
        )

        if "domain_id" in eval_env.obs_space:
            eval_hi = _max_high(eval_env.obs_space["domain_id"])
            assert eval_hi <= expected_hi + 1e-6, (
                f"[ENV BUG] EVAL domain_id high={eval_hi} exceeds num_domains-1={expected_hi}. "
                f"num_domains={dreamerv3_config.num_domains}"
            )

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

    embodied.run.train_vlm_weather(agent, env, eval_env, replay, eval_replay, logger, args)

if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
