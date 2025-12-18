import datetime
import warnings

import embodied
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


def main(argv=None):
    model_configs = yaml.YAML(typ="safe").load((embodied.Path(__file__).parent / "dreamerv3.yaml").read())
    config = embodied.Config({"dreamerv3": model_configs["defaults"]})
    # config = config.update({"dreamerv3": model_configs["small"]})

    parsed, other = embodied.Flags(task=["carla_navigation"]).parse_known(argv)
    for name in parsed.task:
        print("Using task: ", name)
        eval_name = name + "_eval"
        train_name = name + "_train"
        print("Using train task: ", train_name)
        print("Using eval task: ", eval_name)
        env, env_config = car_dreamer.create_task(train_name, argv)

        print("car_dreamer loaded from:", __import__("car_dreamer").__file__)
        print("enabled:", getattr(env_config.env.observation, "enabled", None))
        print("has domain_id cfg:", hasattr(env_config.env.observation, "domain_id"))

        print("RAW gym obs keys:", list(env.observation_space.spaces.keys()))
        print("RAW gym domain_id space:", env.observation_space.spaces.get("domain_id", None))

        eval_env, eval_env_config = car_dreamer.create_task(eval_name, argv)
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
    assert int(dreamerv3_config.num_domains) == 4, (
        f"[CONFIG BUG] dreamerv3.num_domains must be 4, got {dreamerv3_config.num_domains}. "
        "This usually means a YAML/CLI override or env_config.update() changed it."
    )

    env = from_gym.FromGym(env)
    env = wrap_env(env, dreamerv3_config)
    env = embodied.BatchEnv([env], parallel=False)

    eval_env = from_gym.FromGym(eval_env)
    eval_env = wrap_env(eval_env, dreamerv3_config)
    eval_env = embodied.BatchEnv([eval_env], parallel=False)

    print("Wrapped obs keys:", list(env.obs_space.keys()))
    print("Wrapped act keys:", list(env.act_space.keys()))
    print("Wrapped domain_id space:", env.obs_space.get("domain_id", None))
    
    import numpy as np

    def _max_high(space):
        # Works for scalar or vector highs.
        return float(np.max(np.asarray(space.high)))

    expected_hi = float(int(dreamerv3_config.num_domains) - 1)

    assert "domain_id" in env.obs_space, "[ENV BUG] domain_id missing from train env obs_space."
    assert "domain_id" in eval_env.obs_space, "[ENV BUG] domain_id missing from eval env obs_space."

    train_hi = _max_high(env.obs_space["domain_id"])
    eval_hi  = _max_high(eval_env.obs_space["domain_id"])

    assert abs(train_hi - expected_hi) < 1e-6, (
        f"[ENV/CONFIG MISMATCH] Train env domain_id high={train_hi}, expected {expected_hi}. "
        "Env and model disagree on num_domains."
    )
    assert abs(eval_hi - expected_hi) < 1e-6, (
        f"[ENV/CONFIG MISMATCH] Eval env domain_id high={eval_hi}, expected {expected_hi}. "
        "Env and model disagree on num_domains."
    )

    print(f"[OK] num_domains={int(dreamerv3_config.num_domains)} and env domain_id range is [0, {expected_hi}].")


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

    embodied.run.train_vlm(agent, env, eval_env, replay, eval_replay, logger, args)


if __name__ == "__main__":
    main()
