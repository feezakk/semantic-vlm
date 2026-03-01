import datetime
import warnings

import embodied
import ruamel.yaml as yaml

import car_dreamer
import dreamerv3

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

import csv
import numpy as np

class CSVScalarOutput:
    """
    Writes metrics in long format:
      step,name,value
    This is robust to changing metric keys over time.
    """
    def __init__(self, logdir, filename="metrics.csv"):
        self._path = embodied.Path(logdir) / filename
        self._file = self._path.open("a")
        self._writer = csv.writer(self._file)
        if self._file.tell() == 0:
            self._writer.writerow(["step", "name", "value"])

    def __call__(self, *args):
        # Support either output(step, metrics) OR output(metrics)
        if len(args) == 2:
            step, metrics = args
        elif len(args) == 1:
            (metrics,) = args
            step = metrics.get("step", None)
        else:
            return

        # Convert step to int if possible
        try:
            step = int(step)
        except Exception:
            try:
                step = int(getattr(step, "value", 0))
            except Exception:
                step = 0

        for k, v in metrics.items():
            if isinstance(v, dict):
                continue
            # Only write scalar-like values
            try:
                arr = np.array(v)
                if arr.shape != ():
                    continue
                self._writer.writerow([step, k, float(arr)])
            except Exception:
                continue

        self._file.flush()



def main(argv=None):
    model_configs = yaml.YAML(typ="safe").load((embodied.Path(__file__).parent / "dreamerv3.yaml").read())
    config = embodied.Config({"dreamerv3": model_configs["defaults"]})
    # config = config.update({"dreamerv3": model_configs["small"]})

    parsed, other = embodied.Flags(task=["carla_navigation"]).parse_known(argv)
    for name in parsed.task:
        print("Using task: ", name)
        eval_name = name + "_test"
        print("Using eval task: ", eval_name)

        print("1*******************************")

        env, env_config = car_dreamer.create_task(name, argv)
        print("2*******************************")
        eval_env,    eval_env_config = car_dreamer.create_task(eval_name, argv)
        print("3*******************************")

        config = config.update(env_config)
        print("4*******************************")
        eval_config = config.update(eval_env_config)
        print("5*******************************")

    print("*******************************")

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
            CSVScalarOutput(logdir, "metrics.csv"),
        ],
    )

    from embodied.envs import from_gym

    dreamerv3_config = config.dreamerv3

    env = from_gym.FromGym(env)
    env = wrap_env(env, dreamerv3_config)
    env = embodied.BatchEnv([env], parallel=False)

    eval_env = from_gym.FromGym(eval_env)
    eval_env = wrap_env(eval_env, dreamerv3_config)
    eval_env = embodied.BatchEnv([eval_env], parallel=False)

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    config_filename = f"config_{timestamp}.yaml"
    config.save(str(logdir / config_filename))
    print(f"[Train] Config saved to {logdir / config_filename}")

    timer = embodied.Timer()

    args = embodied.Config(
        **dreamerv3_config.run,
        logdir=dreamerv3_config.logdir,
        batch_steps=dreamerv3_config.batch_size * dreamerv3_config.batch_length,
        actor_dist_disc=dreamerv3_config.actor_dist_disc,
    )

    teacher_agent = dreamerv3.agent_teacher(env.obs_space, env.act_space, step, dreamerv3_config)
    teacher_replay = embodied.replay.Uniform(dreamerv3_config.batch_length, dreamerv3_config.replay_size, logdir / "teacher_replay")
    timer.wrap("agent", teacher_agent, ["policy", "train", "report", "save"])
    timer.wrap("env", env, ["step"])
    timer.wrap("replay", teacher_replay, ["add", "save"])
    timer.wrap("logger", logger, ["write"])

    expert = embodied.Checkpoint(logdir / "teacher.ckpt")
    timer.wrap("expert", expert, ["save", "load"])
    # expert.step = step
    expert.agent = teacher_agent
    expert.replay = teacher_replay
    expert.load()  

    teacher_policy = lambda *args: teacher_agent.policy(*args, mode="eval")

    tp = teacher_agent.agent.task_behavior.ac.policy
    teacher_wm = teacher_agent.agent.wm
    agent = dreamerv3.agent_student(env.obs_space, env.act_space, teacher_wm, tp,  step, dreamerv3_config)  
    replay = embodied.replay.Uniform(dreamerv3_config.batch_length, dreamerv3_config.replay_size, logdir / "replay")
    eval_replay = embodied.replay.Uniform(dreamerv3_config.batch_length, dreamerv3_config.replay_size, logdir / "eval_replay")  
    
    embodied.run.train_student(agent, teacher_policy, env, eval_env, replay, eval_replay, teacher_replay, logger, args)


if __name__ == "__main__":
    main()
