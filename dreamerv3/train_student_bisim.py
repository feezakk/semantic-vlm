import datetime
import warnings

import embodied
import ruamel.yaml as yaml

import car_dreamer
import dreamerv3

import jax
import jax.numpy as jnp
import numpy as np

warnings.filterwarnings("ignore", ".*truncated to dtype int32.*")

from jax import config
config.update("jax_transfer_guard", "allow")  # or "log" / "warn"


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
        eval_name = name + "_test"
        print("Using eval task: ", eval_name)
        print("1**********************")

        env, env_config = car_dreamer.create_task(name, argv)
        print("2**********************")
        eval_env,    eval_env_config = car_dreamer.create_task(eval_name, argv)
        print("3**********************")

        config = config.update(env_config)
        print("4**********************")
        eval_config = config.update(eval_env_config)
        print("5**********************")

    config = embodied.Flags(config).parse(other)
    eval_config = embodied.Flags(eval_config).parse(other)

    logdir = embodied.Path(config.dreamerv3.logdir)

    # -------------------- student counter + logger --------------------
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

    shared_replay = embodied.replay.Uniform(dreamerv3_config.batch_length, dreamerv3_config.replay_size, logdir / "teacher_replay")

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


    teacher_step = embodied.Counter()  # separate counter for teacher
    teacher_agent = dreamerv3.agent_teacher(env.obs_space, env.act_space, teacher_step, dreamerv3_config)
    teacher_replay = shared_replay

    # teacher_replay = embodied.replay.Uniform(dreamerv3_config.batch_length, dreamerv3_config.replay_size, logdir / "replay")
    timer.wrap("agent", teacher_agent, ["policy", "train", "report", "save"])
    timer.wrap("env", env, ["step"])
    timer.wrap("replay", teacher_replay, ["add", "save"])
    timer.wrap("logger", logger, ["write"])

    expert = embodied.Checkpoint(logdir / "teacher.ckpt")
    timer.wrap("expert", expert, ["save", "load"])
    expert.step = teacher_step
    expert.agent = teacher_agent
    expert.replay = teacher_replay
    expert.load()  

    teacher_policy = lambda *args: teacher_agent.policy(*args, mode="eval")

    tp = teacher_agent.agent.task_behavior.ac.policy
    teacher_wm = teacher_agent.agent.wm

    agent = dreamerv3.agent_student_bisim(env.obs_space, env.act_space, teacher_wm, tp,  step, dreamerv3_config)  

    import jax
    import jax.numpy as jnp
    import numpy as np

    teacher_vars = teacher_agent.save()
    student_vars = agent.save()

    copied = 0
    missing = 0
    shape_mismatch = 0

    for k, v in teacher_vars.items():
        if not k.startswith("agent/wm/"):
            continue
        k_student = k.replace("agent/wm/", "agent/teacher_wm/")

        if k_student not in student_vars:
            print("[MISSING IN STUDENT]", k_student)
            missing += 1
            continue

        if student_vars[k_student].shape != v.shape:
            print("[SHAPE MISMATCH]", k_student,
                "student", student_vars[k_student].shape,
                "teacher", v.shape)
            shape_mismatch += 1
            continue

        student_vars[k_student] = v
        copied += 1

    print(f"[WM copy] Copied {copied} tensors, missing={missing}, shape_mismatch={shape_mismatch}")

    # -------------------- NEW: copy teacher ACTOR -> student teacher_actor --------------------
    copied_actor = 0
    missing_actor = 0
    shape_mismatch_actor = 0

    for k, v in teacher_vars.items():
        # Teacher actor params live here (verify by printing keys if needed)
        if not k.startswith("agent/task_behavior/ac/actor/"):
            continue

        k_student = k.replace("agent/task_behavior/ac/actor/", "agent/task_behavior/ac/teacher_actor/")

        if k_student not in student_vars:
            print("[MISSING IN STUDENT]", k_student)
            missing_actor += 1
            continue

        if student_vars[k_student].shape != v.shape:
            print("[SHAPE MISMATCH]", k_student,
                "student", student_vars[k_student].shape,
                "teacher", v.shape)
            shape_mismatch_actor += 1
            continue

        student_vars[k_student] = v
        copied_actor += 1

    print(f"[Actor copy] Copied {copied_actor} tensors, missing={missing_actor}, shape_mismatch={shape_mismatch_actor}")
    # -----------------------------------------------------------------------------------------



    agent.load(student_vars)

    def subtree_norm(vars_dict, prefix):
        arrs = [v for k, v in vars_dict.items() if k.startswith(prefix)]
        if not arrs:
            print(prefix, "NO PARAMS FOUND")
            return
        flat = np.concatenate([np.ravel(np.array(v)) for v in arrs])
        print(prefix, "L2 =", np.linalg.norm(flat))

    sv = agent.save()
    subtree_norm(sv, "agent/task_behavior/ac/actor/")
    subtree_norm(sv, "agent/task_behavior/ac/teacher_actor/")


    print("***************************************************************")
    print(f"[WM copy] Copied {copied} teacher WM tensors into student WM.")
    print("***************************************************************")
    
    def param_norm(agent_obj, label):
        vars_dict = agent_obj.save()

        wm_arrays = [
            v
            for k, v in vars_dict.items()
            if k.startswith("agent/wm/") and isinstance(v, (np.ndarray, jnp.ndarray))
        ]

        if not wm_arrays:
            print(f"[{label}] No agent/wm/ parameter arrays found.")
            return

        # Explicitly move to host to avoid implicit device->host transfer
        host_arrays = [np.ravel(jax.device_get(v)) for v in wm_arrays]
        flat = np.concatenate(host_arrays)
        norm = np.linalg.norm(flat)

        print(f"[{label}] ||agent/wm||_2 = {norm:.6e}")

    param_norm(teacher_agent, "Teacher")
    param_norm(agent, "Student (after copy)")

    def wm_subtree_norm(vars_dict, prefix):
        arrs = [v for k, v in vars_dict.items()
                if k.startswith(prefix) and isinstance(v, (np.ndarray, jnp.ndarray))]
        if not arrs:
            print(f"[{prefix}] No params found")
            return
        flat = np.concatenate([np.ravel(v) for v in arrs])
        print(f"[{prefix}] L2 norm = {np.linalg.norm(flat):.6e}")

    teacher_vars  = teacher_agent.save()
    student_vars  = agent.save()

    wm_subtree_norm(teacher_vars, "agent/wm/")
    wm_subtree_norm(student_vars, "agent/teacher_wm/")

    replay = shared_replay
    eval_replay = embodied.replay.Uniform(dreamerv3_config.batch_length, dreamerv3_config.replay_size, logdir / "eval_replay")  
    
    embodied.run.train_student_bisim(agent, teacher_policy, env, eval_env, replay, eval_replay, teacher_replay, logger, args)


if __name__ == "__main__":
    main()
