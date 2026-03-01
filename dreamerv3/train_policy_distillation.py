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

        env, env_config = car_dreamer.create_task(name, argv)
        eval_env,    eval_env_config = car_dreamer.create_task(eval_name, argv)

        config = config.update(env_config)
        eval_config = config.update(eval_env_config)

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

    agent = dreamerv3.agent_policy_distillation(env.obs_space, env.act_space, teacher_wm, tp,  step, dreamerv3_config)  

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

    # -------------------------------
    # Copy TEACHER ACTOR -> student teacher_actor
    # -------------------------------
    teacher_vars = teacher_agent.save()
    student_vars = agent.save()

    # 1) Inspect: find teacher actor keys
    teacher_actor_keys = [
        k for k in teacher_vars.keys()
        if ("/task_behavior/" in k and "/ac/" in k and "/actor/" in k)
    ]
    print(f"[Teacher actor] found {len(teacher_actor_keys)} tensors")
    print("\n".join(teacher_actor_keys[:20]))

    # 2) Inspect: ensure student teacher_actor keys exist (creation pass worked)
    student_teacher_actor_keys = [
        k for k in student_vars.keys()
        if ("/task_behavior/" in k and "/ac/" in k and "/teacher_actor/" in k)
    ]
    print(f"[Student teacher_actor] found {len(student_teacher_actor_keys)} tensors")
    print("\n".join(student_teacher_actor_keys[:20]))

    # 3) Copy with mapping: actor -> teacher_actor
    copied = 0
    missing = 0
    shape_mismatch = 0
    mapping = []  # (teacher_key, student_key)

    for k, v in teacher_vars.items():
        if ("/task_behavior/" in k and "/ac/" in k and "/actor/" in k):
            k2 = k.replace("/actor/", "/teacher_actor/")

            if k2 not in student_vars:
                missing += 1
                continue

            if student_vars[k2].shape != v.shape:
                shape_mismatch += 1
                print("[SHAPE MISMATCH]", k, "->", k2,
                    "teacher", v.shape, "student", student_vars[k2].shape)
                continue

            student_vars[k2] = v
            copied += 1
            mapping.append((k, k2))

    print(f"[Actor copy] Copied={copied}, missing={missing}, shape_mismatch={shape_mismatch}")

    # 4) Load once after all copies (WM + teacher_actor)
    agent.load(student_vars)
    print("[Actor copy] Loaded student_vars with teacher_actor weights.")





    agent.load(student_vars)

    import numpy as np
    import jax

    # Re-save AFTER loading to confirm what is actually inside the agent
    student_vars_after = agent.save()

    # Max absolute diff across all copied tensors
    max_abs = 0.0
    for tk, sk in mapping:
        t = np.asarray(jax.device_get(teacher_vars[tk]))
        s = np.asarray(jax.device_get(student_vars_after[sk]))
        max_abs = max(max_abs, float(np.max(np.abs(t - s))))

    print(f"[VERIFY teacher_actor] max |teacher - student.teacher_actor| = {max_abs:.6e}")

    def subtree_l2(vars_dict, prefix):
        arrs = []
        for k, v in vars_dict.items():
            if k.startswith(prefix):
                vv = np.asarray(jax.device_get(v))
                arrs.append(vv.reshape(-1))
        if not arrs:
            return None
        flat = np.concatenate(arrs, axis=0)
        return float(np.linalg.norm(flat))

    # You must match these prefixes to your actual key strings.
    # Print keys earlier to confirm the exact prefix.
    teacher_prefix = "agent/task_behavior/ac/actor/"
    student_prefix = "agent/task_behavior/ac/teacher_actor/"

    tn = subtree_l2(teacher_vars, teacher_prefix)
    sn = subtree_l2(student_vars_after, student_prefix)

    print(f"[VERIFY teacher_actor] teacher actor L2 norm      = {tn}")
    print(f"[VERIFY teacher_actor] student teacher_actor L2 norm = {sn}")



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
    
    embodied.run.train_policy_distillation(agent, teacher_policy, env, eval_env, replay, eval_replay, teacher_replay, logger, args)


if __name__ == "__main__":
    main()
