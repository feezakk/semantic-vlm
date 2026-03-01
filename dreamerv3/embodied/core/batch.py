import numpy as np

from . import base


class BatchEnv(base.Env):
    def __init__(self, envs, parallel):
        assert all(len(env) == 0 for env in envs)
        assert len(envs) > 0
        self._envs = envs
        self._parallel = parallel
        self._keys = list(self.obs_space.keys())

    @property
    def obs_space(self):
        return self._envs[0].obs_space

    @property
    def act_space(self):
        return self._envs[0].act_space

    def __len__(self):
        return len(self._envs)

    def step(self, action):
        assert all(len(v) == len(self._envs) for v in action.values()), (
            len(self._envs),
            {k: v.shape for k, v in action.items()},
        )
        obs, info = [], []
        for i, env in enumerate(self._envs):
            act = {k: v[i] for k, v in action.items()}
            obs_step, info_step = env.step(act)
            obs.append(obs_step)
            info.append(info_step)
        if self._parallel:
            obs = [ob() for ob in obs]
            info = [inf() for inf in info]
        obs = {k: np.array([ob[k] for ob in obs]) for k in obs[0]}
        # info = {k: np.array([inf[k] for inf in info]) for k in info[0]}
        # return obs, info
    
        # Batch `info` robustly: different envs may emit different info keys.
        keys = sorted({k for inf in info for k in inf.keys()})

        def _default_like(x):
            # Match type/shape so np.array stacking stays sane.
            if isinstance(x, np.ndarray):
                return np.zeros_like(x)
            if isinstance(x, (np.bool_, bool)):
                return False
            if isinstance(x, (np.integer, int)):
                return 0
            if isinstance(x, (np.floating, float)):
                return 0.0
            if isinstance(x, str):
                return ""
            return 0

        info_batched = {}
        for k in keys:
            sample = next((inf[k] for inf in info if k in inf), None)
            default = _default_like(sample) if sample is not None else 0
            info_batched[k] = np.array([inf.get(k, default) for inf in info])

        info = info_batched
        return obs, info

    def render(self):
        return np.stack([env.render() for env in self._envs])

    def close(self):
        for env in self._envs:
            try:
                env.close()
            except Exception:
                pass
