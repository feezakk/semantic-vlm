import jax
import jax.numpy as jnp

from dreamerv3.embodied.core import metrics

tree_map = jax.tree_util.tree_map
sg = lambda x: tree_map(jax.lax.stop_gradient, x)

import logging

logger = logging.getLogger()


class CheckTypesFilter(logging.Filter):
    def filter(self, record):
        return "check_types" not in record.getMessage()


logger.addFilter(CheckTypesFilter())

from . import behaviors_teacher, jaxagent_teacher, jaxutils_teacher, nets_vlm
from . import ninjax as nj

# @jax.custom_vjp
# def grad_reverse(x, scale):
#     return x

def grad_reverse(x, scale):
    # scale can be dynamic (e.g., your DANN schedule); we don't want grads w.r.t. it.
    scale = jax.lax.stop_gradient(jnp.asarray(scale, dtype=x.dtype))
    x_sg = jax.lax.stop_gradient(x)
    # Forward: equals x. Backward: multiplies gradient by -scale.
    return x_sg - scale * (x - x_sg)

def _gr_fwd(x, scale):
    return x, scale

def _gr_bwd(scale, g):
    return (-scale * g, None)

# grad_reverse.defvjp(_gr_fwd, _gr_bwd)



@jaxagent_teacher.Wrapper
class Agent(nj.Module):
    def __init__(self, obs_space, act_space, step, config):
        self.config = config
        self.obs_space = obs_space
        self.act_space = act_space["action"]
        self.step = step
        self.wm = WorldModel(obs_space, act_space, config, name="wm")
        self.task_behavior = getattr(behaviors_teacher, config.task_behavior)(self.wm, self.act_space, self.config, name="task_behavior")
        if config.expl_behavior == "None":
            self.expl_behavior = self.task_behavior
        else:
            self.expl_behavior = getattr(behaviors_teacher, config.expl_behavior)(self.wm, self.act_space, self.config, name="expl_behavior")

    def policy_initial(self, batch_size):
        return (
            self.wm.initial(batch_size),
            self.task_behavior.initial(batch_size),
            self.expl_behavior.initial(batch_size),
        )

    def train_initial(self, batch_size):
        return self.wm.initial(batch_size)

    # def policy(self, obs, state, mode="train"):
    #     self.config.jax.jit and print("Tracing policy function.")
    #     obs = self.preprocess(obs)
    #     (prev_latent, prev_action), task_state, expl_state = state
    #     embed = self.wm.encoder(obs)
    #     latent, _ = self.wm.rssm.obs_step(prev_latent, prev_action, embed, obs["is_first"])
    #     latent = self.wm.add_semsty(latent)
    #     self.expl_behavior.policy(latent, expl_state)
    #     task_outs, task_state = self.task_behavior.policy(latent, task_state)
    #     expl_outs, expl_state = self.expl_behavior.policy(latent, expl_state)
        
    #     if mode == "eval":
    #         outs = task_outs
    #         outs["action"] = outs["action"].sample(seed=nj.rng())
    #         outs["log_entropy"] = jnp.zeros(outs["action"].shape[:1])
    #     elif mode == "explore":
    #         outs = expl_outs
    #         outs["log_entropy"] = outs["action"].entropy()
    #         outs["action"] = outs["action"].sample(seed=nj.rng())
    #     elif mode == "train":
    #         outs = task_outs
    #         outs["log_entropy"] = outs["action"].entropy()
    #         outs["action"] = outs["action"].sample(seed=nj.rng())
    #     state = ((latent, outs["action"]), task_state, expl_state)
    #     return outs, state

    def policy(self, obs, state, mode="train"):
        obs = self.preprocess(obs)
        (prev_latent, prev_action), task_state, expl_state = state

        embed = self.wm.encoder(obs)

        # 1) core RSSM latent
        latent_core, _ = self.wm.rssm.obs_step(prev_latent, prev_action, embed, obs["is_first"])

        # 2) derived features for downstream modules (actor/critic/etc.)
        latent = self.wm.add_semsty(latent_core)

        task_outs, task_state = self.task_behavior.policy(latent, task_state)
        expl_outs, expl_state = self.expl_behavior.policy(latent, expl_state)

        if mode == "eval":
            outs = task_outs
            outs["action"] = outs["action"].sample(seed=nj.rng())
            # outs["action"] = outs["action"].mode()
            outs["log_entropy"] = jnp.zeros(outs["action"].shape[:1])
        elif mode == "explore":
            outs = expl_outs
            outs["log_entropy"] = outs["action"].entropy()
            outs["action"] = outs["action"].sample(seed=nj.rng())
        else:
            outs = task_outs
            outs["log_entropy"] = outs["action"].entropy()
            outs["action"] = outs["action"].sample(seed=nj.rng())

        # IMPORTANT: store latent_core, not latent (which includes sem/sty)
        state = ((latent_core, outs["action"]), task_state, expl_state)
        return outs, state


    def train(self, data, state):
        self.config.jax.jit and print("Tracing train function.")
        metrics = {}
        data = self.preprocess(data)
        state, wm_outs, mets = self.wm.train(data, state)
        metrics.update(mets)
        context = {**data, **wm_outs["post"]}
        start = tree_map(lambda x: x.reshape([-1] + list(x.shape[2:])), context)
        _, mets = self.task_behavior.train(self.wm.imagine, start, context)
        metrics.update(mets)
        if self.config.expl_behavior != "None":
            _, mets = self.expl_behavior.train(self.wm.imagine, start, context)
            metrics.update({"expl_" + key: value for key, value in mets.items()})

        if "key" in data.keys():
            outs = {
                "key": data["key"],
                "env_step": data["env_step"],
                "model_loss": metrics["model_loss_raw"].copy(),
                "td_error": metrics["td_error"].copy(),
            }

        else:
            outs = {}

        # Don't need the full model_loss_raw or td_error after the priority calculation, summarize it.
        metrics.update({"model_loss_raw": metrics["model_loss_raw"].mean()})
        metrics.update({"td_error": metrics["td_error"].mean()})

        return outs, state, metrics

    def report(self, data):
        self.config.jax.jit and print("Tracing report function.")
        data = self.preprocess(data)
        report = {}
        report.update(self.wm.report(data))
        mets = self.task_behavior.report(data)
        report.update({f"task_{k}": v for k, v in mets.items()})
        if self.expl_behavior is not self.task_behavior:
            mets = self.expl_behavior.report(data)
            report.update({f"expl_{k}": v for k, v in mets.items()})
        return report

    def preprocess(self, obs):
        obs = obs.copy()
        for key, value in obs.items():
            if key.startswith("log_") or key in ("key", "env_step"):
                continue
            if len(value.shape) > 3 and value.dtype == jnp.uint8:
                value = jaxutils_teacher.cast_to_compute(value) / 255.0
            else:
                value = value.astype(jnp.float32)
            obs[key] = value
        obs["cont"] = 1.0 - obs["is_terminal"].astype(jnp.float32)
        return obs


class WorldModel(nj.Module):
    def __init__(self, obs_space, act_space, config):
        self.obs_space = obs_space
        self.act_space = act_space["action"]
        self.config = config
        shapes = {k: tuple(v.shape) for k, v in obs_space.items()}
        # shapes = {k: v for k, v in shapes.items() if not k.startswith("log_")}
        shapes = {
            k: v for k, v in shapes.items()
            if (not k.startswith("log_")) and (k not in ("domain_id", "domain_mask", "vlm"))
        }

        self.encoder = nets_vlm.MultiEncoder(shapes, **config.encoder, name="enc")
        self.rssm = nets_vlm.RSSM(**config.rssm, name="rssm")
        self.heads = {
            "decoder": nets_vlm.MultiDecoder(shapes, **config.decoder, name="dec"),
            "reward": nets_vlm.MLP((), **config.reward_head, name="rew"),
            "cont": nets_vlm.MLP((), **config.cont_head, name="cont"),
        }
        self.opt = jaxutils_teacher.Optimizer(name="model_opt", **config.model_opt)
        scales = self.config.loss_scales.copy()
        image, vector = scales.pop("image"), scales.pop("vector")
        scales.update({k: image for k in self.heads["decoder"].cnn_shapes})
        scales.update({k: vector for k in self.heads["decoder"].mlp_shapes})
        self.scales = scales

        def _sc(obj, k, default=0.0):
            return obj.get(k, default) if isinstance(obj, dict) else getattr(obj, k, default)

        # Weight for domain adversarial loss (set it in config.loss_scales.domain_adv)
        self.domain_adv_scale = float(_sc(config.loss_scales, "domain_adv", getattr(config, "domain_loss_weight", 0.0)))


        # --- semantic/style factorization (derived from RSSM state) ---
        self.sem_dim = getattr(config, "sem_dim", 512)
        self.sty_dim = getattr(config, "sty_dim", 512)
        self.vlm_dim = getattr(config, "vlm_dim", 512)
        self.sem_horizons = getattr(config, "sem_horizons", (1, 2, 4, 8))
        self.sem_scale = getattr(config, "sem_scale", 1.0)

        # Sem/style projections from RSSM features; keep it simple: use deter only
        # self.sem_proj = nets_vlm.MLP(
        #     None, layers=2, units=self.sem_dim, inputs=["deter"], dims="deter", name="sem_proj"
        # )
        # self.sty_proj = nets_vlm.MLP(
        #     None, layers=2, units=self.sty_dim, inputs=["deter"], dims="deter", name="sty_proj"
        # )

        # Horizon-specific semantic rollout heads: sem -> vlm embedding
        self.sem_heads = {}
        # for h in self.sem_horizons:
        #     self.sem_heads[h] = nets_vlm.MLP(
        #         None, layers=2, units=self.vlm_dim, inputs=["sem"], dims="sem", name=f"sem_head_h{h}"
        #     )

        self.num_domains = self.config.num_domains
        self.domain_scale = self.config.domain_scale
        # self.domain_head = nets_vlm.MLP(
        #     (self.num_domains,), layers=2, units=256, inputs=["sem"], dims="sem",
        #     dist="onehot", name="domain_head"
        # )

        common = dict(act="silu", norm="layer")

        self.sem_proj = nets_vlm.MLP(None, layers=2, units=self.sem_dim,
                                    inputs=["deter"], dims="deter", name="sem_proj", **common)
        self.sty_proj = nets_vlm.MLP(None, layers=2, units=self.sty_dim,
                                    inputs=["deter"], dims="deter", name="sty_proj", **common)

        for h in self.sem_horizons:
            self.sem_heads[h] = nets_vlm.MLP(
                None, layers=2, units=self.vlm_dim,
                inputs=["sem", "sty_ctx"], dims="sem",
                name=f"sem_head_h{h}", **common
            )

        # Style should carry domain info (non-adversarial)
        self.domain_head_sty = nets_vlm.MLP((self.num_domains,), layers=2, units=256,
                                            inputs=["sty_ctx"], dims="sty_ctx", dist="onehot",
                                            name="domain_head_sty", **common)
 
        self.domain_sty_scale = float(getattr(config, "domain_sty_scale", 1.0))

        self.use_semsty = bool(getattr(config, "use_semsty", True))
        self.domain_head_input = str(getattr(config, "domain_head_input", "sem"))  # "sem" or "deter"

        # GRL schedule controls
        self.domain_grl_max = float(getattr(config, "domain_grl_max", getattr(config, "domain_scale", 0.0)))
        self.domain_grl_schedule = str(getattr(config, "domain_grl_schedule", "dann"))  # "none"|"linear"|"dann"

        self.domain_head = nets_vlm.MLP(
            (self.num_domains,), layers=2, units=256,
            inputs=[self.domain_head_input], dims=self.domain_head_input,
            dist="onehot", name="domain_head", **common
        )

        common = dict(act="silu", norm="layer")

        self.style_ctx_dim = getattr(config, "sty_ctx_dim", self.sty_dim)

        self.style_ctx_net = nets_vlm.MLP(
            None,                      # <-- IMPORTANT: no distribution wrapper
            layers=2,
            units=self.style_ctx_dim,  # <-- output dim = style_ctx_dim (like sem_proj/sty_proj)
            inputs=["tensor"],
            dims="tensor",
            name="style_ctx_net",
            **common
        )

        self.domain_probe = nets_vlm.MLP(
            (self.num_domains,), layers=2, units=256,
            inputs=[self.domain_head_input], dims=self.domain_head_input,
            dist="onehot", name="domain_probe", act="silu", norm="layer"
        )

    
    def add_semsty(self, state):
        if not self.use_semsty:
            return state
        sem = self.sem_proj(state)
        sty = self.sty_proj(state)
        return {**state, "sem": sem, "sty": sty}
    
    def _compute_sty_ctx(self, embed, is_first):
        """
        Build a *causal* style context from encoder embeddings.

        embed:    (B, T, E)
        is_first: (B, T) or (B, T, 1)  (1 at episode starts)

        Returns:
        sty_ctx: (B, T, C)
        """
        B, T, E = embed.shape

        if is_first.ndim == 3:
            is_first = is_first[..., 0]
        is_first = is_first.astype(jnp.bool_)

        # alpha = jnp.array(float(getattr(self.config, "sty_ctx_ema", 0.9)), jnp.float32)

        # # Scan over time with an EMA that resets at episode boundaries.
        # emb_TBE = jnp.swapaxes(embed, 0, 1)       # (T, B, E)
        # first_TB = jnp.swapaxes(is_first, 0, 1)   # (T, B)

        # def step(prev, inp):
        #     e_t, first_t = inp                    # e_t:(B,E), first_t:(B,)
        #     first_t = first_t.astype(jnp.bool_)
        #     prev = jnp.where(first_t[:, None], e_t, alpha * prev + (1.0 - alpha) * e_t)
        #     return prev, prev

        # init = emb_TBE[0]
        # _, ema_TBE = jax.lax.scan(step, init, (emb_TBE, first_TB))

        dtype = embed.dtype  # IMPORTANT: keeps scan carry dtype consistent (float16 in your run)
        alpha = jnp.asarray(getattr(self.config, "sty_ctx_ema", 0.9), dtype=dtype)
        one = jnp.asarray(1.0, dtype=dtype)

        emb_TBE = jnp.swapaxes(embed, 0, 1)       # (T, B, E)
        first_TB = jnp.swapaxes(is_first, 0, 1)   # (T, B)

        def step(prev, inp):
            e_t, first_t = inp
            e_t = e_t.astype(dtype)
            first_t = first_t.astype(jnp.bool_)
            prev = jnp.where(
                first_t[:, None],
                e_t,
                alpha * prev + (one - alpha) * e_t,
            ).astype(dtype)
            return prev, prev

        init = emb_TBE[0].astype(dtype)
        _, ema_TBE = jax.lax.scan(step, init, (emb_TBE.astype(dtype), first_TB))





        ema_BTE = jnp.swapaxes(ema_TBE, 0, 1)     # (B, T, E)

        # Your MLP expects 2D; flatten time, apply once, then reshape back.
        ema_flat = ema_BTE.reshape((-1, E))       # (B*T, E)
        sty_flat = self.style_ctx_net({"tensor": ema_flat})  # (B*T, C)
        C = sty_flat.shape[-1]
        return sty_flat.reshape((B, T, C))

    def initial(self, batch_size):
        prev_latent = self.rssm.initial(batch_size)
        prev_action = jnp.zeros((batch_size, *self.act_space.shape))
        return prev_latent, prev_action

    # def train(self, data, state):
    #     modules = [
    #         self.encoder, self.rssm, *self.heads.values(),
    #         self.sem_proj, self.sty_proj,
    #         self.style_ctx_net,              # <-- ADD THIS
    #         self.domain_head, self.domain_head_sty,
    #         self.domain_probe,
    #         *self.sem_heads.values(),
    #     ]

    #     mets, (state, outs, metrics) = self.opt(modules, self.loss, data, state, has_aux=True)
    #     metrics.update(mets)
    #     return state, outs, metrics

    def train(self, data, state):
        # Always used in loss():
        modules = [
            self.encoder,
            self.rssm,
            *self.heads.values(),
            self.style_ctx_net,
        ]

        # sem/style projections only exist (and are used) if enabled
        if self.use_semsty:
            modules += [self.sem_proj, self.sty_proj]

            # semantic rollout heads only needed if sem_rollout loss is on
            sem_w = self._get_scale("sem_rollout", 0.0)
            if sem_w > 0.0:
                modules += list(self.sem_heads.values())

        # domain heads only needed if corresponding losses/probe are on
        dom_w   = self._get_scale("domain_adv", 0.0)
        sty_w   = self._get_scale("domain_sty", 0.0)
        probe_w = self._get_scale("domain_probe", 0.0)

        if dom_w > 0.0:
            modules += [self.domain_head]
        if sty_w > 0.0:
            modules += [self.domain_head_sty]
        if probe_w > 0.0:
            modules += [self.domain_probe]

        mets, (state, outs, metrics) = self.opt(modules, self.loss, data, state, has_aux=True)
        metrics.update(mets)
        return state, outs, metrics

    
    def _get_scale(self, name, default=0.0):
        obj = self.config.loss_scales
        return float(obj.get(name, default) if isinstance(obj, dict) else getattr(obj, name, default))
    
    def _grl_scale(self, data):
        maxs = float(self.domain_grl_max)
        if maxs <= 0.0:
            return jnp.array(0.0, jnp.float32)

        sched = self.domain_grl_schedule.lower()
        if sched == "none":
            return jnp.array(maxs, jnp.float32)

        # Use env_step if available
        if "env_step" not in data:
            print("*************************************************************")
            print("env_step not in data; cannot compute GRL schedule. Using max scale.")
            print("*************************************************************")
            return jnp.array(maxs, jnp.float32)

        cur = jnp.mean(data["env_step"].astype(jnp.float32))
        steps_total = float(getattr(getattr(self.config, "run", self.config), "steps", 1e6))
        p = jnp.clip(cur / steps_total, 0.0, 1.0)

        if sched == "linear":
            return maxs * p

        # "dann" schedule
        return maxs * (2.0 / (1.0 + jnp.exp(-10.0 * p)) - 1.0)

    def loss(self, data, state):
        # ---- Encode + RSSM observe ----
        embed = self.encoder(data)

        prev_latent, prev_action = state

        # Keep RSSM core only (important for compatibility across configs)
        prev_latent = {k: v for k, v in prev_latent.items() if k not in ("sem", "sty", "sty_ctx")}

        prev_actions = jnp.concatenate([prev_action[:, None], data["action"][:, :-1]], 1)
        post_core, prior_core = self.rssm.observe(embed, prev_actions, data["is_first"], prev_latent)

        B, T = data["action"].shape[:2]
        zeros_bt = jnp.zeros((B, T), jnp.float32)
        nan = jnp.array(jnp.nan, jnp.float32)

        # ---- Scales / feature toggles ----
        sem_w = self._get_scale("sem_rollout", 0.0)
        dom_w = self._get_scale("domain_adv", 0.0)
        sty_w = self._get_scale("domain_sty", 0.0)
        probe_w = self._get_scale("domain_probe", 0.0)

        need_domain = (dom_w > 0.0) or (sty_w > 0.0) or (probe_w > 0.0)
        need_vlm = (sem_w > 0.0)

        # For sem rollout you also need sem/sty enabled and vlm present in data
        use_sem = bool(self.use_semsty and need_vlm)
        use_sty = bool(self.use_semsty and (sty_w > 0.0))

        if need_vlm and ("vlm" not in data):
            raise KeyError(
                "[CONFIG/ENV BUG] sem_rollout > 0 but 'vlm' is missing from the batch. "
                "Either include 'vlm' in the env observation or set loss_scales.sem_rollout=0."
            )
        
        if need_domain and ("domain_id" not in data):
            raise KeyError(
                "[CONFIG/ENV BUG] domain losses/probe enabled but 'domain_id' is missing from the batch. "
                "Either include 'domain_id' in the env observation or set domain_adv/domain_sty/domain_probe scales to 0."
            )
       
        # ---- Add sem/sty features if enabled ----
        post = self.add_semsty(post_core) if self.use_semsty else post_core 
        prior = self.add_semsty(prior_core) if self.use_semsty else prior_core

        # ---- Style context (decoder/reward/cont in your config require sty_ctx) ----
        # sty_ctx0 = self.style_ctx_net({"tensor": embed[:, 5]})                                         # (B, C)
        # sty_ctx  = jnp.repeat(sty_ctx0[:, None, :], T, axis=1)                                         # (B, T, C)
        # post  = {**post,  "sty_ctx": sty_ctx}
        # prior = {**prior, "sty_ctx": sty_ctx}

        # ---- Style context (CAUSAL; no future leakage) ----
        sty_ctx = self._compute_sty_ctx(embed, data["is_first"])   # (B, T, C)
        post  = {**post,  "sty_ctx": sty_ctx}
        prior = {**prior, "sty_ctx": sty_ctx}

        losses = {}

        # ---- Domain adversarial/probe/sty losses ----
        losses["domain_adv"] = zeros_bt
        losses["domain_probe"] = zeros_bt
        losses["domain_sty"] = zeros_bt

        # Placeholders for metrics (so we never reference undefined vars)
        dom = None
        dom_oh = None
        dom_dist = None
        probe_dist = None
        dom_dist_sty = None
        grl = jnp.array(0.0, jnp.float32)

        if need_domain:
            dom = data["domain_id"]                                                                         # (B,T)
            if dom.ndim == 3:
                dom = dom[..., 0]                                                                           # -> (B,T)

            dom = jnp.clip(jnp.round(dom), 0, self.num_domains - 1).astype(jnp.int32)
            dom_oh = jax.nn.one_hot(dom, self.num_domains)   

            domain_mask = data.get("domain_mask", None)
            if domain_mask is None:
                domain_mask = jnp.ones_like(dom, dtype=jnp.float32)
            else:
                if domain_mask.ndim == 3:
                    domain_mask = domain_mask[..., 0]
                domain_mask = domain_mask.astype(jnp.float32)  # (B, T)


            if self.domain_head_input not in post:
                raise KeyError(
                    f"[CONFIG BUG] domain_head_input='{self.domain_head_input}' is not in post-state keys. "
                    f"Available keys: {tuple(post.keys())}. "
                    "If use_semsty=False, you almost certainly want --dreamerv3.domain_head_input deter."
                )   

            feat = post[self.domain_head_input]                                                             # (B,T,dim)

             # Adversarial head
            if dom_w > 0.0:
                grl = self._grl_scale(data)
                feat_gr = grad_reverse(feat, grl)
                dom_dist = self.domain_head({self.domain_head_input: feat_gr})
                dom_nll = -dom_dist.log_prob(dom_oh)  
                # losses["domain_adv"] = dom_nll    
                losses["domain_adv"] = dom_nll * domain_mask                                                          # (B,T)

            # Probe head (non-adversarial, stop-grad into representation)
            if probe_w > 0.0:
                feat_sg = sg(feat)                                                                          # stop gradient to SEM features
                probe_dist = self.domain_probe({self.domain_head_input: feat_sg})
                probe_nll  = -probe_dist.log_prob(dom_oh)                                                   # (B,T)
                # losses["domain_probe"] = probe_nll
                losses["domain_probe"] = probe_nll * domain_mask

            # Style domain head (uses sty_ctx)
            if sty_w > 0.0:
                dom_dist_sty = self.domain_head_sty({"sty_ctx": post["sty_ctx"]})
                # losses["domain_sty"] = -dom_dist_sty.log_prob(dom_oh)
                losses["domain_sty"] = -dom_dist_sty.log_prob(dom_oh) * domain_mask


        # ---- World model reconstruction + dynamics losses ----        
        dists = {}
        feats = {**post, "embed": embed}
        for name, head in self.heads.items():
            out = head(feats if name in self.config.grad_heads else sg(feats))
            out = out if isinstance(out, dict) else {name: out}
            dists.update(out)
        
        
        losses["dyn"] = self.rssm.dyn_loss(post, prior, **self.config.dyn_loss)
        losses["rep"] = self.rssm.rep_loss(post, prior, **self.config.rep_loss)
        for key, dist in dists.items():
            loss = -dist.log_prob(data[key].astype(jnp.float32))
            assert loss.shape == embed.shape[:2], (key, loss.shape)
            losses[key] = loss

        # ---- Semantic rollout loss (optional) ----
        sem_loss = jnp.array(0.0, jnp.float32)
        losses["sem_rollout"] = sem_loss

        if use_sem:
            H = int(max(self.sem_horizons))
            valid_T = T - H
            if valid_T > 0:      
                def _l2_normalize(x, eps=1e-6):
                    return x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + eps)

                def _cos_dist(a, b):
                    a = _l2_normalize(a)
                    b = _l2_normalize(b)
                    return 1.0 - jnp.sum(a * b, axis=-1)

                is_first = data["is_first"]
                if is_first.ndim == 3:
                    is_first = is_first[..., 0]
                is_first = is_first.astype(jnp.int32)
                csum_first = jnp.cumsum(is_first, axis=1)

                start = {k: v[:, :valid_T] for k, v in post.items()
                        if k in ("deter","stoch","logit","mean","std","sem","sty","sty_ctx")}
                start = tree_map(lambda x: x.reshape([-1] + list(x.shape[2:])), start)

                acts = jnp.stack([data["action"][:, i:i+valid_T] for i in range(H)], axis=0)
                acts = acts.reshape([H, -1] + list(data["action"].shape[2:]))

                rssm_keys = set(self.rssm.initial(1).keys())

                def scan_step(s, a):
                    sty_ctx_local = s.get("sty_ctx", None)
                    s_core = {k: s[k] for k in rssm_keys}
                    s2 = self.rssm.img_step(s_core, a)
                    s2 = self.add_semsty(s2) if self.use_semsty else s2
                    if sty_ctx_local is not None:
                        s2 = {**s2, "sty_ctx": sty_ctx_local}
                    return s2

            seq = jaxutils_teacher.scan(
                scan_step, acts, start,
                self.config.imag_unroll if hasattr(self.config, "imag_unroll") else 1,
            )

            sem_loss_accum = jnp.array(0.0, jnp.float32)

            for h in self.sem_horizons:
                pred_state_h = tree_map(lambda x: x[h-1], seq)
                pred_vlm = self.sem_heads[h]({"sem": pred_state_h["sem"], 
                                              "sty_ctx": pred_state_h["sty_ctx"]})
                tgt = data["vlm"][:, h:h+valid_T].reshape([-1, data["vlm"].shape[-1]])
                interval = csum_first[:, h:h+valid_T] - csum_first[:, :valid_T]
                roll_mask = (interval == 0).astype(jnp.float32).reshape([-1])
                dist = _cos_dist(pred_vlm, tgt)
                sem_loss_accum = sem_loss_accum + (dist * roll_mask).sum() / (roll_mask.sum() + 1e-6) 

            # sem_loss = sem_loss_accum
            # sem_loss = sem_loss_accum / float(len(self.sem_horizons))
            num_h = float(len(self.sem_horizons))
            sem_loss = sem_loss_accum / max(num_h, 1.0)

            losses["sem_rollout"] = sem_loss

        # ---- Total loss ----
        scaled = {k: v * self.scales[k] for k, v in losses.items() if k in self.scales}
        model_loss = sum(scaled.values())

        # ---- Outputs + next state (RSSM core only) ----
        out = {"embed": embed, "post": post, "prior": prior}
        out.update({f"{k}_loss": v for k, v in losses.items()})

        rssm_keys = set(self.rssm.initial(1).keys())
        last_latent = {k: v[:, -1] for k, v in post.items() if k in rssm_keys}
        last_action = data["action"][:, -1]
        state = last_latent, last_action

        # ---- Metrics ----
        metrics = self._metrics(data, dists, post, prior, losses, model_loss)

        # Keep your expected scalar names
        metrics["model_loss_raw"] = model_loss  
        metrics["sem_rollout"] = sem_loss

        metrics["domain_adv_scale"] = self.domain_adv_scale
        metrics["domain_grl_scale"] = grl

        # Default domain metrics to NaN (so baselines without domain loss don't crash)
        metrics.setdefault("domain_acc", nan)
        metrics.setdefault("domain_entropy", nan)
        metrics.setdefault("domain_chance_acc", nan)
        metrics.setdefault("domain_majority_acc", nan)
        metrics.setdefault("domain_acc_minus_chance", nan)
        metrics.setdefault("domain_acc_minus_majority", nan)
        metrics.setdefault("domain_present_K", nan)
        metrics.setdefault("domain_chance_acc_present", nan)
        metrics.setdefault("domain_acc_minus_chance_present", nan)
        metrics.setdefault("domain_soft_acc", nan)
        metrics.setdefault("domain_nll_prior", nan)
        metrics.setdefault("domain_nll_model", nan)
        metrics.setdefault("domain_nll_gain", nan)
        metrics.setdefault("domain_probe_acc", nan)
        metrics.setdefault("domain_probe_entropy", nan)
        metrics.setdefault("domain_probe_soft_acc", nan)
        metrics.setdefault("sty_domain_acc", nan)

        # Per-class defaults
        for i in range(int(self.num_domains)):
            metrics.setdefault(f"dom_label_frac_{i}", nan)
            metrics.setdefault(f"dom_pred_frac_{i}", nan)
            metrics.setdefault(f"dom_predprob_frac_{i}", nan)
            metrics.setdefault(f"domain_recall_{i}", nan)
            metrics.setdefault(f"sty_domain_recall_{i}", nan)
            metrics.setdefault(f"domain_frac_{i}", nan)

        # Fill domain metrics only if the corresponding head exists
        # if dom_dist is not None:
        #     dom_probs = dom_dist.mean()                                                                     # (B,T,num_domains)
        #     dom_pred = jnp.argmax(dom_probs, axis=-1).astype(jnp.int32)                                     # (B,T)
            
        #     dom_acc = (dom_pred == dom).astype(jnp.float32).mean()
        #     dom_ent = dom_dist.entropy().mean()
        #     dom_chance = 1.0 / float(self.num_domains)

        #     num_dom = int(self.num_domains)

        #     dom_oh_f = dom_oh.astype(jnp.float32)                                     # (B,T,K)
        #     pred_oh = jax.nn.one_hot(dom_pred, num_dom).astype(jnp.float32)                                 # (B,T,K)

        #     dom_soft_acc = (dom_probs * dom_oh_f).sum(axis=-1).mean()  # in [0,1], chance≈0.25
        #     dom_frac  = dom_oh_f.mean(axis=(0, 1))                                                            # (K,)
        #     pred_frac = pred_oh.mean(axis=(0, 1))                                                           # (K,)

        #     dom_majority = dom_frac.max()
        #     dom_acc_minus_chance = dom_acc - dom_chance
        #     dom_acc_minus_majority = dom_acc - dom_majority

        #     present = (dom_frac > 1e-6).astype(jnp.float32)
        #     K_present = jnp.maximum(present.sum(), 1.0)
        #     dom_chance_present = 1.0 / K_present
        
        #     label_prior = jax.lax.stop_gradient(dom_frac) + 1e-6
        #     nll_prior = -jnp.log(jnp.take(label_prior, dom))   # (B,T)
        #     nll_model = -dom_dist.log_prob(dom_oh) 
        
        #     metrics["domain_acc"] = dom_acc
        #     metrics["domain_entropy"] = dom_ent
        #     metrics["domain_chance_acc"] = jnp.array(dom_chance, jnp.float32)
        #     metrics["domain_majority_acc"] = dom_majority
        #     metrics["domain_acc_minus_chance"] = dom_acc_minus_chance
        #     metrics["domain_acc_minus_majority"] = dom_acc_minus_majority
        #     metrics["domain_adv_loss"] = dom_nll.mean()
        #     metrics["domain_present_K"] = K_present
        #     metrics["domain_chance_acc_present"] = dom_chance_present
        #     metrics["domain_acc_minus_chance_present"] = dom_acc - dom_chance_present
        #     metrics["domain_soft_acc"] = dom_soft_acc
        #     metrics["domain_nll_prior"] = nll_prior.mean()
        #     metrics["domain_nll_model"] = nll_model.mean()
        #     metrics["domain_nll_gain"]  = (nll_prior - nll_model).mean()
        #     metrics["domain_probe_acc"] = (jnp.argmax(probe_dist.mean(), -1) == dom).mean()
        #     metrics["domain_probe_entropy"] = probe_dist.entropy().mean()
        #     metrics["domain_probe_soft_acc"] = (probe_dist.mean() * dom_oh).sum(-1).mean()

        #     predprob_frac = dom_probs.mean(axis=(0, 1))

        #     for i in range(int(self.num_domains)):
        #         metrics[f"dom_label_frac_{i}"] = dom_frac[i]
        #         metrics[f"dom_pred_frac_{i}"] = pred_frac[i]
        #         metrics[f"dom_predprob_frac_{i}"] = predprob_frac[i]

        #         mask = (dom == i)
        #         denom = mask.astype(jnp.float32).sum() + 1e-6
        #         metrics[f"domain_recall_{i}"] = ((dom_pred == i) & mask).astype(jnp.float32).sum() / denom
        #         metrics[f"domain_frac_{i}"] = mask.astype(jnp.float32).mean()

        if dom_dist is not None:
            mask_f = domain_mask.astype(jnp.float32)                       # (B,T)
            mask_sum = mask_f.sum()
            valid = mask_sum > 0.5                                  # True only if any supervised domains exist

            def mmean(x_bt):
                return (x_bt * mask_f).sum() / (mask_sum + 1e-6)

            dom_probs = dom_dist.mean()                             # (B,T,K)
            dom_pred  = jnp.argmax(dom_probs, axis=-1).astype(jnp.int32)

            dom_oh_f  = dom_oh.astype(jnp.float32)                  # (B,T,K)
            pred_oh   = jax.nn.one_hot(dom_pred, int(self.num_domains)).astype(jnp.float32)

            dom_acc   = mmean((dom_pred == dom).astype(jnp.float32))
            dom_ent   = mmean(dom_dist.entropy())
            dom_soft  = mmean((dom_probs * dom_oh_f).sum(axis=-1))   # expected prob of true class

            dom_frac  = (dom_oh_f * mask_f[..., None]).sum(axis=(0, 1)) / (mask_sum + 1e-6)
            pred_frac = (pred_oh  * mask_f[..., None]).sum(axis=(0, 1)) / (mask_sum + 1e-6)
            predprob_frac = (dom_probs * mask_f[..., None]).sum(axis=(0, 1)) / (mask_sum + 1e-6)

            dom_chance = 1.0 / float(self.num_domains)
            dom_majority = dom_frac.max()

            present = (dom_frac > 1e-6).astype(jnp.float32)
            K_present = jnp.maximum(present.sum(), 1.0)
            dom_chance_present = 1.0 / K_present

            label_prior = jax.lax.stop_gradient(dom_frac) + 1e-6
            nll_prior = -jnp.log(jnp.take(label_prior, dom))         # (B,T)
            nll_model = -dom_dist.log_prob(dom_oh)                   # (B,T)

            # Masked means
            nll_prior_m = mmean(nll_prior)
            nll_model_m = mmean(nll_model)

            metrics["domain_acc"] = jnp.where(valid, dom_acc, nan)
            metrics["domain_entropy"] = jnp.where(valid, dom_ent, nan)
            metrics["domain_chance_acc"] = jnp.where(valid, jnp.array(dom_chance, jnp.float32), nan)
            metrics["domain_majority_acc"] = jnp.where(valid, dom_majority, nan)
            metrics["domain_acc_minus_chance"] = jnp.where(valid, dom_acc - dom_chance, nan)
            metrics["domain_acc_minus_majority"] = jnp.where(valid, dom_acc - dom_majority, nan)
            metrics["domain_present_K"] = jnp.where(valid, K_present, nan)
            metrics["domain_chance_acc_present"] = jnp.where(valid, dom_chance_present, nan)
            metrics["domain_acc_minus_chance_present"] = jnp.where(valid, dom_acc - dom_chance_present, nan)
            metrics["domain_soft_acc"] = jnp.where(valid, dom_soft, nan)

            metrics["domain_nll_prior"] = jnp.where(valid, nll_prior_m, nan)
            metrics["domain_nll_model"] = jnp.where(valid, nll_model_m, nan)
            metrics["domain_nll_gain"]  = jnp.where(valid, nll_prior_m - nll_model_m, nan)

            # If you want a masked loss number for logging
            metrics["domain_adv_loss"] = jnp.where(valid, mmean(dom_nll), nan)

            for i in range(int(self.num_domains)):
                metrics[f"dom_label_frac_{i}"] = jnp.where(valid, dom_frac[i], nan)
                metrics[f"dom_pred_frac_{i}"] = jnp.where(valid, pred_frac[i], nan)
                metrics[f"dom_predprob_frac_{i}"] = jnp.where(valid, predprob_frac[i], nan)

                label_mask = ((dom == i).astype(jnp.float32) * mask_f)
                denom = label_mask.sum()
                recall = ((dom_pred == i).astype(jnp.float32) * label_mask).sum() / (denom + 1e-6)

                metrics[f"domain_recall_{i}"] = jnp.where(denom > 0.5, recall, nan)
                metrics[f"domain_frac_{i}"] = jnp.where(valid, denom / (mask_sum + 1e-6), nan)


        if probe_dist is not None:
            probe_probs = probe_dist.mean()
            probe_pred = jnp.argmax(probe_probs, axis=-1).astype(jnp.int32)
            metrics["domain_probe_acc"] = (probe_pred == dom).astype(jnp.float32).mean()
            metrics["domain_probe_entropy"] = probe_dist.entropy().mean()
            metrics["domain_probe_soft_acc"] = (probe_probs * dom_oh.astype(jnp.float32)).sum(-1).mean()

        # if dom_dist_sty is not None:
        #     sty_probs = dom_dist_sty.mean()
        #     sty_pred = jnp.argmax(sty_probs, axis=-1).astype(jnp.int32)
        #     metrics["sty_domain_acc"] = (sty_pred == dom).astype(jnp.float32).mean()

        #     for i in range(int(self.num_domains)):
        #         mask = (dom == i)
        #         denom = mask.astype(jnp.float32).sum() + 1e-6
        #         metrics[f"sty_domain_recall_{i}"] = ((sty_pred == i) & mask).astype(jnp.float32).sum() / denom

        if dom_dist_sty is not None:
            # Rebuild masked averaging utils here (do NOT assume dom_dist block ran)
            mask_f = domain_mask.astype(jnp.float32)            # (B,T) domain_mask (1 for supervised domains, 0 otherwise)
            mask_sum = mask_f.sum()
            valid = mask_sum > 0.5                       # only compute metrics if any supervised samples exist

            def mmean(x_bt):
                # masked mean over (B,T)
                return (x_bt * mask_f).sum() / (mask_sum + 1e-6)

            sty_probs = dom_dist_sty.mean()              # (B,T,K)
            sty_pred = jnp.argmax(sty_probs, axis=-1).astype(jnp.int32)

            # Masked accuracy
            sty_acc = mmean((sty_pred == dom).astype(jnp.float32))
            metrics["sty_domain_acc"] = jnp.where(valid, sty_acc, nan)

            # Masked per-class recall
            for i in range(int(self.num_domains)):
                # only count timesteps where (domain==i) AND supervised (mask_f==1)
                label_mask = (dom == i).astype(jnp.float32) * mask_f     # (B,T)
                denom = label_mask.sum()

                recall = ((sty_pred == i).astype(jnp.float32) * label_mask).sum() / (denom + 1e-6)
                metrics[f"sty_domain_recall_{i}"] = jnp.where(denom > 0.5, recall, nan)



        return model_loss.mean(), (state, out, metrics)

    def imagine(self, policy, start, horizon):
        first_cont = (1.0 - start["is_terminal"]).astype(jnp.float32)

        rssm_keys = set(self.rssm.initial(1).keys())

        # ✅ Preserve optional style context (it exists when sty_w > 0 and you attached it in loss()).
        sty_ctx0 = start.get("sty_ctx", None)  # shape (N, C) where N = B*T after flattening

        # ✅ Only the RSSM core state goes into img_step/observe.
        start_core = {k: v for k, v in start.items() if k in rssm_keys}

        # Add sem/sty features for the actor/critic inputs.
        start_feat = self.add_semsty(start_core)

        # ✅ Re-attach style context so heads that require it (e.g., cont) can read it.
        if sty_ctx0 is not None:
            start_feat = {**start_feat, "sty_ctx": sty_ctx0}

        # Sample action from policy using features (sem, etc.)
        start_feat = {**start_feat, "action": policy(start_feat)}

        def step(prev, _):
            # prev contains RSSM core + sem/sty (+ maybe sty_ctx) + action
            sty_ctx = prev.get("sty_ctx", None)
            action = prev["action"]

            # ✅ Feed only RSSM core into img_step (no sem/sty/sty_ctx/action).
            prev_core = {k: prev[k] for k in rssm_keys}
            state_core = self.rssm.img_step(prev_core, action)

            # Add sem/sty again for downstream modules
            state = self.add_semsty(state_core)

            # ✅ Carry style context forward unchanged
            if sty_ctx is not None:
                state = {**state, "sty_ctx": sty_ctx}

            return {**state, "action": policy(state)}

        traj = jaxutils_teacher.scan(step, jnp.arange(horizon), start_feat, self.config.imag_unroll)
        traj = {k: jnp.concatenate([start_feat[k][None], v], 0) for k, v in traj.items()}

        cont = self.heads["cont"](traj).mode()
        traj["cont"] = jnp.concatenate([first_cont[None], cont[1:]], 0)

        discount = 1 - 1 / self.config.horizon
        traj["weight"] = jnp.cumprod(discount * traj["cont"], 0) / discount
        return traj

    def imagine_carry(self, policy, start, horizon, carry):
        first_cont = (1.0 - start["is_terminal"]).astype(jnp.float32)

        rssm_keys = set(self.rssm.initial(1).keys())
        sty_ctx0 = start.get("sty_ctx", None)

        start_core = {k: v for k, v in start.items() if k in rssm_keys}
        start_feat = self.add_semsty(start_core)

        if sty_ctx0 is not None:
            start_feat = {**start_feat, "sty_ctx": sty_ctx0}

        outs, carry = policy(start_feat, carry)
        start_feat = {**start_feat, "action": outs, "carry": carry}

        def step(prev, _):
            carry = prev["carry"]
            sty_ctx = prev.get("sty_ctx", None)
            action = prev["action"]

            prev_core = {k: prev[k] for k in rssm_keys}
            state_core = self.rssm.img_step(prev_core, action)
            state = self.add_semsty(state_core)

            if sty_ctx is not None:
                state = {**state, "sty_ctx": sty_ctx}

            outs, carry = policy(state, carry)
            return {**state, "action": outs, "carry": carry}

        traj = jaxutils_teacher.scan(step, jnp.arange(horizon), start_feat, self.config.imag_unroll)
        traj = {k: jnp.concatenate([start_feat[k][None], v], 0) for k, v in traj.items() if k != "carry"}

        cont = self.heads["cont"](traj).mode()
        traj["cont"] = jnp.concatenate([first_cont[None], cont[1:]], 0)

        discount = 1 - 1 / self.config.horizon
        traj["weight"] = jnp.cumprod(discount * traj["cont"], 0) / discount
        return traj

    def report(self, data):
        state = self.initial(len(data["is_first"]))
        report = {}
        report.update(self.loss(data, state)[-1][-1])

        # --- Encode once (avoid duplicate encoder calls) ---
        embed = self.encoder(data)

        # --- Build a per-sequence style context and repeat over time ---
        # Safe to always compute; extra keys won't hurt heads that don't use it.
        # sty_ctx0 = self.style_ctx_net({"tensor": embed[:, 0]})               # (B, C)
        # K = 5  # first 5 timesteps
        # sty_in = jnp.mean(embed[:, :K], axis=1)        # (B,C)
        # sty_ctx0 = self.style_ctx_net({"tensor": sty_in})
        # sty_ctx  = jnp.repeat(sty_ctx0[:, None, :], embed.shape[1], axis=1)  # (B, T, C)

        sty_ctx = self._compute_sty_ctx(embed, data["is_first"])  # (B, T, C)


        # --- Reconstruct first few steps ---
        context, _ = self.rssm.observe(
            embed[:6, :5],
            data["action"][:6, :5],
            data["is_first"][:6, :5],
        )
        context = {**context, "sty_ctx": sty_ctx[:6, :5]}  # <-- REQUIRED if decoder/cont expects sty_ctx

        # Start state for open-loop imagination: RSSM core keys only
        rssm_keys = set(self.rssm.initial(1).keys())
        start = {k: v[:, -1] for k, v in context.items() if k in rssm_keys}

        # --- Open-loop prediction for remaining steps ---
        openl_state = self.rssm.imagine(data["action"][:6, 5:], start)
        openl_state = {**openl_state, "sty_ctx": sty_ctx[:6, 5:]}  # <-- REQUIRED

        recon = self.heads["decoder"](context)
        openl = self.heads["decoder"](openl_state)

        for key in self.heads["decoder"].cnn_shapes.keys():
            truth = data[key][:6].astype(jnp.float32)
            model = jnp.concatenate([recon[key].mode()[:, :5], openl[key].mode()], 1)
            error = (model - truth + 1) / 2
            video = jnp.concatenate([truth, model, error], 2)
            report[f"openl_{key}"] = jaxutils_teacher.video_grid(video)

        return report


    def _metrics(self, data, dists, post, prior, losses, model_loss):
        entropy = lambda feat: self.rssm.get_dist(feat).entropy()
        metrics = {}
        metrics.update(jaxutils_teacher.tensorstats(entropy(prior), "prior_ent"))
        metrics.update(jaxutils_teacher.tensorstats(entropy(post), "post_ent"))
        metrics.update({f"{k}_loss_mean": v.mean() for k, v in losses.items()})
        metrics.update({f"{k}_loss_std": v.std() for k, v in losses.items()})
        metrics["model_loss_mean"] = model_loss.mean()
        metrics["model_loss_std"] = model_loss.std()
        metrics["reward_max_data"] = jnp.abs(data["reward"]).max()
        metrics["reward_max_pred"] = jnp.abs(dists["reward"].mean()).max()
        if "reward" in dists and not self.config.jax.debug_nans:
            stats = jaxutils_teacher.balance_stats(dists["reward"], data["reward"], 0.1)
            metrics.update({f"reward_{k}": v for k, v in stats.items()})
        if "cont" in dists and not self.config.jax.debug_nans:
            stats = jaxutils_teacher.balance_stats(dists["cont"], data["cont"], 0.5)
            metrics.update({f"cont_{k}": v for k, v in stats.items()})
        if "vlm" in data:
            vlm = data["vlm"].astype(jnp.float32)
            vlm_norm = jnp.linalg.norm(vlm, axis=-1)  # (B,T)
            metrics["vlm_norm_mean"] = vlm_norm.mean()
            metrics["vlm_norm_std"] = vlm_norm.std()
        return metrics


class ImagActorCritic(nj.Module):
    def __init__(self, critics, scales, act_space, config):

        def _parse_keys(x, default):
            if x is None:
                return list(default)
            if isinstance(x, (list, tuple)):
                return list(x)
            if isinstance(x, str):
                return [k.strip() for k in x.split(",") if k.strip()]
            return list(default)

        critics = {k: v for k, v in critics.items() if scales[k]}
        for key, scale in scales.items():
            assert not scale or key in critics, key
        self.critics = {k: v for k, v in critics.items() if scales[k]}
        self.scales = scales
        self.act_space = act_space
        self.config = config
        disc = act_space.discrete
        self.grad = config.actor_grad_disc if disc else config.actor_grad_cont

        actor_inputs = _parse_keys(getattr(config, "actor_inputs", None), default=["sem"])
        actor_dims = str(getattr(config, "actor_dims", actor_inputs[0]))

        actor_kw = dict(config.actor)
        actor_kw["inputs"] = actor_inputs

        self.actor = nets_vlm.MLP(
            name="actor",
            dims=actor_dims,
            shape=act_space.shape,
            **actor_kw,
            dist=config.actor_dist_disc if act_space.discrete else config.actor_dist_cont,
        )

        self.retnorms = {k: jaxutils_teacher.Moments(**config.retnorm, name=f"retnorm_{k}") for k in critics}
        self.opt = jaxutils_teacher.Optimizer(name="actor_opt", **config.actor_opt)

    def initial(self, batch_size):
        return {}

    def policy(self, state, carry):
        return {"action": self.actor(state)}, carry

    def train(self, imagine, start, context):
        def loss(start):
            policy = lambda s: self.actor(sg(s)).sample(seed=nj.rng())
            traj = imagine(policy, start, self.config.imag_horizon)
            loss, metrics = self.loss(traj)
            return loss, (traj, metrics)

        mets, (traj, metrics) = self.opt(self.actor, loss, start, has_aux=True)
        metrics.update(mets)
        for key, critic in self.critics.items():
            mets = critic.train(traj, self.actor)
            metrics.update({f"{key}_critic_{k}": v for k, v in mets.items()})
        return traj, metrics

    def loss(self, traj):
        metrics = {}
        advs = []
        total = sum(self.scales[k] for k in self.critics)
        for key, critic in self.critics.items():
            rew, ret, base = critic.score(traj, self.actor)
            offset, invscale = self.retnorms[key](ret)
            normed_ret = (ret - offset) / invscale
            normed_base = (base - offset) / invscale
            advs.append((normed_ret - normed_base) * self.scales[key] / total)
            metrics.update(jaxutils_teacher.tensorstats(rew, f"{key}_reward"))
            metrics.update(jaxutils_teacher.tensorstats(ret, f"{key}_return_raw"))
            metrics.update(jaxutils_teacher.tensorstats(normed_ret, f"{key}_return_normed"))
            metrics[f"{key}_return_rate"] = (jnp.abs(ret) >= 0.5).mean()

        # if len(self.critics) != 1:
        #  raise NotImplementedError('Must have exactly one critic for TD error calculation.')

        r = jnp.reshape(rew[0], (self.config.batch_size, self.config.batch_length))
        v = jnp.reshape(base[0], (self.config.batch_size, self.config.batch_length))
        disc = jnp.reshape(traj["cont"][0], (self.config.batch_size, self.config.batch_length)) * (1 - 1 / self.config.horizon)
        td_error = r[:, :-1] + disc[:, 1:] * v[:, 1:] - v[:, :-1]
        metrics["td_error"] = td_error  # Store TD error for PER prioritization

        adv = jnp.stack(advs).sum(0)
        policy = self.actor(sg(traj))
        logpi = policy.log_prob(sg(traj["action"]))[:-1]
        loss = {"backprop": -adv, "reinforce": -logpi * sg(adv)}[self.grad]
        ent = policy.entropy()[:-1]
        loss -= self.config.actent * ent
        loss *= sg(traj["weight"])[:-1]
        loss *= self.config.loss_scales.actor
        metrics.update(self._metrics(traj, policy, logpi, ent, adv))
        return loss.mean(), metrics

    def _metrics(self, traj, policy, logpi, ent, adv):
        metrics = {}
        ent = policy.entropy()[:-1]
        rand = (ent - policy.minent) / (policy.maxent - policy.minent)
        rand = rand.mean(range(2, len(rand.shape)))
        act = traj["action"]
        act = jnp.argmax(act, -1) if self.act_space.discrete else act
        metrics.update(jaxutils_teacher.tensorstats(act, "action"))
        metrics.update(jaxutils_teacher.tensorstats(rand, "policy_randomness"))
        metrics.update(jaxutils_teacher.tensorstats(ent, "policy_entropy"))
        metrics.update(jaxutils_teacher.tensorstats(logpi, "policy_logprob"))
        metrics.update(jaxutils_teacher.tensorstats(adv, "adv"))
        metrics["imag_weight_dist"] = jaxutils_teacher.subsample(traj["weight"])
        return metrics


class VFunction(nj.Module):
    def __init__(self, rewfn, config):

        def _parse_keys(x, default):
            if x is None:
                return list(default)
            if isinstance(x, (list, tuple)):
                return list(x)
            if isinstance(x, str):
                return [k.strip() for k in x.split(",") if k.strip()]
            return list(default)

        self.rewfn = rewfn
        self.config = config
        # self.net = nets_vlm.MLP((), name="net", dims="deter", **self.config.critic)
        # self.slow = nets_vlm.MLP((), name="slow", dims="deter", **self.config.critic)
        # critic_kw = dict(self.config.critic)
        # critic_kw["inputs"] = ["sem"]
        # self.net = nets_vlm.MLP((), name="net", dims="sem", **critic_kw)
        # self.slow = nets_vlm.MLP((), name="slow", dims="sem", **critic_kw)
        
        critic_inputs = _parse_keys(getattr(config, "critic_inputs", None), default=["sem"])
        critic_dims = str(getattr(config, "critic_dims", critic_inputs[0]))

        critic_kw = dict(self.config.critic)
        critic_kw["inputs"] = critic_inputs

        self.net = nets_vlm.MLP((), name="net", dims=critic_dims, **critic_kw)
        self.slow = nets_vlm.MLP((), name="slow", dims=critic_dims, **critic_kw)

        self.updater = jaxutils_teacher.SlowUpdater(
            self.net,
            self.slow,
            self.config.slow_critic_fraction,
            self.config.slow_critic_update,
        )
        self.opt = jaxutils_teacher.Optimizer(name="critic_opt", **self.config.critic_opt)

    def train(self, traj, actor):
        target = sg(self.score(traj)[1])
        mets, metrics = self.opt(self.net, self.loss, traj, target, has_aux=True)
        metrics.update(mets)
        self.updater()
        return metrics

    def loss(self, traj, target):
        metrics = {}
        traj = {k: v[:-1] for k, v in traj.items()}
        dist = self.net(traj)
        loss = -dist.log_prob(sg(target))
        if self.config.critic_slowreg == "logprob":
            reg = -dist.log_prob(sg(self.slow(traj).mean()))
        elif self.config.critic_slowreg == "xent":
            reg = -jnp.einsum("...i,...i->...", sg(self.slow(traj).probs), jnp.log(dist.probs))
        else:
            raise NotImplementedError(self.config.critic_slowreg)
        loss += self.config.loss_scales.slowreg * reg
        loss = (loss * sg(traj["weight"])).mean()
        loss *= self.config.loss_scales.critic
        metrics = jaxutils_teacher.tensorstats(dist.mean())
        return loss, metrics

    def score(self, traj, actor=None):
        rew = self.rewfn(traj)
        assert len(rew) == len(traj["action"]) - 1, "should provide rewards for all but last action"
        discount = 1 - 1 / self.config.horizon
        disc = traj["cont"][1:] * discount
        value = self.net(traj).mean()
        vals = [value[-1]]
        interm = rew + disc * value[1:] * (1 - self.config.return_lambda)
        for t in reversed(range(len(disc))):
            vals.append(interm[t] + disc[t] * self.config.return_lambda * vals[-1])
        ret = jnp.stack(list(reversed(vals))[:-1])
        return rew, ret, value[:-1]
