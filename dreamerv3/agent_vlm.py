import jax
import jax.numpy as jnp

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

@jax.custom_vjp
def grad_reverse(x, scale):
    return x

def _gr_fwd(x, scale):
    return x, scale

def _gr_bwd(scale, g):
    return (-scale * g, None)

grad_reverse.defvjp(_gr_fwd, _gr_bwd)



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
        shapes = {k: v for k, v in shapes.items() if not k.startswith("log_")}
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
            self.sem_heads[h] = nets_vlm.MLP(None, layers=2, units=self.vlm_dim,
                                            inputs=["sem"], dims="sem", name=f"sem_head_h{h}", **common)

        self.domain_head = nets_vlm.MLP((self.num_domains,), layers=2, units=256,
                                        inputs=["sem"], dims="sem", dist="onehot", name="domain_head", **common)

        # Style should carry domain info (non-adversarial)
        self.domain_head_sty = nets_vlm.MLP((self.num_domains,), layers=2, units=256,
                                            inputs=["sty"], dims="sty", dist="onehot",
                                            name="domain_head_sty", **common)
 
        self.domain_sty_scale = float(getattr(config, "domain_sty_scale", 1.0))

    def add_semsty(self, state):
        # state: dict with keys incl 'deter'
        sem = self.sem_proj(state)  # (..., sem_dim)
        sty = self.sty_proj(state)  # (..., sty_dim)
        return {**state, "sem": sem, "sty": sty}

    def initial(self, batch_size):
        prev_latent = self.rssm.initial(batch_size)
        prev_action = jnp.zeros((batch_size, *self.act_space.shape))
        return prev_latent, prev_action

    def train(self, data, state):
        # modules = [self.encoder, self.rssm,+  *self.heads.values()]
        modules = [
            self.encoder, self.rssm, *self.heads.values(),
            self.sem_proj, self.sty_proj, self.domain_head, self.domain_head_sty,
            *self.sem_heads.values(),
        ]
        mets, (state, outs, metrics) = self.opt(modules, self.loss, data, state, has_aux=True)
        metrics.update(mets)
        return state, outs, metrics

    def loss(self, data, state):
        embed = self.encoder(data)
        prev_latent, prev_action = state

        # DEFENSIVE: strip sem/sty if they ever sneak in (e.g., from old checkpoint/state)
        prev_latent = {k: v for k, v in prev_latent.items() if k not in ("sem", "sty")}

        prev_actions = jnp.concatenate([prev_action[:, None], data["action"][:, :-1]], 1)

        # --- core RSSM outputs ---
        post_core, prior_core = self.rssm.observe(embed, prev_actions, data["is_first"], prev_latent)

        # --- augmented views (for actor/critic + aux losses) ---
        post = self.add_semsty(post_core)
        prior = self.add_semsty(prior_core)
        # post, prior = self.rssm.observe(embed, prev_actions, data["is_first"], prev_latent)
        # # Attach sem/sty to posterior and prior
        # post = self.add_semsty(post)

        # # domain_id should be one-hot for the head if you use dist="onehot"
        # dom = data["domain_id"].astype(jnp.int32)  # (B,T)
        # dom_oh = jax.nn.one_hot(dom, self.num_domains)  # (B,T,num_domains)

        # sem_gr = grad_reverse(post["sem"], self.domain_scale)  # reverse gradient into sem
        # dom_dist = self.domain_head({"sem": sem_gr})
        # dom_loss = -dom_dist.log_prob(dom_oh).mean()  # scalar
        # losses["domain_adv"] = dom_loss

        dom = data["domain_id"]                    # (B,T,1) in your setup
        if dom.ndim == 3:
            dom = dom[..., 0]                     # -> (B,T)

        # domain_id is float32 (from preprocess), so round+clip before int
        dom = jnp.clip(jnp.round(dom), 0, self.num_domains - 1).astype(jnp.int32)

        dom_oh = jax.nn.one_hot(dom, self.num_domains)   # (B,T,num_domains)

        sem_gr = grad_reverse(post["sem"], self.domain_scale)# --- GRL schedule (0 -> domain_scale) ---
        # Use env_step from the batch as a proxy for training progress.
        # steps_total = float(getattr(self.config.run, "steps", 1e5))  # ensure this exists
        # print("*****************")
        # print(data.keys())
        # cur = jnp.mean(data["env_step"].astype(jnp.float32))
        # p = jnp.clip(cur / steps_total, 0.0, 1.0)

        # # DANN schedule: lambda(p) = 2/(1+exp(-10p)) - 1
        # grl = self.domain_scale * (2.0 / (1.0 + jnp.exp(-10.0 * p)) - 1.0)

        # sem_gr = grad_reverse(post["sem"], grl)
        dom_dist = self.domain_head({"sem": sem_gr})
        dom_loss = -dom_dist.log_prob(dom_oh).mean()

        # Encourage style to be domain-predictive (no gradient reversal)
        dom_dist_sty = self.domain_head_sty({"sty": post["sty"]})
        dom_loss_sty = -dom_dist_sty.log_prob(dom_oh).mean()

        dom_nll = -dom_dist.log_prob(dom_oh)           # (B,T)
        dom_nll_sty = -dom_dist_sty.log_prob(dom_oh)  # (B,T)

        # ---- Domain diagnostics (accuracy + entropy) ----
        # dom: (B,T) int32 labels
        # dom_dist: OneHotDist over num_domains with batch shape (B,T)

        # Some custom OneHot distributions in this repo expose probs=None.
        # mean() is always well-defined and equals probs for one-hot categorical.
        dom_probs = dom_dist.mean()  # (B,T,num_domains)

        

        # Predicted domain id
        dom_pred = jnp.argmax(dom_probs, axis=-1).astype(jnp.int32)  # (B,T)



        # Accuracy (higher = more domain leakage in sem)
        dom_acc = (dom_pred == dom).astype(jnp.float32).mean()

        # Entropy (higher = more confused / invariant)
        # Use dist.entropy() rather than recomputing from probs.
        dom_ent = dom_dist.entropy().mean()

        dom_chance = 1.0 / float(self.num_domains)

        # --- NEW: batch label / prediction balance diagnostics ---
        # dom_flat  = dom.reshape(-1)         # (B*T,)
        # pred_flat = dom_pred.reshape(-1)    # (B*T,)

        # dom_hist  = jnp.bincount(dom_flat,  minlength=self.num_domains).astype(jnp.float32)
        # pred_hist = jnp.bincount(pred_flat, minlength=self.num_domains).astype(jnp.float32)

        # den = jnp.maximum(dom_flat.size, 1)
        # dom_frac  = dom_hist / den
        # pred_frac = pred_hist / den

        # --- NEW: batch label / prediction balance diagnostics ---
        # dom_flat  = dom.reshape(-1)         # (B*T,)
        # pred_flat = dom_pred.reshape(-1)    # (B*T,)

        # dom_hist  = jnp.bincount(dom_flat,  minlength=self.num_domains).astype(jnp.float32)
        # pred_hist = jnp.bincount(pred_flat, minlength=self.num_domains).astype(jnp.float32)

        # den = jnp.maximum(dom_flat.size, 1)
        # dom_frac  = dom_hist / den
        # pred_frac = pred_hist / den

        # --- NEW: batch label / prediction balance diagnostics (JIT-safe) ---
        num_dom = int(self.num_domains)

        dom_oh  = jax.nn.one_hot(dom, num_dom).astype(jnp.float32)       # (B,T,K)
        pred_oh = jax.nn.one_hot(dom_pred, num_dom).astype(jnp.float32)  # (B,T,K)

        # Fractions per class in the current batch
        dom_frac  = dom_oh.mean(axis=(0, 1))    # (K,)
        pred_frac = pred_oh.mean(axis=(0, 1))   # (K,)

        # Majority baseline (best constant predictor for this batch)
        dom_majority = dom_frac.max()



        # dom_majority = dom_frac.max()

        # Class frequencies in the current batch (B,T)
        # dom_freq = jnp.mean(jax.nn.one_hot(dom, self.num_domains).astype(jnp.float32), axis=(0, 1))
        # dom_majority = jnp.max(dom_freq)

        # Helpful “how much above baseline” signals
        dom_acc_minus_chance = dom_acc - dom_chance
        dom_acc_minus_majority = dom_acc - dom_majority



        # Save to metrics
        # (These will appear as train/domain_acc, eval/domain_acc, etc via your logging prefixes.)
        # Put them either in `losses` or directly in `metrics` later—direct metrics is simplest.



        prior = self.add_semsty(prior)
        dists = {}
        feats = {**post, "embed": embed}
        for name, head in self.heads.items():
            out = head(feats if name in self.config.grad_heads else sg(feats))
            out = out if isinstance(out, dict) else {name: out}
            dists.update(out)
        losses = {}
        # losses["domain_adv"] = dom_loss
        # losses["domain_sty"] = dom_loss_sty
        losses["domain_adv"] = dom_nll
        losses["domain_sty"] = dom_nll_sty
        losses["dyn"] = self.rssm.dyn_loss(post, prior, **self.config.dyn_loss)
        losses["rep"] = self.rssm.rep_loss(post, prior, **self.config.rep_loss)
        for key, dist in dists.items():
            loss = -dist.log_prob(data[key].astype(jnp.float32))
            assert loss.shape == embed.shape[:2], (key, loss.shape)
            losses[key] = loss
        # scaled = {k: v * self.scales[k] for k, v in losses.items()}

        def _l2_normalize(x, eps=1e-6):
            return x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + eps)

        def _cos_dist(a, b):
            a = _l2_normalize(a)
            b = _l2_normalize(b)
            return 1.0 - jnp.sum(a * b, axis=-1)

        # Multi-horizon open-loop rollout from each time t using future actions.
        # We roll out up to max horizon H from posterior states.
        H = int(max(self.sem_horizons))
        B, T = data["action"].shape[:2]
        valid_T = T - H
        sem_loss = 0.0

        if valid_T > 0:

            # ---- episode boundary mask prep ----
            is_first = data["is_first"]
            if is_first.ndim == 3:
                is_first = is_first[..., 0]
            is_first = is_first.astype(jnp.int32)          # (B,T)
            csum_first = jnp.cumsum(is_first, axis=1)      # (B,T)

            # Start states: post at times [0..valid_T-1]
            start = {k: v[:, :valid_T] for k, v in post.items() if k in ("deter","stoch","logit","mean","std","sem","sty")}
            start = tree_map(lambda x: x.reshape([-1] + list(x.shape[2:])), start)  # (B*valid_T, ...)

            # Build action sequences of length H for each start time
            # acts[h] corresponds to action at time t+h for each t in [0..valid_T-1]
            acts = jnp.stack([data["action"][:, i:i+valid_T] for i in range(H)], axis=0)  # (H,B,valid_T,Adim)
            acts = acts.reshape([H, -1] + list(data["action"].shape[2:]))               # (H,B*valid_T,Adim)

            # def scan_step(s, a):
            #     s2 = self.rssm.img_step(s, a)
            #     s2 = self.add_semsty(s2)
            #     return s2, s2

            # # lax.scan over horizon; returns states for steps 1..H
            # _, seq = jax.lax.scan(scan_step, start, acts)  # seq: dict with (H, B*valid_T, ...)

            def scan_step(s, a):
                s2 = self.rssm.img_step(s, a)
                s2 = self.add_semsty(s2)
                return s2

            # jaxutils_teacher.scan is used elsewhere in your repo specifically to avoid ninjax RNG leaks
            seq = jaxutils_teacher.scan(
                scan_step,
                acts,                 # shape (H, B*valid_T, act_dim)
                start,                # dict, shape (B*valid_T, ...)
                self.config.imag_unroll if hasattr(self.config, "imag_unroll") else 1,
            )


            # Targets: VLM embeddings at time t+h
            # data["vlm"]: (B,T,D) -> slice to (B, valid_T, D) aligned with future indices
            for h in self.sem_horizons:
                pred_state_h = tree_map(lambda x: x[h-1], seq)  # (B*valid_T,...)
                pred_vlm = self.sem_heads[h]({"sem": pred_state_h["sem"]})  # (B*valid_T, D)
                tgt = data["vlm"][:, h:h+valid_T]                             # (B, valid_T, D)
                tgt = tgt.reshape([-1, tgt.shape[-1]])                        # (B*valid_T, D)
                # sem_loss = sem_loss + _cos_dist(pred_vlm, tgt).mean()+               
                # ---- mask out pairs that cross an episode reset in (t, t+h] ----
                # interval_count[t] = sum(is_first[t+1 ... t+h])
                interval = csum_first[:, h:h+valid_T] - csum_first[:, :valid_T]   # (B,valid_T)
                mask = (interval == 0).astype(jnp.float32).reshape([-1])          # (B*valid_T,)
 
                dist = _cos_dist(pred_vlm, tgt)                                   # (B*valid_T,)
                sem_loss = sem_loss + (dist * mask).sum() / (mask.sum() + 1e-6)

        losses["sem_rollout"] = sem_loss

        # scaled = {k: v * self.scales[k] for k, v in losses.items() if k in self.scales}
        # model_loss = sum(scaled.values()) + self.sem_scale * losses["sem_rollout"]

        scaled = {k: v * self.scales[k] for k, v in losses.items() if k in self.scales}
        model_loss = sum(scaled.values())


        # model_loss = sum(scaled.values())
        out = {"embed": embed, "post": post, "prior": prior}
        out.update({f"{k}_loss": v for k, v in losses.items()})
        last_latent = {k: v[:, -1] for k, v in post.items()}
        last_action = data["action"][:, -1]
        state = last_latent, last_action
        metrics = self._metrics(data, dists, post, prior, losses, model_loss)
        metrics["sem_rollout"] = losses["sem_rollout"]
        metrics["domain_adv_scale"] = self.domain_adv_scale
        metrics["domain_adv_loss"] = losses["domain_adv"].mean()
        metrics["domain_acc"] = dom_acc
        metrics["domain_entropy"] = dom_ent
        metrics["model_loss_raw"] = model_loss  # Store model loss for Curious Replay prioritization
        metrics["domain_chance_acc"] = dom_chance
        metrics["domain_majority_acc"] = dom_majority
        metrics["domain_acc_minus_chance"] = dom_acc_minus_chance
        metrics["domain_acc_minus_majority"] = dom_acc_minus_majority
        metrics["domain_adv_loss"] = dom_nll.mean()
        metrics["domain_sty_loss"] = dom_nll_sty.mean()

        sty_probs = dom_dist_sty.mean()
        sty_pred  = jnp.argmax(sty_probs, axis=-1).astype(jnp.int32)
        metrics["sty_domain_acc"] = (sty_pred == dom).astype(jnp.float32).mean()

        # --- NEW: store balance diagnostics as scalars ---
        metrics["dom_majority_baseline"] = dom_majority
        for i in range(self.num_domains):
            metrics[f"dom_label_frac_{i}"] = dom_frac[i]
            metrics[f"dom_pred_frac_{i}"]  = pred_frac[i]

        # ---- Per-class diagnostics: recall + class fractions ----
        for k in range(self.num_domains):
            mask = (dom == k)
            denom = mask.astype(jnp.float32).sum() + 1e-6
            metrics[f"domain_recall_{k}"] = ((dom_pred == k) & mask).astype(jnp.float32).sum() / denom
            metrics[f"sty_domain_recall_{k}"] = ((sty_pred == k) & mask).astype(jnp.float32).sum() / denom
            metrics[f"domain_frac_{k}"] = mask.astype(jnp.float32).mean()



        return model_loss.mean(), (state, out, metrics)

    def imagine(self, policy, start, horizon):
        first_cont = (1.0 - start["is_terminal"]).astype(jnp.float32)
        keys = list(self.rssm.initial(1).keys())
        # start = {k: v for k, v in start.items() if k in keys}
        # start["action"] = policy(start)

        start = {k: v for k, v in start.items() if k in keys}
        start = self.add_semsty(start)
        start["action"] = policy(start)

        def step(prev, _):
            prev = prev.copy()
            # state = self.rssm.img_step(prev, prev.pop("action"))
            # return {**state, "action": policy(state)}
        
            state = self.rssm.img_step(prev, prev.pop("action"))
            state = self.add_semsty(state)
            return {**state, "action": policy(state)}


        traj = jaxutils_teacher.scan(step, jnp.arange(horizon), start, self.config.imag_unroll)
        traj = {k: jnp.concatenate([start[k][None], v], 0) for k, v in traj.items()}
        cont = self.heads["cont"](traj).mode()
        traj["cont"] = jnp.concatenate([first_cont[None], cont[1:]], 0)
        discount = 1 - 1 / self.config.horizon
        traj["weight"] = jnp.cumprod(discount * traj["cont"], 0) / discount
        return traj

    def imagine_carry(self, policy, start, horizon, carry):
        first_cont = (1.0 - start["is_terminal"]).astype(jnp.float32)
        keys = list(self.rssm.initial(1).keys())
        start = {k: v for k, v in start.items() if k in keys}
        outs, carry = policy(start, carry)
        start["action"] = outs
        start["carry"] = carry

        def step(prev, _):
            prev = prev.copy()
            carry = prev.pop("carry")
            state = self.rssm.img_step(prev, prev.pop("action"))
            outs, carry = policy(state, carry)
            return {**state, "action": outs, "carry": carry}

        traj = jaxutils_teacher.scan(step, jnp.arange(horizon), start, self.config.imag_unroll)
        traj = {k: jnp.concatenate([start[k][None], v], 0) for k, v in traj.items() if k != "carry"}
        cont = self.heads["cont"](traj).mode()
        traj["cont"] = jnp.concatenate([first_cont[None], cont[1:]], 0)
        discount = 1 - 1 / self.config.horizon
        traj["weight"] = jnp.cumprod(discount * traj["cont"], 0) / discount
        return traj

    def report(self, data):
        state = self.initial(len(data["is_first"]))
        report = {}
        report.update(self.loss(data, state)[-1][-1])
        context, _ = self.rssm.observe(self.encoder(data)[:6, :5], data["action"][:6, :5], data["is_first"][:6, :5])
        start = {k: v[:, -1] for k, v in context.items()}
        recon = self.heads["decoder"](context)
        openl = self.heads["decoder"](self.rssm.imagine(data["action"][:6, 5:], start))
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
        return metrics


class ImagActorCritic(nj.Module):
    def __init__(self, critics, scales, act_space, config):
        critics = {k: v for k, v in critics.items() if scales[k]}
        for key, scale in scales.items():
            assert not scale or key in critics, key
        self.critics = {k: v for k, v in critics.items() if scales[k]}
        self.scales = scales
        self.act_space = act_space
        self.config = config
        disc = act_space.discrete
        self.grad = config.actor_grad_disc if disc else config.actor_grad_cont
        # self.actor = nets_vlm.MLP(
        #     name="actor",
        #     dims="deter",
        #     shape=act_space.shape,
        #     **config.actor,
        #     dist=config.actor_dist_disc if disc else config.actor_dist_cont,
        # )

        actor_kw = dict(config.actor)
        actor_kw["inputs"] = ["sem"]
        self.actor = nets_vlm.MLP(
            name="actor",
            dims="sem",
            shape=act_space.shape,
            **actor_kw,
            dist=config.actor_dist_disc if disc else config.actor_dist_cont,
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
        self.rewfn = rewfn
        self.config = config
        # self.net = nets_vlm.MLP((), name="net", dims="deter", **self.config.critic)
        # self.slow = nets_vlm.MLP((), name="slow", dims="deter", **self.config.critic)
        critic_kw = dict(self.config.critic)
        critic_kw["inputs"] = ["sem"]
        self.net = nets_vlm.MLP((), name="net", dims="sem", **critic_kw)
        self.slow = nets_vlm.MLP((), name="slow", dims="sem", **critic_kw)

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
