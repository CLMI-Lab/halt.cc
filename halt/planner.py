"""CEM / MPPI planning with H*-truncated rollouts and abstention (§4.5).

Objective, Eq. (11):

    J(a_{t:t+H*}) = sum_{h<H*} gamma^h r_hat(z_{t+h}, a_{t+h})
                    + gamma^{H*} V_nu(z_{t+H*})
                    - beta sum_{h<H*} lambda_hat q_hat_h

Three things in that expression do real work:

  * the sum stops at H*. Beyond the certified region the planner does not
    trust imagination at all;
  * V_nu bootstraps past the truncation, so stopping early costs value
    estimation error rather than the whole tail of the return;
  * the last term penalises planning into states whose certified quantile is
    already large, which pushes the optimiser toward regions the model
    actually understands.

Abstention rule: H* < H_min  =>  execute pi_safe, one of
{slow, retract, re-observe, request help}.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

Array = np.ndarray

__all__ = ["PlannerConfig", "CEMPlanner", "MPPIPlanner", "SafePolicy", "make_planner"]


@dataclass
class PlannerConfig:
    horizon_max: int = 20
    n_samples: int = 50
    n_elite: int = 10
    n_iters: int = 3
    gamma: float = 0.95
    beta: float = 0.20            # weight on the certified-quantile penalty
    temperature: float = 1.0      # MPPI
    init_std: float = 0.5
    min_std: float = 0.05
    action_low: float = -1.0
    action_high: float = 1.0
    h_min: int = 3                # abstention threshold


class SafePolicy:
    """pi_safe. Which of the four options fires depends on why H* collapsed."""

    MODES = ("slow", "retract", "re_observe", "request_help")

    def __init__(self, action_dim: int, mode: str = "slow"):
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}")
        self.action_dim = action_dim
        self.mode = mode
        self.n_calls = 0

    def __call__(self, z: Array, last_action: Array | None = None) -> Array:
        self.n_calls += 1
        if self.mode == "slow":
            # Keep moving, but at a fraction of the commanded speed, so the
            # model has fewer steps of dynamics to be wrong about.
            return 0.25 * (np.zeros(self.action_dim) if last_action is None else last_action)
        if self.mode == "retract":
            return -0.4 * (np.zeros(self.action_dim) if last_action is None else last_action)
        if self.mode == "re_observe":
            return np.zeros(self.action_dim)
        # request_help: hold still and raise. The caller decides what to do with it.
        return np.zeros(self.action_dim)


class _BasePlanner:
    def __init__(
        self,
        model,
        cfg: PlannerConfig,
        reward_fn: Callable[[Array, Array], float],
        value_fn: Callable[[Array], float] | None = None,
    ):
        self.model = model
        self.cfg = cfg
        self.reward_fn = reward_fn
        self.value_fn = value_fn or (lambda z: 0.0)
        self.stats = {"abstentions": 0, "plans": 0, "mean_h_star": 0.0}

    # -- Eq. (11) ------------------------------------------------------------- #
    def _score(self, z: Array, actions: Array, h_star: int, q_pen: Array,
               rng: np.random.Generator) -> float:
        if h_star <= 0:
            return -np.inf
        acts = actions[:h_star]
        ys = self.model.rollout(z, acts, n_rollouts=1, rng=rng)[0]   # (h_star, d_y)
        g = self.cfg.gamma
        ret = 0.0
        for h in range(h_star):
            ret += (g**h) * self.reward_fn(ys[h], acts[h])
        ret += (g**h_star) * self.value_fn(ys[h_star - 1])
        ret -= self.cfg.beta * float(np.sum(q_pen[:h_star]))
        return float(ret)

    def plan(self, z, h_star, q_penalty, rng):   # pragma: no cover - interface
        raise NotImplementedError


class CEMPlanner(_BasePlanner):
    """Cross-entropy method (Rubinstein, 1999), truncated at H*."""

    def plan(self, z: Array, h_star: int, q_penalty: Array, rng: np.random.Generator) -> Array:
        cfg = self.cfg
        self.stats["plans"] += 1
        self.stats["mean_h_star"] += h_star
        H = int(np.clip(h_star, 0, cfg.horizon_max))
        d_a = self.model.action_dim
        if H <= 0:
            return np.zeros(d_a)

        mu = np.zeros((H, d_a))
        std = np.full((H, d_a), cfg.init_std)
        for _ in range(cfg.n_iters):
            samples = np.clip(
                mu[None] + std[None] * rng.normal(size=(cfg.n_samples, H, d_a)),
                cfg.action_low, cfg.action_high,
            )
            scores = np.array([self._score(z, s, H, q_penalty, rng) for s in samples])
            elite = samples[np.argsort(-scores)[: cfg.n_elite]]
            mu = elite.mean(axis=0)
            std = np.maximum(elite.std(axis=0), cfg.min_std)
        return mu[0]


class MPPIPlanner(_BasePlanner):
    """Model-predictive path integral (Williams et al., 2017), truncated at H*.

    The reason §4.4 matters is visible right here: the softmax weighting is an
    exponential-tilt toward high-scoring sequences, and the score comes from
    the *model*. Sequences the model is optimistically wrong about score well
    and get weight. The executed trajectory is therefore drawn from a
    different distribution than a behaviour-policy calibration rollout, which
    is precisely the exchangeability failure Prop. 4.1 assumes away and
    remedy (R1) repairs.
    """

    def plan(self, z: Array, h_star: int, q_penalty: Array, rng: np.random.Generator) -> Array:
        cfg = self.cfg
        self.stats["plans"] += 1
        self.stats["mean_h_star"] += h_star
        H = int(np.clip(h_star, 0, cfg.horizon_max))
        d_a = self.model.action_dim
        if H <= 0:
            return np.zeros(d_a)

        mu = np.zeros((H, d_a))
        noise = rng.normal(size=(cfg.n_samples, H, d_a)) * cfg.init_std
        samples = np.clip(mu[None] + noise, cfg.action_low, cfg.action_high)
        scores = np.array([self._score(z, s, H, q_penalty, rng) for s in samples])

        finite = np.isfinite(scores)
        if not finite.any():
            return np.zeros(d_a)
        s = np.where(finite, scores, -np.inf)
        w = np.exp((s - s.max()) / max(cfg.temperature, 1e-6))
        w = np.where(finite, w, 0.0)
        w_sum = w.sum()
        if w_sum <= 0:
            return samples[int(np.argmax(s))][0]
        return np.clip((w[:, None, None] * samples).sum(axis=0) / w_sum,
                       cfg.action_low, cfg.action_high)[0]


def make_planner(kind: str, model, cfg: PlannerConfig, reward_fn, value_fn=None):
    kind = kind.lower()
    if kind == "cem":
        return CEMPlanner(model, cfg, reward_fn, value_fn)
    if kind == "mppi":
        return MPPIPlanner(model, cfg, reward_fn, value_fn)
    raise ValueError(f"planner must be 'cem' or 'mppi', got {kind!r}")
