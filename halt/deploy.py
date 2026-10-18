"""Algorithm 1 -- the HALT-MPC deployment loop.

     1: for t = 1, 2, ... do
     2:     z_t <- q_phi(o_<=t, a_<t);  phi_t <- FEATURES(z_t, history)   # no GT
     3:     q_hat_{1:H_max} <- f_xi(phi_t)
     4:     H* <- max{ h : lambda_hat(alpha_t) q_hat_h <= tau }     # fixed unless ACI
     5:     if H* < H_min then
     6:         a_t <- pi_safe(z_t);  log abstention
     7:     else
     8:         a_{t:t+H*} <- argmax J(.) by CEM/MPPI, truncated at H* with V_nu
     9:         push (t, y_hat, H*, alpha_t) onto Q
    10:     execute a_t;  observe o_{t+1}
    11:     if mode = ACI then
    12:         for each entry in Q whose horizon has elapsed do
    13:             miss <- 1[ exists h <= H*: d_h > lambda_hat q_hat_h ]   # needs GT
    14:             alpha_{t+1} <- alpha_t + eta(alpha - miss);  pop

The three modes of Remark 4.2 are first-class here, and the ACI block is
separated so that Static and Static+planner-conditional are visibly the
default. ACI is refused outright when the model's capabilities say the task
state is not observable at deployment -- silently degrading to a guarantee
that does not hold would be the worst available behaviour.

--------------------------------------------------------------------------
ABSTENTION STARVES THE ACI FEEDBACK LOOP
--------------------------------------------------------------------------

Line 9 pushes a rollout onto Q, and it sits inside the ELSE branch of the
abstention test on line 5. So when the controller abstains, nothing is
queued, nothing is scored, and no miss indicator ever reaches line 14. The
two mechanisms are coupled, and the coupling has a fixed point that is easy
to fall into:

    a few early misses  ->  alpha_t falls  ->  lambda_hat(alpha_t) rises
    ->  H* collapses below H_min  ->  the controller abstains
    ->  no rollout is queued  ->  no feedback arrives
    ->  alpha_t never moves again  ->  it abstains forever.

We hit this on the reference suite. In a 25-episode run, the ACI mode
produced 12 feedback items in roughly 2500 control steps, alpha_t settled at
-0.22, and the controller abstained on 99.5% of steps. Nothing recovers it,
because recovery requires exactly the evidence that abstention refuses to
generate. Prop. 5.2 is untouched -- its average is over the steps that were
fed, and those were fine -- but the controller is useless.

`shadow_rollouts` (off by default, because Algorithm 1 as published does not
have it) is one remedy: while abstaining, still imagine a rollout at the
horizon that WOULD have been certified, score it when it elapses, and feed
that to ACI. The safe policy is executed either way, so this costs imagined
rollouts and nothing else -- the robot's behaviour is unchanged. It keeps the
feedback stream alive so alpha_t can climb back and the controller can resume.

Other remedies exist -- treat an abstention as an observed non-miss, floor
alpha_t, or run the ACI update on a separate always-on probe horizon -- and
each biases the long-run average differently. The point is that Algorithm 1
needs one of them, and does not currently say so.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .aci import ACIController, StaticController, lambda_at_level
from .calibrate import CalibratedHALT
from .conformal import certified_horizon
from .metrics import task_error
from .planner import PlannerConfig, SafePolicy, make_planner

Array = np.ndarray

MODES = ("static", "static_planner", "aci")

__all__ = ["DeployConfig", "EpisodeResult", "run_episode", "run_deployment"]


@dataclass
class DeployConfig:
    mode: str = "static"
    h_min: int = 3
    eta: float = 0.05
    l_max: int = 25
    planner: str = "mppi"
    n_episodes: int = 40
    max_steps: int = 100
    safe_mode: str = "slow"
    seed: int = 0
    # Off by default, because Algorithm 1 as published does not have it.
    # See `shadow_rollouts_on_abstention` in the module docstring.
    shadow_rollouts: bool = False

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {self.mode!r}")


@dataclass
class EpisodeResult:
    success: bool
    steps: int
    reward: float
    unsafe_contacts: int
    abstentions: int
    h_star: list[int] = field(default_factory=list)
    misses: list[int] = field(default_factory=list)
    lags: list[int] = field(default_factory=list)


def _reward_fn_factory(goal: Array):
    def reward(y: Array, a: Array) -> float:
        return -float(np.linalg.norm(y[:2] - goal)) - 0.01 * float(np.sum(a**2))
    return reward


def run_episode(
    env,
    model,
    extractor,
    halt: CalibratedHALT,
    cfg: DeployConfig,
    controller,
    rng: np.random.Generator,
    planner_cfg: PlannerConfig | None = None,
) -> EpisodeResult:
    pcfg = planner_cfg or PlannerConfig(h_min=cfg.h_min)
    obs = env.reset(rng)
    goal = np.asarray(obs["goal"], dtype=np.float64)
    planner = make_planner(cfg.planner, model, pcfg, _reward_fn_factory(goal))
    safe = SafePolicy(env.action_dim, cfg.safe_mode)

    obs_hist, act_hist = [obs], []
    residuals: list[float] = []
    queue: list[dict] = []          # Q of line 9
    res = EpisodeResult(False, 0, 0.0, 0, 0)
    last_action = np.zeros(env.action_dim)

    for t in range(cfg.max_steps):
        # -- lines 2-3: encode, feature, quantile curve --------------------- #
        z = model.encode(obs_hist, act_hist)
        probe = rng.uniform(-1, 1, size=(halt.h_max, env.action_dim))
        y_roll = model.rollout(z, probe, n_rollouts=model.capabilities.n_rollouts, rng=rng)
        J = (model.jacobian(z, probe, extractor.cfg.jacobian_horizons)
             if model.capabilities.differentiable_readout else None)
        phi = extractor.extract(
            y_rollouts=y_roll, jacobians=J,
            latent_novelty=model.latent_novelty(z) if hasattr(model, "latent_novelty") else 0.0,
            tactile_residual=float(obs.get("tactile", 0.0)),
            contact_onset_prob=float(obs.get("contact", False)),
            error_memory=np.asarray(residuals[-extractor.cfg.memory_m:]) if residuals else None,
        )
        q = halt.q_hat(phi)[0]

        # -- line 4: the certified horizon, at the current effective level --- #
        if cfg.mode == "aci":
            lam = lambda_at_level(halt.calibration.scores, controller.level())
        else:
            lam = halt.calibration.lambda_hat
        h_star = certified_horizon(q, lam, halt.tau)
        res.h_star.append(h_star)

        # -- lines 5-9: abstain or plan -------------------------------------- #
        if h_star < cfg.h_min:
            a = safe(z, last_action)
            res.abstentions += 1
            if cfg.mode == "aci" and cfg.shadow_rollouts:
                # Imagine, at the horizon that WOULD have been certified, so
                # ACI keeps receiving evidence while the robot stays safe.
                # The executed action is still pi_safe; only the imagination
                # is extra. Without this the loop above deadlocks.
                h_shadow = max(1, min(cfg.h_min, halt.h_max))
                y_pred = model.rollout(z, np.tile(a, (h_shadow, 1)), 1, rng)[0]
                queue.append({"t": t, "h_star": h_shadow, "y_pred": y_pred,
                              "bound": lam * q[:h_shadow],
                              "state": env.get_state(),
                              "action": a.copy(), "shadow": True})
        else:
            a = planner.plan(z, h_star, lam * q, rng)
            y_pred = model.rollout(z, np.tile(a, (h_star, 1)), 1, rng)[0]
            queue.append({"t": t, "h_star": h_star, "y_pred": y_pred,
                          "bound": lam * q[:h_star],
                          "state": env.get_state(),
                          "action": a.copy()})

        # -- line 10: execute ------------------------------------------------- #
        obs, r, done, info = env.step(a)
        obs_hist.append(obs)
        act_hist.append(a)
        last_action = a
        res.reward += r
        res.steps += 1
        res.unsafe_contacts += int(info.get("unsafe", False))

        if len(obs_hist) > 1:
            pred = model.one_step_prediction(z, a)
            residuals.append(float(task_error(env.task_state(), pred, env.task_spec)))

        # -- lines 11-14: the optional ACI block ------------------------------ #
        if cfg.mode == "aci":
            still: list[dict] = []
            for entry in queue:
                elapsed = t - entry["t"]
                if elapsed >= entry["h_star"]:
                    # The miss indicator needs y_{t+h}, i.e. ground truth at
                    # deployment. This is exactly the requirement Remark 4.1(c)
                    # names and §10 lists as a limitation.
                    y_true = _replay_truth(env, entry)
                    d = task_error(y_true, entry["y_pred"], env.task_spec)
                    miss = int(np.any(d > entry["bound"]))
                    lag = max(0, elapsed - entry["h_star"])
                    controller.launch(miss, min(lag, controller.l_max))
                    res.misses.append(miss)
                    res.lags.append(lag)
                else:
                    still.append(entry)
            queue = still
            controller.step()

        if done:
            res.success = bool(info.get("success", False))
            break

    return res


def _replay_truth(env, entry: dict) -> Array:
    """Recover y_{t+1..t+H*} for a finished rollout by replaying from its state.

    In simulation this is exact. On hardware the equivalent is a mocap rig or
    a perception stack, with the estimator noise that implies -- the paper's
    ACI experiment uses sigma_est = 0.02 in tolerance-normalised units for
    precisely that reason, rather than exact ground truth, because exact
    ground truth would misrepresent the deployment setting.
    """
    saved = env.get_state()
    env.set_state(entry["state"])
    H = entry["h_star"]
    out = np.zeros((H, env.task_spec.dim))
    for h in range(H):
        env.step(entry["action"])
        out[h] = env.task_state()
    env.set_state(saved)
    return out


def run_deployment(env, model, extractor, halt: CalibratedHALT, cfg: DeployConfig,
                   planner_cfg: PlannerConfig | None = None, verbose: bool = True) -> dict:
    """Run cfg.n_episodes and aggregate. Refuses ACI without deployment-time y."""
    if cfg.mode == "aci" and not model.capabilities.aci_available():
        raise RuntimeError(
            "mode='aci' requires observing y_{t+h} at deployment, and this "
            "model reports task_state_at_deployment=False. Per Remark 4.2 the "
            "available modes are 'static' and 'static_planner'. Refusing to "
            "run ACI without the feedback its guarantee depends on."
        )

    rng = np.random.default_rng(cfg.seed)
    controller = (
        ACIController(halt.alpha, eta=cfg.eta, l_max=cfg.l_max)
        if cfg.mode == "aci"
        else StaticController(halt.alpha)
    )

    results: list[EpisodeResult] = []
    for ep in range(cfg.n_episodes):
        results.append(run_episode(env, model, extractor, halt, cfg, controller, rng, planner_cfg))
        if verbose and (ep + 1) % 10 == 0:
            sr = np.mean([r.success for r in results])
            print(f"  episode {ep + 1}/{cfg.n_episodes}, success so far {sr:.1%}")

    all_h = [h for r in results for h in r.h_star]
    all_miss = [m for r in results for m in r.misses]
    out = {
        "mode": cfg.mode,
        "planner": cfg.planner,
        "n_episodes": cfg.n_episodes,
        "success_rate": float(np.mean([r.success for r in results])),
        "mean_reward": float(np.mean([r.reward for r in results])),
        "unsafe_rate": float(
            np.sum([r.unsafe_contacts for r in results]) / max(np.sum([r.steps for r in results]), 1)
        ),
        "abstention_rate": float(
            np.sum([r.abstentions for r in results]) / max(np.sum([r.steps for r in results]), 1)
        ),
        "mean_h_star": float(np.mean(all_h)) if all_h else 0.0,
        "std_h_star": float(np.std(all_h)) if all_h else 0.0,
        "realised_miss_rate": float(np.mean(all_miss)) if all_miss else float("nan"),
        "n_feedback": len(all_miss),
    }
    if cfg.mode == "aci":
        out.update({f"aci_{k}": v for k, v in controller.summary().items()})
    return out
