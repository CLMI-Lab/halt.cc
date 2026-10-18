#!/usr/bin/env python3
"""Step 5 -- planner-induced distribution shift and its two remedies (§4.4).

The experiment the section asks for, run honestly:

  1. calibrate on behaviour-policy rollouts, as Prop. 4.1 assumes;
  2. collect anchors under the PLANNER instead, which preferentially selects
     actions the model is wrong about;
  3. measure the coverage shortfall that opens up;
  4. apply remedy (R1) -- recalibrate on planner-induced scores -- and measure
     both the restored coverage and the horizon it costs;
  5. apply remedy (R2) -- the conservative deflation kappa_plan -- for
     comparison, as the paper reports it is dominated by (R1).

Writes results/planner_shift.csv.
"""

from __future__ import annotations

import numpy as np

from _common import H_MAX, K_ROLLOUTS, banner, base_parser, get_run, make_env, make_extractor

from halt.calibrate import CalibratedHALT
from halt.collect import CollectConfig, anchors_to_arrays, collect_anchors
from halt.conformal import (choose_kappa_plan, conformal_quantile, conformal_scores,
                            certified_horizon_batch, simultaneous_coverage)
from halt.io import write_csv
from halt.models import EnsembleWorldModel
from halt.planner import PlannerConfig, make_planner


class PlannerPolicyEnv:
    """Wraps the environment so `behaviour_action` returns the PLANNER's action.

    Everything downstream -- collection, feature extraction, ground-truth
    replay -- is unchanged, so the only difference between the two anchor sets
    is the distribution the actions were drawn from. That is the controlled
    comparison §4.4 needs.
    """

    def __init__(self, env, model, halt, h_plan: int, seed: int = 0):
        self._env = env
        self._model = model
        self._halt = halt
        self._rng = np.random.default_rng(seed)
        cfg = PlannerConfig(horizon_max=h_plan, n_samples=24, n_iters=2)
        self._planner = make_planner("mppi", model, cfg, self._reward)
        self._h_plan = h_plan
        self._last_obs = None

    # -- delegate the Environment protocol ---------------------------------- #
    task_spec = property(lambda self: self._env.task_spec)
    action_dim = property(lambda self: self._env.action_dim)
    p = property(lambda self: self._env.p)

    def _reward(self, y, a):
        g = self._env._state[6:8]
        return -float(np.linalg.norm(y[:2] - g)) - 0.01 * float(np.sum(a**2))

    def reset(self, rng=None):
        self._last_obs = self._env.reset(rng)
        return self._last_obs

    def step(self, a):
        out = self._env.step(a)
        self._last_obs = out[0]
        return out

    def task_state(self):
        return self._env.task_state()

    def get_state(self):
        return self._env.get_state()

    def set_state(self, s):
        return self._env.set_state(s)

    def supports_reset_to_state(self):
        return self._env.supports_reset_to_state()

    def observe(self):
        return self._env.observe()

    def behaviour_action(self):
        z = self._model.encode([self._last_obs or self._env.observe()], [])
        q = self._halt.q_hat(z if False else self._probe(z))[0]
        return self._planner.plan(z, self._h_plan, self._halt.calibration.lambda_hat * q, self._rng)

    def _probe(self, z):
        acts = self._rng.uniform(-1, 1, size=(H_MAX, self._env.action_dim))
        y = self._model.rollout(z, acts, n_rollouts=self._model.capabilities.n_rollouts,
                                rng=self._rng)
        from _common import make_extractor as _mk
        ex = _mk(self._model.capabilities)
        J = (self._model.jacobian(z, acts, ex.cfg.jacobian_horizons)
             if self._model.capabilities.differentiable_readout else None)
        return ex.extract(y_rollouts=y, jacobians=J,
                          latent_novelty=self._model.latent_novelty(z))[None, :]


def main() -> None:
    p = base_parser(__doc__)
    p.add_argument("--kind", default="ensemble")
    p.add_argument("--episodes", type=int, default=40)
    args = p.parse_args()

    run = get_run(args)
    halt = CalibratedHALT.load(run.ckpt(f"halt_{args.kind}"))
    model = EnsembleWorldModel.load(run.ckpt(f"world_model_{args.kind}.pt"))
    extractor = make_extractor(model.capabilities)

    banner("[05] collecting PLANNER-induced anchors", args.quiet)
    h_plan = max(3, int(round(halt.certified_horizon(
        np.zeros((1, extractor.dim)))[0])) or 5)
    penv = PlannerPolicyEnv(make_env(), model, halt, h_plan, seed=args.seed + 300)
    cfg = CollectConfig(n_episodes=args.episodes, h_max=H_MAX, anchors_per_episode=5,
                        n_rollouts=K_ROLLOUTS, seed=args.seed + 301,
                        jacobian_every=10_000, max_episode_steps=penv.p.max_steps)
    anchors, _ = collect_anchors(penv, model, extractor, cfg, verbose=False)
    pl = anchors_to_arrays(anchors)
    print(f"  {pl['phi'].shape[0]} planner anchors (planning horizon {h_plan})")

    q_pl = halt.q_hat(pl["phi"])
    s_pl = conformal_scores(pl["d"], q_pl)
    s_beh = halt.calibration.scores

    # ---- 3: the shortfall under the behaviour-policy certificate ---------- #
    half = pl["phi"].shape[0] // 2
    before = simultaneous_coverage(pl["d"], q_pl, halt.calibration.lambda_hat, halt.tau)

    # ---- 4: remedy (R1) ---------------------------------------------------- #
    lam_r1 = conformal_quantile(s_pl[:half], halt.alpha)
    after = simultaneous_coverage(pl["d"][half:], q_pl[half:], lam_r1, halt.tau)

    # ---- 5: remedy (R2) ---------------------------------------------------- #
    kappa = choose_kappa_plan(s_beh, s_pl)
    h_r2 = np.floor(kappa * certified_horizon_batch(
        q_pl[half:], halt.calibration.lambda_hat, halt.tau)).astype(int)
    cov_r2 = float(np.mean([
        bool(np.all(pl["d"][half + i, : h_r2[i]] <= halt.tau)) if h_r2[i] > 0 else True
        for i in range(h_r2.size)
    ]))

    rows = [
        {"setting": "behaviour-policy calibration, behaviour-policy test",
         "lambda_hat": halt.calibration.lambda_hat,
         "coverage": "", "mean_h_star": "", "note": "the Prop. 4.1 regime (see coverage_*.csv)"},
        {"setting": "behaviour-policy calibration, PLANNER test",
         "lambda_hat": halt.calibration.lambda_hat,
         "coverage": before["coverage"], "mean_h_star": before["mean_h_star"],
         "note": "exchangeability broken; this is the shortfall of §4.4"},
        {"setting": "(R1) planner-conditional calibration, PLANNER test",
         "lambda_hat": lam_r1, "coverage": after["coverage"],
         "mean_h_star": after["mean_h_star"],
         "note": "coverage restored; horizon is the price"},
        {"setting": "(R2) conservative horizon kappa_plan, PLANNER test",
         "lambda_hat": halt.calibration.lambda_hat, "coverage": cov_r2,
         "mean_h_star": float(h_r2.mean()),
         "note": f"kappa_plan = {kappa:.3f}"},
    ]
    for r in rows:
        r["alpha"] = halt.alpha
        r["tau"] = halt.tau
    write_csv(run.csv("planner_shift.csv"), rows)

    banner("[05] result", args.quiet)
    print(f"  target 1 - alpha                     = {1 - halt.alpha:.3f}")
    print(f"  coverage, behaviour cal / planner test = {before['coverage']:.3f}  "
          f"(H* {before['mean_h_star']:.2f})")
    print(f"  coverage, (R1) planner-conditional     = {after['coverage']:.3f}  "
          f"(H* {after['mean_h_star']:.2f})")
    print(f"  lambda_hat ratio R1 / behaviour        = "
          f"{lam_r1 / max(halt.calibration.lambda_hat, 1e-12):.3f}x")
    if before["mean_h_star"] > 0:
        shrink = 1 - after["mean_h_star"] / before["mean_h_star"]
        print(f"  horizon shortening from (R1)           = {shrink:+.1%}")
    print(f"  (R2) kappa_plan = {kappa:.3f}, coverage {cov_r2:.3f}, "
          f"H* {h_r2.mean():.2f}")


if __name__ == "__main__":
    main()
