#!/usr/bin/env python3
"""Step 7 -- closed-loop deployment: HALT-truncated MPC vs fixed horizons.

The Table 3 analogue. HALT-MPC runs Algorithm 1; each baseline replaces line 4
with a constant horizon and never abstains. Success rate, unsafe-contact rate
and abstention rate are measured over the same episode seeds for every method,
so the comparison is paired.
"""

from __future__ import annotations

import numpy as np

from _common import banner, base_parser, get_run, make_env, make_extractor

from halt.calibrate import CalibratedHALT
from halt.deploy import DeployConfig, run_deployment
from halt.io import write_csv
from halt.models import EnsembleWorldModel
from halt.planner import PlannerConfig


class FixedHorizonHALT:
    """A CalibratedHALT whose certified horizon is a constant.

    Implemented by wrapping rather than by branching inside the deployment
    loop, so the baseline runs through exactly the same planner, the same
    feature extraction and the same episode seeds. The only difference is
    line 4 of Algorithm 1.
    """

    def __init__(self, halt: CalibratedHALT, h_fixed: int):
        self._halt = halt
        self.h_fixed = int(h_fixed)
        self.tau = halt.tau
        self.alpha = halt.alpha
        self.h_max = halt.h_max
        self.calibration = halt.calibration

    def q_hat(self, phi):
        """A step curve: comfortably under tau for h < h_fixed, far over it after.

        Definition 4.1 applied to this returns exactly h_fixed for every
        state, which is precisely what a fixed-horizon controller does. The
        curve is non-decreasing, so it satisfies the same contract the
        monotone head does.
        """
        n = np.atleast_2d(phi).shape[0]
        lam = max(self.calibration.lambda_hat, 1e-12)
        low = 0.5 * self.tau / lam          # lam * low = tau/2 <= tau
        high = 1e9
        h_idx = np.arange(self.h_max)[None, :]
        return np.where(h_idx < self.h_fixed, low, high).repeat(n, axis=0)


def main() -> None:
    p = base_parser(__doc__)
    p.add_argument("--kind", default="ensemble")
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--fixed", type=int, nargs="*", default=[1, 5, 10, 15])
    p.add_argument("--modes", nargs="*", default=["static", "aci"])
    args = p.parse_args()

    run = get_run(args)
    halt = CalibratedHALT.load(run.ckpt(f"halt_{args.kind}"))
    model = EnsembleWorldModel.load(run.ckpt(f"world_model_{args.kind}.pt"))
    extractor = make_extractor(model.capabilities)
    pcfg = PlannerConfig(horizon_max=halt.h_max, n_samples=24, n_iters=2, h_min=3)

    rows = []
    banner("[07] closed-loop deployment", args.quiet)

    for h in args.fixed:
        env = make_env()
        cfg = DeployConfig(mode="static", n_episodes=args.episodes, seed=args.seed,
                           max_steps=env.p.max_steps, planner="mppi", h_min=0)
        out = run_deployment(env, model, extractor, FixedHorizonHALT(halt, h), cfg,
                             pcfg, verbose=False)
        out["method"] = f"fixed H={h}"
        rows.append(out)
        print(f"  {out['method']:18s} success {out['success_rate']:.1%}  "
              f"unsafe {out['unsafe_rate']:.3f}  H* {out['mean_h_star']:.1f}")

    for mode in args.modes:
        if mode == "aci" and not model.capabilities.aci_available():
            print(f"  skipping mode={mode}: the model reports no deployment-time task state")
            continue
        env = make_env()
        cfg = DeployConfig(mode=mode, n_episodes=args.episodes, seed=args.seed,
                           max_steps=env.p.max_steps, planner="mppi", h_min=3,
                           l_max=halt.h_max)
        out = run_deployment(env, model, extractor, halt, cfg, pcfg, verbose=False)
        out["method"] = f"HALT ({mode})"
        rows.append(out)
        print(f"  {out['method']:18s} success {out['success_rate']:.1%}  "
              f"unsafe {out['unsafe_rate']:.3f}  H* {out['mean_h_star']:.1f}  "
              f"abstain {out['abstention_rate']:.1%}")

    write_csv(run.csv("deployment.csv"), rows,
              meta={"episodes": args.episodes, "planner": "mppi"})

    halt_rows = [r for r in rows if r["method"].startswith("HALT")]
    fixed_rows = [r for r in rows if r["method"].startswith("fixed")]
    if halt_rows and fixed_rows:
        best_fixed = max(fixed_rows, key=lambda r: r["success_rate"])
        best_halt = max(halt_rows, key=lambda r: r["success_rate"])
        print(f"\n  best fixed baseline: {best_fixed['method']} at "
              f"{best_fixed['success_rate']:.1%} success, "
              f"{best_fixed['unsafe_rate']:.3f} unsafe")
        print(f"  best HALT mode     : {best_halt['method']} at "
              f"{best_halt['success_rate']:.1%} success, "
              f"{best_halt['unsafe_rate']:.3f} unsafe")
        print("\n  H3 was pre-registered as falsified if a tuned fixed horizon "
              "matches HALT within noise. With "
              f"{args.episodes} episodes the standard error on a success rate "
              f"near 50% is about {0.5 / np.sqrt(args.episodes):.1%}; read the "
              "comparison against that, not against the point estimates.")


if __name__ == "__main__":
    main()
