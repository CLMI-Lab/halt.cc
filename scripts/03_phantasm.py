#!/usr/bin/env python3
"""Step 3 -- the PHANTASM six-axis stress sweep (§9.3, Table 6 analogue).

For each axis at a fixed severity: collect FRESH episodes under the perturbed
dynamics, re-measure coverage with the UNCHANGED calibration, and compare
against a fixed-horizon baseline. The calibration is deliberately not redone;
the question is whether a certificate calibrated on nominal dynamics survives
the shift, which is what a deployment would actually face.
"""

from __future__ import annotations

import numpy as np

from _common import H_MAX, K_ROLLOUTS, banner, base_parser, get_run, make_extractor, make_model

from halt.calibrate import CalibratedHALT
from halt.collect import CollectConfig, anchors_to_arrays, collect_anchors
from halt.conformal import certified_horizon_batch, simultaneous_coverage
from halt.envs import AXES, StressSpec, make_stressed_env
from halt.io import write_csv
from halt.models import EnsembleWorldModel


def fixed_horizon_coverage(d: np.ndarray, tau: float, h_fixed: int) -> float:
    """What a fixed horizon actually delivers: P[ for all h <= H_fixed: d_h <= tau ]."""
    H = min(h_fixed, d.shape[1])
    return float(np.mean(np.all(d[:, :H] <= tau, axis=1)))


def main() -> None:
    p = base_parser(__doc__)
    p.add_argument("--kind", default="ensemble")
    p.add_argument("--severity", type=int, default=2)
    p.add_argument("--episodes", type=int, default=40)
    p.add_argument("--anchors-per-episode", type=int, default=5)
    p.add_argument("--h-fixed", type=int, default=15)
    args = p.parse_args()

    run = get_run(args)
    halt = CalibratedHALT.load(run.ckpt(f"halt_{args.kind}"))
    model = EnsembleWorldModel.load(run.ckpt(f"world_model_{args.kind}.pt"))
    extractor = make_extractor(model.capabilities)

    rows = []
    banner(f"[03] PHANTASM sweep at severity {args.severity}", args.quiet)
    print(f"  calibration is NOT redone: lambda_hat = {halt.calibration.lambda_hat:.3f}, "
          f"tau = {halt.tau:.3f}\n")

    for axis in ("nominal",) + AXES:
        spec = None if axis == "nominal" else StressSpec(axis=axis, severity=args.severity)
        env = make_stressed_env(spec)
        cfg = CollectConfig(
            n_episodes=args.episodes, h_max=H_MAX,
            anchors_per_episode=args.anchors_per_episode, n_rollouts=K_ROLLOUTS,
            seed=args.seed + 100 + AXES.index(axis) if axis != "nominal" else args.seed + 99,
            jacobian_every=10_000,            # grounding probes are not needed here
            max_episode_steps=env.p.max_steps,
        )
        anchors, diag = collect_anchors(env, model, extractor, cfg, stress=spec, verbose=False)
        if not anchors:
            print(f"  {axis:14s} no anchors collected, skipping")
            continue
        arr = anchors_to_arrays(anchors)
        q = halt.q_hat(arr["phi"])
        out = simultaneous_coverage(arr["d"], q, halt.calibration.lambda_hat, halt.tau)
        fixed_cov = fixed_horizon_coverage(arr["d"], halt.tau, args.h_fixed)

        rows.append({
            "axis": axis, "severity": 0 if spec is None else args.severity,
            "n_anchors": out["n"],
            "halt_coverage": out["coverage"],
            "halt_mean_h_star": out["mean_h_star"],
            "halt_abstention_rate": out["abstention_rate"],
            "fixed_coverage": fixed_cov,
            "fixed_h": args.h_fixed,
            "contact_rate": diag["contact_rate"],
            "coverage_gap": out["coverage"] - fixed_cov,
        })
        print(f"  {axis:14s} HALT cov {out['coverage']:.3f}  H* {out['mean_h_star']:5.2f}  |  "
              f"fixed(H={args.h_fixed}) cov {fixed_cov:.3f}  |  "
              f"contact {diag['contact_rate']:.1%}")

    write_csv(run.csv("phantasm_sweep.csv"), rows,
              meta={"severity": args.severity, "h_fixed": args.h_fixed})

    if rows:
        hc = np.array([r["halt_coverage"] for r in rows])
        fc = np.array([r["fixed_coverage"] for r in rows])
        print(f"\n  HALT coverage across axes : {hc.min():.3f} to {hc.max():.3f}")
        print(f"  Fixed coverage across axes: {fc.min():.3f} to {fc.max():.3f}")
        print(f"  worst-case gap from target: "
              f"{np.max(np.abs(hc - (1 - halt.alpha))):.3f}")


if __name__ == "__main__":
    main()
