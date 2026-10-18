#!/usr/bin/env python3
"""Step 1 -- Algorithm G.1: fit the head, then calibrate lambda_hat.

Writes:
  checkpoints/halt_<kind>/        head.pt, calibration.json, scores.npy
  results/head_training_<kind>.csv
  results/calibration_<kind>.csv
"""

from __future__ import annotations

import numpy as np

from _common import ALPHA, H_MAX, banner, base_parser, get_run

from halt.calibrate import calibrate_tolerance, run_algorithm_g1
from halt.conformal import bonferroni_calibrate, bonferroni_certified_horizon, certified_horizon_batch
from halt.envs import episode_level_split
from halt.io import write_csv
from halt.nn.head import HeadConfig


def main() -> None:
    p = base_parser(__doc__)
    p.add_argument("--kind", default="ensemble", choices=["ensemble", "video"])
    p.add_argument("--alpha", type=float, default=ALPHA)
    p.add_argument("--tau", type=float, default=None, help="default: Appendix H.2 heuristic")
    p.add_argument("--epochs", type=int, default=80)
    args = p.parse_args()

    run = get_run(args)
    z = np.load(run.csv(f"anchors_{args.kind}.npz"))
    phi, d, ep = z["phi"], z["d"], z["episode_id"]

    import json
    meta = json.loads(run.csv(f"collect_{args.kind}.meta.json").read_text())
    names = meta["feature_names"]

    banner("[01] episode-level splits (never transition-level)", args.quiet)
    m = episode_level_split(ep, (0.6, 0.2, 0.2), seed=args.seed)
    for k, v in m.items():
        print(f"  {k:5s}: {v.sum():5d} anchors from {len(set(ep[v]))} episodes")

    banner("[01] Algorithm G.1", args.quiet)
    tau = args.tau if args.tau is not None else calibrate_tolerance(d[m["train"]])
    cfg = HeadConfig(d_phi=phi.shape[1], h_max=H_MAX, alpha=args.alpha,
                     epochs=args.epochs, width=192, seed=args.seed)
    halt = run_algorithm_g1(
        phi[m["train"]], d[m["train"]], phi[m["cal"]], d[m["cal"]],
        alpha=args.alpha, h_max=H_MAX, feature_names=names, tau=tau,
        head_cfg=cfg, verbose=not args.quiet,
    )
    halt.save(run.ckpt(f"halt_{args.kind}"))
    write_csv(run.csv(f"head_training_{args.kind}.csv"), halt.history)

    # ---- the Bonferroni baseline Eq. (8) replaces --------------------------- #
    q_cal = halt.q_hat(phi[m["cal"]])
    q_te = halt.q_hat(phi[m["test"]])
    lam_b = bonferroni_calibrate(d[m["cal"]], q_cal, args.alpha, H_MAX)
    h_halt = certified_horizon_batch(q_te, halt.calibration.lambda_hat, tau)
    h_bonf = np.array([bonferroni_certified_horizon(r, lam_b, tau) for r in q_te])

    rows = [{
        "method": "HALT max-score (Eq. 8)",
        "lambda_hat": halt.calibration.lambda_hat,
        "mean_h_star": float(h_halt.mean()), "std_h_star": float(h_halt.std()),
        "n_cal": halt.calibration.n_cal, "tau": tau, "alpha": args.alpha,
    }, {
        "method": "Bonferroni per-horizon",
        "lambda_hat": float(np.mean(lam_b)),
        "mean_h_star": float(h_bonf.mean()), "std_h_star": float(h_bonf.std()),
        "n_cal": halt.calibration.n_cal, "tau": tau, "alpha": args.alpha,
    }]
    write_csv(run.csv(f"calibration_{args.kind}.csv"), rows)

    print(f"\n  tau              = {tau:.4f}  (Appendix H.2 heuristic)")
    print(f"  lambda_hat       = {halt.calibration.lambda_hat:.4f}")
    print(f"  mean H* (HALT)   = {h_halt.mean():.2f} +/- {h_halt.std():.2f}")
    print(f"  mean H* (Bonf.)  = {h_bonf.mean():.2f} +/- {h_bonf.std():.2f}")
    print(f"  horizon the union bound costs: "
          f"{h_halt.mean() - h_bonf.mean():+.2f} steps")
    if not np.all(np.isfinite(lam_b)):
        n_inf = int(np.sum(~np.isfinite(lam_b)))
        print(
            f"\n  NOTE: {n_inf}/{H_MAX} Bonferroni horizons have lambda = inf. "
            f"Each is calibrated at alpha/H_max = {args.alpha / H_MAX:.5f}, which "
            f"needs ceil((n+1)(1-alpha/H_max)) <= n calibration points; with "
            f"n_cal = {halt.calibration.n_cal} that is impossible, so the union "
            "bound certifies nothing at all here. That is not a bug in the "
            "baseline -- it is the cost Eq. (8) is designed to avoid, and it "
            "shows up as an infeasible calibration long before it shows up as "
            "a merely shorter horizon."
        )


if __name__ == "__main__":
    main()
