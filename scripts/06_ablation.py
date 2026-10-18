#!/usr/bin/env python3
"""Step 6 -- feature-family ablation (§9.2, Table 5 analogue).

Each cell refits the head on the SAME anchors with all but the kept families
zeroed, recalibrates, and re-measures on the same held-out split. Zeroing
rather than removing keeps the input dimension and the architecture fixed, so
a difference between cells is attributable to information rather than to
capacity.

The quantity worth watching is not coverage alone but the coverage/horizon
pair. A blunter feature set does not merely cover less -- it certifies
FURTHER, because the head cannot see the states where error is about to spike.
Confidently wrong for longer is a worse failure than visibly uncertain.
"""

from __future__ import annotations

import itertools

import numpy as np

from _common import ALPHA, H_MAX, banner, base_parser, get_run, make_extractor, make_model

from halt.calibrate import run_algorithm_g1
from halt.conformal import simultaneous_coverage
from halt.envs import episode_level_split
from halt.features import FAMILIES
from halt.io import write_csv
from halt.metrics import horizon_calibration_error
from halt.nn.head import HeadConfig

# The subsets Table 5 reports, mapped onto this implementation's family names.
SUBSETS = [
    ("full (5 families)", FAMILIES),
    ("disagreement only", ("disagreement",)),
    ("action-Jacobian only", ("jacobian",)),
    ("task residuals only", ("memory",)),
    ("tactile and contact", ("tactile",)),
    ("error memory and novelty", ("memory", "novelty")),
    ("disagreement, Jacobian, tactile", ("disagreement", "jacobian", "tactile")),
]


def main() -> None:
    p = base_parser(__doc__)
    p.add_argument("--kind", default="ensemble")
    p.add_argument("--epochs", type=int, default=60)
    args = p.parse_args()

    run = get_run(args)
    z = np.load(run.csv(f"anchors_{args.kind}.npz"))
    phi, d, ep = z["phi"], z["d"], z["episode_id"]
    model = make_model(args.kind, seed=args.seed)
    extractor = make_extractor(model.capabilities)
    m = episode_level_split(ep, (0.6, 0.2, 0.2), seed=args.seed)

    import json
    tau = json.loads((run.ckpt(f"halt_{args.kind}") / "calibration.json").read_text())["tau"]

    banner("[06] feature-family ablation", args.quiet)
    rows = []
    for label, keep in SUBSETS:
        phi_m = extractor.mask_to(phi, keep=keep)
        cfg = HeadConfig(d_phi=phi.shape[1], h_max=H_MAX, alpha=ALPHA,
                         epochs=args.epochs, width=192, seed=args.seed)
        halt = run_algorithm_g1(
            phi_m[m["train"]], d[m["train"]], phi_m[m["cal"]], d[m["cal"]],
            alpha=ALPHA, h_max=H_MAX, feature_names=extractor.names(),
            tau=tau, head_cfg=cfg, verbose=False,
        )
        out = simultaneous_coverage(
            d[m["test"]], halt.q_hat(phi_m[m["test"]]),
            halt.calibration.lambda_hat, tau,
        )
        hce = horizon_calibration_error(out["h_star"], out["covered"].astype(float), ALPHA)
        rows.append({
            "feature_group": label, "n_families": len(keep),
            "coverage": out["coverage"], "mean_h_star": out["mean_h_star"],
            "std_h_star": float(np.std(out["h_star"])),
            "abstention_rate": out["abstention_rate"],
            "hce": hce, "lambda_hat": halt.calibration.lambda_hat,
            "n_certified": out["n_certified"],
        })
        print(f"  {label:34s} cov {out['coverage']:.3f}  "
              f"H* {out['mean_h_star']:5.2f}  HCE {hce:.4f}")

    write_csv(run.csv("feature_ablation.csv"), rows, meta={"tau": tau, "alpha": ALPHA})

    full = rows[0]
    print(f"\n  Reading the trade-off, not just coverage:")
    for r in rows[1:]:
        dh = r["mean_h_star"] - full["mean_h_star"]
        dc = r["coverage"] - full["coverage"]
        verdict = ("certifies further on worse information"
                   if dh > 0 and dc < 0 else "strictly worse" if dc < 0 else "comparable")
        print(f"    {r['feature_group']:34s} dH* {dh:+5.2f}  dcov {dc:+.3f}  -> {verdict}")


if __name__ == "__main__":
    main()
