#!/usr/bin/env python3
"""Step 2 -- coverage, horizons, grounding, silent failure (§9.1, 9.5, 9.6).

Writes:
  results/coverage_<kind>.csv        the Prop. 4.1 check on held-out anchors
  results/horizon_profile_<kind>.csv H* against state (contact vs free space)
  results/silent_failure_<kind>.csv  SFR and the appearance-vs-TTCD correlation
  results/grounding_summary_<kind>.csv  CAS, rho, spectral entropy, verdict
"""

from __future__ import annotations

import csv

import numpy as np

from _common import ALPHA, H_MAX, banner, base_parser, get_run

from halt.calibrate import CalibratedHALT
from halt.conformal import certified_horizon_batch, simultaneous_coverage
from halt.envs import episode_level_split
from halt.io import write_csv
from halt.metrics import (aurc_c, classify_grounding, horizon_calibration_error,
                          rank_correlation, silent_failure_rate, ttcd)


def main() -> None:
    p = base_parser(__doc__)
    p.add_argument("--kind", default="ensemble", choices=["ensemble", "video"])
    args = p.parse_args()

    run = get_run(args)
    z = np.load(run.csv(f"anchors_{args.kind}.npz"))
    phi, d, ep = z["phi"], z["d"], z["episode_id"]
    contact, app = z["contact"], z["appearance"]
    halt = CalibratedHALT.load(run.ckpt(f"halt_{args.kind}"))
    m = episode_level_split(ep, (0.6, 0.2, 0.2), seed=args.seed)
    te = m["test"]

    # ---- Prop. 4.1 on held-out anchors ------------------------------------ #
    banner("[02] coverage (Proposition 4.1)", args.quiet)
    q_te = halt.q_hat(phi[te])
    out = simultaneous_coverage(d[te], q_te, halt.calibration.lambda_hat, halt.tau)
    hce = horizon_calibration_error(out["h_star"], out["covered"].astype(float), halt.alpha)

    print(f"  target 1 - alpha        = {1 - halt.alpha:.3f}")
    print(f"  simultaneous coverage   = {out['coverage']:.3f}  "
          f"(n = {out['n_certified']} certified of {out['n']})")
    print(f"  abstention rate         = {out['abstention_rate']:.3f}")
    print(f"  mean H*                 = {out['mean_h_star']:.2f}")
    print(f"  horizon calibration err = {hce:.4f}")

    write_csv(run.csv(f"coverage_{args.kind}.csv"), [{
        "alpha": halt.alpha, "target": 1 - halt.alpha, "tau": halt.tau,
        "lambda_hat": halt.calibration.lambda_hat,
        "coverage": out["coverage"],
        "coverage_including_abstentions": out["coverage_including_abstentions"],
        "abstention_rate": out["abstention_rate"],
        "mean_h_star": out["mean_h_star"], "mean_h_star_certified": out["mean_h_star_certified"],
        "hce": hce, "n_test": out["n"], "n_certified": out["n_certified"],
        "within_1pct_of_target": bool(abs(out["coverage"] - (1 - halt.alpha)) <= 0.01),
    }])

    # ---- H* is state-dependent, which is the whole claim ------------------- #
    banner("[02] is H* actually state-dependent?", args.quiet)
    h_star = out["h_star"]
    c_te = contact[te]
    rows = []
    for label, sel in (("contact", c_te), ("free_space", ~c_te)):
        if sel.sum() == 0:
            continue
        rows.append({
            "regime": label, "n": int(sel.sum()),
            "mean_h_star": float(h_star[sel].mean()),
            "std_h_star": float(h_star[sel].std()),
            "mean_d1": float(d[te][sel, 0].mean()),
            "mean_d_final": float(d[te][sel, -1].mean()),
        })
        print(f"  {label:11s} n={sel.sum():4d}  mean H* = {h_star[sel].mean():5.2f}  "
              f"mean d_1 = {d[te][sel,0].mean():7.2f}")
    if len(rows) == 2:
        ratio = rows[0]["mean_h_star"] / max(rows[1]["mean_h_star"], 1e-9)
        print(f"  contact / free-space horizon ratio: {ratio:.2f}x  "
              f"(a fixed horizon has to be wrong in one regime or the other)")
    # H* against the spread of realised error, binned
    for lo, hi in [(0, 5), (5, 10), (10, 20), (20, H_MAX + 1)]:
        sel = (h_star >= lo) & (h_star < hi)
        if sel.sum():
            rows.append({"regime": f"h_star_in_[{lo},{hi})", "n": int(sel.sum()),
                         "mean_h_star": float(h_star[sel].mean()),
                         "std_h_star": float(h_star[sel].std()),
                         "mean_d1": float(d[te][sel, 0].mean()),
                         "mean_d_final": float(d[te][sel, -1].mean())})
    write_csv(run.csv(f"horizon_profile_{args.kind}.csv"), rows)

    # ---- silent failure: appearance vs control-relevant validity ----------- #
    banner("[02] silent failure (Definition 3.2, H1)", args.quiet)
    h_task = max(2, H_MAX // 2)
    t = ttcd(d[te], halt.tau)
    corr = rank_correlation(app[te], t)
    sfr_rows = []
    for kappa in (0.50, 0.75, 0.90):
        sfr = silent_failure_rate(d[te], app[te], halt.tau, kappa, h_task)
        sfr_rows.append({"tau": halt.tau, "kappa": kappa, "h_task": h_task, "sfr": sfr})
        print(f"  SFR(tau={halt.tau:.2f}, kappa={kappa:.2f}) = {sfr:.3f}")
    print(f"  rank corr(appearance, TTCD) = {corr:.3f}   "
          f"(H1 falsifier was > 0.70)")
    sfr_rows.append({"tau": halt.tau, "kappa": "", "h_task": h_task,
                     "appearance_ttcd_rank_corr": corr,
                     "h1_falsified": bool(corr > 0.70)})
    write_csv(run.csv(f"silent_failure_{args.kind}.csv"), sfr_rows)

    # ---- AURC-C: is the trust signal any good, threshold-free? ------------- #
    loss = (d[te][:, : max(1, int(np.median(h_star)))] > halt.tau).any(axis=1).astype(float)
    aurc_halt = aurc_c(h_star.astype(float), loss)
    aurc_rand = aurc_c(np.random.default_rng(0).random(h_star.size), loss)
    print(f"  AURC-C: HALT {aurc_halt:.4f} vs random trust {aurc_rand:.4f} (lower is better)")

    # ---- grounding summary from the collection probes ---------------------- #
    gpath = run.csv(f"grounding_{args.kind}.csv")
    if gpath.exists() and gpath.stat().st_size > 0:
        banner("[02] action grounding (Eq. 4-5, Table 7 analogue)", args.quiet)
        with gpath.open() as fh:
            g = list(csv.DictReader(fh))
        grows = []
        for h in sorted({int(r["h"]) for r in g}):
            sub = [r for r in g if int(r["h"]) == h]
            cas_v = np.array([float(r["cas"]) for r in sub])
            rho_v = np.array([float(r["rho"]) for r in sub if np.isfinite(float(r["rho"]))])
            verdict = classify_grounding(float(cas_v.mean()), float(rho_v.mean()))
            grows.append({"h": h, "n": len(sub),
                          "cas_mean": cas_v.mean(), "cas_std": cas_v.std(),
                          "rho_mean": rho_v.mean(), "rho_std": rho_v.std(),
                          "verdict": verdict})
            print(f"  h={h}: CAS = {cas_v.mean():.3f} +/- {cas_v.std():.3f}, "
                  f"rho = {rho_v.mean():+.3f} +/- {rho_v.std():.3f}  -> {verdict}")
        grows.append({"h": "", "n": len(g), "aurc_halt": aurc_halt, "aurc_random": aurc_rand})
        write_csv(run.csv(f"grounding_summary_{args.kind}.csv"), grows)


if __name__ == "__main__":
    main()
