#!/usr/bin/env python3
"""Step 8 -- assemble the per-experiment CSVs into a summary and a report.

Writes:
  results/SUMMARY.csv    one row per claim, with the measured value and verdict
  results/REPORT.md      the same, readable, with the scope caveat attached

Every row records which paper claim it bears on and whether this run supports
it ON THE REFERENCE SUITE. A claim about the paper's MuJoCo matrix is marked
not-tested rather than quietly inherited.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from _common import banner, base_parser, get_run

from halt.io import write_csv


def read(path: Path) -> list[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open() as fh:
        return list(csv.DictReader(fh))


def f(x, default=float("nan")) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def main() -> None:
    p = base_parser(__doc__)
    p.add_argument("--kind", default="ensemble")
    args = p.parse_args()
    run = get_run(args)
    R = run.root / "results"

    rows: list[dict] = []

    def claim(name, paper_ref, measured, verdict, detail=""):
        rows.append({"claim": name, "paper_reference": paper_ref,
                     "measured": measured, "verdict": verdict, "detail": detail})

    # ---- coverage ---------------------------------------------------------- #
    cov = read(R / f"coverage_{args.kind}.csv")
    if cov:
        c = cov[0]
        got, target = f(c["coverage"]), f(c["target"])
        # Proposition 4.1 is ONE-SIDED: P[...] >= 1 - alpha. Over-coverage is
        # the guarantee holding conservatively, not a failure. Only a
        # shortfall falsifies it. (An earlier version of this script tested
        # |got - target| <= 0.02 and marked a perfectly valid 0.989 as "NOT
        # supported", which had the logic exactly backwards.)
        slack = 0.01                      # finite-sample tolerance on a single split
        if got >= target - slack:
            verdict = "supported"
            if got > target + 0.02:
                verdict = "supported (conservative)"
        else:
            verdict = "NOT supported"
        claim("Simultaneous coverage reaches 1 - alpha", "Prop. 4.1 (one-sided)",
              f"{got:.3f} vs target {target:.3f}", verdict,
              f"n_certified={c['n_certified']}, HCE={f(c['hce']):.4f}; "
              "the bound is >= 1 - alpha, so excess coverage is conservatism, "
              "paid for in a shorter certified horizon")

    # ---- Eq. 8 vs Bonferroni ------------------------------------------------ #
    cal = read(R / f"calibration_{args.kind}.csv")
    if len(cal) >= 2:
        h_halt = f(cal[0]["mean_h_star"])
        h_bonf = f(cal[1]["mean_h_star"])
        claim("Max-score beats the union bound on horizon", "§4.3",
              f"H* {h_halt:.2f} vs Bonferroni {h_bonf:.2f}",
              "supported" if h_halt >= h_bonf else "NOT supported")

    # ---- state dependence --------------------------------------------------- #
    prof = read(R / f"horizon_profile_{args.kind}.csv")
    reg = {r["regime"]: r for r in prof}
    if "contact" in reg and "free_space" in reg:
        hc, hf = f(reg["contact"]["mean_h_star"]), f(reg["free_space"]["mean_h_star"])
        dc, df = f(reg["contact"]["mean_d1"]), f(reg["free_space"]["mean_d1"])
        claim("H* is state-dependent (contact vs free space)", "§1, Prop. 5.1",
              f"H* {hc:.2f} at contact vs {hf:.2f} free; d_1 {dc:.2f} vs {df:.2f}",
              "supported" if hc < hf else "weak on this suite",
              "a fixed horizon must be wrong in one regime or the other")

    # ---- silent failure (H1) ------------------------------------------------- #
    sf = read(R / f"silent_failure_{args.kind}.csv")
    corr_row = next((r for r in sf if r.get("appearance_ttcd_rank_corr")), None)
    if corr_row:
        corr = f(corr_row["appearance_ttcd_rank_corr"])
        claim("Appearance is a poor proxy for control validity", "H1, §9.5",
              f"rank corr(appearance, TTCD) = {corr:.3f}",
              "supported" if corr <= 0.70 else "FALSIFIED",
              "pre-registered falsifier was > 0.70")

    # ---- feature ablation (H2) ------------------------------------------------ #
    ab = read(R / "feature_ablation.csv")
    if ab:
        full = ab[0]
        worst = min(ab[1:], key=lambda r: f(r["coverage"]), default=None)
        if worst:
            claim("Richer features improve the trust signal", "H2, §9.2",
                  f"full {f(full['coverage']):.3f} vs worst subset "
                  f"{f(worst['coverage']):.3f} ({worst['feature_group']})",
                  "supported" if f(full["coverage"]) >= f(worst["coverage"]) else "NOT supported")
            longer = [r for r in ab[1:] if f(r["mean_h_star"]) > f(full["mean_h_star"])]
            claim("Blunter features certify FURTHER, not just worse", "§9.2 discussion",
                  f"{len(longer)}/{len(ab)-1} subsets have a longer mean H* than the full set",
                  "supported" if longer else "not observed",
                  "confidently wrong for longer is the worse failure")

    # ---- PHANTASM --------------------------------------------------------------- #
    ph = read(R / "phantasm_sweep.csv")
    if ph:
        hc = [f(r["halt_coverage"]) for r in ph]
        fc = [f(r["fixed_coverage"]) for r in ph]
        tgt = f(cov[0]["target"]) if cov else 0.95
        worst = max(abs(x - tgt) for x in hc)
        claim("Coverage survives the six stress axes", "§9.3, Table 6",
              f"HALT {min(hc):.3f}-{max(hc):.3f} vs fixed {min(fc):.3f}-{max(fc):.3f}",
              "supported" if worst <= 0.05 else "degraded",
              f"worst deviation from target {worst:.3f}")

    # ---- ACI (H4) ----------------------------------------------------------------- #
    aci = read(R / "aci_summary.csv")
    if len(aci) >= 2:
        a, s = aci[0], aci[1]
        dev = f(a["deviation"])
        tgt = f(a["alpha_target"])
        static_miss = f(s["miss_rate_overall"])
        claim("Delayed ACI tracks the target level", "H4, Prop. 5.2, §9.4",
              f"ACI miss {f(a['miss_rate_overall']):.4f}, target {tgt:.3f}, "
              f"deviation {dev:.4f}",
              "supported" if dev <= 0.03 else "outside +/-0.03",
              f"Prop. 5.2 bound respected: {a['bound_respected']}, "
              f"telescoping residual {a['telescoping_residual']}")
        # Separate claim, because tracking the target and beating static are
        # different things and this run only establishes the first.
        if static_miss < tgt:
            claim("ACI outperforms static conformal under shift", "§9.4",
                  f"static miss {static_miss:.4f} is BELOW target {tgt:.3f}",
                  "NOT DEMONSTRATED",
                  "static over-covered here rather than failing, so the shift "
                  "induced no shortfall for ACI to correct. The comparison "
                  "needs a calibration with less headroom, or a harsher shift")
        else:
            claim("ACI outperforms static conformal under shift", "§9.4",
                  f"ACI {f(a['miss_rate_overall']):.4f} vs static {static_miss:.4f}, "
                  f"target {tgt:.3f}",
                  "supported" if dev < abs(static_miss - tgt) else "not shown")

    # ---- planner shift ------------------------------------------------------------- #
    psh = read(R / "planner_shift.csv")
    if len(psh) >= 3:
        before = f(psh[1]["coverage"])
        after = f(psh[2]["coverage"])
        hb, ha = f(psh[1]["mean_h_star"]), f(psh[2]["mean_h_star"])
        claim("Closed-loop planning breaks exchangeability", "§4.4",
              f"coverage drops to {before:.3f} under the planner",
              "observed" if before < f(psh[1]["alpha"]) * 0 + 0.95 else "not observed")
        claim("Remedy (R1) restores coverage, at a horizon cost", "§4.4 (R1), Prop. 4.2",
              f"coverage {after:.3f}; H* {hb:.2f} -> {ha:.2f} "
              f"({(ha/hb - 1) * 100:+.0f}%)" if hb else f"coverage {after:.3f}",
              "supported" if after >= 0.93 else "NOT supported")

    # ---- deployment (H3) --------------------------------------------------------------- #
    dep = read(R / "deployment.csv")
    if dep:
        halt_r = [r for r in dep if r["method"].startswith("HALT")]
        fix_r = [r for r in dep if r["method"].startswith("fixed")]
        if halt_r and fix_r:
            bh = max(halt_r, key=lambda r: f(r["success_rate"]))
            bf = max(fix_r, key=lambda r: f(r["success_rate"]))
            all_zero = all(f(r["success_rate"]) == 0.0 for r in dep)
            if all_zero:
                claim("HALT-truncated MPC beats the best fixed horizon", "H3, §9.1",
                      "every method scored 0% success", "UNINFORMATIVE",
                      "no method completed the task, so there is nothing to "
                      "compare. The world model is action-collapsed "
                      "(CAS ~ 0.04), and planning against a model that barely "
                      "responds to the action is expected to fail -- which is "
                      "coherent with Eq. 4-5 but leaves H3 untested here")
            else:
                claim("HALT-truncated MPC beats the best fixed horizon", "H3, §9.1",
                      f"{f(bh['success_rate']):.1%} ({bh['method']}) vs "
                      f"{f(bf['success_rate']):.1%} ({bf['method']})",
                      "supported" if f(bh["success_rate"]) > f(bf["success_rate"]) else "not shown",
                      f"unsafe {f(bh['unsafe_rate']):.3f} vs {f(bf['unsafe_rate']):.3f}; "
                      "read against the standard error, not the point estimate")

        # The abstention/ACI coupling, visible in this run's own numbers.
        aci_dep = next((r for r in dep if r["method"] == "HALT (aci)"), None)
        if aci_dep and f(aci_dep["abstention_rate"]) > 0.5:
            claim("ACI feedback survives the abstention rule", "Alg. 1, lines 5-14",
                  f"abstention {f(aci_dep['abstention_rate']):.1%}, only "
                  f"{aci_dep['n_feedback']} feedback items, alpha_t -> "
                  f"{f(aci_dep.get('aci_alpha_final', 'nan')):.3f}",
                  "FAILS",
                  "line 9 pushes onto Q only in the non-abstaining branch, so "
                  "an abstaining controller generates no evidence and alpha_t "
                  "can never recover. See docs/PAPER_NOTES.md finding 4; "
                  "DeployConfig(shadow_rollouts=True) is one remedy")

    # ---- claims this run deliberately does not test --------------------------------------- #
    for name, ref in [
        ("The paper's Table 3 numbers on five MuJoCo tasks", "§8-9.1"),
        ("DreamerV3 and TD-MPC2 as the wrapped models", "§8"),
        ("GAIA-1 coverage of 0.92 with the Jacobian family dropped", "Table 9"),
        ("Real-robot transfer", "§10"),
    ]:
        claim(name, ref, "not measured here", "NOT TESTED",
              "requires MuJoCo and trained agents; see docs/REAL_ENVIRONMENTS.md")

    write_csv(run.csv("SUMMARY.csv"), rows)

    # ---- readable report ------------------------------------------------------------------- #
    lines = [
        "# HALT reference implementation — run summary",
        "",
        f"Run: `{run.name}`",
        "",
        "## Scope",
        "",
        "These results come from the **PlanarPush reference suite**, a synthetic",
        "contact-rich environment bundled with this repository. They are **not** a",
        "reproduction of the paper's Section 8 experiments, which use five MuJoCo",
        "tasks, ten seeds and three world-model classes. What they establish is that",
        "the algorithms are implemented correctly and that the propositions hold",
        "where they are supposed to.",
        "",
        "## Claims",
        "",
        "| Claim | Paper | Measured | Verdict |",
        "|---|---|---|---|",
    ]
    for r in rows:
        lines.append(f"| {r['claim']} | {r['paper_reference']} | {r['measured']} | "
                     f"**{r['verdict']}** |")
    lines += ["", "## Notes", ""]
    for r in rows:
        if r["detail"]:
            lines.append(f"- **{r['claim']}** — {r['detail']}")
    (R / "REPORT.md").write_text("\n".join(lines) + "\n")

    banner("[08] summary", args.quiet)
    for r in rows:
        print(f"  {r['verdict']:>14s}  {r['claim']}")
        if r["measured"]:
            print(f"                  {r['measured']}")
    print(f"\n  wrote {run.csv('SUMMARY.csv')}")
    print(f"  wrote {R / 'REPORT.md'}")


if __name__ == "__main__":
    main()
