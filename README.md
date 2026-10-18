# HALT — reference implementation

Certified, state-dependent trust horizons for action-conditioned world models.

> Mohammad, Sheikh and Habib. *HALT: Certifying Trust Horizons in
> Action-Conditioned Robotic World Models.* NeurIPS 2026 Workshop on Robot
> Learning with World Models.

```bash
pip install -r requirements.txt
pytest tests/ -q                 # 60 tests: the propositions, numerically
bash scripts/run_all.sh          # the full pipeline, end to end, ~10 min on 2 CPUs
```

---

## Read this before the results

The repository has two layers, and conflating them would be the one mistake
that matters next to a paper about honest uncertainty.

**Layer 1 — the algorithms.** Every method in the paper, implemented faithfully
and tested: the monotone quantile head (Eq. 6), the pinball objective (Eq. 7),
simultaneous split-conformal calibration (Eq. 8), the certified horizon
(Definition 4.1), the planning objective (Eq. 11), delayed-feedback ACI
(Eq. 12), all six metrics of Table B.1, Algorithms 1 and G.1, and the PHANTASM
axes. This layer is environment-independent and is what you would use on a real
robot.

**Layer 2 — a runnable suite.** `PlanarPush`, a synthetic contact-rich
environment with exact reset-to-state, bundled so the whole pipeline executes
on a laptop with no MuJoCo. It reproduces the *phenomenon* the paper is about —
a model that is accurate in free space and dynamically meaningless within a few
steps of contact — and it is what the shipped CSVs were produced from.

**The CSVs in `runs/` are not a reproduction of the paper's Table 3.** That
would need DreamerV3 and TD-MPC2 trained on five MuJoCo tasks across ten seeds.
Every CSV carries a `provenance` column and a `.meta.json` sidecar saying so,
and `results/SUMMARY.csv` marks the paper's MuJoCo claims **NOT TESTED** rather
than quietly inheriting them. Swapping in the real thing is one adapter file:
[`docs/REAL_ENVIRONMENTS.md`](docs/REAL_ENVIRONMENTS.md).

---

## Paper → code

| Paper | Code |
|---|---|
| Eq. (2) task error `d_h` | `halt.metrics.task_error` |
| Def. 3.1 TTCD | `halt.metrics.ttcd` |
| Def. 3.2 silent failure | `halt.metrics.silent_failure_rate` |
| Eq. (3) action Jacobians | `halt.jacobian` |
| Eq. (4) CAS, Eq. (5) ρ | `halt.metrics.cas`, `halt.metrics.rho` |
| Table 1 interface | `halt.types.InterfaceCapabilities` |
| §4.1 five feature families | `halt.features.FeatureExtractor` |
| Eq. (6) monotone head | `halt.nn.head.MonotoneQuantileHead` |
| Eq. (7) pinball loss | `halt.nn.head.pinball_loss` |
| Eq. (8) conformal calibration | `halt.conformal.calibrate` |
| Def. 4.1 certified horizon | `halt.conformal.certified_horizon` |
| Prop. 4.1 / 4.2 | `halt.theory` |
| §4.4 planner shift, R1 / R2 | `halt.conformal.choose_kappa_plan`, `scripts/05` |
| Eq. (11) planning objective | `halt.planner` |
| Eq. (12) delayed ACI | `halt.aci.ACIController` |
| Prop. 5.1 simulation lemma | `halt.theory.prop_51_bound` |
| Prop. 5.2 long-run coverage | `halt.aci.prop_52_bound` |
| §6 PHANTASM | `halt.envs.phantasm` |
| Alg. G.1 offline calibration | `halt.calibrate.run_algorithm_g1` |
| Alg. 1 deployment loop | `halt.deploy.run_episode` |

---

## The pipeline

| Step | Script | What it does |
|---|---|---|
| 0 | `00_collect.py` | fit the world model; collect anchors (G.1, 1–4) |
| 1 | `01_calibrate.py` | fit the head, calibrate `λ̂` (G.1, 5–7); Bonferroni baseline |
| 2 | `02_evaluate.py` | coverage, state-dependence, SFR, grounding, AURC-C |
| 3 | `03_phantasm.py` | six-axis stress sweep against fixed-horizon |
| 4 | `04_aci.py` | delayed ACI vs static conformal under a real shift |
| 5 | `05_planner_shift.py` | §4.4: break exchangeability, then R1 and R2 |
| 6 | `06_ablation.py` | feature families, refit and recalibrated per cell |
| 7 | `07_deploy.py` | closed-loop MPC vs fixed horizons |
| 8 | `08_tables.py` | `SUMMARY.csv` and `REPORT.md`, one row per claim |

Each step reads the previous step's artefacts, so you can iterate on one
without rerunning the rest.

---

## What the tests actually check

`tests/test_theory.py` tries to falsify the propositions rather than restate
them:

- **Prop. 4.1** — coverage lands at `1 − α` across four levels, and the claim
  holds *simultaneously* over every horizon inside `H*`.
- **Eq. (8) vs Bonferroni** — the max-score certifies at least as far as
  per-horizon correction at `α/H_max`, at equal coverage.
- **Prop. 4.2** — the planner shift is constructed, the coverage shortfall is
  measured, R1 restores it, and R1's horizon cost is asserted.
- **Prop. 5.1** — the bound dominates a measured value gap; the confidence term
  is exactly linear in `α`; the optimal horizon moves with `τ` and `ε_V`.
- **Prop. 5.2** — ACI tracks the target under shift at three lag bounds, the
  telescoping identity holds to 1e-9, and a lag exceeding `L_max` raises.
- **Static conformal fails** under the same shift, so the comparison is real.
- **Algorithm 1 deadlocks** when abstention starves the ACI feedback loop, and
  the remedy restores it.

Three findings from writing them, all recorded in the code and in
[`docs/PAPER_NOTES.md`](docs/PAPER_NOTES.md):

**Clipping `α_t` breaks Prop. 5.2.** Eq. (12) has no clip. With one, the
telescoping identity stops holding and the realised deviation exceeded the
bound (1.3e-2 against 3.7e-3 at `L_max = 20`) while the controller looked fine.
The recursion is self-bounding without it, because a negative `α_t` widens the
certificate until misses stop. `ACIController` reports
`telescoping_residual`; the tests assert it is ~0.

**The Prop. 5.1 minimiser is at a boundary, not interior.** Both `H`-dependent
terms carry the same `γ^H`, so

```
d/dH [ L_r τ (1−γ^H)/(1−γ) + γ^H ε_V ] = γ^H |log γ| ( L_r τ/(1−γ) − ε_V )
```

has a sign independent of `H`, and the optimum switches at
`L_r τ/(1−γ) = ε_V`. The interior optimum one expects comes from the *classical*
simulation lemma, where model error compounds with `H` instead of saturating at
`τ` — replacing that compounding is exactly what the certificate buys. The
paper's claim survives: the optimal horizon depends on `τ` and `ε_V`, both
state-dependent, so no fixed hyperparameter tracks it. That is what the test
asserts. Worth a sentence in the paper, since a reviewer will differentiate it.

**Abstention starves the ACI feedback loop.** Line 9 of Algorithm 1 pushes a
rollout onto `Q` only in the *non-abstaining* branch, so a controller that
abstains generates no miss indicators, `α_t` freezes, and it abstains forever.
Measured: 12 feedback items in ~2500 control steps, `α_t → −0.22`, 99.5%
abstention. Prop. 5.2 is untouched — its average is over the steps that *were*
fed — so the bound holds while the controller is useless, which is why it does
not surface as a violated guarantee. `DeployConfig(shadow_rollouts=True)` is
one remedy; it is off by default so the default path stays faithful to the
published algorithm.

---

## Layout

```
halt/
  types.py          TaskStateSpec, InterfaceCapabilities, the WorldModel protocol
  conformal.py      Eq. 8, Def. 4.1, Bonferroni baseline, κ_plan
  aci.py            Eq. 12 with a bounded-lag queue, Prop. 5.2
  metrics.py        CAS, ρ, spectral entropy, TTCD, SFR, AURC-C, HCE
  theory.py         the propositions as evaluable bounds
  features.py       the five families, gated by Table 1
  jacobian.py       autodiff J_hat, finite-difference J*
  planner.py        CEM / MPPI truncated at H*, abstention
  collect.py        Algorithm G.1 lines 1–4
  calibrate.py      Algorithm G.1 lines 5–7, the Appendix H.2 τ heuristic
  deploy.py         Algorithm 1, three modes of Remark 4.2
  io.py             CSV + provenance sidecars
  nn/head.py        Eq. 6 and Eq. 7
  envs/             PlanarPush, PHANTASM, episode-level splits
  models/           K-ensemble, video stub, Dreamer/TD-MPC2/MuJoCo adapters
scripts/            the nine pipeline steps plus run_all.sh
tests/              60 tests
docs/               REAL_ENVIRONMENTS.md
```

---

## Design decisions worth knowing

**Features never see `y`.** `FeatureExtractor.extract` takes no ground-truth
argument at all, and a test asserts the signature stays that way. Remark 4.1's
three uses of the task state — features (never), calibration (offline),
ACI (at deployment) — are kept apart structurally rather than by discipline.

**Disabled feature families are zero-filled, not removed.** The input dimension
stays fixed across every ablation cell, so differences are attributable to
information rather than capacity.

**Episode-level splits only.** `episode_level_split` raises on leakage.
Anchors within an episode share a trajectory; splitting transitions destroys
the exchangeability Prop. 4.1 rests on in a way no amount of calibration data
repairs.

**ACI is refused, not degraded.** `run_deployment` raises when
`mode="aci"` meets `task_state_at_deployment=False`. Silently falling back to a
guarantee that does not hold is the worst available behaviour.

**No stub world models.** The Dreamer and TD-MPC2 adapters raise `ImportError`
when the upstream package is missing rather than returning something that
runs. A certificate computed against a stub is a certificate about nothing.

---

## Citation

```bibtex
@inproceedings{mohammad2026halt,
  title     = {{HALT}: Certifying Trust Horizons in Action-Conditioned Robotic World Models},
  author    = {Mohammad, Noor Islam S. and Sheikh, Md. Faisal and Habib, Jakaria},
  booktitle = {NeurIPS 2026 Workshop on Robot Learning with World Models:
               Capabilities, Frontiers, and Challenges},
  year      = {2026}
}
```

MIT licensed. The paper's text, figures and results remain the authors'.

---

## What the shipped run actually found

`runs/reference/results/REPORT.md` has the full table. The honest highlights,
including the rows that did not go the paper's way:

| Claim | Verdict on the reference suite |
|---|---|
| Prop. 4.1 coverage ≥ 1−α | **supported (conservative)** — 0.989 vs 0.950 |
| Eq. 8 beats the union bound | **supported** — H\* 7.34 vs 4.96; and Bonferroni is *infeasible* at n_cal = 400 |
| H\* is state-dependent | **supported** — 6.45 at contact vs 7.64 in free space |
| Appearance ≉ control validity (H1) | **supported** — rank corr 0.071, falsifier was > 0.70 |
| Blunter features certify *further* | **supported** |
| Coverage survives all six axes | **degraded** — five hold, sensing falls to 0.757 |
| Delayed ACI tracks the target | **supported** — deviation 0.0107, bound 0.0370, residual 0 |
| ACI beats static under shift | **NOT DEMONSTRATED** — static over-covered (0.0023), nothing to correct |
| Planning breaks exchangeability (§4.4) | **observed** — coverage drops to 0.647 |
| R1 restores coverage | **supported** — 1.000, at −65% horizon |
| H3: HALT-MPC beats fixed horizons | **UNINFORMATIVE** — every method scored 0% |
| ACI survives the abstention rule | **FAILS** — see Algorithm 1 above |
| Anything about MuJoCo, Dreamer, GAIA-1 | **NOT TESTED** |

Two of these deserve a sentence rather than a row.

**H3 is uninformative, not negative.** No method completed the task, including
all four fixed-horizon baselines, so there is nothing to compare. That is
coherent: the reference world model is action-collapsed (CAS₁ ≈ 0.04), and
planning against a model that barely responds to the action should fail. It is
a small vindication of Eq. 4–5 as a pre-flight check — the grounding metrics
predicted the planner would not work before it was run — but H3 needs a better
world model, not more episodes.

**AURC-C is where the trust signal clearly pays.** 0.0066 against 0.0288 for a
random trust score, a 4.4× improvement, threshold-free. That is the cleanest
positive result in the run and the one least sensitive to how τ was chosen.
