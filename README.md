# HALT — reference implementation

Certified, state-dependent trust horizons for action-conditioned world models.

> Mohammad, Sheikh and Habib. *HALT: Certifying Trust Horizons in
> Action-Conditioned Robotic World Models.* NeurIPS 2026 Workshop on Robot
> Learning with World Models.

```bash
pip install -r requirements.txt
pytest tests/ -q          # 60 tests, ~50 s
bash scripts/run_all.sh   # full pipeline, ~15 min on 2 CPUs
```

---

## Scope

The repository has two layers.

**The algorithms** — environment-independent, and what you would run on a real
robot: the monotone quantile head, the pinball objective, simultaneous
split-conformal calibration, the certified horizon, the planning objective,
delayed-feedback ACI, the six action-conditioned metrics, both algorithms, and
the PHANTASM stress axes.

**A runnable suite** — `PlanarPush`, a synthetic contact-rich environment with
exact reset-to-state, bundled so the pipeline executes without MuJoCo. It is
what the shipped CSVs were produced from.

The CSVs in `runs/` are **not** a reproduction of the paper's Table 3, which
requires DreamerV3 and TD-MPC2 trained on five MuJoCo tasks across ten seeds.
Each CSV carries a `provenance` column and a `.meta.json` sidecar recording
this, and `results/SUMMARY.csv` marks the MuJoCo claims as not tested. To run
against real tasks, see [`docs/REAL_ENVIRONMENTS.md`](docs/REAL_ENVIRONMENTS.md).

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

## Pipeline

| Step | Script | What it does |
|---|---|---|
| 0 | `00_collect.py` | fit the world model; collect anchors (G.1, 1–4) |
| 1 | `01_calibrate.py` | fit the head, calibrate `λ̂` (G.1, 5–7); Bonferroni baseline |
| 2 | `02_evaluate.py` | coverage, state-dependence, SFR, grounding, AURC-C |
| 3 | `03_phantasm.py` | six-axis stress sweep against a fixed horizon |
| 4 | `04_aci.py` | delayed ACI against static conformal under shift |
| 5 | `05_planner_shift.py` | §4.4: planner-induced shift, then R1 and R2 |
| 6 | `06_ablation.py` | feature families, refit and recalibrated per cell |
| 7 | `07_deploy.py` | closed-loop MPC against fixed horizons |
| 8 | `08_tables.py` | `SUMMARY.csv` and `REPORT.md`, one row per claim |

Each step reads the previous step's artefacts, so a single step can be rerun
without repeating the others. All steps take `--run` and `--seed`.

Results land in `runs/<name>/`:

```
results/       CSVs, each with a .meta.json provenance sidecar
checkpoints/   world models, trained heads, calibration scores
logs/          the full run log
```

---

## Tests

`pytest tests/ -q` runs 60 tests across three files.

- `test_theory.py` — the propositions, evaluated numerically: marginal and
  simultaneous coverage at four miscoverage levels, the max-score against a
  Bonferroni baseline, the planner-shift shortfall and its remedy, the
  Prop. 5.1 bound against a measured value gap, and ACI tracking under shift
  at three lag bounds.
- `test_components.py` — head monotonicity and positivity, the pinball
  minimiser, feature gating, every metric, episode-level splits, exact
  reset-to-state, and agreement between the autodiff and finite-difference
  Jacobians.
- `test_pipeline.py` — collect → calibrate → certify → deploy end to end,
  including checkpoint round-tripping and the abstention rule.

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
  deploy.py         Algorithm 1, the three deployment modes
  io.py             CSV writers and provenance sidecars
  nn/head.py        Eq. 6 and Eq. 7
  envs/             PlanarPush, PHANTASM, episode-level splits
  models/           K-ensemble, video model, Dreamer/TD-MPC2/MuJoCo adapters
configs/            reference.yaml, mujoco_pusher.yaml
scripts/            the nine pipeline steps plus run_all.sh
tests/              60 tests
docs/               REAL_ENVIRONMENTS.md
```

---

## Implementation notes

**Features never receive the task state.** `FeatureExtractor.extract` takes no
ground-truth argument, and a test asserts the signature stays that way. The
three uses of `y` — features, calibration, ACI feedback — are separated
structurally.

**Disabled feature families are zero-filled, not removed.** The input
dimension is constant across ablation cells, so differences reflect
information rather than capacity.

**Splits are at the episode level.** `episode_level_split` raises on leakage.
Anchors within an episode share a trajectory, and splitting transitions
breaks exchangeability.

**ACI is refused rather than degraded.** `run_deployment` raises when
`mode="aci"` meets `task_state_at_deployment=False`, instead of running a
guarantee whose precondition is absent.

**Adapters do not stub.** The DreamerV3 and TD-MPC2 adapters raise
`ImportError` when the upstream package is missing.

**Rollouts are clipped to the training range.** Autoregressive rollouts can
leave the training manifold and diverge over long horizons; a few such
anchors would dominate the conformal score.

---

## Requirements

Python ≥ 3.10, NumPy, SciPy, PyTorch. MuJoCo, DreamerV3 and TD-MPC2 are
optional and only needed for the real-task path.

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
