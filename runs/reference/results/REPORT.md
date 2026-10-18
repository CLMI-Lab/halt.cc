# HALT reference implementation — run summary

Run: `reference`

## Scope

These results come from the **PlanarPush reference suite**, a synthetic
contact-rich environment bundled with this repository. They are **not** a
reproduction of the paper's Section 8 experiments, which use five MuJoCo
tasks, ten seeds and three world-model classes. What they establish is that
the algorithms are implemented correctly and that the propositions hold
where they are supposed to.

## Claims

| Claim | Paper | Measured | Verdict |
|---|---|---|---|
| Simultaneous coverage reaches 1 - alpha | Prop. 4.1 (one-sided) | 0.989 vs target 0.950 | **supported (conservative)** |
| Max-score beats the union bound on horizon | §4.3 | H* 7.34 vs Bonferroni 4.96 | **supported** |
| H* is state-dependent (contact vs free space) | §1, Prop. 5.1 | H* 6.45 at contact vs 7.64 free; d_1 2.39 vs 2.61 | **supported** |
| Appearance is a poor proxy for control validity | H1, §9.5 | rank corr(appearance, TTCD) = 0.071 | **supported** |
| Richer features improve the trust signal | H2, §9.2 | full 0.985 vs worst subset 0.978 (disagreement only) | **supported** |
| Blunter features certify FURTHER, not just worse | §9.2 discussion | 1/6 subsets have a longer mean H* than the full set | **supported** |
| Coverage survives the six stress axes | §9.3, Table 6 | HALT 0.757-1.000 vs fixed 0.364-0.763 | **degraded** |
| Delayed ACI tracks the target level | H4, Prop. 5.2, §9.4 | ACI miss 0.0393, target 0.050, deviation 0.0107 | **supported** |
| ACI outperforms static conformal under shift | §9.4 | static miss 0.0023 is BELOW target 0.050 | **NOT DEMONSTRATED** |
| Closed-loop planning breaks exchangeability | §4.4 | coverage drops to 0.647 under the planner | **observed** |
| Remedy (R1) restores coverage, at a horizon cost | §4.4 (R1), Prop. 4.2 | coverage 1.000; H* 5.72 -> 1.99 (-65%) | **supported** |
| HALT-truncated MPC beats the best fixed horizon | H3, §9.1 | every method scored 0% success | **UNINFORMATIVE** |
| ACI feedback survives the abstention rule | Alg. 1, lines 5-14 | abstention 99.5%, only 12 feedback items, alpha_t -> -0.220 | **FAILS** |
| The paper's Table 3 numbers on five MuJoCo tasks | §8-9.1 | not measured here | **NOT TESTED** |
| DreamerV3 and TD-MPC2 as the wrapped models | §8 | not measured here | **NOT TESTED** |
| GAIA-1 coverage of 0.92 with the Jacobian family dropped | Table 9 | not measured here | **NOT TESTED** |
| Real-robot transfer | §10 | not measured here | **NOT TESTED** |

## Notes

- **Simultaneous coverage reaches 1 - alpha** — n_certified=730, HCE=0.0396; the bound is >= 1 - alpha, so excess coverage is conservatism, paid for in a shorter certified horizon
- **H* is state-dependent (contact vs free space)** — a fixed horizon must be wrong in one regime or the other
- **Appearance is a poor proxy for control validity** — pre-registered falsifier was > 0.70
- **Blunter features certify FURTHER, not just worse** — confidently wrong for longer is the worse failure
- **Coverage survives the six stress axes** — worst deviation from target 0.193
- **Delayed ACI tracks the target level** — Prop. 5.2 bound respected: True, telescoping residual 0
- **ACI outperforms static conformal under shift** — static over-covered here rather than failing, so the shift induced no shortfall for ACI to correct. The comparison needs a calibration with less headroom, or a harsher shift
- **HALT-truncated MPC beats the best fixed horizon** — no method completed the task, so there is nothing to compare. The world model is action-collapsed (CAS ~ 0.04), and planning against a model that barely responds to the action is expected to fail -- which is coherent with Eq. 4-5 but leaves H3 untested here
- **ACI feedback survives the abstention rule** — line 9 pushes onto Q only in the non-abstaining branch, so an abstaining controller generates no evidence and alpha_t can never recover. See docs/PAPER_NOTES.md finding 4; DeployConfig(shadow_rollouts=True) is one remedy
- **The paper's Table 3 numbers on five MuJoCo tasks** — requires MuJoCo and trained agents; see docs/REAL_ENVIRONMENTS.md
- **DreamerV3 and TD-MPC2 as the wrapped models** — requires MuJoCo and trained agents; see docs/REAL_ENVIRONMENTS.md
- **GAIA-1 coverage of 0.92 with the Jacobian family dropped** — requires MuJoCo and trained agents; see docs/REAL_ENVIRONMENTS.md
- **Real-robot transfer** — requires MuJoCo and trained agents; see docs/REAL_ENVIRONMENTS.md
