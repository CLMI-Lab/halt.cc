# Notes for the authors, from implementing the paper

Three things surfaced while making the methods executable. None of them
invalidates anything; the first two are worth a sentence in the paper because a
careful reviewer will find them, and the third is a reproducibility detail.

---

## 1. Proposition 5.1's minimiser is at a boundary, not interior

The paper reads:

> Terms one and two trade off, so an optimal horizon exists, it is
> state-dependent, and no fixed hyperparameter can track it.

The trade-off is real — the terms have opposite signs in `H`. But both carry
the same `γ^H` factor, so differentiating Eq. (13) in `H` gives

```
d/dH [ L_r·τ·(1 − γ^H)/(1 − γ)  +  γ^H·ε_V ]
    = γ^H · |log γ| · ( L_r·τ/(1 − γ)  −  ε_V )
```

whose sign does not depend on `H`. The minimiser is therefore bang-bang: `H_max`
when `L_r·τ/(1 − γ) < ε_V`, and `0` otherwise, switching exactly at
`L_r·τ/(1 − γ) = ε_V`.

The interior optimum one intuitively expects comes from the *classical*
simulation lemma, where the model-error term compounds with `H` rather than
saturating at the certified tolerance — and replacing that compounding with a
constant `τ` is precisely what the certificate buys. So the substantive claim
survives intact: the optimal horizon is a function of `τ` and `ε_V`, both of
which vary from state to state, and no fixed hyperparameter tracks it.

**Suggestion.** Phrase it as a regime switch rather than an interior optimum,
and state the switching condition — it is a cleaner result than "an optimum
exists", and it makes the state-dependence explicit in one line.

Tests: `test_prop_51_optimal_horizon_is_state_dependent`,
`test_prop_51_regime_switch_is_where_the_algebra_says`.

---

## 2. Clipping α_t silently breaks Proposition 5.2

Eq. (12) contains no clipping, and it must not. The bound comes from
telescoping the recursion:

```
Σ_i miss_i = n·α + (α_1 − α_{n+1}) / η
```

Clipping discards part of an increment, the identity stops holding, and the
realised deviation can exceed the bound while the controller looks perfectly
well behaved. Measured, with `α = 0.05`, `η = 0.05`, `T = 6000`, `L_max = 20`:

| | clipped to [−η, 1+η] | unclipped |
|---|---|---|
| telescoping residual | 3e-4 | < 1e-12 |
| realised deviation | **1.26e-2** | 2.7e-4 |
| Prop. 5.2 bound | 3.67e-3 | 3.67e-3 |
| bound respected | **no** | yes |

No clip is needed. The recursion is self-bounding whenever the miss process
responds to the level: if `α_t` drops below 0 the certificate is at its widest,
misses go to zero, and the next update raises `α_t` by `η·α`. That is the
interval Gibbs & Candès *assume*, not one to impose.

The paper does not mention clipping, so nothing here is wrong — but the
reference implementation does mention it loudly, because clipping is the
obvious defensive thing for an implementer to add, and it is exactly wrong.
`ACIController.summary()` reports `telescoping_residual` so a run can be
audited; the tests assert it is ~0 before asserting the bound.

**Suggestion.** One sentence in §4.6 or Appendix C.4: no clipping, and why.

---

## 3. CAS and ρ are doing their job, and the reference model fails them

On the bundled suite, CAS₁ ≈ 0.04–0.07 and ρ₁ is strongly negative — textbook
action collapse, the pathology §3 predicts for reconstruction-trained models.
Inspecting the Jacobians shows why: the model puts almost all of its action
sensitivity on the *contact indicator* coordinate, while the true Jacobian puts
it on the *object position*. The two are nearly orthogonal after Frobenius
normalisation, so CAS ≈ 0.

This is worth stating as a usage note, because it is easy to read a low CAS as
a bug. Two checks separate the cases:

- autodiff vs a finite difference **of the same network** → agreement must be
  CAS > 0.99, or `torch_rollout_jacobian` is broken (we measured 0.9992);
- autodiff vs the **environment** → a low value is a finding about the model.

A detached sample inside the latent step collapses the first check to zero
without any error, and the Jacobian feature family then contributes a constant
block that the head quietly ignores. Worth flagging in any released code.

Test: `test_autodiff_jacobian_matches_a_finite_difference_of_the_same_model`.

---

## 4. Smaller reproducibility notes

**The Bonferroni baseline is often infeasible, not merely worse.** With
`H_max = 30` and `α = 0.05`, each horizon is calibrated at `α/H_max ≈ 0.0017`,
which needs `⌈(n+1)(1 − α/H_max)⌉ ≤ n`. Below roughly 600 calibration anchors
that is impossible and the union bound certifies *nothing*. The paper frames
the comparison as a loss of horizon; at realistic calibration sizes it is
sharper than that, and the sharper version is a better argument for Eq. (8).

**`H_task < H_max` in Definition 3.2.** Since TTCD uses `min ∅ := H_max`,
setting `H_task = H_max` makes every non-diverging rollout satisfy
`TTCD ≤ H_task`, and SFR then reports every plausible-looking rollout as a
silent failure. Our implementation raises rather than returning the degenerate
number. The paper's `H_task` is presumably well below `H_max`; saying so would
prevent the mistake.

**Table 1 row 4 is load-bearing and easy to default wrong.** Implementations
naturally default `task_state_at_deployment=True` because it is true in
simulation. `run_deployment` refuses ACI when it is `False` instead of
degrading to a guarantee that does not hold — that refusal is worth making
explicit in the paper's deployment-mode discussion, since the failure is
silent otherwise.

---

## 5. Algorithm 1: abstention starves the ACI feedback loop

**This is the finding I would most want you to see.**

Line 9 (`push (t, ŷ, H*, α_t) onto Q`) sits inside the **else** branch of the
line-5 abstention test. So a controller that abstains queues nothing, scores
nothing, and delivers no miss indicator to line 14. The ACI feedback loop and
the abstention rule are coupled, and the coupling has an absorbing state:

```
a few early misses  →  α_t falls  →  λ̂(α_t) rises
→  H* collapses below H_min  →  the controller abstains
→  nothing is queued  →  no feedback arrives
→  α_t never moves again  →  it abstains forever
```

Measured on the reference suite, 25 episodes, ACI mode:

| | |
|---|---|
| control steps | ~2500 |
| feedback items reaching line 14 | **12** |
| α_t at the end | **−0.22** |
| abstention rate | **99.5%** |
| mean H* | 0.03 |

Nothing recovers it, because recovery requires exactly the evidence that
abstention refuses to generate. Note that **Proposition 5.2 is untouched** —
its average runs over the steps that were fed, and those twelve were fine.
The bound holds and the controller is useless at the same time, which is why
this does not show up as a violated guarantee.

### Remedies, and what each costs

- **Shadow rollouts.** While abstaining, still imagine a rollout at the
  horizon that *would* have been certified, score it when it elapses, feed
  that to ACI. The safe policy is still executed, so the robot's behaviour is
  unchanged and the cost is imagined rollouts only. Implemented as
  `DeployConfig(shadow_rollouts=True)`, **off by default** so the default
  path stays faithful to Algorithm 1 as published.
- **Treat an abstention as an observed non-miss.** Cheapest, but biases the
  long-run average downward: α_t drifts up and the controller becomes
  over-confident precisely in the states it was unsure about.
- **Floor α_t.** Stops the absorbing state but discards the adaptivity that
  motivated ACI in the first place.
- **A separate always-on probe horizon.** Cleanest statistically, since the
  probe distribution does not depend on the abstention decision, but it
  changes what the certificate is about.

**Suggestion.** Move line 9 outside the else-branch, or add a sentence to
§4.6 naming the coupling and the chosen remedy. Any of the four is defensible;
having none is not, and an implementer following Algorithm 1 literally will
hit the deadlock.

Test: `test_aci_deadlocks_on_abstention_and_shadow_rollouts_fix_it`.

---

## 6. Where the reference run did NOT support the paper

Reported here rather than buried, because they are the interesting rows.

**Sensing degradation breaks the certificate.** Table 6 reports coverage
within 2% of target on all six axes. On the reference suite five axes hold
(0.876–1.000) but **sensing degradation falls to 0.757**, with H* collapsing
from ~7 to 2.5. The head does become less confident, which is the right
direction, but not nearly enough. Occlusion and actuation latency change the
*observation* process, so the features themselves become unreliable — and
every other axis changes the dynamics while leaving the features trustworthy.
That asymmetry may be worth separating in the paper's axis taxonomy. Whether
a better-trained head closes the gap is an open question on this suite.

**The ACI-vs-static comparison did not reproduce.** Static conformal did not
fail under the shift: its miss rate was **0.0023 against a target of 0.05**,
i.e. heavily over-covering, so there was no shortfall for ACI to correct. The
cause is that λ̂ was calibrated conservatively (coverage 0.989), leaving so
much headroom that a severity-3 parametric shift could not consume it. ACI
still tracked the target correctly and the Prop. 5.2 bound held with a
telescoping residual of exactly 0 — the *mechanism* is verified, the
*comparison* is not. A sharper demonstration needs a calibration with less
headroom or a harsher shift.

**H3 is untested here, not refuted.** Every method scored 0% success in
closed-loop deployment, including all four fixed-horizon baselines, so there
is nothing to compare. This is coherent rather than mysterious: the reference
world model is action-collapsed (CAS₁ ≈ 0.04), and planning against a model
that barely responds to the action should fail. It is a small vindication of
Eq. 4–5 as a pre-flight check — the grounding metrics predicted that planning
would not work before the planner was run — but it means H3 needs a better
world model, not more episodes.
