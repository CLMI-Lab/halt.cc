# Running HALT on real tasks

Everything in `halt/` is written against the `WorldModel` protocol in
`halt/types.py`, which *is* the Table 1 interface. Swapping the reference suite
for MuJoCo + DreamerV3 means supplying that interface and nothing else.

This document is the honest version of "how do I reproduce the paper". The
short answer is that you cannot reproduce it from this repository alone,
because the paper's numbers depend on trained DreamerV3 and TD-MPC2 agents on
five MuJoCo tasks across ten seeds. What you can do is run the same algorithms
against those agents once you have them.

---

## 1. What you need

| Piece | Why | Where it plugs in |
|---|---|---|
| MuJoCo + the five tasks | the dynamics | `halt/envs/mujoco_adapter.py` |
| A trained world model | the thing being wrapped | `halt/models/adapters.py` |
| A **task-state readout** `h_ω` | error is measured in `y`, not pixels | your adapter's `_readout` |
| Per-coordinate tolerances `σ` | makes `τ = 1` mean something | `MuJoCoTaskSpecs` |
| Reset-to-state | ground-truth replay and `J*` | your env wrapper |

The third row is the one that catches people. Dreamer ships an image decoder,
not a pose head. A certificate over pixel reconstruction error certifies the
wrong thing.

---

## 2. The environment adapter

Implement the `Environment` protocol. Six methods, and one property that is not
optional:

```python
class MuJoCoTask:
    task_spec = MuJoCoTaskSpecs.by_name("peginsertion-v2")
    action_dim = 7

    def reset(self, rng): ...
    def step(self, action): ...                 # -> (obs, reward, done, info)
    def task_state(self):                       # -> y, the readout. NOT the qpos vector.
        return np.concatenate([peg_pos, peg_euler, [float(in_contact)]])
    def get_state(self):                        # mujoco: (qpos, qvel, act, time)
        return np.concatenate([d.qpos, d.qvel, d.act, [d.time]])
    def set_state(self, s): ...                 # and mj_forward afterwards
    def supports_reset_to_state(self): return True
```

**`get_state` / `set_state` must round-trip exactly.** Two things depend on it:
ground-truth replay during collection, and the central differences that give
`J*`. The test to copy is `test_env_reset_to_state_is_exact`; run it against
your adapter before anything else. A `set_state` that forgets `mj_forward`
leaves derived quantities stale and the replay silently drifts.

`info` should carry `contact`, `tactile`, `success` and `unsafe`. The first two
feed the multimodal-surprise family; the last two are what the deployment
script scores.

---

## 3. The world-model adapter

Subclass `_TorchLatentAdapter` in `halt/models/adapters.py` and supply three
differentiable callables:

```python
def _encode(self, obs_history, act_history):  # -> latent
def _step(self, z, a):                        # -> next latent, DIFFERENTIABLE
def _readout(self, z):                        # -> task state, DIFFERENTIABLE
```

Then declare what you actually have:

```python
InterfaceCapabilities(
    stochastic_rollouts=True,            # K >= 2 sampled rollouts
    differentiable_readout=True,         # autodiff reaches the task state
    task_state_at_calibration=True,      # y available offline
    task_state_at_deployment=False,      # on hardware this is almost always False
    n_rollouts=5,
)
```

Declare these honestly. They are not configuration — they gate which feature
families exist and which deployment modes are legal. Setting
`task_state_at_deployment=True` on a robot with no mocap gets you an ACI run
whose guarantee does not hold, which is worse than no certificate at all.
`run_deployment` refuses ACI when the flag is `False`; that refusal is the
feature.

### Gotchas, in the order people hit them

**A detached sample kills the Jacobian family silently.** If `_step` samples
with `.detach()` or `torch.no_grad()`, `torch_rollout_jacobian` returns zeros,
the Jacobian features become a constant block, and the head quietly ignores
them. Use the reparameterised sample. Verify with the test pattern in
`test_autodiff_jacobian_matches_a_finite_difference_of_the_same_model`: finite-
difference your own `_step` and compare. A CAS below 0.99 there is a bug in the
adapter; a low CAS against the *environment* is a finding about the model.

**TD-MPC2 calibration is already planner-conditional.** Its rollouts come from
its own MPPI, so you get remedy (R1) by default. Good — but the Table 4
comparison then needs a separate collection under a non-planning policy, or
you are comparing planner scores against planner scores and measuring nothing.

**Autoregressive rollouts diverge.** Over 30-50 steps a learned model fed its
own output can leave the training manifold entirely. In the reference
implementation this showed up as a mean `d_30` of 1.8e4 against a median of 48:
a handful of runaway anchors, each of which would dominate the conformal score
and inflate `λ̂` for everything else. `EnsembleWorldModel` clips the readout to
the training range widened by 20%. Do the same in your adapter, and check the
mean-to-median ratio of `d_h` at your largest horizon before trusting a
calibration.

---

## 4. Choosing σ and τ

`σ` is a task-design decision. Set each coordinate to the error that would
actually matter: 5 mm on peg position because that is the clearance, 2° on
orientation because that is when insertion jams. Then `τ = 1` means "at the
edge of task tolerance" and transfers across tasks.

`τ` is also a design decision, not a statistical one. The certificate is valid
for whatever `τ` you supply; `τ` decides *what is being certified*, not whether
the guarantee holds. `calibrate_tolerance` implements the Appendix H.2
heuristic (`τ = √(P75(d₁₀)·P75(d₂₀))`) as a starting point, but it is a way of
landing somewhere a planner can operate, not a principled choice.

Two failure modes, both real:

- **σ far too tight.** The reference suite first used 5 mm on object position
  and 2 cm/s on velocity. A competent one-step model then sat at `d₁ ≈ 9`
  tolerance units, every horizon was already beyond tolerance, and the
  certificate had nothing to say. Setting σ an order of magnitude below what
  the task requires is not conservatism — it is asking a miscalibrated
  question.
- **Non-task-defining coordinates in `y`.** Object *velocity* was in the
  reference readout initially. It is a state variable, not a task-defining
  quantity — nobody specifies a pushing task in terms of instantaneous
  velocity — and because it jumps discontinuously at contact it dominated `d_h`
  with error that no one cares about. §3 lists what belongs in `y`: object
  poses, end-effector pose, contact indicators, gripper aperture, and other
  task-defining quantities.

---

## 5. The run

```bash
python3 scripts/00_collect.py       --run mujoco_pusher --kind dreamer
python3 scripts/01_calibrate.py     --run mujoco_pusher --kind dreamer
python3 scripts/02_evaluate.py      --run mujoco_pusher --kind dreamer
python3 scripts/03_phantasm.py      --run mujoco_pusher
python3 scripts/04_aci.py           --run mujoco_pusher
python3 scripts/05_planner_shift.py --run mujoco_pusher
python3 scripts/06_ablation.py      --run mujoco_pusher
python3 scripts/07_deploy.py        --run mujoco_pusher
python3 scripts/08_tables.py        --run mujoco_pusher
```

For the paper's matrix, loop over `--seed 0..9` and three model kinds, then
aggregate. The scripts take `--seed` precisely so this is a shell loop rather
than a code change.

Budget honestly: the dominant cost is **not** HALT. Calibration is one forward
pass per anchor plus a quantile. The costs are training the world models in the
first place, and the `2·d_a` simulator resets per `J*` probe — which is why
`CollectConfig.jacobian_every` subsamples, and why `J*` is needed only for
*evaluating* CAS and ρ, never at deployment.

---

## 6. Before you believe a number

1. `pytest tests/ -q` passes against your adapter.
2. `test_env_reset_to_state_is_exact` passes with your `get_state`/`set_state`.
3. Autodiff and finite-difference Jacobians of *your own* `_step` agree to CAS > 0.99.
4. `d_h` mean-to-median ratio at your largest horizon is below ~2 (no runaway rollouts).
5. Splits are at the episode level — `episode_level_split` raises if they leak.
6. Coverage on held-out *behaviour-policy* anchors lands at `1 − α` before you
   interpret anything about shift. If Prop. 4.1 does not hold in the regime it
   assumes, nothing downstream means anything.
7. `aci_summary.csv` shows `telescoping_residual ≈ 0`. If it does not, the
   recursion was not applied exactly and the Prop. 5.2 bound does not follow.
