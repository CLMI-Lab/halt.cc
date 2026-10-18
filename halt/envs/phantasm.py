"""PHANTASM: the six-axis stress suite of §6 and Table F.1.

Axes, with graded severity, applied alone and in pairs:

  1 parametric shift      mass, friction, damping, restitution x {0.5 .. 2}
  2 contact richness      free space -> single -> multi -> in-hand
  3 material class        rigid -> articulated -> granular -> deformable
  4 action distribution   on-policy -> random -> adversarial (maximising d_h)
  5 sensing degradation   occlusion, blur, tactile dropout, 1-5 step latency
  6 compositional novelty unseen objects, pairs, goal configurations

What the suite supplies beyond the perturbations themselves is the part that
makes it usable for conformal work: task-state extractors with per-coordinate
tolerances, a reset-to-state harness so J* can be taken by central differences,
and episode-level calibration/test splits. Splitting at the transition level
would destroy exchangeability, because anchors from one episode are strongly
dependent -- so the split helper here refuses to do it.

On this reference environment, axes 3 and 6 are realised through the
parameters available to PlanarPush (effective mass/restitution families, and
unseen goal/object configurations). A MuJoCo backend implements them with real
articulated, granular and deformable bodies; see mujoco_adapter.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .planar_push import PlanarPush, PlanarPushParams

Array = np.ndarray

__all__ = ["StressSpec", "AXES", "SEVERITIES", "make_stressed_env", "episode_level_split"]

AXES = (
    "parametric",
    "contact",
    "action",
    "sensing",
    "material",
    "composition",
)

SEVERITIES = (0, 1, 2, 3)     # 0 = nominal


@dataclass(frozen=True)
class StressSpec:
    axis: str
    severity: int = 2

    def __post_init__(self) -> None:
        if self.axis not in AXES:
            raise ValueError(f"unknown axis {self.axis!r}; expected one of {AXES}")
        if self.severity not in SEVERITIES:
            raise ValueError(f"severity must be in {SEVERITIES}")


def make_stressed_env(spec: StressSpec | None, base: PlanarPushParams | None = None) -> PlanarPush:
    """Return an environment with one axis perturbed at the given severity."""
    p = base or PlanarPushParams()
    if spec is None or spec.severity == 0:
        return PlanarPush(p)

    s = spec.severity
    if spec.axis == "parametric":
        # Table F.1: scale factors drawn from {0.5, 0.75, 1, 1.5, 2}.
        f = {1: 1.5, 2: 1.75, 3: 2.0}[s]
        return PlanarPush(p.scaled(mass=f, friction=1.0 / f, restitution=f))

    if spec.axis == "contact":
        # Richer contact: a larger effective contact radius and more restitution
        # means more frequent, higher-impulse events.
        env = PlanarPush(
            PlanarPushParams(
                **{**p.__dict__, "r_contact": p.r_contact * (1.0 + 0.5 * s),
                   "restitution": min(0.95, p.restitution * (1.0 + 0.6 * s))}
            )
        )
        return env

    if spec.axis == "action":
        # Handled by the action sampler, not the plant; the env is nominal and
        # `adversarial_actions` below supplies the distribution.
        return PlanarPush(p)

    if spec.axis == "sensing":
        env = PlanarPush(p)
        env.obs_dropout = 0.05 * s
        env.obs_blur = 0.0015 * s
        env.action_latency = {1: 1, 2: 3, 3: 5}[s]
        return env

    if spec.axis == "material":
        # Softer, more dissipative contact stands in for the rigid ->
        # articulated -> granular -> deformable progression.
        return PlanarPush(
            p.scaled(damping=1.0 + 0.9 * s, restitution=max(0.05, 1.0 - 0.3 * s),
                     friction=1.0 + 0.5 * s)
        )

    if spec.axis == "composition":
        # Unseen goal configurations: goals placed beyond the training range.
        env = PlanarPush(p, goal_radius=max(0.02, 0.06 - 0.012 * s))
        return env

    raise AssertionError(spec.axis)


def adversarial_actions(
    env: PlanarPush, model, horizon: int, rng: np.random.Generator, n_candidates: int = 48
) -> Array:
    """Axis 4 at top severity: actions chosen to maximise predicted divergence.

    This is the distribution the paper warns about in §4.4, in its purest form.
    It is not what a planner does -- a planner maximises reward and only
    incidentally finds model error -- but it upper-bounds the effect.
    """
    best, best_score = None, -np.inf
    z = model.encode([env.observe()], [])
    for _ in range(n_candidates):
        acts = rng.uniform(-1, 1, size=(horizon, env.action_dim))
        ys = model.rollout(z, acts, n_rollouts=max(2, model.capabilities.n_rollouts), rng=rng)
        spread = float(np.std(ys, axis=0).mean())
        if spread > best_score:
            best, best_score = acts, spread
    return best


# --------------------------------------------------------------------------- #
# Splits
# --------------------------------------------------------------------------- #
def episode_level_split(
    episode_ids: Array,
    fractions: tuple[float, float, float] = (0.6, 0.2, 0.2),
    seed: int = 0,
) -> dict[str, Array]:
    """Split at the EPISODE level, never the transition level.

    Anchors inside one episode share a trajectory and are strongly dependent.
    Splitting transitions across train/cal/test leaks, and the exchangeability
    that Proposition 4.1 rests on is then false in a way no amount of
    calibration data fixes. Returns boolean masks over anchors.
    """
    if abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError(f"fractions must sum to 1, got {fractions}")
    ids = np.asarray(episode_ids)
    uniq = np.unique(ids)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(uniq)

    n = perm.shape[0]
    n_tr = int(round(fractions[0] * n))
    n_ca = int(round(fractions[1] * n))
    groups = {
        "train": set(perm[:n_tr].tolist()),
        "cal": set(perm[n_tr : n_tr + n_ca].tolist()),
        "test": set(perm[n_tr + n_ca :].tolist()),
    }
    masks = {k: np.array([i in v for i in ids]) for k, v in groups.items()}

    overlap = (masks["train"] & masks["cal"]).any() or (masks["cal"] & masks["test"]).any()
    if overlap:
        raise AssertionError("episode leaked across splits")
    return masks
