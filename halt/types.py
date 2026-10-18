"""Core types and the Table 1 interface contract.

The paper is explicit (§3.1, Table 1) that HALT's guarantee is only meaningful
when the wrapped world model exposes a particular interface. That contract is
expressed here as a runtime-checkable protocol plus a capability record, so a
model that cannot satisfy part of it fails loudly at wrap time rather than
silently producing a certificate that means nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Protocol, Sequence, runtime_checkable

import numpy as np

Array = np.ndarray


# --------------------------------------------------------------------------- #
# Task-state specification
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TaskStateSpec:
    """The readout y and its per-coordinate tolerances sigma_i (Eq. 2).

    Error is measured in y, never in pixels. Dividing by sigma_i makes d_h
    dimensionless, so tau = 1 means 'at the edge of task tolerance' in every
    environment and one threshold suffices across tasks.
    """

    names: tuple[str, ...]
    sigma: Array  # (d_y,) strictly positive per-coordinate task tolerances

    def __post_init__(self) -> None:
        sigma = np.asarray(self.sigma, dtype=np.float64)
        if sigma.ndim != 1:
            raise ValueError(f"sigma must be 1-D, got shape {sigma.shape}")
        if len(self.names) != sigma.shape[0]:
            raise ValueError(
                f"{len(self.names)} names but {sigma.shape[0]} sigmas"
            )
        if not np.all(sigma > 0):
            bad = [n for n, s in zip(self.names, sigma) if s <= 0]
            raise ValueError(f"sigma must be positive; non-positive for {bad}")
        object.__setattr__(self, "sigma", sigma)

    @property
    def dim(self) -> int:
        return self.sigma.shape[0]

    def weighted_norm(self, delta: Array) -> Array:
        """||delta||_Sigma with Sigma = diag(sigma^-2), i.e. Eq. (2).

        Accepts (..., d_y) and reduces the last axis, so it works for a single
        error vector, a horizon of them, or a whole batch of rollouts.
        """
        delta = np.asarray(delta, dtype=np.float64)
        if delta.shape[-1] != self.dim:
            raise ValueError(
                f"last axis must be d_y={self.dim}, got {delta.shape[-1]}"
            )
        return np.sqrt(np.sum((delta / self.sigma) ** 2, axis=-1))


# --------------------------------------------------------------------------- #
# Interface capabilities (Table 1)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class InterfaceCapabilities:
    """Which rows of Table 1 a given world model satisfies.

    stochastic_rollouts      K >= 2 rollouts            -> feature D_h
    differentiable_readout   autodiff through h_omega   -> Jacobian features
    task_state_at_calibration                           -> the certificate itself
    task_state_at_deployment                            -> ACI miss indicator
    """

    stochastic_rollouts: bool
    differentiable_readout: bool
    task_state_at_calibration: bool
    task_state_at_deployment: bool
    n_rollouts: int = 1

    def missing_for_certificate(self) -> list[str]:
        """What is missing for any certificate at all to be meaningful."""
        missing = []
        if not self.task_state_at_calibration:
            missing.append(
                "task_state_at_calibration (no y offline => q_hat and "
                "lambda_hat cannot be fitted; Table 1 says fall back to a "
                "fixed horizon)"
            )
        return missing

    def disabled_feature_families(self) -> list[str]:
        """Feature families that must be dropped, with the Table 1 reasons."""
        off = []
        if not self.stochastic_rollouts or self.n_rollouts < 2:
            off.append("disagreement")
        if not self.differentiable_readout:
            off.append("jacobian")
        return off

    def aci_available(self) -> bool:
        """ACI needs y_{t+h} observed at deployment (Remark 4.1(c))."""
        return self.task_state_at_deployment

    def as_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- #
# World-model protocol
# --------------------------------------------------------------------------- #
@runtime_checkable
class WorldModel(Protocol):
    """The minimal surface HALT wraps. M_theta is frozen; HALT never trains it."""

    task_spec: TaskStateSpec
    capabilities: InterfaceCapabilities
    action_dim: int

    def encode(self, obs_history: Sequence[dict], act_history: Sequence[Array]) -> Array:
        """z_t ~ q_phi(z_t | o_<=t, a_<t). Returns the latent."""
        ...

    def rollout(self, z: Array, actions: Array, n_rollouts: int, rng: np.random.Generator) -> Array:
        """Imagine forward.

        actions : (H, d_a)
        returns : (K, H, d_y) decoded task states y_hat, one per stochastic rollout
        """
        ...

    def jacobian(self, z: Array, actions: Array, horizons: Sequence[int]) -> Array | None:
        """J_hat_h = d y_hat_{t+h} / d a_t by autodiff through the latent rollout.

        Returns (len(horizons), d_y, d_a), or None when the model exposes no
        differentiable task readout (Table 1 row 2).
        """
        ...


@runtime_checkable
class Environment(Protocol):
    """Ground truth source. Only used offline, and by PHANTASM for J*."""

    task_spec: TaskStateSpec
    action_dim: int

    def reset(self, rng: np.random.Generator) -> dict: ...
    def step(self, action: Array) -> tuple[dict, float, bool, dict]: ...
    def task_state(self) -> Array: ...
    def get_state(self) -> Array: ...
    def set_state(self, state: Array) -> None: ...
    def supports_reset_to_state(self) -> bool: ...


# --------------------------------------------------------------------------- #
# Records written to disk
# --------------------------------------------------------------------------- #
@dataclass
class Anchor:
    """One calibration/evaluation anchor: a state, its features, its realised errors."""

    episode_id: int
    t: int
    phi: Array          # (d_phi,) deployment-available features
    d: Array            # (H_max,) realised task-space errors d_h, Eq. (2)
    contact: bool = False
    appearance_percentile: float = 0.0   # pi_app, for SFR (Definition 3.2)
    meta: dict = field(default_factory=dict)


@dataclass
class CalibrationResult:
    """Output of Algorithm G.1."""

    lambda_hat: float
    alpha: float
    n_cal: int
    tau: float
    h_max: int
    scores: Array
    planner_conditional: bool = False

    def as_dict(self) -> dict:
        d = asdict(self)
        d.pop("scores")
        d["score_mean"] = float(np.mean(self.scores))
        d["score_p95"] = float(np.quantile(self.scores, 0.95))
        return d
