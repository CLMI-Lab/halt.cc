"""Numerically evaluable forms of the paper's propositions.

Each function returns the bound the proposition states, so that a test can
compare a measured quantity against it rather than against a remembered
number. Nothing here is a proof; these are the inequalities the proofs
deliver, written so they can be falsified by a run.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Array = np.ndarray

__all__ = [
    "SimulationLemmaParams",
    "prop_51_bound",
    "optimal_horizon",
    "prop_41_coverage_bound",
    "prop_42_coverage_bound",
    "prop_52_bound",
]

from .aci import prop_52_bound  # re-exported: the delayed-ACI bound lives there


# --------------------------------------------------------------------------- #
# Proposition 5.1 -- truncated-rollout suboptimality
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SimulationLemmaParams:
    """Assumption 5.1: reward Lipschitz in y and bounded, value error, discount."""

    L_r: float        # Lipschitz constant of the reward in the task state
    R_max: float      # |r| <= R_max
    eps_V: float      # error of the learned value at the truncation state
    gamma: float      # discount in (0,1)

    def __post_init__(self) -> None:
        if not 0.0 < self.gamma < 1.0:
            raise ValueError(f"gamma must lie in (0,1), got {self.gamma}")
        for name in ("L_r", "R_max", "eps_V"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")


def prop_51_bound(p: SimulationLemmaParams, tau: float, alpha: float, h_star: int | Array):
    """| V^pi - V_hat^pi |  <=  L_r tau (1-gamma^H)/(1-gamma)
                               + gamma^H eps_V
                               + 2 alpha R_max / (1-gamma)

    Three readings, which are the design argument of the whole method:

    1. The classical simulation lemma pays a *global* model error compounding
       over a fixed horizon. The first term here pays only the *certified local
       tolerance* tau -- that substitution is what the certificate buys.
    2. Terms one and two trade off: the first grows with H, the second decays.
       So an optimal horizon exists, it depends on tau and eps_V, and therefore
       on the state. No fixed hyperparameter can track it.
    3. The third term is the honest price of a probabilistic certificate.
       Driving alpha down shrinks it linearly, at the cost of a shorter H* and
       more frequent abstention.
    """
    H = np.asarray(h_star, dtype=np.float64)
    g = p.gamma
    term_model = p.L_r * tau * (1.0 - g**H) / (1.0 - g)
    term_value = (g**H) * p.eps_V
    term_conf = 2.0 * alpha * p.R_max / (1.0 - g)
    total = term_model + term_value + term_conf
    return total if total.ndim else float(total)


def optimal_horizon(p: SimulationLemmaParams, tau: float, alpha: float, h_max: int) -> int:
    """argmin_H of the Prop. 5.1 bound, by direct enumeration.

    Exists because of reading (2) above. Used in tests to show the optimum is a
    function of tau and eps_V, i.e. state-dependent.
    """
    hs = np.arange(0, h_max + 1)
    return int(hs[np.argmin(prop_51_bound(p, tau, alpha, hs))])


# --------------------------------------------------------------------------- #
# Propositions 4.1 and 4.2 -- coverage targets
# --------------------------------------------------------------------------- #
def prop_41_coverage_bound(alpha: float) -> float:
    """Open-loop: P[...] >= 1 - alpha under exchangeability."""
    return 1.0 - alpha


def prop_42_coverage_bound(alpha: float, n_val: int) -> float:
    """Planner-conditional: >= 1 - alpha - 1/(n_val + 1).

    The extra term is the standard finite-sample split-conformal correction
    with n_val calibration points. The substantive requirement is not this
    term but the one the proof names and the paper repeats: D_plan must be
    representative of deployment, which for sim-to-real is exactly the hard
    part. Under a real shift the guarantee is not preserved and ACI is the
    appropriate fallback.
    """
    if n_val < 1:
        raise ValueError("n_val must be at least 1")
    return 1.0 - alpha - 1.0 / (n_val + 1.0)
