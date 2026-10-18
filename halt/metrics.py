"""The six action-conditioned metrics of Table B.1, plus SFR and TTCD.

  CAS_h     Eq. (4)         directional action grounding
  rho_h     Eq. (5)         gain grounding: collapse (<<0) vs chaos (>>0)
  H(J_hat)                  spectral entropy, catches rank-collapsed responses
                            that the Frobenius norm alone would miss
  TTCD(tau) Definition 3.1  first step with d_h > tau; usable horizon
  SFR(tau,kappa) Def. 3.2   the appearance-physics gap
  AURC-C                    threshold-free quality of the trust signal
  HCE                       calibration of the horizon claim

Everything here is measured in task space, so every function takes either
pre-normalised d_h values or a TaskStateSpec to normalise with.
"""

from __future__ import annotations

import numpy as np

from .types import TaskStateSpec

Array = np.ndarray

__all__ = [
    "task_error",
    "ttcd",
    "silent_failure_rate",
    "cas",
    "rho",
    "spectral_entropy",
    "aurc_c",
    "horizon_calibration_error",
    "rank_correlation",
    "classify_grounding",
]


# --------------------------------------------------------------------------- #
# Eq. (2) and Definition 3.1
# --------------------------------------------------------------------------- #
def task_error(y_true: Array, y_pred: Array, spec: TaskStateSpec) -> Array:
    """d_h = || y_{t+h} - y_hat_{t+h} ||_Sigma. Reduces the last axis."""
    return spec.weighted_norm(np.asarray(y_true) - np.asarray(y_pred))


def ttcd(d: Array, tau: float, h_max: int | None = None) -> int | Array:
    """TTCD(tau) = min{ h >= 1 : d_h > tau }, with min of empty := H_max.

    Accepts (H,) for a single rollout or (n, H) for a batch.
    Note h is 1-indexed in the paper; d[..., 0] is h = 1.
    """
    d = np.asarray(d, dtype=np.float64)
    single = d.ndim == 1
    if single:
        d = d[None, :]
    H = h_max if h_max is not None else d.shape[1]
    d = d[:, :H]

    exceeds = d > tau
    any_ex = exceeds.any(axis=1)
    first = np.where(any_ex, np.argmax(exceeds, axis=1) + 1, H)
    return int(first[0]) if single else first.astype(int)


def silent_failure_rate(
    d: Array, appearance_percentile: Array, tau: float, kappa: float, h_task: int
) -> float:
    """SFR(tau, kappa) = P[ pi_app >= kappa, TTCD(tau) <= H_task ]  (Definition 3.2).

    A rollout is a silent failure when it looks good -- its appearance
    percentile is at or above kappa -- and has nevertheless already diverged
    in task space within the task horizon. This is a joint probability, not a
    conditional one: it is the fraction of all rollouts that are both
    plausible-looking and wrong.
    """
    d = np.asarray(d, dtype=np.float64)
    pi = np.asarray(appearance_percentile, dtype=np.float64)
    if pi.shape[0] != d.shape[0]:
        raise ValueError(f"{d.shape[0]} rollouts but {pi.shape[0]} appearance scores")
    if h_task >= d.shape[1]:
        # TTCD uses min of empty := H_max, so with h_task >= H_max a rollout
        # that never diverges still satisfies TTCD <= h_task and the metric
        # reports every plausible-looking rollout as a silent failure. That is
        # a degenerate configuration rather than a measurement.
        raise ValueError(
            f"h_task={h_task} must be strictly less than the number of "
            f"horizons ({d.shape[1]}), otherwise SFR counts non-diverging "
            "rollouts as failures"
        )
    t = ttcd(d, tau)
    return float(np.mean((pi >= kappa) & (t <= h_task)))


# --------------------------------------------------------------------------- #
# Eq. (4) and Eq. (5)
# --------------------------------------------------------------------------- #
def cas(j_hat: Array, j_true: Array, eps: float = 1e-12) -> float:
    """CAS_h = 1 - (1/sqrt 2) || J_hat/||J_hat||_F - J*/||J*||_F ||_F,  in [0,1].

    Both Jacobians are normalised to unit Frobenius norm first, so CAS sees
    only direction. The 1/sqrt(2) is what maps the worst case (antiparallel
    unit matrices, distance 2) onto 0 and the best case onto 1.
    """
    a = np.asarray(j_hat, dtype=np.float64)
    b = np.asarray(j_true, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"Jacobian shapes differ: {a.shape} vs {b.shape}")
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < eps or nb < eps:
        return 0.0
    diff = np.linalg.norm(a / na - b / nb)
    return float(np.clip(1.0 - diff / np.sqrt(2.0), 0.0, 1.0))


def rho(j_hat: Array, j_true: Array, eps: float = 1e-12) -> float:
    """rho_h = log( ||J_hat||_F / ||J*||_F ).

    rho << 0 is action collapse: the model has learned an essentially
    unconditional prior and barely responds to the action, so the objective
    landscape a CEM/MPPI optimiser sees is flat and it is fitting noise.
    rho >> 0 with low CAS is action chaos: over-reaction along ungrounded
    directions, fabricating affordances.
    """
    na = float(np.linalg.norm(np.asarray(j_hat, dtype=np.float64)))
    nb = float(np.linalg.norm(np.asarray(j_true, dtype=np.float64)))
    if na < eps or nb < eps:
        return float("nan")
    return float(np.log(na / nb))


def spectral_entropy(j: Array, eps: float = 1e-12) -> float:
    """H(J) = - sum_i s_bar_i log s_bar_i over normalised singular values.

    Low entropy means the action response has collapsed onto a few directions.
    The Frobenius norm cannot see this: a rank-1 Jacobian and a well-conditioned
    one can share a norm exactly.
    """
    s = np.linalg.svd(np.asarray(j, dtype=np.float64), compute_uv=False)
    total = s.sum()
    if total < eps:
        return 0.0
    p = s / total
    p = p[p > eps]
    return float(-np.sum(p * np.log(p)))


def classify_grounding(cas_value: float, rho_value: float,
                       rho_collapse: float = -0.10, rho_chaos: float = 0.10,
                       cas_low: float = 0.70) -> str:
    """Name the pathology, using the thresholds the paper's discussion implies."""
    if rho_value < rho_collapse:
        return "action_collapse"
    if rho_value > rho_chaos and cas_value < cas_low:
        return "action_chaos"
    if cas_value < cas_low:
        return "poorly_grounded"
    return "grounded"


# --------------------------------------------------------------------------- #
# Risk-coverage and calibration
# --------------------------------------------------------------------------- #
def aurc_c(trust_score: Array, loss: Array) -> float:
    """Area under the risk-coverage curve, threshold-free.

    Sort by trust (most trusted first), sweep coverage from 1/n to 1, and
    average the selective risk. Lower is better. This is the quantity that
    makes trust signals comparable without picking a threshold per domain --
    the thing ensemble disagreement and pessimism penalties cannot offer.
    """
    t = np.asarray(trust_score, dtype=np.float64)
    l = np.asarray(loss, dtype=np.float64)
    if t.shape != l.shape:
        raise ValueError(f"trust {t.shape} and loss {l.shape} must match")
    order = np.argsort(-t)          # most trusted first
    risks = np.cumsum(l[order]) / np.arange(1, l.size + 1)
    return float(np.mean(risks))


def horizon_calibration_error(
    h_star: Array, covered: Array, alpha: float, n_bins: int = 10
) -> float:
    """HCE = sum_m (n_m/n) | cov_m - (1-alpha) |, binned by certified horizon.

    A predictor can hit 1-alpha on average while being badly over-confident at
    long horizons and over-conservative at short ones. HCE is what catches that:
    it asks whether the coverage claim holds *at each horizon length*, weighted
    by how often that length is issued.
    """
    h = np.asarray(h_star, dtype=np.float64)
    c = np.asarray(covered, dtype=np.float64)
    if h.size == 0:
        return float("nan")
    lo, hi = h.min(), h.max()
    if hi <= lo:
        return float(abs(c.mean() - (1.0 - alpha)))
    edges = np.linspace(lo, hi + 1e-9, n_bins + 1)
    idx = np.clip(np.digitize(h, edges) - 1, 0, n_bins - 1)
    n = h.size
    out = 0.0
    for m in range(n_bins):
        sel = idx == m
        nm = int(sel.sum())
        if nm:
            out += (nm / n) * abs(c[sel].mean() - (1.0 - alpha))
    return float(out)


def rank_correlation(a: Array, b: Array) -> float:
    """Spearman rho. Used for rho_s(D_h, d_h) and for appearance-vs-TTCD.

    The H1 falsifier is stated in these terms: the hypothesis that silent
    failure is real would have been falsified by a rank correlation above 0.7
    between appearance metrics and TTCD.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size < 2:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = np.sqrt((ra**2).sum() * (rb**2).sum())
    return float(ra @ rb / denom) if denom > 0 else float("nan")
