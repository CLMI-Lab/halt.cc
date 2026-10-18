"""Simultaneous split-conformal calibration and the certified trust horizon.

Implements, in order:

  Eq. (8)        R^(i) = max_h  d_h^(i) / q_hat_h(phi^(i))
                 lambda_hat = Quantile_{ceil((n+1)(1-alpha))/n}( {R^(i)} )
  Definition 4.1 H*(phi; tau, alpha) = max{ h <= H_max : lambda_hat q_hat_h(phi) <= tau }
  Prop. 4.1      open-loop validity under exchangeability
  Prop. 4.2      planner-conditional validity, with the 1/(n_val+1) correction
  §4.4 (R2)      the conservative horizon deflation kappa_plan

The structural point the paper makes about Eq. (8) is that taking the maximum
over h inside the score gives *simultaneous* coverage over all horizons at no
extra cost, where a per-horizon construction would need a Bonferroni correction
across H_max horizons and lose a great deal of horizon to it. `bonferroni_
calibrate` below implements that baseline so the loss is measurable rather than
asserted.
"""

from __future__ import annotations

import numpy as np

from .types import CalibrationResult

Array = np.ndarray

__all__ = [
    "conformal_scores",
    "conformal_quantile",
    "calibrate",
    "certified_horizon",
    "certified_horizon_batch",
    "bonferroni_calibrate",
    "bonferroni_certified_horizon",
    "empirical_coverage",
    "simultaneous_coverage",
    "choose_kappa_plan",
]


# --------------------------------------------------------------------------- #
# Eq. (8)
# --------------------------------------------------------------------------- #
def conformal_scores(d: Array, q_hat: Array, h_max: int | None = None) -> Array:
    """The scale-free max-score R^(i) of Eq. (8).

    Parameters
    ----------
    d      : (n, H) realised task-space errors d_h
    q_hat  : (n, H) predicted (1-alpha)-quantiles q_hat_h(phi^(i))
    h_max  : truncate the max to the first h_max horizons (default: all)

    Dividing by q_hat before the max is what makes the score scale-free: a
    horizon where the head already predicts a large error does not dominate
    the maximum simply because its absolute error is large.
    """
    d = np.asarray(d, dtype=np.float64)
    q = np.asarray(q_hat, dtype=np.float64)
    if d.shape != q.shape:
        raise ValueError(f"d {d.shape} and q_hat {q.shape} must match")
    if np.any(q <= 0):
        raise ValueError(
            "q_hat must be strictly positive; the cumulative-softplus head "
            "of Eq. (6) guarantees this, so a non-positive value means the "
            "head was bypassed or clipped"
        )
    if h_max is not None:
        d, q = d[:, :h_max], q[:, :h_max]
    return np.max(d / q, axis=1)


def conformal_quantile(scores: Array, alpha: float) -> float:
    """The finite-sample conformal quantile, ceil((n+1)(1-alpha))/n.

    This is the split-conformal correction, not the plain empirical quantile.
    With n calibration points it returns the k-th order statistic for
    k = ceil((n+1)(1-alpha)); when that exceeds n the set is uninformative and
    we return +inf, which is the honest answer (H* then collapses to 0).
    """
    scores = np.asarray(scores, dtype=np.float64)
    n = scores.shape[0]
    if n == 0:
        raise ValueError("no calibration scores")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must lie in (0,1), got {alpha}")
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    if k > n:
        return float("inf")
    return float(np.sort(scores)[k - 1])


def calibrate(
    d_cal: Array,
    q_cal: Array,
    alpha: float,
    tau: float,
    h_max: int | None = None,
    planner_conditional: bool = False,
) -> CalibrationResult:
    """Algorithm G.1, lines 6-7: scores then the conformal scale lambda_hat."""
    scores = conformal_scores(d_cal, q_cal, h_max=h_max)
    lam = conformal_quantile(scores, alpha)
    return CalibrationResult(
        lambda_hat=lam,
        alpha=float(alpha),
        n_cal=int(scores.shape[0]),
        tau=float(tau),
        h_max=int(h_max if h_max is not None else d_cal.shape[1]),
        scores=scores,
        planner_conditional=bool(planner_conditional),
    )


# --------------------------------------------------------------------------- #
# Definition 4.1
# --------------------------------------------------------------------------- #
def certified_horizon(q_hat: Array, lambda_hat: float, tau: float) -> int:
    """H* = max{ h <= H_max : lambda_hat * q_hat_h <= tau },  max of empty := 0.

    q_hat is non-decreasing by construction (Eq. 6 is a cumulative sum of
    softplus increments), so the set is a prefix and the certified horizon is a
    single threshold crossing. No post-hoc isotonic repair is needed, and none
    is done here: if a caller hands in a non-monotone curve that is a bug in
    the caller, so we assert rather than silently patch it.
    """
    q = np.asarray(q_hat, dtype=np.float64).ravel()
    if q.size == 0:
        return 0
    if np.any(np.diff(q) < -1e-9):
        raise ValueError(
            "q_hat is not non-decreasing in h; Eq. (6) guarantees monotonicity, "
            "so this curve did not come from the monotone head"
        )
    if not np.isfinite(lambda_hat):
        return 0
    ok = lambda_hat * q <= tau
    if not ok[0]:
        return 0
    # prefix length
    idx = np.argmax(~ok) if np.any(~ok) else q.size
    return int(idx)


def certified_horizon_batch(q_hat: Array, lambda_hat: float, tau: float) -> Array:
    """Vectorised Definition 4.1 over a batch of feature vectors. (n, H) -> (n,)."""
    q = np.asarray(q_hat, dtype=np.float64)
    if np.any(np.diff(q, axis=1) < -1e-9):
        raise ValueError("q_hat is not non-decreasing in h for at least one row")
    if not np.isfinite(lambda_hat):
        return np.zeros(q.shape[0], dtype=int)
    ok = lambda_hat * q <= tau
    return ok.cumprod(axis=1).sum(axis=1).astype(int)


# --------------------------------------------------------------------------- #
# Bonferroni baseline (Stankeviciute et al., 2021) -- what Eq. (8) avoids
# --------------------------------------------------------------------------- #
def bonferroni_calibrate(d_cal: Array, q_cal: Array, alpha: float, h_max: int) -> Array:
    """Per-horizon conformal scales at level alpha/H_max.

    This is the union-bound construction HALT replaces. It gives simultaneous
    coverage too, but each horizon is calibrated at alpha/H_max rather than
    alpha, so every scale is inflated and the certified horizon shortens.
    Returns (H_max,) per-horizon lambda values.
    """
    d = np.asarray(d_cal, dtype=np.float64)[:, :h_max]
    q = np.asarray(q_cal, dtype=np.float64)[:, :h_max]
    per_h_alpha = alpha / h_max
    return np.array(
        [conformal_quantile(d[:, h] / q[:, h], per_h_alpha) for h in range(h_max)]
    )


def bonferroni_certified_horizon(q_hat: Array, lambdas: Array, tau: float) -> int:
    """Definition 4.1 with a per-horizon scale."""
    q = np.asarray(q_hat, dtype=np.float64).ravel()[: lambdas.shape[0]]
    ok = lambdas * q <= tau
    if ok.size == 0 or not ok[0]:
        return 0
    return int(np.argmax(~ok)) if np.any(~ok) else int(q.size)


# --------------------------------------------------------------------------- #
# Coverage measurement
# --------------------------------------------------------------------------- #
def simultaneous_coverage(d_test: Array, q_test: Array, lambda_hat: float, tau: float) -> dict:
    """Did the certificate hold, simultaneously, on each test anchor?

    The event of Prop. 4.1 is  for all h <= H*:  d_h <= tau.  Note the bound
    that matters is tau, not lambda_hat*q_hat_h: inside the certified region the
    latter is <= tau by Definition 4.1, so tau is the user-facing claim.

    Anchors with H* = 0 are abstentions. They vacuously satisfy the claim, so
    reporting coverage over all anchors would reward a predictor that abstains
    always. Both figures are returned, and `coverage` -- the paper's quantity --
    is computed over certified anchors only.
    """
    d = np.asarray(d_test, dtype=np.float64)
    q = np.asarray(q_test, dtype=np.float64)
    h_star = certified_horizon_batch(q, lambda_hat, tau)

    n = d.shape[0]
    covered = np.ones(n, dtype=bool)
    for i in range(n):
        H = h_star[i]
        if H > 0:
            covered[i] = bool(np.all(d[i, :H] <= tau))

    certified = h_star > 0
    n_cert = int(certified.sum())
    return {
        "coverage": float(covered[certified].mean()) if n_cert else float("nan"),
        "coverage_including_abstentions": float(covered.mean()),
        "abstention_rate": float((~certified).mean()),
        "mean_h_star": float(h_star.mean()),
        "mean_h_star_certified": float(h_star[certified].mean()) if n_cert else 0.0,
        "n": n,
        "n_certified": n_cert,
        "h_star": h_star,
        "covered": covered,
    }


def empirical_coverage(d_test: Array, q_test: Array, lambda_hat: float) -> float:
    """P[R^(n+1) <= lambda_hat] -- the score-level event the proof uses."""
    scores = conformal_scores(d_test, q_test)
    return float(np.mean(scores <= lambda_hat))


# --------------------------------------------------------------------------- #
# §4.4 remedy (R2): conservative horizon
# --------------------------------------------------------------------------- #
def choose_kappa_plan(
    scores_behaviour: Array,
    scores_planner: Array,
    grid: Array | None = None,
) -> float:
    """Largest kappa in (0,1] at which the behaviour-policy score distribution
    stochastically dominates the planner-induced one after deflation.

    (R2) in §4.4. The paper reports kappa_plan in [0.59, 0.72] and prefers (R1),
    so this exists to be measurable, not to be the recommended path.

    Deflating the horizon by kappa shrinks the max in Eq. (8) to the first
    kappa*H_max horizons; we approximate the resulting score distribution by
    scaling, then ask for first-order stochastic dominance on a quantile grid.
    """
    sb = np.sort(np.asarray(scores_behaviour, dtype=np.float64))
    sp = np.sort(np.asarray(scores_planner, dtype=np.float64))
    if grid is None:
        grid = np.linspace(0.40, 1.00, 61)
    probs = np.linspace(0.01, 0.99, 99)
    qb = np.quantile(sb, probs)
    best = grid[0]
    for kappa in grid:
        qp = np.quantile(sp, probs) * kappa
        if np.all(qb >= qp - 1e-12):
            best = kappa
    return float(best)
