"""Numerical checks of the paper's propositions.

Each test states the claim it is checking and fails if the implementation
violates it. These are not proofs; they are falsification attempts, which is
the only thing a test can be.
"""

from __future__ import annotations

import numpy as np
import pytest

from halt.aci import ACIController, StaticController, prop_52_bound
from halt.conformal import (
    bonferroni_calibrate,
    bonferroni_certified_horizon,
    calibrate,
    certified_horizon,
    certified_horizon_batch,
    conformal_quantile,
    conformal_scores,
    simultaneous_coverage,
)
from halt.theory import (
    SimulationLemmaParams,
    optimal_horizon,
    prop_41_coverage_bound,
    prop_42_coverage_bound,
    prop_51_bound,
)


# --------------------------------------------------------------------------- #
# Synthetic generator: exchangeable anchors with a monotone quantile curve
# --------------------------------------------------------------------------- #
def make_anchors(n: int, h_max: int = 30, seed: int = 0, excess: float = 1.0):
    """d_h = base_h * lognormal(state) * excess, q_hat = the head's prediction.

    Exchangeable by construction: every anchor is an i.i.d. draw. That is the
    regime Prop. 4.1 assumes, so coverage must land at 1-alpha here. Anywhere
    it does not, the conformal code is wrong.

    `excess` inflates the REALISED error only, leaving q_hat where a head
    fitted on behaviour-policy data would have put it. That asymmetry is the
    whole content of planner-induced shift: the head is stale, not retrained,
    and the scores R = d/q_hat rise. Scaling both d and q_hat together would
    leave every score identical and model nothing.
    """
    rng = np.random.default_rng(seed)
    base = np.cumsum(rng.uniform(0.02, 0.03, size=h_max))       # increasing in h
    sigma = 0.45
    s = rng.lognormal(0.0, sigma, size=n)                        # per-anchor difficulty
    noise = rng.lognormal(0.0, 0.30, size=(n, h_max))
    mean_curve = s[:, None] * base[None, :]
    d = excess * mean_curve * noise
    # The head predicts the 90th percentile of the UNSHIFTED error process.
    q = np.exp(np.log(mean_curve) + 0.30 * 1.2816)
    return d, q


# --------------------------------------------------------------------------- #
# Proposition 4.1
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("alpha", [0.01, 0.05, 0.10, 0.20])
def test_prop_41_marginal_coverage(alpha):
    """P[R_test <= lambda_hat] >= 1 - alpha under exchangeability."""
    d, q = make_anchors(4000, seed=1)
    d_cal, q_cal, d_te, q_te = d[:2000], q[:2000], d[2000:], q[2000:]

    cal = calibrate(d_cal, q_cal, alpha, tau=1.0)
    cov = float(np.mean(conformal_scores(d_te, q_te) <= cal.lambda_hat))

    target = prop_41_coverage_bound(alpha)
    # Finite-sample slack: the conformal guarantee is >= 1-alpha in
    # expectation over calibration sets; a single split fluctuates by O(1/sqrt n).
    assert cov >= target - 0.02, f"alpha={alpha}: coverage {cov:.4f} < {target:.4f}"
    assert cov <= target + 0.05, f"alpha={alpha}: coverage {cov:.4f} is far above target"


def test_prop_41_simultaneity_over_horizons():
    """The claim is simultaneous over h, not per-horizon.

    This is the structural point of Eq. (8). On the certificate event, the
    bound holds for EVERY h <= H* at once -- so the fraction of test anchors
    where any horizon inside the certified prefix exceeds tau must be <= alpha.
    """
    alpha, tau = 0.05, 0.6
    d, q = make_anchors(6000, seed=2)
    cal = calibrate(d[:3000], q[:3000], alpha, tau=tau)
    out = simultaneous_coverage(d[3000:], q[3000:], cal.lambda_hat, tau)

    assert out["n_certified"] > 100, "test is vacuous if almost nothing is certified"
    assert out["coverage"] >= 1 - alpha - 0.02, (
        f"simultaneous coverage {out['coverage']:.4f} below 1-alpha={1-alpha}"
    )


def test_conformal_quantile_is_the_finite_sample_one():
    """ceil((n+1)(1-alpha))/n, not the plain empirical quantile."""
    scores = np.arange(1.0, 101.0)            # n = 100
    lam = conformal_quantile(scores, 0.05)
    k = int(np.ceil(101 * 0.95))              # 96
    assert lam == float(scores[k - 1]) == 96.0
    # And it is strictly more conservative than numpy's quantile here.
    assert lam >= float(np.quantile(scores, 0.95))


def test_conformal_quantile_returns_inf_when_uninformative():
    """With too few calibration points for the level, the honest answer is inf."""
    assert conformal_quantile(np.arange(5.0), 0.01) == float("inf")
    # and H* then collapses to 0 rather than silently certifying something
    assert certified_horizon(np.linspace(0.1, 1.0, 10), float("inf"), tau=10.0) == 0


# --------------------------------------------------------------------------- #
# Definition 4.1 and the monotone head
# --------------------------------------------------------------------------- #
def test_certified_horizon_is_a_prefix():
    q = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    assert certified_horizon(q, lambda_hat=1.0, tau=0.35) == 3
    assert certified_horizon(q, lambda_hat=1.0, tau=0.05) == 0
    assert certified_horizon(q, lambda_hat=1.0, tau=99.0) == 5


def test_certified_horizon_rejects_non_monotone_input():
    """Eq. (6) guarantees monotonicity; a non-monotone curve is a caller bug."""
    with pytest.raises(ValueError, match="non-decreasing"):
        certified_horizon(np.array([0.1, 0.3, 0.2]), 1.0, 1.0)


def test_certified_horizon_batch_matches_scalar():
    rng = np.random.default_rng(3)
    q = np.cumsum(rng.uniform(0.01, 0.05, size=(200, 25)), axis=1)
    batch = certified_horizon_batch(q, 1.3, 0.4)
    scalar = np.array([certified_horizon(row, 1.3, 0.4) for row in q])
    np.testing.assert_array_equal(batch, scalar)


def test_horizon_is_monotone_in_tau_and_in_alpha():
    """Looser tolerance certifies further; lower alpha certifies less far.

    The second is the paper's 'honest price of confidence': driving alpha down
    shrinks the Prop. 5.1 confidence term linearly, at the cost of a shorter
    H* and more frequent fallback.
    """
    d, q = make_anchors(3000, seed=4)
    q_probe = q[0]

    taus = [0.1, 0.3, 0.6, 1.2]
    cal = calibrate(d[:1500], q[:1500], 0.05, tau=1.0)
    hs = [certified_horizon(q_probe, cal.lambda_hat, t) for t in taus]
    assert hs == sorted(hs), f"H* not non-decreasing in tau: {hs}"

    alphas = [0.20, 0.10, 0.05, 0.01]
    hs_a = [
        certified_horizon(q_probe, calibrate(d[:1500], q[:1500], a, tau=0.5).lambda_hat, 0.5)
        for a in alphas
    ]
    assert hs_a == sorted(hs_a, reverse=True), f"H* not non-increasing as alpha falls: {hs_a}"


# --------------------------------------------------------------------------- #
# Eq. (8) versus the Bonferroni baseline
# --------------------------------------------------------------------------- #
def test_max_score_beats_bonferroni_on_horizon_at_equal_coverage():
    """The union bound costs horizon; the max-score does not.

    Both constructions give simultaneous coverage. The claim being checked is
    that calibrating one scale-free max-score certifies at least as far as
    correcting each horizon at alpha/H_max -- which is why §4.3 does it that way.
    """
    alpha, tau, h_max = 0.05, 0.5, 30
    d, q = make_anchors(5000, h_max=h_max, seed=5)
    d_cal, q_cal, d_te, q_te = d[:2500], q[:2500], d[2500:], q[2500:]

    cal = calibrate(d_cal, q_cal, alpha, tau=tau, h_max=h_max)
    halt_h = certified_horizon_batch(q_te, cal.lambda_hat, tau)

    lam_bonf = bonferroni_calibrate(d_cal, q_cal, alpha, h_max)
    bonf_h = np.array([bonferroni_certified_horizon(row, lam_bonf, tau) for row in q_te])

    assert halt_h.mean() >= bonf_h.mean(), (
        f"max-score certified {halt_h.mean():.2f} steps, Bonferroni "
        f"{bonf_h.mean():.2f}: the union bound should not win"
    )
    out = simultaneous_coverage(d_te, q_te, cal.lambda_hat, tau)
    assert out["coverage"] >= 1 - alpha - 0.02


# --------------------------------------------------------------------------- #
# Proposition 4.2
# --------------------------------------------------------------------------- #
def test_prop_42_correction_is_the_finite_sample_term():
    assert prop_42_coverage_bound(0.05, 999) == pytest.approx(0.95 - 1 / 1000)
    # the correction vanishes as n_val grows, and is never free
    assert prop_42_coverage_bound(0.05, 10) < prop_42_coverage_bound(0.05, 1000)


def test_planner_shift_breaks_coverage_and_r1_restores_it():
    """§4.4, measured rather than asserted.

    A planner selects actions the model is wrong about, so its score
    distribution is shifted upward relative to a behaviour policy. Calibrating
    on behaviour scores then under-covers at deployment; recalibrating on
    planner-induced scores (R1) restores coverage and shortens the horizon.
    """
    alpha, tau = 0.05, 0.5
    d_beh, q_beh = make_anchors(3000, seed=6, excess=1.0)
    d_pln, q_pln = make_anchors(3000, seed=7, excess=1.55)   # the adversarial tilt

    cal_beh = calibrate(d_beh, q_beh, alpha, tau=tau)
    under = simultaneous_coverage(d_pln, q_pln, cal_beh.lambda_hat, tau)

    cal_r1 = calibrate(d_pln[:1500], q_pln[:1500], alpha, tau=tau, planner_conditional=True)
    fixed = simultaneous_coverage(d_pln[1500:], q_pln[1500:], cal_r1.lambda_hat, tau)

    assert under["coverage"] < 1 - alpha, (
        "behaviour-policy calibration should under-cover under planner shift; "
        f"got {under['coverage']:.4f}"
    )
    assert fixed["coverage"] >= 1 - alpha - 0.02, (
        f"(R1) failed to restore coverage: {fixed['coverage']:.4f}"
    )
    assert cal_r1.lambda_hat > cal_beh.lambda_hat, "R1 should inflate lambda_hat"
    assert fixed["mean_h_star"] < under["mean_h_star"], (
        "R1 should shorten the certified horizon -- that is its cost"
    )


# --------------------------------------------------------------------------- #
# Proposition 5.1
# --------------------------------------------------------------------------- #
def test_prop_51_optimal_horizon_is_state_dependent():
    """Reading (2): the horizon that minimises the bound varies with the state.

    A subtlety worth recording, because it is easy to over-read the paper's
    phrasing. Both H-dependent terms of Eq. (13) carry the same gamma^H
    factor:

        d/dH [ L_r tau (1-g^H)/(1-g) + g^H eps_V ]
            = g^H |log g| ( L_r tau / (1-g) - eps_V )

    The sign of that derivative does not depend on H, so the minimiser of
    *this* bound is at a boundary, switching at L_r tau / (1-gamma) = eps_V.
    The interior optimum one might expect comes from the classical simulation
    lemma, where the model-error term compounds with H instead of saturating
    at the certified tolerance -- replacing that compounding with a constant
    tau is exactly what the certificate buys.

    The paper's actual claim survives intact and is what we test: the optimal
    horizon is a function of tau and eps_V, both of which vary from state to
    state, so no fixed hyperparameter can track it.
    """
    gamma, L_r, h_max = 0.95, 1.0, 60
    eps_V = 0.8
    switch = eps_V * (1 - gamma) / L_r          # tau at which the regime flips

    p = SimulationLemmaParams(L_r=L_r, R_max=1.0, eps_V=eps_V, gamma=gamma)
    h_tight = optimal_horizon(p, tau=switch * 0.25, alpha=0.05, h_max=h_max)
    h_loose = optimal_horizon(p, tau=switch * 4.00, alpha=0.05, h_max=h_max)

    assert h_tight > h_loose, (
        f"a tighter tolerance must not certify a shorter horizon: "
        f"{h_tight} vs {h_loose}"
    )

    # Same tau, different value-function quality: a better V_nu makes early
    # truncation cheaper, so the optimum shortens. State-dependence again.
    tau = switch
    worse_v = SimulationLemmaParams(L_r=L_r, R_max=1.0, eps_V=eps_V * 4, gamma=gamma)
    better_v = SimulationLemmaParams(L_r=L_r, R_max=1.0, eps_V=eps_V / 4, gamma=gamma)
    assert optimal_horizon(worse_v, tau, 0.05, h_max) > optimal_horizon(better_v, tau, 0.05, h_max)


def test_prop_51_regime_switch_is_where_the_algebra_says():
    """The boundary sits at L_r tau / (1-gamma) = eps_V, to within one step."""
    gamma, L_r, eps_V, h_max = 0.95, 1.0, 0.5, 80
    p = SimulationLemmaParams(L_r=L_r, R_max=1.0, eps_V=eps_V, gamma=gamma)
    switch = eps_V * (1 - gamma) / L_r
    assert optimal_horizon(p, switch * 0.9, 0.05, h_max) == h_max
    assert optimal_horizon(p, switch * 1.1, 0.05, h_max) == 0


def test_prop_51_confidence_term_is_linear_in_alpha():
    """Reading (3): driving alpha down shrinks the third term linearly."""
    p = SimulationLemmaParams(L_r=1.0, R_max=2.0, eps_V=0.1, gamma=0.9)
    b1 = prop_51_bound(p, tau=0.1, alpha=0.10, h_star=10)
    b2 = prop_51_bound(p, tau=0.1, alpha=0.05, h_star=10)
    delta = b1 - b2
    expected = 2.0 * (0.10 - 0.05) * p.R_max / (1 - p.gamma)
    assert delta == pytest.approx(expected, rel=1e-9)


def test_prop_51_bound_dominates_a_measured_value_gap():
    """The inequality must actually hold on simulated returns."""
    rng = np.random.default_rng(8)
    p = SimulationLemmaParams(L_r=1.0, R_max=1.0, eps_V=0.05, gamma=0.9)
    tau, alpha, H = 0.08, 0.05, 12

    worst = 0.0
    for _ in range(400):
        # Model error inside the certified region respects tau (the certificate
        # event); outside it the value function takes over.
        err = rng.uniform(0, tau, size=H)
        gap = sum((p.gamma**h) * p.L_r * err[h] for h in range(H)) + (p.gamma**H) * p.eps_V
        worst = max(worst, gap)
    assert worst <= prop_51_bound(p, tau, alpha, H) + 1e-9


# --------------------------------------------------------------------------- #
# Proposition 5.2
# --------------------------------------------------------------------------- #
def test_prop_52_bound_shape():
    """The L_max term is the entire price of delay, and it is linear in L_max."""
    b0 = prop_52_bound(0.05, 0.05, 0, 1000)
    b10 = prop_52_bound(0.05, 0.05, 10, 1000)
    assert b10 > b0
    assert (b10 - b0) == pytest.approx(0.05 * 10 / (0.05 * 1000), rel=1e-9)
    # and it vanishes as T grows: the bound is asymptotic, which is exactly
    # the caveat the paper attaches to it.
    assert prop_52_bound(0.05, 0.05, 10, 10**6) < 1e-3


@pytest.mark.parametrize("l_max", [0, 5, 20])
def test_aci_respects_prop_52_under_shift(l_max):
    """ACI drives the long-run miss rate to alpha, within the stated bound."""
    rng = np.random.default_rng(9 + l_max)
    alpha, eta, T = 0.05, 0.05, 6000
    ctl = ACIController(alpha, eta=eta, l_max=l_max)

    for t in range(T):
        # Shift at T/3: the realised miss propensity jumps to 6x the level.
        infl = 1.0 if t < T // 3 else 6.0
        p = min(0.98, ctl.level() * infl)
        ctl.launch(int(rng.random() < p), int(rng.integers(0, l_max + 1)))
        ctl.step()

    s = ctl.summary()
    # The identity the proof telescopes must hold exactly. If it does not,
    # something has interfered with the recursion and the bound below is
    # meaningless regardless of whether it happens to be satisfied.
    assert s["telescoping_residual"] < 1e-9, (
        f"the ACI recursion was not applied exactly: residual "
        f"{s['telescoping_residual']:.3e}"
    )
    assert s["bound_respected"], (
        f"deviation {s['deviation']:.4f} exceeds the Prop. 5.2 bound "
        f"{s['prop_5_2_bound']:.4f} at L_max={l_max}"
    )
    assert abs(s["miss_rate"] - alpha) < 0.03, (
        f"ACI did not track the target under shift: miss rate {s['miss_rate']:.4f}"
    )


def test_static_conformal_fails_under_the_same_shift():
    """The comparison that makes §4.6 worth having.

    Static conformal has no mechanism to recover: its level is fixed, so when
    the realised miss propensity jumps it simply misses more, forever.
    """
    rng = np.random.default_rng(11)
    alpha, T = 0.05, 6000
    ctl = StaticController(alpha)
    for t in range(T):
        infl = 1.0 if t < T // 3 else 6.0
        ctl.observe(int(rng.random() < min(0.98, alpha * infl)))
    assert ctl.trace.miss_rate > 3 * alpha, (
        "static conformal should degrade badly under this shift; if it does "
        "not, the shift is too weak for the comparison to be informative"
    )


def test_aci_rejects_a_lag_exceeding_l_max():
    """Prop. 5.2 assumes L_t <= L_max a.s.; a violation must surface."""
    ctl = ACIController(0.05, eta=0.05, l_max=3)
    with pytest.raises(ValueError, match="exceeds L_max"):
        ctl.launch(1, lag=4)
