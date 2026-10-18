"""Component tests: head, features, metrics, splits, environment, interface gating."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from halt.envs import PlanarPush, PlanarPushParams, StressSpec, episode_level_split, make_stressed_env
from halt.features import FeatureConfig, FeatureExtractor
from halt.metrics import (
    aurc_c,
    cas,
    classify_grounding,
    horizon_calibration_error,
    rank_correlation,
    rho,
    silent_failure_rate,
    spectral_entropy,
    task_error,
    ttcd,
)
from halt.nn.head import HeadConfig, MonotoneQuantileHead, pinball_loss, train_head
from halt.types import InterfaceCapabilities, TaskStateSpec


# --------------------------------------------------------------------------- #
# Eq. (6): monotonicity and positivity are architectural, not learned
# --------------------------------------------------------------------------- #
def test_head_is_monotone_and_positive_at_initialisation():
    """Before any training. If this needed training it would not be a guarantee."""
    cfg = HeadConfig(d_phi=12, h_max=40)
    head = MonotoneQuantileHead(cfg)
    q = head.predict(np.random.default_rng(0).normal(size=(64, 12)))
    assert q.shape == (64, 40)
    assert np.all(np.diff(q, axis=1) > 0), "q_hat must be strictly increasing in h"
    assert np.all(q > 0), "Eq. (8) divides by q_hat; it must be strictly positive"


def test_head_stays_monotone_after_training_on_adversarial_targets():
    """Fit against deliberately NON-monotone targets; the head cannot follow."""
    rng = np.random.default_rng(1)
    n, d_phi, h = 512, 6, 20
    phi = rng.normal(size=(n, d_phi))
    d = np.abs(rng.normal(size=(n, h))) * 0.3     # i.i.d. across h, no trend
    cfg = HeadConfig(d_phi=d_phi, h_max=h, epochs=8, width=64)
    head, _ = train_head(phi, d, cfg, verbose=False)
    q = head.predict(phi[:64])
    assert np.all(np.diff(q, axis=1) > 0)


def test_pinball_minimiser_is_the_target_quantile():
    """rho_{1-alpha} is minimised at the (1-alpha)-quantile. Checked directly."""
    rng = np.random.default_rng(2)
    sample = torch.as_tensor(rng.normal(5.0, 2.0, size=20000), dtype=torch.float32)
    alpha = 0.10
    truth = float(np.quantile(sample.numpy(), 1 - alpha))

    grid = torch.linspace(truth - 2.0, truth + 2.0, 201)
    losses = [
        float(pinball_loss(g.repeat(sample.shape[0], 1).T.reshape(-1, 1),
                           sample.reshape(-1, 1), alpha, torch.ones(1)))
        for g in grid
    ]
    best = float(grid[int(np.argmin(losses))])
    assert abs(best - truth) < 0.1, f"pinball minimiser {best:.3f} vs quantile {truth:.3f}"


def test_head_roundtrips_through_a_checkpoint(tmp_path):
    cfg = HeadConfig(d_phi=8, h_max=15, epochs=2, width=32)
    rng = np.random.default_rng(3)
    phi = rng.normal(size=(128, 8))
    d = np.cumsum(np.abs(rng.normal(size=(128, 15))) * 0.1, axis=1)
    head, _ = train_head(phi, d, cfg, verbose=False)
    before = head.predict(phi[:10])

    head.save(tmp_path / "head.pt")
    after = MonotoneQuantileHead.load(tmp_path / "head.pt").predict(phi[:10])
    np.testing.assert_allclose(before, after, rtol=1e-6)


# --------------------------------------------------------------------------- #
# §4.1 features
# --------------------------------------------------------------------------- #
def _spec(d_y=4):
    return TaskStateSpec(names=tuple(f"c{i}" for i in range(d_y)), sigma=np.full(d_y, 0.01))


def _caps(**kw):
    base = dict(stochastic_rollouts=True, differentiable_readout=True,
                task_state_at_calibration=True, task_state_at_deployment=True, n_rollouts=5)
    base.update(kw)
    return InterfaceCapabilities(**base)


def test_feature_dim_is_fixed_regardless_of_which_families_are_gated_off():
    """Disabled families are zero-filled, so one head serves every ablation."""
    spec, cfg = _spec(), FeatureConfig()
    full = FeatureExtractor(spec, _caps(), cfg)
    no_jac = FeatureExtractor(spec, _caps(differentiable_readout=False), cfg)
    no_dis = FeatureExtractor(spec, _caps(stochastic_rollouts=False, n_rollouts=1), cfg)
    assert full.dim == no_jac.dim == no_dis.dim
    assert "jacobian" in no_jac.disabled and "jacobian" not in full.disabled
    assert "disagreement" in no_dis.disabled


def test_disagreement_is_zero_for_identical_rollouts_and_grows_with_spread():
    spec = _spec()
    ex = FeatureExtractor(spec, _caps(), FeatureConfig(h_probe=(1, 5)))
    H, d_y = 10, spec.dim
    same = np.tile(np.linspace(0, 1, H)[None, :, None], (5, 1, d_y))
    phi_same = ex.extract(y_rollouts=same)
    sl = ex.family_slices()["disagreement"]
    assert np.allclose(phi_same[sl], 0.0)

    rng = np.random.default_rng(4)
    spread = same + rng.normal(0, 0.05, size=same.shape)
    assert np.all(ex.extract(y_rollouts=spread)[sl] > 0)


def test_features_never_see_the_task_state():
    """Remark 4.1(a), enforced structurally: `extract` takes no ground truth."""
    import inspect

    params = set(inspect.signature(FeatureExtractor.extract).parameters)
    for forbidden in ("y_true", "task_state", "env", "y"):
        assert forbidden not in params, (
            f"FeatureExtractor.extract exposes {forbidden!r}; features must be "
            "buildable from model internals and sensors alone"
        )


def test_mask_to_keeps_only_the_named_families():
    spec = _spec()
    ex = FeatureExtractor(spec, _caps(), FeatureConfig())
    phi = np.arange(ex.dim, dtype=float) + 1.0
    masked = ex.mask_to(phi, keep=("tactile",))
    sl = ex.family_slices()
    assert np.all(masked[sl["tactile"]] == phi[sl["tactile"]])
    assert np.all(masked[sl["disagreement"]] == 0.0)
    assert np.all(masked[sl["jacobian"]] == 0.0)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def test_task_error_is_dimensionless_and_tau_one_is_the_tolerance_edge():
    spec = TaskStateSpec(names=("x", "y"), sigma=np.array([0.005, 0.01]))
    # An error of exactly one tolerance in a single coordinate gives d = 1.
    assert task_error(np.array([0.005, 0.0]), np.zeros(2), spec) == pytest.approx(1.0)
    assert task_error(np.array([0.0, 0.01]), np.zeros(2), spec) == pytest.approx(1.0)
    # Scaling the physical units without scaling sigma changes nothing dimensionally.
    assert task_error(np.array([0.010, 0.0]), np.zeros(2), spec) == pytest.approx(2.0)


def test_ttcd_definition():
    d = np.array([0.1, 0.2, 0.9, 0.3])
    assert ttcd(d, tau=0.5) == 3                   # 1-indexed, first exceedance
    assert ttcd(d, tau=5.0) == 4                   # min of empty := H_max
    np.testing.assert_array_equal(ttcd(np.stack([d, d * 10]), 0.5), [3, 1])


def test_cas_bounds_and_extremes():
    J = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert cas(J, J) == pytest.approx(1.0)                      # identical direction
    assert cas(J, -J) == pytest.approx(0.0, abs=1e-9)           # antiparallel
    assert 0.0 <= cas(J, np.array([[1.0, 1.0], [0.0, 1.0]])) <= 1.0
    # Scale-invariant: CAS sees direction only.
    assert cas(J, 7.3 * J) == pytest.approx(1.0)


def test_rho_detects_collapse_and_chaos():
    J = np.eye(3)
    assert rho(0.1 * J, J) < 0 and classify_grounding(0.95, rho(0.1 * J, J)) == "action_collapse"
    assert rho(10 * J, J) > 0
    assert classify_grounding(0.4, rho(10 * J, J)) == "action_chaos"
    assert classify_grounding(0.95, 0.0) == "grounded"


def test_spectral_entropy_sees_rank_collapse_that_frobenius_cannot():
    full = np.eye(4)
    collapsed = np.zeros((4, 4))
    collapsed[0, 0] = 2.0                       # same Frobenius norm as eye(4)
    assert np.linalg.norm(full) == pytest.approx(np.linalg.norm(collapsed))
    assert spectral_entropy(collapsed) < spectral_entropy(full)


def test_sfr_is_the_joint_probability_of_looking_good_and_being_wrong():
    d = np.array([[0.1] * 10, [0.1] * 10, [9.0] * 10, [9.0] * 10])
    app = np.array([0.9, 0.1, 0.9, 0.1])
    # only row 2 is both plausible-looking (>= 0.75) and already diverged
    assert silent_failure_rate(d, app, tau=1.0, kappa=0.75, h_task=8) == pytest.approx(0.25)


def test_sfr_refuses_a_degenerate_h_task():
    """With h_task >= H_max every non-diverging rollout counts as a failure."""
    d = np.full((4, 10), 0.1)
    with pytest.raises(ValueError, match="strictly less"):
        silent_failure_rate(d, np.ones(4), tau=1.0, kappa=0.75, h_task=10)


def test_aurc_rewards_a_trust_signal_that_ranks_loss():
    rng = np.random.default_rng(5)
    loss = rng.random(500)
    good = -loss                       # perfectly ranks: high trust <=> low loss
    bad = rng.random(500)              # uninformative
    assert aurc_c(good, loss) < aurc_c(bad, loss)


def test_hce_penalises_horizon_dependent_miscalibration():
    """Average coverage can be right while every bucket is wrong."""
    n = 2000
    half = n // 2
    h = np.concatenate([np.full(half, 2.0), np.full(half, 20.0)])
    # Short horizons cover 1.00, long ones cover 0.90. The mean is exactly
    # 0.95, so marginal coverage looks perfect while both buckets are off by
    # 0.05 -- which is precisely the failure HCE exists to catch.
    covered = np.concatenate([np.ones(half), np.zeros(half)])
    covered[half : half + int(0.90 * half)] = 1.0
    assert covered.mean() == pytest.approx(0.95, abs=1e-9)

    hce = horizon_calibration_error(h, covered, alpha=0.05)
    assert hce == pytest.approx(0.05, abs=1e-6), (
        f"HCE should report the per-bucket error of 0.05, got {hce:.4f}"
    )
    # A genuinely well-calibrated predictor scores near zero on the same data.
    good = np.zeros(n)
    good[: int(0.95 * half)] = 1.0
    good[half : half + int(0.95 * half)] = 1.0
    assert horizon_calibration_error(h, good, alpha=0.05) < 1e-6


def test_rank_correlation_matches_scipy():
    from scipy.stats import spearmanr

    rng = np.random.default_rng(6)
    a, b = rng.normal(size=300), rng.normal(size=300)
    assert rank_correlation(a, b) == pytest.approx(spearmanr(a, b).statistic, abs=1e-9)


# --------------------------------------------------------------------------- #
# Splits and the environment
# --------------------------------------------------------------------------- #
def test_splits_are_episode_level_with_no_leakage():
    ids = np.repeat(np.arange(60), 8)
    m = episode_level_split(ids, (0.6, 0.2, 0.2), seed=0)
    assert m["train"].sum() + m["cal"].sum() + m["test"].sum() == ids.size
    for a, b in (("train", "cal"), ("cal", "test"), ("train", "test")):
        shared = set(ids[m[a]]) & set(ids[m[b]])
        assert not shared, f"episodes {sorted(shared)[:5]} leaked between {a} and {b}"


def test_env_reset_to_state_is_exact():
    """Required for ground-truth replay and for finite-difference Jacobians."""
    env = PlanarPush(PlanarPushParams(process_noise=0.0, obs_noise=0.0))
    rng = np.random.default_rng(7)
    env.reset(rng)
    for _ in range(10):
        env.step(rng.uniform(-1, 1, size=2))

    saved = env.get_state()
    a = rng.uniform(-1, 1, size=(6, 2))
    first = np.array([env.step(x)[0]["y"] for x in a])
    env.set_state(saved)
    second = np.array([env.step(x)[0]["y"] for x in a])
    np.testing.assert_allclose(first, second, atol=1e-12)


def test_env_produces_contact_and_contact_is_where_error_spikes():
    """The phenomenon the paper is about must actually occur in the suite."""
    env = PlanarPush()
    rng = np.random.default_rng(8)
    env.reset(rng)
    contacts = 0
    for _ in range(400):
        _, _, done, info = env.step(env.behaviour_action())
        contacts += int(info["contact"])
        if done:
            env.reset(rng)
    assert contacts > 10, f"only {contacts} contacts in 400 steps; the suite is too easy"


@pytest.mark.parametrize("axis", ["parametric", "contact", "sensing", "material", "composition"])
def test_every_phantasm_axis_constructs_and_runs(axis):
    env = make_stressed_env(StressSpec(axis=axis, severity=2))
    rng = np.random.default_rng(9)
    env.reset(rng)
    for _ in range(30):
        obs, _, done, _ = env.step(env.behaviour_action())
        assert np.all(np.isfinite(obs["y"]))
        if done:
            break


def test_interface_gating_reports_the_table_1_consequences():
    caps = InterfaceCapabilities(False, False, False, False, n_rollouts=1)
    assert set(caps.disabled_feature_families()) == {"disagreement", "jacobian"}
    assert caps.missing_for_certificate()           # no y at calibration => no certificate
    assert not caps.aci_available()

    ok = _caps()
    assert ok.disabled_feature_families() == []
    assert ok.missing_for_certificate() == []
    assert ok.aci_available()


# --------------------------------------------------------------------------- #
# Eq. (3): the autodiff Jacobian must be correct independently of model quality
# --------------------------------------------------------------------------- #
def test_autodiff_jacobian_matches_a_finite_difference_of_the_same_model():
    """Validates J_hat itself, not the model it is taken through.

    This separation matters. A low CAS against the *environment* is a finding
    about the world model -- action collapse -- and the metric is doing its
    job. A disagreement between autodiff and a finite difference of the same
    network would instead be a bug in `torch_rollout_jacobian`. Only the
    second is a defect, so only the second is asserted here.
    """
    from halt.envs import TASK_SPEC
    from halt.models import EnsembleConfig, EnsembleWorldModel

    rng = np.random.default_rng(12)
    model = EnsembleWorldModel(
        EnsembleConfig(d_y=TASK_SPEC.dim, d_a=2, d_context=2, K=2, width=32, epochs=1),
        TASK_SPEC,
    )
    model._ctx = np.zeros(2)
    z = rng.normal(size=TASK_SPEC.dim) * 0.1
    acts = rng.uniform(-1, 1, size=(6, 2))
    horizons = (1, 3)

    j_ad = model.jacobian(z, acts, horizons)

    eps = 1e-4
    j_fd = np.zeros_like(j_ad)

    def roll(a0, h):
        y = np.asarray(z, dtype=np.float64).copy()
        A = acts.copy()
        A[0] = a0
        for k in range(h):
            y, _ = model._step_np(y, A[k], 0)
        return y

    for hi, h in enumerate(horizons):
        for j in range(2):
            e = np.zeros(2)
            e[j] = eps
            j_fd[hi, :, j] = (roll(acts[0] + e, h) - roll(acts[0] - e, h)) / (2 * eps)

    for hi in range(len(horizons)):
        assert cas(j_ad[hi], j_fd[hi]) > 0.99, (
            f"autodiff and finite-difference Jacobians disagree at h="
            f"{horizons[hi]}: CAS {cas(j_ad[hi], j_fd[hi]):.4f}"
        )
        np.testing.assert_allclose(j_ad[hi], j_fd[hi], atol=1e-3)


def test_video_model_drops_the_jacobian_family(tmp_path):
    """Table 9 mechanism: no differentiable readout => no Jacobian features."""
    from halt.envs import TASK_SPEC
    from halt.models import EnsembleConfig, VideoWorldModel

    vm = VideoWorldModel(
        EnsembleConfig(d_y=TASK_SPEC.dim, d_a=2, d_context=2, K=3, width=16, epochs=1),
        TASK_SPEC,
    )
    assert vm.jacobian(np.zeros(TASK_SPEC.dim), np.zeros((3, 2)), (1,)) is None
    assert "jacobian" in vm.capabilities.disabled_feature_families()
    assert "disagreement" not in vm.capabilities.disabled_feature_families()
    assert not vm.capabilities.aci_available()


def test_deploy_refuses_aci_without_deployment_time_ground_truth():
    """Rather than silently degrading to a guarantee that does not hold."""
    from halt.deploy import DeployConfig, run_deployment
    from halt.envs import TASK_SPEC
    from halt.models import EnsembleConfig, VideoWorldModel

    vm = VideoWorldModel(
        EnsembleConfig(d_y=TASK_SPEC.dim, d_a=2, d_context=2, K=2, width=16, epochs=1),
        TASK_SPEC,
    )
    with pytest.raises(RuntimeError, match="task_state_at_deployment=False"):
        run_deployment(None, vm, None, None, DeployConfig(mode="aci"), verbose=False)
