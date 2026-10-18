"""End-to-end integration: collect -> calibrate -> certify -> deploy.

Tiny by design (it runs in well under a minute), because its job is to catch
the integration bugs unit tests miss -- a feature-dimension drift, a split
leak, a checkpoint that does not round-trip, a deployment loop that silently
never abstains -- not to produce results.
"""

from __future__ import annotations

import numpy as np
import pytest

from halt.calibrate import calibrate_tolerance, run_algorithm_g1
from halt.collect import CollectConfig, anchors_to_arrays, collect_anchors
from halt.conformal import certified_horizon_batch, simultaneous_coverage
from halt.deploy import DeployConfig, run_deployment
from halt.envs import PlanarPush, PlanarPushParams, TASK_SPEC, episode_level_split
from halt.features import FeatureConfig, FeatureExtractor
from halt.models import EnsembleConfig, EnsembleWorldModel
from halt.nn.head import HeadConfig
from halt.planner import PlannerConfig

H_MAX = 12


@pytest.fixture(scope="module")
def fitted():
    """A small trained model plus anchors, shared across the tests below."""
    env = PlanarPush(PlanarPushParams(max_steps=60))
    rng = np.random.default_rng(0)

    Y, C, A, YN = [], [], [], []
    for _ in range(60):
        obs = env.reset(rng)
        for _ in range(60):
            y = env.task_state()
            a = env.behaviour_action()
            obs, _, done, _ = env.step(a)
            Y.append(y); C.append(obs["goal"]); A.append(a); YN.append(env.task_state())
            if done:
                break

    model = EnsembleWorldModel(
        EnsembleConfig(d_y=TASK_SPEC.dim, d_a=2, d_context=2, K=3,
                       width=64, epochs=6, seed=0),
        TASK_SPEC,
    )
    model.fit(np.array(Y), np.array(C), np.array(A), np.array(YN), verbose=False)

    extractor = FeatureExtractor(
        TASK_SPEC, model.capabilities,
        FeatureConfig(h_probe=(1, 5, 10), jacobian_horizons=(1,), memory_m=3),
    )
    anchors, diag = collect_anchors(
        env, model, extractor,
        CollectConfig(n_episodes=60, h_max=H_MAX, anchors_per_episode=5,
                      n_rollouts=3, seed=1, jacobian_every=10_000,
                      max_episode_steps=60),
        verbose=False,
    )
    return env, model, extractor, anchors_to_arrays(anchors), diag


def test_collection_yields_usable_anchors(fitted):
    _, _, extractor, arr, diag = fitted
    assert arr["phi"].shape[0] >= 100, "too few anchors for the rest of the suite"
    assert arr["phi"].shape[1] == extractor.dim
    assert arr["d"].shape[1] == H_MAX
    assert np.all(np.isfinite(arr["phi"])) and np.all(np.isfinite(arr["d"]))
    assert diag["contact_rate"] > 0.02, "the suite must actually produce contact"

    # No runaway rollouts: a mean far above the median at the longest horizon
    # means a few anchors exploded and would dominate the conformal score.
    tail = arr["d"][:, -1]
    assert tail.mean() / max(np.median(tail), 1e-9) < 3.0, (
        f"d_H mean/median = {tail.mean() / np.median(tail):.1f}; "
        "autoregressive rollouts are diverging"
    )


def test_error_grows_monotonically_with_horizon(fitted):
    """Not guaranteed per-anchor, but the median curve must increase."""
    _, _, _, arr, _ = fitted
    med = np.median(arr["d"], axis=0)
    assert np.all(np.diff(med) > -1e-9), f"median d_h is not increasing: {med}"
    assert med[-1] > 2 * med[0], "error should compound meaningfully over the horizon"


def test_full_calibration_and_coverage(fitted, tmp_path):
    _, _, extractor, arr, _ = fitted
    phi, d, ep = arr["phi"], arr["d"], arr["episode_id"]
    m = episode_level_split(ep, (0.6, 0.2, 0.2), seed=0)

    tau = calibrate_tolerance(d[m["train"]], h_a=5, h_b=10)
    assert tau > 0

    halt = run_algorithm_g1(
        phi[m["train"]], d[m["train"]], phi[m["cal"]], d[m["cal"]],
        alpha=0.10, h_max=H_MAX, feature_names=extractor.names(), tau=tau,
        head_cfg=HeadConfig(d_phi=phi.shape[1], h_max=H_MAX, alpha=0.10,
                            epochs=25, width=64, seed=0),
        verbose=False,
    )

    out = simultaneous_coverage(d[m["test"]], halt.q_hat(phi[m["test"]]),
                                halt.calibration.lambda_hat, tau)
    assert out["n_certified"] > 0, "nothing was certified; the test is vacuous"
    assert out["coverage"] >= 0.80, (
        f"coverage {out['coverage']:.3f} far below the 0.90 target -- with a "
        "small split some slack is expected, this much is a bug"
    )

    # the checkpoint must round-trip exactly
    halt.save(tmp_path / "halt")
    from halt.calibrate import CalibratedHALT

    reloaded = CalibratedHALT.load(tmp_path / "halt")
    np.testing.assert_allclose(reloaded.q_hat(phi[:8]), halt.q_hat(phi[:8]), rtol=1e-6)
    assert reloaded.calibration.lambda_hat == pytest.approx(halt.calibration.lambda_hat)
    np.testing.assert_array_equal(
        reloaded.certified_horizon(phi[:8]), halt.certified_horizon(phi[:8])
    )


def test_deployment_loop_runs_and_abstains_when_it_should(fitted):
    """Algorithm 1 must actually fire the abstention rule at a high H_min."""
    env, model, extractor, arr, _ = fitted
    phi, d, ep = arr["phi"], arr["d"], arr["episode_id"]
    m = episode_level_split(ep, (0.6, 0.2, 0.2), seed=0)
    halt = run_algorithm_g1(
        phi[m["train"]], d[m["train"]], phi[m["cal"]], d[m["cal"]],
        alpha=0.10, h_max=H_MAX, feature_names=extractor.names(),
        tau=calibrate_tolerance(d[m["train"]], 5, 10),
        head_cfg=HeadConfig(d_phi=phi.shape[1], h_max=H_MAX, alpha=0.10,
                            epochs=20, width=64, seed=0),
        verbose=False,
    )
    pcfg = PlannerConfig(horizon_max=H_MAX, n_samples=8, n_iters=1, h_min=3)

    out = run_deployment(
        env, model, extractor, halt,
        DeployConfig(mode="static", n_episodes=2, max_steps=25, seed=0,
                     planner="mppi", h_min=3),
        pcfg, verbose=False,
    )
    assert 0.0 <= out["success_rate"] <= 1.0
    assert out["mean_h_star"] >= 0
    assert out["abstention_rate"] <= 1.0

    # With H_min above H_max the rule must fire on every single step.
    always = run_deployment(
        env, model, extractor, halt,
        DeployConfig(mode="static", n_episodes=1, max_steps=12, seed=0,
                     planner="mppi", h_min=H_MAX + 1),
        pcfg, verbose=False,
    )
    assert always["abstention_rate"] == pytest.approx(1.0), (
        "the abstention rule did not fire with H_min above H_max"
    )


def test_aci_mode_runs_end_to_end(fitted):
    """The ACI block of Algorithm 1 (lines 11-14) with real delayed feedback."""
    env, model, extractor, arr, _ = fitted
    phi, d, ep = arr["phi"], arr["d"], arr["episode_id"]
    m = episode_level_split(ep, (0.6, 0.2, 0.2), seed=0)
    halt = run_algorithm_g1(
        phi[m["train"]], d[m["train"]], phi[m["cal"]], d[m["cal"]],
        alpha=0.10, h_max=H_MAX, feature_names=extractor.names(),
        tau=calibrate_tolerance(d[m["train"]], 5, 10),
        head_cfg=HeadConfig(d_phi=phi.shape[1], h_max=H_MAX, alpha=0.10,
                            epochs=20, width=64, seed=0),
        verbose=False,
    )
    out = run_deployment(
        env, model, extractor, halt,
        DeployConfig(mode="aci", n_episodes=2, max_steps=25, seed=0,
                     planner="mppi", h_min=2, l_max=H_MAX, eta=0.1),
        PlannerConfig(horizon_max=H_MAX, n_samples=8, n_iters=1, h_min=2),
        verbose=False,
    )
    assert "aci_miss_rate" in out
    assert out["aci_telescoping_residual"] < 1e-9, (
        "the ACI recursion was not applied exactly inside the deployment loop"
    )


def test_aci_deadlocks_on_abstention_and_shadow_rollouts_fix_it(fitted):
    """Algorithm 1 couples the ACI feedback loop to the abstention rule.

    Line 9 (push onto Q) is inside the else-branch of the line-5 abstention
    test, so an abstaining controller generates no feedback, alpha_t freezes,
    and it abstains forever. This test pins the failure and the remedy.
    """
    env, model, extractor, arr, _ = fitted
    phi, d, ep = arr["phi"], arr["d"], arr["episode_id"]
    m = episode_level_split(ep, (0.6, 0.2, 0.2), seed=0)
    halt = run_algorithm_g1(
        phi[m["train"]], d[m["train"]], phi[m["cal"]], d[m["cal"]],
        alpha=0.10, h_max=H_MAX, feature_names=extractor.names(),
        tau=calibrate_tolerance(d[m["train"]], 5, 10),
        head_cfg=HeadConfig(d_phi=phi.shape[1], h_max=H_MAX, alpha=0.10,
                            epochs=20, width=64, seed=0),
        verbose=False,
    )
    pcfg = PlannerConfig(horizon_max=H_MAX, n_samples=8, n_iters=1, h_min=H_MAX - 1)

    def run(shadow):
        return run_deployment(
            env, model, extractor, halt,
            DeployConfig(mode="aci", n_episodes=1, max_steps=30, seed=3,
                         planner="mppi", h_min=H_MAX - 1, l_max=H_MAX,
                         eta=0.2, shadow_rollouts=shadow),
            pcfg, verbose=False,
        )

    # H_min just below H_max forces near-total abstention.
    starved = run(False)
    assert starved["abstention_rate"] > 0.8, "set-up failed to induce abstention"
    assert starved["n_feedback"] <= 2, (
        f"expected the feedback stream to be starved, got "
        f"{starved['n_feedback']} items -- if this rises, Algorithm 1 was "
        "changed and this finding needs rechecking"
    )

    fed = run(True)
    assert fed["n_feedback"] > starved["n_feedback"], (
        "shadow rollouts did not restore the feedback stream"
    )
