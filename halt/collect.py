"""Anchor collection -- Algorithm G.1, lines 1-4.

For every sampled anchor t in every episode: encode z_t, roll out K imagined
trajectories of length H_max, build phi_t from deployment-available quantities
only, and record the realised d_h against the ground truth the episode
actually produced.

The ground truth is recorded by replaying the *same* action sequence in the
simulator from a saved state. That is what makes d_h the error of the
imagination rather than the error of a different trajectory, and it is why
reset-to-state is a hard requirement of the harness rather than a convenience.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .features import FeatureExtractor
from .metrics import task_error
from .types import Anchor

Array = np.ndarray

__all__ = ["CollectConfig", "collect_anchors", "anchors_to_arrays"]


@dataclass
class CollectConfig:
    n_episodes: int = 240
    h_max: int = 50
    anchors_per_episode: int = 10
    n_rollouts: int = 5
    jacobian_every: int = 8        # J* is expensive: 2*d_a resets per probe
    memory_m: int = 5
    seed: int = 0
    policy: str = "behaviour"      # "behaviour" | "random"
    max_episode_steps: int = 100


def collect_anchors(
    env,
    model,
    extractor: FeatureExtractor,
    cfg: CollectConfig,
    stress=None,
    verbose: bool = True,
) -> tuple[list[Anchor], dict]:
    """Returns the anchors plus a diagnostics dict (grounding probes, contact rate)."""
    rng = np.random.default_rng(cfg.seed)
    anchors: list[Anchor] = []
    grounding: list[dict] = []
    n_contact = 0
    n_steps = 0

    for ep in range(cfg.n_episodes):
        obs = env.reset(rng)
        # The exact simulator state is cached alongside every observation, so
        # ground truth can be replayed from any anchor without re-simulating
        # the episode prefix. This is the only place the harness needs
        # reset-to-state during collection; a MuJoCo backend caches
        # `env.get_state()` here in exactly the same way.
        obs["_state_cache"] = env.get_state()
        traj_obs, traj_act, traj_info = [obs], [], []
        residuals: list[float] = []

        # ---- run one episode under the behaviour policy ------------------- #
        for t in range(cfg.max_episode_steps):
            a = (
                env.behaviour_action()
                if cfg.policy == "behaviour"
                else rng.uniform(-1, 1, size=env.action_dim)
            )
            obs, _, done, info = env.step(a)
            obs["_state_cache"] = env.get_state()
            traj_act.append(a)
            traj_obs.append(obs)
            traj_info.append(info)
            n_steps += 1
            n_contact += int(info.get("contact", False))
            if done:
                break

        T = len(traj_act)
        if T < cfg.h_max + 2:
            continue

        # ---- sample anchors, far enough from the end to have H_max of truth -- #
        valid = np.arange(1, T - cfg.h_max)
        if valid.size == 0:
            continue
        picks = rng.choice(valid, size=min(cfg.anchors_per_episode, valid.size), replace=False)

        for t in sorted(picks):
            z = model.encode(traj_obs[: t + 1], traj_act[:t])
            actions = np.asarray(traj_act[t : t + cfg.h_max])

            # ---- ground truth by replaying the same actions from this state -- #
            # Saved before imagination so the replay is exact.
            saved = env.get_state()
            env.set_state(traj_obs[t]["_state_cache"])
            y_true = np.zeros((cfg.h_max, env.task_spec.dim))
            for h in range(cfg.h_max):
                env.step(actions[h])
                y_true[h] = env.task_state()
            env.set_state(saved)

            # ---- imagination -------------------------------------------------- #
            y_hat = model.rollout(z, actions, n_rollouts=cfg.n_rollouts, rng=rng)
            y_mean = y_hat.mean(axis=0)
            d = task_error(y_true, y_mean, env.task_spec)       # (H_max,)

            # ---- features: model internals and sensors only, never y ---------- #
            J_hat = (
                model.jacobian(z, actions, extractor.cfg.jacobian_horizons)
                if model.capabilities.differentiable_readout
                else None
            )
            phi = extractor.extract(
                y_rollouts=y_hat,
                jacobians=J_hat,
                latent_novelty=model.latent_novelty(z) if hasattr(model, "latent_novelty") else 0.0,
                tactile_residual=float(traj_obs[t].get("tactile", 0.0)),
                contact_onset_prob=float(traj_info[t - 1].get("contact", False)) if t > 0 else 0.0,
                error_memory=np.asarray(residuals[-cfg.memory_m :]) if residuals else None,
            )

            # pi_app: a stand-in for an appearance percentile. Here it is the
            # one-step prediction quality, which is what an appearance metric
            # effectively rewards -- locally plausible continuation -- and is
            # deliberately NOT a function of the long-horizon task error. SFR
            # then measures the gap between the two, as Definition 3.2 intends.
            one_step = model.one_step_prediction(z, actions[0])
            app = float(np.exp(-task_error(y_true[0], one_step, env.task_spec) / 3.0))

            anchors.append(
                Anchor(
                    episode_id=ep,
                    t=int(t),
                    phi=phi,
                    d=d,
                    contact=bool(traj_info[t - 1].get("contact", False)) if t > 0 else False,
                    appearance_percentile=app,
                    meta={"stress": None if stress is None else stress.axis,
                          "severity": 0 if stress is None else stress.severity},
                )
            )
            residuals.append(float(d[0]))

            # ---- grounding probe: CAS and rho need J*, so subsample ----------- #
            if J_hat is not None and len(anchors) % cfg.jacobian_every == 0:
                from .jacobian import finite_difference_jacobian
                from .metrics import cas, rho

                try:
                    J_true = finite_difference_jacobian(
                        env, traj_obs[t]["_state_cache"], actions,
                        extractor.cfg.jacobian_horizons,
                    )
                    env.set_state(saved)
                    for k, h in enumerate(extractor.cfg.jacobian_horizons):
                        grounding.append({
                            "episode": ep, "t": int(t), "h": h,
                            "cas": cas(J_hat[k], J_true[k]),
                            "rho": rho(J_hat[k], J_true[k]),
                            "jhat_norm": float(np.linalg.norm(J_hat[k])),
                            "jtrue_norm": float(np.linalg.norm(J_true[k])),
                        })
                except RuntimeError:
                    pass   # environment without reset-to-state: skip grounding

        if verbose and (ep + 1) % 50 == 0:
            print(f"  episode {ep + 1}/{cfg.n_episodes}, {len(anchors)} anchors")

    diagnostics = {
        "n_anchors": len(anchors),
        "n_episodes": cfg.n_episodes,
        "contact_rate": n_contact / max(n_steps, 1),
        "grounding": grounding,
    }
    return anchors, diagnostics


def anchors_to_arrays(anchors: list[Anchor]) -> dict[str, Array]:
    """Stack anchors into the arrays the rest of the pipeline consumes."""
    if not anchors:
        raise ValueError("no anchors collected")
    return {
        "phi": np.stack([a.phi for a in anchors]),
        "d": np.stack([a.d for a in anchors]),
        "episode_id": np.array([a.episode_id for a in anchors]),
        "t": np.array([a.t for a in anchors]),
        "contact": np.array([a.contact for a in anchors]),
        "appearance": np.array([a.appearance_percentile for a in anchors]),
    }
