"""Adapters for the three world-model classes of §8.

These are the files you edit to run HALT against a real system. Everything
else in the package is written against the `WorldModel` protocol of
`halt/types.py`, which is exactly the Table 1 interface, so nothing downstream
changes when the model behind it does.

Each adapter is a thin, documented shim with the integration points marked.
They raise a clear ImportError when the upstream package is absent rather than
pretending to work -- a silently stubbed world model would produce a
certificate about nothing, which is the one failure mode this codebase must
not have.
"""

from __future__ import annotations

import numpy as np

from ..types import InterfaceCapabilities, TaskStateSpec

Array = np.ndarray

__all__ = ["DreamerV3Adapter", "TDMPC2Adapter", "MuJoCoTaskSpecs"]


_MISSING = (
    "{pkg} is not installed in this environment. Install it and point this "
    "adapter at your trained checkpoint; see docs/REAL_ENVIRONMENTS.md. "
    "HALT deliberately does not fall back to a stub model, because a "
    "certificate computed against a stub is meaningless."
)


class _TorchLatentAdapter:
    """Shared machinery for latent-imagination models (Dreamer, TD-MPC2).

    Subclasses provide three callables and nothing else:
      _encode(obs_history, act_history) -> latent tensor
      _step(z, a)                       -> next latent tensor   (differentiable)
      _readout(z)                       -> task state tensor    (differentiable)
    """

    def __init__(self, task_spec: TaskStateSpec, action_dim: int, n_rollouts: int = 5,
                 task_state_at_deployment: bool = False):
        self.task_spec = task_spec
        self.action_dim = action_dim
        self.capabilities = InterfaceCapabilities(
            stochastic_rollouts=n_rollouts >= 2,
            differentiable_readout=True,
            task_state_at_calibration=True,
            # Default False: on hardware you do not observe y_{t+h} at
            # deployment, so ACI is unavailable and Static or
            # Static+planner-conditional is the mode you get (Remark 4.2).
            task_state_at_deployment=task_state_at_deployment,
            n_rollouts=n_rollouts,
        )
        self._z_cache = None

    # -- to be provided by the concrete adapter ------------------------------- #
    def _encode(self, obs_history, act_history): raise NotImplementedError
    def _step(self, z, a): raise NotImplementedError
    def _readout(self, z): raise NotImplementedError

    # -- WorldModel protocol ---------------------------------------------------- #
    def encode(self, obs_history, act_history) -> Array:
        z = self._encode(obs_history, act_history)
        self._z_cache = z
        return z.detach().cpu().numpy() if hasattr(z, "detach") else np.asarray(z)

    def rollout(self, z, actions, n_rollouts: int, rng: np.random.Generator) -> Array:
        import torch

        actions_t = torch.as_tensor(np.asarray(actions, dtype=np.float32))
        H = actions_t.shape[0]
        K = max(1, min(n_rollouts, self.capabilities.n_rollouts))
        out = np.zeros((K, H, self.task_spec.dim))
        with torch.no_grad():
            for k in range(K):
                # Seeding per member keeps rollouts reproducible given rng.
                torch.manual_seed(int(rng.integers(0, 2**31 - 1)))
                zz = self._z_cache
                for h in range(H):
                    zz = self._step(zz, actions_t[h])
                    out[k, h] = self._readout(zz).detach().cpu().numpy().ravel()
        return out

    def jacobian(self, z, actions, horizons):
        from ..jacobian import torch_rollout_jacobian
        import torch

        return torch_rollout_jacobian(
            self._step, self._readout, self._z_cache,
            torch.as_tensor(np.asarray(actions, dtype=np.float32)), horizons,
        )


class DreamerV3Adapter(_TorchLatentAdapter):
    """DreamerV3 (Hafner et al., 2023).

    Integration points, in order:

    1. Load your trained agent and keep the RSSM and the decoder frozen. HALT
       never trains the world model; that is the entire premise of wrapping.
    2. `_encode`  -> the RSSM posterior given the observation history.
    3. `_step`    -> `rssm.img_step`, which must stay differentiable. Use the
       reparameterised sample, not a detached one, or `jacobian` returns zeros
       and the Jacobian feature family silently contributes nothing.
    4. `_readout` -> a *task-state* head h_omega. Dreamer ships an image
       decoder; the certificate needs object poses, end-effector pose, contact
       flags. If your agent has no such head, train one on the frozen latents
       (cheap, supervised, offline) -- or declare
       `differentiable_readout=False` and accept the Table-9 coverage cost.
    """

    def __init__(self, checkpoint_path: str, task_spec: TaskStateSpec, action_dim: int, **kw):
        try:
            import dreamerv3  # noqa: F401
        except ImportError as exc:
            raise ImportError(_MISSING.format(pkg="dreamerv3")) from exc
        super().__init__(task_spec, action_dim, **kw)
        self.checkpoint_path = checkpoint_path
        raise NotImplementedError(
            "Fill in _encode / _step / _readout against your DreamerV3 "
            "checkpoint. The four integration points are listed in this "
            "class's docstring."
        )


class TDMPC2Adapter(_TorchLatentAdapter):
    """TD-MPC2 (Hansen et al., 2024).

    Easier than Dreamer in one respect and harder in another. Easier: TD-MPC2
    already plans in latent space with MPPI, so the planner side of Eq. (11)
    maps onto its existing machinery -- you truncate at H* and bootstrap with
    its own learned value, which it already has. Harder: its latent is trained
    for value prediction, not reconstruction, so a task-state readout usually
    has to be added and fitted offline on frozen latents.

    One caution specific to this model. TD-MPC2's planner is part of the agent,
    so the rollouts you collect for calibration are already planner-induced.
    That is remedy (R1) of §4.4 by default, which is good -- but it means the
    behaviour-policy comparison in Table 4 needs a separate data collection
    under a non-planning policy, or the comparison is vacuous.
    """

    def __init__(self, checkpoint_path: str, task_spec: TaskStateSpec, action_dim: int, **kw):
        try:
            import tdmpc2  # noqa: F401
        except ImportError as exc:
            raise ImportError(_MISSING.format(pkg="tdmpc2")) from exc
        super().__init__(task_spec, action_dim, **kw)
        self.checkpoint_path = checkpoint_path
        raise NotImplementedError(
            "Fill in _encode / _step / _readout against your TD-MPC2 "
            "checkpoint; see this class's docstring."
        )


class MuJoCoTaskSpecs:
    """Task-state readouts and per-coordinate tolerances for the five §8 tasks.

    These are the sigma values that make tau = 1 mean 'at the edge of task
    tolerance' for each task. They are engineering choices about the task, not
    statistical ones -- the certificate is valid for whatever sigma and tau you
    supply, and changing them changes what is being certified, not whether the
    guarantee holds.

    The values below are reasonable defaults for each manipulation task; set
    them from your own task requirements before trusting a number.
    """

    @staticmethod
    def pusher() -> TaskStateSpec:
        return TaskStateSpec(
            names=("obj_x", "obj_y", "obj_z", "tip_x", "tip_y", "tip_z"),
            sigma=np.array([0.005, 0.005, 0.005, 0.005, 0.005, 0.005]),
        )

    @staticmethod
    def door() -> TaskStateSpec:
        return TaskStateSpec(
            names=("hinge_angle", "latch_angle", "grip_x", "grip_y", "grip_z", "contact"),
            sigma=np.array([np.deg2rad(2.0), np.deg2rad(2.0), 0.005, 0.005, 0.005, 0.5]),
        )

    @staticmethod
    def pick_place() -> TaskStateSpec:
        return TaskStateSpec(
            names=("obj_x", "obj_y", "obj_z", "obj_qx", "obj_qy", "obj_qz",
                   "ee_x", "ee_y", "ee_z", "gripper"),
            sigma=np.array([0.005, 0.005, 0.005, 0.035, 0.035, 0.035,
                            0.005, 0.005, 0.005, 0.004]),
        )

    @staticmethod
    def stack() -> TaskStateSpec:
        return TaskStateSpec(
            names=("a_x", "a_y", "a_z", "b_x", "b_y", "b_z", "ee_x", "ee_y", "ee_z", "contact"),
            sigma=np.array([0.004, 0.004, 0.004, 0.004, 0.004, 0.004,
                            0.005, 0.005, 0.005, 0.5]),
        )

    @staticmethod
    def peg_insertion() -> TaskStateSpec:
        # The tightest tolerances of the five, which is why it has the shortest
        # certified horizons in the paper: tau is harder to satisfy.
        return TaskStateSpec(
            names=("peg_x", "peg_y", "peg_z", "peg_rx", "peg_ry", "peg_rz", "contact"),
            sigma=np.array([0.002, 0.002, 0.002,
                            np.deg2rad(2.0), np.deg2rad(2.0), np.deg2rad(2.0), 0.5]),
        )

    @classmethod
    def by_name(cls, name: str) -> TaskStateSpec:
        table = {
            "pusher-v5": cls.pusher, "door-v2": cls.door,
            "pickplace-v2": cls.pick_place, "stack-v2": cls.stack,
            "peginsertion-v2": cls.peg_insertion,
        }
        key = name.lower()
        if key not in table:
            raise KeyError(f"unknown task {name!r}; known: {sorted(table)}")
        return table[key]()
