"""An ensemble of K one-step latent models -- the third model class of §8.

This is a real learned world model, trained on transitions collected from the
environment, not an oracle with noise bolted on. That matters: the error curve
d_h that HALT calibrates against has to come from genuine model error that
compounds with the horizon and spikes at contact, or the whole exercise is
circular.

It satisfies the full Table 1 interface:
  * K >= 2 stochastic members              -> the disagreement feature
  * a differentiable task readout          -> the Jacobian features
  * task state available offline           -> the certificate

`VideoWorldModel` at the bottom is the same machinery with the differentiable
readout withheld, standing in for the GAIA-1 case of §9.8: it demonstrates the
coverage cost of dropping the Jacobian family, which is the mechanism Table 9
reports. It does not reproduce GAIA-1's numbers and does not claim to.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ..jacobian import torch_rollout_jacobian
from ..types import InterfaceCapabilities, TaskStateSpec

Array = np.ndarray

__all__ = ["EnsembleConfig", "EnsembleWorldModel", "VideoWorldModel"]


@dataclass
class EnsembleConfig:
    d_y: int
    d_a: int
    d_context: int = 2          # the goal, appended to the state
    K: int = 5
    width: int = 256
    depth: int = 3
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 40
    batch_size: int = 256
    seed: int = 0
    # Member-level output noise, learned as a free parameter per member.
    # Stochastic rollouts are what make D_h meaningful; a deterministic
    # ensemble of identical members would report zero disagreement forever.
    init_log_sigma: float = -4.0

    def as_dict(self) -> dict:
        return asdict(self)


class _Member(nn.Module):
    """One one-step residual model: (y, goal, a) -> delta y."""

    def __init__(self, cfg: EnsembleConfig):
        super().__init__()
        d_in = cfg.d_y + cfg.d_context + cfg.d_a
        layers: list[nn.Module] = []
        d = d_in
        for _ in range(cfg.depth - 1):
            layers += [nn.Linear(d, cfg.width), nn.SiLU()]
            d = cfg.width
        layers += [nn.Linear(d, cfg.d_y)]
        self.net = nn.Sequential(*layers)
        self.log_sigma = nn.Parameter(torch.full((cfg.d_y,), cfg.init_log_sigma))

    def forward(self, y: torch.Tensor, ctx: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return y + self.net(torch.cat([y, ctx, a], dim=-1))


class EnsembleWorldModel:
    """K one-step members, rolled forward autoregressively."""

    def __init__(self, cfg: EnsembleConfig, task_spec: TaskStateSpec):
        self.cfg = cfg
        self.task_spec = task_spec
        self.action_dim = cfg.d_a
        torch.manual_seed(cfg.seed)
        self.members = nn.ModuleList([_Member(cfg) for _ in range(cfg.K)])
        self.capabilities = InterfaceCapabilities(
            stochastic_rollouts=True,
            differentiable_readout=True,
            task_state_at_calibration=True,
            task_state_at_deployment=True,   # simulation; a real robot would say False
            n_rollouts=cfg.K,
        )
        # Standardisation statistics, fitted on the training transitions.
        self._y_mu = np.zeros(cfg.d_y)
        self._y_sd = np.ones(cfg.d_y)
        self._ref_mu = np.zeros(cfg.d_y)
        self._ref_cov_inv = np.eye(cfg.d_y)
        self._ctx = np.zeros(cfg.d_context)
        # Physical bounds on the readout, fitted from the training data and
        # widened. An autoregressive rollout feeds its own output back in, so
        # without a bound a few trajectories leave the training manifold and
        # diverge: before this guard, mean d_30 over anchors was 1.8e4 because
        # a handful of rollouts ran away, while the median stayed at 48. A
        # single exploded anchor would then dominate the conformal score and
        # inflate lambda_hat for everything else. The plant itself clips to the
        # workspace, so clipping the model to the same region is a physical
        # constraint the model should have learned, not a convenience.
        self._y_lo = np.full(cfg.d_y, -np.inf)
        self._y_hi = np.full(cfg.d_y, np.inf)

    # -- training ------------------------------------------------------------ #
    def fit(self, y: Array, ctx: Array, a: Array, y_next: Array, verbose: bool = True) -> list[dict]:
        """Train every member on bootstrap resamples of the transition set.

        Bootstrapping is what makes the members disagree in a way that tracks
        epistemic uncertainty rather than just initialisation noise.
        """
        cfg = self.cfg
        y = np.asarray(y, dtype=np.float32)
        ctx = np.asarray(ctx, dtype=np.float32)
        a = np.asarray(a, dtype=np.float32)
        y_next = np.asarray(y_next, dtype=np.float32)

        self._y_mu = y.mean(axis=0).astype(np.float64)
        sd = y.std(axis=0)
        sd[sd < 1e-8] = 1.0
        self._y_sd = sd.astype(np.float64)

        # Reference density for the latent-novelty feature: a Gaussian fitted
        # to the training task states. Deliberately simple, and used with the
        # caveat that likelihood-based OOD scores are unreliable alone.
        self._ref_mu = y.mean(axis=0).astype(np.float64)
        cov = np.cov(y.T) + 1e-6 * np.eye(cfg.d_y)
        self._ref_cov_inv = np.linalg.inv(cov)

        lo, hi = y.min(axis=0).astype(np.float64), y.max(axis=0).astype(np.float64)
        pad = 0.2 * np.maximum(hi - lo, 1e-6)
        self._y_lo, self._y_hi = lo - pad, hi + pad

        Y = torch.as_tensor(y)
        C = torch.as_tensor(ctx)
        A = torch.as_tensor(a)
        YN = torch.as_tensor(y_next)
        n = Y.shape[0]
        rng = np.random.default_rng(cfg.seed)

        history: list[dict] = []
        for k, member in enumerate(self.members):
            idx_boot = torch.as_tensor(rng.integers(0, n, size=n))
            opt = torch.optim.AdamW(member.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
            for epoch in range(cfg.epochs):
                perm = idx_boot[torch.randperm(n)]
                tot, nb = 0.0, 0
                for i in range(0, n, cfg.batch_size):
                    b = perm[i : i + cfg.batch_size]
                    opt.zero_grad(set_to_none=True)
                    pred = member(Y[b], C[b], A[b])
                    # Heteroscedastic Gaussian NLL: the members learn their own
                    # output noise, which is what the rollout then samples.
                    sig = torch.exp(member.log_sigma).clamp(1e-5, 1.0)
                    loss = (((pred - YN[b]) / sig) ** 2 * 0.5 + torch.log(sig)).mean()
                    loss.backward()
                    opt.step()
                    tot += float(loss.item())
                    nb += 1
                if epoch == cfg.epochs - 1:
                    history.append({"member": k, "epoch": epoch, "nll": tot / max(nb, 1)})
                    if verbose:
                        print(f"  member {k}: final NLL {tot / max(nb, 1):+.4f}")
        return history

    # -- WorldModel protocol --------------------------------------------------- #
    def encode(self, obs_history, act_history) -> Array:
        """Fully observed here, so the 'latent' is the task state itself.

        A documented simplification: PlanarPush exposes y directly, so q_phi is
        the identity. DreamerV3 and TD-MPC2 adapters supply a real encoder; the
        rest of HALT is unchanged either way, which is the point of wrapping at
        this interface.
        """
        obs = obs_history[-1]
        self._ctx = np.asarray(obs.get("goal", np.zeros(self.cfg.d_context)), dtype=np.float64)
        return np.asarray(obs["y"], dtype=np.float64)

    def _step_np(self, y: Array, a: Array, member: int) -> tuple[Array, Array]:
        with torch.no_grad():
            m = self.members[member]
            out = m(
                torch.as_tensor(y, dtype=torch.float32)[None],
                torch.as_tensor(self._ctx, dtype=torch.float32)[None],
                torch.as_tensor(a, dtype=torch.float32)[None],
            )[0].numpy().astype(np.float64)
            sig = torch.exp(m.log_sigma).clamp(1e-5, 1.0).numpy().astype(np.float64)
        return out, sig

    @torch.no_grad()
    def rollout(self, z: Array, actions: Array, n_rollouts: int, rng: np.random.Generator) -> Array:
        """(K, H, d_y) imagined task states, one trajectory per member.

        Kept entirely in torch. An earlier version called a numpy helper once
        per (member, step), paying a tensor construction and a numpy
        conversion K*H times. Measured A/B on an idle machine, K=5:
        12.7 ms -> 7.5 ms at H=30, 2.4 ms -> 1.6 ms at H=5, so about 1.7x.

        (An earlier draft of this comment claimed 194x. That number came from
        profiling the old path while the full pipeline was saturating both
        cores, and was almost entirely contention. The honest figure is 1.7x.)

        The horizon is inherently sequential, but the members are not, so all
        K trajectories advance together and every tensor is allocated once,
        outside the loop.
        """
        actions_np = np.atleast_2d(np.asarray(actions, dtype=np.float32))
        H = actions_np.shape[0]
        K = min(n_rollouts, self.cfg.K)

        acts = torch.as_tensor(actions_np)                       # (H, d_a)
        ctx = torch.as_tensor(self._ctx, dtype=torch.float32)[None]
        y = torch.as_tensor(np.asarray(z, dtype=np.float32))[None].repeat(K, 1)
        lo = torch.as_tensor(self._y_lo, dtype=torch.float32)
        hi = torch.as_tensor(self._y_hi, dtype=torch.float32)

        sigs = torch.stack([
            torch.exp(self.members[k].log_sigma).clamp(1e-5, 1.0) for k in range(K)
        ])                                                        # (K, d_y)
        # Draw the whole noise field up front from the caller's generator, so
        # the trajectory is reproducible given rng and independent of how the
        # loop is structured.
        noise = torch.as_tensor(
            rng.normal(size=(H, K, self.cfg.d_y)).astype(np.float32)
        ) * sigs[None]

        out = torch.empty((K, H, self.cfg.d_y))
        ctx_k = ctx.repeat(K, 1)
        for h in range(H):
            a_h = acts[h][None].repeat(K, 1)
            step = torch.stack([
                self.members[k](y[k : k + 1], ctx_k[k : k + 1], a_h[k : k + 1])[0]
                for k in range(K)
            ])
            y = torch.clamp(step + noise[h], lo, hi)
            out[:, h] = y
        return out.numpy().astype(np.float64)

    def jacobian(self, z: Array, actions: Array, horizons) -> Array | None:
        """J_hat_h by reverse-mode autodiff through member 0's rollout.

        Noise is not resampled inside the differentiated path, so this is the
        Jacobian of the mean trajectory of that member -- the reparameterised
        case with the noise draw held fixed at zero.
        """
        ctx = torch.as_tensor(self._ctx, dtype=torch.float32)[None]
        member = self.members[0]

        def step_fn(y, a):
            return member(y, ctx, a[None])

        def readout_fn(y):
            return y[0]

        z0 = torch.as_tensor(np.asarray(z, dtype=np.float32))[None]
        acts = torch.as_tensor(np.asarray(actions, dtype=np.float32))
        return torch_rollout_jacobian(step_fn, readout_fn, z0, acts, horizons)

    # -- features that need model internals ----------------------------------- #
    def latent_novelty(self, z: Array) -> float:
        """u_t = -log p_ref(z_t), up to an additive constant."""
        d = np.asarray(z, dtype=np.float64) - self._ref_mu
        return float(0.5 * d @ self._ref_cov_inv @ d)

    def one_step_prediction(self, z: Array, a: Array) -> Array:
        y, _ = self._step_np(np.asarray(z, dtype=np.float64), np.asarray(a, dtype=np.float64), 0)
        return y

    # -- persistence ----------------------------------------------------------- #
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "cfg": self.cfg.as_dict(),
                "state_dict": self.members.state_dict(),
                "task_spec": {"names": list(self.task_spec.names),
                              "sigma": self.task_spec.sigma.tolist()},
                "stats": {
                    "y_mu": self._y_mu.tolist(),
                    "y_sd": self._y_sd.tolist(),
                    "ref_mu": self._ref_mu.tolist(),
                    "ref_cov_inv": self._ref_cov_inv.tolist(),
                    "y_lo": self._y_lo.tolist(),
                    "y_hi": self._y_hi.tolist(),
                },
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "EnsembleWorldModel":
        blob = torch.load(Path(path), map_location="cpu", weights_only=False)
        spec = TaskStateSpec(
            names=tuple(blob["task_spec"]["names"]),
            sigma=np.asarray(blob["task_spec"]["sigma"]),
        )
        m = cls(EnsembleConfig(**blob["cfg"]), spec)
        m.members.load_state_dict(blob["state_dict"])
        st = blob["stats"]
        m._y_mu = np.asarray(st["y_mu"])
        m._y_sd = np.asarray(st["y_sd"])
        m._ref_mu = np.asarray(st["ref_mu"])
        m._ref_cov_inv = np.asarray(st["ref_cov_inv"])
        m._y_lo = np.asarray(st["y_lo"])
        m._y_hi = np.asarray(st["y_hi"])
        return m


class VideoWorldModel(EnsembleWorldModel):
    """Same model, differentiable task readout withheld (§9.8, Table 9).

    GAIA-1 satisfies Table 1's first requirement -- it can be sampled
    stochastically -- but exposes no differentiable task readout, so the
    Jacobian family must be dropped and coverage falls. `jacobian` returns
    None rather than raising, because dropping a family is a supported
    configuration, not an error.
    """

    def __init__(self, cfg: EnsembleConfig, task_spec: TaskStateSpec):
        super().__init__(cfg, task_spec)
        self.capabilities = InterfaceCapabilities(
            stochastic_rollouts=True,
            differentiable_readout=False,
            task_state_at_calibration=True,
            task_state_at_deployment=False,   # no estimator -> no ACI
            n_rollouts=cfg.K,
        )

    def jacobian(self, z, actions, horizons):
        return None
