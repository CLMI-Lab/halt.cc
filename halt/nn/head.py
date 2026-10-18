"""The monotone quantile head, Eq. (6), and the pinball objective, Eq. (7).

    q_hat_h(phi) = sum_{j=1..h} softplus( delta_j(phi) ),   delta = f_xi(phi)

Monotonicity in h is enforced *architecturally*, by construction, not by a
penalty term. That matters for two reasons the paper names:

  * the certified horizon is then a single threshold crossing, so Definition
    4.1 is an argmax over a prefix rather than a search, and
  * no post-hoc isotonic repair is ever needed, which would otherwise break
    the exchangeability the conformal step depends on.

Positivity is also guaranteed, which Eq. (8) needs: the score divides by
q_hat_h, so a head that could emit zero would be unusable.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

Array = np.ndarray

__all__ = ["MonotoneQuantileHead", "HeadConfig", "pinball_loss", "train_head"]


@dataclass
class HeadConfig:
    d_phi: int
    h_max: int = 50
    width: int = 256
    depth: int = 3
    activation: str = "gelu"          # paper: GELU at width 256; SiLU/128 in the ref run
    gamma_w: float = 0.97             # w_h = gamma_w^h, Eq. (7)
    alpha: float = 0.05
    lr: float = 3e-4
    weight_decay: float = 1e-2
    batch_size: int = 256
    epochs: int = 60
    min_increment: float = 1e-6       # keeps q_hat strictly positive for Eq. (8)
    seed: int = 0

    def as_dict(self) -> dict:
        return asdict(self)


_ACT = {"gelu": nn.GELU, "silu": nn.SiLU, "relu": nn.ReLU, "tanh": nn.Tanh}


class MonotoneQuantileHead(nn.Module):
    """f_xi followed by the cumulative-softplus construction of Eq. (6)."""

    def __init__(self, cfg: HeadConfig):
        super().__init__()
        self.cfg = cfg
        if cfg.activation not in _ACT:
            raise ValueError(f"activation must be one of {sorted(_ACT)}")
        act = _ACT[cfg.activation]

        layers: list[nn.Module] = []
        d_in = cfg.d_phi
        for _ in range(cfg.depth - 1):
            layers += [nn.Linear(d_in, cfg.width), act()]
            d_in = cfg.width
        layers += [nn.Linear(d_in, cfg.h_max)]
        self.net = nn.Sequential(*layers)

        # Feature standardisation is part of the head, so it travels with the
        # checkpoint. A head calibrated against one normalisation and deployed
        # against another is a silent source of miscoverage.
        self.register_buffer("phi_mean", torch.zeros(cfg.d_phi))
        self.register_buffer("phi_std", torch.ones(cfg.d_phi))

    # -- standardisation ---------------------------------------------------- #
    def fit_normaliser(self, phi: Array) -> None:
        phi = np.asarray(phi, dtype=np.float32)
        self.phi_mean.copy_(torch.as_tensor(phi.mean(axis=0)))
        std = phi.std(axis=0)
        std[std < 1e-6] = 1.0
        self.phi_std.copy_(torch.as_tensor(std))

    def _norm(self, phi: torch.Tensor) -> torch.Tensor:
        return (phi - self.phi_mean) / self.phi_std

    # -- Eq. (6) ------------------------------------------------------------ #
    def forward(self, phi: torch.Tensor) -> torch.Tensor:
        """(B, d_phi) -> (B, H_max), non-decreasing and strictly positive in h."""
        delta = self.net(self._norm(phi))
        inc = F.softplus(delta) + self.cfg.min_increment
        return torch.cumsum(inc, dim=-1)

    # -- numpy convenience --------------------------------------------------- #
    @torch.no_grad()
    def predict(self, phi: Array) -> Array:
        self.eval()
        x = torch.as_tensor(np.atleast_2d(np.asarray(phi, dtype=np.float32)))
        return self.forward(x).cpu().numpy().astype(np.float64)

    # -- persistence --------------------------------------------------------- #
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"cfg": self.cfg.as_dict(), "state_dict": self.state_dict()}, path)

    @classmethod
    def load(cls, path: str | Path) -> "MonotoneQuantileHead":
        blob = torch.load(Path(path), map_location="cpu", weights_only=False)
        head = cls(HeadConfig(**blob["cfg"]))
        head.load_state_dict(blob["state_dict"])
        head.eval()
        return head


# --------------------------------------------------------------------------- #
# Eq. (7)
# --------------------------------------------------------------------------- #
def pinball_loss(
    q_hat: torch.Tensor, d: torch.Tensor, alpha: float, weights: torch.Tensor
) -> torch.Tensor:
    """L_pin = E[ sum_h w_h * rho_{1-alpha}( d_h - q_hat_h ) ]

    with rho_beta(u) = max{ beta u, (beta-1) u }. At beta = 1 - alpha the
    minimiser is the conditional (1-alpha)-quantile of d_h, which is exactly
    what Eq. (8) then needs as its denominator.

    w_h = gamma_w^h down-weights horizons that are rarely reached, so the head
    spends its capacity where the planner actually operates.
    """
    beta = 1.0 - alpha
    u = d - q_hat
    rho = torch.maximum(beta * u, (beta - 1.0) * u)
    return (rho * weights).sum(dim=-1).mean()


def train_head(
    phi_train: Array,
    d_train: Array,
    cfg: HeadConfig,
    phi_val: Array | None = None,
    d_val: Array | None = None,
    verbose: bool = True,
) -> tuple[MonotoneQuantileHead, list[dict]]:
    """Algorithm G.1 line 5: fit xi on D_train by minimising Eq. (7).

    Returns the trained head and a per-epoch history suitable for a CSV.
    """
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    phi_train = np.asarray(phi_train, dtype=np.float32)
    d_train = np.asarray(d_train, dtype=np.float32)[:, : cfg.h_max]
    if phi_train.shape[0] != d_train.shape[0]:
        raise ValueError(
            f"{phi_train.shape[0]} feature rows but {d_train.shape[0]} error rows"
        )
    if d_train.shape[1] != cfg.h_max:
        raise ValueError(
            f"d has {d_train.shape[1]} horizons but cfg.h_max={cfg.h_max}"
        )

    head = MonotoneQuantileHead(cfg)
    head.fit_normaliser(phi_train)

    w = torch.tensor(
        [cfg.gamma_w ** (h + 1) for h in range(cfg.h_max)], dtype=torch.float32
    )
    opt = torch.optim.AdamW(head.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(cfg.epochs, 1))

    X = torch.as_tensor(phi_train)
    Y = torch.as_tensor(d_train)
    n = X.shape[0]
    has_val = phi_val is not None and d_val is not None
    if has_val:
        Xv = torch.as_tensor(np.asarray(phi_val, dtype=np.float32))
        Yv = torch.as_tensor(np.asarray(d_val, dtype=np.float32)[:, : cfg.h_max])

    history: list[dict] = []
    for epoch in range(cfg.epochs):
        head.train()
        perm = torch.randperm(n)
        total, nb = 0.0, 0
        for i in range(0, n, cfg.batch_size):
            idx = perm[i : i + cfg.batch_size]
            opt.zero_grad(set_to_none=True)
            loss = pinball_loss(head(X[idx]), Y[idx], cfg.alpha, w)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 10.0)
            opt.step()
            total += float(loss.item())
            nb += 1
        sched.step()

        rec = {"epoch": epoch, "train_pinball": total / max(nb, 1),
               "lr": float(sched.get_last_lr()[0])}
        if has_val:
            head.eval()
            with torch.no_grad():
                qv = head(Xv)
                rec["val_pinball"] = float(pinball_loss(qv, Yv, cfg.alpha, w).item())
                # marginal coverage of the *uncalibrated* head, pre-conformal.
                # It will not equal 1-alpha; that gap is what lambda_hat repairs.
                rec["val_pointwise_coverage"] = float((Yv <= qv).float().mean().item())
        history.append(rec)
        if verbose and (epoch % 10 == 0 or epoch == cfg.epochs - 1):
            extra = f" val={rec.get('val_pinball', float('nan')):.5f}" if has_val else ""
            print(f"  epoch {epoch:3d}  pinball={rec['train_pinball']:.5f}{extra}")

    head.eval()
    return head, history
