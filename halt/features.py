"""The five deployment-available feature families of §4.1.

The defining constraint, stated in Remark 4.1(a): features never use the task
state y. They are built from model internals and sensor residuals only, because
at time t the robot has no ground truth. Calibration uses y offline (b), and
ACI uses it at deployment (c) -- but phi_t never does. `FeatureExtractor`
therefore has no access to an Environment at all, which makes the constraint
structural rather than a matter of discipline.

  1. Rollout disagreement     D_h over K stochastic rollouts or ensemble heads
  2. Self-Jacobian statistics log||J_hat||_F, H(J_hat), condition number
  3. Latent novelty           u_t = -log p_ref(z_t), or an RND score
  4. Multimodal surprise      one-step tactile residual, contact-onset prob
  5. Realised error memory    the last m one-step residuals

Families 1 and 2 are gated by Table 1: without K >= 2 rollouts there is no
disagreement, and without a differentiable task readout there are no Jacobian
features. Dropping either is handled here, not by the caller, and the dropped
block is zero-filled so the feature dimension stays fixed across models.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .metrics import spectral_entropy
from .types import InterfaceCapabilities, TaskStateSpec

Array = np.ndarray

__all__ = ["FeatureConfig", "FeatureExtractor", "FAMILIES"]

FAMILIES = ("disagreement", "jacobian", "novelty", "tactile", "memory")


@dataclass
class FeatureConfig:
    h_probe: tuple[int, ...] = (1, 5, 10, 20)   # horizons at which to probe
    memory_m: int = 5                            # last m realised residuals
    families: tuple[str, ...] = FAMILIES
    jacobian_horizons: tuple[int, ...] = (1, 5)

    def __post_init__(self) -> None:
        bad = set(self.families) - set(FAMILIES)
        if bad:
            raise ValueError(f"unknown feature families: {sorted(bad)}")


class FeatureExtractor:
    """Builds phi_t. Deterministic given its inputs; never touches y."""

    def __init__(
        self,
        spec: TaskStateSpec,
        caps: InterfaceCapabilities,
        cfg: FeatureConfig | None = None,
    ):
        self.spec = spec
        self.caps = caps
        self.cfg = cfg or FeatureConfig()

        # Table 1 gating, applied once at construction.
        requested = set(self.cfg.families)
        self.disabled = set(caps.disabled_feature_families()) & requested
        self.active = tuple(f for f in self.cfg.families if f not in self.disabled)

        self._layout: list[tuple[str, int]] = []
        for fam in self.cfg.families:          # layout covers requested, not active
            self._layout.append((fam, self._family_dim(fam)))
        self.dim = sum(d for _, d in self._layout)

    # -- layout -------------------------------------------------------------- #
    def _family_dim(self, fam: str) -> int:
        if fam == "disagreement":
            return len(self.cfg.h_probe)
        if fam == "jacobian":
            return 3 * len(self.cfg.jacobian_horizons)   # log norm, entropy, log cond
        if fam == "novelty":
            return 1
        if fam == "tactile":
            return 2                                      # residual, contact-onset prob
        if fam == "memory":
            return self.cfg.memory_m
        raise AssertionError(fam)

    def names(self) -> list[str]:
        out: list[str] = []
        for fam, _ in self._layout:
            if fam == "disagreement":
                out += [f"D_h{h}" for h in self.cfg.h_probe]
            elif fam == "jacobian":
                for h in self.cfg.jacobian_horizons:
                    out += [f"logJnorm_h{h}", f"Jentropy_h{h}", f"logJcond_h{h}"]
            elif fam == "novelty":
                out += ["latent_novelty"]
            elif fam == "tactile":
                out += ["tactile_residual", "contact_onset_prob"]
            elif fam == "memory":
                out += [f"resid_lag{k+1}" for k in range(self.cfg.memory_m)]
        return out

    # -- family 1 ------------------------------------------------------------ #
    def _disagreement(self, y_rollouts: Array) -> Array:
        """D_h = C(K,2)^-1 sum_{i<j} || y_hat_h^(i) - y_hat_h^(j) ||_Sigma.

        y_rollouts : (K, H, d_y)
        """
        K = y_rollouts.shape[0]
        if K < 2:
            return np.zeros(len(self.cfg.h_probe))
        out = []
        for h in self.cfg.h_probe:
            idx = min(h, y_rollouts.shape[1]) - 1
            ys = y_rollouts[:, idx, :]                       # (K, d_y)
            diffs = ys[:, None, :] - ys[None, :, :]          # (K, K, d_y)
            norms = self.spec.weighted_norm(diffs)           # (K, K)
            iu = np.triu_indices(K, k=1)
            out.append(float(norms[iu].mean()))
            # mean over the C(K,2) unordered pairs, which is what the
            # C(K,2)^-1 sum_{i<j} in §4.1 denotes.
        return np.asarray(out)

    # -- family 2 ------------------------------------------------------------ #
    def _jacobian_stats(self, jacobians: Array | None) -> Array:
        """log ||J_hat||_F, spectral entropy, log condition number.

        Computing J_hat needs no ground truth -- only *evaluating* CAS does,
        because CAS compares against J*. That asymmetry is what lets the
        Jacobian family be a deployment feature at all.
        """
        n = len(self.cfg.jacobian_horizons)
        if jacobians is None:
            return np.zeros(3 * n)
        out = []
        for i in range(n):
            J = np.asarray(jacobians[i], dtype=np.float64)
            fro = float(np.linalg.norm(J))
            s = np.linalg.svd(J, compute_uv=False)
            smax, smin = float(s.max()), float(s.min())
            cond = smax / smin if smin > 1e-12 else 1e12
            out += [
                np.log(fro + 1e-12),
                spectral_entropy(J),
                np.log(cond + 1e-12),
            ]
        return np.asarray(out)

    # -- assemble ------------------------------------------------------------ #
    def extract(
        self,
        y_rollouts: Array,
        jacobians: Array | None = None,
        latent_novelty: float = 0.0,
        tactile_residual: float = 0.0,
        contact_onset_prob: float = 0.0,
        error_memory: Array | None = None,
    ) -> Array:
        """Return phi_t, shape (self.dim,). Disabled families contribute zeros."""
        blocks: list[Array] = []
        for fam, d in self._layout:
            if fam in self.disabled:
                blocks.append(np.zeros(d))
                continue
            if fam == "disagreement":
                blocks.append(self._disagreement(np.asarray(y_rollouts)))
            elif fam == "jacobian":
                blocks.append(self._jacobian_stats(jacobians))
            elif fam == "novelty":
                # Used with the caveat of Nalisnick et al. (2019): likelihood-
                # based OOD scores are unreliable in isolation. It is one of
                # five families here, never a standalone gate.
                blocks.append(np.asarray([latent_novelty]))
            elif fam == "tactile":
                # Contact dominates dynamical divergence and tactile is its
                # earliest observable, which is why this family carries the
                # largest marginal gain in the paper's ablation.
                blocks.append(np.asarray([tactile_residual, contact_onset_prob]))
            elif fam == "memory":
                m = self.cfg.memory_m
                em = np.zeros(m) if error_memory is None else np.asarray(error_memory, dtype=np.float64)
                if em.shape[0] < m:
                    em = np.concatenate([np.zeros(m - em.shape[0]), em])
                blocks.append(em[-m:])
        phi = np.concatenate(blocks)
        if phi.shape[0] != self.dim:
            raise AssertionError(f"feature dim drift: {phi.shape[0]} vs {self.dim}")
        return np.nan_to_num(phi, nan=0.0, posinf=0.0, neginf=0.0)

    # -- ablation support ----------------------------------------------------- #
    def family_slices(self) -> dict[str, slice]:
        out, i = {}, 0
        for fam, d in self._layout:
            out[fam] = slice(i, i + d)
            i += d
        return out

    def mask_to(self, phi: Array, keep: tuple[str, ...]) -> Array:
        """Zero out every family except `keep`. Used by the Table-5 ablation.

        Zeroing rather than removing keeps the head's input dimension fixed, so
        one architecture serves every ablation cell and differences are
        attributable to information rather than capacity.
        """
        phi = np.array(phi, dtype=np.float64, copy=True)
        sl = self.family_slices()
        for fam, s in sl.items():
            if fam not in keep:
                phi[..., s] = 0.0
        return phi
