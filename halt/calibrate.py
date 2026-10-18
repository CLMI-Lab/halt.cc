"""Algorithm G.1 -- offline calibration -- and the tolerance heuristic of H.2.

    Require: M_theta; disjoint EPISODE-level splits D_train, D_cal; tau; alpha; H_max
     1-4: collect anchors (halt/collect.py)
     5:   fit xi on D_train by minimising L_pin with the monotone head (Eq. 6-7)
     6:   R^(i) <- max_h d_h^(i) / q_hat_h(phi^(i))        for i in D_cal
     7:   lambda_hat <- ceil((n+1)(1-alpha))/n empirical quantile of {R^(i)}
         Optional: for planner-conditional (R1), replace {R^(i)} by
                   planner-induced scores from D_plan
    Ensure: (xi, lambda_hat)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .conformal import calibrate as _conformal_calibrate
from .nn.head import HeadConfig, MonotoneQuantileHead, train_head
from .types import CalibrationResult

Array = np.ndarray

__all__ = ["calibrate_tolerance", "run_algorithm_g1", "CalibratedHALT"]


def calibrate_tolerance(d_train: Array, h_a: int = 10, h_b: int = 20) -> float:
    """tau = sqrt( P75(d_{h_a}) * P75(d_{h_b}) ), the heuristic of Appendix H.2.

    Worth being clear about what this is and is not. It is a *design choice*,
    not a statistical result: the certificate is valid for whatever tau is
    supplied, and tau determines what is being certified rather than whether
    the guarantee holds. The geometric mean of two mid-horizon upper quartiles
    simply lands tau somewhere a planner can usefully operate -- short enough
    to be meaningful, long enough not to abstain constantly.

    On a real task, set tau from the task's own tolerances instead and let
    sigma do the normalising; then tau = 1 is the edge of task tolerance.
    """
    d = np.asarray(d_train, dtype=np.float64)
    if d.shape[1] < max(h_a, h_b):
        raise ValueError(f"need at least {max(h_a, h_b)} horizons, have {d.shape[1]}")
    p_a = float(np.quantile(d[:, h_a - 1], 0.75))
    p_b = float(np.quantile(d[:, h_b - 1], 0.75))
    return float(np.sqrt(p_a * p_b))


@dataclass
class CalibratedHALT:
    """(xi, lambda_hat) plus everything needed to reproduce and audit them."""

    head: MonotoneQuantileHead
    calibration: CalibrationResult
    tau: float
    alpha: float
    h_max: int
    feature_names: list[str]
    history: list[dict]

    # -- the two operations deployment needs ---------------------------------- #
    def q_hat(self, phi: Array) -> Array:
        return self.head.predict(phi)

    def certified_horizon(self, phi: Array, lambda_hat: float | None = None) -> Array:
        from .conformal import certified_horizon_batch

        lam = self.calibration.lambda_hat if lambda_hat is None else lambda_hat
        return certified_horizon_batch(self.q_hat(phi), lam, self.tau)

    # -- persistence ----------------------------------------------------------- #
    def save(self, directory: str | Path) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        self.head.save(d / "head.pt")
        np.save(d / "calibration_scores.npy", self.calibration.scores)
        import json

        (d / "calibration.json").write_text(
            json.dumps(
                {
                    **self.calibration.as_dict(),
                    "tau": self.tau,
                    "alpha": self.alpha,
                    "h_max": self.h_max,
                    "feature_names": self.feature_names,
                },
                indent=2,
            )
        )

    @classmethod
    def load(cls, directory: str | Path) -> "CalibratedHALT":
        import json

        d = Path(directory)
        meta = json.loads((d / "calibration.json").read_text())
        scores = np.load(d / "calibration_scores.npy")
        head = MonotoneQuantileHead.load(d / "head.pt")
        cal = CalibrationResult(
            lambda_hat=meta["lambda_hat"], alpha=meta["alpha"], n_cal=meta["n_cal"],
            tau=meta["tau"], h_max=meta["h_max"], scores=scores,
            planner_conditional=meta.get("planner_conditional", False),
        )
        return cls(head, cal, meta["tau"], meta["alpha"], meta["h_max"],
                   meta["feature_names"], history=[])


def run_algorithm_g1(
    phi_train: Array,
    d_train: Array,
    phi_cal: Array,
    d_cal: Array,
    alpha: float,
    h_max: int,
    feature_names: list[str],
    tau: float | None = None,
    head_cfg: HeadConfig | None = None,
    planner_scores: Array | None = None,
    verbose: bool = True,
) -> CalibratedHALT:
    """Algorithm G.1 end to end.

    `planner_scores`, when supplied, replaces the behaviour-policy scores at
    line 7 -- remedy (R1) of §4.4. The resulting lambda_hat is larger and the
    certified horizon shorter; that shortening is the honest cost of
    closed-loop planning, not a defect of the method.
    """
    if tau is None:
        tau = calibrate_tolerance(d_train)
        if verbose:
            print(f"  tau calibrated from training data (Appendix H.2): {tau:.4f}")

    cfg = head_cfg or HeadConfig(d_phi=phi_train.shape[1], h_max=h_max, alpha=alpha)
    if cfg.d_phi != phi_train.shape[1] or cfg.h_max != h_max or cfg.alpha != alpha:
        raise ValueError(
            "HeadConfig disagrees with the data: "
            f"d_phi {cfg.d_phi} vs {phi_train.shape[1]}, "
            f"h_max {cfg.h_max} vs {h_max}, alpha {cfg.alpha} vs {alpha}"
        )

    if verbose:
        print(f"  fitting the monotone quantile head on {phi_train.shape[0]} anchors")
    head, history = train_head(phi_train, d_train, cfg, phi_cal, d_cal, verbose=verbose)

    q_cal = head.predict(phi_cal)
    if planner_scores is None:
        cal = _conformal_calibrate(d_cal, q_cal, alpha, tau, h_max=h_max)
    else:
        lam = __import__("halt.conformal", fromlist=["conformal_quantile"]).conformal_quantile(
            planner_scores, alpha
        )
        cal = CalibrationResult(
            lambda_hat=lam, alpha=alpha, n_cal=int(planner_scores.shape[0]),
            tau=tau, h_max=h_max, scores=planner_scores, planner_conditional=True,
        )

    if verbose:
        kind = "planner-conditional" if cal.planner_conditional else "behaviour-policy"
        print(f"  lambda_hat ({kind}) = {cal.lambda_hat:.4f} on n_cal = {cal.n_cal}")

    return CalibratedHALT(
        head=head, calibration=cal, tau=float(tau), alpha=float(alpha),
        h_max=int(h_max), feature_names=list(feature_names), history=history,
    )
