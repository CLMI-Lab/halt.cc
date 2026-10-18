"""Adaptive conformal inference with delayed, state-dependent feedback (§4.6).

The update is Eq. (12):

    alpha_{t+1} = alpha_t + eta ( alpha - 1[ miss_{t - L_t} ] )
    miss_s      = 1[ exists h <= H*_s : d_h^(s) > lambda_hat(alpha_s) q_hat_h ]

The robotics-specific wrinkle, and the only place the setting departs from
Gibbs & Candes (2021), is that the miss indicator for the rollout launched at
time s is not observable until s + H*_s. The feedback stream therefore arrives
with a variable, state-dependent lag L_t <= L_max. That delay is modelled here
as an explicit queue, exactly as Algorithm 1 lines 12-14 describe, rather than
by assuming feedback is instantaneous.

Proposition 5.2 bounds the long-run deviation by

    | T^-1 sum_t miss_t - alpha |  <=  ( alpha_1 + eta (1 + L_max) ) / ( eta T )

and `prop_52_bound` evaluates exactly that expression. Two things about it are
worth keeping in view when reading results: it is asymptotic in T, and it says
nothing finite-sample. An empirical deviation inside +/-0.03 is consistent with
the bound but is not implied by it.

Running ACI at deployment requires observing y_{t+h}, which is generally
unavailable outside simulation or mocap. `ACIController.requires_ground_truth`
is True for that reason, and `deploy.py` refuses to enable the ACI mode when
the model's InterfaceCapabilities say the task state is not observable then.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

Array = np.ndarray

__all__ = ["ACIController", "ACITrace", "prop_52_bound", "StaticController"]


def prop_52_bound(alpha_1: float, eta: float, l_max: int, T: int) -> float:
    """( alpha_1 + eta (1 + L_max) ) / ( eta T ), Proposition 5.2."""
    if eta <= 0 or T <= 0:
        return float("inf")
    return (alpha_1 + eta * (1.0 + l_max)) / (eta * T)


@dataclass
class ACITrace:
    """Everything needed to audit a run, and to write it to CSV."""

    alpha: list[float] = field(default_factory=list)
    miss: list[int] = field(default_factory=list)
    lag: list[int] = field(default_factory=list)
    running_error: list[float] = field(default_factory=list)
    pending: list[int] = field(default_factory=list)

    def as_arrays(self) -> dict[str, Array]:
        return {k: np.asarray(v) for k, v in self.__dict__.items()}

    @property
    def miss_rate(self) -> float:
        return float(np.mean(self.miss)) if self.miss else float("nan")


class StaticController:
    """No recalibration: alpha_t = alpha for all t.

    Finite-sample coverage under exchangeability (Prop. 4.1), and no guarantee
    at all once the dynamics shift. Kept as a first-class mode because, per
    Remark 4.2, most deployments will run it -- ACI needs ground truth they do
    not have.
    """

    requires_ground_truth = False
    name = "static"

    def __init__(self, alpha: float):
        self.alpha_target = float(alpha)
        self.alpha_t = float(alpha)
        self.trace = ACITrace()
        self._n = 0
        self._misses = 0

    def level(self) -> float:
        return self.alpha_t

    def observe(self, miss: int | None, lag: int = 0) -> None:
        if miss is None:
            return
        self._n += 1
        self._misses += int(miss)
        self.trace.alpha.append(self.alpha_t)
        self.trace.miss.append(int(miss))
        self.trace.lag.append(int(lag))
        self.trace.running_error.append(self._misses / self._n - self.alpha_target)
        self.trace.pending.append(0)


class ACIController:
    """Eq. (12) with a bounded-lag feedback queue.

    Parameters
    ----------
    alpha   : the target miss rate the long-run average is driven to
    eta     : step size. Larger tracks shift faster and oscillates more.
    l_max   : the bound L_max on the feedback lag. Only used to report the
              Prop. 5.2 bound; the queue itself accepts whatever lag it is given
              and raises if one exceeds l_max, since a violated assumption
              should surface rather than quietly invalidate the bound.
    clip    : OFF by default, and it should stay off.

              Eq. (12) contains no clipping, and adding one breaks the proof.
              The bound comes from telescoping the recursion:

                  sum_i miss_i = n alpha + (alpha_1 - alpha_{n+1}) / eta

              A clip discards part of an increment, the identity stops holding,
              and the realised deviation can exceed the Prop. 5.2 bound even
              though the controller is behaving sensibly. We measured exactly
              that: with clipping on at L_max = 20, the telescoping residual
              stayed at 3e-4 while the realised deviation rose to 1.3e-2,
              past a bound of 3.7e-3.

              No clip is needed, because the recursion is self-bounding
              whenever the miss process responds to the level. If alpha_t
              drops below 0 then `level()` returns 0, the certificate is at
              its widest, the miss indicator goes to 0, and the next update
              moves alpha_t back up by eta*alpha. So alpha_t stays in
              [-eta, 1+eta] on its own, which is the interval Gibbs & Candes
              assume rather than impose.

              `summary()` reports `level_saturated_frac`. If that is far from
              zero, the self-bounding argument is not operating -- the miss
              process is not responding to the level -- and Prop. 5.2's
              premises are not met in that run. That is a finding about the
              deployment, not a bug to clip away.
    """

    requires_ground_truth = True
    name = "aci"

    def __init__(self, alpha: float, eta: float = 0.05, l_max: int = 10, clip: bool = False):
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must lie in (0,1), got {alpha}")
        if eta <= 0:
            raise ValueError(f"eta must be positive, got {eta}")
        self.alpha_target = float(alpha)
        self.alpha_1 = float(alpha)
        self.alpha_t = float(alpha)
        self.eta = float(eta)
        self.l_max = int(l_max)
        self.clip = bool(clip)

        self._queue: list[tuple[int, int]] = []   # (arrival_time, miss)
        self._t = 0
        self._n_fed = 0
        self._misses = 0
        self._n_saturated = 0
        self.trace = ACITrace()

    # -- the level actually used to size the certificate this step ---------- #
    def level(self) -> float:
        """alpha_t clamped into [0,1] for use as a miscoverage level.

        The clamp is on the *use* of alpha_t, not on the recursion, so the
        telescoping identity behind Prop. 5.2 is untouched.
        """
        if self.alpha_t < 0.0 or self.alpha_t > 1.0:
            self._n_saturated += 1
        return float(np.clip(self.alpha_t, 0.0, 1.0))

    # -- launch: record a rollout whose outcome will be known later --------- #
    def launch(self, miss: int, lag: int) -> None:
        """Queue the miss indicator for the rollout started now.

        `miss` is the realised outcome; it is queued rather than applied, and
        only becomes visible to the controller `lag` steps later. In a real
        deployment `miss` is not known at launch time either -- the caller
        computes it when the horizon elapses -- but queueing it here keeps the
        simulation and the deployment loop on identical code paths.
        """
        if lag > self.l_max:
            raise ValueError(
                f"lag {lag} exceeds L_max={self.l_max}; Proposition 5.2 assumes "
                "L_t <= L_max almost surely, so either raise l_max or fix the "
                "horizon distribution"
            )
        self._queue.append((self._t + int(lag), int(miss)))

    # -- step: consume all feedback that has arrived ------------------------ #
    def step(self) -> int:
        """Advance one control step; apply every arrived update. Returns #applied."""
        arrived = [m for (a, m) in self._queue if a <= self._t]
        self._queue = [(a, m) for (a, m) in self._queue if a > self._t]

        for miss in arrived:
            self.alpha_t = self.alpha_t + self.eta * (self.alpha_target - miss)
            if self.clip:
                self.alpha_t = float(np.clip(self.alpha_t, -self.eta, 1.0 + self.eta))
            self._n_fed += 1
            self._misses += miss

        self.trace.alpha.append(self.alpha_t)
        self.trace.pending.append(len(self._queue))
        if arrived:
            self.trace.miss.extend(arrived)
            self.trace.lag.extend([0] * len(arrived))
            for _ in arrived:
                self.trace.running_error.append(
                    self._misses / self._n_fed - self.alpha_target
                )
        self._t += 1
        return len(arrived)

    # -- convenience for offline simulation --------------------------------- #
    def observe(self, miss: int | None, lag: int = 0) -> None:
        if miss is not None:
            self.launch(miss, lag)
        self.step()

    @property
    def realised_miss_rate(self) -> float:
        return self._misses / self._n_fed if self._n_fed else float("nan")

    def bound(self, T: int | None = None) -> float:
        return prop_52_bound(self.alpha_1, self.eta, self.l_max, T or max(self._n_fed, 1))

    def telescoping_residual(self) -> float:
        """| mean(miss) - alpha | - | alpha_1 - alpha_{n+1} | / (eta n).

        Zero to machine precision whenever the recursion is applied exactly.
        A non-zero value means something interfered with the update -- a clip,
        a dropped feedback item, a double application -- and the Prop. 5.2
        bound no longer follows. Asserted in the test suite.
        """
        n = max(self._n_fed, 1)
        lhs = abs(self.realised_miss_rate - self.alpha_target)
        rhs = abs(self.alpha_1 - self.alpha_t) / (self.eta * n)
        return float(abs(lhs - rhs))

    def summary(self) -> dict:
        T = max(self._n_fed, 1)
        dev = abs(self.realised_miss_rate - self.alpha_target)
        b = self.bound(T)
        return {
            "controller": self.name,
            "alpha_target": self.alpha_target,
            "alpha_final": self.alpha_t,
            "eta": self.eta,
            "l_max": self.l_max,
            "T_fed": self._n_fed,
            "miss_rate": self.realised_miss_rate,
            "deviation": dev,
            "prop_5_2_bound": b,
            "bound_respected": bool(dev <= b + 1e-12),
            "telescoping_residual": self.telescoping_residual(),
            "level_saturated_frac": self._n_saturated / max(self._t, 1),
            "pending_at_end": len(self._queue),
        }


def lambda_at_level(scores: Array, alpha_t: float) -> float:
    """lambda_hat(alpha_t) -- recalibrate the scale at the current effective level.

    ACI moves alpha_t, and Eq. (12)'s miss indicator compares against
    lambda_hat(alpha_s) q_hat_h, so the scale must be re-read from the stored
    calibration scores rather than held fixed. This is a quantile lookup on an
    array already in memory, not a re-calibration: cost is O(log n).
    """
    from .conformal import conformal_quantile

    a = float(np.clip(alpha_t, 1e-6, 1 - 1e-6))
    return conformal_quantile(scores, a)
