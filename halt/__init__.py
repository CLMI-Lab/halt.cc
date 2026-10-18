"""HALT -- Horizon-Adaptive Latent Trust: reference implementation.

Certified, state-dependent trust horizons for action-conditioned world models,
after:

    Mohammad, Sheikh and Habib, "HALT: Certifying Trust Horizons in
    Action-Conditioned Robotic World Models", NeurIPS 2026 Workshop on Robot
    Learning with World Models.

Scope, stated once here and repeated in every artefact this package writes:
the algorithms are faithful to the paper; the bundled environment is a
synthetic contact-rich suite, not one of the five MuJoCo tasks of Section 8.
Results produced here validate the implementation and the propositions. They
are not a reproduction of the paper's reported numbers, and nothing in this
package will claim otherwise.

Map from paper to code:

    Eq. (2)  task error d_h          halt.metrics.task_error
    Def. 3.1 TTCD                    halt.metrics.ttcd
    Def. 3.2 silent failure          halt.metrics.silent_failure_rate
    Eq. (3)  action Jacobians        halt.jacobian
    Eq. (4)  CAS                     halt.metrics.cas
    Eq. (5)  rho                     halt.metrics.rho
    Table 1  interface               halt.types.InterfaceCapabilities
    §4.1     features                halt.features.FeatureExtractor
    Eq. (6)  monotone head           halt.nn.head.MonotoneQuantileHead
    Eq. (7)  pinball loss            halt.nn.head.pinball_loss
    Eq. (8)  conformal calibration   halt.conformal.calibrate
    Def. 4.1 certified horizon       halt.conformal.certified_horizon
    Prop 4.1/4.2                     halt.theory
    §4.4     planner-induced shift   halt.conformal.choose_kappa_plan, scripts/06
    Eq. (11) planning objective      halt.planner
    Eq. (12) delayed ACI             halt.aci.ACIController
    Prop 5.1 simulation lemma        halt.theory.prop_51_bound
    Prop 5.2 long-run coverage       halt.aci.prop_52_bound
    §6       PHANTASM                halt.envs.phantasm
    Alg. G.1 offline calibration     halt.calibrate.run_algorithm_g1
    Alg. 1   deployment loop         halt.deploy.run_episode
"""

__version__ = "0.1.0"

from .types import TaskStateSpec, InterfaceCapabilities, Anchor, CalibrationResult
from .conformal import calibrate, certified_horizon, certified_horizon_batch
from .aci import ACIController, StaticController, prop_52_bound
from .calibrate import run_algorithm_g1, CalibratedHALT, calibrate_tolerance
from .features import FeatureExtractor, FeatureConfig, FAMILIES

__all__ = [
    "__version__",
    "TaskStateSpec", "InterfaceCapabilities", "Anchor", "CalibrationResult",
    "calibrate", "certified_horizon", "certified_horizon_batch",
    "ACIController", "StaticController", "prop_52_bound",
    "run_algorithm_g1", "CalibratedHALT", "calibrate_tolerance",
    "FeatureExtractor", "FeatureConfig", "FAMILIES",
]
