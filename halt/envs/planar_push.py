"""PlanarPush: a small contact-rich environment that runs anywhere.

This is the reference environment for the implementation. It is NOT one of the
five MuJoCo tasks of §8 and reproduces none of their numbers. It exists so that
every algorithm in the paper can be executed end to end on a machine without
MuJoCo, and so that the propositions can be checked numerically rather than
asserted. `halt/envs/mujoco_adapter.py` is where the real tasks plug in.

What it does reproduce is the *phenomenon* the paper is about: a regime where a
learned model stays accurate for many steps (free space), and a contact event
after which it becomes dynamically meaningless within a few steps. That
discontinuity is what makes a fixed horizon wrong and a state-dependent one
necessary, and it is built in rather than assumed.

Dynamics (2-D, analytic, deterministic given the noise draw):
    end effector   p_e   velocity-controlled, first-order lag
    object         p_o   free-sliding with Coulomb-like drag
    contact        when ||p_e - p_o|| < r_contact, an impulse transfers along
                   the contact normal, scaled by restitution and inverse mass

Task state  y = [o_x, o_y, o_vx, o_vy, e_x, e_y]  with per-coordinate
tolerances sigma, so tau = 1 is the edge of task tolerance exactly as Eq. (2)
intends.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from ..types import TaskStateSpec

Array = np.ndarray

__all__ = ["PlanarPushParams", "PlanarPush", "TASK_SPEC"]


# The task state y is a READOUT, not the simulator state. Section 3 lists what
# belongs in it: "object six-degree-of-freedom poses, end-effector pose,
# contact indicators, gripper aperture, and any other task-defining quantity".
# Object velocity is a state variable but not a task-defining one -- nobody
# specifies a pushing task in terms of the object's instantaneous velocity --
# so it drives the dynamics here and stays out of y. The simulator still
# carries it; `task_state()` simply does not expose it.
TASK_SPEC = TaskStateSpec(
    names=("obj_x", "obj_y", "ee_x", "ee_y", "contact"),
    #  1 cm on object position, 5 mm on the end effector, and half a unit on
    #  the contact indicator, so a full contact misprediction costs 2 tolerance
    #  units. These are task-design choices, not statistical ones: they decide
    #  what tau = 1 means, and the certificate is valid for whatever they are.
    sigma=np.array([0.010, 0.010, 0.005, 0.005, 0.5]),
)


@dataclass(frozen=True)
class PlanarPushParams:
    dt: float = 0.05
    mass: float = 1.0
    friction: float = 0.6
    damping: float = 0.25
    restitution: float = 0.35
    r_contact: float = 0.045
    # Tuned so the end effector can traverse the workspace in ~10 steps and
    # the behaviour policy actually makes contact. With the original values
    # the effector moved 0.0096 per step against a 0.09 standoff and touched
    # the object on 1% of steps, which left almost no contact events for the
    # certificate to be interesting about.
    ee_lag: float = 0.80          # first-order tracking of the velocity command
    action_scale: float = 0.90
    process_noise: float = 0.0015
    obs_noise: float = 0.0008
    tactile_noise: float = 0.01
    max_steps: int = 100
    workspace: float = 1.0

    def scaled(self, mass=1.0, friction=1.0, damping=1.0, restitution=1.0) -> "PlanarPushParams":
        """Parametric shift, PHANTASM axis 1: multiplicative scaling."""
        return replace(
            self,
            mass=self.mass * mass,
            friction=self.friction * friction,
            damping=self.damping * damping,
            restitution=self.restitution * restitution,
        )


class PlanarPush:
    """Environment protocol implementation with exact reset-to-state."""

    task_spec = TASK_SPEC
    action_dim = 2

    def __init__(self, params: PlanarPushParams | None = None, goal_radius: float = 0.06):
        self.p = params or PlanarPushParams()
        self.goal_radius = goal_radius
        self._rng = np.random.default_rng(0)
        self._state = np.zeros(8)     # [o_x,o_y,o_vx,o_vy,e_x,e_y,goal_x,goal_y]
        self._t = 0
        self._contact = False
        self._tactile = 0.0
        # Sensing degradation (PHANTASM axis 5) is applied at the observation
        # boundary, never to the underlying state.
        self.obs_dropout = 0.0
        self.obs_blur = 0.0
        self.action_latency = 0
        self._action_queue: list[Array] = []

    # -- lifecycle ----------------------------------------------------------- #
    def reset(self, rng: np.random.Generator | None = None) -> dict:
        if rng is not None:
            self._rng = rng
        r = self._rng
        obj = r.uniform(-0.35, 0.35, size=2)
        ee = obj + r.uniform(-0.30, -0.12, size=2) * r.choice([-1.0, 1.0], size=2)
        goal = obj + r.uniform(0.18, 0.40, size=2) * r.choice([-1.0, 1.0], size=2)
        self._state = np.concatenate([obj, np.zeros(2), ee, goal])
        self._t = 0
        self._contact = False
        self._tactile = 0.0
        self._action_queue = [np.zeros(2) for _ in range(self.action_latency)]
        return self.observe()

    def step(self, action: Array) -> tuple[dict, float, bool, dict]:
        a = np.clip(np.asarray(action, dtype=np.float64).ravel(), -1.0, 1.0)

        # Actuation latency: the commanded action reaches the plant late.
        if self.action_latency > 0:
            self._action_queue.append(a)
            a = self._action_queue.pop(0)

        o, v, e, g = self._state[:2], self._state[2:4], self._state[4:6], self._state[6:8]
        p, dt = self.p, self.p.dt

        # End effector: first-order lag toward the commanded velocity.
        e_new = e + p.ee_lag * p.action_scale * a * dt / p.dt * dt

        # Contact test and impulse.
        delta = o - e_new
        dist = float(np.linalg.norm(delta))
        contact = dist < p.r_contact
        v_new = v.copy()
        if contact and dist > 1e-9:
            n = delta / dist
            v_cmd = p.action_scale * a
            approach = float(np.dot(v_cmd, n))
            if approach > 0:
                impulse = (1.0 + p.restitution) * approach / p.mass
                v_new = v_new + impulse * n
            penetration = p.r_contact - dist
            v_new = v_new + 0.5 * penetration * n / (p.mass * dt)

        # Object: drag, then integrate.
        speed = float(np.linalg.norm(v_new))
        if speed > 1e-9:
            drag = min(speed, p.friction * dt + p.damping * speed * dt)
            v_new = v_new * (1.0 - drag / speed)
        o_new = o + v_new * dt

        if p.process_noise > 0:
            o_new = o_new + self._rng.normal(0, p.process_noise, size=2)
            v_new = v_new + self._rng.normal(0, p.process_noise * 2, size=2)

        w = p.workspace
        o_new = np.clip(o_new, -w, w)
        e_new = np.clip(e_new, -w, w)

        self._state = np.concatenate([o_new, v_new, e_new, g])
        self._t += 1
        self._contact = bool(contact)
        self._tactile = (
            float(np.linalg.norm(v_new - v)) if contact else 0.0
        ) + abs(self._rng.normal(0, p.tactile_noise))

        dist_goal = float(np.linalg.norm(o_new - g))
        reward = -dist_goal - 0.01 * float(np.sum(a**2))
        success = dist_goal < self.goal_radius
        done = success or self._t >= p.max_steps
        info = {
            "contact": self._contact,
            "tactile": self._tactile,
            "dist_goal": dist_goal,
            "success": success,
            # An unsafe contact is a high-impulse one: the thing a trust
            # horizon is supposed to let the planner avoid.
            "unsafe": bool(contact and float(np.linalg.norm(v_new - v)) > 0.45),
        }
        return self.observe(), reward, done, info

    # -- observation and state ----------------------------------------------- #
    def observe(self) -> dict:
        y = self.task_state()
        if self.obs_blur > 0:
            y = y + self._rng.normal(0, self.obs_blur, size=y.shape)
        if self.obs_dropout > 0 and self._rng.random() < self.obs_dropout:
            y = y.copy()
            y[:4] = 0.0            # occlusion: the object is not seen this step
        if self.p.obs_noise > 0:
            y = y + self._rng.normal(0, self.p.obs_noise, size=y.shape)
        return {
            "y": y,
            "goal": self._state[6:8].copy(),
            "tactile": self._tactile,
            "contact": self._contact,
            "t": self._t,
        }

    def task_state(self) -> Array:
        """y -- the readout error is measured in, never pixels.

        Object position, end-effector position, contact indicator. The
        object's velocity drives the dynamics but is not task-defining, so it
        lives in the simulator state and not in y.
        """
        return np.array([
            self._state[0], self._state[1],          # object position
            self._state[4], self._state[5],          # end-effector position
            1.0 if self._contact else 0.0,           # contact indicator
        ])

    def get_state(self) -> Array:
        return np.concatenate([self._state, [self._t]])

    def set_state(self, state: Array) -> None:
        s = np.asarray(state, dtype=np.float64)
        self._state = s[:8].copy()
        self._t = int(s[8]) if s.shape[0] > 8 else self._t

    def supports_reset_to_state(self) -> bool:
        return True

    # -- scripted behaviour policy -------------------------------------------- #
    def behaviour_action(self, noise: float = 0.35) -> Array:
        """A simple push-toward-goal policy, used to collect D_train and D_cal.

        Deliberately not the planner: the whole point of §4.4 is that
        calibrating on behaviour-policy rollouts and deploying under a planner
        breaks exchangeability, and that gap can only be measured if the two
        are genuinely different distributions.
        """
        o, e, g = self._state[:2], self._state[4:6], self._state[6:8]
        to_goal = g - o
        n = np.linalg.norm(to_goal)
        if n < 1e-9:
            return self._rng.normal(0, noise, size=2)

        # Stage on the far side of the object from the goal, at a standoff
        # INSIDE the contact radius, so the approach phase terminates in
        # contact rather than hovering just outside it.
        standoff = 0.6 * self.p.r_contact
        behind = o - standoff * to_goal / n
        gap = behind - e
        if np.linalg.norm(gap) > 0.5 * self.p.r_contact:
            a = gap * 12.0                      # approach
        else:
            a = to_goal / n * 1.5               # push through the object
        return np.clip(a + self._rng.normal(0, noise, size=2), -1.0, 1.0)
