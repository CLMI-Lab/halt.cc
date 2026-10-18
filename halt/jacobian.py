"""Model and true action-Jacobians, Eq. (3).

    J_hat_h(s,a) = d y_hat_{t+h} / d a_t      autodiff through the latent rollout
    J*_h(s,a)    = d y_{t+h}     / d a_t      central differences in the simulator

The true Jacobian costs 2*d_a simulator resets per probe and requires
reset-to-state, which is why PHANTASM provides that harness and why J* is
computed on a subsampled set rather than at every anchor.

The asymmetry between the two is the point: J_hat is a *deployment* quantity
(it feeds the Jacobian feature family and needs no ground truth), whereas J*
is an *evaluation* quantity (it is needed only to score CAS and rho offline).
"""

from __future__ import annotations

import numpy as np

Array = np.ndarray

__all__ = ["finite_difference_jacobian", "torch_rollout_jacobian"]


def finite_difference_jacobian(
    env,
    state: Array,
    actions: Array,
    horizons: list[int] | tuple[int, ...],
    eps: float = 1e-3,
) -> Array:
    """J*_h by central differences in the simulator, with reset-to-state.

    Parameters
    ----------
    env       : an Environment supporting get_state / set_state
    state     : the state to reset to before every probe
    actions   : (H, d_a) the nominal action sequence; only a_t (row 0) is perturbed
    horizons  : which h to return
    eps       : step in *normalised* action units (the paper uses 1e-3)

    Returns (len(horizons), d_y, d_a). Costs 2*d_a resets.
    """
    if not env.supports_reset_to_state():
        raise RuntimeError(
            "true Jacobians need reset-to-state; this environment does not "
            "provide it, so CAS and rho cannot be evaluated here"
        )
    actions = np.asarray(actions, dtype=np.float64)
    H = actions.shape[0]
    d_a = actions.shape[1]
    d_y = env.task_spec.dim
    hs = list(horizons)
    h_max = max(hs)
    if h_max > H:
        raise ValueError(f"need {h_max} actions, got {H}")

    def rollout_true(a0: Array) -> Array:
        env.set_state(state)
        acts = actions.copy()
        acts[0] = a0
        ys = np.zeros((h_max, d_y))
        for h in range(h_max):
            env.step(acts[h])
            ys[h] = env.task_state()
        return ys

    J = np.zeros((len(hs), d_y, d_a))
    a0 = actions[0]
    for j in range(d_a):
        e = np.zeros(d_a)
        e[j] = eps
        y_plus = rollout_true(a0 + e)
        y_minus = rollout_true(a0 - e)
        for k, h in enumerate(hs):
            J[k, :, j] = (y_plus[h - 1] - y_minus[h - 1]) / (2.0 * eps)
    return J


def torch_rollout_jacobian(
    step_fn,
    readout_fn,
    z0,
    actions,
    horizons: list[int] | tuple[int, ...],
):
    """J_hat_h by reverse-mode autodiff through the latent rollout.

    Parameters
    ----------
    step_fn    : (z, a) -> z'     differentiable latent transition
    readout_fn : z -> y           differentiable task readout h_omega
    z0         : torch tensor, the initial latent
    actions    : torch tensor (H, d_a); gradients are taken w.r.t. row 0
    horizons   : which h to return

    Returns a numpy array (len(horizons), d_y, d_a).

    Stochastic rollouts are differentiated through the reparameterised sample,
    so the noise is held fixed across the backward passes and the Jacobian is
    the derivative of *this* imagined trajectory rather than of its mean.
    """
    import torch

    hs = list(horizons)
    h_max = max(hs)
    a = actions.clone().detach().requires_grad_(True)

    z = z0
    ys: dict[int, "torch.Tensor"] = {}
    for h in range(1, h_max + 1):
        z = step_fn(z, a[h - 1])
        if h in hs:
            ys[h] = readout_fn(z)

    d_a = a.shape[1]
    out = []
    for h in hs:
        y = ys[h]
        d_y = y.shape[-1]
        rows = []
        for i in range(d_y):
            g = torch.autograd.grad(
                y[..., i].sum(), a, retain_graph=True, create_graph=False,
                allow_unused=True,
            )[0]
            rows.append(
                torch.zeros(d_a) if g is None else g[0].detach()
            )
        out.append(torch.stack(rows).cpu().numpy())
    return np.stack(out)
