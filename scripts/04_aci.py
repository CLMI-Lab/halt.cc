#!/usr/bin/env python3
"""Step 4 -- delayed-feedback ACI against static conformal under shift (§9.4).

The miss stream here is generated from REAL anchors: nominal anchors before
the shift, PHANTASM-perturbed anchors after it, each scored against the
certificate at the controller's current effective level. So the comparison is
between two controllers on the same realised error process, not between two
synthetic Bernoulli streams.

The lag is the real one. A rollout launched at step s is only scored once its
own certified horizon has elapsed, so L_t tracks H*_s and varies with the
state, exactly as §4.6 describes.
"""

from __future__ import annotations

import numpy as np

from _common import H_MAX, K_ROLLOUTS, banner, base_parser, get_run, make_extractor

from halt.aci import ACIController, StaticController, lambda_at_level, prop_52_bound
from halt.calibrate import CalibratedHALT
from halt.collect import CollectConfig, anchors_to_arrays, collect_anchors
from halt.conformal import certified_horizon
from halt.envs import StressSpec, make_stressed_env
from halt.io import write_csv
from halt.models import EnsembleWorldModel


def gather(model, extractor, spec, episodes, seed):
    env = make_stressed_env(spec)
    cfg = CollectConfig(n_episodes=episodes, h_max=H_MAX, anchors_per_episode=6,
                        n_rollouts=K_ROLLOUTS, seed=seed, jacobian_every=10_000,
                        max_episode_steps=env.p.max_steps)
    anchors, _ = collect_anchors(env, model, extractor, cfg, stress=spec, verbose=False)
    return anchors_to_arrays(anchors)


def run_controller(controller, phi, d, halt, rng, l_max):
    """Feed the anchor stream through a controller, honouring the real lag."""
    trace = []
    q_all = halt.q_hat(phi)
    n = phi.shape[0]
    for t in range(n):
        lam = (lambda_at_level(halt.calibration.scores, controller.level())
               if isinstance(controller, ACIController) else halt.calibration.lambda_hat)
        q = q_all[t]
        h = certified_horizon(q, lam, halt.tau)
        if h > 0:
            miss = int(np.any(d[t, :h] > halt.tau))
            # The outcome is observable only after the horizon elapses.
            lag = int(min(h, l_max))
        else:
            miss, lag = 0, 0
        if isinstance(controller, ACIController):
            controller.launch(miss, lag)
            controller.step()
        else:
            controller.observe(miss, lag)
        trace.append({"t": t, "alpha_t": controller.level(), "h_star": h,
                      "miss": miss, "lag": lag, "lambda": lam})
    return trace


def main() -> None:
    p = base_parser(__doc__)
    p.add_argument("--kind", default="ensemble")
    p.add_argument("--eta", type=float, default=0.05)
    p.add_argument("--episodes", type=int, default=70)
    p.add_argument("--severity", type=int, default=3)
    p.add_argument("--shift-axis", default="parametric")
    args = p.parse_args()

    run = get_run(args)
    halt = CalibratedHALT.load(run.ckpt(f"halt_{args.kind}"))
    model = EnsembleWorldModel.load(run.ckpt(f"world_model_{args.kind}.pt"))
    extractor = make_extractor(model.capabilities)

    banner("[04] building the miss stream from real anchors", args.quiet)
    pre = gather(model, extractor, None, args.episodes // 2, args.seed + 200)
    post = gather(model, extractor,
                  StressSpec(axis=args.shift_axis, severity=args.severity),
                  args.episodes // 2, args.seed + 201)
    phi = np.concatenate([pre["phi"], post["phi"]])
    d = np.concatenate([pre["d"], post["d"]])
    shift_at = pre["phi"].shape[0]
    print(f"  {shift_at} nominal anchors, then {post['phi'].shape[0]} under "
          f"{args.shift_axis} severity {args.severity}")

    l_max = H_MAX
    rng = np.random.default_rng(args.seed)

    aci = ACIController(halt.alpha, eta=args.eta, l_max=l_max)
    stat = StaticController(halt.alpha)
    t_aci = run_controller(aci, phi, d, halt, rng, l_max)
    t_stat = run_controller(stat, phi, d, halt, rng, l_max)

    for name, trace in (("aci", t_aci), ("static", t_stat)):
        for r in trace:
            r["controller"] = name
    write_csv(run.csv("aci_trace.csv"), t_aci + t_stat,
              meta={"shift_at": shift_at, "eta": args.eta, "l_max": l_max})

    def post_rate(trace):
        m = [r["miss"] for r in trace[shift_at:]]
        return float(np.mean(m)) if m else float("nan")

    s = aci.summary()
    rows = [
        {"controller": "ACI (Eq. 12, delayed)", "alpha_target": halt.alpha,
         "miss_rate_overall": s["miss_rate"], "miss_rate_post_shift": post_rate(t_aci),
         "deviation": s["deviation"], "prop_5_2_bound": s["prop_5_2_bound"],
         "bound_respected": s["bound_respected"],
         "telescoping_residual": s["telescoping_residual"],
         "alpha_final": s["alpha_final"], "T_fed": s["T_fed"], "l_max": l_max, "eta": args.eta},
        {"controller": "static conformal", "alpha_target": halt.alpha,
         "miss_rate_overall": stat.trace.miss_rate, "miss_rate_post_shift": post_rate(t_stat),
         "deviation": abs(stat.trace.miss_rate - halt.alpha),
         "prop_5_2_bound": "", "bound_respected": "", "telescoping_residual": "",
         "alpha_final": halt.alpha, "T_fed": len(t_stat), "l_max": 0, "eta": 0.0},
    ]
    write_csv(run.csv("aci_summary.csv"), rows)

    banner("[04] result", args.quiet)
    print(f"  target alpha                 = {halt.alpha:.3f}")
    print(f"  ACI    miss rate (overall)   = {s['miss_rate']:.4f}")
    print(f"  ACI    miss rate (post-shift)= {post_rate(t_aci):.4f}")
    print(f"  static miss rate (overall)   = {stat.trace.miss_rate:.4f}")
    print(f"  static miss rate (post-shift)= {post_rate(t_stat):.4f}")
    print(f"  Prop. 5.2 bound at T={s['T_fed']}, L_max={l_max}: {s['prop_5_2_bound']:.4f}")
    print(f"  ACI deviation {s['deviation']:.4f} -> bound respected: {s['bound_respected']}")
    print(f"  telescoping residual = {s['telescoping_residual']:.2e} (must be ~0)")
    print(f"  alpha_t drifted {halt.alpha:.3f} -> {s['alpha_final']:.4f}")
    if s["level_saturated_frac"] > 0.01:
        print(f"  WARNING: alpha_t left [0,1] on {s['level_saturated_frac']:.1%} of steps; "
              "the self-bounding argument behind Prop. 5.2 is not operating here")


if __name__ == "__main__":
    main()
