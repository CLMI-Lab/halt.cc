#!/usr/bin/env python3
"""Step 0 -- train the world model, then collect anchors (Algorithm G.1, 1-4).

Writes:
  checkpoints/world_model_<kind>.pt
  results/anchors_<kind>.npz          phi, d, episode ids, contact, appearance
  results/grounding_<kind>.csv        CAS and rho probes (Table 7 analogue)
  results/world_model_training.csv
"""

from __future__ import annotations

import numpy as np

from _common import (ALPHA, H_MAX, K_ROLLOUTS, banner, base_parser, get_run,
                     make_env, make_extractor, make_model)

from halt.collect import CollectConfig, anchors_to_arrays, collect_anchors
from halt.io import write_csv, write_json


def collect_transitions(env, n_episodes: int, seed: int):
    """Transitions for fitting the world model. Behaviour policy, no planner."""
    rng = np.random.default_rng(seed)
    Y, C, A, YN = [], [], [], []
    for _ in range(n_episodes):
        obs = env.reset(rng)
        for _ in range(env.p.max_steps):
            y = env.task_state()
            a = env.behaviour_action()
            obs, _, done, _ = env.step(a)
            Y.append(y)
            C.append(obs["goal"])
            A.append(a)
            YN.append(env.task_state())
            if done:
                break
    return map(np.asarray, (Y, C, A, YN))


def main() -> None:
    p = base_parser(__doc__)
    p.add_argument("--kind", default="ensemble", choices=["ensemble", "video"])
    p.add_argument("--train-episodes", type=int, default=260)
    p.add_argument("--anchor-episodes", type=int, default=200)
    p.add_argument("--anchors-per-episode", type=int, default=8)
    args = p.parse_args()

    run = get_run(args)
    env = make_env()
    model = make_model(args.kind, seed=args.seed)

    banner(f"[00] fitting the {args.kind} world model", args.quiet)
    Y, C, A, YN = collect_transitions(env, args.train_episodes, args.seed)
    print(f"  {Y.shape[0]} transitions from {args.train_episodes} episodes")
    hist = model.fit(Y, C, A, YN, verbose=not args.quiet)
    model.save(run.ckpt(f"world_model_{args.kind}.pt"))
    write_csv(run.csv("world_model_training.csv"), hist)

    banner(f"[00] collecting anchors (Algorithm G.1, lines 1-4)", args.quiet)
    extractor = make_extractor(model.capabilities)
    print(f"  feature dim {extractor.dim}; disabled families: "
          f"{sorted(extractor.disabled) or 'none'}")

    cfg = CollectConfig(
        n_episodes=args.anchor_episodes, h_max=H_MAX,
        anchors_per_episode=args.anchors_per_episode,
        n_rollouts=K_ROLLOUTS, seed=args.seed + 1,
        max_episode_steps=env.p.max_steps,
    )
    anchors, diag = collect_anchors(env, model, extractor, cfg, verbose=not args.quiet)
    arr = anchors_to_arrays(anchors)

    np.savez_compressed(run.csv(f"anchors_{args.kind}.npz").with_suffix(".npz"), **arr)
    write_csv(run.csv(f"grounding_{args.kind}.csv"), diag["grounding"])
    write_json(
        run.csv(f"collect_{args.kind}.meta.json"),
        {"config": cfg.__dict__, "n_anchors": diag["n_anchors"],
         "contact_rate": diag["contact_rate"],
         "feature_names": extractor.names(),
         "capabilities": model.capabilities.as_dict()},
    )

    d = arr["d"]
    print(f"\n  {arr['phi'].shape[0]} anchors, feature dim {arr['phi'].shape[1]}")
    print(f"  contact rate during collection: {diag['contact_rate']:.1%}")
    print(f"  d_h quantiles (the Table H.2 analogue):")
    print(f"    {'h':>4} {'mean':>9} {'median':>9} {'p75':>9} {'p90':>9} {'p95':>9}")
    for h in (1, 5, 10, 15, 20, H_MAX):
        c = d[:, h - 1]
        print(f"    {h:>4} {c.mean():9.4f} {np.median(c):9.4f} "
              f"{np.quantile(c,.75):9.4f} {np.quantile(c,.90):9.4f} {np.quantile(c,.95):9.4f}")

    rows = [{"h": h, "mean": d[:, h-1].mean(), "median": float(np.median(d[:, h-1])),
             "p75": float(np.quantile(d[:, h-1], .75)),
             "p90": float(np.quantile(d[:, h-1], .90)),
             "p95": float(np.quantile(d[:, h-1], .95))}
            for h in range(1, H_MAX + 1)]
    write_csv(run.csv(f"error_distribution_{args.kind}.csv"), rows)


if __name__ == "__main__":
    main()
