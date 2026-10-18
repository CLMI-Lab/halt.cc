"""Shared setup for the pipeline scripts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from halt.envs import PlanarPush, PlanarPushParams, TASK_SPEC          # noqa: E402
from halt.features import FeatureConfig, FeatureExtractor              # noqa: E402
from halt.io import Run                                                # noqa: E402
from halt.models import EnsembleConfig, EnsembleWorldModel, VideoWorldModel  # noqa: E402

DEFAULT_RUN = "reference"
H_MAX = 30          # the reference suite uses a shorter H_max than the paper's 50
ALPHA = 0.05
K_ROLLOUTS = 5


def base_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--run", default=DEFAULT_RUN, help="run directory name under results/")
    p.add_argument("--root", default=str(ROOT / "runs"), help="where runs are written")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--quiet", action="store_true")
    return p


def get_run(args) -> Run:
    return Run(args.root, args.run)


def make_env(params: PlanarPushParams | None = None) -> PlanarPush:
    return PlanarPush(params)


def make_extractor(caps, memory_m: int = 5) -> FeatureExtractor:
    return FeatureExtractor(
        TASK_SPEC,
        caps,
        FeatureConfig(h_probe=(1, 5, 10, 20), memory_m=memory_m,
                      jacobian_horizons=(1, 5)),
    )


def make_model(kind: str = "ensemble", seed: int = 0, K: int = K_ROLLOUTS):
    cfg = EnsembleConfig(d_y=TASK_SPEC.dim, d_a=2, d_context=2, K=K, seed=seed,
                         width=192, epochs=25)
    cls = {"ensemble": EnsembleWorldModel, "video": VideoWorldModel}[kind]
    return cls(cfg, TASK_SPEC)


def banner(text: str, quiet: bool = False) -> None:
    if not quiet:
        print("\n" + "=" * 72)
        print(text)
        print("=" * 72)


def mean_sd(x) -> tuple[float, float]:
    a = np.asarray(x, dtype=np.float64)
    return float(a.mean()), float(a.std())
