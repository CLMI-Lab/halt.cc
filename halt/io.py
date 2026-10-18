"""CSV and checkpoint plumbing, with provenance stamped into every artefact.

Every CSV this package writes carries a `provenance` column and a sidecar
`.meta.json` recording the code version, the git commit if there is one, the
config, the seed, and -- most importantly -- whether the row came from the
reference environment or from a real MuJoCo task. A results file that does not
say which is which is a liability next to a paper about honest uncertainty.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

__all__ = ["write_csv", "write_json", "provenance_block", "Run"]

REFERENCE_SUITE = "planar-push-reference-suite"


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return "not-a-git-checkout"


def provenance_block(suite: str = REFERENCE_SUITE, **extra) -> dict:
    from . import __version__

    return {
        "suite": suite,
        "halt_version": __version__,
        "git_commit": _git_commit(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "note": (
            "Produced by the HALT reference implementation. The reference "
            "suite is a synthetic contact-rich environment, NOT one of the "
            "five MuJoCo tasks of the paper's Section 8. These numbers "
            "validate the implementation and the propositions; they are not "
            "a reproduction of the paper's reported results."
            if suite == REFERENCE_SUITE
            else "Produced by the HALT reference implementation on a user-supplied task."
        ),
        **extra,
    }


def _jsonable(obj):
    if is_dataclass(obj) and not isinstance(obj, type):
        return {k: _jsonable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def write_json(path: str | Path, payload) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(_jsonable(payload), indent=2))
    return p


def write_csv(
    path: str | Path,
    rows: list[dict],
    suite: str = REFERENCE_SUITE,
    meta: dict | None = None,
) -> Path:
    """Write rows to CSV and a sidecar .meta.json. Column order follows row 0."""
    import csv

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        p.write_text("")
        write_json(p.with_suffix(".meta.json"), {"provenance": provenance_block(suite), "rows": 0})
        return p

    cols: list[str] = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    if "provenance" not in cols:
        cols.append("provenance")

    with p.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            out = {k: _fmt(r.get(k, "")) for k in cols}
            out["provenance"] = suite
            w.writerow(out)

    write_json(
        p.with_suffix(".meta.json"),
        {"provenance": provenance_block(suite), "rows": len(rows),
         "columns": cols, **( {"config": meta} if meta else {} )},
    )
    return p


def _fmt(v):
    if isinstance(v, float):
        return f"{v:.6g}"
    if isinstance(v, (np.floating,)):
        return f"{float(v):.6g}"
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    return v


class Run:
    """A results directory with a fixed layout, created once per experiment."""

    def __init__(self, root: str | Path, name: str):
        self.root = Path(root) / name
        self.name = name
        for sub in ("results", "checkpoints", "figures", "logs"):
            (self.root / sub).mkdir(parents=True, exist_ok=True)

    def csv(self, filename: str) -> Path:
        return self.root / "results" / filename

    def ckpt(self, filename: str) -> Path:
        return self.root / "checkpoints" / filename

    def fig(self, filename: str) -> Path:
        return self.root / "figures" / filename

    def log(self, filename: str) -> Path:
        return self.root / "logs" / filename

    def __repr__(self) -> str:
        return f"Run({self.root})"
