#!/usr/bin/env python3
"""In-house refinement ladder of the anchor ``BEM-HULME-HEMISPHERE`` at one ka.

The anchor grades the panel solver at 1024 constant panels and quotes a worst
band error of 1.15 % against the Hulme multipole series.  Those two numbers --
1.15 % (solver, in-house) and 3.36 % (independent code vs reference) -- are
quoted side by side as if they were commensurable, but the solver's own
discretisation error has never been separated from its solution: the case rungs
are 256 and 1024 panels only, i.e. two points, and two points cannot carry an
observed convergence order.

This driver adds the missing rungs at the worst band point (ka = 2, where
B33 carries the largest error) so that ``tensorbem.benchmarks.
hemisphere_independent_convergence`` can fit the same three-parameter
convergence model to the in-house ladder that it fits to the independent code's
ladder, and compare continuum limits rather than discretised values.

The single-cell work is delegated to
``tensorbem.benchmarks.hulme_hemisphere_case.run_cell`` -- no solver code is
duplicated here, so the ladder cannot diverge from the graded case.

Crash-safe bookkeeping
----------------------
The first attempt at this ladder lost everything: the ``subdivisions = 5``
(4096 panel) rung died with ``malloc(): invalid size`` and the payload, which
was written only after the last rung, never reached disk.  Hence:

* the artifact is rewritten after *every* rung (partial progress is evidence);
* a rung that raises is recorded as ``{"subdivisions": s, "error": ...}`` in
  ``failures`` *and then re-raised*, so the process still exits non-zero and
  the job status stays honest;
* ``--resume`` skips rungs already present in the output file, so a crash at
  the expensive rung costs only that rung;
* each row carries peak RSS of the process after the rung, which is what
  separates "out of memory" from "heap corruption" after the fact.

Cost (measured on a 128-core host, one BLAS thread, one cell per process):
64 panels ~0.2 s, 256 ~1.4 s, 1024 ~25 s, 4096 ~7 min.  Run the top rung
through the compute gate, not the control plane.

Usage
-----
    python scripts/hulme_hemisphere_inhouse_ladder.py --ka 2.0 \
        --subdivisions 2 3 4 5 --resume \
        --out benchmarks/verified/casepool_bem_hulme_hemisphere/\
inhouse_ladder_ka2.json
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from tensorbem.benchmarks.hemisphere_independent_convergence import (
    environment_consistency,
)
from tensorbem.benchmarks.hulme_hemisphere_case import run_cell

DEFAULT_OUT = Path(
    "benchmarks/verified/casepool_bem_hulme_hemisphere/inhouse_ladder_ka2.json"
)
QUANTITIES = (
    "A11_dimless",
    "B11_dimless",
    "A33_dimless",
    "B33_dimless",
    "Fx_dimless",
    "Fz_dimless",
)
THREAD_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def _peak_rss_mb() -> float:
    """Peak resident set size of this process, MB (Linux: ru_maxrss in kB)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _code_revision() -> dict[str, Any]:
    """Commit this ladder was computed from, so "same code" is checkable.

    The m21674 environment comparison had to *assert* that two runs shared a
    commit; no artifact recorded one, so the claim could not be audited after
    the fact.  Every row now carries the revision it was produced by, and
    ``dirty`` says whether the tree had uncommitted edits at the time.
    """
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip() or None
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                text=True,
                check=False,
            ).stdout.strip()
        )
    except OSError:  # pragma: no cover - git not installed
        return {"commit": None, "dirty": None}
    return {"commit": commit, "dirty": dirty}


def _skeleton(ka: float) -> dict[str, Any]:
    return {
        "case_code": "BEM-HULME-HEMISPHERE",
        "purpose": (
            "in-house panel-solver refinement ladder at the worst band point, "
            "so that the solver's own convergence order and continuum limit can "
            "be extrapolated and compared with the reference series and with "
            "the independent code's extrapolated limit"
        ),
        "ka": ka,
        "quantities": list(QUANTITIES),
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "host": platform.node(),
            "revision": _code_revision(),
            "thread_env": {v: os.environ.get(v) for v in THREAD_VARS},
            "generated_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(
                timespec="seconds"
            ),
        },
        "rows": [],
        "failures": [],
    }


def _load(path: Path, ka: float) -> dict[str, Any]:
    """Existing payload if it belongs to this ka, else a fresh skeleton."""
    if path.exists():
        payload = json.loads(path.read_text())
        if float(payload.get("ka", float("nan"))) == float(ka):
            payload.setdefault("rows", [])
            payload.setdefault("failures", [])
            return payload
    return _skeleton(ka)


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)  # atomic: a kill cannot leave a truncated artifact


def build(
    ka: float,
    subdivisions: list[int],
    *,
    out: Path = DEFAULT_OUT,
    resume: bool = False,
) -> dict[str, Any]:
    """Run the requested rungs, persisting the artifact after every rung."""
    payload = _load(out, ka) if resume else _skeleton(ka)
    done = {int(r["subdivisions"]) for r in payload["rows"]} if resume else set()
    for sub in subdivisions:
        if sub in done:
            print(f"subdivisions {sub:2d} -- reused from {out}", flush=True)
            continue
        started = time.time()
        try:
            cell = run_cell(ka, sub)
        except BaseException as exc:  # recorded, then re-raised: honest failure
            payload["failures"].append(
                {
                    "subdivisions": sub,
                    "error": f"{type(exc).__name__}: {exc}",
                    "wall_seconds": time.time() - started,
                    "peak_rss_mb": _peak_rss_mb(),
                }
            )
            _write(out, payload)
            raise
        cell["wall_seconds"] = time.time() - started
        cell["peak_rss_mb"] = _peak_rss_mb()
        # Per-row provenance, not just per-file: a ladder can legitimately be
        # assembled from rungs computed in different processes (or on different
        # hosts, when the expensive rung goes through the compute gate), and the
        # reader must be able to see which machine produced which rung without
        # trusting the file-level ``environment`` block.
        cell["host"] = platform.node()
        cell["numpy"] = np.__version__
        cell["revision"] = _code_revision()["commit"]
        payload["rows"] = sorted(
            [r for r in payload["rows"] if int(r["subdivisions"]) != sub] + [cell],
            key=lambda r: r["panels"],
        )
        _write(out, payload)
        print(
            f"subdivisions {sub:2d} = {cell['panels']:5d} panels  "
            f"A33 {cell['A33_dimless']:.6f}  B33 {cell['B33_dimless']:.6f}  "
            f"{cell['wall_seconds']:.1f} s  peak RSS {cell['peak_rss_mb']:.0f} MB",
            flush=True,
        )
    return payload


def _numeric_environment(row: dict[str, Any]) -> tuple[str | None, str | None]:
    """The part of a rung's provenance that can change its arithmetic."""
    numpy_version = row.get("numpy")
    revision = row.get("revision")
    return (
        None if numpy_version is None else str(numpy_version),
        None if revision is None else str(revision),
    )


def _replicate_entry(
    row: dict[str, Any], canonical: dict[str, Any], quantities: Sequence[str]
) -> dict[str, Any]:
    """A same-panel rung computed in a different numeric environment.

    Kept as a *replicate* of the ladder rung instead of a second rung: two
    measurements of the same discretisation are not two refinement steps.  What
    they measure is whether the environments agree, which is exactly the control
    the mixed ladder needs before its refinement shape may be read.
    """
    values = {q: float(row[q]) for q in quantities if q in row}
    return {
        "host": row.get("host"),
        "numpy": row.get("numpy"),
        "revision": row.get("revision"),
        "values": values,
        "delta_vs_rung": {
            q: abs(values[q] / float(canonical[q]) - 1.0) for q in values
        },
        "radiation_relative_residual": row.get("radiation_relative_residual"),
        "solve_warnings": list(row.get("solve_warnings") or []),
    }


def merge_ladders(
    *paths: Path, quantities: Sequence[str] = QUANTITIES, allow_replicates: bool = True
) -> dict[str, Any]:
    """Combine single-rung artifacts from separate processes into one ladder.

    A crash at the top rung must not force the cheap rungs to be recomputed on
    a different host, but the rungs then come from different processes; the
    per-row provenance (host, numpy, code revision, thread env, RSS) travels
    with each row, and every contributing file is named in ``provenance`` so the
    mixture stays auditable.

    Two rows for the *same* panel count are only accepted when
    ``allow_replicates`` is set and their numeric environments (numpy version,
    code revision) differ: they are then stored as ``replicates`` of the ladder
    rung -- a cross-environment control, not a refinement step.  A genuine
    duplicate (same panel count, same environment) is still an error, because
    nothing can justify it.
    """
    merged = _skeleton(0.0)
    merged["rows"] = []
    merged["failures"] = []
    merged["provenance"] = []
    ka: float | None = None
    raw: list[dict[str, Any]] = []
    for path in paths:
        payload = json.loads(Path(path).read_text())
        this_ka = float(payload["ka"])
        if ka is None:
            ka = this_ka
            merged["ka"] = this_ka
        elif this_ka != ka:
            raise ValueError(f"{path}: ka {this_ka} != {ka}")
        file_env = payload.get("environment", {})
        for row in payload.get("rows", []):
            # older rungs predate per-row provenance: stamp them from the
            # file-level environment and say so, rather than inventing a host
            if not row.get("host"):
                row["host"] = file_env.get("host")
                row["host_source"] = "file_environment"
            if not row.get("numpy"):
                row["numpy"] = file_env.get("numpy")
            if not row.get("revision"):
                revision = file_env.get("revision")
                row["revision"] = (
                    revision.get("commit") if isinstance(revision, dict) else revision
                )
                if row["revision"]:
                    row["revision_source"] = "file_environment"
            raw.append(row)
        merged["failures"].extend(payload.get("failures", []))
        merged["provenance"].append(
            {
                "file": str(path),
                "rows": [int(r["subdivisions"]) for r in payload.get("rows", [])],
                "host": payload.get("environment", {}).get("host"),
                "numpy": payload.get("environment", {}).get("numpy"),
                "revision": (
                    payload.get("environment", {}).get("revision", {}) or {}
                ).get("commit")
                if isinstance(payload.get("environment", {}).get("revision"), dict)
                else payload.get("environment", {}).get("revision"),
            }
        )

    groups: dict[int, list[dict[str, Any]]] = {}
    for row in raw:
        groups.setdefault(int(row["subdivisions"]), []).append(row)
    for sub, rows in groups.items():
        environments = {_numeric_environment(r) for r in rows}
        if len(rows) > 1 and len(environments) == 1:
            raise ValueError(
                f"duplicate rung subdivisions={sub}: {len(rows)} copies from the "
                "same numeric environment cannot both be the ladder rung"
            )
        if len(environments) > 1 and not allow_replicates:
            raise ValueError(
                f"duplicate rung subdivisions={sub} from different numeric "
                f"environments {sorted(map(str, environments))}"
            )
        # the canonical rung is the one that can testify to its own health: a
        # recorded residual beats an unrecorded one, otherwise file order wins
        canonical = max(
            rows,
            key=lambda r: (r.get("radiation_relative_residual") is not None, -rows.index(r)),
        )
        canonical = dict(canonical)
        replicates = [
            _replicate_entry(r, canonical, quantities)
            for r in rows
            if r is not canonical and _numeric_environment(r) != _numeric_environment(canonical)
        ]
        if replicates:
            existing = list(canonical.get("replicates") or [])
            canonical["replicates"] = existing + replicates
        merged["rows"].append(canonical)
    merged["rows"] = sorted(merged["rows"], key=lambda r: r["panels"])
    merged["environment_consistency"] = environment_consistency(merged["rows"])
    return merged


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ka", type=float, default=2.0)
    parser.add_argument("--subdivisions", type=int, nargs="+", default=[2, 3, 4, 5])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="reuse rungs already present in --out instead of recomputing them",
    )
    parser.add_argument(
        "--merge",
        type=Path,
        nargs="+",
        default=None,
        help="merge these single-process ladder artifacts into --out and exit",
    )
    args = parser.parse_args(argv)
    if args.merge:
        payload = merge_ladders(*args.merge)
        _write(args.out, payload)
        print(
            json.dumps(
                {
                    "out": str(args.out),
                    "panels": [r["panels"] for r in payload["rows"]],
                    "failures": [f["subdivisions"] for f in payload["failures"]],
                    "environment_consistency": payload["environment_consistency"],
                }
            )
        )
        return 0
    payload = build(
        args.ka, args.subdivisions, out=args.out, resume=args.resume
    )
    print(
        json.dumps(
            {
                "out": str(args.out),
                "panels": [r["panels"] for r in payload["rows"]],
                "failures": [f["subdivisions"] for f in payload["failures"]],
            }
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - script entry point
    raise SystemExit(main())