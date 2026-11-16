"""Is the C9 band-top slope identifiable? -- span sensitivity on committed artifacts.

Gate ``C9`` reads the log-log slope of the diagonal damping at the band top and
compares it with the algebraic-decay bar ``p > 2.05``.  The slope is fitted over the
last ``window`` samples of the sweep, so the *answer depends on the window*: a short
window measures the local tangent at the band top, a long window measures a secant
across a large part of the band.  If those two disagree by more than the distance to
the bar, the gate is not deciding anything -- the number is a property of the fit,
not of the physics.

This script quantifies that on the artifacts that already exist (no BEM solve):

* ``sweep_extended_probe_coarse.npz`` / ``sweep_extended_probe_medium.npz`` -- the
  earlier 12..20 rad/s probes, which carry the *only* band-top data above 12 rad/s
  in the repository;
* ``sweep_medium.npz`` -- the committed 0.2..12 rad/s sweep, as the 12 rad/s control.

For every artifact and DOF it reports the decay exponent ``p`` (convention of
``tensorbem.causality.local_tail_exponent``: ``B ~ omega^-p``) fitted over increasing
spans, the span in decades, the fit quality, and the spread of ``p`` across spans.

Usage::

    PYTHONPATH=src python3 scripts/cummins_tail_span_sensitivity.py
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

WORKTREE = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKTREE / "src"))

from tensorbem.benchmarks import case_bem_cummins_wigley as case  # noqa: E402

CASE_DIR = WORKTREE / "benchmarks" / "verified" / "casepool_bem_cummins_wigley"
ARTIFACTS = ("sweep_extended_probe_coarse.npz", "sweep_extended_probe_medium.npz", "sweep_medium.npz")
SLOPE_BAR = 2.05
SPANS = (3, 4, 5, 7, 10)


def fit(omega: np.ndarray, damping: np.ndarray) -> dict:
    """Decay exponent of ``B ~ omega^-p`` over the given samples."""
    x = np.log(np.asarray(omega, dtype=float))
    y = np.log(np.asarray(damping, dtype=float))
    if x.size < 3 or not np.all(np.isfinite(y)):
        return {"decay_exponent": float("nan"), "decades": float(x[-1] - x[0]) / np.log(10.0),
                "r_squared": float("nan"), "n": int(x.size)}
    coefficients = np.polyfit(x, y, 1)
    fitted = np.polyval(coefficients, x)
    residual = y - fitted
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return {
        "decay_exponent": -float(coefficients[0]),
        "decades": float(x[-1] - x[0]) / np.log(10.0),
        "r_squared": 1.0 - float(np.sum(residual**2)) / ss_tot if ss_tot > 0 else 1.0,
        "max_relative_residual": float(np.max(np.abs(np.expm1(residual)))),
        "n": int(x.size),
    }


def analyze(path: pathlib.Path) -> dict:
    sweep = case.load_sweep(path)
    rows = []
    for index, name in enumerate(case.DOF_NAMES):
        column = sweep.damping[:, index, index]
        spans = {}
        for span in SPANS:
            span = min(int(span), sweep.omega.size)
            spans[str(span)] = fit(sweep.omega[-span:], column[-span:])
        values = [spans[key]["decay_exponent"] for key in spans]
        values = [value for value in values if np.isfinite(value)]
        rows.append(
            {
                "dof": int(index),
                "dof_name": name,
                "slope_by_span": spans,
                "spread": float(max(values) - min(values)) if len(values) > 1 else float("nan"),
                "verdict_by_span": {
                    key: bool(np.isfinite(spans[key]["decay_exponent"]) and spans[key]["decay_exponent"] > SLOPE_BAR)
                    for key in spans
                },
                "verdict_flips": len(
                    {
                        bool(
                            np.isfinite(spans[key]["decay_exponent"])
                            and spans[key]["decay_exponent"] > SLOPE_BAR
                        )
                        for key in spans
                    }
                )
                > 1,
            }
        )
    return {
        "artifact": path.name,
        "label": sweep.label,
        "mesh": [int(sweep.n_sections), int(sweep.n_levels)],
        "panel_count": int(sweep.panel_count),
        "band": [float(sweep.omega[0]), float(sweep.omega[-1])],
        "n_omega": int(sweep.omega.size),
        "slope_bar": SLOPE_BAR,
        "spans_samples": list(SPANS),
        "per_dof": rows,
    }


def main(argv=None) -> int:
    payload = {
        "question": (
            "gate C9 reads p at the band top; does p depend on the fit span, i.e. is "
            "the C9 verdict identifiable from the committed data?"
        ),
        "artifacts": [],
    }
    for name in ARTIFACTS:
        path = CASE_DIR / name
        if not path.exists():
            print(f"missing artifact {path}")
            continue
        payload["artifacts"].append(analyze(path))
    payload["findings"] = findings(payload["artifacts"])
    target = CASE_DIR / "tail_span_sensitivity.json"
    target.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {target}\n")
    for block in payload["artifacts"]:
        print(f"{block['artifact']}  mesh {block['mesh']} {block['panel_count']} panels  band {block['band']}")
        print(f"  {'dof':6s} " + "  ".join(f"p({s:>2d})" for s in SPANS) + "   spread  flips")
        for row in block["per_dof"]:
            values = "  ".join(
                f"{row['slope_by_span'][str(min(s, block['n_omega']))]['decay_exponent']:6.2f}"
                for s in SPANS
            )
            print(f"  {row['dof_name']:6s} {values}  {row['spread']:6.2f}  {row['verdict_flips']}")
        print()
    print(json.dumps(payload["findings"], indent=1))
    return 0


def findings(blocks) -> dict:
    flippers = {}
    for block in blocks:
        names = [row["dof_name"] for row in block["per_dof"] if row["verdict_flips"]]
        if names:
            flippers[block["artifact"]] = names
    unstable = {}
    for block in blocks:
        names = [
            row["dof_name"]
            for row in block["per_dof"]
            if np.isfinite(row["spread"]) and row["spread"] > abs(SLOPE_BAR)
        ]
        if names:
            unstable[block["artifact"]] = names
    return {
        "verdict_flips_within_one_artifact": flippers,
        "spread_exceeds_bar": unstable,
        "spans_in_decades_by_artifact": {
            block["artifact"]: {
                str(min(span, block["n_omega"])): block["per_dof"][0]["slope_by_span"][
                    str(min(span, block["n_omega"]))
                ]["decades"]
                for span in SPANS
            }
            for block in blocks
        },
        "reading": (
            "p at the band top changes verdict with the fit span inside a single "
            "artifact, so the C9 per-DOF decision is a property of the estimator, not "
            "of the data; a band-top slope can only be quoted with its span"
            if flippers
            else "p at the band top is stable across fit spans on these artifacts"
        ),
    }


if __name__ == "__main__":
    sys.exit(main())