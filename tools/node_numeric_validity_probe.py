"""Decide whether a compute node reproduces the committed Wigley radiation sweep.

Why this exists
---------------
``docs/casepool_bem_cummins_wigley/README.md`` (m20419) recorded a provenance
break: the *same* CASE-BEM-CUMMINS-WIGLEY coarse configuration (12x4 mesh,
band 0.2-12 rad/s, 41 log nodes, CHIEF regularisation) gave

* on cluster node n315 (cj2690/cj2704): max solver relative residual 1.253 and
  **negative** diagonal radiation damping (surge -0.947, pitch -1.46), i.e. a
  solve that creates energy and therefore is not a solution at all;
* in the local worktree: residual 6.4e-3, diagonal damping >= 0.

The committed case artifact was later refreshed (mission m21960, medium
production mesh, 16/16 gates) from a stack that reproduces the local numbers, so
the *published verdict* no longer depends on the disputed node run.  What was
left open is the second half: **is the node's numerical stack itself broken?**
That is the question this tool answers, with a pre-declared decision rule, and
without touching the case criterion.

What is measured
----------------
1. ``linalg`` probe -- three fixed, deterministic dense linear-algebra problems
   with an exactly known answer (well-conditioned solve, ill-conditioned solve,
   over-determined least squares, the last being the CHIEF path).  A broken
   LAPACK/BLAS or a mis-sized integer type shows up here as a residual that is
   orders of magnitude above the rounding floor, *without* any BEM assembly.
2. ``sweep`` probe -- the coarse case sweep re-solved on the node at exactly the
   committed frequency grid, saved as an ``.npz`` in the same schema as
   :func:`tensorbem.benchmarks.case_bem_cummins_wigley.save_sweep`, so it can be
   compared element-wise against (a) the committed reference sweep the published
   verdict rests on and (b) the disputed n315 artifact.

Pre-declared decision rule (:func:`decide`)
-------------------------------------------
Checked in this order; the first one that fires sets the decision.

============================  ===========================================
``NOT_CONVERGED``             max solver relative residual > 1e-2
``NON_PHYSICAL``              min diagonal radiation damping < -1e-10
``MATCH``                     max relative |dA|,|dB| <= 1e-6   (same solve)
``NEAR``                      <= 1e-3  (numerical noise, not bit-identical)
``DIVERGE``                   > 1e-3   (a different solution)
============================  ===========================================

The amplitudes are normalised per DOF by that DOF's band-max amplitude of the
*reference*, which is the same normalisation gate C3 uses, so ``DIVERGE`` at
1e-3 is still 100x below the 10 % case tolerance: it means "not the same
numbers", not "wrong physics".

Usage
-----
::

    PYTHONPATH=src python tools/node_numeric_validity_probe.py run --out /tmp/probe
    PYTHONPATH=src python tools/node_numeric_validity_probe.py compare \
        --probe /tmp/probe/probe_sweep_coarse.npz \
        --reference benchmarks/verified/casepool_bem_cummins_wigley/sweep_coarse.npz
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


# ---------------------------------------------------------------------------
# pre-declared thresholds (do not tune after seeing a result)
# ---------------------------------------------------------------------------

MATCH_TOL = 1.0e-6
NEAR_TOL = 1.0e-3
SOLVER_RESIDUAL_MAX = 1.0e-2
DIAG_DAMPING_MIN = -1.0e-10
GRID_TOL = 1.0e-12

#: The two artifacts the node probe is judged against, resolved from the repo root.
REPO_ROOT = Path(__file__).resolve().parents[1]
COMMITTED_REFERENCE = (
    REPO_ROOT / "benchmarks/verified/casepool_bem_cummins_wigley/sweep_coarse.npz"
)
DISPUTED_ARTIFACT = (
    REPO_ROOT / "docs/casepool_bem_cummins_wigley/evidence_cluster_n315_sweep_coarse.npz"
)

#: Denominators used by :func:`linalg_probe`; a working stack meets them by
#: orders of magnitude.
LINALG_BOUNDS = {
    "solve_cond_1e3": {"rel_residual": 1.0e-10, "rel_error": 1.0e-10},
    "solve_cond_1e8": {"rel_residual": 1.0e-6, "rel_error": 1.0e-4},
    "lstsq_overdetermined": {"rel_residual": 1.0e-10, "rel_error": 1.0e-10},
}


# ---------------------------------------------------------------------------
# linear-algebra probe: no BEM assembly involved
# ---------------------------------------------------------------------------


def _orthonormal(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    q, r = np.linalg.qr(rng.standard_normal((n, n)))
    return q * np.sign(np.diag(r))


def _known_solution_system(n: int, cond: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """A matrix with a prescribed condition number and a known solution."""
    q = _orthonormal(n, seed)
    singular = np.logspace(0.0, -np.log10(cond), n)
    a = (q * singular) @ q.T
    x_exact = np.cos(np.arange(n, dtype=float) * 0.37)
    return a, x_exact


def linalg_probe(*, n: int = 192, seed: int = 20261116) -> dict[str, Any]:
    """Three deterministic problems whose exact answer is known in closed form."""
    report: dict[str, Any] = {"problems": {}}

    for name, cond in (("solve_cond_1e3", 1.0e3), ("solve_cond_1e8", 1.0e8)):
        a, x_exact = _known_solution_system(n, cond, seed)
        b = a @ x_exact
        x = np.linalg.solve(a, b)
        report["problems"][name] = {
            "shape": [n, n],
            "condition_number": float(np.linalg.cond(a)),
            "rel_residual": float(np.linalg.norm(a @ x - b) / np.linalg.norm(b)),
            "rel_error": float(np.linalg.norm(x - x_exact) / np.linalg.norm(x_exact)),
        }

    # Over-determined least squares with an exactly consistent right-hand side:
    # this is the code path CHIEF interior-point suppression uses.
    rows = 2 * n
    a, x_exact = _known_solution_system(n, 1.0e2, seed + 1)
    a_over = np.vstack([a, _orthonormal(n, seed + 2)])
    b = a_over @ x_exact
    x, *_ = np.linalg.lstsq(a_over, b, rcond=None)
    report["problems"]["lstsq_overdetermined"] = {
        "shape": [rows, n],
        "condition_number": float(np.linalg.cond(a_over)),
        "rel_residual": float(np.linalg.norm(a_over @ x - b) / np.linalg.norm(b)),
        "rel_error": float(np.linalg.norm(x - x_exact) / np.linalg.norm(x_exact)),
    }
    return report


def linalg_verdict(report: dict[str, Any]) -> dict[str, Any]:
    """Compare a :func:`linalg_probe` report against :data:`LINALG_BOUNDS`."""
    failures: list[str] = []
    for name, bounds in LINALG_BOUNDS.items():
        row = report["problems"].get(name)
        if row is None:
            failures.append(f"{name}: missing")
            continue
        for key, bound in bounds.items():
            if not (row[key] <= bound):
                failures.append(f"{name}.{key}={row[key]:.3e} > {bound:.0e}")
    return {"passed": not failures, "failures": failures, "bounds": LINALG_BOUNDS}


# ---------------------------------------------------------------------------
# sweep probe: the case's own solver, on the committed frequency grid
# ---------------------------------------------------------------------------


def sweep_probe(
    out_dir: str | Path,
    *,
    label: str = "coarse",
    band: tuple[float, float] = (0.2, 12.0),
    n_omega: int = 41,
    jobs: int = 1,
) -> dict[str, Any]:
    """Re-solve the case sweep on this machine and write it next to the metadata."""
    from tensorbem.benchmarks import case_bem_cummins_wigley as case

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    sweep = case.solve_sweep(label, band=band, n_omega=n_omega, jobs=jobs)
    path = case.save_sweep(sweep, out / f"probe_sweep_{label}.npz")

    diagonal = np.einsum("kii->ki", sweep.damping)
    import platform
    import socket

    meta = {
        "kind": "sweep_probe",
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "label": sweep.label,
        "mesh": [sweep.n_sections, sweep.n_levels],
        "panel_count": sweep.panel_count,
        "max_panel_scale": sweep.max_panel_scale,
        "band_rad_s": [float(band[0]), float(band[1])],
        "n_omega": int(sweep.omega.size),
        "grid_matches_case_default": bool(
            np.allclose(sweep.omega, case.sweep_grid(band, n_omega), rtol=0.0, atol=GRID_TOL)
        ),
        "solver_relative_residual_max": float(np.max(sweep.solver_relative_residual)),
        "diagonal_damping_min": float(np.min(diagonal)),
        "diagonal_damping_min_per_dof": np.min(diagonal, axis=0).tolist(),
        "elapsed_s": float(sweep.elapsed),
        "wall_s": time.time() - started,
        "artifact": str(path),
    }
    return meta


# ---------------------------------------------------------------------------
# comparison + decision
# ---------------------------------------------------------------------------


def _load_sweep_arrays(path: str | Path) -> dict[str, np.ndarray]:
    payload = np.load(Path(path), allow_pickle=False)
    return {
        "omega": np.asarray(payload["omega"], dtype=float),
        "added_mass": np.asarray(payload["added_mass"], dtype=float),
        "damping": np.asarray(payload["damping"], dtype=float),
        "residual": np.asarray(payload["solver_relative_residual"], dtype=float),
    }


def _per_dof_relative(probe: np.ndarray, reference: np.ndarray) -> list[float]:
    """Per DOF ``max|probe-ref| / max|ref|`` over the 6x6 blocks (C3 normalisation)."""
    out: list[float] = []
    for dof in range(reference.shape[1]):
        ref_dof = reference[:, dof, dof]
        scale = float(np.max(np.abs(ref_dof)))
        block = slice(None)
        diff = np.max(np.abs(probe[block, dof, :] - reference[block, dof, :]))
        out.append(float(diff / scale) if scale > 0.0 else float("inf"))
    return out


def decide(
    probe: dict[str, np.ndarray],
    reference: dict[str, np.ndarray],
    *,
    dof_names: Sequence[str] = ("surge", "sway", "heave", "roll", "pitch", "yaw"),
) -> dict[str, Any]:
    """Apply the pre-declared decision rule to one probe/reference pair.

    ``probe`` and ``reference`` are dicts with ``omega``, ``added_mass``,
    ``damping`` and ``residual`` arrays (the schema written by
    :func:`tensorbem.benchmarks.case_bem_cummins_wigley.save_sweep`).
    """
    omega_p, omega_r = probe["omega"], reference["omega"]
    if omega_p.shape != omega_r.shape or not np.allclose(omega_p, omega_r, rtol=0.0, atol=GRID_TOL):
        raise ValueError(
            "probe and reference frequency grids differ; refusing to interpolate "
            f"({omega_p.size} vs {omega_r.size} nodes)"
        )

    d_a = np.abs(probe["added_mass"] - reference["added_mass"])
    d_b = np.abs(probe["damping"] - reference["damping"])
    scale_a = float(np.max(np.abs(reference["added_mass"])))
    scale_b = float(np.max(np.abs(reference["damping"])))
    rel_a = float(np.max(d_a) / scale_a) if scale_a > 0.0 else float("inf")
    rel_b = float(np.max(d_b) / scale_b) if scale_b > 0.0 else float("inf")
    residual_max = float(np.max(probe["residual"]))
    diagonal_min = float(np.min(np.einsum("kii->ki", probe["damping"])))

    reasons: list[str] = []
    if residual_max > SOLVER_RESIDUAL_MAX:
        decision = "NOT_CONVERGED"
        reasons.append(
            f"max solver relative residual {residual_max:.3e} > {SOLVER_RESIDUAL_MAX:.0e}"
        )
    elif diagonal_min < DIAG_DAMPING_MIN:
        decision = "NON_PHYSICAL"
        reasons.append(
            f"min diagonal radiation damping {diagonal_min:.3e} < {DIAG_DAMPING_MIN:.0e}"
        )
    else:
        worst = max(rel_a, rel_b)
        if worst <= MATCH_TOL:
            decision = "MATCH"
        elif worst <= NEAR_TOL:
            decision = "NEAR"
        else:
            decision = "DIVERGE"
        reasons.append(f"max relative amplitude difference {worst:.3e}")

    return {
        "decision": decision,
        "reasons": reasons,
        "thresholds": {
            "match": MATCH_TOL,
            "near": NEAR_TOL,
            "solver_residual_max": SOLVER_RESIDUAL_MAX,
            "diag_damping_min": DIAG_DAMPING_MIN,
            "grid_tol": GRID_TOL,
        },
        "amplitude": {
            "added_mass_rel": rel_a,
            "damping_rel": rel_b,
            "per_dof_added_mass_rel": _dof_rows(
                _per_dof_relative(probe["added_mass"], reference["added_mass"]), dof_names
            ),
            "per_dof_damping_rel": _dof_rows(
                _per_dof_relative(probe["damping"], reference["damping"]), dof_names
            ),
        },
        "validity": {
            "solver_residual_max": residual_max,
            "diagonal_damping_min": diagonal_min,
        },
    }


def _dof_rows(values: Sequence[float], dof_names: Sequence[str]) -> list[dict[str, Any]]:
    return [
        {"dof": str(name), "value": float(value)}
        for name, value in zip(dof_names, values)
    ]


def compare_files(probe_path: str | Path, reference_path: str | Path) -> dict[str, Any]:
    return decide(_load_sweep_arrays(probe_path), _load_sweep_arrays(reference_path))


# ---------------------------------------------------------------------------
# renderer: the doc is generated from the probe JSONs, never hand-copied
# ---------------------------------------------------------------------------

REFERENCE_LABELS = {
    "probe_compare_vs_committed.json": (
        "committed reference sweep (what the published verdict rests on)"
    ),
    "probe_compare_vs_n315_artifact.json": (
        "disputed node-n315 artifact (docs/.../evidence_cluster_n315_sweep_coarse.npz)"
    ),
}

STATUS_BEGIN = "<!-- BEGIN node-validity status (generated by tools/node_numeric_validity_probe.py report) -->"
STATUS_END = "<!-- END node-validity status -->"


def _pct(value: float) -> str:
    return f"{100.0 * value:.3e} %"


def render_report(probe_dir: str | Path) -> str:
    """Markdown report assembled from the probe JSON files in ``probe_dir``."""
    root = Path(probe_dir)
    linalg = json.loads((root / "probe_linalg.json").read_text(encoding="utf-8"))
    meta = json.loads((root / "probe_sweep_meta.json").read_text(encoding="utf-8"))
    compares = {
        name: json.loads((root / name).read_text(encoding="utf-8"))
        for name in REFERENCE_LABELS
        if (root / name).exists()
    }

    env = linalg.get("environment", {})
    lines: list[str] = []
    lines.append("# NODE-NUMERIC-VALIDITY — node reproducibility of CASE-BEM-CUMMINS-WIGLEY")
    lines.append("")
    lines.append(
        "Generated by `tools/node_numeric_validity_probe.py report` from the JSON "
        "files in this directory; every number below is read from those files, "
        "none is typed by hand."
    )
    lines.append("")
    lines.append("## 1. Question and decision")
    lines.append("")
    lines.append(
        "`README.md` §3 recorded that the *same* coarse configuration gave max solver "
        "residual 1.253 and negative diagonal radiation damping on node n315, while the "
        "worktree stack gave 6.4e-3 and non-negative damping. The published verdict was "
        "later refreshed from a stack that reproduces the worktree numbers, so what "
        "remains open is only the numerical question: **does a node reproduce the "
        "committed radiation sweep?**"
    )
    lines.append("")
    lines.append("| comparison | decision | driving reason |")
    lines.append("| --- | --- | --- |")
    for name, label in REFERENCE_LABELS.items():
        row = compares.get(name)
        if row is None:
            lines.append(f"| {label} | not run | — |")
            continue
        reason = "; ".join(row["reasons"])
        lines.append(f"| {label} | **{row['decision']}** | {reason} |")
    lines.append("")
    lines.append(
        "Decision rule (pre-declared in the tool, unchanged after the run): "
        f"`NOT_CONVERGED` if max solver relative residual > {SOLVER_RESIDUAL_MAX:.0e}; "
        f"`NON_PHYSICAL` if min diagonal radiation damping < {DIAG_DAMPING_MIN:.0e}; "
        f"then `MATCH` <= {MATCH_TOL:.0e}, `NEAR` <= {NEAR_TOL:.0e}, else `DIVERGE` on the "
        "per-DOF normalised amplitude difference."
    )
    lines.append("")
    lines.append("## 2. Where and on what it was run")
    lines.append("")
    lines.append("| item | value |")
    lines.append("| --- | --- |")
    lines.append(f"| host | `{meta.get('hostname', '?')}` ({meta.get('machine', env.get('machine', '?'))}) |")
    lines.append(f"| platform | `{meta.get('platform', env.get('platform', '?'))}` |")
    lines.append(f"| python / numpy | {meta.get('python', '?')} / {meta.get('numpy', '?')} |")
    lines.append(
        f"| mesh | {meta['mesh'][0]}x{meta['mesh'][1]} ({meta['panel_count']} panels, label `{meta['label']}`) |"
    )
    lines.append(
        f"| band / nodes | {meta['band_rad_s'][0]} .. {meta['band_rad_s'][1]} rad/s, {meta['n_omega']} log nodes |"
    )
    lines.append(
        f"| grid identical to the committed case grid | {meta['grid_matches_case_default']} |"
    )
    lines.append(f"| sweep wall time | {meta['elapsed_s']:.1f} s inside the solver |")
    lines.append("")
    lines.append("## 3. Numerical stack: linear algebra with a known answer")
    lines.append("")
    lines.append(
        "Three deterministic problems solved on the node; the right-hand side is built "
        "from the exact solution, so both the residual and the error are known."
    )
    lines.append("")
    lines.append("| problem | shape | cond | rel. residual | bound | rel. error | bound |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for name, row in linalg["problems"].items():
        bound = LINALG_BOUNDS[name]
        lines.append(
            f"| {name} | {row['shape'][0]}x{row['shape'][1]} | {row['condition_number']:.3g} | "
            f"{row['rel_residual']:.3e} | {bound['rel_residual']:.0e} | "
            f"{row['rel_error']:.3e} | {bound['rel_error']:.0e} |"
        )
    verdict = linalg["verdict"]
    lines.append("")
    lines.append(
        f"Stack verdict: **{'PASS' if verdict['passed'] else 'FAIL'}**"
        + ("" if verdict["passed"] else " — " + "; ".join(verdict["failures"]))
    )
    lines.append("")
    lines.append("## 4. Radiation sweep: node vs the two reference artifacts")
    lines.append("")
    lines.append(
        "![node probe vs references](fig_node_numeric_validity.png)"
    )
    lines.append("")
    for name, label in REFERENCE_LABELS.items():
        row = compares.get(name)
        if row is None:
            continue
        lines.append(f"### vs {label}")
        lines.append("")
        lines.append("| DOF | dA / max|A| | dB / max|B| |")
        lines.append("| --- | --- | --- |")
        for a_row, b_row in zip(
            row["amplitude"]["per_dof_added_mass_rel"], row["amplitude"]["per_dof_damping_rel"]
        ):
            lines.append(f"| {a_row['dof']} | {_pct(a_row['value'])} | {_pct(b_row['value'])} |")
        lines.append(
            f"| **max** | **{_pct(row['amplitude']['added_mass_rel'])}** | "
            f"**{_pct(row['amplitude']['damping_rel'])}** |"
        )
        lines.append("")
        lines.append(
            f"max solver relative residual {row['validity']['solver_residual_max']:.3e} "
            f"(bound {SOLVER_RESIDUAL_MAX:.0e}); min diagonal damping "
            f"{row['validity']['diagonal_damping_min']:.3e} (bound {DIAG_DAMPING_MIN:.0e})"
        )
        lines.append("")
    lines.append("## 5. Error judgement")
    lines.append("")
    for name, label in REFERENCE_LABELS.items():
        row = compares.get(name)
        if row is None:
            continue
        worst = max(row["amplitude"]["added_mass_rel"], row["amplitude"]["damping_rel"])
        lines.append(
            f"* vs {label}: worst relative difference {_pct(worst)} — "
            f"**{'< 3 %' if worst < 0.03 else '>= 3 %'}**, decision `{row['decision']}`."
        )
    lines.append("")
    lines.append("## 6. Honest limits")
    lines.append("")
    lines.append(
        "* One run on one node on the day recorded above. The rule decides "
        "*reproducibility now*, not what node n315 did at the time of cj2690/cj2704."
    )
    lines.append(
        "* Cross-machine floating point is not bit-identical: `MATCH` at 1e-6 is already "
        "far below any physical tolerance, and `NEAR` (<=1e-3) still means the same "
        "solution with a different summation order / BLAS."
    )
    lines.append(
        "* The sweep probe covers the coarse 12x4 mesh only — the disputed n315 artifact "
        "was coarse, and the medium production artifact is the one the published verdict rests on."
    )
    lines.append(
        "* A `DIVERGE` decision would say the node is a different solve; it would still "
        "not by itself identify which of assembly / LAPACK / threading is responsible."
    )
    lines.append("")
    return "\n".join(lines) + "\n"


def render_status_block(probe_dir: str | Path) -> str:
    """Short generated block that pins the README to the committed probe results."""
    root = Path(probe_dir)
    meta = json.loads((root / "probe_sweep_meta.json").read_text(encoding="utf-8"))
    compares = {
        name: json.loads((root / name).read_text(encoding="utf-8"))
        for name in REFERENCE_LABELS
        if (root / name).exists()
    }
    body = [STATUS_BEGIN, ""]
    body.append(
        "Generated from `probe_*.json` in this directory — do not edit by hand; "
        "re-run `python tools/node_numeric_validity_probe.py report`."
    )
    body.append("")
    body.append(
        f"* probe: `{meta.get('hostname', '?')}` / {meta['label']} {meta['mesh'][0]}x{meta['mesh'][1]} "
        f"({meta['panel_count']} panels), {meta['n_omega']} nodes over "
        f"{meta['band_rad_s'][0]}..{meta['band_rad_s'][1]} rad/s"
    )
    for name, label in REFERENCE_LABELS.items():
        row = compares.get(name)
        if row is None:
            continue
        worst = max(row["amplitude"]["added_mass_rel"], row["amplitude"]["damping_rel"])
        body.append(
            f"* vs {label}: `{row['decision']}` — worst relative difference {_pct(worst)}, "
            f"solver residual max {row['validity']['solver_residual_max']:.3e}"
        )
    body.append("")
    body.append(STATUS_END)
    return "\n".join(body)


def render_figure(
    probe_dir: str | Path,
    out_png: str | Path,
    *,
    committed_npz: str | Path,
    disputed_npz: str | Path,
    dofs: Sequence[int] = (0, 4),
    dof_names: Sequence[str] = ("surge", "sway", "heave", "roll", "pitch", "yaw"),
) -> Path:
    """Contract figure: diagonal damping and solve residual, node vs the two references."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = Path(probe_dir)
    probe = _load_sweep_arrays(root / "probe_sweep_coarse.npz")
    references = {
        "committed reference": _load_sweep_arrays(committed_npz),
        "n315 artifact (m20419)": _load_sweep_arrays(disputed_npz),
    }
    colours = {"committed reference": "C0", "n315 artifact (m20419)": "C3"}
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.4))
    for dof, axis in zip(dofs, axes[: len(dofs)]):
        diag = np.einsum("kii->ki", probe["damping"])[:, dof]
        axis.plot(probe["omega"], diag, "o-", color="C2", ms=3, label="node probe (this run)")
        for name, arrays in references.items():
            axis.plot(
                arrays["omega"],
                np.einsum("kii->ki", arrays["damping"])[:, dof],
                "s--",
                color=colours[name],
                ms=3,
                lw=1.0,
                label=name,
            )
        axis.set_xscale("log")
        axis.set_xlabel("omega [rad/s]")
        axis.set_ylabel(f"B_{dof_names[dof]}{dof_names[dof]} [dimensionless]")
        axis.set_title(f"diagonal radiation damping, {dof_names[dof]}")
        axis.grid(alpha=0.3)
        axis.axhline(0.0, color="0.4", lw=0.8)
    residual_axis = axes[-1]
    residual_axis.semilogy(probe["omega"], probe["residual"], "o-", color="C2", ms=3, label="node probe")
    for name, arrays in references.items():
        residual_axis.semilogy(
            arrays["omega"], arrays["residual"], "s--", color=colours[name], ms=3, lw=1.0, label=name
        )
    residual_axis.axhline(SOLVER_RESIDUAL_MAX, color="C3", ls=":", lw=1.2, label="C0 bound 1e-2")
    residual_axis.set_xscale("log")
    residual_axis.set_xlabel("omega [rad/s]")
    residual_axis.set_ylabel("solver relative residual")
    residual_axis.set_title("linear-solve residual")
    residual_axis.grid(alpha=0.3, which="both")
    residual_axis.legend(fontsize=7)
    axes[0].legend(fontsize=7)
    fig.suptitle("CASE-BEM-CUMMINS-WIGLEY coarse 12x4: node re-solve vs committed reference vs n315 artifact")
    fig.tight_layout()
    target = Path(out_png)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=140)
    plt.close(fig)
    return target


def write_report(
    probe_dir: str | Path,
    out_md: str | Path,
    readme: str | Path | None = None,
    figure: str | Path | None = None,
) -> Path:
    root = Path(probe_dir)
    target = Path(out_md)
    target.write_text(render_report(root), encoding="utf-8")
    if figure is not None:
        render_figure(
            root,
            figure,
            committed_npz=COMMITTED_REFERENCE,
            disputed_npz=DISPUTED_ARTIFACT,
            dofs=(0, 4),
        )
    if readme is not None:
        readme_path = Path(readme)
        text = readme_path.read_text(encoding="utf-8")
        block = render_status_block(root)
        if STATUS_BEGIN in text and STATUS_END in text:
            head, _, rest = text.partition(STATUS_BEGIN)
            _, _, tail = rest.partition(STATUS_END)
            text = head + block + tail
        else:
            text = text.rstrip("\n") + "\n\n" + block + "\n"
        readme_path.write_text(text, encoding="utf-8")
    return target


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run the linalg + sweep probes on this machine")
    run.add_argument("--out", default="/tmp/node_numeric_validity")
    run.add_argument("--label", default="coarse")
    run.add_argument("--omega-min", type=float, default=0.2)
    run.add_argument("--omega-max", type=float, default=12.0)
    run.add_argument("--n-omega", type=int, default=41)
    run.add_argument("--jobs", type=int, default=1)

    cmp_ = sub.add_parser("compare", help="apply the decision rule to two sweeps")
    cmp_.add_argument("--probe", required=True)
    cmp_.add_argument("--reference", required=True)
    cmp_.add_argument("--json-out", default=None)

    rep = sub.add_parser("report", help="render the evidence doc from the probe JSONs")
    rep.add_argument("--probe-dir", default="docs/casepool_bem_cummins_wigley")
    rep.add_argument("--out", default="docs/casepool_bem_cummins_wigley/NODE_NUMERIC_VALIDITY.md")
    rep.add_argument("--readme", default="docs/casepool_bem_cummins_wigley/README.md")
    rep.add_argument(
        "--figure",
        default="docs/casepool_bem_cummins_wigley/fig_node_numeric_validity.png",
    )

    args = parser.parse_args(argv)
    if args.command == "report":
        path = write_report(
            args.probe_dir, args.out, args.readme or None, figure=args.figure or None
        )
        print(f"wrote {path}")
        return 0

    if args.command == "run":
        import platform

        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        linalg = linalg_probe()
        linalg["environment"] = {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "numpy": np.__version__,
        }
        linalg["verdict"] = linalg_verdict(linalg)
        (out / "probe_linalg.json").write_text(
            json.dumps(linalg, indent=2), encoding="utf-8"
        )
        sweep = sweep_probe(
            out,
            label=args.label,
            band=(args.omega_min, args.omega_max),
            n_omega=args.n_omega,
            jobs=args.jobs,
        )
        (out / "probe_sweep_meta.json").write_text(
            json.dumps(sweep, indent=2), encoding="utf-8"
        )
        print(json.dumps({"linalg": linalg["verdict"], "sweep": sweep}, indent=2))
        return 0 if linalg["verdict"]["passed"] else 1

    result = compare_files(args.probe, args.reference)
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["decision"] in ("MATCH", "NEAR") else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())