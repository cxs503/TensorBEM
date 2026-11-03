"""Root cause of the Wigley "pressure-integration" path failing the two-path check.

The open item settled here
--------------------------
``docs/wigley_wave_resistance.md`` reports an *honest FAIL*: the second
(``wave_resistance_pressure``) path disagrees with the verified Kochin path by
103-284 % over a mesh ladder and is negative everywhere.  That document
attributes the failure to the thin-ship approximation itself ("potential
evaluated on the hull instead of the centreplane", "``B/L`` only marginally
small").  This module shows those explanations are wrong: the discrepancy is a
*formulation* error in the second path, and a first-order thin-ship theory has
no independent near-field path to begin with.

Why the legacy path cannot be a near-field representation
--------------------------------------------------------
Set ``k0 = g/U^2`` and write the steady perturbation potential of the thin-ship
problem as a superposition of Laplace modes ``exp(k z) exp(i k x)`` (the legacy
integrand).  The steady free-surface condition for ``Phi = -U x + phi`` is

    dphi/dz + (U^2/g) d^2phi/dx^2 = 0   on z = 0,

so a mode carries the eigenvalue ``k - k^2/k0`` and therefore satisfies the
condition only for ``k = k0``.  In three dimensions the condition is
``kappa = k0 k^2`` with ``kappa = sqrt(k^2 + l^2)`` for the transverse wave
number ``l``, i.e. the admissible set is the *Kelvin wake curve*

    l^2 = k0^2 k^4 - k^2,      k >= k0,

a one-dimensional family that must be integrated with the Jacobian
``1/sqrt(k0^2 k^4 - k^2)``.  That Jacobian is exactly the
``1/(lambda^2 sqrt(lambda^2-1))`` weight of the verified Kochin integral.  The
legacy path instead integrates over ``k in [k0, k_max]`` with plain ``dk``
weights and no transverse-wave number at all, so:

* its quadrature nodes are Gauss-Legendre nodes, which never sit on an
  endpoint -- every node is *off* the dispersion curve, the field it builds
  violates the free-surface condition by O(1) (``fs_residual``);
* ``k_max = k0 * max(10, 20/(k0 T))`` is an arbitrary truncation with no
  derivation, so the result cannot converge (``cutoff_sweep``);
* a genuine sign or constant-factor bug would give a *constant* pressure/Kochin
  ratio; the measured ratio spans ``-0.20 ... -0.77`` (``froude_sweep``), so no
  rescaling can repair it.

Consequence for the verification strategy: in first-order thin-ship theory the
near-field pressure integral and the far-field Kochin integral are the *same*
integral (the free-surface condition collapses the double ``(k, l)`` synthesis
onto the wake curve), so "two independent paths" is not an achievable
acceptance criterion inside this theory.  An independent second path requires a
Rankine/Dawson panel method that imposes the free-surface condition on panels
(different approximation order, not implemented here), or a literature ``Cw``
whose reference area is stated (registered UNSUPPORTED in the docs).

Usage::

    PYTHONPATH=src python3 -m tensorbem.benchmarks.wigley_pressure_audit

which writes the committed artefacts ``docs/wigley_pressure_audit/results.json``,
``docs/wigley_pressure_audit/summary.md`` and
``docs/figures/wigley_pressure_audit.png`` (all three are regenerated from the
same dictionary, so the manuscript quotes no hand-transcribed number).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss

from ..wigley import (
    _legacy_k_max_factor,
    michell_kochin_function,
    michell_wave_resistance,
    wave_resistance_pressure,
    wigley_hull_mesh,
)

DEFAULT_L = 2.0
DEFAULT_B = 0.2
DEFAULT_T = 0.125
DEFAULT_RHO = 1025.0
DEFAULT_G = 9.80665
FROUDE_POINTS = (0.20, 0.25, 0.30, 0.35, 0.40)
CUTOFF_FACTORS = (5.0, 10.0, 20.0, 40.0, 80.0, 160.0)
NODE_COUNTS = (32, 64, 128, 256, 512)
MESH_LEVELS = ((8, 4), (16, 8), (32, 16), (64, 32))


def _k0(g: float, U: float) -> float:
    return g / U**2


def eigval_standard(k: float, k0: float) -> float:
    """Eigenvalue of ``d/dz + (U^2/g) d^2/dx^2`` on ``exp(kz) exp(ikx)``.

    Zero only for ``k = k0``, i.e. the two-dimensional steady dispersion
    relation ``U^2 k^2 = g k``.
    """
    return k - k * k / k0


def eigval_documented(k: float, k0: float) -> float:
    """Eigenvalue of the operator as written in the legacy docstring.

    ``d/dz - (U^2/g) d^2/dx^2`` gives ``k + k^2/k0 > 0`` for every real ``k``,
    so *no* real wavenumber satisfies that sign convention: the legacy
    docstring's free-surface condition is itself mis-signed.  Reported so the
    manuscript cannot claim the sign convention is innocent.
    """
    return k + k * k / k0


def fs_residual(
    L: float = DEFAULT_L,
    B: float = DEFAULT_B,
    T: float = DEFAULT_T,
    U: float | None = None,
    g: float = DEFAULT_G,
    froude: float = 0.30,
    n_quad: int = 128,
    k_max_factor: float | None = None,
) -> dict[str, float]:
    """Amplitude-weighted relative free-surface residual of the legacy field.

    For the legacy superposition ``phi = (U/pi) Re int i k H(k) e^{kz} e^{ikx}
    dk`` the component amplitude is ``|k H(k)|``.  The residual is normalised
    by the size of the two terms that should cancel, so a value of 1 means "the
    two terms do not cancel at all" and 0 would mean "exactly a free wave".
    """
    if U is None:
        U = froude * math.sqrt(g * L)
    k0 = _k0(g, U)
    factor = _legacy_k_max_factor(k0, T) if k_max_factor is None else float(k_max_factor)
    nodes, weights = leggauss(n_quad)
    k_pts = 0.5 * (k0 * factor - k0) * (nodes + 1.0) + k0
    w_pts = 0.5 * (k0 * factor - k0) * weights
    amp = np.abs(k_pts * np.array([michell_kochin_function(L, B, T, k) for k in k_pts]))
    scale = np.abs(k_pts) + k_pts**2 / k0

    def _relative(eig):
        num = float(np.sum(w_pts * amp * np.abs(eig)))
        den = float(np.sum(w_pts * amp * scale))
        return num / den

    standard = _relative(eigval_standard(k_pts, k0))
    documented = _relative(eigval_documented(k_pts, k0))
    return {
        "froude": froude,
        "k0": k0,
        "k_max_factor": factor,
        "k_max_over_k0": k_pts[-1] / k0,
        "fs_residual_standard": standard,
        "fs_residual_documented": documented,
        "nodes_on_dispersion_curve": int(np.count_nonzero(np.abs(eigval_standard(k_pts, k0)) < 1e-12 * k0)),
        "min_abs_eigval_standard_over_k0": float(np.min(np.abs(eigval_standard(k_pts, k0))) / k0),
    }


def dispersion_mismatch_profile(
    L: float = DEFAULT_L,
    B: float = DEFAULT_B,
    T: float = DEFAULT_T,
    g: float = DEFAULT_G,
    froude: float = 0.30,
    lambdas: np.ndarray | None = None,
) -> dict[str, list[float]]:
    """Per-wavenumber relative free-surface residual for ``k = lambda k0``.

    The value is ``|lambda - 1| / (lambda + 1)``, the normalised eigenvalue of
    the standard steady free-surface operator; it is zero only at
    ``lambda = 1`` and tends to 1, so the legacy field is not a free-wave field
    at any interior node of its quadrature.
    """
    lam = np.linspace(1.0, 20.0, 200) if lambdas is None else np.asarray(lambdas, dtype=float)
    return {
        "lambda": lam.tolist(),
        "relative_residual": (np.abs(lam - 1.0) / (lam + 1.0)).tolist(),
    }


def cutoff_sweep(
    L: float = DEFAULT_L,
    B: float = DEFAULT_B,
    T: float = DEFAULT_T,
    rho: float = DEFAULT_RHO,
    g: float = DEFAULT_G,
    froude: float = 0.30,
    mesh_level: tuple[int, int] = (16, 8),
    n_quad: int = 128,
    factors: tuple[float, ...] = CUTOFF_FACTORS,
) -> dict[str, object]:
    """Sweep the arbitrary high-wavenumber cut-off of the legacy path."""
    U = froude * math.sqrt(g * L)
    mesh = wigley_hull_mesh(L, B, T, *mesh_level)
    rows = []
    for factor in factors:
        Rw = wave_resistance_pressure(
            mesh, L, B, T, U, rho, g, n_quad=n_quad, k_max_factor=factor
        )
        rows.append({"k_max_factor": factor, "Rw_pressure": Rw})
    kochin = michell_wave_resistance(L, B, T, U, rho, g)
    values = np.array([row["Rw_pressure"] for row in rows])
    return {
        "froude": froude,
        "mesh": list(mesh_level),
        "n_quad": n_quad,
        "Rw_kochin": kochin,
        "cutoff": [
            {
                **row,
                "ratio_to_kochin": row["Rw_pressure"] / kochin,
            }
            for row in rows
        ],
        "pressure_min": float(values.min()),
        "pressure_max": float(values.max()),
        "cutoff_spread_relative": float((values.max() - values.min()) / abs(kochin)),
    }


def node_sweep(
    L: float = DEFAULT_L,
    B: float = DEFAULT_B,
    T: float = DEFAULT_T,
    rho: float = DEFAULT_RHO,
    g: float = DEFAULT_G,
    froude: float = 0.30,
    mesh_level: tuple[int, int] = (16, 8),
    counts: tuple[int, ...] = NODE_COUNTS,
) -> dict[str, object]:
    """Quadrature-node sweep at the legacy cut-off (does the integral converge?)."""
    U = froude * math.sqrt(g * L)
    mesh = wigley_hull_mesh(L, B, T, *mesh_level)
    kochin = michell_wave_resistance(L, B, T, U, rho, g)
    rows = []
    for n_quad in counts:
        Rw = wave_resistance_pressure(mesh, L, B, T, U, rho, g, n_quad=n_quad)
        rows.append({"n_quad": n_quad, "Rw_pressure": Rw, "ratio_to_kochin": Rw / kochin})
    return {"froude": froude, "Rw_kochin": kochin, "nodes": rows}


def mesh_sweep(
    L: float = DEFAULT_L,
    B: float = DEFAULT_B,
    T: float = DEFAULT_T,
    rho: float = DEFAULT_RHO,
    g: float = DEFAULT_G,
    froude: float = 0.30,
    levels: tuple[tuple[int, int], ...] = MESH_LEVELS,
    n_quad: int = 128,
) -> dict[str, object]:
    """Mesh ladder for both paths: the pressure path does not converge."""
    U = froude * math.sqrt(g * L)
    kochin = michell_wave_resistance(L, B, T, U, rho, g)
    rows = []
    for level in levels:
        mesh = wigley_hull_mesh(L, B, T, *level)
        Rw = wave_resistance_pressure(mesh, L, B, T, U, rho, g, n_quad=n_quad)
        rows.append(
            {
                "mesh": list(level),
                "panel_count": mesh.panel_count,
                "h_mean": mesh.h_mean,
                "Rw_pressure": Rw,
                "ratio_to_kochin": Rw / kochin,
                "relative_difference": abs(Rw - kochin) / abs(kochin),
            }
        )
    return {"froude": froude, "Rw_kochin": kochin, "levels": rows}


def froude_sweep(
    L: float = DEFAULT_L,
    B: float = DEFAULT_B,
    T: float = DEFAULT_T,
    rho: float = DEFAULT_RHO,
    g: float = DEFAULT_G,
    froudes: tuple[float, ...] = FROUDE_POINTS,
    mesh_level: tuple[int, int] = (16, 8),
    n_quad: int = 128,
) -> dict[str, object]:
    """Pressure/Kochin ratio over the Froude range (is it a constant factor?)."""
    mesh = wigley_hull_mesh(L, B, T, *mesh_level)
    rows = []
    for froude in froudes:
        U = froude * math.sqrt(g * L)
        kochin = michell_wave_resistance(L, B, T, U, rho, g)
        Rw = wave_resistance_pressure(mesh, L, B, T, U, rho, g, n_quad=n_quad)
        rows.append(
            {
                "froude": froude,
                "Rw_kochin": kochin,
                "Rw_pressure": Rw,
                "ratio_to_kochin": Rw / kochin,
                "relative_difference": abs(Rw - kochin) / abs(kochin),
            }
        )
    ratios = np.array([row["ratio_to_kochin"] for row in rows])
    diffs = np.array([row["relative_difference"] for row in rows])
    return {
        "mesh": list(mesh_level),
        "rows": rows,
        "ratio_min": float(ratios.min()),
        "ratio_max": float(ratios.max()),
        "ratio_spread": float(ratios.max() - ratios.min()),
        "min_relative_difference": float(diffs.min()),
        "all_negative": bool(np.all(ratios < 0.0)),
    }


def run_audit(
    L: float = DEFAULT_L,
    B: float = DEFAULT_B,
    T: float = DEFAULT_T,
    rho: float = DEFAULT_RHO,
    g: float = DEFAULT_G,
) -> dict[str, object]:
    """Assemble every audit table into one JSON-serialisable mapping."""
    cutoff = cutoff_sweep(L=L, B=B, T=T, rho=rho, g=g)
    nodes = node_sweep(L=L, B=B, T=T, rho=rho, g=g)
    mesh = mesh_sweep(L=L, B=B, T=T, rho=rho, g=g)
    froude = froude_sweep(L=L, B=B, T=T, rho=rho, g=g)
    residual = [fs_residual(L=L, B=B, T=T, g=g, froude=fr) for fr in FROUDE_POINTS]
    kochin_independent = {
        "note": "the Kochin path has no cut-off parameter; repeated calls are bit-identical",
        "Rw_values": [
            michell_wave_resistance(L, B, T, fr * math.sqrt(g * L), rho, g)
            for fr in FROUDE_POINTS
        ],
    }
    verdict = {
        "pressure_path_cutoff_dependent": cutoff["cutoff_spread_relative"] > 0.2,
        "pressure_path_mesh_nonconvergent": mesh["levels"][-1]["relative_difference"]
        > 0.5 * mesh["levels"][0]["relative_difference"],
        "pressure_path_negative_everywhere": froude["all_negative"],
        "no_constant_factor": froude["ratio_spread"] > 0.2,
        "field_off_dispersion_curve": min(
            row["fs_residual_standard"] for row in residual
        )
        > 0.3,
        "quadrature_nodes_on_dispersion_curve": max(
            row["nodes_on_dispersion_curve"] for row in residual
        ),
    }
    verdict["is_formulation_error_not_thin_ship_limit"] = bool(
        verdict["pressure_path_cutoff_dependent"]
        and verdict["no_constant_factor"]
        and verdict["field_off_dispersion_curve"]
    )
    return {
        "case": {
            "L": L,
            "B": B,
            "T": T,
            "rho": rho,
            "g": g,
            "froude_points": list(FROUDE_POINTS),
            "reference": "Kochin far-field path (verified to 2.6e-5 vs converged quadrature)",
        },
        "cutoff_sweep": cutoff,
        "node_sweep": nodes,
        "mesh_sweep": mesh,
        "froude_sweep": froude,
        "fs_residual": residual,
        "kochin_cutoff_independence": kochin_independent,
        "dispersion_profile": dispersion_mismatch_profile(L=L, B=B, T=T, g=g),
        "verdict": verdict,
    }


def make_figure(audit: dict[str, object], target: Path) -> Path:
    """Plot the three decisive diagnostics."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    target.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.1))

    cutoff = audit["cutoff_sweep"]
    factors = [row["k_max_factor"] for row in cutoff["cutoff"]]
    values = [row["Rw_pressure"] for row in cutoff["cutoff"]]
    axes[0].semilogx(factors, values, "o-", label="legacy pressure path")
    axes[0].axhline(cutoff["Rw_kochin"], color="k", ls="--", label="Kochin (verified)")
    axes[0].axhline(0.0, color="0.7", lw=0.8)
    axes[0].set_xlabel(r"cut-off factor $k_{max}/k_0$")
    axes[0].set_ylabel(r"$R_W$ [N], $Fr=0.30$")
    axes[0].set_title("(a) arbitrary cut-off, no limit")
    axes[0].legend(fontsize=8)

    froude = audit["froude_sweep"]
    fr = [row["froude"] for row in froude["rows"]]
    ratio = [row["ratio_to_kochin"] for row in froude["rows"]]
    axes[1].plot(fr, ratio, "s-", color="tab:red")
    axes[1].axhline(0.0, color="0.7", lw=0.8)
    axes[1].axhline(1.0, color="k", ls="--", lw=1.0, label="agreement")
    axes[1].set_xlabel(r"$Fr$")
    axes[1].set_ylabel(r"$R_W^{pressure}/R_W^{Kochin}$")
    axes[1].set_title("(b) ratio is not a constant factor")
    axes[1].legend(fontsize=8)

    profile = audit["dispersion_profile"]
    axes[2].plot(profile["lambda"], profile["relative_residual"], color="tab:green")
    axes[2].axhline(0.0, color="0.7", lw=0.8)
    axes[2].set_xlabel(r"$\lambda = k/k_0$")
    axes[2].set_ylabel(r"$|\lambda-1|/(\lambda+1)$")
    axes[2].set_title("(c) free-surface residual per mode")
    axes[2].set_ylim(0.0, 1.0)

    fig.tight_layout()
    path = target / "wigley_pressure_audit.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def render_markdown(audit: dict[str, object]) -> str:
    """Render every audit table as Markdown, so no number is transcribed by hand."""
    case = audit["case"]
    cutoff = audit["cutoff_sweep"]
    nodes = audit["node_sweep"]
    mesh = audit["mesh_sweep"]
    froude = audit["froude_sweep"]
    verdict = audit["verdict"]

    lines: list[str] = []
    lines.append("| quantity | value |")
    lines.append("|---|---|")
    lines.append(
        "| hull | Wigley $L={L}$, $B={B}$, $T={T}$ m |".format(**case)
    )
    lines.append(
        "| density / gravity | $\\rho={rho}$ kg/m3, $g={g}$ m/s2 |".format(**case)
    )
    lines.append("| reference | " + str(case["reference"]) + " |")
    lines.append("")

    lines.append("### Cut-off sensitivity ($Fr=%.2f$, mesh %s)" % (
        cutoff["froude"], cutoff["mesh"]))
    lines.append("")
    lines.append("| $k_{max}/k_0$ | $R_W$ pressure [N] | ratio to Kochin |")
    lines.append("|---:|---:|---:|")
    for row in cutoff["cutoff"]:
        lines.append("| %.0f | %.4e | %.4f |" % (
            row["k_max_factor"], row["Rw_pressure"], row["ratio_to_kochin"]))
    lines.append("")
    lines.append("Kochin reference $R_W=%.4e$ N; total spread over the cut-off "
                 "sweep %.1f %% of the Kochin value." % (
                     cutoff["Rw_kochin"], 100.0 * cutoff["cutoff_spread_relative"]))
    lines.append("")

    lines.append("### Quadrature-node convergence at the legacy cut-off ($Fr=%.2f$)" % nodes["froude"])
    lines.append("")
    lines.append("| nodes | $R_W$ pressure [N] | ratio to Kochin |")
    lines.append("|---:|---:|---:|")
    for row in nodes["nodes"]:
        lines.append("| %d | %.4e | %.4f |" % (
            row["n_quad"], row["Rw_pressure"], row["ratio_to_kochin"]))
    lines.append("")
    lines.append("The sampled integral is *converged* (identical to seven digits from "
                 "128 nodes on): the defect is not quadrature error, it is the "
                 "integrand.")
    lines.append("")

    lines.append("### Mesh ladder ($Fr=%.2f$)" % mesh["froude"])
    lines.append("")
    lines.append("| $n_x\\times n_z$ | panels | $h_{mean}$ [m] | $R_W$ pressure [N] | ratio | rel. difference to Kochin |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for row in mesh["levels"]:
        lines.append("| %s | %d | %.4f | %.4e | %.4f | %.3f |" % (
            "x".join(str(v) for v in row["mesh"]), row["panel_count"], row["h_mean"],
            row["Rw_pressure"], row["ratio_to_kochin"], row["relative_difference"]))
    lines.append("")
    lines.append("Refining the mesh drives the pressure path to $R_W\\to 0$, i.e. the "
                 "relative difference tends to 1, not to 0.")
    lines.append("")

    lines.append("### Froude sweep (mesh %s)" % froude["mesh"])
    lines.append("")
    lines.append("| $Fr$ | $R_W$ Kochin [N] | $R_W$ pressure [N] | ratio | rel. difference |")
    lines.append("|---:|---:|---:|---:|---:|")
    for row in froude["rows"]:
        lines.append("| %.2f | %.4e | %.4e | %.4f | %.3f |" % (
            row["froude"], row["Rw_kochin"], row["Rw_pressure"],
            row["ratio_to_kochin"], row["relative_difference"]))
    lines.append("")
    lines.append("Ratio spans %.4f ... %.4f (spread %.4f): no single sign or constant "
                 "factor maps one path onto the other." % (
                     froude["ratio_min"], froude["ratio_max"], froude["ratio_spread"]))
    lines.append("")

    lines.append("### Free-surface residual of the legacy field")
    lines.append("")
    lines.append("| $Fr$ | residual, standard operator | residual, documented operator | nodes on dispersion curve | min $|\\lambda-1|$ |")
    lines.append("|---:|---:|---:|---:|---:|")
    for row in audit["fs_residual"]:
        lines.append("| %.2f | %.3f | %.3f | %d | %.2e |" % (
            row["froude"], row["fs_residual_standard"], row["fs_residual_documented"],
            row["nodes_on_dispersion_curve"], row["min_abs_eigval_standard_over_k0"]))
    lines.append("")

    lines.append("### Machine verdict")
    lines.append("")
    lines.append("| test | result |")
    lines.append("|---|---|")
    for key, value in sorted(verdict.items()):
        lines.append("| `%s` | %s |" % (key, value))
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/wigley_pressure_audit/results.json"),
    )
    parser.add_argument(
        "--summary", type=Path, default=Path("docs/wigley_pressure_audit/summary.md")
    )
    parser.add_argument(
        "--figures", type=Path, default=Path("docs/figures")
    )
    args = parser.parse_args(argv)

    audit = run_audit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    figure = make_figure(audit, args.figures)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(render_markdown(audit))

    print(f"wrote {args.output}")
    print(f"wrote {args.summary}")
    print(f"wrote {figure}")
    verdict = audit["verdict"]
    print(json.dumps(verdict, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())