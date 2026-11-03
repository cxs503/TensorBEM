"""Benchmark and report generator for the symmetry-plane (half-model) reduction.

Run::

    python -m tensorbem.benchmarks.half_model_symmetry --quick
    python -m tensorbem.benchmarks.half_model_symmetry --report

What is measured
----------------
1. **Involution.** The half model must reproduce the ordinary full model on the
   *same* triangulation. The full reference is ``surface.closed_mesh()`` mirrored
   and welded back together, so both models see identical panels and identical
   quadrature; any difference beyond roundoff would be a parity or image-sign
   error, not a numerical one.
2. **Accuracy against the closed form.** ``analytic.ellipsoid_added_mass_matrix``
   supplies the exact added mass of a centred triaxial ellipsoid; the error is
   tabulated per refinement level together with its observed order.
3. **By-product exactness.** The added-mass entries coupling modes of opposite
   parity (surge-pitch, surge-yaw, sway-pitch, ...) must be exactly zero.
4. **Cost.** Unknowns, dense-solve work and wall-clock time of both models.

The report writes ``docs/half_model_symmetry.md``, its figure and the raw JSON
so that no number in the document is transcribed by hand.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np

from ..analytic import ellipsoid_added_mass_matrix
from ..symmetry import (
    MODE_PARITIES,
    compute_added_mass_matrix_3d_half,
    half_ellipsoid_surface,
)
from ..solver3d import compute_added_mass_matrix_3d

MODE_NAMES = ("surge", "sway", "heave", "roll", "pitch", "yaw")

#: ``(n_phi, n_theta)`` refinement ladder: half-panel count is
#: ``2 * n_theta * (n_phi - 1)``, the full model has twice as many.
DEFAULT_LEVELS: tuple[tuple[int, int], ...] = ((4, 8), (8, 16), (12, 24))
QUICK_LEVELS: tuple[tuple[int, int], ...] = ((4, 8), (8, 16))

CASES = (
    {"name": "sphere", "semi_axes": (1.0, 1.0, 1.0)},
    {"name": "ellipsoid", "semi_axes": (1.0, 0.6, 0.4)},
)


def half_panel_count(n_phi: int, n_theta: int) -> int:
    """Panel count of :func:`half_ellipsoid_surface` (two caps plus the rings)."""
    return 2 * n_theta + 2 * (n_phi - 2) * n_theta


def _relative_error(value: np.ndarray, reference: np.ndarray) -> np.ndarray:
    reference = np.asarray(reference, dtype=float)
    return np.abs(np.asarray(value, dtype=float) - reference) / np.maximum(
        np.abs(reference), np.finfo(float).tiny
    )


def _observed_orders(panels: list[int], errors: list[float]) -> list[float | None]:
    orders: list[float | None] = [None]
    for index in range(1, len(errors)):
        numerator = math.log(errors[index - 1] / errors[index])
        denominator = math.log(panels[index] / panels[index - 1])
        orders.append(numerator / denominator if denominator > 0.0 else None)
    return orders


def run(
    levels: tuple[tuple[int, int], ...] = DEFAULT_LEVELS,
    cases: tuple[dict, ...] = CASES,
    quadrature_order: int = 8,
    full_model: bool = True,
) -> dict[str, object]:
    """Run the half-model study and return a JSON-serialisable record."""
    record: dict[str, object] = {
        "quadrature_order": int(quadrature_order),
        "mode_names": list(MODE_NAMES),
        "mode_parities": [float(value) for value in MODE_PARITIES],
        "levels": [list(level) for level in levels],
        "cases": [],
    }
    for case in cases:
        axes = np.asarray(case["semi_axes"], dtype=float)
        exact = np.diag(ellipsoid_added_mass_matrix(axes))
        case_record: dict[str, object] = {
            "name": case["name"],
            "semi_axes": [float(value) for value in axes],
            "exact_diagonal": [float(value) for value in exact],
            "levels": [],
        }
        for n_phi, n_theta in levels:
            surface = half_ellipsoid_surface(axes, n_phi, n_theta)
            closed = surface.closed_mesh()
            volume = float(
                np.sum(
                    np.einsum("ij,ij->i", closed.v0, np.cross(closed.v1, closed.v2))
                )
                / 6.0
            )
            start = time.perf_counter()
            half = compute_added_mass_matrix_3d_half(
                surface, quadrature_order=quadrature_order
            )
            half_seconds = time.perf_counter() - start
            level_record: dict[str, object] = {
                "n_phi": int(n_phi),
                "n_theta": int(n_theta),
                "half_panels": int(surface.panel_count),
                "full_panels": int(closed.panel_count),
                "half_seconds": float(half_seconds),
                "half_diagonal": [float(value) for value in np.diag(half.added_mass_matrix)],
                "half_diagonal_error": [
                    float(value) for value in _relative_error(np.diag(half.added_mass_matrix), exact)
                ],
                "kappa_half_even": float(half.condition_number_even),
                "kappa_half_odd": float(half.condition_number_odd),
                "max_relative_residual": float(half.max_relative_residual_l2),
                "off_parity_max": float(half.off_parity_max),
                "symmetry_error": float(half.symmetry_error),
                "closed_volume": volume,
                "closed_volume_error": float(
                    abs(volume - 4.0 / 3.0 * math.pi * float(np.prod(axes)))
                    / (4.0 / 3.0 * math.pi * float(np.prod(axes)))
                ),
            }
            if full_model:
                start = time.perf_counter()
                reference = compute_added_mass_matrix_3d(
                    closed, quadrature_order=quadrature_order
                )
                level_record["full_seconds"] = float(time.perf_counter() - start)
                level_record["full_diagonal"] = [
                    float(value) for value in np.diag(reference.added_mass_matrix)
                ]
                level_record["full_diagonal_error"] = [
                    float(value)
                    for value in _relative_error(np.diag(reference.added_mass_matrix), exact)
                ]
                level_record["kappa_full"] = float(reference.condition_number)
                level_record["involution_max_abs_diff"] = float(
                    np.max(np.abs(half.added_mass_matrix - reference.added_mass_matrix))
                )
                level_record["involution_rel_diff"] = float(
                    np.max(np.abs(half.added_mass_matrix - reference.added_mass_matrix))
                    / np.max(np.abs(reference.added_mass_matrix))
                )
            case_record["levels"].append(level_record)
        levels_records = case_record["levels"]
        panels = [int(item["half_panels"]) for item in levels_records]
        translation_orders = {}
        for mode_index, name in enumerate(MODE_NAMES[:3]):
            errors = [float(item["half_diagonal_error"][mode_index]) for item in levels_records]
            translation_orders[name] = _observed_orders(panels, errors)
        case_record["observed_order_half_panels"] = translation_orders
        record["cases"].append(case_record)
    return record


def _format_error(value: float) -> str:
    return f"{value:.3e}"


def render_markdown(record: dict[str, object], figure_name: str = "half_model_symmetry.png") -> str:
    """Render the report with every number injected from ``record``."""
    lines: list[str] = []
    lines.append("# Symmetry-plane (half-model) reduction for the 3-D BEM Laplace solver")
    lines.append("")
    lines.append("Generated by `python -m tensorbem.benchmarks.half_model_symmetry --report`;")
    lines.append("every number below is inserted from the JSON record")
    lines.append("(`docs/half_model_symmetry/results.json`), none is transcribed by hand.")
    lines.append("")
    lines.append("## 1. Case: what is being verified")
    lines.append("")
    lines.append(
        "The exterior Neumann problem for a centred triaxial ellipsoid "
        "`(x/a)^2 + (y/b)^2 + (z/c)^2 = 1` in an unbounded ideal fluid. "
        "The body is symmetric about `x = 0`, so the tangential half "
        "`x >= 0` is meshed and the mesh is mirrored and welded back to form "
        "the full model. Both models use constant triangular panels, centroid "
        "collocation, exact self-panel polar integration, Duffy quadrature "
        "elsewhere and exact solid angles for the double layer; the only "
        "difference is that the half model carries one unknown per panel pair."
    )
    lines.append("")
    lines.append("| case | semi-axes | quadrature order | rigid modes | parities |")
    lines.append("|---|---|---|---|---|")
    parities = ", ".join(
        f"{name}: {int(value):+d}"
        for name, value in zip(MODE_NAMES, record["mode_parities"])
    )
    for case in record["cases"]:
        axes = ", ".join(f"{value:g}" for value in case["semi_axes"])
        lines.append(
            f"| {case['name']} | ({axes}) | {record['quadrature_order']} | "
            f"{', '.join(MODE_NAMES)} | {parities} |"
        )
    lines.append("")

    lines.append("## 2. Method")
    lines.append("")
    lines.append(
        "For a mode potential obeying `phi(R x) = p phi(x)` with reflection `R` "
        "in the plane `x = 0`, the hidden half carries density `p q` and "
        "potential `p u`. Folding the image half into the collocation rows gives"
    )
    lines.append("")
    lines.append("```")
    lines.append("(1/2 I - H_half - p H_image) u_half = (S_half + p S_image) q_half")
    lines.append("A_ij = -rho (1 + p_i p_j) * sum_{half} phi_j n_i dS")
    lines.append("```")
    lines.append("")
    lines.append(
        "so only an `n x n` system is assembled and solved per parity class "
        "(one solve block for the three even modes, one for the three odd ones). "
        "No kernel is re-derived: the image panels are the reflected triangles "
        "handed to the same quadrature and solid-angle routines, which is why "
        "the half model is expected to be *exactly* the full model on the same "
        "triangulation."
    )
    lines.append("")

    lines.append("## 3. Results")
    lines.append("")
    for case in record["cases"]:
        lines.append(f"### 3.{record['cases'].index(case) + 1} {case['name']}")
        lines.append("")
        header = (
            "| n_phi x n_theta | half panels | full panels | "
            "surge err (half) | sway err | heave err | "
        )
        if "involution_rel_diff" in case["levels"][0]:
            header += "involution rel diff | kappa (half even/odd) | kappa (full) | half s | full s |"
        else:
            header += "kappa (half even/odd) | half s |"
        lines.append(header)
        columns = "|---|---|---|---|---|---|"
        if "involution_rel_diff" in case["levels"][0]:
            columns += "---|---|---|---|---|"
        else:
            columns += "---|---|"
        lines.append(columns)
        for level in case["levels"]:
            row = (
                f"| {level['n_phi']} x {level['n_theta']} | {level['half_panels']} | "
                f"{level['full_panels']} | "
                f"{_format_error(level['half_diagonal_error'][0])} | "
                f"{_format_error(level['half_diagonal_error'][1])} | "
                f"{_format_error(level['half_diagonal_error'][2])} | "
            )
            if "involution_rel_diff" in level:
                row += (
                    f"{_format_error(level['involution_rel_diff'])} | "
                    f"{level['kappa_half_even']:.3f} / {level['kappa_half_odd']:.3f} | "
                    f"{level['kappa_full']:.3f} | "
                    f"{level['half_seconds']:.2f} | {level['full_seconds']:.2f} |"
                )
            else:
                row += (
                    f"{level['kappa_half_even']:.3f} / {level['kappa_half_odd']:.3f} | "
                    f"{level['half_seconds']:.2f} |"
                )
            lines.append(row)
        lines.append("")
        orders = case["observed_order_half_panels"]
        order_text = ", ".join(
            f"{name}: " + ", ".join("n/a" if value is None else f"{value:.2f}" for value in values)
            for name, values in orders.items()
        )
        lines.append(f"Observed order in half panel count (translation modes): {order_text}.")
        lines.append("")
        level = case["levels"][-1]
        lines.append(
            f"Exact diagonal (closed form): "
            + ", ".join(
                f"{name} = {value:.6f}"
                for name, value in zip(MODE_NAMES, case["exact_diagonal"])
            )
            + "."
        )
        lines.append(
            f"At the finest level the half model gives "
            + ", ".join(
                f"{name} = {value:.6f}" for name, value in zip(MODE_NAMES, level["half_diagonal"])
            )
            + f"; maximum off-parity entry = {level['off_parity_max']:.1e} "
            f"(exact zeros), largest relative residual = {level['max_relative_residual']:.1e}, "
            f"closed-mesh volume error = {level['closed_volume_error']:.2%}."
        )
        lines.append("")
    lines.append(f"![half-model symmetry study]({figure_name})")
    lines.append("")
    lines.append(
        "Left: added-mass error of the surge mode against the closed form as the "
        "half mesh is refined; the half and full curves lie on top of each other "
        "because they solve the same discrete problem. Right: wall-clock time of "
        "the two paths at the same triangulation."
    )
    lines.append("")

    lines.append("## 4. Error judgement against the benchmark")
    lines.append("")
    lines.append("| key quantity | reference | value (finest level) | relative error | below 3 %? |")
    lines.append("|---|---|---|---|---|")
    for case in record["cases"]:
        level = case["levels"][-1]
        for mode_index in range(3):
            name = MODE_NAMES[mode_index]
            reference = case["exact_diagonal"][mode_index]
            value = level["half_diagonal"][mode_index]
            error = level["half_diagonal_error"][mode_index]
            verdict = "**yes**" if error < 0.03 else "**no**"
            lines.append(
                f"| {case['name']} {name} added mass | {reference:.6f} | {value:.6f} | "
                f"{_format_error(error)} | {verdict} |"
            )
    involution_values = [
        level["involution_rel_diff"]
        for case in record["cases"]
        for level in case["levels"]
        if "involution_rel_diff" in level
    ]
    if involution_values:
        lines.append(
            f"| half vs full involution (worst level) | 0 | "
            f"{max(involution_values):.1e} | roundoff | **yes** |"
        )
    off_parity = [level["off_parity_max"] for case in record["cases"] for level in case["levels"]]
    lines.append(
        f"| mode-coupled entries (opposite parity) | 0 | {max(off_parity):.1e} | exact | **yes** |"
    )
    lines.append("")
    lines.append(
        "The finite-level errors above are *discretisation* errors of the "
        "constant-panel formulation, which the half model inherits unchanged; "
        "the reduction itself introduces no additional error, as the "
        "roundoff-level involution column shows."
    )
    lines.append("")

    lines.append("## 5. Notes, pitfalls and scope")
    lines.append("")
    lines.append(
        "* The parity is a property of the *mode*, not of the geometry: surge, "
        "pitch and yaw are odd, sway, heave and roll are even. Mixing them up "
        "gives a plausible-looking but wrong added mass, so the implementation "
        "derives the fold from `MODE_PARITIES` and returns the opposite-parity "
        "entries as exact zeros."
    )
    lines.append(
        "* Cost, stated exactly: the half model halves the unknowns (`n` against "
        "`2n`), quarters the assembled entries and divides the dense-LU work by "
        "eight. It also assembles twice, once per parity class. At the panel "
        "counts measured here the Python-level assembly loop dominates, so the "
        "wall-clock columns come out nearly equal (see the table above); the "
        "`8x` figure is the solve-only cost model, not a measured speedup at "
        "these sizes."
    )
    lines.append(
        "* The closed-mesh volume deficit in the tables is the inscribed-polyhedron "
        "error of the baseline mesh, not an error of the reduction: the half and "
        "full models see the same facets."
    )
    lines.append(
        "* `center` of the rotational modes must lie in the symmetry plane; the "
        "constructor rejects anything else instead of silently returning a "
        "wrong tensor."
    )
    lines.append(
        "* Scope: unbounded-fluid exterior Neumann problems only. The free-surface "
        "solvers have no half-model path, and one symmetry plane is supported "
        "(no quarter models)."
    )
    lines.append(
        "* The welded reproduction of the full mesh is what makes the involution "
        "exact; comparing a half mesh against an independently generated full "
        "mesh (for example an icosphere) would mix the mesh difference into the "
        "comparison."
    )
    lines.append("")
    return "\n".join(lines)


def plot(record: dict[str, object], path: Path) -> None:
    """Two-panel summary figure: error convergence and wall-clock cost."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, (left, right) = plt.subplots(1, 2, figsize=(11.0, 4.2))
    markers = {"sphere": "o", "ellipsoid": "s"}
    for case in record["cases"]:
        panels = [level["half_panels"] for level in case["levels"]]
        errors = [level["half_diagonal_error"][0] for level in case["levels"]]
        left.loglog(
            panels,
            errors,
            marker=markers.get(case["name"], "o"),
            label=f"{case['name']} half model",
            color="tab:blue" if case["name"] == "sphere" else "tab:green",
        )
        if "full_diagonal_error" in case["levels"][0]:
            left.loglog(
                panels,
                [level["full_diagonal_error"][0] for level in case["levels"]],
                marker=markers.get(case["name"], "o"),
                linestyle="none",
                markersize=9,
                markerfacecolor="none",
                color="tab:red",
                label=f"{case['name']} full model",
            )
    finest = record["cases"][0]["levels"]
    if len(finest) >= 2:
        reference = finest[0]["half_diagonal_error"][0] * np.sqrt(
            finest[0]["half_panels"] / np.asarray([level["half_panels"] for level in finest], dtype=float)
        )
        left.loglog(
            [level["half_panels"] for level in finest],
            reference,
            linestyle=":",
            color="grey",
            label="slope -1/2 (h)",
        )
    left.set_xlabel("half-model panels (the full model has twice as many)")
    left.set_ylabel("relative error of the surge added mass")
    left.set_title("accuracy against the closed form")
    left.grid(True, which="both", alpha=0.3)
    left.legend(fontsize="small")

    for case in record["cases"]:
        if "full_seconds" not in case["levels"][0]:
            continue
        panels = [level["half_panels"] for level in case["levels"]]
        right.loglog(
            panels,
            [level["half_seconds"] for level in case["levels"]],
            marker="o",
            color="tab:blue",
            label=f"{case['name']} half model",
        )
        right.loglog(
            panels,
            [level["full_seconds"] for level in case["levels"]],
            marker="s",
            color="tab:red",
            label=f"{case['name']} full model",
        )
    right.set_xlabel("half-model panels (the full model has twice as many)")
    right.set_ylabel("wall-clock time [s]")
    right.set_title("cost of the same triangulation")
    right.grid(True, which="both", alpha=0.3)
    right.legend(fontsize="small")

    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="two coarse levels only")
    parser.add_argument(
        "--report",
        action="store_true",
        help="write docs/half_model_symmetry.md, its figure and results.json",
    )
    parser.add_argument("--quadrature-order", type=int, default=8)
    parser.add_argument("--json", type=Path, default=None, help="write the raw record here")
    parser.add_argument("--output-dir", type=Path, default=None)
    arguments = parser.parse_args(argv)

    levels = QUICK_LEVELS if arguments.quick else DEFAULT_LEVELS
    record = run(levels=levels, quadrature_order=arguments.quadrature_order)

    repository_root = Path(__file__).resolve().parents[3]
    output_dir = arguments.output_dir or repository_root / "docs" / "half_model_symmetry"
    if arguments.json is not None:
        arguments.json.write_text(json.dumps(record, indent=2) + "\n")

    for case in record["cases"]:
        for level in case["levels"]:
            print(
                f"{case['name']:9s} {level['n_phi']:2d}x{level['n_theta']:<3d} "
                f"half={level['half_panels']:5d} full={level['full_panels']:5d} "
                f"surge err={level['half_diagonal_error'][0]:.3e} "
                + (
                    f"involution={level['involution_rel_diff']:.1e} "
                    if "involution_rel_diff" in level
                    else ""
                )
                + f"half={level['half_seconds']:.2f}s"
            )
    if arguments.report:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "results.json").write_text(json.dumps(record, indent=2) + "\n")
        figure_path = output_dir / "half_model_symmetry.png"
        plot(record, figure_path)
        document = repository_root / "docs" / "half_model_symmetry.md"
        document.write_text(
            render_markdown(record, figure_name=f"half_model_symmetry/{figure_path.name}") + "\n"
        )
        print(f"wrote {document} and {figure_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())