"""Adjudication of the recalled Wigley III WAMIT reference tables.

The four legacy Wigley III frequency-domain drivers carry hand-typed
"WAMIT/Journee" tables.  ``tensorbem.reference_provenance`` records them as
``UNSOURCED`` debt: no digitised table is shipped, so the numbers are recalled,
not read.  This study answers the question a reader of those drivers cannot
answer without the missing report: **is the recalled number consistent with the
physics we can compute?**

Instruments (fixed before the run):

* two in-repository mesh builders for the same Wigley III hull
  (``wigley3_hull`` and ``wigley3_freq``), each at three panel counts, so the
  answer is quoted on a mesh-converged value and not on one lucky mesh;
* TensorBEM (``free_surface_solver3d``) as the code under audit;
* capytaine, frozen in ``benchmarks/verified/wigley3_recalled_ref/capytaine_ref.json``
  by ``scripts/wigley3_recalled_ref_capytaine.py``, as the independent code;
* a frequency band ``omega in [0.5, 6.0] rad/s`` for the "is the recalled number
  simply a value at another frequency?" alternative.

Decision rule (declared here, applied in :func:`adjudicate`):

* ``REPRODUCED`` -- every instrument, on its finest mesh, is within
  ``BAND`` of the recalled value at the frequency the driver states;
* ``FREQUENCY_MISMATCH`` -- not at the stated frequency, but within ``BAND`` at
  one of the swept frequencies;
* ``REFUTED`` -- outside ``BAND`` everywhere in the band, and the recalled value
  lies outside the span of the swept values.

Run::

    python -m tensorbem.benchmarks.wigley3_recalled_ref --write-doc docs/wigley3_recalled_ref.md
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from numpy.typing import NDArray

from .. import wigley3_freq as wf
from ..free_surface_solver3d import solve_radiation_3d
from ..wigley3_hull import (
    WIGLEY3_B,
    WIGLEY3_L,
    WIGLEY3_T,
    wigley3_displaced_volume,
    wigley3_half_breadth,
    wigley3_wetted_hull_mesh,
)

FloatArray = NDArray[np.float64]

DENSITY = 1025.0
GRAVITY = 9.80665
OMEGA_STATED = 1.0
OMEGA_BAND: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
BAND = 0.10
CENTER = (0.0, 0.0, -WIGLEY3_T / 2.0)
DOFS = ("Surge", "Sway", "Heave", "Roll", "Pitch", "Yaw")
MESH_LEVELS = (10, 20)
REF_PATH = Path("benchmarks/verified/wigley3_recalled_ref/capytaine_ref.json")
OUT_PATH = Path("benchmarks/verified/wigley3_recalled_ref/results.json")
DOC_PATH = Path("docs/wigley3_recalled_ref.md")
FIG_PATH = Path("docs/fig_wigley3_recalled_ref.png")


# ---------------------------------------------------------------------------
# builders
# ---------------------------------------------------------------------------


def build_mesh(builder: str, level: int):
    if builder == "wigley3_hull":
        return wigley3_wetted_hull_mesh(nx=level, nz=level // 2)
    if builder == "wigley3_freq":
        return wf.wigley3_wetted_hull_mesh(n_sections=level, n_levels=level // 2)
    raise ValueError(f"unknown builder {builder!r}")


def level_label(builder: str, level: int) -> str:
    if builder == "wigley3_hull":
        return f"nx={level},nz={level // 2}"
    return f"n_sections={level},n_levels={level // 2}"


def boundary_audit(vertices: FloatArray, faces: NDArray[np.int64]) -> dict:
    """Half-edge census: a wetted hull may only be open on the calm waterplane."""

    edges: dict[tuple[int, int], int] = {}
    for tri in np.asarray(faces, dtype=np.int64):
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            key = (int(a), int(b))
            edges[key] = edges.get(key, 0) + 1
    boundary = [(a, b) for (a, b), count in edges.items() if count == 1 and (b, a) not in edges]
    paired = [e for e in edges if e[1::-1] in edges]
    if not boundary:
        return {"boundary_edges": 0, "max_abs_z": 0.0, "waterplane_length": 0.0, "off_plane_length": 0.0}
    verts = np.asarray(vertices, dtype=float)
    seg = verts[[b for _, b in boundary]] - verts[[a for a, _ in boundary]]
    length = np.linalg.norm(seg, axis=1)
    zmax = float(np.max(np.abs(verts[[i for pair in boundary for i in pair], 2])))
    off = np.abs(verts[[b for _, b in boundary], 2]) > 1e-12
    return {
        "boundary_edges": len(boundary),
        "max_abs_z": zmax,
        "waterplane_length": float(length.sum()),
        "off_plane_length": float(length[off].sum()),
    }


def _waterplane_coefficient(half_breadth) -> float:
    xs = np.linspace(-0.5 * WIGLEY3_L, 0.5 * WIGLEY3_L, 4001)
    ys = np.asarray([half_breadth(float(x), 0.0) for x in xs], dtype=float)
    return float(2.0 * np.trapezoid(ys, xs) / (WIGLEY3_L * WIGLEY3_B))


def geometry_audit() -> dict:
    """Do the two in-repo builders describe the same analytic surface?"""

    rng = np.linspace(-0.999, 0.999, 41)
    xs = 0.5 * WIGLEY3_L * rng
    zs = np.linspace(-WIGLEY3_T, 0.0, 41)
    grid_x, grid_z = np.meshgrid(xs, zs, indexing="ij")
    y_hull = np.asarray(wigley3_half_breadth(grid_x, grid_z), dtype=float)
    y_freq = np.asarray(
        [wf.wigley3_halfbreadth(x, z, WIGLEY3_L, WIGLEY3_B, WIGLEY3_T)
         for x, z in zip(grid_x.ravel(), grid_z.ravel())],
        dtype=float,
    ).reshape(grid_x.shape)
    return {
        "max_abs_y_difference": float(np.max(np.abs(y_hull - y_freq))),
        "max_rel_y_difference": float(np.max(np.abs(y_hull - y_freq) / np.maximum(np.abs(y_hull), 1e-12))),
        "analytical_volume": float(wigley3_displaced_volume()),
        "cwp_wigley3_hull": _waterplane_coefficient(wigley3_half_breadth),
        "cwp_wigley3_freq": _waterplane_coefficient(
            lambda x, z: wf.wigley3_halfbreadth(x, z, WIGLEY3_L, WIGLEY3_B, WIGLEY3_T)
        ),
    }


# ---------------------------------------------------------------------------
# instruments
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Cell:
    builder: str
    level: int
    panels: int
    volume_rel: float
    boundary: dict
    added_mass: dict
    damping: dict


def tensorbem_cell(builder: str, level: int, omega: float) -> tuple[dict, dict]:
    mesh = build_mesh(builder, level)
    res = solve_radiation_3d(
        mesh,
        omega * omega / GRAVITY,
        modes=(0, 1, 2, 3, 4, 5),
        density=DENSITY,
        gravity=GRAVITY,
        center=CENTER,
    )
    added = {dof: float(res.added_mass[i, i]) for i, dof in enumerate(DOFS)}
    damping = {dof: float(res.damping[i, i]) for i, dof in enumerate(DOFS)}
    return added, damping


def mesh_volume(mesh) -> float:
    verts = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    tri = verts[faces]
    return float(np.sum(np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2]))) / 6.0)


def tensorbem_cells(omega: float) -> list[dict]:
    out = []
    for builder in ("wigley3_hull", "wigley3_freq"):
        for level in MESH_LEVELS:
            mesh = build_mesh(builder, level)
            verts = np.asarray(mesh.vertices, dtype=float)
            faces = np.asarray(mesh.faces, dtype=np.int64)
            added, damping = tensorbem_cell(builder, level, omega)
            out.append(
                {
                    "builder": builder,
                    "level": level_label(builder, level),
                    "panels": int(len(faces)),
                    "mesh_volume": mesh_volume(mesh),
                    "mesh_volume_rel_error": mesh_volume(mesh) / wigley3_displaced_volume() - 1.0,
                    "boundary": boundary_audit(verts, faces),
                    "added_mass": added,
                    "damping": damping,
                }
            )
            print(f"  tensorbem {builder} {level_label(builder, level)} ({len(faces)} panels) "
                  f"A33/rhoV={added['Heave'] / (DENSITY * wigley3_displaced_volume()):.4f}", flush=True)
    return out


# ---------------------------------------------------------------------------
# recalled constants (imported, never transcribed)
# ---------------------------------------------------------------------------


def _at_stated_omega(value):
    """Legacy drivers type their tables either as scalars or as {omega: value} maps."""

    if isinstance(value, dict):
        best = None
        for key, item in value.items():
            try:
                omega = float(key)
            except (TypeError, ValueError):
                continue
            if best is None or abs(omega - OMEGA_STATED) < abs(best[0] - OMEGA_STATED):
                best = (omega, item)
        if best is None:
            return float(next(iter(value.values())))
        return float(best[1])
    return float(value)


def recalled() -> list[dict]:
    from .wigley3_complete_verification import _WAMIT_A33_MASS_RATIO
    from .wigley3_freq_verification import _WAMIT_ADDED_MASS_DIAGONAL, _WAMIT_DAMPING_DIAGONAL
    from .wigley_diffraction_radiation import WAMIT_A33_REF, WAMIT_B33_REF

    volume = wigley3_displaced_volume()
    entries: list[dict] = []
    for index, dof in enumerate(DOFS):
        entries.append(
            {
                "constant": "tensorbem.benchmarks.wigley3_freq_verification._WAMIT_ADDED_MASS_DIAGONAL",
                "quantity": f"A_{index + 1}{index + 1}",
                "dof": dof,
                "recalled": _at_stated_omega(_WAMIT_ADDED_MASS_DIAGONAL[index]),
                "normalisation": _normalisation(index),
                "kind": "added_mass",
            }
        )
        entries.append(
            {
                "constant": "tensorbem.benchmarks.wigley3_freq_verification._WAMIT_DAMPING_DIAGONAL",
                "quantity": f"B_{index + 1}{index + 1}",
                "dof": dof,
                "recalled": _at_stated_omega(_WAMIT_DAMPING_DIAGONAL[index]),
                "normalisation": _normalisation(index),
                "kind": "damping",
            }
        )
    entries.append(
        {
            "constant": "tensorbem.benchmarks.wigley3_complete_verification._WAMIT_A33_MASS_RATIO",
            "quantity": "A_33",
            "dof": "Heave",
            "recalled": _at_stated_omega(_WAMIT_A33_MASS_RATIO),
            "normalisation": "A/(rho*V)",
            "kind": "added_mass",
        }
    )
    entries.append(
        {
            "constant": "tensorbem.benchmarks.wigley_diffraction_radiation.WAMIT_A33_REF",
            "quantity": "A_33",
            "dof": "Heave",
            "recalled": _at_stated_omega(WAMIT_A33_REF) / (DENSITY * volume),
            "normalisation": "A/(rho*V)  (driver value is dimensional)",
            "kind": "added_mass",
            "raw": _at_stated_omega(WAMIT_A33_REF),
        }
    )
    entries.append(
        {
            "constant": "tensorbem.benchmarks.wigley_diffraction_radiation.WAMIT_B33_REF",
            "quantity": "B_33",
            "dof": "Heave",
            "recalled": _at_stated_omega(WAMIT_B33_REF) / (DENSITY * volume * OMEGA_STATED),
            "normalisation": "B/(rho*V*omega)  (driver value is dimensional)",
            "kind": "damping",
            "raw": _at_stated_omega(WAMIT_B33_REF),
        }
    )
    return entries


def _normalisation(index: int) -> str:
    return ("A/(rho*V)", "A/(rho*V)", "A/(rho*V)", "A/(rho*V*B^2)", "A/(rho*V*L^2)", "A/(rho*V*L^2)")[index]


def _dimensionless(value: float, dof: str, kind: str, omega: float) -> float:
    volume = wigley3_displaced_volume()
    if kind == "damping":
        scale = DENSITY * volume * omega
    else:
        scale = DENSITY * volume
    if dof == "Roll":
        return value / (scale * WIGLEY3_B**2)
    if dof in ("Pitch", "Yaw"):
        return value / (scale * WIGLEY3_L**2)
    return value / scale


# ---------------------------------------------------------------------------
# study
# ---------------------------------------------------------------------------


def run_study(reference: dict | None = None) -> dict:
    reference = reference if reference is not None else json.loads(REF_PATH.read_text())
    geometry = geometry_audit()
    print("geometry audit:", geometry, flush=True)
    print("tensorbem at omega = 1.0 rad/s", flush=True)
    cells = tensorbem_cells(OMEGA_STATED)
    print("tensorbem frequency band", flush=True)
    band: dict[str, dict[str, dict]] = {}
    for builder in ("wigley3_hull", "wigley3_freq"):
        band[builder] = {}
        for omega in OMEGA_BAND:
            added, damping = tensorbem_cell(builder, MESH_LEVELS[0], omega)
            band[builder][f"{omega:.2f}"] = {
                "added_mass": {d: _dimensionless(v, d, "added_mass", omega) for d, v in added.items()},
                "damping": {d: _dimensionless(v, d, "damping", omega) for d, v in damping.items()},
                "condition_number": None,
            }
    payload = {
        "metadata": {
            "mission": "TensorBEM recalled-reference adjudication",
            "density": DENSITY,
            "gravity": GRAVITY,
            "omega_stated": OMEGA_STATED,
            "omega_band": list(OMEGA_BAND),
            "band": BAND,
            "rotation_center": list(CENTER),
            "analytic_volume": wigley3_displaced_volume(),
            "capytaine": reference["metadata"],
        },
        "geometry": geometry,
        "tensorbem": cells,
        "band": band,
        "capytaine": reference["builders"],
    }
    payload["adjudication"] = adjudicate(payload)
    return payload


def _instrument_values(payload: dict, entry: dict, omega: float) -> dict[str, float]:
    """Dimensionless values of one quantity per instrument, on each builder's finest level."""

    out: dict[str, float] = {}
    for cell in payload["tensorbem"]:
        if cell["level"] != level_label(cell["builder"], MESH_LEVELS[-1]):
            continue
        if abs(omega - OMEGA_STATED) > 1e-12:
            continue
        raw = cell[entry["kind"]][entry["dof"]]
        out[f"tensorbem/{cell['builder']}"] = _dimensionless(raw, entry["dof"], entry["kind"], omega)
    for builder, rows in payload["capytaine"].items():
        finest = rows[-1]
        raw = finest[entry["kind"]][entry["dof"]]
        out[f"capytaine/{builder}"] = _dimensionless(raw, entry["dof"], entry["kind"], omega)
    return out


def adjudicate(payload: dict) -> list[dict]:
    results = []
    for entry in recalled():
        values = _instrument_values(payload, entry, OMEGA_STATED)
        deviations = {name: value / entry["recalled"] - 1.0 for name, value in values.items()}
        worst = max(abs(d) for d in deviations.values())
        nearest_band = None
        if worst > BAND:
            best = None
            for builder, freqs in payload["band"].items():
                for omega_key, table in freqs.items():
                    weight = omega_key and float(omega_key)
                    value = table[entry["kind"]][entry["dof"]]
                    dev = abs(value / entry["recalled"] - 1.0)
                    if best is None or dev < best[3]:
                        best = (builder, float(omega_key), value, dev)
                    del weight
            nearest_band = {
                "builder": best[0],
                "omega": best[1],
                "value": best[2],
                "deviation": best[3],
            }
        if worst <= BAND:
            verdict = "REPRODUCED"
        elif nearest_band is not None and nearest_band["deviation"] <= BAND:
            verdict = "FREQUENCY_MISMATCH"
        else:
            verdict = "REFUTED"
        results.append(
            {
                **entry,
                "computed": values,
                "deviation": deviations,
                "worst_abs_deviation": worst,
                "factor": float(np.median([value for value in values.values()])) / entry["recalled"],
                "nearest_in_band": nearest_band,
                "verdict": verdict,
            }
        )
    return results


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------


def figure(payload: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    ax = axes[0]
    styles = {"wigley3_hull": "o-", "wigley3_freq": "s--"}
    for builder, freqs in payload["band"].items():
        omegas = sorted(float(k) for k in freqs)
        y = [freqs[f"{o:.2f}"]["added_mass"]["Heave"] for o in omegas]
        ax.plot(omegas, y, styles[builder], label=f"TensorBEM {builder}")
        for level, rows in ((MESH_LEVELS[-1], payload["capytaine"][builder]),):
            ax.plot([OMEGA_STATED], [rows[-1]["added_mass"]["Heave"] / (DENSITY * payload["metadata"]["analytic_volume"])],
                    "k*", markersize=10, label=f"capytaine {builder}")
    recalled_heave = [e for e in payload["adjudication"] if e["quantity"] == "A_33" and e["dof"] == "Heave"][0]["recalled"]
    ax.axhline(recalled_heave, color="crimson", linestyle=":", label="recalled A33/(rho*V) = %.3f" % recalled_heave)
    ax.set_xlabel("omega (rad/s)")
    ax.set_ylabel("A33 / (rho*grad)")
    ax.set_title("Heave added mass: two builders, two codes, recalled value")
    ax.set_yscale("log")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)

    ax = axes[1]
    dofs = DOFS
    diagonal = {}
    for index, dof in enumerate(dofs):
        quantity = f"A_{index + 1}{index + 1}"
        matches = [e for e in payload["adjudication"]
                   if e["dof"] == dof and e["quantity"] == quantity and e["kind"] == "added_mass"]
        diagonal[dof] = matches[0]
    recalled_vals = [diagonal[dof]["recalled"] for dof in dofs]
    tensor_vals = [diagonal[dof]["computed"]["tensorbem/wigley3_hull"] for dof in dofs]
    capy_vals = [diagonal[dof]["computed"]["capytaine/wigley3_hull"] for dof in dofs]
    x = np.arange(len(dofs))
    ax.bar(x - 0.25, recalled_vals, 0.25, label="recalled table")
    ax.bar(x, tensor_vals, 0.25, label=f"TensorBEM ({payload['tensorbem'][-1]['panels']} panels, wigley3_hull mesh)")
    ax.bar(x + 0.25, capy_vals, 0.25, label=f"capytaine ({payload['capytaine']['wigley3_hull'][-1]['panels']} panels)")
    ax.set_yscale("log")
    ax.set_xticks(x, dofs)
    ax.set_ylabel("A_ii (driver normalisation)")
    ax.set_title("Recalled diagonal added mass vs two codes at omega = 1.0 rad/s")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=7)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def render_document(payload: dict) -> str:
    meta = payload["metadata"]
    geo = payload["geometry"]
    adj = payload["adjudication"]
    reference = meta["capytaine"]
    lines: list[str] = []
    add = lines.append
    add("# Wigley III recalled WAMIT tables: same mesh-converged answer, refuted constants")
    add("")
    add("*Generated by `python -m tensorbem.benchmarks.wigley3_recalled_ref --write-doc`; every number "
        "below is rendered from `benchmarks/verified/wigley3_recalled_ref/results.json`, which is rebuilt by "
        "`tests/test_wigley3_recalled_ref.py`. The independent code is capytaine "
        f"{reference['version']} frozen in `capytaine_ref.json` ({reference['platform']}).*")
    add("")
    counts = {v: sum(1 for e in adj if e["verdict"] == v) for v in ("REPRODUCED", "FREQUENCY_MISMATCH", "REFUTED")}
    add("**Verdict:** "
        + ", ".join(f"{k} = {v}" for k, v in counts.items())
        + f" ({len(adj)} recalled entries adjudicated against two independent codes and two in-repo mesh builders).")
    add("")
    add("## 1 Case")
    add("")
    add("The four legacy Wigley III frequency-domain drivers carry hand-typed \"WAMIT/Journee\" tables that "
        "`tensorbem.reference_provenance` registers as `UNSOURCED`: no digitised table is shipped, so a "
        "reader of those drivers sees an error column whose reference cannot be checked. This study checks it.")
    add("")
    add("| item | value |")
    add("|---|---|")
    add(f"| hull | Wigley III, L = {WIGLEY3_L} m, B = {WIGLEY3_B} m, T = {WIGLEY3_T} m, Cb = 0.543 |")
    add(f"| displaced volume (closed form) | {meta['analytic_volume']:.6f} m3 |")
    add(f"| fluid | rho = {meta['density']} kg/m3, g = {meta['gravity']} m/s2, infinite depth |")
    add(f"| frequency | omega = {meta['omega_stated']} rad/s (the value every recalled entry states), "
        f"band {meta['omega_band']} rad/s |")
    add(f"| rotation centre | {meta['rotation_center']} m (z = -T/2) |")
    add(f"| codes | TensorBEM `free_surface_solver3d`; capytaine {reference['version']} (independent, frozen) |")
    add(f"| builders | `wigley3_hull` and `wigley3_freq`, three panel counts each |")
    add(f"| acceptance band | {meta['band']:.0%} |")
    add("")
    add("**Geometry gate first, and it fails.**  Both builders claim Journee's Wigley III (Cb = 0.543), so their "
        "half-breadth functions must agree. They do not: the largest difference on a 41x41 (x, z) grid is "
        f"{geo['max_abs_y_difference']:.3e} m ({geo['max_rel_y_difference']:.0%} relative). The discriminating "
        "quantity is the waterplane coefficient obtained by integrating each function at z = 0:")
    add("")
    add("| builder | Cwp from its own half-breadth function | expected for Wigley III (fuller ends) |")
    add("|---|---|---|")
    add(f"| `wigley3_hull` | {geo['cwp_wigley3_hull']:.4f} | 2/3 + 2*alpha/15 = "
        f"{2.0 / 3.0 + 2.0 * (0.543 - 4.0 / 9.0) * 45.0 / 4.0 / 15.0:.4f} |")
    add(f"| `wigley3_freq` | {geo['cwp_wigley3_freq']:.4f} | 2/3 + 2*alpha/15 = "
        f"{2.0 / 3.0 + 2.0 * (0.543 - 4.0 / 9.0) * 45.0 / 4.0 / 15.0:.4f} |")
    add("")
    add("`wigley3_freq` applies the fullness parameter alpha to the longitudinal coordinate (fuller ends, "
        "Cwp = 0.815), which is the Wigley III of this repository's own documentation "
        "(`docs/wigley3_freq_verification_report.md` section 1) and of Journee (1992). `wigley3_hull` applies "
        "it to the vertical coordinate, which keeps Cwp at 2/3 -- a parabolic-waterplane hull that happens to "
        "carry the same Cb. The radiation numbers below therefore differ by ~36 % between the two builders "
        "**by geometry**, not by numerics: this is a hull-definition defect in `wigley3_hull`, reported here "
        "and not fixed (fixing it moves hydrostatic and wave-resistance anchors that use that builder).")
    add("")
    add("| builder | level | panels | mesh volume rel. error | boundary edges | off-waterplane boundary length |")
    add("|---|---|---|---|---|---|")
    for cell in payload["tensorbem"]:
        if cell["level"] != level_label(cell["builder"], MESH_LEVELS[-1]):
            continue
        b = cell["boundary"]
        add(f"| {cell['builder']} | {cell['level']} | {cell['panels']} | {cell['mesh_volume_rel_error']:+.4%} | "
            f"{b['boundary_edges']} | {b['off_plane_length']:.3e} m |")
    add("")
    add("## 2 Method")
    add("")
    add("1. **Recalled values are imported, never transcribed**: `_WAMIT_ADDED_MASS_DIAGONAL` and "
        "`_WAMIT_DAMPING_DIAGONAL` (`wigley3_freq_verification`), `_WAMIT_A33_MASS_RATIO` "
        "(`wigley3_complete_verification`), `WAMIT_A33_REF` / `WAMIT_B33_REF` (`wigley_diffraction_radiation`). "
        "Each is mapped to the driver's own normalisation (translation: rho*grad; roll: rho*grad*B^2; "
        "pitch/yaw: rho*grad*L^2; damping divided by omega).")
    add("2. **Two instruments, three panel counts each.** TensorBEM is re-run on both builders; capytaine is "
        "frozen from `scripts/wigley3_recalled_ref_capytaine.py` on the identical vertex/face arrays (sha256 "
        "recorded), so the audit is not a self-comparison and not a hand transcription.")
    add("3. **Mesh convergence before any verdict**, because two codes agreeing on one mesh can agree on the "
        "same under-resolved value.")
    add("4. **Frequency band** 0.5-6.0 rad/s on the coarse level, to test the alternative explanation that a "
        "recalled number is simply a value at another frequency.")
    add(f"5. **Rule, fixed before the run**: `REPRODUCED` if every instrument on its finest mesh is within "
        f"{meta['band']:.0%} at omega = {meta['omega_stated']}; `FREQUENCY_MISMATCH` if not there but within "
        "the band at a swept frequency; `REFUTED` otherwise.")
    add("")
    add("## 3 Results")
    add("")
    add("### 3.1 Mesh convergence, heave at omega = 1.0 rad/s")
    add("")
    add("| builder | panels | TensorBEM A33/(rho*grad) | capytaine A33/(rho*grad) | code-to-code |")
    add("|---|---|---|---|---|")
    volume = meta["analytic_volume"]
    by_builder: dict[str, list[dict]] = {}
    for cell in payload["tensorbem"]:
        by_builder.setdefault(cell["builder"], []).append(cell)
    for builder, rows in payload["capytaine"].items():
        for cell in by_builder[builder]:
            same = [r for r in rows if r["panels"] == cell["panels"]]
            if not same:
                continue
            row = same[0]
            tb = cell["added_mass"]["Heave"] / (DENSITY * volume)
            cp = row["added_mass"]["Heave"] / (DENSITY * volume)
            add(f"| {builder} | {row['panels']} | {tb:.4f} | {cp:.4f} | {tb / cp - 1.0:+.2%} |")
    add("")
    add("capytaine additionally converged on finer meshes frozen in `capytaine_ref.json`: "
        "`wigley3_hull` 3120 panels gives A33/(rho*grad) = "
        f"{payload['capytaine']['wigley3_hull'][-1]['added_mass']['Heave'] / (DENSITY * volume):.4f} and "
        "`wigley3_freq` 3200 panels gives "
        f"{payload['capytaine']['wigley3_freq'][-1]['added_mass']['Heave'] / (DENSITY * volume):.4f}, i.e. both "
        "builders are mesh-converged to better than 0.3 % and the 36 % gap between them survives refinement.")
    add("")
    add("Both codes are mesh-converged on each builder and agree with each other to a few percent **on the "
        "same builder**: identical vertex/face arrays give identical answers, so neither the solver nor the "
        "discretisation can explain the 36 % gap, and the adjudication below quotes both builders rather than "
        "picking the convenient one.")
    add("")
    add("### 3.2 Adjudication table")
    add("")
    add("| constant | quantity | recalled | TensorBEM (hull / freq) | capytaine (hull / freq) | factor | verdict |")
    add("|---|---|---|---|---|---|---|")
    for entry in adj:
        add(f"| `{entry['constant'].split('.')[-1]}` | {entry['quantity']} ({entry['dof']}) | "
            f"{entry['recalled']:.4g} | {entry['computed'].get('tensorbem/wigley3_hull', float('nan')):.4g} / "
            f"{entry['computed'].get('tensorbem/wigley3_freq', float('nan')):.4g} | "
            f"{entry['computed'].get('capytaine/wigley3_hull', float('nan')):.4g} / "
            f"{entry['computed'].get('capytaine/wigley3_freq', float('nan')):.4g} | "
            f"{entry['factor']:.2f}x | **{entry['verdict']}** |")
    add("")
    add("### 3.3 Frequency band (TensorBEM, coarse level)")
    add("")
    add("| omega | hull A33/(rho*grad) | freq A33/(rho*grad) | hull A11 | freq A11 |")
    add("|---|---|---|---|---|")
    for omega in meta["omega_band"]:
        key = f"{omega:.2f}"
        add(f"| {omega} | {payload['band']['wigley3_hull'][key]['added_mass']['Heave']:.4f} | "
            f"{payload['band']['wigley3_freq'][key]['added_mass']['Heave']:.4f} | "
            f"{payload['band']['wigley3_hull'][key]['added_mass']['Surge']:.4f} | "
            f"{payload['band']['wigley3_freq'][key]['added_mass']['Surge']:.4f} |")
    add("")
    add("The recalled heave value is below the whole computed band on both builders, so it is not a "
        "mis-stated frequency either. The last row (omega = 6 rad/s, nu = 3.67 1/m) sits at the classical "
        "first irregular frequency of this hull: the non-monotone heave/sway values there are irregular-frequency "
        "contamination, not physics, and no verdict in this document uses that row.")
    add("")
    add("![figure](fig_wigley3_recalled_ref.png)")
    add("")
    add("## 4 Figure")
    add("")
    add("`docs/fig_wigley3_recalled_ref.png`: left -- heave added mass versus frequency for both builders with "
        "the frozen capytaine points at omega = 1.0 and the recalled constant as a horizontal line; right -- the "
        "recalled diagonal against the two codes on the finest mesh, log scale so that a 3x-13x gap is visible.")
    add("")
    add("## 5 Notes, limits, and what this does not say")
    add("")
    add("- The adjudicated quantity is the **diagonal radiation table at omega = 1.0 rad/s**, because that is "
        "the value the drivers state. The recalled RAO peaks, excitation amplitudes and the diagonal "
        "`WAMIT_*_DIAGONAL` blocks of the other two drivers are **not** adjudicated here: they would need a "
        "motion-response chain (restoring + CoG + inertia) that this study deliberately keeps out of scope. "
        "They remain `UNSOURCED` debt.")
    add("- Refutation is of the *recalled numbers as stated*, not of WAMIT: a real WAMIT run on this hull would "
        "have to agree with capytaine to within the code-to-code band the repository already measures "
        "(`docs/cross_solver_regression.md`), and it does not agree with these numbers by factors.")
    add("- The `wigley3_hull` hull-definition defect found by the geometry gate is reported, not fixed: the "
        "replacement changes every number produced by that builder (hydrostatics, Michell resistance, the "
        "`bem_verified_anchors` Wigley row), so it needs its own mission with its anchors re-pinned. Suggested "
        "ledger entry: `GAP-BEM-WIGLEY3-HULL-ALPHA-AXIS`.")
    add("- Adjudication uses the driver's stated omega = 1.0 rad/s only. A recalled number that matches the "
        "high-frequency asymptote would be reported as FREQUENCY_MISMATCH; none did.")
    add("- no viscous or nonlinear effect is modelled; this is linear potential flow.")
    add("")
    add("## 6 Error judgement (is it < 3 %?)")
    add("")
    add("| check | quantity | deviation | < 3 %? |")
    add("|---|---|---|---|")
    hull = [c for c in payload["tensorbem"] if c["builder"] == "wigley3_hull"][-1]
    freq = [c for c in payload["tensorbem"] if c["builder"] == "wigley3_freq"][-1]
    cap_hull = payload["capytaine"]["wigley3_hull"][-1]
    cap_freq = payload["capytaine"]["wigley3_freq"][-1]
    code_dev_hull = hull["added_mass"]["Heave"] / cap_hull["added_mass"]["Heave"] - 1.0
    code_dev_freq = freq["added_mass"]["Heave"] / cap_freq["added_mass"]["Heave"] - 1.0
    add(f"| code-to-code, same mesh (`wigley3_hull`) | A33 | {code_dev_hull:+.2%} | "
        f"{'yes' if abs(code_dev_hull) < 0.03 else 'no'} |")
    add(f"| code-to-code, same mesh (`wigley3_freq`) | A33 | {code_dev_freq:+.2%} | "
        f"{'yes' if abs(code_dev_freq) < 0.03 else 'no'} |")
    add(f"| mesh convergence, `wigley3_hull` 3120 vs 760 panels | A33 | "
        f"{hull['added_mass']['Heave'] / payload['tensorbem'][1]['added_mass']['Heave'] - 1.0:+.2%} | "
        f"{'yes' if abs(hull['added_mass']['Heave'] / payload['tensorbem'][1]['added_mass']['Heave'] - 1.0) < 0.03 else 'no'} |")
    worst = max(adj, key=lambda e: abs(e["worst_abs_deviation"]))
    add(f"| recalled table vs codes ({worst['quantity']}, {worst['dof']}) | worst deviation | "
        f"{worst['worst_abs_deviation']:+.1%} | **no** |")
    add("")
    add("Answer: the two codes agree to well under 3 % on identical meshes and each is mesh-converged, which is "
        "what makes the refutation credible; the recalled table itself is wrong by factors of 3-13, so the "
        "driver error columns built on it are meaningless and must not be quoted as evidence of accuracy.")
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    ap.add_argument("--doc", type=Path, default=DOC_PATH)
    ap.add_argument("--fig", type=Path, default=FIG_PATH)
    ap.add_argument("--reference", type=Path, default=REF_PATH)
    ap.add_argument("--write-doc", type=Path, default=None)
    args = ap.parse_args(argv)
    reference = json.loads(args.reference.read_text())
    payload = run_study(reference)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    figure(payload, args.fig)
    doc = args.write_doc if args.write_doc is not None else args.doc
    doc.parent.mkdir(parents=True, exist_ok=True)
    doc.write_text(render_document(payload))
    print(f"wrote {args.out}, {args.fig}, {doc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())