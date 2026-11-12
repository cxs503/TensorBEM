"""Generate the GAP-BEM-COEFF-IO evidence bundle (numbers, figure, README).

Every number in ``docs/coefficient_io/README.md`` comes from this script, which
writes ``evidence.json`` first and renders the document from it -- nothing is
transcribed by hand.

Run:  PYTHONPATH=src python3 scripts/coefficients_io_evidence.py
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
from pathlib import Path

import numpy as np

from tensorbem.coefficients import (
    COEFFICIENT_SCHEMA_VERSION,
    DOF_ORDER,
    HydrodynamicCoefficients,
    from_radiation_3d,
    stack_frequencies,
)

OUT = Path("docs/coefficient_io")
OMEGA = np.array([0.5, 1.0, 2.0])


def _rel(actual, expected) -> float:
    actual = np.asarray(actual, dtype=np.complex128)
    expected = np.asarray(expected, dtype=np.complex128)
    return float(np.max(np.abs(actual - expected) / np.maximum(np.abs(expected), 1e-300)))


def _synthetic() -> HydrodynamicCoefficients:
    rng = np.random.default_rng(20261112)
    added = rng.normal(scale=1.0e4, size=(OMEGA.size, 3, 3))
    damp = rng.normal(scale=1.0e5, size=(OMEGA.size, 3, 3))
    added[:, 0, 1] *= 3.0  # deliberate asymmetry: I/O must not symmetrise
    force = rng.normal(scale=1.0e6, size=(OMEGA.size, 2, 3)) + 1j * rng.normal(
        scale=1.0e6, size=(OMEGA.size, 2, 3)
    )
    return HydrodynamicCoefficients(
        omega=OMEGA,
        added_mass=added,
        damping=damp,
        dof_labels=DOF_ORDER[:3],
        excitation=force,
        headings_deg=np.array([0.0, 135.0]),
        rho=1025.0,
        length_scale=2.0,
        provenance={"solver": "synthetic", "note": "round-trip stress case"},
    )


def _sphere_sweep() -> HydrodynamicCoefficients:
    from tensorbem.free_surface_solver3d import solve_radiation_3d
    from tensorbem.geometry3d import submerged_sphere_mesh

    mesh = submerged_sphere_mesh(subdivisions=1, radius=0.25, submergence=1.0)
    slices = []
    for nu in (0.4, 0.8, 1.2, 1.6):
        result = solve_radiation_3d(mesh, nu, modes=(0, 2))  # surge + heave
        slices.append(
            from_radiation_3d(
                result,
                provenance={
                    "solver": "tensorbem.free_surface_solver3d.solve_radiation_3d",
                    "case": "submerged sphere r=0.25, submergence=1.0, subdivisions=1",
                    "nu": nu,
                    "n_panels": int(mesh.faces.shape[0]),
                    "condition_number": float(result.condition_number),
                    "relative_residual": float(result.relative_residual),
                },
            )
        )
    return stack_frequencies(slices)


def _capytaine_interop(tmp: Path) -> dict:
    try:
        import capytaine.io.wamit as wamit
        import xarray as xr
    except Exception as exc:  # pragma: no cover - environment dependent
        return {"available": False, "reason": f"{type(exc).__name__}: {exc}"}

    dofs = ["Surge", "Heave", "Pitch"]
    omega = np.array([0.6, 1.1])
    added = np.zeros((omega.size, 3, 3))
    damp = np.zeros_like(added)
    for f in range(omega.size):
        for i in range(3):
            for j in range(3):
                added[f, i, j] = 1.0e3 * (1 + i + 2 * j) * (1.0 + 0.3 * f)
                damp[f, i, j] = 1.0e2 * (2 + 3 * i + j) * (1.0 + 0.1 * f)
    force = np.zeros((omega.size, 2, 3), dtype=complex)
    for f in range(omega.size):
        for beta in range(2):
            for i in range(3):
                force[f, beta, i] = 1.0e4 * (1 + i) * (1 + beta) * complex(
                    math.cos(0.3 * (i + 1)), math.sin(0.3 * (i + 1))
                )
    ds = xr.Dataset(
        data_vars={
            "added_mass": (("omega", "influenced_dof", "radiating_dof"), added),
            "radiation_damping": (("omega", "influenced_dof", "radiating_dof"), damp),
            "excitation_force": (("omega", "wave_direction", "influenced_dof"), force),
        },
        coords={
            "omega": omega,
            "influenced_dof": dofs,
            "radiating_dof": dofs,
            "wave_direction": np.array([0.0, math.pi / 2]),
            "rho": 1025.0,
            "g": 9.80665,
            "forward_speed": 0.0,
        },
    ).assign_coords(period=("omega", 2.0 * np.pi / omega))

    length_scale = 2.0
    path1 = tmp / "capytaine.1"
    path3 = tmp / "capytaine.3"
    wamit.check_dataset_ready_for_export(ds)
    wamit.export_wamit_1(ds, str(path1), length_scale=length_scale)
    wamit.export_wamit_3(ds, str(path3))

    read1 = HydrodynamicCoefficients.read_wamit_1(path1, rho=1025.0, length_scale=length_scale)
    read3 = HydrodynamicCoefficients.read_wamit_3(
        path3, rho=1025.0, g=9.80665, length_scale=1.0, conjugate=True
    )
    return {
        "available": True,
        "capytaine_version": _maybe_version("capytaine"),
        "wamit_1_dof_labels": list(read1.dof_labels),
        "wamit_1_omega_rel_error": _rel(read1.omega, omega),
        "wamit_1_added_mass_rel_error": _rel(read1.added_mass, added),
        "wamit_1_damping_rel_error": _rel(read1.damping, damp),
        "wamit_3_modulus_rel_error": _rel(np.abs(read3.excitation), np.abs(force)),
        "wamit_3_complex_rel_error": _rel(read3.excitation, force),
        "wamit_3_headings_deg": read3.headings_deg.tolist(),
    }


def _maybe_version(package: str) -> str:
    try:
        module = __import__(package)
        return str(getattr(module, "__version__", "unknown"))
    except Exception:  # pragma: no cover
        return "absent"


def collect() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = OUT / "tmp"
    tmp.mkdir(exist_ok=True)

    synthetic = _synthetic()
    json_path = synthetic.write_json(tmp / "synthetic.json")
    native = HydrodynamicCoefficients.read_json(json_path)
    wamit1 = HydrodynamicCoefficients.read_wamit_1(
        synthetic.write_wamit_1(tmp / "synthetic.1"), rho=synthetic.rho, length_scale=2.0
    )
    wamit3 = HydrodynamicCoefficients.read_wamit_3(
        synthetic.write_wamit_3(tmp / "synthetic.3"), rho=synthetic.rho, length_scale=2.0
    )
    tec = HydrodynamicCoefficients.read_nemoh_tec(synthetic.write_nemoh_tec(tmp / "synthetic.tec"))
    tec_f = HydrodynamicCoefficients.read_nemoh_excitation_tec(
        synthetic.write_nemoh_excitation_tec(tmp / "synthetic-F.tec")
    )

    tampered = json.loads(json_path.read_text(encoding="utf-8"))
    tampered["added_mass"][0][0][0] *= 1.0000001
    tamper_path = tmp / "tampered.json"
    tamper_path.write_text(json.dumps(tampered), encoding="utf-8")
    try:
        HydrodynamicCoefficients.read_json(tamper_path)
        tamper_detected = False
    except Exception:
        tamper_detected = True

    sphere = _sphere_sweep()
    sphere_json = HydrodynamicCoefficients.read_json(sphere.write_json(tmp / "sphere.json"))
    sphere_1 = HydrodynamicCoefficients.read_wamit_1(
        sphere.write_wamit_1(tmp / "sphere.1"), rho=sphere.rho
    )
    sphere_tec = HydrodynamicCoefficients.read_nemoh_tec(sphere.write_nemoh_tec(tmp / "sphere.tec"))

    heave_index = sphere.dof_labels.index("heave")
    a33 = sphere.added_mass[:, heave_index, heave_index] / sphere.rho  # nondim by density

    evidence = {
        "anchor": "GAP-BEM-COEFF-IO",
        "generated_by": "scripts/coefficients_io_evidence.py",
        "schema_version": COEFFICIENT_SCHEMA_VERSION,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "capytaine": _maybe_version("capytaine"),
            "xarray": _maybe_version("xarray"),
        },
        "acceptance": {
            "native_roundtrip_max_rel": {
                "added_mass": _rel(native.added_mass, synthetic.added_mass),
                "damping": _rel(native.damping, synthetic.damping),
                "excitation": _rel(native.excitation, synthetic.excitation),
            },
            "dialect_roundtrip_max_rel": {
                "wamit_1_added_mass": _rel(wamit1.added_mass, synthetic.added_mass),
                "wamit_1_damping": _rel(wamit1.damping, synthetic.damping),
                "wamit_1_omega": _rel(wamit1.omega, synthetic.omega),
                "wamit_3_excitation": _rel(wamit3.excitation, synthetic.excitation),
                "nemoh_tec_added_mass": _rel(tec.added_mass, synthetic.added_mass),
                "nemoh_tec_damping": _rel(tec.damping, synthetic.damping),
                "nemoh_tec_excitation": _rel(tec_f.excitation, synthetic.excitation),
            },
            "tamper_detected": tamper_detected,
            "tampered_value": float(tampered["added_mass"][0][0][0]),
            "fingerprint_original": synthetic.fingerprint(),
        },
        "capytaine_interop": _capytaine_interop(tmp),
        "solver_case": {
            "case": "submerged sphere r=0.25, submergence=1.0, icosphere subdivisions=1",
            "dof_labels": list(sphere.dof_labels),
            "n_panels": sphere.provenance.get("n_panels"),
            "omega": sphere.omega.tolist(),
            "added_mass_surge": sphere.added_mass[:, 0, 0].tolist(),
            "added_mass_heave": sphere.added_mass[:, heave_index, heave_index].tolist(),
            "damping_heave": sphere.damping[:, heave_index, heave_index].tolist(),
            "a33_over_rho": a33.tolist(),
            "propagated_rel_error": {
                "native": _rel(sphere_json.added_mass[:, heave_index, heave_index], sphere.added_mass[:, heave_index, heave_index]),
                "wamit_1": _rel(sphere_1.added_mass[:, heave_index, heave_index], sphere.added_mass[:, heave_index, heave_index]),
                "nemoh_tec": _rel(sphere_tec.added_mass[:, heave_index, heave_index], sphere.added_mass[:, heave_index, heave_index]),
            },
            "condition_numbers": [
                entry.get("condition_number")
                for entry in (sphere.provenance.get("slices") or [])
            ],
            "residuals": [
                entry.get("relative_residual") for entry in (sphere.provenance.get("slices") or [])
            ],
        },
        "files": {},
    }
    for path in sorted(tmp.glob("*")):
        if path.suffix in {".json", ".1", ".3", ".tec"}:
            evidence["files"][path.name] = {
                "bytes": path.stat().st_size,
                "sha256_16": hashlib.sha256(path.read_bytes()).hexdigest()[:16],
            }
    (OUT / "evidence.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    return evidence


def _figure(evidence: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    acc = evidence["acceptance"]
    labels = ["native .json\n(A,B,F)", ".1 added\nmass", ".1 damping", ".3 excitation", ".tec radiation", ".tec excitation"]
    values = [
        max(acc["native_roundtrip_max_rel"].values()),
        acc["dialect_roundtrip_max_rel"]["wamit_1_added_mass"],
        acc["dialect_roundtrip_max_rel"]["wamit_1_damping"],
        acc["dialect_roundtrip_max_rel"]["wamit_3_excitation"],
        acc["dialect_roundtrip_max_rel"]["nemoh_tec_added_mass"],
        acc["dialect_roundtrip_max_rel"]["nemoh_tec_excitation"],
    ]
    floor = 1e-18
    plotted = [max(value, floor) for value in values]

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6))
    colours = ["#1b6ca8"] + ["#c46a1b"] * (len(labels) - 1)
    axes[0].bar(range(len(labels)), plotted, color=colours)
    axes[0].set_yscale("log")
    axes[0].axhline(1e-12, color="k", ls="--", lw=1.0, label="acceptance: native 1e-12")
    axes[0].axhline(1e-6, color="r", ls=":", lw=1.2, label="acceptance: dialect 1e-6")
    axes[0].set_xticks(range(len(labels)))
    axes[0].set_xticklabels(labels, fontsize=8)
    axes[0].set_ylabel("max relative round-trip error")
    axes[0].set_title("Round-trip error by container dialect")
    axes[0].legend(fontsize=8, loc="lower right")
    for index, value in enumerate(plotted):
        axes[0].text(index, value * 1.6, f"{value:.1e}", ha="center", fontsize=7)

    case = evidence["solver_case"]
    omega = np.array(case["omega"])
    heave = np.array(case["added_mass_heave"]) / evidence.get("rho", 1025.0) if False else np.array(case["a33_over_rho"])
    axes[1].plot(omega, heave, "o-", color="#1b6ca8", label="A33/rho from BEM solve")
    propagated = case["propagated_rel_error"]
    worst = max(propagated.values())
    axes[1].set_xlabel("omega [rad/s]")
    axes[1].set_ylabel("A33 / rho [m^3]")
    axes[1].set_title(f"Solved case survives the wire (worst rel. error {worst:.1e})")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_roundtrip.png", dpi=160)
    plt.close(fig)


def _readme(evidence: dict) -> None:
    acc = evidence["acceptance"]
    native = acc["native_roundtrip_max_rel"]
    dialect = acc["dialect_roundtrip_max_rel"]
    interop = evidence["capytaine_interop"]
    case = evidence["solver_case"]

    def pct(value: float) -> str:
        return f"{value:.2e}"

    lines = [
        "# GAP-BEM-COEFF-IO -- hydrodynamic coefficient interchange layer",
        "",
        "Generated by `scripts/coefficients_io_evidence.py` from `evidence.json`; every number below is read",
        "from that file, none is transcribed.",
        "",
        "## 1 Case",
        "",
        "Not a flow case: the object is the *wire* between solvers. TensorBEM had no way to read a",
        "hydrodynamic coefficient file back, so every verification driver compared against hand-typed",
        "constants. The container implemented here carries added mass `A_ij(omega)`, radiation damping",
        "`B_ij(omega)` and complex exciting force `F_i(omega, beta)` for a DOF subset, in four dialects:",
        "canonical JSON, WAMIT `.1`, WAMIT `.3`, Nemoh Tecplot `.tec`.",
        "",
        "## 2 Method",
        "",
        "1. One validated frozen container (`tensorbem.coefficients.HydrodynamicCoefficients`); shapes, finiteness,",
        "   strictly increasing frequencies and heading/force consistency are checked on construction.",
        "2. Native JSON stores every float at full `repr` precision and stamps a SHA-256 of the numeric payload;",
        "   reading verifies the stamp and fails closed on a changed number.",
        "3. Dialect reader/writer pairs implement the WAMIT nondimensionalisation",
        "   `A_ij/(rho L^k)`, `B_ij/(omega rho L^k)` with `k=3/4/5` and the excitation normalisation",
        "   `F_i/(rho g A L^m)` with `m=2/3`.",
        "4. Interop is judged against an *independent implementation*: capytaine "
        f"{interop.get('capytaine_version')} writes the `.1`/`.3` files, this reader parses them.",
        "5. End-to-end: a real 3-D BEM radiation solve (submerged sphere, `solve_radiation_3d`) is pushed through",
        "   every dialect and the nondimensional heave added mass is re-derived after the round trip.",
        "",
        "## 3 Results",
        "",
        "| check | max relative error | acceptance | verdict |",
        "|---|---:|---|---|",
        f"| native JSON, added mass | {pct(native['added_mass'])} | < 1e-12 | {'PASS' if native['added_mass'] < 1e-12 else 'FAIL'} |",
        f"| native JSON, damping | {pct(native['damping'])} | < 1e-12 | {'PASS' if native['damping'] < 1e-12 else 'FAIL'} |",
        f"| native JSON, excitation | {pct(native['excitation'])} | < 1e-12 | {'PASS' if native['excitation'] < 1e-12 else 'FAIL'} |",
        f"| WAMIT `.1`, added mass | {pct(dialect['wamit_1_added_mass'])} | < 1e-6 | {'PASS' if dialect['wamit_1_added_mass'] < 1e-6 else 'FAIL'} |",
        f"| WAMIT `.1`, damping | {pct(dialect['wamit_1_damping'])} | < 1e-6 | {'PASS' if dialect['wamit_1_damping'] < 1e-6 else 'FAIL'} |",
        f"| WAMIT `.3`, excitation | {pct(dialect['wamit_3_excitation'])} | < 1e-6 | {'PASS' if dialect['wamit_3_excitation'] < 1e-6 else 'FAIL'} |",
        f"| Nemoh `.tec`, added mass | {pct(dialect['nemoh_tec_added_mass'])} | < 1e-6 | {'PASS' if dialect['nemoh_tec_added_mass'] < 1e-6 else 'FAIL'} |",
        f"| Nemoh `.tec`, excitation | {pct(dialect['nemoh_tec_excitation'])} | < 1e-6 | {'PASS' if dialect['nemoh_tec_excitation'] < 1e-6 else 'FAIL'} |",
        f"| tampered JSON detected | {'yes' if acc['tamper_detected'] else 'NO'} | must be yes | {'PASS' if acc['tamper_detected'] else 'FAIL'} |",
        "",
        "External implementation (capytaine-written files, parsed here):",
        "",
    ]
    if interop.get("available"):
        lines += [
            f"| capytaine `.1` | max relative error | acceptance |",
            "|---|---:|---|",
            f"| added mass | {pct(interop['wamit_1_added_mass_rel_error'])} | < 1e-6 |",
            f"| damping | {pct(interop['wamit_1_damping_rel_error'])} | < 1e-6 |",
            f"| frequency grid | {pct(interop['wamit_1_omega_rel_error'])} | < 1e-6 |",
            "",
            f"| capytaine `.3` | max relative error | acceptance |",
            "|---|---:|---|",
            f"| modulus | {pct(interop['wamit_3_modulus_rel_error'])} | < 1e-6 |",
            f"| complex (phase column, 7 significant digits) | {pct(interop['wamit_3_complex_rel_error'])} | < 1e-5 |",
        ]
    else:
        lines += [f"NOT RUN: {interop.get('reason')}"]

    lines += [
        "",
        "Solved case through the wire (`submerged sphere`, heave added mass / rho):",
        "",
        "| omega [rad/s] | A33/rho [m^3] | B33 [kg/s] |",
        "|---:|---:|---:|",
    ]
    for index, value in enumerate(case["omega"]):
        lines.append(
            f"| {value:.3f} | {case['a33_over_rho'][index]:.6f} | {case['damping_heave'][index]:.6f} |"
        )
    propagated = case["propagated_rel_error"]
    lines += [
        "",
        f"Propagated relative error of A33 on that case: native {pct(propagated['native'])}, "
        f"`.1` {pct(propagated['wamit_1'])}, `.tec` {pct(propagated['nemoh_tec'])}.",
        "",
        "![round-trip](fig_roundtrip.png)",
        "",
        "## 4 Notes / limitations",
        "",
        "* The WAMIT *block* `.1` variant (six rows of six columns) is **not** implemented: no authoritative",
        "  sample exists in this repository, so such a parser could not be verified -- only the line variant",
        "  (`PER I J A B`), which is what capytaine exports and what BEMRosetta/Nemoh post-processing consumes.",
        "* In the line variant column 2 is the *radiating* and column 3 the *influenced* DOF index (observed in",
        "  capytaine's exporter and verified by round trip); an asymmetrised matrix is preserved, never symmetrised.",
        "* The `.3` phase column is in degrees and its 7 significant digits limit the recovered complex value to",
        "  ~1e-5 relative even though the modulus returns to 1e-6; WAMIT's own `e^{+i omega t}` convention is",
        "  absorbed by an explicit `conjugate=True` switch rather than an invisible default.",
        "* BEMRosetta was **not** used to open the written files (not available offline here): format conformance is",
        "  established against capytaine's writer, not against a spreadsheet export of BEMRosetta.",
        "* The Wigley verification drivers still use their hard-coded reference constants; rewiring them to",
        "  file-injected references is the remaining half of the anchor's acceptance and is not claimed here.",
        "",
    ]
    (OUT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    evidence = collect()
    _figure(evidence)
    _readme(evidence)
    cap = evidence["capytaine_interop"]
    print(
        json.dumps(
            {
                "native_max": max(evidence["acceptance"]["native_roundtrip_max_rel"].values()),
                "dialect_max": max(evidence["acceptance"]["dialect_roundtrip_max_rel"].values()),
                "tamper_detected": evidence["acceptance"]["tamper_detected"],
                "capytaine_available": cap.get("available"),
                "capytaine_worst": (
                    max(
                        cap.get("wamit_1_added_mass_rel_error", 0.0),
                        cap.get("wamit_1_damping_rel_error", 0.0),
                        cap.get("wamit_3_modulus_rel_error", 0.0),
                    )
                    if cap.get("available")
                    else None
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()