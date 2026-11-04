"""Idempotent patch: add the interior-harmonic group to the declared-number registry."""
from pathlib import Path

path = Path("src/tensorbem/benchmarks/declared_numbers.py")
text = path.read_text()

# 1) import the benchmark module
old_imports = """from tensorbem.benchmarks import (
    cylinder_added_mass as cylinder_benchmark,
    ellipsoid_added_mass as ellipsoid_benchmark,
    ursell_cylinder as ursell_benchmark,
)"""
new_imports = """from tensorbem.benchmarks import (
    cylinder_added_mass as cylinder_benchmark,
    ellipsoid_added_mass as ellipsoid_benchmark,
    interior_harmonics as interior_harmonics_benchmark,
    ursell_cylinder as ursell_benchmark,
)"""
if "interior_harmonics as interior_harmonics_benchmark" not in text:
    assert old_imports in text
    text = text.replace(old_imports, new_imports)

# 2) register the 1280-panel interior-harmonic rows after the Ursell 256 rows
anchor = """        declared=6.8412e-3,
        tier="heavy",
    ),
)"""
entries = """        declared=6.8412e-3,
        tier="heavy",
    ),
    # --- 3-D interior solid harmonics (Laplace, exact polynomial truth) -----
    DeclaredNumber(
        key="interior_harmonic_deg1_flux_1280",
        description="flux relative L2 error, degree 1 (x), 1280 panels",
        source="docs/classical_benchmarks.md section 5.2 table 3",
        group="interior_harmonics",
        quantity="normal_derivative_relative_l2_error",
        parameters={"harmonic": "x", "subdivisions": 3},
        declared=3.7640e-3,
        tier="standard",
    ),
    DeclaredNumber(
        key="interior_harmonic_deg2_flux_1280",
        description="flux relative L2 error, degree 2 (x2-y2), 1280 panels",
        source="docs/classical_benchmarks.md section 5.2 table 3",
        group="interior_harmonics",
        quantity="normal_derivative_relative_l2_error",
        parameters={"harmonic": "x2-y2", "subdivisions": 3},
        declared=7.3437e-3,
        tier="standard",
    ),
    DeclaredNumber(
        key="interior_harmonic_deg3_flux_1280",
        description="flux relative L2 error, degree 3 (x3-3xy2), 1280 panels",
        source="docs/classical_benchmarks.md section 5.2 table 3",
        group="interior_harmonics",
        quantity="normal_derivative_relative_l2_error",
        parameters={"harmonic": "x3-3xy2", "subdivisions": 3},
        declared=1.1908e-2,
        tier="standard",
    ),
    DeclaredNumber(
        key="interior_harmonic_deg1_field_rmse_1280",
        description="interior probe RMSE, degree 1 (x), 1280 panels",
        source="docs/classical_benchmarks.md section 5.2 table 3",
        group="interior_harmonics",
        quantity="field_rmse",
        parameters={"harmonic": "x", "subdivisions": 3},
        declared=2.293e-4,
        tier="standard",
    ),
    DeclaredNumber(
        key="interior_harmonic_deg2_field_rmse_1280",
        description="interior probe RMSE, degree 2 (x2-y2), 1280 panels",
        source="docs/classical_benchmarks.md section 5.2 table 3",
        group="interior_harmonics",
        quantity="field_rmse",
        parameters={"harmonic": "x2-y2", "subdivisions": 3},
        declared=2.308e-4,
        tier="standard",
    ),
    DeclaredNumber(
        key="interior_harmonic_deg3_field_rmse_1280",
        description="interior probe RMSE, degree 3 (x3-3xy2), 1280 panels",
        source="docs/classical_benchmarks.md section 5.2 table 3",
        group="interior_harmonics",
        quantity="field_rmse",
        parameters={"harmonic": "x3-3xy2", "subdivisions": 3},
        declared=1.824e-4,
        tier="standard",
    ),
)"""
if "interior_harmonic_deg1_flux_1280" not in text:
    assert text.count(anchor) == 1
    text = text.replace(anchor, entries)

# 3) dispatch branch
anchor2 = """    if group == "ursell_reference":"""
branch = """    if group == "interior_harmonics":
        row = interior_harmonics_benchmark.run_case(
            str(parameters["harmonic"]), int(parameters["subdivisions"])
        )
        return {
            "normal_derivative_relative_l2_error": float(
                row["normal_derivative_relative_l2_error"]
            ),
            "field_rmse": float(row["field_rmse"]),
            "panels": float(row["panels"]),
        }
    if group == "ursell_reference":"""
if 'group == "interior_harmonics"' not in text:
    assert text.count(anchor2) == 1
    text = text.replace(anchor2, branch)

path.write_text(text)
print("patched")