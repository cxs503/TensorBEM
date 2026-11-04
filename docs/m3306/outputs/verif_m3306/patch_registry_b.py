"""Second patch: pin the reproducible B11 order, add the passivity row, and
correct table 10's order column with a provenance note.

Idempotent guard: raises if already applied.
"""
from pathlib import Path

# ---------------------------------------------------- registry: B11 order
path = Path("src/tensorbem/benchmarks/declared_numbers.py")
text = path.read_text()
assert "declared=1.94" not in text, "already patched"

old = """        description="observed order of B11' between 64 and 256 panels",
        source="docs/marine_bem_benchmarks.md table 10 (section 4.10)",
        group="hulme_hemisphere",
        quantity="observed_order_b11",
        parameters={"subdivisions": [2, 3], "ka": 1.0},
        declared=1.88,"""
new = """        description="observed order of B11' between 64 and 256 panels",
        source="docs/marine_bem_benchmarks.md table 10 (section 4.10)",
        group="hulme_hemisphere",
        quantity="observed_order_b11",
        parameters={"subdivisions": [2, 3], "ka": 1.0},
        # Table 10 used to quote 1.88 for this cell: that is the same ratio
        # evaluated with the nominal refinement factor 2 instead of the mesh's
        # actual h_mean ratio 1.9642. The errors it is derived from (1.30e-1 and
        # 3.53e-2) have always been reproduced; only the derived order was off.
        declared=1.94,"""
assert old in text
text = text.replace(old, new, 1)

# ------------------------------------------- registry: passivity indicator
old = """    DeclaredNumber(
        key="hulme_irregular_chief_condition","""
new = """    DeclaredNumber(
        key="hulme_irregular_passivity_violated",
        description="1 when the un-augmented BIE returns B33' < 0 at Ka=2.58 (non-physical damping)",
        source="docs/marine_bem_benchmarks.md table 11 (section 4.10)",
        group="hulme_irregular",
        quantity="passivity_violated",
        parameters={"subdivisions": 3, "ka": 2.58},
        declared=1.0,
        tier="standard",
    ),
    DeclaredNumber(
        key="hulme_irregular_chief_condition","""
assert old in text
text = text.replace(old, new, 1)

old = """            "damping_magnitude": abs(float(unreg["b33_dimless"])),"""
new = """            "damping_magnitude": abs(float(unreg["b33_dimless"])),
            "passivity_violated": 1.0 if float(unreg["b33_dimless"]) < 0.0 else 0.0,"""
assert old in text
text = text.replace(old, new, 1)
path.write_text(text)

# ------------------------------------------------------- docs: order cell
docs = Path("docs/marine_bem_benchmarks.md")
dtext = docs.read_text()
old = "| 3 | 256 | 2.783 | **\\(7.41\\times10^{-3}\\)** | **\\(1.18\\times10^{-2}\\)** | **\\(1.30\\times10^{-2}\\)** | **\\(3.53\\times10^{-2}\\)** | **\\(5.67\\times10^{-3}\\)** | **2.06 / 1.98 / 2.01 / 1.88 / 2.02** |"
assert old in dtext, "table 10 row 3 not found"
new = "| 3 | 256 | 2.783 | **\\(7.41\\times10^{-3}\\)** | **\\(1.18\\times10^{-2}\\)** | **\\(1.30\\times10^{-2}\\)** | **\\(3.53\\times10^{-2}\\)** | **\\(5.67\\times10^{-3}\\)** | **2.06 / 1.98 / 2.01 / 1.94 / 2.02** |"
dtext = dtext.replace(old, new, 1)
docs.write_text(dtext)
print("patched registry + table 10 order cell")