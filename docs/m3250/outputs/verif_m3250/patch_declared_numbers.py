"""Splice the 3-D free-surface registry block into declared_numbers.py (idempotent)."""

from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
TARGET = ROOT / "src" / "tensorbem" / "benchmarks" / "declared_numbers.py"
HERE = pathlib.Path(__file__).resolve().parent

text = TARGET.read_text(encoding="utf-8")

if "_free_surface_three_dimensional_numbers" in text:
    print("already patched")
    sys.exit(0)

OLD_IMPORTS = (
    "from tensorbem.benchmarks import (\n"
    "    cylinder_added_mass as cylinder_benchmark,\n"
    "    ellipsoid_added_mass as ellipsoid_benchmark,\n"
    "    interior_harmonics as interior_harmonics_benchmark,\n"
    "    ursell_cylinder as ursell_benchmark,\n"
    ")"
)

NEW_IMPORTS = (
    "from tensorbem.benchmarks import (\n"
    "    cylinder_added_mass as cylinder_benchmark,\n"
    "    ellipsoid_added_mass as ellipsoid_benchmark,\n"
    "    floating_hemisphere as floating_hemisphere_benchmark,\n"
    "    interior_harmonics as interior_harmonics_benchmark,\n"
    "    submerged_sphere_wave as submerged_sphere_wave_benchmark,\n"
    "    ursell_cylinder as ursell_benchmark,\n"
    ")"
)

OLD_STDLIB = "import argparse\nimport json\nimport math\n"
NEW_STDLIB = "import argparse\nimport json\nimport math\nimport numbers\n"

OLD_PROVENANCE = """``dev_report_tensorbem_declared_numbers.md`` rather than silently overwritten.
\"\"\""""

NEW_PROVENANCE = """``dev_report_tensorbem_declared_numbers.md`` rather than silently overwritten.

Provenance note (2026-11-04). The three-dimensional free-surface tables of
``docs/marine_bem_benchmarks.md`` (sections 4.10-4.11: the Hulme floating
hemisphere refinement, the interior-CHIEF irregular-frequency comparison and
the Wang/Linton submerged-sphere refinement) were documentation-only: the
numbers came from the ``floating_hemisphere`` and ``submerged_sphere_wave``
command-line studies, and the tests that did exist guard the *physics* at a
few-percent threshold, not the quoted convergence columns. Every column of
those three tables is now registered (``hemisphere3d``,
``hemisphere3d_chief``, ``submerged_sphere3d`` groups) and recomputed on the
``standard`` tier, which costs about a minute of three-dimensional solves.
\"\"\""""

OLD_GROUP_DOC = '''    """Which benchmark to run: ``cylinder``, ``ellipsoid``, ``ursell`` or
    ``ursell_reference``."""'''

NEW_GROUP_DOC = '''    """Which benchmark to run: ``cylinder``, ``ellipsoid``,
    ``interior_harmonics``, ``ursell``, ``ursell_reference``, ``hemisphere3d``,
    ``hemisphere3d_chief`` or ``submerged_sphere3d``."""'''

OLD_REGISTRY_ANCHOR = "\n#: The registry. ``source`` cites"
BLOCK = (HERE / "fs3d_block.txt").read_text(encoding="utf-8")

OLD_LITERAL_END = "\n)\n\n_URSELL_RADIUS = 1.0"
NEW_LITERAL_END = "\n    *_free_surface_three_dimensional_numbers(),\n)\n\n_URSELL_RADIUS = 1.0"

RUNNER = (HERE / "fs3d_runner.txt").read_text(encoding="utf-8")
OLD_RUNNER_ANCHOR = '    raise ValueError(f"unknown benchmark group: {group!r}")'
NEW_RUNNER_ANCHOR = RUNNER + "\n" + OLD_RUNNER_ANCHOR

for old, new, label in (
    (OLD_STDLIB, NEW_STDLIB, "stdlib imports"),
    (OLD_IMPORTS, NEW_IMPORTS, "benchmark imports"),
    (OLD_PROVENANCE, NEW_PROVENANCE, "module docstring"),
    (OLD_GROUP_DOC, NEW_GROUP_DOC, "group docstring"),
    (OLD_RUNNER_ANCHOR, NEW_RUNNER_ANCHOR, "group runner"),
    (OLD_REGISTRY_ANCHOR, "\n" + BLOCK + "\n" + OLD_REGISTRY_ANCHOR, "registry block"),
    (OLD_LITERAL_END, NEW_LITERAL_END, "registry literal"),
):
    if text.count(old) != 1:
        raise SystemExit(f"anchor {label!r} matched {text.count(old)} times")
    text = text.replace(old, new)

TARGET.write_text(text, encoding="utf-8")
print("patched", TARGET)