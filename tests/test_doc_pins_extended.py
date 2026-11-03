"""Pins for the remaining milestone tables (sphere, Hulme, Wang/Linton).

``docs/marine_bem_benchmarks.md`` Tables 2, 10 and 12 quote concrete numbers
that were previously prose-only claims (see
``dev_report_tensorbem_declared_numbers.md`` §5: "其余里程碑发布数字（球体
20–1280 面元、Hulme 半球、Wang/Linton 潜没球）同样只有散文声明").  This module
registers them in ``tensorbem.benchmarks.doc_pins`` so that any drift between
the code and the documentation is a red build.

The fast rows are enforced by ``test_documented_benchmarks.py`` through
``doc_pins.check``; the 1280-panel sphere row (the README headline 3-D
number) is re-executed here because it is the single most cited 3-D result.
"""

from __future__ import annotations

import json
import math

from tensorbem.benchmarks import doc_pins
from tensorbem.benchmarks.sphere_added_mass import run as run_sphere

REPO_ROOT = doc_pins.REPO_ROOT if hasattr(doc_pins, "REPO_ROOT") else None
from pathlib import Path

PIN_PATH = Path(__file__).resolve().parents[1] / "docs" / "benchmark_pins.json"

# Declared values from docs/marine_bem_benchmarks.md Table 2 (1280 panels).
DECLARED_1280 = {
    "h_max": 0.16465,
    "added_mass": 2137.16456,
    "added_mass_relative_error": 4.4674e-3,
    "normal_derivative_relative_l2_error": 6.8458e-3,
}


def test_pin_file_covers_all_three_tables():
    document = json.loads(PIN_PATH.read_text(encoding="utf-8"))
    pins = document["pins"]
    for name in (
        "sphere_added_mass_3d_20",
        "sphere_added_mass_3d_80",
        "sphere_added_mass_3d_320",
        "sphere_added_mass_3d_1280",
        "hulme_hemisphere_16",
        "hulme_hemisphere_64",
        "hulme_hemisphere_256",
        "submerged_sphere_20",
        "submerged_sphere_80",
        "submerged_sphere_320",
    ):
        assert name in pins, f"{name} missing from {PIN_PATH}"


def test_sphere_1280_pin_matches_documented_row():
    """The headline 3-D number must still be reproduced by the code."""
    document = json.loads(PIN_PATH.read_text(encoding="utf-8"))
    entry = document["pins"]["sphere_added_mass_1280"] if "sphere_added_mass_1280" in document["pins"] else document["pins"]["sphere_added_mass_3d_1280"]
    measured = run_sphere(subdivisions=3)
    for metric, declared in DECLARED_1280.items():
        actual = float(measured[metric])
        pinned = float(entry["metrics"][metric])
        # pin file and live code must agree to machine precision
        assert math.isclose(actual, pinned, rel_tol=1.0e-9), (
            f"{metric}: live {actual!r} vs pin {pinned!r}"
        )
        # and both must agree with the documented value at printed precision
        assert math.isclose(actual, declared, rel_tol=5.0e-3), (
            f"{metric}: live {actual:.6g} vs documented {declared:.6g}"
        )
