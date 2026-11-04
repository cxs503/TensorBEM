"""Cross-registry agreement: the same documented cells, registered twice.

`main` (merged via m3150/m3196-style work) and `agent/m3181` both encoded the
3-D free-surface tables of ``docs/marine_bem_benchmarks.md`` (tables 10-12) into
``DECLARED_NUMBERS``, under different group names and different key spellings.
After the union merge both registries coexist, so the two independently written
transcriptions can be compared cell by cell -- a free cross-check of the merge:
if the union had picked the wrong side of a hunk, the agreement would collapse.

The mapping below is *derived*, not hand-listed: keys are parsed into
``(family, level, canonical quantity)`` and intersected.
"""

from __future__ import annotations

import re

#: main-side short quantity name -> canonical quantity name.
_MAIN_QUANTITY = {
    "heave_added_mass_error": "a33_rel_error",
    "heave_damping_error": "b33_rel_error",
    "surge_added_mass_error": "a11_rel_error",
    "surge_damping_error": "b11_rel_error",
    "force_z_error": "fz_rel_error",
    "force_x_error": "fx_rel_error",
    "excitation_force_z_error": "fz_rel_error",
    "excitation_force_x_error": "fx_rel_error",
    "condition_number": "condition_number",
    "rotational_coupling": "rot_coupling",
    "order_a33": "order_a33",
    "order_b33": "order_b33",
    "order_a11": "order_a11",
    "order_fz": "order_fz",
    "unregularized_condition_number": "unreg_condition_number",
    "unregularized_a33": "unreg_a33",
    "unregularized_b33": "unreg_b33",
    "chief_condition_number": "chief_condition_number",
    "chief_a33": "chief_a33",
    "chief_b33": "chief_b33",
}

#: m3181-side quantity name -> canonical quantity name.
_OTHER_QUANTITY = {
    "heave_added_mass_rel_error": "a33_rel_error",
    "heave_damping_rel_error": "b33_rel_error",
    "surge_added_mass_rel_error": "a11_rel_error",
    "surge_damping_rel_error": "b11_rel_error",
    "excitation_force_z_rel_error": "fz_rel_error",
    "excitation_force_x_rel_error": "fx_rel_error",
    "condition_number": "condition_number",
    "rotational_added_mass_max_ratio": "rot_coupling",
    "observed_order_a33": "order_a33",
    "observed_order_b33": "order_b33",
    "observed_order_a11": "order_a11",
    "observed_order_fz": "order_fz",
    "unregularized_condition_number": "unreg_condition_number",
    "unregularized_a33_dimless": "unreg_a33",
    "unregularized_b33_dimless": "unreg_b33",
    "chief_regularized_condition_number": "chief_condition_number",
    "chief_regularized_a33_dimless": "chief_a33",
    "chief_regularized_b33_dimless": "chief_b33",
    "exact_a33_dimless": "exact_a33",
    "exact_b33_dimless": "exact_b33",
    "chief_point_count": "chief_point_count",
}

_MAIN_KEY = re.compile(
    r"^(?P<family>hulme_hemisphere|submerged_sphere)_(?P<level>\d+)_(?P<q>[a-z0-9_]+)$"
)
_MAIN_CHIEF_KEY = re.compile(r"^hulme_chief_2_58_(?P<q>[a-z0-9_]+)$")
_OTHER_KEY = re.compile(r"^(?P<family>hulme|wang_linton)_(?P<q>[a-z0-9_]+)_(?P<level>\d+)$")
_OTHER_CHIEF_KEY = re.compile(r"^hulme_chief_(?P<q>[a-z0-9_]+)$")

_FAMILY = {
    "hulme_hemisphere": "hulme",
    "submerged_sphere": "wang_linton",
    "hulme": "hulme",
    "wang_linton": "wang_linton",
}


def canonical(entries, side: str) -> dict[tuple[str, str, str], tuple[str, float]]:
    table = _MAIN_QUANTITY if side == "main" else _OTHER_QUANTITY
    out: dict[tuple[str, str, str], tuple[str, float]] = {}
    for entry in entries:
        key, value = entry.key, float(entry.declared)
        match = _MAIN_KEY.match(key) if side == "main" else _OTHER_KEY.match(key)
        if match:
            q = table.get(match.group("q"))
            if q is None:
                continue
            out[(_FAMILY[match.group("family")], match.group("level"), q)] = (key, value)
            continue
        match = _MAIN_CHIEF_KEY.match(key) if side == "main" else _OTHER_CHIEF_KEY.match(key)
        if match:
            q = table.get(match.group("q"))
            if q is None:
                continue
            out[("hulme_chief", "ka2.58", q)] = (key, value)
    return out


def compare(main_entries, other_entries) -> dict:
    left = canonical(main_entries, "main")
    right = canonical(other_entries, "m3181")
    shared = sorted(set(left) & set(right))
    cells = []
    for cell in shared:
        main_key, main_value = left[cell]
        other_key, other_value = right[cell]
        rel = (
            abs(main_value - other_value) / abs(main_value) if main_value else float("inf")
        )
        cells.append(
            {
                "family": cell[0],
                "level": cell[1],
                "quantity": cell[2],
                "main_key": main_key,
                "main": main_value,
                "m3181_key": other_key,
                "m3181": other_value,
                "rel_diff": rel,
            }
        )
    diffs = [c["rel_diff"] for c in cells if c["rel_diff"] != float("inf")]
    return {
        "cells": cells,
        "shared": len(cells),
        "max_rel_diff": max(diffs) if diffs else 0.0,
        "median_rel_diff": sorted(diffs)[len(diffs) // 2] if diffs else 0.0,
        "within_3pct": sum(1 for d in diffs if d < 0.03),
        "exact": sum(1 for d in diffs if d == 0.0),
        "only_main": len(set(left) - set(right)),
        "only_m3181": len(set(right) - set(left)),
    }