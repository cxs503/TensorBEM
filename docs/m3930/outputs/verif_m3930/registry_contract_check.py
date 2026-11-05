"""Bounded verification of the merged declared-number registry (no long solves).

The full standard-tier sweep of `tests/test_declared_numbers.py` needs ~2 minutes
on an idle machine and much more on the loaded host this merge was healed on, so
this script proves the parts the merge could plausibly break, cheaply:

1. every registered entry resolves to a quantity that its group handler returns
   -- checked with the heavy benchmark runners monkeypatched by *synthetic*
   study payloads, so both pin conventions (main's ``sub<level>_<quantity>`` and
   m3306's finest-level keys) are exercised without assembling a 256-panel
   matrix;
2. the two 3-D handlers really do return the union of both conventions from a
   single study dict;
3. the tier bookkeeping of the merged registry (which entries the default CI
   sweep runs) is reported, because the union adds entries.

The numeric reproduction of every 3-D table cell is covered by
`tests/test_declared_numbers.py` (both sides' tests are kept) and by
`adjudicate_table10_cell.py`, which runs the real 16/64/256-panel solves.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from tensorbem.benchmarks import declared_numbers as dn

OUT = Path(__file__).resolve().parent / "registry_contract_check.json"


def synthetic_study(levels=(1, 2, 3), ka=1.0):
    """Study dict shaped like the real run_refinement_study output."""
    rows = []
    for i, sub in enumerate(levels):
        h = 0.5 / (2.0**i)
        err = 0.4 * (h / 0.5) ** 2
        rows.append(
            {
                "subdivisions": sub,
                "panel_count": 16 * sub**2,
                "h_mean": h,
                "ka": ka,
                "condition_number": 3.0 + i,
                "heave_added_mass_rel_error": err,
                "heave_damping_rel_error": err * 1.1,
                "surge_added_mass_rel_error": err * 1.2,
                "surge_damping_rel_error": err * 1.3,
                "vertical_force_rel_error": err * 0.9,
                "excitation_force_x_rel_error": err * 0.8,
                "excitation_force_z_rel_error": err * 0.7,
                "rotational_coupling_ratio": 2.6e-5 * (1 + 0.1 * i),
                "observed_order_a33": 2.0,
                "observed_order_b33": 1.9,
                "observed_order_a11": 2.0,
                "observed_order_fz": 2.0,
                "far_field_damping_rel_diff_heave": 5.45e-4,
                "haskind_reciprocity_rel_diff_x": 2.09e-4,
                "haskind_reciprocity_rel_diff_z": 8.09e-4,
                "rotational_added_mass_max_ratio": 2.65e-5,
            }
        )
    return {"benchmark": "synthetic", "levels": rows}


def synthetic_irregular(ka=2.58):
    return [
        {
            "ka": ka,
            "unregularized": {
                "condition_number": 145.66,
                "a33_dimless": 0.3481,
                "b33_dimless": -0.1042,
            },
            "chief_regularized": {
                "condition_number": 4.24,
                "a33_dimless": 0.3919,
                "b33_dimless": 0.0687,
                "chief_point_count": 5.0,
            },
            "exact_a33": 0.4007373824061046,
            "exact_b33": 0.06312,
        }
    ]


class FakeModule:
    def __init__(self):
        self.calls = 0

    def run_refinement_study(self, **kwargs):
        self.calls += 1
        return synthetic_study(levels=tuple(kwargs.get("subdivisions", (1, 2, 3))))

    def run_irregular_frequency_study(self, **kwargs):
        self.calls += 1
        return synthetic_irregular()


problems: list[str] = []
fake = FakeModule()
original = {
    "hemisphere": dn.hemisphere_benchmark,
    "sphere": dn.sphere_benchmark,
    "irregular": dn.hemisphere_benchmark.run_irregular_frequency_study,
}
dn.hemisphere_benchmark = fake
dn.sphere_benchmark = fake

groups_3d = {"hulme_hemisphere", "submerged_sphere", "hulme_chief", "hulme_irregular"}
resolved: dict[tuple[str, str], dict[str, float]] = {}
per_entry = []
for entry in dn.DECLARED_NUMBERS:
    if entry.group not in groups_3d:
        continue
    ck = (entry.group, json.dumps(entry.parameters, sort_keys=True))
    if ck not in resolved:
        resolved[ck] = dn._run_group(entry.group, entry.parameters)
    values = resolved[ck]
    if entry.quantity not in values:
        problems.append(f"{entry.key}: quantity {entry.quantity!r} missing from group {entry.group}")
    per_entry.append({"key": entry.key, "quantity": entry.quantity, "tier": entry.tier})

dn.hemisphere_benchmark = original["hemisphere"]
dn.sphere_benchmark = original["sphere"]

# the union glue: both conventions must come out of one 3-D study
hemi_keys = dn._run_group.__wrapped__ if hasattr(dn._run_group, "__wrapped__") else None
union_probe = {}
dn.hemisphere_benchmark = fake
study = fake.run_refinement_study(subdivisions=(1, 2, 3))
main_rows = dn._study_rows(study)
m3306_rows = dn._finest_level(study, extra_orders={"observed_order_b11": "surge_damping_rel_error"})
union_probe["main_convention_keys"] = sorted(main_rows)[:6]
union_probe["m3306_convention_keys"] = sorted(m3306_rows)
union_probe["overlap"] = sorted(set(main_rows) & set(m3306_rows))
union_probe["union_size"] = len(set(main_rows) | set(m3306_rows))
if not union_probe["m3306_convention_keys"]:
    problems.append("_finest_level returned nothing on the synthetic study")
if not any(k.startswith("sub2_") for k in main_rows):
    problems.append("_study_rows did not emit sub<level>_ keys")
dn.hemisphere_benchmark = original["hemisphere"]

tier_counts: dict[str, int] = {}
group_tier: dict[str, dict[str, int]] = {}
for entry in dn.DECLARED_NUMBERS:
    tier_counts[entry.tier] = tier_counts.get(entry.tier, 0) + 1
    group_tier.setdefault(entry.group, {}).setdefault(entry.tier, 0)
    group_tier[entry.group][entry.tier] += 1

payload = {
    "registry_len": len(dn.DECLARED_NUMBERS),
    "unique_keys": len({e.key for e in dn.DECLARED_NUMBERS}),
    "entries_checked_against_synthetic_handlers": len(per_entry),
    "tier_counts": tier_counts,
    "group_tier_counts": group_tier,
    "union_probe": union_probe,
    "problems": problems,
    "note": (
        "synthetic payloads only exercise the dispatch/quantity contract; the "
        "numbers themselves are recomputed by tests/test_declared_numbers.py and "
        "adjudicate_table10_cell.py"
    ),
}
OUT.write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps({k: payload[k] for k in ("registry_len", "unique_keys", "tier_counts", "union_probe", "problems")}, indent=2))
raise SystemExit(1 if problems else 0)