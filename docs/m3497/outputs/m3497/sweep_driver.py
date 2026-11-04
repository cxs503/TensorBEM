"""Before/after mutation sweep on a uniform, fast 2-D guard set.

``run_mutant`` from :mod:`mutation_fs2d` is reused unchanged; only the guard
file list differs, so the before/after numbers are comparable.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mutation_fs2d as M

ROOT = Path(__file__).resolve().parents[2]
BEFORE = [
    "tests/test_mutation_gaps.py",
    "tests/test_free_surface_kernel.py",
    "tests/test_finite_depth.py",
]
AFTER = BEFORE + ["tests/test_mutation_gaps_freesurface_2d.py"]

out = {"before": [], "after": []}
for mid, desc, patch in M.MUTANTS:
    rb = M.run_mutant(mid, patch, BEFORE, ROOT)
    rb["description"] = desc
    ra = M.run_mutant(mid, patch, AFTER, ROOT)
    ra["description"] = desc
    out["before"].append(rb)
    out["after"].append(ra)
    print(f"{mid:42s} before={'KILL' if rb['killed'] else 'ALIVE'} "
          f"after={'KILL' if ra['killed'] else 'ALIVE'}", flush=True)

for key in ("before", "after"):
    rows = out[key]
    n = len(rows)
    k = sum(r["killed"] for r in rows)
    out[key + "_kill_ratio"] = k / n if n else 0.0
    print(f"{key}: {k}/{n} = {out[key + '_kill_ratio']:.3f}")

Path(ROOT / "outputs/m3497/mutation_sweep.json").write_text(
    json.dumps(out, indent=2), encoding="utf-8"
)
print("-> outputs/m3497/mutation_sweep.json")
