"""Re-judge the mutants that the 'before' sweep did not kill, after the guards.

Adding tests can only turn survivors into kills, so the killed rows of the
'before' sweep stay valid; only the un-killed ones are re-run.  This keeps the
after-measurement cheap and exactly comparable (same null sample, same seed).
"""
from __future__ import annotations

import json, sys, tempfile, shutil, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from tensorbem.benchmarks import mutation_audit as ma  # noqa: E402

before = Path(sys.argv[1]); after = Path(sys.argv[2])
extra = tuple(sys.argv[3:])
budget = float(sys.argv[4]) if len(sys.argv) > 4 else 80.0

base = next(t for t in ma.TARGETS if t.module == "tensorbem.analytic")
target = ma.Target(base.module, base.source, base.unit_tests + extra)
data = json.loads(before.read_text())
open_keys = {r["key"] for r in data["mutants"] if r["verdict"] != "killed"}
source = base.source.read_text()
pool = {(m.lineno, m.col, m.operator): m for m in ma.enumerate_mutants(base.module, source, base.source)}

after.parent.mkdir(parents=True, exist_ok=True)
done = {}
if after.exists():
    done = {r["key"]: r for r in json.loads(after.read_text())["mutants"]}

rows = {r["key"]: r for r in data["mutants"]}
todo = [m for k, m in pool.items() if m.key in open_keys and m.key not in done]
started = time.time()
if todo:
    staging = Path(tempfile.mkdtemp(prefix="tensorbem-rejudge-"))
    try:
        ma._shadow_package(staging); ma._check_isolation(staging / "src")
        for mutant in sorted(todo, key=lambda m: (m.lineno, m.col)):
            if time.time() - started >= budget:
                break
            row = ma._run_one(target, mutant, staging, 240.0, True)
            done[row["key"]] = row
            print(row["verdict"], row["key"], "|", row["summary"][:80], "| stage", row.get("stage"), flush=True)
    finally:
        shutil.rmtree(staging, ignore_errors=True)

rows.update(done)
ordered = [rows[r["key"]] for r in data["mutants"] if r["key"] in rows]
payload = {
    "module": "tensorbem.analytic",
    "seed": data["seed"], "max_per_module": data["max_per_module"],
    "extra_tests": list(extra), "unit_tests": list(target.unit_tests),
    "mutants": ordered,
    "score": {k: sum(1 for r in ordered if r["verdict"] == k) for k in ("killed", "survived", "invalid", "timeout")},
}
payload["score"]["selected"] = len(data["mutants"]); payload["score"]["done"] = len(ordered)
after.write_text(json.dumps(payload, indent=2, sort_keys=True))
print("rejudged", len(done), "of", len(open_keys), "->", payload["score"])
