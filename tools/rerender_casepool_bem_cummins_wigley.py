"""Re-render the committed CASE-BEM-CUMMINS-WIGLEY artifacts without a BEM solve.

The case writes ``case_result.json`` *after* the figure, so a headless compute
node without matplotlib leaves the artifact directory holding the solved sweeps
but neither the JSON nor the Markdown.  This script closes that gap without
solving anything again: it loads ``sweep_<label>.npz``, rebuilds the figure,
and re-renders ``CASE.md`` from the JSON it just re-wrote.

Usage::

    PYTHONPATH=src python tools/rerender_casepool_bem_cummins_wigley.py \
        benchmarks/verified/casepool_bem_cummins_wigley
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tensorbem.benchmarks import case_bem_cummins_wigley as case  # noqa: E402


def main(argv: list[str]) -> int:
    out = Path(argv[1] if len(argv) > 1 else "benchmarks/verified/casepool_bem_cummins_wigley")
    results = json.loads((out / "case_result.json").read_text(encoding="utf-8"))
    sweeps = {
        label: case.load_sweep(out / f"sweep_{label}.npz")
        for label in results["settings"]["meshes"]
        if (out / f"sweep_{label}.npz").exists()
    }
    results["figure"] = case._make_figure(sweeps, results, out)
    (out / "case_result.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (out / "CASE.md").write_text(case.render_markdown(results), encoding="utf-8")
    print(json.dumps(results["verdict"], indent=2))
    if "C0" in results["gates"]:
        print("C0 solver residual max:", results["gates"]["C0"]["values"]["solver_relative_residual_max"])
    print("figure:", results["figure"])
    return 0 if results["verdict"]["passed"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv))