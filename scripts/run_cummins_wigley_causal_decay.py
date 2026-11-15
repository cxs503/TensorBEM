"""Driver for the C5 attribution audit of CASE-BEM-CUMMINS-WIGLEY.

Stages::

    python scripts/run_cummins_wigley_causal_decay.py --levels coarse,medium \\
        --out benchmarks/verified/casepool_bem_cummins_wigley/causal_decay_audit.json \\
        --figures docs/case_bem_cummins_wigley_c5_attribution

Every number written to the JSON comes from the frozen case-pool sweeps
(``benchmarks/verified/casepool_bem_cummins_wigley/sweep_*.npz``); nothing is
re-solved and nothing is typed by hand.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from tensorbem.benchmarks import cummins_wigley_causal_decay as audit  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--levels", default="coarse,medium")
    parser.add_argument("--out", default=audit.ARTIFACT_PATH)
    parser.add_argument("--figures", default="docs/case_bem_cummins_wigley_c5_attribution")
    args = parser.parse_args(argv)

    levels = [lv.strip() for lv in args.levels.split(",") if lv.strip()]
    payload = {
        "case": "CASE-BEM-CUMMINS-WIGLEY",
        "audit": "C5 attribution",
        "gates_definition": audit.GATES,
        "levels": {},
    }
    for level in levels:
        result = audit.run_audit(level)
        failed = [k for k, g in result["gates"].items() if g["passed"] is False]
        undecided = [k for k, g in result["gates"].items() if g["passed"] is None]
        print(
            f"[{level}] {len(result['gates']) - len(failed) - len(undecided)}/{len(result['gates'])} pass "
            f"failed={failed or 'none'} undecided={undecided or 'none'} "
            f"| zeta legacy={result['legacy_route']['zeta_decay']:.5f} "
            f"ref={result['legacy_reference']['zeta_frequency_domain']:.5f} "
            f"exact={result['causal_model']['zeta_exact']:.5f} "
            f"causal={result['causal_model']['zeta_decay_time_domain']:.5f} "
            f"| {result['attribution']['verdict']}",
            flush=True,
        )
        result.pop("records", None)
        payload["levels"][level] = result

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1, default=audit._jsonable))
    print(f"[write] {out}")

    if args.figures:
        from tensorbem.benchmarks import cummins_wigley_causal_decay_figures as figs

        target = pathlib.Path(args.figures)
        target.mkdir(parents=True, exist_ok=True)
        for name, path in figs.make_all(target).items():
            print(f"[figure] {name} -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
