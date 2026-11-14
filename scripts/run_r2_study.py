"""Producer driver for the round-2 capytaine cross-validation study.

``python -m tensorbem.benchmarks.capytaine_crosscheck_r2`` runs every case
serially; the finest meshes of this round (1024 / 1026 panels) make that a
~12 min single-core job.  This driver splits the cases over ``--jobs``
processes and caches each case's solve, so a re-assembly of the document and
the figures costs nothing and a single case can be re-run in isolation:

    python scripts/run_r2_study.py --jobs 4            # full study
    python scripts/run_r2_study.py --assemble-only     # doc/figures from cache

The cache lives under ``benchmarks/verified/capytaine_crosscheck_r2/solves`` and
is keyed by case name; the capytaine reference is never re-run here.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from tensorbem.benchmarks import capytaine_crosscheck_r2 as r2  # noqa: E402

CACHE = pathlib.Path(r2.ARTIFACT_ROOT) / "solves"


def _solve_case(case: str) -> tuple[str, dict]:
    started = time.time()
    block = r2.run_case(case)
    print(
        "  {}: {} levels, {} panels, {:.1f} s".format(
            case,
            len(block["levels"]),
            block["reference_faces"],
            time.time() - started,
        ),
        flush=True,
    )
    return case, block


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--assemble-only", action="store_true")
    parser.add_argument("--refresh", action="store_true", help="ignore the cache")
    parser.add_argument("--reference", type=pathlib.Path, default=None)
    args = parser.parse_args(argv)

    CACHE.mkdir(parents=True, exist_ok=True)
    tensorbem: dict[str, dict] = {}
    if not args.assemble_only:
        todo = [
            case
            for case in r2.CASES
            if args.refresh or not (CACHE / (case + ".json")).exists()
        ]
        if args.jobs > 1 and len(todo) > 1:
            with mp.Pool(min(args.jobs, len(todo))) as pool:
                solved = dict(pool.map(_solve_case, todo))
        else:
            solved = dict(_solve_case(case) for case in todo)
        for case, block in solved.items():
            (CACHE / (case + ".json")).write_text(json.dumps(block, sort_keys=True))

    for case in r2.CASES:
        path = CACHE / (case + ".json")
        if not path.exists():
            print("missing cached solve for {}; run without --assemble-only".format(case))
            return 2
        tensorbem[case] = json.loads(path.read_text())

    reference = r2.load_reference(args.reference)
    study = {
        "density": r2.DENSITY,
        "gravity": r2.GRAVITY,
        "nu_grid": list(r2.NU_GRID),
        "dof_order": list(r2.DOF_ORDER),
        "tensorbem": tensorbem,
        "deep_submergence": r2.deep_submergence(),
    }
    study = r2.assemble(study, reference)
    paths = r2.write_outputs(study)
    print("wrote {}".format(paths["result"]))
    print("wrote {}".format(paths["document"]))
    for fig in paths["figures"]:
        print("wrote {}".format(fig))
    print(
        "verdict {} | worst active {:.2%} | worst invariant ratio {:.2e} | "
        "deep {}".format(
            study["comparison"]["verdict"],
            study["comparison"]["worst_active_relative"],
            study["comparison"]["worst_invariant_ratio"],
            (study["deep_submergence"] or {}).get("verdict"),
        )
    )
    for case, block in study["comparison"]["cases"].items():
        if not block.get("compared"):
            continue
        print(
            "  {}: active {} null {} invariant {} conflict {} undeclared {} "
            "| worst active {:.2%} | {} | {} panels vs capytaine {}".format(
                case,
                block["n_active"],
                block["n_null"],
                block["n_invariant"],
                block["n_conflict"],
                block["n_undeclared_null"],
                block["worst_active_relative"],
                block["verdict"],
                block["tensorbem_faces"],
                block["capytaine_faces"],
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())