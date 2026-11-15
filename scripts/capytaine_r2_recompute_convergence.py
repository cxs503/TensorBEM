"""Re-derive the round-2 mesh-convergence ladder from the frozen matrices.

The committed ``benchmarks/verified/capytaine_crosscheck_r2/results.json`` stores
**every** 6x6 matrix it compared (TensorBEM at each mesh level, capytaine at its
finest level).  The convergence ladder is a pure function of those matrices, so
it can be re-derived without solving anything -- which is what this script does:

    python scripts/capytaine_r2_recompute_convergence.py [--check]

``--check`` only reports the difference and exits non-zero if the committed
block is stale; without it the block, ``docs/capytaine_crosscheck_r2.md`` and
``docs/fig_capytaine_r2_convergence.png`` are rewritten in place.

Why this exists: the ladder used to walk ``added_mass`` only, so it could not
see the damping entry that leaves the registered diagonal band on
``hemisphere_r1`` at 1024 panels (``damping A[heave,heave]``, +5.82 %), and the
document's limitation section still described the pre-round-2 mesh budget.  Both
are arithmetic over the frozen artifact, hence regenerable offline; the numbers
in the prose are then the numbers in the artifact by construction, not by hand.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensorbem.benchmarks import capytaine_crosscheck_r2 as r2  # noqa: E402


def load_study() -> tuple[pathlib.Path, dict]:
    path = ROOT / r2.ARTIFACT_ROOT / r2.RESULT_FILE
    return path, json.loads(path.read_text())


def recompute(study: dict) -> dict:
    ladder = r2.convergence(study["tensorbem"], study["capytaine"])
    out = dict(study)
    out["convergence"] = ladder
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="do not write; fail if the committed convergence block is stale",
    )
    args = parser.parse_args(argv)

    path, study = load_study()
    fresh = recompute(study)
    stale = json.dumps(study["convergence"], sort_keys=True) != json.dumps(
        fresh["convergence"], sort_keys=True
    )
    for case, rows in fresh["convergence"].items():
        for row in rows:
            print(
                "{} level {} ({} faces): diagonal A {:.2f}%  B {:.2f}%  "
                "coupling A {:.2f}%  B {:.2f}%  -> worst diagonal {:.2f}%".format(
                    case,
                    row["level"],
                    row["n_faces"],
                    100.0 * row["worst_diagonal_added_mass"],
                    100.0 * row["worst_diagonal_damping"],
                    100.0 * row["worst_coupling_added_mass"],
                    100.0 * row["worst_coupling_damping"],
                    100.0 * row["worst_diagonal"],
                )
            )
    if args.check:
        if stale:
            print("STALE: committed convergence block differs from the recomputation")
            return 1
        print("OK: committed convergence block reproduces from the frozen matrices")
        return 0

    fresh["figures"] = study["figures"]
    doc = r2.render_document(fresh)
    stale_doc = (ROOT / r2.DOC_PATH).read_text() != doc
    if not stale and not stale_doc:
        print("nothing to do: committed convergence block already reproduces")
        return 0

    if stale:
        fresh["figures"] = [str(p) for p in r2.render_figures(fresh)]
        path.write_text(json.dumps(fresh, indent=2, sort_keys=True) + "\n")
        print("wrote {}".format(path.relative_to(ROOT)))
        for fig in fresh["figures"]:
            print("wrote {}".format(fig))
    if stale_doc:
        r2.DOC_PATH.write_text(doc)
        print("wrote {}".format(r2.DOC_PATH))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())