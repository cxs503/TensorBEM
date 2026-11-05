"""One-shot conflict resolver for the m3245 -> main merge (mission m3795).

Resolution policy: semantic union. Both sides' declarations are kept; the
dispatcher accepts every quantity key either side produces and both spellings
of the submerged-sphere group.

Run:  python3 artifacts/merge_resolve_m3795.py
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REG = ROOT / "src/tensorbem/benchmarks/declared_numbers.py"
TEST = ROOT / "tests/test_declared_numbers.py"


def split_conflicts(text):
    """Yield (start, end, ours, theirs) line-index blocks for each conflict."""
    lines = text.splitlines(keepends=True)
    blocks, i = [], 0
    while i < len(lines):
        if lines[i].startswith("<<<<<<<"):
            start = i
            i += 1
            ours = []
            while not lines[i].startswith("======="):
                ours.append(lines[i])
                i += 1
            i += 1
            theirs = []
            while not lines[i].startswith(">>>>>>>"):
                theirs.append(lines[i])
                i += 1
            blocks.append((start, i, ours, theirs))
        i += 1
    return lines, blocks


def resolve(path, handlers):
    lines, blocks = split_conflicts(path.read_text())
    assert len(blocks) == len(handlers), (path, len(blocks))
    out, cursor = [], 0
    for (start, end, ours, theirs), handler in zip(blocks, handlers):
        out.extend(lines[cursor:start])
        out.extend(handler(ours, theirs))
        cursor = end + 1
    out.extend(lines[cursor:])
    path.write_text("".join(out))
    print(f"{path.name}: resolved {len(blocks)} conflict(s)")


def imports(ours, theirs):
    """Union of both alias sets; the merged dispatcher uses the explicit names."""
    return [
        "    floating_hemisphere as floating_hemisphere_benchmark,\n",
        "    interior_harmonics as interior_harmonics_benchmark,\n",
        "    submerged_sphere_wave as submerged_sphere_wave_benchmark,\n",
    ]


def declared_numbers(ours, theirs):
    """Keep main's hand-written rows *and* m3245's generated rows."""
    return (
        ["    # --- main: hand-written table 10/11/12 rows (kept verbatim) ---------\n"]
        + ours
        + [
            "    # --- m3245: the same documented tables generated from the registry\n"
            "    # run JSON, plus the Wang/Linton submerged-sphere rows that main\n"
            "    # does not carry. Both sets are kept: they are two independent\n"
            "    # extraction paths of the same document cells, and the generated\n"
            "    # set is the only one covering sections 4.11's 56 sphere rows.\n"
        ]
        + theirs
    )


DISPATCH = '''    if group == "hulme_hemisphere":
        study = floating_hemisphere_benchmark.run_refinement_study(
            subdivisions=tuple(int(n) for n in parameters.get("subdivisions", (1, 2, 3))),
            ka=float(parameters.get("ka", 1.0)),
            radius=float(parameters.get("radius", 1.0)),
            density=float(parameters.get("density", 1025.0)),
            gravity=float(parameters.get("gravity", 9.80665)),
        )
        # Union of both quantity-key conventions: main's ``sub<level>_<q>``
        # (from ``_study_rows``) and m3245's ``panels<N>.<q>`` / ``reference.<q>``
        # (from ``_flatten_*``).
        values = _study_rows(study)
        values.update(_flatten_levels(study["levels"]))
        values.update(_flatten_reference(study["exact_reference"]))
        return values
    if group in _SUBMERGED_SPHERE_GROUPS:
        study = submerged_sphere_wave_benchmark.run_refinement_study(
            subdivisions=tuple(int(n) for n in parameters.get("subdivisions", (0, 1, 2))),
            ka=float(parameters.get("ka", 0.8)),
            radius=float(parameters.get("radius", 1.0)),
            submergence=float(parameters.get("submergence", 1.5)),
            density=float(parameters.get("density", 1025.0)),
            gravity=float(parameters.get("gravity", 9.80665)),
        )
        values = _study_rows(study)
        values.update(_flatten_levels(study["levels"]))
        values.update(_flatten_reference(study["exact_reference"]))
        return values
    if group == "hulme_chief":
        rows = floating_hemisphere_benchmark.run_irregular_frequency_study(
            subdivision=int(parameters.get("subdivision", 3)),
            ka_values=tuple(float(v) for v in parameters.get("ka_values", (2.58,))),
            radius=float(parameters.get("radius", 1.0)),
            density=float(parameters.get("density", 1025.0)),
            gravity=float(parameters.get("gravity", 9.80665)),
        )
        values = _chief_rows(rows)
        values.update(_flatten_chief_rows(rows))
        return values
'''


def dispatch(ours, theirs):
    return [DISPATCH]


def test_groups(ours, theirs):
    """Both spellings of the submerged-sphere group are registered."""
    return ['        "submerged_sphere",\n', '        "submerged_sphere_wave",\n']


resolve(REG, [imports, declared_numbers, dispatch])
resolve(TEST, [test_groups])