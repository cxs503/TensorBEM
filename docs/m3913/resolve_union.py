"""Mechanical 3-way union resolver for PR #128 (agent/m3181 <- main).

Not a generic tool: each conflict hunk is resolved by an explicit rule that
keeps the additions from *both* sides, so no side's content is dropped.
"""
import re
import sys

CONFLICT = re.compile(
    r"<<<<<<< HEAD\n(.*?)=======\n(.*?)>>>>>>> origin/main\n", re.S
)


def split_hunks(path):
    text = open(path, encoding="utf-8").read()
    return text, list(CONFLICT.finditer(text))


def apply(path, rules):
    text, hunks = split_hunks(path)
    if len(hunks) != len(rules):
        raise SystemExit(f"{path}: {len(hunks)} hunks, {len(rules)} rules")
    out, cursor = [], 0
    for match, rule in zip(hunks, rules):
        ours, theirs = match.group(1), match.group(2)
        out.append(text[cursor:match.start()])
        out.append(rule(ours, theirs))
        cursor = match.end()
    out.append(text[cursor:])
    open(path, "w", encoding="utf-8").write("".join(out))
    print(f"{path}: resolved {len(hunks)} hunks")


# ---------------------------------------------------------------- declared_numbers
DECL = "src/tensorbem/benchmarks/declared_numbers.py"


def imports(ours, theirs):
    """Union the benchmark-module aliases: main and this branch picked
    different alias names for the same two modules; keep all four."""
    return (
        "    floating_hemisphere as hemisphere_benchmark,\n"
        "    floating_hemisphere as hulme_benchmark,\n"
        "    interior_harmonics as interior_harmonics_benchmark,\n"
        "    submerged_sphere_wave as sphere_benchmark,\n"
        "    submerged_sphere_wave as wang_linton_benchmark,\n"
    )


def registry(ours, theirs):
    """Keep main's inline DeclaredNumber rows, then append this branch's
    programmatic ``_free_surface_3d_rows()`` extension after the tuple."""
    assert theirs.rstrip().endswith(")"), "unexpected main tail"
    body = theirs.rstrip()
    assert body.endswith("\n)")
    return body[: -len("\n)")] + "\n) + _free_surface_3d_rows()\n"


def dispatchers(ours, theirs):
    """Both sides add disjoint benchmark groups to ``_run_group``; keep both."""
    return ours + theirs


apply(DECL, [imports, registry, dispatchers])

# ---------------------------------------------------------------- tests
TESTS = "tests/test_declared_numbers.py"


def sign_comment(ours, theirs):
    return (
        "        # A quote may legitimately be non-positive: the unregularized\n"
        "        # 3-D BIE produces an unphysical negative heave damping (table 11\n"
        "        # quotes it as the defect CHIEF removes), and that sign is exactly\n"
        "        # the claim the table makes, so only zero is illegal.\n"
    )


def group_union(ours, theirs):
    e_ours = [line.strip() for line in ours.strip().splitlines()]
    e_theirs = [line.strip() for line in theirs.strip().splitlines()]
    merged = sorted(set(e_ours) | set(e_theirs))
    return "".join(f"        {item}\n" for item in merged)


apply(TESTS, [sign_comment, group_union])

# ---------------------------------------------------------------- shadow check
print("\n-- duplicate top-level definitions --")
src = open(DECL, encoding="utf-8").read()
names = re.findall(r"^def (\w+)", src, re.M)
dups = sorted({n for n in names if names.count(n) > 1})
print("duplicates:", dups or "none")
sys.exit(1 if dups else 0)