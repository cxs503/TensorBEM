"""Zero-silent-loss verifier for the PR #128 3-way merge (agent/m3181 <- main).

Mechanical overlays have silently dropped content in this repo before, so the
heal is checked line by line: every line that *either* parent added relative to
the merge base must still be present in the merged file, except for the handful
of lines that a documented rule rewrote on purpose.

Rules (see docs/m3913/pr128_merge_resolution.md):

* ``R1 imports``  - the two parents chose different local aliases for the same
  benchmark modules; the union keeps both aliases, so no single alias line
  survives verbatim (renamed/duplicated on purpose).
* ``R2 brief``    - the tests' rationale comment was fused into one sentence.
* ``R3 registry``- the tuple tail ``)`` became ``) + _free_surface_3d_rows()``.
* ``R4 group set``- the test's group literal is a sorted set union.
* ``R5 rename``   - main's ``_chief_rows(rows)`` study flattener was renamed
  ``_chief_study_rows`` because this branch already owns ``_chief_rows()``;
  Python would otherwise keep only the later definition (silent loss).

Usage::

    python3 docs/m3913/verify_union.py --repo . --merged worktree
"""

from __future__ import annotations

import argparse
import subprocess
import sys

FILES = (
    "src/tensorbem/benchmarks/declared_numbers.py",
    "tests/test_declared_numbers.py",
    "docs/marine_bem_benchmarks.md",
)

#: Additions a rule intentionally rewrote instead of keeping verbatim.
DRULES = {
    "R1_imports": (
        "    floating_hemisphere as hemisphere_benchmark,",
        "    floating_hemisphere as hulme_benchmark,",
        "    submerged_sphere_wave as sphere_benchmark,",
        "    submerged_sphere_wave as wang_linton_benchmark,",
    ),
    "R3_registry_tail": (")", ") + _free_surface_3d_rows()"),
    "R4_group_set": (
        "        \"hulme3d\",",
        "        \"hulme3d_chief\",",
        "        \"hulme_chief\",",
        "        \"hulme_hemisphere\",",
    ),
    "R5_rename": (
        "def _chief_rows(rows) -> dict[str, float]:",
        "        return _chief_rows(rows)",
    ),
    "R2_comment_fused": (
        "        # sign-bearing quantities exist (e.g. the unphysical negative B33' that",
        "        # table 11 quotes as the defect CHIEF removes), so only zero is illegal",
        "        # sign is exactly the claim the table makes.",
        "        # 3-D BIE produces an unphysical negative heave damping, and that",
    ),
}
REWRITTEN = {line for group in DRULES.values() for line in group}


def run(*args: str) -> str:
    return subprocess.run(args, check=True, capture_output=True, text=True).stdout


def added_lines(rev: str, base: str, path: str) -> list[str]:
    """Lines added by ``rev`` relative to ``base`` in ``path``."""
    diff = run("git", "diff", "-U0", base, rev, "--", path)
    out, keep = [], False
    for line in diff.splitlines():
        if line.startswith("@@"):
            keep = True
            continue
        if not keep or line.startswith("---") or line.startswith("+++"):
            continue
        if line.startswith("+"):
            out.append(line[1:])
    return out


def squash(lines: list[str]) -> str:
    return "\n".join(" ".join(line.split()) for line in lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    ap.add_argument("--main", default="origin/main")
    ap.add_argument("--branch", default="origin/agent/m3181")
    ap.add_argument("--merged", default="HEAD")
    args = ap.parse_args()

    worktree = args.merged.upper() == "WORKTREE"
    if not worktree:
        subprocess.run(["git", "-C", args.repo, "rev-parse", args.merged], check=True,
                       capture_output=True)
    base = run("git", "-C", args.repo, "merge-base", args.main, args.branch).strip()
    print(f"merge-base = {base}")
    failures = 0
    for path in FILES:
        if worktree:
            merged = open(f"{args.repo}/{path}", encoding="utf-8").read()
        else:
            merged = run("git", "-C", args.repo, "show", f"{args.merged}:{path}")
        merged_flat = squash(merged.splitlines())
        for side, rev in (("main", args.main), ("branch", args.branch)):
            kept = dropped = rule_handled = 0
            for line in added_lines(rev, base, path):
                if not line.strip():
                    continue
                norm = " ".join(line.split())
                if norm in merged_flat:
                    kept += 1
                elif line in REWRITTEN:
                    rule_handled += 1
                elif norm in squash(merged.splitlines()):
                    kept += 1
                else:
                    dropped += 1
                    print(f"  !! LOST from {side}: {path}: {line!r}")
            print(f"{path}: {side} additions kept={kept} rule-rewritten={rule_handled} lost={dropped}")
            failures += dropped
    print(f"\nverdict: {'PASS - zero silent loss' if failures == 0 else f'FAIL - {failures} line(s) lost'}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())