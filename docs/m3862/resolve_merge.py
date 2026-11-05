"""Resolve the two conflicted files of PR #136 (agent/m3250 -> main) by hand.

Resolution policy (documented in docs/m3862/merge_report.md):

* ``src/tensorbem/benchmarks/declared_numbers.py``: take the ``agent/m3250``
  side of every hunk -- it is the *superset* (85 rows / 63 three-dimensional
  rows versus 84 / 62 on main, generator-based registry, renamed groups
  ``hemisphere3d``, ``hemisphere3d_chief``, ``submerged_sphere3d``) -- and graft
  main's sharper pins that the narrower registry did not carry (rotational
  coupling 2.151e-4 / 2.6481e-5 instead of the two-significant-digit
  2.2e-4 / 2.6e-5; the 4-digit observed orders 1.8021 / 1.9377 as a comment
  under the two-decimal doc pin 1.80 / 1.94).
* ``tests/test_declared_numbers.py``: keep *both* sides -- main's two tests are
  re-pointed at the renamed groups instead of being dropped.

The script is deterministic and idempotent: it refuses to run on a file that no
longer carries conflict markers.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def resolve_take_theirs(text: str) -> str:
    """Drop every ``HEAD`` side, keep the ``origin/agent/m3250`` side."""
    out: list[str] = []
    lines = text.splitlines(keepends=True)
    index = 0
    while index < len(lines):
        line = lines[index]
        if line.startswith("<<<<<<< "):
            index += 1
            while not lines[index].startswith("======="):
                index += 1
            index += 1
            while not lines[index].startswith(">>>>>>> "):
                out.append(lines[index])
                index += 1
            index += 1
            continue
        out.append(line)
        index += 1
    return "".join(out)


def main() -> int:
    target = ROOT / "src/tensorbem/benchmarks/declared_numbers.py"
    text = target.read_text(encoding="utf-8")
    if "<<<<<<< " not in text:
        print("no conflict markers: already resolved")
        return 0
    resolved = resolve_take_theirs(text)
    if "<<<<<<< " in resolved or ">>>>>>> " in resolved:
        raise SystemExit("unresolved markers remain")
    target.write_text(resolved, encoding="utf-8")
    print("resolved", target.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())