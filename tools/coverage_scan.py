"""Per-module coverage scan: how many tests / docs mention each ``tensorbem`` module.

Portable by construction: the scan root defaults to the repository that contains
this file (``tools/`` parent), and can be overridden with ``--root`` for other
checkouts.  An earlier revision hard-coded an absolute path to another agent's
worktree (``/root/tp-agent/work/m16890``), so running it from this repository
silently reported coverage for a *different* tree.
"""
from __future__ import annotations

import argparse
import glob
import os
import re
from pathlib import Path

DEFAULT_ROOT = Path(__file__).resolve().parents[1]


def scan(root: Path) -> list[tuple[int, int, str]]:
    root = Path(root)
    mods = sorted(os.path.basename(p)[:-3] for p in glob.glob(str(root / "src/tensorbem/*.py")))
    docs = []
    for pattern in ("docs/**/*.md", "*.md", "docs/*.json"):
        for d in glob.glob(str(root / pattern), recursive=True):
            try:
                docs.append((d, open(d, errors="ignore").read()))
            except Exception:
                pass
    tests = []
    for d in glob.glob(str(root / "tests/**/*.py"), recursive=True):
        try:
            tests.append((d, open(d, errors="ignore").read()))
        except Exception:
            pass
    rows = []
    for m in mods:
        pat = re.compile(r"\btensorbem\." + re.escape(m) + r"\b|\bimport " + re.escape(m) + r"\b")
        nd = sum(1 for _, t in docs if pat.search(t))
        nt = sum(1 for _, t in tests if pat.search(t))
        rows.append((nt, nd, m))
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(DEFAULT_ROOT),
                        help="repository root to scan (default: repository containing this file)")
    args = parser.parse_args(argv)
    rows = scan(Path(args.root))
    print(f"# scan root: {Path(args.root).resolve()}")
    for nt, nd, m in sorted(rows):
        print(f"tests={nt:3d} docs={nd:3d}  {m}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())