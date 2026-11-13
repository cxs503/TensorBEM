#!/usr/bin/env python3
"""Command-line front end for the case-pool registry.

The platform dispatches anchors by ``case_code``; this script is the generic
caller that no longer needs to know which driver answers which anchor::

    PYTHONPATH=src python3 scripts/case_pool_run.py --list
    PYTHONPATH=src python3 scripts/case_pool_run.py --check
    PYTHONPATH=src python3 scripts/case_pool_run.py --run BEM-SUBMERGED-SPHERE-HAVELOCK -- --stage cheap

``--check`` verifies that every registered entry/artefact/document exists and
prints the verdict stored in the artefact; ``--run`` executes the registered
driver (extra arguments are passed through after ``--``).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from tensorbem.benchmarks import case_pool_registry as registry


def _list() -> int:
    for entry in registry.list_cases():
        missing = entry.missing()
        print(f"{entry.case_code:<40} tol={entry.tolerance:<5} "
              f"entry={entry.entry} {'MISSING:' + ','.join(missing) if missing else ''}")
    return 0


def _check(code: str | None) -> int:
    codes = [registry.normalise(code)] if code else [e.case_code for e in registry.list_cases()]
    bad = 0
    for case_code in codes:
        entry = registry.resolve(case_code)
        missing = entry.missing()
        try:
            verdict = registry.artifact_verdict(case_code)
        except Exception as exc:  # noqa: BLE001
            verdict = f"unreadable ({exc})"
        status = "OK" if not missing else "MISSING"
        print(json.dumps({
            "case_code": entry.case_code,
            "status": status,
            "verdict": verdict,
            "missing": missing,
            "repro_cmd": entry.repro_cmd,
        }, ensure_ascii=False))
        bad += 1 if missing else 0
    return 0 if bad == 0 else 1


def _run(code: str, extra: list[str]) -> int:
    entry = registry.resolve(code)
    if not entry.path(entry.entry).is_file():
        print(f"entry not found: {entry.entry}", file=sys.stderr)
        return 2
    env = dict(os.environ)
    src = str(registry.REPO_ROOT / "src")
    env["PYTHONPATH"] = src if not env.get("PYTHONPATH") else f"{src}:{env['PYTHONPATH']}"
    cmd = [sys.executable, str(entry.path(entry.entry)), *entry.entry_args, *extra]
    print("+ " + " ".join(cmd), file=sys.stderr)
    return subprocess.run(cmd, cwd=registry.REPO_ROOT, env=env, check=False).returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--check", nargs="?", const="", default=None,
                        help="verify registered paths; optional case code")
    parser.add_argument("--run", metavar="CASE_CODE", default=None)
    parser.add_argument("extra", nargs=argparse.REMAINDER,
                        help="arguments forwarded to the driver after --")
    args = parser.parse_args(argv)

    if args.list:
        return _list()
    if args.check is not None:
        return _check(args.check or None)
    if args.run:
        extra = [item for item in args.extra if item != "--"]
        return _run(args.run, extra)
    parser.print_help()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())