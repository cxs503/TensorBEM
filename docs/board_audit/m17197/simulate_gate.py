#!/usr/bin/env python3
"""Dry-run the platform maturity gate against a candidate verdict payload.

This does not write anything: it imports /root/tp-agent/maturity_gate.py and calls
judge(level, evidence, repo, audit_mid) exactly as the gate daemon does, so the audit
can see whether the declared level survives machine re-check (and why not).

Run: python3 docs/board_audit/m17197/simulate_gate.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from insert_verdict import BY, LEVEL, REPO, SCOPE, build_evidence  # noqa: E402


def load_gate():
    spec = importlib.util.spec_from_file_location("maturity_gate", "/root/tp-agent/maturity_gate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    gate = load_gate()
    ev = build_evidence()
    level, reasons = gate.judge(LEVEL, ev, REPO, 17197)
    print(json.dumps({"declared": LEVEL, "gate_level": level, "reasons": reasons,
                      "repo": REPO, "scope": SCOPE, "verdict_by": BY},
                     indent=2, ensure_ascii=False))
    if level != LEVEL:
        raise SystemExit(f"GATE WOULD DOWNGRADE: declared {LEVEL} -> {level}")


if __name__ == "__main__":
    main()