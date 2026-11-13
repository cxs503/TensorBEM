#!/usr/bin/env python3
"""Insert the board-audit verdict for m17339 (scope: m17296 CASE-BEM-CYL-DRIFT-ASYMP).

Idempotent on (repo, scope, verdict_by).  Before writing, the evidence bundle is
dry-run through the maturity_gate JUDGE_SCHEME (same code the gate daemon uses)
so a malformed pointer cannot silently reopen the verdict.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import psycopg

HERE = Path(__file__).resolve().parent
sys.path.insert(0, "/root/tp-agent")
import maturity_gate as gate  # noqa: E402

REPO = "TensorBEM"
SCOPE = "m17296 CASE-BEM-CYL-DRIFT-ASYMP"
LEVEL = 4
MISSION = 17339
BY = "board-audit"

SQL = """
INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by)
VALUES (%s, %s, %s, %s, %s::jsonb, %s)
ON CONFLICT (repo, scope, verdict_by)
DO UPDATE SET level = EXCLUDED.level, evidence = EXCLUDED.evidence,
              mission_id = EXCLUDED.mission_id, created_at = now()
RETURNING id, gate_state, gate_level
"""

if __name__ == "__main__":
    evidence = json.loads((HERE / "evidence.json").read_text())

    dry_level, reasons = gate.judge(LEVEL, evidence, REPO, MISSION)
    print(f"dry-run maturity_gate.judge -> gate_level={dry_level} reasons={reasons}")
    if dry_level != LEVEL:
        raise SystemExit(f"REFUSING to write: evidence only supports L{dry_level}")

    with psycopg.connect("dbname=tensorplatform user=postgres") as conn:
        with conn.cursor() as cur:
            cur.execute(SQL, (REPO, SCOPE, MISSION, LEVEL, json.dumps(evidence, ensure_ascii=False), BY))
            print("verdict row:", cur.fetchone())
        conn.commit()