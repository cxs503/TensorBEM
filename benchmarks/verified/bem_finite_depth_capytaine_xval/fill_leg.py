"""Fill one grid row of the auto-mode cross-validation (worker for local runs).

    python3 fill_leg.py 5.0:0.3 5.0:0.6 ...   # h/R:ka pairs, subdivision 2
    python3 fill_leg.py --sub 3 5.0:1.0
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from auto_mode_xval import CASES, case_id, solve  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("points", nargs="*", help="depth_ratio:ka")
    ap.add_argument("--sub", type=int, default=2)
    args = ap.parse_args()
    CASES.mkdir(parents=True, exist_ok=True)
    for spec in args.points:
        h_s, ka_s = spec.split(":")
        h, ka = float(h_s), float(ka_s)
        cid = case_id(h, ka, args.sub)
        path = CASES / f"{cid}.json"
        if path.exists():
            print(cid, "cached", flush=True)
            continue
        try:
            payload = solve(h, ka, args.sub)
            payload["status"] = "ok"
        except Exception:  # noqa: BLE001
            payload = {"status": "error", "error": traceback.format_exc()}
        payload["case_id"] = cid
        path.write_text(json.dumps(payload, indent=1, sort_keys=True))
        print(cid, payload["status"], payload.get("seconds"), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())