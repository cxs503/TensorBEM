"""Fill the cheap sub-1 rows of the auto-mode grid locally (light budget).

Used when the cluster queue is congested: subdivision 1 (80 panels) solves in
seconds, so the coarse leg of the mesh trend does not need a batch job.

    python3 fill_sub1.py 5.0      # h/R station
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from auto_mode_xval import CASES, KA, case_id, solve  # noqa: E402


def main() -> int:
    depth_ratio = float(sys.argv[1])
    CASES.mkdir(parents=True, exist_ok=True)
    for ka in KA:
        cid = case_id(depth_ratio, ka, 1)
        path = CASES / f"{cid}.json"
        if path.exists():
            print(cid, "cached")
            continue
        payload = solve(depth_ratio, ka, 1)
        payload["status"] = "ok"
        payload["case_id"] = cid
        path.write_text(json.dumps(payload, indent=1, sort_keys=True))
        print(cid, payload["status"], payload["seconds"], flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())