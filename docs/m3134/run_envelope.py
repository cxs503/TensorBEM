"""Incremental driver for the (Ka, f/a) envelope study (m3134).

The shell harness kills the whole process group after 120 s, so the envelope is
computed in resumable chunks: every finished working point is appended to a
JSONL file and skipped on the next invocation.

    PYTHONPATH=src python3 docs/m3134/run_envelope.py --subdivision 2 \
        --ratios 1.1 --kas 0.2 0.8 1.2 2.0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tensorbem.benchmarks.submerged_sphere_envelope import evaluate_point

OUT = Path(__file__).resolve().parent / "outputs" / "points.jsonl"


def _key(row: dict) -> tuple[float, float, int]:
    return (round(row["ka"], 6), round(row["f_over_a"], 6), int(row["panels"]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subdivision", type=int, default=2)
    ap.add_argument("--kas", type=float, nargs="+", required=True)
    ap.add_argument("--ratios", type=float, nargs="+", required=True)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--budget", action="store_true", help="also record coarser levels")
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    done: set[tuple[float, float, int]] = set()
    if args.out.exists():
        for line in args.out.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(_key(json.loads(line)))

    subdivisions = [0, 1, args.subdivision] if args.budget else [args.subdivision]
    written = 0
    with args.out.open("a", encoding="utf-8") as handle:
        for ratio in args.ratios:
            for ka in args.kas:
                for sub in subdivisions:
                    probe = {"ka": ka, "f_over_a": ratio, "panels": 0}
                    row = evaluate_point(ka, ratio, subdivision=sub)
                    probe["panels"] = row["panels"]
                    if _key(probe) in done:
                        continue
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
                    handle.flush()
                    written += 1
                    print(
                        f"f/a={ratio:5.2f} ka={ka:4.2f} sub={sub} "
                        f"panels={row['panels']:4d} "
                        f"err_a33_scaled={row['err_a33_scaled']:.4e} "
                        f"err_b33_scaled={row['err_b33_scaled']:.4e}",
                        flush=True,
                    )
    print(f"wrote {written} new rows -> {args.out}")


if __name__ == "__main__":
    main()