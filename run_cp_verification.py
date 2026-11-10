#!/usr/bin/env python
"""Run the full cathodic protection verification study on the cluster."""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from tensorbem.benchmarks.cp_verification_m9204 import run_full_verification

if __name__ == "__main__":
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "artifacts/cp_verification"
    results = run_full_verification(output_dir)
    
    # Print summary
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    criteria = results.get("acceptance_criteria", {})
    for name, c in criteria.items():
        status = "PASS" if c.get("pass") else "FAIL"
        val = c.get("value", c.get("gci_percent", c.get("max_imbalance", "N/A")))
        print(f"  {name}: {status} (value={val})")
    print(f"\nOverall: {'ALL PASS' if results.get('overall_pass') else 'SOME FAIL'}")
    
    # Save summary
    summary_path = Path(output_dir) / "summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "overall_pass": results.get("overall_pass"),
            "criteria": {k: {"pass": v.get("pass"), "value": v.get("value", v.get("gci_percent", v.get("max_imbalance")))} for k, v in criteria.items()},
        }, f, indent=2)
    print(f"\nSummary saved to {summary_path}")
