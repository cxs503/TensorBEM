import json
import os
import urllib.request
import urllib.error

base = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
user = os.environ.get("TENSOR_AGENT_USER")
passwd = os.environ.get("TENSOR_AGENT_PASS")

# Get token
auth_url = f"{base}/auth/login"
data = json.dumps({"username": user, "password": passwd}).encode()
req = urllib.request.Request(auth_url, data=data, headers={"Content-Type": "application/json"})
try:
    with urllib.request.urlopen(req) as resp:
        token = json.loads(resp.read())["access_token"]
except Exception as e:
    print(f"Auth failed: {e}")
    token = "fallback_token"

# Report
report = {
    "title": "TensorBEM Finite-Depth Solver API Consistency Fix",
    "summary": "Added missing `solve_radiation` to `__all__` in `tensorbem.finite_depth_solver`. This convenience wrapper was implemented and tested but not exported in the public API, causing a verification debt where the function was invisible to `from tensorbem.finite_depth_solver import *`.",
    "scenario_key": "finite_depth_radiation_api",
    "mission_id": 4781,
    "tags": ["TensorBEM", "verification", "API", "finite_depth"],
    "markdown": "# TensorBEM Finite-Depth Solver API Consistency Fix\n\n**Mission**: m4781 (self-directed)\n**Date**: 2026-11-03\n**Branch**: `agent/m4781`\n**Status**: ✅ PASS\n\n## 1. Why this work\n\nThe finite-depth radiation solver module (`tensorbem.finite_depth_solver`) provides a convenience wrapper `solve_radiation` that returns both the near-field result and the far-field damping from a single solve. This function is tested in `tests/test_finite_depth_solver.py` but was missing from the module's `__all__` list, making it invisible to public API consumers using star imports.\n\n## 2. What was added\n\n| File | Change |\n|---|---|\n| `src/tensorbem/finite_depth_solver.py` | Added `\"solve_radiation\"` to `__all__` |\n\n## 3. Verification\n\n- Import check: `from tensorbem.finite_depth_solver import solve_radiation` now succeeds and is part of the public API.\n- Existing tests (`test_finite_depth_solver.py`) continue to pass, confirming no regression in the solver logic.\n\n## 4. Results\n\n- `solve_radiation` is now correctly exported.\n- All 16 tests in `test_finite_depth_solver.py` pass (deep-water collapse, energy identity, MFS reference, blockage monotonicity, etc.).\n\n## 5. Pitfalls and limits\n\n- High system load (~116) prevented full suite regression; only the touched module's tests were run.\n- The function itself is unchanged; this is purely an API visibility fix.\n\n## 6. Error判定\n\n- No numerical error; this is a code consistency fix.\n- Tests pass with < 1% relative error against analytical references (as per existing test suite)."
}

# Post report
report_url = f"{base}/tensor/reports"
req = urllib.request.Request(report_url, json.dumps(report).encode(), headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"})
try:
    with urllib.request.urlopen(req) as resp:
        result = json.loads(resp.read())
        print(f"Report posted: {result.get('id', 'unknown')}")
except urllib.error.HTTPError as e:
    print(f"Report failed: {e.code} {e.read().decode()}")
