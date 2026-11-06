#!/usr/bin/env python3
"""Submit report for m4662 mission - simplified version"""
import json
import os
import sys
import requests

# Simplified report
report_data = {
    "title": "m4623 Pipeline Smoke Test Marker - PR#306 Re-open",
    "summary": "Re-implemented m4623 deliverables: added docs/smoke/smoke-m4623.md (3 lines). Branch agent/m4662 ready for PR#306 re-open.",
    "scenario_key": "tensorbem_m4623_smoke",
    "mission_id": 4662,
    "tags": ["TensorBEM", "smoke-test", "documentation"],
    "markdown": """# m4623 Pipeline Smoke Test Marker

## 案例介绍
Pipeline smoke test marker for m4623 mission. Re-open of PR#306 (closed 2026-11-05).

## 方法
1. Examined agent/m4623 branch
2. Created docs/smoke/smoke-m4623.md from main (a261e234)
3. Validated with pytest (30 tests pass)
4. Committed to agent/m4662

## 结果
- File: docs/smoke/smoke-m4623.md (43 bytes, 3 lines)
- Content: PIPELINE-SMOKE-A m4623 / a261e234 / 2026-11-06
- Tests: test_doc_inventory.py (19 pass), test_capability_evidence.py (11 pass)
- Commit: 55c8c92

## 结果图
N/A - Documentation-only task, no computational results.

## 注意点
- Full test suite times out (>120s); ran targeted tests only
- Environment-specific test failure in test_doc_claims.py (unrelated)
- Smoke marker is minimal validation artifact

## 误差判定
N/A - Documentation task, no numerical results to compare.

## 交付状态
✅ File created at docs/smoke/smoke-m4623.md
✅ Content matches m4623 spec
✅ Tests pass (30/30)
✅ Committed to agent/m4662
⏳ Awaiting PR submission and merge
"""
}

api_base = os.environ.get('TENSOR_API_BASE', 'http://127.0.0.1:8010/api/v1')
username = os.environ.get('TENSOR_AGENT_USER', 'eco_admin')
password = os.environ.get('TENSOR_AGENT_PASS', 'EcoTest!2026pass')

try:
    # Get token
    login_resp = requests.post(
        f'{api_base}/auth/login',
        json={'username': username, 'password': password},
        timeout=10
    )
    login_resp.raise_for_status()
    token = login_resp.json()['access_token']
    print(f"Login successful, token: {token[:20]}...")
    
    # Submit report
    report_resp = requests.post(
        f'{api_base}/tensor/reports',
        headers={
            'Authorization': f'Bearer {token}',
            'Content-Type': 'application/json'
        },
        json=report_data,
        timeout=30
    )
    
    print(f"Response status: {report_resp.status_code}")
    print(f"Response body: {report_resp.text[:500]}")
    
    report_resp.raise_for_status()
    print("Report submitted successfully!")
    print(json.dumps(report_resp.json(), indent=2))
    
except requests.exceptions.HTTPError as e:
    print(f"HTTP Error: {e}", file=sys.stderr)
    print(f"Response: {e.response.text if e.response else 'N/A'}", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f"Error: {e}", file=sys.stderr)
    sys.exit(1)
