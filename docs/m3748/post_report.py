#!/usr/bin/env python3
"""Publish the mission-3748 progress report to the platform.

Credentials come from the environment (TENSOR_AGENT_USER / TENSOR_AGENT_PASS /
TENSOR_API_BASE) so nothing secret is written to disk or into a command line.

usage: post_report.py /tmp/report.json
"""
from __future__ import annotations

import json
import os
import sys
import urllib.request

BASE = os.environ["TENSOR_API_BASE"].rstrip("/")


def post(path: str, payload: dict, token: str | None = None) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{BASE}{path}", data=data, method="POST",
        headers={"Content-Type": "application/json",
                 **({"Authorization": f"Bearer {token}"} if token else {})},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())


def main(path: str) -> int:
    creds = {"username": os.environ["TENSOR_AGENT_USER"],
             "password": os.environ["TENSOR_AGENT_PASS"]}
    try:
        auth = post("/auth/login", creds)
    except urllib.error.HTTPError as exc:  # noqa: F821
        auth = post("/auth/token", creds)
    token = auth.get("access_token") or auth.get("token")
    assert token, auth
    report = json.load(open(path))
    out = post("/tensor/reports", report, token)
    print(json.dumps({k: out[k] for k in list(out)[:8]}, ensure_ascii=False)[:800])
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/report.json"))