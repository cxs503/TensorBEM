#!/usr/bin/env python3
"""Log in to the Tensor platform and POST the mission-3796 report.

Credentials come from the environment (TENSOR_AGENT_USER / TENSOR_AGENT_PASS) so
nothing secret ever appears on a command line.  The response is written to
/tmp/report_resp.json for an append-first receipt.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path

BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010").rstrip("/")
USER = os.environ.get("TENSOR_AGENT_USER", "")
PASS = os.environ.get("TENSOR_AGENT_PASS", "")


def post(path: str, payload: dict, token: str | None = None, timeout: int = 120) -> tuple[int, str]:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(BASE + path, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read().decode()[:4000]
    except urllib.error.HTTPError as exc:  # pragma: no cover - network path
        return exc.code, exc.read().decode()[:4000]
    except Exception as exc:  # pragma: no cover
        return -1, repr(exc)


def main() -> int:
    receipt: dict = {"api": BASE}
    status, body = post("/auth/login", {"username": USER, "password": PASS})
    receipt["login_status"] = status
    token = ""
    if status == 200:
        try:
            doc = json.loads(body)
            token = doc.get("access_token") or doc.get("token") or doc.get("data", {}).get("access_token", "")
        except json.JSONDecodeError:
            receipt["login_body"] = body[:200]
    if not token:
        receipt["error"] = "no token"
        Path("/tmp/report_resp.json").write_text(json.dumps(receipt, indent=2))
        print("login failed", status, body[:200])
        return 1
    payload = json.loads(Path("/tmp/report_3796.json").read_text())
    status, body = post("/tensor/reports", payload, token=token)
    receipt["report_status"] = status
    receipt["report_body"] = body[:2000]
    Path("/tmp/report_resp.json").write_text(json.dumps(receipt, indent=2))
    print("report post status:", status)
    print(body[:400])
    return 0 if status in (200, 201) else 1


if __name__ == "__main__":
    raise SystemExit(main())