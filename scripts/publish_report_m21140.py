"""Publish the m21140 report to the platform (token from the agent environment)."""

from __future__ import annotations

import json
import os
import pathlib
import urllib.request

BASE = os.environ["TENSOR_API_BASE"].rstrip("/")
PAYLOAD = pathlib.Path("/tmp/report_m21140.json")


def post(path: str, body: dict, token: str | None = None) -> dict:
    request = urllib.request.Request(
        f"{BASE}{path}", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 **({"Authorization": f"Bearer {token}"} if token else {})},
        method="POST")
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.loads(response.read().decode())


def main() -> int:
    login = post("/auth/login", {"username": os.environ["TENSOR_AGENT_USER"],
                                 "password": os.environ["TENSOR_AGENT_PASS"]})
    token = login.get("access_token") or login.get("token")
    if not token:
        raise SystemExit(f"no token in login response: {sorted(login)}")
    payload = json.loads(PAYLOAD.read_text())
    created = post("/tensor/reports", payload, token)
    print(json.dumps({k: created.get(k) for k in ("id", "title", "status", "created_at")},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())