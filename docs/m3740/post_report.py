"""Post the m3740 merge report to the platform reports API.

Credentials come from the environment (never printed / never committed).
Tries the documented endpoint shape and the /api/v1-prefixed one.
"""
from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010").rstrip("/")
USER = os.environ.get("TENSOR_AGENT_USER", "")
PASSWORD = os.environ.get("TENSOR_AGENT_PASS", "")


def request(url: str, payload: dict | None = None, token: str | None = None):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, method="POST" if data else "GET")
    if data:
        req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=120) as response:
            return response.status, response.read().decode()
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode()
    except Exception as error:  # noqa: BLE001
        return 0, str(error)


def login() -> str:
    attempts = [
        ("/auth/login", {"username": USER, "password": PASSWORD}),
        ("/api/v1/auth/login", {"username": USER, "password": PASSWORD}),
        ("/tensor/login", {"username": USER, "password": PASSWORD}),
        ("/api/v1/tensor/login", {"username": USER, "password": PASSWORD}),
    ]
    for path, payload in attempts:
        status, body = request(BASE + path, payload)
        if status == 200:
            data = json.loads(body)
            token = (
                data.get("access_token")
                or data.get("token")
                or (data.get("data") or {}).get("access_token")
            )
            if token:
                print("login ok via", path)
                return token
        print("login failed", path, status, body[:120])
    raise SystemExit("login failed")


def main() -> int:
    payload = json.loads(open("/tmp/report.json", encoding="utf-8").read())
    token = login()
    for path in ("/tensor/reports", "/api/v1/tensor/reports"):
        status, body = request(BASE + path, payload, token)
        print(path, status, body[:300])
        if status in (200, 201):
            return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())