import json, os, pathlib, urllib.request

base = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8001/api/v1").rstrip("/")
user = os.environ.get("TENSOR_AGENT_USER", "agent")
pw = os.environ.get("TENSOR_AGENT_PASS", "agent")

def call(path, payload=None, token=None, method=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(base + path, data=data, method=method or ("POST" if data else "GET"))
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", "Bearer " + token)
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode())

token = os.environ.get("TOKEN")
if not token:
    login = call("/auth/login", {"username": user, "password": pw})
    token = login.get("access_token") or login.get("token") or login.get("data", {}).get("access_token")
report = json.loads(pathlib.Path("/tmp_report_m21275.json").read_text())
out = call("/tensor/reports", report, token=token)
print(json.dumps(out.get("data", out))[:400])
