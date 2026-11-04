import json, os, urllib.request

base = os.environ["TENSOR_API_BASE"]
user = os.environ["TENSOR_AGENT_USER"]
pw = os.environ["TENSOR_AGENT_PASS"]

req = urllib.request.Request(
    base + "/auth/login",
    data=json.dumps({"username": user, "password": pw}).encode(),
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urllib.request.urlopen(req) as r:
    data = json.load(r)
tok = data.get("access_token") or data.get("token") or ""
open("/tmp/m3371_token", "w").write(tok)
print("LOGIN_OK len=", len(tok))
