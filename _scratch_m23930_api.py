import json, os, sys, urllib.request

base = os.environ["TENSOR_API_BASE"]
user = os.environ["TENSOR_AGENT_USER"]
pw = os.environ["TENSOR_AGENT_PASS"]

def req(method, path, body=None, token=None):
    url = base + path
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(url, data=data, method=method)
    r.add_header("Content-Type", "application/json")
    if token:
        r.add_header("Authorization", "Bearer " + token)
    try:
        with urllib.request.urlopen(r, timeout=30) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        return {"error": str(e)}

tok = req("POST", "/auth/login", {"username": user, "password": pw})
token = tok.get("access_token") or tok.get("token") or ""
print("login:", "ok" if token else tok)
if not token:
    sys.exit(1)

for p in sys.argv[1:]:
    m = req("GET", p, token=token)
    print("==", p)
    print(json.dumps(m, ensure_ascii=False, indent=1)[:6000])
