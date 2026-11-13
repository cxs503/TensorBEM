import json, os, sys, urllib.request, urllib.error

BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")

def login():
    body = json.dumps({"username": os.environ["TENSOR_AGENT_USER"],
                       "password": os.environ["TENSOR_AGENT_PASS"]}).encode()
    req = urllib.request.Request(BASE + "/auth/login", data=body,
                                 headers={"Content-Type": "application/json"})
    d = json.load(urllib.request.urlopen(req, timeout=20))
    return d.get("access_token") or d.get("token")

def call(path, method="GET", body=None, token=None):
    data = json.dumps(body).encode() if body is not None else None
    h = {"Content-Type": "application/json"}
    if token:
        h["Authorization"] = "Bearer " + token
    req = urllib.request.Request(BASE + path, data=data, headers=h, method=method)
    try:
        r = urllib.request.urlopen(req, timeout=120)
        return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()[:3000]

if __name__ == "__main__":
    tok = login()
    print("token_ok", bool(tok), file=sys.stderr)
    path = sys.argv[1] if len(sys.argv) > 1 else "/health"
    print(json.dumps(call(path, token=tok), ensure_ascii=False)[:6000])