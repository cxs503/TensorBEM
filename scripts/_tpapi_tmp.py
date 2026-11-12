import os, sys, json, urllib.request

BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1").rstrip("/")
USER = os.environ["TENSOR_AGENT_USER"]
PWD = os.environ["TENSOR_AGENT_PASS"]


def req(path, method="GET", body=None, token=None):
    url = BASE + path
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(url, data=data, method=method)
    r.add_header("Content-Type", "application/json")
    if token:
        r.add_header("Authorization", "Bearer " + token)
    with urllib.request.urlopen(r, timeout=60) as f:
        return json.loads(f.read().decode())


def login():
    d = req("/auth/login", "POST", {"username": USER, "password": PWD})
    return d.get("access_token") or d.get("token") or d.get("data", {}).get("access_token")


def main():
    tok = login()
    cmd = sys.argv[1]
    if cmd == "reports":
        d = req("/tensor/reports?limit=20", token=tok)
        items = d.get("items") or d.get("reports") or d.get("data") or []
        for r in items[:20]:
            print(r.get("id"), "|", r.get("scenario_key"), "|", (r.get("title") or "")[:100])
    elif cmd == "mission":
        print(json.dumps(req("/tensor/missions/" + sys.argv[2], token=tok), ensure_ascii=False)[:3000])
    print("TOKEN_OK", bool(tok))


if __name__ == "__main__":
    main()