import os, json, urllib.request

base = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
tok = open("/tmp/tok.txt").read().strip()

def get(path):
    r = urllib.request.Request(base + path)
    r.add_header("Authorization", "Bearer " + tok)
    try:
        with urllib.request.urlopen(r, timeout=60) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode() or "{}")
    except Exception as e:
        return -1, {"err": str(e)}

s, d = get("/tensor/reports/16344")
print("REPORT", s)
if s == 200:
    json.dump(d, open("/tmp/report_16344.json", "w"), ensure_ascii=False, indent=1)
    md = d.get("markdown", "")
    print("md len:", len(md))
    print(md[:3000])
