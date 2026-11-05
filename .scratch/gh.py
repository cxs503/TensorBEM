import json
import sys
import urllib.request
url = "https://api.github.com" + sys.argv[1]
req = urllib.request.Request(url, headers={"User-Agent": "tp-agent", "Accept": "application/vnd.github+json"})
try:
    with urllib.request.urlopen(req, timeout=30) as r:
        print(json.dumps(json.load(r), ensure_ascii=False)[: int(sys.argv[2]) if len(sys.argv) > 2 else 3000])
except Exception as e:
    print("ERR", type(e).__name__, e)
