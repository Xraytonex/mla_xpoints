"""Fetch every archive member whose name matches a regex and whose size is under a cap.

Usage: python -I getmany.py <index.json> <out-root> <regex> [max_usize]
"""
import json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rangezip

idx, root, pat = sys.argv[1], sys.argv[2], re.compile(sys.argv[3])
cap = int(sys.argv[4]) if len(sys.argv) > 4 else 1 << 62
for e in json.load(open(idx)):
    if e["name"].endswith("/") or not pat.search(e["name"]) or e["usize"] > cap:
        continue
    out = os.path.join(root, e["name"])
    if os.path.exists(out) and os.path.getsize(out) == e["usize"]:
        continue
    os.makedirs(os.path.dirname(out), exist_ok=True)
    n = rangezip.extract(e, out)
    print(f"{n:>11} {e['name']}")
