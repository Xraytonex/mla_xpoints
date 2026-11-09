"""Render the per-sample table for FORMAT.md from allsamples.json and the archive index."""
import glob
import json
import re
import sys

idx = {e["name"]: e for e in json.load(open("data/index.json"))}
res = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "scratch/allsamples.json"))
print("| Sample | Frames | Spectra (log) | Spectra (.jkf) | uint16 by size | uint16 in .jkf | Index tiles file | Headers checked | 0-size spots |")
print("|---|---|---|---|---|---|---|---|---|")
for s in sorted(res):
    r = res[s]
    log = open(glob.glob(f"data/zip/Samples/{s}/Measurements/GXMAP/{s}_GXMAP.txt")[0], errors="replace").read()
    n_log = int(re.search(r"(\d+) x-ray collected", log).group(1))
    size = idx[f"Samples/{s}/Measurements/GXMAP/Processing Stages/X-Ray/{s}_GXMAP_spx.xpoints"]["usize"]
    by_size = (size - 2080 * n_log) / 2000
    ok = "yes" if r["contiguous"] and r["tiles_file"] and not r["other_sizes"] else "NO"
    hm = f"{r['headers_checked']:,} {'match' if r['headers_match'] else 'MISMATCH'}"
    print(f"| {s} | {r['frames']} | {n_log:,} | {r['points']:,} | {by_size:,.0f} | {r['n_uint16']:,} | {ok} | {hm} | {r['zero_size_spots']} |")
