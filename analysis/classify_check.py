"""Nearest-reference classification of the test frame vs MLA's classification_report.csv.

Usage: python analysis/classify_check.py <xpoints> <jkx> <classification_report.csv>
"""
import csv
import sys
from collections import Counter

import numpy as np

from mla_xpoints import XPointsFile, read_jkx

LO, HI = 38, 1100  # MLA "Channel Limit" 38; above ~11 keV there is little signal


def main(xp_path: str, jkx_path: str, report_path: str) -> None:
    refs = [r for r in read_jkx(jkx_path) if r.spectrum.size]
    R = np.array([r.spectrum[LO:HI] for r in refs], float)
    R /= R.sum(1, keepdims=True)
    with XPointsFile(xp_path) as xp:
        S = xp.spectra()[:, LO:HI].astype(float)
    k = S.sum(1)
    best = np.array([(((s - kk * R) ** 2) / (kk * R + 1)).sum(1).argmin() for s, kk in zip(S, k)])
    ours = Counter(refs[b].mineral for b in best)
    rows = [r for r in csv.reader(open(report_path))][2:]
    mla = Counter()
    for r in rows:
        if len(r) > 3:
            mla[r[0]] += int(r[-1])
    print(f"{'mineral':28s} {'MLA':>5s} {'nearest':>7s}")
    for m in sorted(set(mla) | set(ours), key=lambda m: -mla[m]):
        print(f"{m:28s} {mla[m]:5d} {ours[m]:7d}")
    print("overlap", sum(min(mla[m], ours[m]) for m in mla), "of", len(S))


if __name__ == "__main__":
    main(*sys.argv[1:4])
