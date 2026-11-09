"""Command line: ``python -m mla_xpoints info|export|check``."""

from __future__ import annotations

import argparse
import sys

import numpy as np

from .jkf import JkfFile
from .records import XPointsFile


def _info(args: argparse.Namespace) -> int:
    with XPointsFile(args.xpoints) as xp:
        h = xp.headers
        n16 = int(xp.table.is_uint16.sum())
        print(f"file            {xp.path}")
        print(f"bytes           {xp.size}")
        print(f"records         {len(xp)}  (uint8 {len(xp) - n16}, uint16 {n16})")
        print(f"channels        {xp.n_channels}")
        if len(xp):
            print(f"x range         {h['x'].min()}..{h['x'].max()}   y range {h['y'].min()}..{h['y'].max()}")
            sizes, counts = np.unique(h["width"], return_counts=True)
            print("spot sizes      " + ", ".join(f"{s}px x{c}" for s, c in zip(sizes, counts)))
            if args.counts:
                tot = xp.total_counts()
                print(f"counts/spectrum min {tot.min()}  median {int(np.median(tot))}  max {tot.max()}")
    return 0


MAX_EXPORT = 200_000  # ~0.8 GB of uint16 spectra in memory


def _export(args: argparse.Namespace) -> int:
    with XPointsFile(args.xpoints) as xp:
        idx = np.arange(len(xp))[args.start : args.stop]
        if len(idx) > MAX_EXPORT and not args.force:
            print(
                f"error: {len(idx)} records would need ~{len(idx) * xp.n_channels * 2 / 1e9:.1f} GB; "
                f"export at most {MAX_EXPORT} at a time with --start/--stop, or pass --force",
                file=sys.stderr,
            )
            return 2
        out = args.out if str(args.out).endswith(".npz") else f"{args.out}.npz"
        np.savez_compressed(out, headers=xp.headers[idx], spectra=xp.spectra(idx), record=idx)
    print(f"wrote {len(idx)} spectra to {out}")
    return 0


def _check(args: argparse.Namespace) -> int:
    """Compare every indexed .xpoints header with the X-Ray .jkf index (matched by offset)."""
    with XPointsFile(args.xpoints) as xp, JkfFile(args.jkf) as jk:
        idx = jk.point_index()
        h, t = xp.headers, xp.table
        want = idx["offset"].astype(np.int64)
        if len(xp):
            pos = np.minimum(np.searchsorted(t.offsets, want), len(xp) - 1)
            found = t.offsets[pos] == want
        else:
            pos, found = np.zeros(len(idx), np.int64), np.zeros(len(idx), bool)
        problems = []
        if not found.all():
            problems.append(f"{int((~found).sum())} index entries point between records or past the end")
        pos, sel = pos[found], idx[found]
        checks = {
            "record_bytes": sel["record_bytes"] == t.payload_bytes[pos] + 80,
            "x": sel["x"] == h["x"][pos],
            "y": sel["y"] == h["y"][pos],
            "width": sel["width"] == h["width"][pos],
            "height": sel["height"] == h["height"][pos],
            "bse_mode": sel["bse_mode"] == h["bse_mode"][pos],
            "bse_mode_fraction": sel["bse_mode_fraction"] == h["bse_mode_fraction"][pos],
        }
        problems += [f"{k}: {int((~v).sum())} mismatches" for k, v in checks.items() if not v.all()]
        unindexed = len(xp) - len(np.unique(pos))
        print(f"{len(xp)} records, {len(jk.frames)} frames, {len(idx)} index entries")
        if unindexed:
            print(f"{unindexed} records are not referenced by the .jkf (e.g. frames removed in processing)")
        print("\n".join(problems) if problems else "all indexed headers match the .jkf index")
        return 1 if problems else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="mla_xpoints", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("info", help="summarise an .xpoints file")
    p.add_argument("xpoints")
    p.add_argument("--counts", action="store_true", help="also sum every spectrum")
    p.set_defaults(func=_info)
    p = sub.add_parser("export", help="write headers and spectra to .npz")
    p.add_argument("xpoints")
    p.add_argument("out")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--stop", type=int, default=None)
    p.add_argument("--force", action="store_true", help=f"allow more than {200_000} records")
    p.set_defaults(func=_export)
    p = sub.add_parser("check", help="verify an .xpoints file against its X-Ray .jkf")
    p.add_argument("xpoints")
    p.add_argument("jkf")
    p.set_defaults(func=_check)
    args = ap.parse_args(argv)
    try:
        return args.func(args)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
