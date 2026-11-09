"""Read members of a remote ZIP (incl. ZIP64) with HTTP Range requests.

Usage:
  python -I rangezip.py list  <out.json>
  python -I rangezip.py get   <member-name> <out-path> [max_uncompressed_bytes]
"""
import json
import struct
import sys
import urllib.request
import zlib

URL = "https://rodare.hzdr.de/record/4896/files/Datenpublikation_RODARE.zip"
TOTAL = 15821751662
CHUNK = 8 << 20


def fetch(start, end_incl):
    req = urllib.request.Request(URL, headers={"Range": f"bytes={start}-{end_incl}",
                                               "User-Agent": "mla-xpoints-research"})
    with urllib.request.urlopen(req, timeout=120) as r:
        data = r.read()
    if len(data) != end_incl - start + 1:
        raise IOError(f"short read {len(data)} for {start}-{end_incl}")
    return data


def central_directory():
    tail = fetch(TOTAL - 65536, TOTAL - 1)
    eocd = tail.rfind(b"PK\x05\x06")
    if eocd < 0:
        raise ValueError("no EOCD")
    loc = tail.rfind(b"PK\x06\x07", 0, eocd)
    if loc < 0:
        raise ValueError("no ZIP64 locator")
    (z64_off,) = struct.unpack_from("<Q", tail, loc + 8)
    z64 = fetch(z64_off, z64_off + 55)
    if z64[:4] != b"PK\x06\x06":
        raise ValueError("bad ZIP64 EOCD")
    n_entries, cd_size, cd_off = struct.unpack_from("<QQQ", z64, 32)
    cd = fetch(cd_off, cd_off + cd_size - 1)
    entries, p = [], 0
    for _ in range(n_entries):
        if cd[p:p + 4] != b"PK\x01\x02":
            raise ValueError(f"bad CD header at {p}")
        (method,) = struct.unpack_from("<H", cd, p + 10)
        csize, usize = struct.unpack_from("<II", cd, p + 20)
        nlen, xlen, clen = struct.unpack_from("<HHH", cd, p + 28)
        (lho,) = struct.unpack_from("<I", cd, p + 42)
        name = cd[p + 46:p + 46 + nlen].decode("utf-8", "replace")
        extra = cd[p + 46 + nlen:p + 46 + nlen + xlen]
        q = 0
        while q + 4 <= len(extra):
            hid, hlen = struct.unpack_from("<HH", extra, q)
            if hid == 1:
                vals, r = [], q + 4
                for flag in (usize, csize, lho):
                    if flag == 0xFFFFFFFF:
                        vals.append(struct.unpack_from("<Q", extra, r)[0]); r += 8
                    else:
                        vals.append(flag)
                usize, csize, lho = vals
            q += 4 + hlen
        entries.append({"name": name, "method": method, "csize": csize,
                        "usize": usize, "offset": lho})
        p += 46 + nlen + xlen + clen
    return entries


def extract(entry, out_path, max_bytes=None):
    hdr = fetch(entry["offset"], entry["offset"] + 29)
    nlen, xlen = struct.unpack_from("<HH", hdr, 26)
    start = entry["offset"] + 30 + nlen + xlen
    end = start + entry["csize"]
    dec = zlib.decompressobj(-15) if entry["method"] == 8 else None
    written = 0
    with open(out_path, "wb") as f:
        pos = start
        while pos < end:
            chunk = fetch(pos, min(pos + CHUNK, end) - 1)
            pos += len(chunk)
            out = dec.decompress(chunk) if dec else chunk
            if max_bytes is not None:
                out = out[:max_bytes - written]
            f.write(out)
            written += len(out)
            if max_bytes is not None and written >= max_bytes:
                break
        if dec and (max_bytes is None or written < max_bytes):
            tail = dec.flush()
            f.write(tail); written += len(tail)
    return written


def main(argv):
    if argv[1] == "list":
        ents = central_directory()
        with open(argv[2], "w") as f:
            json.dump(ents, f, indent=0)
        print(len(ents), "entries")
    elif argv[1] == "get":
        with open(argv[2]) as f:
            ents = json.load(f)
        matches = [e for e in ents if e["name"] == argv[3]]
        if not matches:
            raise SystemExit("no such member")
        mx = int(argv[5]) if len(argv) > 5 else None
        n = extract(matches[0], argv[4], mx)
        print(n, "bytes ->", argv[4])


if __name__ == "__main__":
    main(sys.argv)
