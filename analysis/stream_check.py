"""Stream a remote .xpoints member, parse every record, compare headers with a saved .jkf index.

Usage: python -I stream_check.py <index.json> <member> <index.npy> <out.npz>
Stores only per-record summaries (counts, payload type, header bytes of uint16 records).
"""
import json, sys, os, zlib, struct, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'tools'))
import rangezip

ents = {e['name']: e for e in json.load(open(sys.argv[1]))}
e = ents[sys.argv[2]]
IDX = np.load(sys.argv[3]); N = len(IDX)
g = lambda A, o, t: A[:, o:o+np.dtype(t).itemsize].copy().view(t).ravel()
ix, iy, iw, ih = g(IDX,20,'<u4'), g(IDX,24,'<u4'), g(IDX,40,'<u4'), g(IDX,44,'<u4')
ib, i2, ioff, ilen = g(IDX,56,'<f4'), g(IDX,60,'<f4'), g(IDX,8,'<u8'), g(IDX,16,'<u4')

hdr = rangezip.fetch(e['offset'], e['offset']+29)
nlen, xlen = struct.unpack_from('<HH', hdr, 26)
start = e['offset'] + 30 + nlen + xlen; end = start + e['csize']
dec = zlib.decompressobj(-15)
buf = bytearray(); pos = start; k = 0; fpos = 0
counts = np.zeros(N, np.int64); kind = np.zeros(N, np.uint8); maxc = np.zeros(N, np.int32)
bytehist = np.zeros((2, 80, 256), np.int64); H16 = []; mism = 0; first_mism = []; sumspec = np.zeros(2000, np.int64)
t0 = time.time()
while pos < end or len(buf) >= 80:
    if pos < end and len(buf) < 4 << 20:
        chunk = rangezip.fetch(pos, min(pos + rangezip.CHUNK, end) - 1); pos += len(chunk)
        buf += dec.decompress(chunk)
        if pos >= end: buf += dec.flush()
        continue
    p = 0
    while p + 80 <= len(buf):
        plen = struct.unpack_from('<I', buf, p+76)[0]
        if p + 80 + plen > len(buf): break
        h = bytes(buf[p:p+80])
        x, y = struct.unpack_from('<HH', h, 12); w, hh = struct.unpack_from('<HH', h, 40)
        fb, f2 = struct.unpack_from('<ff', h, 48)
        ok = (fpos == ioff[k] and 80+plen == ilen[k] and x == ix[k] and y == iy[k] and w == iw[k]
              and hh == ih[k] and fb == ib[k] and f2 == i2[k] and h[0] == 0x5A and h[70:72] == b'\xd0\x07')
        if not ok:
            mism += 1
            if len(first_mism) < 20: first_mism.append((k, fpos, h.hex()))
        dt = np.uint8 if plen == 2000 else np.dtype('<u2')
        s = np.frombuffer(bytes(buf[p+80:p+80+plen]), dt)
        counts[k] = int(s.sum()); maxc[k] = int(s.max()); kind[k] = 1 if plen == 2000 else 2
        sumspec += s
        bytehist[0 if plen == 2000 else 1, np.arange(80), np.frombuffer(h, np.uint8)] += 1
        if plen != 2000: H16.append(h)
        fpos += 80 + plen; p += 80 + plen; k += 1
    del buf[:p]
    if k % 100000 < 2000:
        print(f"{k} records, {pos-start}/{end-start} compressed, {time.time()-t0:.0f}s, mism={mism}", flush=True)
print("DONE records", k, "expected", N, "file bytes", fpos, "mismatches", mism, "leftover", len(buf))
np.savez(sys.argv[4], bytehist=bytehist, counts=counts, kind=kind, maxc=maxc, sumspec=sumspec,
         h16=np.frombuffer(b''.join(H16), np.uint8).reshape(-1, 80) if H16 else np.zeros((0,80),np.uint8),
         first_mism=np.array([str(m) for m in first_mism]))
