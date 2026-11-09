"""For every GXMAP sample: fetch its X-Ray .jkf, check the index tiles the .xpoints, compare
the first records' headers with a partial inflate. Writes scratch/allsamples.json."""
import json, os, re, struct, sys, zlib
sys.path.insert(0, 'tools'); sys.path.insert(0, 'src')
import numpy as np
import rangezip
from mla_xpoints import JkfFile, scan_records, HEADER_DTYPE
from mla_xpoints.records import headers_from_buffer

ents = {e['name']: e for e in json.load(open('data/index.json'))}
samples = sorted({m.group(1) for n in ents for m in [re.match(r'Samples/([^/]+)/Measurements/GXMAP/Processing Stages/X-Ray/\1_GXMAP_spx\.xpoints$', n)] if m})
out = {}
if os.path.exists('scratch/allsamples.json'):
    out = json.load(open('scratch/allsamples.json'))
for s in samples:
    if s in out:
        continue
    base = f'Samples/{s}/Measurements/GXMAP/Processing Stages/X-Ray/{s}_GXMAP_spx'
    xe = ents[base + '.xpoints']; je = ents[base + '.jkf']
    tmp = f'data/tmp/{s}.jkf'
    rangezip.extract(je, tmp)
    with JkfFile(tmp) as jk:
        idx = jk.point_index()
        nframes = len(jk.frames)
    os.remove(tmp)
    off, rb = idx['offset'].astype(np.int64), idx['record_bytes'].astype(np.int64)
    contiguous = bool(off[0] == 0 and np.array_equal(off[1:], off[:-1] + rb[:-1]))
    tiles = int(off[-1] + rb[-1]) == xe['usize']
    # partial inflate of the first ~8 MB compressed
    hdr = rangezip.fetch(xe['offset'], xe['offset'] + 29)
    nlen, xlen = struct.unpack_from('<HH', hdr, 26)
    start = xe['offset'] + 30 + nlen + xlen
    raw = zlib.decompressobj(-15).decompress(rangezip.fetch(start, start + (8 << 20) - 1))
    # trim to whole records using the index
    k = int(np.searchsorted(off + rb, len(raw), side='right'))
    part = raw[: int(off[k - 1] + rb[k - 1])] if k else b''
    t = scan_records(part)
    h = headers_from_buffer(part, t)
    same = all(np.array_equal(idx[f][:k], h[f]) for f in ('x', 'y', 'width', 'height', 'bse_mode', 'bse_mode_fraction'))
    same = same and np.array_equal(idx['offset'][:k], t.offsets)
    out[s] = dict(frames=nframes, points=int(len(idx)), n_uint16=int((rb == 4080).sum()),
                  other_sizes=sorted(set(rb.tolist()) - {2080, 4080}), contiguous=contiguous,
                  tiles_file=tiles, headers_checked=k, headers_match=bool(same),
                  zero_size_spots=int((idx['width'] == 0).sum()))
    print(s, out[s], flush=True)
    json.dump(out, open('scratch/allsamples.json', 'w'), indent=1)
print('DONE', len(out))
