# mla-xpoints

Reader for FEI/Thermo MLA raw X-ray spectra (`.xpoints`), the X-Ray/Backscatter stage
`.jkf` files that place them, and `.jkx` mineral lists. Decoded from RODARE 4896
(MLA Suite 3.1.4.686); the format description and the evidence are in [FORMAT.md](FORMAT.md).

## Install

```bash
python -m venv .venv && .venv/bin/pip install -e ".[test]"
```

## Use

```python
from mla_xpoints import XPointsFile, JkfFile, energy_axis_kev, pixel_to_stage

with XPointsFile("3_test_X-Ray.xpoints") as xp:          # memory-mapped
    print(len(xp), xp.n_channels)                         # 2981 2000
    h = xp.headers                                        # x, y, width, height, bse_mode, ...
    spectra = xp.spectra(slice(0, 100))                   # (100, 2000) uint16
    total = xp.sum_spectrum()                             # chunked, fine for 2.5 GB files
energy = energy_axis_kev()                                # fitted: -10 eV + 10.016 eV/ch

with JkfFile("3_test_X-Ray.jkf") as jk:
    idx = jk.point_index(with_grains=True)                # frame, particle, grain per spectrum
    frame = jk.frame(int(idx["frame"][0]))
    sx, sy = pixel_to_stage(frame, idx["x"], idx["y"])    # stage um
```

Command line:

```bash
python -m mla_xpoints info 3_test_X-Ray.xpoints --counts
python -m mla_xpoints check 3_test_X-Ray.xpoints 3_test_X-Ray.jkf
python -m mla_xpoints export 3_test_X-Ray.xpoints out.npz --start 0 --stop 1000
```

A full GXMAP file holds ~1.2 million spectra (2.5 GB on disk, ~5 GB as uint16 in memory),
so use `iter_chunks` rather than `spectra()` on those.

## Getting the data without 15.8 GB

`tools/rangezip.py` reads single members of the RODARE ZIP over HTTP Range requests:

```bash
python -I tools/rangezip.py list data/index.json
python -I tools/getmany.py data/index.json data/zip '^Samples/3/Measurements/test/'
```

## Tests

```bash
.venv/bin/python -m pytest --cov=mla_xpoints
```

Synthetic files cover the reader; tests marked `realdata` run only when the RODARE test
frame is under `data/zip/`. `analysis/` holds the scripts behind the evidence in FORMAT.md.
