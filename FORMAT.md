# MLA `.xpoints` raw X-ray format

Decoded from RODARE 4896 (HZDR, CC BY 4.0): FEI Quanta 650F, two Bruker X-Flash 5030
detectors, FEI MLA Suite 3.1.4.686, GXMAP mode. Everything below holds for this dataset and
software version; other versions may differ.

Status: the `.xpoints` record layout is fully decoded and verified on all 20 full-size files
(see [Evidence](#evidence)). The meaning of five constant header fields is still unknown.
The `.jkf` and `.jkx` layouts are decoded only as far as needed to place and check spectra.

## 1. `.xpoints`: a plain sequence of records

There is no file header and no footer. Records follow each other with no gaps:

    record  = 80-byte header + payload
    payload = n_channels counts
              uint8          when payload_bytes == n_channels      (2000 bytes)
              uint16 (LE)    when payload_bytes == 2 * n_channels  (4000 bytes)

MLA writes a record as uint16 only when one channel would overflow a byte. In sample 3, all
180 uint16 records have a channel above 255 and no uint8 record does. There is no separate
type flag: `payload_bytes` is the only switch.

Record order = acquisition order = the order of the index records in the X-Ray `.jkf`
(frame by frame, particle by particle). The `.jkf` may skip records whose frames were
dropped in processing (sample 5-1a), so always walk the `.xpoints` file itself.

### Header (80 bytes, little-endian)

| Offset | Type | Field | Values seen | Meaning / confidence |
|---|---|---|---|---|
| 0x00 | u8 | tag | 0x5A in every record | record marker (certain) |
| 0x01 | u8 | (padding) | anything | uninitialised memory |
| 0x02 | u16 | version_a | 3 | constant; meaning unknown |
| 0x04 | u16 | version_b | 2 | constant; meaning unknown |
| 0x06 | 2 bytes | (padding) | anything | uninitialised memory |
| 0x08 | u32 | | 0 | always zero |
| 0x0C | u16 | x | 0..499 | pixel column in the 500×500 frame (certain) |
| 0x0E | u16 | y | 0..499 | pixel row in the frame (certain) |
| 0x10 | u16 | | 0 | always zero |
| 0x12 | 6 bytes | (padding) | anything | uninitialised memory |
| 0x18 | 16 bytes | | 0 | always zero |
| 0x28 | u16 | width | 10, 1, rarely 0 | side of the area the spectrum stands for: the GXMAP step size (10 px) or a single spot (1 px) |
| 0x2A | u16 | height | = width | as above |
| 0x2C | u16 | unknown_44 | 1 | constant; meaning unknown |
| 0x2E | 2 bytes | (padding) | anything | uninitialised memory |
| 0x30 | f32 | bse_mode | integer-valued, ~8..255 | most frequent BSE grey level in the spectrum's grain (see §4) |
| 0x34 | f32 | bse_mode_fraction | 0.05..1, 3 decimals | fraction of the grain's pixels at that grey level, truncated to 3 decimals |
| 0x38 | 12 bytes | | 0 | always zero |
| 0x44 | u8 | | 4 | constant (same for uint8 and uint16 records) |
| 0x45 | u8 | (padding) | anything | uninitialised memory |
| 0x46 | u16 | n_channels | 2000 | channel count (certain) |
| 0x48 | u8 | | 4 | constant (same for uint8 and uint16 records) |
| 0x49 | 3 bytes | (padding) | anything | uninitialised memory |
| 0x4C | u32 | payload_bytes | 2000 or 4000 | payload size in bytes (certain) |

"Always zero" and "constant" were checked over all 1,216,798 headers of sample 3. The
padding bytes are zero in some records and hold stale memory in others (the byte
histograms show all 256 values), so a reader must not rely on them. The earlier note that
bytes 0x44/0x48 read `04 08` in uint16 records came from this padding: both are 4 in every
record of both kinds.

Width/height 0 occurs in 0 to a few hundred records per sample (see the per-sample table). In sample 3 all three such spectra are the only spectrum of their grain, and each grain has `bse_mode` 21, just below the GXMAP trigger's `Grey Low = 22`; that may be the rule, unconfirmed.

### Counts

Counts per spectrum in sample 3: 304 to 5,891, median 3,375. MLA's "Minimum Counts = 800"
applies at classification time; low-count spectra are still stored.

## 2. Energy axis

The files hold no calibration. `Maximum Energy = 20 KV` and 2000 channels give a nominal
10 eV/channel from 0. A least-squares fit of nine K lines (Mg Kα, Al Kα, Si Kα, K Kα,
Ca Kα, Ca Kβ, Ti Kα, Fe Kα, Fe Kβ) to Gaussian centroids in the test frame's summed spectrum
gives

    E(eV) = -10.04 + 10.0163 × channel    (channel index from 0; residuals within ±3.3 eV)

O Kα sits 5 eV low (normal for low-energy lines). The zero (noise) peak occupies channels
0–8; channels ~9–23 are empty (detector threshold). Fitted on one frame of one sample only.

## 3. Placing spectra: the X-Ray stage `.jkf`

Each measurement's `Processing Stages/X-Ray/<name>_spx.jkf` indexes the `.xpoints` file.
`JK` magic, version 4, file size as u64 at 0x10, a 256-entry BGRA palette block, then one
block per frame. Every block starts `u32 id, u32 type, u64 length` (length from block start).

| Block | Type | Content |
|---|---|---|
| frame | 0x5DC | +0x10 u32 frame number (twice), +0x18 u32 particle count, +0x28 u32 width (500), +0x2C u32 height (500), +0x48 f32 magnification, +0x4C/+0x50 f32 frame-centre stage X/Y (µm), +0x60 first particle block |
| particle | 0x64A | +0x24 u32 xmin, ymin, xmax, ymax (frame pixels, max exclusive), +0x40 image sub-block |
| image | 0xA28 | +0x10 u32 width, height, stride (width rounded up to 4), +0x1C u64 data length, +0x28 PackBits data → u8 image of the particle's bounding box |
| points | 0xA64 | `u32 id, u32 type, u32 length, u32 count`, then `count` × 64-byte index records |

In the X-Ray stage the particle image holds **grain labels** (0 = outside, 1..255). In the
Backscatter stage (`Backscatter/<name>.jkf`) it holds **BSE grey levels**, one particle per
frame covering the full frame. Particles with no spectra have no point sub-block.

64-byte index record (one per spectrum, same order as the `.xpoints` file):

| Offset | Type | Field |
|---|---|---|
| 0x00 | u16 | 2 |
| 0x02 | 6 bytes | uninitialised memory |
| 0x08 | u64 | byte offset of the record in `.xpoints` |
| 0x10 | u32 | record size (2080 or 4080) |
| 0x14 | u32 | x |
| 0x18 | u32 | y |
| 0x28 | u32 | width |
| 0x2C | u32 | height |
| 0x30 | i32 | -1 |
| 0x38 | f32 | bse_mode (copy of the `.xpoints` header) |
| 0x3C | f32 | bse_mode_fraction (copy) |

### Frame geometry and stage coordinates

Frames are 500×500 px at 3 µm/px (`processing-stage-pixel-size = 3.0`), on a serpentine grid
with a 1,499 µm step (sample 3: 15 columns × 24 rows; row 0 runs +X, row 1 runs −X, ...).
Edge continuity between neighbouring BSE frames shows image x runs along stage +X and image
rows run along stage −Y:

    stage_X = cx + (x + 0.5 − 250) × 3 µm
    stage_Y = cy − (y + 0.5 − 250) × 3 µm

(cx, cy = frame centre from the frame block; the half-pixel origin is assumed, not tested.)

The `test` measurement of sample 3 is frame 151 of the GXMAP run, and its `.xpoints` is
byte-identical to `GXMAP/.../3_GXMAP_spx_sel.xpoints`. Its Backscatter `.jkf` holds the 3×3
frames around frame 151, starting with frame 147.

## 4. What `bse_mode` and `bse_mode_fraction` are

Both floats are the same for every spectrum in a grain (421,185 grains in sample 3, zero
exceptions). Against the frame-151 BSE image and grain labels:

* grain mean BSE vs `bse_mode`: r = 0.98;
* histogram mode of the grain's BSE pixels (ties broken toward the brighter level) equals
  `bse_mode`, **and** `floor(1000 × count_at_mode / grain_area) / 1000` equals
  `bse_mode_fraction`, for 548 of 631 grains (87%);
* small grains make up the misses (area 5–11 px). Any one-pixel shift of the label image drops
  the match to ≤26%, so alignment is right. MLA probably computed the statistic on a slightly
  different pixel set (for example before small-grain clean-up).

## 5. `.jkx` mineral list

`JKXF`, version 4; u32 entry count at 0x88; first entry offset at 0x90. Each entry:
+0x18 u32 channel count, +0x1C char[32] X-ray reference name (`XRay0`, `XRay0_0`, `Hornbl`,
...), +0x5C u32 entry length, +0x70 2000 × u32 reference counts (same channel scale as
`.xpoints`), then an `XNFO` block with the mineral name and formula as u32-length-prefixed
cp1252 strings at XNFO+0x7A. The three special classes (`Unknown`, `Low_Counts`, `No_XRay`)
have a channel count of 1 and no spectrum. The Kindisch list holds 34 entries, 31 reference
spectra, in the same order as `classification_report.csv`.

## Evidence

1. **File-size arithmetic, all 20 GXMAP files.** With N = "x-ray collected" from each
   `<sample>_GXMAP.txt`, `(size − 2080·N)` is a non-negative multiple of 2000 in every file,
   giving the number of uint16 records (180 to 24,853, i.e. 0.01–2.0%).
2. **Full stream of sample 3** (2,531,299,840 bytes, never stored): 1,216,798 records parse
   back to back to the last byte. Every header (offset, size, x, y, width, height, both
   floats) equals its `.jkf` index record: **0 mismatches**.
3. **Index tiling, all 20 samples.** In 19 samples the X-Ray `.jkf` index starts at offset 0,
   is contiguous, ends exactly at the `.xpoints` size, counts exactly the logged spectra, and
   has exactly as many 4080-byte records as the size arithmetic predicts. Sample 5-1a has
   gaps where frames were dropped from the `.jkf` (see below). In all 20, the first ~16,000
   headers (partial inflate) equal the index. See [Per-sample results](#per-sample-results).
4. **Physics.** Peaks of the summed test-frame spectrum fall on C, O, Na, Mg, Al, Si, P, K, Ca,
   Ti, Cr and Fe lines (§2). Nearest-reference classification of the 2,981 test spectra
   against all 31 `.jkx` references (`analysis/classify_check.py`) gives plagioclase 835 vs
   MLA 889, amphibole (both kinds) 819 vs 773, albite 86 vs 84, muscovite 47 vs 45, calcite
   21 vs 19, ilmenite 43 vs 34, chromite 10 vs 12, quartz 0 vs 0. Most disagreement is among
   similar Fe–Mg silicates (actinolite vs hornblende, enstatite, biotite vs chlorite) and
   MLA's 100 "Unknown", which a nearest-match rule cannot produce.
5. **Grain statistics** (§4).
6. **Published mosaics.** Each BSE frame decoded from the Backscatter `.jkf` (frames 0, 151,
   200, 359 tested) matches the 500×500 block of `Results/3_GXMAP_BSE_SAMPLE.bmp` (7495×11992,
   bottom-up 8-bit BMP) at column round((cx − min cx)/3) and row round((max cy − cy)/3) from
   the top. Every interior pixel is identical; only the outermost row or column differs,
   where the next frame was pasted over it (frames are 500 px on a 499.67 px pitch). A
   vertically flipped frame gives r ≤ 0.1. This confirms the frame placement and axis
   directions of §3 (`analysis/mosaic_check.py`).
7. **Mineral map, per spectrum.** `Results/3_GXMAP_MINERAL_SAMPLE.bmp` (7439×11924; palette
   index = position in `Granite_Legend.png`, 0 = background) is cropped differently from the
   BSE mosaic. Searching offsets within ±150 px, the MLA mineral under each test-frame
   spectrum agrees with the nearest-reference mineral for 64% of the 2,981 spectra at a
   single sharp optimum (39 rows, 5 columns). It is 84% with the two amphiboles merged,
   against ~23% at other offsets and ≤16% with the frame flipped either way.

### Per-sample results

| Sample | Frames | Spectra (log) | Spectra (.jkf) | uint16 by size | uint16 in .jkf | Index tiles file | Headers checked | 0-size spots |
|---|---|---|---|---|---|---|---|---|
| 1b-a | 400 | 1,095,745 | 1,095,745 | 13,604 | 13,604 | yes | 15,938 match | 0 |
| 1b-c | 360 | 989,748 | 989,748 | 10,511 | 10,511 | yes | 16,633 match | 0 |
| 1c-a | 352 | 954,793 | 954,793 | 9,925 | 9,925 | yes | 16,346 match | 7 |
| 1c-b | 400 | 1,088,195 | 1,088,195 | 13,855 | 13,855 | yes | 16,435 match | 42 |
| 2 | 374 | 1,068,413 | 1,068,413 | 4,585 | 4,585 | yes | 15,021 match | 8 |
| 3 | 360 | 1,216,798 | 1,216,798 | 180 | 180 | yes | 15,231 match | 3 |
| 4 | 384 | 1,051,984 | 1,051,984 | 13,455 | 13,455 | yes | 16,089 match | 0 |
| 4b-a | 425 | 1,212,276 | 1,212,276 | 13,625 | 13,625 | yes | 16,130 match | 0 |
| 4b-b | 374 | 990,094 | 990,094 | 12,425 | 12,425 | yes | 16,097 match | 547 |
| 5-1a | 432 | 1,218,824 | 1,166,549 | 24,853 | 24,020 | gaps, see below | 16,759 match | 0 |
| 6-1a | 400 | 1,117,586 | 1,117,586 | 18,554 | 18,554 | yes | 16,001 match | 406 |
| 6-1b | 459 | 1,293,888 | 1,293,888 | 17,036 | 17,036 | yes | 15,824 match | 1 |
| 6-3c | 442 | 1,268,903 | 1,268,903 | 14,059 | 14,059 | yes | 16,143 match | 9 |
| 7a | 416 | 1,152,655 | 1,152,655 | 18,334 | 18,334 | yes | 15,994 match | 0 |
| 7b | 384 | 1,051,429 | 1,051,429 | 15,457 | 15,457 | yes | 15,983 match | 0 |
| 7b-1 | 408 | 1,201,093 | 1,201,093 | 9,141 | 9,141 | yes | 16,207 match | 0 |
| 7c | 384 | 1,040,943 | 1,040,943 | 16,116 | 16,116 | yes | 16,029 match | 1 |
| 8-2c | 408 | 1,098,502 | 1,098,502 | 16,671 | 16,671 | yes | 15,948 match | 238 |
| 8-2d | 374 | 1,042,563 | 1,042,563 | 13,540 | 13,540 | yes | 16,165 match | 4 |
| 8c | 408 | 1,220,858 | 1,220,858 | 11,672 | 11,672 | yes | 16,457 match | 0 |

In 5-1a the `.jkf` no longer references 23 frames (17–18, 53, 89–90, … at the ends of
serpentine rows): it has 1,166,549 index entries against 1,218,824 records, in 12 gaps. The
records are still in the `.xpoints` file: the first gap (10,122,160 bytes) parses as 4,752
valid records (119 uint16) starting exactly on a record boundary, and the size arithmetic
holds for the whole file. So a reader must walk the `.xpoints` file itself and must not
assume the index tiles it. `mla_xpoints check` matches by offset and reports such records.

## Open questions

* Meaning of the constant fields 0x02 (3), 0x04 (2), 0x2C (1), 0x44 (4), 0x48 (4).
* Why some records have width = height = 0.
* The 13% of grains whose `bse_mode` statistics cannot be reproduced exactly.
* `.jkf` file header, palette block and the remaining frame/particle fields.
* Whether other MLA versions or modes (XBSE, SPL, XMOD) write the same layout.
