"""Check frame placement against Results/*_GXMAP_BSE_SAMPLE.bmp and spectra against the
mineral map (evidence 6 and 7 in FORMAT.md). Run from the project root after downloading
Results/3_GXMAP_{BSE,MINERAL}_SAMPLE.bmp, the sample-3 Backscatter .jkf and the test frame.
"""
import numpy as np

from mla_xpoints import JkfFile, XPointsFile, read_jkx

Z = "data/zip/"
S3 = Z + "Samples/3/Measurements/"
LEGEND = [None, "Apatite", "Biotite", "Calcite", "Chamosite", "Chlorite", "Epidote?*", "Ilmenite with Mn",
          "Feldspar (Albite)", "Feldspar (Orthoclase)", "Feldspar (Plagioclase)", "Amphibole (Actinolite)",
          "Amphibole (Hornblende)", "Pyroxene (Enstatite)", "Monazite", "Hematite", "Magnetite_Ti", "Muscovite",
          "Pyrrhotite", "Quartz", "Sphalerite with mediumFe", "Titanite", "Unknown", "Low_Counts", "No_XRay", "Chromite"]


def bmp8(path: str) -> np.ndarray:
    head = np.fromfile(path, np.uint8, 54)
    w, h = (int(v) for v in head[18:26].view("<i4"))
    stride = (w + 3) // 4 * 4
    return np.memmap(path, np.uint8, "r", offset=1078, shape=(h, stride))[::-1, :w]


def main() -> None:
    mos = bmp8(Z + "Results/3_GXMAP_BSE_SAMPLE.bmp")
    with JkfFile(S3 + "GXMAP/Processing Stages/Backscatter/3_GXMAP.jkf") as b:
        cx0 = min(f.center_um[0] for f in b.frames)
        cy1 = max(f.center_um[1] for f in b.frames)
        for n in (0, 151, 200, 359):
            f = b.frame(n)
            r, c = round((cy1 - f.center_um[1]) / 3), round((f.center_um[0] - cx0) / 3)
            diff = np.asarray(mos[r:r + 500, c:c + 500]) != b.frame_image(n)
            print(f"frame {n}: mosaic block at row {r}, col {c}: interior identical "
                  f"{not diff[1:-1, 1:-1].any()}, {int(diff.sum())} edge pixels differ")
        f151 = b.frame(151)
        r151, c151 = round((cy1 - f151.center_um[1]) / 3), round((f151.center_um[0] - cx0) / 3)

    minmap = np.asarray(bmp8(Z + "Results/3_GXMAP_MINERAL_SAMPLE.bmp"))
    refs = [r for r in read_jkx(Z + "Mineral_List/2024_-1_08_GraniteLausitz/2024_-1_08_GraniteLausitz.jkx") if r.spectrum.size]
    R = np.array([r.spectrum[38:1100] for r in refs], float)
    R /= R.sum(1, keepdims=True)
    with XPointsFile(S3 + "test/Processing Stages/X-Ray/3_test_X-Ray.xpoints") as xp:
        S = xp.spectra()[:, 38:1100].astype(float)
        x, y = xp.headers["x"].astype(int), xp.headers["y"].astype(int)
    k = S.sum(1)
    ours = np.array([LEGEND.index(refs[(((s - kk * R) ** 2) / (kk * R + 1)).sum(1).argmin()].mineral) for s, kk in zip(S, k)])
    best = (0.0, 0, 0)
    for dy in range(-150, 151):
        for dx in range(-150, 151):
            rr, cc = r151 + y - dy, c151 + x - dx
            if rr.min() < 0 or cc.min() < 0 or rr.max() >= minmap.shape[0] or cc.max() >= minmap.shape[1]:
                continue
            best = max(best, (float(np.mean(minmap[rr, cc] == ours)), dy, dx))
    print(f"mineral map: best per-spectrum agreement {best[0]:.3f} at offset (rows {best[1]}, cols {best[2]})")


if __name__ == "__main__":
    main()
