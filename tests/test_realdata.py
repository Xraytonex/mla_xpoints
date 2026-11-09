"""Checks against the RODARE 4896 test frame; skipped unless the files are under data/."""

from pathlib import Path

import numpy as np
import pytest

from mla_xpoints import JkfFile, XPointsFile, energy_axis_kev

ROOT = Path(__file__).resolve().parents[1] / "data/zip/Samples/3/Measurements/test/Processing Stages"
XP = ROOT / "X-Ray/3_test_X-Ray.xpoints"
JK = ROOT / "X-Ray/3_test_X-Ray.jkf"
pytestmark = [pytest.mark.realdata, pytest.mark.skipif(not XP.exists(), reason="RODARE 4896 test frame not downloaded")]


def test_test_frame_layout():
    with XPointsFile(XP) as xp:
        assert len(xp) == 2981 == xp.size // 2080
        h = xp.headers
        assert (h["tag"] == 0x5A).all() and (h["version_a"] == 3).all() and (h["version_b"] == 2).all()
        assert set(np.unique(h["width"])) == {1, 10}
        tot = xp.total_counts()
        assert tot.min() == 1754 and tot.max() == 5077


def test_headers_match_jkf_index():
    with XPointsFile(XP) as xp, JkfFile(JK) as jk:
        idx = jk.point_index(with_grains=True)
        assert jk.frames[0].number == 151
        np.testing.assert_array_equal(idx["offset"], xp.table.offsets)
        for f in ("x", "y", "width", "height", "bse_mode", "bse_mode_fraction"):
            np.testing.assert_array_equal(idx[f], xp.headers[f])
        assert (idx["grain"] > 0).all()


def test_major_peaks_land_on_expected_lines():
    with XPointsFile(XP) as xp:
        s = xp.sum_spectrum()
    e = energy_axis_kev()
    for line_kev in (1.4866, 1.7398, 3.6917, 6.4039):  # Al, Si, Ca, Fe K-alpha
        window = np.abs(e - line_kev) < 0.06
        peak = e[window][np.argmax(s[window])]
        assert abs(peak - line_kev) < 0.015
