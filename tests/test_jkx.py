from pathlib import Path

import numpy as np
import pytest
from synth import jkx_entry, jkx_file

from mla_xpoints import JkxFormatError, read_jkx

REAL = Path(__file__).resolve().parents[1] / "data/zip/Mineral_List/2024_-1_08_GraniteLausitz/2024_-1_08_GraniteLausitz.jkx"


def test_reads_references_and_special_entries(tmp_path):
    spec = np.arange(2000)
    path = tmp_path / "m.jkx"
    path.write_bytes(jkx_file([
        jkx_entry("XRay0", spec, "Chlorite", "(Mg,Fe)3(OH)6·x"),
        jkx_entry("Unknown", None),
        jkx_entry("XRay0_0", spec * 2, "Quartz", "SiO2"),
    ]))
    refs = read_jkx(path)
    assert [r.mineral for r in refs] == ["Chlorite", "Unknown", "Quartz"]
    assert refs[0].formula.endswith("·x")
    assert [r.xray_name for r in refs] == ["XRay0", "Unknown", "XRay0_0"]
    np.testing.assert_array_equal(refs[2].spectrum, spec * 2)
    assert refs[1].spectrum.size == 0


def test_rejects_bad_files(tmp_path):
    bad = tmp_path / "bad.jkx"
    bad.write_bytes(b"NOPE" + bytes(200))
    with pytest.raises(JkxFormatError, match="magic"):
        read_jkx(bad)
    entry = bytearray(jkx_entry("XRay0", np.ones(2000), "Q", "SiO2"))
    entry[0x70 + 8000 : 0x70 + 8004] = b"XXXX"
    bad.write_bytes(jkx_file([bytes(entry)]))
    with pytest.raises(JkxFormatError, match="XNFO"):
        read_jkx(bad)


@pytest.mark.realdata
@pytest.mark.skipif(not REAL.exists(), reason="RODARE 4896 mineral list not downloaded")
def test_real_mineral_list():
    refs = read_jkx(REAL)
    assert len(refs) == 34
    assert [r.mineral for r in refs if r.spectrum.size == 0] == ["Unknown", "Low_Counts", "No_XRay"]
    quartz = [r for r in refs if r.mineral == "Quartz"][0]
    assert quartz.formula == "SiO2" and quartz.spectrum.size == 2000
    assert 170 <= int(np.argmax(quartz.spectrum[38:])) + 38 <= 178  # Si K-alpha


def test_truncated_and_odd_bytes(tmp_path):
    path = tmp_path / "t.jkx"
    path.write_bytes(b"JKXF" + bytes(10))
    with pytest.raises(JkxFormatError):
        read_jkx(path)
    good = jkx_file([jkx_entry("XRay0", np.ones(2000), "Q", "SiO2")])
    path.write_bytes(good[:3000])
    with pytest.raises(JkxFormatError):
        read_jkx(path)
    odd = jkx_entry("XRay0", np.ones(2000), "Q", "SiO2").replace(b"SiO2", b"Si\x81\x8d")
    path.write_bytes(jkx_file([odd]))
    assert read_jkx(path)[0].formula == "Si��"
