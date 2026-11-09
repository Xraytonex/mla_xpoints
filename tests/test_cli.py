import numpy as np
from synth import jkf_file, jkf_frame, jkf_particle, point, random_spectra, xpoints_record

from mla_xpoints.cli import main


def make_pair(tmp_path, bse_in_index=80.0):
    spec = random_spectra(2)
    xp = tmp_path / "a.xpoints"
    xp.write_bytes(xpoints_record(spec[0], x=5, y=2, bse=80.0) + xpoints_record(spec[1], x=6, y=3, bse=90.0))
    img = np.ones((10, 20), np.uint8)
    part = jkf_particle(1, (0, 0, 20, 10), img, [point(0, 2080, 5, 2, bse=bse_in_index), point(2080, 2080, 6, 3, bse=90.0)])
    jk = tmp_path / "a.jkf"
    jk.write_bytes(jkf_file([jkf_frame(0, (0.0, 0.0), [part])]))
    return str(xp), str(jk), spec


def test_info(tmp_path, capsys):
    xp, _, spec = make_pair(tmp_path)
    assert main(["info", xp, "--counts"]) == 0
    out = capsys.readouterr().out
    assert "records         2  (uint8 2, uint16 0)" in out
    assert f"max {spec.sum(1).max()}" in out


def test_export_round_trip(tmp_path):
    xp, _, spec = make_pair(tmp_path)
    out = tmp_path / "o.npz"
    assert main(["export", xp, str(out), "--start", "1"]) == 0
    z = np.load(out)
    np.testing.assert_array_equal(z["spectra"], spec[1:])
    np.testing.assert_array_equal(z["record"], [1])


def test_check_passes_and_fails(tmp_path, capsys):
    xp, jk, _ = make_pair(tmp_path)
    assert main(["check", xp, jk]) == 0
    assert "all indexed headers match" in capsys.readouterr().out
    xp, jk, _ = make_pair(tmp_path, bse_in_index=81.0)
    assert main(["check", xp, jk]) == 1
    assert "bse_mode: 1 mismatches" in capsys.readouterr().out


def test_bad_file_returns_error_code(tmp_path, capsys):
    bad = tmp_path / "bad.xpoints"
    bad.write_bytes(b"\x00" * 100)
    assert main(["info", str(bad)]) == 2
    assert "error:" in capsys.readouterr().err


def test_check_reports_unindexed_records(tmp_path, capsys):
    spec = random_spectra(3)
    xp = tmp_path / "b.xpoints"
    xp.write_bytes(b"".join(xpoints_record(s, x=i, y=i) for i, s in enumerate(spec)))
    part = jkf_particle(1, (0, 0, 20, 10), np.ones((10, 20), np.uint8), [point(0, 2080, 0, 0, bse=88.0), point(4160, 2080, 2, 2, bse=88.0)])
    jk = tmp_path / "b.jkf"
    jk.write_bytes(jkf_file([jkf_frame(0, (0.0, 0.0), [part])]))
    assert main(["check", str(xp), str(jk)]) == 0
    out = capsys.readouterr().out
    assert "1 records are not referenced" in out and "all indexed headers match" in out


def test_export_refuses_huge_ranges(tmp_path, capsys, monkeypatch):
    import mla_xpoints.cli as cli

    xp, _, _ = make_pair(tmp_path)
    monkeypatch.setattr(cli, "MAX_EXPORT", 1)
    assert main(["export", xp, str(tmp_path / "o")]) == 2
    assert "export at most 1" in capsys.readouterr().err
    assert main(["export", xp, str(tmp_path / "o"), "--force"]) == 0
    assert (tmp_path / "o.npz").exists()
