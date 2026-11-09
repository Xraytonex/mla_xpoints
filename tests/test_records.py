import os

import numpy as np
import pytest
from synth import N_CH, random_spectra, xpoints_record

from mla_xpoints import HEADER_DTYPE, XPointsFile, XPointsFormatError, read_xpoints, scan_records


def write(tmp_path, records: list[bytes], name: str = "f.xpoints") -> str:
    path = tmp_path / name
    path.write_bytes(b"".join(records))
    return str(path)


def test_header_dtype_is_80_bytes():
    assert HEADER_DTYPE.itemsize == 80


def test_reads_uniform_uint8_file(tmp_path):
    spec = random_spectra(5, peak=200)
    recs = [xpoints_record(s, x=i, y=10 * i, size=1 if i % 2 else 10, bse=70.0 + i, frac=0.25) for i, s in enumerate(spec)]
    path = write(tmp_path, recs)

    with XPointsFile(path) as xp:
        assert len(xp) == 5
        assert xp.n_channels == N_CH
        np.testing.assert_array_equal(xp.table.offsets, np.arange(5) * 2080)
        np.testing.assert_array_equal(xp.headers["x"], np.arange(5))
        np.testing.assert_array_equal(xp.headers["y"], np.arange(5) * 10)
        np.testing.assert_array_equal(xp.headers["width"], [10, 1, 10, 1, 10])
        np.testing.assert_allclose(xp.headers["bse_mode"], 70.0 + np.arange(5))
        np.testing.assert_array_equal(xp.spectra(), spec)
        assert not xp.table.is_uint16.any()


def test_reads_mixed_uint8_and_uint16_records(tmp_path):
    spec = random_spectra(6, peak=200)
    spec[2, 500] = 300
    spec[4, 7] = 65535
    path = write(tmp_path, [xpoints_record(s) for s in spec])

    with XPointsFile(path) as xp:
        np.testing.assert_array_equal(xp.table.is_uint16, [0, 0, 1, 0, 1, 0])
        np.testing.assert_array_equal(xp.table.offsets, [0, 2080, 4160, 8240, 10320, 14400])
        np.testing.assert_array_equal(xp.spectra(), spec)
        np.testing.assert_array_equal(xp.spectrum(4), spec[4])


def test_mixed_file_with_uniform_looking_size_is_not_misread(tmp_path):
    # 26 uint16 records add 26 * 2000 bytes = 25 * 2080, so the size is a multiple of 2080.
    spec = random_spectra(30, peak=200)
    spec[:26, 0] = 1000
    path = write(tmp_path, [xpoints_record(s) for s in spec])
    assert os.path.getsize(path) % 2080 == 0

    with XPointsFile(path) as xp:
        assert len(xp) == 30
        np.testing.assert_array_equal(xp.spectra(), spec)


def test_padding_garbage_is_ignored(tmp_path):
    rng = np.random.default_rng(1)
    spec = random_spectra(3)
    recs = [xpoints_record(s, x=7, y=9, garbage=rng.integers(0, 256, 80, dtype=np.uint8).tobytes()) for s in spec]
    with XPointsFile(write(tmp_path, recs)) as xp:
        assert (xp.headers["x"] == 7).all() and (xp.headers["y"] == 9).all()
        np.testing.assert_array_equal(xp.spectra(), spec)


def test_empty_file_has_no_records(tmp_path):
    with XPointsFile(write(tmp_path, [])) as xp:
        assert len(xp) == 0
        assert xp.spectra().shape == (0, 0)


@pytest.mark.parametrize(
    "corrupt, message",
    [
        (lambda b: b[:-5], "payload runs past end"),
        (lambda b: b + b"\x5a" * 10, "truncated header"),
        (lambda b: b[:2080] + b"\x00" + b[2081:], "tag 0x00"),
        (lambda b: b[:2080 + 76] + b"\x07\x00\x00\x00" + b[2080 + 80 :], "payload_bytes=7"),
    ],
)
def test_malformed_files_raise_with_record_position(tmp_path, corrupt, message):
    data = b"".join(xpoints_record(s) for s in random_spectra(3))
    path = tmp_path / "bad.xpoints"
    path.write_bytes(corrupt(data))
    with pytest.raises(XPointsFormatError, match=message):
        XPointsFile(path)


def test_channel_count_change_mid_file_is_rejected():
    data = xpoints_record(np.ones(N_CH, int)) + xpoints_record(np.ones(1000, int))
    with pytest.raises(XPointsFormatError, match="n_channels=1000"):
        scan_records(data)


def test_chunked_sums_match_full_read(tmp_path):
    spec = random_spectra(23, peak=250)
    spec[5, 100] = 4000
    with XPointsFile(write(tmp_path, [xpoints_record(s) for s in spec])) as xp:
        np.testing.assert_array_equal(xp.sum_spectrum(chunk=4), spec.sum(0))
        np.testing.assert_array_equal(xp.total_counts(chunk=4), spec.sum(1))
        seen = np.concatenate([idx for idx, _ in xp.iter_chunks(chunk=10)])
        np.testing.assert_array_equal(seen, np.arange(23))


def test_read_xpoints_slice(tmp_path):
    spec = random_spectra(8)
    path = write(tmp_path, [xpoints_record(s, x=i) for i, s in enumerate(spec)])
    headers, spectra = read_xpoints(path, start=2, stop=5)
    np.testing.assert_array_equal(headers["x"], [2, 3, 4])
    np.testing.assert_array_equal(spectra, spec[2:5])


def test_spectra_accepts_slices_and_rejects_bad_indices(tmp_path):
    spec = random_spectra(4)
    with XPointsFile(write(tmp_path, [xpoints_record(s) for s in spec])) as xp:
        np.testing.assert_array_equal(xp.spectra(slice(1, 3)), spec[1:3])
        np.testing.assert_array_equal(xp.spectra([3, 0]), spec[[3, 0]])
        with pytest.raises(IndexError):
            xp.spectra([4])
        with pytest.raises(IndexError):
            xp.spectra([-1])


def test_boolean_mask_and_bad_index_types(tmp_path):
    spec = random_spectra(4)
    recs = [xpoints_record(s, size=10 if i % 2 else 1) for i, s in enumerate(spec)]
    with XPointsFile(write(tmp_path, recs)) as xp:
        mask = xp.headers["width"] == 10
        np.testing.assert_array_equal(xp.spectra(mask), spec[[1, 3]])
        with pytest.raises(IndexError, match="mask"):
            xp.spectra(np.array([True, False]))
        with pytest.raises(IndexError, match="integers"):
            xp.spectra(np.array([0.0, 1.5]))


def test_all_uint16_file_and_mixed_chunks(tmp_path):
    spec = random_spectra(7, peak=60000)
    with XPointsFile(write(tmp_path, [xpoints_record(s) for s in spec])) as xp:
        assert xp.table.is_uint16.all()
        np.testing.assert_array_equal(np.concatenate([b for _, b in xp.iter_chunks(3)]), spec)


def test_handle_is_closed_when_parsing_fails(tmp_path):
    path = tmp_path / "bad.xpoints"
    path.write_bytes(b"\x00" * 100)
    with pytest.raises(XPointsFormatError):
        XPointsFile(path)
    path.unlink()  # would fail on Windows if the handle leaked; here just exercise the path
