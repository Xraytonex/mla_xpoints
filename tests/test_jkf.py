import struct

import numpy as np
import pytest
from synth import jkf_file, jkf_frame, jkf_particle, packbits, point

from mla_xpoints import JkfFile, JkfFormatError, pixel_to_stage, unpack_packbits


def labels(h: int, w: int, value: int) -> np.ndarray:
    img = np.zeros((h, w), np.uint8)
    img[: h // 2] = value
    img[h // 2 :] = value + 1
    return img


@pytest.fixture
def two_frame_jkf(tmp_path):
    # frame 0: particle 1 (two grains) with 2 spectra, particle 2 with no spectra
    # frame 1: particle 1 (odd width -> padded stride) with 1 spectrum
    p1 = jkf_particle(1, (0, 0, 20, 10), labels(10, 20, 3), [point(0, 2080, 5, 2, bse=80.0), point(2080, 4080, 6, 8, bse=90.0)])
    p2 = jkf_particle(2, (100, 100, 110, 104), labels(4, 10, 9), [])
    p3 = jkf_particle(1, (10, 50, 17, 56), labels(6, 7, 200), [point(6160, 2080, 12, 54, size=1, bse=120.0)])
    data = jkf_file([jkf_frame(0, (-63210.0, -21013.0), [p1, p2]), jkf_frame(1, (-61711.0, -21013.0), [p3])])
    path = tmp_path / "x.jkf"
    path.write_bytes(data)
    return str(path)


def test_packbits_round_trip():
    raw = bytes([0] * 300 + [1, 2, 3, 4] + [7] * 5 + list(range(200)))
    enc = packbits(raw)
    assert unpack_packbits(enc, 0, len(enc), expected=len(raw)) == raw


def test_packbits_known_vector():
    # Apple Technical Note TN1023 example
    enc = bytes.fromhex("FEAA0280002AFDAA0380002A22F7AA")
    expected = bytes.fromhex("AAAAAA80002AAAAAAAAA80002A22AAAAAAAAAAAAAAAAAAAA")
    assert unpack_packbits(enc, 0, len(enc)) == expected


def test_packbits_rejects_wrong_length():
    enc = packbits(b"\x05" * 10)
    with pytest.raises(JkfFormatError):
        unpack_packbits(enc, 0, len(enc), expected=11)


def test_frames_particles_and_points(two_frame_jkf):
    with JkfFile(two_frame_jkf) as j:
        assert [f.number for f in j.frames] == [0, 1]
        assert j.frames[0].center_um == (-63210.0, -21013.0)
        assert [p.pid for p in j.frames[0].particles] == [1, 2]
        assert len(j.frames[0].particles[1].points) == 0
        idx = j.point_index(with_grains=True)
    np.testing.assert_array_equal(idx["offset"], [0, 2080, 6160])
    np.testing.assert_array_equal(idx["record_bytes"], [2080, 4080, 2080])
    np.testing.assert_array_equal(idx["frame"], [0, 0, 1])
    np.testing.assert_array_equal(idx["particle"], [1, 1, 1])
    np.testing.assert_array_equal(idx["grain"], [3, 4, 201])
    np.testing.assert_array_equal(idx["width"], [10, 10, 1])
    np.testing.assert_allclose(idx["bse_mode"], [80, 90, 120])


def test_point_index_without_grains_marks_minus_one(two_frame_jkf):
    with JkfFile(two_frame_jkf) as j:
        assert (j.point_index()["grain"] == -1).all()


def test_frame_image_places_particles(two_frame_jkf):
    with JkfFile(two_frame_jkf) as j:
        img = j.frame_image(0)
        assert img.shape == (500, 500)
        assert img[0, 0] == 3 and img[9, 19] == 4 and img[101, 105] == 9 and img[300, 300] == 0
        sub = j.particle_image(j.frame(1).particles[0])
        assert sub.shape == (6, 7)
        with pytest.raises(KeyError):
            j.frame(7)


def test_pixel_to_stage_orientation(two_frame_jkf):
    with JkfFile(two_frame_jkf) as j:
        f = j.frame(0)
    x, y = pixel_to_stage(f, [0, 499], [0, 499])
    np.testing.assert_allclose(x, [-63210.0 - 748.5, -63210.0 + 748.5])
    np.testing.assert_allclose(y, [-21013.0 + 748.5, -21013.0 - 748.5])


def test_rejects_non_jkf(tmp_path):
    path = tmp_path / "not.jkf"
    path.write_bytes(b"XX" + bytes(100))
    with pytest.raises(JkfFormatError, match="magic"):
        JkfFile(path)


def test_rejects_corrupt_point_block(tmp_path):
    p = bytearray(jkf_particle(1, (0, 0, 4, 4), labels(4, 4, 1), [point(0, 2080, 1, 1)]))
    p[-64 - 4 : -64] = (5).to_bytes(4, "little")  # wrong point count
    path = tmp_path / "bad.jkf"
    path.write_bytes(jkf_file([jkf_frame(0, (0.0, 0.0), [bytes(p)])]))
    with pytest.raises(JkfFormatError, match="point sub-block"):
        JkfFile(path)


def _write(tmp_path, data: bytes) -> str:
    path = tmp_path / "t.jkf"
    path.write_bytes(data)
    return str(path)


def _frame_with(particles: list[bytes], n_particles: int | None = None) -> bytes:
    frame = bytearray(jkf_frame(0, (0.0, 0.0), particles))
    if n_particles is not None:
        frame[24:28] = n_particles.to_bytes(4, "little")
    return bytes(frame)


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda p: p[:8] + (0).to_bytes(8, "little") + p[16:], "particle length 0"),
        (lambda p: p[:8] + (10**9).to_bytes(8, "little") + p[16:], "does not fit its frame"),
        (lambda p: p[:36] + struct.pack("<4I", 0, 0, 600, 4) + p[52:], "outside"),
        (lambda p: p[: 64 + 24] + (3).to_bytes(4, "little") + p[64 + 28 :], "stride 3"),
    ],
)
def test_rejects_corrupt_particles(tmp_path, mutate, message):
    good = jkf_particle(1, (0, 0, 4, 4), labels(4, 4, 1), [point(0, 2080, 1, 1)])
    path = _write(tmp_path, jkf_file([_frame_with([mutate(good)])]))
    with pytest.raises(JkfFormatError, match=message):
        JkfFile(path)


def test_rejects_impossible_particle_count_quickly(tmp_path):
    good = jkf_particle(1, (0, 0, 4, 4), labels(4, 4, 1), [])
    path = _write(tmp_path, jkf_file([_frame_with([good], n_particles=4_000_000_000)]))
    with pytest.raises(JkfFormatError, match="cannot fit"):
        JkfFile(path)


def test_rejects_truncated_file_and_trailing_garbage(tmp_path):
    good = jkf_file([_frame_with([jkf_particle(1, (0, 0, 4, 4), labels(4, 4, 1), [point(0, 2080, 1, 1)])])])
    with pytest.raises(JkfFormatError):
        JkfFile(_write(tmp_path, good[:-40]))
    with pytest.raises(JkfFormatError):
        JkfFile(_write(tmp_path, good + bytes(64)))


def test_point_outside_particle_gets_grain_minus_one(tmp_path):
    part = jkf_particle(1, (5, 5, 7, 7), labels(2, 2, 4), [point(0, 2080, 1, 1), point(2080, 2080, 6, 6)])
    with JkfFile(_write(tmp_path, jkf_file([_frame_with([part])]))) as j:
        np.testing.assert_array_equal(j.point_index(with_grains=True)["grain"], [-1, 5])


@pytest.mark.parametrize("enc", [bytes([0xFE]), bytes([0x05, 1, 2]), bytes([0x80, 0xFF])])
def test_packbits_truncated_packets_raise(enc):
    with pytest.raises(JkfFormatError, match="overruns"):
        unpack_packbits(enc, 0, len(enc))


def test_packbits_stops_once_output_exceeds_expected():
    enc = bytes([0x81, 0]) * 1000
    with pytest.raises(JkfFormatError, match="exceeds"):
        unpack_packbits(enc, 0, len(enc), expected=100)


def test_first_frame_found_without_palette_block(tmp_path):
    data = bytearray(jkf_file([_frame_with([jkf_particle(1, (0, 0, 4, 4), labels(4, 4, 1), [point(0, 2080, 1, 1)])])]))
    data[0x120:0x130] = bytes(16)  # wipe the palette block header
    data[0x200:0x204] = (0x5DC).to_bytes(4, "little")  # decoy frame tag inside the palette
    with JkfFile(_write(tmp_path, bytes(data))) as j:
        assert [f.number for f in j.frames] == [0] and j.frames[0].start == 0x530
