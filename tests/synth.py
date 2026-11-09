"""Builders for synthetic .xpoints and .jkf files that follow the decoded layout."""

from __future__ import annotations

import struct

import numpy as np

N_CH = 2000


def xpoints_record(
    counts: np.ndarray,
    x: int = 1,
    y: int = 2,
    size: int = 10,
    bse: float = 88.0,
    frac: float = 0.113,
    wide: bool | None = None,
    garbage: bytes | None = None,
) -> bytes:
    """One record. ``garbage`` fills the padding bytes, as MLA does with stale memory."""
    counts = np.asarray(counts)
    nch = len(counts)
    wide = bool(counts.max() > 255) if wide is None else wide
    payload = counts.astype("<u2").tobytes() if wide else counts.astype(np.uint8).tobytes()
    h = bytearray(garbage[:80] if garbage else bytes(80))
    for off, val in ((8, 0), (16, 0), (24, 0), (28, 0), (32, 0), (36, 0), (56, 0), (60, 0), (64, 0)):
        struct.pack_into("<I", h, off, val)
    h[0] = 0x5A
    struct.pack_into("<HH", h, 2, 3, 2)
    struct.pack_into("<HH", h, 12, x, y)
    struct.pack_into("<HHH", h, 40, size, size, 1)
    struct.pack_into("<ff", h, 48, bse, frac)
    h[68] = 4
    struct.pack_into("<H", h, 70, nch)
    h[72] = 4
    struct.pack_into("<I", h, 76, len(payload))
    return bytes(h) + payload


def random_spectra(n: int, seed: int = 0, nch: int = N_CH, peak: int = 200) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, peak, size=(n, nch))


def packbits(data: bytes) -> bytes:
    """Simple PackBits encoder (runs of >= 3 become repeat packets)."""
    out, i, n = bytearray(), 0, len(data)
    while i < n:
        run = 1
        while i + run < n and run < 128 and data[i + run] == data[i]:
            run += 1
        if run >= 3:
            out += bytes((257 - run, data[i]))
            i += run
            continue
        j = i
        while j < n and j - i < 128 and not (j + 2 < n and data[j] == data[j + 1] == data[j + 2]):
            j += 1
        out += bytes((j - i - 1,)) + data[i:j]
        i = j
    return bytes(out)


def _point(offset: int, rec_bytes: int, x: int, y: int, size: int, bse: float, frac: float) -> bytes:
    r = bytearray(64)
    struct.pack_into("<H", r, 0, 2)
    r[2:8] = b"\x72\x71\x00\x00\x1d\x05"  # stale memory, as in real files
    struct.pack_into("<QIII", r, 8, offset, rec_bytes, x, y)
    struct.pack_into("<II", r, 40, size, size)
    struct.pack_into("<iI", r, 48, -1, 0)
    struct.pack_into("<ff", r, 56, bse, frac)
    return bytes(r)


def jkf_particle(pid: int, bbox: tuple[int, int, int, int], image: np.ndarray, points: list[bytes]) -> bytes:
    xmin, ymin, xmax, ymax = bbox
    h, w = image.shape
    stride = (w + 3) // 4 * 4
    padded = np.zeros((h, stride), np.uint8)
    padded[:, :w] = image
    data = packbits(padded.tobytes())
    img = struct.pack("<IIQIIIQI", 1, 0xA28, 40 + len(data), w, h, stride, len(data), 2348 + pid) + data
    pts = b""
    if points:
        pts = struct.pack("<IIII", 2, 0xA64, 16 + 64 * len(points), len(points)) + b"".join(points)
    body = struct.pack("<IIII", 0, pid - 1, 0, 0) + struct.pack("<I4III", 800, xmin, ymin, xmax, ymax, 0, 2)
    body += struct.pack("<I", 0xCB)
    length = 16 + len(body) + len(img) + len(pts)
    return struct.pack("<IIQ", pid, 0x64A, length) + body + img + pts


def jkf_frame(number: int, center: tuple[float, float], particles: list[bytes]) -> bytes:
    hdr = struct.pack("<III", number, number, len(particles)) + struct.pack("<QI", 0, 0)
    hdr += struct.pack("<II", 500, 500) + struct.pack("<I", 1) + bytes(0x48 - 0x34)
    hdr += struct.pack("<fffff", 158.667, center[0], center[1], -63210.0, -21013.0) + struct.pack("<I", 0xCB)
    assert len(hdr) == 0x60 - 16
    body = hdr + b"".join(particles)
    return struct.pack("<IIQ", number + 3, 0x5DC, 16 + len(body)) + body


def jkf_file(frames: list[bytes]) -> bytes:
    head = bytearray(0x530)
    head[0:4] = b"JK\x04\x00"
    struct.pack_into("<III", head, 0x120, 2, 0xC1C, 0x410)
    body = bytes(head) + b"".join(frames) + b"\x00\x00"
    struct.pack_into("<Q", head, 0x10, len(body))
    return bytes(head) + b"".join(frames) + b"\x00\x00"


def point(offset: int, rec_bytes: int, x: int, y: int, size: int = 10, bse: float = 88.0, frac: float = 0.113) -> bytes:
    return _point(offset, rec_bytes, x, y, size, bse, frac)


def jkx_entry(xray_name: str, spectrum: np.ndarray | None, mineral: str = "", formula: str = "") -> bytes:
    nch = 1 if spectrum is None else len(spectrum)
    head = bytearray(0x70)
    struct.pack_into("<III", head, 0, 4, 1, 100)
    struct.pack_into("<I", head, 0x18, nch)
    head[0x1C : 0x1C + len(xray_name)] = xray_name.encode()
    if spectrum is None:
        body = bytes(head) + bytes(284 - 0x70)
    else:
        xnfo = bytearray(b"XNFO" + bytes(0x7A - 4))
        for text in (mineral, formula):
            raw = text.encode("cp1252")
            xnfo += struct.pack("<I", len(raw)) + raw
        body = bytes(head) + np.asarray(spectrum, "<u4").tobytes() + bytes(xnfo) + bytes(16)
    body = bytearray(body)
    struct.pack_into("<I", body, 0x5C, len(body))
    return bytes(body)


def jkx_file(entries: list[bytes]) -> bytes:
    head = bytearray(0x94)
    head[0:8] = b"JKXF\x04\x00\x00\x00"
    struct.pack_into("<I4xI", head, 0x88, len(entries), 0x94)
    return bytes(head) + b"".join(entries) + b"trailing data"
