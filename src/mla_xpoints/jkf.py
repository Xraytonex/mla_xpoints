"""Partial reader for MLA ``.jkf`` processing-stage files (X-Ray and Backscatter stages).

Only the parts needed to place ``.xpoints`` spectra are decoded; see FORMAT.md. Layout::

    file header (0x00..)          "JK", version 4, file size (u64 at 0x10), ...
    palette block                 256 x BGRA
    frame block   (type 0x5DC)    one per frame, in frame order
      particle block (type 0x64A) one per particle in the frame
        image sub-block (0xA28)   PackBits-compressed u8 image of the particle bounding box
                                  (grain labels in the X-Ray stage, BSE grey in Backscatter)
        point sub-block (0xA64)   64-byte index record per spectrum (X-Ray stage only)

Every block starts with ``u32 id, u32 type, u64 length`` (length counted from the block
start) except the point sub-block, which stores ``u32 length, u32 count`` after its type.
Every length and count is checked against its parent block before use, so a corrupt file
raises :class:`JkfFormatError` instead of reading out of bounds.
"""

from __future__ import annotations

import mmap
import os
import struct
from dataclasses import dataclass, field

import numpy as np

FRAME_TYPE = 0x5DC
PARTICLE_TYPE = 0x64A
IMAGE_TYPE = 0xA28
POINTS_TYPE = 0xA64
POINT_RECORD_SIZE = 64
FRAME_SIZE_PX = 500
FRAME_STEP_UM = 1499.0
PIXEL_SIZE_UM = 3.0

POINT_DTYPE = np.dtype(
    {
        "names": ["offset", "record_bytes", "x", "y", "width", "height", "bse_mode", "bse_mode_fraction"],
        "formats": ["<u8", "<u4", "<u4", "<u4", "<u4", "<u4", "<f4", "<f4"],
        "offsets": [8, 16, 20, 24, 40, 44, 56, 60],
        "itemsize": POINT_RECORD_SIZE,
    }
)
"""Index record for one spectrum (fields mirror the .xpoints header plus its file offset)."""

INDEX_DTYPE = np.dtype(
    [
        ("offset", "<u8"),
        ("record_bytes", "<u4"),
        ("x", "<u4"),
        ("y", "<u4"),
        ("width", "<u4"),
        ("height", "<u4"),
        ("bse_mode", "<f4"),
        ("bse_mode_fraction", "<f4"),
        ("frame", "<u4"),
        ("particle", "<u4"),
        ("grain", "<i2"),
    ]
)


class JkfFormatError(ValueError):
    """Raised when a ``.jkf`` file does not match the decoded layout."""


def unpack_packbits(src, start: int, end: int, expected: int | None = None) -> bytes:
    """Decode Apple PackBits data in ``src[start:end]``.

    Control byte ``c``: 0..127 copies the next ``c + 1`` bytes, 129..255 repeats the next
    byte ``257 - c`` times, 128 is a no-op.
    """
    if start < 0 or end > len(src) or start > end:
        raise JkfFormatError(f"PackBits span {start}..{end} outside the buffer")
    limit = expected if expected is not None else None
    out = bytearray()
    i = start
    while i < end:
        c = src[i]
        i += 1
        if c < 128:
            if i + c + 1 > end:
                raise JkfFormatError(f"PackBits literal at {i - 1} overruns its block end {end}")
            out += src[i : i + c + 1]
            i += c + 1
        elif c > 128:
            if i >= end:
                raise JkfFormatError(f"PackBits repeat at {i - 1} overruns its block end {end}")
            out += bytes((src[i],)) * (257 - c)
            i += 1
        if limit is not None and len(out) > limit:
            raise JkfFormatError(f"PackBits output exceeds the expected {limit} bytes")
    if expected is not None and len(out) != expected:
        raise JkfFormatError(f"PackBits decoded {len(out)} bytes, expected {expected}")
    return bytes(out)


@dataclass(frozen=True)
class Particle:
    pid: int
    start: int
    end: int
    bbox: tuple[int, int, int, int]  # xmin, ymin, xmax, ymax (max exclusive), frame pixels
    image_span: tuple[int, int, int, int, int]  # data start, data end, width, height, stride
    points: np.ndarray  # POINT_DTYPE records

    def image(self, buf) -> np.ndarray:
        """Decode this particle's u8 image (grain labels or BSE) as ``(height, width)``."""
        d0, d1, w, h, stride = self.image_span
        raw = unpack_packbits(buf, d0, d1, expected=h * stride)
        return np.frombuffer(raw, np.uint8).reshape(h, stride)[:, :w]


@dataclass(frozen=True)
class Frame:
    number: int
    block_id: int
    start: int
    end: int
    center_um: tuple[float, float]  # stage position of the frame centre
    magnification: float
    width: int
    height: int
    particles: tuple[Particle, ...] = field(default_factory=tuple)


PALETTE_TYPE = 0xC1C
PALETTE_BLOCK_OFFSET = 0x120
FRAME_HEADER_SIZE = 0x60
PARTICLE_HEADER_SIZE = 64
IMAGE_HEADER_SIZE = 40
MAX_TAIL_BYTES = 16  # real files end 2 bytes after the last frame block


def _unpack(fmt: str, buf, pos: int, limit: int) -> tuple:
    """struct.unpack_from that refuses to read past ``limit`` and raises JkfFormatError."""
    if pos < 0 or pos + struct.calcsize(fmt) > limit:
        raise JkfFormatError(f"byte {pos}: {fmt} field runs past its block (limit {limit})")
    return struct.unpack_from(fmt, buf, pos)


def _block(buf, pos: int, limit: int) -> tuple[int, int, int]:
    return _unpack("<IIQ", buf, pos, limit)


def _parse_image(buf, img: int, end: int, bbox: tuple[int, int, int, int]) -> tuple[tuple, int]:
    _, ityp, ilen = _block(buf, img, end)
    if ityp != IMAGE_TYPE:
        raise JkfFormatError(f"byte {img}: expected image sub-block, got type {ityp:#x}")
    w, h, stride, dlen = _unpack("<IIIQ", buf, img + 16, end)
    d0 = img + IMAGE_HEADER_SIZE
    if d0 + dlen != img + ilen or img + ilen > end:
        raise JkfFormatError(f"byte {img}: image sub-block length {ilen} inconsistent")
    if w != bbox[2] - bbox[0] or h != bbox[3] - bbox[1] or stride < w:
        raise JkfFormatError(f"byte {img}: image {w}x{h} (stride {stride}) does not match bbox {bbox}")
    return (d0, d0 + dlen, w, h, stride), d0 + dlen


def _parse_points(buf, p: int, end: int) -> np.ndarray:
    _, ptyp, plen, count = _unpack("<IIII", buf, p, end)
    if ptyp != POINTS_TYPE or plen != 16 + POINT_RECORD_SIZE * count or p + plen != end:
        raise JkfFormatError(f"byte {p}: malformed point sub-block")
    return np.frombuffer(buf, POINT_DTYPE, count=count, offset=p + 16).copy()


def _parse_particle(buf, pos: int, limit: int, frame_size: tuple[int, int]) -> Particle:
    pid, typ, length = _block(buf, pos, limit)
    if typ != PARTICLE_TYPE:
        raise JkfFormatError(f"byte {pos}: expected particle block, got type {typ:#x}")
    end = pos + length
    if length < PARTICLE_HEADER_SIZE + IMAGE_HEADER_SIZE or end > limit:
        raise JkfFormatError(f"byte {pos}: particle length {length} does not fit its frame")
    bbox = _unpack("<4I", buf, pos + 36, end)
    if not (bbox[0] < bbox[2] <= frame_size[0] and bbox[1] < bbox[3] <= frame_size[1]):
        raise JkfFormatError(f"byte {pos}: particle bbox {bbox} outside the {frame_size} frame")
    span, p = _parse_image(buf, pos + PARTICLE_HEADER_SIZE, end, bbox)
    points = _parse_points(buf, p, end) if p < end else np.zeros(0, POINT_DTYPE)
    return Particle(pid, pos, end, bbox, span, points)


def _parse_frame(buf, pos: int, size: int) -> Frame:
    block_id, typ, length = _block(buf, pos, size)
    end = pos + length
    if typ != FRAME_TYPE or length < FRAME_HEADER_SIZE or end > size:
        raise JkfFormatError(f"byte {pos}: bad frame block (type {typ:#x}, length {length})")
    number, _, n_particles = _unpack("<III", buf, pos + 16, end)
    width, height = _unpack("<II", buf, pos + 40, end)
    mag, cx, cy = _unpack("<fff", buf, pos + 0x48, end)
    max_particles = (length - FRAME_HEADER_SIZE) // (PARTICLE_HEADER_SIZE + IMAGE_HEADER_SIZE)
    if n_particles > max_particles:
        raise JkfFormatError(f"frame {number}: {n_particles} particles cannot fit in {length} bytes")
    particles, p = [], pos + FRAME_HEADER_SIZE
    for _ in range(n_particles):
        part = _parse_particle(buf, p, end, (width, height))
        particles.append(part)
        p = part.end
    if p != end:
        raise JkfFormatError(f"frame {number}: particle chain ends at {p}, block ends at {end}")
    return Frame(number, block_id, pos, end, (cx, cy), mag, width, height, tuple(particles))


def _first_frame(buf, size: int) -> int:
    """Offset of the first frame block: right after the palette block at 0x120 (all files
    seen); otherwise the first plausible frame header in the first 64 KiB."""
    try:
        _, typ, length = _block(buf, PALETTE_BLOCK_OFFSET, size)
        start = PALETTE_BLOCK_OFFSET + length
        if typ == PALETTE_TYPE and _block(buf, start, size)[1] == FRAME_TYPE:
            return start
    except JkfFormatError:
        pass
    tag = struct.pack("<I", FRAME_TYPE)
    pos = 3
    while True:
        pos = buf.find(tag, pos + 1, min(size, 1 << 16))
        if pos < 0:
            raise JkfFormatError("no frame block found")
        start = pos - 4
        try:
            _parse_frame(buf, start, size)
            return start
        except JkfFormatError:
            continue


class JkfFile:
    """Memory-mapped ``.jkf`` file with its frames and particles parsed."""

    def __init__(self, path: str | os.PathLike):
        self.path = os.fspath(path)
        self._fh = open(self.path, "rb")
        try:
            self.size = os.fstat(self._fh.fileno()).st_size
            if self._fh.read(2) != b"JK":
                raise JkfFormatError("missing 'JK' magic")
            if self.size < PALETTE_BLOCK_OFFSET:
                raise JkfFormatError(f"file too short for a .jkf ({self.size} bytes)")
            self._mm = mmap.mmap(self._fh.fileno(), 0, access=mmap.ACCESS_READ)
            self.frames = self._parse_frames()
        except Exception:
            self.close()
            raise

    def _parse_frames(self) -> tuple[Frame, ...]:
        frames, pos = [], _first_frame(self._mm, self.size)
        while self.size - pos > MAX_TAIL_BYTES:
            frame = _parse_frame(self._mm, pos, self.size)
            frames.append(frame)
            pos = frame.end
        return tuple(frames)

    def frame(self, number: int) -> Frame:
        for f in self.frames:
            if f.number == number:
                return f
        raise KeyError(f"frame {number} not in {self.path}")

    def particle_image(self, particle: Particle) -> np.ndarray:
        return particle.image(self._mm)

    def frame_image(self, number: int, fill: int = 0) -> np.ndarray:
        """Compose all particle images of a frame into one ``(height, width)`` u8 array."""
        f = self.frame(number)
        out = np.full((f.height, f.width), fill, np.uint8)
        for part in f.particles:
            xmin, ymin, xmax, ymax = part.bbox
            img = part.image(self._mm)
            mask = img != 0
            out[ymin:ymax, xmin:xmax][mask] = img[mask]
        return out

    def point_index(self, with_grains: bool = False) -> np.ndarray:
        """One :data:`INDEX_DTYPE` row per spectrum, in ``.xpoints`` file order.

        ``grain`` is the label at (x, y) in the particle's label image when
        ``with_grains`` is set (slower: every label image is decoded), else -1.
        """
        parts = []
        for f in self.frames:
            for part in f.particles:
                pts = part.points
                if not len(pts):
                    continue
                rows = np.zeros(len(pts), INDEX_DTYPE)
                for name in POINT_DTYPE.names:
                    rows[name] = pts[name]
                rows["frame"] = f.number
                rows["particle"] = part.pid
                rows["grain"] = -1
                if with_grains:
                    rows["grain"] = self._grains(part)
                parts.append(rows)
        index = np.concatenate(parts) if parts else np.zeros(0, INDEX_DTYPE)
        return index[np.argsort(index["offset"], kind="stable")]

    def _grains(self, part: Particle) -> np.ndarray:
        """Grain label under each spectrum of a particle; -1 where the point lies outside."""
        labels = part.image(self._mm)
        xmin, ymin = part.bbox[:2]
        dx = part.points["x"].astype(np.int64) - xmin
        dy = part.points["y"].astype(np.int64) - ymin
        inside = (dx >= 0) & (dx < labels.shape[1]) & (dy >= 0) & (dy < labels.shape[0])
        out = np.full(len(dx), -1, np.int16)
        out[inside] = labels[dy[inside], dx[inside]]
        return out

    def close(self) -> None:
        mm = getattr(self, "_mm", None)
        if mm is not None:
            mm.close()
        self._fh.close()

    def __enter__(self) -> "JkfFile":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def pixel_to_stage(frame: Frame, x, y, pixel_size_um: float = PIXEL_SIZE_UM):
    """Stage coordinates (um) of pixel centres in a frame.

    Image x runs along stage +X and image rows run along stage -Y (checked by edge
    continuity between neighbouring frames). The half-pixel origin is an assumption.
    """
    cx, cy = frame.center_um
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    return (cx + (x + 0.5 - frame.width / 2) * pixel_size_um, cy - (y + 0.5 - frame.height / 2) * pixel_size_um)
