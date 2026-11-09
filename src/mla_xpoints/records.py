"""Reader for MLA ``.xpoints`` raw X-ray spectrum files.

A ``.xpoints`` file is a plain concatenation of records with no file header::

    record  = 80-byte header + payload
    payload = n_channels counts, uint8 (payload_bytes == n_channels)
              or little-endian uint16 (payload_bytes == 2 * n_channels)

All header fields are little-endian. See ``FORMAT.md`` for the field table and the
evidence behind each field. Bytes not listed in :data:`HEADER_DTYPE` are either zero
or hold uninitialised memory (struct padding); readers must ignore them.
"""

from __future__ import annotations

import mmap
import os
from dataclasses import dataclass
from typing import Iterator

import numpy as np

HEADER_SIZE = 80
RECORD_TAG = 0x5A
_TAG_OFFSET = 0
_N_CHANNELS_OFFSET = 70
_PAYLOAD_BYTES_OFFSET = 76

HEADER_DTYPE = np.dtype(
    {
        "names": [
            "tag",
            "version_a",
            "version_b",
            "x",
            "y",
            "width",
            "height",
            "unknown_44",
            "bse_mode",
            "bse_mode_fraction",
            "n_channels",
            "payload_bytes",
        ],
        "formats": ["u1", "<u2", "<u2", "<u2", "<u2", "<u2", "<u2", "<u2", "<f4", "<f4", "<u2", "<u4"],
        "offsets": [0, 2, 4, 12, 14, 40, 42, 44, 48, 52, 70, 76],
        "itemsize": HEADER_SIZE,
    }
)
"""Decoded header fields (see FORMAT.md):

tag                 always 0x5A
version_a/b         always 3 and 2 in the dataset studied (meaning unknown)
x, y                pixel position of the spectrum in its 500x500 frame
width, height       side of the area the spectrum stands for: 1 (single spot) or the
                    GXMAP step size (10)
unknown_44          always 1 in the dataset studied
bse_mode            most frequent BSE grey level of the grain the spectrum belongs to
bse_mode_fraction   fraction of grain pixels at that grey level, truncated to 3 decimals
n_channels          number of channels (2000)
payload_bytes       n_channels (uint8 counts) or 2 * n_channels (uint16 counts)
"""


class XPointsFormatError(ValueError):
    """Raised when a file does not follow the ``.xpoints`` record layout."""


@dataclass(frozen=True)
class RecordTable:
    """Byte offset and payload size of every record in a file."""

    offsets: np.ndarray  # int64, byte offset of each record header
    payload_bytes: np.ndarray  # uint32
    n_channels: int

    @property
    def is_uint16(self) -> np.ndarray:
        return self.payload_bytes == 2 * self.n_channels

    def __len__(self) -> int:
        return len(self.offsets)


def _u16(buf, off: int) -> int:
    return int.from_bytes(buf[off : off + 2], "little")


def _u32(buf, off: int) -> int:
    return int.from_bytes(buf[off : off + 4], "little")


def _check_header(buf, off: int, index: int, size: int) -> tuple[int, int]:
    if off + HEADER_SIZE > size:
        raise XPointsFormatError(f"record {index} at byte {off}: truncated header ({size - off} bytes left)")
    if buf[off + _TAG_OFFSET] != RECORD_TAG:
        raise XPointsFormatError(f"record {index} at byte {off}: tag {buf[off]:#04x}, expected {RECORD_TAG:#04x}")
    nch = _u16(buf, off + _N_CHANNELS_OFFSET)
    plen = _u32(buf, off + _PAYLOAD_BYTES_OFFSET)
    if nch == 0 or plen not in (nch, 2 * nch):
        raise XPointsFormatError(
            f"record {index} at byte {off}: payload_bytes={plen} does not match n_channels={nch}"
        )
    if off + HEADER_SIZE + plen > size:
        raise XPointsFormatError(f"record {index} at byte {off}: payload runs past end of file")
    return nch, plen


def _looks_uniform(buf, size: int, nch: int, stride: int, samples: int = 64) -> bool:
    """Cheap pre-check: are evenly spaced records valid uint8 headers on the fixed stride?"""
    n = size // stride
    for i in np.unique(np.linspace(0, n - 1, min(n, samples)).astype(np.int64)).tolist():
        off = i * stride
        if (
            buf[off + _TAG_OFFSET] != RECORD_TAG
            or _u16(buf, off + _N_CHANNELS_OFFSET) != nch
            or _u32(buf, off + _PAYLOAD_BYTES_OFFSET) != nch
        ):
            return False
    return True


def _uniform_table(buf, size: int, nch: int) -> RecordTable | None:
    """Fast path: every record has uint8 counts, so records sit on a fixed stride."""
    stride = HEADER_SIZE + nch
    if size % stride or not _looks_uniform(buf, size, nch, stride):
        return None
    n = size // stride
    view = np.ndarray((n, stride), np.uint8, buffer=buf)
    plen = view[:, _PAYLOAD_BYTES_OFFSET : _PAYLOAD_BYTES_OFFSET + 4].copy().view("<u4").ravel()
    chan = view[:, _N_CHANNELS_OFFSET : _N_CHANNELS_OFFSET + 2].copy().view("<u2").ravel()
    if not (np.all(view[:, _TAG_OFFSET] == RECORD_TAG) and np.all(plen == nch) and np.all(chan == nch)):
        return None
    return RecordTable(np.arange(n, dtype=np.int64) * stride, plen.astype(np.uint32), nch)


def scan_records(buf, size: int | None = None) -> RecordTable:
    """Walk the record chain of an ``.xpoints`` buffer and validate every header.

    ``buf`` is any object supporting the buffer protocol (bytes, mmap, memoryview).
    Raises :class:`XPointsFormatError` on the first malformed record.
    """
    size = len(buf) if size is None else size
    if size == 0:
        return RecordTable(np.zeros(0, np.int64), np.zeros(0, np.uint32), 0)
    nch, _ = _check_header(buf, 0, 0, size)
    fast = _uniform_table(buf, size, nch)
    if fast is not None:
        return fast
    offsets, plens = [], []
    off, index = 0, 0
    while off < size:
        rec_nch, plen = _check_header(buf, off, index, size)
        if rec_nch != nch:
            raise XPointsFormatError(f"record {index} at byte {off}: n_channels={rec_nch}, file uses {nch}")
        offsets.append(off)
        plens.append(plen)
        off += HEADER_SIZE + plen
        index += 1
    return RecordTable(np.asarray(offsets, np.int64), np.asarray(plens, np.uint32), nch)


def _gather_rows(src: np.ndarray, starts: np.ndarray, width: int) -> np.ndarray:
    """Copy ``width`` bytes from each start offset into a ``(len(starts), width)`` array.

    Equally spaced starts (the common all-uint8 case) are copied through one strided view;
    anything else falls back to a per-row copy. No large index arrays are built.
    """
    out = np.empty((len(starts), width), np.uint8)
    if len(starts) == 0:
        return out
    if starts.min() < 0 or starts.max() + width > len(src):
        raise ValueError("record table does not fit the buffer")
    steps = np.diff(starts)
    if len(starts) == 1 or (steps[0] > 0 and np.all(steps == steps[0])):
        step = int(steps[0]) if len(starts) > 1 else width
        base = src[int(starts[0]) :]
        out[:] = np.lib.stride_tricks.as_strided(base, (len(starts), width), (step, 1), writeable=False)
        return out
    for row, start in enumerate(starts.tolist()):
        out[row] = src[start : start + width]
    return out


def headers_from_buffer(buf, table: RecordTable) -> np.ndarray:
    """Return a structured array (dtype :data:`HEADER_DTYPE`) with one row per record."""
    src = np.frombuffer(buf, np.uint8)
    return _gather_rows(src, table.offsets, HEADER_SIZE).view(HEADER_DTYPE).ravel()


def spectra_from_buffer(buf, table: RecordTable, indices: np.ndarray) -> np.ndarray:
    """Return counts for the given record indices as a ``(k, n_channels)`` uint16 array."""
    indices = np.asarray(indices)
    if indices.dtype == bool:
        if indices.shape != (len(table),):
            raise IndexError(f"boolean mask has shape {indices.shape}, expected ({len(table)},)")
        indices = np.flatnonzero(indices)
    elif indices.size and not np.issubdtype(indices.dtype, np.integer):
        raise IndexError(f"record indices must be integers or a boolean mask, got {indices.dtype}")
    indices = indices.astype(np.int64).ravel()
    if len(indices) and (indices.min() < 0 or indices.max() >= len(table)):
        raise IndexError(f"record index out of range 0..{len(table) - 1}")
    nch = table.n_channels
    out = np.empty((len(indices), nch), np.uint16)
    src = np.frombuffer(buf, np.uint8)
    starts = table.offsets[indices] + HEADER_SIZE
    wide = table.is_uint16[indices]
    narrow_rows = np.flatnonzero(~wide)
    if len(narrow_rows):
        out[narrow_rows] = _gather_rows(src, starts[narrow_rows], nch)
    wide_rows = np.flatnonzero(wide)
    if len(wide_rows):
        out[wide_rows] = _gather_rows(src, starts[wide_rows], 2 * nch).view("<u2")
    return out


class XPointsFile:
    """Memory-mapped ``.xpoints`` file.

    >>> with XPointsFile("3_test_X-Ray.xpoints") as xp:
    ...     xp.headers["x"][:3], xp.spectra([0, 1]).shape
    """

    def __init__(self, path: str | os.PathLike):
        self.path = os.fspath(path)
        self._fh = open(self.path, "rb")
        self._mm = b""
        try:
            self.size = os.fstat(self._fh.fileno()).st_size
            if self.size:
                self._mm = mmap.mmap(self._fh.fileno(), 0, access=mmap.ACCESS_READ)
            self.table = scan_records(self._mm, self.size)
            self.headers = headers_from_buffer(self._mm, self.table)
        except Exception:
            self.close()
            raise

    @property
    def n_channels(self) -> int:
        return self.table.n_channels

    def __len__(self) -> int:
        return len(self.table)

    def spectrum(self, index: int) -> np.ndarray:
        return self.spectra([index])[0]

    def spectra(self, indices=None) -> np.ndarray:
        """Counts for ``indices`` (all records if None) as a ``(k, n_channels)`` uint16 array."""
        if indices is None:
            indices = np.arange(len(self))
        elif isinstance(indices, slice):
            indices = np.arange(len(self))[indices]
        return spectra_from_buffer(self._mm, self.table, np.atleast_1d(np.asarray(indices)))

    def iter_chunks(self, chunk: int = 50_000) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield ``(record_indices, spectra)`` in chunks so large files fit in memory."""
        for start in range(0, len(self), chunk):
            idx = np.arange(start, min(start + chunk, len(self)))
            yield idx, self.spectra(idx)

    def sum_spectrum(self, chunk: int = 50_000) -> np.ndarray:
        total = np.zeros(self.n_channels, np.int64)
        for _, block in self.iter_chunks(chunk):
            total += block.sum(axis=0, dtype=np.int64)
        return total

    def total_counts(self, chunk: int = 50_000) -> np.ndarray:
        out = np.empty(len(self), np.int64)
        for idx, block in self.iter_chunks(chunk):
            out[idx] = block.sum(axis=1, dtype=np.int64)
        return out

    def close(self) -> None:
        if isinstance(self._mm, mmap.mmap):
            self._mm.close()
        self._fh.close()

    def __enter__(self) -> "XPointsFile":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def read_xpoints(path: str | os.PathLike, start: int = 0, stop: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Read headers and counts of records ``start:stop`` into memory.

    Returns ``(headers, spectra)``: a structured array with :data:`HEADER_DTYPE` and a
    ``(k, n_channels)`` uint16 array. A full-size GXMAP file (~1.2 million spectra) needs
    about 5 GB as uint16, so use :class:`XPointsFile` and ``iter_chunks`` for those.
    """
    with XPointsFile(path) as xp:
        idx = np.arange(len(xp))[start:stop]
        return xp.headers[idx].copy(), xp.spectra(idx)
