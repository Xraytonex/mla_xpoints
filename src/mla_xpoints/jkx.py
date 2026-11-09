"""Reader for MLA ``.jkx`` mineral lists (reference spectra used by the classifier).

Layout found in RODARE 4896 (see FORMAT.md)::

    0x00  "JKXF", u32 version (4)
    0x88  u32 number of entries
    0x90  u32 offset of the first entry
    entry +0x00  u32 version (4)
          +0x18  u32 n_channels (2000; 1 for the special classes Unknown/Low_Counts/No_XRay)
          +0x1C  char[32] X-ray reference name (e.g. "XRay0", "XRay0_0")
          +0x5C  u32 entry length in bytes
          +0x70  n_channels x u32 counts
          then   "XNFO" block; at XNFO+0x7A: u32 length + mineral name, u32 length + formula
"""

from __future__ import annotations

import os
import struct
from dataclasses import dataclass

import numpy as np

_ENCODING = "cp1252"


class JkxFormatError(ValueError):
    """Raised when a ``.jkx`` file does not match the decoded layout."""


@dataclass(frozen=True)
class Reference:
    mineral: str
    formula: str
    xray_name: str
    spectrum: np.ndarray  # uint32 counts; empty for the special classes


def _u32(buf: bytes, pos: int) -> int:
    if pos < 0 or pos + 4 > len(buf):
        raise JkxFormatError(f"byte {pos}: field past end of file")
    return struct.unpack_from("<I", buf, pos)[0]


def _text(raw: bytes) -> str:
    return raw.decode(_ENCODING, errors="replace")


def _string(buf: bytes, pos: int) -> tuple[str, int]:
    n = _u32(buf, pos)
    if n > 4096 or pos + 4 + n > len(buf):
        raise JkxFormatError(f"byte {pos}: implausible string length {n}")
    return _text(buf[pos + 4 : pos + 4 + n]), pos + 4 + n


def _entry(buf: bytes, pos: int) -> tuple[Reference, int]:
    nch = _u32(buf, pos + 0x18)
    xray_name = _text(buf[pos + 0x1C : pos + 0x3C].split(b"\0")[0])
    length = _u32(buf, pos + 0x5C)
    if length <= 0x70 or pos + length > len(buf):
        raise JkxFormatError(f"byte {pos}: bad entry length {length}")
    if nch <= 1:  # special classes carry no spectrum
        return Reference(xray_name, "", xray_name, np.zeros(0, np.uint32)), pos + length
    spec_end = pos + 0x70 + 4 * nch
    if spec_end + 4 > pos + length or buf[spec_end : spec_end + 4] != b"XNFO":
        raise JkxFormatError(f"byte {spec_end}: expected XNFO after {nch}-channel spectrum")
    spectrum = np.frombuffer(buf, "<u4", count=nch, offset=pos + 0x70).copy()
    mineral, p = _string(buf, spec_end + 0x7A)
    formula, _ = _string(buf, p)
    return Reference(mineral, formula, xray_name, spectrum), pos + length


def read_jkx(path: str | os.PathLike) -> list[Reference]:
    """Return every entry of a ``.jkx`` mineral list in file order."""
    with open(path, "rb") as fh:
        buf = fh.read()
    if buf[:4] != b"JKXF":
        raise JkxFormatError("missing 'JKXF' magic")
    count, first = _u32(buf, 0x88), _u32(buf, 0x90)
    if count > len(buf) // 0x70:
        raise JkxFormatError(f"entry count {count} cannot fit in {len(buf)} bytes")
    refs, pos = [], first
    for _ in range(count):
        ref, pos = _entry(buf, pos)
        refs.append(ref)
    return refs
