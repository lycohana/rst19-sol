from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from rst19.fits import _parse_card_value, decode_auxiliary, read_fits


def _card(key: str, value: str) -> bytes:
    return f"{key:<8}= {value:>20}".ljust(80).encode("ascii")


def _make_fits(path: Path, *, width: int = 16, height: int = 16) -> tuple[float, ...]:
    header = b"".join(
        [
            _card("SIMPLE", "T"),
            _card("BITPIX", "16"),
            _card("NAXIS", "2"),
            _card("NAXIS1", str(width)),
            _card("NAXIS2", str(height)),
            b"END".ljust(80),
        ]
    )
    header = header.ljust(2880, b" ")
    values = tuple(float(index) for index in range(26))
    aux = bytearray(struct.pack("<26d", *values))
    for index in range(0, len(aux), 2):
        aux[index], aux[index + 1] = aux[index + 1], aux[index]
    data = bytearray(np.arange(width * height, dtype=">i2").tobytes())
    data[: len(aux)] = aux
    path.write_bytes(header + data)
    return values


def test_read_fits_decodes_shape_and_auxiliary(tmp_path: Path) -> None:
    path = tmp_path / "frame.fits"
    expected_values = _make_fits(path)

    frame = read_fits(path)

    assert frame.data.shape == (16, 16)
    assert frame.data.dtype == np.dtype(">i2")
    assert frame.data_offset == 2880
    assert frame.auxiliary is not None
    assert frame.auxiliary.values == expected_values
    assert frame.auxiliary.as_dict()["ra"] == 2.0


def test_decode_auxiliary_rejects_wrong_length() -> None:
    try:
        decode_auxiliary(b"short")
    except ValueError as exc:
        assert "208" in str(exc)
    else:
        raise AssertionError("decode_auxiliary should reject a short buffer")


def test_parse_card_value_handles_current_numeric_card_variant() -> None:
    assert _parse_card_value("0000          01500") == 1500
    assert _parse_card_value("          0270.00000 0000.00000") == 270.0
