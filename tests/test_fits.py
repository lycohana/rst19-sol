from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from rst19.fits import _parse_card_value, decode_auxiliary, exposure_milliseconds, exposure_seconds, read_fits


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


def test_exposure_seconds_prefers_standard_exptime_in_seconds() -> None:
    assert exposure_seconds({"EXPTIME": 1.5, "EXPOSURE": 1500}) == 1.5


def test_exposure_seconds_preserves_rst19_exposure_milliseconds() -> None:
    assert exposure_seconds({"EXPOSURE": 1500}) == 1.5
    assert exposure_seconds({"EXPOSURE": "1.5 s"}) == 1.5
    assert exposure_seconds({"EXPOSURE": 1500, "EXPOSURE_UNIT": "ms"}) == 1.5


def test_exposure_seconds_honours_explicit_unit_and_default() -> None:
    assert exposure_seconds({"EXPTIME": 1500, "EXPTIME_UNIT": "ms"}) == 1.5
    assert exposure_seconds({"EXPOSURE": 1.5, "TIMEUNIT": "s"}) == 1.5
    assert exposure_seconds({}) == 1.0
    assert exposure_milliseconds({}) is None
    assert exposure_milliseconds({"EXPTIME": 1.5}) == 1500.0
