from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from rst19.format_audit import run_format_audit, write_format_audit_artifacts


def _card(key: str, value: str) -> bytes:
    return f"{key:<8}= {value:>20}".ljust(80).encode("ascii")


def _make_fits(path: Path, *, timestamp: str | None = None) -> None:
    cards = [
        _card("SIMPLE", "T"),
        _card("BITPIX", "16"),
        _card("NAXIS", "2"),
        _card("NAXIS1", "16"),
        _card("NAXIS2", "16"),
        _card("BSCALE", "1"),
        _card("BZERO", "0"),
    ]
    if timestamp is not None:
        cards.append(_card("DATE-OBS", f"'{timestamp}'"))
    cards.append(b"END".ljust(80))
    header = b"".join(cards).ljust(2880, b" ")
    values = tuple(float(index) for index in range(26))
    auxiliary = bytearray(struct.pack("<26d", *values))
    for index in range(0, len(auxiliary), 2):
        auxiliary[index], auxiliary[index + 1] = auxiliary[index + 1], auxiliary[index]
    data = bytearray(np.arange(16 * 16, dtype=">i2").tobytes())
    data[: len(auxiliary)] = auxiliary
    path.write_bytes(header + data)


def test_format_audit_reports_actual_storage_and_writes_artifacts(tmp_path: Path) -> None:
    _make_fits(tmp_path / "b.fits")
    _make_fits(tmp_path / "a.fits")

    result = run_format_audit(tmp_path)

    assert result.summary["frame_count"] == 2
    assert result.summary["shape_counts"] == {"16x16": 2}
    assert result.summary["all_file_sizes_match_header"] is True
    assert result.summary["all_shapes_match_header"] is True
    assert result.summary["complete_auxiliary_frames"] == 2
    assert result.summary["frames_with_any_standard_wcs_card"] == 0
    assert result.summary["frames_with_minimal_wcs_core"] == 0
    assert all(row.wcs_standard_card_count == 0 for row in result.frame_rows)
    assert all("CRPIX1" in row.wcs_standard_cards_missing for row in result.frame_rows)
    assert all(row.exact_minus_one_count == 0 for row in result.frame_rows)
    assert all(row.unsigned_diagnostic_max is not None for row in result.frame_rows)

    output = write_format_audit_artifacts(result, tmp_path / "out")
    assert (output / "format_audit_frames.csv").is_file()
    assert (output / "format_audit_velocity.csv").is_file()
    assert (output / "format_audit.json").is_file()


def test_format_audit_rejects_invalid_fraction(tmp_path: Path) -> None:
    _make_fits(tmp_path / "frame.fits")
    with pytest.raises(ValueError, match="extreme_fraction"):
        run_format_audit(tmp_path, extreme_fraction=0.0)
