from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from rst19.attitude import (
    audit_auxiliary_boresight,
    auxiliary_boresight_radec,
    quaternion_to_active_matrix,
    summarize_auxiliary_boresight,
    vector_to_radec_deg,
    write_auxiliary_boresight_artifacts,
)
from rst19.models import AuxiliaryData


def _card(key: str, value: str) -> bytes:
    return f"{key:<8}= {value:>20}".ljust(80).encode("ascii")


def _make_fits(path: Path) -> None:
    header = b"".join(
        [
            _card("SIMPLE", "T"),
            _card("BITPIX", "16"),
            _card("NAXIS", "2"),
            _card("NAXIS1", "16"),
            _card("NAXIS2", "16"),
            _card("AZIMUTH", "270.0"),
            _card("ELEVATIO", "0.0"),
            _card("DATE-OBS", "'2026-03-30T16:32:05.4132'"),
            b"END".ljust(80),
        ]
    ).ljust(2880, b" ")
    values = [0.0] * 26
    values[0] = 270.0
    values[1] = 0.0
    values[2] = 0.0
    values[3] = 0.0
    values[12] = 2.0**-0.5
    values[13] = 2.0**-0.5
    aux = bytearray(struct.pack("<26d", *values))
    for index in range(0, len(aux), 2):
        aux[index], aux[index + 1] = aux[index + 1], aux[index]
    data = bytearray(np.zeros(16 * 16, dtype=">i2").tobytes())
    data[: len(aux)] = aux
    path.write_bytes(header + data)


def test_quaternion_axis_projection_matches_expected_radec() -> None:
    angle = np.pi / 2.0
    quaternion = (0.0, 0.0, float(np.sin(angle / 2.0)), float(np.cos(angle / 2.0)))
    matrix = quaternion_to_active_matrix(quaternion)
    assert np.allclose(matrix @ np.array((0.0, -1.0, 0.0)), np.array((1.0, 0.0, 0.0)))
    assert vector_to_radec_deg((1.0, 0.0, 0.0)) == (0.0, 0.0)
    values = [0.0] * 26
    values[10:14] = quaternion
    auxiliary = AuxiliaryData.from_values(values)
    assert np.allclose(auxiliary_boresight_radec(auxiliary), (0.0, 0.0), atol=1e-12)


def test_auxiliary_audit_checks_boresight_and_header(tmp_path: Path) -> None:
    path = tmp_path / "frame.fits"
    _make_fits(path)

    rows = audit_auxiliary_boresight([path])

    assert len(rows) == 1
    row = rows[0]
    assert row.status == "ok"
    assert row.header_pointing_match is True
    assert row.quaternion_norm == 1.0
    assert row.angular_residual_arcsec is not None
    assert row.angular_residual_arcsec < 1e-8
    summary = summarize_auxiliary_boresight(rows)
    assert summary["ok_frame_count"] == 1
    assert summary["header_pointing_match_count"] == 1

    output = write_auxiliary_boresight_artifacts(rows, tmp_path / "audit")
    assert (output / "auxiliary_boresight_audit.csv").is_file()
    assert (output / "auxiliary_boresight_audit.json").is_file()
