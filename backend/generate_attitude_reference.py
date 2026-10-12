"""Freeze auxiliary-boresight audit cases from Python reference FITS files."""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import struct
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rst19.attitude import audit_auxiliary_boresight, summarize_auxiliary_boresight
from rst19.fits import read_fits


def card(key: str, value: str) -> bytes:
    return f"{key:<8}= {value:>20}".ljust(80).encode("ascii")


def make_fits(path: Path, values: list[float] | None, *, header_azimuth: str | None = "270.0", header_elevation: str | None = "0.0") -> None:
    cards = [
        card("SIMPLE", "T"),
        card("BITPIX", "16"),
        card("NAXIS", "2"),
        card("NAXIS1", "16"),
        card("NAXIS2", "16"),
    ]
    if header_azimuth is not None:
        cards.append(card("AZIMUTH", header_azimuth))
    if header_elevation is not None:
        cards.append(card("ELEVATIO", header_elevation))
    cards.extend([card("DATE-OBS", "'2026-03-30T16:32:05.4132'"), b"END".ljust(80)])
    header = b"".join(cards).ljust(2880, b" ")
    data = bytearray(np.zeros(16 * 16, dtype=">i2").tobytes())
    if values is not None:
        auxiliary = bytearray(struct.pack("<26d", *values))
        for index in range(0, len(auxiliary), 2):
            auxiliary[index], auxiliary[index + 1] = auxiliary[index + 1], auxiliary[index]
        data[: len(auxiliary)] = auxiliary
    path.write_bytes(header + data)


def main() -> None:
    quaternion = [0.0, 0.0, 2.0**-0.5, 2.0**-0.5]
    first = [0.0] * 26
    first[0:4] = [270.0, 0.0, 0.0, 0.0]
    first[10:14] = quaternion
    first[14:17] = [1.0, -2.0, 3.5]
    wrap = first.copy()
    wrap[2:4] = [359.99995, -0.0001]
    wrap[14:17] = [-4.0, 2.0, 8.0]
    specs = [
        {"path": "aligned.fits", "values": first, "header_azimuth": "270.0", "header_elevation": "0.0"},
        {"path": "ra-wrap.fits", "values": wrap, "header_azimuth": "270.0", "header_elevation": None},
        {"path": "missing-aux.fits", "values": None, "header_azimuth": "bad", "header_elevation": "-5.5"},
    ]
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        records = []
        paths = []
        for spec in specs:
            path = root / spec["path"]
            make_fits(path, spec["values"], header_azimuth=spec["header_azimuth"], header_elevation=spec["header_elevation"])
            paths.append(path)
            frame = read_fits(path)
            records.append({
                "path": spec["path"],
                "header": frame.header,
                "auxiliary_values": None if frame.auxiliary is None else list(frame.auxiliary.values),
            })
        default_axis_rows = tuple(
            replace(row, path=spec["path"])
            for row, spec in zip(audit_auxiliary_boresight(paths), specs, strict=True)
        )
        default_case = {
            "label": "default_axis_wrap_headers_and_missing_auxiliary",
            "frames": records,
            "camera_body_axis": None,
            "expected": {
                "schema_version": 1,
                "summary": summarize_auxiliary_boresight(default_axis_rows),
                "frames": [row.as_dict() for row in default_axis_rows],
            },
        }
        alternate_axis = [1.0, 0.0, 0.0]
        alternate_rows = tuple(
            replace(row, path=spec["path"])
            for row, spec in zip(audit_auxiliary_boresight(paths[:1], camera_body_axis=alternate_axis), specs[:1], strict=True)
        )
        alternate_case = {
            "label": "alternate_camera_axis",
            "frames": records[:1],
            "camera_body_axis": alternate_axis,
            "expected": {
                "schema_version": 1,
                "summary": summarize_auxiliary_boresight(alternate_rows, camera_body_axis=alternate_axis),
                "frames": [row.as_dict() for row in alternate_rows],
            },
        }
    source_hashes = {
        name: hashlib.sha256((ROOT / "src/rst19" / name).read_bytes()).hexdigest()
        for name in ("attitude.py", "fits.py")
    }
    target = ROOT / "backend/tests/fixtures/attitude.json"
    target.write_text(
        json.dumps({"source_hashes": source_hashes, "cases": [default_case, alternate_case]}, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(target)


if __name__ == "__main__":
    main()
