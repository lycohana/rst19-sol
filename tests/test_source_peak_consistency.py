from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from rst19.models import FitsFrame
from rst19.source_peak_consistency import (
    run_source_peak_consistency,
    write_source_peak_consistency_artifacts,
)


def _write_catalog(path: Path) -> None:
    fields = (
        "detection_id",
        "feature_class",
        "feature_class_label",
        "quality_passed",
        "peak_x",
        "peak_y",
    )
    rows = (
        (1, "compact_quality", "compact", "True", 5, 5),
        (2, "range_anomaly", "range", "False", 8, 5),
        (3, "crowded_blend", "blend", "False", 12, 5),
    )
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(fields)
        writer.writerows(rows)


def _frame(tmp_path: Path) -> FitsFrame:
    image = np.full((20, 20), 10.0, dtype=np.float32)
    image[5, 5] = 100.0
    # Candidate 2 is not the raw local maximum: a brighter adjacent pixel wins.
    image[5, 7] = 30.0
    image[5, 8] = 20.0
    image[5, 9] = 90.0
    # Candidate 3 is a unique raw local maximum.
    image[5, 12] = 80.0
    return FitsFrame(path=tmp_path / "synthetic.fits", header={}, data=image, auxiliary=None, data_offset=0)


def test_source_peak_consistency_reports_category_and_target_gap(tmp_path: Path, monkeypatch) -> None:
    catalog = tmp_path / "source_catalog.csv"
    _write_catalog(catalog)
    frame = _frame(tmp_path)
    monkeypatch.setattr("rst19.source_peak_consistency.read_fits", lambda _path: frame)

    result = run_source_peak_consistency(catalog, frame.path, target_ids=(2, 3))
    rows = {row.detection_id: row for row in result.rows}
    assert rows[1].raw_local_maximum_r1 is True
    assert rows[2].raw_local_maximum_r1 is False
    assert rows[2].raw_local_max_gap_adu_r1 == 70.0
    assert rows[3].raw_unique_local_maximum_r1 is True

    summaries = {row.feature_class: row for row in result.class_summaries}
    assert summaries["range_anomaly"].nonlocal_peak_count_r1 == 1
    assert summaries["range_anomaly"].nonlocal_peak_gap_median_adu_r1 == 70.0
    assert len(result.targets) == 6
    target = next(row for row in result.targets if row.detection_id == 2 and row.radius_px == 1)
    assert (target.local_max_x, target.local_max_y) == (9, 5)
    assert target.local_max_gap_adu == 70.0


def test_source_peak_consistency_writes_machine_artifacts(tmp_path: Path, monkeypatch) -> None:
    catalog = tmp_path / "source_catalog.csv"
    _write_catalog(catalog)
    frame = _frame(tmp_path)
    monkeypatch.setattr("rst19.source_peak_consistency.read_fits", lambda _path: frame)
    result = run_source_peak_consistency(catalog, frame.path, target_ids=(2,))

    output = write_source_peak_consistency_artifacts(result, tmp_path / "out")
    assert (output / "source_peak_consistency.csv").is_file()
    assert (output / "source_peak_consistency_class_summary.csv").is_file()
    assert (output / "source_peak_consistency_targets.csv").is_file()
    payload = json.loads((output / "source_peak_consistency.json").read_text(encoding="utf-8"))
    assert payload["targets"][0]["detection_id"] == 2
    assert "充分条件" in payload["interpretation_boundary"]


def test_source_peak_consistency_rejects_non_stable_radius_set(tmp_path: Path, monkeypatch) -> None:
    catalog = tmp_path / "source_catalog.csv"
    _write_catalog(catalog)
    frame = _frame(tmp_path)
    monkeypatch.setattr("rst19.source_peak_consistency.read_fits", lambda _path: frame)

    try:
        run_source_peak_consistency(catalog, frame.path, local_max_radii_px=(1, 2))
    except ValueError as exc:
        assert "stable radius set" in str(exc)
    else:
        raise AssertionError("expected non-stable radius set to be rejected")
