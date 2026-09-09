from __future__ import annotations

import csv

from rst19.feature_flag_profile import run_feature_flag_profile, write_feature_flag_profile_artifacts


def _write_catalog(path) -> None:
    rows = [
        {"detection_id": "82931", "feature_class": "range_anomaly", "quality_passed": "False", "flags": "CODE_PATTERN"},
        {"detection_id": "82934", "feature_class": "crowded_blend", "quality_passed": "False", "flags": "UNRESOLVED_BLEND|LOW_FLUX_SNR"},
        {"detection_id": "1", "feature_class": "crowded_blend", "quality_passed": "False", "flags": "UNRESOLVED_BLEND"},
        {"detection_id": "2", "feature_class": "compact_quality", "quality_passed": "True", "flags": ""},
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_flag_profile_preserves_overlapping_flags_and_targets(tmp_path) -> None:
    catalog = tmp_path / "source_catalog.csv"
    _write_catalog(catalog)
    result = run_feature_flag_profile(catalog, target_ids=("82931", "82934"))

    rows = {(row.feature_class, row.flag): row for row in result.rows}
    assert rows[("crowded_blend", "UNRESOLVED_BLEND")].flag_count == 2
    assert rows[("crowded_blend", "LOW_FLUX_SNR")].flag_count == 1
    assert rows[("compact_quality", "<none>")].flag_count == 1
    assert {row.detection_id for row in result.target_rows} == {"82931", "82934"}


def test_flag_profile_writes_artifacts(tmp_path) -> None:
    catalog = tmp_path / "source_catalog.csv"
    _write_catalog(catalog)
    output = write_feature_flag_profile_artifacts(
        run_feature_flag_profile(catalog),
        tmp_path / "out",
    )
    assert (output / "feature_flag_profile.csv").exists()
    assert (output / "feature_flag_profile_targets.csv").exists()
    assert (output / "feature_flag_profile.json").exists()
