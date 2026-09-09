from __future__ import annotations

import csv

import pytest

from rst19.feature_raw_evidence_audit import (
    RAW_EVIDENCE_METRICS,
    run_feature_raw_evidence_audit,
    write_feature_raw_evidence_artifacts,
)


def _write_csv(path, fields, rows) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_inputs(tmp_path):
    catalog = tmp_path / "source_catalog.csv"
    _write_csv(
        catalog,
        ["detection_id", "feature_class", "feature_class_label"],
        [
            {"detection_id": "1", "feature_class": "compact_quality", "feature_class_label": "紧凑"},
            {"detection_id": "2", "feature_class": "compact_quality", "feature_class_label": "紧凑"},
            {"detection_id": "3", "feature_class": "linear_artifact", "feature_class_label": "线状"},
            {"detection_id": "4", "feature_class": "linear_artifact", "feature_class_label": "线状"},
            {"detection_id": "5", "feature_class": "weak_or_background", "feature_class_label": "弱"},
        ],
    )
    raw_summary = tmp_path / "stratified_raw_source_summary.csv"
    fields = ["detection_id", "origin", *RAW_EVIDENCE_METRICS]
    def row(detection_id: int, origin: str, core: float, noise: float, negative: int) -> dict[str, str]:
        values = {
            "detection_id": str(detection_id),
            "origin": origin,
            "core_fraction_median": str(core),
            "noise_median": str(noise),
            "zero_count_max": "0",
            "frames_zero_count_gt0": "0",
            "frames_negative_extreme_count_gt0": str(negative),
            "frames_repeated_code_count_gt0": "0",
            "frames_peak_snr_ge5": "15",
            "frames_flux_snr_ge5": "15",
            "frames_support_ge3": "15",
        }
        return values
    _write_csv(
        raw_summary,
        fields,
        [
            row(1, "control", 0.8, 8.0, 0),
            row(2, "control", 0.7, 10.0, 0),
            row(3, "diagnostic", 0.2, 12.0, 3),
            row(4, "diagnostic", 0.4, 14.0, 5),
            row(5, "diagnostic", 0.5, 9.0, 0),
        ],
    )
    return catalog, raw_summary


def test_raw_evidence_joins_classes_and_summarizes_axes(tmp_path) -> None:
    catalog, raw_summary = _write_inputs(tmp_path)
    result = run_feature_raw_evidence_audit(catalog, raw_summary)

    assert result.feature_classes == ("compact_quality", "linear_artifact", "weak_or_background")
    assert result.sample_counts == {
        "compact_quality": 2,
        "linear_artifact": 2,
        "weak_or_background": 1,
    }
    assert result.origin_counts["compact_quality"] == {"control": 2}
    core = next(
        row
        for row in result.rows
        if row.feature_class == "compact_quality" and row.metric == "core_fraction_median"
    )
    assert core.median == pytest.approx(0.75)
    assert core.p10 == pytest.approx(0.71)
    assert core.p90 == pytest.approx(0.79)
    assert core.definition_overlap.startswith("间接相关")
    negative = next(
        row
        for row in result.rows
        if row.feature_class == "linear_artifact" and row.metric == "frames_negative_extreme_count_gt0"
    )
    assert negative.median == pytest.approx(4.0)
    assert negative.nonzero_fraction == 1.0

    output = write_feature_raw_evidence_artifacts(result, tmp_path / "out")
    assert (output / "feature_raw_evidence_audit.json").is_file()
    assert (output / "feature_raw_evidence_summary.csv").is_file()


def test_raw_evidence_rejects_missing_or_duplicate_ids(tmp_path) -> None:
    catalog, raw_summary = _write_inputs(tmp_path)
    fields = ["detection_id", "origin", *RAW_EVIDENCE_METRICS]
    rows = []
    for _ in range(2):
        values = {field: "0" for field in fields}
        values.update({"detection_id": "999", "origin": "diagnostic"})
        rows.append(values)
    _write_csv(raw_summary, fields, rows)
    with pytest.raises(ValueError, match="absent from source catalog"):
        run_feature_raw_evidence_audit(catalog, raw_summary)

    catalog, raw_summary = _write_inputs(tmp_path)
    with raw_summary.open("a", encoding="utf-8", newline="") as stream:
        stream.write("1,control," + ",".join("0" for _ in RAW_EVIDENCE_METRICS) + "\n")
    with pytest.raises(ValueError, match="duplicate detection_id=1"):
        run_feature_raw_evidence_audit(catalog, raw_summary)
