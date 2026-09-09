from __future__ import annotations

import csv
import json

import pytest

from rst19.feature_evidence_axes import (
    run_feature_evidence_axes,
    write_feature_evidence_axes_artifacts,
)


_FIELDNAMES = [
    "feature_class",
    "feature_class_label",
    "candidate_count",
    "candidate_presence_ge_required_count",
    "quality_response_ge_required_count",
    "spatial_local_correlation_median",
    "spatial_local_residual_median",
]


def _write_matrix(path) -> None:
    rows = [
        {
            "feature_class": "compact_quality",
            "feature_class_label": "compact",
            "candidate_count": "100",
            "candidate_presence_ge_required_count": "95",
            "quality_response_ge_required_count": "85",
            "spatial_local_correlation_median": "0.84",
            "spatial_local_residual_median": "0.54",
        },
        {
            "feature_class": "crowded_blend",
            "feature_class_label": "blend",
            "candidate_count": "100",
            "candidate_presence_ge_required_count": "60",
            "quality_response_ge_required_count": "1",
            "spatial_local_correlation_median": "0.17",
            "spatial_local_residual_median": "0.98",
        },
        {
            "feature_class": "masked_or_edge",
            "feature_class_label": "masked",
            "candidate_count": "100",
            "candidate_presence_ge_required_count": "80",
            "quality_response_ge_required_count": "20",
            "spatial_local_correlation_median": "0.67",
            "spatial_local_residual_median": "0.74",
        },
        {
            "feature_class": "linear_artifact",
            "feature_class_label": "line",
            "candidate_count": "10",
            "candidate_presence_ge_required_count": "2",
            "quality_response_ge_required_count": "1",
            "spatial_local_correlation_median": "0.40",
            "spatial_local_residual_median": "0.91",
        },
        {
            "feature_class": "other_rejected",
            "feature_class_label": "other",
            "candidate_count": "0",
            "candidate_presence_ge_required_count": "0",
            "quality_response_ge_required_count": "0",
            "spatial_local_correlation_median": "",
            "spatial_local_residual_median": "",
        },
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def test_evidence_axes_classifies_patterns_and_keeps_q_given_p_descriptive(tmp_path) -> None:
    matrix = tmp_path / "feature_evidence_matrix.csv"
    _write_matrix(matrix)

    result = run_feature_evidence_axes(matrix)
    rows = {row.feature_class: row for row in result.rows}

    assert rows["compact_quality"].evidence_axis_pattern == "aligned_quality_psf"
    assert rows["compact_quality"].quality_response_given_candidate_persistent_fraction == pytest.approx(85 / 95)
    assert rows["crowded_blend"].evidence_axis_pattern == "persistent_without_quality"
    assert rows["crowded_blend"].quality_response_given_candidate_persistent_fraction == pytest.approx(1 / 60)
    assert rows["masked_or_edge"].evidence_axis_pattern == "location_persistent_quality_sparse"
    assert rows["linear_artifact"].evidence_axis_pattern == "sparse_or_structure_sensitive"
    assert rows["other_rejected"].evidence_axis_pattern == "no_sample"
    assert "逐源质量通过率" in rows["crowded_blend"].caution
    sensitivity = {(row.configuration, row.feature_class): row.evidence_axis_pattern for row in result.sensitivity_rows}
    assert sensitivity[("baseline", "compact_quality")] == "aligned_quality_psf"
    assert sensitivity[("strict_psf", "compact_quality")] == "location_persistent_quality_sparse"
    assert sensitivity[("strict_quality_gap", "crowded_blend")] == "persistent_without_quality"


def test_evidence_axes_rejects_inconsistent_counts(tmp_path) -> None:
    matrix = tmp_path / "feature_evidence_matrix.csv"
    _write_matrix(matrix)
    text = matrix.read_text(encoding="utf-8-sig").replace(",60,1,0.17", ",60,101,0.17")
    matrix.write_text(text, encoding="utf-8-sig")

    with pytest.raises(ValueError, match="quality response count exceeds"):
        run_feature_evidence_axes(matrix)


def test_evidence_axes_writes_csv_and_json(tmp_path) -> None:
    matrix = tmp_path / "feature_evidence_matrix.csv"
    _write_matrix(matrix)
    output = write_feature_evidence_axes_artifacts(
        run_feature_evidence_axes(matrix),
        tmp_path / "out",
    )

    assert (output / "feature_evidence_axes.csv").is_file()
    assert (output / "feature_evidence_axes_sensitivity.csv").is_file()
    assert (output / "feature_evidence_axes.json").is_file()
    payload = json.loads((output / "feature_evidence_axes.json").read_text(encoding="utf-8"))
    assert payload["rows"][0]["feature_class"] == "compact_quality"
