from __future__ import annotations

import csv
import json

import pytest

from rst19.feature_gate_temporal_route import (
    run_feature_gate_temporal_route,
    write_feature_gate_temporal_route_artifacts,
)


def _write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_inputs(tmp_path):
    route = tmp_path / "route.csv"
    _write_csv(
        route,
        [
            {
                "detection_id": "1",
                "feature_class": "compact_quality",
                "feature_class_label": "紧凑",
                "quality_passed": "True",
                "route": "quality_passed",
                "numeric_failure_reasons": "",
                "structural_flags": "",
                "flags": "",
            },
            {
                "detection_id": "2",
                "feature_class": "crowded_blend",
                "feature_class_label": "拥挤",
                "quality_passed": "False",
                "route": "rejected_structural_only",
                "numeric_failure_reasons": "",
                "structural_flags": "UNRESOLVED_BLEND",
                "flags": "UNRESOLVED_BLEND",
            },
            {
                "detection_id": "3",
                "feature_class": "weak_or_background",
                "feature_class_label": "弱",
                "quality_passed": "False",
                "route": "rejected_numeric_only",
                "numeric_failure_reasons": "flux_snr",
                "structural_flags": "",
                "flags": "LOW_FLUX_SNR",
            },
        ],
    )
    diagnostic = tmp_path / "diagnostic.csv"
    _write_csv(
        diagnostic,
        [
            {
                "detection_id": "2",
                "feature_class": "crowded_blend",
                "feature_class_label": "拥挤",
                "diagnostic_subgroup": "blend_unresolved",
                "diagnostic_subgroup_label": "拥挤",
                "diagnostic_subgroup_definition": "未解析混合",
                "candidate_presence": "12",
                "candidate_same_subgroup_presence": "11",
                "quality_presence": "0",
                "quality_same_subgroup_presence": "0",
            },
            {
                "detection_id": "3",
                "feature_class": "weak_or_background",
                "feature_class_label": "弱",
                "diagnostic_subgroup": "weak_low_flux_snr",
                "diagnostic_subgroup_label": "弱",
                "diagnostic_subgroup_definition": "低通量信噪比",
                "candidate_presence": "15",
                "candidate_same_subgroup_presence": "15",
                "quality_presence": "2",
                "quality_same_subgroup_presence": "1",
            },
        ],
    )
    persistence = tmp_path / "persistence.json"
    persistence.write_text(
        json.dumps(
            {
                "frame_count": 15,
                "required_presence": 12,
                "association_radius_px": 1.0,
                "association_method": "reciprocal_nearest_one_to_one",
            }
        ),
        encoding="utf-8",
    )
    return route, diagnostic, persistence


def test_temporal_route_preserves_missing_quality_source_as_unavailable(tmp_path) -> None:
    route, diagnostic, persistence = _write_inputs(tmp_path)
    result = run_feature_gate_temporal_route(route, diagnostic, persistence)
    sources = {row.detection_id: row for row in result.source_rows}
    assert sources["1"].temporal_source_covered is False
    assert sources["1"].candidate_presence is None
    assert sources["2"].candidate_presence == 12
    assert sources["3"].quality_presence == 2

    crowded = next(row for row in result.summary_rows if row.feature_class == "crowded_blend")
    assert crowded.temporal_source_count == 1
    assert crowded.candidate_presence_ge_required_count == 1
    assert crowded.candidate_same_subgroup_ge_required_count == 0
    crowded_subgroup = next(
        row
        for row in result.subgroup_summary_rows
        if row.diagnostic_subgroup == "blend_unresolved"
    )
    assert crowded_subgroup.subgroup_count == 1
    assert crowded_subgroup.route_rejected_structural_only_count == 1
    assert crowded_subgroup.candidate_presence_ge_required_count == 1
    assert crowded_subgroup.candidate_same_subgroup_ge_required_count == 0
    assert crowded_subgroup.mechanism_gap_fraction == 1.0
    compact = next(row for row in result.summary_rows if row.feature_class == "compact_quality")
    assert compact.temporal_source_count == 0
    assert compact.candidate_presence_ge_required_count is None


def test_temporal_route_requires_rejected_candidates_to_be_covered(tmp_path) -> None:
    route, diagnostic, persistence = _write_inputs(tmp_path)
    with diagnostic.open("r", encoding="utf-8-sig", newline="") as stream:
        diagnostic_rows = list(csv.DictReader(stream))
    _write_csv(diagnostic, diagnostic_rows[:1])
    with pytest.raises(ValueError, match="missing temporal row"):
        run_feature_gate_temporal_route(route, diagnostic, persistence)


def test_temporal_route_can_profile_requested_targets_within_subgroup(tmp_path) -> None:
    route, diagnostic, persistence = _write_inputs(tmp_path)
    result = run_feature_gate_temporal_route(
        route,
        diagnostic,
        persistence,
        target_ids=("2", "3"),
    )
    targets = {row.detection_id: row for row in result.target_rows}
    assert targets["2"].diagnostic_subgroup == "blend_unresolved"
    assert targets["2"].subgroup_count == 1
    assert targets["2"].candidate_presence_le_fraction == 1.0
    assert targets["2"].candidate_presence_ge_fraction == 1.0
    assert targets["2"].all_three_ge_required is False
    assert targets["3"].candidate_same_subgroup_presence == 15
    assert targets["3"].candidate_same_subgroup_presence_le_fraction == 1.0


def test_temporal_route_writes_artifacts(tmp_path) -> None:
    route, diagnostic, persistence = _write_inputs(tmp_path)
    output = write_feature_gate_temporal_route_artifacts(
        run_feature_gate_temporal_route(route, diagnostic, persistence),
        tmp_path / "out",
    )
    assert (output / "feature_gate_temporal_route_sources.csv").exists()
    assert (output / "feature_gate_temporal_route_summary.csv").exists()
    assert (output / "feature_gate_temporal_subgroup_summary.csv").exists()
    assert (output / "feature_gate_temporal_targets.csv").exists()
    assert (output / "feature_gate_temporal_routes.json").exists()
