from __future__ import annotations

import json

import pytest

from rst19.sequence_feature_comparison import (
    compare_sequence_feature_runs,
    write_sequence_feature_comparison_artifacts,
)


def _payload(*, frame_count: int = 2, required_presence: int = 2) -> dict[str, object]:
    frame_rows = []
    for frame_index in range(1, frame_count + 1):
        frame_rows.append(
            {
                "frame_index": frame_index,
                "path": f"frame-{frame_index}.fits",
                "candidate_count": 100 if frame_index == 1 else 101,
                "quality_count": 40 if frame_index == 1 else 41,
            }
        )
    return {
        "frame_count": frame_count,
        "association_radius_px": 1.0,
        "required_presence": required_presence,
        "frame_rows": frame_rows,
        "persistence_rows": [
            {
                "feature_class": "compact_quality",
                "feature_class_label": "紧凑质量源",
                "anchor_candidate_count": 10,
                "candidate_presence_ge_required_count": 9,
                "candidate_presence_all_frames_count": 8,
                "anchor_quality_count": 8,
                "quality_presence_ge_required_count": 7,
                "quality_presence_all_frames_count": 6,
            }
        ],
        "class_transition_rows": [
            {
                "layer": "quality",
                "anchor_feature_class": "compact_quality",
                "response_feature_class": "compact_quality",
                "anchor_count": 8,
                "frame_count": 2,
                "possible_match_count": 16,
                "matched_count": 14,
                "match_rate": 0.875,
                "mean_matches_per_anchor": 1.75,
            }
        ],
    }


def _write_payload(path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_compare_sequence_feature_runs_separates_candidate_and_quality_layers(tmp_path) -> None:
    low = tmp_path / "low.json"
    high = tmp_path / "high.json"
    _write_payload(low, _payload())
    high_payload = _payload()
    high_payload["persistence_rows"][0]["quality_presence_ge_required_count"] = 8  # type: ignore[index]
    _write_payload(high, high_payload)

    result = compare_sequence_feature_runs({"SNR3": low, "SNR9": high})

    assert [summary.unique_frame_count for summary in result.runs] == [2, 2]
    assert result.runs[0].first_frame_candidate_count == 100
    assert result.runs[0].first_frame_quality_fraction == 0.4
    low_row = next(row for row in result.rows if row.configuration == "SNR3")
    high_row = next(row for row in result.rows if row.configuration == "SNR9")
    assert low_row.candidate_persistence_fraction == 0.9
    assert low_row.quality_persistence_fraction == 0.875
    assert high_row.quality_persistence_fraction == 1.0
    assert result.observations["same_sequence_geometry"] is True

    output = write_sequence_feature_comparison_artifacts(result, tmp_path / "out")
    assert (output / "sequence_feature_parameter_comparison.json").is_file()
    assert (output / "sequence_feature_parameter_runs.csv").is_file()
    assert (output / "sequence_feature_parameter_profiles.csv").is_file()
    assert (output / "sequence_feature_parameter_transitions.csv").is_file()


def test_compare_sequence_feature_runs_rejects_mismatched_frame_count(tmp_path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    _write_payload(first, _payload(frame_count=2))
    _write_payload(second, _payload(frame_count=3, required_presence=3))

    with pytest.raises(ValueError, match="frame_count"):
        compare_sequence_feature_runs({"first": first, "second": second})
