from __future__ import annotations

import csv

import pytest

from rst19.feature_transition_flow import (
    summarize_sequence_feature_transition_flow,
    write_sequence_feature_transition_flow_artifacts,
)


FIELDS = [
    "layer",
    "anchor_feature_class",
    "anchor_feature_class_label",
    "response_feature_class",
    "response_feature_class_label",
    "anchor_count",
    "frame_count",
    "possible_match_count",
    "matched_count",
    "match_rate",
    "mean_matches_per_anchor",
]


def _write_rows(path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def test_transition_flow_separates_any_same_and_cross_class(tmp_path) -> None:
    source = tmp_path / "transition.csv"
    _write_rows(
        source,
        [
            {
                "layer": "candidate",
                "anchor_feature_class": "compact_quality",
                "anchor_feature_class_label": "紧凑",
                "response_feature_class": "compact_quality",
                "response_feature_class_label": "紧凑",
                "anchor_count": 10,
                "frame_count": 2,
                "possible_match_count": 20,
                "matched_count": 12,
            },
            {
                "layer": "candidate",
                "anchor_feature_class": "compact_quality",
                "anchor_feature_class_label": "紧凑",
                "response_feature_class": "masked_or_edge",
                "response_feature_class_label": "边缘",
                "anchor_count": 10,
                "frame_count": 2,
                "possible_match_count": 20,
                "matched_count": 8,
            },
            {
                "layer": "quality",
                "anchor_feature_class": "masked_or_edge",
                "anchor_feature_class_label": "边缘",
                "response_feature_class": "compact_quality",
                "response_feature_class_label": "紧凑",
                "anchor_count": 0,
                "frame_count": 2,
                "possible_match_count": 0,
                "matched_count": 0,
            },
        ],
    )

    result = summarize_sequence_feature_transition_flow(source)

    assert result.input_row_count == 3
    assert result.included_group_count == 1
    assert result.skipped_zero_anchor_group_count == 1
    row = result.rows[0]
    assert row.matched_count == 20
    assert row.same_class_match_count == 12
    assert row.q_any == 1.0
    assert row.q_same == 0.6
    assert row.top_non_same_feature_class == "masked_or_edge"
    assert row.top_non_same_matched_count == 8
    assert row.top_non_same_fraction == 0.4

    output = write_sequence_feature_transition_flow_artifacts(result, tmp_path / "out")
    assert (output / "sequence_feature_transition_flow.csv").is_file()
    assert (output / "sequence_feature_transition_flow.json").is_file()


def test_transition_flow_rejects_inconsistent_group_metadata(tmp_path) -> None:
    source = tmp_path / "transition.csv"
    rows = [
        {
            "layer": "candidate",
            "anchor_feature_class": "compact_quality",
            "anchor_feature_class_label": "紧凑",
            "response_feature_class": "compact_quality",
            "response_feature_class_label": "紧凑",
            "anchor_count": 10,
            "frame_count": 2,
            "possible_match_count": 20,
            "matched_count": 1,
        },
        {
            "layer": "candidate",
            "anchor_feature_class": "compact_quality",
            "anchor_feature_class_label": "紧凑",
            "response_feature_class": "masked_or_edge",
            "response_feature_class_label": "边缘",
            "anchor_count": 11,
            "frame_count": 2,
            "possible_match_count": 22,
            "matched_count": 1,
        },
    ]
    _write_rows(source, rows)

    with pytest.raises(ValueError, match="元数据.*不一致"):
        summarize_sequence_feature_transition_flow(source)
