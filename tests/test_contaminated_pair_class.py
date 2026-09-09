from __future__ import annotations

import csv
import json

import numpy as np

from rst19.contaminated_pair_class import (
    run_feature_class_contaminated_pair_audit,
    select_feature_class_anchors,
    write_feature_class_contaminated_pair_artifacts,
)
from rst19.detection import DetectionResult
from rst19.models import FitsFrame


def _write_catalog(path) -> None:
    rows = [
        {
            "detection_id": "1",
            "feature_class": "compact_quality",
            "feature_class_label": "compact",
            "x": "10",
            "y": "10",
            "filter_snr": "20",
            "flux_snr": "10",
            "peak": "100",
            "quality_passed": "true",
            "flags": "",
        },
        {
            "detection_id": "2",
            "feature_class": "compact_quality",
            "feature_class_label": "compact",
            "x": "30",
            "y": "30",
            "filter_snr": "40",
            "flux_snr": "20",
            "peak": "200",
            "quality_passed": "true",
            "flags": "",
        },
        {
            "detection_id": "3",
            "feature_class": "linear_artifact",
            "feature_class_label": "line",
            "x": "20",
            "y": "20",
            "filter_snr": "100",
            "flux_snr": "50",
            "peak": "300",
            "quality_passed": "false",
            "flags": "LINE_ARTIFACT",
        },
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_select_feature_class_anchors_is_deterministic_and_in_bounds(tmp_path) -> None:
    catalog = tmp_path / "source_catalog.csv"
    _write_catalog(catalog)
    rows = select_feature_class_anchors(
        catalog,
        feature_classes=("compact_quality", "linear_artifact"),
        image_shape=(40, 40),
        max_separation_px=4.123,
    )
    assert [row.detection_id for row in rows] == [1, 3]
    assert rows[0].selection_mode == "median"
    assert rows[1].quality_passed is False


def test_class_audit_wraps_injection_and_writes_selection(tmp_path, monkeypatch) -> None:
    catalog = tmp_path / "source_catalog.csv"
    _write_catalog(catalog)
    image = np.full((40, 40), 10.0, dtype=np.float32)
    frame = FitsFrame(path=tmp_path / "synthetic.fits", header={}, data=image, auxiliary=None, data_offset=0)

    monkeypatch.setattr(
        "rst19.contaminated_pair_injection.detect_sources",
        lambda image, **_options: DetectionResult(
            image_shape=image.shape,
            background=10.0,
            noise=2.0,
            threshold=18.0,
            candidate_count=0,
            sources=(),
            quality_count=0,
            parameters={},
        ),
    )
    monkeypatch.setattr("rst19.contaminated_pair_injection.auxiliary_mask", lambda shape: np.zeros(shape, dtype=bool))
    monkeypatch.setattr("rst19.contaminated_pair_injection._inject_source_signal", lambda *_args, **_kwargs: 1.0)
    result = run_feature_class_contaminated_pair_audit(
        frame,
        catalog,
        feature_classes=("compact_quality",),
        separations_px=(4.0,),
        secondary_to_primary_ratios=(1.0,),
    )
    assert result.selected_rows[0].detection_id == 1
    output = write_feature_class_contaminated_pair_artifacts(result, tmp_path / "out")
    assert (output / "contaminated_pair_class_selection.csv").exists()
    assert (output / "contaminated_pair_class_summary.csv").exists()
    assert (output / "contaminated_pair_quality_reason_summary.csv").exists()
    assert (output / "contaminated_pair_injection.json").exists()
    payload = json.loads((output / "contaminated_pair_class_audit.json").read_text(encoding="utf-8"))
    assert payload["selected_rows"][0]["feature_class"] == "compact_quality"
    assert payload["class_summary"][0]["feature_class"] == "compact_quality"
    assert payload["quality_reason_summary"][0]["combined_quality_reason_counts"] == '{"NO_CANDIDATE":2}'
