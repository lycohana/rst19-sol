from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from rst19.detection import Detection, DetectionResult
from rst19.signed_null_forward_overlap import (
    run_signed_null_forward_overlap,
    write_signed_null_forward_overlap_artifacts,
)


def _source(
    detection_id: int,
    x: float,
    y: float,
    *,
    quality_passed: bool,
    flags: tuple[str, ...] = (),
) -> Detection:
    return Detection(
        detection_id=detection_id,
        x=x,
        y=y,
        peak=100.0,
        flux=200.0,
        background=10.0,
        noise=2.0,
        snr=20.0,
        fwhm=2.0,
        flags=flags,
        flux_snr=10.0,
        filter_snr=20.0,
        quality_passed=quality_passed,
        peak_x=round(x),
        peak_y=round(y),
        footprint_pixels=8,
        psf_support_pixels=5,
    )


def _analysis(sources: tuple[Detection, ...]) -> SimpleNamespace:
    return SimpleNamespace(
        detection=DetectionResult(
            image_shape=(32, 32),
            background=10.0,
            noise=2.0,
            threshold=18.0,
            candidate_count=len(sources),
            sources=sources,
            quality_count=sum(source.quality_passed for source in sources),
            parameters={},
        )
    )


def _write_reverse_csv(path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("frame_index", "detection_id", "x", "y", "raw_evidence_layer", "flux_snr"),
        )
        writer.writeheader()
        writer.writerows(
            (
                {
                    "frame_index": 1,
                    "detection_id": 101,
                    "x": 10.0,
                    "y": 10.0,
                    "raw_evidence_layer": "raw_negative_anomaly",
                    "flux_snr": 20.0,
                },
                {
                    "frame_index": 2,
                    "detection_id": 202,
                    "x": 20.0,
                    "y": 20.0,
                    "raw_evidence_layer": "extreme_range_transfer",
                    "flux_snr": 30.0,
                },
                {
                    "frame_index": 2,
                    "detection_id": 203,
                    "x": 0.0,
                    "y": 0.0,
                    "raw_evidence_layer": "no_negative_evidence",
                    "flux_snr": 5.0,
                },
            )
        )


def test_forward_overlap_separates_candidate_and_quality_neighbors(monkeypatch, tmp_path) -> None:
    reverse_csv = tmp_path / "reverse.csv"
    _write_reverse_csv(reverse_csv)
    frame_a = tmp_path / "a.fits"
    frame_b = tmp_path / "b.fits"

    def fake_analyze(path: Path, **_kwargs: object) -> SimpleNamespace:
        if path.name == "a.fits":
            return _analysis((_source(1, 10.8, 10.0, quality_passed=False, flags=("SPIKE",)),))
        return _analysis((_source(2, 20.8, 20.0, quality_passed=True),))

    monkeypatch.setattr("rst19.signed_null_forward_overlap.analyze_frame", fake_analyze)
    result = run_signed_null_forward_overlap((frame_a, frame_b), reverse_csv)

    assert result.frame_count == 2
    assert result.reverse_quality_source_count == 3
    assert result.overlap_class_counts == {
        "candidate_only_within_2px": 1,
        "quality_counterpart_within_2px": 1,
        "no_forward_candidate_within_4px": 1,
    }
    first, second, third = result.source_rows
    assert first.candidate_count_r1 == 1
    assert first.quality_count_r4 == 0
    assert first.nearest_candidate_quality_passed is False
    assert second.quality_count_r2 == 1
    assert second.nearest_quality_detection_id == 2
    assert third.nearest_candidate_distance_px > 4.0

    output = write_signed_null_forward_overlap_artifacts(result, tmp_path / "out")
    assert (output / "signed_null_forward_overlap.csv").is_file()
    payload = json.loads(
        (output / "signed_null_forward_overlap_summary.json").read_text(encoding="utf-8")
    )
    assert payload["source_row_count"] == 3
    assert len(payload["frame_rows"]) == 2
    assert payload["frame_summaries"] == payload["frame_rows"]
    assert payload["detector_parameters"] == payload["parameters"]
    assert "detector-level" in payload["conclusion"]


def test_forward_overlap_rejects_missing_reverse_fields(tmp_path) -> None:
    reverse_csv = tmp_path / "reverse.csv"
    reverse_csv.write_text("frame_index,x,y\n1,1,1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing fields"):
        run_signed_null_forward_overlap((tmp_path / "a.fits",), reverse_csv)
