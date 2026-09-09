from __future__ import annotations

import json

import numpy as np

from rst19.contaminated_pair_injection import (
    ContaminatedPairAnchor,
    run_contaminated_pair_injection_audit,
    write_contaminated_pair_injection_artifacts,
)
from rst19.detection import Detection, DetectionResult
from rst19.models import FitsFrame


def _detection(detection_id: int, x: float, y: float, *, quality: bool) -> Detection:
    return Detection(
        detection_id=detection_id,
        x=x,
        y=y,
        peak=100.0,
        flux=100.0,
        background=10.0,
        noise=2.0,
        snr=20.0,
        fwhm=2.0,
        quality_passed=quality,
        flags=(),
        flux_snr=10.0,
        peak_x=round(x),
        peak_y=round(y),
    )


def test_contaminated_pair_injection_separates_baseline_and_new_hits(tmp_path, monkeypatch) -> None:
    image = np.full((40, 40), 10.0, dtype=np.float32)
    frame = FitsFrame(path=tmp_path / "synthetic.fits", header={}, data=image, auxiliary=None, data_offset=0)
    calls = {"count": 0}

    def fake_detect(image, **_options):
        calls["count"] += 1
        if calls["count"] == 1:
            sources = (_detection(1, 22.0, 20.0, quality=True),)
        else:
            sources = (
                _detection(1, 22.0, 20.0, quality=True),
                _detection(2, 18.0, 20.0, quality=True),
            )
        return DetectionResult(
            image_shape=image.shape,
            background=10.0,
            noise=2.0,
            threshold=18.0,
            candidate_count=len(sources),
            sources=sources,
            quality_count=sum(source.quality_passed for source in sources),
            parameters={},
        )

    monkeypatch.setattr("rst19.contaminated_pair_injection.detect_sources", fake_detect)
    monkeypatch.setattr("rst19.contaminated_pair_injection.auxiliary_mask", lambda shape: np.zeros(shape, dtype=bool))
    monkeypatch.setattr(
        "rst19.contaminated_pair_injection._inject_source_signal",
        lambda image, *_args, **_kwargs: 1.0,
    )
    result = run_contaminated_pair_injection_audit(
        frame,
        (ContaminatedPairAnchor("synthetic", 20.0, 20.0, 1.0, 0.0),),
        separations_px=(4.0,),
        secondary_to_primary_ratios=(1.0,),
        total_peak_excess_adu=100.0,
    )

    row = result.rows[0]
    assert row.baseline_candidate_primary is True
    assert row.baseline_candidate_secondary is False
    assert row.injected_candidate_primary is True
    assert row.injected_candidate_secondary is True
    assert row.new_candidate_secondary is True
    assert row.candidate_pair_resolved is True
    assert row.quality_pair_resolved is True
    assert row.injected_primary_match_id == 1
    assert row.injected_secondary_match_id == 2
    assert row.injected_primary_match_distance_px == 0.0
    assert row.injected_secondary_match_distance_px == 0.0
    assert row.injected_primary_quality_reason == "QUALITY_PASS"
    assert row.injected_secondary_quality_reason == "QUALITY_PASS"
    assert "baseline" in row.note

    output = write_contaminated_pair_injection_artifacts(result, tmp_path / "out")
    assert (output / "contaminated_pair_injection.csv").exists()
    payload = json.loads((output / "contaminated_pair_injection.json").read_text(encoding="utf-8"))
    assert payload["rows"][0]["new_candidate_hit_count"] == 1


def test_contaminated_pair_anchor_is_normalised_and_validated(monkeypatch, tmp_path) -> None:
    image = np.ones((20, 20), dtype=np.float32)
    frame = FitsFrame(path=tmp_path / "synthetic.fits", header={}, data=image, auxiliary=None, data_offset=0)
    monkeypatch.setattr(
        "rst19.contaminated_pair_injection.detect_sources",
        lambda image, **_options: DetectionResult(
            image_shape=image.shape,
            background=1.0,
            noise=1.0,
            threshold=2.0,
            candidate_count=0,
            sources=(),
            quality_count=0,
            parameters={},
        ),
    )
    result = run_contaminated_pair_injection_audit(
        frame,
        (ContaminatedPairAnchor("a", 10.0, 10.0, 3.0, 4.0),),
        separations_px=(2.0,),
        secondary_to_primary_ratios=(1.0,),
    )
    assert result.anchors[0].direction_x == 0.6
    assert result.anchors[0].direction_y == 0.8


def test_local_roi_scope_offsets_detections_and_marks_count_scope(tmp_path, monkeypatch) -> None:
    image = np.full((40, 40), 10.0, dtype=np.float32)
    frame = FitsFrame(path=tmp_path / "synthetic.fits", header={}, data=image, auxiliary=None, data_offset=0)
    calls = {"count": 0}

    def fake_detect(image, **_options):
        calls["count"] += 1
        if calls["count"] == 1:
            sources = (_detection(1, 22.0, 20.0, quality=True),)
        else:
            # The ROI is x/y=[8, 33), so these local coordinates become
            # the full-frame truth positions (22, 20) and (18, 20).
            sources = (
                _detection(1, 14.0, 12.0, quality=True),
                _detection(2, 10.0, 12.0, quality=True),
            )
        return DetectionResult(
            image_shape=image.shape,
            background=10.0,
            noise=2.0,
            threshold=18.0,
            candidate_count=len(sources),
            sources=sources,
            quality_count=len(sources),
            parameters={},
        )

    monkeypatch.setattr("rst19.contaminated_pair_injection.detect_sources", fake_detect)
    monkeypatch.setattr("rst19.contaminated_pair_injection.auxiliary_mask", lambda shape: np.zeros(shape, dtype=bool))
    monkeypatch.setattr(
        "rst19.contaminated_pair_injection._inject_source_signal",
        lambda image, *_args, **_kwargs: 1.0,
    )
    result = run_contaminated_pair_injection_audit(
        frame,
        (ContaminatedPairAnchor("synthetic", 20.0, 20.0, 1.0, 0.0),),
        separations_px=(4.0,),
        secondary_to_primary_ratios=(1.0,),
        total_peak_excess_adu=100.0,
        analysis_scope="local_roi",
        roi_half_size_px=12,
    )

    row = result.rows[0]
    assert row.analysis_scope == "local_roi"
    assert row.roi_bounds == (8, 8, 33, 33)
    assert row.injected_candidate_count == 2
    assert row.candidate_pair_resolved is True
    assert row.quality_pair_resolved is True
    assert result.parameters["injected_count_scope"] == "local_roi"
    assert row.injected_primary_match_id == 1
    assert row.injected_secondary_match_id == 2
    assert row.injected_primary_quality_reason == "QUALITY_PASS"


def test_injection_records_quality_rejection_flags(tmp_path, monkeypatch) -> None:
    image = np.full((40, 40), 10.0, dtype=np.float32)
    frame = FitsFrame(path=tmp_path / "synthetic.fits", header={}, data=image, auxiliary=None, data_offset=0)

    def fake_detect(image, **_options):
        source = Detection(
            detection_id=7,
            x=22.0,
            y=20.0,
            peak=100.0,
            flux=100.0,
            background=10.0,
            noise=2.0,
            snr=20.0,
            fwhm=2.0,
            flags=("UNRESOLVED_BLEND", "LOW_FLUX_SNR"),
            flux_snr=2.0,
            quality_passed=False,
            peak_x=22.0,
            peak_y=20.0,
        )
        return DetectionResult(
            image_shape=image.shape,
            background=10.0,
            noise=2.0,
            threshold=18.0,
            candidate_count=1,
            sources=(source,),
            quality_count=0,
            parameters={},
        )

    monkeypatch.setattr("rst19.contaminated_pair_injection.detect_sources", fake_detect)
    monkeypatch.setattr("rst19.contaminated_pair_injection.auxiliary_mask", lambda shape: np.zeros(shape, dtype=bool))
    monkeypatch.setattr(
        "rst19.contaminated_pair_injection._inject_source_signal",
        lambda image, *_args, **_kwargs: 1.0,
    )
    result = run_contaminated_pair_injection_audit(
        frame,
        (ContaminatedPairAnchor("synthetic", 20.0, 20.0, 1.0, 0.0),),
        separations_px=(4.0,),
        secondary_to_primary_ratios=(1.0,),
        total_peak_excess_adu=100.0,
    )

    row = result.rows[0]
    assert row.injected_primary_match_id == 7
    assert row.injected_secondary_match_id is None
    assert row.injected_primary_quality_reason == "UNRESOLVED_BLEND|LOW_FLUX_SNR"
    assert row.injected_secondary_quality_reason == "NO_CANDIDATE"
