from __future__ import annotations

import json

import numpy as np

from rst19.detection import Detection, DetectionResult
from rst19.fits import FitsFrame
from rst19.pair_support_audit_cli import _load_shifts
from rst19.pair_support_audit import (
    _assign_target_sources,
    _support_metrics,
    run_pair_support_audit,
    write_pair_support_artifacts,
)


def test_load_shifts_accepts_sequence_and_pair_audit_json_schemas(tmp_path) -> None:
    cumulative = tmp_path / "sequence.json"
    cumulative.write_text(
        json.dumps({"cumulative_shifts": [[0.0, 0.0], [1.25, -0.5]]}),
        encoding="utf-8",
    )
    frame = tmp_path / "pair-fit.json"
    frame.write_text(
        json.dumps({"frame_shifts": [[0.0, 0.0], [1.25, -0.5]]}),
        encoding="utf-8",
    )

    assert _load_shifts(cumulative) == ((0.0, 0.0), (1.25, -0.5))
    assert _load_shifts(frame) == ((0.0, 0.0), (1.25, -0.5))


def _source(detection_id: int, x: float, y: float, *, quality: bool = True, flags: tuple[str, ...] = ()) -> Detection:
    return Detection(
        detection_id=detection_id,
        x=x,
        y=y,
        peak=100.0,
        flux=500.0,
        background=20.0,
        noise=3.0,
        snr=30.0,
        fwhm=2.0,
        flags=flags,
        flux_snr=20.0,
        quality_passed=quality,
    )


def test_pair_assignment_never_reuses_one_candidate() -> None:
    sources = (_source(1, 1.0, 0.0),)

    assigned = _assign_target_sources(
        sources,
        ((0.0, 0.0), (2.0, 0.0)),
        match_radius_px=2.0,
    )

    assert len(assigned) == 1
    assert assigned[0][1] == 0


def test_support_metrics_separate_shared_and_partition_pixels() -> None:
    image = np.full((41, 41), 20.0, dtype=np.float64)
    image[20, 20] = 120.0
    image[20, 23] = 80.0
    image[20, 16] = 60.0
    mask = np.zeros_like(image, dtype=bool)

    metrics = _support_metrics(
        image,
        mask,
        (20.0, 20.0),
        (23.0, 20.0),
        crop_origin=(0, 0),
        aperture_radius=4,
    )

    assert metrics.total_valid_pixels > metrics.exclusive_valid_pixels > 0
    assert metrics.partition_valid_pixels > 0
    assert metrics.total_positive_excess_adu is not None
    assert metrics.shared_positive_fraction is not None
    assert 0.0 < metrics.shared_positive_fraction < 1.0


def test_pair_support_audit_reuses_global_assignment_and_writes_artifacts(monkeypatch, tmp_path) -> None:
    image = np.full((64, 64), 20.0, dtype=np.float32)
    frame_path = tmp_path / "frame.fits"
    frame = FitsFrame(path=frame_path, header={}, data=image, auxiliary=None, data_offset=0)
    monkeypatch.setattr("rst19.pair_support_audit.read_fits", lambda _path: frame)

    def fake_detect(crop, **_kwargs):
        # target crop origin is (20, 20) for the coordinates below.
        sources = (_source(0, 10.0, 10.0), _source(1, 13.0, 10.0, quality=False, flags=("MASKED",)))
        return DetectionResult(
            image_shape=tuple(crop.shape),
            background=20.0,
            noise=3.0,
            threshold=32.0,
            candidate_count=2,
            sources=sources,
            parameters={},
            quality_count=1,
        )

    monkeypatch.setattr("rst19.pair_support_audit.detect_sources", fake_detect)
    progress: list[tuple[int, int]] = []

    result = run_pair_support_audit(
        (frame_path,),
        (30.0, 30.0),
        (33.0, 30.0),
        primary_detection_id=82931,
        secondary_detection_id=82934,
        frame_shifts=((0.0, 0.0),),
        mask_modes=("raw", "shared_aperture_masked"),
        crop_padding_px=10.0,
        aperture_radius=4,
        progress=lambda index, total: progress.append((index, total)),
    )

    assert progress == [(1, 1)]
    assert len(result.rows) == 2
    assert all(row.assigned_candidate_count == 2 for row in result.rows)
    assert result.rows[0].primary_re_detected_id == 0
    assert result.rows[0].secondary_re_detected_id == 1
    assert result.rows[1].masked_pixel_count > result.rows[0].masked_pixel_count

    output = write_pair_support_artifacts(result, tmp_path / "pair-support")
    assert (output / "pair_support_audit.csv").is_file()
    assert (output / "pair_support_summary.csv").is_file()
    payload = json.loads((output / "pair_support_audit.json").read_text(encoding="utf-8"))
    assert payload["rows"][0]["assigned_candidate_count"] == 2
    assert payload["summary"][0]["assigned_two_count"] == 1
    assert payload["summary"][0]["quality_two_count"] == 0
    assert "共同孔径" in payload["conclusion"]
