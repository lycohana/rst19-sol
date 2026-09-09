from __future__ import annotations

import json

import numpy as np

from rst19.aperture_sensitivity import (
    run_aperture_sensitivity_audit,
    write_aperture_sensitivity_artifacts,
)
from rst19.detection import Detection, DetectionResult
from rst19.experiments import EmpiricalPSF
from rst19.fits import FitsFrame


def _source(*, quality: bool = True) -> Detection:
    return Detection(
        detection_id=7,
        x=8.1,
        y=8.0,
        peak=100.0,
        flux=80.0,
        background=20.0,
        noise=3.0,
        snr=26.0,
        fwhm=2.2,
        flags=() if quality else ("LOW_FLUX_SNR",),
        flux_error=10.0,
        flux_snr=8.0 if quality else 4.0,
        filter_snr=20.0,
        fwhm_x=2.2,
        fwhm_y=2.2,
        ellipticity=0.0,
        sharpness=0.2,
        footprint_pixels=8,
        psf_support_pixels=5,
        quality_passed=quality,
    )


def test_aperture_sensitivity_keeps_paired_control_and_writes_artifacts(monkeypatch, tmp_path) -> None:
    image = np.full((64, 64), 20.0, dtype=np.float32)
    frame_path = tmp_path / "frame.fits"
    frame = FitsFrame(path=frame_path, header={}, data=image, auxiliary=None, data_offset=0)
    monkeypatch.setattr("rst19.aperture_sensitivity.read_fits", lambda _path: frame)
    calls: list[tuple[int, int]] = []

    def fake_detect(values, **kwargs):
        values_array = np.asarray(values)
        calls.append((int(kwargs["aperture_radius"]), len(calls)))
        if len(calls) == 1:
            return DetectionResult(
                image_shape=values_array.shape,
                background=20.0,
                noise=3.0,
                threshold=32.0,
                candidate_count=0,
                sources=(),
                parameters={"saturation_level": 32735.0, "filter_noise": 3.0},
                quality_count=0,
            )
        if len(calls) in {2, 4}:
            return DetectionResult(
                image_shape=values_array.shape,
                background=20.0,
                noise=3.0,
                threshold=32.0,
                candidate_count=10,
                sources=(_source(quality=True),),
                parameters={"filter_noise": 3.0},
                quality_count=1,
            )
        return DetectionResult(
            image_shape=values_array.shape,
            background=20.0,
            noise=3.0,
            threshold=32.0,
            candidate_count=11,
            sources=(_source(quality=True),),
            parameters={"filter_noise": 3.0},
            quality_count=1,
        )

    monkeypatch.setattr("rst19.aperture_sensitivity.detect_sources", fake_detect)
    monkeypatch.setattr(
        "rst19.aperture_sensitivity.estimate_empirical_psf",
        lambda *_args, **_kwargs: EmpiricalPSF(
            kernel=np.ones((7, 7), dtype=np.float32),
            support_radius=3,
            source_count=4,
            median_fwhm_px=2.4,
            median_ellipticity=0.05,
        ),
    )
    monkeypatch.setattr("rst19.aperture_sensitivity._inject_source_signal", lambda *_args, **_kwargs: 4.0)

    progress: list[tuple[int, int]] = []
    result = run_aperture_sensitivity_audit(
        frame_path,
        aperture_radii=(3, 4),
        reference_aperture_radius=4,
        control_signal_adu=512.0,
        positions=(("r0c0", 8.0, 8.0, 20.0, 3.0),),
        progress=lambda index, total: progress.append((index, total)),
    )

    assert calls == [(4, 0), (3, 1), (3, 2), (4, 3), (4, 4)]
    assert len(result.rows) == 2
    assert all(row.candidate_recovered for row in result.rows)
    assert all(row.quality_recovered for row in result.rows)
    assert all(row.candidate_delta_vs_control == 1 for row in result.rows)
    assert all(row.quality_delta_vs_control == 0 for row in result.rows)
    assert result.psf_kernel_sum == 49.0
    assert np.isclose(result.psf_encircled_energy_by_radius["3"], 29.0 / 49.0)
    assert progress == [(0, 4), (1, 4), (2, 4), (3, 4), (4, 4)]

    output = write_aperture_sensitivity_artifacts(result, tmp_path / "artifacts")
    assert (output / "aperture_sensitivity_audit.csv").is_file()
    payload = json.loads((output / "aperture_sensitivity_audit.json").read_text(encoding="utf-8"))
    assert payload["aperture_radii"] == [3, 4]
    assert len(payload["summary_by_aperture"]) == 2
    assert "质量回收率最高" in payload["conclusion"]


def test_aperture_sensitivity_rejects_invalid_radius() -> None:
    frame = FitsFrame(
        path="small.fits",
        header={},
        data=np.full((32, 32), 20.0, dtype=np.float32),
        auxiliary=None,
        data_offset=0,
    )
    try:
        run_aperture_sensitivity_audit(frame, aperture_radii=(0,))
    except ValueError as exc:
        assert "positive" in str(exc)
    else:
        raise AssertionError("non-positive aperture radius should be rejected")
