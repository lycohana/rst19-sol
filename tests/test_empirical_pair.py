from __future__ import annotations

import json

import numpy as np

from rst19.detection import Detection
from rst19.empirical_pair import (
    load_detection_catalog_csv,
    run_empirical_psf_pair_audit,
    write_empirical_psf_pair_audit_artifacts,
)
from rst19.experiments import _gaussian_source
from rst19.models import FitsFrame


def _source(
    detection_id: int,
    x: float,
    y: float,
    *,
    quality: bool,
) -> Detection:
    return Detection(
        detection_id=detection_id,
        x=x,
        y=y,
        peak=149.0,
        flux=1000.0,
        background=21.0,
        noise=5.0,
        snr=25.0,
        fwhm=3.0,
        flags=() if quality else ("UNRESOLVED_BLEND",),
        flux_snr=50.0,
        ellipticity=0.1,
        quality_passed=quality,
    )


def test_empirical_pair_audit_builds_global_and_local_templates(monkeypatch, tmp_path) -> None:
    image = np.full((128, 128), 21.0, dtype=np.float32)
    template_positions = ((20.0, 20.0), (108.0, 20.0), (20.0, 108.0), (108.0, 108.0))
    sources = []
    for index, (x, y) in enumerate(template_positions):
        _gaussian_source(image, x, y, 128.0, 3.0)
        sources.append(_source(index, x, y, quality=True))
    _gaussian_source(image, 62.0, 64.0, 128.0, 3.0)
    _gaussian_source(image, 66.0, 64.0, 64.0, 3.0)
    sources.extend(
        (
            _source(82931, 62.0, 64.0, quality=False),
            _source(82934, 66.0, 64.0, quality=False),
        )
    )
    frame = FitsFrame(path=tmp_path / "frame.fits", header={}, data=image, auxiliary=None, data_offset=0)
    paths = (tmp_path / "frame-a.fits", tmp_path / "frame-b.fits")
    monkeypatch.setattr("rst19.empirical_pair.read_fits", lambda _path: frame)
    progress: list[tuple[int, int]] = []

    result = run_empirical_psf_pair_audit(
        paths,
        (62.0, 64.0),
        (66.0, 64.0),
        template_sources=sources,
        primary_detection_id=82931,
        secondary_detection_id=82934,
        template_frame=frame,
        frame_shifts=((0.0, 0.0), (0.25, -0.2)),
        grid_size=1,
        support_radius=7,
        progress=lambda index, total: progress.append((index, total)),
    )

    assert progress == [(1, 2), (2, 2)]
    assert result.frame_count == 2
    assert result.global_psf is not None
    assert result.global_psf.source_count == 4
    assert result.local_psf is not None
    assert result.local_psf.source_count == 4
    assert all(row.local_template_available for row in result.rows)
    assert all(row.primary_global_correlation is not None for row in result.rows)
    assert all(row.secondary_global_relative_residual is not None for row in result.rows)
    assert "星表身份" in result.conclusion

    output = write_empirical_psf_pair_audit_artifacts(result, tmp_path / "empirical-pair")
    assert (output / "empirical_pair_psf_audit.csv").is_file()
    assert (output / "empirical_pair_psf_audit.json").is_file()
    assert (output / "empirical_pair_psf_similarity.png").is_file()
    payload = json.loads((output / "empirical_pair_psf_audit.json").read_text(encoding="utf-8"))
    assert payload["global_psf"]["source_count"] == 4
    assert payload["local_psf"]["source_count"] == 4


def test_empirical_pair_audit_reports_missing_local_template(monkeypatch, tmp_path) -> None:
    image = np.full((96, 96), 21.0, dtype=np.float32)
    _gaussian_source(image, 48.0, 48.0, 128.0, 3.0)
    sources = (_source(82931, 46.0, 48.0, quality=False), _source(82934, 50.0, 48.0, quality=False))
    frame = FitsFrame(path=tmp_path / "frame.fits", header={}, data=image, auxiliary=None, data_offset=0)
    path = tmp_path / "frame.fits"
    monkeypatch.setattr("rst19.empirical_pair.read_fits", lambda _path: frame)

    result = run_empirical_psf_pair_audit(
        (path,),
        (46.0, 48.0),
        (50.0, 48.0),
        template_sources=sources,
        template_frame=frame,
        grid_size=1,
    )

    assert result.global_psf is None
    assert result.local_psf is None
    assert result.rows[0].local_template_available is False
    assert result.rows[0].primary_global_correlation is None
    assert "模板不足" in result.rows[0].note


def test_load_detection_catalog_csv_preserves_detection_fields(tmp_path) -> None:
    path = tmp_path / "source_catalog.csv"
    path.write_text(
        "detection_id,x,y,peak,flux,background,noise,snr,fwhm,flags,flux_error,flux_snr,filter_snr,"
        "fwhm_x,fwhm_y,ellipticity,sharpness,footprint_pixels,psf_support_pixels,quality_passed,"
        "peak_x,peak_y,centroid_shift_px,proposal_methods,proposal_scales,proposal_snr,"
        "nearest_gaussian_px,deblend_delta_bic,deblend_component_snr,repeated_code_count,"
        "range_anomaly_pixel_count,repeated_code_values\n"
        "7,10.5,11.5,120,900,21,5,24,3.2,,2,18,20,3.1,3.0,0.1,0.2,8,5,true,"
        "10,12,0.5,gaussian|dog_narrow,1.732|2.0,22,1.5,4.0,3.0,0,0,3990|3991\n",
        encoding="utf-8",
    )

    sources = load_detection_catalog_csv(path)

    assert len(sources) == 1
    source = sources[0]
    assert source.detection_id == 7
    assert source.quality_passed is True
    assert source.proposal_methods == ("gaussian", "dog_narrow")
    assert source.proposal_scales == (1.732, 2.0)
    assert source.repeated_code_values == (3990, 3991)
