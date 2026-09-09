from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from rst19.detection import Detection
from rst19.empirical_pair_fit import (
    run_empirical_pair_fit_audit,
    write_empirical_pair_fit_artifacts,
)
from rst19.experiments import EmpiricalPSF, _gaussian_source
from rst19.models import FitsFrame


def _template_source(detection_id: int, x: float, y: float) -> Detection:
    return Detection(
        detection_id=detection_id,
        x=x,
        y=y,
        peak=120.0,
        flux=800.0,
        background=20.0,
        noise=5.0,
        snr=24.0,
        fwhm=3.0,
        ellipticity=0.1,
        quality_passed=True,
        flags=(),
        flux_snr=40.0,
    )


def test_empirical_pair_fit_uses_same_window_and_writes_artifacts(monkeypatch, tmp_path) -> None:
    image = np.full((96, 96), 20.0, dtype=np.float32)
    _gaussian_source(image, 46.0, 48.0, 120.0, 3.0)
    _gaussian_source(image, 48.5, 48.0, 70.0, 3.0)
    image[48, 47] = 3990.0
    grid = np.arange(-3, 4, dtype=np.float64)
    yy, xx = np.meshgrid(grid, grid, indexing="ij")
    kernel = np.exp(-0.5 * (xx**2 + yy**2) / 1.3**2).astype(np.float32)
    kernel /= np.max(kernel)
    empirical_psf = EmpiricalPSF(
        kernel=kernel,
        support_radius=3,
        source_count=4,
        median_fwhm_px=3.0,
        median_ellipticity=0.1,
    )
    frame = FitsFrame(
        path=tmp_path / "frame.fits",
        header={},
        data=image,
        auxiliary=None,
        data_offset=0,
    )
    paths = (tmp_path / "frame-a.fits", tmp_path / "frame-b.fits")
    monkeypatch.setattr("rst19.empirical_pair_fit.read_fits", lambda _path: frame)
    progress: list[tuple[int, int]] = []

    result = run_empirical_pair_fit_audit(
        paths,
        (46.0, 48.0),
        (48.5, 48.0),
        template_sources=(
            _template_source(1, 12.0, 12.0),
            _template_source(2, 80.0, 12.0),
        ),
        primary_detection_id=82931,
        secondary_detection_id=82934,
        template_frame=frame,
        frame_shifts=((0.0, 0.0), (0.25, -0.2)),
        coordinate_mode="centroid",
        mask_modes=("raw", "sentinel_masked", "repeated_code_masked"),
        repeated_code_values=(3990,),
        support_radius=3,
        patch_padding_px=10.0,
        best_single_search_radius_px=1.0,
        best_single_grid_step_px=0.5,
        empirical_psf=empirical_psf,
        progress=lambda index, total: progress.append((index, total)),
    )

    assert progress == [(1, 2), (2, 2)]
    assert result.frame_count == 2
    assert len(result.rows) == 6
    assert all(row.patch_width > 0 and row.patch_height > 0 for row in result.rows)
    assert all(row.fixed_delta_bic is not None for row in result.rows)
    assert all(row.best_single_grid_delta_bic is not None for row in result.rows)
    assert all(row.best_single_grid_evaluated_count > 0 for row in result.rows)
    raw_row = next(row for row in result.rows if row.mask_mode == "raw" and row.frame_index == 1)
    code_row = next(
        row for row in result.rows if row.mask_mode == "repeated_code_masked" and row.frame_index == 1
    )
    assert code_row.masked_pixel_count > raw_row.masked_pixel_count
    assert "位置锚定" in result.conclusion

    output = write_empirical_pair_fit_artifacts(result, tmp_path / "empirical-pair-fit")
    assert (output / "empirical_pair_fit.csv").is_file()
    assert (output / "empirical_pair_fit.json").is_file()
    assert (output / "empirical_pair_fit_bic.png").is_file()
    payload = json.loads((output / "empirical_pair_fit.json").read_text(encoding="utf-8"))
    assert payload["empirical_psf"]["source_count"] == 4
    assert payload["rows"][0]["best_single_grid_evaluated_count"] > 0


def test_empirical_pair_fit_rejects_small_patch_padding(monkeypatch, tmp_path) -> None:
    image = np.full((32, 32), 20.0, dtype=np.float32)
    kernel = np.ones((7, 7), dtype=np.float32)
    psf = EmpiricalPSF(kernel, 3, 1, 3.0, 0.1)
    path = tmp_path / "frame.fits"
    monkeypatch.setattr(
        "rst19.empirical_pair_fit.read_fits",
        lambda _path: FitsFrame(path=path, header={}, data=image, auxiliary=None, data_offset=0),
    )
    try:
        run_empirical_pair_fit_audit(
            (Path(path),),
            (12.0, 12.0),
            (15.0, 12.0),
            template_sources=(_template_source(1, 4.0, 4.0),),
            template_frame=image,
            support_radius=3,
            patch_padding_px=3.0,
            empirical_psf=psf,
        )
    except ValueError as exc:
        assert "patch_padding_px" in str(exc)
    else:
        raise AssertionError("expected small patch padding to be rejected")
