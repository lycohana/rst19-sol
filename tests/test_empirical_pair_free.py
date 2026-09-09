from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from rst19.detection import Detection
from rst19.empirical_pair_free import (
    run_empirical_free_pair_audit,
    write_empirical_free_pair_artifacts,
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


def test_empirical_free_pair_fit_writes_bounded_model_artifacts(monkeypatch, tmp_path) -> None:
    image = np.full((96, 96), 20.0, dtype=np.float32)
    _gaussian_source(image, 46.0, 48.0, 120.0, 3.0)
    _gaussian_source(image, 48.5, 48.0, 70.0, 3.0)
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

    result = run_empirical_free_pair_audit(
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
        mask_modes=("raw", "repeated_code_masked"),
        repeated_code_values=(3990,),
        support_radius=3,
        patch_padding_px=10.0,
        single_position_radius_px=2.0,
        double_position_radius_px=1.25,
        empirical_psf=empirical_psf,
        progress=lambda index, total: progress.append((index, total)),
    )

    assert progress == [(1, 2), (2, 2)]
    assert result.frame_count == 2
    assert len(result.rows) == 4
    assert all(row.single_bic is not None for row in result.rows)
    assert all(row.double_bic is not None for row in result.rows)
    assert all(row.delta_bic_single_minus_double is not None for row in result.rows)
    assert all(len(row.double_fitted_positions) == 2 for row in result.rows)
    assert all(len(row.double_positions_at_bound) == 2 for row in result.rows)
    assert "自由位置" in result.conclusion

    output = write_empirical_free_pair_artifacts(result, tmp_path / "empirical-pair-free")
    assert (output / "empirical_pair_free.csv").is_file()
    assert (output / "empirical_pair_free.json").is_file()
    assert (output / "empirical_pair_free_bic.png").is_file()
    payload = json.loads((output / "empirical_pair_free.json").read_text(encoding="utf-8"))
    assert payload["empirical_psf"]["source_count"] == 4
    assert payload["rows"][0]["double_fitted_positions"]


def test_empirical_free_pair_rejects_duplicate_targets(tmp_path: Path) -> None:
    try:
        run_empirical_free_pair_audit(
            (tmp_path / "missing.fits",),
            (10.0, 10.0),
            (10.0, 10.0),
            template_sources=(_template_source(1, 4.0, 4.0),),
        )
    except ValueError as exc:
        assert "differ" in str(exc)
    else:
        raise AssertionError("expected duplicate target coordinates to be rejected")


def test_empirical_free_pair_can_recover_a_well_separated_injection(monkeypatch, tmp_path) -> None:
    image = np.full((96, 96), 20.0, dtype=np.float32)
    _gaussian_source(image, 42.0, 48.0, 1200.0, 3.0)
    _gaussian_source(image, 48.0, 48.0, 900.0, 3.0)
    grid = np.arange(-3, 4, dtype=np.float64)
    yy, xx = np.meshgrid(grid, grid, indexing="ij")
    kernel = np.exp(-0.5 * (xx**2 + yy**2) / 1.3**2).astype(np.float32)
    kernel /= np.max(kernel)
    empirical_psf = EmpiricalPSF(kernel, 3, 4, 3.0, 0.1)
    frame = FitsFrame(
        path=tmp_path / "injection.fits",
        header={},
        data=image,
        auxiliary=None,
        data_offset=0,
    )
    monkeypatch.setattr("rst19.empirical_pair_fit.read_fits", lambda _path: frame)

    result = run_empirical_free_pair_audit(
        (tmp_path / "injection.fits",),
        (42.0, 48.0),
        (48.0, 48.0),
        template_sources=(_template_source(1, 10.0, 10.0),),
        template_frame=frame,
        mask_modes=("raw",),
        support_radius=3,
        patch_padding_px=10.0,
        single_position_radius_px=3.0,
        double_position_radius_px=3.0,
        minimum_fitted_separation_px=2.0,
        empirical_psf=empirical_psf,
    )
    row = result.rows[0]
    assert row.confirmation_line
    assert row.delta_bic_single_minus_double is not None
    assert row.delta_bic_single_minus_double >= 10.0
    assert row.fitted_separation_px is not None
    assert row.fitted_separation_px >= 2.0
    assert row.double_second_component_snr is not None
    assert row.double_second_component_snr >= 5.0
