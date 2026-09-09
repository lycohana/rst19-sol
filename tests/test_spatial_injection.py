from __future__ import annotations

import json

import numpy as np

from rst19.detection import DetectionResult
from rst19.fits import FitsFrame
from rst19.spatial_injection import run_spatial_injection_audit, write_spatial_injection_artifacts


def test_spatial_injection_audit_keeps_cell_denominators_and_writes_artifacts(monkeypatch, tmp_path) -> None:
    image = np.full((128, 128), 20.0, dtype=np.float32)
    frame_path = tmp_path / "frame.fits"
    frame = FitsFrame(path=frame_path, header={}, data=image, auxiliary=None, data_offset=0)
    monkeypatch.setattr("rst19.spatial_injection.read_fits", lambda _path: frame)
    calls: list[int] = []

    def fake_detect(values, **_kwargs):
        calls.append(int(np.asarray(values).shape[0]))
        return DetectionResult(
            image_shape=tuple(np.asarray(values).shape),
            background=20.0,
            noise=3.0,
            threshold=32.0,
            candidate_count=0,
            sources=(),
            parameters={"filter_noise": 3.0},
            quality_count=0,
        )

    monkeypatch.setattr("rst19.spatial_injection.detect_sources", fake_detect)
    progress: list[tuple[int, int]] = []
    result = run_spatial_injection_audit(
        frame_path,
        grid_size=2,
        peak_levels=(56.0,),
        trials_per_level=1,
        sources_per_cell=1,
        psf_fwhm=2.0,
        proposal_mode="hybrid",
        signal_normalization="peak_excess",
        progress=lambda index, total: progress.append((index, total)),
    )

    assert len(calls) == 6  # original baseline, same-dtype paired control, plus four cell injections
    assert len(result.rows) == 4
    assert {row.cell_id for row in result.rows} == {"r0c0", "r0c1", "r1c0", "r1c1"}
    assert all(row.injected_count == 1 for row in result.rows)
    assert all(row.candidate_recovered_count == 0 for row in result.rows)
    assert all(row.quality_recovered_count == 0 for row in result.rows)
    assert all(row.paired_control_candidate_count == 0 for row in result.rows)
    assert all(row.candidate_delta_vs_paired_control == 0.0 for row in result.rows)
    assert all(row.mean_filter_noise == 3.0 for row in result.rows)
    assert all(row.filter_noise_delta_vs_paired_control == 0.0 for row in result.rows)
    assert result.paired_control_candidate_count == 0
    assert result.paired_control_quality_count == 0
    assert result.paired_control_filter_noise == 3.0
    assert progress == [(0, 4), (1, 4), (2, 4), (3, 4), (4, 4)]

    output = write_spatial_injection_artifacts(result, tmp_path / "artifacts")
    assert (output / "spatial_injection_audit.csv").is_file()
    payload = json.loads((output / "spatial_injection_audit.json").read_text(encoding="utf-8"))
    assert payload["grid_size"] == 2
    assert len(payload["rows"]) == 4
    assert "候选/质量回收" in payload["conclusion"]


def test_spatial_injection_audit_rejects_small_cells() -> None:
    image = np.full((32, 32), 20.0, dtype=np.float32)
    frame = FitsFrame(path="small.fits", header={}, data=image, auxiliary=None, data_offset=0)

    # The real detector is not reached: each 2x2 cell is too small for the
    # requested PSF/aperture safety margin.
    try:
        run_spatial_injection_audit(frame, grid_size=2, peak_levels=(56.0,))
    except ValueError as exc:
        assert "cell" in str(exc)
    else:
        raise AssertionError("small spatial cells should be rejected")
