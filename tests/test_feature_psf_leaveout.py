from __future__ import annotations

import json

import numpy as np

from rst19.detection import Detection
from rst19.experiments import _gaussian_source
from rst19.feature_psf_leaveout import (
    run_feature_psf_leaveout_audit,
    write_feature_psf_leaveout_artifacts,
)


def _source(detection_id: int, x: float, y: float) -> Detection:
    return Detection(
        detection_id=detection_id,
        x=x,
        y=y,
        peak=140.0,
        flux=900.0,
        background=20.0,
        noise=5.0,
        snr=28.0,
        fwhm=3.0,
        flags=(),
        flux_snr=45.0,
        ellipticity=0.1,
        quality_passed=True,
    )


def test_feature_psf_leaveout_excludes_the_evaluated_source_and_reports_progress(tmp_path) -> None:
    image = np.full((96, 96), 20.0, dtype=np.float32)
    positions = ((16.0, 16.0), (48.0, 16.0), (16.0, 48.0), (48.0, 48.0))
    sources = tuple(_source(index, x, y) for index, (x, y) in enumerate(positions))
    for x, y in positions:
        _gaussian_source(image, x, y, 120.0, 3.0)

    progress: list[tuple[int, int]] = []
    result = run_feature_psf_leaveout_audit(
        image,
        sources,
        per_class=4,
        support_radius=3,
        max_template_sources=4,
        progress=lambda index, total: progress.append((index, total)),
    )

    compact = next(row for row in result.class_rows if row.feature_class == "compact_quality")
    assert compact.sample_count == 4
    assert compact.in_sample_valid_count == 4
    assert compact.leaveout_valid_count == 4
    assert compact.paired_count == 4
    assert compact.leaveout_template_source_count_median == 3.0
    assert compact.leaveout_correlation_median is not None
    assert compact.leaveout_correlation_median > 0.95
    assert progress == [(1, 4), (2, 4), (3, 4), (4, 4)]
    assert all(row.leaveout_template_source_count == 3 for row in result.source_rows)

    output = write_feature_psf_leaveout_artifacts(result, tmp_path / "leaveout")
    assert (output / "feature_psf_leaveout.csv").is_file()
    assert (output / "feature_psf_leaveout_source.csv").is_file()
    payload = json.loads((output / "feature_psf_leaveout.json").read_text(encoding="utf-8"))
    assert payload["global_psf"]["source_count"] == 4
    assert len(payload["source_rows"]) == 4
