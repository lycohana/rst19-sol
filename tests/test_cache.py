from __future__ import annotations

from pathlib import Path

import numpy as np

from rst19.cache import cache_key, clear_cache, load_analysis, save_analysis
from rst19.detection import Detection, DetectionResult
from rst19.models import FitsFrame
from rst19.photometry import FaintestSource
from rst19.pipeline import FrameAnalysis


def _analysis(frame: FitsFrame) -> FrameAnalysis:
    source = Detection(
        0,
        5.0,
        6.0,
        100.0,
        20.0,
        10.0,
        2.0,
        45.0,
        2.0,
        (),
        flux_error=1.5,
        flux_snr=13.333,
        filter_snr=22.0,
        fwhm_x=2.1,
        fwhm_y=1.9,
        ellipticity=0.095,
        sharpness=0.31,
        footprint_pixels=9,
        quality_passed=True,
    )
    detection = DetectionResult(
        image_shape=(12, 16),
        background=10.0,
        noise=2.0,
        threshold=18.0,
        candidate_count=1,
        sources=(source,),
        parameters={"threshold_sigma": 4.0, "max_sources": -1},
        quality_count=1,
    )
    faintest = FaintestSource(
        0,
        5.0,
        6.0,
        20.0,
        45.0,
        -3.252574989,
        None,
        (),
        flux_snr=13.333,
        flux_rate=10.0,
    )
    return FrameAnalysis(frame=frame, detection=detection, matching=None, faintest=faintest)


def test_analysis_cache_round_trip_and_clear(tmp_path: Path) -> None:
    frame_path = tmp_path / "frame.fits"
    frame_path.write_bytes(b"frame")
    frame = FitsFrame(frame_path, {"BITPIX": 16}, np.zeros((12, 16), dtype=np.int16), None, 0)
    key = cache_key(frame, threshold_sigma=4.0, min_distance=4, aperture_radius=4, max_sources=None, zero_point=None)
    cache_dir = tmp_path / ".rst19-cache"
    analysis = _analysis(frame)

    save_analysis(cache_dir, key, analysis)
    restored = load_analysis(cache_dir, key, frame)

    assert restored is not None
    assert restored.detection.sources[0].flux == 20.0
    restored_source = restored.detection.sources[0]
    assert restored_source.flux_error == 1.5
    assert restored_source.flux_snr == 13.333
    assert restored_source.filter_snr == 22.0
    assert restored_source.footprint_pixels == 9
    assert restored.detection.star_count == 1
    assert restored.faintest is not None
    assert restored.faintest.detection_id == 0
    assert restored.faintest.flux_rate == 10.0
    assert clear_cache(cache_dir) == 1
    assert load_analysis(cache_dir, key, frame) is None


def test_cache_key_changes_when_scientific_parameters_change(tmp_path: Path) -> None:
    frame_path = tmp_path / "frame.fits"
    frame_path.write_bytes(b"frame")
    frame = FitsFrame(frame_path, {"BITPIX": 16}, np.zeros((4, 4), dtype=np.int16), None, 0)

    base = cache_key(frame, threshold_sigma=4.0, min_distance=3, aperture_radius=4, max_sources=None, zero_point=None)
    assert cache_key(frame, threshold_sigma=4.0, min_distance=3, aperture_radius=4, max_sources=None, zero_point=None, psf_fwhm=4.0) != base
    assert cache_key(frame, threshold_sigma=4.0, min_distance=3, aperture_radius=4, max_sources=None, zero_point=None, min_flux_snr=8.0) != base


def test_clear_cache_does_not_remove_unrelated_files(tmp_path: Path) -> None:
    cache_dir = tmp_path / ".rst19-cache"
    cache_dir.mkdir()
    (cache_dir / "result.json.gz").write_bytes(b"cache")
    (cache_dir / "partial.tmp").write_bytes(b"cache")
    (cache_dir / "legacy-result.json").write_bytes(b"legacy cache")
    keep = cache_dir / "README.txt"
    keep.write_text("keep", encoding="utf-8")

    assert clear_cache(cache_dir) == 3
    assert keep.read_text(encoding="utf-8") == "keep"
