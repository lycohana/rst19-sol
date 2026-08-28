from __future__ import annotations

import numpy as np

from rst19.detection import detect_sources, sigma_clipped_stats


def _add_gaussian(image: np.ndarray, x0: float, y0: float, amplitude: float, sigma: float) -> None:
    yy, xx = np.indices(image.shape, dtype=np.float64)
    image += amplitude * np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2.0 * sigma**2))


def test_sigma_clipped_stats_resists_bright_sources() -> None:
    image = np.full((64, 64), 100.0)
    image[30, 30] = 100_000.0

    background, noise = sigma_clipped_stats(image)

    assert background == 100.0
    assert noise > 0


def test_detect_sources_returns_injected_peaks() -> None:
    rng = np.random.default_rng(19)
    image = rng.normal(100.0, 2.0, size=(96, 96))
    _add_gaussian(image, 24.5, 31.5, 120.0, 1.4)
    _add_gaussian(image, 70.0, 64.0, 80.0, 1.7)
    mask = np.zeros(image.shape, dtype=bool)
    mask[0, :10] = True

    result = detect_sources(image, mask=mask, threshold_sigma=5.0, min_distance=4, aperture_radius=4)

    assert result.star_count >= 2
    positions = np.array([(source.x, source.y) for source in result.sources])
    assert np.min(np.linalg.norm(positions - np.array([24.5, 31.5]), axis=1)) < 1.5
    assert np.min(np.linalg.norm(positions - np.array([70.0, 64.0]), axis=1)) < 1.5
    assert all(source.snr > 5 for source in result.sources[:2])
    assert all(source.flux_error is not None and source.flux_snr is not None for source in result.sources)
    assert result.star_count == 2
    assert result.returned_count == result.candidate_count


def test_detect_sources_only_truncates_when_limit_is_explicit() -> None:
    image = np.zeros((48, 48), dtype=float)
    for y, x in ((8, 8), (8, 24), (8, 40), (24, 8), (24, 24), (24, 40), (40, 8), (40, 24), (40, 40)):
        image[y, x] = 100.0

    full = detect_sources(image, threshold_sigma=4.0, min_distance=3, aperture_radius=2)
    limited = detect_sources(image, threshold_sigma=4.0, min_distance=3, aperture_radius=2, max_sources=2)

    assert full.candidate_count == 9
    assert full.returned_count == 9
    assert limited.candidate_count == 9
    assert limited.returned_count == 2
    assert limited.truncated
    assert full.star_count == 0
    assert all("NARROW" in source.flags or "SPIKE" in source.flags for source in full.sources)


def test_detect_sources_rejects_a_long_connected_trail_without_hiding_candidates() -> None:
    rng = np.random.default_rng(3)
    image = rng.normal(20.0, 2.0, size=(128, 128))
    for y in range(10, 118):
        image[y, 64] += 80.0 + 20.0 * np.sin(y * 0.8)
        image[y, 65] += 50.0 + 15.0 * np.cos(y * 0.6)

    result = detect_sources(image, threshold_sigma=4.0, min_distance=3, aperture_radius=4, psf_fwhm=3.0)

    line_sources = [source for source in result.sources if "LINE_ARTIFACT" in source.flags]
    assert result.candidate_count >= len(line_sources) >= 1
    assert line_sources
    assert all(not source.quality_passed for source in line_sources)
    assert result.star_count < result.returned_count


def test_detect_sources_marks_a_masked_pixel_inside_the_geometric_aperture() -> None:
    rng = np.random.default_rng(31)
    image = rng.normal(20.0, 2.0, size=(64, 64))
    _add_gaussian(image, 32.0, 32.0, 1_000.0, 1.4)
    mask = np.zeros(image.shape, dtype=bool)
    mask[32, 35] = True

    result = detect_sources(image, mask=mask, threshold_sigma=5.0, min_distance=4, aperture_radius=4)

    assert any("MASKED" in source.flags for source in result.sources)
    assert all(not source.quality_passed for source in result.sources if "MASKED" in source.flags)


def test_detect_sources_marks_an_in_aperture_saturated_pixel() -> None:
    rng = np.random.default_rng(32)
    image = np.rint(rng.normal(20.0, 2.0, size=(64, 64))).astype(np.int16)
    image[31:34, 31:34] += 500
    image[32, 35] = 32750

    result = detect_sources(image, threshold_sigma=5.0, min_distance=4, aperture_radius=4)

    assert any("SATURATED" in source.flags for source in result.sources)
    assert all(not source.quality_passed for source in result.sources if "SATURATED" in source.flags)
