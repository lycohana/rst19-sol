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
