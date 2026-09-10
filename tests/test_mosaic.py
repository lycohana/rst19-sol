from __future__ import annotations

import numpy as np

from rst19.mosaic import (
    build_registered_mosaic,
    load_mosaic_cache,
    render_mosaic_preview,
    save_mosaic_cache,
)


def test_registered_mosaic_uses_union_footprint_and_coverage() -> None:
    frame_a = np.full((3, 4), 10.0, dtype=np.float32)
    frame_b = np.full((3, 4), 20.0, dtype=np.float32)

    result = build_registered_mosaic(
        [frame_a, frame_b],
        [(0.0, 0.0), (1.0, 0.0)],
        combine_mode="median",
        mask_zero_pixels=False,
    )

    assert result.source_shape == (3, 4)
    assert result.output_shape == (3, 5)
    assert result.common_origin_xy == (-1.0, 0.0)
    assert np.array_equal(result.coverage[1], np.array([1, 2, 2, 2, 1], dtype=np.uint16))
    assert np.allclose(result.image[1], np.array([20, 15, 15, 15, 10], dtype=np.float32))
    assert result.overlap_pixel_count == 6
    assert result.max_coverage == 2
    assert result.covered_bbox_xy == (0, 1, 4, 2)
    assert result.uncovered_pixel_count == 5


def test_registered_mosaic_masks_auxiliary_and_zero_pixels() -> None:
    frame = np.ones((2, 110), dtype=np.float32)
    frame[0, 0] = 999.0
    frame[1, 10] = 0.0

    result = build_registered_mosaic([frame], [(0.0, 0.0)])

    assert np.isnan(result.image[0, 0])
    assert np.isnan(result.image[1, 10])
    assert result.covered_pixel_count == (2 * 110 - 104 - 1)


def test_mosaic_cache_round_trip(tmp_path) -> None:
    frame = np.arange(12, dtype=np.float32).reshape(3, 4)
    result = build_registered_mosaic([frame], [(0.0, 0.0)], mask_zero_pixels=False)
    key = "test-mosaic"

    save_mosaic_cache(tmp_path, key, result)
    restored = load_mosaic_cache(tmp_path, key)

    assert restored is not None
    assert restored.as_dict() == result.as_dict()
    assert np.array_equal(restored.coverage, result.coverage)
    assert np.allclose(restored.image, result.image, equal_nan=True)


def test_preview_keeps_uncovered_footprint_transparent() -> None:
    image = np.array([[1.0, np.nan], [2.0, 3.0]], dtype=np.float32)

    preview = render_mosaic_preview(image, mode="raw", max_side=64)

    assert preview.mode == "RGBA"
    assert preview.getpixel((1, 0))[3] == 0
    assert preview.getpixel((0, 0))[3] == 255
