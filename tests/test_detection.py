from __future__ import annotations

import numpy as np
import pytest
from scipy import ndimage

from rst19.detection import (
    Detection,
    _pair_psf_evidence,
    _guard_unresolved_gaussian_pairs,
    _repeated_high_code_values,
    _source_from_peak,
    _resize_grid,
    _response_proposals,
    build_background_model,
    detect_sources,
    sigma_clipped_stats,
)
from rst19.fits import auxiliary_mask


def _add_gaussian(image: np.ndarray, x0: float, y0: float, amplitude: float, sigma: float) -> None:
    yy, xx = np.indices(image.shape, dtype=np.float64)
    image += amplitude * np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2.0 * sigma**2))


def test_sigma_clipped_stats_resists_bright_sources() -> None:
    image = np.full((64, 64), 100.0)
    image[30, 30] = 100_000.0

    background, noise = sigma_clipped_stats(image)

    assert background == 100.0
    assert noise > 0


def test_resize_grid_keeps_zoom_crop_coordinates_without_oversized_array() -> None:
    grid = np.arange(16 * 16, dtype=np.float32).reshape(16, 16)
    target_shape = (64, 80)
    zoom = (
        (target_shape[0] - 1) / (grid.shape[0] - 1),
        (target_shape[1] - 1) / (grid.shape[1] - 1),
    )
    reference = ndimage.zoom(grid, zoom=zoom, order=1, mode="nearest", prefilter=False)
    expected = np.full(target_shape, reference[-1, -1], dtype=np.float32)
    expected[: min(target_shape[0], reference.shape[0]), : min(target_shape[1], reference.shape[1])] = reference[
        : target_shape[0], : target_shape[1]
    ]

    resized = _resize_grid(grid, target_shape)

    assert resized.shape == target_shape
    assert np.max(np.abs(resized - expected)) < 3e-5


def test_response_peak_window_does_not_hide_a_pair_at_the_nms_boundary() -> None:
    response = np.zeros((32, 32), dtype=float)
    response[16, 10] = 20.0
    response[16, 14] = 18.0
    valid = np.ones(response.shape, dtype=bool)

    proposals = _response_proposals(
        response,
        valid,
        threshold_sigma=4.0,
        min_distance=4,
        scale_fwhm=3.0,
        method="gaussian",
    )

    assert {(proposal.x, proposal.y) for proposal in proposals} == {(10, 16), (14, 16)}


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
    assert all(source.psf_support_pixels is not None and source.psf_support_pixels >= 3 for source in result.quality_sources)


def test_hybrid_proposals_recover_a_candidate_missed_by_single_scale_gaussian() -> None:
    rng = np.random.default_rng(1988)
    image = rng.normal(100.0, 2.0, size=(96, 96))
    _add_gaussian(image, 48.2, 47.7, 8.0, 0.8)

    gaussian = detect_sources(
        image,
        threshold_sigma=6.0,
        min_distance=4,
        aperture_radius=4,
        min_flux_snr=2.0,
        proposal_mode="gaussian",
    )
    hybrid = detect_sources(
        image,
        threshold_sigma=6.0,
        min_distance=4,
        aperture_radius=4,
        min_flux_snr=2.0,
        proposal_mode="hybrid",
        dog_threshold_sigma=5.0,
    )

    gaussian_near = [
        source
        for source in gaussian.sources
        if (float(source.peak_x) - 48.2) ** 2 + (float(source.peak_y) - 47.7) ** 2 < 9.0
    ]
    hybrid_near = [
        source
        for source in hybrid.sources
        if (float(source.peak_x) - 48.2) ** 2 + (float(source.peak_y) - 47.7) ** 2 < 9.0
    ]

    assert not gaussian_near
    assert len(hybrid_near) == 1
    assert set(hybrid_near[0].proposal_methods) == {"dog_narrow", "dog_broad"}
    assert hybrid_near[0].proposal_snr is not None and hybrid_near[0].proposal_snr >= 5.0
    # 宽筛补回候选不等于强行通过质量层；该弱源仍保留 LOW_FLUX_SNR。
    assert not hybrid_near[0].quality_passed
    assert "LOW_FLUX_SNR" in hybrid_near[0].flags


def test_hybrid_proposals_merge_algorithm_hits_without_double_counting() -> None:
    rng = np.random.default_rng(1919)
    image = rng.normal(100.0, 2.0, size=(96, 96))
    _add_gaussian(image, 48.0, 48.0, 160.0, 1.4)

    result = detect_sources(
        image,
        threshold_sigma=5.0,
        min_distance=4,
        aperture_radius=4,
        proposal_mode="hybrid",
        dog_threshold_sigma=5.0,
    )

    near = [
        source
        for source in result.sources
        if (float(source.peak_x) - 48.0) ** 2 + (float(source.peak_y) - 48.0) ** 2 < 9.0
    ]
    assert len(near) == 1
    assert near[0].quality_passed
    assert near[0].proposal_methods[0] == "gaussian"
    assert {"dog_narrow", "dog_broad"}.issubset(near[0].proposal_methods)
    assert result.parameters["proposal_count_merged"] == result.candidate_count


def test_ensemble_adds_starlet_evidence_without_duplicate_sources() -> None:
    rng = np.random.default_rng(1920)
    image = rng.normal(100.0, 2.0, size=(96, 96))
    _add_gaussian(image, 48.0, 48.0, 160.0, 1.4)

    result = detect_sources(
        image,
        threshold_sigma=5.0,
        min_distance=4,
        aperture_radius=4,
        proposal_mode="ensemble",
        dog_threshold_sigma=5.0,
        starlet_threshold_sigma=4.0,
    )

    near = [
        source
        for source in result.sources
        if (float(source.peak_x) - 48.0) ** 2 + (float(source.peak_y) - 48.0) ** 2 < 9.0
    ]
    assert len(near) == 1
    assert near[0].quality_passed
    assert "gaussian" in near[0].proposal_methods
    assert any(method.startswith("starlet_") for method in near[0].proposal_methods)
    assert int(result.parameters["proposal_count_starlet_s1"]) > 0
    assert int(result.parameters["proposal_count_starlet_s2"]) > 0
    assert result.parameters["proposal_count_merged"] == result.candidate_count


def test_dog_only_near_neighbor_is_released_only_after_pair_psf_evidence() -> None:
    rng = np.random.default_rng(3100)
    image = rng.normal(100.0, 2.0, size=(96, 96))
    _add_gaussian(image, 40.0, 48.0, 160.0, 1.4)
    _add_gaussian(image, 46.0, 48.0, 40.0, 0.8)

    result = detect_sources(
        image,
        threshold_sigma=6.0,
        min_distance=4,
        aperture_radius=4,
        min_flux_snr=2.0,
        proposal_mode="hybrid",
        dog_threshold_sigma=5.0,
    )

    dog_only = [source for source in result.sources if "gaussian" not in source.proposal_methods]
    assert len(dog_only) == 1
    assert dog_only[0].nearest_gaussian_px == pytest.approx(6.0)
    assert "UNRESOLVED_BLEND" not in dog_only[0].flags
    assert dog_only[0].deblend_delta_bic is not None and dog_only[0].deblend_delta_bic > 10.0
    assert dog_only[0].deblend_component_snr is not None and dog_only[0].deblend_component_snr > 5.0
    assert dog_only[0].quality_passed


def test_local_deblend_recovers_a_companion_hidden_inside_one_response_peak() -> None:
    rng = np.random.default_rng(4004)
    image = rng.normal(100.0, 2.0, size=(96, 96))
    _add_gaussian(image, 40.0, 48.0, 160.0, 1.4)
    _add_gaussian(image, 44.0, 48.0, 40.0, 1.4)

    result = detect_sources(
        image,
        mask=auxiliary_mask(image.shape),
        threshold_sigma=6.0,
        min_distance=4,
        aperture_radius=4,
        min_flux_snr=2.0,
        proposal_mode="gaussian",
        enable_local_deblend=True,
    )

    deblended = [source for source in result.sources if "deblend_local" in source.proposal_methods]
    assert len(deblended) == 1
    assert deblended[0].deblend_delta_bic is not None and deblended[0].deblend_delta_bic >= 10.0
    assert deblended[0].deblend_component_snr is not None and deblended[0].deblend_component_snr >= 5.0
    assert deblended[0].quality_passed
    assert result.star_count == 2
    assert np.hypot(float(deblended[0].peak_x) - 44.0, float(deblended[0].peak_y) - 48.0) <= 3.0


def test_local_deblend_does_not_split_an_isolated_gaussian_star() -> None:
    rng = np.random.default_rng(3111)
    image = rng.normal(100.0, 2.0, size=(96, 96))
    _add_gaussian(image, 40.0, 48.0, 160.0, 1.4)

    result = detect_sources(
        image,
        mask=auxiliary_mask(image.shape),
        threshold_sigma=6.0,
        min_distance=4,
        aperture_radius=4,
        min_flux_snr=2.0,
        proposal_mode="gaussian",
    )

    assert not [source for source in result.sources if "deblend_local" in source.proposal_methods]
    assert result.star_count == 1


def test_pair_psf_evidence_rejects_a_bright_star_wing_without_a_second_source() -> None:
    rng = np.random.default_rng(3101)
    single = rng.normal(100.0, 2.0, size=(96, 96))
    _add_gaussian(single, 40.0, 48.0, 160.0, 1.4)
    pair = single.copy()
    _add_gaussian(pair, 46.0, 48.0, 40.0, 1.4)
    mask = np.zeros(single.shape, dtype=bool)

    single_delta, single_snr = _pair_psf_evidence(
        single,
        mask,
        (40.0, 48.0),
        (46.0, 48.0),
        psf_fwhm=3.0,
    )
    pair_delta, pair_snr = _pair_psf_evidence(
        pair,
        mask,
        (40.0, 48.0),
        (46.0, 48.0),
        psf_fwhm=3.0,
    )

    assert single_delta is None or single_delta < 10.0 or single_snr is None or single_snr < 5.0
    assert pair_delta is not None and pair_delta > 10.0
    assert pair_snr is not None and pair_snr > 5.0


def test_detect_sources_rejects_unknown_proposal_mode() -> None:
    with pytest.raises(ValueError, match="proposal_mode"):
        detect_sources(np.ones((32, 32), dtype=float), proposal_mode="unknown")


def test_detect_sources_keeps_integer_peak_and_reports_centroid_shift() -> None:
    rng = np.random.default_rng(19019)
    image = rng.normal(100.0, 2.0, size=(96, 96))
    _add_gaussian(image, 48.35, 41.70, 160.0, 1.5)

    result = detect_sources(image, threshold_sigma=5.0, min_distance=4, aperture_radius=4)

    source = min(result.sources, key=lambda item: np.hypot(item.x - 48.35, item.y - 41.70))
    assert source.peak_x is not None and source.peak_y is not None
    assert source.peak_x == round(source.peak_x)
    assert source.peak_y == round(source.peak_y)
    assert source.centroid_shift_px is not None
    assert source.centroid_shift_px == pytest.approx(np.hypot(source.x - source.peak_x, source.y - source.peak_y))
    assert source.centroid_shift_px < 2.0


def test_float32_sequence_path_preserves_integer_adu_candidates() -> None:
    image = np.full((64, 64), 100, dtype=np.int16)
    image[20, 20] = 1_000
    image[44, 45] = 800

    precise = detect_sources(image, threshold_sigma=4.0, min_distance=4, aperture_radius=2)
    fast = detect_sources(image, threshold_sigma=4.0, min_distance=4, aperture_radius=2, use_float32=True)

    assert fast.candidate_count == precise.candidate_count
    assert [(source.x, source.y) for source in fast.sources] == [(source.x, source.y) for source in precise.sources]
    assert fast.parameters["use_float32"] == 1
    assert precise.parameters["use_float32"] == 0


def test_shared_background_model_preserves_integer_peak_coordinates() -> None:
    image = np.full((96, 96), 100, dtype=np.int16)
    image[20, 20] = 1_000
    image[70, 64] = 800
    model = build_background_model(
        image,
        mask=auxiliary_mask(image.shape),
        background_box_size=32,
        background_sample_limit=512,
        refine_local_background=False,
        use_float32=True,
    )

    local = detect_sources(
        image,
        threshold_sigma=4.0,
        min_distance=4,
        aperture_radius=2,
        background_box_size=32,
        background_sample_limit=512,
        refine_local_background=False,
        use_float32=True,
    )
    shared = detect_sources(
        image,
        threshold_sigma=4.0,
        min_distance=4,
        aperture_radius=2,
        background_box_size=32,
        background_sample_limit=512,
        refine_local_background=False,
        use_float32=True,
        background_model=model,
    )

    assert shared.parameters["background_model_mode"] == "shared_sequence_pilot"
    assert [(source.x, source.y) for source in shared.sources] == [(source.x, source.y) for source in local.sources]


def test_detect_sources_reports_progress_through_source_measurement() -> None:
    image = np.zeros((48, 48), dtype=float)
    image[24, 24] = 100.0
    events: list[tuple[float, str]] = []

    detect_sources(image, threshold_sigma=4.0, min_distance=3, aperture_radius=2, progress=lambda value, label: events.append((value, label)))

    assert events
    assert events[0][1] == "准备检测"
    assert events[-1][0] == 98.0
    assert any("源级测量" in label for _value, label in events)


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


def test_detect_sources_rejects_an_isolated_impulse_without_psf_support() -> None:
    rng = np.random.default_rng(19031)
    image = rng.normal(100.0, 2.0, size=(96, 96))
    image[48, 52] += 400.0

    result = detect_sources(image, threshold_sigma=5.0, min_distance=4, aperture_radius=4)

    source = min(result.sources, key=lambda item: np.hypot(item.peak_x - 52.0, item.peak_y - 48.0))
    assert source.psf_support_pixels is not None
    assert source.psf_support_pixels <= 2
    assert "INSUFFICIENT_PSF_SUPPORT" in source.flags
    assert not source.quality_passed


def test_detect_sources_rejects_an_impossible_psf_support_requirement() -> None:
    image = np.full((48, 48), 100.0)

    with pytest.raises(ValueError, match="between 1 and 9"):
        detect_sources(image, min_psf_support_pixels=10)


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


def test_detect_sources_tolerates_an_isolated_auto_zero_inside_a_source_aperture() -> None:
    rng = np.random.default_rng(3101)
    image = rng.normal(20.0, 2.0, size=(96, 96))
    _add_gaussian(image, 48.0, 48.0, 1_000.0, 1.4)
    image = np.rint(image).astype(np.int16)
    # This is a sparse exact-zero bad pixel in the outer part of the aperture,
    # not an arbitrary caller-supplied mask. The source center remains valid.
    image[44, 48] = 0

    result = detect_sources(
        image,
        threshold_sigma=5.0,
        min_distance=4,
        aperture_radius=4,
        mask_zero_pixels=True,
    )

    source = min(result.sources, key=lambda item: np.hypot(item.peak_x - 48.0, item.peak_y - 48.0))
    assert "PARTIAL_MASKED" in source.flags
    assert "MASKED" not in source.flags
    assert source.quality_passed


def test_detect_sources_marks_an_in_aperture_saturated_pixel() -> None:
    rng = np.random.default_rng(32)
    image = np.rint(rng.normal(20.0, 2.0, size=(64, 64))).astype(np.int16)
    image[31:34, 31:34] += 500
    image[32, 35] = 32750

    result = detect_sources(image, threshold_sigma=5.0, min_distance=4, aperture_radius=4)

    assert any("SATURATED" in source.flags for source in result.sources)
    assert all(not source.quality_passed for source in result.sources if "SATURATED" in source.flags)


def test_detect_sources_rejects_extreme_negative_signed_code_in_aperture() -> None:
    rng = np.random.default_rng(33)
    image = rng.normal(20.0, 2.0, size=(96, 96))
    _add_gaussian(image, 48.0, 48.0, 1_000.0, 1.4)
    image = np.rint(image).astype(np.int16)
    # A negative excursion close to the signed int16 limit is not ordinary
    # background noise in this data format. It must remain in the candidate
    # audit but invalidate aperture photometry for the affected source.
    image[49, 50] = -32_000

    result = detect_sources(image, threshold_sigma=5.0, min_distance=4, aperture_radius=4)

    source = min(result.sources, key=lambda item: np.hypot(item.peak_x - 48.0, item.peak_y - 48.0))
    assert result.parameters["negative_overflow_limit"] == pytest.approx(-0.9 * np.iinfo(np.int16).max)
    assert "NEGATIVE_OVERFLOW" in source.flags
    assert not source.quality_passed


def test_repeated_high_code_audit_uses_full_frame_frequency() -> None:
    image = np.full((256, 256), 20, dtype=np.int16)
    image.flat[:160] = 3991
    mask = np.zeros(image.shape, dtype=bool)

    codes = _repeated_high_code_values(image, mask, background=20.0, noise=2.0)

    assert codes == (3991,)


def test_code_pattern_rejects_a_local_high_code_and_negative_pair() -> None:
    # 候选峰本身落在全幅异常重复的高位码上，且孔径还含远离背景的负值，
    # 才构成 CODE_PATTERN。这里把峰值直接设为重复码 3991。
    image = np.full((64, 64), 20, dtype=np.int16)
    image[32, 32] = 3_991
    image[31, 32] = -20_000
    mask = np.zeros(image.shape, dtype=bool)
    background = np.full(image.shape, 20.0)
    noise = np.full(image.shape, 2.0)

    source = _source_from_peak(
        image.astype(np.float64),
        mask,
        None,
        32,
        32,
        background,
        noise,
        aperture_radius=4,
        saturation_level=None,
        filter_snr=20.0,
        min_flux_snr=2.0,
        min_fwhm=0.8,
        max_fwhm=12.0,
        max_ellipticity=0.65,
        min_sharpness=0.005,
        max_sharpness=0.85,
        min_footprint_pixels=2,
        min_psf_support_pixels=3,
        gain_e_per_adu=None,
        read_noise_adu=0.0,
        refine_local_background=True,
        psf_fwhm=2.0,
        proposal_methods=("gaussian",),
        proposal_scales=(2.0,),
        proposal_snr=20.0,
        nearest_gaussian_px=None,
        dog_blend_radius_px=5.0,
        negative_overflow_limit=-29_490.3,
        repeated_high_code_values=(3991,),
    )

    assert source.repeated_code_count == 1
    assert source.range_anomaly_pixel_count == 1
    assert "CODE_PATTERN" in source.flags
    assert not source.quality_passed


def test_code_pattern_does_not_reject_bright_peak_with_bleed_codes() -> None:
    # 真实首帧的亮星会在饱和/溢出出血列里同时带入 3991/3992 等重复码
    # 和 -20628 这类负值，但峰值仍是一个正常高亮度值（如 20893）。这种
    # 星点不能被 CODE_PATTERN 误拒：只有峰值本身落在重复码上才算。
    image = np.full((64, 64), 20, dtype=np.int16)
    image[32, 32] = 20_893
    image[32, 31] = 3_991
    image[31, 32] = -20_628
    mask = np.zeros(image.shape, dtype=bool)
    background = np.full(image.shape, 20.0)
    noise = np.full(image.shape, 2.0)

    source = _source_from_peak(
        image.astype(np.float64),
        mask,
        None,
        32,
        32,
        background,
        noise,
        aperture_radius=4,
        saturation_level=None,
        filter_snr=20.0,
        min_flux_snr=2.0,
        min_fwhm=0.8,
        max_fwhm=12.0,
        max_ellipticity=0.65,
        min_sharpness=0.005,
        max_sharpness=0.85,
        min_footprint_pixels=2,
        min_psf_support_pixels=3,
        gain_e_per_adu=None,
        read_noise_adu=0.0,
        refine_local_background=True,
        psf_fwhm=2.0,
        proposal_methods=("gaussian",),
        proposal_scales=(2.0,),
        proposal_snr=20.0,
        nearest_gaussian_px=None,
        dog_blend_radius_px=5.0,
        negative_overflow_limit=-29_490.3,
        repeated_high_code_values=(3991,),
    )

    assert source.repeated_code_count == 1
    assert source.range_anomaly_pixel_count == 1
    assert "CODE_PATTERN" not in source.flags


def test_code_pattern_does_not_count_masked_negative_pixels() -> None:
    image = np.full((64, 64), 20, dtype=np.int16)
    image[32, 32] = 3_991
    image[31, 32] = -20_000
    mask = np.zeros(image.shape, dtype=bool)
    mask[31, 32] = True
    background = np.full(image.shape, 20.0)
    noise = np.full(image.shape, 2.0)

    source = _source_from_peak(
        image.astype(np.float64),
        mask,
        None,
        32,
        32,
        background,
        noise,
        aperture_radius=4,
        saturation_level=None,
        filter_snr=20.0,
        min_flux_snr=2.0,
        min_fwhm=0.8,
        max_fwhm=12.0,
        max_ellipticity=0.65,
        min_sharpness=0.005,
        max_sharpness=0.85,
        min_footprint_pixels=2,
        min_psf_support_pixels=3,
        gain_e_per_adu=None,
        read_noise_adu=0.0,
        refine_local_background=True,
        psf_fwhm=2.0,
        proposal_methods=("gaussian",),
        proposal_scales=(2.0,),
        proposal_snr=20.0,
        nearest_gaussian_px=None,
        dog_blend_radius_px=5.0,
        negative_overflow_limit=-29_490.3,
        repeated_high_code_values=(3991,),
    )

    assert source.repeated_code_count == 1
    assert source.range_anomaly_pixel_count == 0
    assert "CODE_PATTERN" not in source.flags


def test_close_gaussian_pair_guard_keeps_stronger_source(monkeypatch) -> None:
    image = np.zeros((96, 96), dtype=np.float64)
    mask = np.zeros(image.shape, dtype=bool)
    sources = [
        Detection(
            detection_id=1,
            x=40.0,
            y=48.0,
            peak=200.0,
            flux=500.0,
            background=20.0,
            noise=2.0,
            snr=90.0,
            fwhm=2.0,
            flags=(),
            filter_snr=100.0,
            quality_passed=True,
            peak_x=40.0,
            peak_y=48.0,
            proposal_methods=("gaussian",),
        ),
        Detection(
            detection_id=2,
            x=42.5,
            y=48.0,
            peak=80.0,
            flux=220.0,
            background=20.0,
            noise=2.0,
            snr=30.0,
            fwhm=2.0,
            flags=(),
            filter_snr=40.0,
            quality_passed=True,
            peak_x=42.0,
            peak_y=48.0,
            proposal_methods=("gaussian",),
        ),
    ]
    monkeypatch.setattr("rst19.detection._pair_psf_evidence", lambda *args, **kwargs: (0.0, 1.0))

    guarded, tested, guarded_count, radius = _guard_unresolved_gaussian_pairs(
        sources,
        image,
        mask,
        psf_fwhm=2.0,
        delta_bic_min=10.0,
        component_snr_min=5.0,
    )

    assert tested == 1
    assert guarded_count == 1
    assert radius == pytest.approx(3.0)
    assert guarded[0].quality_passed
    assert not guarded[1].quality_passed
    assert "UNRESOLVED_BLEND" in guarded[1].flags


def test_detect_sources_can_keep_original_negative_code_limit_after_float_conversion() -> None:
    rng = np.random.default_rng(34)
    image = rng.normal(20.0, 2.0, size=(96, 96))
    _add_gaussian(image, 48.0, 48.0, 1_000.0, 1.4)
    image[49, 50] = -32_000.0

    result = detect_sources(
        image.astype(np.float32),
        threshold_sigma=5.0,
        min_distance=4,
        aperture_radius=4,
        negative_overflow_limit=-0.9 * np.iinfo(np.int16).max,
    )

    source = min(result.sources, key=lambda item: np.hypot(item.peak_x - 48.0, item.peak_y - 48.0))
    assert result.parameters["negative_overflow_limit"] == pytest.approx(-0.9 * np.iinfo(np.int16).max)
    assert "NEGATIVE_OVERFLOW" in source.flags
    assert not source.quality_passed
