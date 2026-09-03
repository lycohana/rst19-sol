from __future__ import annotations

import queue
from types import SimpleNamespace

from rst19.gui import (
    GUI_DEFAULT_CANDIDATE_CONSENSUS_MIN_SNR,
    GUI_DEFAULT_LOCAL_DEBLEND,
    GUI_DEFAULT_MIN_DISTANCE,
    GUI_DEFAULT_MIN_FLUX_SNR,
    GUI_DEFAULT_PROPOSAL_MODE,
    GUI_DEFAULT_PSF_FWHM,
    GUI_DEFAULT_SEQUENCE_FULL,
    GUI_DEFAULT_TEMPORAL_CANDIDATE_MIN_SNR,
    GUI_DEFAULT_TEMPORAL_MIN_PSF_CORRELATION,
    GUI_DEFAULT_TEMPORAL_MULTISCALE,
    GUI_DEFAULT_TEMPORAL_PROPOSAL_MODE,
    GUI_DEFAULT_TEMPORAL_REFERENCE_MIN_SNR,
    GUI_DEFAULT_THRESHOLD_SIGMA,
    StarfieldApp,
    _make_preview,
    _make_preview_variants,
    source_quality_reason,
)
from rst19.sequence import FrameSequenceSummary, MotionFeaturePoint, MotionFeatureTrack, SequenceResult
from rst19.wcs import AffineWCSCalibration


def test_gui_defaults_use_current_balanced_high_recall_profile() -> None:
    """单帧和序列 UI 的初始值必须与已复测的默认口径保持一致。"""

    assert GUI_DEFAULT_THRESHOLD_SIGMA == 4.0
    assert GUI_DEFAULT_MIN_DISTANCE == 4
    assert GUI_DEFAULT_PSF_FWHM == 2.0
    assert GUI_DEFAULT_MIN_FLUX_SNR == 5.0
    assert GUI_DEFAULT_PROPOSAL_MODE == "hybrid"
    assert GUI_DEFAULT_LOCAL_DEBLEND is False
    assert GUI_DEFAULT_SEQUENCE_FULL is False
    assert GUI_DEFAULT_TEMPORAL_PROPOSAL_MODE == "median"
    assert GUI_DEFAULT_CANDIDATE_CONSENSUS_MIN_SNR == 15.0
    assert GUI_DEFAULT_TEMPORAL_CANDIDATE_MIN_SNR == 7.5
    assert GUI_DEFAULT_TEMPORAL_REFERENCE_MIN_SNR == 15.0
    assert GUI_DEFAULT_TEMPORAL_MULTISCALE is False
    assert GUI_DEFAULT_TEMPORAL_MIN_PSF_CORRELATION == 0.8


def test_sequence_evidence_summary_uses_frame_metadata_without_starting_tk() -> None:
    frames = tuple(
        FrameSequenceSummary(
            index,
            f"frame-{index}.fits",
            100,
            100,
            50,
            timestamp=timestamp,
            exposure_ms=1500.0,
            auxiliary=(
                ("roll", 10.0 + index),
                ("pitch", 20.0 + index * 0.5),
                ("yaw", 30.0 - index * 0.25),
            ),
            width_px=4096,
            height_px=4096,
        )
        for index, timestamp in enumerate(("2026-03-30T16:32:05.4132", "2026-03-30T16:32:06.8862", "2026-03-30T16:32:08.4152"))
    )
    points = tuple(
        MotionFeaturePoint(index, 100.0 + index * 10.0, 200.0 - index * 4.0, 100.0 + index * 10.0, 200.0 - index * 4.0, 20.0, 40, 50.0, 3.0, -21.8, (0, 0, 10, 10), False)
        for index in range(3)
    )
    moving = MotionFeatureTrack(4, "moving", points, 22.4, 11.2, 0.1)
    result = SequenceResult(
        frames,
        ((0.0, 0.0), (0.1, 0.2), (0.2, 0.3)),
        (),
        4.0,
        2,
        2.0,
        0.75,
        (moving,),
        source_working_limit=6000,
    )
    label = SimpleNamespace(text="")
    label.config = lambda **kwargs: setattr(label, "text", kwargs["text"])
    app = object.__new__(StarfieldApp)
    app.sequence_evidence_label = label

    app._render_sequence_evidence(result)

    assert "3 帧" in label.text
    assert "4096×4096" in label.text
    assert "moving" in label.text
    assert "姿态变化" in label.text
    assert "无 WCS/像元尺度" in label.text
    assert "前 6,000" in label.text


def test_motion_overlay_uses_each_point_frame_registration_shift() -> None:
    points = tuple(
        MotionFeaturePoint(
            index,
            100.0 + index * 10.0,
            100.0,
            100.0 + index * 10.0,
            100.0,
            20.0,
            40,
            30.0,
            3.0,
            0.0,
            (90, 90, 120, 110),
            False,
        )
        for index in range(3)
    )
    track = MotionFeatureTrack(1, "moving", points, 20.0, 10.0, 0.1)
    frame = FrameSequenceSummary(0, "frame.fits", 1, 1, 1)
    result = SequenceResult(
        (frame, frame, frame),
        ((0.0, 0.0), (2.0, 0.0), (5.0, 0.0)),
        (),
        4.0,
        2,
        2.0,
        0.75,
        (track,),
    )

    class CaptureDraw:
        def __init__(self) -> None:
            self.lines: list[object] = []

        def line(self, points: object, **_kwargs: object) -> None:
            self.lines.append(points)

        def ellipse(self, *_args: object, **_kwargs: object) -> None:
            return None

    app = object.__new__(StarfieldApp)
    app.sequence_result = result
    app.preview_scale_x = 1.0
    app.preview_scale_y = 1.0
    app._draw_image_label = lambda *_args: None
    draw = CaptureDraw()

    app._draw_motion_trajectory_overlay(draw, 2, 0, 0, 1.0, (500, 500))

    assert draw.lines[0] == [(100.0, 100.0), (112.0, 100.0), (125.0, 100.0)]


def test_sequence_progress_is_rendered_in_the_main_task_state() -> None:
    app = object.__new__(StarfieldApp)
    app.result_queue = queue.Queue()
    app.frame_token = 7
    app.active_job_kind = "sequence"
    app.status_var = SimpleNamespace(set=lambda value: setattr(app.status_var, "value", value), value="")
    progress: list[tuple[float, str]] = []
    app._set_progress = lambda value, text=None: progress.append((float(value), str(text)))
    app._set_sequence_progress = lambda value, text=None: progress.append((float(value), f"sequence:{text}"))
    app.sequence_frame_states = ["pending"] * 15
    app.after = lambda *_args: None
    app.result_queue.put(("analysis-progress", 7, (63.0, "完成 8/15 · F08 源级测量")))

    app._poll_result()

    assert progress == [
        (63.0, "完成 8/15 · F08 源级测量"),
        (63.0, "sequence:完成 8/15 · F08 源级测量"),
    ]
    assert "完成 8/15" in app.status_var.value
    assert app.sequence_frame_states[:8] == ["completed"] * 8
    assert app.sequence_frame_states[8:] == ["pending"] * 7


def test_sequence_progress_is_visible_in_the_right_evidence_card() -> None:
    app = object.__new__(StarfieldApp)
    label = SimpleNamespace(text="")
    label.config = lambda **kwargs: setattr(label, "text", kwargs["text"])
    app.sequence_evidence_label = label
    app.active_job_kind = "sequence"

    app._render_running_progress(63.0, "完成 8/15 · F08 源级测量")

    assert "运行中 · 63%" in label.text
    assert "完成 8/15" in label.text
    assert "逐帧检测完成后" in label.text


def test_sequence_completion_shows_stable_overlay_by_default() -> None:
    frame = FrameSequenceSummary(0, "frame-0.fits", 10, 8, 3)
    result = SequenceResult(
        (frame, frame),
        ((0.0, 0.0), (0.0, 0.0)),
        (),
        4.0,
        2,
        2.0,
        0.75,
    )
    app = object.__new__(StarfieldApp)
    app.result_queue = queue.Queue()
    app.frame_token = 5
    app.active_job_token = 5
    app.busy = True
    app.active_job_kind = "sequence"
    app.long_trails = ()
    app.overlay_mode_var = SimpleNamespace(value="motion")
    app.overlay_mode_var.set = lambda value: setattr(app.overlay_mode_var, "value", value)
    app.status_var = SimpleNamespace(value="")
    app.status_var.set = lambda value: setattr(app.status_var, "value", value)
    app._set_job_controls = lambda: None
    app._set_progress = lambda *_args: None
    app._set_sequence_progress = lambda *_args: None
    app._mark_sequence_ledger = lambda *_args: None
    app._update_overlay_hint = lambda: None
    app._render_sequence_evidence = lambda *_args: None
    app._draw_preview = lambda: None
    app.after = lambda *_args: None
    app.result_queue.put(("sequence", 5, (result, "序列缓存命中", True)))

    app._poll_result()

    assert app.overlay_mode_var.value == "stable"
    assert "当前显示稳定星场" in app.status_var.value


def test_preview_zoom_caps_at_fifteen_times() -> None:
    app = object.__new__(StarfieldApp)
    app.preview = SimpleNamespace(width=1600, height=1600)
    app.preview_zoom = 14.0
    app.preview_pan_x = 0.0
    app.preview_pan_y = 0.0
    app.canvas = SimpleNamespace(winfo_width=lambda: 800, winfo_height=lambda: 600)
    app.status_var = SimpleNamespace(set=lambda value: setattr(app.status_var, "value", value), value="")
    app._draw_preview = lambda: None

    app._zoom_at(SimpleNamespace(x=300, y=220), 2.0)

    assert app.preview_zoom == 15.0
    assert "15.00×" in app.status_var.value


def test_preview_nonlinear_stretch_keeps_weak_peak_above_background() -> None:
    import numpy as np

    image = np.full((128, 128), 21.0, dtype=np.float32)
    image[64, 64] = 32.0
    image[20, 20] = 195.0
    image[30, 30] = 500.0

    preview = _make_preview(image, max_side=128)

    assert preview.getpixel((64, 64))[0] > preview.getpixel((63, 64))[0]


def test_preview_keeps_background_dark_after_noise_suppression() -> None:
    import numpy as np

    rng = np.random.default_rng(19)
    image = rng.normal(21.0, 8.0, (128, 128)).astype(np.float32)
    image[64, 64] = 195.0

    preview = _make_preview(image, max_side=128)

    pixels = np.asarray(preview)[..., 0]
    background = float(np.percentile(pixels, 50))
    source = preview.getpixel((64, 64))[0]
    assert background < 40
    assert source > background + 40


def test_preview_default_keeps_sensor_resolution_for_high_zoom() -> None:
    import numpy as np

    image = np.full((1601, 1601), 21.0, dtype=np.float32)
    image[800, 800] = 32.0
    image[400, 400] = 500.0

    preview = _make_preview(image)

    assert preview.size == (1601, 1601)


def test_preview_variants_include_raw_and_noise_views_without_changing_input() -> None:
    import numpy as np

    image = np.full((64, 64), 21.0, dtype=np.float32)
    image[32, 32] = 80.0
    original = image.copy()

    variants = _make_preview_variants(image, max_side=64)

    assert set(variants) == {"enhanced", "raw", "noise"}
    assert all(preview.size == (64, 64) for preview in variants.values())
    assert np.array_equal(image, original)
    # 增亮噪声层不做显示层高斯平滑，邻近背景像素仍能保留各自的颗粒差异。
    noisy = np.random.default_rng(19).normal(21.0, 8.0, (64, 64)).astype(np.float32)
    noisy_variants = _make_preview_variants(noisy, max_side=64)
    noise_pixels = np.asarray(noisy_variants["noise"])[..., 0]
    assert len(np.unique(noise_pixels)) > 20


def test_preview_completion_clears_loading_status_and_refreshes_controls() -> None:
    app = object.__new__(StarfieldApp)
    preview = SimpleNamespace()
    app.preview_mode_var = SimpleNamespace(get=lambda: "enhanced")
    app.hover_info_var = SimpleNamespace(value="")
    app.hover_info_var.set = lambda value: setattr(app.hover_info_var, "value", value)
    app.status_var = SimpleNamespace(value="正在载入预览")
    app.status_var.set = lambda value: setattr(app.status_var, "value", value)
    calls: list[str] = []
    app._set_job_controls = lambda: calls.append("controls")
    app._update_overlay_hint = lambda: calls.append("hint")
    app._draw_preview = lambda: calls.append("draw")

    app._apply_preview_result((4096, 4096), {"enhanced": preview})

    assert app.preview is preview
    assert app.preview_shape == (4096, 4096)
    assert app.preview_zoom == 1.0
    assert "预览已载入" in app.status_var.value
    assert "正在载入" not in app.status_var.value
    assert "分析当前帧" in app.hover_info_var.value
    assert calls == ["controls", "hint", "draw"]


def test_source_quality_reason_explains_rejection_and_compact_label() -> None:
    source = SimpleNamespace(
        quality_passed=False,
        flags=("MASKED", "LOW_FLUX_SNR", "INSUFFICIENT_PSF_SUPPORT"),
        flux_snr=4.2,
        psf_support_pixels=1,
    )

    detailed = source_quality_reason(
        source,
        {"min_flux_snr": 5.0, "min_psf_support_pixels": 3},
    )
    compact = source_quality_reason(source, compact=True)

    assert "孔径内含掩膜" in detailed
    assert "通量 SNR 4.20 < 门槛 5.00" in detailed
    assert "PSF 支持 1/3" in detailed
    assert compact == "MASKED; LOW SNR; LOW PSF"


def test_source_quality_reason_explains_negative_overflow() -> None:
    source = SimpleNamespace(
        quality_passed=False,
        flags=("NEGATIVE_OVERFLOW",),
        flux_snr=10.0,
        psf_support_pixels=4,
    )

    detailed = source_quality_reason(
        source,
        {"min_flux_snr": 5.0, "min_psf_support_pixels": 3, "negative_overflow_limit": -29490.3},
    )
    compact = source_quality_reason(source, compact=True)

    assert "疑似有符号溢出/饱和邻域" in detailed
    assert "RANGE?" in compact


def test_source_hover_exposes_snr_provenance_and_rejection_reason() -> None:
    source = SimpleNamespace(
        detection_id=18836,
        x=11.5,
        y=22.5,
        peak_x=12.0,
        peak_y=23.0,
        flux=42.0,
        flux_error=6.3,
        flux_snr=6.7,
        snr=8.1,
        filter_snr=7.5,
        fwhm=3.0,
        ellipticity=0.1,
        flags=("LOW_FLUX_SNR",),
        quality_passed=False,
        psf_support_pixels=4,
        centroid_shift_px=0.2,
        proposal_methods=("gaussian", "dog_narrow"),
        proposal_scales=(1.27, 2.41),
        proposal_snr=5.4,
        nearest_gaussian_px=1.3,
        deblend_delta_bic=2.0,
        deblend_component_snr=3.2,
    )
    app = object.__new__(StarfieldApp)
    app.overlay_mode_var = SimpleNamespace(get=lambda: "candidates")
    app.hover_info_var = SimpleNamespace(value="")
    app.hover_info_var.set = lambda value: setattr(app.hover_info_var, "value", value)
    app.hover_source_id = None
    app.hover_source = None
    app.hover_motion_track = None
    app.hover_catalog_match = None
    app.analysis = SimpleNamespace(
        detection=SimpleNamespace(
            parameters={
                "min_flux_snr": 7.0,
                "min_psf_support_pixels": 3,
                "dog_blend_radius_px": 4.0,
            }
        )
    )
    app.exposure_s = 1.5
    app._find_source_at = lambda _event: source
    app._draw_preview = lambda: None

    app._on_canvas_motion(SimpleNamespace(x=10, y=20))

    assert "ID 18836" in app.hover_info_var.value
    assert "flux SNR 6.70" in app.hover_info_var.value
    assert "filter SNR 7.50" in app.hover_info_var.value
    assert "提案 gaussian+dog_narrow" in app.hover_info_var.value
    assert "去混叠 ΔBIC 2.0" in app.hover_info_var.value
    assert "通量 SNR 6.70 < 门槛 7.00" in app.hover_info_var.value


def test_single_frame_trail_preview_is_rendered_before_full_analysis() -> None:
    point = MotionFeaturePoint(
        0,
        42.0,
        48.0,
        42.0,
        48.0,
        60.0,
        120,
        72.0,
        3.0,
        90.0,
        (40, 10, 44, 82),
        False,
    )
    track = MotionFeatureTrack(9, "candidate", (point,), 0.0, 0.0, None)
    app = object.__new__(StarfieldApp)
    app.result_queue = queue.Queue()
    app.frame_token = 7
    app.active_job_kind = "single"
    app.long_trails = ()
    app.status_var = SimpleNamespace(value="")
    app.status_var.set = lambda value: setattr(app.status_var, "value", value)
    app.overlay_mode_var = SimpleNamespace(value="")
    app.overlay_mode_var.set = lambda value: setattr(app.overlay_mode_var, "value", value)
    app._update_overlay_hint = lambda: None
    app._draw_preview = lambda: None
    app.after = lambda *_args: None
    app.result_queue.put(("trail-preview", 7, (track,)))

    app._poll_result()

    assert app.long_trails == (track,)
    assert app.overlay_mode_var.value == "motion"
    assert "单帧长线" in app.status_var.value


def test_innovation_summary_rows_expose_motion_and_wcs_boundaries() -> None:
    frames = tuple(
        FrameSequenceSummary(
            index,
            f"frame-{index}.fits",
            100 + index,
            100 + index,
            50 + index,
            timestamp=timestamp,
            exposure_ms=1500.0,
            width_px=4096,
            height_px=4096,
        )
        for index, timestamp in enumerate(
            (
                "2026-03-30T16:32:05.4132",
                "2026-03-30T16:32:06.8862",
                "2026-03-30T16:32:08.4152",
            )
        )
    )
    points = tuple(
        MotionFeaturePoint(
            index,
            100.0 + index * 10.0,
            200.0 - index * 4.0,
            100.0 + index * 10.0,
            200.0 - index * 4.0,
            20.0,
            40,
            50.0,
            3.0,
            -21.8,
            (0, 0, 10, 10),
            False,
        )
        for index in range(3)
    )
    moving = MotionFeatureTrack(4, "moving", points, 21.5, 10.75, 0.1)
    result = SequenceResult(frames, ((0.0, 0.0), (0.1, 0.2), (0.2, 0.3)), (), 4.0, 2, 2.0, 0.75, (moving,))

    rows = StarfieldApp._innovation_summary_rows(result)
    values = {(group, metric): value for group, metric, value, _meaning in rows}

    assert values[("时序关系", "观测帧数")] == "3 帧"
    assert values[("固定星场", "最大累计配准")] == "0.361 px"
    assert values[("线状 0004", "状态 / 帧范围")] == "moving 候选 · F01–F03"
    assert "px/s" in values[("线状 0004", "位移 / 速度")]
    assert "[95%" in values[("线状 0004", "位移 / 速度")]
    assert "° (95%)" in values[("线状 0004", "方向 / 拟合 RMS")]
    assert "+7.505 s" in values[("线状 0004", "短期图像外推")]
    assert "95% X±" in values[("线状 0004", "短期图像外推")]
    assert "/ Y±" in values[("线状 0004", "短期图像外推")]
    assert values[("结论边界", "WCS / 物理速度")] == "待标定"


def test_innovation_summary_adds_tangent_plane_rate_after_local_calibration() -> None:
    frames = tuple(
        FrameSequenceSummary(
            index,
            f"frame-{index}.fits",
            100,
            100,
            50,
            timestamp=timestamp,
            exposure_ms=1500.0,
            width_px=4096,
            height_px=4096,
        )
        for index, timestamp in enumerate(
            (
                "2026-03-30T16:32:05.4132",
                "2026-03-30T16:32:06.8862",
                "2026-03-30T16:32:08.4152",
            )
        )
    )
    points = tuple(
        MotionFeaturePoint(
            index,
            100.0 + index * 10.0,
            200.0 - index * 4.0,
            100.0 + index * 10.0,
            200.0 - index * 4.0,
            20.0,
            40,
            50.0,
            3.0,
            -21.8,
            (0, 0, 10, 10),
            False,
        )
        for index in range(3)
    )
    moving = MotionFeatureTrack(4, "moving", points, 21.5, 10.75, 0.1)
    result = SequenceResult(frames, ((0.0, 0.0), (0.1, 0.2), (0.2, 0.3)), (), 4.0, 2, 2.0, 0.75, (moving,))
    calibration = AffineWCSCalibration(
        center_ra_deg=10.0,
        center_dec_deg=20.0,
        matrix_px_per_arcsec=((1.0, 0.0), (0.0, 1.0)),
        offset_px=(50.0, 50.0),
        plate_scale_arcsec_per_pixel=1.0,
        rotation_deg=0.0,
        parity=1,
        anisotropy_ratio=1.0,
        matched_count=12,
        inlier_count=11,
        rms_residual_px=0.2,
        max_residual_px=0.5,
        all_rms_residual_px=0.3,
        all_max_residual_px=2.0,
        condition_number=2.0,
        inlier_source_ids=tuple(f"s{index}" for index in range(11)),
    )

    rows = StarfieldApp._innovation_summary_rows(result, calibration)
    values = {(group, metric): value for group, metric, value, _meaning in rows}

    assert "arcsec/s" in values[("线状 0004", "切平面角速度")]
    assert values[("WCS 校准", "匹配 / 内点")] == "11/12 (91.7%)"
    assert values[("结论边界", "WCS / 物理速度")] == "已有局部角尺度"
