from __future__ import annotations

import queue
from dataclasses import replace
from pathlib import Path
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
    _contiguous_motion_point_runs,
    _format_observed_frame_ranges,
    _motion_overlay_candidate_is_allowed,
    _single_frame_model_is_review_candidate,
    _single_frame_exposure_rate_text,
    _make_preview,
    _make_preview_variants,
    source_quality_reason,
)


def test_single_frame_exposure_rate_uses_exposure_and_marks_edge_lower_bound() -> None:
    interior = MotionFeaturePoint(
        0, 1.0, 1.0, 1.0, 1.0, 10.0, 20, 30.0, 2.0, 0.0, (0, 0, 4, 4), False
    )
    edge = MotionFeaturePoint(
        0, 1.0, 1.0, 1.0, 1.0, 10.0, 20, 30.0, 2.0, 0.0, (0, 0, 4, 4), True
    )

    assert _single_frame_exposure_rate_text(interior, 1.5) == "线长估计/1.500s ≈ 20.0px/s"
    assert _single_frame_exposure_rate_text(edge, 1.5) == "可见线长下限/1.500s ≈ 20.0px/s"
from rst19.sequence import FrameSequenceSummary, MotionFeaturePoint, MotionFeatureTrack, SequenceResult, SourceTrack, TrackPoint
from rst19.wcs import AffineWCSCalibration


def test_gui_frame_ranges_show_reconnected_track_gaps() -> None:
    points = tuple(
        SimpleNamespace(frame_index=index)
        for index in (4, 5, 6, 7, 10, 11, 11)
    )

    assert _format_observed_frame_ranges(points) == "F05–F08、F11–F12"


def test_gui_history_renderer_splits_reconnected_track_runs() -> None:
    points = tuple(SimpleNamespace(frame_index=index) for index in (4, 5, 6, 7, 10, 11, 12, 13))

    runs = _contiguous_motion_point_runs(points)

    assert [[point.frame_index for point in run] for run in runs] == [
        [4, 5, 6, 7],
        [10, 11, 12, 13],
    ]


def test_single_frame_review_layer_keeps_a_line_model_above_weak_bic_boundary() -> None:
    point = MotionFeaturePoint(
        0,
        42.0,
        48.0,
        42.0,
        48.0,
        8.0,
        30,
        12.0,
        2.0,
        90.0,
        (36, 42, 48, 54),
        False,
        0.45,
        0.42,
        single_frame_model="line",
        line_template_snr=5.2,
        line_vs_point_delta_bic=2.8,
        line_vs_double_delta_bic=2.3,
        line_vs_background_delta_bic=2.1,
        line_vs_elliptical_delta_bic=2.0,
        single_frame_model_margin_bic=2.0,
    )
    track = SourceTrack(
        3,
        "candidate",
        (point,),
        0.0,
        0.0,
        None,
        evidence_level="single_frame_trail",
    )
    app = object.__new__(StarfieldApp)
    app.active_job_kind = "single"
    app.long_trails = (track,)

    assert _single_frame_model_is_review_candidate(point) is True
    assert app._motion_feature_points(0) == ()
    assert app._motion_feature_points(0, include_weak=True) == ((track, point),)


def test_single_frame_review_layer_hides_compact_source_bic_tie() -> None:
    """线模型几乎不胜椭圆模型时，不能把亮星画成 EXP TRAIL?。"""

    point = MotionFeaturePoint(
        0,
        1270.9,
        3464.3,
        1270.9,
        3464.3,
        3956.1,
        16,
        8.0,
        2.0,
        160.0,
        (1265, 3461, 1276, 3468),
        False,
        1.0,
        0.72,
        single_frame_model="line",
        line_template_snr=3560.1,
        line_vs_point_delta_bic=32.0,
        line_vs_double_delta_bic=8.8,
        line_vs_background_delta_bic=48.1,
        line_vs_elliptical_delta_bic=0.48,
        single_frame_model_margin_bic=0.48,
    )
    track = MotionFeatureTrack(
        11,
        "candidate",
        (point,),
        0.0,
        0.0,
        None,
        evidence_level="single_frame_trail",
    )
    app = object.__new__(StarfieldApp)
    app.active_job_kind = "single"
    app.long_trails = (track,)

    assert _single_frame_model_is_review_candidate(point) is False
    assert app._motion_feature_points(0) == ()
    assert app._motion_feature_points(0, include_weak=True) == ()


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


def test_single_frame_overlay_hint_does_not_claim_sequence_evidence() -> None:
    """单帧页的图例不能把跨帧 moving 语义带进当前曝光结果。"""

    class Value:
        def __init__(self, value: object) -> None:
            self.value = value

        def get(self) -> object:
            return self.value

        def set(self, value: object) -> None:
            self.value = value

    app = object.__new__(StarfieldApp)
    app.task_mode = "single"
    app.preview_mode_var = Value("enhanced")
    app.overlay_mode_var = Value("motion")
    app.motion_overlay_var = Value(False)
    app.overlay_hint_var = Value("")

    app._update_overlay_hint()

    assert "只使用当前这一张图" in app.overlay_hint_var.value
    assert "moving" not in app.overlay_hint_var.value
    assert "跨帧线状" not in app.overlay_hint_var.value
    assert "+5 帧" not in app.overlay_hint_var.value


def test_workspace_buttons_are_locked_while_analysis_is_running() -> None:
    class FakeButton:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def config(self, **kwargs: object) -> None:
            self.calls.append(kwargs)

    app = object.__new__(StarfieldApp)
    app.busy = True
    app.selected_frame = 0
    app.active_job_kind = "single"
    app.sequence_result = None
    app.single_mode_button = FakeButton()
    app.sequence_mode_button = FakeButton()

    app._set_job_controls()

    assert app.single_mode_button.calls == [{"state": "disabled"}]
    assert app.sequence_mode_button.calls == [{"state": "disabled"}]


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
            0.88,
            0.82,
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
    app.task_mode = "sequence"
    app.sequence_result = result
    app.preview_scale_x = 1.0
    app.preview_scale_y = 1.0
    app._draw_image_label = lambda *_args: None
    draw = CaptureDraw()

    app._draw_motion_trajectory_overlay(draw, 2, 0, 0, 1.0, (500, 500))

    # The path is stored in registered coordinates and is projected as a
    # whole onto the currently displayed raw frame (F03 shift = +5 px).
    assert draw.lines[0] == [(105.0, 100.0), (115.0, 100.0), (125.0, 100.0)]


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


def test_sequence_completion_shows_quality_overlay_by_default() -> None:
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

    assert app.overlay_mode_var.value == "quality"
    assert "当前显示可信源主层" in app.status_var.value


def test_sequence_completion_repopulates_main_frame_result_without_fake_photometry() -> None:
    frames = (
        FrameSequenceSummary(0, "frame-0.fits", 120, 80, 60),
        FrameSequenceSummary(1, "frame-1.fits", 130, 90, 70),
    )
    track = SourceTrack(
        3,
        "static",
        (TrackPoint(1, 42, 12.5, 20.5, 12.5, 20.5, 18.0),),
        0.0,
        0.0,
        None,
    )
    result = SequenceResult(frames, ((0.0, 0.0), (0.0, 0.0)), (track,), 4.0, 2, 2.0, 0.75)

    class Widget:
        def __init__(self) -> None:
            self.values: dict[str, object] = {}

        def config(self, **kwargs: object) -> None:
            self.values.update(kwargs)

    class Tree:
        def __init__(self) -> None:
            self.rows: list[tuple[object, ...]] = []

        def get_children(self) -> tuple[object, ...]:
            return ()

        def delete(self, _item: object) -> None:
            return None

        def insert(self, _parent: str, _position: object, *, values: tuple[object, ...]) -> None:
            self.rows.append(values)

    app = object.__new__(StarfieldApp)
    app.task_mode = "sequence"
    app.sequence_result = result
    app.selected_frame = Path("frame-1.fits")
    app.frames = [Path("frame-0.fits"), Path("frame-1.fits")]
    app.metric_values = {name: Widget() for name in ("candidate", "returned", "background", "noise", "faintest")}
    app.faintest_metric_title = Widget()
    app.faintest_detail = Widget()
    app.faintest_physical_label = Widget()
    app.faintest_note = Widget()
    app.table_count_label = Widget()
    app.source_tree = Tree()

    app._render_sequence_frame_result(result)

    assert app.metric_values["candidate"].values["text"] == "130"
    assert app.metric_values["returned"].values["text"] == "90"
    assert app.metric_values["background"].values["text"] == "—"
    assert app.metric_values["faintest"].values["text"] == "1"
    assert "当前 F02" in app.faintest_detail.values["text"]
    assert "已排除运动轨迹点" in app.faintest_detail.values["text"]
    assert "未保留" in app.faintest_physical_label.values["text"]
    assert app.table_count_label.values["text"] == "1 / 1 跨帧可信源"
    assert app.source_tree.rows == [("0042", "12.5 / 20.5", "序列层未保留", "18.0", "序列层未保留", "—", "—")]


def test_sequence_job_does_not_label_the_single_frame_button_as_running() -> None:
    class Button:
        def __init__(self) -> None:
            self.values: dict[str, object] = {}

        def config(self, **kwargs: object) -> None:
            self.values.update(kwargs)

    app = object.__new__(StarfieldApp)
    app.busy = True
    app.active_job_kind = "sequence"
    app.task_mode = "sequence"
    app.sequence_result = None
    app.mosaic_result = None
    app.selected_frame = Path("frame-0.fits")
    app.run_button = Button()
    app.motion_button = Button()
    app.mosaic_button = Button()
    app.cache_button = Button()
    app.evidence_button = Button()

    app._set_job_controls()

    assert app.run_button.values["text"] == "单帧分析（15帧运行中）"
    assert app.motion_button.values["text"] == "15 帧分析中…"


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
    assert "一键分析单张" in app.hover_info_var.value
    assert calls == ["controls", "hint", "draw"]


def test_preview_event_is_not_blocked_by_previous_single_frame_analysis() -> None:
    app = object.__new__(StarfieldApp)
    app._closing = False
    app.result_queue = queue.Queue()
    app.frame_token = 11
    app.cache_generation = 0
    app.selected_frame = Path("current.fits")
    app.analysis = SimpleNamespace()  # stale single-frame context is intentional
    app.after = lambda *_args: None
    received: list[tuple[object, object, object]] = []

    def apply_preview(shape: object, previews: object, *, frame_path: object = None) -> None:
        received.append((shape, previews, frame_path))

    app._apply_preview_result = apply_preview
    app.result_queue.put(
        ("preview", 11, (Path("current.fits"), (32, 32), {"enhanced": object()}))
    )

    app._poll_result()

    assert received
    assert received[0][0] == (32, 32)
    assert received[0][2] == Path("current.fits")


def test_sequence_frame_index_uses_positional_fallback_only_for_same_length() -> None:
    frame_a = FrameSequenceSummary(0, "local-a.fits", 10, 8, 3)
    frame_b = FrameSequenceSummary(1, "local-b.fits", 10, 8, 3)
    result = SequenceResult(
        (frame_a, frame_b),
        ((0.0, 0.0), (0.0, 0.0)),
        (),
        4.0,
        2,
        2.0,
        0.75,
    )
    app = object.__new__(StarfieldApp)
    app.task_mode = "sequence"
    app.sequence_result = result
    app.selected_frame = Path("local-b.fits")
    app.frames = [Path("local-a.fits"), Path("local-b.fits")]

    assert app._sequence_frame_index() == 1


def test_sequence_frame_index_does_not_use_same_length_for_a_different_queue() -> None:
    frame_a = FrameSequenceSummary(0, "other-a.fits", 10, 8, 3)
    frame_b = FrameSequenceSummary(1, "other-b.fits", 10, 8, 3)
    result = SequenceResult(
        (frame_a, frame_b),
        ((0.0, 0.0), (0.0, 0.0)),
        (),
        4.0,
        2,
        2.0,
        0.75,
    )
    app = object.__new__(StarfieldApp)
    app.task_mode = "sequence"
    app.sequence_result = result
    app.selected_frame = Path("local-b.fits")
    app.frames = [Path("local-a.fits"), Path("local-b.fits")]

    assert app._sequence_frame_index() is None


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
    app.source_photometry_by_id = {
        18836: SimpleNamespace(
            catalog_mg_gspphot=4.2,
            catalog_mg_gspphot_lower=3.9,
            catalog_mg_gspphot_upper=4.5,
            catalog_mg_gspphot_source="Gaia DR3 GSP-Phot: mg_gspphot",
            flags=(),
        )
    }
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
    assert "Gaia模型 M_G = 4.200 [3.900, 4.500]" in app.hover_info_var.value


def test_single_frame_trail_preview_renders_orange_shape_evidence_immediately() -> None:
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
        single_frame_model="line",
        line_template_snr=8.0,
        line_vs_point_delta_bic=10.0,
        line_vs_double_delta_bic=10.0,
        line_vs_elliptical_delta_bic=10.0,
    )
    track = MotionFeatureTrack(
        9,
        "candidate",
        (point,),
        0.0,
        0.0,
        None,
        evidence_level="single_frame_trail",
    )
    app = object.__new__(StarfieldApp)
    app.result_queue = queue.Queue()
    app.frame_token = 7
    app.active_job_kind = "single"
    app.long_trails = ()
    app.status_var = SimpleNamespace(value="")
    app.status_var.set = lambda value: setattr(app.status_var, "value", value)
    app.overlay_mode_var = SimpleNamespace(value="")
    app.overlay_mode_var.set = lambda value: setattr(app.overlay_mode_var, "value", value)
    app.motion_overlay_var = SimpleNamespace(get=lambda: True)
    app._update_overlay_hint = lambda: None
    draw_calls: list[bool] = []
    app._draw_preview = lambda: draw_calls.append(True)
    app.after = lambda *_args: None
    app.result_queue.put(("trail-preview", 7, (track,)))

    app._poll_result()

    assert app.long_trails == (track,)
    assert app.overlay_mode_var.value == "quality"
    assert draw_calls == [True]
    assert track.is_single_frame_motion_candidate is True
    assert "已先发现 1 条单帧曝光线形候选" in app.status_var.value
    assert "橙线已叠加" in app.status_var.value


def test_single_frame_motion_points_do_not_use_stale_sequence_result() -> None:
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
        0.88,
        0.82,
        single_frame_model="line",
        line_template_snr=8.0,
        line_vs_point_delta_bic=10.0,
        line_vs_double_delta_bic=10.0,
        line_vs_elliptical_delta_bic=10.0,
    )
    single_frame_track = MotionFeatureTrack(
        9,
        "candidate",
        (point,),
        0.0,
        0.0,
        None,
        evidence_level="single_frame_trail",
    )
    app = object.__new__(StarfieldApp)
    app.active_job_kind = "single"
    app.long_trails = (single_frame_track,)
    app.sequence_result = object()

    assert app._motion_feature_points(0) == ((single_frame_track, point),)


def test_single_frame_workspace_hides_retained_sequence_evidence() -> None:
    """切回单帧工作区后，内存中的 15 帧结果也不能进入绘图入口。"""

    app = object.__new__(StarfieldApp)
    app.task_mode = "single"
    app.sequence_result = object()
    app.long_trails = ()

    assert app._sequence_workspace_active() is False
    assert app._motion_evidence_result() is None
    assert app._motion_evidence_frame_index() is None
    assert app._motion_feature_points(0) == ()


def test_single_frame_features_render_from_local_result_after_run() -> None:
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
        0.88,
        0.82,
        single_frame_model="line",
        line_template_snr=8.0,
        line_vs_point_delta_bic=10.0,
        line_vs_double_delta_bic=10.0,
        line_vs_elliptical_delta_bic=10.0,
    )
    restored_track = MotionFeatureTrack(
        12,
        "candidate",
        (point,),
        0.0,
        0.0,
        None,
        evidence_level="single_frame_trail",
    )
    app = object.__new__(StarfieldApp)
    app.active_job_kind = None
    app.sequence_result = None
    app.long_trails = (restored_track,)

    assert app._motion_feature_points(0) == ((restored_track, point),)


def test_sequence_workspace_does_not_fallback_to_single_frame_trails() -> None:
    """15 帧工作区没有序列结果时，不能把单帧预检冒充成跨帧证据。"""

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
        0.88,
        0.82,
    )
    single_frame_track = MotionFeatureTrack(12, "candidate", (point,), 0.0, 0.0, None)
    app = object.__new__(StarfieldApp)
    app.task_mode = "sequence"
    app.sequence_result = None
    app.long_trails = (single_frame_track,)

    assert app._motion_evidence_result() is None
    assert app._motion_feature_points(0) == ()


def test_single_frame_motion_points_without_shape_evidence_are_not_rendered() -> None:
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
        single_frame_model="line",
        line_template_snr=8.0,
        line_vs_point_delta_bic=10.0,
        line_vs_double_delta_bic=10.0,
        line_vs_elliptical_delta_bic=10.0,
    )
    track = MotionFeatureTrack(
        9,
        "candidate",
        (point,),
        0.0,
        0.0,
        None,
        evidence_level="single_frame_trail",
    )
    app = object.__new__(StarfieldApp)
    app.active_job_kind = "single"
    app.long_trails = (track,)

    assert app._motion_feature_points(0) == ()


def test_single_frame_motion_view_hides_low_fill_peak_pairs() -> None:
    weak_point = MotionFeaturePoint(
        0, 30.0, 40.0, 30.0, 40.0, 40.0, 12, 18.0, 4.0, 20.0,
        (20, 30, 40, 50), False, 0.22, 0.18,
    )
    strong_point = MotionFeaturePoint(
        0, 80.0, 60.0, 80.0, 60.0, 40.0, 60, 42.0, 3.0, -35.0,
        (55, 35, 105, 85), False, 0.88, 0.82,
        single_frame_model="line",
        line_template_snr=8.0,
        line_vs_point_delta_bic=10.0,
        line_vs_double_delta_bic=10.0,
        line_vs_elliptical_delta_bic=10.0,
    )
    # “弱”只表示剖面连续度不足；有限线段模型本身仍须有完整的
    # SNR/Delta-BIC 证据，否则它属于审计记录而不属于运动叠加层。
    weak_point = replace(
        weak_point,
        single_frame_model="line",
        line_template_snr=8.0,
        line_vs_point_delta_bic=10.0,
        line_vs_double_delta_bic=10.0,
        line_vs_elliptical_delta_bic=10.0,
    )
    weak_track = MotionFeatureTrack(
        1, "candidate", (weak_point,), 0.0, 0.0, None, evidence_level="single_frame_trail"
    )
    strong_track = MotionFeatureTrack(
        2, "candidate", (strong_point,), 0.0, 0.0, None, evidence_level="single_frame_trail"
    )
    app = object.__new__(StarfieldApp)
    app.active_job_kind = "single"
    app.long_trails = (weak_track, strong_track)

    assert app._motion_feature_points(0) == ((strong_track, strong_point),)
    assert app._motion_feature_points(0, include_weak=True) == (
        (weak_track, weak_point),
        (strong_track, strong_point),
    )


def test_single_frame_weak_motion_view_hides_non_line_model_winners() -> None:
    line_point = MotionFeaturePoint(
        0, 30.0, 40.0, 30.0, 40.0, 40.0, 12, 18.0, 4.0, 20.0,
        (20, 30, 40, 50), False, 0.22, 0.18,
        single_frame_model="line",
    )
    double_point = MotionFeaturePoint(
        0, 80.0, 60.0, 80.0, 60.0, 40.0, 60, 42.0, 3.0, -35.0,
        (55, 35, 105, 85), False, 0.88, 0.82,
        single_frame_model="double",
    )
    line_point = replace(
        line_point,
        line_template_snr=8.0,
        line_vs_point_delta_bic=10.0,
        line_vs_double_delta_bic=10.0,
        line_vs_elliptical_delta_bic=10.0,
    )
    line_track = MotionFeatureTrack(
        1, "candidate", (line_point,), 0.0, 0.0, None, evidence_level="single_frame_trail"
    )
    double_track = MotionFeatureTrack(
        2, "candidate", (double_point,), 0.0, 0.0, None, evidence_level="single_frame_trail"
    )
    app = object.__new__(StarfieldApp)
    app.active_job_kind = "single"
    app.long_trails = (line_track, double_track)

    # 默认复合层同样不能让一个“形态看起来够长”的双星通过；否则虽然
    # 画布不一定画出它，悬浮命中仍可能把它解释成曝光线。
    assert app._motion_feature_points(0) == ()
    assert app._motion_feature_points(0, include_weak=True) == ((line_track, line_point),)


def test_single_frame_overlay_rejects_unmodelled_legacy_trail() -> None:
    """旧单帧缓存没有模型证据时不能重新变成橙色运动线。"""

    point = MotionFeaturePoint(
        0, 80.0, 60.0, 80.0, 60.0, 40.0, 60, 42.0, 3.0, -35.0,
        (55, 35, 105, 85), False, 0.88, 0.82,
    )
    track = MotionFeatureTrack(
        2,
        "candidate",
        (point,),
        0.0,
        0.0,
        None,
        evidence_level="single_frame_trail",
    )
    app = object.__new__(StarfieldApp)
    app.active_job_kind = "single"
    app.long_trails = (track,)

    assert app._motion_feature_points(0, include_weak=True) == ()


def test_motion_overlay_keeps_long_artifact_candidate_as_explicit_review_line() -> None:
    """超长线不能静默消失，但仍只能以 ARTIFACT? 候选显示。"""

    point = MotionFeaturePoint(
        0,
        80.0,
        60.0,
        80.0,
        60.0,
        40.0,
        120,
        685.0,
        12.0,
        90.0,
        (20, 0, 40, 700),
        False,
        0.90,
        0.90,
        line_artifact_overlap_fraction=1.0,
        single_frame_model="long_line_profile",
        line_template_snr=20.0,
        line_vs_background_delta_bic=20.0,
    )
    track = MotionFeatureTrack(
        4,
        "candidate",
        (point,),
        0.0,
        0.0,
        None,
        evidence_level="single_frame_trail",
    )
    app = object.__new__(StarfieldApp)
    app.active_job_kind = "single"
    app.long_trails = (track,)

    assert app._motion_feature_points(0) == ((track, point),)
    assert app._motion_feature_points(0, include_weak=True) == ((track, point),)


def test_sequence_overlay_keeps_long_artifact_candidate_visible_for_review() -> None:
    """15帧工作区也不能把算法保留的超长线过滤到画面外。"""

    point = MotionFeaturePoint(
        0,
        80.0,
        60.0,
        80.0,
        60.0,
        40.0,
        120,
        685.0,
        12.0,
        90.0,
        (20, 0, 40, 700),
        False,
        0.90,
        0.90,
        line_artifact_overlap_fraction=1.0,
        single_frame_model="long_line_profile",
        line_template_snr=20.0,
        line_vs_background_delta_bic=20.0,
    )
    track = MotionFeatureTrack(
        4,
        "candidate",
        (point,),
        0.0,
        0.0,
        None,
        evidence_level="single_frame_trail",
    )
    app = object.__new__(StarfieldApp)
    app.task_mode = "sequence"
    app.sequence_result = SimpleNamespace(motion_features=(track,))

    assert app._motion_feature_points(0) == ((track, point),)
    assert app._motion_feature_points(0, include_weak=True) == ((track, point),)


def test_motion_overlay_keeps_multi_frame_artifact_conflict_as_uncertain_candidate() -> None:
    """跨帧几何证据不能因伪影提示被静默抹掉，但仍保持 candidate。"""

    points = tuple(
        MotionFeaturePoint(
            frame_index=index,
            x=100.0 + index * 30.0,
            y=200.0 - index * 4.0,
            aligned_x=100.0 + index * 30.0,
            aligned_y=200.0 - index * 4.0,
            residual_snr=40.0,
            area_pixels=120,
            length_px=32.0,
            width_px=3.0,
            angle_deg=-7.6,
            bbox=(0, 0, 200, 200),
            touches_edge=False,
            continuity_ratio=0.88,
            profile_fill_ratio=0.86,
            line_artifact_overlap_fraction=1.0,
            single_frame_model="line",
            line_template_snr=20.0,
            line_vs_point_delta_bic=20.0,
            line_vs_double_delta_bic=20.0,
            line_vs_background_delta_bic=20.0,
            line_vs_elliptical_delta_bic=20.0,
        )
        for index in range(4)
    )
    track = MotionFeatureTrack(
        5,
        "candidate",
        points,
        90.0,
        30.0,
        1.0,
        evidence_level="center_motion",
    )

    assert _motion_overlay_candidate_is_allowed(track, points[0]) is True


def test_single_frame_weak_motion_view_keeps_overlength_geometry_candidate() -> None:
    point = MotionFeaturePoint(
        0, 30.0, 40.0, 30.0, 40.0, 40.0, 120, 685.0, 12.0, 90.0,
        (20, 0, 40, 700), False, 0.90, 0.90,
        single_frame_model="not_fit",
    )
    track = MotionFeatureTrack(
        3, "candidate", (point,), 0.0, 0.0, None, evidence_level="single_frame_trail"
    )
    app = object.__new__(StarfieldApp)
    app.active_job_kind = "single"
    app.long_trails = (track,)

    assert app._motion_feature_points(0, include_weak=True) == ()


def test_single_frame_composite_can_show_line_candidates_without_sequence_data() -> None:
    line_point = MotionFeaturePoint(
        0, 30.0, 40.0, 30.0, 40.0, 40.0, 12, 18.0, 4.0, 20.0,
        (20, 30, 40, 50), False, 0.22, 0.18,
        single_frame_model="line",
    )
    line_point = replace(
        line_point,
        line_template_snr=8.0,
        line_vs_point_delta_bic=10.0,
        line_vs_double_delta_bic=10.0,
        line_vs_elliptical_delta_bic=10.0,
    )
    track = MotionFeatureTrack(
        1, "candidate", (line_point,), 0.0, 0.0, None, evidence_level="single_frame_trail"
    )
    app = object.__new__(StarfieldApp)
    app.active_job_kind = "single"
    app.long_trails = (track,)
    app.sequence_result = None

    assert app._sequence_workspace_active() is False
    assert app._motion_feature_points(0, include_weak=True) == ((track, line_point),)


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
        0.88,
        0.82,
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
