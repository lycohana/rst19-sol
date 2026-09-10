"""rst19 的纯 Python Tkinter 桌面界面。"""

from __future__ import annotations

import argparse
import csv
import json
import math
import queue
import re
import threading
from collections import Counter
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import tkinter as tk
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageTk
from tkinter import filedialog, messagebox, ttk

from .cache import (
    cache_key,
    clear_cache,
    load_analysis,
    load_sequence_result,
    save_analysis,
    save_sequence_result,
    sequence_cache_key,
)
from .catalog import CatalogSource, load_catalog_csv
from .experiments import (
    DetectionPSFSweepRow,
    DetectionSweepRow,
    InjectionRecoveryRow,
    RealBackgroundInjectionRow,
    SingleFrameTrailAuditRow,
    run_detection_threshold_sweep,
    run_detection_psf_sweep,
    run_injection_recovery,
    run_real_background_injection,
    run_single_frame_trail_audit,
    write_detection_sweep_artifacts,
    write_detection_psf_sweep_artifacts,
    write_detection_source_artifacts,
    write_sequence_source_audit,
    write_injection_artifacts,
    write_real_background_injection_artifacts,
    write_single_frame_trail_artifacts,
)
from .feedback import ManualThresholdFeedback, append_manual_feedback, load_manual_feedback
from .fits import auxiliary_mask, read_fits
from .innovation import _fit_constant_velocity, _telemetry_consistency, _telemetry_prediction, build_innovation_report_from_payload, write_innovation_artifacts
from .photometry import instrumental_magnitude
from .pipeline import FrameAnalysis, analyze_frame
from .matching import MatchResult
from .sequence import (
    DEFAULT_FAST_POINT_FIT_RMS_PX,
    DEFAULT_FAST_POINT_GATE_PX,
    DEFAULT_FAST_POINT_MAX_STEP_PX,
    DEFAULT_FAST_POINT_MIN_SNR,
    DEFAULT_SEQUENCE_SOURCE_WORKING_LIMIT,
    DEFAULT_SEQUENCE_BACKGROUND_SAMPLE_LIMIT,
    DEFAULT_SEQUENCE_WORKERS,
    MotionFeaturePoint,
    MotionFeatureTrack,
    MotionFrameAudit,
    SequenceResult,
    SourceTrack,
    TrackPoint,
    analyze_sequence,
    detect_single_frame_long_trails,
)
from .wcs import AffineWCSCalibration, TangentPlaneWCS, fit_affine_wcs_from_matches
from .wcs_validation import WCSValidationReport, run_sequence_wcs_validation, write_wcs_validation_artifacts

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "doc" / "00-项目资料" / "原始数据"

# 深空观测台主题：主背景是带蓝紫色相的近黑色，星点用暖白/金色，
# 通过状态才使用青绿色。避免米白底、青灰卡片和“AI 仪表盘”式平均配色。
# 这些常量仍沿用旧名字，是为了让研究弹窗和结果渲染代码共享同一套主题。
NAVY = "#0b1224"
NAVY_DARK = "#070c19"
NAVY_SOFT = "#1b2b4a"
PAPER = "#0a1020"
PAPER_LIGHT = "#101a30"
PAPER_LINE = "#263a5d"
INK = "#e7edf9"
INK_SOFT = "#91a6c9"
AMBER = "#e2a84b"
AMBER_LIGHT = "#ffd477"
MINT = "#65ddc0"
SKY = "#77b8ff"
SKY_LIGHT = "#b9d7ff"
# 预览图上的源标记单独使用高亮色，确保在黑灰 FITS 上有足够对比度。
STAR_POINT = "#63e8ff"
DOG_ONLY_POINT = "#ffe07a"
STAR_POINT_REJECTED = "#c3a5ff"
STATIC_STAR_POINT = "#67e4c3"
PERSISTENT_STAR_POINT = "#90caff"
TEMPORAL_STAR_POINT = "#bba6ff"
STACK_FAINT_POINT = "#8bd6aa"
MOTION_TRAIL = "#ff6bd6"
MOTION_CANDIDATE = "#ff9a5c"
# 点状高速目标使用金橙色，与可信源青色、线状候选洋红色分开；
# 这条线代表逐帧点源质心拟合，不是把点源涂成长线。
POINT_MOTION = "#ffd166"
FORECAST = "#8edbbb"
WHITE = "#f2f5ff"
MONO = "Consolas"
SANS = "Segoe UI"
MAX_PREVIEW_ZOOM = 15.0
# 当前真实 FITS 抽测后的 GUI 默认分析口径。这里集中保存“默认值”，
# 让单帧检测、15 帧序列的缓存键和实际调用不会再次发生参数漂移。
GUI_DEFAULT_THRESHOLD_SIGMA = 4.0
GUI_DEFAULT_MIN_DISTANCE = 4
GUI_DEFAULT_PSF_FWHM = 2.0
GUI_DEFAULT_MIN_FLUX_SNR = 5.0
GUI_DEFAULT_PROPOSAL_MODE = "hybrid"
GUI_DEFAULT_TEMPORAL_PROPOSAL_MODE = "median"
GUI_DEFAULT_CANDIDATE_CONSENSUS_MIN_SNR = 15.0
GUI_DEFAULT_TEMPORAL_CANDIDATE_MIN_SNR = 7.5
GUI_DEFAULT_TEMPORAL_REFERENCE_MIN_SNR = 15.0
GUI_DEFAULT_TEMPORAL_MULTISCALE = False
GUI_DEFAULT_TEMPORAL_MIN_PSF_CORRELATION = 0.8
GUI_DEFAULT_LOCAL_DEBLEND = False
GUI_DEFAULT_SEQUENCE_FULL = False
# 这些参数只作用于观察预览，不参与候选检测、孔径测光或星等计算。
# 真实 FITS 的背景中位数约为 21 ADU，噪声约为 8--9 ADU；若直接对
# 1--99.5 分位做平方根拉伸，背景会被抬到约 78/255，整幅图看起来像
# 灰色噪声。这里用稳健 IQR 估计噪声底，把背景压回暗部，再做很轻的
# 显示层高斯去噪。检测器仍然只读取原始 ADU。
PREVIEW_LOW_PERCENTILE = 1.0
PREVIEW_HIGH_PERCENTILE = 99.5
PREVIEW_LOW_SIGMA = 0.5
PREVIEW_HIGH_SIGMA = 12.0
PREVIEW_DISPLAY_GAMMA = 1.15
PREVIEW_DENOISE_RADIUS = 0.6
# 当前比赛 FITS 为 4096²；保留这一级预览，15× 才是在真实传感器
# 空间上放大，而不是先把弱星压缩到 1600² 后再放大插值。
PREVIEW_MAX_SIDE = 4096
PREVIEW_MODE_ENHANCED = "enhanced"
PREVIEW_MODE_RAW = "raw"
PREVIEW_MODE_NOISE = "noise"
PREVIEW_MODE_LABELS = {
    PREVIEW_MODE_ENHANCED: "增强显示 · 轻度降噪",
    PREVIEW_MODE_RAW: "原始显示 · 未降噪",
    PREVIEW_MODE_NOISE: "增亮噪声 · 看弱点",
}
PREVIEW_MODES = tuple(PREVIEW_MODE_LABELS)
# ImageDraw 默认字体只有很有限的字符集；Windows 目标环境优先使用
# 中文字体，避免图内标签把“候选/掩膜”等文字画成方框。ASCII 标签仍
# 作为跨平台回退，因此即使没有这些系统字体也不会出现乱码。
OVERLAY_FONT_CANDIDATES = (
    Path(r"C:\Windows\Fonts\msyh.ttc"),
    Path(r"C:\Windows\Fonts\simhei.ttf"),
    Path(r"C:\Windows\Fonts\simsun.ttc"),
    Path(r"C:\Windows\Fonts\Deng.ttf"),
)
MANUAL_THRESHOLD_MIN = 1.0
MANUAL_THRESHOLD_MAX = 20.0
MANUAL_SNR_MIN = 1.0
MANUAL_SNR_MAX = 10.0


def _preview_values_and_stats(data: np.ndarray) -> tuple[np.ndarray, tuple[float, float, float, float, float, float]]:
    """准备显示用数组和稳健统计量；不修改 FITS 数据。"""

    # 预览只需要把输入映射到 8-bit；对当前 16-bit FITS 用 float32 足够
    # 精确，并减少切帧时的临时内存和等待。检测器不会复用这个数组。
    values = np.asarray(data, dtype=np.float32).copy()
    values[auxiliary_mask(values.shape)] = np.nan
    valid = values[np.isfinite(values)]
    if valid.size == 0:
        raise ValueError("图像没有可预览的有限像素")
    low_percentile, q25, center, q75, high_percentile = np.percentile(
        valid,
        [PREVIEW_LOW_PERCENTILE, 25.0, 50.0, 75.0, PREVIEW_HIGH_PERCENTILE],
    )
    # 对含有大量背景像素的星图，用 IQR/1.349 估计背景噪声比全局标准差
    # 稳健；后者会被亮星和饱和/线状目标明显拉高。
    robust_noise = max((float(q75) - float(q25)) / 1.3489795, np.finfo(np.float64).eps)
    return values, (
        float(low_percentile),
        float(q25),
        float(center),
        float(q75),
        float(high_percentile),
        robust_noise,
    )


def _render_preview(
    values: np.ndarray,
    stats: tuple[float, float, float, float, float, float],
    *,
    mode: str,
    max_side: int,
) -> Image.Image:
    """按指定显示口径生成 8-bit 预览；这里只影响人眼观察。"""

    low_percentile, _q25, center, _q75, high_percentile, robust_noise = stats
    if mode == PREVIEW_MODE_RAW:
        # 保留原始像素的噪声起伏，不做平滑；百分位只是显示映射，
        # 不改变任何分析数据。
        low = low_percentile
        high = max(high_percentile, low + 1.0)
        gamma = 1.0
        denoise_radius = 0.0
    elif mode == PREVIEW_MODE_NOISE:
        # 把背景噪声抬到可见范围，便于判断弱点是否只是颗粒噪声。
        # 这里故意不做高斯平滑，也不把结果送回检测器。
        low = max(low_percentile, center - 2.0 * robust_noise)
        high = max(center + 7.0 * robust_noise, low + 1.0)
        gamma = 0.82
        denoise_radius = 0.0
    else:
        # 增强显示：压住背景阴影，同时用轻度平滑抑制单像素颗粒。
        low = max(low_percentile, center - PREVIEW_LOW_SIGMA * robust_noise)
        high = max(high_percentile, center + PREVIEW_HIGH_SIGMA * robust_noise)
        gamma = PREVIEW_DISPLAY_GAMMA
        denoise_radius = PREVIEW_DENOISE_RADIUS
    normalized = np.clip((values - low) / (high - low), 0.0, 1.0)
    normalized = np.power(normalized, gamma)
    normalized[~np.isfinite(normalized)] = 0.0
    image = Image.fromarray(np.rint(normalized * 255.0).astype(np.uint8), mode="L").convert("RGB")
    if denoise_radius > 0:
        # 这是 8-bit 显示层的轻度平滑，保留约 3--4 px 的点扩散斑，
        # 只抑制单像素颗粒；任何定量结果都不从这张图读取。
        image = image.filter(ImageFilter.GaussianBlur(denoise_radius))
    image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return image


def _make_preview(
    data: np.ndarray,
    max_side: int = PREVIEW_MAX_SIDE,
    mode: str = PREVIEW_MODE_ENHANCED,
) -> Image.Image:
    """创建可缩放的观察预览；检测点在 Canvas 当前缩放级别上动态绘制。

    预览故意和检测器分开：这里的背景抑制、色调映射和轻度去噪只为让
    人眼看清星点，不会回写 FITS，也不会改变检测器使用的原始 ADU。
    """
    if mode not in PREVIEW_MODES:
        raise ValueError(f"未知预览模式：{mode}")
    values, stats = _preview_values_and_stats(data)
    return _render_preview(values, stats, mode=mode, max_side=max_side)


def _make_preview_variants(data: np.ndarray, max_side: int = PREVIEW_MAX_SIDE) -> dict[str, Image.Image]:
    """一次统计、生成三种显示层，切换时不重复扫描 FITS。"""

    values, stats = _preview_values_and_stats(data)
    return {
        mode: _render_preview(values, stats, mode=mode, max_side=max_side)
        for mode in PREVIEW_MODES
    }


FLAG_REASON_LABELS = {
    "EDGE": "靠近图像边缘，测光孔径不完整",
    "MASKED": "孔径内含掩膜/无效像素",
    "PARTIAL_MASKED": "孔径内有少量精确零值坏像素，已按剩余有效像素保留",
    "SATURATED": "孔径内含饱和像素",
    "NEGATIVE_OVERFLOW": "孔径内含接近有符号整型下限的极端负码，疑似溢出/饱和邻域",
    "CODE_PATTERN": "候选峰本身落在全幅异常重复高位码上，且孔径含远离背景的负值，数据有效性不足",
    "LINE_ARTIFACT": "落在线状伪影或亮线区域",
    "BACKGROUND_UNCERTAIN": "背景环可用像素不足，背景估计不确定",
    "NON_POSITIVE_FLUX": "背景扣除后的净通量非正",
    "LOW_FLUX_SNR": "通量 SNR 低于当前门槛",
    "NO_SHAPE": "无法可靠计算点源形态",
    "NARROW": "光斑过窄，疑似尖峰/坏点",
    "BROAD": "光斑过宽，不符合当前 PSF 范围",
    "ELONGATED": "光斑过于细长，疑似拖影或线状目标",
    "DIFFUSE": "光斑过于弥散",
    "SPIKE": "峰值过尖，疑似单像素/尖峰伪影",
    "SMALL_FOOTPRINT": "有效足迹像素过少",
    "INSUFFICIENT_PSF_SUPPORT": "候选峰周围 3×3 PSF 支持不足",
    "UNRESOLVED_BLEND": "补充算法峰紧邻更强 Gaussian 源，单/双 PSF 证据尚不足",
}
FLAG_SHORT_LABELS = {
    "EDGE": "EDGE",
    "MASKED": "MASKED",
    "PARTIAL_MASKED": "ZERO BAD",
    "SATURATED": "SATURATED",
    "NEGATIVE_OVERFLOW": "RANGE?",
    "CODE_PATTERN": "CODE?",
    "LINE_ARTIFACT": "LINE",
    "BACKGROUND_UNCERTAIN": "BG?",
    "NON_POSITIVE_FLUX": "NO FLUX",
    "LOW_FLUX_SNR": "LOW SNR",
    "NO_SHAPE": "NO SHAPE",
    "NARROW": "NARROW",
    "BROAD": "BROAD",
    "ELONGATED": "ELONGATED",
    "DIFFUSE": "DIFFUSE",
    "SPIKE": "SPIKE",
    "SMALL_FOOTPRINT": "SMALL FOOT",
    "INSUFFICIENT_PSF_SUPPORT": "LOW PSF",
    "UNRESOLVED_BLEND": "BLEND?",
}


def source_quality_reason(
    source: Any,
    parameters: dict[str, Any] | None = None,
    *,
    compact: bool = False,
) -> str:
    """返回用户可读的质量判定/落选原因；不会把候选数包装成真星数。"""

    flags = tuple(str(flag) for flag in getattr(source, "flags", ()) if str(flag))
    if bool(getattr(source, "quality_passed", False)):
        if "PARTIAL_MASKED" in flags:
            return "PASS/ZERO BAD" if compact else "通过质量筛选（孔径内少量精确零值已按坏像素忽略）"
        return "PASS" if compact else "通过质量筛选"
    if not flags:
        return "REJECT" if compact else "未通过质量筛选，但结果未记录具体标志"
    parameters = parameters or {}
    reasons: list[str] = []
    for flag in flags:
        if compact:
            reasons.append(FLAG_SHORT_LABELS.get(flag, flag))
            continue
        reason = FLAG_REASON_LABELS.get(flag, flag)
        if flag == "LOW_FLUX_SNR":
            value = getattr(source, "flux_snr", None)
            threshold = parameters.get("min_flux_snr")
            if isinstance(value, (int, float)) and isinstance(threshold, (int, float)):
                reason = f"通量 SNR {float(value):.2f} < 门槛 {float(threshold):.2f}"
        elif flag == "INSUFFICIENT_PSF_SUPPORT":
            support = getattr(source, "psf_support_pixels", None)
            required = parameters.get("min_psf_support_pixels", 3)
            if isinstance(support, (int, float)):
                reason = f"PSF 支持 {int(support)}/{int(required)}，不足"
        elif flag == "UNRESOLVED_BLEND":
            distance = getattr(source, "nearest_gaussian_px", None)
            limit = parameters.get("dog_blend_radius_px")
            if isinstance(distance, (int, float)) and isinstance(limit, (int, float)):
                reason = f"距最近 Gaussian 主峰 {float(distance):.2f}px < 去混叠半径 {float(limit):.2f}px"
        elif flag == "NEGATIVE_OVERFLOW":
            limit = parameters.get("negative_overflow_limit")
            if isinstance(limit, (int, float)):
                reason = f"孔径含极端负码 ≤ {float(limit):.0f}，疑似有符号溢出/饱和邻域，不能可靠测光"
        elif flag in {"NARROW", "BROAD"}:
            fwhm = getattr(source, "fwhm", None)
            if isinstance(fwhm, (int, float)):
                minimum = parameters.get("min_fwhm", 0.8)
                maximum = parameters.get("max_fwhm", 12.0)
                reason = f"FWHM {float(fwhm):.2f}px，不在 {float(minimum):.2f}–{float(maximum):.2f}px 范围"
        elif flag == "ELONGATED":
            ellipticity = getattr(source, "ellipticity", None)
            limit = parameters.get("max_ellipticity", 0.65)
            if isinstance(ellipticity, (int, float)):
                reason = f"椭圆率 {float(ellipticity):.2f} > 上限 {float(limit):.2f}"
        elif flag in {"SPIKE", "DIFFUSE"}:
            sharpness = getattr(source, "sharpness", None)
            if isinstance(sharpness, (int, float)):
                minimum = parameters.get("min_sharpness", 0.005)
                maximum = parameters.get("max_sharpness", 0.85)
                reason = f"尖锐度 {float(sharpness):.3f}，不在 {float(minimum):.3f}–{float(maximum):.3f} 范围"
        reasons.append(reason)
    return "; ".join(reasons)


@lru_cache(maxsize=16)
def _overlay_font(size: int) -> ImageFont.ImageFont:
    """加载支持中文的图内标注字体，找不到时回退到 PIL 内置字体。"""

    safe_size = max(9, min(32, int(size)))
    for path in OVERLAY_FONT_CANDIDATES:
        if not path.is_file():
            continue
        try:
            return ImageFont.truetype(str(path), safe_size)
        except OSError:
            continue
    return ImageFont.load_default()


def moving_points_for_frame(
    result: SequenceResult | None,
    frame_index: int,
) -> tuple[tuple[SourceTrack, TrackPoint], ...]:
    """只返回指定帧中已经被序列算法判定为 ``moving`` 的轨迹点。"""

    if result is None:
        return ()
    return tuple(
        (track, point)
        for track in result.tracks
        if track.classification == "moving"
        for point in track.points
        if point.frame_index == frame_index
    )


def stable_points_for_frame(
    result: SequenceResult | None,
    frame_index: int,
) -> tuple[tuple[SourceTrack, TrackPoint], ...]:
    """Return static/persistent tracks present in this frame.

    ``static`` needs the strict default presence threshold (about 80% of
    frames); ``persistent`` is a lower-confidence stationary candidate that
    still appears in at least half of the sequence and fits the registered
    position without measurable motion.
    """

    if result is None:
        return ()
    return tuple(
        (track, point)
        for track in result.tracks
        if track.classification in {"static", "persistent"}
        and track.evidence_level != "stack_faint"
        for point in track.points
        if point.frame_index == frame_index
    )


def trusted_points_for_frame(
    result: SequenceResult | None,
    frame_index: int,
) -> tuple[tuple[SourceTrack, TrackPoint], ...]:
    """Return per-frame quality-passed sources from a sequence result.

    A ``SequenceResult`` intentionally drops the heavy frame-level
    ``Detection`` objects after tracking.  The quality-passed ``TrackPoint``
    flag is the compact provenance retained for restoring the trusted-source
    layer in the GUI; consensus and stack-faint supplements remain excluded.
    """

    if result is None:
        return ()
    return tuple(
        (track, point)
        for track in result.tracks
        if track.evidence_level == "quality"
        for point in track.points
        if point.frame_index == frame_index and point.quality_passed
    )


def stack_faint_points_for_frame(
    result: SequenceResult | None,
    frame_index: int,
) -> tuple[tuple[SourceTrack, TrackPoint], ...]:
    """返回当前帧中由叠加参考图恢复的暗星轨迹点。

    这些点来自 15 帧注册中值/稳健叠加的降噪参考图，经逐帧强制测光确认
    后标为 ``evidence_level="stack_faint"``。它们是单帧质量门下的低置信
    补充层，不能与严格静态/持续候选或官方逐星真值混写。
    """

    if result is None:
        return ()
    return tuple(
        (track, point)
        for track in result.tracks
        if track.evidence_level == "stack_faint"
        for point in track.points
        if point.frame_index == frame_index
    )


def sequence_frame_index_for_frame(
    result: SequenceResult | None,
    selected_frame: Path | None,
    frame_paths: Sequence[Path],
) -> int | None:
    """Map the GUI-selected FITS to the index stored in a sequence result.

    Cached/CLI sequence results may store relative paths while the GUI keeps
    resolved ``Path`` objects. Falling back to list position alone can ask the
    stable layer for the wrong frame when the result was produced from a
    different working directory or after the directory order changed.
    """

    if result is None or selected_frame is None:
        return None

    def resolved(path: str | Path) -> Path:
        try:
            return Path(path).expanduser().resolve()
        except OSError:
            return Path(path).expanduser().absolute()

    selected_path = resolved(selected_frame)
    for frame in result.frames:
        stored_path = Path(frame.path).expanduser()
        candidates = (stored_path,) if stored_path.is_absolute() else (stored_path, PROJECT_ROOT / stored_path)
        if any(resolved(candidate) == selected_path for candidate in candidates):
            return int(frame.frame_index)

    # Results created directly by the current GUI have the same ordering even
    # when an older/relative path cannot be resolved from the current cwd.
    for index, path in enumerate(frame_paths):
        if resolved(path) == selected_path and index < len(result.frames):
            return int(result.frames[index].frame_index)
    return None


def motion_features_for_frame(
    result: SequenceResult | None,
    frame_index: int,
) -> tuple[tuple[MotionFeatureTrack, MotionFeaturePoint], ...]:
    """返回当前帧的线状运动候选；静态星点不会进入此层。"""

    if result is None:
        return ()
    return tuple(
        (track, point)
        for track in result.motion_features
        if track.classification in {"moving", "candidate"}
        for point in track.points
        if point.frame_index == frame_index
    )


def source_peak_position(source: Any) -> tuple[float, float]:
    """Return the matched-filter peak used for the visible source marker.

    ``Detection.x/y`` remain the measured sub-pixel centroid.  The integer
    candidate peak is a more stable visual anchor when weak-source noise has
    pulled the photometric centroid away from the actual compact core.
    """

    peak_x = getattr(source, "peak_x", None)
    peak_y = getattr(source, "peak_y", None)
    if peak_x is not None and peak_y is not None and np.isfinite(peak_x) and np.isfinite(peak_y):
        return float(peak_x), float(peak_y)
    return float(source.x), float(source.y)


class StarfieldApp(tk.Tk):
    """本地星图检测工作台。"""

    def __init__(self, data_dir: Path, cache_dir: Path | None = None) -> None:
        super().__init__()
        self.data_dir = data_dir.resolve()
        self.cache_dir = (cache_dir or PROJECT_ROOT / ".rst19-cache").resolve()
        self.frames: list[Path] = []
        self.selected_frame: Path | None = None
        self.analysis: FrameAnalysis | None = None
        self.catalog_analysis: FrameAnalysis | None = None
        self.catalog_match_result: MatchResult | None = None
        self.catalog_wcs: TangentPlaneWCS | None = None
        self.catalog_calibration: AffineWCSCalibration | None = None
        self.catalog_path: Path | None = None
        self.catalog_frame_path: Path | None = None
        self.catalog_window: tk.Toplevel | None = None
        self.sequence_result: SequenceResult | None = None
        self.long_trails: tuple[MotionFeatureTrack, ...] = ()
        self.sequence_evidence_window: tk.Toplevel | None = None
        self.sequence_source_audit_window: tk.Toplevel | None = None
        self.source_study_window: tk.Toplevel | None = None
        self.source_export_in_progress = False
        self.preview: Image.Image | None = None
        self.preview_variants: dict[str, Image.Image] = {}
        self.preview_photo: ImageTk.PhotoImage | None = None
        self.preview_shape: tuple[int, int] | None = None
        self.preview_scale_x = 1.0
        self.preview_scale_y = 1.0
        self.preview_zoom = 1.0
        self.preview_pan_x = 0.0
        self.preview_pan_y = 0.0
        self.drag_start: tuple[int, int] | None = None
        self.hover_source_id: int | None = None
        self.hover_source: Any | None = None
        self.hover_motion_track: tuple[
            SourceTrack | MotionFeatureTrack,
            TrackPoint | MotionFeaturePoint,
        ] | None = None
        self.hover_trusted_track: tuple[SourceTrack, TrackPoint] | None = None
        self.hover_catalog_match: Any | None = None
        self.source_grid: dict[tuple[int, int], list[Any]] = {}
        self.exposure_s = 1.0
        self.frame_token = 0
        self.result_queue: queue.Queue[tuple[str, int, Any]] = queue.Queue()
        self.sequence_frame_states: list[str] = []
        self.busy = False
        self.active_job_token: int | None = None
        self.active_job_kind: str | None = None
        self._closing = False
        self.cache_generation = 0
        self.cache_lock = threading.Lock()
        self.manual_tuning_dirty = False
        self.manual_feedback_path = PROJECT_ROOT / "tmp" / "manual-threshold-feedback.jsonl"

        self.title("RST19 · Starfield Lab")
        self.geometry("1440x900")
        self.minsize(1120, 720)
        self.configure(bg=PAPER)
        self._configure_styles()
        self._build_layout_starfield()
        self._load_frames()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(100, self._poll_result)

    def _configure_styles(self) -> None:
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("Treeview", background=PAPER_LIGHT, fieldbackground=PAPER_LIGHT, foreground=INK, rowheight=27, font=(MONO, 9), borderwidth=0)
        style.configure("Treeview.Heading", background=PAPER, foreground=INK_SOFT, font=(MONO, 8, "bold"), relief="flat", borderwidth=0)
        style.map("Treeview", background=[("selected", NAVY_SOFT)], foreground=[("selected", WHITE)])
        style.configure("TScrollbar", background=PAPER_LINE, troughcolor=PAPER, bordercolor=PAPER, arrowcolor=INK_SOFT)
        style.configure("RST19.TCombobox", fieldbackground=PAPER, background=PAPER_LIGHT, foreground=INK, arrowcolor=SKY, bordercolor=PAPER_LINE)
        style.map("RST19.TCombobox", fieldbackground=[("readonly", PAPER)], foreground=[("readonly", INK)])
        style.configure("RST19.TNotebook", background=PAPER_LIGHT, borderwidth=0)
        style.configure("RST19.TNotebook.Tab", background=PAPER, foreground=INK_SOFT, padding=(12, 7), borderwidth=0)
        style.map("RST19.TNotebook.Tab", background=[("selected", NAVY_SOFT)], foreground=[("selected", WHITE)])
        style.configure(
            "RST19.Horizontal.TProgressbar",
            troughcolor=PAPER_LINE,
            background=MINT,
            bordercolor=PAPER,
            lightcolor=MINT,
            darkcolor=MINT,
        )

    def _label(self, parent: tk.Misc, text: str, *, color: str = INK, size: int = 10, bold: bool = False, **kwargs: Any) -> tk.Label:
        return tk.Label(parent, text=text, bg=kwargs.pop("bg", parent.cget("bg")), fg=color, font=(SANS, size, "bold" if bold else "normal"), **kwargs)

    def _mono_label(self, parent: tk.Misc, text: str, *, color: str = INK_SOFT, size: int = 9, **kwargs: Any) -> tk.Label:
        return tk.Label(parent, text=text, bg=kwargs.pop("bg", parent.cget("bg")), fg=color, font=(MONO, size), **kwargs)

    def _build_layout(self) -> None:
        self.sidebar = tk.Frame(self, bg=NAVY, width=260)
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False)
        self.main = tk.Frame(self, bg=PAPER)
        self.main.pack(side="left", fill="both", expand=True)

        brand = tk.Frame(self.sidebar, bg=NAVY)
        brand.pack(fill="x", padx=22, pady=(25, 0))
        mark = tk.Canvas(brand, width=28, height=28, bg=NAVY, highlightthickness=0)
        mark.pack(side="left", padx=(0, 11))
        mark.create_rectangle(2, 17, 8, 26, fill=AMBER_LIGHT, outline="")
        mark.create_rectangle(11, 7, 17, 26, fill=WHITE, outline="")
        mark.create_rectangle(20, 12, 26, 26, fill=MINT, outline="")
        brand_text = tk.Frame(brand, bg=NAVY)
        brand_text.pack(side="left")
        self._label(brand_text, "RST19", color=WHITE, size=17, bold=True, bg=NAVY).pack(anchor="w")
        self._mono_label(brand_text, "STARFIELD LAB", color="#aeb7c4", size=8, bg=NAVY).pack(anchor="w", pady=(4, 0))
        tk.Frame(self.sidebar, bg=NAVY_SOFT, height=1).pack(fill="x", padx=22, pady=(27, 21))

        self._mono_label(self.sidebar, "FRAME QUEUE", color="#aeb7c4", size=8, bg=NAVY).pack(anchor="w", padx=31)
        self.frame_list = tk.Listbox(self.sidebar, bg=NAVY, fg="#d4d9df", selectbackground=NAVY_SOFT, selectforeground=WHITE, activestyle="none", bd=0, highlightthickness=0, font=(MONO, 9), relief="flat")
        self.frame_list.pack(fill="both", expand=True, padx=22, pady=(8, 15))
        self.frame_list.bind("<<ListboxSelect>>", self._on_frame_selected)

        dataset = tk.Frame(self.sidebar, bg=NAVY, highlightbackground=NAVY_SOFT, highlightthickness=1)
        dataset.pack(fill="x", padx=22, pady=(0, 24))
        self._mono_label(dataset, "LOCAL DATASET", color="#aeb7c4", size=8, bg=NAVY).pack(anchor="w", padx=13, pady=(13, 0))
        dataset_line = tk.Frame(dataset, bg=NAVY)
        dataset_line.pack(anchor="w", padx=13, pady=(7, 0))
        self.frame_count_label = self._label(dataset_line, "—", color=AMBER_LIGHT, size=27, bg=NAVY)
        self.frame_count_label.pack(side="left")
        self._mono_label(dataset_line, " frames", color="#aeb7c4", size=9, bg=NAVY).pack(side="left", padx=(6, 0), pady=(12, 0))
        self._mono_label(dataset, "开运一号 · 1500 ms", color="#aeb7c4", size=9, bg=NAVY).pack(anchor="w", padx=13, pady=(1, 0))
        status = tk.Frame(dataset, bg=NAVY)
        status.pack(anchor="w", padx=13, pady=(12, 13))
        tk.Label(status, text="●", bg=NAVY, fg=MINT, font=(SANS, 9)).pack(side="left")
        self._label(status, "原始 FITS 已就绪", color="#bce0d1", size=9, bg=NAVY).pack(side="left", padx=(5, 0))

        topbar = tk.Frame(self.main, bg=PAPER, height=54)
        topbar.pack(fill="x", padx=33)
        topbar.pack_propagate(False)
        self._label(topbar, "RST19 星图识别与分析", color=NAVY_DARK, size=10, bold=True, bg=PAPER).pack(side="left", pady=18)
        self._mono_label(topbar, "●  本地 FITS · 不上传", color=MINT, size=8, bg=PAPER).pack(side="right", pady=19)
        tk.Frame(self.main, bg=PAPER_LINE, height=1).pack(fill="x", padx=33)

        header = tk.Frame(self.main, bg=PAPER)
        header.pack(fill="x", padx=38, pady=(18, 14))
        heading = tk.Frame(header, bg=PAPER)
        heading.pack(side="left")
        self._label(heading, "观测分析台", color=NAVY_DARK, size=25, bold=True, bg=PAPER).pack(anchor="w")
        self._label(heading, "核对星点、最暗可信源与 15 帧运动证据", color=INK_SOFT, size=10, bg=PAPER).pack(anchor="w", pady=(4, 0))
        method = tk.Frame(header, bg=PAPER)
        method.pack(side="right", anchor="s", pady=2)
        self._mono_label(method, "当前计算口径", color=INK_SOFT, size=8, bg=PAPER).pack(anchor="e")
        self._label(method, "单图全量精测  ·  序列限集并行", color=AMBER, size=10, bold=True, bg=PAPER).pack(anchor="e", pady=(4, 0))

        self._build_controls()
        self._build_metrics()
        self._build_content()

    def _build_controls(self) -> None:
        controls = tk.Frame(self.main, bg=PAPER_LIGHT, highlightbackground=PAPER_LINE, highlightthickness=1)
        controls.pack(fill="x", padx=38, pady=(0, 14))

        control_header = tk.Frame(controls, bg=PAPER_LIGHT)
        control_header.pack(fill="x", padx=15, pady=(11, 8))
        self._label(control_header, "检测参数", color=NAVY_DARK, size=10, bold=True, bg=PAPER_LIGHT).pack(side="left")
        self._mono_label(control_header, "单图：全量候选与精细测量", color=MINT, size=8, bg=PAPER_LIGHT).pack(side="right")
        self._mono_label(control_header, f"15 帧：默认快速工作集 · {DEFAULT_SEQUENCE_WORKERS} 线程", color=SKY, size=8, bg=PAPER_LIGHT).pack(side="right", padx=(0, 16))
        self.research_menu = tk.Menu(
            self,
            tearoff=False,
            bg=PAPER_LIGHT,
            fg=INK,
            activebackground="#e9d8b7",
            activeforeground=NAVY_DARK,
            bd=1,
            relief="solid",
            font=(SANS, 9),
        )
        self.research_menu.add_command(label="打开星表核验", command=self._show_catalog_match)
        self.research_menu.add_separator()
        self.research_menu.add_command(label="导出星点研究表", command=self.export_source_study, state="disabled")
        self.research_export_menu_index = int(self.research_menu.index("end"))
        self.source_export_button = tk.Button(
            control_header,
            text="研究工具 ▾",
            command=self._show_research_menu,
            bg=PAPER,
            fg=INK,
            activebackground="#e9d8b7",
            activeforeground=NAVY_DARK,
            relief="flat",
            bd=0,
            padx=9,
            pady=4,
            font=(SANS, 8, "bold"),
        )
        self.source_export_button.pack(side="right", padx=(0, 8))

        # 15 帧任务的进度必须出现在主工作区，而不是只放在底部状态栏。
        # 并行检测时“当前帧”和“总体百分比”是两个不同维度，拆成一个
        # 固定进度条和一行短状态，避免长日志把布局顶开。
        sequence_progress_row = tk.Frame(controls, bg=PAPER_LIGHT)
        sequence_progress_row.pack(fill="x", padx=15, pady=(0, 10))
        self.sequence_progress_scope_label = self._mono_label(
            sequence_progress_row,
            "SEQ / —",
            color=SKY,
            size=8,
            bg=PAPER_LIGHT,
        )
        self.sequence_progress_scope_label.pack(side="left", padx=(0, 12))
        self.sequence_progress_percent_var = tk.DoubleVar(value=0.0)
        self.sequence_progress_text_var = tk.StringVar(value="序列待运行")
        self.sequence_progress_bar = ttk.Progressbar(
            sequence_progress_row,
            style="RST19.Horizontal.TProgressbar",
            orient="horizontal",
            mode="determinate",
            maximum=100.0,
            variable=self.sequence_progress_percent_var,
            length=260,
        )
        self.sequence_progress_bar.pack(side="left", fill="x", expand=True, pady=2)
        self.sequence_progress_label = tk.Label(
            sequence_progress_row,
            textvariable=self.sequence_progress_text_var,
            bg=PAPER_LIGHT,
            fg=INK_SOFT,
            font=(MONO, 8),
            anchor="e",
            width=34,
        )
        self.sequence_progress_label.pack(side="right", padx=(12, 0))

        # 15 帧任务再增加一条固定的帧账本。百分比只能说明总体进度，
        # 账本能让用户一眼看到 F01…F15 哪些已经完成、哪一帧正在计算；
        # 它不依赖日志滚动，也不会被右侧长摘要顶掉。
        sequence_ledger_row = tk.Frame(controls, bg=PAPER_LIGHT)
        sequence_ledger_row.pack(fill="x", padx=15, pady=(0, 9))
        self._mono_label(sequence_ledger_row, "FRAME STATUS", color=INK_SOFT, size=8, bg=PAPER_LIGHT).pack(side="left", padx=(0, 12))
        self.sequence_ledger_canvas = tk.Canvas(
            sequence_ledger_row,
            height=22,
            bg=PAPER_LIGHT,
            highlightthickness=0,
            bd=0,
        )
        self.sequence_ledger_canvas.pack(side="left", fill="x", expand=True)
        self.sequence_ledger_canvas.bind("<Configure>", lambda _event: self._draw_sequence_ledger())

        parameter_row = tk.Frame(controls, bg=PAPER_LIGHT)
        parameter_row.pack(fill="x", padx=15)
        for column in range(5):
            parameter_row.grid_columnconfigure(column, weight=1)

        threshold = tk.Frame(parameter_row, bg=PAPER_LIGHT)
        threshold.grid(row=0, column=0, sticky="w")
        self._mono_label(threshold, "检测阈值", color=INK_SOFT, size=8, bg=PAPER_LIGHT).pack(side="left", padx=(0, 6))
        self.threshold_var = tk.StringVar(value=f"{GUI_DEFAULT_THRESHOLD_SIGMA:.1f}")
        threshold_entry = tk.Entry(threshold, textvariable=self.threshold_var, width=5, bg=PAPER, fg=INK, insertbackground=INK, relief="flat", highlightbackground=PAPER_LINE, highlightthickness=1, font=(MONO, 10))
        threshold_entry.pack(side="left")
        threshold_entry.bind("<FocusOut>", lambda _event: self._sync_manual_controls_from_entries())
        self._mono_label(threshold, "σ", color=INK_SOFT, size=8, bg=PAPER_LIGHT).pack(side="left", padx=(4, 0))

        distance = tk.Frame(parameter_row, bg=PAPER_LIGHT)
        distance.grid(row=0, column=1)
        self._mono_label(distance, "最小间距", color=INK_SOFT, size=8, bg=PAPER_LIGHT).pack(side="left", padx=(0, 6))
        self.min_distance_var = tk.StringVar(value=str(GUI_DEFAULT_MIN_DISTANCE))
        tk.Entry(distance, textvariable=self.min_distance_var, width=4, bg=PAPER, fg=INK, insertbackground=INK, relief="flat", highlightbackground=PAPER_LINE, highlightthickness=1, font=(MONO, 10)).pack(side="left")
        self._mono_label(distance, "px", color=INK_SOFT, size=8, bg=PAPER_LIGHT).pack(side="left", padx=(4, 0))

        psf = tk.Frame(parameter_row, bg=PAPER_LIGHT)
        psf.grid(row=0, column=2)
        self._mono_label(psf, "PSF FWHM", color=INK_SOFT, size=8, bg=PAPER_LIGHT).pack(side="left", padx=(0, 6))
        self.psf_fwhm_var = tk.StringVar(value=f"{GUI_DEFAULT_PSF_FWHM:.1f}")
        tk.Entry(psf, textvariable=self.psf_fwhm_var, width=4, bg=PAPER, fg=INK, insertbackground=INK, relief="flat", highlightbackground=PAPER_LINE, highlightthickness=1, font=(MONO, 10)).pack(side="left")
        self._mono_label(psf, "px", color=INK_SOFT, size=8, bg=PAPER_LIGHT).pack(side="left", padx=(4, 0))

        snr = tk.Frame(parameter_row, bg=PAPER_LIGHT)
        snr.grid(row=0, column=3)
        self._mono_label(snr, "通量 SNR", color=INK_SOFT, size=8, bg=PAPER_LIGHT).pack(side="left", padx=(0, 6))
        self.min_flux_snr_var = tk.StringVar(value=f"{GUI_DEFAULT_MIN_FLUX_SNR:.1f}")
        min_flux_snr_entry = tk.Entry(snr, textvariable=self.min_flux_snr_var, width=4, bg=PAPER, fg=INK, insertbackground=INK, relief="flat", highlightbackground=PAPER_LINE, highlightthickness=1, font=(MONO, 10))
        min_flux_snr_entry.pack(side="left")
        min_flux_snr_entry.bind("<FocusOut>", lambda _event: self._sync_manual_controls_from_entries())
        self._mono_label(snr, "σ", color=INK_SOFT, size=8, bg=PAPER_LIGHT).pack(side="left", padx=(4, 0))

        source_limit = tk.Frame(parameter_row, bg=PAPER_LIGHT)
        source_limit.grid(row=0, column=4, sticky="e")
        self._mono_label(source_limit, "单图源上限", color=INK_SOFT, size=8, bg=PAPER_LIGHT).pack(side="left", padx=(0, 6))
        self.max_sources_var = tk.StringVar(value="")
        tk.Entry(source_limit, textvariable=self.max_sources_var, width=7, bg=PAPER, fg=INK, insertbackground=INK, relief="flat", highlightbackground=PAPER_LINE, highlightthickness=1, font=(MONO, 10)).pack(side="left")
        self._mono_label(source_limit, "留空=全量", color=MINT, size=8, bg=PAPER_LIGHT).pack(side="left", padx=(5, 0))

        proposal_row = tk.Frame(controls, bg=PAPER_LIGHT)
        proposal_row.pack(fill="x", padx=15, pady=(9, 0))
        self._mono_label(proposal_row, "宽筛模式", color=INK_SOFT, size=8, bg=PAPER_LIGHT).pack(side="left", padx=(0, 8))
        # 当前实测的默认平衡口径：hybrid 在单帧质量层增加少量有效源，
        # 15 帧持续候选也高于 Gaussian；Starlet 仍是实验模式。局部去混叠
        # 需要分钟级逐候选拟合，保持为显式选项，不随 hybrid 自动打开。
        self.proposal_mode_var = tk.StringVar(value=GUI_DEFAULT_PROPOSAL_MODE)
        for value, label in (
            ("hybrid", "推荐平衡 · +DoG+双PSF"),
            ("gaussian", "Gaussian基线 · 速度优先"),
            ("ensemble", "小波实验 · +Starlet"),
        ):
            tk.Radiobutton(
                proposal_row,
                text=label,
                value=value,
                variable=self.proposal_mode_var,
                command=self._mark_manual_tuning_dirty,
                bg=PAPER_LIGHT,
                fg=INK,
                activebackground=PAPER_LIGHT,
                activeforeground=INK,
                selectcolor=PAPER,
                font=(MONO, 8),
                highlightthickness=0,
                bd=0,
            ).pack(side="left", padx=(0, 12))
        self.local_deblend_var = tk.BooleanVar(value=GUI_DEFAULT_LOCAL_DEBLEND)
        tk.Checkbutton(
            proposal_row,
            text="局部去混叠（精测，较慢）",
            variable=self.local_deblend_var,
            command=self._mark_manual_tuning_dirty,
            bg=PAPER_LIGHT,
            fg=INK,
            activebackground=PAPER_LIGHT,
            activeforeground=NAVY_DARK,
            selectcolor=PAPER,
            font=(MONO, 8),
            highlightthickness=0,
            bd=0,
        ).pack(side="left", padx=(3, 0))
        self.sequence_full_var = tk.BooleanVar(value=GUI_DEFAULT_SEQUENCE_FULL)
        tk.Checkbutton(
            proposal_row,
            text="15帧全量关联（慢）",
            variable=self.sequence_full_var,
            command=self._mark_manual_tuning_dirty,
            bg=PAPER_LIGHT,
            fg=INK,
            activebackground=PAPER_LIGHT,
            activeforeground=NAVY_DARK,
            selectcolor=PAPER,
            font=(MONO, 8),
            highlightthickness=0,
            bd=0,
        ).pack(side="left", padx=(12, 0))
        temporal_row = tk.Frame(controls, bg=PAPER_LIGHT)
        temporal_row.pack(fill="x", padx=15, pady=(5, 0))
        self._mono_label(temporal_row, "15帧弱星补提案", color=INK_SOFT, size=8, bg=PAPER_LIGHT).pack(side="left", padx=(0, 8))
        self.temporal_proposal_mode_var = tk.StringVar(value=GUI_DEFAULT_TEMPORAL_PROPOSAL_MODE)
        for value, label in (
            ("median", "稳健中值（默认）"),
            ("coadd", "稳健叠加（召回优先）"),
            ("both", "两者并集（最慢）"),
        ):
            tk.Radiobutton(
                temporal_row,
                text=label,
                value=value,
                variable=self.temporal_proposal_mode_var,
                command=self._mark_manual_tuning_dirty,
                bg=PAPER_LIGHT,
                fg=INK,
                activebackground=PAPER_LIGHT,
                activeforeground=INK,
                selectcolor=PAPER,
                font=(MONO, 8),
                highlightthickness=0,
                bd=0,
            ).pack(side="left", padx=(0, 12))
        self._mono_label(temporal_row, "参考 SNR≥", color=INK_SOFT, size=8, bg=PAPER_LIGHT).pack(side="left", padx=(2, 4))
        self.temporal_reference_min_snr_var = tk.StringVar(value=f"{GUI_DEFAULT_TEMPORAL_REFERENCE_MIN_SNR:.1f}")
        temporal_reference_entry = tk.Entry(
            temporal_row,
            textvariable=self.temporal_reference_min_snr_var,
            width=4,
            bg=PAPER,
            fg=INK,
            insertbackground=INK,
            relief="flat",
            highlightbackground=PAPER_LINE,
            highlightthickness=1,
            font=(MONO, 9),
        )
        temporal_reference_entry.pack(side="left")
        temporal_reference_entry.bind("<FocusOut>", lambda _event: self._mark_manual_tuning_dirty())
        self._mono_label(
            temporal_row,
            "默认15；12可做召回抽测，10仅审计",
            color=MINT,
            size=8,
            bg=PAPER_LIGHT,
        ).pack(side="left", padx=(5, 0))
        stack_faint_row = tk.Frame(controls, bg=PAPER_LIGHT)
        stack_faint_row.pack(fill="x", padx=15, pady=(5, 0))
        self._mono_label(stack_faint_row, "叠加暗星恢复", color=INK_SOFT, size=8, bg=PAPER_LIGHT).pack(side="left", padx=(0, 8))
        self.stack_faint_recovery_var = tk.BooleanVar(value=True)
        tk.Checkbutton(
            stack_faint_row,
            text="15帧叠加恢复暗星（默认开）",
            variable=self.stack_faint_recovery_var,
            command=self._mark_manual_tuning_dirty,
            bg=PAPER_LIGHT,
            fg=INK,
            activebackground=PAPER_LIGHT,
            activeforeground=NAVY_DARK,
            selectcolor=PAPER,
            font=(MONO, 8),
            highlightthickness=0,
            bd=0,
        ).pack(side="left")
        proposal_note = tk.Frame(controls, bg=PAPER_LIGHT)
        proposal_note.pack(fill="x", padx=15, pady=(2, 0))
        self._mono_label(
            proposal_note,
            "深筛/局部去混叠和叠加并集较慢；附加候选仍经逐帧原图质量规则，近邻峰需 ΔBIC≥10 且次分量SNR≥5",
            color=MINT,
            size=8,
            bg=PAPER_LIGHT,
        ).pack(side="left")

        # 人工调参只改变下一次检测的参数，不在拖动时反复启动 4096² 计算。
        # 这样用户可以快速试不同的“候选阈值”和“可信 SNR”，并把实际运行
        # 过的结果记录成后续实验的反馈样本。
        manual_row = tk.Frame(controls, bg=PAPER_LIGHT)
        manual_row.pack(fill="x", padx=15, pady=(10, 0))
        self._mono_label(manual_row, "人工调参", color=AMBER, size=8, bg=PAPER_LIGHT).pack(side="left", padx=(0, 10))

        threshold_tuning = tk.Frame(manual_row, bg=PAPER_LIGHT)
        threshold_tuning.pack(side="left", fill="x", expand=True, padx=(0, 10))
        self._mono_label(threshold_tuning, "候选 σ", color=INK_SOFT, size=8, bg=PAPER_LIGHT).pack(side="left", padx=(0, 4))
        self.manual_threshold_scale_var = tk.DoubleVar(value=4.0)
        self.manual_threshold_scale = tk.Scale(
            threshold_tuning,
            from_=MANUAL_THRESHOLD_MIN,
            to=MANUAL_THRESHOLD_MAX,
            resolution=0.5,
            orient="horizontal",
            variable=self.manual_threshold_scale_var,
            showvalue=False,
            highlightthickness=0,
            bd=0,
            bg=PAPER_LIGHT,
            fg=INK_SOFT,
            troughcolor=PAPER_LINE,
            activebackground=AMBER_LIGHT,
            sliderlength=15,
            width=12,
            command=self._on_manual_threshold_changed,
        )
        self.manual_threshold_scale.pack(side="left", fill="x", expand=True)
        self.manual_threshold_value_label = self._mono_label(threshold_tuning, "4.0σ", color=AMBER, size=8, bg=PAPER_LIGHT, width=5, anchor="e")
        self.manual_threshold_value_label.pack(side="left", padx=(4, 0))

        snr_tuning = tk.Frame(manual_row, bg=PAPER_LIGHT)
        snr_tuning.pack(side="left", fill="x", expand=True, padx=(0, 10))
        self._mono_label(snr_tuning, "可信 SNR", color=INK_SOFT, size=8, bg=PAPER_LIGHT).pack(side="left", padx=(0, 4))
        self.manual_snr_scale_var = tk.DoubleVar(value=5.0)
        self.manual_snr_scale = tk.Scale(
            snr_tuning,
            from_=MANUAL_SNR_MIN,
            to=MANUAL_SNR_MAX,
            resolution=0.5,
            orient="horizontal",
            variable=self.manual_snr_scale_var,
            showvalue=False,
            highlightthickness=0,
            bd=0,
            bg=PAPER_LIGHT,
            fg=INK_SOFT,
            troughcolor=PAPER_LINE,
            activebackground=MINT,
            sliderlength=15,
            width=12,
            command=self._on_manual_snr_changed,
        )
        self.manual_snr_scale.pack(side="left", fill="x", expand=True)
        self.manual_snr_value_label = self._mono_label(snr_tuning, "5.0σ", color=MINT, size=8, bg=PAPER_LIGHT, width=5, anchor="e")
        self.manual_snr_value_label.pack(side="left", padx=(4, 0))

        feedback_tuning = tk.Frame(manual_row, bg=PAPER_LIGHT)
        feedback_tuning.pack(side="left", padx=(0, 8))
        self._mono_label(feedback_tuning, "反馈", color=INK_SOFT, size=8, bg=PAPER_LIGHT).pack(side="left", padx=(0, 4))
        self.manual_feedback_judgement_var = tk.StringVar(value="当前平衡")
        self.manual_feedback_judgement = ttk.Combobox(
            feedback_tuning,
            textvariable=self.manual_feedback_judgement_var,
            values=("保留弱星", "减少伪点", "当前平衡"),
            state="readonly",
            width=9,
        )
        self.manual_feedback_judgement.pack(side="left")
        self.manual_apply_button = tk.Button(
            manual_row,
            text="应用并分析",
            command=self.run_analysis,
            bg=AMBER,
            fg=NAVY_DARK,
            activebackground=AMBER_LIGHT,
            activeforeground=NAVY_DARK,
            relief="flat",
            bd=0,
            padx=10,
            pady=6,
            font=(SANS, 9, "bold"),
        )
        self.manual_apply_button.pack(side="left", padx=(0, 6))
        self.manual_feedback_button = tk.Button(
            manual_row,
            text="记录反馈",
            command=self._record_manual_feedback,
            bg=PAPER,
            fg=INK,
            activebackground="#e9d8b7",
            activeforeground=NAVY_DARK,
            relief="flat",
            bd=0,
            padx=9,
            pady=6,
            font=(SANS, 9, "bold"),
            state="disabled",
        )
        self.manual_feedback_button.pack(side="left")
        self.manual_tuning_status_label = self._mono_label(
            controls,
            "拖动后点击“应用并分析”；反馈只记录已实际运行的阈值，不会自动修改算法",
            color=INK_SOFT,
            size=8,
            bg=PAPER_LIGHT,
        )
        self.manual_tuning_status_label.pack(anchor="w", padx=85, pady=(3, 0))

        action_row = tk.Frame(controls, bg=PAPER_LIGHT)
        action_row.pack(fill="x", padx=15, pady=(10, 12))
        self._mono_label(action_row, "标定零点", color=INK_SOFT, size=8, bg=PAPER_LIGHT).pack(side="left")
        self.zero_point_var = tk.StringVar(value="")
        tk.Entry(action_row, textvariable=self.zero_point_var, width=6, bg=PAPER, fg=INK, insertbackground=INK, relief="flat", highlightbackground=PAPER_LINE, highlightthickness=1, font=(MONO, 10)).pack(side="left", padx=(6, 5))
        self._mono_label(action_row, "可选，仅用于 m_cal", color=INK_SOFT, size=8, bg=PAPER_LIGHT).pack(side="left")
        self.cache_button = tk.Button(action_row, text="清缓存", command=self.clear_detection_cache, bg=PAPER, fg=INK, activebackground="#e9d8b7", activeforeground=NAVY_DARK, relief="flat", bd=0, padx=10, pady=7, font=(SANS, 9, "bold"))
        self.cache_button.pack(side="left", padx=(15, 0))
        self.evidence_button = tk.Button(action_row, text="15 帧证据", command=self._show_sequence_evidence, bg=PAPER, fg=INK, activebackground="#dcecf0", activeforeground=NAVY_DARK, relief="flat", bd=0, padx=13, pady=9, font=(SANS, 9, "bold"), state="disabled")
        self.evidence_button.pack(side="right", padx=(8, 0))
        self.motion_button = tk.Button(action_row, text="分析 15 帧", command=self.run_sequence_analysis, bg=SKY, fg=NAVY_DARK, activebackground=SKY_LIGHT, activeforeground=NAVY_DARK, relief="flat", bd=0, padx=14, pady=8, font=(SANS, 10, "bold"))
        self.motion_button.pack(side="right", padx=(8, 0))
        self.run_button = tk.Button(action_row, text="分析当前帧", command=self.run_analysis, bg=NAVY, fg=WHITE, activebackground=NAVY_SOFT, activeforeground=WHITE, relief="flat", bd=0, padx=15, pady=8, font=(SANS, 10, "bold"))
        self.run_button.pack(side="right", padx=(14, 0))

    def _build_metrics(self) -> None:
        metrics = tk.Frame(self.main, bg=PAPER)
        metrics.pack(fill="x", padx=38, pady=(0, 14))
        metrics.grid_columnconfigure(0, weight=4)
        metrics.grid_columnconfigure(1, weight=3)
        metrics.grid_columnconfigure(2, weight=3)
        self.metric_values: dict[str, tk.Label] = {}

        source_box = tk.Frame(metrics, bg=NAVY, highlightbackground=NAVY, highlightthickness=1)
        source_box.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        self._mono_label(source_box, "源检测计数 · 不等于天体真值", color="#b1bbc9", size=8, bg=NAVY).pack(anchor="w", padx=14, pady=(10, 5))
        source_values = tk.Frame(source_box, bg=NAVY)
        source_values.pack(fill="x", padx=14, pady=(0, 10))
        for key, title, color in (("candidate", "候选峰", AMBER_LIGHT), ("returned", "可信星点", WHITE)):
            group = tk.Frame(source_values, bg=NAVY)
            group.pack(side="left", expand=True, fill="x")
            self._mono_label(group, title, color="#b1bbc9", size=8, bg=NAVY).pack(anchor="w")
            value = self._label(group, "—", color=color, size=19, bg=NAVY)
            value.pack(anchor="w", pady=(3, 0))
            self.metric_values[key] = value

        baseline_box = tk.Frame(metrics, bg=PAPER_LIGHT, highlightbackground=PAPER_LINE, highlightthickness=1)
        baseline_box.grid(row=0, column=1, sticky="nsew", padx=(0, 6))
        self._mono_label(baseline_box, "图像基线 · ADU", color=INK_SOFT, size=8, bg=PAPER_LIGHT).pack(anchor="w", padx=14, pady=(10, 5))
        baseline_values = tk.Frame(baseline_box, bg=PAPER_LIGHT)
        baseline_values.pack(fill="x", padx=14, pady=(0, 10))
        for key, title in (("background", "背景"), ("noise", "噪声")):
            group = tk.Frame(baseline_values, bg=PAPER_LIGHT)
            group.pack(side="left", expand=True, fill="x")
            self._mono_label(group, title, color=INK_SOFT, size=8, bg=PAPER_LIGHT).pack(anchor="w")
            value = self._label(group, "—", color=INK, size=17, bg=PAPER_LIGHT)
            value.pack(anchor="w", pady=(3, 0))
            self.metric_values[key] = value

        faintest_box = tk.Frame(metrics, bg=PAPER_LIGHT, highlightbackground=MINT, highlightthickness=1)
        faintest_box.grid(row=0, column=2, sticky="nsew")
        self._mono_label(faintest_box, "最暗可信源 · m_inst", color=MINT, size=8, bg=PAPER_LIGHT).pack(anchor="w", padx=14, pady=(10, 5))
        faintest_value = self._label(faintest_box, "—", color=MINT, size=19, bg=PAPER_LIGHT)
        faintest_value.pack(anchor="w", padx=14, pady=(3, 10))
        self.metric_values["faintest"] = faintest_value

    def _build_content(self) -> None:
        content = tk.Frame(self.main, bg=PAPER)
        self.content_panel = content
        content.grid_columnconfigure(0, weight=3)
        content.grid_columnconfigure(1, weight=2)
        # 右栏的摘要是主工作流反馈，优先保证它和主图有稳定高度；
        # 源表仍保留在下方，但不能把运动速度/方向摘要压成不可见。
        content.grid_rowconfigure(0, weight=1)
        # 星点表是摘要，不应和右侧证据卡争夺高度；完整表格从“研究工具”打开。
        # 设为 0 权重后，窗口变矮时优先保住 15 帧摘要和进度反馈。
        content.grid_rowconfigure(1, weight=0)

        viewer = tk.Frame(content, bg=PAPER_LIGHT, highlightbackground=PAPER_LINE, highlightthickness=1)
        viewer.grid(row=0, column=0, rowspan=2, sticky="nsew", padx=(0, 10))
        heading = tk.Frame(viewer, bg=PAPER_LIGHT)
        heading.pack(fill="x", padx=16, pady=(14, 10))
        # 先放右侧图层选择，再放左侧标题；如果标题先以 top pack，
        # Tk 会把 mode_box 挤到下一行，矮窗口下主图因此只剩 1 px。
        mode_box = tk.Frame(heading, bg=PAPER_LIGHT)
        mode_box.pack(side="right", anchor="s")
        self.overlay_mode_var = tk.StringVar(value="quality")
        self._mono_label(mode_box, "OVERLAY", color=INK_SOFT, size=8, bg=PAPER_LIGHT).pack(anchor="e")
        choices = (("quality", "可信星点"), ("stable", "稳定星场"), ("stack-faint", "叠加暗星"), ("candidates", "全部候选"), ("motion", "运动候选"), ("catalog", "星表匹配"))
        choice_row = tk.Frame(mode_box, bg=PAPER_LIGHT)
        choice_row.pack(anchor="e", pady=(3, 0))
        for value, label in choices:
            tk.Radiobutton(
                choice_row,
                text=label,
                variable=self.overlay_mode_var,
                value=value,
                command=self._on_overlay_mode_changed,
                bg=PAPER_LIGHT,
                fg=INK,
                activebackground=PAPER_LIGHT,
                activeforeground=NAVY_DARK,
                selectcolor=PAPER,
                relief="flat",
                bd=0,
                font=(SANS, 9),
            ).pack(side="left", padx=(8, 0))
        self._mono_label(heading, "FRAME VIEWER", color=INK_SOFT, size=8, bg=PAPER_LIGHT).pack(anchor="w")
        self.frame_title_label = self._label(heading, "选择一个观测帧", color=NAVY_DARK, size=14, bold=True, bg=PAPER_LIGHT)
        self.frame_title_label.pack(anchor="w", pady=(3, 0))
        display_row = tk.Frame(viewer, bg=PAPER_LIGHT)
        display_row.pack(fill="x", padx=16, pady=(0, 8))
        self._mono_label(display_row, "IMAGE DISPLAY", color=INK_SOFT, size=8, bg=PAPER_LIGHT).pack(side="left", padx=(0, 7))
        self.preview_mode_var = tk.StringVar(value=PREVIEW_MODE_ENHANCED)
        for value, label in PREVIEW_MODE_LABELS.items():
            tk.Radiobutton(
                display_row,
                text=label,
                variable=self.preview_mode_var,
                value=value,
                command=self._on_preview_mode_changed,
                bg=PAPER_LIGHT,
                fg=INK,
                activebackground=PAPER_LIGHT,
                activeforeground=NAVY_DARK,
                selectcolor=PAPER,
                relief="flat",
                bd=0,
                font=(SANS, 9),
            ).pack(side="left", padx=(7, 0))
        self.canvas = tk.Canvas(viewer, bg=NAVY_DARK, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        self.canvas.bind("<Configure>", lambda _event: self._draw_preview())
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind("<Button-4>", lambda event: self._zoom_at(event, 1.15))
        self.canvas.bind("<Button-5>", lambda event: self._zoom_at(event, 1 / 1.15))
        self.canvas.bind("<ButtonPress-1>", self._on_pan_start)
        self.canvas.bind("<B1-Motion>", self._on_pan_move)
        self.canvas.bind("<ButtonRelease-1>", lambda _event: setattr(self, "drag_start", None))
        self.canvas.bind("<Motion>", self._on_canvas_motion)
        self.canvas.bind("<Leave>", lambda _event: self._clear_hover())
        self.hover_info_var = tk.StringVar(value="将鼠标移到候选点查看坐标、通量、误差、SNR、形状和仪器星等")
        self.hover_info_label = tk.Label(
            viewer,
            textvariable=self.hover_info_var,
            bg=PAPER_LIGHT,
            fg=INK_SOFT,
            font=(MONO, 8),
            anchor="w",
            justify="left",
            width=1,
            height=4,
            wraplength=640,
        )
        self.hover_info_label.pack(fill="x", padx=16, pady=(0, 4))
        viewer.bind("<Configure>", self._on_viewer_configure)
        self.overlay_hint_var = tk.StringVar(value="显示：已抑噪 · 滚轮缩放≤15× · 高亮青点 = 通过质量筛选 · 绿色环 = 最暗可信源")
        tk.Label(viewer, textvariable=self.overlay_hint_var, bg=PAPER_LIGHT, fg=INK_SOFT, font=(MONO, 8), anchor="w").pack(anchor="w", padx=16, pady=(0, 13))

        detail = tk.Frame(content, bg=PAPER_LIGHT, highlightbackground=PAPER_LINE, highlightthickness=1)
        self.detail_panel = detail
        detail.grid(row=0, column=1, sticky="nsew", pady=(0, 10))
        self.faintest_section_label = self._mono_label(detail, "FAINTEST SOURCE", color=MINT, size=8, bg=PAPER_LIGHT)
        self.faintest_section_label.pack(anchor="w", padx=16, pady=(15, 0))
        self.faintest_detail = self._label(detail, "尚未运行分析", color=NAVY_DARK, size=14, bold=True, bg=PAPER_LIGHT, justify="left", anchor="w", wraplength=280)
        self.faintest_detail.pack(fill="x", padx=16, pady=(8, 3))
        self.faintest_note = self._label(
            detail,
            "按局部通量 SNR、点源形状、边缘、掩膜和饱和状态筛选；m_inst = −2.5 log10(flux_rate)，有零点后才显示 m_cal。",
            color=INK_SOFT,
            size=9,
            bg=PAPER_LIGHT,
            justify="left",
            wraplength=330,
        )
        self.faintest_note.pack(fill="x", padx=16, pady=(0, 15))
        self.detail_separator = tk.Frame(detail, bg=PAPER_LINE, height=1)
        self.detail_separator.pack(fill="x", padx=16)
        self.evidence_section_label = self._mono_label(detail, "15-FRAME EVIDENCE", color=SKY, size=8, bg=PAPER_LIGHT)
        self.evidence_section_label.pack(anchor="w", padx=16, pady=(12, 5))
        self.sequence_evidence_label = self._label(
            detail,
            "尚未完成序列分析\n点击“分析 15 帧动目标”后显示时间、配准和轨迹摘要",
            color=INK_SOFT,
            size=8,
            bg=PAPER_LIGHT,
            justify="left",
            anchor="w",
            wraplength=330,
        )
        self.sequence_evidence_label.pack(fill="x", padx=16, pady=(0, 14))
        detail.bind("<Configure>", self._on_detail_configure)
        register = tk.Frame(content, bg=PAPER_LIGHT, highlightbackground=PAPER_LINE, highlightthickness=1)
        self.source_register_panel = register
        register.grid(row=1, column=1, sticky="nsew")
        top = tk.Frame(register, bg=PAPER_LIGHT)
        top.pack(fill="x", padx=16, pady=(13, 7))
        self._mono_label(top, "SOURCE REGISTER · TOP FLUX SNR", color=INK_SOFT, size=8, bg=PAPER_LIGHT).pack(side="left")
        self.table_count_label = self._mono_label(top, "0 rows", color=AMBER, size=8, bg=PAPER_LIGHT)
        self.table_count_label.pack(side="right")
        columns = ("id", "xy", "peak", "snr", "mag")
        self.source_tree = ttk.Treeview(register, columns=columns, show="headings", height=2)
        for column, title, width in (("id", "ID", 36), ("xy", "X / Y", 82), ("peak", "PEAK", 48), ("snr", "SNR", 44), ("mag", "m_inst", 48)):
            self.source_tree.heading(column, text=title)
            self.source_tree.column(column, width=width, minwidth=34, anchor="w", stretch=True)
        self.source_tree_scrollbar = ttk.Scrollbar(register, orient="vertical", command=self.source_tree.yview)
        self.source_tree.configure(yscrollcommand=self.source_tree_scrollbar.set)
        self.source_tree_scrollbar.pack(side="right", fill="y", padx=(0, 10), pady=(0, 10))
        self.source_tree.pack(side="left", fill="both", expand=True, padx=(10, 0), pady=(0, 10))
        self.content_compact = False
        content.bind("<Configure>", self._on_content_configure)

        self.status_var = tk.StringVar(value="就绪 · 请选择一张 FITS")
        self.progress_percent_var = tk.DoubleVar(value=0.0)
        self.progress_text_var = tk.StringVar(value="就绪")
        status = tk.Frame(self.main, bg=NAVY_DARK, height=29)
        status.pack(fill="x", side="bottom")
        status.pack_propagate(False)
        tk.Label(status, text="●", bg=NAVY_DARK, fg=MINT, font=(SANS, 9)).pack(side="left", padx=(14, 7))
        progress_group = tk.Frame(status, bg=NAVY_DARK)
        progress_group.pack(side="right", padx=(8, 14))
        self.progress_text_label = tk.Label(progress_group, textvariable=self.progress_text_var, bg=NAVY_DARK, fg="#bce0d1", font=(MONO, 8), width=24, anchor="e")
        self.progress_text_label.pack(side="left", padx=(0, 7))
        self.progress_bar = ttk.Progressbar(
            progress_group,
            style="RST19.Horizontal.TProgressbar",
            orient="horizontal",
            mode="determinate",
            maximum=100.0,
            variable=self.progress_percent_var,
            length=220,
        )
        self.progress_bar.pack(side="left", pady=8)
        tk.Label(status, text="RST19 · PYTHON / OFFLINE", bg=NAVY_DARK, fg="#96a1b1", font=(MONO, 8)).pack(side="right", padx=(12, 0))
        tk.Label(status, textvariable=self.status_var, bg=NAVY_DARK, fg="#d4d9df", font=(MONO, 9), anchor="w").pack(side="left", fill="x", expand=True)
        # 状态栏先占据固定高度，内容区再填满剩余空间；否则较矮窗口中
        # 先 pack 的 expand 内容会把最后创建的进度栏压缩到 1 px。
        content.pack(fill="both", expand=True, padx=38, pady=(0, 16))

    def _on_content_configure(self, event: tk.Event) -> None:
        """在矮窗口中收起次要星点表，把空间让给主图和 15 帧证据。"""

        content = self.__dict__.get("content_panel")
        detail = self.__dict__.get("detail_panel")
        register = self.__dict__.get("source_register_panel")
        if content is None or detail is None or register is None:
            return
        compact = int(getattr(event, "height", 0)) < 300
        if compact == bool(self.__dict__.get("content_compact", False)):
            return
        self.content_compact = compact
        if compact:
            # 完整星表仍可从“研究工具”打开；主工作区在 720p 高度下不再
            # 让 Treeview 的最小请求高度挤掉运行进度和运动摘要。
            register.grid_remove()
            detail.grid_configure(rowspan=2, pady=(0, 0))
            for widget in (
                self.faintest_section_label,
                self.faintest_detail,
                self.faintest_note,
                self.detail_separator,
            ):
                widget.pack_forget()
        else:
            detail.grid_configure(rowspan=1, pady=(0, 10))
            register.grid(row=1, column=1, sticky="nsew")
            evidence_section = self.evidence_section_label
            self.faintest_section_label.pack(anchor="w", padx=16, pady=(15, 0), before=evidence_section)
            self.faintest_detail.pack(fill="x", padx=16, pady=(8, 3), before=evidence_section)
            self.faintest_note.pack(fill="x", padx=16, pady=(0, 15), before=evidence_section)
            self.detail_separator.pack(fill="x", padx=16, before=evidence_section)

    # ------------------------------------------------------------------
    # 星空观测台前端
    #
    # 旧版布局保留在上方作为历史实现，新的启动入口使用下面这组构建器。
    # 后端分析、缓存、结果队列和研究弹窗不在这里重写；这里的工作只是
    # 把“单张一键分析”提升为主任务，并让证据层级在视觉上可读。
    # ------------------------------------------------------------------

    def _star_button(
        self,
        parent: tk.Misc,
        text: str,
        command: Any,
        *,
        kind: str = "quiet",
        padx: int = 12,
        pady: int = 7,
        size: int = 9,
        state: str = "normal",
    ) -> tk.Button:
        palette = {
            "primary": (AMBER, NAVY_DARK, AMBER_LIGHT, NAVY_DARK),
            "secondary": (SKY, NAVY_DARK, SKY_LIGHT, NAVY_DARK),
            "quiet": (PAPER, INK, NAVY_SOFT, WHITE),
            "ghost": (PAPER_LIGHT, INK_SOFT, NAVY_SOFT, WHITE),
        }
        background, foreground, active_background, active_foreground = palette.get(
            kind,
            palette["quiet"],
        )
        return tk.Button(
            parent,
            text=text,
            command=command,
            state=state,
            bg=background,
            fg=foreground,
            activebackground=active_background,
            activeforeground=active_foreground,
            disabledforeground="#52698f",
            relief="flat",
            bd=0,
            padx=padx,
            pady=pady,
            font=(SANS, size, "bold"),
            cursor="hand2",
        )

    @staticmethod
    def _draw_star_motif(canvas: tk.Canvas) -> None:
        """在标题旁绘制克制的星图标记，不使用渐变或装饰性大图。"""

        canvas.delete("all")
        width = max(1, int(canvas.winfo_reqwidth()))
        height = max(1, int(canvas.winfo_reqheight()))
        canvas.create_arc(
            width - 150,
            -45,
            width + 42,
            height + 105,
            start=188,
            extent=136,
            outline=PAPER_LINE,
            width=1,
        )
        canvas.create_arc(
            width - 112,
            -28,
            width + 18,
            height + 76,
            start=188,
            extent=136,
            outline=NAVY_SOFT,
            width=1,
        )
        stars = (
            (24, 42, 1.3, STAR_POINT),
            (61, 22, 1.0, AMBER_LIGHT),
            (106, 51, 1.7, WHITE),
            (151, 29, 1.0, SKY),
            (190, 63, 1.4, AMBER_LIGHT),
            (225, 35, 0.9, MINT),
            (274, 55, 1.2, STAR_POINT),
        )
        for x, y, radius, color in stars:
            canvas.create_oval(
                x - radius,
                y - radius,
                x + radius,
                y + radius,
                fill=color,
                outline="",
            )
        canvas.create_line(106, 51, 151, 29, fill=NAVY_SOFT, width=1)
        canvas.create_line(151, 29, 190, 63, fill=NAVY_SOFT, width=1)

    def _build_layout_starfield(self) -> None:
        """构建以单张观测为主任务的深空分析台。"""

        self.sidebar = tk.Frame(self, bg=NAVY, width=248)
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False)
        self.main = tk.Frame(self, bg=PAPER)
        self.main.pack(side="left", fill="both", expand=True)

        brand = tk.Frame(self.sidebar, bg=NAVY)
        brand.pack(fill="x", padx=22, pady=(24, 0))
        mark = tk.Canvas(brand, width=34, height=34, bg=NAVY, highlightthickness=0)
        mark.pack(side="left", padx=(0, 11))
        mark.create_oval(13, 13, 21, 21, fill=AMBER_LIGHT, outline="")
        mark.create_oval(3, 4, 7, 8, fill=STAR_POINT, outline="")
        mark.create_oval(27, 5, 31, 9, fill=MINT, outline="")
        mark.create_line(7, 7, 14, 15, fill=NAVY_SOFT, width=1)
        mark.create_line(21, 17, 28, 7, fill=NAVY_SOFT, width=1)
        brand_text = tk.Frame(brand, bg=NAVY)
        brand_text.pack(side="left")
        self._label(brand_text, "RST19", color=WHITE, size=18, bold=True, bg=NAVY).pack(anchor="w")
        self._mono_label(brand_text, "ORBITAL STARFIELD LAB", color=SKY_LIGHT, size=7, bg=NAVY).pack(anchor="w", pady=(4, 0))
        self._mono_label(
            self.sidebar,
            "LOCAL OBSERVATION / NO UPLOAD",
            color=INK_SOFT,
            size=7,
            bg=NAVY,
        ).pack(anchor="w", padx=24, pady=(17, 0))
        tk.Frame(self.sidebar, bg=NAVY_SOFT, height=1).pack(fill="x", padx=24, pady=(17, 18))

        self._mono_label(
            self.sidebar,
            "DATASET  /  FRAME QUEUE",
            color=SKY,
            size=8,
            bg=NAVY,
        ).pack(anchor="w", padx=24)
        self.frame_list = tk.Listbox(
            self.sidebar,
            bg=NAVY,
            fg="#c7d7f1",
            selectbackground=NAVY_SOFT,
            selectforeground=WHITE,
            activestyle="none",
            bd=0,
            highlightthickness=0,
            font=(MONO, 8),
            relief="flat",
            selectborderwidth=0,
        )
        self.frame_list.pack(fill="both", expand=True, padx=18, pady=(8, 15))
        self.frame_list.bind("<<ListboxSelect>>", self._on_frame_selected)

        dataset = tk.Frame(
            self.sidebar,
            bg=PAPER_LIGHT,
            highlightbackground=PAPER_LINE,
            highlightthickness=1,
        )
        dataset.pack(fill="x", padx=20, pady=(0, 22))
        self._mono_label(dataset, "LOCAL FITS SET", color=INK_SOFT, size=7, bg=PAPER_LIGHT).pack(
            anchor="w",
            padx=13,
            pady=(13, 0),
        )
        dataset_line = tk.Frame(dataset, bg=PAPER_LIGHT)
        dataset_line.pack(anchor="w", padx=13, pady=(5, 0))
        self.frame_count_label = self._label(
            dataset_line,
            "—",
            color=AMBER_LIGHT,
            size=29,
            bg=PAPER_LIGHT,
        )
        self.frame_count_label.pack(side="left")
        self._mono_label(dataset_line, " FRAMES", color=INK_SOFT, size=8, bg=PAPER_LIGHT).pack(
            side="left",
            padx=(5, 0),
            pady=(14, 0),
        )
        self._mono_label(
            dataset,
            "开运一号  ·  1500 ms  ·  FITS / 16 bit",
            color=INK_SOFT,
            size=7,
            bg=PAPER_LIGHT,
        ).pack(anchor="w", padx=13, pady=(0, 0))
        status = tk.Frame(dataset, bg=PAPER_LIGHT)
        status.pack(anchor="w", padx=13, pady=(11, 13))
        tk.Label(status, text="●", bg=PAPER_LIGHT, fg=MINT, font=(SANS, 8)).pack(side="left")
        self._label(status, "原始数据已就绪", color=MINT, size=8, bg=PAPER_LIGHT).pack(
            side="left",
            padx=(5, 0),
        )

        topbar = tk.Frame(self.main, bg=PAPER, height=45)
        topbar.pack(fill="x", padx=32)
        topbar.pack_propagate(False)
        self._mono_label(topbar, "RST19  /  STARFIELD LAB", color=SKY, size=8, bg=PAPER).pack(
            side="left",
            pady=15,
        )
        self._mono_label(topbar, "PYTHON PIPELINE  ·  LOCAL FITS", color=INK_SOFT, size=8, bg=PAPER).pack(
            side="right",
            pady=15,
        )
        tk.Frame(self.main, bg=PAPER_LINE, height=1).pack(fill="x", padx=32)

        header = tk.Frame(self.main, bg=PAPER)
        header.pack(fill="x", padx=36, pady=(17, 11))
        heading = tk.Frame(header, bg=PAPER)
        heading.pack(side="left")
        self._mono_label(heading, "SINGLE FRAME OBSERVATION", color=AMBER, size=8, bg=PAPER).pack(
            anchor="w",
        )
        self._label(heading, "单张星图 · 证据分析", color=WHITE, size=25, bold=True, bg=PAPER).pack(
            anchor="w",
            pady=(4, 0),
        )
        self._label(
            heading,
            "按论文路线从原始 ADU 走到可复核源表：宽筛保召回，细筛辨机制。",
            color=INK_SOFT,
            size=9,
            bg=PAPER,
        ).pack(anchor="w", pady=(5, 0))
        motif = tk.Canvas(header, width=310, height=84, bg=PAPER, highlightthickness=0)
        motif.pack(side="right", anchor="e")
        self._draw_star_motif(motif)

        self._build_controls_starfield()
        self._build_metrics_starfield()

        self.status_var = tk.StringVar(value="就绪 · 请选择一张 FITS")
        self.progress_percent_var = tk.DoubleVar(value=0.0)
        self.progress_text_var = tk.StringVar(value="就绪")
        status_bar = tk.Frame(self.main, bg=NAVY_DARK, height=31)
        status_bar.pack(fill="x", side="bottom")
        status_bar.pack_propagate(False)
        tk.Label(status_bar, text="●", bg=NAVY_DARK, fg=MINT, font=(SANS, 9)).pack(
            side="left",
            padx=(14, 7),
        )
        tk.Label(
            status_bar,
            textvariable=self.status_var,
            bg=NAVY_DARK,
            fg=INK,
            font=(MONO, 8),
            anchor="w",
        ).pack(side="left", fill="x", expand=True)
        progress_group = tk.Frame(status_bar, bg=NAVY_DARK)
        progress_group.pack(side="right", padx=(8, 14))
        self.progress_text_label = tk.Label(
            progress_group,
            textvariable=self.progress_text_var,
            bg=NAVY_DARK,
            fg=INK_SOFT,
            font=(MONO, 8),
            width=24,
            anchor="e",
        )
        self.progress_text_label.pack(side="left", padx=(0, 7))
        self.progress_bar = ttk.Progressbar(
            progress_group,
            style="RST19.Horizontal.TProgressbar",
            orient="horizontal",
            mode="determinate",
            maximum=100.0,
            variable=self.progress_percent_var,
            length=190,
        )
        self.progress_bar.pack(side="left", pady=8)
        self._mono_label(
            status_bar,
            "FITS ADU  ·  NO WCS CLAIM",
            color=INK_SOFT,
            size=7,
            bg=NAVY_DARK,
        ).pack(side="right", padx=(12, 0))

        self._build_content_starfield()

    def _build_controls_starfield(self) -> None:
        """只把复现关键参数常驻，复杂调参进入高级区。"""

        controls = tk.Frame(
            self.main,
            bg=PAPER_LIGHT,
            highlightbackground=PAPER_LINE,
            highlightthickness=1,
        )
        controls.pack(fill="x", padx=36, pady=(0, 12))

        header = tk.Frame(controls, bg=PAPER_LIGHT)
        header.pack(fill="x", padx=16, pady=(12, 8))
        heading = tk.Frame(header, bg=PAPER_LIGHT)
        heading.pack(side="left")
        self._mono_label(
            heading,
            "ANALYSIS ROUTE  /  ONE CLICK",
            color=SKY,
            size=8,
            bg=PAPER_LIGHT,
        ).pack(anchor="w")
        self._label(
            heading,
            "一键分析当前帧",
            color=WHITE,
            size=13,
            bold=True,
            bg=PAPER_LIGHT,
        ).pack(anchor="w", pady=(3, 0))

        self.source_export_button = self._star_button(
            header,
            "研究工具 ▾",
            self._show_research_menu,
            kind="ghost",
            padx=9,
            pady=6,
            size=8,
        )
        self.source_export_button.pack(side="right", padx=(8, 0))
        self.cache_button = self._star_button(
            header,
            "清缓存",
            self.clear_detection_cache,
            kind="ghost",
            padx=9,
            pady=6,
            size=8,
        )
        self.cache_button.pack(side="right", padx=(8, 0))
        self.motion_button = self._star_button(
            header,
            "15 帧动目标",
            self.run_sequence_analysis,
            kind="secondary",
            padx=11,
            pady=8,
            size=8,
        )
        self.motion_button.pack(side="right", padx=(8, 0))
        self.evidence_button = self._star_button(
            header,
            "15 帧证据",
            self._show_sequence_evidence,
            kind="ghost",
            padx=9,
            pady=7,
            size=8,
            state="disabled",
        )
        self.evidence_button.pack(side="right", padx=(8, 0))
        self.run_button = self._star_button(
            header,
            "✦  一键分析单张",
            self.run_analysis,
            kind="primary",
            padx=17,
            pady=9,
            size=10,
        )
        self.run_button.pack(side="right", padx=(12, 0))

        route = tk.Frame(controls, bg=PAPER_LIGHT)
        route.pack(fill="x", padx=16, pady=(0, 10))
        steps = (
            ("01", "Gaussian + DoG 提案", AMBER_LIGHT),
            ("02", "原始 ADU 细筛", SKY),
            ("03", "PSF / 值域 / 去混叠", TEMPORAL_STAR_POINT),
            ("04", "可信源输出", MINT),
        )
        for index, (number, label, color) in enumerate(steps):
            step = tk.Frame(route, bg=PAPER_LIGHT)
            step.pack(side="left", fill="x", expand=True)
            self._mono_label(step, number, color=color, size=8, bg=PAPER_LIGHT).pack(anchor="w")
            self._label(step, label, color=INK, size=8, bg=PAPER_LIGHT).pack(anchor="w", pady=(3, 0))
            if index < len(steps) - 1:
                self._mono_label(route, "→", color=PAPER_LINE, size=10, bg=PAPER_LIGHT).pack(
                    side="left",
                    padx=7,
                    pady=(2, 0),
                )

        sequence_progress_row = tk.Frame(controls, bg=PAPER_LIGHT)
        sequence_progress_row.pack(fill="x", padx=16, pady=(0, 8))
        self.sequence_progress_scope_label = self._mono_label(
            sequence_progress_row,
            "SEQ / —",
            color=SKY,
            size=8,
            bg=PAPER_LIGHT,
        )
        self.sequence_progress_scope_label.pack(side="left", padx=(0, 11))
        self.sequence_progress_percent_var = tk.DoubleVar(value=0.0)
        self.sequence_progress_text_var = tk.StringVar(value="15 帧任务待运行")
        self.sequence_progress_bar = ttk.Progressbar(
            sequence_progress_row,
            style="RST19.Horizontal.TProgressbar",
            orient="horizontal",
            mode="determinate",
            maximum=100.0,
            variable=self.sequence_progress_percent_var,
            length=220,
        )
        self.sequence_progress_bar.pack(side="left", fill="x", expand=True, pady=2)
        self.sequence_progress_label = tk.Label(
            sequence_progress_row,
            textvariable=self.sequence_progress_text_var,
            bg=PAPER_LIGHT,
            fg=INK_SOFT,
            font=(MONO, 8),
            anchor="e",
            width=31,
        )
        self.sequence_progress_label.pack(side="right", padx=(11, 0))

        ledger_row = tk.Frame(controls, bg=PAPER_LIGHT)
        ledger_row.pack(fill="x", padx=16, pady=(0, 10))
        self._mono_label(ledger_row, "FRAME LEDGER", color=INK_SOFT, size=7, bg=PAPER_LIGHT).pack(
            side="left",
            padx=(0, 11),
        )
        self.sequence_ledger_canvas = tk.Canvas(
            ledger_row,
            height=20,
            bg=PAPER_LIGHT,
            highlightthickness=0,
            bd=0,
        )
        self.sequence_ledger_canvas.pack(side="left", fill="x", expand=True)
        self.sequence_ledger_canvas.bind("<Configure>", lambda _event: self._draw_sequence_ledger())

        profile = tk.Frame(controls, bg=NAVY_SOFT, highlightbackground=NAVY_SOFT, highlightthickness=1)
        profile.pack(fill="x", padx=16, pady=(0, 11))
        self._mono_label(profile, "论文默认口径", color=AMBER_LIGHT, size=8, bg=NAVY_SOFT).pack(
            side="left",
            padx=(12, 8),
            pady=8,
        )
        self._label(
            profile,
            "hybrid  ·  Gaussian + DoG  ·  原图 ADU 细筛  ·  FWHM 2 px  ·  flux SNR 5",
            color=WHITE,
            size=8,
            bg=NAVY_SOFT,
        ).pack(side="left", pady=8)
        self.advanced_controls_visible = False
        self.advanced_toggle = self._star_button(
            profile,
            "高级参数  ＋",
            self._toggle_advanced_controls,
            kind="ghost",
            padx=8,
            pady=4,
            size=8,
        )
        self.advanced_toggle.pack(side="right", padx=8, pady=4)

        self.advanced_controls_panel = tk.Frame(controls, bg=PAPER_LIGHT)
        advanced = self.advanced_controls_panel
        self._mono_label(advanced, "ADVANCED / REPRODUCIBLE CONTROLS", color=INK_SOFT, size=7, bg=PAPER_LIGHT).pack(
            anchor="w",
            padx=16,
            pady=(1, 7),
        )
        parameter_row = tk.Frame(advanced, bg=PAPER_LIGHT)
        parameter_row.pack(fill="x", padx=16)
        for column in range(6):
            parameter_row.grid_columnconfigure(column, weight=1)

        def entry_field(
            column: int,
            label: str,
            variable: tk.StringVar,
            suffix: str = "",
            width: int = 5,
            *,
            focus_sync: bool = False,
        ) -> None:
            field = tk.Frame(parameter_row, bg=PAPER_LIGHT)
            field.grid(row=0, column=column, sticky="w", padx=(0, 12))
            self._mono_label(field, label, color=INK_SOFT, size=7, bg=PAPER_LIGHT).pack(anchor="w")
            line = tk.Frame(field, bg=PAPER_LIGHT)
            line.pack(anchor="w", pady=(4, 0))
            entry = tk.Entry(
                line,
                textvariable=variable,
                width=width,
                bg=PAPER,
                fg=INK,
                insertbackground=WHITE,
                relief="flat",
                highlightbackground=PAPER_LINE,
                highlightcolor=SKY,
                highlightthickness=1,
                font=(MONO, 9),
            )
            entry.pack(side="left")
            if focus_sync:
                entry.bind("<FocusOut>", lambda _event: self._sync_manual_controls_from_entries())
            if suffix:
                self._mono_label(line, suffix, color=INK_SOFT, size=7, bg=PAPER_LIGHT).pack(
                    side="left",
                    padx=(4, 0),
                )

        self.threshold_var = tk.StringVar(value=f"{GUI_DEFAULT_THRESHOLD_SIGMA:.1f}")
        self.min_distance_var = tk.StringVar(value=str(GUI_DEFAULT_MIN_DISTANCE))
        self.psf_fwhm_var = tk.StringVar(value=f"{GUI_DEFAULT_PSF_FWHM:.1f}")
        self.min_flux_snr_var = tk.StringVar(value=f"{GUI_DEFAULT_MIN_FLUX_SNR:.1f}")
        self.max_sources_var = tk.StringVar(value="")
        self.zero_point_var = tk.StringVar(value="")
        entry_field(0, "候选阈值", self.threshold_var, "σ", focus_sync=True)
        entry_field(1, "最小峰距", self.min_distance_var, "px")
        entry_field(2, "PSF FWHM", self.psf_fwhm_var, "px")
        entry_field(3, "通量门", self.min_flux_snr_var, "σ", focus_sync=True)
        entry_field(4, "单图源上限", self.max_sources_var, "留空=全量", width=7)
        entry_field(5, "测光零点", self.zero_point_var, "可选", width=7)

        proposal_row = tk.Frame(advanced, bg=PAPER_LIGHT)
        proposal_row.pack(fill="x", padx=16, pady=(10, 0))
        self._mono_label(proposal_row, "宽筛提案", color=INK_SOFT, size=7, bg=PAPER_LIGHT).pack(
            side="left",
            padx=(0, 8),
        )
        self.proposal_mode_var = tk.StringVar(value=GUI_DEFAULT_PROPOSAL_MODE)
        for value, label in (
            ("hybrid", "hybrid / Gaussian+DoG"),
            ("gaussian", "Gaussian 基线"),
            ("ensemble", "Starlet 实验"),
        ):
            tk.Radiobutton(
                proposal_row,
                text=label,
                value=value,
                variable=self.proposal_mode_var,
                command=self._mark_manual_tuning_dirty,
                bg=PAPER_LIGHT,
                fg=INK,
                activebackground=PAPER_LIGHT,
                activeforeground=WHITE,
                selectcolor=NAVY_SOFT,
                font=(MONO, 8),
                highlightthickness=0,
                bd=0,
            ).pack(side="left", padx=(0, 12))
        self.local_deblend_var = tk.BooleanVar(value=GUI_DEFAULT_LOCAL_DEBLEND)
        tk.Checkbutton(
            proposal_row,
            text="局部联合 PSF（较慢）",
            variable=self.local_deblend_var,
            command=self._mark_manual_tuning_dirty,
            bg=PAPER_LIGHT,
            fg=INK,
            activebackground=PAPER_LIGHT,
            activeforeground=WHITE,
            selectcolor=NAVY_SOFT,
            font=(MONO, 8),
            highlightthickness=0,
            bd=0,
        ).pack(side="left", padx=(3, 0))

        temporal_row = tk.Frame(advanced, bg=PAPER_LIGHT)
        temporal_row.pack(fill="x", padx=16, pady=(7, 0))
        self._mono_label(temporal_row, "15 帧补提案", color=INK_SOFT, size=7, bg=PAPER_LIGHT).pack(
            side="left",
            padx=(0, 8),
        )
        self.temporal_proposal_mode_var = tk.StringVar(value=GUI_DEFAULT_TEMPORAL_PROPOSAL_MODE)
        for value, label in (
            ("median", "稳健中值"),
            ("coadd", "稳健叠加"),
            ("both", "两者并集"),
        ):
            tk.Radiobutton(
                temporal_row,
                text=label,
                value=value,
                variable=self.temporal_proposal_mode_var,
                command=self._mark_manual_tuning_dirty,
                bg=PAPER_LIGHT,
                fg=INK,
                activebackground=PAPER_LIGHT,
                activeforeground=WHITE,
                selectcolor=NAVY_SOFT,
                font=(MONO, 8),
                highlightthickness=0,
                bd=0,
            ).pack(side="left", padx=(0, 11))
        self._mono_label(temporal_row, "参考 SNR", color=INK_SOFT, size=7, bg=PAPER_LIGHT).pack(
            side="left",
            padx=(5, 4),
        )
        self.temporal_reference_min_snr_var = tk.StringVar(
            value=f"{GUI_DEFAULT_TEMPORAL_REFERENCE_MIN_SNR:.1f}",
        )
        tk.Entry(
            temporal_row,
            textvariable=self.temporal_reference_min_snr_var,
            width=5,
            bg=PAPER,
            fg=INK,
            insertbackground=WHITE,
            relief="flat",
            highlightbackground=PAPER_LINE,
            highlightcolor=SKY,
            highlightthickness=1,
            font=(MONO, 8),
        ).pack(side="left")
        self.sequence_full_var = tk.BooleanVar(value=GUI_DEFAULT_SEQUENCE_FULL)
        tk.Checkbutton(
            temporal_row,
            text="15 帧全量关联",
            variable=self.sequence_full_var,
            command=self._mark_manual_tuning_dirty,
            bg=PAPER_LIGHT,
            fg=INK,
            activebackground=PAPER_LIGHT,
            activeforeground=WHITE,
            selectcolor=NAVY_SOFT,
            font=(MONO, 8),
            highlightthickness=0,
            bd=0,
        ).pack(side="left", padx=(14, 0))
        self.stack_faint_recovery_var = tk.BooleanVar(value=True)
        tk.Checkbutton(
            temporal_row,
            text="叠加暗星恢复",
            variable=self.stack_faint_recovery_var,
            command=self._mark_manual_tuning_dirty,
            bg=PAPER_LIGHT,
            fg=INK,
            activebackground=PAPER_LIGHT,
            activeforeground=WHITE,
            selectcolor=NAVY_SOFT,
            font=(MONO, 8),
            highlightthickness=0,
            bd=0,
        ).pack(side="left", padx=(12, 0))

        manual_row = tk.Frame(advanced, bg=PAPER_LIGHT)
        manual_row.pack(fill="x", padx=16, pady=(9, 1))
        self._mono_label(manual_row, "人工复核", color=AMBER, size=7, bg=PAPER_LIGHT).pack(
            side="left",
            padx=(0, 9),
        )
        threshold_tuning = tk.Frame(manual_row, bg=PAPER_LIGHT)
        threshold_tuning.pack(side="left", fill="x", expand=True, padx=(0, 10))
        self._mono_label(threshold_tuning, "候选 σ", color=INK_SOFT, size=7, bg=PAPER_LIGHT).pack(
            side="left",
            padx=(0, 4),
        )
        self.manual_threshold_scale_var = tk.DoubleVar(value=GUI_DEFAULT_THRESHOLD_SIGMA)
        self.manual_threshold_scale = tk.Scale(
            threshold_tuning,
            from_=MANUAL_THRESHOLD_MIN,
            to=MANUAL_THRESHOLD_MAX,
            resolution=0.5,
            orient="horizontal",
            variable=self.manual_threshold_scale_var,
            showvalue=False,
            highlightthickness=0,
            bd=0,
            bg=PAPER_LIGHT,
            fg=INK_SOFT,
            troughcolor=PAPER_LINE,
            activebackground=AMBER_LIGHT,
            sliderlength=15,
            width=12,
            command=self._on_manual_threshold_changed,
        )
        self.manual_threshold_scale.pack(side="left", fill="x", expand=True)
        self.manual_threshold_value_label = self._mono_label(
            threshold_tuning,
            "4.0σ",
            color=AMBER_LIGHT,
            size=7,
            bg=PAPER_LIGHT,
            width=5,
            anchor="e",
        )
        self.manual_threshold_value_label.pack(side="left", padx=(4, 0))

        snr_tuning = tk.Frame(manual_row, bg=PAPER_LIGHT)
        snr_tuning.pack(side="left", fill="x", expand=True, padx=(0, 10))
        self._mono_label(snr_tuning, "可信 SNR", color=INK_SOFT, size=7, bg=PAPER_LIGHT).pack(
            side="left",
            padx=(0, 4),
        )
        self.manual_snr_scale_var = tk.DoubleVar(value=GUI_DEFAULT_MIN_FLUX_SNR)
        self.manual_snr_scale = tk.Scale(
            snr_tuning,
            from_=MANUAL_SNR_MIN,
            to=MANUAL_SNR_MAX,
            resolution=0.5,
            orient="horizontal",
            variable=self.manual_snr_scale_var,
            showvalue=False,
            highlightthickness=0,
            bd=0,
            bg=PAPER_LIGHT,
            fg=INK_SOFT,
            troughcolor=PAPER_LINE,
            activebackground=MINT,
            sliderlength=15,
            width=12,
            command=self._on_manual_snr_changed,
        )
        self.manual_snr_scale.pack(side="left", fill="x", expand=True)
        self.manual_snr_value_label = self._mono_label(
            snr_tuning,
            "5.0σ",
            color=MINT,
            size=7,
            bg=PAPER_LIGHT,
            width=5,
            anchor="e",
        )
        self.manual_snr_value_label.pack(side="left", padx=(4, 0))

        feedback_tuning = tk.Frame(manual_row, bg=PAPER_LIGHT)
        feedback_tuning.pack(side="left", padx=(0, 7))
        self._mono_label(feedback_tuning, "判断", color=INK_SOFT, size=7, bg=PAPER_LIGHT).pack(
            side="left",
            padx=(0, 4),
        )
        self.manual_feedback_judgement_var = tk.StringVar(value="当前平衡")
        self.manual_feedback_judgement = ttk.Combobox(
            feedback_tuning,
            textvariable=self.manual_feedback_judgement_var,
            values=("保留弱星", "减少伪点", "当前平衡"),
            state="readonly",
            width=9,
            style="RST19.TCombobox",
        )
        self.manual_feedback_judgement.pack(side="left")
        self.manual_apply_button = self._star_button(
            manual_row,
            "应用并分析",
            self.run_analysis,
            kind="primary",
            padx=10,
            pady=6,
            size=8,
        )
        self.manual_apply_button.pack(side="left", padx=(7, 5))
        self.manual_feedback_button = self._star_button(
            manual_row,
            "记录反馈",
            self._record_manual_feedback,
            kind="ghost",
            padx=9,
            pady=6,
            size=8,
            state="disabled",
        )
        self.manual_feedback_button.pack(side="left")
        self.manual_tuning_status_label = self._mono_label(
            advanced,
            "默认口径已锁定；高级参数只在展开后可改，反馈不会自动修改算法。",
            color=INK_SOFT,
            size=7,
            bg=PAPER_LIGHT,
        )
        self.manual_tuning_status_label.pack(anchor="w", padx=90, pady=(2, 8))

        self.research_menu = tk.Menu(
            self,
            tearoff=False,
            bg=PAPER_LIGHT,
            fg=INK,
            activebackground=NAVY_SOFT,
            activeforeground=WHITE,
            bd=1,
            relief="solid",
            font=(SANS, 9),
        )
        self.research_menu.add_command(label="打开星表核验", command=self._show_catalog_match)
        self.research_menu.add_separator()
        self.research_menu.add_command(label="导出星点研究表", command=self.export_source_study, state="disabled")
        self.research_export_menu_index = int(self.research_menu.index("end"))

    def _toggle_advanced_controls(self) -> None:
        panel = self.__dict__.get("advanced_controls_panel")
        button = self.__dict__.get("advanced_toggle")
        if panel is None or button is None:
            return
        if self.advanced_controls_visible:
            panel.pack_forget()
            self.advanced_controls_visible = False
            button.config(text="高级参数  ＋")
        else:
            panel.pack(fill="x")
            self.advanced_controls_visible = True
            button.config(text="收起高级参数  −")

    def _build_metrics_starfield(self) -> None:
        metrics = tk.Frame(self.main, bg=PAPER)
        metrics.pack(fill="x", padx=36, pady=(0, 12))
        for column, weight in enumerate((5, 4, 4)):
            metrics.grid_columnconfigure(column, weight=weight)
        self.metric_values: dict[str, tk.Label] = {}

        def card(
            column: int,
            eyebrow: str,
            title: str,
            accent: str,
            background: str = PAPER_LIGHT,
        ) -> tk.Frame:
            panel = tk.Frame(
                metrics,
                bg=background,
                highlightbackground=PAPER_LINE if accent != MINT else MINT,
                highlightthickness=1,
            )
            panel.grid(row=0, column=column, sticky="nsew", padx=(0 if column == 0 else 6, 0))
            top = tk.Frame(panel, bg=background)
            top.pack(fill="x", padx=14, pady=(10, 0))
            self._mono_label(top, eyebrow, color=accent, size=7, bg=background).pack(side="left")
            self._mono_label(top, title, color=INK_SOFT, size=7, bg=background).pack(side="right")
            return panel

        source_box = card(0, "DETECTION ACCOUNT", "not star truth", AMBER_LIGHT, NAVY_SOFT)
        source_values = tk.Frame(source_box, bg=NAVY_SOFT)
        source_values.pack(fill="x", padx=14, pady=(5, 11))
        for key, title, color in (
            ("candidate", "候选峰", AMBER_LIGHT),
            ("returned", "可信源", WHITE),
        ):
            group = tk.Frame(source_values, bg=NAVY_SOFT)
            group.pack(side="left", expand=True, fill="x")
            self._mono_label(group, title, color=INK_SOFT, size=7, bg=NAVY_SOFT).pack(anchor="w")
            value = self._label(group, "—", color=color, size=21, bg=NAVY_SOFT)
            value.pack(anchor="w", pady=(2, 0))
            self.metric_values[key] = value

        baseline_box = card(1, "IMAGE BASELINE", "raw ADU", SKY)
        baseline_values = tk.Frame(baseline_box, bg=PAPER_LIGHT)
        baseline_values.pack(fill="x", padx=14, pady=(5, 11))
        for key, title in (("background", "背景"), ("noise", "噪声")):
            group = tk.Frame(baseline_values, bg=PAPER_LIGHT)
            group.pack(side="left", expand=True, fill="x")
            self._mono_label(group, title, color=INK_SOFT, size=7, bg=PAPER_LIGHT).pack(anchor="w")
            value = self._label(group, "—", color=INK, size=19, bg=PAPER_LIGHT)
            value.pack(anchor="w", pady=(2, 0))
            self.metric_values[key] = value

        faintest_box = card(2, "FAINTEST ACCEPTED", "m_inst", MINT)
        faintest_value = self._label(faintest_box, "—", color=MINT, size=21, bg=PAPER_LIGHT)
        faintest_value.pack(anchor="w", padx=14, pady=(5, 11))
        self.metric_values["faintest"] = faintest_value

    def _build_content_starfield(self) -> None:
        content = tk.Frame(self.main, bg=PAPER)
        self.content_panel = content
        content.grid_columnconfigure(0, weight=3)
        content.grid_columnconfigure(1, weight=2)
        content.grid_rowconfigure(0, weight=1)
        content.grid_rowconfigure(1, weight=0)

        viewer = tk.Frame(
            content,
            bg=PAPER_LIGHT,
            highlightbackground=PAPER_LINE,
            highlightthickness=1,
        )
        viewer.grid(row=0, column=0, rowspan=2, sticky="nsew", padx=(0, 10))
        viewer_header = tk.Frame(viewer, bg=PAPER_LIGHT)
        viewer_header.pack(fill="x", padx=16, pady=(13, 9))
        mode_box = tk.Frame(viewer_header, bg=PAPER_LIGHT)
        mode_box.pack(side="right", anchor="s")
        self.overlay_mode_var = tk.StringVar(value="quality")
        # 可信源是主证据层；运动轨迹作为可叠加层，不再抢占整张图的显示权。
        # 默认打开叠加，单帧长线/15 帧运动证据出现时可以直接对照星点。
        self.motion_overlay_var = tk.BooleanVar(value=True)
        self._mono_label(mode_box, "PRIMARY LAYER · COMPOSITE", color=INK_SOFT, size=7, bg=PAPER_LIGHT).pack(anchor="e")
        choice_row = tk.Frame(mode_box, bg=PAPER_LIGHT)
        choice_row.pack(anchor="e", pady=(3, 0))
        choices = (
            ("quality", "可信源", MINT),
            ("stable", "稳定星场", SKY),
            ("stack-faint", "叠加暗星", STACK_FAINT_POINT),
            ("candidates", "全部候选", AMBER_LIGHT),
            ("motion", "运动候选", MOTION_TRAIL),
            ("catalog", "星表匹配", TEMPORAL_STAR_POINT),
        )
        for value, label, color in choices:
            tk.Radiobutton(
                choice_row,
                text=label,
                variable=self.overlay_mode_var,
                value=value,
                command=self._on_overlay_mode_changed,
                bg=PAPER_LIGHT,
                fg=color,
                activebackground=PAPER_LIGHT,
                activeforeground=WHITE,
                selectcolor=NAVY_SOFT,
                relief="flat",
                bd=0,
                font=(SANS, 8),
            ).pack(side="left", padx=(7, 0))
        composite_row = tk.Frame(mode_box, bg=PAPER_LIGHT)
        composite_row.pack(anchor="e", pady=(3, 0))
        self._mono_label(composite_row, "叠加", color=INK_SOFT, size=7, bg=PAPER_LIGHT).pack(side="left", padx=(0, 3))
        self.motion_overlay_checkbutton = tk.Checkbutton(
            composite_row,
            text="运动轨迹",
            variable=self.motion_overlay_var,
            command=self._on_overlay_mode_changed,
            bg=PAPER_LIGHT,
            fg=MOTION_TRAIL,
            activebackground=PAPER_LIGHT,
            activeforeground=WHITE,
            selectcolor=NAVY_SOFT,
            relief="flat",
            bd=0,
            highlightthickness=0,
            font=(SANS, 8),
        )
        self.motion_overlay_checkbutton.pack(side="left")
        self._mono_label(viewer_header, "FRAME VIEWER  /  RAW + OVERLAY", color=SKY, size=7, bg=PAPER_LIGHT).pack(anchor="w")
        self.frame_title_label = self._label(
            viewer_header,
            "选择一个观测帧",
            color=WHITE,
            size=14,
            bold=True,
            bg=PAPER_LIGHT,
        )
        self.frame_title_label.pack(anchor="w", pady=(3, 0))

        display_row = tk.Frame(viewer, bg=PAPER_LIGHT)
        display_row.pack(fill="x", padx=16, pady=(0, 8))
        self._mono_label(display_row, "DISPLAY", color=INK_SOFT, size=7, bg=PAPER_LIGHT).pack(
            side="left",
            padx=(0, 7),
        )
        self.preview_mode_var = tk.StringVar(value=PREVIEW_MODE_ENHANCED)
        for value, label in PREVIEW_MODE_LABELS.items():
            tk.Radiobutton(
                display_row,
                text=label,
                variable=self.preview_mode_var,
                value=value,
                command=self._on_preview_mode_changed,
                bg=PAPER_LIGHT,
                fg=INK,
                activebackground=PAPER_LIGHT,
                activeforeground=WHITE,
                selectcolor=NAVY_SOFT,
                relief="flat",
                bd=0,
                font=(SANS, 8),
            ).pack(side="left", padx=(7, 0))

        self.canvas = tk.Canvas(
            viewer,
            bg=NAVY_DARK,
            highlightbackground=PAPER_LINE,
            highlightthickness=1,
        )
        self.canvas.pack(fill="both", expand=True, padx=10, pady=(0, 9))
        self.canvas.bind("<Configure>", lambda _event: self._draw_preview())
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind("<Button-4>", lambda event: self._zoom_at(event, 1.15))
        self.canvas.bind("<Button-5>", lambda event: self._zoom_at(event, 1 / 1.15))
        self.canvas.bind("<ButtonPress-1>", self._on_pan_start)
        self.canvas.bind("<B1-Motion>", self._on_pan_move)
        self.canvas.bind("<ButtonRelease-1>", lambda _event: setattr(self, "drag_start", None))
        self.canvas.bind("<Motion>", self._on_canvas_motion)
        self.canvas.bind("<Leave>", lambda _event: self._clear_hover())
        self.hover_info_var = tk.StringVar(value="将鼠标移到星点上查看 ID、SNR、形态、值域与落选原因")
        self.hover_info_label = tk.Label(
            viewer,
            textvariable=self.hover_info_var,
            bg=PAPER_LIGHT,
            fg=INK_SOFT,
            font=(MONO, 8),
            anchor="w",
            justify="left",
            width=1,
            height=4,
            wraplength=640,
        )
        self.hover_info_label.pack(fill="x", padx=16, pady=(0, 3))
        viewer.bind("<Configure>", self._on_viewer_configure)
        self.overlay_hint_var = tk.StringVar(
            value="显示：增强显示 · 可信源主层 · 洋红=运动轨迹叠加 · 滚轮缩放≤15×",
        )
        tk.Label(
            viewer,
            textvariable=self.overlay_hint_var,
            bg=PAPER_LIGHT,
            fg=INK_SOFT,
            font=(MONO, 7),
            anchor="w",
        ).pack(anchor="w", padx=16, pady=(0, 12))

        detail = tk.Frame(
            content,
            bg=PAPER_LIGHT,
            highlightbackground=PAPER_LINE,
            highlightthickness=1,
        )
        self.detail_panel = detail
        detail.grid(row=0, column=1, sticky="nsew", pady=(0, 10))
        self.faintest_section_label = self._mono_label(
            detail,
            "FAINTEST ACCEPTED SOURCE",
            color=MINT,
            size=7,
            bg=PAPER_LIGHT,
        )
        self.faintest_section_label.pack(anchor="w", padx=16, pady=(14, 0))
        self.faintest_detail = self._label(
            detail,
            "尚未运行分析",
            color=WHITE,
            size=14,
            bold=True,
            bg=PAPER_LIGHT,
            justify="left",
            anchor="w",
            wraplength=280,
        )
        self.faintest_detail.pack(fill="x", padx=16, pady=(8, 3))
        self.faintest_note = self._label(
            detail,
            "按通量 SNR、点源形状、边缘、掩膜和饱和状态筛选；m_inst 是仪器星等，有零点后才显示 m_cal。",
            color=INK_SOFT,
            size=8,
            bg=PAPER_LIGHT,
            justify="left",
            wraplength=330,
        )
        self.faintest_note.pack(fill="x", padx=16, pady=(0, 14))
        self.detail_separator = tk.Frame(detail, bg=PAPER_LINE, height=1)
        self.detail_separator.pack(fill="x", padx=16)
        self.evidence_section_label = self._mono_label(
            detail,
            "15-FRAME EVIDENCE",
            color=SKY,
            size=7,
            bg=PAPER_LIGHT,
        )
        self.evidence_section_label.pack(anchor="w", padx=16, pady=(12, 5))
        self.sequence_evidence_label = self._label(
            detail,
            "尚未完成序列分析\n点击“15 帧动目标”后显示时间、配准和轨迹摘要",
            color=INK_SOFT,
            size=8,
            bg=PAPER_LIGHT,
            justify="left",
            anchor="w",
            wraplength=330,
        )
        self.sequence_evidence_label.pack(fill="x", padx=16, pady=(0, 14))
        detail.bind("<Configure>", self._on_detail_configure)

        register = tk.Frame(
            content,
            bg=PAPER_LIGHT,
            highlightbackground=PAPER_LINE,
            highlightthickness=1,
        )
        self.source_register_panel = register
        register.grid(row=1, column=1, sticky="nsew")
        table_header = tk.Frame(register, bg=PAPER_LIGHT)
        table_header.pack(fill="x", padx=16, pady=(11, 6))
        self._mono_label(table_header, "SOURCE REGISTER  /  TOP FLUX SNR", color=SKY, size=7, bg=PAPER_LIGHT).pack(
            side="left",
        )
        self.table_count_label = self._mono_label(table_header, "0 rows", color=AMBER_LIGHT, size=7, bg=PAPER_LIGHT)
        self.table_count_label.pack(side="right")
        columns = ("id", "xy", "peak", "snr", "mag")
        self.source_tree = ttk.Treeview(register, columns=columns, show="headings", height=2)
        for column, title, width in (
            ("id", "ID", 40),
            ("xy", "X / Y", 88),
            ("peak", "PEAK", 52),
            ("snr", "SNR", 48),
            ("mag", "m_inst", 54),
        ):
            self.source_tree.heading(column, text=title)
            self.source_tree.column(column, width=width, minwidth=34, anchor="w", stretch=True)
        self.source_tree_scrollbar = ttk.Scrollbar(register, orient="vertical", command=self.source_tree.yview)
        self.source_tree.configure(yscrollcommand=self.source_tree_scrollbar.set)
        self.source_tree_scrollbar.pack(side="right", fill="y", padx=(0, 10), pady=(0, 10))
        self.source_tree.pack(side="left", fill="both", expand=True, padx=(10, 0), pady=(0, 10))
        self.content_compact = False
        content.bind("<Configure>", self._on_content_configure)
        content.pack(fill="both", expand=True, padx=36, pady=(0, 15))

    def _set_progress(self, value: float, text: str | None = None) -> None:
        """更新底部进度条；所有后台线程只能通过主线程轮询调用本方法。"""

        if not hasattr(self, "progress_percent_var"):
            return
        bounded = max(0.0, min(100.0, float(value)))
        self.progress_percent_var.set(bounded)
        if text is not None:
            self.progress_text_var.set(text)

    def _set_sequence_progress(self, value: float, text: str | None = None) -> None:
        """更新检测参数区的显式序列进度。"""

        if not hasattr(self, "sequence_progress_percent_var"):
            return
        bounded = max(0.0, min(100.0, float(value)))
        self.sequence_progress_percent_var.set(bounded)
        if text is not None:
            self.sequence_progress_text_var.set(text)

    def _reset_sequence_ledger(self, total: int | None = None, *, state: str = "pending") -> None:
        """重置 15 帧账本；状态只在主线程更新。"""

        count = int(total if total is not None else len(self.frames))
        self.sequence_frame_states = [state] * max(0, count)
        self._draw_sequence_ledger()

    def _update_sequence_ledger(self, value: float, progress_text: str) -> None:
        """从阶段进度事件推断逐帧状态，不把日志当作唯一反馈。"""

        states = self.__dict__.get("sequence_frame_states", [])
        if self.active_job_kind != "sequence" or not states:
            return
        completed_match = re.search(r"(?:已完成|完成)\s*(\d+)\s*/\s*(\d+)\s*(?:帧)?", progress_text)
        if completed_match:
            completed = min(len(states), max(0, int(completed_match.group(1))))
            for index in range(completed):
                states[index] = "completed"
        frame_match = re.search(r"F(\d{1,3})", progress_text)
        if frame_match:
            frame_index = int(frame_match.group(1)) - 1
            if 0 <= frame_index < len(states):
                if float(value) >= 100.0 or "完成" in progress_text and "帧" in progress_text:
                    states[frame_index] = "completed"
                elif states[frame_index] != "completed":
                    states[frame_index] = "running"
        if float(value) >= 100.0:
            states[:] = ["completed"] * len(states)
        self._draw_sequence_ledger()

    def _mark_sequence_ledger(self, state: str) -> None:
        states = self.__dict__.get("sequence_frame_states", [])
        if states:
            states[:] = [state] * len(states)
            self._draw_sequence_ledger()

    def _draw_sequence_ledger(self) -> None:
        """绘制紧凑的 F01…F15 状态带。"""

        canvas = self.__dict__.get("sequence_ledger_canvas")
        if canvas is None:
            return
        try:
            width = max(1, int(canvas.winfo_width()))
        except tk.TclError:
            return
        canvas.delete("all")
        states = self.sequence_frame_states
        if not states:
            canvas.create_text(0, 11, anchor="w", text="无 FITS", fill=INK_SOFT, font=(MONO, 8))
            return
        colors = {
            "pending": PAPER_LINE,
            "running": AMBER_LIGHT,
            "completed": MINT,
            "error": "#c46656",
        }
        gap = 3
        cell_width = max(9.0, (width - gap * (len(states) - 1)) / len(states))
        for index, state in enumerate(states):
            x0 = index * (cell_width + gap)
            x1 = min(width, x0 + cell_width)
            color = colors.get(state, PAPER_LINE)
            canvas.create_rectangle(x0, 2, x1, 19, fill=color, outline=color)
            if cell_width >= 22:
                canvas.create_text((x0 + x1) / 2, 10.5, text=f"F{index + 1:02d}", fill=NAVY_DARK if state != "pending" else INK_SOFT, font=(MONO, 7))

    def _update_research_menu_state(self) -> None:
        """只把当前可执行的研究动作开放给菜单，避免隐藏按钮竞争布局高度。"""

        menu = self.__dict__.get("research_menu")
        menu_index = self.__dict__.get("research_export_menu_index")
        if menu is None or menu_index is None:
            return
        state = "normal" if (not self.busy and self.analysis is not None and not self.source_export_in_progress) else "disabled"
        menu.entryconfigure(int(menu_index), state=state)

    def _show_research_menu(self) -> None:
        """在主参数栏打开星表核验和论文式星点表导出入口。"""

        menu = self.__dict__.get("research_menu")
        button = self.__dict__.get("source_export_button")
        if menu is None or button is None:
            return
        self._update_research_menu_state()
        try:
            menu.tk_popup(button.winfo_rootx(), button.winfo_rooty() + button.winfo_height())
        finally:
            menu.grab_release()

    def _load_frames(self) -> None:
        self.frames = sorted(self.data_dir.glob("*.fits"))
        self._reset_sequence_ledger(len(self.frames))
        self.frame_list.delete(0, tk.END)
        for index, path in enumerate(self.frames, start=1):
            self.frame_list.insert(tk.END, f"{index:02d}  {path.stem}")
        self.frame_count_label.config(text=str(len(self.frames)))
        self.sequence_progress_scope_label.config(text=f"SEQ / {len(self.frames)}")
        if self.frames:
            self.frame_list.selection_set(0)
            self.frame_list.activate(0)
            self.selected_frame = self.frames[0]
            self._select_frame(self.selected_frame)
        else:
            self._set_sequence_progress(0.0, "无 FITS · 序列不可用")
            self.status_var.set(f"数据目录为空 · {self.data_dir}")

    def _on_frame_selected(self, _event: tk.Event) -> None:
        selection = self.frame_list.curselection()
        if not selection:
            return
        self._select_frame(self.frames[selection[0]])

    def _select_frame(self, frame_path: Path) -> None:
        """切帧时先使旧结果失效，再异步载入新帧的无标注预览。"""

        self.frame_token += 1
        token = self.frame_token
        self.selected_frame = frame_path
        self._set_job_controls()
        self.frame_title_label.config(text=frame_path.stem)
        self.analysis = None
        self.catalog_analysis = None
        self.catalog_match_result = None
        self.catalog_wcs = None
        self.catalog_calibration = None
        self.catalog_frame_path = None
        self.long_trails = ()
        self.source_export_in_progress = False
        self.preview = None
        self.preview_variants = {}
        self.preview_shape = None
        self.source_grid = {}
        self.hover_source_id = None
        self.hover_source = None
        self.hover_motion_track = None
        self.hover_trusted_track = None
        self.hover_catalog_match = None
        self.preview_zoom = 1.0
        self.preview_pan_x = 0.0
        self.preview_pan_y = 0.0
        # 新帧尚未完成单帧检测时，不能沿用上一帧可能选中的“全部候选”
        # 图层；否则用户会看到密集的审计候选，并误以为降噪/质量筛选失效。
        # 如果已有 15 帧结果，则保留运动/稳定图层，让序列证据仍能跨帧查看。
        if self.sequence_result is None:
            self.overlay_mode_var.set("quality")
            self._update_overlay_hint()
        self._reset_result_widgets()
        self._set_job_controls()
        self._set_progress(0.0, "就绪")
        if not self.busy and self.sequence_result is None:
            self._set_sequence_progress(0.0, f"{len(self.frames)} 帧待运行")
        self.canvas.delete("all")
        self.hover_info_var.set("正在载入当前帧预览 · 尚未运行检测")
        self.status_var.set(f"已切换 {frame_path.name} · 正在载入预览")

        def worker() -> None:
            try:
                frame = read_fits(frame_path)
                previews = _make_preview_variants(frame.data)
                self.result_queue.put(("preview", token, (frame_path, frame.data.shape, previews)))
            except Exception as exc:  # noqa: BLE001 - worker must return a user-facing error
                self.result_queue.put(("preview-error", token, exc))

        threading.Thread(target=worker, daemon=True).start()

    def _read_parameters(self) -> tuple[float, int, int | None, float | None, float, float]:
        try:
            threshold = float(self.threshold_var.get())
            min_distance = int(self.min_distance_var.get())
            max_text = self.max_sources_var.get().strip()
            max_sources = int(max_text) if max_text else None
            zero_text = self.zero_point_var.get().strip()
            zero_point = float(zero_text) if zero_text else None
            psf_fwhm = float(self.psf_fwhm_var.get())
            min_flux_snr = float(self.min_flux_snr_var.get())
        except ValueError as exc:
            raise ValueError("阈值、最小间距、返回上限、零点、PSF FWHM 和通量 SNR 必须是有效数字") from exc
        if not 0 < threshold <= 20:
            raise ValueError("检测阈值应在 0 至 20σ 之间")
        if not 1 <= min_distance <= 100:
            raise ValueError("最小间距应在 1 至 100 像素之间")
        if max_sources is not None and max_sources < 1:
            raise ValueError("返回上限留空表示全量，否则必须为正整数")
        if psf_fwhm <= 0 or min_flux_snr <= 0:
            raise ValueError("PSF FWHM 和通量 SNR 必须为正数")
        return threshold, min_distance, max_sources, zero_point, psf_fwhm, min_flux_snr

    @staticmethod
    def _format_tuning_value(value: float) -> str:
        return f"{float(value):.1f}"

    def _set_manual_tuning_status(self, text: str, *, color: str = INK_SOFT) -> None:
        label = self.__dict__.get("manual_tuning_status_label")
        if label is not None:
            label.config(text=text, fg=color)

    def _mark_manual_tuning_dirty(self) -> None:
        self.manual_tuning_dirty = True
        self._set_manual_tuning_status("阈值已调整 · 点击“应用并分析”后才能记录这次反馈", color=AMBER)

    def _on_manual_threshold_changed(self, value: str) -> None:
        try:
            threshold = float(value)
        except (TypeError, ValueError):
            return
        self.threshold_var.set(self._format_tuning_value(threshold))
        self.manual_threshold_value_label.config(text=f"{threshold:.1f}σ")
        self._mark_manual_tuning_dirty()

    def _on_manual_snr_changed(self, value: str) -> None:
        try:
            min_flux_snr = float(value)
        except (TypeError, ValueError):
            return
        self.min_flux_snr_var.set(self._format_tuning_value(min_flux_snr))
        self.manual_snr_value_label.config(text=f"{min_flux_snr:.1f}σ")
        self._mark_manual_tuning_dirty()

    def _sync_manual_controls_from_entries(self) -> None:
        """把手动输入框同步到滑块；输入框仍允许探测器支持的更宽范围。"""

        try:
            threshold = float(self.threshold_var.get())
            min_flux_snr = float(self.min_flux_snr_var.get())
        except (TypeError, ValueError):
            return
        threshold_slider = min(MANUAL_THRESHOLD_MAX, max(MANUAL_THRESHOLD_MIN, threshold))
        snr_slider = min(MANUAL_SNR_MAX, max(MANUAL_SNR_MIN, min_flux_snr))
        self.manual_threshold_scale_var.set(threshold_slider)
        self.manual_snr_scale_var.set(snr_slider)
        self.manual_threshold_value_label.config(text=f"{threshold:.1f}σ")
        self.manual_snr_value_label.config(text=f"{min_flux_snr:.1f}σ")
        self._mark_manual_tuning_dirty()

    def _record_manual_feedback(self) -> None:
        """保存一次用户确认过的参数与结果，作为后续离线调参样本。"""

        if self.busy or self.analysis is None or self.selected_frame is None:
            return
        if self.manual_tuning_dirty:
            self.status_var.set("当前阈值尚未应用 · 请先点击“应用并分析”，再记录反馈")
            return
        try:
            threshold, min_distance, _max_sources, _zero_point, psf_fwhm, min_flux_snr = self._read_parameters()
        except ValueError as exc:
            messagebox.showerror("参数错误", str(exc))
            return

        detection = self.analysis.detection
        flag_counts = Counter(
            flag
            for source in detection.sources
            for flag in source.flags
        )
        feedback = ManualThresholdFeedback(
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            frame_path=str(self.selected_frame.resolve()),
            threshold_sigma=threshold,
            min_flux_snr=min_flux_snr,
            min_distance=min_distance,
            psf_fwhm=psf_fwhm,
            candidate_count=int(detection.candidate_count),
            quality_count=int(detection.star_count),
            returned_count=int(detection.returned_count),
            flag_counts=dict(sorted(flag_counts.items())),
            judgement=self.manual_feedback_judgement_var.get() or "当前平衡",
        )
        try:
            append_manual_feedback(self.manual_feedback_path, feedback)
            try:
                feedback_count = len(load_manual_feedback(self.manual_feedback_path))
            except ValueError:
                feedback_count = None
        except OSError as exc:
            messagebox.showerror("反馈保存失败", str(exc))
            return
        count_text = f"第 {feedback_count} 条" if feedback_count is not None else "已写入"
        self._set_manual_tuning_status(f"{count_text}人工反馈已保存 · 后续调参可读取", color=MINT)
        self.status_var.set(f"已保存人工阈值反馈 · {self.manual_feedback_path}")

    def _set_job_controls(self) -> None:
        """根据当前后台任务状态同步按钮，避免切帧把旧任务误报成空闲。"""

        if self.busy:
            self.run_button.config(state="disabled", text="正在分析单张…")
            busy_text = "15 帧分析中…" if self.active_job_kind == "sequence" else "正在分析…"
            self.motion_button.config(state="disabled", text=busy_text)
            self.cache_button.config(state="normal")
            self.evidence_button.config(state="disabled")
        else:
            self.run_button.config(state="normal", text="✦  一键分析单张")
            self.motion_button.config(state="normal", text="15 帧动目标")
            self.cache_button.config(state="normal")
            self.evidence_button.config(state="normal" if self.sequence_result is not None else "disabled")
        if "manual_apply_button" in self.__dict__:
            self.manual_apply_button.config(state="disabled" if self.busy or self.selected_frame is None else "normal")
            self.manual_feedback_button.config(
                state="normal" if (not self.busy and self.analysis is not None) else "disabled"
            )
            scale_state = "disabled" if self.busy else "normal"
            self.manual_threshold_scale.config(state=scale_state)
            self.manual_snr_scale.config(state=scale_state)
        if "source_export_button" in self.__dict__:
            self.source_export_button.config(
                state="disabled" if (self.busy or self.source_export_in_progress) else "normal"
            )
            self._update_research_menu_state()

    def _selected_frame_index(self) -> int | None:
        if self.selected_frame is None:
            return None
        try:
            return self.frames.index(self.selected_frame)
        except ValueError:
            return None

    def _sequence_frame_index(self) -> int | None:
        """Return the selected frame's index in the loaded sequence result."""

        if self.sequence_result is None:
            return self._selected_frame_index()
        return sequence_frame_index_for_frame(self.sequence_result, self.selected_frame, self.frames)

    def _on_preview_mode_changed(self) -> None:
        """切换观察层；不重新检测，也不改变原始 ADU 或缓存结果。"""

        mode = self.preview_mode_var.get()
        preview = self.preview_variants.get(mode)
        if preview is None:
            self.status_var.set("当前显示层还在载入 · 检测仍使用原始 FITS")
            return
        self.preview = preview
        # 三个显示层尺寸相同，保留用户当前缩放和平移位置，便于逐层核对
        # 同一个候选；只有切帧时才重置视图。
        self._update_overlay_hint()
        self.status_var.set(
            f"图像显示：{PREVIEW_MODE_LABELS.get(mode, mode)} · 仅影响观察，检测仍使用原始 ADU"
        )
        self._draw_preview()

    def _apply_preview_result(
        self,
        shape: tuple[int, int],
        previews: dict[str, Image.Image],
    ) -> None:
        """Commit an asynchronously rendered preview to the main-thread UI.

        Preview loading is not a background *analysis* job, so ``busy`` remains
        false while the worker runs.  The completion path must nevertheless
        refresh both status and controls; otherwise the visible state remains
        stuck on ``正在载入预览`` even though the image has already arrived.
        """

        self.preview_shape = shape
        self.preview_variants = dict(previews)
        selected_mode = self.preview_mode_var.get()
        self.preview = self.preview_variants.get(selected_mode) or self.preview_variants.get(
            PREVIEW_MODE_ENHANCED
        )
        self.preview_zoom = 1.0
        self.preview_pan_x = 0.0
        self.preview_pan_y = 0.0
        mode_label = PREVIEW_MODE_LABELS.get(selected_mode, "增强显示")
        self.hover_info_var.set(f"预览已载入 · {mode_label} · 点击“一键分析单张”运行检测")
        self.status_var.set(f"预览已载入 · {mode_label} · 检测使用原始 FITS ADU")
        self._set_job_controls()
        self._update_overlay_hint()
        self._draw_preview()

    def _motion_overlay_enabled(self) -> bool:
        """Return whether the motion evidence should be composited over the primary layer."""

        # 部分 GUI 单元测试用 ``object.__new__`` 构造轻量对象，没有初始化
        # Tk 的 ``self.tk``；不能用 getattr 触发 Tkinter 的属性回退链。
        variable = self.__dict__.get("motion_overlay_var")
        if variable is None:
            return False
        try:
            return bool(variable.get())
        except (AttributeError, tk.TclError):
            return False

    def _on_overlay_mode_changed(self) -> None:
        mode = self.overlay_mode_var.get()
        motion_overlay = self._motion_overlay_enabled()
        if mode == "quality":
            overlay_text = " + 运动轨迹叠加" if motion_overlay else ""
            self.status_var.set(f"可信源主层{overlay_text} · 质量通过的星点优先显示")
        elif mode == "motion" and self.sequence_result is None and not self.long_trails:
            self.status_var.set("尚未完成 15 帧动目标分析 · 请先点击“分析 15 帧动目标”")
        elif mode == "stable" and self.sequence_result is None:
            self.status_var.set("尚未完成 15 帧分析 · 稳定星场需要跨帧持续性证据")
        elif mode == "stable":
            frame_index = self._sequence_frame_index()
            visible_count = (
                len(stable_points_for_frame(self.sequence_result, frame_index))
                if frame_index is not None
                else 0
            )
            frame_text = f"F{frame_index + 1:02d}" if frame_index is not None else "当前帧"
            self.status_var.set(
                f"稳定星场层 · {frame_text} 显示 {visible_count:,} 个点 · "
                "绿色=严格静态，青色=滤波共识，紫色=时间中值补检"
            )
        elif mode == "stack-faint":
            if self.sequence_result is None:
                self.status_var.set("尚未完成 15 帧分析 · 叠加暗星层需要跨帧降噪参考图")
            else:
                frame_index = self._sequence_frame_index()
                visible_count = (
                    len(stack_faint_points_for_frame(self.sequence_result, frame_index))
                    if frame_index is not None
                    else 0
                )
                frame_text = f"F{frame_index + 1:02d}" if frame_index is not None else "当前帧"
                self.status_var.set(
                    f"叠加暗星层 · {frame_text} 显示 {visible_count:,} 个点 · "
                    "绿灰=叠加恢复的暗星，经逐帧强制测光确认，非官方逐星真值"
                )
        elif mode == "motion":
            if self.sequence_result is None:
                self.status_var.set("单帧长轨迹层 · 当前橙线只是形状候选，需跨帧才能确认运动目标")
            else:
                self.status_var.set("运动候选层 · 金橙=高速点源轨迹 · 洋红=线状目标 · 橙=待复核线")
        elif mode == "catalog":
            if self.catalog_match_result is None:
                self.status_var.set("尚未完成星表核验 · 请打开“星表核验”输入目录和已标定 WCS")
            else:
                self.status_var.set("星表匹配层 · 绿色标出目录预测位置，连线显示检测位置与预测位置残差")
        elif mode == "candidates":
            self.status_var.set("全部候选层 · 包含被质量规则剔除的审计候选")
        self._update_overlay_hint()
        self._draw_preview()

    def _update_overlay_hint(self) -> None:
        # 说明保持短小，避免提示文本反向撑宽 viewer、挤压右侧证据栏；
        # 完整的显示口径写入 README/DESIGN。
        display_mode = getattr(self, "preview_mode_var", None)
        display_value = display_mode.get() if display_mode is not None else PREVIEW_MODE_ENHANCED
        display_note = f"图像：{PREVIEW_MODE_LABELS.get(display_value, '增强显示')}"
        hints = {
            "quality": f"{display_note} · 滚轮缩放≤15× · 高亮青点 = Gaussian质量源 · 亮黄点 = 补充算法质量源 · 绿色环 = 最暗可信源{(' · 洋红线 = 运动轨迹叠加' if self._motion_overlay_enabled() else ' · 可勾选叠加运动轨迹')}",
            "candidates": f"{display_note} · 滚轮缩放≤15× · 青点 = Gaussian质量源 · 黄点 = 补充算法质量源 · 紫红点 = 被剔除候选（悬停查看原因）",
            "stable": f"{display_note} · 亮青绿点 = 严格静态源 · 亮蓝点 = 持续源候选 · 该层是序列工作集，不等于全图恒星总数",
            "stack-faint": f"{display_note} · 绿灰点 = 叠加参考图恢复的暗星 · 经逐帧强制测光确认 · 低置信待复核，非官方逐星真值",
            "motion": f"{display_note} · 滚轮缩放≤15× · 金橙实线 = 高速点源 moving · 洋红线 = 线状 moving · 青绿虚线 = 末帧 +5 帧图像平面外推 · 橙线 = 单帧/待复核",
            "catalog": f"{display_note} · 滚轮缩放≤15× · 高亮青点 = 图像检测位置 · 绿色环 = 星表预测位置 · 仅用于身份核验",
        }
        self.overlay_hint_var.set(hints.get(self.overlay_mode_var.get(), hints["quality"]))

    def _on_viewer_configure(self, event: tk.Event) -> None:
        """让悬浮信息栏随 viewer 换行，不用长文本撑大右侧布局。"""

        self.hover_info_label.config(wraplength=max(240, int(event.width) - 32))

    def _on_detail_configure(self, event: tk.Event) -> None:
        """约束右侧说明文字宽度，避免内容反向改变主栏与右栏的分配。"""

        wraplength = max(180, int(event.width) - 32)
        self.faintest_detail.config(wraplength=wraplength)
        self.faintest_note.config(wraplength=wraplength)
        self.sequence_evidence_label.config(wraplength=wraplength)

    def _render_running_progress(self, value: float, progress_text: str) -> None:
        """把后台进度同步到右侧证据卡，避免进度只藏在底部状态栏。"""

        label = self.__dict__.get("sequence_evidence_label")
        if label is None:
            return
        percent = max(0.0, min(100.0, float(value)))
        if self.active_job_kind == "sequence":
            label.config(
                text=(
                    f"运行中 · {percent:.0f}%\n"
                    f"{progress_text}\n"
                    f"逐帧检测完成后，这里会替换为时间、配准、速度、方向和预测摘要"
                )
            )
        elif self.active_job_kind == "single":
            label.config(
                text=(
                    f"当前帧运行中 · {percent:.0f}%\n"
                    f"{progress_text}\n"
                    f"长线预检和全量星点精测分阶段完成"
                )
            )

    def run_analysis(self) -> None:
        if self.busy or self.selected_frame is None:
            if self.selected_frame is None:
                show_message = "请先选择数据目录中的 FITS 文件"
                messagebox.showinfo("RST19", show_message)
            return
        # 允许用户直接编辑输入框后点击“分析当前帧”，不要求先碰一下滑块。
        self._sync_manual_controls_from_entries()
        try:
            threshold, min_distance, max_sources, zero_point, psf_fwhm, min_flux_snr = self._read_parameters()
        except ValueError as exc:
            messagebox.showerror("参数错误", str(exc))
            return
        proposal_mode = self.proposal_mode_var.get()
        self.manual_tuning_dirty = False
        enable_local_deblend = bool(self.local_deblend_var.get())
        deblend_note = " · 局部去混叠开启" if enable_local_deblend else ""
        self._set_manual_tuning_status(
            f"本次运行参数 · { {'gaussian': 'Gaussian基线', 'hybrid': 'hybrid平衡', 'ensemble': 'Starlet实验'}.get(proposal_mode, proposal_mode) }{deblend_note} · 候选 {threshold:.1f}σ · 可信 SNR {min_flux_snr:.1f}σ",
            color=SKY,
        )
        self.catalog_analysis = None
        self.catalog_match_result = None
        self.catalog_wcs = None
        self.catalog_calibration = None
        self.catalog_frame_path = None
        self.hover_catalog_match = None
        self.busy = True
        self.active_job_kind = "single"
        self.active_job_token = self.frame_token
        self._set_job_controls()
        self._set_progress(2.0, "准备")
        self._set_sequence_progress(2.0, "单图 · 准备")
        self._reset_sequence_ledger(len(self.frames), state="pending")
        self._render_running_progress(2.0, "单图 · 准备")
        self.status_var.set(f"正在分析 {self.selected_frame.name} · 全量检测模式" if max_sources is None else f"正在分析 {self.selected_frame.name} · 返回上限 {max_sources}")
        frame_path = self.selected_frame
        token = self.frame_token
        cache_generation = self.cache_generation

        def worker() -> None:
            try:
                self.result_queue.put(("analysis-progress", token, (4.0, "读取 1/1")))
                frame = read_fits(frame_path)
                self.result_queue.put(("analysis-progress", token, (8.0, "读取完成 · 先找单帧长线")))

                # 长线检测不依赖全量星点孔径测光。先用同一套背景、掩膜和
                # 连通域几何规则做轻量候选筛选，让明显长轨迹先进入预览；
                # 后台随后继续完成数万候选的星点属性和星等精测。
                def trail_preview_progress(value: float, label: str) -> None:
                    self.result_queue.put((
                        "analysis-progress",
                        token,
                        (8.0 + 6.0 * float(value) / 100.0, f"单帧长线 · {label}"),
                    ))

                try:
                    long_trails = detect_single_frame_long_trails(
                        frame,
                        psf_fwhm=psf_fwhm,
                        progress=trail_preview_progress,
                    )
                except Exception as exc:  # noqa: BLE001 - supplementary preview must not block star measurement
                    long_trails = ()
                    self.result_queue.put(
                        ("analysis-progress", token, (14.0, f"单帧长线预检跳过 · {type(exc).__name__}"))
                    )
                # 这是单帧形状候选，不是已经确认的 moving 目标；UI 用橙色
                # 显示，并在最终分析结果到达时用同一结果替换。
                self.result_queue.put(("trail-preview", token, long_trails))

                def progress(value: float, label: str) -> None:
                    self.result_queue.put(("analysis-progress", token, (14.0 + 0.80 * float(value), label)))

                self.result_queue.put(("analysis-progress", token, (14.5, "读取单图缓存")))
                key = cache_key(
                    frame,
                    threshold_sigma=threshold,
                    min_distance=min_distance,
                    aperture_radius=4,
                    max_sources=max_sources,
                    zero_point=zero_point,
                    psf_fwhm=psf_fwhm,
                    min_flux_snr=min_flux_snr,
                    proposal_mode=proposal_mode,
                    enable_local_deblend=enable_local_deblend,
                    allow_partial_zero_mask=None,
                )
                analysis = load_analysis(self.cache_dir, key, frame)
                cache_hit = analysis is not None
                if analysis is None:
                    analysis = analyze_frame(
                        frame,
                        threshold_sigma=threshold,
                        min_distance=min_distance,
                        max_sources=max_sources,
                    zero_point=zero_point,
                    psf_fwhm=psf_fwhm,
                    min_flux_snr=min_flux_snr,
                    proposal_mode=proposal_mode,
                        enable_local_deblend=enable_local_deblend,
                        allow_partial_zero_mask=None,
                        reject_linear_artifacts=True,
                    progress=progress,
                )
                    try:
                        with self.cache_lock:
                            if token != self.frame_token or cache_generation != self.cache_generation:
                                cache_state = "已跳过过期缓存写入"
                            else:
                                save_analysis(self.cache_dir, key, analysis)
                                cache_state = "已写入缓存"
                    except OSError as exc:
                        cache_state = f"缓存写入失败：{exc}"
                else:
                    cache_state = "缓存命中"
                    self.result_queue.put(("analysis-progress", token, (94.0, "缓存命中")))
                if not cache_hit:
                    self.result_queue.put(("analysis-progress", token, (94.0, "完成星点检测")))
                self.result_queue.put(("analysis-progress", token, (94.0, "生成预览")))
                previews = _make_preview_variants(frame.data)
                self.result_queue.put(("analysis-progress", token, (100.0, "完成")))
                self.result_queue.put(("analysis", token, (analysis, frame.data.shape, previews, cache_state, cache_hit, long_trails)))
            except Exception as exc:  # noqa: BLE001 - worker must return a user-facing error
                self.result_queue.put(("analysis-error", token, exc))

        threading.Thread(target=worker, daemon=True).start()

    def export_source_study(self) -> None:
        """导出当前已完成检测的全量源表和 SNR 研究图，不重新运行检测。"""

        if self.analysis is None or self.selected_frame is None or self.busy:
            return
        analysis = self.analysis
        frame_path = self.selected_frame
        token = self.frame_token
        self.source_export_in_progress = True
        self.source_export_button.config(state="disabled")
        self._update_research_menu_state()
        self.status_var.set(f"正在导出 {frame_path.name} 的全量星点研究表…")
        output_dir = PROJECT_ROOT / "tmp" / "gui-source-audit"

        def worker() -> None:
            try:
                output = write_detection_source_artifacts(analysis.detection, output_dir, frame=analysis.frame)
            except Exception as exc:  # noqa: BLE001 - worker returns a visible export error
                self.result_queue.put(("source-artifacts-error", token, (self.source_export_button, frame_path, str(exc))))
                return
            self.result_queue.put(("source-artifacts", token, (self.source_export_button, frame_path, output)))

        threading.Thread(target=worker, daemon=True).start()

    def _show_sequence_source_audit(self, output: Path) -> None:
        """展示 15 帧点源时序审计图，不把抽样图误标为真值。"""

        if self.sequence_source_audit_window is not None:
            try:
                self.sequence_source_audit_window.destroy()
            except tk.TclError:
                pass
        window = tk.Toplevel(self)
        self.sequence_source_audit_window = window
        window.title("RST19 · 星点时序审计")
        window.geometry("1180x820")
        window.minsize(980, 620)
        window.configure(bg=PAPER)
        window.transient(self)

        summary: dict[str, Any] = {}
        summary_path = output / "source_track_audit_summary.json"
        if summary_path.is_file():
            try:
                loaded = json.loads(summary_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    summary = loaded
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                summary = {}
        frame_count = summary.get("frame_count", "—")
        strict_count = summary.get("strict_static_count", "—")
        persistent_count = summary.get("persistent_count", "—")
        transient_count = summary.get("transient_count", "—")
        sampled_count = summary.get("sampled_track_count", "—")
        threshold = summary.get("persistent_min_presence", "—")

        header = tk.Frame(window, bg=PAPER)
        header.pack(fill="x", padx=24, pady=(18, 12))
        self._mono_label(header, "SOURCE TRACK AUDIT / RAW FITS CUTOUTS", color=INK_SOFT, size=8, bg=PAPER).pack(anchor="w")
        self._label(header, "星点时序审计", color=NAVY_DARK, size=20, bold=True, bg=PAPER).pack(anchor="w", pady=(5, 0))
        self._label(
            header,
            (
                f"{frame_count} 帧 · 严格静态 {strict_count:,} · 持续候选 {persistent_count:,} · "
                f"瞬态轨迹 {transient_count:,} · 持续门槛 {threshold}/{frame_count} · 抽样 {sampled_count} 条"
                if all(isinstance(value, int) for value in (frame_count, strict_count, persistent_count, transient_count, sampled_count, threshold))
                else f"{frame_count} 帧 · 严格静态 {strict_count} · 持续候选 {persistent_count} · 瞬态轨迹 {transient_count} · 抽样 {sampled_count} 条"
            ),
            color=INK,
            size=9,
            bg=PAPER,
        ).pack(anchor="w", pady=(4, 0))
        self._mono_label(
            header,
            "青绿色 = 严格静态，蓝色 = 低置信持续候选，琥珀色 = 瞬态；灰框代表该帧没有检测到对应点。抽样图用于人工复核，不是逐星真值。",
            color=AMBER,
            size=8,
            bg=PAPER,
        ).pack(anchor="w", pady=(7, 0))

        body = tk.Frame(window, bg=NAVY_DARK)
        body.pack(fill="both", expand=True, padx=24, pady=(0, 24))
        canvas = tk.Canvas(body, bg=NAVY_DARK, highlightthickness=0)
        vertical = ttk.Scrollbar(body, orient="vertical", command=canvas.yview)
        horizontal = ttk.Scrollbar(body, orient="horizontal", command=canvas.xview)
        canvas.configure(yscrollcommand=vertical.set, xscrollcommand=horizontal.set)
        canvas.grid(row=0, column=0, sticky="nsew")
        vertical.grid(row=0, column=1, sticky="ns")
        horizontal.grid(row=1, column=0, sticky="ew")
        body.grid_rowconfigure(0, weight=1)
        body.grid_columnconfigure(0, weight=1)
        image_path = output / "source_track_contact_sheet.png"
        if image_path.is_file():
            image = Image.open(image_path).convert("RGB")
            image.thumbnail((1080, 1800), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(image)
            canvas.create_image(12, 12, image=photo, anchor="nw")
            canvas.image = photo
            canvas.configure(scrollregion=(0, 0, image.width + 24, image.height + 24))
        else:
            canvas.create_text(24, 24, text="未找到星点时序审计图，请重新生成", fill=SKY_LIGHT, anchor="nw", font=(SANS, 11))
        window.protocol("WM_DELETE_WINDOW", lambda: self._close_sequence_source_audit(window))

    def _close_sequence_source_audit(self, window: tk.Toplevel) -> None:
        try:
            window.destroy()
        except tk.TclError:
            pass
        if self.sequence_source_audit_window is window:
            self.sequence_source_audit_window = None

    def _show_source_study_artifacts(self, output: Path, frame_path: Path) -> None:
        """在桌面窗口中展示已导出的论文式星点表图，不重新运行检测。"""

        if self.source_study_window is not None:
            try:
                self.source_study_window.destroy()
            except tk.TclError:
                pass
        window = tk.Toplevel(self)
        self.source_study_window = window
        window.title(f"RST19 · 星点研究表图 · {frame_path.stem}")
        window.geometry("1180x820")
        window.minsize(1060, 700)
        window.configure(bg=PAPER)
        window.transient(self)

        header = tk.Frame(window, bg=PAPER)
        header.pack(fill="x", padx=24, pady=(18, 12))
        self._label(header, "星点检测研究表图", color=NAVY_DARK, size=20, bold=True, bg=PAPER).pack(anchor="w")
        self._label(
            header,
            f"{frame_path.name} · 结果来自当前已完成检测，不重新计算",
            color=INK_SOFT,
            size=9,
            bg=PAPER,
        ).pack(anchor="w", pady=(4, 0))
        self._mono_label(
            header,
            "候选峰、返回源、质量源、拒绝标志和空间分布分别呈现；图表不是恒星真值。",
            color=AMBER,
            size=8,
            bg=PAPER,
        ).pack(anchor="w", pady=(7, 0))

        notebook = ttk.Notebook(window)
        notebook.pack(fill="both", expand=True, padx=24, pady=(0, 12))
        charts_tab = tk.Frame(notebook, bg=PAPER_LIGHT)
        metrics_tab = tk.Frame(notebook, bg=PAPER_LIGHT)
        notebook.add(charts_tab, text="论文表图")
        notebook.add(metrics_tab, text="指标与口径")

        chart_specs = (
            ("SNR 排名", "source_snr_rank.png"),
            ("通量 SNR 分布", "source_flux_snr_distribution.png"),
            ("质量标志计数", "source_quality_flags.png"),
            ("空间密度与通过率", "source_spatial_density.png"),
        )
        chart_images: list[tuple[Image.Image, tk.Label]] = []
        chart_photos: list[ImageTk.PhotoImage] = []
        for column in range(2):
            charts_tab.grid_columnconfigure(column, weight=1)
        for row in range(2):
            charts_tab.grid_rowconfigure(row, weight=1)
        for index, (title, filename) in enumerate(chart_specs):
            panel = tk.Frame(charts_tab, bg=PAPER_LIGHT)
            panel.grid(row=index // 2, column=index % 2, sticky="nsew", padx=(12 if index % 2 == 0 else 6, 12 if index % 2 else 6), pady=(12 if index < 2 else 6, 12))
            self._label(panel, title, color=NAVY_DARK, size=10, bold=True, bg=PAPER_LIGHT).pack(anchor="w", pady=(0, 6))
            chart_path = output / filename
            label = tk.Label(panel, text=f"未找到 {filename}", bg=NAVY_DARK, fg=SKY_LIGHT, font=(SANS, 9))
            label.pack(fill="both", expand=True)
            if chart_path.is_file():
                original = Image.open(chart_path).convert("RGB")
                preview = original.copy()
                # 2×2 图证在 1180×820 窗口内完整可见；原图仍保存在输出目录。
                preview.thumbnail((500, 220), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(preview)
                label.config(image=photo, text="")
                chart_photos.append(photo)
                chart_images.append((original, label))
        charts_tab.chart_photos = chart_photos  # type: ignore[attr-defined]
        charts_tab.chart_images = chart_images  # type: ignore[attr-defined]

        columns = ("metric", "value", "definition")
        tree = ttk.Treeview(metrics_tab, columns=columns, show="headings", height=18)
        for column, title, width in (("metric", "指标", 220), ("value", "数值", 150), ("definition", "定义与边界", 650)):
            tree.heading(column, text=title)
            tree.column(column, width=width, minwidth=90, stretch=column == "definition", anchor="w")
        summary_path = output / "source_quality_summary.csv"
        if summary_path.is_file():
            with summary_path.open("r", encoding="utf-8-sig", newline="") as stream:
                for row in csv.DictReader(stream):
                    tree.insert("", tk.END, values=(row.get("metric", ""), row.get("value", ""), row.get("definition", "")))
        y_scroll = ttk.Scrollbar(metrics_tab, orient="vertical", command=tree.yview)
        x_scroll = ttk.Scrollbar(metrics_tab, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)
        tree.grid(row=0, column=0, sticky="nsew", padx=(12, 0), pady=(12, 0))
        y_scroll.grid(row=0, column=1, sticky="ns", pady=(12, 0))
        x_scroll.grid(row=1, column=0, sticky="ew", padx=(12, 0), pady=(0, 12))
        metrics_tab.grid_rowconfigure(0, weight=1)
        metrics_tab.grid_columnconfigure(0, weight=1)

        self._mono_label(window, f"输出目录 · {output}", color=INK_SOFT, size=8, bg=PAPER).pack(anchor="w", padx=24, pady=(0, 14))

    def run_sequence_analysis(self) -> None:
        """对当前目录的全部 FITS 做跨帧关联，单独显示点轨迹和线状目标。"""

        if self.busy or not self.frames:
            if not self.frames:
                messagebox.showinfo("RST19", "当前数据目录没有 FITS 文件")
            return
        if len(self.frames) < 2:
            messagebox.showinfo("动目标分析", "至少需要 2 帧 FITS 才能进行跨帧运动判定")
            return
        try:
            threshold, min_distance, max_sources, zero_point, psf_fwhm, min_flux_snr = self._read_parameters()
        except ValueError as exc:
            messagebox.showerror("参数错误", str(exc))
            return
        proposal_mode = self.proposal_mode_var.get()
        temporal_proposal_mode = self.temporal_proposal_mode_var.get()
        try:
            temporal_reference_min_snr = float(self.temporal_reference_min_snr_var.get())
        except (TypeError, ValueError) as exc:
            messagebox.showerror("参数错误", "15帧参考 SNR 必须是有效数字")
            return
        if not 0 < temporal_reference_min_snr <= 15.0:
            messagebox.showerror("参数错误", "15帧参考 SNR 应在 0 至 15σ 之间")
            return
        sequence_full = bool(self.sequence_full_var.get())
        stack_faint_recovery = bool(self.stack_faint_recovery_var.get())
        stack_reference_mode = temporal_proposal_mode if temporal_proposal_mode in {"median", "coadd"} else "median"
        self.catalog_analysis = None
        self.catalog_match_result = None
        self.catalog_wcs = None
        self.catalog_calibration = None
        self.catalog_frame_path = None
        self.hover_catalog_match = None
        # 旧序列结果在新一轮计算期间不能继续伪装成当前结果；完成后再写回。
        self.sequence_result = None
        self.long_trails = ()
        self.sequence_evidence_label.config(text=f"正在分析 {len(self.frames)} 帧 · 当前阶段和帧号会显示在此处")
        token = self.frame_token
        frame_paths = tuple(self.frames)
        cache_generation = self.cache_generation
        sequence_parameters = {
            "threshold_sigma": threshold,
            "min_distance": min_distance,
            "aperture_radius": 4,
            "max_sources": None if sequence_full else max_sources,
            "sequence_max_sources": None if sequence_full else (DEFAULT_SEQUENCE_SOURCE_WORKING_LIMIT if max_sources is None else max_sources),
            "candidate_consensus_min_snr": GUI_DEFAULT_CANDIDATE_CONSENSUS_MIN_SNR,
            "temporal_candidate_min_snr": GUI_DEFAULT_TEMPORAL_CANDIDATE_MIN_SNR,
            "temporal_reference_min_snr": temporal_reference_min_snr,
            "temporal_multiscale": GUI_DEFAULT_TEMPORAL_MULTISCALE,
            "temporal_min_psf_correlation": GUI_DEFAULT_TEMPORAL_MIN_PSF_CORRELATION,
            "temporal_proposal_mode": temporal_proposal_mode,
            "stack_faint_recovery": stack_faint_recovery,
            "stack_reference_mode": stack_reference_mode,
            "stack_threshold_sigma": GUI_DEFAULT_THRESHOLD_SIGMA,
            "stack_min_flux_snr": GUI_DEFAULT_MIN_FLUX_SNR,
            "stack_frame_min_flux_snr": 3.0,
            "zero_point": zero_point,
            "psf_fwhm": psf_fwhm,
            "background_box_size": 256,
            "background_sample_limit": DEFAULT_SEQUENCE_BACKGROUND_SAMPLE_LIMIT,
            "min_flux_snr": min_flux_snr,
            "min_fwhm": 0.8,
            "max_fwhm": 12.0,
            "max_ellipticity": 0.65,
            "min_sharpness": 0.005,
            "max_sharpness": 0.85,
            "min_footprint_pixels": 2,
            "min_psf_support_pixels": 3,
            "gain_e_per_adu": None,
            "read_noise_adu": 0.0,
            "mask_zero_pixels": None,
            "allow_partial_zero_mask": None,
            "reject_linear_artifacts": True,
            "proposal_mode": proposal_mode,
            "refine_local_background": False,
            "use_float32": True,
            "fast_sequence": True,
            "link_radius_px": 4.0,
            "min_presence": None,
            "persistent_min_presence": None,
            "motion_min_displacement_px": 2.0,
            "registration_radius_px": 8.0,
            "max_motion_fit_rms_px": 0.75,
            "fast_point_motion": True,
            "fast_point_min_snr": DEFAULT_FAST_POINT_MIN_SNR,
            "fast_point_max_step_px": DEFAULT_FAST_POINT_MAX_STEP_PX,
            "fast_point_gate_px": DEFAULT_FAST_POINT_GATE_PX,
            "fast_point_max_fit_rms_px": DEFAULT_FAST_POINT_FIT_RMS_PX,
        }
        self.busy = True
        self.active_job_kind = "sequence"
        self.active_job_token = token
        self._set_job_controls()
        self._set_progress(2.0, f"0/{len(frame_paths)}")
        self._set_sequence_progress(2.0, f"准备 · 0/{len(frame_paths)}")
        self._reset_sequence_ledger(len(frame_paths), state="pending")
        self._render_running_progress(2.0, f"准备 · 0/{len(frame_paths)}")
        working_limit = None if sequence_full else (DEFAULT_SEQUENCE_SOURCE_WORKING_LIMIT if max_sources is None else max_sources)
        working_text = "全量源" if working_limit is None else f"≤{working_limit:,} 源"
        self.status_var.set(
            f"正在分析 {len(frame_paths)} 帧 · {DEFAULT_SEQUENCE_WORKERS} 个并行 worker · "
            f"每帧配准工作集 {working_text} · 进度将按帧更新…"
        )

        def worker() -> None:
            try:
                self.result_queue.put(("analysis-progress", token, (2.5, f"准备序列缓存 · 0/{len(frame_paths)}")))
                key = sequence_cache_key(frame_paths, parameters=sequence_parameters)
                with self.cache_lock:
                    result = load_sequence_result(self.cache_dir, key)
                cache_hit = result is not None
                if result is None:
                    frame_progress_values = [0.0] * len(frame_paths)
                    frame_progress_lock = threading.Lock()

                    def progress(stage: str, index: int, total: int) -> None:
                        if stage == "prepare":
                            value, label = 4.0, f"准备 0/{total}"
                        elif stage == "background":
                            value, label = 6.0, "建立共享背景模型"
                        elif stage == "frame":
                            with frame_progress_lock:
                                measured = sum(frame_progress_values) / (100.0 * max(1, total))
                            value = max(8.0 + 72.0 * index / max(1, total), 8.0 + 72.0 * measured)
                            label = f"已完成 {index}/{total} 帧"
                        elif stage == "registration":
                            value, label = 84.0, "配准关联"
                        elif stage == "fast-point-motion":
                            value, label = 86.0, "高速点源关联"
                        elif stage == "sentinel-audit":
                            value, label = 83.5, "固定值异常码审计"
                        elif stage == "consensus":
                            value, label = 84.0, "全量候选跨帧共识"
                        elif stage == "temporal-coadd":
                            value, label = 84.0, "构建多帧稳健叠加提案"
                        elif stage == "stack-faint":
                            value, label = 85.0, "叠加参考图暗星恢复"
                        elif stage == "motion-detail":
                            value = 84.0 + 10.0 * max(0, min(100, index)) / max(1, total)
                            label = f"线状筛选 · {index}%"
                        elif stage == "motion":
                            value, label = 94.0, "线状筛选"
                        else:
                            value, label = 100.0, "完成"
                        self.result_queue.put(("analysis-progress", token, (value, label)))

                    def detail_progress(frame_index: int, total: int, value: float, label: str) -> None:
                        bounded = max(0.0, min(100.0, float(value)))
                        with frame_progress_lock:
                            position = max(0, min(total - 1, frame_index - 1))
                            frame_progress_values[position] = max(frame_progress_values[position], bounded)
                            completed_equivalent = sum(frame_progress_values) / 100.0
                            finished = sum(item >= 100.0 for item in frame_progress_values)
                        overall = 8.0 + 72.0 * completed_equivalent / max(1, total)
                        progress_label = f"完成 {finished}/{total} · F{frame_index:02d} {label}"
                        self.result_queue.put(("analysis-progress", token, (overall, progress_label)))

                    result = analyze_sequence(
                        frame_paths,
                        threshold_sigma=threshold,
                        min_distance=min_distance,
                        max_sources=None if sequence_full else max_sources,
                        background_box_size=256,
                        sequence_max_sources=working_limit,
                        sequence_workers=DEFAULT_SEQUENCE_WORKERS,
                        candidate_consensus_min_snr=GUI_DEFAULT_CANDIDATE_CONSENSUS_MIN_SNR,
                        temporal_candidate_min_snr=GUI_DEFAULT_TEMPORAL_CANDIDATE_MIN_SNR,
                        temporal_reference_min_snr=temporal_reference_min_snr,
                        temporal_multiscale=GUI_DEFAULT_TEMPORAL_MULTISCALE,
                        temporal_min_psf_correlation=GUI_DEFAULT_TEMPORAL_MIN_PSF_CORRELATION,
                        temporal_proposal_mode=temporal_proposal_mode,
                        stack_faint_recovery=stack_faint_recovery,
                        stack_reference_mode=stack_reference_mode,
                        stack_threshold_sigma=GUI_DEFAULT_THRESHOLD_SIGMA,
                        stack_min_flux_snr=GUI_DEFAULT_MIN_FLUX_SNR,
                        stack_frame_min_flux_snr=3.0,
                        zero_point=zero_point,
                        psf_fwhm=psf_fwhm,
                        min_flux_snr=min_flux_snr,
                        proposal_mode=proposal_mode,
                        reject_linear_artifacts=True,
                        refine_local_background=False,
                        fast_point_motion=True,
                        fast_point_min_snr=DEFAULT_FAST_POINT_MIN_SNR,
                        fast_point_max_step_px=DEFAULT_FAST_POINT_MAX_STEP_PX,
                        fast_point_gate_px=DEFAULT_FAST_POINT_GATE_PX,
                        fast_point_max_fit_rms_px=DEFAULT_FAST_POINT_FIT_RMS_PX,
                        progress=progress,
                        detail_progress=detail_progress,
                    )
                    try:
                        with self.cache_lock:
                            if token != self.frame_token or cache_generation != self.cache_generation:
                                cache_state = "已跳过过期序列缓存写入"
                            else:
                                save_sequence_result(self.cache_dir, key, result)
                                cache_state = "已写入序列缓存"
                    except OSError as exc:
                        cache_state = f"序列缓存写入失败：{exc}"
                else:
                    cache_state = "序列缓存命中"
                    self.result_queue.put(("analysis-progress", token, (100.0, "缓存命中")))
                self.result_queue.put(("analysis-progress", token, (100.0, "完成")))
                self.result_queue.put(("sequence", token, (result, cache_state, cache_hit)))
            except Exception as exc:  # noqa: BLE001 - worker must return a user-facing error
                self.result_queue.put(("sequence-error", token, exc))

        threading.Thread(target=worker, daemon=True).start()

    def _poll_result(self) -> None:
        if self.__dict__.get("_closing", False):
            return
        try:
            kind, token, payload = self.result_queue.get_nowait()
        except queue.Empty:
            self.after(100, self._poll_result)
            return
        if kind == "analysis-progress":
            # 检测器会在每帧源级测量时产生很多细粒度事件。逐条以 100 ms
            # 消费会让 UI 落后几十秒，甚至在后台已完成后还看不到完成状态；
            # 这里只合并连续的同一任务进度，保留队列中的最终结果事件。
            latest_progress = (kind, token, payload)
            while True:
                try:
                    queued = self.result_queue.get_nowait()
                except queue.Empty:
                    break
                if queued[0] == "analysis-progress" and queued[1] == token:
                    latest_progress = queued
                    continue
                self.result_queue.put(queued)
                break
            _, token, payload = latest_progress
            if token == self.frame_token:
                value, progress_text = payload
                self._set_progress(float(value), str(progress_text))
                self.status_var.set(f"{progress_text} · 当前后台分析")
                if self.active_job_kind == "sequence":
                    self._set_sequence_progress(float(value), str(progress_text))
                    self._update_sequence_ledger(float(value), str(progress_text))
                else:
                    self._set_sequence_progress(float(value), f"单图 · {progress_text}")
                self._render_running_progress(float(value), str(progress_text))
            self.after(60, self._poll_result)
            return
        if kind == "trail-preview":
            # 单帧长线与全量星点检测并行推进。只有当前帧、当前单图任务
            # 的候选才能写入预览，切帧或启动 15 帧任务后丢弃旧事件。
            if token == self.frame_token and self.active_job_kind == "single":
                self.long_trails = tuple(payload)
                if self.long_trails:
                    # 长线只是叠加证据，不能把可信源主层替换掉。
                    self.overlay_mode_var.set("quality")
                    self._update_overlay_hint()
                    self.status_var.set(
                        f"已先发现 {len(self.long_trails):,} 条单帧长线候选 · 已叠加在可信源上 · 星点精测继续进行"
                    )
                    self._draw_preview()
            self.after(60, self._poll_result)
            return
        if kind == "injection-progress":
            window, status, progress_text = payload
            try:
                exists = bool(window.winfo_exists())
            except tk.TclError:
                exists = False
            if exists:
                status.config(text=progress_text)
            self.after(100, self._poll_result)
            return
        if kind == "injection":
            window, render, buttons, status, rows, output, mode = payload
            try:
                exists = bool(window.winfo_exists())
            except tk.TclError:
                exists = False
            if exists:
                render(rows, output, mode)
                for button in buttons:
                    button.config(state="normal")
                status.config(text="完成")
            self.after(100, self._poll_result)
            return
        if kind == "injection-error":
            window, buttons, status, error_text = payload
            try:
                exists = bool(window.winfo_exists())
            except tk.TclError:
                exists = False
            if exists:
                for button in buttons:
                    button.config(state="normal")
                status.config(text=f"失败：{error_text}")
            self.after(100, self._poll_result)
            return
        if kind == "sweep":
            window, render, buttons, status, rows, output = payload
            try:
                exists = bool(window.winfo_exists())
            except tk.TclError:
                exists = False
            if exists:
                render(rows, output)
                for button in buttons:
                    button.config(state="normal")
                status.config(text="完成")
            self.after(100, self._poll_result)
            return
        if kind == "sweep-error":
            window, buttons, status, error_text = payload
            try:
                exists = bool(window.winfo_exists())
            except tk.TclError:
                exists = False
            if exists:
                for button in buttons:
                    button.config(state="normal")
                status.config(text=f"失败：{error_text}")
            self.after(100, self._poll_result)
            return
        if kind == "psf-sweep":
            window, render, run_button, status, rows, output = payload
            try:
                exists = bool(window.winfo_exists())
            except tk.TclError:
                exists = False
            if exists:
                render(rows, output)
                run_button.config(state="normal", text="重新扫描")
                status.config(text="完成")
            self.after(100, self._poll_result)
            return
        if kind == "psf-sweep-error":
            window, buttons, status, error_text = payload
            try:
                exists = bool(window.winfo_exists())
            except tk.TclError:
                exists = False
            if exists:
                for button in buttons:
                    button.config(state="normal")
                status.config(text=f"失败：{error_text}")
            self.after(100, self._poll_result)
            return
        if kind == "innovation-artifacts":
            window, render, run_button, status, output = payload
            try:
                exists = bool(window.winfo_exists())
            except tk.TclError:
                exists = False
            if exists:
                render(output)
                run_button.config(state="normal", text="重新生成")
                status.config(text="已生成")
            self.after(100, self._poll_result)
            return
        if kind == "innovation-error":
            window, run_button, status, error_text = payload
            try:
                exists = bool(window.winfo_exists())
            except tk.TclError:
                exists = False
            if exists:
                run_button.config(state="normal", text="重试")
                status.config(text=f"失败：{error_text}")
            self.after(100, self._poll_result)
            return
        if kind == "sequence-source-audit":
            window, run_button, status, output = payload
            try:
                exists = bool(window.winfo_exists())
            except tk.TclError:
                exists = False
            if exists:
                run_button.config(state="normal", text="重新审计星点")
                status.config(text="星点审计完成")
                self._show_sequence_source_audit(Path(output))
            self.after(100, self._poll_result)
            return
        if kind == "sequence-source-audit-error":
            window, run_button, status, error_text = payload
            try:
                exists = bool(window.winfo_exists())
            except tk.TclError:
                exists = False
            if exists:
                run_button.config(state="normal", text="重试星点审计")
                status.config(text=f"星点审计失败：{error_text}")
            self.after(100, self._poll_result)
            return
        if kind == "trail-audit-progress":
            window, status, progress_text = payload
            try:
                exists = bool(window.winfo_exists())
            except tk.TclError:
                exists = False
            if exists:
                status.config(text=progress_text)
            self.after(100, self._poll_result)
            return
        if kind == "trail-audit":
            window, render, run_button, status, rows, output = payload
            try:
                exists = bool(window.winfo_exists())
            except tk.TclError:
                exists = False
            if exists:
                render(rows, output)
                run_button.config(state="normal", text="重新审计")
                status.config(text="已完成")
            self.after(100, self._poll_result)
            return
        if kind == "trail-audit-error":
            window, run_button, status, error_text = payload
            try:
                exists = bool(window.winfo_exists())
            except tk.TclError:
                exists = False
            if exists:
                run_button.config(state="normal", text="重试审计")
                status.config(text=f"失败：{error_text}")
            self.after(100, self._poll_result)
            return
        if kind == "source-artifacts":
            button, frame_path, output = payload
            if token == self.frame_token and self.selected_frame == frame_path:
                self.source_export_in_progress = False
                button.config(state="normal")
                self._update_research_menu_state()
                self.status_var.set(f"星点研究表已导出 · {output}")
                self._show_source_study_artifacts(Path(output), frame_path)
            self.after(100, self._poll_result)
            return
        if kind == "source-artifacts-error":
            button, frame_path, error_text = payload
            if token == self.frame_token and self.selected_frame == frame_path:
                self.source_export_in_progress = False
                button.config(state="normal")
                self._update_research_menu_state()
                self.status_var.set(f"星点研究表导出失败 · {error_text}")
            self.after(100, self._poll_result)
            return
        if kind == "catalog":
            window, render, run_button, status, analysis, wcs, catalog_path, frame_path, catalog = payload
            try:
                exists = bool(window.winfo_exists())
            except tk.TclError:
                exists = False
            if exists:
                if token != self.frame_token or self.selected_frame != frame_path:
                    run_button.config(state="normal", text="重试")
                    status.config(text="当前帧已切换 · 结果未写入")
                else:
                    render(analysis, wcs, catalog_path, frame_path, catalog)
                    run_button.config(state="normal", text="重新核验")
                    status.config(text="已完成")
            self.after(100, self._poll_result)
            return
        if kind == "catalog-calibration":
            window, render, button, status, calibration = payload
            try:
                exists = bool(window.winfo_exists())
            except tk.TclError:
                exists = False
            if exists and token == self.frame_token:
                render(calibration)
                button.config(state="normal", text="根据匹配拟合 WCS")
            self.after(100, self._poll_result)
            return
        if kind == "catalog-validation-progress":
            window, status, index, total = payload
            try:
                exists = bool(window.winfo_exists())
            except tk.TclError:
                exists = False
            if exists and token == self.frame_token:
                status.config(text=f"正在验证 15 帧 WCS · {index}/{total}")
            self.after(100, self._poll_result)
            return
        if kind == "catalog-validation":
            window, button, status, report, output, render = payload
            try:
                exists = bool(window.winfo_exists())
            except tk.TclError:
                exists = False
            if exists and token == self.frame_token:
                render(report, output)
                button.config(state="normal", text="重新验证 15 帧 WCS")
                status.config(text=f"15 帧 WCS 验证完成 · {report.validated_count}/{report.frame_count} 帧有效")
            self.after(100, self._poll_result)
            return
        if kind == "catalog-validation-error":
            window, button, status, error_text = payload
            try:
                exists = bool(window.winfo_exists())
            except tk.TclError:
                exists = False
            if exists and token == self.frame_token:
                button.config(state="normal", text="重试 15 帧 WCS")
                status.config(text=f"15 帧 WCS 验证失败：{error_text}")
            self.after(100, self._poll_result)
            return
        if kind == "catalog-calibration-error":
            window, button, status, error_text = payload
            try:
                exists = bool(window.winfo_exists())
            except tk.TclError:
                exists = False
            if exists and token == self.frame_token:
                button.config(state="normal", text="重试 WCS 拟合")
                status.config(text=f"WCS 拟合失败：{error_text}")
            self.after(100, self._poll_result)
            return
        if kind == "catalog-error":
            window, run_button, status, error_text = payload
            try:
                exists = bool(window.winfo_exists())
            except tk.TclError:
                exists = False
            if exists:
                run_button.config(state="normal", text="重试")
                status.config(text=f"失败：{error_text}")
            self.after(100, self._poll_result)
            return
        if token != self.frame_token:
            if self.active_job_token == token and kind in {"analysis", "analysis-error", "sequence", "sequence-error"}:
                self.active_job_token = None
                self.busy = False
                self.active_job_kind = None
                self._set_job_controls()
                self._set_sequence_progress(0.0, "任务已切帧取消 · 可重试")
                self._reset_sequence_ledger(len(self.frames), state="pending")
            self.after(100, self._poll_result)
            return
        if kind == "preview":
            if self.analysis is None:
                _frame_path, shape, previews = payload
                self._apply_preview_result(shape, previews)
        elif kind == "preview-error":
            self.status_var.set(f"预览载入失败 · {payload}")
            messagebox.showerror("预览载入失败", str(payload))
        elif kind == "analysis-error":
            self.active_job_token = None
            self.busy = False
            self.active_job_kind = None
            self._set_job_controls()
            self._set_progress(0.0, "失败")
            self._set_sequence_progress(0.0, "单图失败 · 可重试")
            self.sequence_evidence_label.config(text=f"单图分析失败\n{payload}\n请检查参数、FITS 文件和缓存后重试")
            self.status_var.set(f"分析失败 · {payload}")
            messagebox.showerror("分析失败", str(payload))
        elif kind == "analysis":
            self.active_job_token = None
            self.busy = False
            self.active_job_kind = None
            self._set_job_controls()
            self._set_progress(100.0, "完成")
            self.analysis, shape, previews, cache_state, _cache_hit, self.long_trails = payload
            self.preview_variants = dict(previews)
            self.preview = self.preview_variants.get(self.preview_mode_var.get()) or self.preview_variants.get(PREVIEW_MODE_ENHANCED)
            # 单帧分析是新的证据上下文，不能继续显示上一次 15 帧分析的轨迹。
            self.sequence_result = None
            self._set_sequence_progress(100.0, "当前帧完成 · 序列待运行")
            self._set_job_controls()
            self.preview_shape = shape
            self.preview_zoom = 1.0
            self.preview_pan_x = 0.0
            self.preview_pan_y = 0.0
            self._prepare_source_grid()
            detection = self.analysis.detection
            single_frame_lines = [
                "当前为单帧分析",
                "单帧长线只能作为候选；需要 15 帧序列才能确认 moving",
            ]
            for trail in sorted(
                (item for item in self.long_trails if item.points),
                key=lambda item: item.points[0].length_px,
                reverse=True,
            )[:3]:
                point = trail.points[0]
                edge_text = "触边" if point.touches_edge else "未触边"
                single_frame_lines.append(
                    f"TRAIL? {trail.track_id:04d} · 长 {point.length_px:.1f}px · 宽 {point.width_px:.1f}px · "
                    f"方向 {point.angle_deg:.1f}° · residual SNR {point.residual_snr:.1f} · {edge_text}"
                )
            if len(self.long_trails) > 3:
                single_frame_lines.append(f"其余 {len(self.long_trails) - 3} 条候选见运动层悬停信息")
            self.sequence_evidence_label.config(text="\n".join(single_frame_lines))
            trail_text = f" · 单帧长线候选 {len(self.long_trails):,}" if self.long_trails else " · 未发现满足几何门槛的单帧长线"
            self.status_var.set(f"{cache_state} · 候选峰 {detection.candidate_count:,} · 可信星点 {detection.star_count:,} · 审计返回 {detection.returned_count:,}{trail_text}")
            # 单帧分析完成后始终回到可信源主层。长线候选保留为叠加证据，
            # 不再让“发现长线”把整张图切成只显示轨迹的模式。
            self.overlay_mode_var.set("quality")
            self._update_overlay_hint()
            self._render_analysis()
        elif kind == "sequence-error":
            self.active_job_token = None
            self.busy = False
            self.active_job_kind = None
            self._set_job_controls()
            self._set_progress(0.0, "失败")
            self._set_sequence_progress(0.0, f"{len(self.frames)} 帧失败 · 可重试")
            self._mark_sequence_ledger("error")
            self.sequence_evidence_label.config(
                text=f"15 帧分析失败\n{payload}\n请检查帧序列、参数和缓存后重试"
            )
            self.status_var.set(f"动目标分析失败 · {payload}")
            messagebox.showerror("动目标分析失败", str(payload))
        elif kind == "sequence":
            self.active_job_token = None
            self.busy = False
            self.active_job_kind = None
            self._set_job_controls()
            self._set_progress(100.0, "完成")
            result, cache_state, _cache_hit = payload
            self.sequence_result = result
            self._set_sequence_progress(100.0, f"{len(result.frames)} 帧完成 · {len(result.frames)}/{len(result.frames)}")
            self._mark_sequence_ledger("completed")
            self.long_trails = ()
            self._set_job_controls()
            # 序列完成后仍以当前帧可信源作为主层；稳定星场和运动证据
            # 都可以单独切换或通过“运动轨迹”勾选叠加查看。
            self.overlay_mode_var.set("quality")
            self._update_overlay_hint()
            display_layer = "可信源主层 + 运动轨迹叠加" if self._motion_overlay_enabled() else "可信源主层"
            moving_features = sum(track.classification == "moving" for track in result.motion_features)
            feature_candidates = sum(track.classification == "candidate" for track in result.motion_features)
            self.status_var.set(
                f"{cache_state} · {len(result.frames)} 帧分析完成 · 严格静态 {result.stable_source_count:,} · "
                f"持续候选 {result.persistent_source_count:,} · "
                f"叠加暗星 {result.stack_faint_count:,} · "
                f"点轨迹 {result.moving_track_count:,}（高速点 {result.fast_point_motion_count:,}） · "
                f"线状目标 {moving_features:,} · "
                f"待复核线 {feature_candidates:,} · 当前显示{display_layer} · "
                f"配准工作集 {'≤' + format(result.source_working_limit, ',') if result.source_working_limit is not None else '全量'}"
            )
            self._render_sequence_evidence(result)
            self._draw_preview()
        self.after(100, self._poll_result)

    def _reset_result_widgets(self) -> None:
        for value in self.metric_values.values():
            value.config(text="—")
        self.faintest_detail.config(text="尚未运行分析")
        self.faintest_note.config(
            text="按局部通量 SNR、点源形状、边缘、掩膜和饱和状态筛选；m_inst 为仪器星等。"
        )
        if self.sequence_result is None:
            self.sequence_evidence_label.config(text="尚未完成序列分析\n点击“分析 15 帧动目标”后显示时间、配准和轨迹摘要")
        else:
            self._render_sequence_evidence(self.sequence_result)
        self.table_count_label.config(text="0 rows")
        self.hover_info_var.set("将鼠标移到候选点查看坐标、通量、误差、SNR、形状和仪器星等")
        for item in self.source_tree.get_children():
            self.source_tree.delete(item)

    def _prepare_source_grid(self) -> None:
        self.source_grid = {}
        if self.analysis is None:
            return
        for source in self.analysis.detection.sources:
            positions = ((float(source.x), float(source.y)), source_peak_position(source))
            cells = {(int(x // 32), int(y // 32)) for x, y in positions}
            for cell in cells:
                self.source_grid.setdefault(cell, []).append(source)

    def _render_analysis(self) -> None:
        assert self.analysis is not None
        detection = self.analysis.detection
        self.manual_tuning_dirty = False
        applied_threshold = float(detection.parameters.get("threshold_sigma", self.threshold_var.get()))
        applied_min_flux_snr = float(detection.parameters.get("min_flux_snr", self.min_flux_snr_var.get()))
        applied_proposal_mode = str(detection.parameters.get("proposal_mode", "gaussian"))
        self._set_manual_tuning_status(
            f"已应用 · { {'gaussian': 'Gaussian基线', 'hybrid': 'hybrid平衡', 'ensemble': 'Starlet实验'}.get(applied_proposal_mode, applied_proposal_mode) } · 候选 {applied_threshold:.1f}σ · 可信 SNR {applied_min_flux_snr:.1f}σ · 可记录反馈",
            color=MINT,
        )
        exposure_ms = self.analysis.frame.header.get("EXPOSURE")
        self.exposure_s = float(exposure_ms) / 1000.0 if isinstance(exposure_ms, (int, float)) and float(exposure_ms) > 0 else 1.0
        self.metric_values["candidate"].config(text=f"{detection.candidate_count:,}")
        self.metric_values["returned"].config(text=f"{detection.star_count:,}")
        self.metric_values["background"].config(text=f"{detection.background:.2f}")
        self.metric_values["noise"].config(text=f"{detection.noise:.2f}")
        faintest = self.analysis.faintest
        if faintest is None:
            self.metric_values["faintest"].config(text="—")
            self.faintest_detail.config(text="没有满足质量条件的源")
            self.faintest_note.config(text="请降低阈值或检查掩膜/边缘筛选结果；m_inst 只在质量源上定义。")
        else:
            self.metric_values["faintest"].config(text=f"{faintest.instrumental_magnitude:.2f}")
            calibrated = f"\n校准星等：{faintest.calibrated_magnitude:.2f}" if faintest.calibrated_magnitude is not None else ""
            self.faintest_detail.config(text=f"ID {faintest.detection_id:04d}\nm_inst = {faintest.instrumental_magnitude:.3f}{calibrated}\nX {faintest.x:.1f}  /  Y {faintest.y:.1f}")
            signal_snr = faintest.flux_snr if faintest.flux_snr is not None else faintest.snr
            rate_text = f"{faintest.flux_rate:.1f}" if faintest.flux_rate is not None else "—"
            self.faintest_note.config(
                text=(
                    f"通量 {faintest.flux:.1f} ADU ({rate_text} ADU/s) · flux SNR {signal_snr:.1f}\n"
                    "m_inst 为仪器星等；无波段转换时不称 Mv。已在左侧用绿色环标出"
                )
            )
        table_count = min(40, detection.star_count)
        self.table_count_label.config(text=f"{table_count:,} / {detection.star_count:,}")
        for item in self.source_tree.get_children():
            self.source_tree.delete(item)
        quality_sources = detection.quality_sources
        for source in sorted(quality_sources, key=lambda item: (item.flux_snr if item.flux_snr is not None else item.snr), reverse=True)[:40]:
            magnitude = instrumental_magnitude(source.flux, exposure_s=self.exposure_s)
            magnitude_text = f"{magnitude:.2f}" if magnitude is not None else "—"
            signal_snr = source.flux_snr if source.flux_snr is not None else source.snr
            self.source_tree.insert("", tk.END, values=(f"{source.detection_id:04d}", f"{source.x:.1f} / {source.y:.1f}", f"{source.peak:.0f}", f"{signal_snr:.1f}", magnitude_text))
        self._draw_preview()

    def _render_sequence_evidence(self, result: SequenceResult) -> None:
        """在右侧显示短摘要，完整逐帧数据由 innovation 导出器保存。"""

        frames = result.frames
        if not frames:
            self.sequence_evidence_label.config(text="序列结果为空")
            return
        timestamps: list[datetime] = []
        for frame in frames:
            if not frame.timestamp:
                continue
            try:
                parsed = datetime.fromisoformat(frame.timestamp.replace("Z", "+00:00"))
            except ValueError:
                continue
            timestamps.append(parsed.replace(tzinfo=None))
        duration = (timestamps[-1] - timestamps[0]).total_seconds() if len(timestamps) >= 2 else None
        intervals = [
            (right - left).total_seconds()
            for left, right in zip(timestamps, timestamps[1:], strict=False)
            if (right - left).total_seconds() > 0
        ]
        interval_text = f"{float(np.median(intervals)):.3f}±{float(np.std(intervals)):.3f}s" if intervals else "—"
        duration_text = f"{duration:.3f}s" if duration is not None else "—"
        final_shift = result.cumulative_shifts[-1] if result.cumulative_shifts else (0.0, 0.0)
        shift_norm = float(np.hypot(final_shift[0], final_shift[1]))
        moving_features = [track for track in result.motion_features if track.classification == "moving"]
        moving_point_tracks = [track for track in result.tracks if track.classification == "moving"]
        first_frame = frames[0]
        size_text = f"{first_frame.width_px}×{first_frame.height_px}" if first_frame.width_px and first_frame.height_px else "同尺寸"
        exposure_text = f"{first_frame.exposure_ms:g} ms" if first_frame.exposure_ms is not None else "曝光未知"
        candidate_values = [int(frame.candidate_count) for frame in frames]
        quality_values = [int(frame.quality_count) for frame in frames]
        candidate_range = f"{min(candidate_values):,}–{max(candidate_values):,}" if candidate_values else "—"
        quality_range = f"{min(quality_values):,}–{max(quality_values):,}" if quality_values else "—"
        feature_groups: list[str] = []
        for track in result.motion_features:
            if not track.points:
                continue
            first_index = track.points[0].frame_index + 1
            last_index = track.points[-1].frame_index + 1
            frame_range = f"F{first_index:02d}" if first_index == last_index else f"F{first_index:02d}–F{last_index:02d}"
            state = "moving" if track.classification == "moving" else "candidate"
            feature_groups.append(f"{frame_range} {state}")
        group_text = "；".join(feature_groups[:3]) if feature_groups else "无"
        if len(feature_groups) > 3:
            group_text += f"；另 {len(feature_groups) - 3} 组"
        posture_text = "姿态变化已记录" if any(frame.auxiliary for frame in frames) else "姿态字段未提供"
        fixed_audit = result.fixed_sentinel_audit
        if fixed_audit is None:
            fixed_audit_text = ""
        elif fixed_audit.fixed_coordinate_count:
            fixed_audit_text = (
                f"固定值 {fixed_audit.sentinel_value}："
                f"{fixed_audit.fixed_coordinate_count:,} 坐标/{fixed_audit.frame_count} 帧恒定"
                f" · 共 {fixed_audit.total_occurrences:,} 次（仅诊断）"
            )
        elif not fixed_audit.same_shape:
            fixed_audit_text = "固定值审计：帧尺寸不一致，未计算固定坐标（仅诊断）"
        else:
            fixed_audit_text = (
                f"固定值 {fixed_audit.sentinel_value}：未发现跨帧恒定坐标"
                "（仅诊断）"
            )
        impact_audit = result.fixed_sentinel_impact_audit
        if impact_audit is None:
            fixed_impact_text = ""
        elif impact_audit.candidate_peak_data_available:
            fixed_impact_text = (
                f"固定值影响：宽筛 {impact_audit.candidate_peak_affected_count:,}/"
                f"{impact_audit.candidate_peak_count:,} · 返回源 "
                f"{impact_audit.affected_returned_source_count:,}"
                f"（质量 {impact_audit.affected_quality_source_count:,}）"
            )
        else:
            fixed_impact_text = (
                f"固定值影响：返回源 {impact_audit.affected_returned_source_count:,}"
                f"（质量 {impact_audit.affected_quality_source_count:,}）"
                " · 宽筛峰证据不可用"
            )
        compact = bool(self.__dict__.get("content_compact", False))
        if compact:
            # 720p 高度下右栏只承担“现在算到哪、主要目标是什么”的职责；
            # 完整逐帧审计仍在“15 帧证据”窗口中，避免文字被裁切后误以为没有结果。
            lines = [
                f"{len(frames)} 帧 · {size_text} · {duration_text} · Δt {interval_text}",
                f"候选 {candidate_range}/帧 · 质量 {quality_range}/帧 · 工作集前 {result.source_working_limit:,}" if result.source_working_limit is not None else f"候选 {candidate_range}/帧 · 质量 {quality_range}/帧 · 工作集全量",
                f"配准 {shift_norm:.2f}px · {posture_text} · 严格静态 {result.stable_source_count:,} · 持续候选 {result.persistent_source_count:,} · 叠加暗星 {result.stack_faint_count:,} · 补检 {result.candidate_consensus_count:,}（中值 {result.temporal_reference_count:,}） · 点源 moving {len(moving_point_tracks):,}",
            ]
        else:
            lines = [
                f"{len(frames)} 帧 · {size_text} · {duration_text} · Δt {interval_text} · 曝光 {exposure_text}",
                f"候选 {candidate_range}/帧 · 质量 {quality_range}/帧 · 工作集前 {result.source_working_limit:,}" if result.source_working_limit is not None else f"候选 {candidate_range}/帧 · 质量 {quality_range}/帧 · 工作集全量",
                f"配准末端 {shift_norm:.3f}px · {posture_text} · 严格静态 {result.stable_source_count:,} · 持续候选 {result.persistent_source_count:,} · 叠加暗星 {result.stack_faint_count:,} · 补检 {result.candidate_consensus_count:,}（中值 {result.temporal_reference_count:,}） · 点源 moving {len(moving_point_tracks):,} · 线状 {group_text}",
            ]
        if fixed_audit_text:
            lines.append(fixed_audit_text)
        if fixed_impact_text:
            lines.append(fixed_impact_text)
        if result.fast_sequence:
            sample_text = (
                f"每个背景块最多抽样 {result.background_sample_limit:,} 点"
                if result.background_sample_limit is not None
                else "背景网格使用序列快速抽样"
            )
            model_text = {
                "shared_sequence_pilot": "共享首帧背景/RMS",
                "shared_sequence_pilot_with_fallback": "共享背景/RMS（异常尺寸帧回退）",
                "per_frame_local": "逐帧背景/RMS",
            }.get(result.background_model_mode, result.background_model_mode)
            lines.append(f"快速路径：{sample_text} · {result.calculation_dtype} · {model_text} · 线审计保留")
        elif result.background_sample_limit is not None:
            lines.append(f"背景网格：每块最多 {result.background_sample_limit:,} 点 · 非快速路径")
        if moving_point_tracks:
            track = max(moving_point_tracks, key=lambda item: item.presence)
            first = track.points[0]
            last = track.points[-1]
            kinematics = self._track_kinematics(result, track)
            speed_text = f"{track.speed_px_per_frame:.3f}px/frame"
            direction_text = f"{np.degrees(np.arctan2(last.aligned_y - first.aligned_y, last.aligned_x - first.aligned_x)):.1f}°"
            if kinematics is not None:
                speed_text = f"{float(kinematics['speed_px_per_s']):.2f}px/s"
                direction_text = f"{float(kinematics['direction_deg_image']):.1f}°"
            evidence_text = "高速点源补充关联" if track.evidence_level == "fast_point_motion" else "严格点轨迹"
            lines.append(
                f"点状 moving F{first.frame_index + 1:02d}–F{last.frame_index + 1:02d} · {evidence_text} · "
                f"{speed_text} · 方向 {direction_text} · 位移 {track.displacement_px:.1f}px · 拟合 RMS {track.fit_rms_px:.2f}px"
            )
        if moving_features:
            track = max(moving_features, key=lambda item: item.presence)
            point = track.points[0]
            last = track.points[-1]
            kinematics = self._track_kinematics(result, track)
            speed_text = "—"
            direction_text = "—"
            forecast_text = ""
            if kinematics is not None:
                speed_text = f"{float(kinematics['speed_px_per_s']):.2f}px/s"
                if kinematics["speed_ci95_low_px_per_s"] is not None and kinematics["speed_ci95_high_px_per_s"] is not None:
                    speed_text += f" [95% {float(kinematics['speed_ci95_low_px_per_s']):.2f}–{float(kinematics['speed_ci95_high_px_per_s']):.2f}]"
                direction_text = f"{float(kinematics['direction_deg_image']):.1f}°"
                if kinematics["direction_ci95_half_width_deg"] is not None:
                    direction_text += f"±{float(kinematics['direction_ci95_half_width_deg']):.1f}°"
                if kinematics["predicted_x_px"] is not None and kinematics["predicted_y_px"] is not None:
                    forecast_text = f"  ·  外推 X {float(kinematics['predicted_x_px']):.1f}/Y {float(kinematics['predicted_y_px']):.1f}"
            lines.append(f"主轨迹 F{point.frame_index + 1:02d}–F{last.frame_index + 1:02d} · {speed_text} · 方向 {direction_text}{forecast_text}")
            lines.append("图像平面 px/s · 无 WCS/像元尺度时不换算真实速度")
        else:
            lines.append("尚无跨帧 moving · 单帧长线只能作为 candidate 复核")
        self.sequence_evidence_label.config(text="\n".join(lines))

    def _show_catalog_match(self) -> None:
        """打开可选的星表核验窗口；星表和 WCS 由用户显式提供。"""

        if self.selected_frame is None:
            messagebox.showinfo("星表核验", "请先选择一个 FITS 文件。")
            return
        if self.catalog_window is not None and self.catalog_window.winfo_exists():
            self.catalog_window.deiconify()
            self.catalog_window.lift()
            self.catalog_window.focus_force()
            return

        frame_path = self.selected_frame
        window = tk.Toplevel(self)
        self.catalog_window = window
        window.title("RST19 · 星表核验")
        window.geometry("1060x720")
        window.minsize(900, 620)
        window.configure(bg=PAPER)

        def close_window() -> None:
            self.catalog_window = None
            window.destroy()

        window.protocol("WM_DELETE_WINDOW", close_window)
        header = tk.Frame(window, bg=PAPER)
        header.pack(fill="x", padx=24, pady=(20, 12))
        self._mono_label(header, "CATALOG CROSS-CHECK / PRIOR WCS → LOCAL FIT", color=INK_SOFT, size=8, bg=PAPER).pack(anchor="w")
        self._label(header, "星表核验", color=NAVY_DARK, size=20, bold=True, bg=PAPER).pack(anchor="w", pady=(5, 0))
        self._label(
            header,
            "星表先用于身份核验；匹配点足够且几何稳定时，可进一步拟合视场内局部 WCS。它不替代图像检测，也不能把匹配数直接当作全部恒星数。",
            color=INK_SOFT,
            size=9,
            bg=PAPER,
            justify="left",
            wraplength=950,
        ).pack(anchor="w", pady=(4, 0))

        form = tk.Frame(window, bg=PAPER_LIGHT, highlightbackground=PAPER_LINE, highlightthickness=1)
        form.pack(fill="x", padx=24, pady=(0, 12))
        for column in (1, 3, 5):
            form.grid_columnconfigure(column, weight=1)
        catalog_path_var = tk.StringVar(value=str(self.catalog_path) if self.catalog_path else "")
        center_ra = ""
        center_dec = ""
        if self.analysis is not None and self.analysis.frame.auxiliary is not None:
            center_ra = f"{self.analysis.frame.auxiliary.ra_deg:.8f}"
            center_dec = f"{self.analysis.frame.auxiliary.dec_deg:.8f}"
        elif self.sequence_result is not None:
            selected_index = self._selected_frame_index()
            if selected_index is not None and selected_index < len(self.sequence_result.frames):
                auxiliary = self.sequence_result.frames[selected_index].auxiliary_dict
                if "ra" in auxiliary:
                    center_ra = f"{auxiliary['ra']:.8f}"
                if "dec" in auxiliary:
                    center_dec = f"{auxiliary['dec']:.8f}"
        ra_var = tk.StringVar(value=center_ra)
        dec_var = tk.StringVar(value=center_dec)
        scale_var = tk.StringVar(value="")
        rotation_var = tk.StringVar(value="0")
        parity_var = tk.StringVar(value="1")
        radius_var = tk.StringVar(value="3")

        self._mono_label(form, "CATALOG CSV", color=AMBER, size=8, bg=PAPER_LIGHT).grid(row=0, column=0, padx=(14, 7), pady=(13, 7), sticky="e")
        path_entry = tk.Entry(form, textvariable=catalog_path_var, bg=PAPER, fg=INK, insertbackground=INK, relief="flat", highlightbackground=PAPER_LINE, highlightthickness=1, font=(MONO, 9))
        path_entry.grid(row=0, column=1, columnspan=5, padx=(0, 8), pady=(10, 7), sticky="ew")

        def choose_catalog() -> None:
            chosen = filedialog.askopenfilename(
                parent=window,
                title="选择 CSV 星表",
                filetypes=(("CSV catalog", "*.csv"), ("All files", "*.*")),
            )
            if chosen:
                catalog_path_var.set(chosen)

        tk.Button(form, text="选择文件", command=choose_catalog, bg=PAPER, fg=INK, activebackground="#dcecf0", relief="flat", bd=0, padx=10, pady=6, font=(SANS, 9, "bold")).grid(row=0, column=6, padx=(0, 14), pady=(10, 7))

        field_specs = (
            ("CENTER RA / deg", ra_var, 1, 0),
            ("CENTER DEC / deg", dec_var, 1, 2),
            ("PIXEL SCALE / arcsec px⁻¹", scale_var, 1, 4),
            ("ROTATION / deg", rotation_var, 2, 0),
            ("PARITY ±1", parity_var, 2, 2),
            ("MATCH RADIUS / px", radius_var, 2, 4),
        )
        for label_text, variable, row, label_column in field_specs:
            self._mono_label(form, label_text, color=INK_SOFT, size=8, bg=PAPER_LIGHT).grid(row=row, column=label_column, padx=(14 if label_column == 0 else 12, 7), pady=(4, 13), sticky="e")
            tk.Entry(form, textvariable=variable, width=12, bg=PAPER, fg=INK, insertbackground=INK, relief="flat", highlightbackground=PAPER_LINE, highlightthickness=1, font=(MONO, 9)).grid(row=row, column=label_column + 1, padx=(0, 8), pady=(4, 13), sticky="ew")

        note = self._label(
            form,
            "当前 FITS 没有标准 WCS。RA/DEC 可用首行辅助字段作为光轴先验；先运行身份匹配，再点击“根据匹配拟合 WCS”。拟合失败或内点不足时继续保持待标定，不能凭截图猜尺度。",
            color=INK_SOFT,
            size=8,
            bg=PAPER_LIGHT,
            justify="left",
            wraplength=950,
        )
        note.grid(row=3, column=0, columnspan=7, padx=14, pady=(0, 12), sticky="w")

        body = tk.Frame(window, bg=PAPER_LIGHT, highlightbackground=PAPER_LINE, highlightthickness=1)
        body.pack(fill="both", expand=True, padx=24, pady=(0, 24))
        toolbar = tk.Frame(body, bg=PAPER_LIGHT)
        toolbar.pack(fill="x", padx=14, pady=(12, 8))
        status = self._mono_label(toolbar, "尚未运行 · 需要 CSV 与已标定 WCS", color=INK_SOFT, size=8, bg=PAPER_LIGHT)
        status.pack(side="left")
        run_button = tk.Button(toolbar, text="运行星表核验", bg=NAVY, fg=WHITE, activebackground=NAVY_SOFT, activeforeground=WHITE, relief="flat", bd=0, padx=13, pady=7, font=(SANS, 9, "bold"))
        run_button.pack(side="right")
        calibrate_button = tk.Button(toolbar, text="根据匹配拟合 WCS", state="disabled", bg=MINT, fg=WHITE, activebackground="#43856f", activeforeground=WHITE, relief="flat", bd=0, padx=13, pady=7, font=(SANS, 9, "bold"))
        calibrate_button.pack(side="right", padx=(0, 8))
        export_calibration_button = tk.Button(toolbar, text="导出校准 JSON", state="disabled", bg=PAPER, fg=INK, activebackground="#dcecf0", relief="flat", bd=0, padx=11, pady=7, font=(SANS, 9, "bold"))
        export_calibration_button.pack(side="right", padx=(0, 8))
        validation_button = tk.Button(toolbar, text="验证 15 帧 WCS", state="disabled", bg=AMBER, fg=NAVY_DARK, activebackground=AMBER_LIGHT, activeforeground=NAVY_DARK, relief="flat", bd=0, padx=11, pady=7, font=(SANS, 9, "bold"))
        validation_button.pack(side="right", padx=(0, 8))
        summary = self._label(body, "匹配结果将在这里显示", color=INK_SOFT, size=9, bg=PAPER_LIGHT, justify="left", anchor="w")
        summary.pack(fill="x", padx=14, pady=(0, 8))
        calibration_summary = self._mono_label(body, "校准状态：尚未根据匹配点拟合 · 至少需要 6 个有效且非共线匹配", color=INK_SOFT, size=8, bg=PAPER_LIGHT, anchor="w", justify="left", wraplength=950)
        calibration_summary.pack(fill="x", padx=14, pady=(0, 8))
        columns = ("det", "source", "det_xy", "pred_xy", "residual", "mag")
        match_tree = ttk.Treeview(body, columns=columns, show="headings")
        for column, title, width in (("det", "DETECTION ID", 110), ("source", "CATALOG ID", 180), ("det_xy", "DETECTED X / Y", 150), ("pred_xy", "PREDICTED X / Y", 150), ("residual", "RESIDUAL / px", 120), ("mag", "CATALOG MAG", 120)):
            match_tree.heading(column, text=title)
            match_tree.column(column, width=width, anchor="w", stretch=column in {"source", "pred_xy"})
        scroll = ttk.Scrollbar(body, orient="vertical", command=match_tree.yview)
        match_tree.configure(yscrollcommand=scroll.set)
        match_tree.pack(side="left", fill="both", expand=True, padx=(14, 0), pady=(0, 14))
        scroll.pack(side="right", fill="y", padx=(0, 14), pady=(0, 14))

        current_analysis: FrameAnalysis | None = None
        current_wcs: TangentPlaneWCS | None = None
        current_catalog: tuple[CatalogSource, ...] = ()
        current_frame: Path | None = None
        validation_window: tk.Toplevel | None = None

        def render(analysis: FrameAnalysis, wcs: TangentPlaneWCS, catalog_path: Path, matched_frame: Path, catalog: tuple[CatalogSource, ...]) -> None:
            nonlocal current_analysis, current_wcs, current_catalog, current_frame
            matching = analysis.matching
            if matching is None:
                return
            current_analysis = analysis
            current_wcs = wcs
            current_catalog = catalog
            current_frame = matched_frame
            self.catalog_analysis = analysis
            self.catalog_match_result = matching
            self.catalog_wcs = wcs
            self.catalog_calibration = None
            self.catalog_path = catalog_path
            self.catalog_frame_path = matched_frame
            self.hover_catalog_match = None
            calibrate_button.config(state="normal" if matching.matched_count >= 6 else "disabled", text="根据匹配拟合 WCS")
            export_calibration_button.config(state="disabled")
            validation_button.config(state="normal", text="验证 15 帧 WCS")
            calibration_summary.config(text="校准状态：可以开始局部仿射拟合 · 仅使用唯一匹配点，不是盲解算" if matching.matched_count >= 6 else "校准状态：匹配点不足 6 个，暂不输出像元角尺度或角速度")
            for item in match_tree.get_children():
                match_tree.delete(item)
            for match in matching.matches:
                magnitude = f"{match.catalog_magnitude:.3f}" if match.catalog_magnitude is not None else "—"
                match_tree.insert(
                    "",
                    tk.END,
                    values=(
                        f"{match.detection_id:04d}",
                        match.source_id,
                        f"{match.detection_x:.1f} / {match.detection_y:.1f}",
                        f"{match.predicted_x:.1f} / {match.predicted_y:.1f}",
                        f"{match.residual_px:.3f}",
                        magnitude,
                    ),
                )
            rms = f"{matching.rms_residual_px:.3f}px" if matching.rms_residual_px is not None else "—"
            summary.config(text=f"检测源 {len(analysis.detection.quality_sources):,} · 匹配成功 {matching.matched_count:,} · 未匹配检测源 {len(matching.unmatched_detection_ids):,} · 未匹配目录源 {len(matching.unmatched_catalog_ids):,}\n匹配半径 {matching.radius_px:.2f}px · RMS 残差 {rms} · 检测源匹配比例 {matching.inlier_ratio:.2%} · 一对一分配 {matching.assignment_mode}\n绿色环为目录预测位置；连线长度是位置残差，不能解释为运动轨迹。")
            self.overlay_mode_var.set("catalog")
            self._update_overlay_hint()
            self._draw_preview()

        def render_calibration(calibration: AffineWCSCalibration) -> None:
            self.catalog_calibration = calibration
            leave_one_out_text = f"{calibration.leave_one_out_rms_residual_px:.3f}px" if calibration.leave_one_out_rms_residual_px is not None else "—"
            calibration_summary.config(
                text=(
                    f"局部 WCS 已拟合：内点 {calibration.inlier_count}/{calibration.matched_count} "
                    f"({calibration.inlier_ratio:.1%}) · 像元角尺度 {calibration.plate_scale_arcsec_per_pixel:.4f} arcsec/px "
                    f"· 旋转 {calibration.rotation_deg:.3f}° · parity {calibration.parity:+d} "
                    f"· 内点 RMS {calibration.rms_residual_px:.3f}px · 留一 RMS "
                    f"{leave_one_out_text} · 各向异性 {calibration.anisotropy_ratio:.5f}"
                )
            )
            export_calibration_button.config(state="normal")
            status.config(text="已完成星表核验与局部 WCS 拟合 · 重新打开 15 帧证据可查看角速度换算")

        def calibrate() -> None:
            if current_analysis is None or current_wcs is None or current_frame is None:
                return
            matching = current_analysis.matching
            if matching is None:
                return
            calibrate_button.config(state="disabled", text="拟合中…")
            status.config(text="正在根据唯一匹配点拟合局部仿射 WCS…")
            token = self.frame_token
            wcs = current_wcs
            catalog = current_catalog

            def worker() -> None:
                try:
                    calibration = fit_affine_wcs_from_matches(matching.matches, catalog, wcs)
                    self.result_queue.put(("catalog-calibration", token, (window, render_calibration, calibrate_button, status, calibration)))
                except Exception as exc:  # noqa: BLE001 - worker returns a visible calibration error
                    self.result_queue.put(("catalog-calibration-error", token, (window, calibrate_button, status, str(exc))))

            threading.Thread(target=worker, daemon=True).start()

        def export_calibration() -> None:
            calibration = self.catalog_calibration
            if calibration is None or current_wcs is None or current_frame is None:
                return
            chosen = filedialog.asksaveasfilename(
                parent=window,
                title="导出局部 WCS 校准证据",
                defaultextension=".json",
                initialfile="catalog_wcs_calibration.json",
                filetypes=(("JSON", "*.json"), ("All files", "*.*")),
            )
            if not chosen:
                return
            payload = {
                "schema_version": 1,
                "frame_path": str(current_frame),
                "catalog_path": str(self.catalog_path) if self.catalog_path is not None else None,
                "reference_wcs": current_wcs.as_dict(),
                "calibration": calibration.as_dict(),
                "note": "局部仿射反校准；依赖先验 WCS 引导的唯一匹配，不是全天空盲解算或轨道速度。",
            }
            try:
                Path(chosen).write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
            except OSError as exc:
                messagebox.showerror("导出失败", str(exc), parent=window)
                return
            status.config(text=f"校准证据已导出 · {Path(chosen)}")

        def render_validation(report: WCSValidationReport, output: Path) -> None:
            nonlocal validation_window
            if validation_window is not None and validation_window.winfo_exists():
                validation_window.destroy()
            validation_window = tk.Toplevel(window)
            validation_window.title("RST19 · 15 帧 WCS 验证")
            validation_window.geometry("1180x760")
            validation_window.minsize(920, 620)
            validation_window.configure(bg=PAPER)
            validation_window.protocol("WM_DELETE_WINDOW", validation_window.destroy)
            header = tk.Frame(validation_window, bg=PAPER)
            header.pack(fill="x", padx=22, pady=(18, 10))
            self._mono_label(header, "WCS VALIDATION / 15 FRAME HOLDOUT EVIDENCE", color=INK_SOFT, size=8, bg=PAPER).pack(anchor="w")
            self._label(header, "逐帧局部 WCS 验证", color=NAVY_DARK, size=18, bold=True, bg=PAPER).pack(anchor="w", pady=(4, 0))
            validated_rows = [row for row in report.frame_rows if row.status == "validated"]
            rms_values = [row.rms_residual_px for row in validated_rows if row.rms_residual_px is not None]
            scale_values = [row.plate_scale_arcsec_per_pixel for row in validated_rows if row.plate_scale_arcsec_per_pixel is not None]
            summary_text = (
                f"有效帧 {report.validated_count}/{report.frame_count} ({report.validation_ratio:.1%}) · "
                f"匹配星表源 {report.catalog_source_count} · 最少匹配 {report.min_matches} · "
                f"RMS 中位 {float(np.median(rms_values)):.3f}px" if rms_values else
                f"有效帧 {report.validated_count}/{report.frame_count} ({report.validation_ratio:.1%}) · 匹配星表源 {report.catalog_source_count} · 最少匹配 {report.min_matches}"
            )
            if scale_values:
                summary_text += f" · 像元角尺度中位 {float(np.median(scale_values)):.5f} arcsec/px"
            self._label(
                header,
                summary_text,
                color=INK,
                size=9,
                bg=PAPER,
                anchor="w",
                justify="left",
                wraplength=1080,
            ).pack(anchor="w", pady=(5, 0))
            self._label(
                header,
                f"方法：{report.method}。只有 status=validated 的帧才进入汇总；不把匹配数量当作恒星真值。结果目录：{output}",
                color=INK_SOFT,
                size=8,
                bg=PAPER,
                anchor="w",
                justify="left",
                wraplength=1080,
            ).pack(anchor="w", pady=(4, 0))
            body = tk.Frame(validation_window, bg=PAPER_LIGHT, highlightbackground=PAPER_LINE, highlightthickness=1)
            body.pack(fill="both", expand=True, padx=22, pady=(0, 22))
            table_frame = tk.Frame(body, bg=PAPER_LIGHT)
            table_frame.pack(fill="both", expand=True, padx=12, pady=(12, 6))
            columns = ("frame", "status", "matched", "inlier", "rms", "loo", "scale", "rotation", "parity", "anisotropy")
            tree = ttk.Treeview(table_frame, columns=columns, show="headings", height=10)
            specs = (
                ("frame", "FRAME", 70),
                ("status", "STATUS", 100),
                ("matched", "MATCHED", 90),
                ("inlier", "INLIER", 90),
                ("rms", "RMS / px", 100),
                ("loo", "LOO RMS / px", 120),
                ("scale", "SCALE / arcsec px⁻¹", 155),
                ("rotation", "ROTATION / °", 115),
                ("parity", "PARITY", 75),
                ("anisotropy", "ANISOTROPY", 115),
            )
            for column, title, column_width in specs:
                tree.heading(column, text=title)
                tree.column(column, width=column_width, anchor="w", stretch=column in {"status", "scale"})
            scroll = ttk.Scrollbar(table_frame, orient="vertical", command=tree.yview)
            tree.configure(yscrollcommand=scroll.set)
            tree.pack(side="left", fill="both", expand=True)
            scroll.pack(side="right", fill="y")
            for row in report.frame_rows:
                tree.insert(
                    "",
                    tk.END,
                    values=(
                        f"F{row.frame_index + 1:02d}",
                        row.status,
                        row.matched_count,
                        row.inlier_count,
                        f"{row.rms_residual_px:.3f}" if row.rms_residual_px is not None else "—",
                        f"{row.leave_one_out_rms_residual_px:.3f}" if row.leave_one_out_rms_residual_px is not None else "—",
                        f"{row.plate_scale_arcsec_per_pixel:.5f}" if row.plate_scale_arcsec_per_pixel is not None else "—",
                        f"{row.rotation_deg:.3f}" if row.rotation_deg is not None else "—",
                        f"{row.parity:+d}" if row.parity is not None else "—",
                        f"{row.anisotropy_ratio:.5f}" if row.anisotropy_ratio is not None else "—",
                    ),
                )
            charts = tk.Frame(body, bg=PAPER_LIGHT)
            charts.pack(fill="x", padx=12, pady=(0, 12))
            chart_photos: list[ImageTk.PhotoImage] = []
            for chart_name in ("wcs_validation_residual.png", "wcs_validation_scale.png"):
                chart_path = output / chart_name
                if not chart_path.is_file():
                    continue
                image = Image.open(chart_path).convert("RGB")
                image.thumbnail((520, 260), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(image)
                chart_photos.append(photo)
                tk.Label(charts, image=photo, bg=PAPER_LIGHT).pack(side="left", fill="x", expand=True, padx=5)
            charts.chart_photos = chart_photos  # type: ignore[attr-defined]

        def validate_sequence() -> None:
            if not self.frames:
                messagebox.showerror("WCS 验证", "当前数据目录没有 FITS 文件", parent=window)
                return
            try:
                catalog_path = Path(catalog_path_var.get().strip()).expanduser().resolve()
                if not catalog_path.is_file():
                    raise ValueError("请选择存在的 CSV 星表文件")
                center_ra_value = float(ra_var.get())
                center_dec_value = float(dec_var.get())
                pixel_scale_value = float(scale_var.get())
                rotation_value = float(rotation_var.get())
                parity_value = int(parity_var.get())
                radius_value = float(radius_var.get())
                if pixel_scale_value <= 0 or radius_value <= 0:
                    raise ValueError("像元角尺度和匹配半径必须为正数")
                if parity_value not in (-1, 1):
                    raise ValueError("PARITY 只能填写 1 或 -1")
                threshold, min_distance, max_sources, zero_point, psf_fwhm, min_flux_snr = self._read_parameters()
                width = self.preview_shape[1] if self.preview_shape is not None else 4096
                height = self.preview_shape[0] if self.preview_shape is not None else 4096
                wcs = TangentPlaneWCS(
                    center_ra_deg=center_ra_value,
                    center_dec_deg=center_dec_value,
                    pixel_scale_arcsec=pixel_scale_value,
                    crpix_x=width / 2.0,
                    crpix_y=height / 2.0,
                    rotation_deg=rotation_value,
                    parity=parity_value,
                )
            except (OSError, ValueError) as exc:
                messagebox.showerror("WCS 验证参数错误", str(exc), parent=window)
                return
            validation_button.config(state="disabled", text="验证中…")
            status.config(text="正在逐帧检测、匹配并拟合 WCS…")
            token = self.frame_token
            frame_paths = tuple(self.frames)
            output_dir = PROJECT_ROOT / "tmp" / "gui-wcs-validation"

            def worker() -> None:
                try:
                    catalog = load_catalog_csv(catalog_path)

                    def progress(index: int, total: int) -> None:
                        self.result_queue.put(("catalog-validation-progress", token, (window, status, index, total)))

                    report = run_sequence_wcs_validation(
                        frame_paths,
                        catalog,
                        wcs,
                        detector_kwargs={
                            "threshold_sigma": threshold,
                            "min_distance": min_distance,
                            "max_sources": max_sources,
                            "zero_point": zero_point,
                            "psf_fwhm": psf_fwhm,
                            "min_flux_snr": min_flux_snr,
                            "proposal_mode": self.proposal_mode_var.get(),
                            "reject_linear_artifacts": True,
                        },
                        match_radius_px=radius_value,
                        min_matches=6,
                        progress=progress,
                    )
                    output = write_wcs_validation_artifacts(report, output_dir)
                except Exception as exc:  # noqa: BLE001 - worker returns a visible validation error
                    self.result_queue.put(("catalog-validation-error", token, (window, validation_button, status, str(exc))))
                    return
                self.result_queue.put(("catalog-validation", token, (window, validation_button, status, report, output, render_validation)))

            threading.Thread(target=worker, daemon=True).start()

        def run() -> None:
            try:
                catalog_path = Path(catalog_path_var.get().strip()).expanduser().resolve()
                if not catalog_path.is_file():
                    raise ValueError("请选择存在的 CSV 星表文件")
                center_ra_value = float(ra_var.get())
                center_dec_value = float(dec_var.get())
                pixel_scale_value = float(scale_var.get())
                rotation_value = float(rotation_var.get())
                parity_value = int(parity_var.get())
                radius_value = float(radius_var.get())
                if pixel_scale_value <= 0 or radius_value <= 0:
                    raise ValueError("像元角尺度和匹配半径必须为正数")
                if parity_value not in (-1, 1):
                    raise ValueError("PARITY 只能填写 1 或 -1")
                threshold, min_distance, max_sources, zero_point, psf_fwhm, min_flux_snr = self._read_parameters()
                width = self.preview_shape[1] if self.preview_shape is not None else 4096
                height = self.preview_shape[0] if self.preview_shape is not None else 4096
                wcs = TangentPlaneWCS(
                    center_ra_deg=center_ra_value,
                    center_dec_deg=center_dec_value,
                    pixel_scale_arcsec=pixel_scale_value,
                    crpix_x=width / 2.0,
                    crpix_y=height / 2.0,
                    rotation_deg=rotation_value,
                    parity=parity_value,
                )
            except (OSError, ValueError) as exc:
                messagebox.showerror("星表参数错误", str(exc), parent=window)
                return
            run_button.config(state="disabled", text="核验中…")
            status.config(text="后台读取 FITS、星表并执行唯一匹配…")
            token = self.frame_token

            def worker() -> None:
                try:
                    frame = read_fits(frame_path)
                    catalog = load_catalog_csv(catalog_path)
                    analysis = analyze_frame(
                        frame,
                        catalog=catalog,
                        wcs=wcs,
                        threshold_sigma=threshold,
                        min_distance=min_distance,
                        max_sources=max_sources,
                        zero_point=zero_point,
                        psf_fwhm=psf_fwhm,
                        min_flux_snr=min_flux_snr,
                        proposal_mode=self.proposal_mode_var.get(),
                        match_radius_px=radius_value,
                        reject_linear_artifacts=True,
                    )
                    self.result_queue.put(("catalog", token, (window, render, run_button, status, analysis, wcs, catalog_path, frame_path, catalog)))
                except Exception as exc:  # noqa: BLE001 - worker must return a user-facing error
                    self.result_queue.put(("catalog-error", token, (window, run_button, status, str(exc))))

            threading.Thread(target=worker, daemon=True).start()

        run_button.config(command=run)
        calibrate_button.config(command=calibrate)
        export_calibration_button.config(command=export_calibration)
        validation_button.config(command=validate_sequence)

    def _show_sequence_evidence(self) -> None:
        """打开逐帧、逐轨迹和图表证据窗口。"""

        if self.sequence_result is None:
            messagebox.showinfo("15 帧证据", "请先完成“分析 15 帧动目标”。")
            return
        if self.sequence_evidence_window is not None and self.sequence_evidence_window.winfo_exists():
            self.sequence_evidence_window.deiconify()
            self.sequence_evidence_window.lift()
            self.sequence_evidence_window.focus_force()
            return
        result = self.sequence_result
        window = tk.Toplevel(self)
        self.sequence_evidence_window = window
        window.title("RST19 · 15 帧证据")
        window.geometry("1120x760")
        window.minsize(900, 620)
        window.configure(bg=PAPER)
        window.protocol("WM_DELETE_WINDOW", lambda: self._close_sequence_evidence(window))

        header = tk.Frame(window, bg=PAPER)
        header.pack(fill="x", padx=24, pady=(20, 13))
        self._mono_label(header, "SEQUENCE EVIDENCE / REPRODUCIBLE SUMMARY", color=INK_SOFT, size=8, bg=PAPER).pack(anchor="w")
        self._label(header, "15 帧观测证据", color=NAVY_DARK, size=20, bold=True, bg=PAPER).pack(anchor="w", pady=(5, 0))
        self._label(header, "逐帧计数、配准稳定性、差分图与线状轨迹分开呈现；图证按需生成，不在打开窗口时提前计算。", color=INK_SOFT, size=9, bg=PAPER).pack(anchor="w", pady=(4, 0))

        datetimes = self._sequence_datetimes(result)
        interval_values = [
            (datetimes[index] - datetimes[index - 1]).total_seconds()
            for index in range(1, len(datetimes))
            if datetimes[index] is not None
            and datetimes[index - 1] is not None
            and (datetimes[index] - datetimes[index - 1]).total_seconds() > 0
        ]
        median_interval = float(np.median(interval_values)) if interval_values else None
        duration = None
        if datetimes and datetimes[0] is not None and datetimes[-1] is not None:
            duration = (datetimes[-1] - datetimes[0]).total_seconds()
        max_shift = max(
            (float(np.hypot(shift[0], shift[1])) for shift in result.cumulative_shifts),
            default=0.0,
        )
        feature_groups: list[str] = []
        for track in result.motion_features:
            if not track.points:
                continue
            first_frame = track.points[0].frame_index + 1
            last_frame = track.points[-1].frame_index + 1
            if first_frame == last_frame:
                feature_groups.append(f"第 {first_frame} 帧单帧候选")
            else:
                state = "连续 moving 候选" if track.classification == "moving" else "连续候选"
                feature_groups.append(f"第 {first_frame}–{last_frame} 帧{state}")
        relation_box = tk.Frame(window, bg=PAPER_LIGHT, highlightbackground=PAPER_LINE, highlightthickness=1)
        relation_box.pack(fill="x", padx=24, pady=(0, 13))
        exposure_text = "—"
        if result.frames and result.frames[0].exposure_ms is not None:
            exposure_text = f"{result.frames[0].exposure_ms:g} ms"
        timing_text = f"{len(result.frames)} 帧连续观测 · 曝光 {exposure_text}"
        if duration is not None:
            timing_text += f" · 时间跨度 {duration:.3f} s"
        if median_interval is not None:
            timing_text += f" · 中位 Δt {median_interval:.3f} s"
        relation_line = (
            f"{timing_text} · 静态配准最大位移 {max_shift:.3f} px · "
            f"严格静态 {result.stable_source_count:,} · 持续候选 {result.persistent_source_count:,} · "
            f"叠加暗星 {result.stack_faint_count:,} · "
            f"补检 {result.candidate_consensus_count:,}（中值 {result.temporal_reference_count:,}） · "
            f"点源 moving {result.moving_track_count:,}（高速点 {result.fast_point_motion_count:,}） · "
            f"点源工作集 {'≤' + format(result.source_working_limit, ',') if result.source_working_limit is not None else '全量'}"
        )
        self._mono_label(relation_box, relation_line, color=INK, size=8, bg=PAPER_LIGHT, anchor="w").pack(
            fill="x", padx=12, pady=(9, 3)
        )
        audit = dict(result.candidate_consensus_audit)
        audit_line = (
            f"候选共识审计：锚点 {audit.get('anchor_total', 0):,} · "
            f"达到持续门槛 {audit.get('anchor_presence_ge_persistent', 0):,} · "
            f"通过 {audit.get('accepted', 0):,} · "
            f"重复/已有质量源 {audit.get('reject_quality_overlap', 0):,} · "
            f"PSF 支持不足 {audit.get('reject_temporal_support', 0):,} · "
            f"有效帧不足 {audit.get('reject_temporal_valid', 0):,} · "
            f"形状 {audit.get('reject_temporal_shape', 0):,}"
        )
        self._mono_label(relation_box, audit_line, color=INK_SOFT, size=8, bg=PAPER_LIGHT, anchor="w").pack(
            fill="x", padx=12, pady=(0, 3)
        )
        temporal_mode_label = {
            "median": "时间中值",
            "coadd": "稳健叠加",
            "both": "中值+叠加并集",
        }.get(result.temporal_proposal_mode, result.temporal_proposal_mode)
        temporal_line = (
            f"时序补提案：{temporal_mode_label} · 中值种子 {result.temporal_reference_candidate_count:,} · "
            f"叠加种子 {result.temporal_coadd_candidate_count:,} · "
            f"逐帧响应 SNR≥{result.temporal_candidate_min_snr:.1f} · "
            f"参考 SNR≥{result.temporal_reference_min_snr:.1f} · "
            f"PSF相关≥{result.temporal_min_psf_correlation:.2f} · "
            f"局部峰搜索={'窄/基准/宽' if result.temporal_multiscale else '基准'} ±1px；"
            "叠加只提出坐标，仍需逐帧原图细筛"
        )
        self._mono_label(relation_box, temporal_line, color=INK_SOFT, size=8, bg=PAPER_LIGHT, anchor="w").pack(
            fill="x", padx=12, pady=(0, 3)
        )
        group_text = "；".join(feature_groups) if feature_groups else "当前没有线状候选轨迹"
        self._label(
            relation_box,
            f"线状证据分组：{group_text}。两组不同时间/位置的线不会自动合并；候选峰数量也不会跨帧相加为恒星总数。",
            color=INK_SOFT,
            size=8,
            bg=PAPER_LIGHT,
            anchor="w",
            justify="left",
            wraplength=1050,
        ).pack(fill="x", padx=12, pady=(0, 9))

        notebook = ttk.Notebook(window)
        notebook.pack(fill="both", expand=True, padx=24, pady=(0, 24))
        frame_tab = tk.Frame(notebook, bg=PAPER_LIGHT)
        motion_tab = tk.Frame(notebook, bg=PAPER_LIGHT)
        motion_audit_tab = tk.Frame(notebook, bg=PAPER_LIGHT)
        innovation_tab = tk.Frame(notebook, bg=PAPER_LIGHT)
        chart_tab = tk.Frame(notebook, bg=PAPER_LIGHT)
        telemetry_tab = tk.Frame(notebook, bg=PAPER_LIGHT)
        telemetry_position_tab = tk.Frame(notebook, bg=PAPER_LIGHT)
        visual_tab = tk.Frame(notebook, bg=PAPER_LIGHT)
        injection_tab = tk.Frame(notebook, bg=PAPER_LIGHT)
        sweep_tab = tk.Frame(notebook, bg=PAPER_LIGHT)
        trail_audit_tab = tk.Frame(notebook, bg=PAPER_LIGHT)
        notebook.add(frame_tab, text="逐帧表")
        notebook.add(motion_tab, text="线状轨迹")
        notebook.add(motion_audit_tab, text="线状诊断")
        notebook.add(innovation_tab, text="创新摘要")
        notebook.add(chart_tab, text="图表")
        notebook.add(telemetry_tab, text="辅助遥测")
        notebook.add(telemetry_position_tab, text="遥测轨迹")
        notebook.add(visual_tab, text="图证")
        notebook.add(injection_tab, text="星点实验")
        notebook.add(sweep_tab, text="阈值扫描")
        notebook.add(trail_audit_tab, text="单图长线")

        frame_columns = ("frame", "time", "dt", "candidate", "quality", "shift", "attitude")
        frame_tree = ttk.Treeview(frame_tab, columns=frame_columns, show="headings")
        frame_specs = (
            ("frame", "FRAME", 70),
            ("time", "DATE-OBS", 190),
            ("dt", "Δt / s", 80),
            ("candidate", "CANDIDATE", 100),
            ("quality", "QUALITY / WORKSET" if result.source_working_limit is not None else "QUALITY", 135),
            ("shift", "SHIFT / px", 100),
            ("attitude", "ROLL / PITCH / YAW", 250),
        )
        for column, title, width in frame_specs:
            frame_tree.heading(column, text=title)
            frame_tree.column(column, width=width, anchor="w", stretch=column in {"time", "attitude"})
        frame_scroll = ttk.Scrollbar(frame_tab, orient="vertical", command=frame_tree.yview)
        frame_tree.configure(yscrollcommand=frame_scroll.set)
        frame_tree.pack(side="left", fill="both", expand=True, padx=(14, 0), pady=14)
        frame_scroll.pack(side="right", fill="y", padx=(0, 14), pady=14)
        for index, frame in enumerate(result.frames):
            interval = "—"
            if index > 0 and datetimes[index] is not None and datetimes[index - 1] is not None:
                interval = f"{(datetimes[index] - datetimes[index - 1]).total_seconds():.3f}"
            shift = result.cumulative_shifts[index] if index < len(result.cumulative_shifts) else (0.0, 0.0)
            shift_text = f"{float(np.hypot(shift[0], shift[1])):.3f}"
            auxiliary = frame.auxiliary_dict
            attitude = " / ".join(f"{auxiliary.get(key, float('nan')):.3f}" for key in ("roll", "pitch", "yaw")) if auxiliary else "—"
            frame_tree.insert("", tk.END, values=(f"{frame.frame_index + 1:02d}", frame.timestamp or "—", interval, f"{frame.candidate_count:,}", f"{frame.quality_count:,}", shift_text, attitude))

        motion_columns = ("id", "state", "frames", "duration", "speed", "direction", "length", "snr", "rms", "forecast")
        motion_tree = ttk.Treeview(motion_tab, columns=motion_columns, show="headings")
        motion_specs = (
            ("id", "ID", 70),
            ("state", "STATE", 100),
            ("frames", "FRAMES", 90),
            ("duration", "DURATION / s", 100),
            ("speed", "SPEED / px/s [95%]", 150),
            ("direction", "DIRECTION / ° [95%]", 145),
            ("length", "LENGTH / px", 100),
            ("snr", "RESIDUAL SNR", 110),
            ("rms", "FIT RMS / px", 100),
            ("forecast", "FORECAST X / Y [95%]", 190),
        )
        for column, title, width in motion_specs:
            motion_tree.heading(column, text=title)
            motion_tree.column(column, width=width, anchor="w", stretch=column == "forecast")
        motion_scroll = ttk.Scrollbar(motion_tab, orient="vertical", command=motion_tree.yview)
        motion_tree.configure(yscrollcommand=motion_scroll.set)
        motion_tree.pack(side="left", fill="both", expand=True, padx=(14, 0), pady=14)
        motion_scroll.pack(side="right", fill="y", padx=(0, 14), pady=14)
        for track in result.motion_features:
            points = track.points
            frame_text = f"{points[0].frame_index + 1}–{points[-1].frame_index + 1}" if points else "—"
            duration = ((datetimes[points[-1].frame_index] - datetimes[points[0].frame_index]).total_seconds() if points and datetimes[points[-1].frame_index] is not None and datetimes[points[0].frame_index] is not None else None)
            kinematics = self._track_kinematics(result, track)
            speed = float(kinematics["speed_px_per_s"]) if kinematics is not None else None
            direction = float(kinematics["direction_deg_image"]) if kinematics is not None else points[0].angle_deg if points else None
            speed_text = f"{speed:.2f}" if speed is not None else "—"
            if kinematics is not None and kinematics["speed_ci95_low_px_per_s"] is not None and kinematics["speed_ci95_high_px_per_s"] is not None:
                speed_text = f"{speed_text} [{float(kinematics['speed_ci95_low_px_per_s']):.2f},{float(kinematics['speed_ci95_high_px_per_s']):.2f}]"
            direction_text = f"{direction:.1f}" if direction is not None else "—"
            if kinematics is not None and kinematics["direction_ci95_half_width_deg"] is not None:
                direction_text += f" ±{float(kinematics['direction_ci95_half_width_deg']):.1f}"
            forecast = "—"
            if kinematics is not None and kinematics["predicted_x_px"] is not None and kinematics["predicted_y_px"] is not None:
                forecast = f"{float(kinematics['predicted_x_px']):.1f} / {float(kinematics['predicted_y_px']):.1f}"
                if kinematics["predicted_x_ci95_half_width_px"] is not None and kinematics["predicted_y_ci95_half_width_px"] is not None:
                    forecast += f" ±{float(kinematics['predicted_x_ci95_half_width_px']):.1f}/{float(kinematics['predicted_y_ci95_half_width_px']):.1f}"
            motion_tree.insert("", tk.END, values=(
                f"{track.track_id:04d}",
                track.classification,
                frame_text,
                f"{duration:.3f}" if duration is not None else "—",
                speed_text,
                direction_text,
                f"{np.median([point.length_px for point in points]):.1f}" if points else "—",
                f"{max(point.residual_snr for point in points):.1f}" if points else "—",
                f"{track.fit_rms_px:.2f}" if track.fit_rms_px is not None else "—",
                forecast,
            ))

        self._build_telemetry_table(telemetry_tab, result)
        self._build_telemetry_position_tab(telemetry_position_tab, result)
        self._build_innovation_summary_tab(innovation_tab, result)
        self._build_motion_audit_tab(motion_audit_tab, result)
        self._build_visual_evidence_tab(visual_tab, window, result)
        self._build_injection_tab(injection_tab, window, result)
        self._build_detection_sweep_tab(sweep_tab, window, result)
        self._build_single_frame_trail_audit_tab(trail_audit_tab, window, result)

        chart_canvas = tk.Canvas(chart_tab, bg=PAPER_LIGHT, highlightthickness=0)
        chart_canvas.pack(fill="both", expand=True, padx=14, pady=14)
        chart_canvas.bind("<Configure>", lambda _event: self._draw_evidence_charts(chart_canvas, result))
        self._draw_evidence_charts(chart_canvas, result)

    def _build_motion_audit_tab(self, parent: tk.Misc, result: SequenceResult) -> None:
        """显示线状残差从连通域到最终候选的逐帧筛选证据。"""

        header = tk.Frame(parent, bg=PAPER_LIGHT)
        header.pack(fill="x", padx=14, pady=(14, 6))
        self._label(
            header,
            "线状诊断：解释残差亮点如何被逐级筛选，而不是把所有亮点都标成运动目标",
            color=INK,
            size=10,
            bg=PAPER_LIGHT,
            anchor="w",
        ).pack(side="left", fill="x", expand=True)
        self._mono_label(
            header,
            "COMPONENT → AREA → GEOMETRY → EDGE → LINE",
            color=AMBER,
            size=8,
            bg=PAPER_LIGHT,
        ).pack(side="right")
        self._label(
            parent,
            "每一行只描述本帧残差筛选的中间计数：连通域是阈值后的原始结构，面积/几何是逐级通过数，触边剔除单独记录，最终线才进入跨帧关联。它不是星点总数，也不能单独计算误检率。",
            color=INK_SOFT,
            size=9,
            bg=PAPER_LIGHT,
            anchor="w",
            justify="left",
            wraplength=1050,
        ).pack(fill="x", padx=14, pady=(0, 8))

        audits: tuple[MotionFrameAudit, ...] = tuple(sorted(result.motion_frame_audits, key=lambda item: item.frame_index))
        if not audits:
            self._label(
                parent,
                "当前序列缓存没有逐帧线状诊断。请清空旧序列缓存后重新运行 15 帧分析。",
                color=AMBER,
                size=9,
                bg=PAPER_LIGHT,
                anchor="w",
                justify="left",
                wraplength=1050,
            ).pack(fill="x", padx=14, pady=(6, 14))
            return

        body = tk.Frame(parent, bg=PAPER_LIGHT)
        body.pack(fill="both", expand=True, padx=14, pady=(0, 14))
        body.grid_columnconfigure(0, weight=1)
        body.grid_rowconfigure(0, weight=3)
        body.grid_rowconfigure(1, weight=2)

        table_frame = tk.Frame(body, bg=PAPER_LIGHT)
        table_frame.grid(row=0, column=0, sticky="nsew")
        columns = (
            "frame",
            "threshold",
            "noise",
            "valid",
            "support",
            "components",
            "area",
            "geometry",
            "edge",
            "lines",
            "snr",
        )
        tree = ttk.Treeview(table_frame, columns=columns, show="headings", height=8)
        specs = (
            ("frame", "FRAME", 62),
            ("threshold", "THRESH / ADU", 100),
            ("noise", "DIFF σ / ADU", 100),
            ("valid", "VALID PX", 88),
            ("support", "SUPPORT PX", 98),
            ("components", "COMPONENTS", 100),
            ("area", "AREA PASS", 88),
            ("geometry", "GEOMETRY PASS", 112),
            ("edge", "EDGE REJECT", 100),
            ("lines", "EMITTED LINES", 108),
            ("snr", "MAX RES SNR", 108),
        )
        for column, title, width in specs:
            tree.heading(column, text=title)
            tree.column(column, width=width, anchor="w", stretch=False)
        y_scroll = ttk.Scrollbar(table_frame, orient="vertical", command=tree.yview)
        x_scroll = ttk.Scrollbar(table_frame, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)
        tree.grid(row=0, column=0, sticky="nsew")
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll.grid(row=1, column=0, sticky="ew")
        table_frame.grid_rowconfigure(0, weight=1)
        table_frame.grid_columnconfigure(0, weight=1)
        tree.tag_configure("hit", foreground="#c7359e")
        tree.tag_configure("edge", foreground="#b34f3e")
        for audit in audits:
            tags = ("edge",) if audit.edge_rejected_count else (("hit",) if audit.feature_count else ())
            tree.insert(
                "",
                tk.END,
                tags=tags,
                values=(
                    f"{audit.frame_index + 1:02d}",
                    f"{audit.threshold_adu:.1f}",
                    f"{audit.residual_noise_adu:.1f}",
                    f"{audit.valid_pixel_count:,}",
                    f"{audit.support_pixel_count:,}",
                    f"{audit.component_count:,}",
                    audit.area_pass_count,
                    audit.geometry_pass_count,
                    audit.edge_rejected_count,
                    audit.feature_count,
                    f"{audit.max_feature_residual_snr:.1f}" if audit.max_feature_residual_snr is not None else "—",
                ),
            )

        chart = tk.Canvas(body, bg=PAPER_LIGHT, highlightthickness=0, height=245)
        chart.grid(row=1, column=0, sticky="nsew", pady=(8, 0))
        chart.bind("<Configure>", lambda _event: self._draw_motion_audit_chart(chart, result))
        self._draw_motion_audit_chart(chart, result)

    @staticmethod
    def _draw_motion_audit_chart(canvas: tk.Canvas, result: SequenceResult) -> None:
        """在 Tk Canvas 上画逐帧线状筛选漏斗。"""

        canvas.delete("all")
        audits: tuple[MotionFrameAudit, ...] = tuple(sorted(result.motion_frame_audits, key=lambda item: item.frame_index))
        width = max(620, int(canvas.winfo_width()))
        height = max(220, int(canvas.winfo_height()))
        if not audits:
            canvas.create_text(width // 2, height // 2, text="暂无逐帧线状诊断", fill=INK_SOFT, font=(SANS, 10))
            return
        left, top, right, bottom = 58, 24, width - 28, height - 42
        max_count = max(
            1,
            max(
                max(audit.component_count, audit.area_pass_count, audit.geometry_pass_count, audit.feature_count)
                for audit in audits
            ),
        )
        max_log_count = math.log1p(max_count)
        canvas.create_text(left, 10, text="LOG1P(COUNT)", anchor="w", fill=INK_SOFT, font=(MONO, 8))
        canvas.create_text(right, height - 14, text="FRAME", anchor="e", fill=INK_SOFT, font=(MONO, 8))
        for tick in range(0, 4):
            fraction = tick / 3.0
            y = bottom - fraction * (bottom - top)
            value = int(round(math.expm1(max_log_count * fraction)))
            canvas.create_line(left, y, right, y, fill="#e4ded1", width=1)
            canvas.create_text(left - 8, y, text=str(value), anchor="e", fill=INK_SOFT, font=(MONO, 8))
        canvas.create_line(left, top, left, bottom, fill=INK_SOFT, width=1)
        canvas.create_line(left, bottom, right, bottom, fill=INK_SOFT, width=1)
        series = (
            ("components", lambda audit: audit.component_count, AMBER),
            ("area pass", lambda audit: audit.area_pass_count, AMBER_LIGHT),
            ("geometry pass", lambda audit: audit.geometry_pass_count, SKY),
            ("emitted lines", lambda audit: audit.feature_count, MOTION_TRAIL),
        )
        for series_index, (name, getter, color) in enumerate(series):
            points = []
            for index, audit in enumerate(audits):
                x = left if len(audits) == 1 else left + index / (len(audits) - 1) * (right - left)
                y = bottom - math.log1p(float(getter(audit))) / max_log_count * (bottom - top)
                points.append((x, y))
            if len(points) >= 2:
                canvas.create_line(points, fill=color, width=2)
            for x, y in points:
                canvas.create_oval(x - 2, y - 2, x + 2, y + 2, fill=color, outline=color)
            legend_x = left + series_index * 150
            canvas.create_line(legend_x, top - 8, legend_x + 16, top - 8, fill=color, width=3)
            canvas.create_text(legend_x + 22, top - 8, text=name, anchor="w", fill=INK, font=(MONO, 8))
        for index, audit in enumerate(audits):
            if index == 0 or index == len(audits) - 1 or len(audits) <= 8:
                x = left if len(audits) == 1 else left + index / (len(audits) - 1) * (right - left)
                canvas.create_text(x, bottom + 14, text=f"F{audit.frame_index + 1:02d}", fill=INK_SOFT, font=(MONO, 8))

    @staticmethod
    def _innovation_summary_rows(
        result: SequenceResult,
        calibration: AffineWCSCalibration | None = None,
    ) -> tuple[tuple[str, str, str, str], ...]:
        """把序列结果整理成答辩用的集中式创新指标表。"""

        datetimes = StarfieldApp._sequence_datetimes(result)
        intervals = [
            (datetimes[index] - datetimes[index - 1]).total_seconds()
            for index in range(1, len(datetimes))
            if datetimes[index] is not None
            and datetimes[index - 1] is not None
            and (datetimes[index] - datetimes[index - 1]).total_seconds() > 0
        ]
        duration = None
        if datetimes and datetimes[0] is not None and datetimes[-1] is not None:
            duration = (datetimes[-1] - datetimes[0]).total_seconds()
        max_shift = max(
            (float(np.hypot(shift[0], shift[1])) for shift in result.cumulative_shifts),
            default=0.0,
        )
        exposure_values = [frame.exposure_ms for frame in result.frames if frame.exposure_ms is not None]
        rows: list[tuple[str, str, str, str]] = [
            ("时序关系", "观测帧数", f"{len(result.frames)} 帧", "按 DATE-OBS 排序的连续曝光"),
            ("时序关系", "时间跨度", f"{duration:.3f} s" if duration is not None else "—", "首帧到末帧的实际时间"),
            ("时序关系", "中位帧间隔", f"{float(np.median(intervals)):.3f} s" if intervals else "—", "由相邻 DATE-OBS 差分得到"),
            ("时序关系", "曝光", f"{float(np.median(exposure_values)):g} ms" if exposure_values else "—", "FITS 辅助字段口径"),
            ("固定星场", "最大累计配准", f"{max_shift:.3f} px", "平移基线；不是完整 WCS"),
            (
                "固定星场",
                "点源配准工作集",
                f"每帧 ≤{result.source_working_limit:,}" if result.source_working_limit is not None else "全量",
                "候选总数仍全量审计；质量数和点轨迹为工作集口径",
            ),
            ("固定星场", "像素计算精度", result.calculation_dtype, "序列整数 FITS 的 float32 仅是实现级优化；不改变输入 ADU"),
            (
                "计算口径",
                "背景网格统计",
                (
                    f"每块 ≤{result.background_sample_limit:,} 点"
                    if result.background_sample_limit is not None
                    else "完整网格像素"
                ),
                "序列快速路径使用确定性步进抽样；单图精测使用完整网格统计",
            ),
            (
                "计算口径",
                "掩膜卷积",
                "稀疏单遍" if result.fast_sequence else "归一化双遍",
                "无效像素稀疏时的速度优化；质量规则和线状审计仍单独保留",
            ),
            (
                "计算口径",
                "时序补提案",
                {
                    "median": "时间中值",
                    "coadd": "稳健叠加",
                    "both": "中值 + 叠加并集",
                }.get(result.temporal_proposal_mode, result.temporal_proposal_mode),
                f"中值种子 {result.temporal_reference_candidate_count:,} · 叠加种子 {result.temporal_coadd_candidate_count:,} · "
                f"逐帧响应 SNR≥{result.temporal_candidate_min_snr:.1f} · "
                f"参考 SNR≥{result.temporal_reference_min_snr:.1f} · "
                f"PSF相关≥{result.temporal_min_psf_correlation:.2f} · "
                f"局部峰搜索={'窄/基准/宽' if result.temporal_multiscale else '基准'} ±1px；只作宽筛提案",
            ),
            (
                "计算口径",
                "背景模型",
                {
                    "shared_sequence_pilot": "共享首帧 background/RMS",
                    "shared_sequence_pilot_with_fallback": "共享模型 + 尺寸异常回退",
                    "per_frame_local": "逐帧 background/RMS",
                }.get(result.background_model_mode, result.background_model_mode),
                "同尺寸连续帧复用局部背景；尺寸不一致时安全回退，不广播错误模型",
            ),
            ("固定星场", "严格静态点轨迹", f"{result.stable_source_count:,}", "至少约 80% 帧出现且未呈现运动的点轨迹"),
            ("固定星场", "持续点源候选", f"{result.persistent_source_count:,}", "至少约 50% 帧出现、无显著位移；不是恒星真值"),
            ("运动证据", "点源 moving 轨迹", f"{result.moving_track_count:,}", "包含普通关联与高速点源补充关联；不等于物理真值"),
            ("运动证据", "高速点源补充轨迹", f"{result.fast_point_motion_count:,}", "宽筛候选 + 原图 flux SNR/PSF 复核；独立于固定星场 4 px 最近邻"),
            ("运动证据", "线状证据组", f"{len(result.motion_features):,}", "单帧候选与跨帧候选分开统计"),
        ]
        for track in result.motion_features:
            points = track.points
            if not points:
                continue
            first_frame = points[0].frame_index + 1
            last_frame = points[-1].frame_index + 1
            frame_text = f"F{first_frame:02d}" if first_frame == last_frame else f"F{first_frame:02d}–F{last_frame:02d}"
            state = "moving 候选" if track.classification == "moving" else "单帧/待复核候选"
            rows.append((f"线状 {track.track_id:04d}", "状态 / 帧范围", f"{state} · {frame_text}", "必须结合原图、差分和人工复核"))
            snrs = [float(point.residual_snr) for point in points]
            if len(points) >= 2 and intervals:
                first_time = datetimes[points[0].frame_index] if points[0].frame_index < len(datetimes) else None
                last_time = datetimes[points[-1].frame_index] if points[-1].frame_index < len(datetimes) else None
                track_duration = (last_time - first_time).total_seconds() if first_time is not None and last_time is not None else None
                horizon = float(np.median(intervals) * 5.0)
                kinematics: dict[str, Any] | None = None
                if track_duration is not None and track_duration > 0 and first_time is not None:
                    point_times = [
                        datetimes[point.frame_index]
                        if point.frame_index < len(datetimes)
                        else None
                        for point in points
                    ]
                    if all(item is not None for item in point_times):
                        elapsed = [(item - first_time).total_seconds() for item in point_times if item is not None]
                        kinematics = _fit_constant_velocity(
                            elapsed,
                            [point.aligned_x for point in points],
                            [point.aligned_y for point in points],
                            horizon,
                        )
                speed = float(kinematics["speed_px_per_s"]) if kinematics is not None else None
                direction = float(kinematics["direction_deg_image"]) if kinematics is not None else float(
                    np.degrees(np.arctan2(points[-1].aligned_y - points[0].aligned_y, points[-1].aligned_x - points[0].aligned_x))
                )
                speed_low = float(kinematics["speed_ci95_low_px_per_s"]) if kinematics is not None and kinematics["speed_ci95_low_px_per_s"] is not None else None
                speed_high = float(kinematics["speed_ci95_high_px_per_s"]) if kinematics is not None and kinematics["speed_ci95_high_px_per_s"] is not None else None
                direction_half = float(kinematics["direction_ci95_half_width_deg"]) if kinematics is not None and kinematics["direction_ci95_half_width_deg"] is not None else None
                speed_text = f"{speed:.2f} px/s" if speed is not None else "—"
                if speed_low is not None and speed_high is not None:
                    speed_text += f" [95% {speed_low:.2f}, {speed_high:.2f}]"
                direction_text = f"{direction:.1f}°"
                if direction_half is not None:
                    direction_text += f" ±{direction_half:.1f}° (95%)"
                fit_rms = float(kinematics["fit_rms_px"]) if kinematics is not None else track.fit_rms_px
                rows.extend(
                    (
                        (f"线状 {track.track_id:04d}", "位移 / 速度", f"{track.displacement_px:.1f} px · {speed_text}", "实际 DATE-OBS 常速度 OLS；区间未包含配准系统误差"),
                        (f"线状 {track.track_id:04d}", "方向 / 拟合 RMS", f"{direction_text} · {fit_rms:.2f} px" if fit_rms is not None else f"{direction_text} · —", "图像 X/Y 方向；95% 区间为小样本 delta method"),
                    )
                )
                if calibration is not None and kinematics is not None:
                    dx_per_s = float(kinematics["velocity_x_px_per_s"])
                    dy_per_s = float(kinematics["velocity_y_px_per_s"])
                    east_rate, north_rate = calibration.pixel_velocity_to_tangent_arcsec(dx_per_s, dy_per_s)
                    angular_speed = float(np.hypot(east_rate, north_rate))
                    sky_direction = float(np.degrees(np.arctan2(north_rate, east_rate)))
                    rows.append(
                        (
                            f"线状 {track.track_id:04d}",
                            "切平面角速度",
                            f"{angular_speed:.4f} arcsec/s · 东 {east_rate:+.4f} / 北 {north_rate:+.4f}",
                            f"由星表局部仿射校准换算；天空切平面方向 {sky_direction:.1f}°，不是轨道速度",
                        )
                    )
                if kinematics is not None:
                    forecast_x = float(kinematics["predicted_x_px"])
                    forecast_y = float(kinematics["predicted_y_px"])
                    x_half = kinematics["predicted_x_ci95_half_width_px"]
                    y_half = kinematics["predicted_y_ci95_half_width_px"]
                    uncertainty = f" · 95% X±{float(x_half):.1f} / Y±{float(y_half):.1f} px" if x_half is not None and y_half is not None else " · 区间不可估"
                    rows.append((f"线状 {track.track_id:04d}", "短期图像外推", f"X {forecast_x:.1f} / Y {forecast_y:.1f} · +{horizon:.3f} s{uncertainty}", "OLS 轨迹均值区间；不是单次观测预测区间或轨道预测"))
            else:
                point = points[0]
                rows.extend(
                    (
                        (f"线状 {track.track_id:04d}", "形态长度 / 宽度", f"{point.length_px:.1f} / {point.width_px:.1f} px", "单帧形状证据，不能单独确认运动"),
                        (f"线状 {track.track_id:04d}", "残差 SNR / 方向", f"{max(snrs):.1f}σ / {point.angle_deg:.1f}°", "方向是线形主轴，不是跨帧速度方向"),
                    )
                )

        for prefix, label in (("j2000", "J2000"), ("wgs84", "WGS84")):
            velocity_norms: list[float] = []
            position_deltas: list[float] = []
            first = result.frames[0].auxiliary_dict if result.frames else {}
            first_position = np.asarray([first.get(f"{prefix}_{axis}", np.nan) for axis in ("x", "y", "z")], dtype=np.float64)
            for frame in result.frames:
                auxiliary = frame.auxiliary_dict
                velocity = np.asarray([auxiliary.get(f"{prefix}_{axis}v", np.nan) for axis in ("x", "y", "z")], dtype=np.float64)
                position = np.asarray([auxiliary.get(f"{prefix}_{axis}", np.nan) for axis in ("x", "y", "z")], dtype=np.float64)
                if np.all(np.isfinite(velocity)):
                    velocity_norms.append(float(np.linalg.norm(velocity)))
                if np.all(np.isfinite(position)) and np.all(np.isfinite(first_position)):
                    position_deltas.append(float(np.linalg.norm(position - first_position)))
            if velocity_norms:
                rows.append(("辅助遥测", f"{label} 速度模长", f"{min(velocity_norms):.2f}–{max(velocity_norms):.2f} m/s?", "载体辅助字段；单位仍待主办方确认"))
            if position_deltas:
                rows.append(("辅助遥测", f"{label} 相对首帧位置", f"{position_deltas[-1]:.2f} m", "位置字段差值，不是图像平面位移"))
        if calibration is None:
            rows.append(("结论边界", "WCS / 物理速度", "待标定", "没有完整 WCS 和像元角尺度，不输出角秒或真实天体速度"))
        else:
            leave_one_out_text = f"{calibration.leave_one_out_rms_residual_px:.3f}" if calibration.leave_one_out_rms_residual_px is not None else "—"
            rows.extend(
                (
                    ("WCS 校准", "匹配 / 内点", f"{calibration.inlier_count}/{calibration.matched_count} ({calibration.inlier_ratio:.1%})", "先验 WCS 引导的局部仿射校准；不是盲解算"),
                    ("WCS 校准", "像元角尺度", f"{calibration.plate_scale_arcsec_per_pixel:.4f} arcsec/px", "由匹配星的切平面坐标拟合"),
                    ("WCS 校准", "残差 / 留一验证", f"{calibration.rms_residual_px:.3f} / {leave_one_out_text} px", "内点 RMS / 留一 RMS；留一不是独立帧验证"),
                    ("WCS 校准", "各向异性", f"{calibration.anisotropy_ratio:.5f}", "偏离 1 时保留完整仿射证据，不强行压成标量 WCS"),
                    ("结论边界", "WCS / 物理速度", "已有局部角尺度", "可报告视场内角速度；仍不是目标轨道速度，也不替代正式 WCS 验证"),
                )
            )
        return tuple(rows)

    def _build_innovation_summary_tab(self, parent: tk.Misc, result: SequenceResult) -> None:
        """集中显示可用于答辩的序列关系、运动量和解释边界。"""

        header = tk.Frame(parent, bg=PAPER_LIGHT)
        header.pack(fill="x", padx=14, pady=(14, 8))
        self._label(
            header,
            "创新摘要：把 15 帧关系、图像平面运动量和辅助遥测放在同一张可复核表中",
            color=INK,
            size=10,
            bg=PAPER_LIGHT,
            anchor="w",
        ).pack(side="left", fill="x", expand=True)
        badge = "LOCAL WCS → ARCSEC" if self.catalog_calibration is not None else "NO WCS → NO ARCSEC"
        self._mono_label(header, badge, color=MINT if self.catalog_calibration is not None else AMBER, size=8, bg=PAPER_LIGHT).pack(side="right")

        note = self._label(
            parent,
            "速度与方向只对当前图像坐标中的线状候选成立；若完成星表局部 WCS 校准，才额外给出视场切平面角速度。辅助遥测的 m/s? 是载体状态量；第 1 帧单帧长线与第 9–15 帧连续候选不会自动合并。",
            color=INK_SOFT,
            size=9,
            bg=PAPER_LIGHT,
            anchor="w",
            justify="left",
            wraplength=1050,
        )
        note.pack(fill="x", padx=14, pady=(0, 8))

        columns = ("group", "metric", "value", "meaning")
        tree = ttk.Treeview(parent, columns=columns, show="headings")
        specs = (
            ("group", "GROUP", 110),
            ("metric", "METRIC", 165),
            ("value", "VALUE", 300),
            ("meaning", "INTERPRETATION", 520),
        )
        for column, title, width in specs:
            tree.heading(column, text=title)
            tree.column(column, width=width, anchor="w", stretch=column in {"value", "meaning"})
        tree.tag_configure("motion", foreground="#9b5d19")
        tree.tag_configure("boundary", foreground="#b34f3e")
        y_scroll = ttk.Scrollbar(parent, orient="vertical", command=tree.yview)
        x_scroll = ttk.Scrollbar(parent, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)
        tree.pack(side="left", fill="both", expand=True, padx=(14, 0), pady=(0, 14))
        y_scroll.pack(side="right", fill="y", padx=(0, 14), pady=(0, 14))
        tree.configure(xscrollcommand=x_scroll.set)
        x_scroll.pack(side="bottom", fill="x", padx=(14, 14), pady=(0, 14))
        for group, metric, value, meaning in self._innovation_summary_rows(result, self.catalog_calibration):
            tags = ("boundary",) if group == "结论边界" else (("motion",) if group.startswith("线状") else ())
            tree.insert("", tk.END, values=(group, metric, value, meaning), tags=tags)

    def _build_single_frame_trail_audit_tab(
        self,
        parent: tk.Misc,
        window: tk.Toplevel,
        result: SequenceResult,
    ) -> None:
        """按需显示每一帧独立长线证据，避免把单帧形状冒充跨帧运动。"""

        header = tk.Frame(parent, bg=PAPER_LIGHT)
        header.pack(fill="x", padx=14, pady=(14, 7))
        note = self._label(
            header,
            "逐帧独立运行长线几何筛选；这里只回答“本帧有无长线候选”，不直接给出 moving 结论",
            color=INK_SOFT,
            size=9,
            bg=PAPER_LIGHT,
            anchor="w",
            justify="left",
        )
        note.pack(side="left", fill="x", expand=True)
        status = self._mono_label(header, "未运行", color=INK_SOFT, size=8, bg=PAPER_LIGHT)
        status.pack(side="right", padx=(8, 0))
        run_button = tk.Button(
            header,
            text="运行单图审计",
            bg=AMBER,
            fg=NAVY_DARK,
            activebackground=AMBER_LIGHT,
            activeforeground=NAVY_DARK,
            relief="flat",
            bd=0,
            padx=11,
            pady=6,
            font=(SANS, 9, "bold"),
        )
        run_button.pack(side="right", padx=(8, 0))
        source_audit_button = tk.Button(
            header,
            text="星点时序审计",
            bg=PAPER,
            fg=INK,
            activebackground="#dcecf0",
            activeforeground=NAVY_DARK,
            relief="solid",
            bd=1,
            padx=10,
            pady=5,
            font=(SANS, 9, "bold"),
        )
        source_audit_button.pack(side="right", padx=(8, 0))

        parameter_note = self._mono_label(
            parent,
            "检测口径：当前阈值 / 最小间距 / PSF FWHM / 通量 SNR / PSF支持≥3/9 · 线状门槛：残差 ≥15σ 且 ≥100 ADU、面积 ≥40 px、轴比 ≥4",
            color=INK_SOFT,
            size=8,
            bg=PAPER_LIGHT,
            anchor="w",
        )
        parameter_note.pack(fill="x", padx=14, pady=(0, 7))

        summary = self._label(
            parent,
            "审计后显示每帧候选数、最长线长度、宽度、残差 SNR、触边状态和包围盒。",
            color=INK,
            size=9,
            bg=PAPER_LIGHT,
            anchor="w",
            justify="left",
            wraplength=1040,
        )
        summary.pack(fill="x", padx=14, pady=(0, 7))

        body = tk.Frame(parent, bg=PAPER_LIGHT)
        body.pack(fill="both", expand=True, padx=14, pady=(0, 14))
        body.grid_columnconfigure(0, weight=1)
        body.grid_rowconfigure(0, weight=3)
        body.grid_rowconfigure(1, weight=2)

        table_frame = tk.Frame(body, bg=PAPER_LIGHT)
        table_frame.grid(row=0, column=0, sticky="nsew")
        columns = (
            "frame",
            "time",
            "candidate",
            "quality",
            "trails",
            "length",
            "width",
            "snr",
            "edge",
            "angle",
            "bbox",
        )
        tree = ttk.Treeview(table_frame, columns=columns, show="headings", height=9)
        specs = (
            ("frame", "FRAME", 62),
            ("time", "DATE-OBS", 178),
            ("candidate", "CANDIDATE", 92),
            ("quality", "QUALITY", 82),
            ("trails", "TRAILS", 65),
            ("length", "MAX LEN / px", 96),
            ("width", "WIDTH / px", 86),
            ("snr", "RES SNR", 86),
            ("edge", "EDGE", 58),
            ("angle", "ANGLE / °", 82),
            ("bbox", "BBOX", 160),
        )
        for column, title, width in specs:
            tree.heading(column, text=title)
            tree.column(column, width=width, anchor="w", stretch=column in {"time", "bbox"})
        tree.tag_configure("trail", foreground="#9b5d19")
        tree.tag_configure("edge", foreground="#b34f3e")
        y_scroll = ttk.Scrollbar(table_frame, orient="vertical", command=tree.yview)
        x_scroll = ttk.Scrollbar(table_frame, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)
        tree.grid(row=0, column=0, sticky="nsew")
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll.grid(row=1, column=0, sticky="ew")
        table_frame.grid_rowconfigure(0, weight=1)
        table_frame.grid_columnconfigure(0, weight=1)

        chart_frame = tk.Frame(body, bg=PAPER_LIGHT)
        chart_frame.grid(row=1, column=0, sticky="nsew", pady=(8, 0))
        chart_frame.grid_columnconfigure(0, weight=1)
        chart_frame.grid_columnconfigure(1, weight=1)
        count_label = tk.Label(
            chart_frame,
            text="运行审计后查看候选数量曲线",
            bg=NAVY_DARK,
            fg=SKY_LIGHT,
            font=(SANS, 9),
        )
        count_label.grid(row=0, column=0, sticky="nsew", padx=(0, 4))
        length_label = tk.Label(
            chart_frame,
            text="运行审计后查看最长线长度曲线",
            bg=NAVY_DARK,
            fg=SKY_LIGHT,
            font=(SANS, 9),
        )
        length_label.grid(row=0, column=1, sticky="nsew", padx=(4, 0))
        chart_frame.grid_rowconfigure(0, weight=1)

        def set_chart(label: tk.Label, path: Path, empty_text: str) -> None:
            if not path.is_file():
                label.config(image="", text=empty_text)
                label.image = None
                return
            image = Image.open(path).convert("RGB")
            image.thumbnail((500, 245), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(image)
            label.config(image=photo, text="")
            label.image = photo

        def render(rows: tuple[SingleFrameTrailAuditRow, ...], output: Path) -> None:
            for item in tree.get_children():
                tree.delete(item)
            for row in rows:
                row_tags = ("edge",) if row.edge_trail_count else (("trail",) if row.trail_count else ())
                tree.insert(
                    "",
                    tk.END,
                    tags=row_tags,
                    values=(
                        f"{row.frame_number:02d}",
                        row.timestamp or "—",
                        f"{row.candidate_count:,}",
                        f"{row.quality_count:,}",
                        row.trail_count,
                        f"{row.max_length_px:.1f}" if row.max_length_px is not None else "—",
                        f"{row.max_width_px:.1f}" if row.max_width_px is not None else "—",
                        f"{row.max_residual_snr:.1f}" if row.max_residual_snr is not None else "—",
                        row.edge_trail_count,
                        f"{row.longest_angle_deg:.1f}" if row.longest_angle_deg is not None else "—",
                        row.longest_bbox or "—",
                    ),
                )
            hit_frames = [row.frame_number for row in rows if row.trail_count > 0]
            groups = "、".join(str(frame) for frame in hit_frames) if hit_frames else "无"
            total_trails = sum(row.trail_count for row in rows)
            summary.config(
                text=(
                    f"完成 {len(rows)} 帧 · 独立长线候选 {total_trails} 条 · 命中帧：{groups}。"
                    " 命中帧仍是形状候选；只有结合连续帧、配准坐标和拟合残差后，才进入 moving 复核。"
                )
            )
            set_chart(count_label, output / "single_frame_trail_counts.png", "没有候选数量曲线")
            set_chart(length_label, output / "single_frame_trail_lengths.png", "没有最长线曲线")

        def run() -> None:
            try:
                threshold, min_distance, max_sources, _zero_point, psf_fwhm, min_flux_snr = self._read_parameters()
            except ValueError as exc:
                messagebox.showerror("参数错误", str(exc), parent=window)
                return
            frame_paths = tuple(Path(frame.path) for frame in result.frames)
            if not frame_paths or not all(path.is_file() for path in frame_paths):
                messagebox.showerror("单图审计", "序列结果中的 FITS 路径已失效，请重新载入数据并运行 15 帧分析。", parent=window)
                return
            run_button.config(state="disabled", text="审计中…")
            status.config(text=f"0/{len(frame_paths)}")
            source_limit = "全量" if max_sources is None else str(max_sources)
            parameter_note.config(
                text=(
                    f"检测口径：threshold {threshold:g}σ · min distance {min_distance}px · PSF FWHM {psf_fwhm:g}px · "
                    f"flux SNR ≥{min_flux_snr:g}σ · PSF支持 ≥3/9 · 返回 {source_limit}；线状门槛：残差 ≥15σ 且 ≥100 ADU、面积 ≥40 px、轴比 ≥4"
                )
            )
            output_dir = PROJECT_ROOT / "tmp" / "gui-single-frame-trails"

            def progress(index: int, total: int) -> None:
                self.result_queue.put(("trail-audit-progress", self.frame_token, (window, status, f"{index}/{total}")))

            def worker() -> None:
                try:
                    rows = run_single_frame_trail_audit(
                        frame_paths,
                        threshold_sigma=threshold,
                        min_distance=min_distance,
                        aperture_radius=4,
                        psf_fwhm=psf_fwhm,
                        background_box_size=128,
                        min_flux_snr=min_flux_snr,
                        max_sources=max_sources,
                        progress=progress,
                    )
                    output = write_single_frame_trail_artifacts(rows, output_dir)
                except Exception as exc:  # noqa: BLE001 - worker returns a visible audit error
                    self.result_queue.put(("trail-audit-error", self.frame_token, (window, run_button, status, str(exc))))
                    return
                self.result_queue.put(("trail-audit", self.frame_token, (window, render, run_button, status, rows, output)))

            threading.Thread(target=worker, daemon=True).start()

        run_button.config(command=run)

    def _build_visual_evidence_tab(self, parent: tk.Misc, window: tk.Toplevel, result: SequenceResult) -> None:
        """按需生成并展示注册中值拼图、差分图和轨迹图。"""

        header = tk.Frame(parent, bg=PAPER_LIGHT)
        header.pack(fill="x", padx=14, pady=(14, 8))
        note = self._label(
            header,
            "图证来自当前序列结果：拼图用于检查固定星场，差分图显示注册后相对中值参考的正/负残差",
            color=INK_SOFT,
            size=9,
            bg=PAPER_LIGHT,
            anchor="w",
        )
        note.pack(side="left", fill="x", expand=True)
        status = self._mono_label(header, "未生成", color=INK_SOFT, size=8, bg=PAPER_LIGHT)
        status.pack(side="right", padx=(8, 0))
        run_button = tk.Button(
            header,
            text="生成图证",
            bg=NAVY,
            fg=WHITE,
            activebackground=NAVY_SOFT,
            activeforeground=WHITE,
            relief="flat",
            bd=0,
            padx=11,
            pady=6,
            font=(SANS, 9, "bold"),
        )
        run_button.pack(side="right", padx=(8, 0))

        body = tk.Frame(parent, bg=PAPER_LIGHT)
        body.pack(fill="both", expand=True, padx=14, pady=(0, 14))
        body.grid_columnconfigure(0, weight=1)
        body.grid_rowconfigure(0, weight=3)
        body.grid_rowconfigure(1, weight=2)
        image_row = tk.Frame(body, bg=PAPER_LIGHT)
        image_row.grid(row=0, column=0, sticky="nsew")
        mosaic_frame = tk.LabelFrame(image_row, text="REGISTERED MEDIAN MOSAIC", bg=PAPER_LIGHT, fg=INK_SOFT, font=(MONO, 8, "bold"), bd=1, relief="solid")
        mosaic_frame.pack(side="left", fill="both", expand=True, padx=(0, 7))
        difference_frame = tk.LabelFrame(image_row, text="REGISTERED DIFFERENCE", bg=PAPER_LIGHT, fg=INK_SOFT, font=(MONO, 8, "bold"), bd=1, relief="solid")
        difference_frame.pack(side="left", fill="both", expand=True, padx=(7, 0))
        mosaic_label = tk.Label(mosaic_frame, text="点击“生成图证”", bg=NAVY_DARK, fg=SKY_LIGHT, font=(SANS, 10))
        mosaic_label.pack(fill="both", expand=True, padx=8, pady=8)
        difference_label = tk.Label(difference_frame, text="点击“生成图证”", bg=NAVY_DARK, fg=SKY_LIGHT, font=(SANS, 10))
        difference_label.pack(fill="both", expand=True, padx=8, pady=8)
        evidence_row = tk.Frame(body, bg=PAPER_LIGHT)
        evidence_row.grid(row=1, column=0, sticky="nsew", pady=(8, 0))
        cutout_frame = tk.LabelFrame(evidence_row, text="MOTION CUTOUT CONTACT SHEET", bg=PAPER_LIGHT, fg=INK_SOFT, font=(MONO, 8, "bold"), bd=1, relief="solid")
        cutout_frame.pack(side="left", fill="both", expand=True, padx=(0, 7))
        trajectory_frame = tk.LabelFrame(evidence_row, text="FITTED IMAGE-PLANE TRAJECTORY", bg=PAPER_LIGHT, fg=INK_SOFT, font=(MONO, 8, "bold"), bd=1, relief="solid")
        trajectory_frame.pack(side="left", fill="both", expand=True, padx=(7, 0))
        cutout_label = tk.Label(cutout_frame, text="生成图证后查看逐帧裁剪", bg=NAVY_DARK, fg=SKY_LIGHT, font=(SANS, 10))
        cutout_label.pack(fill="both", expand=True, padx=8, pady=8)
        trajectory_label = tk.Label(trajectory_frame, text="生成图证后查看拟合轨迹", bg=NAVY_DARK, fg=SKY_LIGHT, font=(SANS, 10))
        trajectory_label.pack(fill="both", expand=True, padx=8, pady=8)

        def set_image(label: tk.Label, path: Path, empty_text: str, max_size: tuple[int, int]) -> None:
            if not path.is_file():
                label.config(image="", text=empty_text)
                label.image = None
                return
            image = Image.open(path).convert("RGB")
            image.thumbnail(max_size, Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(image)
            label.config(image=photo, text="")
            label.image = photo

        def render(output: Path) -> None:
            set_image(mosaic_label, output / "registered_mosaic.png", "未找到中值拼图", (500, 300))
            set_image(difference_label, output / "registered_difference.png", "未找到注册差分图", (500, 300))
            set_image(cutout_label, output / "motion_cutout_contact_sheet.png", "没有线状候选裁剪", (520, 220))
            set_image(trajectory_label, output / "motion_trajectory.png", "没有拟合轨迹图", (520, 220))
            status.config(text=f"已生成 · {output.name}")

        def run() -> None:
            run_button.config(state="disabled", text="生成中…")
            status.config(text="后台读取 15 帧")
            sequence_payload = result.as_dict()
            source_hint = PROJECT_ROOT / "tmp" / "rst19-gui-sequence.json"
            output_dir = PROJECT_ROOT / "tmp" / "gui-innovation"

            def worker() -> None:
                try:
                    report = build_innovation_report_from_payload(sequence_payload, source_hint=source_hint)
                    write_innovation_artifacts(report, output_dir)
                except Exception as exc:  # noqa: BLE001 - display evidence generation failures in the tab
                    self.result_queue.put(("innovation-error", self.frame_token, (window, run_button, status, str(exc))))
                    return
                self.result_queue.put(("innovation-artifacts", self.frame_token, (window, render, run_button, status, output_dir)))

            threading.Thread(target=worker, daemon=True).start()

        def run_source_audit() -> None:
            source_audit_button.config(state="disabled", text="审计中…")
            status.config(text="读取原图并抽取时序小窗")
            output_dir = PROJECT_ROOT / "tmp" / "gui-sequence-source-audit"

            def worker() -> None:
                try:
                    output = write_sequence_source_audit(result, output_dir)
                except Exception as exc:  # noqa: BLE001 - worker returns a visible audit error
                    self.result_queue.put(("sequence-source-audit-error", self.frame_token, (window, source_audit_button, status, str(exc))))
                    return
                self.result_queue.put(("sequence-source-audit", self.frame_token, (window, source_audit_button, status, output)))

            threading.Thread(target=worker, daemon=True).start()

        run_button.config(command=run)
        source_audit_button.config(command=run_source_audit)

    def _build_telemetry_table(self, parent: tk.Misc, result: SequenceResult) -> None:
        """显示辅助字段向量摘要，并把单位线索与自洽检查边界写在表头下。"""

        telemetry_rows = [
            {"timestamp": frame.timestamp, **frame.auxiliary_dict}
            for frame in result.frames
        ]
        consistency = {
            prefix: _telemetry_consistency(telemetry_rows, prefix)
            for prefix in ("j2000", "wgs84")
        }
        consistency_parts: list[str] = []
        for prefix, label in (("j2000", "J2000"), ("wgs84", "WGS84")):
            summary = consistency[prefix]
            rate_median = summary.get("position_rate_norm_m_per_s_median")
            median = summary.get("interior_relative_error_assuming_m_per_s_median")
            if rate_median is not None and median is not None:
                consistency_parts.append(
                    f"{label} Δr/Δt |v| {float(rate_median):.2f} m/s，中间帧相对差 {float(median) * 100:.3f}%"
                )
            elif median is not None:
                consistency_parts.append(f"{label} 中间帧相对差 {float(median) * 100:.3f}%")
        consistency_text = "；".join(consistency_parts) if consistency_parts else "暂无足够的有效位置/时间/速度字段"

        note = self._label(
            parent,
            "FITS 辅助字段摘要 · 位置按格式说明为 m；速度字段表格原文写 m，以下按 m/s? 展示并用 Δr/Δt 做自洽检查（待主办方确认）。 "
            f"{consistency_text}。这不是图像平面运动目标的速度。",
            color=INK_SOFT,
            size=9,
            bg=PAPER_LIGHT,
            anchor="w",
            justify="left",
            wraplength=1050,
        )
        note.pack(fill="x", padx=14, pady=(14, 4))
        frame_times = self._sequence_datetimes(result)
        interval_values = [
            (frame_times[index] - frame_times[index - 1]).total_seconds()
            for index in range(1, len(frame_times))
            if frame_times[index] is not None
            and frame_times[index - 1] is not None
            and (frame_times[index] - frame_times[index - 1]).total_seconds() > 0
        ]
        prediction_horizon_s = float(np.median(interval_values) * 5.0) if interval_values else None
        prediction_parts: list[str] = []
        for prefix, label in (("j2000", "J2000"), ("wgs84", "WGS84")):
            prediction = _telemetry_prediction(telemetry_rows, prefix, prediction_horizon_s)
            coordinates = prediction.get("predicted_position_m_assuming_m_per_s")
            if prediction.get("status") == "available" and isinstance(coordinates, list) and len(coordinates) == 3:
                coordinate_text = ", ".join(f"{float(value):.1f}" for value in coordinates)
                prediction_parts.append(f"{label} +{float(prediction_horizon_s):.3f}s → ({coordinate_text}) m?")
        prediction_text = "；".join(prediction_parts) if prediction_parts else "暂无足够的末帧位置/速度字段生成预测"
        prediction_note = self._mono_label(
            parent,
            f"短期遥测外推：{prediction_text} · 末帧常速度基线，不是轨道传播或位置保证。",
            color=AMBER,
            size=8,
            bg=PAPER_LIGHT,
            anchor="w",
            justify="left",
            wraplength=1050,
        )
        prediction_note.pack(fill="x", padx=14, pady=(0, 8))
        columns = ("frame", "time", "jpos", "jdelta", "jvel", "jaz", "jel", "wpos", "wdelta", "wvel", "waz", "wel")
        tree = ttk.Treeview(parent, columns=columns, show="headings")
        specs = (
            ("frame", "FRAME", 70),
            ("time", "DATE-OBS", 170),
            ("jpos", "J2000 |r| / m", 110),
            ("jdelta", "J2000 Δr / m", 115),
            ("jvel", "J2000 |v| / m/s?", 125),
            ("jaz", "J2000 AZ / °", 100),
            ("jel", "J2000 EL / °", 100),
            ("wpos", "WGS84 |r| / m", 110),
            ("wdelta", "WGS84 Δr / m", 115),
            ("wvel", "WGS84 |v| / m/s?", 125),
            ("waz", "WGS84 AZ / °", 100),
            ("wel", "WGS84 EL / °", 100),
        )
        for column, title, width in specs:
            tree.heading(column, text=title)
            tree.column(column, width=width, anchor="w", stretch=column == "time")
        scroll = ttk.Scrollbar(parent, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=scroll.set)
        tree.pack(side="left", fill="both", expand=True, padx=(14, 0), pady=(0, 14))
        scroll.pack(side="right", fill="y", padx=(0, 14), pady=(0, 14))

        first_auxiliary = result.frames[0].auxiliary_dict if result.frames else {}
        for frame in result.frames:
            auxiliary = frame.auxiliary_dict
            values: list[str] = [f"{frame.frame_index + 1:02d}", frame.timestamp or "—"]
            for prefix in ("j2000", "wgs84"):
                vector = np.asarray([auxiliary.get(f"{prefix}_{axis}", np.nan) for axis in ("x", "y", "z")], dtype=np.float64)
                velocity = np.asarray([auxiliary.get(f"{prefix}_{axis}v", np.nan) for axis in ("x", "y", "z")], dtype=np.float64)
                if np.all(np.isfinite(vector)):
                    values.append(f"{float(np.linalg.norm(vector)):.2f}")
                    first_vector = np.asarray([first_auxiliary.get(f"{prefix}_{axis}", np.nan) for axis in ("x", "y", "z")], dtype=np.float64)
                    values.append(f"{float(np.linalg.norm(vector - first_vector)):.2f}" if np.all(np.isfinite(first_vector)) else "—")
                else:
                    values.extend(("—", "—"))
                if np.all(np.isfinite(velocity)):
                    values.extend(
                        (
                            f"{float(np.linalg.norm(velocity)):.2f}",
                            f"{float(np.degrees(np.arctan2(velocity[1], velocity[0]))):.2f}",
                            f"{float(np.degrees(np.arctan2(velocity[2], np.hypot(velocity[0], velocity[1])))):.2f}",
                        )
                    )
                else:
                    values.extend(("—", "—", "—"))
            tree.insert("", tk.END, values=tuple(values))

    def _build_telemetry_position_tab(self, parent: tk.Misc, result: SequenceResult) -> None:
        """显示辅助遥测的相对位置轨迹和末帧常速度外推。"""

        header = tk.Frame(parent, bg=PAPER_LIGHT)
        header.pack(fill="x", padx=14, pady=(14, 4))
        self._label(
            header,
            "遥测轨迹 · 相对首帧的 XY 投影；末帧外推只作为短期常速度基线，不是轨道传播",
            color=INK,
            size=10,
            bg=PAPER_LIGHT,
            anchor="w",
        ).pack(side="left", fill="x", expand=True)
        self._mono_label(
            header,
            "J2000 / WGS84 · m?",
            color=AMBER,
            size=8,
            bg=PAPER_LIGHT,
        ).pack(side="right")
        note = self._label(
            parent,
            "每个面板都以第 1 帧为原点，避免把两个参考系的绝对坐标硬叠在一起；洋红/青色为观测序列，琥珀色为末帧向后 5 个中位帧间隔的外推。坐标单位沿用格式说明的 m，速度单位仍以 m/s? 标注，待主办方确认。",
            color=INK_SOFT,
            size=9,
            bg=PAPER_LIGHT,
            anchor="w",
            justify="left",
            wraplength=1050,
        )
        note.pack(fill="x", padx=14, pady=(0, 6))
        canvas = tk.Canvas(parent, bg=PAPER_LIGHT, highlightthickness=0, height=470)
        canvas.pack(fill="both", expand=True, padx=14, pady=(0, 14))
        canvas.bind("<Configure>", lambda _event: self._draw_telemetry_position_chart(canvas, result))
        self._draw_telemetry_position_chart(canvas, result)

    def _draw_telemetry_position_chart(self, canvas: tk.Canvas, result: SequenceResult) -> None:
        """在 Tk Canvas 中画 J2000/WGS84 相对位置轨迹，保持 XY 等比例。"""

        if not canvas.winfo_exists():
            return
        width = max(700, canvas.winfo_width())
        height = max(440, canvas.winfo_height())
        canvas.delete("all")
        canvas.create_text(
            20,
            14,
            text="TELEMETRY TRAJECTORY · RELATIVE XY / CONSTANT-VELOCITY FORECAST",
            anchor="nw",
            fill=INK_SOFT,
            font=(MONO, 9),
        )
        rows = [
            {"timestamp": frame.timestamp, **frame.auxiliary_dict}
            for frame in result.frames
        ]
        datetimes = self._sequence_datetimes(result)
        intervals = [
            (datetimes[index] - datetimes[index - 1]).total_seconds()
            for index in range(1, len(datetimes))
            if datetimes[index] is not None
            and datetimes[index - 1] is not None
            and (datetimes[index] - datetimes[index - 1]).total_seconds() > 0
        ]
        horizon_s = float(np.median(intervals) * 5.0) if intervals else None
        predictions = {
            prefix: _telemetry_prediction(rows, prefix, horizon_s)
            for prefix in ("j2000", "wgs84")
        }
        panel_top = 48
        panel_bottom = height - 32
        panel_gap = 24
        panel_left = 42
        panel_width = max(260.0, (width - panel_left - 24 - panel_gap) / 2.0)

        for panel_index, (prefix, label, color) in enumerate(
            (("j2000", "J2000", MOTION_TRAIL), ("wgs84", "WGS84", SKY))
        ):
            left = panel_left + panel_index * (panel_width + panel_gap)
            right = left + panel_width
            points: list[tuple[float, float]] = []
            for row in rows:
                try:
                    point = (float(row[f"{prefix}_x"]), float(row[f"{prefix}_y"]))
                except (KeyError, TypeError, ValueError):
                    continue
                if all(np.isfinite(point)):
                    points.append(point)
            canvas.create_rectangle(left, panel_top, right, panel_bottom, outline=PAPER_LINE, fill=PAPER)
            canvas.create_text(
                left + 10,
                panel_top + 9,
                text=f"{label} · ΔX / ΔY from F01",
                anchor="nw",
                fill=INK,
                font=(MONO, 8, "bold"),
            )
            if not points:
                canvas.create_text(left + 12, panel_top + 42, text="no valid position", anchor="nw", fill=INK_SOFT, font=(MONO, 8))
                continue
            origin = np.asarray(points[0], dtype=np.float64)
            observed = [tuple(float(value) for value in (np.asarray(point) - origin)) for point in points]
            prediction_values = predictions[prefix].get("predicted_position_m_assuming_m_per_s")
            predicted: tuple[float, float] | None = None
            if isinstance(prediction_values, list) and len(prediction_values) >= 2:
                predicted_array = np.asarray([float(prediction_values[0]), float(prediction_values[1])], dtype=np.float64) - origin
                if np.all(np.isfinite(predicted_array)):
                    predicted = (float(predicted_array[0]), float(predicted_array[1]))
            plotted = observed + ([predicted] if predicted is not None else [])
            x_values = [point[0] for point in plotted] + [0.0]
            y_values = [point[1] for point in plotted] + [0.0]
            x_min, x_max = min(x_values), max(x_values)
            y_min, y_max = min(y_values), max(y_values)
            x_span = max(x_max - x_min, 1.0)
            y_span = max(y_max - y_min, 1.0)
            plot_left, plot_right = left + 28, right - 14
            plot_top, plot_bottom = panel_top + 38, panel_bottom - 28
            scale = min((plot_right - plot_left) / x_span, (plot_bottom - plot_top) / y_span)
            origin_screen_x = (plot_left + plot_right) / 2.0 - ((x_min + x_max) / 2.0) * scale
            origin_screen_y = (plot_top + plot_bottom) / 2.0 + ((y_min + y_max) / 2.0) * scale

            def project(point: tuple[float, float]) -> tuple[float, float]:
                return origin_screen_x + point[0] * scale, origin_screen_y - point[1] * scale

            zero_x, zero_y = project((0.0, 0.0))
            if plot_left <= zero_x <= plot_right:
                canvas.create_line(zero_x, plot_top, zero_x, plot_bottom, fill=PAPER_LINE, dash=(3, 3))
            if plot_top <= zero_y <= plot_bottom:
                canvas.create_line(plot_left, zero_y, plot_right, zero_y, fill=PAPER_LINE, dash=(3, 3))
            observed_screen = [project(point) for point in observed]
            if len(observed_screen) >= 2:
                canvas.create_line(*observed_screen, fill=color, width=2)
            for index, point in enumerate(observed_screen):
                canvas.create_oval(point[0] - 3, point[1] - 3, point[0] + 3, point[1] + 3, fill=color, outline=color)
                if index == 0:
                    canvas.create_text(point[0] + 7, point[1] - 7, text="F01", anchor="sw", fill=INK_SOFT, font=(MONO, 7))
                elif index == len(observed_screen) - 1:
                    canvas.create_text(point[0] + 7, point[1] - 7, text=f"F{len(observed_screen):02d}", anchor="sw", fill=INK_SOFT, font=(MONO, 7))
            if predicted is not None:
                predicted_screen = project(predicted)
                last_screen = observed_screen[-1]
                canvas.create_line(*last_screen, *predicted_screen, fill=AMBER, width=2, dash=(5, 3))
                canvas.create_oval(
                    predicted_screen[0] - 5,
                    predicted_screen[1] - 5,
                    predicted_screen[0] + 5,
                    predicted_screen[1] + 5,
                    outline=AMBER,
                    width=2,
                )
                canvas.create_text(predicted_screen[0] + 7, predicted_screen[1] - 7, text="forecast", anchor="sw", fill=AMBER, font=(MONO, 7))
            canvas.create_text(left + 10, panel_bottom - 16, text="ΔX / m?  ·  ΔY / m?", anchor="w", fill=INK_SOFT, font=(MONO, 7))
        if horizon_s is not None:
            canvas.create_text(width - 18, 16, text=f"forecast +{horizon_s:.3f}s · m/s? assumption", anchor="ne", fill=AMBER, font=(MONO, 8))

    def _close_sequence_evidence(self, window: tk.Toplevel) -> None:
        if window.winfo_exists():
            window.destroy()
        if self.sequence_evidence_window is window:
            self.sequence_evidence_window = None

    @staticmethod
    def _sequence_datetimes(result: SequenceResult) -> list[datetime | None]:
        values: list[datetime | None] = []
        for frame in result.frames:
            if not frame.timestamp:
                values.append(None)
                continue
            try:
                value = datetime.fromisoformat(frame.timestamp.replace("Z", "+00:00"))
            except ValueError:
                values.append(None)
            else:
                values.append(value.replace(tzinfo=None))
        return values

    @staticmethod
    def _track_kinematics(result: SequenceResult, track: MotionFeatureTrack) -> dict[str, Any] | None:
        """使用证据页与主预览共用的真实时间常速度拟合。"""

        if len(track.points) < 2:
            return None
        datetimes = StarfieldApp._sequence_datetimes(result)
        point_times = [datetimes[point.frame_index] if 0 <= point.frame_index < len(datetimes) else None for point in track.points]
        if any(value is None for value in point_times):
            return None
        first_time = point_times[0]
        if first_time is None:
            return None
        intervals = [
            (right - left).total_seconds()
            for left, right in zip(datetimes, datetimes[1:], strict=False)
            if left is not None and right is not None and (right - left).total_seconds() > 0
        ]
        if not intervals:
            return None
        elapsed = [(value - first_time).total_seconds() for value in point_times if value is not None]
        if len(elapsed) != len(track.points) or float(np.ptp(elapsed)) <= 0:
            return None
        return _fit_constant_velocity(
            elapsed,
            [point.aligned_x for point in track.points],
            [point.aligned_y for point in track.points],
            float(np.median(intervals) * 5.0),
        )

    def _draw_evidence_charts(self, canvas: tk.Canvas, result: SequenceResult) -> None:
        if not canvas.winfo_exists():
            return
        width = max(600, canvas.winfo_width())
        height = max(420, canvas.winfo_height())
        canvas.delete("all")
        canvas.create_text(20, 15, text="SEQUENCE PLOTS · COUNTS / REGISTRATION / TELEMETRY", anchor="nw", fill=INK_SOFT, font=(MONO, 9))

        def plot_box(
            top: int,
            bottom: int,
            title: str,
            values: Sequence[float],
            color: str,
            second: tuple[Sequence[float], str, str] | None = None,
            x_bounds: tuple[int, int] | None = None,
            first_label: str = "candidate",
        ) -> None:
            left, right = x_bounds or (65, width - 35)
            canvas.create_rectangle(left, top, right, bottom, outline=PAPER_LINE, fill=PAPER)
            canvas.create_text(left + 10, top + 9, text=title, anchor="nw", fill=INK, font=(MONO, 8, "bold"))
            all_values = [float(value) for value in values]
            if second is not None:
                all_values.extend(float(value) for value in second[0])
            maximum = max(all_values, default=1.0)
            minimum = min(0.0, min(all_values, default=0.0))
            if maximum <= minimum:
                maximum = minimum + 1.0
            plot_top, plot_bottom = top + 34, bottom - 28
            canvas.create_line(left, plot_top, left, plot_bottom, fill=INK_SOFT)
            canvas.create_line(left, plot_bottom, right, plot_bottom, fill=INK_SOFT)
            canvas.create_text(left - 8, plot_top, text=f"{maximum:.1f}", anchor="e", fill=INK_SOFT, font=(MONO, 7))
            canvas.create_text(left - 8, plot_bottom, text=f"{minimum:.1f}", anchor="e", fill=INK_SOFT, font=(MONO, 7))
            x_values = list(range(len(values)))
            x_span = max(1, len(values) - 1)

            def points_for(series: Sequence[float]) -> list[tuple[float, float]]:
                return [
                    (left + index / x_span * (right - left), plot_bottom - (float(value) - minimum) / (maximum - minimum) * (plot_bottom - plot_top))
                    for index, value in enumerate(series)
                ]

            first_points = points_for(values)
            if len(first_points) >= 2:
                canvas.create_line(*first_points, fill=color, width=2)
            for point in first_points:
                canvas.create_oval(point[0] - 2, point[1] - 2, point[0] + 2, point[1] + 2, fill=color, outline=color)
            canvas.create_text(right - 10, top + 10, text=first_label, anchor="ne", fill=color, font=(MONO, 8))
            if second is not None:
                second_points = points_for(second[0])
                if len(second_points) >= 2:
                    canvas.create_line(*second_points, fill=second[1], width=2)
                for point in second_points:
                    canvas.create_oval(point[0] - 2, point[1] - 2, point[0] + 2, point[1] + 2, fill=second[1], outline=second[1])
                canvas.create_text(right - 10, top + 25, text=second[2], anchor="ne", fill=second[1], font=(MONO, 8))
            canvas.create_text((left + right) / 2, bottom - 16, text="frame", fill=INK_SOFT, font=(MONO, 7))

        candidate_values = [float(frame.candidate_count) for frame in result.frames]
        quality_values = [float(frame.quality_count) for frame in result.frames]
        top_bottom = min(235, height // 2 - 12)
        plot_box(38, top_bottom, "SOURCE COUNTS", candidate_values, AMBER, (quality_values, MINT, "quality"))
        shift_values = [float(np.hypot(*result.cumulative_shifts[index])) if index < len(result.cumulative_shifts) else 0.0 for index in range(len(result.frames))]
        second_top = min(275, height // 2 + 12)
        split = max(350, int(width * 0.52))
        left_bounds = (65, min(split, width - 260))
        right_bounds = (max(left_bounds[1] + 42, int(width * 0.57)), width - 35)
        plot_box(second_top, height - 35, "CUMULATIVE SHIFT / PX", shift_values, SKY, x_bounds=left_bounds, first_label="shift")
        telemetry_values: dict[str, list[float]] = {"j2000": [], "wgs84": []}
        for frame in result.frames:
            auxiliary = frame.auxiliary_dict
            for prefix in telemetry_values:
                vector = np.asarray([auxiliary.get(f"{prefix}_{axis}v", np.nan) for axis in ("x", "y", "z")], dtype=np.float64)
                telemetry_values[prefix].append(float(np.linalg.norm(vector)) if np.all(np.isfinite(vector)) else 0.0)
        plot_box(
            second_top,
            height - 35,
            "AUXILIARY |v| / m/s?",
            telemetry_values["j2000"],
            MOTION_TRAIL,
            (telemetry_values["wgs84"], SKY, "WGS84"),
            x_bounds=right_bounds,
            first_label="J2000",
        )

    def _build_injection_tab(self, parent: tk.Misc, window: tk.Toplevel, result: SequenceResult) -> None:
        """按需运行理想背景或真实 FITS 背景的注入-回收实验。"""

        header = tk.Frame(parent, bg=PAPER_LIGHT)
        header.pack(fill="x", padx=14, pady=(14, 5))
        note = self._label(
            header,
            "可切换理想背景与真实 FITS 背景；真实背景中的已有检测不冒充误检率",
            color=INK_SOFT,
            size=9,
            bg=PAPER_LIGHT,
            anchor="w",
        )
        note.pack(side="left", fill="x", expand=True)
        status = self._mono_label(header, "未运行", color=INK_SOFT, size=8, bg=PAPER_LIGHT)
        status.pack(side="right", padx=(8, 0))
        run_button = tk.Button(
            header,
            text="理想背景",
            bg=NAVY,
            fg=WHITE,
            activebackground=NAVY_SOFT,
            activeforeground=WHITE,
            relief="flat",
            bd=0,
            padx=11,
            pady=6,
            font=(SANS, 9, "bold"),
        )
        run_button.pack(side="right", padx=(8, 0))
        real_button = tk.Button(
            header,
            text="真实 FITS 背景",
            bg=AMBER,
            fg=NAVY_DARK,
            activebackground=AMBER_LIGHT,
            activeforeground=NAVY_DARK,
            relief="flat",
            bd=0,
            padx=11,
            pady=6,
            font=(SANS, 9, "bold"),
        )
        real_button.pack(side="right", padx=(8, 0))

        columns = (
            "peak",
            "injected",
            "candidate",
            "quality",
            "candidate_recall",
            "quality_recall",
            "net_candidate",
            "net_quality",
        )
        tree = ttk.Treeview(parent, columns=columns, show="headings", height=8)
        specs = (
            ("peak", "PEAK EXCESS / ADU", 150),
            ("injected", "INJECTED", 90),
            ("candidate", "CANDIDATE TP", 110),
            ("quality", "QUALITY TP", 100),
            ("candidate_recall", "CANDIDATE RECALL", 150),
            ("quality_recall", "QUALITY RECALL", 140),
            ("net_candidate", "NET CANDIDATE Δ", 140),
            ("net_quality", "NET QUALITY Δ", 130),
        )
        for column, title, width in specs:
            tree.heading(column, text=title)
            tree.column(column, width=width, anchor="w", stretch=column in {"candidate_recall", "quality_recall"})
        tree.pack(fill="x", padx=14, pady=(0, 8))
        chart = tk.Canvas(parent, bg=PAPER_LIGHT, highlightthickness=0, height=300)
        chart.pack(fill="both", expand=True, padx=14, pady=(0, 8))
        footer = self._label(
            parent,
            "真实背景结果中的背景源数只是注入保护区外的检测数；没有逐星真值时，不把它写成 precision 或误检率。",
            color=INK_SOFT,
            size=8,
            bg=PAPER_LIGHT,
            anchor="w",
            justify="left",
            wraplength=1050,
        )
        footer.pack(fill="x", padx=14, pady=(0, 14))

        def render(
            rows: tuple[InjectionRecoveryRow, ...] | tuple[RealBackgroundInjectionRow, ...],
            output: Path,
            mode: str,
        ) -> None:
            try:
                exists = bool(window.winfo_exists())
            except tk.TclError:
                exists = False
            if not exists:
                return
            for item in tree.get_children():
                tree.delete(item)
            for row in rows:
                if isinstance(row, RealBackgroundInjectionRow):
                    net_candidate = f"{row.net_candidate_delta:+.1f}"
                    net_quality = f"{row.net_quality_delta:+.1f}"
                else:
                    net_candidate = "—"
                    net_quality = "—"
                tree.insert("", tk.END, values=(
                    f"{row.peak_excess_adu:g}",
                    row.injected_count,
                    row.candidate_recovered_count,
                    row.quality_recovered_count,
                    f"{row.candidate_recall:.3f}",
                    f"{row.quality_recall:.3f}",
                    net_candidate,
                    net_quality,
                ))
            self._draw_injection_chart(chart, rows)
            psf_note = ""
            if rows and isinstance(rows[0], RealBackgroundInjectionRow):
                first_row = rows[0]
                psf_note = (
                    f" · PSF={first_row.psf_model}, 样本={first_row.psf_source_count}, "
                    f"中位 FWHM={first_row.psf_median_fwhm_px:.3f} px"
                    if first_row.psf_median_fwhm_px is not None
                    else f" · PSF={first_row.psf_model}, 样本={first_row.psf_source_count}"
                )
            footer.config(
                text=(
                    f"{mode} · 输出：{output} · "
                    f"召回率按注入真值计算{psf_note}；候选/质量数量仍不能直接等同于真实恒星数"
                )
            )

        buttons = (run_button, real_button)

        def run(mode: str) -> None:
            for button in buttons:
                button.config(state="disabled")
            status.config(text=f"{mode}后台计算")

            def worker() -> None:
                try:
                    if mode == "真实 FITS 背景":
                        if not result.frames:
                            raise ValueError("当前序列没有可用帧")
                        frame_path = Path(result.frames[0].path)
                        if not frame_path.is_file():
                            frame_path = PROJECT_ROOT / frame_path
                        if not frame_path.is_file():
                            raise FileNotFoundError(f"找不到真实首帧 FITS：{frame_path}")
                        rows = run_real_background_injection(
                            frame_path,
                            trials_per_level=1,
                            sources_per_trial=12,
                            min_distance=4,
                            psf_fwhm=3.0,
                            min_flux_snr=5.0,
                            psf_model="empirical",
                            progress=lambda index, total: self.result_queue.put(
                                (
                                    "injection-progress",
                                    self.frame_token,
                                    (window, status, f"真实背景实验 · 强度档位 {index}/{total}"),
                                )
                            ),
                        )
                        output = write_real_background_injection_artifacts(
                            rows,
                            PROJECT_ROOT / "tmp" / "gui-real-background-injection",
                        )
                    else:
                        rows = run_injection_recovery()
                        output = write_injection_artifacts(
                            rows,
                            PROJECT_ROOT / "tmp" / "gui-injection",
                        )
                except Exception as exc:  # noqa: BLE001 - display experiment failures in the tab
                    self.result_queue.put(("injection-error", self.frame_token, (window, buttons, status, str(exc))))
                    return
                self.result_queue.put(("injection", self.frame_token, (window, render, buttons, status, rows, output, mode)))

            threading.Thread(target=worker, daemon=True).start()

        run_button.config(command=lambda: run("理想背景"))
        real_button.config(command=lambda: run("真实 FITS 背景"))

    def _build_detection_sweep_tab(self, parent: tk.Misc, window: tk.Toplevel, result: SequenceResult) -> None:
        """按需在真实序列首帧上扫描检测阈值，并展示数量敏感性。"""

        header = tk.Frame(parent, bg=PAPER_LIGHT)
        header.pack(fill="x", padx=14, pady=(14, 5))
        note = self._label(
            header,
            "真实序列第 1 帧 · 可分别扫描检测阈值和匹配滤波 PSF；候选峰、质量源和线状审计分开统计",
            color=INK_SOFT,
            size=9,
            bg=PAPER_LIGHT,
            anchor="w",
        )
        note.pack(side="left", fill="x", expand=True)
        status = self._mono_label(header, "未运行", color=INK_SOFT, size=8, bg=PAPER_LIGHT)
        status.pack(side="right", padx=(8, 0))
        run_button = tk.Button(
            header,
            text="运行扫描",
            bg=NAVY,
            fg=WHITE,
            activebackground=NAVY_SOFT,
            activeforeground=WHITE,
            relief="flat",
            bd=0,
            padx=11,
            pady=6,
            font=(SANS, 9, "bold"),
        )
        run_button.pack(side="right", padx=(8, 0))
        psf_button = tk.Button(
            header,
            text="PSF 扫描",
            bg=AMBER,
            fg=NAVY_DARK,
            activebackground=AMBER_LIGHT,
            activeforeground=NAVY_DARK,
            relief="flat",
            bd=0,
            padx=11,
            pady=6,
            font=(SANS, 9, "bold"),
        )
        psf_button.pack(side="right", padx=(8, 0))

        columns = ("threshold", "candidate", "returned", "quality", "rejected", "line", "noise")
        tree = ttk.Treeview(parent, columns=columns, show="headings", height=8)
        specs = (
            ("threshold", "THRESHOLD / σ OR FWHM / px", 190),
            ("candidate", "CANDIDATE", 110),
            ("returned", "RETURNED", 110),
            ("quality", "QUALITY", 110),
            ("rejected", "REJECTED", 110),
            ("line", "LINE FLAG", 100),
            ("noise", "NOISE / ADU", 110),
        )
        for column, title, width in specs:
            tree.heading(column, text=title)
            tree.column(column, width=width, anchor="w", stretch=column in {"candidate", "quality"})
        tree.pack(fill="x", padx=14, pady=(0, 8))
        chart = tk.Canvas(parent, bg=PAPER_LIGHT, highlightthickness=0, height=300)
        chart.pack(fill="both", expand=True, padx=14, pady=(0, 8))
        footer = self._label(
            parent,
            "阈值升高通常会减少候选尾部，但不能由数量接近外部示例就认定为真实恒星数；需结合注入回收、星表匹配和人工抽检。",
            color=INK_SOFT,
            size=8,
            bg=PAPER_LIGHT,
            anchor="w",
            justify="left",
            wraplength=900,
        )
        footer.pack(fill="x", padx=14, pady=(0, 14))

        def render(rows: tuple[DetectionSweepRow, ...], output: Path) -> None:
            try:
                if not window.winfo_exists():
                    return
            except tk.TclError:
                return
            for item in tree.get_children():
                tree.delete(item)
            for row in rows:
                tree.insert(
                    "",
                    tk.END,
                    values=(
                        f"{row.threshold_sigma:g}",
                        f"{row.candidate_count:,}",
                        f"{row.returned_count:,}",
                        f"{row.quality_count:,}",
                        f"{row.rejected_count:,}",
                        f"{row.line_artifact_count:,}",
                        f"{row.noise_adu:.3f}",
                    ),
                )
            self._draw_detection_sweep_chart(chart, rows)
            footer.config(text=f"输出已保存至 {output} · 阈值扫描只改变候选门槛，不改变 FITS 原图")

        def render_psf(rows: tuple[DetectionPSFSweepRow, ...], output: Path) -> None:
            try:
                if not window.winfo_exists():
                    return
            except tk.TclError:
                return
            for item in tree.get_children():
                tree.delete(item)
            for row in rows:
                tree.insert(
                    "",
                    tk.END,
                    values=(
                        f"{row.psf_fwhm_px:g}",
                        f"{row.candidate_count:,}",
                        f"{row.returned_count:,}",
                        f"{row.quality_count:,}",
                        f"{row.rejected_count:,}",
                        f"{row.line_artifact_count:,}",
                        f"{row.noise_adu:.3f}",
                    ),
                )
            self._draw_detection_psf_sweep_chart(chart, rows)
            footer.config(text=f"输出已保存至 {output} · PSF 扫描只改变匹配滤波尺度，不改变 FITS 原图")

        buttons = (run_button, psf_button)

        def run() -> None:
            for button in buttons:
                button.config(state="disabled")
            status.config(text="后台读取真实首帧")
            frame_path = Path(result.frames[0].path) if result.frames else PROJECT_ROOT / "missing.fits"
            if not frame_path.is_file():
                frame_path = PROJECT_ROOT / frame_path
            output_dir = PROJECT_ROOT / "tmp" / "gui-detection-sweep"

            def worker() -> None:
                try:
                    rows = run_detection_threshold_sweep(frame_path)
                    output = write_detection_sweep_artifacts(rows, output_dir)
                except Exception as exc:  # noqa: BLE001 - display experiment failures in the tab
                    self.result_queue.put(("sweep-error", self.frame_token, (window, buttons, status, str(exc))))
                    return
                self.result_queue.put(("sweep", self.frame_token, (window, render, buttons, status, rows, output)))

            threading.Thread(target=worker, daemon=True).start()

        def run_psf() -> None:
            for button in buttons:
                button.config(state="disabled")
            status.config(text="后台扫描真实首帧 PSF FWHM")
            frame_path = Path(result.frames[0].path) if result.frames else PROJECT_ROOT / "missing.fits"
            if not frame_path.is_file():
                frame_path = PROJECT_ROOT / frame_path
            output_dir = PROJECT_ROOT / "tmp" / "gui-detection-psf-sweep"

            def worker() -> None:
                try:
                    rows = run_detection_psf_sweep(frame_path)
                    output = write_detection_psf_sweep_artifacts(rows, output_dir)
                except Exception as exc:  # noqa: BLE001 - display experiment failures in the tab
                    self.result_queue.put(("psf-sweep-error", self.frame_token, (window, buttons, status, str(exc))))
                    return
                self.result_queue.put(("psf-sweep", self.frame_token, (window, render_psf, buttons, status, rows, output)))

            threading.Thread(target=worker, daemon=True).start()

        run_button.config(command=run)
        psf_button.config(command=run_psf)

    @staticmethod
    def _draw_detection_sweep_chart(canvas: tk.Canvas, rows: tuple[DetectionSweepRow, ...]) -> None:
        if not canvas.winfo_exists():
            return
        width = max(600, canvas.winfo_width())
        height = max(260, canvas.winfo_height())
        canvas.delete("all")
        canvas.create_text(18, 14, text="REAL FRAME · CANDIDATE / QUALITY COUNTS", anchor="nw", fill=INK_SOFT, font=(MONO, 9))
        left, top, right, bottom = 76, 42, width - 36, height - 44
        canvas.create_rectangle(left, top, right, bottom, outline=PAPER_LINE, fill=PAPER)
        canvas.create_line(left, top, left, bottom, fill=INK_SOFT)
        canvas.create_line(left, bottom, right, bottom, fill=INK_SOFT)
        if not rows:
            canvas.create_text((left + right) / 2, (top + bottom) / 2, text="点击“运行扫描”生成曲线", fill=INK_SOFT, font=(SANS, 10))
            return
        maximum = max((row.candidate_count for row in rows), default=1)
        maximum = max(1, maximum)
        x_span = max(1, len(rows) - 1)
        canvas.create_text(left - 8, top, text=f"{maximum:,}", anchor="e", fill=INK_SOFT, font=(MONO, 7))
        canvas.create_text(left - 8, bottom, text="0", anchor="e", fill=INK_SOFT, font=(MONO, 7))
        canvas.create_text((left + right) / 2, bottom + 21, text="candidate threshold / sigma", fill=INK_SOFT, font=(MONO, 7))

        def points(values: Sequence[int]) -> list[tuple[float, float]]:
            return [
                (
                    left + index / x_span * (right - left),
                    bottom - float(value) / maximum * (bottom - top),
                )
                for index, value in enumerate(values)
            ]

        for label, color, values in (
            ("candidate", AMBER, [row.candidate_count for row in rows]),
            ("quality", MINT, [row.quality_count for row in rows]),
        ):
            series = points(values)
            if len(series) >= 2:
                canvas.create_line(*series, fill=color, width=2)
            for point in series:
                canvas.create_oval(point[0] - 3, point[1] - 3, point[0] + 3, point[1] + 3, fill=color, outline=color)
            legend_y = top + 10 + (0 if label == "candidate" else 20)
            canvas.create_line(right - 135, legend_y + 5, right - 117, legend_y + 5, fill=color, width=2)
            canvas.create_text(right - 111, legend_y, text=label, anchor="nw", fill=color, font=(MONO, 8))
        for index, row in enumerate(rows):
            x = left + index / x_span * (right - left)
            canvas.create_text(x, bottom + 5, text=f"{row.threshold_sigma:g}", anchor="n", fill=INK_SOFT, font=(MONO, 7))

    @staticmethod
    def _draw_detection_psf_sweep_chart(canvas: tk.Canvas, rows: tuple[DetectionPSFSweepRow, ...]) -> None:
        if not canvas.winfo_exists():
            return
        width = max(600, canvas.winfo_width())
        height = max(260, canvas.winfo_height())
        canvas.delete("all")
        canvas.create_text(18, 14, text="REAL FRAME · PSF FWHM / CANDIDATE / QUALITY", anchor="nw", fill=INK_SOFT, font=(MONO, 9))
        left, top, right, bottom = 76, 42, width - 36, height - 44
        canvas.create_rectangle(left, top, right, bottom, outline=PAPER_LINE, fill=PAPER)
        canvas.create_line(left, top, left, bottom, fill=INK_SOFT)
        canvas.create_line(left, bottom, right, bottom, fill=INK_SOFT)
        if not rows:
            canvas.create_text((left + right) / 2, (top + bottom) / 2, text="点击“PSF 扫描”生成曲线", fill=INK_SOFT, font=(SANS, 10))
            return
        maximum = max((row.candidate_count for row in rows), default=1)
        maximum = max(1, maximum)
        x_span = max(1, len(rows) - 1)
        canvas.create_text(left - 8, top, text=f"{maximum:,}", anchor="e", fill=INK_SOFT, font=(MONO, 7))
        canvas.create_text(left - 8, bottom, text="0", anchor="e", fill=INK_SOFT, font=(MONO, 7))
        canvas.create_text((left + right) / 2, bottom + 21, text="matched-filter PSF FWHM / px", fill=INK_SOFT, font=(MONO, 7))

        def points(values: Sequence[int]) -> list[tuple[float, float]]:
            return [
                (
                    left + index / x_span * (right - left),
                    bottom - float(value) / maximum * (bottom - top),
                )
                for index, value in enumerate(values)
            ]

        for label, color, values in (
            ("candidate", AMBER, [row.candidate_count for row in rows]),
            ("quality", MINT, [row.quality_count for row in rows]),
        ):
            series = points(values)
            if len(series) >= 2:
                canvas.create_line(*series, fill=color, width=2)
            for point in series:
                canvas.create_oval(point[0] - 3, point[1] - 3, point[0] + 3, point[1] + 3, fill=color, outline=color)
            legend_y = top + 10 + (0 if label == "candidate" else 20)
            canvas.create_line(right - 135, legend_y + 5, right - 117, legend_y + 5, fill=color, width=2)
            canvas.create_text(right - 111, legend_y, text=label, anchor="nw", fill=color, font=(MONO, 8))
        for index, row in enumerate(rows):
            x = left + index / x_span * (right - left)
            canvas.create_text(x, bottom + 5, text=f"{row.psf_fwhm_px:g}", anchor="n", fill=INK_SOFT, font=(MONO, 7))

    @staticmethod
    def _draw_injection_chart(canvas: tk.Canvas, rows: tuple[InjectionRecoveryRow, ...]) -> None:
        if not canvas.winfo_exists():
            return
        width = max(600, canvas.winfo_width())
        height = max(260, canvas.winfo_height())
        canvas.delete("all")
        canvas.create_text(18, 14, text="INJECTION RECOVERY · RECALL CURVE", anchor="nw", fill=INK_SOFT, font=(MONO, 9))
        left, top, right, bottom = 70, 42, width - 36, height - 42
        canvas.create_rectangle(left, top, right, bottom, outline=PAPER_LINE, fill=PAPER)
        canvas.create_line(left, top, left, bottom, fill=INK_SOFT)
        canvas.create_line(left, bottom, right, bottom, fill=INK_SOFT)
        if not rows:
            canvas.create_text((left + right) / 2, (top + bottom) / 2, text="点击“运行实验”生成曲线", fill=INK_SOFT, font=(SANS, 10))
            return
        minimum = min(row.peak_excess_adu for row in rows)
        maximum = max(row.peak_excess_adu for row in rows)
        if maximum <= minimum:
            maximum = minimum + 1.0
        x_span = max(1.0, maximum - minimum)
        canvas.create_text(left - 8, top, text="1.0", anchor="e", fill=INK_SOFT, font=(MONO, 7))
        canvas.create_text(left - 8, bottom, text="0.0", anchor="e", fill=INK_SOFT, font=(MONO, 7))
        canvas.create_text((left + right) / 2, bottom + 20, text="injected peak excess / ADU", fill=INK_SOFT, font=(MONO, 7))
        for label, color, values in (
            ("candidate", AMBER, [row.candidate_recall for row in rows]),
            ("quality", MINT, [row.quality_recall for row in rows]),
        ):
            points = []
            for row, value in zip(rows, values, strict=True):
                x = left + (row.peak_excess_adu - minimum) / x_span * (right - left)
                y = bottom - float(value) * (bottom - top)
                points.append((x, y))
            if len(points) >= 2:
                canvas.create_line(*points, fill=color, width=2)
            for point in points:
                canvas.create_oval(point[0] - 3, point[1] - 3, point[0] + 3, point[1] + 3, fill=color, outline=color)
            legend_y = top + 12 + (0 if label == "candidate" else 18)
            canvas.create_line(right - 130, legend_y + 5, right - 112, legend_y + 5, fill=color, width=2)
            canvas.create_text(right - 106, legend_y, text=label, anchor="nw", fill=color, font=(MONO, 8))

    def _draw_preview(self) -> None:
        if self.preview is None or not self.canvas.winfo_exists():
            return
        canvas_width = max(100, self.canvas.winfo_width())
        canvas_height = max(100, self.canvas.winfo_height())
        base_width, base_height = self.preview.size
        fit_scale = min((canvas_width - 14) / base_width, (canvas_height - 14) / base_height)
        scale = max(0.01, fit_scale * self.preview_zoom)
        scaled_width = max(1, round(base_width * scale))
        scaled_height = max(1, round(base_height * scale))
        origin_x = (canvas_width - scaled_width) / 2 + self.preview_pan_x
        origin_y = (canvas_height - scaled_height) / 2 + self.preview_pan_y

        left = max(0, min(base_width - 1, int(max(0, -origin_x) / scale)))
        top = max(0, min(base_height - 1, int(max(0, -origin_y) / scale)))
        right = min(base_width, max(left + 1, int((canvas_width - origin_x) / scale) + 1))
        bottom = min(base_height, max(top + 1, int((canvas_height - origin_y) / scale) + 1))
        image = self.preview.crop((left, top, right, bottom))
        image = image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))), Image.Resampling.LANCZOS)
        draw = ImageDraw.Draw(image)
        if self.preview_shape is not None:
            original_height, original_width = self.preview_shape
            self.preview_scale_x = base_width / original_width
            self.preview_scale_y = base_height / original_height
            mode = self.overlay_mode_var.get()
            if self.analysis is not None and mode in {"quality", "candidates"}:
                sources = self.analysis.detection.quality_sources if mode == "quality" else self.analysis.detection.sources
                # 视场总览时一个传感器像素通常不到 1 px；每个候选画 3×3
                # 方框会把 2--3 万个源铺成一层橙色噪点。缩小时只画 1 px
                # 证据点，放大后再扩成小方框，数量口径不变但不遮住原图。
                marker_radius = min(3, int(round(scale * 0.65)))
                if mode == "candidates":
                    marker_radius = min(1, marker_radius)
                for source in sources:
                    source_x, source_y = source_peak_position(source)
                    point_x = source_x * self.preview_scale_x
                    point_y = source_y * self.preview_scale_y
                    if not (left <= point_x < right and top <= point_y < bottom):
                        continue
                    x = (point_x - left) * scale
                    y = (point_y - top) * scale
                    extra_only = bool(source.proposal_methods) and "gaussian" not in source.proposal_methods
                    color = (DOG_ONLY_POINT if extra_only else STAR_POINT) if source.quality_passed else STAR_POINT_REJECTED
                    center_x = int(round(x))
                    center_y = int(round(y))
                    if marker_radius == 0:
                        draw.point((center_x, center_y), fill=color)
                    else:
                        # 先加一圈深色外缘，放大观察时避免标记融入亮星核心；
                        # 颜色本身仍保持为高亮青/紫红，不改变检测数据。
                        draw.rectangle(
                            (
                                center_x - marker_radius - 1,
                                center_y - marker_radius - 1,
                                center_x + marker_radius + 1,
                                center_y + marker_radius + 1,
                            ),
                            fill=NAVY_DARK,
                        )
                        draw.rectangle(
                            (
                                center_x - marker_radius,
                                center_y - marker_radius,
                                center_x + marker_radius,
                                center_y + marker_radius,
                            ),
                            fill=color,
                        )

            # 序列分析会释放每帧完整 Detection，只保留轨迹中的质量通过标记。
            # 因此序列完成后从轻量 TrackPoint 恢复“可信源”主层，避免切回
            # quality 后出现空图；候选共识/叠加暗星仍不会混进可信源。
            if self.analysis is None and mode == "quality" and self.sequence_result is not None:
                frame_index = self._sequence_frame_index()
                trusted_points = trusted_points_for_frame(self.sequence_result, frame_index) if frame_index is not None else ()
                marker_radius = max(1, min(3, int(round(max(1.0, scale) * 0.65))))
                for _track, point in trusted_points:
                    point_x = point.x * self.preview_scale_x
                    point_y = point.y * self.preview_scale_y
                    if not (left <= point_x < right and top <= point_y < bottom):
                        continue
                    x = (point_x - left) * scale
                    y = (point_y - top) * scale
                    center_x = int(round(x))
                    center_y = int(round(y))
                    if marker_radius == 1:
                        draw.point((center_x, center_y), fill=STAR_POINT)
                    else:
                        draw.ellipse(
                            (
                                center_x - marker_radius - 1,
                                center_y - marker_radius - 1,
                                center_x + marker_radius + 1,
                                center_y + marker_radius + 1,
                            ),
                            outline=NAVY_DARK,
                            width=2,
                        )
                        draw.ellipse(
                            (
                                center_x - marker_radius,
                                center_y - marker_radius,
                                center_x + marker_radius,
                                center_y + marker_radius,
                            ),
                            outline=STAR_POINT,
                            width=max(1, min(2, marker_radius)),
                        )
                        draw.point((center_x, center_y), fill=STAR_POINT)

            if mode == "stable" and self.sequence_result is not None:
                frame_index = self._sequence_frame_index()
                stable_points = stable_points_for_frame(self.sequence_result, frame_index) if frame_index is not None else ()
                # 视场总览时仍保证至少一个实心像素；否则 1.0× 适配显示
                # 中，稳定点只剩一个极细的空心轮廓，肉眼很容易认为没有结果。
                marker_radius = max(1, min(4, int(round(max(1.0, scale) * 0.8))))
                for _track, point in stable_points:
                    point_x = point.x * self.preview_scale_x
                    point_y = point.y * self.preview_scale_y
                    if not (left <= point_x < right and top <= point_y < bottom):
                        continue
                    x = (point_x - left) * scale
                    y = (point_y - top) * scale
                    center_x = int(round(x))
                    center_y = int(round(y))
                    if _track.classification == "static":
                        marker_color = STATIC_STAR_POINT
                    elif _track.evidence_level == "temporal_reference":
                        marker_color = TEMPORAL_STAR_POINT
                    else:
                        marker_color = PERSISTENT_STAR_POINT
                    draw.ellipse(
                        (
                            center_x - marker_radius - 1,
                            center_y - marker_radius - 1,
                            center_x + marker_radius + 1,
                            center_y + marker_radius + 1,
                        ),
                        outline=NAVY_DARK,
                        width=2,
                    )
                    draw.ellipse(
                        (
                            center_x - marker_radius,
                            center_y - marker_radius,
                            center_x + marker_radius,
                            center_y + marker_radius,
                        ),
                        fill=marker_color if marker_radius == 1 else None,
                        outline=marker_color,
                        width=max(1, min(2, marker_radius)),
                    )
                    draw.point((center_x, center_y), fill=marker_color)

            if mode == "stack-faint" and self.sequence_result is not None:
                frame_index = self._sequence_frame_index()
                faint_points = stack_faint_points_for_frame(self.sequence_result, frame_index) if frame_index is not None else ()
                marker_radius = max(1, min(4, int(round(max(1.0, scale) * 0.8))))
                for _track, point in faint_points:
                    point_x = point.x * self.preview_scale_x
                    point_y = point.y * self.preview_scale_y
                    if not (left <= point_x < right and top <= point_y < bottom):
                        continue
                    x = (point_x - left) * scale
                    y = (point_y - top) * scale
                    center_x = int(round(x))
                    center_y = int(round(y))
                    draw.ellipse(
                        (
                            center_x - marker_radius - 1,
                            center_y - marker_radius - 1,
                            center_x + marker_radius + 1,
                            center_y + marker_radius + 1,
                        ),
                        outline=NAVY_DARK,
                        width=2,
                    )
                    draw.ellipse(
                        (
                            center_x - marker_radius,
                            center_y - marker_radius,
                            center_x + marker_radius,
                            center_y + marker_radius,
                        ),
                        fill=STACK_FAINT_POINT if marker_radius == 1 else None,
                        outline=STACK_FAINT_POINT,
                        width=max(1, min(2, marker_radius)),
                    )
                    draw.point((center_x, center_y), fill=STACK_FAINT_POINT)

            # 运动证据默认叠加在可信源主层上；选择“运动候选”时即使用户
            # 关闭复合开关也仍然只显示运动层，保持原有的专门审计视图。
            if mode == "motion" or self._motion_overlay_enabled():
                frame_index = self._sequence_frame_index()
                motion_points = moving_points_for_frame(self.sequence_result, frame_index) if frame_index is not None else ()
                for track, point in motion_points:
                    point_x = point.x * self.preview_scale_x
                    point_y = point.y * self.preview_scale_y
                    if not (left <= point_x < right and top <= point_y < bottom):
                        continue
                    x = (point_x - left) * scale
                    y = (point_y - top) * scale
                    point_color = (
                        POINT_MOTION
                        if track.evidence_level == "fast_point_motion"
                        else SKY_LIGHT
                    )
                    draw.ellipse((x - 7, y - 7, x + 7, y + 7), outline=NAVY_DARK, width=4)
                    draw.ellipse((x - 6, y - 6, x + 6, y + 6), outline=point_color, width=2)
                    draw.point((x, y), fill=point_color)
                    label = "POINT MOV" if track.evidence_level == "fast_point_motion" else "MOV"
                    self._draw_image_label(draw, f"{label} {track.track_id:04d}", x, y, image.size)
                self._draw_motion_trajectory_overlay(draw, frame_index, left, top, scale, image.size)
                feature_points = self._motion_feature_points(frame_index) if frame_index is not None else ()
                for track, point in feature_points:
                    point_x = point.x * self.preview_scale_x
                    point_y = point.y * self.preview_scale_y
                    if not (left <= point_x < right and top <= point_y < bottom):
                        continue
                    x = (point_x - left) * scale
                    y = (point_y - top) * scale
                    half_length = 0.5 * point.length_px * scale
                    angle = np.deg2rad(point.angle_deg)
                    dx = np.cos(angle) * half_length * self.preview_scale_x
                    dy = np.sin(angle) * half_length * self.preview_scale_y
                    color = MOTION_TRAIL if track.classification == "moving" else MOTION_CANDIDATE
                    line_width = max(2, int(round(point.width_px * scale)))
                    if track.classification == "moving":
                        draw.line((x - dx, y - dy, x + dx, y + dy), fill=NAVY_DARK, width=line_width + 4)
                    draw.line((x - dx, y - dy, x + dx, y + dy), fill=color, width=line_width)
                    draw.ellipse((x - 6, y - 6, x + 6, y + 6), outline=color, width=2)
                    label = f"TRAIL {track.track_id:04d}" if track.classification == "moving" else f"TRAIL? {track.track_id:04d}"
                    self._draw_image_label(draw, label, x, y, image.size)

            if mode == "catalog" and self.catalog_match_result is not None:
                for match in self.catalog_match_result.matches:
                    detected_x = match.detection_x * self.preview_scale_x
                    detected_y = match.detection_y * self.preview_scale_y
                    predicted_x = match.predicted_x * self.preview_scale_x
                    predicted_y = match.predicted_y * self.preview_scale_y
                    if not (
                        (left <= detected_x < right or left <= predicted_x < right)
                        and (top <= detected_y < bottom or top <= predicted_y < bottom)
                    ):
                        continue
                    dx = (detected_x - left) * scale
                    dy = (detected_y - top) * scale
                    px = (predicted_x - left) * scale
                    py = (predicted_y - top) * scale
                    draw.line((dx, dy, px, py), fill="#9bc9a7", width=1)
                    draw.ellipse((px - 7, py - 7, px + 7, py + 7), outline=MINT, width=2)
                    draw.point((dx, dy), fill=STAR_POINT)
                    if self.hover_catalog_match is match:
                        draw.ellipse((dx - 10, dy - 10, dx + 10, dy + 10), outline="#f3dfac", width=2)
                        self._draw_image_label(draw, f"CAT {match.source_id} · {match.residual_px:.2f}px", dx, dy, image.size)

            if self.analysis is not None and mode == "quality" and self.analysis.faintest is not None:
                faintest = self.analysis.faintest
                point_x, point_y = source_peak_position(faintest)
                point_x *= self.preview_scale_x
                point_y *= self.preview_scale_y
                if left <= point_x < right and top <= point_y < bottom:
                    x = (point_x - left) * scale
                    y = (point_y - top) * scale
                    draw.ellipse((x - 8, y - 8, x + 8, y + 8), outline=MINT, width=2)
                    self._draw_image_label(draw, f"FNT {faintest.detection_id:04d} · m_inst {faintest.instrumental_magnitude:.2f}", x, y, image.size)

            if self.analysis is not None and mode in {"quality", "candidates"} and self.hover_source is not None:
                source = self.hover_source
                point_x, point_y = source_peak_position(source)
                point_x *= self.preview_scale_x
                point_y *= self.preview_scale_y
                if left <= point_x < right and top <= point_y < bottom:
                    x = (point_x - left) * scale
                    y = (point_y - top) * scale
                    draw.ellipse((x - 9, y - 9, x + 9, y + 9), outline="#f3dfac", width=2)
                    magnitude = instrumental_magnitude(source.flux, exposure_s=self.exposure_s)
                    magnitude_text = f"{magnitude:.2f}" if magnitude is not None else "—"
                    self._draw_image_label(
                        draw,
                        self._source_overlay_label(source, magnitude_text=magnitude_text),
                        x,
                        y,
                        image.size,
                    )
            if mode == "quality" and self.hover_trusted_track is not None:
                _track, trusted_point = self.hover_trusted_track
                point_x = trusted_point.x * self.preview_scale_x
                point_y = trusted_point.y * self.preview_scale_y
                if left <= point_x < right and top <= point_y < bottom:
                    x = (point_x - left) * scale
                    y = (point_y - top) * scale
                    draw.ellipse((x - 10, y - 10, x + 10, y + 10), outline=AMBER_LIGHT, width=2)
                    snr_text = f"{trusted_point.flux_snr:.1f}" if trusted_point.flux_snr is not None else "-"
                    self._draw_image_label(draw, f"SRC {_track.track_id:04d} · FSNR {snr_text}", x, y, image.size)
            if (mode in {"motion", "stable", "stack-faint"} or self._motion_overlay_enabled()) and self.hover_motion_track is not None:
                track, point = self.hover_motion_track
                point_x = point.x * self.preview_scale_x
                point_y = point.y * self.preview_scale_y
                if left <= point_x < right and top <= point_y < bottom:
                    x = (point_x - left) * scale
                    y = (point_y - top) * scale
                    draw.ellipse((x - 10, y - 10, x + 10, y + 10), outline="#f3dfac", width=2)
                    if isinstance(point, MotionFeaturePoint):
                        self._draw_image_label(draw, f"TRAIL {track.track_id:04d} · {point.residual_snr:.1f}σ", x, y, image.size)
                    elif mode == "stack-faint":
                        self._draw_image_label(
                            draw,
                            f"FAINT {track.track_id:04d} · {track.presence}/{len(self.sequence_result.frames) if self.sequence_result is not None else '?'}F",
                            x,
                            y,
                            image.size,
                        )
                    else:
                        stable_label = (
                            "STATIC"
                            if track.classification == "static"
                            else "TEMP"
                            if track.evidence_level == "temporal_reference"
                            else "PERSIST"
                        )
                        label = (
                            f"{stable_label} {track.track_id:04d} · "
                            f"{track.presence}/{len(self.sequence_result.frames) if self.sequence_result is not None else '?'}F"
                            if mode == "stable"
                            else f"MOV {track.track_id:04d} · {track.displacement_px:.2f}px"
                        )
                        self._draw_image_label(draw, label, x, y, image.size)
            if mode == "catalog" and self.hover_catalog_match is not None:
                match = self.hover_catalog_match
                point_x = match.detection_x * self.preview_scale_x
                point_y = match.detection_y * self.preview_scale_y
                if left <= point_x < right and top <= point_y < bottom:
                    x = (point_x - left) * scale
                    y = (point_y - top) * scale
                    draw.ellipse((x - 10, y - 10, x + 10, y + 10), outline="#f3dfac", width=2)
                    self._draw_image_label(draw, f"CAT {match.source_id} · {match.residual_px:.2f}px", x, y, image.size)

        screen_x = origin_x + left * scale
        screen_y = origin_y + top * scale
        self.preview_photo = ImageTk.PhotoImage(image)
        self.canvas.delete("all")
        self.canvas.create_image(screen_x, screen_y, image=self.preview_photo, anchor="nw")

    def _draw_motion_trajectory_overlay(
        self,
        draw: ImageDraw.ImageDraw,
        frame_index: int | None,
        left: int,
        top: int,
        scale: float,
        image_size: tuple[int, int],
    ) -> None:
        """绘制已观测路径和末帧五帧外推，不把外推线当成检测结果。"""

        if self.sequence_result is None or frame_index is None:
            return
        if frame_index >= len(self.sequence_result.cumulative_shifts):
            return
        scale_x = self.preview_scale_x
        scale_y = self.preview_scale_y

        def to_image(x: float, y: float, source_frame_index: int) -> tuple[float, float]:
            if source_frame_index >= len(self.sequence_result.cumulative_shifts):
                return (-1.0, -1.0)
            shift_x, shift_y = self.sequence_result.cumulative_shifts[source_frame_index]
            raw_x = (x + shift_x) * scale_x
            raw_y = (y + shift_y) * scale_y
            return (raw_x - left) * scale, (raw_y - top) * scale

        # 点状 moving 轨迹与线状 streak 是两类证据。点目标每帧只占一个
        # 小 PSF，之前这里只画当前帧圆圈，没有把历史点连起来，所以用户
        # 看到的是“已标记但没有线”。现在用常速度关联得到的观测点画实线；
        # 金橙色专门表示高速点源，洋红色仍保留给线状候选。
        for track in self.sequence_result.tracks:
            if track.classification != "moving" or len(track.points) < 2:
                continue
            observed = [
                to_image(point.aligned_x, point.aligned_y, point.frame_index)
                for point in track.points
            ]
            point_color = (
                POINT_MOTION
                if track.evidence_level == "fast_point_motion"
                else SKY_LIGHT
            )
            line_width = max(2, min(6, int(round(2.0 * scale))))
            draw.line(observed, fill=NAVY_DARK, width=line_width + 4, joint="curve")
            draw.line(observed, fill=point_color, width=line_width, joint="curve")
            radius = max(2, min(5, int(round(2.5 * scale))))
            for point_x, point_y in observed:
                draw.ellipse(
                    (point_x - radius, point_y - radius, point_x + radius, point_y + radius),
                    fill=point_color,
                    outline=NAVY_DARK,
                    width=1,
                )

        for track in self.sequence_result.motion_features:
            if track.classification != "moving" or len(track.points) < 2:
                continue
            # 每个观测点必须使用自己所在帧的 shift；不能把当前查看帧的
            # shift 套到整条历史轨迹上，否则较大视场抖动时路径会被错误扭移。
            observed = [
                to_image(point.aligned_x, point.aligned_y, point.frame_index)
                for point in track.points
            ]
            draw.line(observed, fill=NAVY_DARK, width=5, joint="curve")
            draw.line(observed, fill=SKY_LIGHT, width=2, joint="curve")
            for point in track.points:
                point_x, point_y = to_image(point.aligned_x, point.aligned_y, point.frame_index)
                draw.ellipse((point_x - 3, point_y - 3, point_x + 3, point_y + 3), fill=MOTION_TRAIL, outline=NAVY_DARK)
            last = track.points[-1]
            if frame_index != last.frame_index:
                continue
            kinematics = self._track_kinematics(self.sequence_result, track)
            if kinematics is None or kinematics["predicted_x_px"] is None or kinematics["predicted_y_px"] is None:
                continue
            forecast_end = to_image(
                float(kinematics["predicted_x_px"]),
                float(kinematics["predicted_y_px"]),
                last.frame_index,
            )
            start = to_image(last.aligned_x, last.aligned_y, last.frame_index)
            for segment_index in range(10):
                if segment_index % 2:
                    continue
                fraction_start = segment_index / 10.0
                fraction_end = min(1.0, fraction_start + 0.1)
                x0 = start[0] + (forecast_end[0] - start[0]) * fraction_start
                y0 = start[1] + (forecast_end[1] - start[1]) * fraction_start
                x1 = start[0] + (forecast_end[0] - start[0]) * fraction_end
                y1 = start[1] + (forecast_end[1] - start[1]) * fraction_end
                draw.line((x0, y0, x1, y1), fill=NAVY_DARK, width=4)
                draw.line((x0, y0, x1, y1), fill=FORECAST, width=2)
            draw.ellipse((forecast_end[0] - 4, forecast_end[1] - 4, forecast_end[0] + 4, forecast_end[1] + 4), outline=FORECAST, width=2)
            x_half = kinematics.get("predicted_x_ci95_half_width_px")
            y_half = kinematics.get("predicted_y_ci95_half_width_px")
            if x_half is not None and y_half is not None:
                draw.ellipse(
                    (
                        forecast_end[0] - float(x_half) * scale_x * scale,
                        forecast_end[1] - float(y_half) * scale_y * scale,
                        forecast_end[0] + float(x_half) * scale_x * scale,
                        forecast_end[1] + float(y_half) * scale_y * scale,
                    ),
                    outline=FORECAST,
                    width=1,
                )
            self._draw_image_label(draw, f"FORECAST +5F · {track.track_id:04d}", forecast_end[0], forecast_end[1], image_size)

    def _motion_feature_points(self, frame_index: int) -> tuple[tuple[MotionFeatureTrack, MotionFeaturePoint], ...]:
        """返回序列线状证据，或当前单帧的长线候选。"""

        if self.sequence_result is not None:
            return motion_features_for_frame(self.sequence_result, frame_index)
        # 单帧候选的局部 frame_index 固定为 0；它代表当前正在查看的帧，
        # 而不是数据目录中的第 1 帧。因此切到第 N 帧后仍应显示当前分析结果。
        return tuple(
            (track, point)
            for track in self.long_trails
            for point in track.points
            if point.frame_index == 0
        )

    def _source_overlay_label(self, source: Any, *, magnitude_text: str | None = None) -> str:
        """生成图内短标签；使用 ASCII，避免没有中文字体时出现方框。"""

        signal_snr = getattr(source, "flux_snr", None)
        if signal_snr is None:
            signal_snr = getattr(source, "snr", None)
        snr_text = f"{float(signal_snr):.1f}" if isinstance(signal_snr, (int, float)) else "-"
        if magnitude_text is None:
            magnitude = instrumental_magnitude(source.flux, exposure_s=self.exposure_s)
            magnitude_text = f"{magnitude:.2f}" if magnitude is not None else "-"
        parameters = self.analysis.detection.parameters if self.analysis is not None else None
        reason = source_quality_reason(source, parameters, compact=True)
        decision = reason if reason.startswith("PASS") else f"REJECT/{reason}"
        return f"ID {source.detection_id:04d} | FSNR {snr_text} | m_inst {magnitude_text} | {decision}"

    def _draw_image_label(self, draw: ImageDraw.ImageDraw, text: str, x: float, y: float, image_size: tuple[int, int]) -> None:
        # 12px 在 1× 预览中仍可读，放大后不会因为默认位图字体过小而
        # 看不清；Windows 下优先使用微软雅黑等中文字体，标签内容本身
        # 也尽量采用 ASCII，保证字体回退时不产生乱码。
        font = _overlay_font(12)
        box = draw.textbbox((0, 0), text, font=font)
        width = box[2] - box[0]
        height = box[3] - box[1]
        left = min(max(5, x + 11), max(5, image_size[0] - width - 10))
        top = min(max(5, y - height - 8), max(5, image_size[1] - height - 6))
        draw.rectangle((left - 3, top - 2, left + width + 5, top + height + 3), fill=NAVY_DARK)
        draw.text((left, top), text, fill="#bce0d1", font=font)

    def _view_transform(self) -> tuple[float, float, float, float]:
        if self.preview is None:
            return 1.0, 0.0, 0.0, 1.0
        canvas_width = max(100, self.canvas.winfo_width())
        canvas_height = max(100, self.canvas.winfo_height())
        fit_scale = min((canvas_width - 14) / self.preview.width, (canvas_height - 14) / self.preview.height)
        scale = max(0.01, fit_scale * self.preview_zoom)
        origin_x = (canvas_width - self.preview.width * scale) / 2 + self.preview_pan_x
        origin_y = (canvas_height - self.preview.height * scale) / 2 + self.preview_pan_y
        return scale, origin_x, origin_y, fit_scale

    def _zoom_at(self, event: tk.Event, factor: float) -> None:
        if self.preview is None:
            return
        old_scale, old_origin_x, old_origin_y, _ = self._view_transform()
        base_x = (event.x - old_origin_x) / old_scale
        base_y = (event.y - old_origin_y) / old_scale
        self.preview_zoom = min(MAX_PREVIEW_ZOOM, max(0.35, self.preview_zoom * factor))
        new_scale, _, _, _ = self._view_transform()
        canvas_width = max(100, self.canvas.winfo_width())
        canvas_height = max(100, self.canvas.winfo_height())
        self.preview_pan_x = event.x - base_x * new_scale - (canvas_width - self.preview.width * new_scale) / 2
        self.preview_pan_y = event.y - base_y * new_scale - (canvas_height - self.preview.height * new_scale) / 2
        self.status_var.set(f"视图缩放 {self.preview_zoom:.2f}× · 滚轮继续缩放，左键拖拽平移")
        self._draw_preview()

    def _on_mousewheel(self, event: tk.Event) -> None:
        self._zoom_at(event, 1.15 if event.delta > 0 else 1 / 1.15)

    def _on_pan_start(self, event: tk.Event) -> None:
        if self.preview is not None:
            self.drag_start = (event.x, event.y)

    def _on_pan_move(self, event: tk.Event) -> None:
        if self.drag_start is None:
            return
        previous_x, previous_y = self.drag_start
        self.preview_pan_x += event.x - previous_x
        self.preview_pan_y += event.y - previous_y
        self.drag_start = (event.x, event.y)
        self._draw_preview()

    def _find_source_at(self, event: tk.Event) -> Any | None:
        if self.analysis is None or self.preview is None or self.preview_shape is None:
            return None
        scale, origin_x, origin_y, _ = self._view_transform()
        base_x = (event.x - origin_x) / scale
        base_y = (event.y - origin_y) / scale
        if not (0 <= base_x < self.preview.width and 0 <= base_y < self.preview.height):
            return None
        original_height, original_width = self.preview_shape
        scale_x = self.preview.width / original_width
        scale_y = self.preview.height / original_height
        original_x = base_x / scale_x
        original_y = base_y / scale_y
        cell_size = 32
        cell_x = int(original_x // cell_size)
        cell_y = int(original_y // cell_size)
        screen_radius = 12.0
        best = None
        best_distance = screen_radius * screen_radius
        quality_only = self.overlay_mode_var.get() == "quality"
        for gx in range(cell_x - 1, cell_x + 2):
            for gy in range(cell_y - 1, cell_y + 2):
                for source in self.source_grid.get((gx, gy), ()):
                    if quality_only and not source.quality_passed:
                        continue
                    for source_x, source_y in ((float(source.x), float(source.y)), source_peak_position(source)):
                        point_x = origin_x + source_x * scale_x * scale
                        point_y = origin_y + source_y * scale_y * scale
                        distance = (point_x - event.x) ** 2 + (point_y - event.y) ** 2
                    if distance <= best_distance:
                        best = source
                        best_distance = distance
        return best

    def _find_trusted_track_at(self, event: tk.Event) -> tuple[SourceTrack, TrackPoint] | None:
        """Find a quality-passed sequence source near the pointer."""

        if self.preview is None or self.preview_shape is None or self.sequence_result is None:
            return None
        frame_index = self._sequence_frame_index()
        if frame_index is None:
            return None
        scale, origin_x, origin_y, _ = self._view_transform()
        original_height, original_width = self.preview_shape
        scale_x = self.preview.width / original_width
        scale_y = self.preview.height / original_height
        best: tuple[SourceTrack, TrackPoint] | None = None
        best_distance = 12.0 * 12.0
        for track, point in trusted_points_for_frame(self.sequence_result, frame_index):
            point_x = origin_x + point.x * scale_x * scale
            point_y = origin_y + point.y * scale_y * scale
            distance = (point_x - event.x) ** 2 + (point_y - event.y) ** 2
            if distance <= best_distance:
                best = (track, point)
                best_distance = distance
        return best

    def _find_motion_track_at(
        self,
        event: tk.Event,
    ) -> tuple[SourceTrack | MotionFeatureTrack, TrackPoint | MotionFeaturePoint] | None:
        if self.preview is None or self.preview_shape is None:
            return None
        frame_index = self._sequence_frame_index()
        if frame_index is None:
            return None
        scale, origin_x, origin_y, _ = self._view_transform()
        original_height, original_width = self.preview_shape
        scale_x = self.preview.width / original_width
        scale_y = self.preview.height / original_height
        best: tuple[SourceTrack | MotionFeatureTrack, TrackPoint | MotionFeaturePoint] | None = None
        best_distance = 12.0 * 12.0
        for track, point in moving_points_for_frame(self.sequence_result, frame_index):
            point_x = origin_x + point.x * scale_x * scale
            point_y = origin_y + point.y * scale_y * scale
            distance = (point_x - event.x) ** 2 + (point_y - event.y) ** 2
            if distance <= best_distance:
                best = (track, point)
                best_distance = distance
        for track, point in self._motion_feature_points(frame_index):
            point_x = origin_x + point.x * scale_x * scale
            point_y = origin_y + point.y * scale_y * scale
            half_length = 0.5 * point.length_px * scale
            angle = np.deg2rad(point.angle_deg)
            dx = np.cos(angle) * half_length * scale_x
            dy = np.sin(angle) * half_length * scale_y
            start_x, start_y = point_x - dx, point_y - dy
            segment_x, segment_y = 2.0 * dx, 2.0 * dy
            segment_length_sq = segment_x**2 + segment_y**2
            if segment_length_sq <= 0:
                distance = (point_x - event.x) ** 2 + (point_y - event.y) ** 2
            else:
                projection = ((event.x - start_x) * segment_x + (event.y - start_y) * segment_y) / segment_length_sq
                projection = min(1.0, max(0.0, projection))
                closest_x = start_x + projection * segment_x
                closest_y = start_y + projection * segment_y
                distance = (closest_x - event.x) ** 2 + (closest_y - event.y) ** 2
            hit_radius = max(12.0, 0.5 * point.width_px * scale + 4.0)
            if distance <= hit_radius**2 and distance <= best_distance:
                best = (track, point)
                best_distance = distance
        return best

    def _find_stable_track_at(self, event: tk.Event) -> tuple[SourceTrack, TrackPoint] | None:
        """Find the nearest persistent static track point for the stable layer."""

        if self.preview is None or self.preview_shape is None or self.sequence_result is None:
            return None
        frame_index = self._sequence_frame_index()
        if frame_index is None:
            return None
        scale, origin_x, origin_y, _ = self._view_transform()
        original_height, original_width = self.preview_shape
        scale_x = self.preview.width / original_width
        scale_y = self.preview.height / original_height
        best: tuple[SourceTrack, TrackPoint] | None = None
        best_distance = 12.0 * 12.0
        for track, point in stable_points_for_frame(self.sequence_result, frame_index):
            point_x = origin_x + point.x * scale_x * scale
            point_y = origin_y + point.y * scale_y * scale
            distance = (point_x - event.x) ** 2 + (point_y - event.y) ** 2
            if distance <= best_distance:
                best = (track, point)
                best_distance = distance
        return best

    def _find_stack_faint_track_at(self, event: tk.Event) -> tuple[SourceTrack, TrackPoint] | None:
        """Find the nearest stack-faint recovered source for the overlay layer."""

        if self.preview is None or self.preview_shape is None or self.sequence_result is None:
            return None
        frame_index = self._sequence_frame_index()
        if frame_index is None:
            return None
        scale, origin_x, origin_y, _ = self._view_transform()
        original_height, original_width = self.preview_shape
        scale_x = self.preview.width / original_width
        scale_y = self.preview.height / original_height
        best: tuple[SourceTrack, TrackPoint] | None = None
        best_distance = 12.0 * 12.0
        for track, point in stack_faint_points_for_frame(self.sequence_result, frame_index):
            point_x = origin_x + point.x * scale_x * scale
            point_y = origin_y + point.y * scale_y * scale
            distance = (point_x - event.x) ** 2 + (point_y - event.y) ** 2
            if distance <= best_distance:
                best = (track, point)
                best_distance = distance
        return best

    def _find_catalog_match_at(self, event: tk.Event) -> Any | None:
        """在星表核验层按检测点或目录预测点寻找最近匹配。"""

        if self.preview is None or self.preview_shape is None or self.catalog_match_result is None:
            return None
        if self.catalog_frame_path is not None and self.selected_frame != self.catalog_frame_path:
            return None
        scale, origin_x, origin_y, _ = self._view_transform()
        original_height, original_width = self.preview_shape
        scale_x = self.preview.width / original_width
        scale_y = self.preview.height / original_height
        best = None
        best_distance = 14.0 * 14.0
        for match in self.catalog_match_result.matches:
            for raw_x, raw_y in ((match.detection_x, match.detection_y), (match.predicted_x, match.predicted_y)):
                point_x = origin_x + raw_x * scale_x * scale
                point_y = origin_y + raw_y * scale_y * scale
                distance = (point_x - event.x) ** 2 + (point_y - event.y) ** 2
                if distance <= best_distance:
                    best = match
                    best_distance = distance
        return best

    def _update_trusted_hover(self, track_point: tuple[SourceTrack, TrackPoint] | None) -> None:
        """Update hover information for a sequence quality-passed source."""

        track_key = ("trusted", track_point[0].track_id) if track_point is not None else None
        previous_key = (
            ("trusted", self.hover_trusted_track[0].track_id)
            if self.hover_trusted_track is not None
            else None
        )
        if track_key == previous_key:
            return
        self.hover_trusted_track = track_point
        self.hover_motion_track = None
        self.hover_source = None
        self.hover_source_id = None
        self.hover_catalog_match = None
        if track_point is None:
            self.hover_info_var.set("可信源主层只显示当前帧通过单帧质量规则的源；将鼠标移到青色点查看 SNR 和跨帧证据")
        else:
            track, point = track_point
            snr_text = f"{point.flux_snr:.1f}" if point.flux_snr is not None else "—"
            frame_total = len(self.sequence_result.frames) if self.sequence_result is not None else 0
            self.hover_info_var.set(
                f"可信源 {track.track_id:04d}  ·  当前帧 {point.frame_index + 1:02d}  ·  "
                f"X {point.x:.1f} Y {point.y:.1f}  ·  flux SNR {snr_text}  ·  "
                f"质量通过  ·  持续 {track.presence}/{frame_total} 帧  ·  "
                f"分类 {track.classification}  ·  配准后位移 {track.displacement_px:.2f}px"
            )
        self._draw_preview()

    def _update_motion_hover(
        self,
        track_point: tuple[SourceTrack | MotionFeatureTrack, TrackPoint | MotionFeaturePoint] | None,
    ) -> None:
        """Update hover information for a motion layer or a motion composite overlay."""

        track_key = (
            ("feature", track_point[0].track_id)
            if track_point is not None and isinstance(track_point[1], MotionFeaturePoint)
            else ("point", track_point[0].track_id)
            if track_point is not None
            else None
        )
        previous_key = (
            ("feature", self.hover_motion_track[0].track_id)
            if self.hover_motion_track is not None and isinstance(self.hover_motion_track[1], MotionFeaturePoint)
            else ("point", self.hover_motion_track[0].track_id)
            if self.hover_motion_track is not None
            else None
        )
        if track_key == previous_key:
            return
        self.hover_motion_track = track_point
        self.hover_trusted_track = None
        self.hover_source = None
        self.hover_source_id = None
        self.hover_catalog_match = None
        if track_point is None:
            if self.overlay_mode_var.get() == "motion":
                self.hover_info_var.set("运动层只显示跨帧线状候选和严格 moving 点轨迹；将鼠标移到洋红线或橙线查看信息")
            else:
                self.hover_info_var.set("叠加层只显示运动证据；将鼠标移到洋红线或橙线查看轨迹、SNR 和拟合误差")
        else:
            track, point = track_point
            if isinstance(point, MotionFeaturePoint):
                fit_text = f"{track.fit_rms_px:.2f}px" if track.fit_rms_px is not None else "—"
                state_text = "moving" if track.classification == "moving" else "单帧候选"
                selected_frame_index = self._selected_frame_index()
                display_frame = point.frame_index + 1 if self.sequence_result is not None else ((selected_frame_index + 1) if selected_frame_index is not None else 1)
                self.hover_info_var.set(
                    f"TRAIL {track.track_id:04d}  ·  {state_text}  ·  当前帧 {display_frame:02d}  ·  "
                    f"X {point.x:.1f}  Y {point.y:.1f}  ·  residual SNR {point.residual_snr:.1f}  ·  "
                    f"长度 {point.length_px:.1f}px  / 宽度 {point.width_px:.1f}px  ·  "
                    f"角度 {point.angle_deg:.1f}°  ·  位移 {track.displacement_px:.1f}px  ·  "
                    f"拟合 RMS {fit_text}  ·  出现 {track.presence} 帧"
                )
            else:
                fit_text = f"{track.fit_rms_px:.3f}px" if track.fit_rms_px is not None else "—"
                snr_text = f"{point.flux_snr:.1f}" if point.flux_snr is not None else "—"
                motion_type = (
                    "高速点状运动 · 宽筛候选 + 原图复核"
                    if track.evidence_level == "fast_point_motion" and point.candidate_snr is not None
                    else "高速点状运动 · 逐帧质量通过"
                    if track.evidence_level == "fast_point_motion"
                    else "点源 moving"
                )
                candidate_snr_text = (
                    f"  ·  filter SNR {point.candidate_snr:.1f}"
                    if point.candidate_snr is not None
                    else ""
                )
                self.hover_info_var.set(
                    f"MOV {track.track_id:04d}  ·  {motion_type}  ·  当前帧 {point.frame_index + 1:02d}  ·  "
                    f"X {point.x:.1f}  Y {point.y:.1f}  ·  flux SNR {snr_text}  ·  "
                    f"{candidate_snr_text}"
                    f"位移 {track.displacement_px:.2f}px  ·  速度 {track.speed_px_per_frame:.3f}px/frame  ·  "
                    f"拟合 RMS {fit_text}  ·  出现 {track.presence} 帧"
                )
        self._draw_preview()

    def _on_canvas_motion(self, event: tk.Event) -> None:
        if self.overlay_mode_var.get() == "catalog":
            match = self._find_catalog_match_at(event)
            match_key = match.source_id if match is not None else None
            previous_key = self.hover_catalog_match.source_id if self.hover_catalog_match is not None else None
            if match_key == previous_key:
                return
            self.hover_catalog_match = match
            self.hover_source = None
            self.hover_source_id = None
            self.hover_motion_track = None
            self.hover_trusted_track = None
            if match is None:
                self.hover_info_var.set("星表核验层只显示匹配成功的源；将鼠标移到绿色预测环或黄色检测点查看残差")
            else:
                magnitude_text = f"{match.catalog_magnitude:.3f}" if match.catalog_magnitude is not None else "—"
                self.hover_info_var.set(
                    f"CAT {match.source_id}  ·  检测 X {match.detection_x:.1f} Y {match.detection_y:.1f}  ·  "
                    f"预测 X {match.predicted_x:.1f} Y {match.predicted_y:.1f}  ·  残差 {match.residual_px:.3f}px  ·  "
                    f"目录星等 {magnitude_text}"
                )
            self._draw_preview()
            return
        if self.overlay_mode_var.get() == "stable":
            track_point = self._find_stable_track_at(event)
            track_key = ("stable", track_point[0].track_id) if track_point is not None else None
            previous_key = (
                ("stable", self.hover_motion_track[0].track_id)
                if self.hover_motion_track is not None
                and not isinstance(self.hover_motion_track[1], MotionFeaturePoint)
                and self.hover_motion_track[0].classification in {"static", "persistent"}
                else None
            )
            if track_key == previous_key:
                return
            self.hover_motion_track = track_point
            self.hover_trusted_track = None
            self.hover_source = None
            self.hover_source_id = None
            self.hover_catalog_match = None
            if track_point is None:
                self.hover_info_var.set("稳定星场层只显示跨帧持续出现的静态源；将鼠标移到绿色点查看轨迹证据")
            else:
                track, point = track_point
                if point.flux_snr is not None:
                    snr_label = "flux SNR"
                    snr_text = f"{point.flux_snr:.1f}"
                elif point.candidate_snr is not None:
                    snr_label = "candidate SNR"
                    snr_text = f"{point.candidate_snr:.1f}"
                else:
                    snr_label = "SNR"
                    snr_text = "—"
                frame_total = len(self.sequence_result.frames)
                state_text = (
                    "严格静态"
                    if track.classification == "static"
                    else "时间中值补检"
                    if track.evidence_level == "temporal_reference"
                    else "滤波共识补检"
                )
                self.hover_info_var.set(
                    f"{state_text} {track.track_id:04d}  ·  当前帧 {point.frame_index + 1:02d}  ·  "
                    f"X {point.x:.1f} Y {point.y:.1f}  ·  {snr_label} {snr_text}  ·  "
                    f"持续 {track.presence}/{frame_total} 帧  ·  配准后位移 {track.displacement_px:.2f}px  ·  "
                    f"拟合 RMS {track.fit_rms_px:.3f}px"
                )
            self._draw_preview()
            return
        if self.overlay_mode_var.get() == "stack-faint":
            track_point = self._find_stack_faint_track_at(event)
            track_key = ("stack-faint", track_point[0].track_id) if track_point is not None else None
            previous_key = (
                ("stack-faint", self.hover_motion_track[0].track_id)
                if self.hover_motion_track is not None
                and not isinstance(self.hover_motion_track[1], MotionFeaturePoint)
                and self.hover_motion_track[0].evidence_level == "stack_faint"
                else None
            )
            if track_key == previous_key:
                return
            self.hover_motion_track = track_point
            self.hover_trusted_track = None
            self.hover_source = None
            self.hover_source_id = None
            self.hover_catalog_match = None
            if track_point is None:
                self.hover_info_var.set("叠加暗星层只显示叠加参考图恢复、并经逐帧强制测光确认的暗星；将鼠标移到绿灰点查看证据")
            else:
                track, point = track_point
                if point.flux_snr is not None:
                    snr_text = f"{point.flux_snr:.1f}"
                elif point.candidate_snr is not None:
                    snr_text = f"{point.candidate_snr:.1f}"
                else:
                    snr_text = "—"
                frame_total = len(self.sequence_result.frames)
                self.hover_info_var.set(
                    f"叠加暗星 {track.track_id:04d}  ·  当前帧 {point.frame_index + 1:02d}  ·  "
                    f"X {point.x:.1f} Y {point.y:.1f}  ·  逐帧 flux SNR {snr_text}  ·  "
                    f"持续 {track.presence}/{frame_total} 帧  ·  配准后位移 {track.displacement_px:.2f}px  ·  "
                    f"拟合 RMS {track.fit_rms_px:.3f}px"
                )
            self._draw_preview()
            return
        mode = self.overlay_mode_var.get()
        if mode == "motion" or self._motion_overlay_enabled():
            track_point = self._find_motion_track_at(event)
            if track_point is not None or mode == "motion":
                self._update_motion_hover(track_point)
                return
            # 复合层没有命中轨迹时，继续向下寻找可信源，而不是让轨迹
            # 的空命中信息遮住星点本身的 SNR/质量说明。
            self.hover_motion_track = None
        if mode == "quality" and self.analysis is None and self.sequence_result is not None:
            trusted_track = self._find_trusted_track_at(event)
            if trusted_track is not None:
                self._update_trusted_hover(trusted_track)
                return
            self.hover_trusted_track = None
        source = self._find_source_at(event)
        source_id = source.detection_id if source is not None else None
        if source_id == self.hover_source_id:
            return
        self.hover_source_id = source_id
        self.hover_source = source
        self.hover_trusted_track = None
        if source is None:
            self.hover_info_var.set("将鼠标移到候选点查看坐标、通量、误差、SNR、形状和仪器星等")
        else:
            magnitude = instrumental_magnitude(source.flux, exposure_s=self.exposure_s)
            magnitude_text = f"{magnitude:.3f}" if magnitude is not None else "—"
            flags = ", ".join(source.flags) if source.flags else "无"
            signal_snr = source.flux_snr if source.flux_snr is not None else source.snr
            filter_snr_text = f"{source.filter_snr:.2f}" if source.filter_snr is not None else "—"
            peak_snr_text = f"{source.snr:.2f}" if source.snr is not None else "—"
            error_text = f"{source.flux_error:.1f}" if source.flux_error is not None else "—"
            shape_text = f"FWHM {source.fwhm:.2f}px / e {source.ellipticity:.2f}" if source.fwhm is not None and source.ellipticity is not None else "shape —"
            parameters = self.analysis.detection.parameters if self.analysis is not None else None
            quality_text = source_quality_reason(source, parameters)
            peak_x, peak_y = source_peak_position(source)
            shift_text = f"Δcentroid {source.centroid_shift_px:.2f}px" if source.centroid_shift_px is not None else "Δcentroid —"
            psf_required = parameters.get("min_psf_support_pixels", 3) if parameters is not None else 3
            psf_support_text = f"{source.psf_support_pixels}/{psf_required}" if source.psf_support_pixels is not None else "—"
            proposal_methods = "+".join(source.proposal_methods) if source.proposal_methods else "legacy/unknown"
            proposal_scales = ",".join(f"{scale:.2f}" for scale in source.proposal_scales) if source.proposal_scales else "—"
            proposal_snr = f"{source.proposal_snr:.2f}" if source.proposal_snr is not None else "—"
            nearest_gaussian = f"{source.nearest_gaussian_px:.2f}px" if source.nearest_gaussian_px is not None else "—"
            deblend_bic = f"{source.deblend_delta_bic:.1f}" if source.deblend_delta_bic is not None else "—"
            deblend_snr = f"{source.deblend_component_snr:.2f}" if source.deblend_component_snr is not None else "—"
            repeated_code_value = getattr(source, "repeated_code_count", None)
            repeated_code_values = getattr(source, "repeated_code_values", ())
            repeated_code_count = (
                f"{repeated_code_value} [{','.join(str(value) for value in repeated_code_values)}]"
                if repeated_code_value is not None and repeated_code_values
                else "—"
            )
            range_anomaly_value = getattr(source, "range_anomaly_pixel_count", None)
            range_anomaly_count = (
                str(range_anomaly_value)
                if range_anomaly_value is not None
                else "—"
            )
            self.hover_info_var.set(
                f"ID {source.detection_id:04d}  ·  centroid X {source.x:.1f} Y {source.y:.1f}  ·  peak X {peak_x:.0f} Y {peak_y:.0f}\n"
                f"flux {source.flux:.1f} ADU ± {error_text}  ·  flux SNR {signal_snr:.2f}  ·  peak SNR {peak_snr_text}  ·  filter SNR {filter_snr_text}\n"
                f"提案 {proposal_methods}  ·  proposal SNR {proposal_snr}  ·  尺度 {proposal_scales}px  ·  最近Gaussian {nearest_gaussian}\n"
                f"去混叠 ΔBIC {deblend_bic}  ·  次分量SNR {deblend_snr}\n"
                f"数据审计：重复高位码 {repeated_code_count}  ·  异常负值像素 {range_anomaly_count}\n"
                f"{shape_text}  ·  PSF支持 {psf_support_text}  ·  {shift_text}  ·  m_inst {magnitude_text}\n"
                f"判定：{quality_text}  ·  flags：{flags}"
            )
        self._draw_preview()

    def _clear_hover(self) -> None:
        if self.hover_source_id is None and self.hover_motion_track is None and self.hover_trusted_track is None and self.hover_catalog_match is None:
            return
        self.hover_source_id = None
        self.hover_source = None
        self.hover_motion_track = None
        self.hover_trusted_track = None
        self.hover_catalog_match = None
        self.hover_info_var.set("将鼠标移到候选点查看坐标、通量、误差、SNR、形状和仪器星等")
        self._draw_preview()

    def clear_detection_cache(self) -> None:
        if not messagebox.askyesno("清空检测缓存", f"删除本地检测缓存？\n\n{self.cache_dir}\n\n不会删除 FITS 原图。"):
            return
        with self.cache_lock:
            self.cache_generation += 1
            removed = clear_cache(self.cache_dir)
        self.analysis = None
        self.catalog_analysis = None
        self.catalog_match_result = None
        self.catalog_wcs = None
        self.catalog_calibration = None
        self.catalog_frame_path = None
        self.sequence_result = None
        self.source_grid = {}
        self.hover_source = None
        self.hover_source_id = None
        self.hover_motion_track = None
        self.hover_trusted_track = None
        self.hover_catalog_match = None
        self._reset_result_widgets()
        self._set_job_controls()
        self._draw_preview()
        if not self.busy:
            self._set_sequence_progress(0.0, "缓存已清空 · 序列待运行")
        running_note = " · 当前任务仍在运行，完成结果不会写回缓存" if self.busy else ""
        self.status_var.set(f"已清空 {removed} 个缓存文件 · 原图未删除{running_note}")

    def _on_close(self) -> None:
        self._closing = True
        self.destroy()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="启动 rst19 纯 Python Tkinter 星图分析界面")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="FITS 数据目录")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    data_dir = args.data_dir.resolve()
    if not data_dir.is_dir():
        print(f"rst19-gui: 数据目录不存在：{data_dir}")
        return 2
    app = StarfieldApp(data_dir)
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
