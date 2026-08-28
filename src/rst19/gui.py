"""rst19 的纯 Python Tkinter 桌面界面。"""

from __future__ import annotations

import argparse
import queue
import threading
from pathlib import Path
from typing import Any

import numpy as np
import tkinter as tk
from PIL import Image, ImageDraw, ImageFont, ImageTk
from tkinter import messagebox, ttk

from .cache import cache_key, clear_cache, load_analysis, save_analysis
from .fits import auxiliary_mask, read_fits
from .photometry import instrumental_magnitude
from .pipeline import FrameAnalysis, analyze_frame
from .sequence import (
    MotionFeaturePoint,
    MotionFeatureTrack,
    SequenceResult,
    SourceTrack,
    TrackPoint,
    analyze_sequence,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "doc" / "00-项目资料" / "原始数据"

NAVY = "#20293d"
NAVY_DARK = "#182033"
NAVY_SOFT = "#33415b"
PAPER = "#f4f0e7"
PAPER_LIGHT = "#faf8f2"
PAPER_LINE = "#ded8ca"
INK = "#273147"
INK_SOFT = "#626b7d"
AMBER = "#d79432"
AMBER_LIGHT = "#f0b34b"
MINT = "#4f9b83"
SKY = "#72b9d4"
SKY_LIGHT = "#b9e4ef"
WHITE = "#f6f1e7"
MONO = "Consolas"
SANS = "Segoe UI"


def _make_preview(data: np.ndarray, max_side: int = 1600) -> Image.Image:
    """创建可缩放的基础预览；检测点在 Canvas 当前缩放级别上动态绘制。"""

    values = np.asarray(data, dtype=np.float64).copy()
    values[auxiliary_mask(values.shape)] = np.nan
    valid = values[np.isfinite(values)]
    if valid.size == 0:
        raise ValueError("图像没有可预览的有限像素")
    low, high = np.percentile(valid, [1.0, 99.8])
    if high <= low:
        high = low + 1.0
    normalized = np.clip((values - low) / (high - low), 0.0, 1.0)
    normalized[~np.isfinite(normalized)] = 0.0
    image = Image.fromarray(np.rint(normalized * 255.0).astype(np.uint8), mode="L").convert("RGB")
    image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)

    return image


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


class StarfieldApp(tk.Tk):
    """本地星图检测工作台。"""

    def __init__(self, data_dir: Path, cache_dir: Path | None = None) -> None:
        super().__init__()
        self.data_dir = data_dir.resolve()
        self.cache_dir = (cache_dir or PROJECT_ROOT / ".rst19-cache").resolve()
        self.frames: list[Path] = []
        self.selected_frame: Path | None = None
        self.analysis: FrameAnalysis | None = None
        self.sequence_result: SequenceResult | None = None
        self.preview: Image.Image | None = None
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
        self.source_grid: dict[tuple[int, int], list[Any]] = {}
        self.exposure_s = 1.0
        self.frame_token = 0
        self.result_queue: queue.Queue[tuple[str, int, Any]] = queue.Queue()
        self.busy = False
        self.active_job_token: int | None = None
        self.cache_generation = 0
        self.cache_lock = threading.Lock()

        self.title("RST19 · Starfield Lab")
        self.geometry("1440x900")
        self.minsize(1120, 720)
        self.configure(bg=PAPER)
        self._configure_styles()
        self._build_layout()
        self._load_frames()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(100, self._poll_result)

    def _configure_styles(self) -> None:
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("Treeview", background=PAPER_LIGHT, fieldbackground=PAPER_LIGHT, foreground=INK, rowheight=27, font=(MONO, 9))
        style.configure("Treeview.Heading", background=PAPER, foreground=INK_SOFT, font=(MONO, 8, "bold"), relief="flat")
        style.map("Treeview", background=[("selected", "#e9d8b7")], foreground=[("selected", NAVY_DARK)])
        style.configure("TScrollbar", background=PAPER_LINE, troughcolor=PAPER, bordercolor=PAPER, arrowcolor=INK_SOFT)

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

        topbar = tk.Frame(self.main, bg=PAPER, height=70)
        topbar.pack(fill="x", padx=33)
        topbar.pack_propagate(False)
        self._mono_label(topbar, "RST19 SOL  /  观测工作台", color=INK_SOFT, size=9, bg=PAPER).pack(side="left", pady=27)
        self._mono_label(topbar, "●  LOCAL ONLY", color=MINT, size=9, bg=PAPER).pack(side="right", pady=27)
        tk.Frame(self.main, bg=PAPER_LINE, height=1).pack(fill="x", padx=33)

        header = tk.Frame(self.main, bg=PAPER)
        header.pack(fill="x", padx=38, pady=(28, 20))
        heading = tk.Frame(header, bg=PAPER)
        heading.pack(side="left")
        self._mono_label(heading, "OBSERVATION DESK · PYTHON DESKTOP", color=INK_SOFT, size=8, bg=PAPER).pack(anchor="w")
        self._label(heading, "星图检测工作台", color=NAVY_DARK, size=28, bold=True, bg=PAPER).pack(anchor="w", pady=(7, 3))
        self._label(heading, "在本地 FITS 中寻找候选星点，并标记最暗可信源。", color=INK_SOFT, size=10, bg=PAPER).pack(anchor="w")
        self._mono_label(header, "SOURCE DETECTION\n/ RECALL FIRST", color=AMBER, size=10, bg=PAPER, justify="right").pack(side="right", anchor="s", pady=5)

        self._build_controls()
        self._build_metrics()
        self._build_content()

    def _build_controls(self) -> None:
        controls = tk.Frame(self.main, bg=PAPER_LIGHT, highlightbackground=PAPER_LINE, highlightthickness=1)
        controls.pack(fill="x", padx=38, pady=(0, 18))
        controls.grid_columnconfigure(1, weight=1)
        self._mono_label(controls, "RUN CONTROL", color=AMBER, size=8, bg=PAPER_LIGHT).grid(row=0, column=0, padx=(16, 8), pady=15, sticky="w")
        self._mono_label(controls, "THRESHOLD", color=INK_SOFT, size=8, bg=PAPER_LIGHT).grid(row=0, column=2, padx=(15, 7), pady=15, sticky="e")
        self.threshold_var = tk.StringVar(value="4.0")
        tk.Entry(controls, textvariable=self.threshold_var, width=6, bg=PAPER, fg=INK, insertbackground=INK, relief="flat", highlightbackground=PAPER_LINE, highlightthickness=1, font=(MONO, 10)).grid(row=0, column=3, padx=(0, 4), pady=12)
        self._mono_label(controls, "σ", color=INK_SOFT, size=9, bg=PAPER_LIGHT).grid(row=0, column=4, padx=(0, 12), pady=15)
        self._mono_label(controls, "MIN DIST", color=INK_SOFT, size=8, bg=PAPER_LIGHT).grid(row=0, column=5, padx=(0, 7), pady=15, sticky="e")
        self.min_distance_var = tk.StringVar(value="4")
        tk.Entry(controls, textvariable=self.min_distance_var, width=5, bg=PAPER, fg=INK, insertbackground=INK, relief="flat", highlightbackground=PAPER_LINE, highlightthickness=1, font=(MONO, 10)).grid(row=0, column=6, padx=(0, 4), pady=12)
        self._mono_label(controls, "px", color=INK_SOFT, size=9, bg=PAPER_LIGHT).grid(row=0, column=7, padx=(0, 12), pady=15)
        self._mono_label(controls, "MAX SOURCES", color=INK_SOFT, size=8, bg=PAPER_LIGHT).grid(row=0, column=8, padx=(0, 7), pady=15, sticky="e")
        self.max_sources_var = tk.StringVar(value="")
        tk.Entry(controls, textvariable=self.max_sources_var, width=8, bg=PAPER, fg=INK, insertbackground=INK, relief="flat", highlightbackground=PAPER_LINE, highlightthickness=1, font=(MONO, 10)).grid(row=0, column=9, padx=(0, 4), pady=12)
        self._mono_label(controls, "留空 = 全量", color=MINT, size=8, bg=PAPER_LIGHT).grid(row=0, column=10, padx=(0, 14), pady=15)
        self.run_button = tk.Button(controls, text="▶  分析当前帧", command=self.run_analysis, bg=NAVY, fg=WHITE, activebackground=NAVY_SOFT, activeforeground=WHITE, relief="flat", bd=0, padx=16, pady=10, font=(SANS, 10, "bold"))
        self.run_button.grid(row=0, column=11, padx=(0, 15), pady=9)
        self.motion_button = tk.Button(controls, text="✦  分析 15 帧动目标", command=self.run_sequence_analysis, bg=SKY, fg=NAVY_DARK, activebackground=SKY_LIGHT, activeforeground=NAVY_DARK, relief="flat", bd=0, padx=14, pady=10, font=(SANS, 10, "bold"))
        self.motion_button.grid(row=0, column=12, padx=(0, 15), pady=9)
        self._mono_label(controls, "ZERO POINT", color=INK_SOFT, size=8, bg=PAPER_LIGHT).grid(row=1, column=2, padx=(15, 7), pady=(0, 12), sticky="e")
        self.zero_point_var = tk.StringVar(value="")
        tk.Entry(controls, textvariable=self.zero_point_var, width=6, bg=PAPER, fg=INK, insertbackground=INK, relief="flat", highlightbackground=PAPER_LINE, highlightthickness=1, font=(MONO, 10)).grid(row=1, column=3, padx=(0, 4), pady=(0, 12))
        self._mono_label(controls, "可选；用于 m_cal", color=INK_SOFT, size=8, bg=PAPER_LIGHT).grid(row=1, column=4, padx=(0, 10), pady=(0, 12), sticky="w")
        self._mono_label(controls, "PSF FWHM", color=INK_SOFT, size=8, bg=PAPER_LIGHT).grid(row=1, column=5, padx=(0, 7), pady=(0, 12), sticky="e")
        self.psf_fwhm_var = tk.StringVar(value="3.0")
        tk.Entry(controls, textvariable=self.psf_fwhm_var, width=5, bg=PAPER, fg=INK, insertbackground=INK, relief="flat", highlightbackground=PAPER_LINE, highlightthickness=1, font=(MONO, 10)).grid(row=1, column=6, padx=(0, 4), pady=(0, 12))
        self._mono_label(controls, "px", color=INK_SOFT, size=8, bg=PAPER_LIGHT).grid(row=1, column=7, padx=(0, 9), pady=(0, 12))
        self._mono_label(controls, "MIN FLUX SNR", color=INK_SOFT, size=8, bg=PAPER_LIGHT).grid(row=1, column=8, padx=(0, 7), pady=(0, 12), sticky="e")
        self.min_flux_snr_var = tk.StringVar(value="5.0")
        tk.Entry(controls, textvariable=self.min_flux_snr_var, width=5, bg=PAPER, fg=INK, insertbackground=INK, relief="flat", highlightbackground=PAPER_LINE, highlightthickness=1, font=(MONO, 10)).grid(row=1, column=9, padx=(0, 4), pady=(0, 12))
        self._mono_label(controls, "σ", color=INK_SOFT, size=8, bg=PAPER_LIGHT).grid(row=1, column=10, padx=(0, 10), pady=(0, 12))
        self.cache_button = tk.Button(controls, text="清空检测缓存", command=self.clear_detection_cache, bg=PAPER, fg=INK, activebackground="#e9d8b7", activeforeground=NAVY_DARK, relief="flat", bd=0, padx=13, pady=9, font=(SANS, 9, "bold"))
        self.cache_button.grid(row=1, column=11, padx=(0, 15), pady=(0, 12))

    def _build_metrics(self) -> None:
        metrics = tk.Frame(self.main, bg=PAPER)
        metrics.pack(fill="x", padx=38, pady=(0, 18))
        for column in range(5):
            metrics.grid_columnconfigure(column, weight=1)
        self.metric_values: dict[str, tk.Label] = {}
        specs = [
            ("candidate", "候选源", "ALL PEAKS", AMBER),
            ("returned", "可信星点", "QUALITY PASS", INK),
            ("background", "背景基线", "ADU", INK),
            ("noise", "噪声尺度", "ADU", INK),
            ("faintest", "最暗可信源", "M_INST", MINT),
        ]
        for column, (key, title, note, color) in enumerate(specs):
            box = tk.Frame(metrics, bg=NAVY if key == "candidate" else PAPER_LIGHT, highlightbackground=PAPER_LINE, highlightthickness=1)
            box.grid(row=0, column=column, sticky="nsew", padx=(0 if column == 0 else 1, 0))
            bg = NAVY if key == "candidate" else PAPER_LIGHT
            self._mono_label(box, f"{title.upper()}  ·  {note}", color="#b1bbc9" if key == "candidate" else INK_SOFT, size=8, bg=bg).pack(anchor="w", padx=13, pady=(12, 0))
            value = self._label(box, "—", color=color if key != "candidate" else AMBER_LIGHT, size=19, bold=False, bg=bg)
            value.pack(anchor="w", padx=13, pady=(8, 12))
            self.metric_values[key] = value

    def _build_content(self) -> None:
        content = tk.Frame(self.main, bg=PAPER)
        content.pack(fill="both", expand=True, padx=38, pady=(0, 25))
        content.grid_columnconfigure(0, weight=3)
        content.grid_columnconfigure(1, weight=2)
        content.grid_rowconfigure(0, weight=3)
        content.grid_rowconfigure(1, weight=2)

        viewer = tk.Frame(content, bg=PAPER_LIGHT, highlightbackground=PAPER_LINE, highlightthickness=1)
        viewer.grid(row=0, column=0, rowspan=2, sticky="nsew", padx=(0, 10))
        heading = tk.Frame(viewer, bg=PAPER_LIGHT)
        heading.pack(fill="x", padx=16, pady=(14, 10))
        self._mono_label(heading, "FRAME VIEWER", color=INK_SOFT, size=8, bg=PAPER_LIGHT).pack(anchor="w")
        self.frame_title_label = self._label(heading, "选择一个观测帧", color=NAVY_DARK, size=14, bold=True, bg=PAPER_LIGHT)
        self.frame_title_label.pack(anchor="w", pady=(3, 0))
        mode_box = tk.Frame(heading, bg=PAPER_LIGHT)
        mode_box.pack(side="right", anchor="s")
        self.overlay_mode_var = tk.StringVar(value="quality")
        self._mono_label(mode_box, "OVERLAY", color=INK_SOFT, size=8, bg=PAPER_LIGHT).pack(anchor="e")
        choices = (("quality", "可信星点"), ("candidates", "全部候选"), ("motion", "运动候选"))
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
            height=2,
            wraplength=640,
        )
        self.hover_info_label.pack(fill="x", padx=16, pady=(0, 4))
        viewer.bind("<Configure>", self._on_viewer_configure)
        self.overlay_hint_var = tk.StringVar(value="滚轮缩放 · 左键拖拽平移    琥珀色点 = 通过质量筛选    绿色环 = 最暗可信源")
        tk.Label(viewer, textvariable=self.overlay_hint_var, bg=PAPER_LIGHT, fg=INK_SOFT, font=(MONO, 8), anchor="w").pack(anchor="w", padx=16, pady=(0, 13))

        detail = tk.Frame(content, bg=PAPER_LIGHT, highlightbackground=PAPER_LINE, highlightthickness=1)
        detail.grid(row=0, column=1, sticky="nsew", pady=(0, 10))
        self._mono_label(detail, "FAINTEST SOURCE", color=MINT, size=8, bg=PAPER_LIGHT).pack(anchor="w", padx=16, pady=(15, 0))
        self.faintest_detail = self._label(detail, "尚未运行分析", color=NAVY_DARK, size=14, bold=True, bg=PAPER_LIGHT, justify="left")
        self.faintest_detail.pack(anchor="w", padx=16, pady=(8, 3))
        self.faintest_note = self._label(detail, "将按局部通量 SNR、点源形状、边缘、掩膜和饱和状态筛选", color=INK_SOFT, size=9, bg=PAPER_LIGHT, justify="left", wraplength=330)
        self.faintest_note.pack(anchor="w", padx=16, pady=(0, 15))
        tk.Frame(detail, bg=PAPER_LINE, height=1).pack(fill="x", padx=16)
        self._mono_label(detail, "SCIENTIFIC NOTE", color=AMBER, size=8, bg=PAPER_LIGHT).pack(anchor="w", padx=16, pady=(13, 5))
        self._label(detail, "默认显示的是仪器星等 m_inst = −2.5 log10(flux_rate)。提供零点后才显示 m_cal；没有波段转换时不把它命名为 Mv。", color=INK_SOFT, size=9, bg=PAPER_LIGHT, justify="left", wraplength=330).pack(anchor="w", padx=16, pady=(0, 14))

        register = tk.Frame(content, bg=PAPER_LIGHT, highlightbackground=PAPER_LINE, highlightthickness=1)
        register.grid(row=1, column=1, sticky="nsew")
        top = tk.Frame(register, bg=PAPER_LIGHT)
        top.pack(fill="x", padx=16, pady=(13, 7))
        self._mono_label(top, "SOURCE REGISTER · TOP FLUX SNR", color=INK_SOFT, size=8, bg=PAPER_LIGHT).pack(side="left")
        self.table_count_label = self._mono_label(top, "0 rows", color=AMBER, size=8, bg=PAPER_LIGHT)
        self.table_count_label.pack(side="right")
        columns = ("id", "xy", "peak", "snr", "mag")
        self.source_tree = ttk.Treeview(register, columns=columns, show="headings", height=6)
        for column, title, width in (("id", "ID", 42), ("xy", "X / Y", 110), ("peak", "PEAK", 70), ("snr", "F-SNR", 58), ("mag", "m_inst", 65)):
            self.source_tree.heading(column, text=title)
            self.source_tree.column(column, width=width, anchor="w", stretch=False)
        self.source_tree.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        self.status_var = tk.StringVar(value="就绪 · 请选择一张 FITS")
        status = tk.Frame(self.main, bg=NAVY_DARK, height=29)
        status.pack(fill="x", side="bottom")
        status.pack_propagate(False)
        tk.Label(status, text="●", bg=NAVY_DARK, fg=MINT, font=(SANS, 9)).pack(side="left", padx=(14, 7))
        tk.Label(status, textvariable=self.status_var, bg=NAVY_DARK, fg="#d4d9df", font=(MONO, 9), anchor="w").pack(side="left", fill="x", expand=True)
        tk.Label(status, text="RST19 · PYTHON / OFFLINE", bg=NAVY_DARK, fg="#96a1b1", font=(MONO, 8)).pack(side="right", padx=14)

    def _load_frames(self) -> None:
        self.frames = sorted(self.data_dir.glob("*.fits"))
        self.frame_list.delete(0, tk.END)
        for index, path in enumerate(self.frames, start=1):
            self.frame_list.insert(tk.END, f"{index:02d}  {path.stem}")
        self.frame_count_label.config(text=str(len(self.frames)))
        if self.frames:
            self.frame_list.selection_set(0)
            self.frame_list.activate(0)
            self.selected_frame = self.frames[0]
            self._select_frame(self.selected_frame)
        else:
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
        self.preview = None
        self.preview_shape = None
        self.source_grid = {}
        self.hover_source_id = None
        self.hover_source = None
        self.hover_motion_track = None
        self.preview_zoom = 1.0
        self.preview_pan_x = 0.0
        self.preview_pan_y = 0.0
        self._reset_result_widgets()
        self.canvas.delete("all")
        self.hover_info_var.set("正在载入当前帧预览 · 尚未运行检测")
        self.status_var.set(f"已切换 {frame_path.name} · 正在载入预览")

        def worker() -> None:
            try:
                frame = read_fits(frame_path)
                preview = _make_preview(frame.data)
                self.result_queue.put(("preview", token, (frame_path, frame.data.shape, preview)))
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

    def _set_job_controls(self) -> None:
        """根据当前后台任务状态同步按钮，避免切帧把旧任务误报成空闲。"""

        if self.busy:
            self.run_button.config(state="disabled", text="正在分析…")
            self.motion_button.config(state="disabled", text="正在分析…")
            self.cache_button.config(state="normal")
        else:
            self.run_button.config(state="normal", text="▶  分析当前帧")
            self.motion_button.config(state="normal", text="✦  分析 15 帧动目标")
            self.cache_button.config(state="normal")

    def _selected_frame_index(self) -> int | None:
        if self.selected_frame is None:
            return None
        try:
            return self.frames.index(self.selected_frame)
        except ValueError:
            return None

    def _on_overlay_mode_changed(self) -> None:
        mode = self.overlay_mode_var.get()
        if mode == "motion" and self.sequence_result is None:
            self.status_var.set("尚未完成 15 帧动目标分析 · 请先点击“分析 15 帧动目标”")
        elif mode == "motion":
            self.status_var.set("运动候选层 · 只显示跨帧线状目标、待复核线和严格 moving 点轨迹")
        elif mode == "candidates":
            self.status_var.set("全部候选层 · 包含被质量规则剔除的审计候选")
        self._update_overlay_hint()
        self._draw_preview()

    def _update_overlay_hint(self) -> None:
        hints = {
            "quality": "滚轮缩放 · 左键拖拽平移    琥珀色点 = 通过质量筛选    绿色环 = 最暗可信源",
            "candidates": "滚轮缩放 · 左键拖拽平移    琥珀色 = 可信源    暗琥珀 = 被剔除候选（仅审计）",
            "motion": "滚轮缩放 · 左键拖拽平移    蓝线 = 跨帧 moving 线状目标    橙线 = 待复核线    不显示静态背景星",
        }
        self.overlay_hint_var.set(hints.get(self.overlay_mode_var.get(), hints["quality"]))

    def _on_viewer_configure(self, event: tk.Event) -> None:
        """让悬浮信息栏随 viewer 换行，不用长文本撑大右侧布局。"""

        self.hover_info_label.config(wraplength=max(240, int(event.width) - 32))

    def run_analysis(self) -> None:
        if self.busy or self.selected_frame is None:
            if self.selected_frame is None:
                show_message = "请先选择数据目录中的 FITS 文件"
                messagebox.showinfo("RST19", show_message)
            return
        try:
            threshold, min_distance, max_sources, zero_point, psf_fwhm, min_flux_snr = self._read_parameters()
        except ValueError as exc:
            messagebox.showerror("参数错误", str(exc))
            return
        self.busy = True
        self.active_job_token = self.frame_token
        self._set_job_controls()
        self.status_var.set(f"正在分析 {self.selected_frame.name} · 全量检测模式" if max_sources is None else f"正在分析 {self.selected_frame.name} · 返回上限 {max_sources}")
        frame_path = self.selected_frame
        token = self.frame_token
        cache_generation = self.cache_generation

        def worker() -> None:
            try:
                frame = read_fits(frame_path)
                key = cache_key(
                    frame,
                    threshold_sigma=threshold,
                    min_distance=min_distance,
                    aperture_radius=4,
                    max_sources=max_sources,
                    zero_point=zero_point,
                    psf_fwhm=psf_fwhm,
                    min_flux_snr=min_flux_snr,
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
                        reject_linear_artifacts=True,
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
                preview = _make_preview(frame.data)
                self.result_queue.put(("analysis", token, (analysis, frame.data.shape, preview, cache_state, cache_hit)))
            except Exception as exc:  # noqa: BLE001 - worker must return a user-facing error
                self.result_queue.put(("analysis-error", token, exc))

        threading.Thread(target=worker, daemon=True).start()

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
        token = self.frame_token
        frame_paths = tuple(self.frames)
        self.busy = True
        self.active_job_token = token
        self._set_job_controls()
        self.status_var.set(f"正在分析 {len(frame_paths)} 帧 · 配准、关联并筛选运动轨迹…")

        def worker() -> None:
            try:
                result = analyze_sequence(
                    frame_paths,
                    threshold_sigma=threshold,
                    min_distance=min_distance,
                    max_sources=max_sources,
                    zero_point=zero_point,
                    psf_fwhm=psf_fwhm,
                    min_flux_snr=min_flux_snr,
                    reject_linear_artifacts=True,
                )
                self.result_queue.put(("sequence", token, result))
            except Exception as exc:  # noqa: BLE001 - worker must return a user-facing error
                self.result_queue.put(("sequence-error", token, exc))

        threading.Thread(target=worker, daemon=True).start()

    def _poll_result(self) -> None:
        try:
            kind, token, payload = self.result_queue.get_nowait()
        except queue.Empty:
            self.after(100, self._poll_result)
            return
        if token != self.frame_token:
            if self.active_job_token == token and kind in {"analysis", "analysis-error", "sequence", "sequence-error"}:
                self.active_job_token = None
                self.busy = False
                self._set_job_controls()
            self.after(100, self._poll_result)
            return
        if kind == "preview":
            if self.analysis is None:
                _frame_path, shape, preview = payload
                self.preview_shape = shape
                self.preview = preview
                self.preview_zoom = 1.0
                self.preview_pan_x = 0.0
                self.preview_pan_y = 0.0
                self.hover_info_var.set("预览已载入 · 点击“分析当前帧”运行检测")
                self._draw_preview()
        elif kind == "preview-error":
            self.status_var.set(f"预览载入失败 · {payload}")
            messagebox.showerror("预览载入失败", str(payload))
        elif kind == "analysis-error":
            self.active_job_token = None
            self.busy = False
            self._set_job_controls()
            self.status_var.set(f"分析失败 · {payload}")
            messagebox.showerror("分析失败", str(payload))
        elif kind == "analysis":
            self.active_job_token = None
            self.busy = False
            self._set_job_controls()
            self.analysis, shape, self.preview, cache_state, _cache_hit = payload
            self.preview_shape = shape
            self.preview_zoom = 1.0
            self.preview_pan_x = 0.0
            self.preview_pan_y = 0.0
            self._prepare_source_grid()
            detection = self.analysis.detection
            self.status_var.set(f"{cache_state} · 候选峰 {detection.candidate_count:,} · 可信星点 {detection.star_count:,} · 审计返回 {detection.returned_count:,}")
            self._render_analysis()
        elif kind == "sequence-error":
            self.active_job_token = None
            self.busy = False
            self._set_job_controls()
            self.status_var.set(f"动目标分析失败 · {payload}")
            messagebox.showerror("动目标分析失败", str(payload))
        elif kind == "sequence":
            self.active_job_token = None
            self.busy = False
            self._set_job_controls()
            self.sequence_result = payload
            self.overlay_mode_var.set("motion")
            self._update_overlay_hint()
            moving_features = sum(track.classification == "moving" for track in payload.motion_features)
            feature_candidates = sum(track.classification == "candidate" for track in payload.motion_features)
            self.status_var.set(
                f"15 帧分析完成 · 稳定点 {payload.stable_source_count:,} · "
                f"点轨迹 {payload.moving_track_count:,} · 线状目标 {moving_features:,} · "
                f"待复核线 {feature_candidates:,}"
            )
            self._draw_preview()
        self.after(100, self._poll_result)

    def _reset_result_widgets(self) -> None:
        for value in self.metric_values.values():
            value.config(text="—")
        self.faintest_detail.config(text="尚未运行分析")
        self.faintest_note.config(text="将按局部通量 SNR、点源形状、边缘、掩膜和饱和状态筛选")
        self.table_count_label.config(text="0 rows")
        self.hover_info_var.set("将鼠标移到候选点查看坐标、通量、误差、SNR、形状和仪器星等")
        for item in self.source_tree.get_children():
            self.source_tree.delete(item)

    def _prepare_source_grid(self) -> None:
        self.source_grid = {}
        if self.analysis is None:
            return
        for source in self.analysis.detection.sources:
            cell = (int(source.x // 32), int(source.y // 32))
            self.source_grid.setdefault(cell, []).append(source)

    def _render_analysis(self) -> None:
        assert self.analysis is not None
        detection = self.analysis.detection
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
            self.faintest_note.config(text="请降低阈值或检查掩膜/边缘筛选结果")
        else:
            self.metric_values["faintest"].config(text=f"{faintest.instrumental_magnitude:.2f}")
            calibrated = f"\n校准星等：{faintest.calibrated_magnitude:.2f}" if faintest.calibrated_magnitude is not None else ""
            self.faintest_detail.config(text=f"ID {faintest.detection_id:04d}\nm_inst = {faintest.instrumental_magnitude:.3f}{calibrated}\nX {faintest.x:.1f}  /  Y {faintest.y:.1f}")
            signal_snr = faintest.flux_snr if faintest.flux_snr is not None else faintest.snr
            rate_text = f"{faintest.flux_rate:.1f}" if faintest.flux_rate is not None else "—"
            self.faintest_note.config(text=f"通量 {faintest.flux:.1f} ADU ({rate_text} ADU/s) · flux SNR {signal_snr:.1f}\n已在左侧预览用绿色环标出")
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
                for source in sources:
                    point_x = source.x * self.preview_scale_x
                    point_y = source.y * self.preview_scale_y
                    if not (left <= point_x < right and top <= point_y < bottom):
                        continue
                    x = (point_x - left) * scale
                    y = (point_y - top) * scale
                    color = AMBER_LIGHT if source.quality_passed else "#866d45"
                    draw.point((x, y), fill=color)

            if mode == "motion":
                frame_index = self._selected_frame_index()
                motion_points = moving_points_for_frame(self.sequence_result, frame_index) if frame_index is not None else ()
                for track, point in motion_points:
                    point_x = point.x * self.preview_scale_x
                    point_y = point.y * self.preview_scale_y
                    if not (left <= point_x < right and top <= point_y < bottom):
                        continue
                    x = (point_x - left) * scale
                    y = (point_y - top) * scale
                    draw.ellipse((x - 7, y - 7, x + 7, y + 7), outline=SKY_LIGHT, width=2)
                    draw.point((x, y), fill=SKY_LIGHT)
                    self._draw_image_label(draw, f"MOV {track.track_id:04d}", x, y, image.size)
                feature_points = motion_features_for_frame(self.sequence_result, frame_index) if frame_index is not None else ()
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
                    color = SKY_LIGHT if track.classification == "moving" else AMBER_LIGHT
                    line_width = max(2, int(round(point.width_px * scale)))
                    draw.line((x - dx, y - dy, x + dx, y + dy), fill=color, width=line_width)
                    draw.ellipse((x - 6, y - 6, x + 6, y + 6), outline=color, width=2)
                    label = f"TRAIL {track.track_id:04d}" if track.classification == "moving" else f"TRAIL? {track.track_id:04d}"
                    self._draw_image_label(draw, label, x, y, image.size)

            if self.analysis is not None and mode == "quality" and self.analysis.faintest is not None:
                faintest = self.analysis.faintest
                point_x = faintest.x * self.preview_scale_x
                point_y = faintest.y * self.preview_scale_y
                if left <= point_x < right and top <= point_y < bottom:
                    x = (point_x - left) * scale
                    y = (point_y - top) * scale
                    draw.ellipse((x - 8, y - 8, x + 8, y + 8), outline=MINT, width=2)
                    self._draw_image_label(draw, f"FNT {faintest.detection_id:04d} · m_inst {faintest.instrumental_magnitude:.2f}", x, y, image.size)

            if self.analysis is not None and mode in {"quality", "candidates"} and self.hover_source is not None:
                source = self.hover_source
                point_x = source.x * self.preview_scale_x
                point_y = source.y * self.preview_scale_y
                if left <= point_x < right and top <= point_y < bottom:
                    x = (point_x - left) * scale
                    y = (point_y - top) * scale
                    draw.ellipse((x - 9, y - 9, x + 9, y + 9), outline="#f3dfac", width=2)
                    magnitude = instrumental_magnitude(source.flux, exposure_s=self.exposure_s)
                    magnitude_text = f"{magnitude:.2f}" if magnitude is not None else "—"
                    self._draw_image_label(draw, f"ID {source.detection_id:04d} · m_inst {magnitude_text}", x, y, image.size)
            if mode == "motion" and self.hover_motion_track is not None:
                track, point = self.hover_motion_track
                point_x = point.x * self.preview_scale_x
                point_y = point.y * self.preview_scale_y
                if left <= point_x < right and top <= point_y < bottom:
                    x = (point_x - left) * scale
                    y = (point_y - top) * scale
                    draw.ellipse((x - 10, y - 10, x + 10, y + 10), outline="#f3dfac", width=2)
                    if isinstance(point, MotionFeaturePoint):
                        self._draw_image_label(draw, f"TRAIL {track.track_id:04d} · {point.residual_snr:.1f}σ", x, y, image.size)
                    else:
                        self._draw_image_label(draw, f"MOV {track.track_id:04d} · {track.displacement_px:.2f}px", x, y, image.size)

        screen_x = origin_x + left * scale
        screen_y = origin_y + top * scale
        self.preview_photo = ImageTk.PhotoImage(image)
        self.canvas.delete("all")
        self.canvas.create_image(screen_x, screen_y, image=self.preview_photo, anchor="nw")

    def _draw_image_label(self, draw: ImageDraw.ImageDraw, text: str, x: float, y: float, image_size: tuple[int, int]) -> None:
        font = ImageFont.load_default()
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
        self.preview_zoom = min(8.0, max(0.35, self.preview_zoom * factor))
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
                    point_x = origin_x + source.x * scale_x * scale
                    point_y = origin_y + source.y * scale_y * scale
                    distance = (point_x - event.x) ** 2 + (point_y - event.y) ** 2
                    if distance <= best_distance:
                        best = source
                        best_distance = distance
        return best

    def _find_motion_track_at(
        self,
        event: tk.Event,
    ) -> tuple[SourceTrack | MotionFeatureTrack, TrackPoint | MotionFeaturePoint] | None:
        if self.preview is None or self.preview_shape is None:
            return None
        frame_index = self._selected_frame_index()
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
        for track, point in motion_features_for_frame(self.sequence_result, frame_index):
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

    def _on_canvas_motion(self, event: tk.Event) -> None:
        if self.overlay_mode_var.get() == "motion":
            track_point = self._find_motion_track_at(event)
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
            self.hover_source = None
            self.hover_source_id = None
            if track_point is None:
                self.hover_info_var.set("运动层只显示跨帧线状候选和严格 moving 点轨迹；将鼠标移到蓝线或橙线查看信息")
            else:
                track, point = track_point
                if isinstance(point, MotionFeaturePoint):
                    fit_text = f"{track.fit_rms_px:.2f}px" if track.fit_rms_px is not None else "—"
                    state_text = "moving" if track.classification == "moving" else "单帧候选"
                    self.hover_info_var.set(
                        f"TRAIL {track.track_id:04d}  ·  {state_text}  ·  当前帧 {point.frame_index + 1:02d}  ·  "
                        f"X {point.x:.1f}  Y {point.y:.1f}  ·  residual SNR {point.residual_snr:.1f}  ·  "
                        f"长度 {point.length_px:.1f}px  / 宽度 {point.width_px:.1f}px  ·  "
                        f"角度 {point.angle_deg:.1f}°  ·  位移 {track.displacement_px:.1f}px  ·  "
                        f"拟合 RMS {fit_text}  ·  出现 {track.presence} 帧"
                    )
                else:
                    fit_text = f"{track.fit_rms_px:.3f}px" if track.fit_rms_px is not None else "—"
                    snr_text = f"{point.flux_snr:.1f}" if point.flux_snr is not None else "—"
                    self.hover_info_var.set(
                        f"MOV {track.track_id:04d}  ·  当前帧 {point.frame_index + 1:02d}  ·  "
                        f"X {point.x:.1f}  Y {point.y:.1f}  ·  flux SNR {snr_text}  ·  "
                        f"位移 {track.displacement_px:.2f}px  ·  速度 {track.speed_px_per_frame:.3f}px/frame  ·  "
                        f"拟合 RMS {fit_text}  ·  出现 {track.presence} 帧"
                    )
            self._draw_preview()
            return
        source = self._find_source_at(event)
        source_id = source.detection_id if source is not None else None
        if source_id == self.hover_source_id:
            return
        self.hover_source_id = source_id
        self.hover_source = source
        if source is None:
            self.hover_info_var.set("将鼠标移到候选点查看坐标、通量、误差、SNR、形状和仪器星等")
        else:
            magnitude = instrumental_magnitude(source.flux, exposure_s=self.exposure_s)
            magnitude_text = f"{magnitude:.3f}" if magnitude is not None else "—"
            flags = ", ".join(source.flags) if source.flags else "无"
            signal_snr = source.flux_snr if source.flux_snr is not None else source.snr
            error_text = f"{source.flux_error:.1f}" if source.flux_error is not None else "—"
            shape_text = f"FWHM {source.fwhm:.2f}px / e {source.ellipticity:.2f}" if source.fwhm is not None and source.ellipticity is not None else "shape —"
            quality_text = "可信" if source.quality_passed else "剔除"
            self.hover_info_var.set(f"ID {source.detection_id:04d}  ·  X {source.x:.1f}  Y {source.y:.1f}  ·  flux {source.flux:.1f} ADU ± {error_text}  ·  flux SNR {signal_snr:.1f}  ·  {shape_text}  ·  {quality_text}  ·  m_inst {magnitude_text}  ·  flags {flags}")
        self._draw_preview()

    def _clear_hover(self) -> None:
        if self.hover_source_id is None and self.hover_motion_track is None:
            return
        self.hover_source_id = None
        self.hover_source = None
        self.hover_motion_track = None
        self.hover_info_var.set("将鼠标移到候选点查看坐标、通量、误差、SNR、形状和仪器星等")
        self._draw_preview()

    def clear_detection_cache(self) -> None:
        if not messagebox.askyesno("清空检测缓存", f"删除本地检测缓存？\n\n{self.cache_dir}\n\n不会删除 FITS 原图。"):
            return
        with self.cache_lock:
            self.cache_generation += 1
            removed = clear_cache(self.cache_dir)
        self.analysis = None
        self.sequence_result = None
        self.source_grid = {}
        self.hover_source = None
        self.hover_source_id = None
        self.hover_motion_track = None
        self._reset_result_widgets()
        self._draw_preview()
        running_note = " · 当前任务仍在运行，完成结果不会写回缓存" if self.busy else ""
        self.status_var.set(f"已清空 {removed} 个缓存文件 · 原图未删除{running_note}")

    def _on_close(self) -> None:
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
