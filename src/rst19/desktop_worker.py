"""One isolated native-desktop job per process; stdout is strictly JSONL.

The worker invokes the existing public command parsers and scientific routines.
It deliberately does not import Tkinter or change the original FITS data.
"""
from __future__ import annotations

import argparse
import contextlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import traceback

PROTOCOL = sys.stdout
if hasattr(PROTOCOL, "reconfigure"):
    PROTOCOL.reconfigure(encoding="utf-8", errors="strict")
if hasattr(sys.stdin, "reconfigure"):
    sys.stdin.reconfigure(encoding="utf-8", errors="strict")


def emit(kind: str, **fields: object) -> None:
    PROTOCOL.write(json.dumps({"event": kind, **fields}, ensure_ascii=False, allow_nan=False, default=str) + "\n")
    PROTOCOL.flush()


NATIVE_TOOL_NAMES = frozenset(
    {
        "rst19",
        "rst19-format-audit",
        "rst19-feature-sequence-compare",
        "rst19-feature-raw-evidence-audit",
        "rst19-signed-null-overlap",
        "rst19-signed-null-audit",
        "rst19-signed-null-sequence",
        "rst19-aux-audit",
        "rst19-feature-flag-profile",
        "rst19-feature-flag-interaction",
        "rst19-feature-gate-margin",
        "rst19-feature-effect-size",
        "rst19-feature-class-context",
        "rst19-pair-mechanism-context",
        "rst19-pair-mechanism-sensitivity",
        "rst19-pair-pixel-topology",
        "rst19-pair-support-audit",
        "rst19-aperture-peak-ratio",
        "rst19-wcs-validation",
        "rst19-feature-correlation",
        "rst19-feature-transition-flow",
        "rst19-feature-gate-route",
        "rst19-feature-parameter-sensitivity",
        "rst19-feature-cross",
        "rst19-feature-hard-negative",
        "rst19-gaia-merge",
        "rst19-feature-evidence-axes",
        "rst19-direct-neighbor-audit",
        "rst19-feature-cross-axis-audit",
        "rst19-source-group-audit",
        "rst19-source-group-sensitivity",
        "rst19-source-separation-audit",
        "rst19-source-peak-consistency",
        "rst19-source-proposal-peak-audit",
        "rst19-source-proposal-temporal-audit",
        "rst19-source-geometry-audit",
        "rst19-innovation-package",
        "rst19-temporal-codes",
        "rst19-sweep",
        "rst19-feature-psf-leaveout",
        "rst19-relative-photometry",
        "rst19-photometric-report",
        "rst19-mosaic",
    }
)


def _schema_registry() -> dict[str, str]:
    root = Path(__file__).resolve().parents[2]
    if (root / "pyproject.toml").is_file():
        import tomllib
        entries = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]["scripts"]
    else:
        entries = {ep.name: ep.value for ep in importlib.metadata.distribution("rst19-sol").entry_points
                   if ep.group == "console_scripts"}
    # Keep the public argparse schema available for the native CLI's desktop
    # form while the actual calculation stays in Rust.
    entries.setdefault("rst19", "rst19.cli:main")
    # `rst19` is a Rust executable. The Python parser is only used below to
    # describe its desktop form; Python worker execution is refused.
    # The format audit is Rust-only. Keep its form contract here until the
    # remaining Python catalog bootstrap is replaced by native Rust metadata.
    entries.setdefault("rst19-format-audit", "native:rst19-format-audit")
    entries.setdefault("rst19-feature-sequence-compare", "native:rst19-feature-sequence-compare")
    entries.setdefault("rst19-feature-raw-evidence-audit", "native:rst19-feature-raw-evidence-audit")
    entries.setdefault("rst19-signed-null-overlap", "native:rst19-signed-null-overlap")
    entries.setdefault("rst19-signed-null-audit", "native:rst19-signed-null-audit")
    entries.setdefault("rst19-signed-null-sequence", "native:rst19-signed-null-sequence")
    entries.setdefault("rst19-gaia-merge", "native:rst19-gaia-merge")
    entries.setdefault("rst19-relative-photometry", "native:rst19-relative-photometry")
    entries.setdefault("rst19-wcs-validation", "native:rst19-wcs-validation")
    entries.setdefault("rst19-aux-audit", "native:rst19-aux-audit")
    entries.setdefault("rst19-source-geometry-audit", "native:rst19-source-geometry-audit")
    entries.setdefault("rst19-innovation-package", "native:rst19-innovation-package")
    entries.setdefault("rst19-temporal-codes", "native:rst19-temporal-codes")
    entries.setdefault("rst19-photometric-report", "native:rst19-photometric-report")
    entries.setdefault("rst19-pair-support-audit", "native:rst19-pair-support-audit")
    entries.setdefault("rst19-feature-parameter-sensitivity", "native:rst19-feature-parameter-sensitivity")
    entries.setdefault("rst19-feature-cross", "native:rst19-feature-cross")
    return {name: target for name, target in entries.items()
            if target not in ("rst19.gui:main", "rst19.desktop_launcher:main")}


def registry() -> dict[str, str]:
    entries = _schema_registry()
    for name in NATIVE_TOOL_NAMES:
        entries[name] = f"native:{name}"
    return entries


def module_for(name: str):
    target = registry().get(name)
    if target is None:
        raise ValueError(f"未知研究工具：{name}")
    if target.startswith("native:"):
        raise ValueError(f"{name} 已迁移到 Rust，必须由 native worker 执行")
    return importlib.import_module(target.split(":")[0])


def catalog() -> list[dict]:
    tools = []
    schema_entries = _schema_registry()
    entries = registry()
    for name, target in entries.items():
        if target == "native:rst19-format-audit":
            tools.append({"name": name,
                          "description": "审计 FITS 尺寸、字节序、特殊值和辅助位置—速度自洽性，不修改检测器",
                          "fields": [
                              {"key": "input_path", "flag": "", "help": "单个 FITS 或包含 FITS 的目录",
                               "required": True, "default": None, "choices": [], "kind": "value",
                               "nargs": None, "path": True, "repeat": False},
                              {"key": "extreme_fraction", "flag": "--extreme-fraction",
                               "help": "统计有符号 16 bit 正负极值的相对阈值，默认 0.9",
                               "required": False, "default": 0.9, "choices": [], "kind": "value",
                               "nargs": None, "path": False, "repeat": False},
                              {"key": "out_dir", "flag": "--out-dir", "help": "产物输出目录",
                               "required": False, "default": "tmp/format-audit", "choices": [],
                               "kind": "value", "nargs": None, "path": True, "repeat": False},
                          ]})
            continue
        if target == "native:rst19-feature-sequence-compare":
            tools.append({"name": name,
                          "description": "比较多组已生成的序列特征持久性 JSON，不重新运行 FITS 检测",
                          "fields": [
                              {"key": "run", "flag": "--run", "help": "运行标签和 JSON 路径，NAME=JSON；至少两组",
                               "required": True, "default": None, "choices": [], "kind": "value",
                               "nargs": None, "path": False, "repeat": True},
                              {"key": "out_dir", "flag": "--out-dir", "help": "比较表输出目录",
                               "required": False, "default": "tmp/sequence-feature-parameter-comparison",
                               "choices": [], "kind": "value", "nargs": None, "path": True,
                               "repeat": False},
                          ]})
            continue
        if target == "native:rst19-feature-raw-evidence-audit":
            tools.append({"name": name,
                          "description": "连接源目录与原始 FITS 抽样 CSV，比较各特征类别的局部证据",
                          "fields": [
                              {"key": "catalog", "flag": "", "help": "rst19-sources 导出的 source_catalog.csv",
                               "required": True, "default": None, "choices": [], "kind": "value",
                               "nargs": None, "path": True, "repeat": False},
                              {"key": "raw_summary", "flag": "", "help": "stratified_raw_source_summary.csv",
                               "required": True, "default": None, "choices": [], "kind": "value",
                               "nargs": None, "path": True, "repeat": False},
                              {"key": "feature_class", "flag": "--feature-class",
                               "help": "需要汇总的 feature_class；可重复传入",
                               "required": False, "default": [], "choices": [], "kind": "value",
                               "nargs": None, "path": False, "repeat": True},
                              {"key": "out_dir", "flag": "--out-dir", "help": "CSV/JSON 输出目录",
                               "required": False, "default": "tmp/feature-raw-evidence-audit",
                               "choices": [], "kind": "value", "nargs": None, "path": True,
                               "repeat": False},
                          ]})
            continue
        if target == "native:rst19-signed-null-overlap":
            tools.append({"name": name,
                          "description": "逐帧查找符号反相质量源在原图正向检测中的局部近邻",
                          "fields": [
                              {"key": "inputs", "flag": "", "help": "FITS 文件或包含 FITS 的目录",
                               "required": True, "default": None, "choices": [], "kind": "value",
                               "nargs": "+", "path": True, "repeat": False},
                              {"key": "reverse_source_csv", "flag": "--reverse-source-csv",
                               "help": "signed_null_sequence_quality_sources.csv",
                               "required": True, "default": None, "choices": [], "kind": "value",
                               "nargs": None, "path": True, "repeat": False},
                              {"key": "threshold_sigma", "flag": "--threshold-sigma", "help": "检测阈值 sigma",
                               "required": False, "default": 4.0, "choices": [], "kind": "value",
                               "nargs": None, "path": False, "repeat": False},
                              {"key": "min_distance", "flag": "--min-distance", "help": "候选峰最小间距",
                               "required": False, "default": 4, "choices": [], "kind": "value",
                               "nargs": None, "path": False, "repeat": False},
                              {"key": "aperture_radius", "flag": "--aperture-radius", "help": "测光孔径半径",
                               "required": False, "default": 4, "choices": [], "kind": "value",
                               "nargs": None, "path": False, "repeat": False},
                              {"key": "psf_fwhm", "flag": "--psf-fwhm", "help": "检测器 PSF FWHM",
                               "required": False, "default": 2.0, "choices": [], "kind": "value",
                               "nargs": None, "path": False, "repeat": False},
                              {"key": "background_box_size", "flag": "--background-box-size", "help": "背景块尺寸",
                               "required": False, "default": 128, "choices": [], "kind": "value",
                               "nargs": None, "path": False, "repeat": False},
                              {"key": "background_sample_limit", "flag": "--background-sample-limit", "help": "背景采样上限",
                               "required": False, "default": 100000, "choices": [], "kind": "value",
                               "nargs": None, "path": False, "repeat": False},
                              {"key": "min_flux_snr", "flag": "--min-flux-snr", "help": "质量层最低孔径通量 SNR",
                               "required": False, "default": 5.0, "choices": [], "kind": "value",
                               "nargs": None, "path": False, "repeat": False},
                              {"key": "min_psf_support_pixels", "flag": "--min-psf-support-pixels", "help": "最低 PSF 支持像素数",
                               "required": False, "default": 3, "choices": [], "kind": "value",
                               "nargs": None, "path": False, "repeat": False},
                              {"key": "proposal_mode", "flag": "--proposal-mode", "help": "候选提议方式",
                               "required": False, "default": "hybrid", "choices": ["gaussian", "hybrid", "ensemble"],
                               "kind": "value", "nargs": None, "path": False, "repeat": False},
                              {"key": "out_dir", "flag": "--out-dir", "help": "CSV/JSON 输出目录",
                               "required": False, "default": "tmp/signed-null-positive-overlap", "choices": [],
                               "kind": "value", "nargs": None, "path": True, "repeat": False},
                          ]})
            continue
        if target == "native:rst19-signed-null-audit":
            tools.append({"name": name,
                          "description": "用真实 FITS 的背景符号反相图做源级 signed-tail leakage 对照，不输出源目录",
                          "fields": [
                              {"key": "fits_path", "flag": "", "help": "输入 FITS 文件",
                               "required": True, "default": None, "choices": [], "kind": "value",
                               "nargs": None, "path": True, "repeat": False},
                              {"key": "threshold_sigma", "flag": "--threshold-sigma", "help": "匹配滤波候选阈值 sigma",
                               "required": False, "default": 4.0, "choices": [], "kind": "value",
                               "nargs": None, "path": False, "repeat": False},
                              {"key": "min_distance", "flag": "--min-distance", "help": "候选峰最小间距 pixel",
                               "required": False, "default": 4, "choices": [], "kind": "value",
                               "nargs": None, "path": False, "repeat": False},
                              {"key": "aperture_radius", "flag": "--aperture-radius", "help": "孔径测光半径 pixel",
                               "required": False, "default": 4, "choices": [], "kind": "value",
                               "nargs": None, "path": False, "repeat": False},
                              {"key": "psf_fwhm", "flag": "--psf-fwhm", "help": "Gaussian 匹配滤波 FWHM pixel",
                               "required": False, "default": 2.0, "choices": [], "kind": "value",
                               "nargs": None, "path": False, "repeat": False},
                              {"key": "proposal_mode", "flag": "--proposal-mode", "help": "宽筛选提案方式",
                               "required": False, "default": "hybrid", "choices": ["gaussian", "hybrid", "ensemble"],
                               "kind": "value", "nargs": None, "path": False, "repeat": False},
                              {"key": "background_box_size", "flag": "--background-box-size", "help": "局部背景/RMS 网格大小 pixel",
                               "required": False, "default": 128, "choices": [], "kind": "value",
                               "nargs": None, "path": False, "repeat": False},
                              {"key": "min_flux_snr", "flag": "--min-flux-snr", "help": "质量层孔径通量 SNR 下限",
                               "required": False, "default": 5.0, "choices": [], "kind": "value",
                               "nargs": None, "path": False, "repeat": False},
                              {"key": "min_psf_support_pixels", "flag": "--min-psf-support-pixels",
                               "help": "候选峰中心 3x3 内的最少 PSF 支持像素", "required": False,
                               "default": 3, "choices": [], "kind": "value", "nargs": None,
                               "path": False, "repeat": False},
                              {"key": "keep_linear_artifacts", "flag": "--keep-linear-artifacts",
                               "help": "不标记线状结构为 LINE_ARTIFACT", "required": False,
                               "default": False, "choices": [], "kind": "bool", "nargs": 0,
                               "path": False, "toggle_value": True, "repeat": False},
                              {"key": "snr_thresholds", "flag": "--snr-thresholds", "help": "输出 SNR 分档下限，必须严格递增",
                               "required": False, "default": [4.0, 5.0, 7.0, 10.0, 15.0, 20.0],
                               "choices": [], "kind": "value", "nargs": "+", "path": False,
                               "repeat": False},
                              {"key": "out_dir", "flag": "--out-dir", "help": "JSON/CSV 输出目录",
                               "required": False, "default": "tmp/signed-null-audit", "choices": [],
                               "kind": "value", "nargs": None, "path": True, "repeat": False},
                          ]})
            continue
        if target == "native:rst19-signed-null-sequence":
            tools.append({"name": name,
                          "description": "逐帧运行真实 FITS 的符号反相 signed-tail leakage 对照，可传目录或多个 FITS 文件",
                          "fields": [
                              {"key": "inputs", "flag": "", "help": "FITS 文件或包含 FITS 文件的目录",
                               "required": True, "default": None, "choices": [], "kind": "value",
                               "nargs": "+", "path": True, "repeat": False},
                              {"key": "threshold_sigma", "flag": "--threshold-sigma", "help": "匹配滤波候选阈值 sigma",
                               "required": False, "default": 4.0, "choices": [], "kind": "value",
                               "nargs": None, "path": False, "repeat": False},
                              {"key": "min_distance", "flag": "--min-distance", "help": "候选峰最小间距 pixel",
                               "required": False, "default": 4, "choices": [], "kind": "value",
                               "nargs": None, "path": False, "repeat": False},
                              {"key": "aperture_radius", "flag": "--aperture-radius", "help": "孔径测光半径 pixel",
                               "required": False, "default": 4, "choices": [], "kind": "value",
                               "nargs": None, "path": False, "repeat": False},
                              {"key": "psf_fwhm", "flag": "--psf-fwhm", "help": "Gaussian 匹配滤波 FWHM pixel",
                               "required": False, "default": 2.0, "choices": [], "kind": "value",
                               "nargs": None, "path": False, "repeat": False},
                              {"key": "proposal_mode", "flag": "--proposal-mode", "help": "宽筛选提案方式",
                               "required": False, "default": "hybrid", "choices": ["gaussian", "hybrid", "ensemble"],
                               "kind": "value", "nargs": None, "path": False, "repeat": False},
                              {"key": "background_box_size", "flag": "--background-box-size", "help": "局部背景/RMS 网格大小 pixel",
                               "required": False, "default": 128, "choices": [], "kind": "value",
                               "nargs": None, "path": False, "repeat": False},
                              {"key": "min_flux_snr", "flag": "--min-flux-snr", "help": "质量层孔径通量 SNR 下限",
                               "required": False, "default": 5.0, "choices": [], "kind": "value",
                               "nargs": None, "path": False, "repeat": False},
                              {"key": "min_psf_support_pixels", "flag": "--min-psf-support-pixels",
                               "help": "候选峰中心 3x3 内的最少 PSF 支持像素", "required": False,
                               "default": 3, "choices": [], "kind": "value", "nargs": None,
                               "path": False, "repeat": False},
                              {"key": "keep_linear_artifacts", "flag": "--keep-linear-artifacts",
                               "help": "不标记线状结构为 LINE_ARTIFACT", "required": False,
                               "default": False, "choices": [], "kind": "bool", "nargs": 0,
                               "path": False, "toggle_value": True, "repeat": False},
                              {"key": "snr_thresholds", "flag": "--snr-thresholds", "help": "输出 SNR 分档下限，必须严格递增",
                               "required": False, "default": [4.0, 5.0, 7.0, 10.0, 15.0, 20.0],
                               "choices": [], "kind": "value", "nargs": "+", "path": False,
                               "repeat": False},
                              {"key": "out_dir", "flag": "--out-dir", "help": "JSON/CSV 输出目录",
                               "required": False, "default": "tmp/signed-null-sequence", "choices": [],
                               "kind": "value", "nargs": None, "path": True, "repeat": False},
                          ]})
            continue
        if target == "native:rst19-gaia-merge":
            tools.append({"name": name, "description": "合并分层 Gaia DR3 CSV，并继承输入完整性审计",
                          "fields": [
                              {"key": "catalogs", "flag": "", "help": "至少两份 CSV；通常一份亮星定标层、一份深星层",
                               "required": True, "default": None, "choices": [], "kind": "value",
                               "nargs": "+", "path": True, "repeat": False},
                              {"key": "out", "flag": "--out", "help": "合并后的 CatalogSource 兼容 CSV",
                               "required": True, "default": None, "choices": [], "kind": "value",
                               "nargs": None, "path": True, "repeat": False},
                          ]})
            continue
        if target == "native:rst19-relative-photometry":
            tools.append({"name": name, "description": "对观测行拟合相对星等标尺，不联网、不产生绝对星等",
                          "fields": [
                              {"key": "input", "flag": "", "help": "JSON 数组，或包含 observations/rows/items 的 JSON",
                               "required": True, "default": None, "choices": [], "kind": "value",
                               "nargs": None, "path": True, "repeat": False},
                              {"key": "out", "flag": "--out", "help": "可选结果 JSON；不提供时输出到 stdout",
                               "required": False, "default": None, "choices": [], "kind": "value",
                               "nargs": None, "path": True, "repeat": False},
                              {"key": "spatial_order", "flag": "--spatial-order", "help": "空间响应项阶数",
                               "required": False, "default": 0, "choices": [0, 1, 2], "kind": "value",
                               "nargs": None, "path": False, "repeat": False},
                              {"key": "validation_fraction", "flag": "--validation-fraction", "help": "源分块留出比例，默认 0.2",
                               "required": False, "default": 0.2, "choices": [], "kind": "value",
                               "nargs": None, "path": False, "repeat": False},
                              {"key": "random_state", "flag": "--random-state", "help": "留出分块随机种子",
                               "required": False, "default": 0, "choices": [], "kind": "value",
                               "nargs": None, "path": False, "repeat": False},
                              {"key": "mad_threshold", "flag": "--mad-threshold", "help": "迭代 MAD 剔除阈值",
                               "required": False, "default": 5.0, "choices": [], "kind": "value",
                               "nargs": None, "path": False, "repeat": False},
                          ]})
            continue
        if target == "native:rst19-wcs-validation":
            fields = [
                ("data_dir", "", "包含 FITS 序列的目录", True, None, [], True, False),
                ("catalog_csv", "", "用户提供的离线 CSV 星表", True, None, [], True, False),
                ("center_ra", "--center-ra", "先验光轴 RA / deg", True, None, [], False, False),
                ("center_dec", "--center-dec", "先验光轴 DEC / deg", True, None, [], False, False),
                ("pixel_scale", "--pixel-scale", "先验像元角尺度 / arcsec px^-1", True, None, [], False, False),
                ("crpix_x", "--crpix-x", "先验 CRPIX X，默认按 4096 图像中心", False, 2048.0, [], False, False),
                ("crpix_y", "--crpix-y", "先验 CRPIX Y，默认按 4096 图像中心", False, 2048.0, [], False, False),
                ("rotation", "--rotation", "先验旋转 / deg", False, 0.0, [], False, False),
                ("parity", "--parity", "RA 轴奇偶性", False, 1, [-1, 1], False, False),
                ("match_radius", "--match-radius", "先验 WCS 匹配半径 / px", False, 3.0, [], False, False),
                ("min_matches", "--min-matches", "每帧最少唯一匹配点数", False, 6, [], False, False),
                ("epoch", "--epoch", "星表传播目标历元，Julian year", False, None, [], False, False),
                ("threshold_sigma", "--threshold-sigma", "匹配滤波候选阈值 sigma", False, 4.0, [], False, False),
                ("min_distance", "--min-distance", "候选峰最小间距 pixel", False, 4, [], False, False),
                ("aperture_radius", "--aperture-radius", "孔径测光半径 pixel", False, 4, [], False, False),
                ("psf_fwhm", "--psf-fwhm", "Gaussian 匹配滤波 FWHM pixel", False, 3.0, [], False, False),
                ("background_box_size", "--background-box-size", "局部背景/RMS 网格大小 pixel", False, 128, [], False, False),
                ("min_flux_snr", "--min-flux-snr", "质量层通量 SNR 下限", False, 5.0, [], False, False),
                ("max_sources", "--max-sources", "可选返回源上限；留空表示全量", False, None, [], False, False),
            ]
            schema = [{"key": key, "flag": flag, "help": help_text, "required": required,
                       "default": default, "choices": choices, "kind": "value", "nargs": None,
                       "path": is_path, "repeat": repeat}
                      for key, flag, help_text, required, default, choices, is_path, repeat in fields]
            schema.extend([
                {"key": "keep_linear_artifacts", "flag": "--keep-linear-artifacts",
                 "help": "不标记线状结构为 LINE_ARTIFACT", "required": False,
                 "default": False, "choices": [], "kind": "bool", "nargs": 0,
                 "path": False, "toggle_value": True, "repeat": False},
                {"key": "out_dir", "flag": "--out-dir", "help": "CSV/JSON/PNG 输出目录",
                 "required": False, "default": "tmp/wcs-validation", "choices": [],
                 "kind": "value", "nargs": None, "path": True, "repeat": False},
            ])
            tools.append({"name": name, "description": "使用离线星表，对 FITS 序列做逐帧局部 WCS 验证",
                          "fields": schema})
            continue
        if target == "native:rst19-aux-audit":
            tools.append({"name": name, "description": "审计 FITS 辅助姿态四元数与 ra/dec 光轴的一致性",
                          "fields": [
                              {"key": "input_path", "flag": "", "help": "单个 FITS 或包含 FITS 的目录",
                               "required": True, "default": None, "choices": [], "kind": "value",
                               "nargs": None, "path": True, "repeat": False},
                              {"key": "out_dir", "flag": "--out-dir", "help": "CSV/JSON 输出目录",
                               "required": False, "default": "tmp/auxiliary-attitude-audit", "choices": [],
                               "kind": "value", "nargs": None, "path": True, "repeat": False},
                          ]})
            continue
        if target == "native:rst19-source-geometry-audit":
            tools.append({"name": name, "description": "只读取 source_catalog，审计各特征类别的 detector 坐标细长性",
                          "fields": [
                              {"key": "catalog", "flag": "", "help": "rst19-sources 导出的 source_catalog.csv",
                               "required": True, "default": None, "choices": [], "kind": "value",
                               "nargs": None, "path": True, "repeat": False},
                              {"key": "target_class", "flag": "--target-class", "help": "需要审计的 feature_class；可重复传入，默认审计全部类别",
                               "required": False, "default": [], "choices": [], "kind": "value",
                               "nargs": None, "path": False, "repeat": True},
                              {"key": "control_class", "flag": "--control-class", "help": "等样本量抽样对照类别，默认 compact_quality",
                               "required": False, "default": "compact_quality", "choices": [],
                               "kind": "value", "nargs": None, "path": False, "repeat": False},
                              {"key": "trials", "flag": "--trials", "help": "每个类别的对照抽样次数",
                               "required": False, "default": 5000, "choices": [], "kind": "value",
                               "nargs": None, "path": False, "repeat": False},
                              {"key": "seed", "flag": "--seed", "help": "对照抽样随机种子",
                               "required": False, "default": 1909, "choices": [], "kind": "value",
                               "nargs": None, "path": False, "repeat": False},
                              {"key": "out_dir", "flag": "--out-dir", "help": "CSV/JSON 输出目录",
                               "required": False, "default": "tmp/source-geometry-audit", "choices": [],
                               "kind": "value", "nargs": None, "path": True, "repeat": False},
                          ]})
            continue
        if target == "native:rst19-innovation-package":
            tools.append({"name": name,
                          "description": "从真实背景分层注入证据生成条件化创新交付包",
                          "fields": [
                              {"key": "injection_artifact", "flag": "", "help": "分层注入结果 JSON 或 CSV",
                               "required": True, "default": None, "choices": [], "kind": "value",
                               "nargs": None, "path": True, "repeat": False},
                              {"key": "sequence_report", "flag": "--sequence-report", "help": "可选序列报告 JSON",
                               "required": False, "default": None, "choices": [], "kind": "value",
                               "nargs": None, "path": True, "repeat": False},
                              {"key": "feature_matrix", "flag": "--feature-matrix", "help": "可选特征证据矩阵 JSON",
                               "required": False, "default": None, "choices": [], "kind": "value",
                               "nargs": None, "path": True, "repeat": False},
                              {"key": "out_dir", "flag": "--out-dir", "help": "JSON/CSV/Markdown 输出目录",
                               "required": False, "default": "tmp/innovation-package", "choices": [],
                               "kind": "value", "nargs": None, "path": True, "repeat": False},
                          ]})
            continue
        if target == "native:rst19-temporal-codes":
            tools.append({"name": name,
                          "description": "逐帧比较 FITS detector 固定值、低变化结构和指定焦点坐标",
                          "fields": [
                              {"key": "data_dir", "flag": "", "help": "包含 FITS 序列的目录",
                               "required": True, "default": None, "choices": [], "kind": "value",
                               "nargs": None, "path": True, "repeat": False},
                              {"key": "low_variation_span_adu", "flag": "--low-variation-span-adu",
                               "help": "低变化像素允许的跨帧最大范围（ADU）", "required": False,
                               "default": 2, "choices": [], "kind": "value", "nargs": None,
                               "path": False, "repeat": False},
                              {"key": "code_min_adu", "flag": "--code-min-adu", "help": "低变化正码候选下限",
                               "required": False, "default": 3990, "choices": [], "kind": "value",
                               "nargs": None, "path": False, "repeat": False},
                              {"key": "code_max_adu", "flag": "--code-max-adu", "help": "低变化正码候选上限",
                               "required": False, "default": 3993, "choices": [], "kind": "value",
                               "nargs": None, "path": False, "repeat": False},
                              {"key": "sentinel_value", "flag": "--sentinel-value", "help": "固定 sentinel 值",
                               "required": False, "default": -1, "choices": [], "kind": "value",
                               "nargs": None, "path": False, "repeat": False},
                              {"key": "focus_x", "flag": "--focus-x", "help": "正码候选关注坐标 X",
                               "required": False, "default": 1274, "choices": [], "kind": "value",
                               "nargs": None, "path": False, "repeat": False},
                              {"key": "focus_y", "flag": "--focus-y", "help": "正码候选关注坐标 Y",
                               "required": False, "default": 3466, "choices": [], "kind": "value",
                               "nargs": None, "path": False, "repeat": False},
                              {"key": "sentinel_focus_x", "flag": "--sentinel-focus-x", "help": "sentinel 关注坐标 X",
                               "required": False, "default": 1271, "choices": [], "kind": "value",
                               "nargs": None, "path": False, "repeat": False},
                              {"key": "sentinel_focus_y", "flag": "--sentinel-focus-y", "help": "sentinel 关注坐标 Y",
                               "required": False, "default": 3465, "choices": [], "kind": "value",
                               "nargs": None, "path": False, "repeat": False},
                              {"key": "out_dir", "flag": "--out-dir", "help": "JSON/CSV 输出目录",
                               "required": False, "default": "tmp/temporal-code-audit", "choices": [],
                               "kind": "value", "nargs": None, "path": True, "repeat": False},
                          ]})
            continue
        if target == "native:rst19-photometric-report":
            tools.append({"name": name,
                          "description": "审计 rst19 星等 JSON：区分仪器、相对、表观和绝对星等证据",
                          "fields": [
                              {"key": "input", "flag": "", "help": "rst19 JSON 结果或包含 source_rows/frame_rows 的 JSON",
                               "required": True, "default": None, "choices": [], "kind": "value",
                               "nargs": None, "path": True, "repeat": False},
                              {"key": "out", "flag": "--out", "help": "可选审计报告 JSON；不提供时输出到 stdout",
                               "required": False, "default": None, "choices": [], "kind": "value",
                               "nargs": None, "path": True, "repeat": False},
                              {"key": "require_wcs", "flag": "--require-wcs", "help": "把 WCS 作为强制几何条件",
                               "required": False, "default": False, "choices": [], "kind": "bool",
                               "nargs": 0, "path": False, "toggle_value": True, "repeat": False},
                              {"key": "no_require_wcs", "flag": "--no-require-wcs", "help": "不把 WCS 作为强制几何条件",
                               "required": False, "default": False, "choices": [], "kind": "bool",
                               "nargs": 0, "path": False, "toggle_value": True, "repeat": False},
                              {"key": "strict", "flag": "--strict", "help": "发现 false-valid 矛盾时返回失败",
                               "required": False, "default": False, "choices": [], "kind": "bool",
                               "nargs": 0, "path": False, "toggle_value": True, "repeat": False},
                          ]})
            continue
        if target == "native:rst19-pair-support-audit":
            definitions = [
                ("data_dir", "", True, None, True, "value", []),
                ("source_catalog", "--source-catalog", True, None, True, "value", []),
                ("primary_id", "--primary-id", True, None, False, "value", []),
                ("secondary_id", "--secondary-id", True, None, False, "value", []),
                ("mask_modes", "--mask-modes", False,
                 "raw,range_masked,repeated_code_masked,local_anomaly_masked,local_anomaly_repeated_code_masked,shared_aperture_masked,shared_local_anomaly_repeated_code_masked",
                 False, "value", []),
                ("repeated_code_values", "--repeated-code-values", False, None, False, "value", []),
                ("shifts_json", "--shifts-json", False, None, True, "value", []),
                ("crop_padding_px", "--crop-padding-px", False, 128.0, False, "value", []),
                ("aperture_radius", "--aperture-radius", False, 4, False, "value", []),
                ("match_radius_px", "--match-radius-px", False, 2.0, False, "value", []),
                ("threshold_sigma", "--threshold-sigma", False, 4.0, False, "value", []),
                ("min_distance", "--min-distance", False, 4, False, "value", []),
                ("psf_fwhm", "--psf-fwhm", False, 2.0, False, "value", []),
                ("background_box_size", "--background-box-size", False, 128, False, "value", []),
                ("min_flux_snr", "--min-flux-snr", False, 5.0, False, "value", []),
                ("min_psf_support_pixels", "--min-psf-support-pixels", False, 3, False, "value", []),
                ("proposal_mode", "--proposal-mode", False, "hybrid", False, "value", ["gaussian", "hybrid", "ensemble"]),
                ("keep_linear_artifacts", "--keep-linear-artifacts", False, False, False, "bool", []),
                ("out_dir", "--out-dir", False, "tmp/pair-support-audit", True, "value", []),
            ]
            tools.append({"name": name, "description": "在局部裁剪和留出掩膜下审计近邻双候选的独立像素支持",
                          "fields": [
                              {"key": key, "flag": flag, "help": "", "required": required,
                               "default": default, "choices": choices, "kind": kind, "nargs": 0 if kind == "bool" else None,
                               "path": is_path, "toggle_value": True if kind == "bool" else None, "repeat": False}
                              for key, flag, required, default, is_path, kind, choices in definitions
                          ]})
            continue
        if target == "native:rst19-feature-parameter-sensitivity":
            definitions = [
                ("path", "", True, None, True, "value", []),
                ("threshold_levels", "--threshold-levels", False, [4.0, 5.0, 6.0, 8.0], False, "value", []),
                ("flux_snr_levels", "--flux-snr-levels", False, [3.0, 5.0, 7.0, 9.0], False, "value", []),
                ("baseline_threshold_sigma", "--baseline-threshold-sigma", False, 4.0, False, "value", []),
                ("baseline_min_flux_snr", "--baseline-min-flux-snr", False, 5.0, False, "value", []),
                ("min_distance", "--min-distance", False, 4, False, "value", []),
                ("aperture_radius", "--aperture-radius", False, 4, False, "value", []),
                ("psf_fwhm", "--psf-fwhm", False, 2.0, False, "value", []),
                ("background_box_size", "--background-box-size", False, 128, False, "value", []),
                ("background_sample_limit", "--background-sample-limit", False, 100000, False, "value", []),
                ("min_psf_support_pixels", "--min-psf-support-pixels", False, 3, False, "value", []),
                ("proposal_mode", "--proposal-mode", False, "hybrid", False, "value", ["gaussian", "hybrid", "ensemble"]),
                ("max_sources", "--max-sources", False, None, False, "value", []),
                ("association_radius_px", "--association-radius-px", False, 1.5, False, "value", []),
                ("out_dir", "--out-dir", False, "tmp/feature-parameter-sensitivity-current", True, "value", []),
            ]
            tools.append({"name": name, "description": "固定同一真实 FITS，扫描候选阈值/flux SNR 并追踪各类候选稳定性",
                          "fields": [
                              {"key": key, "flag": flag, "help": "", "required": required,
                               "default": default, "choices": choices, "kind": kind, "nargs": None,
                               "path": is_path, "toggle_value": None, "repeat": False}
                              for key, flag, required, default, is_path, kind, choices in definitions
                          ]})
            continue
        if target == "native:rst19-feature-cross":
            definitions = [
                ("source_summary_csv", "", True, None, True, "value", []),
                ("persistence_csv", "", True, None, True, "value", []),
                ("class_transition_csv", "--class-transition-csv", False, None, True, "value", []),
                ("high_snr_threshold", "--high-snr-threshold", False, 10.0, False, "value", []),
                ("out_dir", "--out-dir", False, "tmp/feature-cross-audit", True, "value", []),
            ]
            tools.append({"name": name, "description": "连接单帧特征汇总、跨帧持久性和可选类别转移表",
                          "fields": [
                              {"key": key, "flag": flag, "help": "", "required": required,
                               "default": default, "choices": choices, "kind": kind, "nargs": None,
                               "path": is_path, "toggle_value": None, "repeat": False}
                              for key, flag, required, default, is_path, kind, choices in definitions
                          ]})
            continue
        # Rust-only commands without a Python parser are described by the
        # compiled native catalog; do not synthesize or dispatch them here.
        if target.startswith("native:") and schema_entries.get(name) is None:
            continue
        # The Rust single-frame command reuses argparse's field schema only;
        # the native worker runs the scientific calculation.
        schema_target = schema_entries.get(name)
        module = (
            importlib.import_module(schema_target.split(":")[0])
            if target.startswith("native:")
            and schema_target is not None
            and not schema_target.startswith("native:")
            else module_for(name)
        )
        parser = module.build_parser()
        fields = []
        for action in parser._actions:
            if isinstance(action, argparse._HelpAction):
                continue
            flags = action.option_strings
            fields.append({
                "key": action.dest, "flag": next((f for f in flags if f.startswith("--")), ""),
                "help": action.help or "", "required": action.required,
                "default": None if action.default is argparse.SUPPRESS else action.default,
                "choices": list(action.choices) if action.choices is not None else [],
                "kind": "bool" if action.nargs == 0 else "value",
                "nargs": action.nargs, "path": action.type is Path,
                "toggle_value": getattr(action, "const", True) if action.nargs == 0 else None,
                "repeat": isinstance(action, (argparse._AppendAction, argparse._AppendConstAction)),
            })
        tools.append({"name": name, "description": parser.description or name, "fields": fields})
    for tool in tools:
        tool["native"] = entries[tool["name"]].startswith("native:")
    return tools


class JobLog:
    def __init__(self, path: Path):
        self.file = path.open("w", encoding="utf-8")
        self.pending = ""

    def write(self, value: str) -> int:
        self.file.write(value)
        self.file.flush()
        self.pending += value
        while "\n" in self.pending or len(self.pending) > 4096:
            if "\n" in self.pending:
                line, self.pending = self.pending.split("\n", 1)
            else:
                line, self.pending = self.pending[:4096], self.pending[4096:]
            emit("log", text=line)
        return len(value)

    def flush(self) -> None:
        self.file.flush()

    def close(self) -> None:
        if self.pending:
            emit("log", text=self.pending)
        self.file.close()


def preview(path: str, out: Path, mode: str) -> dict:
    import numpy as np
    from PIL import Image, ImageFilter
    from .fits import read_fits, auxiliary_mask
    frame = read_fits(path)
    values = np.asarray(frame.data, dtype=np.float32).copy()
    values[auxiliary_mask(values.shape)] = np.nan
    valid = values[np.isfinite(values)]
    if valid.size == 0:
        raise ValueError("图像没有可预览的有限像素")
    lo, q25, mid, q75, hi = np.percentile(valid, [0.5, 25, 50, 75, 99.8])
    noise = max(float(q75 - q25) / 1.3489795, 1e-6)
    gamma = 1.0
    if mode == "noise":
        lo, hi, gamma = max(lo, mid - 2 * noise), mid + 7 * noise, 0.82
    elif mode == "enhanced":
        lo, hi, gamma = max(lo, mid - noise), max(hi, mid + 12 * noise), 0.7
    pixels = np.nan_to_num(np.power(np.clip((values - lo) / max(float(hi - lo), 1), 0, 1), gamma))
    image = Image.fromarray(np.rint(pixels * 255).astype(np.uint8)).convert("RGB")
    if mode == "enhanced":
        image = image.filter(ImageFilter.GaussianBlur(0.35))
    # Keep full resolution so zooming preserves original pixel locations.
    target = out / "preview.png"
    image.save(target)
    return {"preview": str(target), "width": frame.width, "height": frame.height,
            "path": str(frame.path), "header": dict(frame.header),
            "auxiliary": frame.auxiliary.as_dict() if frame.auxiliary else None}


def run(request: dict) -> dict:
    out = Path(request["out_dir"]).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if request["operation"] == "preview":
        return preview(request["path"], out, request.get("mode", "raw"))
    name = request["tool"]
    module = module_for(name)
    argv = request.get("args", [])
    if not isinstance(argv, list) or not all(isinstance(arg, str) for arg in argv):
        raise ValueError("args 必须是字符串数组")
    # argparse validates inputs before any scientific work.
    parser = module.build_parser()
    parsed = parser.parse_args(argv)
    result_path = out / "result.json"
    if any("--json-out" in a.option_strings for a in parser._actions) and "--json-out" not in argv:
        argv = [*argv, "--json-out", str(result_path)]
        parsed = parser.parse_args(argv)
    stdout_path = out / "stdout.txt"
    stderr_path = out / "stderr.txt"
    stdout, stderr = JobLog(stdout_path), JobLog(stderr_path)
    try:
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = module.main(argv)
    finally:
        stdout.close()
        stderr.close()
    if code not in (None, 0):
        raise RuntimeError(f"{name} 返回错误码 {code}；详见 {stderr_path}")
    json_out = getattr(parsed, "json_out", None)
    named_out = getattr(parsed, "out", None)
    if named_out is not None and Path(named_out).suffix.lower() != ".json":
        named_out = None
    actual_result = json_out or named_out or result_path
    payload = None
    if Path(actual_result).is_file():
        payload = json.loads(Path(actual_result).read_text(encoding="utf-8"))
    else:
        try:
            payload = json.loads(stdout_path.read_text(encoding="utf-8"))
            result_path.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        except json.JSONDecodeError:
            pass
    response = {"tool": name, "out_dir": str(out), "result_path": str(actual_result), "result": payload}
    # Only single-frame detection carries the exact image/result identity.
    if name == "rst19" and payload is not None:
        response.update(preview(str(parsed.fits_path), out, request.get("mode", "raw")))
    (out / "request.json").write_text(json.dumps(request, ensure_ascii=False, indent=2), encoding="utf-8")
    return response


def main() -> int:
    try:
        if "--catalog" in sys.argv:
            import numpy, scipy, PIL
            emit("done", ok=True, payload={"tools": catalog(), "environment": {
                "python": sys.version.split()[0], "numpy": numpy.__version__,
                "scipy": scipy.__version__, "pillow": PIL.__version__,
                "frozen": bool(getattr(sys, "frozen", False))}})
        else:
            request = json.loads(sys.stdin.readline())
            emit("done", ok=True, payload=run(request))
        return 0
    except (Exception, SystemExit) as exc:
        emit("done", ok=False, error=str(exc), traceback=traceback.format_exc())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
