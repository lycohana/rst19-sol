"""15 帧符号反相阴性对照的聚合研究。

本模块复用单帧 ``run_signed_null_audit``，逐帧运行而不把源目录或检测缓存
带入结果。聚合时同时保留每帧结果和 pooled（计数求和后的）比值，避免把
不同帧的背景、候选数和物理身份错误地压成一个平均数字。
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from .fits import read_fits
from .models import FitsFrame
from .signed_null import (
    SIGNED_NULL_SNR_THRESHOLDS,
    SignedNullAuditResult,
    SignedNullFeatureRow,
    SignedNullThresholdRow,
    run_signed_null_audit,
)


@dataclass(frozen=True, slots=True)
class SignedNullSequenceFrameRow:
    """单帧正向/反相源级计数和比值。"""

    frame_index: int
    frame_path: str
    timestamp: str | None
    background_adu: float
    positive_noise_adu: float
    negative_noise_adu: float
    positive_candidate_count: int
    positive_quality_count: int
    negative_candidate_count: int
    negative_quality_count: int
    negative_quality_count_after_range_exclusion: int
    candidate_leakage_ratio: float | None
    quality_leakage_ratio: float | None
    quality_leakage_ratio_after_range_exclusion: float | None
    negative_range_transfer_candidate_count: int
    negative_range_transfer_quality_count: int

    def as_dict(self) -> dict[str, object]:
        return {
            "frame_index": self.frame_index,
            "frame_path": self.frame_path,
            "timestamp": self.timestamp,
            "background_adu": self.background_adu,
            "positive_noise_adu": self.positive_noise_adu,
            "negative_noise_adu": self.negative_noise_adu,
            "positive_candidate_count": self.positive_candidate_count,
            "positive_quality_count": self.positive_quality_count,
            "negative_candidate_count": self.negative_candidate_count,
            "negative_quality_count": self.negative_quality_count,
            "negative_quality_count_after_range_exclusion": self.negative_quality_count_after_range_exclusion,
            "candidate_leakage_ratio": self.candidate_leakage_ratio,
            "quality_leakage_ratio": self.quality_leakage_ratio,
            "quality_leakage_ratio_after_range_exclusion": self.quality_leakage_ratio_after_range_exclusion,
            "negative_range_transfer_candidate_count": self.negative_range_transfer_candidate_count,
            "negative_range_transfer_quality_count": self.negative_range_transfer_quality_count,
        }


@dataclass(frozen=True, slots=True)
class SignedNullSequenceThresholdRow:
    """单帧某个 SNR 下限的正向/反相计数。"""

    frame_index: int
    frame_path: str
    threshold: float
    positive_filter_count: int
    negative_filter_count: int
    positive_flux_count: int
    negative_flux_count: int
    filter_leakage_ratio: float | None
    flux_leakage_ratio: float | None

    def as_dict(self) -> dict[str, object]:
        return {
            "frame_index": self.frame_index,
            "frame_path": self.frame_path,
            "threshold": self.threshold,
            "positive_filter_count": self.positive_filter_count,
            "negative_filter_count": self.negative_filter_count,
            "positive_flux_count": self.positive_flux_count,
            "negative_flux_count": self.negative_flux_count,
            "filter_leakage_ratio": self.filter_leakage_ratio,
            "flux_leakage_ratio": self.flux_leakage_ratio,
        }


@dataclass(frozen=True, slots=True)
class SignedNullSequenceFeatureRow:
    """单帧控制方向和首要特征类别的计数。"""

    frame_index: int
    frame_path: str
    control: str
    feature_class: str
    feature_class_label: str
    candidate_count: int
    quality_count: int
    quality_fraction: float | None
    median_flux_snr: float | None
    max_flux_snr: float | None
    common_flags: str

    def as_dict(self) -> dict[str, object]:
        return {
            "frame_index": self.frame_index,
            "frame_path": self.frame_path,
            "control": self.control,
            "feature_class": self.feature_class,
            "feature_class_label": self.feature_class_label,
            "candidate_count": self.candidate_count,
            "quality_count": self.quality_count,
            "quality_fraction": self.quality_fraction,
            "median_flux_snr": self.median_flux_snr,
            "max_flux_snr": self.max_flux_snr,
            "common_flags": self.common_flags,
        }


@dataclass(frozen=True, slots=True)
class SignedNullSequenceSourceRow:
    """逐帧反相质量源及其原始孔径值域证据。"""

    frame_index: int
    frame_path: str
    detection_id: int
    x: float
    y: float
    peak_x: float
    peak_y: float
    peak: float
    flux_snr: float | None
    filter_snr: float | None
    fwhm: float | None
    psf_support_pixels: int | None
    raw_peak_adu: float
    raw_negative_pixel_count: int
    raw_negative_anomaly_pixel_count: int
    raw_extreme_negative_pixel_count: int
    raw_minus_one_count: int
    raw_min_adu: float | None
    raw_max_adu: float | None
    range_transfer_overlap: bool
    raw_evidence_layer: str
    flags: str

    def as_dict(self) -> dict[str, object]:
        return {
            "frame_index": self.frame_index,
            "frame_path": self.frame_path,
            "detection_id": self.detection_id,
            "x": self.x,
            "y": self.y,
            "peak_x": self.peak_x,
            "peak_y": self.peak_y,
            "peak": self.peak,
            "flux_snr": self.flux_snr,
            "filter_snr": self.filter_snr,
            "fwhm": self.fwhm,
            "psf_support_pixels": self.psf_support_pixels,
            "raw_peak_adu": self.raw_peak_adu,
            "raw_negative_pixel_count": self.raw_negative_pixel_count,
            "raw_negative_anomaly_pixel_count": self.raw_negative_anomaly_pixel_count,
            "raw_extreme_negative_pixel_count": self.raw_extreme_negative_pixel_count,
            "raw_minus_one_count": self.raw_minus_one_count,
            "raw_min_adu": self.raw_min_adu,
            "raw_max_adu": self.raw_max_adu,
            "range_transfer_overlap": self.range_transfer_overlap,
            "raw_evidence_layer": self.raw_evidence_layer,
            "flags": self.flags,
        }


@dataclass(frozen=True, slots=True)
class SignedNullSequenceRecurrenceRow:
    """1 px 质心邻域内跨不同帧重复的反相质量源簇。"""

    cluster_id: int
    frame_indices: str
    frame_count: int
    source_count: int
    x_median: float
    y_median: float
    max_radius_px: float
    transfer_count: int
    effective_count: int
    median_flux_snr: float | None

    def as_dict(self) -> dict[str, object]:
        return {
            "cluster_id": self.cluster_id,
            "frame_indices": self.frame_indices,
            "frame_count": self.frame_count,
            "source_count": self.source_count,
            "x_median": self.x_median,
            "y_median": self.y_median,
            "max_radius_px": self.max_radius_px,
            "transfer_count": self.transfer_count,
            "effective_count": self.effective_count,
            "median_flux_snr": self.median_flux_snr,
        }


@dataclass(frozen=True, slots=True)
class SignedNullSequenceResult:
    """15 帧符号反相对照的逐帧和 pooled 结果。"""

    frame_count: int
    frame_rows: tuple[SignedNullSequenceFrameRow, ...]
    threshold_rows: tuple[SignedNullSequenceThresholdRow, ...]
    feature_rows: tuple[SignedNullSequenceFeatureRow, ...]
    pooled_positive_candidate_count: int
    pooled_positive_quality_count: int
    pooled_negative_candidate_count: int
    pooled_negative_quality_count: int
    pooled_negative_quality_count_after_range_exclusion: int
    pooled_candidate_leakage_ratio: float | None
    pooled_quality_leakage_ratio: float | None
    pooled_quality_leakage_ratio_after_range_exclusion: float | None
    pooled_negative_range_transfer_candidate_count: int
    pooled_negative_range_transfer_quality_count: int
    frame_candidate_leakage_mean: float | None
    frame_candidate_leakage_median: float | None
    frame_quality_leakage_mean: float | None
    frame_quality_leakage_median: float | None
    parameters: dict[str, object]
    conclusion: str
    quality_source_rows: tuple[SignedNullSequenceSourceRow, ...] = ()
    recurrence_rows: tuple[SignedNullSequenceRecurrenceRow, ...] = ()

    def as_dict(self) -> dict[str, object]:
        return {
            "frame_count": self.frame_count,
            "frame_row_count": len(self.frame_rows),
            "threshold_row_count": len(self.threshold_rows),
            "feature_row_count": len(self.feature_rows),
            "frame_rows": [row.as_dict() for row in self.frame_rows],
            "threshold_rows": [row.as_dict() for row in self.threshold_rows],
            "feature_rows": [row.as_dict() for row in self.feature_rows],
            "quality_source_row_count": len(self.quality_source_rows),
            "quality_source_rows": [row.as_dict() for row in self.quality_source_rows],
            "recurrence_row_count": len(self.recurrence_rows),
            "recurrence_rows": [row.as_dict() for row in self.recurrence_rows],
            "pooled_positive_candidate_count": self.pooled_positive_candidate_count,
            "pooled_positive_quality_count": self.pooled_positive_quality_count,
            "pooled_negative_candidate_count": self.pooled_negative_candidate_count,
            "pooled_negative_quality_count": self.pooled_negative_quality_count,
            "pooled_negative_quality_count_after_range_exclusion": (
                self.pooled_negative_quality_count_after_range_exclusion
            ),
            "pooled_candidate_leakage_ratio": self.pooled_candidate_leakage_ratio,
            "pooled_quality_leakage_ratio": self.pooled_quality_leakage_ratio,
            "pooled_quality_leakage_ratio_after_range_exclusion": (
                self.pooled_quality_leakage_ratio_after_range_exclusion
            ),
            "pooled_negative_range_transfer_candidate_count": self.pooled_negative_range_transfer_candidate_count,
            "pooled_negative_range_transfer_quality_count": self.pooled_negative_range_transfer_quality_count,
            "frame_candidate_leakage_mean": self.frame_candidate_leakage_mean,
            "frame_candidate_leakage_median": self.frame_candidate_leakage_median,
            "frame_quality_leakage_mean": self.frame_quality_leakage_mean,
            "frame_quality_leakage_median": self.frame_quality_leakage_median,
            "parameters": self.parameters,
            "conclusion": self.conclusion,
        }


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return float(numerator / denominator)


def _optional_summary(values: Sequence[float | None], statistic: str) -> float | None:
    finite = np.asarray([value for value in values if value is not None and np.isfinite(value)], dtype=np.float64)
    if finite.size == 0:
        return None
    if statistic == "mean":
        return float(np.mean(finite))
    if statistic == "median":
        return float(np.median(finite))
    raise ValueError(f"unsupported summary statistic: {statistic!r}")


def _as_frame_row(index: int, frame: FitsFrame, audit: SignedNullAuditResult) -> SignedNullSequenceFrameRow:
    raw_timestamp = frame.header.get("DATE-OBS")
    return SignedNullSequenceFrameRow(
        frame_index=index,
        frame_path=str(audit.frame_path),
        timestamp=None if raw_timestamp is None else str(raw_timestamp),
        background_adu=audit.background_adu,
        positive_noise_adu=audit.positive_noise_adu,
        negative_noise_adu=audit.negative_noise_adu,
        positive_candidate_count=audit.positive_candidate_count,
        positive_quality_count=audit.positive_quality_count,
        negative_candidate_count=audit.negative_candidate_count,
        negative_quality_count=audit.negative_quality_count,
        negative_quality_count_after_range_exclusion=audit.negative_quality_count_after_range_exclusion,
        candidate_leakage_ratio=audit.candidate_leakage_ratio,
        quality_leakage_ratio=audit.quality_leakage_ratio,
        quality_leakage_ratio_after_range_exclusion=audit.quality_leakage_ratio_after_range_exclusion,
        negative_range_transfer_candidate_count=audit.negative_range_transfer_candidate_count,
        negative_range_transfer_quality_count=audit.negative_range_transfer_quality_count,
    )


def _as_threshold_rows(index: int, audit: SignedNullAuditResult) -> tuple[SignedNullSequenceThresholdRow, ...]:
    return tuple(
        SignedNullSequenceThresholdRow(
            frame_index=index,
            frame_path=audit.frame_path,
            threshold=row.threshold,
            positive_filter_count=row.positive_filter_count,
            negative_filter_count=row.negative_filter_count,
            positive_flux_count=row.positive_flux_count,
            negative_flux_count=row.negative_flux_count,
            filter_leakage_ratio=row.filter_leakage_ratio,
            flux_leakage_ratio=row.flux_leakage_ratio,
        )
        for row in audit.thresholds
    )


def _as_feature_rows(index: int, audit: SignedNullAuditResult) -> tuple[SignedNullSequenceFeatureRow, ...]:
    return tuple(
        SignedNullSequenceFeatureRow(
            frame_index=index,
            frame_path=audit.frame_path,
            control=row.control,
            feature_class=row.feature_class,
            feature_class_label=row.feature_class_label,
            candidate_count=row.candidate_count,
            quality_count=row.quality_count,
            quality_fraction=row.quality_fraction,
            median_flux_snr=row.median_flux_snr,
            max_flux_snr=row.max_flux_snr,
            common_flags=row.common_flags,
        )
        for row in audit.feature_rows
    )


def _as_source_rows(index: int, audit: SignedNullAuditResult) -> tuple[SignedNullSequenceSourceRow, ...]:
    return tuple(
        SignedNullSequenceSourceRow(
            frame_index=index,
            frame_path=audit.frame_path,
            detection_id=row.detection_id,
            x=row.x,
            y=row.y,
            peak_x=row.peak_x,
            peak_y=row.peak_y,
            peak=row.peak,
            flux_snr=row.flux_snr,
            filter_snr=row.filter_snr,
            fwhm=row.fwhm,
            psf_support_pixels=row.psf_support_pixels,
            raw_peak_adu=row.raw_peak_adu,
            raw_negative_pixel_count=row.raw_negative_pixel_count,
            raw_negative_anomaly_pixel_count=row.raw_negative_anomaly_pixel_count,
            raw_extreme_negative_pixel_count=row.raw_extreme_negative_pixel_count,
            raw_minus_one_count=row.raw_minus_one_count,
            raw_min_adu=row.raw_min_adu,
            raw_max_adu=row.raw_max_adu,
            range_transfer_overlap=row.range_transfer_overlap,
            raw_evidence_layer=row.raw_evidence_layer,
            flags=row.flags,
        )
        for row in audit.negative_quality_sources
    )


def _build_recurrence_rows(
    rows: Sequence[SignedNullSequenceSourceRow],
    *,
    distance_tolerance_px: float = 1.0,
) -> tuple[SignedNullSequenceRecurrenceRow, ...]:
    """按测量质心连接跨帧近邻，保留可疑重复而不判定物理身份。"""

    if distance_tolerance_px <= 0 or not np.isfinite(float(distance_tolerance_px)):
        raise ValueError("distance_tolerance_px must be finite and positive")
    if not rows:
        return ()
    parent = list(range(len(rows)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for left_index, left in enumerate(rows):
        for right_index in range(left_index + 1, len(rows)):
            right = rows[right_index]
            if left.frame_index == right.frame_index:
                continue
            distance = float(np.hypot(left.x - right.x, left.y - right.y))
            if distance <= distance_tolerance_px:
                union(left_index, right_index)

    grouped: dict[int, list[SignedNullSequenceSourceRow]] = {}
    for index, row in enumerate(rows):
        grouped.setdefault(find(index), []).append(row)
    recurrent = [group for group in grouped.values() if len({row.frame_index for row in group}) >= 2]
    recurrent.sort(key=lambda group: (min(row.frame_index for row in group), np.median([row.x for row in group])))
    result: list[SignedNullSequenceRecurrenceRow] = []
    for cluster_id, group in enumerate(recurrent, 1):
        x_median = float(np.median([row.x for row in group]))
        y_median = float(np.median([row.y for row in group]))
        radii = [float(np.hypot(row.x - x_median, row.y - y_median)) for row in group]
        flux_values = [row.flux_snr for row in group if row.flux_snr is not None and np.isfinite(row.flux_snr)]
        result.append(
            SignedNullSequenceRecurrenceRow(
                cluster_id=cluster_id,
                frame_indices=",".join(f"{index:02d}" for index in sorted({row.frame_index for row in group})),
                frame_count=len({row.frame_index for row in group}),
                source_count=len(group),
                x_median=x_median,
                y_median=y_median,
                max_radius_px=max(radii),
                transfer_count=sum(row.range_transfer_overlap for row in group),
                effective_count=sum(not row.range_transfer_overlap for row in group),
                median_flux_snr=(None if not flux_values else float(np.median(flux_values))),
            )
        )
    return tuple(result)


def run_signed_null_sequence(
    fits_paths: Sequence[str | Path] | str | Path,
    *,
    threshold_sigma: float = 4.0,
    min_distance: int = 4,
    aperture_radius: int = 4,
    psf_fwhm: float = 2.0,
    background_box_size: int = 128,
    min_flux_snr: float = 5.0,
    min_psf_support_pixels: int = 3,
    proposal_mode: str = "hybrid",
    reject_linear_artifacts: bool = True,
    snr_thresholds: Sequence[float] = SIGNED_NULL_SNR_THRESHOLDS,
    progress: Callable[[float, str], None] | None = None,
) -> SignedNullSequenceResult:
    """逐帧运行符号反相对照并保留每帧/合并两种口径。"""

    if isinstance(fits_paths, (str, Path)):
        paths = (Path(fits_paths),)
    else:
        paths = tuple(Path(path) for path in fits_paths)
    if not paths:
        raise ValueError("fits_paths must not be empty")

    total = len(paths)
    frame_rows: list[SignedNullSequenceFrameRow] = []
    threshold_rows: list[SignedNullSequenceThresholdRow] = []
    feature_rows: list[SignedNullSequenceFeatureRow] = []
    quality_source_rows: list[SignedNullSequenceSourceRow] = []
    audits: list[SignedNullAuditResult] = []

    def report(value: float, label: str) -> None:
        if progress is not None:
            progress(max(0.0, min(100.0, float(value))), label)

    for zero_index, path in enumerate(paths):
        frame_number = zero_index + 1
        frame = read_fits(path)
        start = 100.0 * zero_index / total
        span = 100.0 / total

        def frame_progress(value: float, label: str, *, _start=start, _span=span, _number=frame_number) -> None:
            report(_start + _span * max(0.0, min(100.0, float(value))) / 100.0, f"F{_number:02d}/{total} · {label}")

        audit = run_signed_null_audit(
            frame,
            threshold_sigma=threshold_sigma,
            min_distance=min_distance,
            aperture_radius=aperture_radius,
            psf_fwhm=psf_fwhm,
            background_box_size=background_box_size,
            min_flux_snr=min_flux_snr,
            min_psf_support_pixels=min_psf_support_pixels,
            proposal_mode=proposal_mode,
            reject_linear_artifacts=reject_linear_artifacts,
            snr_thresholds=snr_thresholds,
            progress=frame_progress,
        )
        audits.append(audit)
        frame_rows.append(_as_frame_row(frame_number, frame, audit))
        threshold_rows.extend(_as_threshold_rows(frame_number, audit))
        feature_rows.extend(_as_feature_rows(frame_number, audit))
        quality_source_rows.extend(_as_source_rows(frame_number, audit))
        report(start + span, f"F{frame_number:02d}/{total} · 完成")

    positive_candidates = sum(audit.positive_candidate_count for audit in audits)
    positive_quality = sum(audit.positive_quality_count for audit in audits)
    negative_candidates = sum(audit.negative_candidate_count for audit in audits)
    negative_quality = sum(audit.negative_quality_count for audit in audits)
    negative_effective_quality = sum(audit.negative_quality_count_after_range_exclusion for audit in audits)
    range_transfer_candidates = sum(audit.negative_range_transfer_candidate_count for audit in audits)
    range_transfer_quality = sum(audit.negative_range_transfer_quality_count for audit in audits)
    candidate_ratios = [audit.candidate_leakage_ratio for audit in audits]
    quality_ratios = [audit.quality_leakage_ratio for audit in audits]
    recurrence_rows = _build_recurrence_rows(
        quality_source_rows,
        distance_tolerance_px=1.0,
    )
    parameters = {
        "frame_count": total,
        "frame_paths": [str(path) for path in paths],
        "detector": audits[0].parameters.get("detector", {}),
        "snr_thresholds": list(float(value) for value in snr_thresholds),
        "aggregation": {
            "pooled_ratio_definition": "sum(sign_flipped count) / sum(positive count)",
            "frame_mean_median_definition": (
                "mean/median of per-frame ratios over defined frame denominators"
            ),
            "feature_rows_definition": (
                "one row per frame, control direction and mutually exclusive primary feature class"
            ),
            "quality_source_rows_definition": (
                "all sign_flipped quality sources only; no full candidate catalog is persisted"
            ),
            "recurrence_definition": (
                "connected components of measured centroids from different frames within 1.0 px; "
                "diagnostic recurrence only, not physical identity"
            ),
        },
        "not_formal_fdr": True,
        "physical_truth_warning": True,
    }
    conclusion = (
        "15 帧结果同时保留逐帧和 pooled 计数；pooled 比值只是计数聚合后的 signed-tail leakage，"
        "不等于 15 帧真实 precision、FDR、恒星完备率或运动目标概率。值域转移、热像素、"
        "宇宙线、亮线和跨帧配准结构必须继续按类别和独立轨迹证据审计。"
    )
    report(100.0, "15 帧符号反相对照完成")
    return SignedNullSequenceResult(
        frame_count=total,
        frame_rows=tuple(frame_rows),
        threshold_rows=tuple(threshold_rows),
        feature_rows=tuple(feature_rows),
        pooled_positive_candidate_count=positive_candidates,
        pooled_positive_quality_count=positive_quality,
        pooled_negative_candidate_count=negative_candidates,
        pooled_negative_quality_count=negative_quality,
        pooled_negative_quality_count_after_range_exclusion=negative_effective_quality,
        pooled_candidate_leakage_ratio=_safe_ratio(negative_candidates, positive_candidates),
        pooled_quality_leakage_ratio=_safe_ratio(negative_quality, positive_quality),
        pooled_quality_leakage_ratio_after_range_exclusion=_safe_ratio(negative_effective_quality, positive_quality),
        pooled_negative_range_transfer_candidate_count=range_transfer_candidates,
        pooled_negative_range_transfer_quality_count=range_transfer_quality,
        frame_candidate_leakage_mean=_optional_summary(candidate_ratios, "mean"),
        frame_candidate_leakage_median=_optional_summary(candidate_ratios, "median"),
        frame_quality_leakage_mean=_optional_summary(quality_ratios, "mean"),
        frame_quality_leakage_median=_optional_summary(quality_ratios, "median"),
        parameters=parameters,
        conclusion=conclusion,
        quality_source_rows=tuple(quality_source_rows),
        recurrence_rows=recurrence_rows,
    )


def _write_rows(path: Path, rows: Sequence[dict[str, object]]) -> None:
    fields = tuple(rows[0].keys()) if rows else ()
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_signed_null_sequence_artifacts(result: SignedNullSequenceResult, out_dir: str | Path) -> Path:
    """写入 15 帧逐帧、SNR、特征和 JSON 汇总。"""

    output = Path(out_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "signed_null_sequence_summary.json").write_text(
        json.dumps(result.as_dict(), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    _write_rows(output / "signed_null_sequence_frames.csv", [row.as_dict() for row in result.frame_rows])
    _write_rows(output / "signed_null_sequence_thresholds.csv", [row.as_dict() for row in result.threshold_rows])
    _write_rows(output / "signed_null_sequence_features.csv", [row.as_dict() for row in result.feature_rows])
    _write_rows(
        output / "signed_null_sequence_quality_sources.csv",
        [row.as_dict() for row in result.quality_source_rows],
    )
    _write_rows(output / "signed_null_sequence_recurrences.csv", [row.as_dict() for row in result.recurrence_rows])
    return output
