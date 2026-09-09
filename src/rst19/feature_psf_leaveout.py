"""按特征类别做逐源留一法经验 PSF 交叉核验。

现有的类别 PSF 表适合快速观察，但全局经验核可能包含正在被评价的
``compact_quality`` 源本身。这里对每个确定性抽样源重建一次模板，并从
模板候选池排除该源；全幅候选表仍用于邻峰隔离。输出只回答“源的局部
形状是否像一组独立模板”，不把 PSF 相似度解释成恒星概率、星表身份或
物理真值，也不接入默认检测、质量层、GUI 或缓存。
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from .detection import Detection
from .experiments import (
    EmpiricalPSF,
    _SOURCE_FEATURE_CLASSES,
    _psf_similarity_metrics,
    classify_source_feature,
    estimate_empirical_psf,
)
from .fits import FitsFrame, auxiliary_mask, read_fits


@dataclass(frozen=True, slots=True)
class FeaturePSFLeaveOutSourceRow:
    """一个抽样源的全局模板与留一模板成对指标。"""

    detection_id: int
    feature_class: str
    feature_class_label: str
    quality_passed: bool
    in_sample_correlation: float | None
    leaveout_correlation: float | None
    correlation_delta_leaveout_minus_in_sample: float | None
    in_sample_residual: float | None
    leaveout_residual: float | None
    residual_delta_leaveout_minus_in_sample: float | None
    in_sample_central_energy_fraction: float | None
    leaveout_central_energy_fraction: float | None
    leaveout_amplitude_adu: float | None
    leaveout_template_source_count: int
    leaveout_template_median_fwhm_px: float | None
    leaveout_template_median_ellipticity: float | None
    in_sample_valid: bool
    leaveout_valid: bool
    note: str

    def as_dict(self) -> dict[str, object]:
        return {
            "detection_id": self.detection_id,
            "feature_class": self.feature_class,
            "feature_class_label": self.feature_class_label,
            "quality_passed": self.quality_passed,
            "in_sample_correlation": self.in_sample_correlation,
            "leaveout_correlation": self.leaveout_correlation,
            "correlation_delta_leaveout_minus_in_sample": self.correlation_delta_leaveout_minus_in_sample,
            "in_sample_residual": self.in_sample_residual,
            "leaveout_residual": self.leaveout_residual,
            "residual_delta_leaveout_minus_in_sample": self.residual_delta_leaveout_minus_in_sample,
            "in_sample_central_energy_fraction": self.in_sample_central_energy_fraction,
            "leaveout_central_energy_fraction": self.leaveout_central_energy_fraction,
            "leaveout_amplitude_adu": self.leaveout_amplitude_adu,
            "leaveout_template_source_count": self.leaveout_template_source_count,
            "leaveout_template_median_fwhm_px": self.leaveout_template_median_fwhm_px,
            "leaveout_template_median_ellipticity": self.leaveout_template_median_ellipticity,
            "in_sample_valid": self.in_sample_valid,
            "leaveout_valid": self.leaveout_valid,
            "note": self.note,
        }


@dataclass(frozen=True, slots=True)
class FeaturePSFLeaveOutClassRow:
    """一个互斥首要特征类别的留一法汇总。"""

    feature_class: str
    feature_class_label: str
    global_psf_source_count: int
    sample_count: int
    in_sample_valid_count: int
    leaveout_valid_count: int
    paired_count: int
    leaveout_template_built_count: int
    leaveout_template_failed_count: int
    leaveout_template_source_count_median: float | None
    correlation_threshold: float
    in_sample_correlation_median: float | None
    leaveout_correlation_p10: float | None
    leaveout_correlation_median: float | None
    leaveout_correlation_p90: float | None
    leaveout_correlation_ge_threshold_count: int
    leaveout_correlation_ge_threshold_fraction: float | None
    in_sample_residual_median: float | None
    leaveout_residual_p10: float | None
    leaveout_residual_median: float | None
    leaveout_residual_p90: float | None
    paired_correlation_delta_median: float | None
    paired_residual_delta_median: float | None
    sample_detection_ids: str

    def as_dict(self) -> dict[str, object]:
        return {
            "feature_class": self.feature_class,
            "feature_class_label": self.feature_class_label,
            "global_psf_source_count": self.global_psf_source_count,
            "sample_count": self.sample_count,
            "in_sample_valid_count": self.in_sample_valid_count,
            "leaveout_valid_count": self.leaveout_valid_count,
            "paired_count": self.paired_count,
            "leaveout_template_built_count": self.leaveout_template_built_count,
            "leaveout_template_failed_count": self.leaveout_template_failed_count,
            "leaveout_template_source_count_median": self.leaveout_template_source_count_median,
            "correlation_threshold": self.correlation_threshold,
            "in_sample_correlation_median": self.in_sample_correlation_median,
            "leaveout_correlation_p10": self.leaveout_correlation_p10,
            "leaveout_correlation_median": self.leaveout_correlation_median,
            "leaveout_correlation_p90": self.leaveout_correlation_p90,
            "leaveout_correlation_ge_threshold_count": self.leaveout_correlation_ge_threshold_count,
            "leaveout_correlation_ge_threshold_fraction": self.leaveout_correlation_ge_threshold_fraction,
            "in_sample_residual_median": self.in_sample_residual_median,
            "leaveout_residual_p10": self.leaveout_residual_p10,
            "leaveout_residual_median": self.leaveout_residual_median,
            "leaveout_residual_p90": self.leaveout_residual_p90,
            "paired_correlation_delta_median": self.paired_correlation_delta_median,
            "paired_residual_delta_median": self.paired_residual_delta_median,
            "sample_detection_ids": self.sample_detection_ids,
        }


@dataclass(frozen=True, slots=True)
class FeaturePSFLeaveOutResult:
    """整幅图的按类别留一法经验 PSF 研究结果。"""

    frame_path: str
    catalog_path: str | None
    image_shape: tuple[int, int]
    per_class: int
    support_radius: int
    max_template_sources: int
    correlation_threshold: float
    global_psf: EmpiricalPSF | None
    class_rows: tuple[FeaturePSFLeaveOutClassRow, ...]
    source_rows: tuple[FeaturePSFLeaveOutSourceRow, ...]
    parameters: dict[str, object]
    conclusion: str

    def as_dict(self) -> dict[str, object]:
        return {
            "frame_path": self.frame_path,
            "catalog_path": self.catalog_path,
            "image_shape": list(self.image_shape),
            "per_class": self.per_class,
            "support_radius": self.support_radius,
            "max_template_sources": self.max_template_sources,
            "correlation_threshold": self.correlation_threshold,
            "global_psf": None if self.global_psf is None else self.global_psf.as_dict(),
            "class_rows": [row.as_dict() for row in self.class_rows],
            "source_rows": [row.as_dict() for row in self.source_rows],
            "parameters": self.parameters,
            "conclusion": self.conclusion,
        }


def _frame_values(frame: FitsFrame | np.ndarray | str | Path) -> tuple[np.ndarray, str]:
    if isinstance(frame, FitsFrame):
        values = np.asarray(frame.data)
        label = str(frame.path)
    elif isinstance(frame, (str, Path)):
        loaded = read_fits(frame)
        values = np.asarray(loaded.data)
        label = str(loaded.path)
    else:
        values = np.asarray(frame)
        label = "<array>"
    if values.ndim != 2:
        raise ValueError(f"expected a 2-D image, got shape {values.shape}")
    return values, label


def _signal_snr(source: Detection) -> float:
    value = source.flux_snr if source.flux_snr is not None else source.snr
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return float("-inf")
    return converted if np.isfinite(converted) else float("-inf")


def _select_samples(sources: Sequence[Detection], per_class: int) -> dict[str, tuple[Detection, ...]]:
    grouped: dict[str, list[Detection]] = {feature_class: [] for feature_class, _label in _SOURCE_FEATURE_CLASSES}
    for source in sources:
        grouped.setdefault(classify_source_feature(source), []).append(source)
    selected: dict[str, tuple[Detection, ...]] = {}
    for feature_class, _label in _SOURCE_FEATURE_CLASSES:
        members = sorted(grouped[feature_class], key=lambda source: (_signal_snr(source), source.detection_id))
        if len(members) > per_class:
            indices = np.unique(np.rint(np.linspace(0, len(members) - 1, per_class)).astype(int))
            members = [members[int(index)] for index in indices]
        selected[feature_class] = tuple(members)
    return selected


def _metric_or_none(metrics: tuple[float, float, float, float] | None, index: int) -> float | None:
    if metrics is None:
        return None
    value = float(metrics[index])
    return value if np.isfinite(value) else None


def _median(values: Sequence[float]) -> float | None:
    return float(np.median(values)) if values else None


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    return float(np.percentile(values, percentile)) if values else None


def run_feature_psf_leaveout_audit(
    frame: FitsFrame | np.ndarray | str | Path,
    sources: Sequence[Detection],
    *,
    catalog_path: str | Path | None = None,
    per_class: int = 16,
    support_radius: int = 7,
    max_template_sources: int = 20,
    correlation_threshold: float = 0.8,
    progress: Callable[[int, int], None] | None = None,
) -> FeaturePSFLeaveOutResult:
    """对每类抽样源做全局模板与逐源留一模板的配对比较。"""

    if per_class < 1 or support_radius < 3 or max_template_sources < 3:
        raise ValueError("per_class must be positive, support_radius at least 3, and max_template_sources at least 3")
    if not 0.0 < float(correlation_threshold) <= 1.0:
        raise ValueError("correlation_threshold must be between 0 and 1")
    values, frame_path = _frame_values(frame)
    if not sources:
        raise ValueError("sources cannot be empty")

    global_psf = estimate_empirical_psf(
        values,
        sources,
        support_radius=support_radius,
        max_sources=max_template_sources,
        isolation_sources=sources,
    )
    selected_by_class = _select_samples(sources, per_class)
    selected_items = [
        (feature_class, feature_label, source)
        for feature_class, feature_label in _SOURCE_FEATURE_CLASSES
        for source in selected_by_class[feature_class]
    ]
    total = len(selected_items)
    aux_mask = auxiliary_mask(values.shape)
    source_rows: list[FeaturePSFLeaveOutSourceRow] = []

    for index, (feature_class, feature_label, source) in enumerate(selected_items, start=1):
        in_sample_metrics = (
            _psf_similarity_metrics(values, source, global_psf, aux_mask) if global_psf is not None else None
        )
        leaveout_psf = estimate_empirical_psf(
            values,
            sources,
            support_radius=support_radius,
            max_sources=max_template_sources,
            isolation_sources=sources,
            exclude_detection_ids=(source.detection_id,),
        )
        leaveout_metrics = (
            _psf_similarity_metrics(values, source, leaveout_psf, aux_mask) if leaveout_psf is not None else None
        )
        in_corr = _metric_or_none(in_sample_metrics, 0)
        leaveout_corr = _metric_or_none(leaveout_metrics, 0)
        in_residual = _metric_or_none(in_sample_metrics, 1)
        leaveout_residual = _metric_or_none(leaveout_metrics, 1)
        leaveout_template_source_count = 0 if leaveout_psf is None else int(leaveout_psf.source_count)
        if leaveout_psf is None:
            note = "留一模板不足三个合格隔离源，未强行填充 PSF"
        elif leaveout_metrics is None:
            note = "留一模板已建立，但该源裁剪无有效形状指标"
        else:
            note = "留一经验 PSF 形状指标；不等于恒星身份"
        source_rows.append(
            FeaturePSFLeaveOutSourceRow(
                detection_id=int(source.detection_id),
                feature_class=feature_class,
                feature_class_label=feature_label,
                quality_passed=bool(source.quality_passed),
                in_sample_correlation=in_corr,
                leaveout_correlation=leaveout_corr,
                correlation_delta_leaveout_minus_in_sample=(
                    leaveout_corr - in_corr if leaveout_corr is not None and in_corr is not None else None
                ),
                in_sample_residual=in_residual,
                leaveout_residual=leaveout_residual,
                residual_delta_leaveout_minus_in_sample=(
                    leaveout_residual - in_residual
                    if leaveout_residual is not None and in_residual is not None
                    else None
                ),
                in_sample_central_energy_fraction=_metric_or_none(in_sample_metrics, 2),
                leaveout_central_energy_fraction=_metric_or_none(leaveout_metrics, 2),
                leaveout_amplitude_adu=_metric_or_none(leaveout_metrics, 3),
                leaveout_template_source_count=leaveout_template_source_count,
                leaveout_template_median_fwhm_px=(
                    None if leaveout_psf is None else leaveout_psf.median_fwhm_px
                ),
                leaveout_template_median_ellipticity=(
                    None if leaveout_psf is None else leaveout_psf.median_ellipticity
                ),
                in_sample_valid=in_sample_metrics is not None,
                leaveout_valid=leaveout_metrics is not None,
                note=note,
            )
        )
        if progress is not None:
            progress(index, total)

    class_rows: list[FeaturePSFLeaveOutClassRow] = []
    global_count = 0 if global_psf is None else int(global_psf.source_count)
    for feature_class, feature_label in _SOURCE_FEATURE_CLASSES:
        members = [row for row in source_rows if row.feature_class == feature_class]
        in_correlations = [float(row.in_sample_correlation) for row in members if row.in_sample_correlation is not None]
        leaveout_correlations = [
            float(row.leaveout_correlation) for row in members if row.leaveout_correlation is not None
        ]
        in_residuals = [float(row.in_sample_residual) for row in members if row.in_sample_residual is not None]
        leaveout_residuals = [float(row.leaveout_residual) for row in members if row.leaveout_residual is not None]
        paired_correlation_deltas = [
            float(row.correlation_delta_leaveout_minus_in_sample)
            for row in members
            if row.correlation_delta_leaveout_minus_in_sample is not None
        ]
        paired_residual_deltas = [
            float(row.residual_delta_leaveout_minus_in_sample)
            for row in members
            if row.residual_delta_leaveout_minus_in_sample is not None
        ]
        template_counts = [
            float(row.leaveout_template_source_count)
            for row in members
            if row.leaveout_template_source_count > 0
        ]
        count_ge_threshold = sum(value >= float(correlation_threshold) for value in leaveout_correlations)
        class_rows.append(
            FeaturePSFLeaveOutClassRow(
                feature_class=feature_class,
                feature_class_label=feature_label,
                global_psf_source_count=global_count,
                sample_count=len(members),
                in_sample_valid_count=len(in_correlations),
                leaveout_valid_count=len(leaveout_correlations),
                paired_count=len(paired_correlation_deltas),
                leaveout_template_built_count=sum(row.leaveout_template_source_count > 0 for row in members),
                leaveout_template_failed_count=sum(row.leaveout_template_source_count == 0 for row in members),
                leaveout_template_source_count_median=_median(template_counts),
                correlation_threshold=float(correlation_threshold),
                in_sample_correlation_median=_median(in_correlations),
                leaveout_correlation_p10=_percentile(leaveout_correlations, 10.0),
                leaveout_correlation_median=_median(leaveout_correlations),
                leaveout_correlation_p90=_percentile(leaveout_correlations, 90.0),
                leaveout_correlation_ge_threshold_count=count_ge_threshold,
                leaveout_correlation_ge_threshold_fraction=(
                    count_ge_threshold / len(leaveout_correlations) if leaveout_correlations else None
                ),
                in_sample_residual_median=_median(in_residuals),
                leaveout_residual_p10=_percentile(leaveout_residuals, 10.0),
                leaveout_residual_median=_median(leaveout_residuals),
                leaveout_residual_p90=_percentile(leaveout_residuals, 90.0),
                paired_correlation_delta_median=_median(paired_correlation_deltas),
                paired_residual_delta_median=_median(paired_residual_deltas),
                sample_detection_ids="|".join(str(row.detection_id) for row in members),
            )
        )

    conclusion = (
        "留一法只把被评价源从经验 PSF 模板候选池排除，并继续用全幅候选表做邻峰隔离。"
        "如果 compact_quality 的留一相关度仍高而线状、拥挤或尖峰类仍低，"
        "类别差异就不主要是模板自相似；但这仍只是 detector-level 形状证据，"
        "不能替代星表/WCS 一对一匹配、独立注入真值或完整噪声似然。"
    )
    return FeaturePSFLeaveOutResult(
        frame_path=frame_path,
        catalog_path=None if catalog_path is None else str(Path(catalog_path)),
        image_shape=(int(values.shape[0]), int(values.shape[1])),
        per_class=int(per_class),
        support_radius=int(support_radius),
        max_template_sources=int(max_template_sources),
        correlation_threshold=float(correlation_threshold),
        global_psf=global_psf,
        class_rows=tuple(class_rows),
        source_rows=tuple(source_rows),
        parameters={
            "sampling": "各互斥首要类别按 flux_snr 升序后等距抽样，最多 per_class 个",
            "template": "质量通过、非边缘、非掩膜、非线状、非溢出、形状合格且全幅隔离的源",
            "leaveout": "每个样本只从模板候选池排除自身；邻峰隔离仍检查完整候选表",
            "metric": "局部背景扣除后的正残差与经验核余弦相关、相对残差和中心 3x3 能量占比",
            "correlation_threshold_is_not_probability": True,
        },
        conclusion=conclusion,
    )


def write_feature_psf_leaveout_artifacts(
    result: FeaturePSFLeaveOutResult,
    out_dir: str | Path,
) -> Path:
    """写入留一法类别表、逐源表和 JSON。"""

    output = Path(out_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    class_rows = [row.as_dict() for row in result.class_rows]
    source_rows = [row.as_dict() for row in result.source_rows]
    class_fields = tuple(class_rows[0].keys()) if class_rows else ()
    source_fields = tuple(source_rows[0].keys()) if source_rows else ()
    with (output / "feature_psf_leaveout.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=class_fields)
        writer.writeheader()
        writer.writerows(class_rows)
    with (output / "feature_psf_leaveout_source.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=source_fields)
        writer.writeheader()
        writer.writerows(source_rows)
    (output / "feature_psf_leaveout.json").write_text(
        json.dumps(result.as_dict(), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return output
