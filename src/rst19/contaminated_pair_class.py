"""按特征类别选择污染锚点并运行统一的双源注入审计。

``contaminated_pair_injection`` 负责检测器实验本身；本模块只负责从源级
目录中可复现地挑选类别锚点，避免论文把少数手写坐标误写成全类别结论。
默认选择每类指定指标接近类别中位数的源，另提供 ``high`` 模式选择高
显著性 hard-negative。可选的 ``local_roi`` 注入范围用于多锚点加速，且会
明确标注 ROI 局部计数；类别标签是 detector-level 审计标签，不是物理星类。
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from .contaminated_pair_injection import (
    ContaminatedPairAnchor,
    ContaminatedPairInjectionResult,
    ContaminatedPairInjectionRow,
    run_contaminated_pair_injection_audit,
    write_contaminated_pair_injection_artifacts,
)
from .fits import FitsFrame, read_fits


CLASS_SELECTION_METRICS: tuple[str, ...] = ("filter_snr", "flux_snr", "peak")
CLASS_SELECTION_MODES: tuple[str, ...] = ("median", "high")
CLASS_SUMMARY_FIELDS: tuple[str, ...] = (
    "feature_class",
    "feature_class_label",
    "separation_px",
    "secondary_to_primary_ratio",
    "anchor_count",
    "condition_count",
    "anchor_detection_ids",
    "baseline_candidate_pair_count",
    "injected_candidate_pair_count",
    "new_candidate_pair_count",
    "baseline_quality_pair_count",
    "injected_quality_pair_count",
    "new_quality_pair_count",
    "merged_candidate_count",
    "merged_quality_count",
    "candidate_pair_resolved_fraction",
    "quality_pair_resolved_fraction",
    "new_candidate_pair_fraction",
    "new_quality_pair_fraction",
    "merged_candidate_fraction",
)
QUALITY_REASON_SUMMARY_FIELDS: tuple[str, ...] = (
    "feature_class",
    "feature_class_label",
    "separation_px",
    "secondary_to_primary_ratio",
    "anchor_count",
    "condition_count",
    "endpoint_count",
    "primary_quality_reason_counts",
    "secondary_quality_reason_counts",
    "combined_quality_reason_counts",
)
DEFAULT_FEATURE_CLASSES: tuple[str, ...] = (
    "compact_quality",
    "crowded_blend",
    "linear_artifact",
    "masked_or_edge",
    "range_anomaly",
    "shape_outlier",
    "spike_or_support",
    "weak_or_background",
)


@dataclass(frozen=True, slots=True)
class FeatureClassAnchorSelection:
    """一个类别锚点的选择证据。"""

    feature_class: str
    feature_class_label: str
    detection_id: int
    x: float
    y: float
    selection_metric: str
    selection_mode: str
    metric_value: float
    class_metric_median: float
    quality_passed: bool
    flags: str
    class_count: int
    eligible_count: int
    selection_rank: int
    direction_x: float
    direction_y: float

    def as_dict(self) -> dict[str, object]:
        return asdict(self)

    def to_anchor(self) -> ContaminatedPairAnchor:
        return ContaminatedPairAnchor(
            name=f"{self.feature_class}_{self.detection_id}",
            x=self.x,
            y=self.y,
            direction_x=self.direction_x,
            direction_y=self.direction_y,
        )


@dataclass(frozen=True, slots=True)
class FeatureClassContaminatedPairResult:
    """类别锚点选择和污染双源注入结果。"""

    catalog_path: str
    source_path: str
    selection_metric: str
    selection_mode: str
    selected_rows: tuple[FeatureClassAnchorSelection, ...]
    injection: ContaminatedPairInjectionResult

    def as_dict(self) -> dict[str, object]:
        return {
            "catalog_path": self.catalog_path,
            "source_path": self.source_path,
            "selection_metric": self.selection_metric,
            "selection_mode": self.selection_mode,
            "selected_rows": [row.as_dict() for row in self.selected_rows],
            "injection": self.injection.as_dict(),
            "class_summary": list(summarize_feature_class_contaminated_pair(self)),
            "quality_reason_summary": list(summarize_feature_class_quality_reasons(self)),
            "interpretation_boundary": (
                "feature class labels and anchor selection are detector-level diagnostics; "
                "injection recovery is not precision, FDR, or physical star truth"
            ),
        }


def _fraction(count: int, total: int) -> float:
    return float(count / total) if total else 0.0


def summarize_feature_class_contaminated_pair(
    result: FeatureClassContaminatedPairResult,
) -> tuple[dict[str, object], ...]:
    """按类别、间距和强度比汇总污染双源结果。

    ``contaminated_pair_injection.csv`` 逐条件保留完整证据；这里提供一张
    可直接用于论文表格的聚合表。聚合同时保留 baseline、injected 和 new，
    避免把锚点在注入前已经存在的候选误当成注入回收。它仍然只是条件回收
    统计，不是 precision、FDR 或物理恒星真值。
    """

    selection_by_anchor = {
        selection.to_anchor().name: selection for selection in result.selected_rows
    }
    grouped: dict[tuple[str, float, float], list[ContaminatedPairInjectionRow]] = {}
    for row in result.injection.rows:
        selection = selection_by_anchor.get(row.anchor_name)
        if selection is None:
            raise ValueError(f"injection row has unknown anchor: {row.anchor_name!r}")
        key = (
            selection.feature_class,
            float(row.separation_px),
            float(row.secondary_to_primary_ratio),
        )
        grouped.setdefault(key, []).append(row)

    summaries: list[dict[str, object]] = []
    for (feature_class, separation, ratio), rows in sorted(
        grouped.items(), key=lambda item: (item[0][0], item[0][1], item[0][2])
    ):
        selections = [selection_by_anchor[row.anchor_name] for row in rows]
        total = len(rows)
        count = lambda field: sum(bool(getattr(row, field)) for row in rows)
        pair_count = lambda primary, secondary: sum(
            bool(getattr(row, primary) and getattr(row, secondary)) for row in rows
        )
        summaries.append(
            {
                "feature_class": feature_class,
                "feature_class_label": selections[0].feature_class_label,
                "separation_px": separation,
                "secondary_to_primary_ratio": ratio,
                "anchor_count": len({row.anchor_name for row in rows}),
                "condition_count": total,
                "anchor_detection_ids": "|".join(
                    str(value)
                    for value in sorted({selection.detection_id for selection in selections})
                ),
                "baseline_candidate_pair_count": pair_count(
                    "baseline_candidate_primary", "baseline_candidate_secondary"
                ),
                "injected_candidate_pair_count": count("candidate_pair_resolved"),
                "new_candidate_pair_count": pair_count(
                    "new_candidate_primary", "new_candidate_secondary"
                ),
                "baseline_quality_pair_count": pair_count(
                    "baseline_quality_primary", "baseline_quality_secondary"
                ),
                "injected_quality_pair_count": count("quality_pair_resolved"),
                "new_quality_pair_count": pair_count(
                    "new_quality_primary", "new_quality_secondary"
                ),
                "merged_candidate_count": count("merged_candidate"),
                "merged_quality_count": count("merged_quality"),
                "candidate_pair_resolved_fraction": _fraction(
                    count("candidate_pair_resolved"), total
                ),
                "quality_pair_resolved_fraction": _fraction(
                    count("quality_pair_resolved"), total
                ),
                "new_candidate_pair_fraction": _fraction(
                    pair_count("new_candidate_primary", "new_candidate_secondary"), total
                ),
                "new_quality_pair_fraction": _fraction(
                    pair_count("new_quality_primary", "new_quality_secondary"), total
                ),
                "merged_candidate_fraction": _fraction(count("merged_candidate"), total),
            }
        )
    return tuple(summaries)


def _reason_tokens(reason: str) -> tuple[str, ...]:
    value = str(reason).strip()
    if not value:
        return ("UNKNOWN",)
    return tuple(token for token in value.split("|") if token) or ("UNKNOWN",)


def _reason_counts_text(values: Sequence[str]) -> str:
    counts: Counter[str] = Counter()
    for value in values:
        counts.update(_reason_tokens(value))
    return json.dumps(dict(sorted(counts.items())), ensure_ascii=False, separators=(",", ":"))


def summarize_feature_class_quality_reasons(
    result: FeatureClassContaminatedPairResult,
) -> tuple[dict[str, object], ...]:
    """汇总注入端点的质量通过/拒绝原因，不把原因混成一个总命中率。

    每个条件有两个注入真值端点；``NO_CANDIDATE`` 表示宽筛就没有找到
    候选，其他非 ``QUALITY_PASS`` 标签来自实际 ``Detection.flags``。这张
    表仍然是污染结构回收诊断，不是原图伪影率或物理恒星分类器性能。
    """

    selection_by_anchor = {
        selection.to_anchor().name: selection for selection in result.selected_rows
    }
    grouped: dict[tuple[str, float, float], list[ContaminatedPairInjectionRow]] = {}
    for row in result.injection.rows:
        selection = selection_by_anchor.get(row.anchor_name)
        if selection is None:
            raise ValueError(f"injection row has unknown anchor: {row.anchor_name!r}")
        key = (
            selection.feature_class,
            float(row.separation_px),
            float(row.secondary_to_primary_ratio),
        )
        grouped.setdefault(key, []).append(row)

    summaries: list[dict[str, object]] = []
    for (feature_class, separation, ratio), rows in sorted(
        grouped.items(), key=lambda item: (item[0][0], item[0][1], item[0][2])
    ):
        selections = [selection_by_anchor[row.anchor_name] for row in rows]
        primary_reasons = [row.injected_primary_quality_reason for row in rows]
        secondary_reasons = [row.injected_secondary_quality_reason for row in rows]
        summaries.append(
            {
                "feature_class": feature_class,
                "feature_class_label": selections[0].feature_class_label,
                "separation_px": separation,
                "secondary_to_primary_ratio": ratio,
                "anchor_count": len({row.anchor_name for row in rows}),
                "condition_count": len(rows),
                "endpoint_count": 2 * len(rows),
                "primary_quality_reason_counts": _reason_counts_text(primary_reasons),
                "secondary_quality_reason_counts": _reason_counts_text(secondary_reasons),
                "combined_quality_reason_counts": _reason_counts_text(
                    [*primary_reasons, *secondary_reasons]
                ),
            }
        )
    return tuple(summaries)


def _finite_float(row: dict[str, str], field: str) -> float | None:
    raw = row.get(field, "")
    if raw is None or not str(raw).strip():
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _bool_value(row: dict[str, str], field: str) -> bool:
    return str(row.get(field, "")).strip().lower() in {"true", "1", "yes", "y", "是"}


def _read_catalog(path: str | Path) -> tuple[Path, list[dict[str, str]]]:
    catalog_path = Path(path)
    try:
        with catalog_path.open("r", encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
    except OSError as exc:
        raise OSError(f"无法读取源级 CSV：{catalog_path}") from exc
    if not rows:
        raise ValueError(f"源级 CSV 为空：{catalog_path}")
    required = {"detection_id", "feature_class", "x", "y", "quality_passed", "flags"}
    missing = sorted(required - set(rows[0]))
    if missing:
        raise ValueError(f"源级 CSV 缺少列：{', '.join(missing)}")
    return catalog_path, rows


def _normalise_direction(direction_x: float, direction_y: float) -> tuple[float, float]:
    dx = float(direction_x)
    dy = float(direction_y)
    norm = math.hypot(dx, dy)
    if not math.isfinite(norm) or norm <= 0:
        raise ValueError("direction must be finite and non-zero")
    return dx / norm, dy / norm


def _in_bounds(
    x: float,
    y: float,
    *,
    image_shape: tuple[int, int] | None,
    half_separation: float,
    direction_x: float,
    direction_y: float,
) -> bool:
    if image_shape is None:
        return True
    height, width = image_shape
    for sign in (-1.0, 1.0):
        point_x = x + sign * half_separation * direction_x
        point_y = y + sign * half_separation * direction_y
        if not (0.0 <= point_x < float(width) and 0.0 <= point_y < float(height)):
            return False
    return True


def select_feature_class_anchors(
    catalog_path: str | Path,
    *,
    feature_classes: Sequence[str] | None = None,
    per_class: int = 1,
    selection_metric: str = "filter_snr",
    selection_mode: str = "median",
    image_shape: tuple[int, int] | None = None,
    max_separation_px: float = 4.123,
    direction_x: float = 1.0,
    direction_y: float = 0.0,
) -> tuple[FeatureClassAnchorSelection, ...]:
    """按类别选择可复现的中位或高显著性锚点。"""

    if per_class < 1:
        raise ValueError("per_class must be positive")
    if selection_metric not in CLASS_SELECTION_METRICS:
        raise ValueError(f"unsupported selection_metric: {selection_metric!r}")
    if selection_mode not in CLASS_SELECTION_MODES:
        raise ValueError(f"unsupported selection_mode: {selection_mode!r}")
    if not math.isfinite(float(max_separation_px)) or max_separation_px <= 0:
        raise ValueError("max_separation_px must be finite and positive")
    unit_x, unit_y = _normalise_direction(direction_x, direction_y)
    catalog, rows = _read_catalog(catalog_path)
    by_class: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        class_name = str(row.get("feature_class", "")).strip()
        if class_name and class_name != "other_rejected":
            by_class.setdefault(class_name, []).append(row)
    if feature_classes is None:
        requested = tuple(name for name in DEFAULT_FEATURE_CLASSES if name in by_class)
        requested += tuple(sorted(set(by_class) - set(requested)))
    else:
        requested = tuple(dict.fromkeys(str(name).strip() for name in feature_classes if str(name).strip()))
    if not requested:
        raise ValueError("feature_classes cannot be empty")
    unknown = [name for name in requested if name not in by_class]
    if unknown:
        raise ValueError(f"source catalog has no feature_class={unknown[0]!r}")

    selected: list[FeatureClassAnchorSelection] = []
    half_separation = 0.5 * float(max_separation_px)
    for class_name in requested:
        class_rows = by_class[class_name]
        metric_rows: list[tuple[float, int, dict[str, str]]] = []
        for row in class_rows:
            value = _finite_float(row, selection_metric)
            x = _finite_float(row, "x")
            y = _finite_float(row, "y")
            if value is None or x is None or y is None:
                continue
            try:
                detection_id = int(row["detection_id"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid detection_id in class {class_name!r}") from exc
            if _in_bounds(
                x,
                y,
                image_shape=image_shape,
                half_separation=half_separation,
                direction_x=unit_x,
                direction_y=unit_y,
            ):
                metric_rows.append((value, detection_id, row))
        if not metric_rows:
            raise ValueError(f"class {class_name!r} has no eligible in-bounds rows")
        class_median = float(np.median([item[0] for item in metric_rows]))
        if selection_mode == "median":
            ordered = sorted(metric_rows, key=lambda item: (abs(item[0] - class_median), item[1]))
        else:
            rejected = [item for item in metric_rows if not _bool_value(item[2], "quality_passed")]
            ordered = sorted(rejected or metric_rows, key=lambda item: (-item[0], item[1]))
        for rank, (value, detection_id, row) in enumerate(ordered[:per_class], start=1):
            x = float(row["x"])
            y = float(row["y"])
            selected.append(
                FeatureClassAnchorSelection(
                    feature_class=class_name,
                    feature_class_label=row.get("feature_class_label", class_name),
                    detection_id=detection_id,
                    x=x,
                    y=y,
                    selection_metric=selection_metric,
                    selection_mode=selection_mode,
                    metric_value=value,
                    class_metric_median=class_median,
                    quality_passed=_bool_value(row, "quality_passed"),
                    flags=row.get("flags", ""),
                    class_count=len(class_rows),
                    eligible_count=len(metric_rows),
                    selection_rank=rank,
                    direction_x=unit_x,
                    direction_y=unit_y,
                )
            )
    return tuple(selected)


def run_feature_class_contaminated_pair_audit(
    frame: str | Path | FitsFrame,
    catalog_path: str | Path,
    *,
    feature_classes: Sequence[str] | None = None,
    per_class: int = 1,
    selection_metric: str = "filter_snr",
    selection_mode: str = "median",
    separations_px: Sequence[float] = (2.738, 4.123),
    secondary_to_primary_ratios: Sequence[float] = (0.143, 1.0),
    total_peak_excess_adu: float = 4096.0,
    psf_fwhm: float = 2.0,
    max_sources: int | None = None,
    analysis_scope: str = "full_frame",
    roi_half_size_px: int = 192,
    progress: Callable[[int, int], None] | None = None,
) -> FeatureClassContaminatedPairResult:
    """选择类别锚点并运行统一污染双源注入实验。"""

    frame_obj = frame if isinstance(frame, FitsFrame) else read_fits(frame)
    image_shape = tuple(int(value) for value in frame_obj.data.shape)
    if len(image_shape) != 2:
        raise ValueError(f"expected a 2-D FITS image, got shape {image_shape}")
    max_separation = max(float(value) for value in separations_px)
    selections = select_feature_class_anchors(
        catalog_path,
        feature_classes=feature_classes,
        per_class=per_class,
        selection_metric=selection_metric,
        selection_mode=selection_mode,
        image_shape=image_shape,
        max_separation_px=max_separation,
    )
    injection = run_contaminated_pair_injection_audit(
        frame_obj,
        tuple(selection.to_anchor() for selection in selections),
        separations_px=separations_px,
        secondary_to_primary_ratios=secondary_to_primary_ratios,
        total_peak_excess_adu=total_peak_excess_adu,
        psf_fwhm=psf_fwhm,
        max_sources=max_sources,
        analysis_scope=analysis_scope,
        roi_half_size_px=roi_half_size_px,
        progress=progress,
    )
    return FeatureClassContaminatedPairResult(
        catalog_path=str(Path(catalog_path)),
        source_path=str(frame_obj.path),
        selection_metric=selection_metric,
        selection_mode=selection_mode,
        selected_rows=selections,
        injection=injection,
    )


def write_feature_class_contaminated_pair_artifacts(
    result: FeatureClassContaminatedPairResult,
    output_dir: str | Path,
) -> Path:
    """写出类别选择、通用注入和汇总 JSON。"""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_contaminated_pair_injection_artifacts(result.injection, output)
    selection_csv = output / "contaminated_pair_class_selection.csv"
    with selection_csv.open("w", encoding="utf-8-sig", newline="") as stream:
        rows = [row.as_dict() for row in result.selected_rows]
        writer = csv.DictWriter(stream, fieldnames=list(FeatureClassAnchorSelection.__dataclass_fields__))
        writer.writeheader()
        writer.writerows(rows)
    summary_csv = output / "contaminated_pair_class_summary.csv"
    with summary_csv.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(CLASS_SUMMARY_FIELDS))
        writer.writeheader()
        writer.writerows(summarize_feature_class_contaminated_pair(result))
    reason_summary_csv = output / "contaminated_pair_quality_reason_summary.csv"
    with reason_summary_csv.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(QUALITY_REASON_SUMMARY_FIELDS))
        writer.writeheader()
        writer.writerows(summarize_feature_class_quality_reasons(result))
    (output / "contaminated_pair_class_audit.json").write_text(
        json.dumps(result.as_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output


__all__ = [
    "CLASS_SELECTION_METRICS",
    "CLASS_SELECTION_MODES",
    "CLASS_SUMMARY_FIELDS",
    "QUALITY_REASON_SUMMARY_FIELDS",
    "DEFAULT_FEATURE_CLASSES",
    "FeatureClassAnchorSelection",
    "FeatureClassContaminatedPairResult",
    "select_feature_class_anchors",
    "summarize_feature_class_contaminated_pair",
    "summarize_feature_class_quality_reasons",
    "run_feature_class_contaminated_pair_audit",
    "write_feature_class_contaminated_pair_artifacts",
]
