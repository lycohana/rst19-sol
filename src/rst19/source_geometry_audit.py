"""源级坐标几何审计。

该模块只读取 ``source_catalog.csv`` 的候选坐标，检查指定特征类别是否在
detector 平面上呈现明显的细长集合。它是对 ``LINE_ARTIFACT`` 等旗标的
坐标层交叉审计，不重新读取 FITS，也不把空间细长性解释成物理运动或伪影
真值。

默认用同一源表中的 ``compact_quality`` 坐标做等样本量的可重复抽样对照。
对照只回答“当前类别的坐标轴比是否远离点源类的经验分布”，不是显著性检验、
FDR、分类器准确率或恒星概率。
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


DEFAULT_CONTROL_CLASS = "compact_quality"


@dataclass(frozen=True, slots=True)
class SourceGeometrySummary:
    """一个特征类别的 PCA 细长性和点源类抽样对照。"""

    feature_class: str
    feature_class_label: str
    candidate_count: int
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    pca_axis_ratio: float
    pca_orientation_deg: float
    perpendicular_residual_median_px: float
    perpendicular_residual_p90_px: float
    perpendicular_residual_max_px: float
    control_class: str
    control_sample_count: int
    control_trial_count: int
    control_axis_ratio_median: float | None
    control_axis_ratio_p95: float | None
    control_axis_ratio_p99: float | None
    control_axis_ratio_max: float | None
    control_axis_ratio_exceed_count: int
    control_axis_ratio_upper_tail_fraction: float | None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SourceGeometryAuditResult:
    """源级坐标几何审计的机器可读结果。"""

    catalog_path: str
    control_class: str
    control_trial_count: int
    random_seed: int
    target_classes: tuple[str, ...]
    summaries: tuple[SourceGeometrySummary, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "catalog_path": self.catalog_path,
            "control_class": self.control_class,
            "control_trial_count": self.control_trial_count,
            "random_seed": self.random_seed,
            "target_classes": list(self.target_classes),
            "summaries": [item.as_dict() for item in self.summaries],
            "interpretation_guardrails": [
                "PCA 细长性是 detector 坐标分布诊断，不是物理运动真值。",
                "control_axis_ratio_upper_tail_fraction 是有限次经验抽样的描述量，不是 p 值或 FDR。",
                "feature_class 和 compact_quality 都来自同一检测器源表，不能视为独立标注真值。",
                "线状候选仍需原始像素、线段模型、跨帧运动和官方真值联合核验。",
            ],
        }


def _finite_float(row: dict[str, str], field: str) -> float | None:
    raw = row.get(field, "")
    if raw is None or not str(raw).strip():
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _read_catalog(catalog_path: str | Path) -> tuple[Path, dict[str, list[tuple[float, float, str]]]]:
    path = Path(catalog_path)
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            fieldnames = set(reader.fieldnames or ())
            required = {"x", "y", "feature_class", "feature_class_label"}
            missing = sorted(required - fieldnames)
            if missing:
                raise ValueError(f"source catalog is missing required columns: {', '.join(missing)}")
            grouped: dict[str, list[tuple[float, float, str]]] = {}
            for row in reader:
                feature_class = str(row.get("feature_class", "")).strip()
                if not feature_class:
                    raise ValueError("source catalog contains a row without feature_class")
                x = _finite_float(row, "x")
                y = _finite_float(row, "y")
                if x is None or y is None:
                    raise ValueError(f"feature_class={feature_class!r} contains non-finite x/y")
                label = str(row.get("feature_class_label", feature_class)).strip() or feature_class
                grouped.setdefault(feature_class, []).append((x, y, label))
    except OSError as exc:
        raise OSError(f"无法读取源级 CSV：{path}") from exc
    if not grouped:
        raise ValueError(f"source catalog is empty: {path}")
    return path, grouped


def _pca_geometry(points: np.ndarray) -> tuple[float, float, np.ndarray]:
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
        raise ValueError("PCA 几何审计至少需要两个二维坐标")
    centered = points - points.mean(axis=0)
    _, singular_values, vectors = np.linalg.svd(centered, full_matrices=False)
    major = float(singular_values[0])
    minor = float(singular_values[1])
    axis_ratio = major / max(minor, np.finfo(np.float64).eps)
    axis = vectors[0]
    angle = math.degrees(math.atan2(float(axis[1]), float(axis[0])))
    orientation = ((angle + 90.0) % 180.0) - 90.0
    residual = np.abs(centered @ vectors[1])
    return axis_ratio, orientation, residual


def _quantile(values: np.ndarray, probability: float) -> float:
    return float(np.quantile(values, probability)) if len(values) else 0.0


def _control_axis_ratios(
    control_points: np.ndarray,
    sample_count: int,
    trial_count: int,
    random_seed: int,
) -> np.ndarray:
    if sample_count < 2:
        raise ValueError("control 类别至少需要两个坐标")
    if trial_count < 1:
        raise ValueError("control_trial_count 必须为正整数")
    if sample_count > len(control_points):
        raise ValueError("control 类别坐标数量不足以构造等样本量对照")
    rng = np.random.default_rng(random_seed)
    ratios = np.empty(trial_count, dtype=np.float64)
    for index in range(trial_count):
        selected = rng.choice(len(control_points), size=sample_count, replace=False)
        ratios[index] = _pca_geometry(control_points[selected])[0]
    return ratios


def run_source_geometry_audit(
    catalog_path: str | Path,
    *,
    target_classes: Iterable[str] | None = None,
    control_class: str = DEFAULT_CONTROL_CLASS,
    control_trial_count: int = 5000,
    random_seed: int = 1909,
) -> SourceGeometryAuditResult:
    """按类别计算坐标 PCA 细长性，并与等样本量点源类抽样对照。"""

    if control_trial_count < 1:
        raise ValueError("control_trial_count 必须为正整数")
    path, grouped = _read_catalog(catalog_path)
    control_rows = grouped.get(control_class)
    if not control_rows:
        raise ValueError(f"source catalog has no control class: {control_class!r}")

    if target_classes is None:
        requested = tuple(sorted(name for name in grouped if name != "other_rejected"))
    else:
        requested = tuple(dict.fromkeys(str(name).strip() for name in target_classes if str(name).strip()))
    if not requested:
        raise ValueError("target_classes 不能为空")
    unknown = [name for name in requested if name not in grouped]
    if unknown:
        raise ValueError(f"source catalog has no target class: {unknown[0]!r}")

    control_points = np.asarray([(x, y) for x, y, _ in control_rows], dtype=np.float64)
    summaries: list[SourceGeometrySummary] = []
    for class_index, feature_class in enumerate(requested):
        class_rows = grouped[feature_class]
        points = np.asarray([(x, y) for x, y, _ in class_rows], dtype=np.float64)
        if len(points) < 2:
            raise ValueError(f"feature_class={feature_class!r} 至少需要两个坐标")
        axis_ratio, orientation, residual = _pca_geometry(points)
        sample_count = min(len(points), len(control_points))
        if feature_class == control_class:
            control_ratios = np.empty(0, dtype=np.float64)
        else:
            control_ratios = _control_axis_ratios(
                control_points,
                sample_count=sample_count,
                trial_count=control_trial_count,
                random_seed=random_seed + class_index,
            )
        exceed_count = int(np.count_nonzero(control_ratios >= axis_ratio))
        summaries.append(
            SourceGeometrySummary(
                feature_class=feature_class,
                feature_class_label=class_rows[0][2],
                candidate_count=len(points),
                x_min=float(points[:, 0].min()),
                x_max=float(points[:, 0].max()),
                y_min=float(points[:, 1].min()),
                y_max=float(points[:, 1].max()),
                pca_axis_ratio=axis_ratio,
                pca_orientation_deg=orientation,
                perpendicular_residual_median_px=float(np.median(residual)),
                perpendicular_residual_p90_px=_quantile(residual, 0.90),
                perpendicular_residual_max_px=float(residual.max()),
                control_class=control_class,
                control_sample_count=sample_count,
                control_trial_count=len(control_ratios),
                control_axis_ratio_median=_quantile(control_ratios, 0.50) if len(control_ratios) else None,
                control_axis_ratio_p95=_quantile(control_ratios, 0.95) if len(control_ratios) else None,
                control_axis_ratio_p99=_quantile(control_ratios, 0.99) if len(control_ratios) else None,
                control_axis_ratio_max=float(control_ratios.max()) if len(control_ratios) else None,
                control_axis_ratio_exceed_count=exceed_count,
                control_axis_ratio_upper_tail_fraction=(
                    float(exceed_count / len(control_ratios)) if len(control_ratios) else None
                ),
            )
        )
    return SourceGeometryAuditResult(
        catalog_path=str(path),
        control_class=control_class,
        control_trial_count=control_trial_count,
        random_seed=random_seed,
        target_classes=requested,
        summaries=tuple(summaries),
    )


def _write_csv(path: Path, rows: Sequence[SourceGeometrySummary]) -> None:
    if not rows:
        path.write_text("\n", encoding="utf-8-sig")
        return
    fieldnames = list(rows[0].as_dict().keys())
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.as_dict())


def write_source_geometry_artifacts(
    result: SourceGeometryAuditResult,
    out_dir: str | Path,
) -> Path:
    """写出类别几何 CSV/JSON 产物。"""

    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "source_geometry_audit.json").write_text(
        json.dumps(result.as_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_csv(output / "source_geometry_summary.csv", result.summaries)
    return output


__all__ = [
    "DEFAULT_CONTROL_CLASS",
    "SourceGeometryAuditResult",
    "SourceGeometrySummary",
    "run_source_geometry_audit",
    "write_source_geometry_artifacts",
]
