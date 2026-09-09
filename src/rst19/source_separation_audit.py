"""源目录最近邻间距的群体控制审计。

这个审计把候选源的测量质心和整数峰坐标分开统计。它用于检查一个局部双框
是否只是非极大值抑制边界附近的常见几何现象；它不做双星判定，也不把最近邻
距离当作独立源的充分证据。
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.spatial import cKDTree


_DEFAULT_RADII_PX = (3.0, 4.0, 4.5, 5.0, 6.0)
_REQUIRED_FIELDS = {
    "detection_id",
    "x",
    "y",
    "peak_x",
    "peak_y",
    "quality_passed",
    "feature_class",
}
_QUANTILE_LABELS = ("p01", "p05", "p10", "p25", "p50", "p75", "p90", "p95", "p99")
_QUANTILE_LEVELS = (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)


@dataclass(frozen=True, slots=True)
class SourceSeparationThresholdRow:
    """给定半径内的最近邻端点计数。"""

    radius_px: float
    centroid_endpoint_count: int
    centroid_endpoint_fraction: float
    centroid_quality_endpoint_count: int
    centroid_quality_endpoint_fraction: float
    peak_endpoint_count: int
    peak_endpoint_fraction: float
    peak_quality_endpoint_count: int
    peak_quality_endpoint_fraction: float

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SourceSeparationClassRow:
    """按首要特征类别分层的最近邻距离。"""

    feature_class: str
    source_count: int
    quality_count: int
    centroid_nn_median_px: float | None
    centroid_nn_p05_px: float | None
    centroid_nn_p95_px: float | None
    centroid_nn_le_4px_count: int
    peak_nn_median_px: float | None
    peak_nn_p05_px: float | None
    peak_nn_p95_px: float | None
    peak_nn_le_4px_count: int

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SourceSeparationTargetRow:
    """用户指定源在总体最近邻分布中的位置。"""

    detection_id: int
    feature_class: str
    quality_passed: bool
    x: float
    y: float
    peak_x: float
    peak_y: float
    centroid_nn_distance_px: float | None
    centroid_nn_detection_id: int | None
    centroid_nn_percentile: float | None
    peak_nn_distance_px: float | None
    peak_nn_detection_id: int | None
    peak_nn_percentile: float | None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SourceSeparationAuditResult:
    """源目录最近邻间距审计结果。"""

    catalog_path: str
    source_count: int
    quality_count: int
    thresholds: tuple[SourceSeparationThresholdRow, ...]
    classes: tuple[SourceSeparationClassRow, ...]
    targets: tuple[SourceSeparationTargetRow, ...]
    centroid_quantiles_px: dict[str, float]
    peak_quantiles_px: dict[str, float]
    parameters: dict[str, object]
    conclusion: str

    def as_dict(self) -> dict[str, object]:
        return {
            "catalog_path": self.catalog_path,
            "source_count": self.source_count,
            "quality_count": self.quality_count,
            "thresholds": [row.as_dict() for row in self.thresholds],
            "classes": [row.as_dict() for row in self.classes],
            "targets": [row.as_dict() for row in self.targets],
            "centroid_quantiles_px": dict(self.centroid_quantiles_px),
            "peak_quantiles_px": dict(self.peak_quantiles_px),
            "parameters": dict(self.parameters),
            "conclusion": self.conclusion,
        }


def _parse_bool(value: str, field: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    raise ValueError(f"catalog field {field!r} is not boolean: {value!r}")


def _load_catalog(path: str | Path) -> tuple[dict[str, str], ...]:
    catalog_path = Path(path)
    with catalog_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or ())
        missing = sorted(_REQUIRED_FIELDS - fieldnames)
        if missing:
            raise ValueError(f"source catalog is missing fields: {', '.join(missing)}")
        rows = tuple(dict(row) for row in reader)
    if len(rows) < 2:
        raise ValueError("source catalog must contain at least two sources")
    return rows


def _nearest_other(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """返回每个点的最近其它点距离和行号。"""

    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
        raise ValueError("points must be an N×2 array with N≥2")
    tree = cKDTree(points)
    distances, indices = tree.query(points, k=2)
    nearest_distance = np.asarray(distances[:, 1], dtype=np.float64)
    nearest_index = np.asarray(indices[:, 1], dtype=np.int64)
    row_indices = np.arange(len(points), dtype=np.int64)
    needs_fallback = nearest_index == row_indices
    for row_index in np.flatnonzero(needs_fallback):
        fallback_k = min(8, len(points))
        fallback_distances, fallback_indices = tree.query(points[row_index], k=fallback_k)
        choices = np.asarray(fallback_indices).reshape(-1)
        choice_distances = np.asarray(fallback_distances).reshape(-1)
        for distance, index in zip(choice_distances, choices, strict=True):
            if int(index) != int(row_index):
                nearest_distance[row_index] = float(distance)
                nearest_index[row_index] = int(index)
                break
    return nearest_distance, nearest_index


def _quantiles(values: np.ndarray) -> dict[str, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {}
    return {
        label: float(value)
        for label, value in zip(
            _QUANTILE_LABELS,
            np.quantile(finite, _QUANTILE_LEVELS),
            strict=True,
        )
    }


def _optional_float(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def run_source_separation_audit(
    catalog_csv: str | Path,
    *,
    target_ids: Iterable[int] = (),
    radii_px: Iterable[float] = _DEFAULT_RADII_PX,
) -> SourceSeparationAuditResult:
    """统计源目录的最近邻间距，并定位用户指定源。"""

    catalog_path = Path(catalog_csv)
    rows = _load_catalog(catalog_path)
    target_id_tuple = tuple(dict.fromkeys(int(value) for value in target_ids))
    radius_tuple = tuple(sorted({float(value) for value in radii_px}))
    if not radius_tuple or any(value <= 0 for value in radius_tuple):
        raise ValueError("radii_px must contain positive values")

    try:
        detection_ids = np.asarray([int(row["detection_id"]) for row in rows], dtype=np.int64)
        xy = np.asarray([(float(row["x"]), float(row["y"])) for row in rows], dtype=np.float64)
        peak_xy = np.asarray(
            [(float(row["peak_x"]), float(row["peak_y"])) for row in rows],
            dtype=np.float64,
        )
        quality = np.asarray([_parse_bool(row["quality_passed"], "quality_passed") for row in rows])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("source catalog contains an invalid numeric or boolean row") from exc
    if len(np.unique(detection_ids)) != len(detection_ids):
        raise ValueError("source catalog contains duplicate detection_id values")
    if not np.all(np.isfinite(xy)) or not np.all(np.isfinite(peak_xy)):
        raise ValueError("source catalog coordinates must be finite")

    centroid_distance, centroid_index = _nearest_other(xy)
    peak_distance, peak_index = _nearest_other(peak_xy)
    source_count = len(rows)
    quality_count = int(quality.sum())
    thresholds = tuple(
        SourceSeparationThresholdRow(
            radius_px=radius,
            centroid_endpoint_count=int(np.sum(centroid_distance <= radius)),
            centroid_endpoint_fraction=float(np.mean(centroid_distance <= radius)),
            centroid_quality_endpoint_count=int(np.sum(centroid_distance[quality] <= radius)),
            centroid_quality_endpoint_fraction=(
                float(np.mean(centroid_distance[quality] <= radius)) if quality_count else 0.0
            ),
            peak_endpoint_count=int(np.sum(peak_distance <= radius)),
            peak_endpoint_fraction=float(np.mean(peak_distance <= radius)),
            peak_quality_endpoint_count=int(np.sum(peak_distance[quality] <= radius)),
            peak_quality_endpoint_fraction=(
                float(np.mean(peak_distance[quality] <= radius)) if quality_count else 0.0
            ),
        )
        for radius in radius_tuple
    )

    feature_classes = np.asarray([row["feature_class"] for row in rows], dtype=object)
    class_rows: list[SourceSeparationClassRow] = []
    for feature_class in sorted(set(str(value) for value in feature_classes)):
        class_mask = feature_classes == feature_class
        class_centroid = centroid_distance[class_mask]
        class_peak = peak_distance[class_mask]
        class_rows.append(
            SourceSeparationClassRow(
                feature_class=feature_class,
                source_count=int(class_mask.sum()),
                quality_count=int(np.sum(quality[class_mask])),
                centroid_nn_median_px=float(np.median(class_centroid)),
                centroid_nn_p05_px=float(np.quantile(class_centroid, 0.05)),
                centroid_nn_p95_px=float(np.quantile(class_centroid, 0.95)),
                centroid_nn_le_4px_count=int(np.sum(class_centroid <= 4.0)),
                peak_nn_median_px=float(np.median(class_peak)),
                peak_nn_p05_px=float(np.quantile(class_peak, 0.05)),
                peak_nn_p95_px=float(np.quantile(class_peak, 0.95)),
                peak_nn_le_4px_count=int(np.sum(class_peak <= 4.0)),
            )
        )

    id_to_index = {int(value): index for index, value in enumerate(detection_ids)}
    missing_targets = sorted(set(target_id_tuple) - set(id_to_index))
    if missing_targets:
        joined = ", ".join(str(value) for value in missing_targets)
        raise ValueError(f"target detection_id values are not in source catalog: {joined}")
    target_rows: list[SourceSeparationTargetRow] = []
    for target_id in target_id_tuple:
        index = id_to_index[target_id]
        centroid_percentile = float(np.mean(centroid_distance <= centroid_distance[index]))
        peak_percentile = float(np.mean(peak_distance <= peak_distance[index]))
        target_rows.append(
            SourceSeparationTargetRow(
                detection_id=target_id,
                feature_class=str(feature_classes[index]),
                quality_passed=bool(quality[index]),
                x=float(xy[index, 0]),
                y=float(xy[index, 1]),
                peak_x=float(peak_xy[index, 0]),
                peak_y=float(peak_xy[index, 1]),
                centroid_nn_distance_px=_optional_float(centroid_distance[index]),
                centroid_nn_detection_id=int(detection_ids[centroid_index[index]]),
                centroid_nn_percentile=centroid_percentile,
                peak_nn_distance_px=_optional_float(peak_distance[index]),
                peak_nn_detection_id=int(detection_ids[peak_index[index]]),
                peak_nn_percentile=peak_percentile,
            )
        )

    centroid_quantiles = _quantiles(centroid_distance)
    peak_quantiles = _quantiles(peak_distance)
    conclusion = (
        "nearest-neighbor counts are candidate-geometry controls: they show how often local "
        "responses approach a radius, but do not establish independent stars, double-source "
        "identity, or physical truth. Peak and centroid coordinates must be interpreted separately."
    )
    return SourceSeparationAuditResult(
        catalog_path=str(catalog_path),
        source_count=source_count,
        quality_count=quality_count,
        thresholds=thresholds,
        classes=tuple(class_rows),
        targets=tuple(target_rows),
        centroid_quantiles_px=centroid_quantiles,
        peak_quantiles_px=peak_quantiles,
        parameters={
            "radii_px": list(radius_tuple),
            "coordinate_note": "x/y are measured centroids; peak_x/peak_y are integer peak coordinates",
            "endpoint_fraction_note": "each source contributes only its nearest-other endpoint; this is not an unordered pair count",
        },
        conclusion=conclusion,
    )


def _write_rows(path: Path, rows: Iterable[object], row_type: type[object]) -> None:
    values = [row.as_dict() for row in rows]  # type: ignore[attr-defined]
    fieldnames = [field.name for field in fields(row_type)]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(values)


def write_source_separation_artifacts(
    result: SourceSeparationAuditResult,
    out_dir: str | Path,
) -> Path:
    """写入总体、类别和指定源结果。"""

    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    _write_rows(output / "source_separation_thresholds.csv", result.thresholds, SourceSeparationThresholdRow)
    _write_rows(output / "source_separation_classes.csv", result.classes, SourceSeparationClassRow)
    _write_rows(output / "source_separation_targets.csv", result.targets, SourceSeparationTargetRow)
    payload = result.as_dict()
    payload["outputs"] = [
        "source_separation_thresholds.csv",
        "source_separation_classes.csv",
        "source_separation_targets.csv",
        "source_separation_summary.json",
    ]
    with (output / "source_separation_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return output
