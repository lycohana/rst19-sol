"""把特征类别和 hard-negative 放回图像空间中做条件化审计。

现有 ``source_feature_spatial.csv`` 统计类别的粗网格分布；本模块补充三个
容易被总数掩盖的问题：

* 某类是否真的集中在边缘或单个 detector 单元；
* 高 ``flux_snr`` 落选候选是否只来自一个空间热点；
* 指定候选（尤其是截图中的两个响应）周围是否确有近邻候选，以及它们
  分别属于什么算法审计类别。

它只读取 ``rst19-sources`` 导出的源级 CSV，不重新检测 FITS，也不把空间
集中、邻近或高 SNR 解释成伪影真值。``feature_class`` 仍是算法审计标签，
``quality_passed`` 仍是当前规则的结果。
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


SPATIAL_CONTEXT_METRIC_FIELDS: tuple[str, ...] = ("flux_snr", "filter_snr", "peak")


@dataclass(frozen=True, slots=True)
class FeatureSpatialClassSummary:
    """一个首要特征类别的空间、边缘和高 SNR 摘要。"""

    feature_class: str
    feature_class_label: str
    class_count: int
    class_quality_count: int
    class_rejected_count: int
    grid_size: int
    nonempty_grid_cells: int
    max_cell_count: int
    max_cell_share: float
    normalized_grid_entropy: float
    edge_margin_px: float
    edge_count: int
    edge_fraction: float
    high_flux_snr_threshold: float
    high_snr_rejected_count: int
    high_snr_max_cell_count: int
    high_snr_max_cell_share: float
    high_snr_hot_grid_row: int | None
    high_snr_hot_grid_column: int | None
    high_snr_hotspot_enrichment: float

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class FeatureSpatialHotspot:
    """一个类别在某个非空网格单元中的候选/高 SNR 计数。"""

    feature_class: str
    feature_class_label: str
    grid_row: int
    grid_column: int
    candidate_count: int
    quality_count: int
    rejected_count: int
    high_snr_rejected_count: int
    candidate_share: float
    high_snr_share: float
    high_snr_enrichment: float

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class FeatureSpatialTarget:
    """被选中的类别代表或显式目标的空间上下文。"""

    detection_id: int
    selection_reason: str
    rank_within_class: int | None
    feature_class: str
    feature_class_label: str
    quality_passed: bool
    flags: str
    x: float
    y: float
    grid_row: int
    grid_column: int
    edge_distance_px: float
    is_edge: bool
    flux_snr: float | None
    filter_snr: float | None
    class_count: int
    class_cell_count: int
    class_cell_share: float
    cell_high_snr_rejected_count: int
    cell_high_snr_share: float
    nearest_other_candidate_px: float | None
    neighbor_count_radius_5px: int
    neighbor_count_radius_10px: int
    neighbor_count_radius_20px: int
    neighbor_quality_count_radius_10px: int
    neighbor_classes_radius_10px: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class FeatureSpatialContextResult:
    """空间条件化审计的机器可读结果。"""

    catalog_path: str
    image_width: int
    image_height: int
    grid_size: int
    edge_margin_px: float
    high_flux_snr_threshold: float
    metric: str
    top_n: int
    excluded_classes: tuple[str, ...]
    summaries: tuple[FeatureSpatialClassSummary, ...]
    hotspots: tuple[FeatureSpatialHotspot, ...]
    targets: tuple[FeatureSpatialTarget, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "catalog_path": self.catalog_path,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "grid_size": self.grid_size,
            "edge_margin_px": self.edge_margin_px,
            "high_flux_snr_threshold": self.high_flux_snr_threshold,
            "metric": self.metric,
            "top_n": self.top_n,
            "excluded_classes": list(self.excluded_classes),
            "summaries": [item.as_dict() for item in self.summaries],
            "hotspots": [item.as_dict() for item in self.hotspots],
            "targets": [item.as_dict() for item in self.targets],
            "interpretation_guardrails": [
                "空间集中只表示候选分布，不能单独证明固定坏区或伪影。",
                "edge_distance_px 使用 detector 像素边界，不是天球边界，也不等于有效孔径。",
                "高 SNR 落选计数是 hard-negative 诊断，不是误检率或真星概率。",
                "邻近候选可能来自同一混合结构；必须结合 PSF、值域、独占像素和星表/WCS。",
            ],
        }


def _finite_float(row: dict[str, str], field: str) -> float | None:
    raw = row.get(field, "")
    if raw in ("", "nan", "NaN", "None"):
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _is_quality(row: dict[str, str]) -> bool:
    return row.get("quality_passed", "").strip().lower() == "true"


def _read_catalog(catalog_path: str | Path) -> tuple[Path, list[dict[str, str]]]:
    path = Path(catalog_path)
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
    except OSError as exc:
        raise OSError(f"无法读取源级 CSV：{path}") from exc
    if not rows:
        raise ValueError(f"源级 CSV 为空：{path}")
    required = {
        "detection_id",
        "x",
        "y",
        "feature_class",
        "feature_class_label",
        "quality_passed",
        "flags",
    }
    missing = sorted(required - set(rows[0]))
    if missing:
        raise ValueError(f"源级 CSV 缺少列：{', '.join(missing)}")
    return path, rows


def _grid_index(value: float, size: int, grid_size: int) -> int:
    return min(grid_size - 1, max(0, int(value / size * grid_size)))


def _grid_cell(row: dict[str, str], width: int, height: int, grid_size: int) -> tuple[int, int]:
    x = _finite_float(row, "x")
    y = _finite_float(row, "y")
    if x is None or y is None:
        raise ValueError(f"detection_id={row.get('detection_id')!r} 缺少有限 x/y")
    return _grid_index(y, height, grid_size), _grid_index(x, width, grid_size)


def _normalized_entropy(counts: Counter[tuple[int, int]], grid_size: int) -> float:
    total = sum(counts.values())
    if total <= 0 or grid_size <= 1:
        return 0.0
    return float(
        -sum((count / total) * math.log(count / total) for count in counts.values() if count > 0)
        / math.log(grid_size * grid_size)
    )


def _neighbor_classes(values: Sequence[dict[str, Any]]) -> str:
    counts = Counter(str(row["feature_class"]) for row in values)
    return "|".join(f"{name}:{counts[name]}" for name in sorted(counts))


def run_feature_spatial_context(
    catalog_path: str | Path,
    *,
    image_width: int = 4096,
    image_height: int = 4096,
    grid_size: int = 4,
    edge_margin_px: float = 16.0,
    high_flux_snr_threshold: float = 10.0,
    top_n: int = 5,
    metric: str = "flux_snr",
    target_ids: Iterable[int] = (),
    exclude_classes: Iterable[str] = ("other_rejected",),
) -> FeatureSpatialContextResult:
    """生成类别空间摘要，并为代表候选计算局部邻域上下文。"""

    if image_width <= 0 or image_height <= 0:
        raise ValueError("image_width/image_height 必须为正整数")
    if grid_size <= 0:
        raise ValueError("grid_size 必须为正整数")
    if edge_margin_px < 0:
        raise ValueError("edge_margin_px 不能为负数")
    if high_flux_snr_threshold < 0:
        raise ValueError("high_flux_snr_threshold 不能为负数")
    if top_n <= 0:
        raise ValueError("top_n 必须为正整数")
    if metric not in SPATIAL_CONTEXT_METRIC_FIELDS:
        raise ValueError(
            f"不支持的空间 hard-negative 指标：{metric!r}；"
            f"可选项为 {', '.join(SPATIAL_CONTEXT_METRIC_FIELDS)}"
        )

    path, raw_rows = _read_catalog(catalog_path)
    excluded = tuple(dict.fromkeys(str(value) for value in exclude_classes))
    requested_targets = tuple(dict.fromkeys(int(value) for value in target_ids))

    rows: list[dict[str, Any]] = []
    by_id: dict[int, dict[str, Any]] = {}
    by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for raw in raw_rows:
        try:
            detection_id = int(raw["detection_id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"源级 CSV 存在非法 detection_id：{raw.get('detection_id')!r}") from exc
        if detection_id in by_id:
            raise ValueError(f"源级 CSV 存在重复 detection_id：{detection_id}")
        x = _finite_float(raw, "x")
        y = _finite_float(raw, "y")
        if x is None or y is None:
            raise ValueError(f"detection_id={detection_id} 缺少有限 x/y")
        feature_class = raw.get("feature_class", "")
        if not feature_class:
            raise ValueError(f"detection_id={detection_id} 缺少 feature_class")
        grid_row, grid_column = _grid_cell(raw, image_width, image_height, grid_size)
        item: dict[str, Any] = {
            **raw,
            "_id": detection_id,
            "_x": x,
            "_y": y,
            "_grid_row": grid_row,
            "_grid_column": grid_column,
            "_quality": _is_quality(raw),
        }
        by_id[detection_id] = item
        rows.append(item)
        if feature_class not in excluded:
            by_class[feature_class].append(item)

    missing_targets = sorted(set(requested_targets) - set(by_id))
    if missing_targets:
        raise ValueError(f"源级 CSV 找不到 detection_id：{missing_targets}")

    class_cells: dict[str, Counter[tuple[int, int]]] = {}
    class_high_cells: dict[str, Counter[tuple[int, int]]] = {}
    summaries: list[FeatureSpatialClassSummary] = []
    hotspots: list[FeatureSpatialHotspot] = []
    top_rows: dict[int, tuple[int, str]] = {}

    for feature_class in sorted(by_class):
        class_rows = by_class[feature_class]
        cells = Counter((row["_grid_row"], row["_grid_column"]) for row in class_rows)
        high_rows = [
            row
            for row in class_rows
            if not row["_quality"]
            and (value := _finite_float(row, "flux_snr")) is not None
            and value >= high_flux_snr_threshold
        ]
        high_cells = Counter((row["_grid_row"], row["_grid_column"]) for row in high_rows)
        class_cells[feature_class] = cells
        class_high_cells[feature_class] = high_cells

        rejected = sum(not row["_quality"] for row in class_rows)
        edge_count = sum(
            min(row["_x"], row["_y"], image_width - 1 - row["_x"], image_height - 1 - row["_y"])
            <= edge_margin_px
            for row in class_rows
        )
        max_cell_count = max(cells.values(), default=0)
        max_high_count = max(high_cells.values(), default=0)
        max_high_cell = high_cells.most_common(1)[0][0] if high_cells else None
        total_high = len(high_rows)
        max_high_share = max_high_count / total_high if total_high else 0.0
        max_high_enrichment = 0.0
        if max_high_cell is not None and cells[max_high_cell] > 0 and class_rows:
            max_high_enrichment = (max_high_count / total_high) / (cells[max_high_cell] / len(class_rows))

        summaries.append(
            FeatureSpatialClassSummary(
                feature_class=feature_class,
                feature_class_label=class_rows[0].get("feature_class_label", feature_class),
                class_count=len(class_rows),
                class_quality_count=len(class_rows) - rejected,
                class_rejected_count=rejected,
                grid_size=grid_size,
                nonempty_grid_cells=len(cells),
                max_cell_count=max_cell_count,
                max_cell_share=max_cell_count / len(class_rows) if class_rows else 0.0,
                normalized_grid_entropy=_normalized_entropy(cells, grid_size),
                edge_margin_px=edge_margin_px,
                edge_count=edge_count,
                edge_fraction=edge_count / len(class_rows) if class_rows else 0.0,
                high_flux_snr_threshold=high_flux_snr_threshold,
                high_snr_rejected_count=total_high,
                high_snr_max_cell_count=max_high_count,
                high_snr_max_cell_share=max_high_share,
                high_snr_hot_grid_row=max_high_cell[0] if max_high_cell is not None else None,
                high_snr_hot_grid_column=max_high_cell[1] if max_high_cell is not None else None,
                high_snr_hotspot_enrichment=max_high_enrichment,
            )
        )

        for cell in sorted(cells):
            candidate_count = cells[cell]
            high_count = high_cells[cell]
            cell_quality_count = sum(
                row["_quality"]
                for row in class_rows
                if (row["_grid_row"], row["_grid_column"]) == cell
            )
            high_share = high_count / total_high if total_high else 0.0
            candidate_share = candidate_count / len(class_rows) if class_rows else 0.0
            hotspots.append(
                FeatureSpatialHotspot(
                    feature_class=feature_class,
                    feature_class_label=class_rows[0].get("feature_class_label", feature_class),
                    grid_row=cell[0],
                    grid_column=cell[1],
                    candidate_count=candidate_count,
                    quality_count=cell_quality_count,
                    rejected_count=candidate_count - cell_quality_count,
                    high_snr_rejected_count=high_count,
                    candidate_share=candidate_share,
                    high_snr_share=high_share,
                    high_snr_enrichment=(high_share / candidate_share if candidate_share else 0.0),
                )
            )

        ranked = sorted(
            (
                (value, row["_id"], row)
                for row in class_rows
                if not row["_quality"]
                and (value := _finite_float(row, metric)) is not None
            ),
            key=lambda item: (-item[0], item[1]),
        )
        for rank, (_value, detection_id, _row) in enumerate(ranked[:top_n], start=1):
            top_rows[detection_id] = (rank, "top_metric")

    for detection_id in requested_targets:
        if detection_id not in top_rows:
            top_rows[detection_id] = (None, "explicit")
        else:
            rank, _reason = top_rows[detection_id]
            top_rows[detection_id] = (rank, "top_metric|explicit")

    targets: list[FeatureSpatialTarget] = []
    for detection_id in sorted(top_rows):
        row = by_id[detection_id]
        feature_class = row["feature_class"]
        if feature_class in excluded:
            class_rows = [row]
            cells = Counter({(row["_grid_row"], row["_grid_column"]): 1})
            high_cells = Counter()
        else:
            class_rows = by_class[feature_class]
            cells = class_cells[feature_class]
            high_cells = class_high_cells[feature_class]
        class_count = len(class_rows)
        cell = (row["_grid_row"], row["_grid_column"])
        total_high = sum(high_cells.values())
        edge_distance = min(
            row["_x"], row["_y"], image_width - 1 - row["_x"], image_height - 1 - row["_y"]
        )
        others = [candidate for candidate in rows if candidate["_id"] != detection_id]
        distances = [math.hypot(candidate["_x"] - row["_x"], candidate["_y"] - row["_y"]) for candidate in others]
        nearest = min(distances) if distances else None
        in_5: list[dict[str, Any]] = []
        in_10: list[dict[str, Any]] = []
        in_20: list[dict[str, Any]] = []
        for candidate in others:
            distance = math.hypot(candidate["_x"] - row["_x"], candidate["_y"] - row["_y"])
            if distance <= 5.0:
                in_5.append(candidate)
            if distance <= 10.0:
                in_10.append(candidate)
            if distance <= 20.0:
                in_20.append(candidate)
        rank, reason = top_rows[detection_id]
        targets.append(
            FeatureSpatialTarget(
                detection_id=detection_id,
                selection_reason=reason,
                rank_within_class=rank,
                feature_class=feature_class,
                feature_class_label=row.get("feature_class_label", feature_class),
                quality_passed=row["_quality"],
                flags=row.get("flags", ""),
                x=row["_x"],
                y=row["_y"],
                grid_row=row["_grid_row"],
                grid_column=row["_grid_column"],
                edge_distance_px=edge_distance,
                is_edge=edge_distance <= edge_margin_px,
                flux_snr=_finite_float(row, "flux_snr"),
                filter_snr=_finite_float(row, "filter_snr"),
                class_count=class_count,
                class_cell_count=cells[cell],
                class_cell_share=cells[cell] / class_count if class_count else 0.0,
                cell_high_snr_rejected_count=high_cells[cell],
                cell_high_snr_share=high_cells[cell] / total_high if total_high else 0.0,
                nearest_other_candidate_px=nearest,
                neighbor_count_radius_5px=len(in_5),
                neighbor_count_radius_10px=len(in_10),
                neighbor_count_radius_20px=len(in_20),
                neighbor_quality_count_radius_10px=sum(candidate["_quality"] for candidate in in_10),
                neighbor_classes_radius_10px=_neighbor_classes(in_10),
            )
        )

    return FeatureSpatialContextResult(
        catalog_path=str(path),
        image_width=image_width,
        image_height=image_height,
        grid_size=grid_size,
        edge_margin_px=edge_margin_px,
        high_flux_snr_threshold=high_flux_snr_threshold,
        metric=metric,
        top_n=top_n,
        excluded_classes=excluded,
        summaries=tuple(summaries),
        hotspots=tuple(hotspots),
        targets=tuple(targets),
    )


def _write_csv(path: Path, rows: Sequence[Any]) -> None:
    if not rows:
        path.write_text("\n", encoding="utf-8-sig")
        return
    fieldnames = list(rows[0].as_dict().keys())
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.as_dict())


def write_feature_spatial_context_artifacts(
    result: FeatureSpatialContextResult,
    out_dir: str | Path,
) -> Path:
    """写出类别摘要、网格热点、代表候选上下文和 JSON。"""

    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "feature_spatial_context.json").write_text(
        json.dumps(result.as_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_csv(output / "feature_spatial_context_summary.csv", result.summaries)
    _write_csv(output / "feature_spatial_context_hotspots.csv", result.hotspots)
    _write_csv(output / "feature_spatial_context_targets.csv", result.targets)
    return output


__all__ = [
    "SPATIAL_CONTEXT_METRIC_FIELDS",
    "FeatureSpatialClassSummary",
    "FeatureSpatialHotspot",
    "FeatureSpatialTarget",
    "FeatureSpatialContextResult",
    "run_feature_spatial_context",
    "write_feature_spatial_context_artifacts",
]
