"""几何条件化的近邻机制背景审计。

这个只读工具不重新读取 FITS，也不把候选目录当作物理真值。它在测量质心
距离和整数峰距离同时满足给定条件的无序候选对中，统计两端的
``feature_class`` 是否一致、是否同时通过当前质量层，并把指定 pair 放回
这个 detector-level 背景中。

它回答的是一个比“局部峰像不像第二颗星”更窄的问题：截图中的两个框是否
具有同一种可解释的检测机制。跨类别、未成对质量通过只能作为联立 PSF、值域
和有效像素审计的升级信号，不能转换成噪点概率、双星概率或物理恒星身份。
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.spatial import cKDTree


_REQUIRED_FIELDS = {
    "detection_id",
    "x",
    "y",
    "peak_x",
    "peak_y",
    "quality_passed",
    "feature_class",
}
_TRUE_VALUES = {"true", "1", "yes", "y"}
_FALSE_VALUES = {"false", "0", "no", "n"}


@dataclass(frozen=True, slots=True)
class PairMechanismContextPairRow:
    """一个满足几何条件的无序候选近邻对。"""

    detection_id_a: int
    detection_id_b: int
    centroid_distance_px: float
    peak_distance_px: float
    feature_class_a: str
    feature_class_b: str
    same_feature_class: bool
    quality_passed_a: bool
    quality_passed_b: bool
    both_quality_passed: bool
    flags_a: str
    flags_b: str
    is_target_pair: bool

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PairMechanismContextSummaryRow:
    """按两端特征类别组合汇总的近邻对。"""

    feature_class_a: str
    feature_class_b: str
    pair_count: int
    cross_class_count: int
    both_quality_count: int
    both_quality_fraction: float
    centroid_distance_median_px: float | None
    peak_distance_median_px: float | None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PairMechanismContextTargetRow:
    """指定目标 pair 的几何位置和机制一致性。"""

    detection_id_a: int
    detection_id_b: int
    centroid_distance_px: float
    peak_distance_px: float
    feature_class_a: str
    feature_class_b: str
    same_feature_class: bool
    quality_passed_a: bool
    quality_passed_b: bool
    both_quality_passed: bool
    flags_a: str
    flags_b: str
    in_geometry_context: bool

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PairMechanismContextResult:
    """几何条件化近邻机制审计结果。"""

    catalog_path: str
    source_count: int
    quality_count: int
    target_ids: tuple[int, int]
    pairs: tuple[PairMechanismContextPairRow, ...]
    summaries: tuple[PairMechanismContextSummaryRow, ...]
    target: PairMechanismContextTargetRow
    parameters: dict[str, object]
    conclusion: str

    def as_dict(self) -> dict[str, object]:
        return {
            "catalog_path": self.catalog_path,
            "source_count": self.source_count,
            "quality_count": self.quality_count,
            "target_ids": list(self.target_ids),
            "pairs": [row.as_dict() for row in self.pairs],
            "summaries": [row.as_dict() for row in self.summaries],
            "target": self.target.as_dict(),
            "parameters": dict(self.parameters),
            "conclusion": self.conclusion,
            "interpretation_guardrails": [
                "近邻对来自当前候选目录，不是物理双星真值。",
                "feature_class 是 detector-level 规则标签，不是物理类别。",
                "几何条件内没有双质量通过，只说明当前候选层/质量层的局部背景，不是噪点率。",
                "跨类别或共享机制只能触发联合 PSF、值域、独占像素和星表/WCS 复核。",
            ],
        }


@dataclass(frozen=True, slots=True)
class _CatalogSource:
    detection_id: int
    x: float
    y: float
    peak_x: float
    peak_y: float
    quality_passed: bool
    feature_class: str
    flags: str


def _finite_float(row: dict[str, str], field: str) -> float:
    raw = str(row.get(field, "")).strip()
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{field} must be numeric, got {raw!r}") from exc
    if not math.isfinite(value):
        raise ValueError(f"{field} must be finite")
    return value


def _parse_bool(value: str, field: str) -> bool:
    normalised = str(value).strip().lower()
    if normalised in _TRUE_VALUES:
        return True
    if normalised in _FALSE_VALUES:
        return False
    raise ValueError(f"{field} must be boolean, got {value!r}")


def _read_catalog(path: str | Path) -> list[_CatalogSource]:
    catalog_path = Path(path)
    with catalog_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        actual_fields = set(reader.fieldnames or ())
        missing = sorted(_REQUIRED_FIELDS - actual_fields)
        if missing:
            raise ValueError(f"catalog missing fields: {', '.join(missing)}")
        rows: list[_CatalogSource] = []
        seen: set[int] = set()
        for raw in reader:
            try:
                detection_id = int(str(raw["detection_id"]).strip())
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"detection_id must be an integer, got {raw.get('detection_id')!r}"
                ) from exc
            if detection_id in seen:
                raise ValueError(f"duplicate detection_id: {detection_id}")
            seen.add(detection_id)
            feature_class = str(raw.get("feature_class", "")).strip()
            if not feature_class:
                raise ValueError(f"feature_class is empty for detection_id {detection_id}")
            rows.append(
                _CatalogSource(
                    detection_id=detection_id,
                    x=_finite_float(raw, "x"),
                    y=_finite_float(raw, "y"),
                    peak_x=_finite_float(raw, "peak_x"),
                    peak_y=_finite_float(raw, "peak_y"),
                    quality_passed=_parse_bool(raw.get("quality_passed", ""), "quality_passed"),
                    feature_class=feature_class,
                    flags=str(raw.get("flags", "")).strip(),
                )
            )
    if not rows:
        raise ValueError("catalog contains no sources")
    return rows


def _validate_geometry(
    max_centroid_distance_px: float,
    peak_distance_min_px: float,
    peak_distance_max_px: float,
) -> tuple[float, float, float]:
    values = (
        ("max_centroid_distance_px", max_centroid_distance_px),
        ("peak_distance_min_px", peak_distance_min_px),
        ("peak_distance_max_px", peak_distance_max_px),
    )
    for name, value in values:
        if not math.isfinite(float(value)) or float(value) <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if float(peak_distance_min_px) > float(peak_distance_max_px):
        raise ValueError("peak_distance_min_px must not exceed peak_distance_max_px")
    return float(max_centroid_distance_px), float(peak_distance_min_px), float(peak_distance_max_px)


def _make_pair_row(
    first: _CatalogSource,
    second: _CatalogSource,
    *,
    target_ids: frozenset[int],
) -> PairMechanismContextPairRow:
    first, second = sorted((first, second), key=lambda source: source.detection_id)
    centroid_distance = float(math.hypot(first.x - second.x, first.y - second.y))
    peak_distance = float(math.hypot(first.peak_x - second.peak_x, first.peak_y - second.peak_y))
    same_class = first.feature_class == second.feature_class
    return PairMechanismContextPairRow(
        detection_id_a=first.detection_id,
        detection_id_b=second.detection_id,
        centroid_distance_px=centroid_distance,
        peak_distance_px=peak_distance,
        feature_class_a=first.feature_class,
        feature_class_b=second.feature_class,
        same_feature_class=same_class,
        quality_passed_a=first.quality_passed,
        quality_passed_b=second.quality_passed,
        both_quality_passed=first.quality_passed and second.quality_passed,
        flags_a=first.flags,
        flags_b=second.flags,
        is_target_pair={first.detection_id, second.detection_id} == target_ids,
    )


def run_pair_mechanism_context(
    catalog_path: str | Path,
    target_ids: Iterable[int],
    *,
    max_centroid_distance_px: float = 3.5,
    peak_distance_min_px: float = 3.5,
    peak_distance_max_px: float = 4.8,
) -> PairMechanismContextResult:
    """把目标 pair 放回相同几何条件的候选近邻背景中。"""

    resolved_ids = tuple(sorted({int(value) for value in target_ids}))
    if len(resolved_ids) != 2:
        raise ValueError("target_ids must contain exactly two distinct detection IDs")
    max_centroid_distance_px, peak_distance_min_px, peak_distance_max_px = _validate_geometry(
        max_centroid_distance_px,
        peak_distance_min_px,
        peak_distance_max_px,
    )
    sources = _read_catalog(catalog_path)
    by_id = {source.detection_id: source for source in sources}
    missing_ids = [value for value in resolved_ids if value not in by_id]
    if missing_ids:
        raise ValueError(f"target detection IDs not in source catalog: {missing_ids}")

    coordinates = np.asarray([(source.x, source.y) for source in sources], dtype=float)
    candidate_pairs = cKDTree(coordinates).query_pairs(
        max_centroid_distance_px,
        output_type="ndarray",
    )
    candidate_pairs = np.asarray(candidate_pairs, dtype=int).reshape(-1, 2)
    target_id_set = frozenset(resolved_ids)
    pairs: list[PairMechanismContextPairRow] = []
    for first_index, second_index in candidate_pairs:
        first = sources[int(first_index)]
        second = sources[int(second_index)]
        peak_distance = math.hypot(first.peak_x - second.peak_x, first.peak_y - second.peak_y)
        if peak_distance < peak_distance_min_px or peak_distance > peak_distance_max_px:
            continue
        pairs.append(_make_pair_row(first, second, target_ids=target_id_set))
    pairs.sort(key=lambda row: (row.centroid_distance_px, row.detection_id_a, row.detection_id_b))

    grouped: dict[tuple[str, str], list[PairMechanismContextPairRow]] = defaultdict(list)
    for row in pairs:
        grouped[(row.feature_class_a, row.feature_class_b)].append(row)
    summaries: list[PairMechanismContextSummaryRow] = []
    for (class_a, class_b), grouped_rows in sorted(grouped.items()):
        both_quality_count = sum(row.both_quality_passed for row in grouped_rows)
        summaries.append(
            PairMechanismContextSummaryRow(
                feature_class_a=class_a,
                feature_class_b=class_b,
                pair_count=len(grouped_rows),
                cross_class_count=sum(not row.same_feature_class for row in grouped_rows),
                both_quality_count=both_quality_count,
                both_quality_fraction=float(both_quality_count / len(grouped_rows)),
                centroid_distance_median_px=float(
                    np.median([row.centroid_distance_px for row in grouped_rows])
                ),
                peak_distance_median_px=float(np.median([row.peak_distance_px for row in grouped_rows])),
            )
        )

    first = by_id[resolved_ids[0]]
    second = by_id[resolved_ids[1]]
    target_pair = _make_pair_row(first, second, target_ids=target_id_set)
    target = PairMechanismContextTargetRow(
        detection_id_a=target_pair.detection_id_a,
        detection_id_b=target_pair.detection_id_b,
        centroid_distance_px=target_pair.centroid_distance_px,
        peak_distance_px=target_pair.peak_distance_px,
        feature_class_a=target_pair.feature_class_a,
        feature_class_b=target_pair.feature_class_b,
        same_feature_class=target_pair.same_feature_class,
        quality_passed_a=target_pair.quality_passed_a,
        quality_passed_b=target_pair.quality_passed_b,
        both_quality_passed=target_pair.both_quality_passed,
        flags_a=target_pair.flags_a,
        flags_b=target_pair.flags_b,
        in_geometry_context=target_pair in pairs,
    )
    same_count = sum(row.same_feature_class for row in pairs)
    cross_count = len(pairs) - same_count
    both_quality_count = sum(row.both_quality_passed for row in pairs)
    conclusion = (
        f"几何条件内找到 {len(pairs)} 对候选近邻，其中同类别 {same_count} 对、跨类别 {cross_count} 对，"
        f"双质量通过 {both_quality_count} 对。目标 pair 的类别为 "
        f"{target.feature_class_a}+{target.feature_class_b}，"
        f"{('在' if target.in_geometry_context else '不在')}该背景窗口内。"
        "该结果只表示候选目录中的机制一致性背景，不是物理双星或噪点真值。"
    )
    parameters = {
        "max_centroid_distance_px": max_centroid_distance_px,
        "peak_distance_min_px": peak_distance_min_px,
        "peak_distance_max_px": peak_distance_max_px,
        "pair_definition": "unordered candidate pair; measured centroid and integer peak geometry",
        "interpretation_boundary": (
            "candidate-catalog mechanism context only; no physical star probability, precision, FDR, "
            "or pair truth"
        ),
    }
    return PairMechanismContextResult(
        catalog_path=str(catalog_path),
        source_count=len(sources),
        quality_count=sum(source.quality_passed for source in sources),
        target_ids=(resolved_ids[0], resolved_ids[1]),
        pairs=tuple(pairs),
        summaries=tuple(summaries),
        target=target,
        parameters=parameters,
        conclusion=conclusion,
    )


def _write_dataclass_rows(path: Path, rows: Iterable[object], row_type: type[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    field_names = [field.name for field in fields(row_type)]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=field_names)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_pair_mechanism_context_artifacts(
    result: PairMechanismContextResult,
    out_dir: str | Path,
) -> Path:
    """写出 CSV 和 JSON，不写入检测缓存。"""

    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    _write_dataclass_rows(
        output / "pair_mechanism_context_pairs.csv",
        result.pairs,
        PairMechanismContextPairRow,
    )
    _write_dataclass_rows(
        output / "pair_mechanism_context_summary.csv",
        result.summaries,
        PairMechanismContextSummaryRow,
    )
    _write_dataclass_rows(
        output / "pair_mechanism_context_target.csv",
        (result.target,),
        PairMechanismContextTargetRow,
    )
    (output / "pair_mechanism_context.json").write_text(
        json.dumps(result.as_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output


__all__ = [
    "PairMechanismContextPairRow",
    "PairMechanismContextResult",
    "PairMechanismContextSummaryRow",
    "PairMechanismContextTargetRow",
    "run_pair_mechanism_context",
    "write_pair_mechanism_context_artifacts",
]
