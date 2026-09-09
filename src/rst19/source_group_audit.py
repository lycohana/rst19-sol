"""候选源组的低成本后处理审计。

检测器的局部极大值是“响应峰”，不一定是独立物理源。这个模块只读取
``rst19-sources`` 已导出的源表，在源级坐标上建立近邻图，把相互混叠的
候选组织为“父源组—子候选”。它不读取 FITS、不重跑卷积、不修改默认
``quality_passed``，也不把源组直接解释成物理恒星。

源组标签只有三种：

``isolated``
    组内只有一个候选；
``unresolved_group``
    组内有多个候选，但质量、值域或独立 PSF 证据不完整；
``independent_group_candidate``
    组内有多个候选，当前字段满足严格的独立性工程线，但仍需要星表、
    注入或人工真值确认。

这是一条解释/路由层，不是新的恒星概率模型。尤其不能因为某个组只有
一个“代表”就把它自动计为一颗恒星；代表只用于界面和复核排序。
"""

from __future__ import annotations

import csv
import json
import math
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Iterable, Sequence

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
    "flags",
}
_TRUE_VALUES = {"true", "1", "yes", "y"}
_FALSE_VALUES = {"false", "0", "no", "n"}
_INDEPENDENCE_FLAGS = frozenset(
    {
        "CODE_PATTERN",
        "NEGATIVE_OVERFLOW",
        "MASKED",
        "SATURATED",
        "LINE_ARTIFACT",
        "UNRESOLVED_BLEND",
        "INSUFFICIENT_PSF_SUPPORT",
        "NARROW",
        "BROAD",
        "ELONGATED",
        "DIFFUSE",
    }
)


@dataclass(frozen=True, slots=True)
class CandidateGroupSource:
    """用于组级审计的源表行。"""

    detection_id: int
    x: float
    y: float
    peak_x: float
    peak_y: float
    quality_passed: bool
    feature_class: str
    flags: tuple[str, ...]
    peak: float | None = None
    flux_snr: float | None = None
    filter_snr: float | None = None
    fwhm: float | None = None
    deblend_delta_bic: float | None = None
    deblend_component_snr: float | None = None


@dataclass(frozen=True, slots=True)
class CandidateGroupRow:
    """一个父源组；成员 ID 和类别使用 ``|`` 连接以便 CSV 阅读。"""

    group_id: int
    detection_ids: str
    member_count: int
    representative_detection_id: int
    classification: str
    quality_member_count: int
    centroid_span_px: float
    peak_span_px: float
    feature_classes: str
    flags: str
    independent_psf_evidence: bool
    reason: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CandidateGroupClassRow:
    """按源的首要特征类别统计其进入近邻组的情况。"""

    feature_class: str
    source_count: int
    quality_source_count: int
    multi_group_source_count: int
    multi_group_source_fraction: float
    unresolved_group_source_count: int
    independent_group_source_count: int

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CandidateGroupTargetRow:
    """用户指定候选对应的父源组。"""

    group_id: int
    target_detection_ids: str
    group_detection_ids: str
    classification: str
    representative_detection_id: int
    member_count: int
    centroid_span_px: float
    peak_span_px: float
    feature_classes: str
    flags: str
    independent_psf_evidence: bool
    reason: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CandidateGroupAuditResult:
    """候选源组审计结果。"""

    catalog_path: str
    source_count: int
    quality_count: int
    group_count: int
    multi_member_group_count: int
    unresolved_group_count: int
    independent_group_candidate_count: int
    groups: tuple[CandidateGroupRow, ...]
    classes: tuple[CandidateGroupClassRow, ...]
    targets: tuple[CandidateGroupTargetRow, ...]
    parameters: dict[str, object]
    conclusion: str

    def as_dict(self) -> dict[str, object]:
        return {
            "catalog_path": self.catalog_path,
            "source_count": self.source_count,
            "quality_count": self.quality_count,
            "group_count": self.group_count,
            "multi_member_group_count": self.multi_member_group_count,
            "unresolved_group_count": self.unresolved_group_count,
            "independent_group_candidate_count": self.independent_group_candidate_count,
            "groups": [row.as_dict() for row in self.groups],
            "classes": [row.as_dict() for row in self.classes],
            "targets": [row.as_dict() for row in self.targets],
            "parameters": dict(self.parameters),
            "conclusion": self.conclusion,
            "interpretation_guardrails": [
                "源组是 detector-level 近邻组织，不是物理恒星分组。",
                "isolated 只表示没有进入当前组半径，不表示已完成星表身份确认。",
                "independent_group_candidate 仍是候选，不能直接写成双星或多星。",
                "代表源只用于显示/排序，不覆盖组内子候选和质量拒绝原因。",
            ],
        }


def _parse_bool(value: str, field: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    raise ValueError(f"catalog field {field!r} is not boolean: {value!r}")


def _optional_float(row: dict[str, str], field: str, row_number: int) -> float | None:
    raw = str(row.get(field, "")).strip()
    if not raw or raw.lower() in {"none", "nan", "null", "—", "-"}:
        return None
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"catalog row {row_number}: {field} must be numeric, got {raw!r}") from exc
    if not math.isfinite(value):
        raise ValueError(f"catalog row {row_number}: {field} must be finite")
    return value


def _required_float(row: dict[str, str], field: str, row_number: int) -> float:
    value = _optional_float(row, field, row_number)
    if value is None:
        raise ValueError(f"catalog row {row_number}: missing {field}")
    return value


def _parse_flags(value: str) -> tuple[str, ...]:
    return tuple(
        token
        for token in (part.strip() for part in re.split(r"[|,;]", str(value)))
        if token
    )


def load_candidate_group_sources(path: str | Path) -> tuple[CandidateGroupSource, ...]:
    """读取 `rst19-sources` 源表的必要字段。"""

    catalog_path = Path(path)
    with catalog_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        actual_fields = set(reader.fieldnames or ())
        missing = sorted(_REQUIRED_FIELDS - actual_fields)
        if missing:
            raise ValueError(f"source catalog is missing fields: {', '.join(missing)}")
        sources: list[CandidateGroupSource] = []
        seen: set[int] = set()
        for row_number, raw in enumerate(reader, start=2):
            try:
                detection_id = int(str(raw.get("detection_id", "")).strip())
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"catalog row {row_number}: detection_id must be an integer"
                ) from exc
            if detection_id in seen:
                raise ValueError(f"catalog row {row_number}: duplicate detection_id {detection_id}")
            seen.add(detection_id)
            feature_class = str(raw.get("feature_class", "")).strip()
            if not feature_class:
                raise ValueError(f"catalog row {row_number}: feature_class is empty")
            sources.append(
                CandidateGroupSource(
                    detection_id=detection_id,
                    x=_required_float(raw, "x", row_number),
                    y=_required_float(raw, "y", row_number),
                    peak_x=_required_float(raw, "peak_x", row_number),
                    peak_y=_required_float(raw, "peak_y", row_number),
                    quality_passed=_parse_bool(raw.get("quality_passed", ""), "quality_passed"),
                    feature_class=feature_class,
                    flags=_parse_flags(raw.get("flags", "")),
                    peak=_optional_float(raw, "peak", row_number),
                    flux_snr=_optional_float(raw, "flux_snr", row_number),
                    filter_snr=_optional_float(raw, "filter_snr", row_number),
                    fwhm=_optional_float(raw, "fwhm", row_number),
                    deblend_delta_bic=_optional_float(raw, "deblend_delta_bic", row_number),
                    deblend_component_snr=_optional_float(raw, "deblend_component_snr", row_number),
                )
            )
    if not sources:
        raise ValueError("source catalog contains no sources")
    return tuple(sources)


def _validate_parameters(
    psf_fwhm: float,
    group_radius_px: float | None,
    delta_bic_min: float,
    component_snr_min: float,
) -> tuple[float, float, float, float]:
    values = (
        ("psf_fwhm", psf_fwhm),
        ("delta_bic_min", delta_bic_min),
        ("component_snr_min", component_snr_min),
    )
    for name, value in values:
        if not math.isfinite(float(value)) or float(value) <= 0:
            raise ValueError(f"{name} must be finite and positive")
    resolved_radius = max(2.5, 1.5 * float(psf_fwhm)) if group_radius_px is None else float(group_radius_px)
    if not math.isfinite(resolved_radius) or resolved_radius <= 0:
        raise ValueError("group_radius_px must be finite and positive")
    return float(psf_fwhm), resolved_radius, float(delta_bic_min), float(component_snr_min)


def _union_find(size: int, edges: np.ndarray) -> tuple[tuple[int, ...], ...]:
    parent = list(range(size))
    rank = [0] * size

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        if rank[left_root] < rank[right_root]:
            left_root, right_root = right_root, left_root
        parent[right_root] = left_root
        if rank[left_root] == rank[right_root]:
            rank[left_root] += 1

    for left, right in np.asarray(edges, dtype=np.int64).reshape(-1, 2):
        union(int(left), int(right))
    groups: dict[int, list[int]] = defaultdict(list)
    for index in range(size):
        groups[find(index)].append(index)
    return tuple(tuple(indices) for indices in groups.values())


def _distance(first: tuple[float, float], second: tuple[float, float]) -> float:
    return float(math.hypot(first[0] - second[0], first[1] - second[1]))


def _span(points: Sequence[tuple[float, float]]) -> float:
    if len(points) < 2:
        return 0.0
    return max(
        _distance(first, second)
        for index, first in enumerate(points[:-1])
        for second in points[index + 1 :]
    )


def _strength(
    source: CandidateGroupSource,
    *,
    delta_bic_min: float,
    component_snr_min: float,
) -> tuple[float, float, float, float, float]:
    """决定显示代表的排序，不产生物理可信度。"""

    def finite_or_floor(value: float | None) -> float:
        return float(value) if value is not None and math.isfinite(value) else -math.inf

    independent = bool(
        source.deblend_delta_bic is not None
        and source.deblend_component_snr is not None
        and source.deblend_delta_bic >= delta_bic_min
        and source.deblend_component_snr >= component_snr_min
    )
    return (
        float(source.quality_passed),
        float(independent),
        finite_or_floor(source.filter_snr),
        finite_or_floor(source.flux_snr),
        finite_or_floor(source.peak),
    )


def _group_row(
    group_id: int,
    sources: Sequence[CandidateGroupSource],
    indices: Sequence[int],
    *,
    delta_bic_min: float,
    component_snr_min: float,
) -> CandidateGroupRow:
    members = tuple(sorted((sources[index] for index in indices), key=lambda source: source.detection_id))
    representative = max(
        members,
        key=lambda source: _strength(
            source,
            delta_bic_min=delta_bic_min,
            component_snr_min=component_snr_min,
        ),
    )
    quality_count = sum(source.quality_passed for source in members)
    union_flags = tuple(sorted({flag for source in members for flag in source.flags}))
    classes = tuple(sorted({source.feature_class for source in members}))
    psf_evidence = any(
        source.deblend_delta_bic is not None
        and source.deblend_component_snr is not None
        and source.deblend_delta_bic >= delta_bic_min
        and source.deblend_component_snr >= component_snr_min
        for source in members
    )
    blocking_flags = tuple(sorted(set(union_flags).intersection(_INDEPENDENCE_FLAGS)))
    independent = bool(
        len(members) > 1
        and quality_count == len(members)
        and not blocking_flags
        and psf_evidence
    )
    if len(members) == 1:
        classification = "isolated"
        reason = "组内只有一个候选；仍需外部身份核验"
    elif independent:
        classification = "independent_group_candidate"
        reason = "当前质量、值域、双PSF工程线均满足；仍需星表/注入真值确认"
    else:
        classification = "unresolved_group"
        reasons = ["多个候选共享当前近邻组"]
        if quality_count < len(members):
            reasons.append("至少一个成员未通过质量层")
        if blocking_flags:
            reasons.append(f"含结构/值域旗标 {','.join(blocking_flags)}")
        if not psf_evidence:
            reasons.append("没有达到双PSF独立证据线")
        reason = "；".join(reasons)
    centroid_points = [(source.x, source.y) for source in members]
    peak_points = [(source.peak_x, source.peak_y) for source in members]
    return CandidateGroupRow(
        group_id=group_id,
        detection_ids="|".join(str(source.detection_id) for source in members),
        member_count=len(members),
        representative_detection_id=representative.detection_id,
        classification=classification,
        quality_member_count=int(quality_count),
        centroid_span_px=_span(centroid_points),
        peak_span_px=_span(peak_points),
        feature_classes="|".join(classes),
        flags="|".join(union_flags),
        independent_psf_evidence=independent,
        reason=reason,
    )


def _class_rows(
    sources: Sequence[CandidateGroupSource],
    groups: Sequence[CandidateGroupRow],
) -> tuple[CandidateGroupClassRow, ...]:
    group_by_source: dict[int, CandidateGroupRow] = {}
    for group in groups:
        for raw_id in group.detection_ids.split("|"):
            group_by_source[int(raw_id)] = group
    rows: list[CandidateGroupClassRow] = []
    for feature_class in sorted({source.feature_class for source in sources}):
        members = [source for source in sources if source.feature_class == feature_class]
        multi = [source for source in members if group_by_source[source.detection_id].member_count > 1]
        unresolved = [
            source
            for source in multi
            if group_by_source[source.detection_id].classification == "unresolved_group"
        ]
        independent = [
            source
            for source in multi
            if group_by_source[source.detection_id].classification == "independent_group_candidate"
        ]
        rows.append(
            CandidateGroupClassRow(
                feature_class=feature_class,
                source_count=len(members),
                quality_source_count=sum(source.quality_passed for source in members),
                multi_group_source_count=len(multi),
                multi_group_source_fraction=float(len(multi) / len(members)) if members else 0.0,
                unresolved_group_source_count=len(unresolved),
                independent_group_source_count=len(independent),
            )
        )
    return tuple(rows)


def run_candidate_group_audit(
    catalog_csv: str | Path,
    *,
    target_ids: Iterable[int] = (),
    psf_fwhm: float = 2.0,
    group_radius_px: float | None = None,
    delta_bic_min: float = 10.0,
    component_snr_min: float = 5.0,
) -> CandidateGroupAuditResult:
    """从源目录建立候选父源组并输出类别条件统计。"""

    psf_fwhm, group_radius_px, delta_bic_min, component_snr_min = _validate_parameters(
        psf_fwhm,
        group_radius_px,
        delta_bic_min,
        component_snr_min,
    )
    sources = load_candidate_group_sources(catalog_csv)
    coordinates = np.asarray([(source.x, source.y) for source in sources], dtype=np.float64)
    edges = cKDTree(coordinates).query_pairs(group_radius_px, output_type="ndarray")
    groups_indices = _union_find(len(sources), edges)
    groups_sorted = sorted(
        groups_indices,
        key=lambda indices: min(sources[index].detection_id for index in indices),
    )
    groups = tuple(
        _group_row(
            group_id=index,
            sources=sources,
            indices=indices,
            delta_bic_min=delta_bic_min,
            component_snr_min=component_snr_min,
        )
        for index, indices in enumerate(groups_sorted, start=1)
    )
    target_id_tuple = tuple(dict.fromkeys(int(value) for value in target_ids))
    source_ids = {source.detection_id for source in sources}
    missing_targets = sorted(set(target_id_tuple) - source_ids)
    if missing_targets:
        raise ValueError(f"target detection_id values are not in source catalog: {', '.join(map(str, missing_targets))}")
    targets: list[CandidateGroupTargetRow] = []
    for group in groups:
        group_ids = {int(value) for value in group.detection_ids.split("|")}
        matching_targets = sorted(group_ids.intersection(target_id_tuple))
        if not matching_targets:
            continue
        targets.append(
            CandidateGroupTargetRow(
                group_id=group.group_id,
                target_detection_ids="|".join(str(value) for value in matching_targets),
                group_detection_ids=group.detection_ids,
                classification=group.classification,
                representative_detection_id=group.representative_detection_id,
                member_count=group.member_count,
                centroid_span_px=group.centroid_span_px,
                peak_span_px=group.peak_span_px,
                feature_classes=group.feature_classes,
                flags=group.flags,
                independent_psf_evidence=group.independent_psf_evidence,
                reason=group.reason,
            )
        )
    multi_groups = [group for group in groups if group.member_count > 1]
    unresolved_groups = [group for group in multi_groups if group.classification == "unresolved_group"]
    independent_groups = [
        group for group in multi_groups if group.classification == "independent_group_candidate"
    ]
    conclusion = (
        f"源级近邻图将 {len(sources):,} 个候选组织为 {len(groups):,} 个父源组，"
        f"其中多成员组 {len(multi_groups):,} 个、未分辨组 {len(unresolved_groups):,} 个、"
        f"当前独立候选组 {len(independent_groups):,} 个。"
        "该结果只改变解释层次，不改变候选数、质量数或物理恒星真值。"
    )
    return CandidateGroupAuditResult(
        catalog_path=str(catalog_csv),
        source_count=len(sources),
        quality_count=sum(source.quality_passed for source in sources),
        group_count=len(groups),
        multi_member_group_count=len(multi_groups),
        unresolved_group_count=len(unresolved_groups),
        independent_group_candidate_count=len(independent_groups),
        groups=groups,
        classes=_class_rows(sources, groups),
        targets=tuple(targets),
        parameters={
            "psf_fwhm": psf_fwhm,
            "group_radius_px": group_radius_px,
            "group_radius_definition": "max(2.5 px, 1.5 * psf_fwhm) unless explicitly overridden",
            "delta_bic_min": delta_bic_min,
            "component_snr_min": component_snr_min,
            "edge_definition": "centroid distance <= group_radius_px; connected components are transitive",
            "independence_definition": (
                "multi-member group + every member quality_passed + no blocking flag + "
                "at least one member reaches delta_bic and component_snr thresholds"
            ),
            "interpretation_boundary": "detector-level parent grouping; no star probability, pair truth, precision, or FDR",
        },
        conclusion=conclusion,
    )


def _write_rows(path: Path, rows: Iterable[object], row_type: type[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    field_names = [field.name for field in fields(row_type)]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=field_names)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_candidate_group_artifacts(
    result: CandidateGroupAuditResult,
    out_dir: str | Path,
) -> Path:
    """写出组表、类别摘要、目标定位和 JSON。"""

    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    _write_rows(output / "source_groups.csv", result.groups, CandidateGroupRow)
    _write_rows(output / "source_group_class_summary.csv", result.classes, CandidateGroupClassRow)
    _write_rows(output / "source_group_targets.csv", result.targets, CandidateGroupTargetRow)
    (output / "source_group_summary.json").write_text(
        json.dumps(result.as_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output


__all__ = [
    "CandidateGroupAuditResult",
    "CandidateGroupClassRow",
    "CandidateGroupRow",
    "CandidateGroupSource",
    "CandidateGroupTargetRow",
    "load_candidate_group_sources",
    "run_candidate_group_audit",
    "write_candidate_group_artifacts",
]
