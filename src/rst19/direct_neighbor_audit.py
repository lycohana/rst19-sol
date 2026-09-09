"""候选的直接近邻边审计。

父源组使用传递连通分量回答“哪些候选最终连成一组”，但半径增大时，
候选 A 与 B、B 与 C 的近邻关系可能把 A、B、C 一起连起来。这个模块
保留每一条直接近邻边，不做传递合并，用来区分局部 pair 证据和父组的
连锁效应。

本模块只读取已经导出的 ``source_catalog.csv``，不读取 FITS、不重跑检测、
不修改 ``quality_passed``、GUI 或缓存。``independent_pair_candidate`` 仍
只是当前字段下的工程候选，不是物理双星或恒星真值。
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from scipy.spatial import cKDTree

from .source_group_audit import CandidateGroupSource, load_candidate_group_sources


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
class DirectNeighborPairRow:
    """一条不经过传递合并的直接近邻边。"""

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
    independent_psf_evidence: bool
    classification: str
    reason: str
    is_target_pair: bool

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class DirectNeighborSummaryRow:
    """按两端类别组合汇总直接近邻边。"""

    feature_class_a: str
    feature_class_b: str
    pair_count: int
    cross_class_count: int
    both_quality_count: int
    unresolved_pair_count: int
    independent_pair_candidate_count: int
    centroid_distance_median_px: float | None
    peak_distance_median_px: float | None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class DirectNeighborTargetRow:
    """指定 pair 的直接近邻定位；即使超出半径也保留距离。"""

    detection_id_a: int
    detection_id_b: int
    centroid_distance_px: float
    peak_distance_px: float
    feature_class_a: str
    feature_class_b: str
    quality_passed_a: bool
    quality_passed_b: bool
    flags_a: str
    flags_b: str
    independent_psf_evidence: bool
    in_pair_radius: bool
    classification: str
    reason: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class DirectNeighborAuditResult:
    """直接近邻边审计结果。"""

    catalog_path: str
    source_count: int
    quality_count: int
    pair_radius_px: float
    direct_pair_count: int
    unresolved_pair_count: int
    independent_pair_candidate_count: int
    target_ids: tuple[int, ...]
    pairs: tuple[DirectNeighborPairRow, ...]
    summaries: tuple[DirectNeighborSummaryRow, ...]
    target: DirectNeighborTargetRow | None
    parameters: dict[str, object]
    conclusion: str

    def as_dict(self) -> dict[str, object]:
        return {
            "catalog_path": self.catalog_path,
            "source_count": self.source_count,
            "quality_count": self.quality_count,
            "pair_radius_px": self.pair_radius_px,
            "direct_pair_count": self.direct_pair_count,
            "unresolved_pair_count": self.unresolved_pair_count,
            "independent_pair_candidate_count": self.independent_pair_candidate_count,
            "target_ids": list(self.target_ids),
            "pairs": [row.as_dict() for row in self.pairs],
            "summaries": [row.as_dict() for row in self.summaries],
            "target": self.target.as_dict() if self.target is not None else None,
            "parameters": dict(self.parameters),
            "conclusion": self.conclusion,
            "interpretation_guardrails": [
                "直接近邻边不包含传递连通关系；边数不能与父源组数相加。",
                "feature_class 和 flags 是 detector-level 审计标签，不是物理天体类别。",
                "independent_pair_candidate 仍需要真实 PSF、注入、星表/WCS 或人工真值确认。",
                "直接近邻距离、候选数和质量数都不是双星概率、precision、FDR 或物理星数。",
            ],
        }


def _validate_radius(psf_fwhm: float, pair_radius_px: float | None) -> tuple[float, float]:
    if not math.isfinite(float(psf_fwhm)) or float(psf_fwhm) <= 0:
        raise ValueError("psf_fwhm must be finite and positive")
    radius = max(2.5, 1.5 * float(psf_fwhm)) if pair_radius_px is None else float(pair_radius_px)
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError("pair_radius_px must be finite and positive")
    return float(psf_fwhm), float(radius)


def _validate_target_ids(target_ids: Iterable[int]) -> tuple[int, ...]:
    values = tuple(dict.fromkeys(int(value) for value in target_ids))
    if len(values) not in (0, 2):
        raise ValueError("target_ids must contain either zero or two distinct detection IDs")
    return values


def _psf_evidence(
    source: CandidateGroupSource,
    *,
    delta_bic_min: float,
    component_snr_min: float,
) -> bool:
    return bool(
        source.deblend_delta_bic is not None
        and source.deblend_component_snr is not None
        and source.deblend_delta_bic >= delta_bic_min
        and source.deblend_component_snr >= component_snr_min
    )


def _pair_fields(
    first: CandidateGroupSource,
    second: CandidateGroupSource,
    *,
    target_ids: frozenset[int],
    delta_bic_min: float,
    component_snr_min: float,
) -> DirectNeighborPairRow:
    first, second = sorted((first, second), key=lambda source: source.detection_id)
    centroid_distance = math.hypot(first.x - second.x, first.y - second.y)
    peak_distance = math.hypot(first.peak_x - second.peak_x, first.peak_y - second.peak_y)
    flags = tuple(sorted(set(first.flags).union(second.flags)))
    blocking_flags = tuple(flag for flag in flags if flag in _INDEPENDENCE_FLAGS)
    both_quality = first.quality_passed and second.quality_passed
    psf_evidence = _psf_evidence(
        first,
        delta_bic_min=delta_bic_min,
        component_snr_min=component_snr_min,
    ) or _psf_evidence(
        second,
        delta_bic_min=delta_bic_min,
        component_snr_min=component_snr_min,
    )
    independent = bool(both_quality and not blocking_flags and psf_evidence)
    if independent:
        classification = "independent_pair_candidate"
        reason = "两端质量、结构旗标和当前双PSF工程线满足；仍需真值确认"
    else:
        classification = "unresolved_pair"
        reasons = ["两端为直接近邻边"]
        if not both_quality:
            reasons.append("至少一端未通过质量层")
        if blocking_flags:
            reasons.append(f"含结构/值域旗标 {','.join(blocking_flags)}")
        if not psf_evidence:
            reasons.append("没有达到双PSF独立证据线")
        reason = "；".join(reasons)
    return DirectNeighborPairRow(
        detection_id_a=first.detection_id,
        detection_id_b=second.detection_id,
        centroid_distance_px=float(centroid_distance),
        peak_distance_px=float(peak_distance),
        feature_class_a=first.feature_class,
        feature_class_b=second.feature_class,
        same_feature_class=first.feature_class == second.feature_class,
        quality_passed_a=first.quality_passed,
        quality_passed_b=second.quality_passed,
        both_quality_passed=both_quality,
        flags_a="|".join(first.flags),
        flags_b="|".join(second.flags),
        independent_psf_evidence=independent,
        classification=classification,
        reason=reason,
        is_target_pair={first.detection_id, second.detection_id} == target_ids,
    )


def _summary_rows(pairs: Sequence[DirectNeighborPairRow]) -> tuple[DirectNeighborSummaryRow, ...]:
    grouped: dict[tuple[str, str], list[DirectNeighborPairRow]] = defaultdict(list)
    for pair in pairs:
        key = tuple(sorted((pair.feature_class_a, pair.feature_class_b)))
        grouped[key].append(pair)
    rows: list[DirectNeighborSummaryRow] = []
    for (class_a, class_b), members in sorted(grouped.items()):
        rows.append(
            DirectNeighborSummaryRow(
                feature_class_a=class_a,
                feature_class_b=class_b,
                pair_count=len(members),
                cross_class_count=sum(not pair.same_feature_class for pair in members),
                both_quality_count=sum(pair.both_quality_passed for pair in members),
                unresolved_pair_count=sum(pair.classification == "unresolved_pair" for pair in members),
                independent_pair_candidate_count=sum(
                    pair.classification == "independent_pair_candidate" for pair in members
                ),
                centroid_distance_median_px=float(np.median([pair.centroid_distance_px for pair in members])),
                peak_distance_median_px=float(np.median([pair.peak_distance_px for pair in members])),
            )
        )
    return tuple(rows)


def _target_row(
    first: CandidateGroupSource,
    second: CandidateGroupSource,
    *,
    in_pair_radius: bool,
    delta_bic_min: float,
    component_snr_min: float,
) -> DirectNeighborTargetRow:
    pair = _pair_fields(
        first,
        second,
        target_ids=frozenset({first.detection_id, second.detection_id}),
        delta_bic_min=delta_bic_min,
        component_snr_min=component_snr_min,
    )
    if in_pair_radius:
        classification = pair.classification
        reason = pair.reason
    else:
        classification = "outside_pair_radius"
        reason = f"直接质心距离 {pair.centroid_distance_px:.3f}px 超出当前半径"
    return DirectNeighborTargetRow(
        detection_id_a=pair.detection_id_a,
        detection_id_b=pair.detection_id_b,
        centroid_distance_px=pair.centroid_distance_px,
        peak_distance_px=pair.peak_distance_px,
        feature_class_a=pair.feature_class_a,
        feature_class_b=pair.feature_class_b,
        quality_passed_a=pair.quality_passed_a,
        quality_passed_b=pair.quality_passed_b,
        flags_a=pair.flags_a,
        flags_b=pair.flags_b,
        independent_psf_evidence=pair.independent_psf_evidence,
        in_pair_radius=in_pair_radius,
        classification=classification,
        reason=reason,
    )


def run_direct_neighbor_audit(
    catalog_csv: str | Path,
    *,
    target_ids: Iterable[int] = (),
    psf_fwhm: float = 2.0,
    pair_radius_px: float | None = None,
    delta_bic_min: float = 10.0,
    component_snr_min: float = 5.0,
) -> DirectNeighborAuditResult:
    """在源表上保留所有直接近邻边，不做传递连通合并。"""

    psf_fwhm, pair_radius_px = _validate_radius(psf_fwhm, pair_radius_px)
    target_tuple = _validate_target_ids(target_ids)
    if not math.isfinite(float(delta_bic_min)) or float(delta_bic_min) <= 0:
        raise ValueError("delta_bic_min must be finite and positive")
    if not math.isfinite(float(component_snr_min)) or float(component_snr_min) <= 0:
        raise ValueError("component_snr_min must be finite and positive")

    sources = load_candidate_group_sources(catalog_csv)
    source_by_id = {source.detection_id: source for source in sources}
    missing = sorted(set(target_tuple).difference(source_by_id))
    if missing:
        raise ValueError(f"target detection_id values are not in source catalog: {', '.join(map(str, missing))}")

    coordinates = np.asarray([(source.x, source.y) for source in sources], dtype=np.float64)
    edges = cKDTree(coordinates).query_pairs(pair_radius_px, output_type="ndarray")
    target_set = frozenset(target_tuple)
    pairs = tuple(
        sorted(
            (
                _pair_fields(
                    sources[int(left)],
                    sources[int(right)],
                    target_ids=target_set,
                    delta_bic_min=float(delta_bic_min),
                    component_snr_min=float(component_snr_min),
                )
                for left, right in np.asarray(edges, dtype=np.int64).reshape(-1, 2)
            ),
            key=lambda pair: (pair.detection_id_a, pair.detection_id_b),
        )
    )

    target = None
    if target_tuple:
        first = source_by_id[target_tuple[0]]
        second = source_by_id[target_tuple[1]]
        centroid_distance = math.hypot(first.x - second.x, first.y - second.y)
        target = _target_row(
            first,
            second,
            in_pair_radius=centroid_distance <= pair_radius_px,
            delta_bic_min=float(delta_bic_min),
            component_snr_min=float(component_snr_min),
        )

    unresolved_count = sum(pair.classification == "unresolved_pair" for pair in pairs)
    independent_count = sum(pair.classification == "independent_pair_candidate" for pair in pairs)
    conclusion = (
        f"在 {pair_radius_px:g}px 质心半径下保留 {len(pairs):,} 条直接近邻边，"
        f"其中未分辨边 {unresolved_count:,} 条、独立候选边 {independent_count:,} 条；"
        "直接边不经过传递合并，不能与父源组数或物理恒星数互换。"
    )
    return DirectNeighborAuditResult(
        catalog_path=str(catalog_csv),
        source_count=len(sources),
        quality_count=sum(source.quality_passed for source in sources),
        pair_radius_px=pair_radius_px,
        direct_pair_count=len(pairs),
        unresolved_pair_count=unresolved_count,
        independent_pair_candidate_count=independent_count,
        target_ids=target_tuple,
        pairs=pairs,
        summaries=_summary_rows(pairs),
        target=target,
        parameters={
            "psf_fwhm": psf_fwhm,
            "pair_radius_px": pair_radius_px,
            "pair_radius_definition": "max(2.5 px, 1.5 * psf_fwhm) unless explicitly overridden",
            "delta_bic_min": float(delta_bic_min),
            "component_snr_min": float(component_snr_min),
            "edge_definition": "centroid distance <= pair_radius_px; no transitive union",
            "interpretation_boundary": "detector-level direct-neighbor audit; no pair truth or star probability",
        },
        conclusion=conclusion,
    )


def _write_rows(path: Path, rows: Iterable[object], row_type: type[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[field.name for field in fields(row_type)])
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def write_direct_neighbor_artifacts(
    result: DirectNeighborAuditResult,
    out_dir: str | Path,
) -> Path:
    """写出直接近邻边、类别汇总、目标和 JSON。"""

    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    _write_rows(output / "direct_neighbor_pairs.csv", result.pairs, DirectNeighborPairRow)
    _write_rows(output / "direct_neighbor_summaries.csv", result.summaries, DirectNeighborSummaryRow)
    if result.target is not None:
        _write_rows(output / "direct_neighbor_targets.csv", (result.target,), DirectNeighborTargetRow)
    else:
        (output / "direct_neighbor_targets.csv").write_text("", encoding="utf-8-sig")
    (output / "direct_neighbor_audit.json").write_text(
        json.dumps(result.as_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output


__all__ = [
    "DirectNeighborAuditResult",
    "DirectNeighborPairRow",
    "DirectNeighborSummaryRow",
    "DirectNeighborTargetRow",
    "run_direct_neighbor_audit",
    "write_direct_neighbor_artifacts",
]
