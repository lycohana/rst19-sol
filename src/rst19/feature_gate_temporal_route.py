"""把质量门路径和真实序列中的跨帧响应连接起来。

本模块是一个只读后处理器：它不重新读取 FITS，也不重新运行检测器。
输入分别是 ``rst19-feature-gate-route`` 的逐候选路径、
``rst19-feature-sequence`` 的非紧凑诊断源表，以及同一次序列实验的
持久性 JSON。输出把“候选被什么规则拦截”和“该坐标附近在 15 帧中
是否反复出现响应”放到同一条记录上。

这里的跨帧 ``presence`` 仍然是注册坐标的小邻域响应，不是星表身份。
特别地，紧凑质量候选没有逐源诊断表时，相关字段必须保持为空，不能把
“未提供逐源时序”误写成 0 次出现。这样可以区分 detector-level 的拒绝
路径、候选级时序证据和真正需要星表/WCS 或注入真值才能回答的问题。

本模块还输出诊断子组汇总：比较“候选位置再次被命中”和“仍保持同一
诊断机制”两种持久性。二者的差额只是 detector-level 的描述性指标，
不是噪点概率、恒星概率或伪影真值。
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any


_ROUTE_FIELDS = {
    "detection_id",
    "feature_class",
    "feature_class_label",
    "quality_passed",
    "route",
    "numeric_failure_reasons",
    "structural_flags",
    "flags",
}
_DIAGNOSTIC_FIELDS = {
    "detection_id",
    "feature_class",
    "feature_class_label",
    "diagnostic_subgroup",
    "diagnostic_subgroup_label",
    "diagnostic_subgroup_definition",
    "candidate_presence",
    "candidate_same_subgroup_presence",
    "quality_presence",
    "quality_same_subgroup_presence",
}
_ROUTE_ORDER = {
    "quality_passed": 0,
    "rejected_numeric_only": 1,
    "rejected_structural_only": 2,
    "rejected_numeric_and_structural": 3,
    "rejected_unexplained": 4,
}


@dataclass(frozen=True, slots=True)
class FeatureGateTemporalSourceRow:
    """逐候选的质量门路径和可用时序字段。"""

    detection_id: str
    feature_class: str
    feature_class_label: str
    quality_passed: bool
    route: str
    numeric_failure_reasons: str
    structural_flags: str
    flags: str
    temporal_source_covered: bool
    diagnostic_subgroup: str
    diagnostic_subgroup_label: str
    diagnostic_subgroup_definition: str
    candidate_presence: int | None
    candidate_same_subgroup_presence: int | None
    quality_presence: int | None
    quality_same_subgroup_presence: int | None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class FeatureGateTemporalRouteSummaryRow:
    """一个特征类别和一条质量门路径的时序汇总。"""

    feature_class: str
    feature_class_label: str
    route: str
    route_count: int
    temporal_source_count: int
    temporal_coverage_fraction: float
    candidate_presence_median: float | None
    candidate_presence_mean: float | None
    candidate_presence_ge_required_count: int | None
    candidate_presence_ge_required_fraction: float | None
    candidate_presence_all_frames_count: int | None
    candidate_presence_all_frames_fraction: float | None
    candidate_same_subgroup_ge_required_count: int | None
    candidate_same_subgroup_ge_required_fraction: float | None
    quality_presence_median: float | None
    quality_presence_mean: float | None
    quality_presence_ge_required_count: int | None
    quality_presence_ge_required_fraction: float | None
    quality_presence_all_frames_count: int | None
    quality_presence_all_frames_fraction: float | None
    quality_presence_any_count: int | None
    quality_presence_any_fraction: float | None
    quality_same_subgroup_ge_required_count: int | None
    quality_same_subgroup_ge_required_fraction: float | None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class FeatureGateTemporalSubgroupSummaryRow:
    """一个诊断子组在质量门路径和跨帧响应中的描述性汇总。"""

    feature_class: str
    feature_class_label: str
    diagnostic_subgroup: str
    diagnostic_subgroup_label: str
    diagnostic_subgroup_definition: str
    subgroup_count: int
    route_quality_passed_count: int
    route_rejected_numeric_only_count: int
    route_rejected_structural_only_count: int
    route_rejected_numeric_and_structural_count: int
    route_rejected_unexplained_count: int
    candidate_presence_median: float | None
    candidate_presence_ge_required_count: int | None
    candidate_presence_ge_required_fraction: float | None
    candidate_presence_all_frames_count: int | None
    candidate_same_subgroup_ge_required_count: int | None
    candidate_same_subgroup_ge_required_fraction: float | None
    mechanism_gap_fraction: float | None
    quality_presence_ge_required_count: int | None
    quality_presence_ge_required_fraction: float | None
    quality_same_subgroup_ge_required_count: int | None
    quality_same_subgroup_ge_required_fraction: float | None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class FeatureGateTemporalTargetRow:
    """指定候选在其诊断子组经验分布中的位置。

    ``*_le_fraction`` / ``*_ge_fraction`` 是含并列值的经验分布比例，
    只用于描述目标相对同子组候选的高低，不是置信区间或异常概率。
    """

    detection_id: str
    feature_class: str
    feature_class_label: str
    route: str
    temporal_source_covered: bool
    diagnostic_subgroup: str
    diagnostic_subgroup_label: str
    diagnostic_subgroup_definition: str
    subgroup_count: int
    candidate_presence: int | None
    candidate_presence_le_fraction: float | None
    candidate_presence_ge_fraction: float | None
    candidate_same_subgroup_presence: int | None
    candidate_same_subgroup_presence_le_fraction: float | None
    candidate_same_subgroup_presence_ge_fraction: float | None
    quality_presence: int | None
    quality_presence_le_fraction: float | None
    quality_presence_ge_fraction: float | None
    candidate_presence_ge_required: bool | None
    candidate_same_subgroup_ge_required: bool | None
    quality_presence_ge_required: bool | None
    candidate_and_same_ge_required: bool | None
    candidate_same_and_quality_ge_required: bool | None
    all_three_ge_required: bool | None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class FeatureGateTemporalRouteResult:
    """质量门路径与逐源时序证据的连接结果。"""

    route_csv: str
    diagnostic_sources_csv: str
    persistence_json: str
    frame_count: int
    required_presence: int
    association_radius_px: float
    association_method: str
    source_rows: tuple[FeatureGateTemporalSourceRow, ...]
    summary_rows: tuple[FeatureGateTemporalRouteSummaryRow, ...]
    subgroup_summary_rows: tuple[FeatureGateTemporalSubgroupSummaryRow, ...] = ()
    target_rows: tuple[FeatureGateTemporalTargetRow, ...] = ()

    def as_dict(self) -> dict[str, object]:
        return {
            "route_csv": self.route_csv,
            "diagnostic_sources_csv": self.diagnostic_sources_csv,
            "persistence_json": self.persistence_json,
            "frame_count": self.frame_count,
            "required_presence": self.required_presence,
            "association_radius_px": self.association_radius_px,
            "association_method": self.association_method,
            "source_row_count": len(self.source_rows),
            "temporal_source_row_count": sum(row.temporal_source_covered for row in self.source_rows),
            "summary_rows": [row.as_dict() for row in self.summary_rows],
            "subgroup_summary_rows": [row.as_dict() for row in self.subgroup_summary_rows],
            "target_rows": [row.as_dict() for row in self.target_rows],
            "interpretation_guardrails": [
                "temporal presence is a registered-coordinate neighborhood response, not a catalog identity",
                "blank temporal fields mean no source-level diagnostic row was supplied, not zero response",
                "a quality gate route is an observed detector-rule decomposition, not a counterfactual cause",
                "same_subgroup_presence is a mechanism-consistency diagnostic, not a physical artifact label",
                "target distribution fractions are tie-inclusive empirical descriptions, not probabilities or confidence intervals",
                "none of these counts is a noise probability, completeness, precision, or confirmed star count",
            ],
        }


def _read_csv(path: Path, required_fields: set[str]) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            missing = sorted(required_fields - set(reader.fieldnames or ()))
            if missing:
                raise ValueError(f"{path}: missing required columns: {', '.join(missing)}")
            rows = list(reader)
    except OSError:
        raise
    if not rows:
        raise ValueError(f"{path}: CSV has no rows")
    return rows


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def _integer(value: Any, *, field: str, path: Path) -> int:
    text = str(value).strip()
    try:
        parsed = float(text)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path}: field {field!r} is not an integer") from exc
    if not math.isfinite(parsed) or not parsed.is_integer():
        raise ValueError(f"{path}: field {field!r} is not an integer")
    return int(parsed)


def _optional_integer(value: Any, *, field: str, path: Path) -> int | None:
    if str(value).strip() == "":
        return None
    parsed = _integer(value, field=field, path=path)
    if parsed < 0:
        raise ValueError(f"{path}: field {field!r} must not be negative")
    return parsed


def _optional_float(value: Any, *, field: str, path: Path) -> float | None:
    if str(value).strip() == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path}: field {field!r} is not numeric") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{path}: field {field!r} is not finite")
    return parsed


def _metadata(path: Path) -> tuple[int, int, float, str]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path}: invalid persistence JSON") from exc
    frame_count = _integer(payload.get("frame_count"), field="frame_count", path=path)
    required_presence = _integer(payload.get("required_presence"), field="required_presence", path=path)
    association_radius = _optional_float(
        payload.get("association_radius_px"), field="association_radius_px", path=path
    )
    if association_radius is None:
        raise ValueError(f"{path}: association_radius_px is required")
    association_method = str(payload.get("association_method", "")).strip()
    if frame_count < 1 or not 1 <= required_presence <= frame_count or association_radius <= 0:
        raise ValueError(f"{path}: invalid frame/presence/association metadata")
    return frame_count, required_presence, association_radius, association_method


def _median(values: list[int]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[middle])
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _mean(values: list[int]) -> float | None:
    return sum(values) / len(values) if values else None


def _fraction(count: int | None, denominator: int) -> float | None:
    if count is None or denominator == 0:
        return None
    return count / denominator


def _empirical_fraction(
    value: int | None,
    values: list[int],
    *,
    relation: str,
) -> float | None:
    if value is None or not values:
        return None
    if relation == "le":
        count = sum(candidate <= value for candidate in values)
    elif relation == "ge":
        count = sum(candidate >= value for candidate in values)
    else:
        raise ValueError(f"unsupported empirical relation: {relation}")
    return count / len(values)


def _summarize_group(
    members: list[FeatureGateTemporalSourceRow],
    *,
    frame_count: int,
    required_presence: int,
) -> FeatureGateTemporalRouteSummaryRow:
    covered = [member for member in members if member.temporal_source_covered]
    candidate_presence = [member.candidate_presence for member in covered if member.candidate_presence is not None]
    candidate_same = [
        member.candidate_same_subgroup_presence
        for member in covered
        if member.candidate_same_subgroup_presence is not None
    ]
    quality_presence = [member.quality_presence for member in covered if member.quality_presence is not None]
    quality_same = [
        member.quality_same_subgroup_presence
        for member in covered
        if member.quality_same_subgroup_presence is not None
    ]
    covered_count = len(covered)
    candidate_ge_required = (
        sum(value >= required_presence for value in candidate_presence) if candidate_presence else None
    )
    candidate_all_frames = sum(value >= frame_count for value in candidate_presence) if candidate_presence else None
    candidate_same_ge_required = sum(value >= required_presence for value in candidate_same) if candidate_same else None
    quality_ge_required = sum(value >= required_presence for value in quality_presence) if quality_presence else None
    quality_all_frames = sum(value >= frame_count for value in quality_presence) if quality_presence else None
    quality_any = sum(value > 0 for value in quality_presence) if quality_presence else None
    quality_same_ge_required = sum(value >= required_presence for value in quality_same) if quality_same else None
    return FeatureGateTemporalRouteSummaryRow(
        feature_class=members[0].feature_class,
        feature_class_label=members[0].feature_class_label,
        route=members[0].route,
        route_count=len(members),
        temporal_source_count=covered_count,
        temporal_coverage_fraction=covered_count / len(members),
        candidate_presence_median=_median(candidate_presence),
        candidate_presence_mean=_mean(candidate_presence),
        candidate_presence_ge_required_count=candidate_ge_required,
        candidate_presence_ge_required_fraction=_fraction(candidate_ge_required, len(candidate_presence)),
        candidate_presence_all_frames_count=candidate_all_frames,
        candidate_presence_all_frames_fraction=_fraction(candidate_all_frames, len(candidate_presence)),
        candidate_same_subgroup_ge_required_count=candidate_same_ge_required,
        candidate_same_subgroup_ge_required_fraction=_fraction(candidate_same_ge_required, len(candidate_same)),
        quality_presence_median=_median(quality_presence),
        quality_presence_mean=_mean(quality_presence),
        quality_presence_ge_required_count=quality_ge_required,
        quality_presence_ge_required_fraction=_fraction(quality_ge_required, len(quality_presence)),
        quality_presence_all_frames_count=quality_all_frames,
        quality_presence_all_frames_fraction=_fraction(quality_all_frames, len(quality_presence)),
        quality_presence_any_count=quality_any,
        quality_presence_any_fraction=_fraction(quality_any, len(quality_presence)),
        quality_same_subgroup_ge_required_count=quality_same_ge_required,
        quality_same_subgroup_ge_required_fraction=_fraction(quality_same_ge_required, len(quality_same)),
    )


def _summarize_subgroup(
    members: list[FeatureGateTemporalSourceRow],
    *,
    frame_count: int,
    required_presence: int,
) -> FeatureGateTemporalSubgroupSummaryRow:
    """汇总一个诊断子组的路径组成与机制一致性差额。"""

    covered = [member for member in members if member.temporal_source_covered]
    candidate_presence = [
        member.candidate_presence for member in covered if member.candidate_presence is not None
    ]
    candidate_same = [
        member.candidate_same_subgroup_presence
        for member in covered
        if member.candidate_same_subgroup_presence is not None
    ]
    quality_presence = [
        member.quality_presence for member in covered if member.quality_presence is not None
    ]
    quality_same = [
        member.quality_same_subgroup_presence
        for member in covered
        if member.quality_same_subgroup_presence is not None
    ]
    candidate_ge = (
        sum(value >= required_presence for value in candidate_presence)
        if candidate_presence
        else None
    )
    candidate_same_ge = (
        sum(value >= required_presence for value in candidate_same) if candidate_same else None
    )
    quality_ge = (
        sum(value >= required_presence for value in quality_presence)
        if quality_presence
        else None
    )
    quality_same_ge = (
        sum(value >= required_presence for value in quality_same) if quality_same else None
    )
    mechanism_gap = None
    if (
        candidate_ge is not None
        and candidate_same_ge is not None
        and len(candidate_presence) == len(candidate_same)
    ):
        mechanism_gap = (candidate_ge - candidate_same_ge) / len(candidate_presence)
    route_counts = {
        route: sum(member.route == route for member in members) for route in _ROUTE_ORDER
    }
    return FeatureGateTemporalSubgroupSummaryRow(
        feature_class=members[0].feature_class,
        feature_class_label=members[0].feature_class_label,
        diagnostic_subgroup=members[0].diagnostic_subgroup,
        diagnostic_subgroup_label=members[0].diagnostic_subgroup_label,
        diagnostic_subgroup_definition=members[0].diagnostic_subgroup_definition,
        subgroup_count=len(members),
        route_quality_passed_count=route_counts["quality_passed"],
        route_rejected_numeric_only_count=route_counts["rejected_numeric_only"],
        route_rejected_structural_only_count=route_counts["rejected_structural_only"],
        route_rejected_numeric_and_structural_count=route_counts["rejected_numeric_and_structural"],
        route_rejected_unexplained_count=route_counts["rejected_unexplained"],
        candidate_presence_median=_median(candidate_presence),
        candidate_presence_ge_required_count=candidate_ge,
        candidate_presence_ge_required_fraction=_fraction(candidate_ge, len(candidate_presence)),
        candidate_presence_all_frames_count=(
            sum(value >= frame_count for value in candidate_presence) if candidate_presence else None
        ),
        candidate_same_subgroup_ge_required_count=candidate_same_ge,
        candidate_same_subgroup_ge_required_fraction=_fraction(candidate_same_ge, len(candidate_same)),
        mechanism_gap_fraction=mechanism_gap,
        quality_presence_ge_required_count=quality_ge,
        quality_presence_ge_required_fraction=_fraction(quality_ge, len(quality_presence)),
        quality_same_subgroup_ge_required_count=quality_same_ge,
        quality_same_subgroup_ge_required_fraction=_fraction(quality_same_ge, len(quality_same)),
    )


def _summarize_target(
    target: FeatureGateTemporalSourceRow,
    members: list[FeatureGateTemporalSourceRow],
    *,
    required_presence: int,
) -> FeatureGateTemporalTargetRow:
    """描述一个目标相对于所属诊断子组的经验位置。"""

    candidate_values = [
        member.candidate_presence for member in members if member.candidate_presence is not None
    ]
    same_values = [
        member.candidate_same_subgroup_presence
        for member in members
        if member.candidate_same_subgroup_presence is not None
    ]
    quality_values = [
        member.quality_presence for member in members if member.quality_presence is not None
    ]
    candidate_ready = (
        target.candidate_presence >= required_presence
        if target.candidate_presence is not None
        else None
    )
    same_ready = (
        target.candidate_same_subgroup_presence >= required_presence
        if target.candidate_same_subgroup_presence is not None
        else None
    )
    quality_ready = (
        target.quality_presence >= required_presence if target.quality_presence is not None else None
    )
    candidate_and_same = (
        candidate_ready and same_ready
        if candidate_ready is not None and same_ready is not None
        else None
    )
    candidate_same_and_quality = (
        candidate_ready and same_ready and quality_ready
        if candidate_ready is not None and same_ready is not None and quality_ready is not None
        else None
    )
    return FeatureGateTemporalTargetRow(
        detection_id=target.detection_id,
        feature_class=target.feature_class,
        feature_class_label=target.feature_class_label,
        route=target.route,
        temporal_source_covered=target.temporal_source_covered,
        diagnostic_subgroup=target.diagnostic_subgroup,
        diagnostic_subgroup_label=target.diagnostic_subgroup_label,
        diagnostic_subgroup_definition=target.diagnostic_subgroup_definition,
        subgroup_count=len(members),
        candidate_presence=target.candidate_presence,
        candidate_presence_le_fraction=_empirical_fraction(
            target.candidate_presence,
            candidate_values,
            relation="le",
        ),
        candidate_presence_ge_fraction=_empirical_fraction(
            target.candidate_presence,
            candidate_values,
            relation="ge",
        ),
        candidate_same_subgroup_presence=target.candidate_same_subgroup_presence,
        candidate_same_subgroup_presence_le_fraction=_empirical_fraction(
            target.candidate_same_subgroup_presence,
            same_values,
            relation="le",
        ),
        candidate_same_subgroup_presence_ge_fraction=_empirical_fraction(
            target.candidate_same_subgroup_presence,
            same_values,
            relation="ge",
        ),
        quality_presence=target.quality_presence,
        quality_presence_le_fraction=_empirical_fraction(
            target.quality_presence,
            quality_values,
            relation="le",
        ),
        quality_presence_ge_fraction=_empirical_fraction(
            target.quality_presence,
            quality_values,
            relation="ge",
        ),
        candidate_presence_ge_required=candidate_ready,
        candidate_same_subgroup_ge_required=same_ready,
        quality_presence_ge_required=quality_ready,
        candidate_and_same_ge_required=candidate_and_same,
        candidate_same_and_quality_ge_required=candidate_same_and_quality,
        all_three_ge_required=candidate_same_and_quality,
    )


def run_feature_gate_temporal_route(
    route_csv: str | Path,
    diagnostic_sources_csv: str | Path,
    persistence_json: str | Path,
    *,
    target_ids: Sequence[str] = (),
) -> FeatureGateTemporalRouteResult:
    """连接质量门路径与逐源跨帧响应，不重新访问 FITS。

    ``target_ids`` 只为指定候选生成相对于所属诊断子组的经验分布行；
    不传入时不生成目标表，也不会改变全量汇总。
    """

    route_path = Path(route_csv)
    diagnostic_path = Path(diagnostic_sources_csv)
    persistence_path = Path(persistence_json)
    frame_count, required_presence, association_radius, association_method = _metadata(persistence_path)
    route_records = _read_csv(route_path, _ROUTE_FIELDS)
    diagnostic_records = _read_csv(diagnostic_path, _DIAGNOSTIC_FIELDS)
    normalized_target_ids = tuple(
        dict.fromkeys(str(target_id).strip() for target_id in target_ids if str(target_id).strip())
    )

    diagnostic_by_id: dict[str, dict[str, str]] = {}
    for record in diagnostic_records:
        detection_id = str(record.get("detection_id", "")).strip()
        if not detection_id:
            raise ValueError(f"{diagnostic_path}: empty detection_id")
        if detection_id in diagnostic_by_id:
            raise ValueError(f"{diagnostic_path}: duplicate detection_id={detection_id}")
        diagnostic_by_id[detection_id] = record

    source_rows: list[FeatureGateTemporalSourceRow] = []
    seen_route_ids: set[str] = set()
    grouped: defaultdict[tuple[str, str, str], list[FeatureGateTemporalSourceRow]] = defaultdict(list)
    subgrouped: defaultdict[
        tuple[str, str, str, str, str], list[FeatureGateTemporalSourceRow]
    ] = defaultdict(list)
    for record in route_records:
        detection_id = str(record.get("detection_id", "")).strip()
        if not detection_id:
            raise ValueError(f"{route_path}: empty detection_id")
        if detection_id in seen_route_ids:
            raise ValueError(f"{route_path}: duplicate detection_id={detection_id}")
        seen_route_ids.add(detection_id)
        feature_class = str(record.get("feature_class", "")).strip()
        feature_label = str(record.get("feature_class_label", "")).strip() or feature_class
        route = str(record.get("route", "")).strip()
        if not feature_class or not route:
            raise ValueError(f"{route_path}: row {detection_id} has empty feature_class or route")
        diagnostic = diagnostic_by_id.get(detection_id)
        if diagnostic is None and route != "quality_passed":
            raise ValueError(
                f"{diagnostic_path}: missing temporal row for rejected route candidate {detection_id}"
            )
        covered = diagnostic is not None
        if diagnostic is not None and str(diagnostic.get("feature_class", "")).strip() != feature_class:
            raise ValueError(f"feature class mismatch for detection_id={detection_id}")
        row = FeatureGateTemporalSourceRow(
            detection_id=detection_id,
            feature_class=feature_class,
            feature_class_label=feature_label,
            quality_passed=_bool(record.get("quality_passed", "")),
            route=route,
            numeric_failure_reasons=str(record.get("numeric_failure_reasons", "")).strip(),
            structural_flags=str(record.get("structural_flags", "")).strip(),
            flags=str(record.get("flags", "")).strip(),
            temporal_source_covered=covered,
            diagnostic_subgroup=(str(diagnostic.get("diagnostic_subgroup", "")).strip() if diagnostic else ""),
            diagnostic_subgroup_label=(
                str(diagnostic.get("diagnostic_subgroup_label", "")).strip() if diagnostic else ""
            ),
            diagnostic_subgroup_definition=(
                str(diagnostic.get("diagnostic_subgroup_definition", "")).strip()
                if diagnostic
                else ""
            ),
            candidate_presence=(
                _optional_integer(
                    diagnostic.get("candidate_presence", ""),
                    field="candidate_presence",
                    path=diagnostic_path,
                )
                if diagnostic
                else None
            ),
            candidate_same_subgroup_presence=(
                _optional_integer(
                    diagnostic.get("candidate_same_subgroup_presence", ""),
                    field="candidate_same_subgroup_presence",
                    path=diagnostic_path,
                )
                if diagnostic
                else None
            ),
            quality_presence=(
                _optional_integer(
                    diagnostic.get("quality_presence", ""),
                    field="quality_presence",
                    path=diagnostic_path,
                )
                if diagnostic
                else None
            ),
            quality_same_subgroup_presence=(
                _optional_integer(
                    diagnostic.get("quality_same_subgroup_presence", ""),
                    field="quality_same_subgroup_presence",
                    path=diagnostic_path,
                )
                if diagnostic
                else None
            ),
        )
        source_rows.append(row)
        grouped[(feature_class, feature_label, route)].append(row)
        if row.temporal_source_covered and row.diagnostic_subgroup:
            subgrouped[
                (
                    row.feature_class,
                    row.feature_class_label,
                    row.diagnostic_subgroup,
                    row.diagnostic_subgroup_label,
                    row.diagnostic_subgroup_definition,
                )
            ].append(row)

    unknown_diagnostic_ids = sorted(set(diagnostic_by_id) - seen_route_ids)
    if unknown_diagnostic_ids:
        raise ValueError(
            f"{diagnostic_path}: contains {len(unknown_diagnostic_ids)} IDs absent from the route CSV"
        )

    summary_rows = [
        _summarize_group(members, frame_count=frame_count, required_presence=required_presence)
        for _key, members in grouped.items()
    ]
    summary_rows.sort(
        key=lambda row: (
            row.feature_class,
            _ROUTE_ORDER.get(row.route, len(_ROUTE_ORDER)),
            row.route,
        )
    )
    subgroup_summary_rows = [
        _summarize_subgroup(
            members,
            frame_count=frame_count,
            required_presence=required_presence,
        )
        for _key, members in subgrouped.items()
    ]
    subgroup_summary_rows.sort(
        key=lambda row: (row.feature_class, row.diagnostic_subgroup)
    )
    source_by_id = {row.detection_id: row for row in source_rows}
    missing_target_ids = sorted(set(normalized_target_ids) - set(source_by_id))
    if missing_target_ids:
        raise ValueError(
            f"{route_path}: target IDs absent from the route CSV: {', '.join(missing_target_ids)}"
        )
    target_rows: list[FeatureGateTemporalTargetRow] = []
    for target_id in normalized_target_ids:
        target = source_by_id[target_id]
        subgroup_members = []
        if target.temporal_source_covered and target.diagnostic_subgroup:
            subgroup_key = (
                target.feature_class,
                target.feature_class_label,
                target.diagnostic_subgroup,
                target.diagnostic_subgroup_label,
                target.diagnostic_subgroup_definition,
            )
            subgroup_members = subgrouped.get(subgroup_key, [])
        target_rows.append(
            _summarize_target(
                target,
                subgroup_members,
                required_presence=required_presence,
            )
        )
    return FeatureGateTemporalRouteResult(
        route_csv=str(route_path),
        diagnostic_sources_csv=str(diagnostic_path),
        persistence_json=str(persistence_path),
        frame_count=frame_count,
        required_presence=required_presence,
        association_radius_px=association_radius,
        association_method=association_method,
        source_rows=tuple(source_rows),
        summary_rows=tuple(summary_rows),
        subgroup_summary_rows=tuple(subgroup_summary_rows),
        target_rows=tuple(target_rows),
    )


def write_feature_gate_temporal_route_artifacts(
    result: FeatureGateTemporalRouteResult,
    output_dir: str | Path,
) -> Path:
    """写出逐候选连接表、路径汇总和 JSON。"""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    source_fields = [field.name for field in fields(FeatureGateTemporalSourceRow)]
    with (output / "feature_gate_temporal_route_sources.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=source_fields)
        writer.writeheader()
        writer.writerows(row.as_dict() for row in result.source_rows)
    summary_fields = [field.name for field in fields(FeatureGateTemporalRouteSummaryRow)]
    with (output / "feature_gate_temporal_route_summary.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(row.as_dict() for row in result.summary_rows)
    subgroup_fields = [field.name for field in fields(FeatureGateTemporalSubgroupSummaryRow)]
    with (output / "feature_gate_temporal_subgroup_summary.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=subgroup_fields)
        writer.writeheader()
        writer.writerows(row.as_dict() for row in result.subgroup_summary_rows)
    target_fields = [field.name for field in fields(FeatureGateTemporalTargetRow)]
    with (output / "feature_gate_temporal_targets.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=target_fields)
        writer.writeheader()
        writer.writerows(row.as_dict() for row in result.target_rows)
    (output / "feature_gate_temporal_routes.json").write_text(
        json.dumps(result.as_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output


__all__ = [
    "FeatureGateTemporalRouteResult",
    "FeatureGateTemporalSourceRow",
    "FeatureGateTemporalRouteSummaryRow",
    "FeatureGateTemporalSubgroupSummaryRow",
    "FeatureGateTemporalTargetRow",
    "run_feature_gate_temporal_route",
    "write_feature_gate_temporal_route_artifacts",
]
