"""交叉审计提议器来源与 15 帧时序持久性。

``source_catalog.csv`` 记录首帧候选由哪些宽筛提议器提出；序列诊断则记录
同一首帧坐标在后续帧中的候选/质量邻域响应。本模块把两类证据按
``detection_id`` 和来源组合连接起来，回答一个容易被类别总平均掩盖的
问题：某个来源子组是否只是首帧容易提出，还是在 15 帧中也具有稳定的
位置响应和质量响应。

非 ``compact_quality`` 候选有逐源诊断记录，因此输出精确的来源—候选行；
紧凑质量候选当前只有按 proposal subgroup 汇总的时序表，因此只输出按
来源组合加权的子组汇总，不能把子组均值伪装成逐源时序。模块只读取
已有 CSV，不重跑 FITS、detector、质量层或缓存。所有数值仍是
detector-level 的注册邻域描述，不是星表身份、噪点概率、precision、FDR
或物理恒星数。
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np

from .source_proposal_peak_audit import METHOD_GROUP_NAMES, proposal_method_group


COMPACT_FEATURE_CLASS = "compact_quality"
COMPACT_SUBGROUP_TO_METHOD_GROUP = {
    "all_three": "all_three",
    "gaussian_only": "gaussian_only",
    "dog_only_deblend": "dog_only",
    "dog_only_no_deblend": "dog_only",
    "compact_other": "partial",
}


@dataclass(frozen=True, slots=True)
class SourceProposalTemporalRow:
    """一个非紧凑候选的来源与逐源 15 帧持久性记录。"""

    detection_id: int
    feature_class: str
    feature_class_label: str
    diagnostic_subgroup: str
    diagnostic_subgroup_label: str
    method_group: str
    proposal_methods: str
    quality_passed: bool
    candidate_presence: int
    candidate_same_subgroup_presence: int
    quality_presence: int
    quality_same_subgroup_presence: int
    frame_count: int
    required_presence: int
    association_radius_px: float
    detail_level: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


SOURCE_PROPOSAL_TEMPORAL_FIELDS = tuple(field.name for field in fields(SourceProposalTemporalRow))


@dataclass(frozen=True, slots=True)
class SourceProposalTemporalSummary:
    """特征类别 × 来源组合的 15 帧时序汇总。"""

    feature_class: str
    feature_class_label: str
    method_group: str
    detail_level: str
    aggregation_basis: str
    candidate_count: int
    quality_count: int
    class_count: int
    class_fraction: float
    frame_count: int
    required_presence: int
    association_radius_px: float
    candidate_presence_median: float | None
    candidate_presence_mean: float | None
    candidate_presence_ge_required_count: int
    candidate_presence_ge_required_fraction: float
    candidate_presence_all_frames_count: int
    candidate_presence_all_frames_fraction: float
    candidate_same_subgroup_presence_median: float | None
    candidate_same_subgroup_presence_mean: float | None
    candidate_same_subgroup_presence_ge_required_count: int | None
    candidate_same_subgroup_presence_ge_required_fraction: float | None
    candidate_same_subgroup_presence_all_frames_count: int | None
    candidate_same_subgroup_presence_all_frames_fraction: float | None
    quality_presence_median: float | None
    quality_presence_mean: float | None
    quality_presence_ge_required_count: int
    quality_presence_ge_required_fraction: float
    quality_presence_all_frames_count: int
    quality_presence_all_frames_fraction: float
    quality_same_subgroup_presence_median: float | None
    quality_same_subgroup_presence_mean: float | None
    quality_same_subgroup_presence_ge_required_count: int | None
    quality_same_subgroup_presence_ge_required_fraction: float | None
    quality_same_subgroup_presence_all_frames_count: int | None
    quality_same_subgroup_presence_all_frames_fraction: float | None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


SOURCE_PROPOSAL_TEMPORAL_SUMMARY_FIELDS = tuple(
    field.name for field in fields(SourceProposalTemporalSummary)
)


@dataclass(frozen=True, slots=True)
class SourceProposalTemporalTarget:
    """指定候选的来源子组与可用时序证据。"""

    detection_id: int
    feature_class: str
    feature_class_label: str
    method_group: str
    proposal_methods: str
    quality_passed: bool
    flags: str
    temporal_detail_available: bool
    detail_level: str
    diagnostic_subgroup: str
    candidate_presence: int | None
    candidate_same_subgroup_presence: int | None
    quality_presence: int | None
    quality_same_subgroup_presence: int | None
    frame_count: int | None
    required_presence: int | None
    association_radius_px: float | None
    method_group_count: int
    method_group_share_in_class: float
    note: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


SOURCE_PROPOSAL_TEMPORAL_TARGET_FIELDS = tuple(
    field.name for field in fields(SourceProposalTemporalTarget)
)


@dataclass(frozen=True, slots=True)
class SourceProposalTemporalAuditResult:
    """提议器来源与 15 帧时序持久性的交叉审计结果。"""

    catalog_path: str
    diagnostic_sources_path: str
    subgroup_persistence_path: str
    frame_count: int
    required_presence: int
    association_radius_px: float
    source_rows: tuple[SourceProposalTemporalRow, ...]
    summaries: tuple[SourceProposalTemporalSummary, ...]
    targets: tuple[SourceProposalTemporalTarget, ...]
    conclusion: str

    def as_dict(self) -> dict[str, object]:
        return {
            "catalog_path": self.catalog_path,
            "diagnostic_sources_path": self.diagnostic_sources_path,
            "subgroup_persistence_path": self.subgroup_persistence_path,
            "frame_count": self.frame_count,
            "required_presence": self.required_presence,
            "association_radius_px": self.association_radius_px,
            "source_row_count": len(self.source_rows),
            "summary_rows": [row.as_dict() for row in self.summaries],
            "target_rows": [row.as_dict() for row in self.targets],
            "conclusion": self.conclusion,
            "interpretation_guardrails": [
                "proposal_methods 是来源记录，不是独立投票或概率",
                "candidate/quality presence 是注册坐标邻域响应，不是星表身份",
                "compact_quality 只有来源子组聚合，空的逐源时序字段不是零响应",
                "同一来源组合内的持久性也不能替代 WCS、星表、PSF 和注入真值",
                "任何比例都不是噪点概率、precision、FDR、完备率或确认星数",
            ],
        }


def _read_csv(path: Path, required_fields: set[str]) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = sorted(required_fields - set(reader.fieldnames or ()))
        if missing:
            raise ValueError(f"{path}: missing required columns: {', '.join(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path}: CSV has no rows")
    return rows


def _integer(value: object, *, field: str, path: Path, row_number: int) -> int:
    text = str(value).strip()
    try:
        parsed = float(text)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path} row {row_number}: field {field!r} is not an integer") from exc
    if not math.isfinite(parsed) or not parsed.is_integer():
        raise ValueError(f"{path} row {row_number}: field {field!r} is not an integer")
    return int(parsed)


def _optional_float(value: object, *, field: str, path: Path, row_number: int) -> float | None:
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    try:
        parsed = float(text)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path} row {row_number}: field {field!r} is not numeric") from exc
    if not math.isfinite(parsed):
        return None
    return parsed


def _bool(value: object, *, field: str, path: Path, row_number: int) -> bool:
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    raise ValueError(f"{path} row {row_number}: field {field!r} is not boolean")


def _read_catalog(path: Path) -> dict[int, dict[str, object]]:
    rows = _read_csv(
        path,
        {
            "detection_id",
            "feature_class",
            "feature_class_label",
            "quality_passed",
            "flags",
            "proposal_methods",
        },
    )
    records: dict[int, dict[str, object]] = {}
    for row_number, raw in enumerate(rows, start=2):
        detection_id = _integer(raw.get("detection_id", ""), field="detection_id", path=path, row_number=row_number)
        if detection_id in records:
            raise ValueError(f"{path}: duplicate detection_id={detection_id}")
        methods = str(raw.get("proposal_methods", "")).strip()
        feature_class = str(raw.get("feature_class", "")).strip()
        if not feature_class:
            raise ValueError(f"{path} row {row_number}: empty feature_class")
        records[detection_id] = {
            "detection_id": detection_id,
            "feature_class": feature_class,
            "feature_class_label": str(raw.get("feature_class_label", "")).strip(),
            "method_group": proposal_method_group(methods),
            "proposal_methods": methods,
            "quality_passed": _bool(
                raw.get("quality_passed", ""),
                field="quality_passed",
                path=path,
                row_number=row_number,
            ),
            "flags": str(raw.get("flags", "")).strip(),
        }
    return records


def _validate_temporal_bounds(
    *,
    candidate_presence: int,
    candidate_same_subgroup_presence: int,
    quality_presence: int,
    quality_same_subgroup_presence: int,
    frame_count: int,
    required_presence: int,
    context: str,
) -> None:
    if frame_count < 1 or not 1 <= required_presence <= frame_count:
        raise ValueError(f"{context}: invalid frame_count/required_presence")
    values = {
        "candidate_presence": candidate_presence,
        "candidate_same_subgroup_presence": candidate_same_subgroup_presence,
        "quality_presence": quality_presence,
        "quality_same_subgroup_presence": quality_same_subgroup_presence,
    }
    for field, value in values.items():
        if not 0 <= value <= frame_count:
            raise ValueError(f"{context}: {field}={value} outside [0, {frame_count}]")
    if candidate_same_subgroup_presence > candidate_presence:
        raise ValueError(f"{context}: same-subgroup candidate presence exceeds candidate presence")
    if quality_same_subgroup_presence > quality_presence:
        raise ValueError(f"{context}: same-subgroup quality presence exceeds quality presence")


def _read_diagnostic_sources(
    path: Path,
    catalog: Mapping[int, Mapping[str, object]],
    *,
    frame_count: int,
    required_presence: int,
    association_radius_px: float,
) -> dict[int, SourceProposalTemporalRow]:
    rows = _read_csv(
        path,
        {
            "detection_id",
            "feature_class",
            "feature_class_label",
            "diagnostic_subgroup",
            "diagnostic_subgroup_label",
            "quality_passed",
            "proposal_methods",
            "candidate_presence",
            "candidate_same_subgroup_presence",
            "quality_presence",
            "quality_same_subgroup_presence",
        },
    )
    records: dict[int, SourceProposalTemporalRow] = {}
    optional_temporal_fields = {"frame_count", "required_presence", "association_radius_px"}
    supplied_optional_fields = optional_temporal_fields.intersection(rows[0])
    if supplied_optional_fields and supplied_optional_fields != optional_temporal_fields:
        missing_optional = sorted(optional_temporal_fields - supplied_optional_fields)
        raise ValueError(
            f"{path}: optional temporal metadata must be supplied as a complete set; "
            f"missing {', '.join(missing_optional)}"
        )
    for row_number, raw in enumerate(rows, start=2):
        detection_id = _integer(raw.get("detection_id", ""), field="detection_id", path=path, row_number=row_number)
        if detection_id in records:
            raise ValueError(f"{path}: duplicate detection_id={detection_id}")
        source = catalog.get(detection_id)
        if source is None:
            raise ValueError(f"{path} row {row_number}: detection_id={detection_id} is not in source catalog")
        feature_class = str(raw.get("feature_class", "")).strip()
        feature_label = str(raw.get("feature_class_label", "")).strip()
        methods = str(raw.get("proposal_methods", "")).strip()
        if (
            feature_class != source["feature_class"]
            or feature_label != source["feature_class_label"]
            or proposal_method_group(methods) != source["method_group"]
        ):
            raise ValueError(
                f"{path} row {row_number}: catalog/diagnostic source mismatch for detection_id={detection_id}"
            )
        quality_passed = _bool(raw.get("quality_passed", ""), field="quality_passed", path=path, row_number=row_number)
        if quality_passed != source["quality_passed"]:
            raise ValueError(f"{path} row {row_number}: quality_passed mismatch for detection_id={detection_id}")
        if supplied_optional_fields:
            row_frame_count = _integer(
                raw.get("frame_count", ""), field="frame_count", path=path, row_number=row_number
            )
            row_required_presence = _integer(
                raw.get("required_presence", ""),
                field="required_presence",
                path=path,
                row_number=row_number,
            )
            row_association_radius_px = _optional_float(
                raw.get("association_radius_px", ""),
                field="association_radius_px",
                path=path,
                row_number=row_number,
            )
            if (
                row_frame_count != frame_count
                or row_required_presence != required_presence
                or row_association_radius_px is None
                or not math.isclose(row_association_radius_px, association_radius_px, rel_tol=0.0, abs_tol=1e-9)
            ):
                raise ValueError(f"{path} row {row_number}: temporal metadata disagrees with subgroup persistence")
        candidate_presence = _integer(
            raw.get("candidate_presence", ""), field="candidate_presence", path=path, row_number=row_number
        )
        candidate_same = _integer(
            raw.get("candidate_same_subgroup_presence", ""),
            field="candidate_same_subgroup_presence",
            path=path,
            row_number=row_number,
        )
        quality_presence = _integer(
            raw.get("quality_presence", ""), field="quality_presence", path=path, row_number=row_number
        )
        quality_same = _integer(
            raw.get("quality_same_subgroup_presence", ""),
            field="quality_same_subgroup_presence",
            path=path,
            row_number=row_number,
        )
        _validate_temporal_bounds(
            candidate_presence=candidate_presence,
            candidate_same_subgroup_presence=candidate_same,
            quality_presence=quality_presence,
            quality_same_subgroup_presence=quality_same,
            frame_count=frame_count,
            required_presence=required_presence,
            context=f"{path} row {row_number}",
        )
        records[detection_id] = SourceProposalTemporalRow(
            detection_id=detection_id,
            feature_class=feature_class,
            feature_class_label=feature_label,
            diagnostic_subgroup=str(raw.get("diagnostic_subgroup", "")).strip(),
            diagnostic_subgroup_label=str(raw.get("diagnostic_subgroup_label", "")).strip(),
            method_group=str(source["method_group"]),
            proposal_methods=str(source["proposal_methods"]),
            quality_passed=quality_passed,
            candidate_presence=candidate_presence,
            candidate_same_subgroup_presence=candidate_same,
            quality_presence=quality_presence,
            quality_same_subgroup_presence=quality_same,
            frame_count=frame_count,
            required_presence=required_presence,
            association_radius_px=association_radius_px,
            detail_level="source_exact",
        )
    return records


def _read_compact_subgroups(path: Path) -> tuple[list[dict[str, object]], int, int, float]:
    rows = _read_csv(
        path,
        {
            "feature_class",
            "feature_class_label",
            "proposal_subgroup",
            "proposal_subgroup_label",
            "anchor_count",
            "anchor_quality_count",
            "frame_count",
            "required_presence",
            "association_radius_px",
            "candidate_mean_presence",
            "candidate_presence_ge_required_count",
            "candidate_presence_all_frames_count",
            "quality_mean_presence",
            "quality_presence_ge_required_count",
            "quality_presence_all_frames_count",
        },
    )
    parsed: list[dict[str, object]] = []
    frame_values: set[int] = set()
    required_values: set[int] = set()
    radius_values: set[float] = set()
    seen_subgroups: set[tuple[str, str]] = set()
    for row_number, raw in enumerate(rows, start=2):
        feature_class = str(raw.get("feature_class", "")).strip()
        subgroup = str(raw.get("proposal_subgroup", "")).strip()
        if (feature_class, subgroup) in seen_subgroups:
            raise ValueError(f"{path}: duplicate compact subgroup {feature_class}/{subgroup}")
        seen_subgroups.add((feature_class, subgroup))
        if feature_class != COMPACT_FEATURE_CLASS:
            raise ValueError(f"{path} row {row_number}: expected {COMPACT_FEATURE_CLASS!r}, got {feature_class!r}")
        method_group = COMPACT_SUBGROUP_TO_METHOD_GROUP.get(subgroup)
        if method_group is None:
            allowed = ", ".join(sorted(COMPACT_SUBGROUP_TO_METHOD_GROUP))
            raise ValueError(f"{path} row {row_number}: unknown compact proposal_subgroup={subgroup!r}; use {allowed}")
        anchor_count = _integer(raw.get("anchor_count", ""), field="anchor_count", path=path, row_number=row_number)
        anchor_quality_count = _integer(
            raw.get("anchor_quality_count", ""), field="anchor_quality_count", path=path, row_number=row_number
        )
        frame_count = _integer(raw.get("frame_count", ""), field="frame_count", path=path, row_number=row_number)
        required_presence = _integer(
            raw.get("required_presence", ""), field="required_presence", path=path, row_number=row_number
        )
        association_radius_px = _optional_float(
            raw.get("association_radius_px", ""),
            field="association_radius_px",
            path=path,
            row_number=row_number,
        )
        candidate_mean = _optional_float(
            raw.get("candidate_mean_presence", ""),
            field="candidate_mean_presence",
            path=path,
            row_number=row_number,
        )
        quality_mean = _optional_float(
            raw.get("quality_mean_presence", ""),
            field="quality_mean_presence",
            path=path,
            row_number=row_number,
        )
        candidate_ge = _integer(
            raw.get("candidate_presence_ge_required_count", ""),
            field="candidate_presence_ge_required_count",
            path=path,
            row_number=row_number,
        )
        candidate_all = _integer(
            raw.get("candidate_presence_all_frames_count", ""),
            field="candidate_presence_all_frames_count",
            path=path,
            row_number=row_number,
        )
        quality_ge = _integer(
            raw.get("quality_presence_ge_required_count", ""),
            field="quality_presence_ge_required_count",
            path=path,
            row_number=row_number,
        )
        quality_all = _integer(
            raw.get("quality_presence_all_frames_count", ""),
            field="quality_presence_all_frames_count",
            path=path,
            row_number=row_number,
        )
        if anchor_count < 1 or not 0 <= anchor_quality_count <= anchor_count:
            raise ValueError(f"{path} row {row_number}: invalid anchor counts")
        if frame_count < 1 or not 1 <= required_presence <= frame_count:
            raise ValueError(f"{path} row {row_number}: invalid frame_count/required_presence")
        if association_radius_px is None or association_radius_px <= 0:
            raise ValueError(f"{path} row {row_number}: association_radius_px must be positive")
        if candidate_mean is not None and not 0 <= candidate_mean <= frame_count:
            raise ValueError(f"{path} row {row_number}: candidate_mean_presence outside frame range")
        if quality_mean is not None and not 0 <= quality_mean <= frame_count:
            raise ValueError(f"{path} row {row_number}: quality_mean_presence outside frame range")
        for field, value in {
            "candidate_presence_ge_required_count": candidate_ge,
            "candidate_presence_all_frames_count": candidate_all,
            "quality_presence_ge_required_count": quality_ge,
            "quality_presence_all_frames_count": quality_all,
        }.items():
            if not 0 <= value <= anchor_count:
                raise ValueError(f"{path} row {row_number}: {field} outside anchor_count")
        frame_values.add(frame_count)
        required_values.add(required_presence)
        radius_values.add(float(association_radius_px))
        parsed.append(
            {
                "feature_class": feature_class,
                "feature_class_label": str(raw.get("feature_class_label", "")).strip(),
                "proposal_subgroup": subgroup,
                "proposal_subgroup_label": str(raw.get("proposal_subgroup_label", "")).strip(),
                "method_group": method_group,
                "anchor_count": anchor_count,
                "anchor_quality_count": anchor_quality_count,
                "frame_count": frame_count,
                "required_presence": required_presence,
                "association_radius_px": float(association_radius_px),
                "candidate_mean_presence": candidate_mean,
                "candidate_presence_ge_required_count": candidate_ge,
                "candidate_presence_all_frames_count": candidate_all,
                "quality_mean_presence": quality_mean,
                "quality_presence_ge_required_count": quality_ge,
                "quality_presence_all_frames_count": quality_all,
            }
        )
    if len(frame_values) != 1 or len(required_values) != 1 or len(radius_values) != 1:
        raise ValueError(f"{path}: temporal parameters are not constant across compact subgroup rows")
    return parsed, frame_values.pop(), required_values.pop(), radius_values.pop()


def _group_catalog(catalog: Mapping[int, Mapping[str, object]]) -> tuple[
    dict[tuple[str, str], list[Mapping[str, object]]],
    dict[str, list[Mapping[str, object]]],
]:
    by_group: dict[tuple[str, str], list[Mapping[str, object]]] = defaultdict(list)
    by_class: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for source in catalog.values():
        by_group[(str(source["feature_class"]), str(source["method_group"]))].append(source)
        by_class[str(source["feature_class"])].append(source)
    return by_group, by_class


def _median(values: Iterable[float]) -> float | None:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.median(np.asarray(finite, dtype=np.float64))) if finite else None


def _mean(values: Iterable[float]) -> float | None:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.mean(np.asarray(finite, dtype=np.float64))) if finite else None


def _fraction(count: int | None, total: int) -> float | None:
    return None if count is None else (count / total if total else 0.0)


def _exact_summary(
    members: Sequence[SourceProposalTemporalRow],
    catalog_group: Sequence[Mapping[str, object]],
    catalog_class: Sequence[Mapping[str, object]],
    *,
    frame_count: int,
    required_presence: int,
    association_radius_px: float,
) -> SourceProposalTemporalSummary:
    if not members:
        raise ValueError("cannot summarize an empty exact source group")
    candidate = [row.candidate_presence for row in members]
    candidate_same = [row.candidate_same_subgroup_presence for row in members]
    quality = [row.quality_presence for row in members]
    quality_same = [row.quality_same_subgroup_presence for row in members]
    candidate_ge = sum(value >= required_presence for value in candidate)
    candidate_all = sum(value == frame_count for value in candidate)
    candidate_same_ge = sum(value >= required_presence for value in candidate_same)
    candidate_same_all = sum(value == frame_count for value in candidate_same)
    quality_ge = sum(value >= required_presence for value in quality)
    quality_all = sum(value == frame_count for value in quality)
    quality_same_ge = sum(value >= required_presence for value in quality_same)
    quality_same_all = sum(value == frame_count for value in quality_same)
    count = len(members)
    class_count = len(catalog_class)
    quality_count = sum(bool(row["quality_passed"]) for row in catalog_group)
    return SourceProposalTemporalSummary(
        feature_class=members[0].feature_class,
        feature_class_label=members[0].feature_class_label,
        method_group=members[0].method_group,
        detail_level="source_exact",
        aggregation_basis="diagnostic_source_rows",
        candidate_count=count,
        quality_count=quality_count,
        class_count=class_count,
        class_fraction=count / class_count,
        frame_count=frame_count,
        required_presence=required_presence,
        association_radius_px=association_radius_px,
        candidate_presence_median=_median(candidate),
        candidate_presence_mean=_mean(candidate),
        candidate_presence_ge_required_count=candidate_ge,
        candidate_presence_ge_required_fraction=candidate_ge / count,
        candidate_presence_all_frames_count=candidate_all,
        candidate_presence_all_frames_fraction=candidate_all / count,
        candidate_same_subgroup_presence_median=_median(candidate_same),
        candidate_same_subgroup_presence_mean=_mean(candidate_same),
        candidate_same_subgroup_presence_ge_required_count=candidate_same_ge,
        candidate_same_subgroup_presence_ge_required_fraction=candidate_same_ge / count,
        candidate_same_subgroup_presence_all_frames_count=candidate_same_all,
        candidate_same_subgroup_presence_all_frames_fraction=candidate_same_all / count,
        quality_presence_median=_median(quality),
        quality_presence_mean=_mean(quality),
        quality_presence_ge_required_count=quality_ge,
        quality_presence_ge_required_fraction=quality_ge / count,
        quality_presence_all_frames_count=quality_all,
        quality_presence_all_frames_fraction=quality_all / count,
        quality_same_subgroup_presence_median=_median(quality_same),
        quality_same_subgroup_presence_mean=_mean(quality_same),
        quality_same_subgroup_presence_ge_required_count=quality_same_ge,
        quality_same_subgroup_presence_ge_required_fraction=quality_same_ge / count,
        quality_same_subgroup_presence_all_frames_count=quality_same_all,
        quality_same_subgroup_presence_all_frames_fraction=quality_same_all / count,
    )


def _weighted_mean(rows: Sequence[Mapping[str, object]], mean_field: str) -> float | None:
    weighted = [
        (int(row["anchor_count"]), row[mean_field])
        for row in rows
        if row[mean_field] is not None
    ]
    if not weighted:
        return None
    total = sum(weight for weight, _value in weighted)
    return sum(weight * float(value) for weight, value in weighted) / total if total else None


def _compact_summary(
    members: Sequence[Mapping[str, object]],
    catalog_group: Sequence[Mapping[str, object]],
    catalog_class: Sequence[Mapping[str, object]],
    *,
    frame_count: int,
    required_presence: int,
    association_radius_px: float,
) -> SourceProposalTemporalSummary:
    if not members:
        raise ValueError("cannot summarize an empty compact subgroup aggregation")
    count = sum(int(row["anchor_count"]) for row in members)
    quality_count = sum(int(row["anchor_quality_count"]) for row in members)
    if count != len(catalog_group) or quality_count != sum(bool(row["quality_passed"]) for row in catalog_group):
        raise ValueError("compact subgroup anchor counts do not match source catalog")
    candidate_ge = sum(int(row["candidate_presence_ge_required_count"]) for row in members)
    candidate_all = sum(int(row["candidate_presence_all_frames_count"]) for row in members)
    quality_ge = sum(int(row["quality_presence_ge_required_count"]) for row in members)
    quality_all = sum(int(row["quality_presence_all_frames_count"]) for row in members)
    return SourceProposalTemporalSummary(
        feature_class=COMPACT_FEATURE_CLASS,
        feature_class_label=str(catalog_class[0]["feature_class_label"]),
        method_group=str(members[0]["method_group"]),
        detail_level="subgroup_aggregate",
        aggregation_basis="compact_proposal_subgroup_persistence",
        candidate_count=count,
        quality_count=quality_count,
        class_count=len(catalog_class),
        class_fraction=count / len(catalog_class),
        frame_count=frame_count,
        required_presence=required_presence,
        association_radius_px=association_radius_px,
        candidate_presence_median=None,
        candidate_presence_mean=_weighted_mean(members, "candidate_mean_presence"),
        candidate_presence_ge_required_count=candidate_ge,
        candidate_presence_ge_required_fraction=candidate_ge / count,
        candidate_presence_all_frames_count=candidate_all,
        candidate_presence_all_frames_fraction=candidate_all / count,
        candidate_same_subgroup_presence_median=None,
        candidate_same_subgroup_presence_mean=None,
        candidate_same_subgroup_presence_ge_required_count=None,
        candidate_same_subgroup_presence_ge_required_fraction=None,
        candidate_same_subgroup_presence_all_frames_count=None,
        candidate_same_subgroup_presence_all_frames_fraction=None,
        quality_presence_median=None,
        quality_presence_mean=_weighted_mean(members, "quality_mean_presence"),
        quality_presence_ge_required_count=quality_ge,
        quality_presence_ge_required_fraction=quality_ge / count,
        quality_presence_all_frames_count=quality_all,
        quality_presence_all_frames_fraction=quality_all / count,
        quality_same_subgroup_presence_median=None,
        quality_same_subgroup_presence_mean=None,
        quality_same_subgroup_presence_ge_required_count=None,
        quality_same_subgroup_presence_ge_required_fraction=None,
        quality_same_subgroup_presence_all_frames_count=None,
        quality_same_subgroup_presence_all_frames_fraction=None,
    )


def _summaries(
    catalog: Mapping[int, Mapping[str, object]],
    source_rows: Mapping[int, SourceProposalTemporalRow],
    compact_rows: Sequence[Mapping[str, object]],
    *,
    frame_count: int,
    required_presence: int,
    association_radius_px: float,
) -> tuple[SourceProposalTemporalSummary, ...]:
    by_group, by_class = _group_catalog(catalog)
    exact_groups: dict[tuple[str, str], list[SourceProposalTemporalRow]] = defaultdict(list)
    for row in source_rows.values():
        exact_groups[(row.feature_class, row.method_group)].append(row)
    summaries: list[SourceProposalTemporalSummary] = []
    for key, members in sorted(exact_groups.items()):
        feature_class, method_group = key
        summaries.append(
            _exact_summary(
                sorted(members, key=lambda row: row.detection_id),
                by_group[key],
                by_class[feature_class],
                frame_count=frame_count,
                required_presence=required_presence,
                association_radius_px=association_radius_px,
            )
        )
    compact_by_group: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in compact_rows:
        compact_by_group[str(row["method_group"])].append(row)
    compact_class = by_class.get(COMPACT_FEATURE_CLASS, [])
    for method_group, members in sorted(compact_by_group.items()):
        key = (COMPACT_FEATURE_CLASS, method_group)
        summaries.append(
            _compact_summary(
                members,
                by_group[key],
                compact_class,
                frame_count=frame_count,
                required_presence=required_presence,
                association_radius_px=association_radius_px,
            )
        )
    expected_groups = set(by_group)
    actual_groups = {(row.feature_class, row.method_group) for row in summaries}
    if expected_groups != actual_groups:
        missing = sorted(expected_groups - actual_groups)
        extra = sorted(actual_groups - expected_groups)
        raise ValueError(f"summary source groups are incomplete: missing={missing!r}, extra={extra!r}")
    return tuple(sorted(summaries, key=lambda row: (row.feature_class, row.method_group)))


def _targets(
    catalog: Mapping[int, Mapping[str, object]],
    source_rows: Mapping[int, SourceProposalTemporalRow],
    summaries: Sequence[SourceProposalTemporalSummary],
    target_ids: Sequence[int],
) -> tuple[SourceProposalTemporalTarget, ...]:
    requested = tuple(dict.fromkeys(int(value) for value in target_ids))
    if not requested:
        return ()
    missing = sorted(set(requested) - set(catalog))
    if missing:
        raise ValueError(f"target detection_id not found in source catalog: {', '.join(map(str, missing))}")
    by_group_count = {(row.feature_class, row.method_group): row.candidate_count for row in summaries}
    by_class_count: dict[str, int] = defaultdict(int)
    for row in summaries:
        by_class_count[row.feature_class] = row.class_count
    output: list[SourceProposalTemporalTarget] = []
    for detection_id in requested:
        source = catalog[detection_id]
        feature_class = str(source["feature_class"])
        method_group = str(source["method_group"])
        temporal = source_rows.get(detection_id)
        if temporal is None:
            output.append(
                SourceProposalTemporalTarget(
                    detection_id=detection_id,
                    feature_class=feature_class,
                    feature_class_label=str(source["feature_class_label"]),
                    method_group=method_group,
                    proposal_methods=str(source["proposal_methods"]),
                    quality_passed=bool(source["quality_passed"]),
                    flags=str(source["flags"]),
                    temporal_detail_available=False,
                    detail_level="subgroup_aggregate",
                    diagnostic_subgroup="",
                    candidate_presence=None,
                    candidate_same_subgroup_presence=None,
                    quality_presence=None,
                    quality_same_subgroup_presence=None,
                    frame_count=None,
                    required_presence=None,
                    association_radius_px=None,
                    method_group_count=by_group_count[(feature_class, method_group)],
                    method_group_share_in_class=(
                        by_group_count[(feature_class, method_group)] / by_class_count[feature_class]
                    ),
                    note="该来源组只有 compact proposal subgroup 汇总，未提供逐源时序；空值不是零响应",
                )
            )
            continue
        output.append(
            SourceProposalTemporalTarget(
                detection_id=detection_id,
                feature_class=feature_class,
                feature_class_label=str(source["feature_class_label"]),
                method_group=method_group,
                proposal_methods=str(source["proposal_methods"]),
                quality_passed=bool(source["quality_passed"]),
                flags=str(source["flags"]),
                temporal_detail_available=True,
                detail_level=temporal.detail_level,
                diagnostic_subgroup=temporal.diagnostic_subgroup,
                candidate_presence=temporal.candidate_presence,
                candidate_same_subgroup_presence=temporal.candidate_same_subgroup_presence,
                quality_presence=temporal.quality_presence,
                quality_same_subgroup_presence=temporal.quality_same_subgroup_presence,
                frame_count=temporal.frame_count,
                required_presence=temporal.required_presence,
                association_radius_px=temporal.association_radius_px,
                method_group_count=by_group_count[(feature_class, method_group)],
                method_group_share_in_class=(
                    by_group_count[(feature_class, method_group)] / by_class_count[feature_class]
                ),
                note="非紧凑候选有逐源诊断时序",
            )
        )
    return tuple(output)


def run_source_proposal_temporal_audit(
    catalog_path: str | Path,
    diagnostic_sources_path: str | Path,
    subgroup_persistence_path: str | Path,
    *,
    target_ids: Sequence[int] = (),
) -> SourceProposalTemporalAuditResult:
    """连接来源组合与 15 帧持久性，并对输入 ID/参数做严格校验。"""

    catalog = Path(catalog_path).resolve()
    diagnostic = Path(diagnostic_sources_path).resolve()
    subgroup = Path(subgroup_persistence_path).resolve()
    catalog_rows = _read_catalog(catalog)
    compact_rows, compact_frame_count, compact_required, compact_radius = _read_compact_subgroups(subgroup)
    diagnostic_rows = _read_diagnostic_sources(
        diagnostic,
        catalog_rows,
        frame_count=compact_frame_count,
        required_presence=compact_required,
        association_radius_px=compact_radius,
    )
    noncompact_ids = {
        detection_id
        for detection_id, source in catalog_rows.items()
        if source["feature_class"] != COMPACT_FEATURE_CLASS
    }
    if set(diagnostic_rows) != noncompact_ids:
        only_catalog = sorted(noncompact_ids - set(diagnostic_rows))
        only_diagnostic = sorted(set(diagnostic_rows) - noncompact_ids)
        raise ValueError(
            "diagnostic source IDs must equal all non-compact catalog IDs: "
            f"catalog_only={only_catalog[:5]!r}, diagnostic_only={only_diagnostic[:5]!r}"
        )
    by_group, _by_class = _group_catalog(catalog_rows)
    compact_expected = {
        method_group: len(members)
        for (feature_class, method_group), members in by_group.items()
        if feature_class == COMPACT_FEATURE_CLASS
    }
    compact_actual: dict[str, int] = defaultdict(int)
    for row in compact_rows:
        compact_actual[str(row["method_group"])] += int(row["anchor_count"])
    if compact_actual != compact_expected:
        raise ValueError(
            "compact subgroup anchor counts do not cover source catalog groups: "
            f"expected={dict(sorted(compact_expected.items()))!r}, "
            f"actual={dict(sorted(compact_actual.items()))!r}"
        )
    summaries = _summaries(
        catalog_rows,
        diagnostic_rows,
        compact_rows,
        frame_count=compact_frame_count,
        required_presence=compact_required,
        association_radius_px=compact_radius,
    )
    targets = _targets(catalog_rows, diagnostic_rows, summaries, target_ids)
    target_text = "；".join(
        f"ID {row.detection_id}={row.method_group}, candidate={row.candidate_presence}, quality={row.quality_presence}"
        for row in targets
    )
    conclusion = (
        "来源—时序交叉审计表明，来源交集、候选位置持久和质量响应持久是三条不同证据轴；"
        "非紧凑候选可逐源核对，compact_quality 只能按来源子组汇总，不能把缺失的逐源时序写成 0。"
        + (f"目标结果：{target_text}。" if target_text else "")
    )
    return SourceProposalTemporalAuditResult(
        catalog_path=str(catalog),
        diagnostic_sources_path=str(diagnostic),
        subgroup_persistence_path=str(subgroup),
        frame_count=compact_frame_count,
        required_presence=compact_required,
        association_radius_px=compact_radius,
        source_rows=tuple(sorted(diagnostic_rows.values(), key=lambda row: row.detection_id)),
        summaries=summaries,
        targets=targets,
        conclusion=conclusion,
    )


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, object]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_source_proposal_temporal_artifacts(
    result: SourceProposalTemporalAuditResult,
    output_dir: str | Path,
) -> Path:
    """写出逐源、来源汇总、目标和 JSON 结果。"""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(
        output / "source_proposal_temporal_rows.csv",
        SOURCE_PROPOSAL_TEMPORAL_FIELDS,
        (row.as_dict() for row in result.source_rows),
    )
    _write_csv(
        output / "source_proposal_temporal_summary.csv",
        SOURCE_PROPOSAL_TEMPORAL_SUMMARY_FIELDS,
        (row.as_dict() for row in result.summaries),
    )
    _write_csv(
        output / "source_proposal_temporal_targets.csv",
        SOURCE_PROPOSAL_TEMPORAL_TARGET_FIELDS,
        (row.as_dict() for row in result.targets),
    )
    (output / "source_proposal_temporal_audit.json").write_text(
        json.dumps(result.as_dict(), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return output


__all__ = [
    "COMPACT_FEATURE_CLASS",
    "COMPACT_SUBGROUP_TO_METHOD_GROUP",
    "METHOD_GROUP_NAMES",
    "SOURCE_PROPOSAL_TEMPORAL_FIELDS",
    "SOURCE_PROPOSAL_TEMPORAL_SUMMARY_FIELDS",
    "SOURCE_PROPOSAL_TEMPORAL_TARGET_FIELDS",
    "SourceProposalTemporalAuditResult",
    "SourceProposalTemporalRow",
    "SourceProposalTemporalSummary",
    "SourceProposalTemporalTarget",
    "run_source_proposal_temporal_audit",
    "write_source_proposal_temporal_artifacts",
]
