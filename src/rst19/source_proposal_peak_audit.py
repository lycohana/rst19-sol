"""交叉审计提议器来源与原始像素峰的一致性。

``source_catalog.csv`` 中的 ``proposal_methods`` 记录同一个候选被哪些宽筛
提议器提出；它不是独立投票，也不是恒星概率。本模块把该来源组合与
``source_peak_consistency.csv`` 的 raw 局部峰结果按 ``detection_id`` 严格
连接，按 ``feature_class × method_group`` 汇总质量通过、raw 峰一致性和
非局部峰差值，用来识别“某类总体如此”与“某个提议器子组如此”的差别。

模块只读取已有 CSV，不重跑 FITS、detector、质量层或缓存。所有字段仍是
detector-level 诊断，不能替代星表、WCS、PSF、注入真值或人工复核。
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np


METHOD_GROUP_NAMES = ("all_three", "gaussian_only", "dog_only", "partial", "no_method")


@dataclass(frozen=True, slots=True)
class SourceProposalPeakRow:
    """一个候选的提议器来源与 raw 峰交叉记录。"""

    detection_id: int
    feature_class: str
    feature_class_label: str
    method_group: str
    method_count: int
    proposal_methods: str
    quality_passed: bool
    flags: str
    flux_snr: float | None
    filter_snr: float | None
    raw_valid: bool
    raw_local_maximum_r1: bool
    raw_unique_local_maximum_r1: bool
    raw_local_max_gap_adu_r1: float | None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


SOURCE_PROPOSAL_PEAK_FIELDS = tuple(field.name for field in fields(SourceProposalPeakRow))


@dataclass(frozen=True, slots=True)
class SourceProposalPeakSummary:
    """一个特征类别与提议器来源子组的汇总。"""

    feature_class: str
    feature_class_label: str
    method_group: str
    candidate_count: int
    class_count: int
    class_fraction: float
    quality_count: int
    quality_fraction: float
    raw_valid_count: int
    raw_local_maximum_count_r1: int
    raw_local_maximum_fraction_r1: float
    raw_unique_local_maximum_count_r1: int
    nonlocal_peak_count_r1: int
    nonlocal_peak_fraction_r1: float
    nonlocal_peak_gap_median_adu_r1: float | None
    nonlocal_peak_gap_p90_adu_r1: float | None
    median_flux_snr: float | None
    median_filter_snr: float | None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


SOURCE_PROPOSAL_PEAK_SUMMARY_FIELDS = tuple(field.name for field in fields(SourceProposalPeakSummary))


@dataclass(frozen=True, slots=True)
class SourceProposalPeakTarget:
    """指定候选在类别—来源子组中的相对位置。"""

    detection_id: int
    feature_class: str
    feature_class_label: str
    method_group: str
    proposal_methods: str
    quality_passed: bool
    flags: str
    flux_snr: float | None
    filter_snr: float | None
    raw_local_maximum_r1: bool
    raw_unique_local_maximum_r1: bool
    raw_local_max_gap_adu_r1: float | None
    method_group_count: int
    method_group_share_in_class: float

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


SOURCE_PROPOSAL_PEAK_TARGET_FIELDS = tuple(field.name for field in fields(SourceProposalPeakTarget))


@dataclass(frozen=True, slots=True)
class SourceProposalPeakAuditResult:
    """提议器来源与 raw 峰交叉审计结果。"""

    catalog_path: str
    peak_consistency_path: str
    rows: tuple[SourceProposalPeakRow, ...]
    summaries: tuple[SourceProposalPeakSummary, ...]
    targets: tuple[SourceProposalPeakTarget, ...]
    conclusion: str

    def as_dict(self) -> dict[str, object]:
        return {
            "catalog_path": self.catalog_path,
            "peak_consistency_path": self.peak_consistency_path,
            "rows": [row.as_dict() for row in self.rows],
            "summaries": [row.as_dict() for row in self.summaries],
            "targets": [row.as_dict() for row in self.targets],
            "conclusion": self.conclusion,
            "interpretation_boundary": (
                "proposal_methods 是相关提议器来源记录，不是独立投票；"
                "raw 局部峰是像素拓扑诊断，不是恒星充分条件；"
                "本审计不推导 precision、FDR、恒星概率、完备率或物理源数"
            ),
        }


def proposal_method_group(raw_methods: str) -> str:
    """按当前三类宽筛提议器归一化来源组合。"""

    methods = frozenset(value.strip() for value in str(raw_methods).split("|") if value.strip())
    if {"gaussian", "dog_narrow", "dog_broad"}.issubset(methods):
        return "all_three"
    if methods == {"gaussian"}:
        return "gaussian_only"
    if methods and "gaussian" not in methods:
        return "dog_only"
    if methods:
        return "partial"
    return "no_method"


def _finite_float(row: Mapping[str, str], key: str) -> float | None:
    raw = str(row.get(key, "")).strip()
    if not raw or raw.lower() in {"nan", "none", "null"}:
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def _required_bool(row: Mapping[str, str], key: str, row_number: int) -> bool:
    value = str(row.get(key, "")).strip().lower()
    if value in {"true", "1", "yes"}:
        return True
    if value in {"false", "0", "no"}:
        return False
    raise ValueError(f"CSV row {row_number}: invalid {key}={value!r}")


def _read_catalog(path: Path) -> dict[int, dict[str, object]]:
    required = {
        "detection_id",
        "feature_class",
        "feature_class_label",
        "quality_passed",
        "flags",
        "proposal_methods",
        "flux_snr",
        "filter_snr",
    }
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = sorted(required - set(reader.fieldnames or ()))
        if missing:
            raise ValueError(f"source catalog is missing required columns: {', '.join(missing)}")
        records: dict[int, dict[str, object]] = {}
        for row_number, raw in enumerate(reader, start=2):
            try:
                detection_id = int(str(raw.get("detection_id", "")).strip())
            except ValueError as exc:
                raise ValueError(f"catalog row {row_number}: invalid detection_id") from exc
            if detection_id in records:
                raise ValueError(f"source catalog contains duplicate detection_id={detection_id}")
            methods = str(raw.get("proposal_methods", "")).strip()
            feature_class = str(raw.get("feature_class", "")).strip()
            if not feature_class:
                raise ValueError(f"catalog row {row_number}: empty feature_class")
            records[detection_id] = {
                "detection_id": detection_id,
                "feature_class": feature_class,
                "feature_class_label": str(raw.get("feature_class_label", "")).strip(),
                "method_group": proposal_method_group(methods),
                "method_count": len({item for item in methods.split("|") if item}),
                "proposal_methods": methods,
                "quality_passed": _required_bool(raw, "quality_passed", row_number),
                "flags": str(raw.get("flags", "")).strip(),
                "flux_snr": _finite_float(raw, "flux_snr"),
                "filter_snr": _finite_float(raw, "filter_snr"),
            }
    if not records:
        raise ValueError(f"source catalog is empty: {path}")
    return records


def _read_peak_consistency(path: Path) -> dict[int, dict[str, object]]:
    required = {
        "detection_id",
        "raw_valid",
        "raw_local_maximum_r1",
        "raw_unique_local_maximum_r1",
        "raw_local_max_gap_adu_r1",
    }
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = sorted(required - set(reader.fieldnames or ()))
        if missing:
            raise ValueError(f"peak consistency CSV is missing required columns: {', '.join(missing)}")
        records: dict[int, dict[str, object]] = {}
        for row_number, raw in enumerate(reader, start=2):
            try:
                detection_id = int(str(raw.get("detection_id", "")).strip())
            except ValueError as exc:
                raise ValueError(f"peak row {row_number}: invalid detection_id") from exc
            if detection_id in records:
                raise ValueError(f"peak consistency CSV contains duplicate detection_id={detection_id}")
            records[detection_id] = {
                "detection_id": detection_id,
                "raw_valid": _required_bool(raw, "raw_valid", row_number),
                "raw_local_maximum_r1": _required_bool(raw, "raw_local_maximum_r1", row_number),
                "raw_unique_local_maximum_r1": _required_bool(raw, "raw_unique_local_maximum_r1", row_number),
                "raw_local_max_gap_adu_r1": _finite_float(raw, "raw_local_max_gap_adu_r1"),
            }
    if not records:
        raise ValueError(f"peak consistency CSV is empty: {path}")
    return records


def _median(values: Iterable[float]) -> float | None:
    finite = [value for value in values if math.isfinite(value)]
    return float(np.median(np.asarray(finite, dtype=np.float64))) if finite else None


def _p90(values: Iterable[float]) -> float | None:
    finite = [value for value in values if math.isfinite(value)]
    return float(np.quantile(np.asarray(finite, dtype=np.float64), 0.90)) if finite else None


def _joined_rows(
    catalog: Mapping[int, Mapping[str, object]],
    peak: Mapping[int, Mapping[str, object]],
) -> tuple[SourceProposalPeakRow, ...]:
    catalog_ids = set(catalog)
    peak_ids = set(peak)
    if catalog_ids != peak_ids:
        only_catalog = sorted(catalog_ids - peak_ids)
        only_peak = sorted(peak_ids - catalog_ids)
        raise ValueError(
            "catalog and peak consistency IDs do not match: "
            f"catalog_only={only_catalog[:5]!r}, peak_only={only_peak[:5]!r}"
        )
    rows: list[SourceProposalPeakRow] = []
    for detection_id in sorted(catalog_ids):
        source = catalog[detection_id]
        raw = peak[detection_id]
        rows.append(
            SourceProposalPeakRow(
                detection_id=detection_id,
                feature_class=str(source["feature_class"]),
                feature_class_label=str(source["feature_class_label"]),
                method_group=str(source["method_group"]),
                method_count=int(source["method_count"]),
                proposal_methods=str(source["proposal_methods"]),
                quality_passed=bool(source["quality_passed"]),
                flags=str(source["flags"]),
                flux_snr=source["flux_snr"] if isinstance(source["flux_snr"], float) else None,
                filter_snr=source["filter_snr"] if isinstance(source["filter_snr"], float) else None,
                raw_valid=bool(raw["raw_valid"]),
                raw_local_maximum_r1=bool(raw["raw_local_maximum_r1"]),
                raw_unique_local_maximum_r1=bool(raw["raw_unique_local_maximum_r1"]),
                raw_local_max_gap_adu_r1=(
                    raw["raw_local_max_gap_adu_r1"]
                    if isinstance(raw["raw_local_max_gap_adu_r1"], float)
                    else None
                ),
            )
        )
    return tuple(rows)


def _summarize(rows: Sequence[SourceProposalPeakRow]) -> tuple[SourceProposalPeakSummary, ...]:
    by_class: dict[str, list[SourceProposalPeakRow]] = defaultdict(list)
    by_group: dict[tuple[str, str], list[SourceProposalPeakRow]] = defaultdict(list)
    labels: dict[str, str] = {}
    for row in rows:
        by_class[row.feature_class].append(row)
        by_group[(row.feature_class, row.method_group)].append(row)
        labels[row.feature_class] = row.feature_class_label

    summaries: list[SourceProposalPeakSummary] = []
    for (feature_class, method_group), members in sorted(by_group.items()):
        class_count = len(by_class[feature_class])
        quality_count = sum(row.quality_passed for row in members)
        raw_valid_count = sum(row.raw_valid for row in members)
        local_count = sum(row.raw_local_maximum_r1 for row in members)
        unique_count = sum(row.raw_unique_local_maximum_r1 for row in members)
        gaps = [
            row.raw_local_max_gap_adu_r1
            for row in members
            if row.raw_valid and not row.raw_local_maximum_r1 and row.raw_local_max_gap_adu_r1 is not None
        ]
        summaries.append(
            SourceProposalPeakSummary(
                feature_class=feature_class,
                feature_class_label=labels[feature_class],
                method_group=method_group,
                candidate_count=len(members),
                class_count=class_count,
                class_fraction=len(members) / class_count,
                quality_count=quality_count,
                quality_fraction=quality_count / len(members),
                raw_valid_count=raw_valid_count,
                raw_local_maximum_count_r1=local_count,
                raw_local_maximum_fraction_r1=local_count / len(members),
                raw_unique_local_maximum_count_r1=unique_count,
                nonlocal_peak_count_r1=len(gaps),
                nonlocal_peak_fraction_r1=len(gaps) / len(members),
                nonlocal_peak_gap_median_adu_r1=_median(gaps),
                nonlocal_peak_gap_p90_adu_r1=_p90(gaps),
                median_flux_snr=_median(
                    row.flux_snr for row in members if row.flux_snr is not None
                ),
                median_filter_snr=_median(
                    row.filter_snr for row in members if row.filter_snr is not None
                ),
            )
        )
    return tuple(summaries)


def _targets(
    rows: Sequence[SourceProposalPeakRow],
    target_ids: Sequence[int],
) -> tuple[SourceProposalPeakTarget, ...]:
    requested = tuple(dict.fromkeys(int(value) for value in target_ids))
    if not requested:
        return ()
    by_id = {row.detection_id: row for row in rows}
    missing = sorted(set(requested) - set(by_id))
    if missing:
        raise ValueError(f"target detection_id not found in joined rows: {', '.join(map(str, missing))}")
    class_group_counts: dict[tuple[str, str], int] = defaultdict(int)
    class_counts: dict[str, int] = defaultdict(int)
    for row in rows:
        class_group_counts[(row.feature_class, row.method_group)] += 1
        class_counts[row.feature_class] += 1
    output: list[SourceProposalPeakTarget] = []
    for detection_id in requested:
        row = by_id[detection_id]
        group_count = class_group_counts[(row.feature_class, row.method_group)]
        output.append(
            SourceProposalPeakTarget(
                detection_id=row.detection_id,
                feature_class=row.feature_class,
                feature_class_label=row.feature_class_label,
                method_group=row.method_group,
                proposal_methods=row.proposal_methods,
                quality_passed=row.quality_passed,
                flags=row.flags,
                flux_snr=row.flux_snr,
                filter_snr=row.filter_snr,
                raw_local_maximum_r1=row.raw_local_maximum_r1,
                raw_unique_local_maximum_r1=row.raw_unique_local_maximum_r1,
                raw_local_max_gap_adu_r1=row.raw_local_max_gap_adu_r1,
                method_group_count=group_count,
                method_group_share_in_class=group_count / class_counts[row.feature_class],
            )
        )
    return tuple(output)


def run_source_proposal_peak_audit(
    catalog_path: str | Path,
    peak_consistency_path: str | Path,
    *,
    target_ids: Sequence[int] = (),
) -> SourceProposalPeakAuditResult:
    """严格连接 source catalog 与 raw 峰一致性表并按来源子组汇总。"""

    catalog = Path(catalog_path).resolve()
    peak = Path(peak_consistency_path).resolve()
    catalog_rows = _read_catalog(catalog)
    peak_rows = _read_peak_consistency(peak)
    rows = _joined_rows(catalog_rows, peak_rows)
    summaries = _summarize(rows)
    targets = _targets(rows, target_ids)
    target_text = "；".join(
        f"ID {row.detection_id}={row.method_group}, raw r1局部峰={row.raw_local_maximum_r1}"
        for row in targets
    )
    conclusion = (
        "提议器来源与 raw 峰交叉审计显示，同一 feature_class 内不同来源子组的"
        "质量和像素拓扑不能直接合并解释；来源组合只用于复核排序，不是独立投票。"
        + (f"目标结果：{target_text}。" if target_text else "")
    )
    return SourceProposalPeakAuditResult(
        catalog_path=str(catalog),
        peak_consistency_path=str(peak),
        rows=rows,
        summaries=summaries,
        targets=targets,
        conclusion=conclusion,
    )


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, object]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_source_proposal_peak_artifacts(
    result: SourceProposalPeakAuditResult,
    output_dir: str | Path,
) -> Path:
    """写出逐源、类别—来源子组、目标和 JSON 结果。"""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(
        output / "source_proposal_peak_rows.csv",
        SOURCE_PROPOSAL_PEAK_FIELDS,
        (row.as_dict() for row in result.rows),
    )
    _write_csv(
        output / "source_proposal_peak_summary.csv",
        SOURCE_PROPOSAL_PEAK_SUMMARY_FIELDS,
        (row.as_dict() for row in result.summaries),
    )
    _write_csv(
        output / "source_proposal_peak_targets.csv",
        SOURCE_PROPOSAL_PEAK_TARGET_FIELDS,
        (row.as_dict() for row in result.targets),
    )
    (output / "source_proposal_peak_audit.json").write_text(
        json.dumps(result.as_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output


__all__ = [
    "METHOD_GROUP_NAMES",
    "SOURCE_PROPOSAL_PEAK_FIELDS",
    "SOURCE_PROPOSAL_PEAK_SUMMARY_FIELDS",
    "SOURCE_PROPOSAL_PEAK_TARGET_FIELDS",
    "SourceProposalPeakAuditResult",
    "SourceProposalPeakRow",
    "SourceProposalPeakSummary",
    "SourceProposalPeakTarget",
    "proposal_method_group",
    "run_source_proposal_peak_audit",
    "write_source_proposal_peak_artifacts",
]
