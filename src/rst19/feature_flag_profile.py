"""按首要特征类别拆解质量/审计旗标。

旗标可以重叠，因此这里不把各旗标百分比相加成一个拒绝概率；输出只用于
说明不同类别的主要审计机制和指定候选的旗标来源。
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True, slots=True)
class FeatureFlagProfileRow:
    """一个类别中一个旗标的出现次数。"""

    feature_class: str
    candidate_count: int
    quality_count: int
    flag: str
    flag_count: int
    flag_fraction: float

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class FeatureFlagProfileTarget:
    """指定候选的类别、质量状态和原始旗标。"""

    detection_id: str
    feature_class: str
    quality_passed: bool
    flags: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class FeatureFlagProfileResult:
    """旗标按类别汇总的完整结果。"""

    catalog_path: str
    targets: tuple[str, ...]
    rows: tuple[FeatureFlagProfileRow, ...]
    target_rows: tuple[FeatureFlagProfileTarget, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "catalog_path": self.catalog_path,
            "targets": list(self.targets),
            "rows": [row.as_dict() for row in self.rows],
            "target_rows": [row.as_dict() for row in self.target_rows],
            "interpretation_boundary": (
                "flags are overlapping detector-level audit labels; flag fractions are not "
                "mutually exclusive probabilities, precision, FDR, or star probabilities"
            ),
        }


def _parse_quality(value: str) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def _flags(value: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(flag.strip() for flag in str(value).split("|") if flag.strip()))


def run_feature_flag_profile(
    catalog_path: str | Path,
    *,
    target_ids: Sequence[str] = ("82931", "82934", "44132"),
) -> FeatureFlagProfileResult:
    """读取源级 CSV，汇总每个特征类别中的重叠质量旗标。"""

    path = Path(catalog_path)
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fieldnames = set(reader.fieldnames or ())
        required = {"detection_id", "feature_class", "quality_passed", "flags"}
        missing = sorted(required - fieldnames)
        if missing:
            raise ValueError(f"source catalog is missing required columns: {', '.join(missing)}")
        records = list(reader)
    if not records:
        raise ValueError("source catalog is empty")

    normalized_targets = tuple(dict.fromkeys(str(value).strip() for value in target_ids if str(value).strip()))
    class_records: dict[str, list[dict[str, str]]] = defaultdict(list)
    for record in records:
        feature_class = str(record.get("feature_class", "")).strip()
        if feature_class:
            class_records[feature_class].append(record)

    output_rows: list[FeatureFlagProfileRow] = []
    for feature_class in sorted(class_records):
        members = class_records[feature_class]
        count = len(members)
        quality_count = sum(_parse_quality(member.get("quality_passed", "")) for member in members)
        counts = Counter(flag for member in members for flag in _flags(member.get("flags", "")))
        if not counts:
            counts["<none>"] = count
        for flag, flag_count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
            output_rows.append(
                FeatureFlagProfileRow(
                    feature_class=feature_class,
                    candidate_count=count,
                    quality_count=quality_count,
                    flag=flag,
                    flag_count=flag_count,
                    flag_fraction=flag_count / count,
                )
            )

    wanted = set(normalized_targets)
    target_rows = tuple(
        FeatureFlagProfileTarget(
            detection_id=str(record.get("detection_id", "")).strip(),
            feature_class=str(record.get("feature_class", "")).strip(),
            quality_passed=_parse_quality(record.get("quality_passed", "")),
            flags="|".join(_flags(record.get("flags", ""))),
        )
        for record in records
        if str(record.get("detection_id", "")).strip() in wanted
    )
    return FeatureFlagProfileResult(
        catalog_path=str(path),
        targets=normalized_targets,
        rows=tuple(output_rows),
        target_rows=tuple(sorted(target_rows, key=lambda row: row.detection_id)),
    )


def write_feature_flag_profile_artifacts(
    result: FeatureFlagProfileResult,
    output_dir: str | Path,
) -> Path:
    """写出旗标统计 CSV、目标 CSV 和 JSON 边界说明。"""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows_path = output / "feature_flag_profile.csv"
    targets_path = output / "feature_flag_profile_targets.csv"
    json_path = output / "feature_flag_profile.json"
    with rows_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=[field.name for field in fields(FeatureFlagProfileRow)])
        writer.writeheader()
        writer.writerows(row.as_dict() for row in result.rows)
    with targets_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=[field.name for field in fields(FeatureFlagProfileTarget)])
        writer.writeheader()
        writer.writerows(row.as_dict() for row in result.target_rows)
    with json_path.open("w", encoding="utf-8") as stream:
        json.dump(result.as_dict(), stream, ensure_ascii=False, indent=2)
    return output
