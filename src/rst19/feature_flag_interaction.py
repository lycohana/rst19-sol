"""统计审计旗标的单项与两项交互。

这个模块只读取已经生成的 ``source_catalog.csv``，不重新检测 FITS，也不把
旗标组合解释成物理伪影真值。它用于回答一个更窄但重要的问题：某个旗标
单独出现时是否仍可能进入当前质量层，而它与另一个旗标叠加后是否会被拒绝。
``quality_fraction`` 只是当前 detector rule 的行为统计，不是 precision、FDR
或恒星概率。
"""

from __future__ import annotations

import csv
import itertools
import json
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True, slots=True)
class FeatureFlagInteractionRow:
    """一个旗标单项或两项组合的存在性与质量层统计。"""

    kind: str
    interaction: str
    flag_a: str
    flag_b: str
    candidate_count: int
    quality_count: int
    quality_fraction: float
    exact_set_count: int
    exact_set_quality_count: int
    high_snr_count: int
    high_snr_rejected_count: int

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class FeatureFlagInteractionResult:
    """旗标交互审计结果及其口径。"""

    catalog_path: str
    record_count: int
    high_snr_threshold: float
    rows: tuple[FeatureFlagInteractionRow, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "catalog_path": self.catalog_path,
            "record_count": self.record_count,
            "high_snr_threshold": self.high_snr_threshold,
            "rows": [row.as_dict() for row in self.rows],
            "interpretation_boundary": [
                "quality_fraction describes the current detector quality rule, not physical truth",
                "flags overlap; singleton and pair rows are not mutually exclusive probabilities",
                "exact_set_count means the complete row flag set equals this singleton or pair",
                "high SNR rejection is a hard-negative diagnostic, not a false-positive rate",
            ],
        }


def _parse_quality(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def _parse_flags(value: object) -> tuple[str, ...]:
    return tuple(sorted({part.strip() for part in str(value).split("|") if part.strip()}))


def _parse_finite_float(value: object) -> float | None:
    raw = str(value).strip()
    if not raw:
        return None
    try:
        parsed = float(raw)
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def _read_catalog(catalog_path: str | Path) -> tuple[Path, list[dict[str, str]]]:
    path = Path(catalog_path)
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"detection_id", "quality_passed", "flags", "flux_snr"}
        missing = sorted(required - set(reader.fieldnames or ()))
        if missing:
            raise ValueError(f"source catalog is missing required columns: {', '.join(missing)}")
        records = list(reader)
    if not records:
        raise ValueError("source catalog is empty")
    return path, records


def run_feature_flag_interaction(
    catalog_path: str | Path,
    *,
    high_snr_threshold: float = 10.0,
) -> FeatureFlagInteractionResult:
    """汇总单旗标和两旗标组合的候选、质量及 hard-negative 统计。

    对含三个或更多旗标的行，仍会计入其中的 pair presence；``exact_set_count``
    只在完整旗标集合恰好等于该 pair 时增加。
    """

    if not math.isfinite(high_snr_threshold):
        raise ValueError("high_snr_threshold must be finite")
    path, records = _read_catalog(catalog_path)
    stats: dict[tuple[str, ...], dict[str, int]] = {}
    seen_ids: set[str] = set()

    for record in records:
        detection_id = str(record.get("detection_id", "")).strip()
        if not detection_id:
            raise ValueError("source catalog contains an empty detection_id")
        if detection_id in seen_ids:
            raise ValueError(f"source catalog contains duplicate detection_id: {detection_id}")
        seen_ids.add(detection_id)

        flags = _parse_flags(record.get("flags", ""))
        if not flags:
            continue
        quality = _parse_quality(record.get("quality_passed", ""))
        flux_snr = _parse_finite_float(record.get("flux_snr", ""))
        high_snr = flux_snr is not None and flux_snr >= high_snr_threshold
        keys: Iterable[tuple[str, ...]] = itertools.chain(
            ((flag,) for flag in flags),
            itertools.combinations(flags, 2),
        )
        for key in keys:
            row_stats = stats.setdefault(
                key,
                {
                    "candidate_count": 0,
                    "quality_count": 0,
                    "exact_set_count": 0,
                    "exact_set_quality_count": 0,
                    "high_snr_count": 0,
                    "high_snr_rejected_count": 0,
                },
            )
            row_stats["candidate_count"] += 1
            row_stats["quality_count"] += int(quality)
            row_stats["exact_set_count"] += int(flags == key)
            row_stats["exact_set_quality_count"] += int(flags == key and quality)
            row_stats["high_snr_count"] += int(high_snr)
            row_stats["high_snr_rejected_count"] += int(high_snr and not quality)

    rows: list[FeatureFlagInteractionRow] = []
    for key, values in stats.items():
        flag_a = key[0]
        flag_b = key[1] if len(key) == 2 else ""
        rows.append(
            FeatureFlagInteractionRow(
                kind="single" if len(key) == 1 else "pair",
                interaction="+".join(key),
                flag_a=flag_a,
                flag_b=flag_b,
                candidate_count=values["candidate_count"],
                quality_count=values["quality_count"],
                quality_fraction=values["quality_count"] / values["candidate_count"],
                exact_set_count=values["exact_set_count"],
                exact_set_quality_count=values["exact_set_quality_count"],
                high_snr_count=values["high_snr_count"],
                high_snr_rejected_count=values["high_snr_rejected_count"],
            )
        )
    rows.sort(key=lambda row: (0 if row.kind == "single" else 1, -row.candidate_count, row.interaction))
    return FeatureFlagInteractionResult(
        catalog_path=str(path),
        record_count=len(records),
        high_snr_threshold=high_snr_threshold,
        rows=tuple(rows),
    )


def write_feature_flag_interaction_artifacts(
    result: FeatureFlagInteractionResult,
    output_dir: str | Path,
) -> Path:
    """写出旗标单项/组合 CSV 与 JSON。"""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows_path = output / "feature_flag_interactions.csv"
    json_path = output / "feature_flag_interactions.json"
    with rows_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=[field.name for field in fields(FeatureFlagInteractionRow)])
        writer.writeheader()
        writer.writerows(row.as_dict() for row in result.rows)
    with json_path.open("w", encoding="utf-8") as stream:
        json.dump(result.as_dict(), stream, ensure_ascii=False, indent=2)
    return output


__all__ = [
    "FeatureFlagInteractionResult",
    "FeatureFlagInteractionRow",
    "run_feature_flag_interaction",
    "write_feature_flag_interaction_artifacts",
]
