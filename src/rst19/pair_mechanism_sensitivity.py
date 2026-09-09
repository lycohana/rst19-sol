"""几何条件化近邻机制审计的窗口敏感性控制。

这个模块重复调用只读的 :mod:`pair_mechanism_context`，把同一目标 pair
放入几组预先声明的质心/整数峰距离窗口中。它的目的不是寻找一组“最有利”
的数字，而是把窗口改变后比较总体如何变化保存下来，防止把局部统计外推
成全图噪点率、双星概率或物理真值。
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Iterable

from .pair_mechanism_context import run_pair_mechanism_context


PairMechanismSensitivityConfiguration = tuple[str, float, float, float]

DEFAULT_PAIR_MECHANISM_SENSITIVITY_CONFIGS: tuple[PairMechanismSensitivityConfiguration, ...] = (
    ("centroid_3.0", 3.0, 3.5, 4.8),
    ("centroid_3.5", 3.5, 3.5, 4.8),
    ("centroid_4.0", 4.0, 3.5, 4.8),
    ("peak_3.0_5.0", 3.5, 3.0, 5.0),
    ("peak_3.5_5.5", 3.5, 3.5, 5.5),
)


@dataclass(frozen=True, slots=True)
class PairMechanismSensitivityRow:
    """一组几何窗口下的候选 pair 背景汇总。"""

    configuration: str
    max_centroid_distance_px: float
    peak_distance_min_px: float
    peak_distance_max_px: float
    pair_count: int
    same_feature_class_count: int
    cross_feature_class_count: int
    both_quality_count: int
    both_quality_fraction: float
    target_in_geometry_context: bool
    target_centroid_distance_px: float
    target_peak_distance_px: float
    target_feature_class_a: str
    target_feature_class_b: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PairMechanismSensitivityResult:
    """多组几何窗口的机制背景敏感性结果。"""

    catalog_path: str
    target_ids: tuple[int, int]
    rows: tuple[PairMechanismSensitivityRow, ...]
    parameters: dict[str, object]
    conclusion: str

    def as_dict(self) -> dict[str, object]:
        return {
            "catalog_path": self.catalog_path,
            "target_ids": list(self.target_ids),
            "rows": [row.as_dict() for row in self.rows],
            "parameters": dict(self.parameters),
            "conclusion": self.conclusion,
            "interpretation_guardrails": [
                "窗口敏感性只描述候选目录中的比较总体，不是物理双星真值。",
                "放宽窗口会改变比较对象，不能把不同配置的 pair 数直接比较成概率。",
                "both_quality_count 是当前质量规则的通过数，不是可靠星数或噪点数。",
                "目标 pair 的 baseline/injected/new 反事实仍需单独审计。",
            ],
        }


def _normalise_configurations(
    configurations: Iterable[PairMechanismSensitivityConfiguration],
) -> tuple[PairMechanismSensitivityConfiguration, ...]:
    normalised: list[PairMechanismSensitivityConfiguration] = []
    seen: set[str] = set()
    for raw in configurations:
        if len(raw) != 4:
            raise ValueError("each configuration must contain name and three distances")
        name = str(raw[0]).strip()
        if not name:
            raise ValueError("configuration name must not be empty")
        if name in seen:
            raise ValueError(f"duplicate configuration name: {name}")
        seen.add(name)
        try:
            centroid_max = float(raw[1])
            peak_min = float(raw[2])
            peak_max = float(raw[3])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"configuration {name!r} contains a non-numeric distance") from exc
        distances = (centroid_max, peak_min, peak_max)
        if any(not math.isfinite(value) or value <= 0 for value in distances):
            raise ValueError(f"configuration {name!r} distances must be finite and positive")
        if peak_min > peak_max:
            raise ValueError(f"configuration {name!r} peak minimum exceeds maximum")
        normalised.append((name, centroid_max, peak_min, peak_max))
    if not normalised:
        raise ValueError("configurations must contain at least one window")
    return tuple(normalised)


def run_pair_mechanism_sensitivity(
    catalog_path: str | Path,
    target_ids: Iterable[int],
    *,
    configurations: Iterable[PairMechanismSensitivityConfiguration] = DEFAULT_PAIR_MECHANISM_SENSITIVITY_CONFIGS,
) -> PairMechanismSensitivityResult:
    """在多个预先声明的几何窗口中重复执行近邻机制背景审计。"""

    resolved_ids = tuple(sorted({int(value) for value in target_ids}))
    if len(resolved_ids) != 2:
        raise ValueError("target_ids must contain exactly two distinct detection IDs")
    configuration_rows = _normalise_configurations(configurations)
    rows: list[PairMechanismSensitivityRow] = []
    for name, centroid_max, peak_min, peak_max in configuration_rows:
        context = run_pair_mechanism_context(
            catalog_path,
            resolved_ids,
            max_centroid_distance_px=centroid_max,
            peak_distance_min_px=peak_min,
            peak_distance_max_px=peak_max,
        )
        pair_count = len(context.pairs)
        same_count = sum(row.same_feature_class for row in context.pairs)
        both_quality_count = sum(row.both_quality_passed for row in context.pairs)
        rows.append(
            PairMechanismSensitivityRow(
                configuration=name,
                max_centroid_distance_px=centroid_max,
                peak_distance_min_px=peak_min,
                peak_distance_max_px=peak_max,
                pair_count=pair_count,
                same_feature_class_count=same_count,
                cross_feature_class_count=pair_count - same_count,
                both_quality_count=both_quality_count,
                both_quality_fraction=float(both_quality_count / pair_count) if pair_count else 0.0,
                target_in_geometry_context=context.target.in_geometry_context,
                target_centroid_distance_px=context.target.centroid_distance_px,
                target_peak_distance_px=context.target.peak_distance_px,
                target_feature_class_a=context.target.feature_class_a,
                target_feature_class_b=context.target.feature_class_b,
            )
        )
    result_rows = tuple(rows)
    conclusion = (
        f"完成 {len(result_rows)} 组几何窗口敏感性审计；结果只用于检查比较总体随窗口变化的方向，"
        "不能转换为噪点率、双星概率或物理恒星数。"
    )
    return PairMechanismSensitivityResult(
        catalog_path=str(catalog_path),
        target_ids=(resolved_ids[0], resolved_ids[1]),
        rows=result_rows,
        parameters={
            "configuration_count": len(configuration_rows),
            "configurations": [
                {
                    "name": name,
                    "max_centroid_distance_px": centroid_max,
                    "peak_distance_min_px": peak_min,
                    "peak_distance_max_px": peak_max,
                }
                for name, centroid_max, peak_min, peak_max in configuration_rows
            ],
            "interpretation_boundary": (
                "descriptive candidate-catalog sensitivity only; no physical star probability, precision, FDR, or truth"
            ),
        },
        conclusion=conclusion,
    )


def write_pair_mechanism_sensitivity_artifacts(
    result: PairMechanismSensitivityResult,
    out_dir: str | Path,
) -> Path:
    """写出窗口敏感性 CSV/JSON，不写检测缓存。"""

    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "pair_mechanism_sensitivity.csv"
    field_names = [field.name for field in fields(PairMechanismSensitivityRow)]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=field_names)
        writer.writeheader()
        for row in result.rows:
            writer.writerow(row.as_dict())
    (output / "pair_mechanism_sensitivity.json").write_text(
        json.dumps(result.as_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output


__all__ = [
    "DEFAULT_PAIR_MECHANISM_SENSITIVITY_CONFIGS",
    "PairMechanismSensitivityConfiguration",
    "PairMechanismSensitivityResult",
    "PairMechanismSensitivityRow",
    "run_pair_mechanism_sensitivity",
    "write_pair_mechanism_sensitivity_artifacts",
]
