"""审计各特征类别中显著性最高、但仍未通过质量层的候选。

该模块只消费 ``rst19-sources`` 已经生成的源级 CSV，不重新检测 FITS。
它回答的是一个容易被单一 SNR 混淆的问题：在每个算法特征类别里，是否
存在 ``quality_passed=False`` 但通量 SNR 很高的候选？这类源是 hard
negative 候选，说明显著性、局部集中性和独立物理身份不是同一个事件。

输出故意保留原始 flags 与形态字段。排名不是恒星概率，也不是真值标注；
它只为后续值域、去混叠、PSF 留出、星表/WCS 和跨帧验证提供可复核的抽样
入口。
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


HARD_NEGATIVE_METRIC_FIELDS: tuple[str, ...] = ("flux_snr", "filter_snr", "peak")
HARD_NEGATIVE_DETAIL_FIELDS: tuple[str, ...] = (
    "flux_snr",
    "filter_snr",
    "filter_flux_snr_ratio",
    "peak",
    "fwhm",
    "ellipticity",
    "sharpness",
    "psf_support_pixels",
    "footprint_pixels",
)


@dataclass(frozen=True, slots=True)
class FeatureHardNegativeSummary:
    """一个特征类别的质量层计数和 hard-negative 排名摘要。"""

    feature_class: str
    feature_class_label: str
    class_count: int
    class_quality_count: int
    class_rejected_count: int
    class_rejected_fraction: float
    metric: str
    top_metric_value: float | None
    top_detection_id: int | None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class FeatureHardNegativeRow:
    """类别内一个高显著性落选候选的原始审计字段。"""

    detection_id: int
    rank_within_class: int
    feature_class: str
    feature_class_label: str
    flags: str
    metric: str
    metric_value: float
    flux_snr: float | None
    filter_snr: float | None
    filter_flux_snr_ratio: float | None
    peak: float | None
    fwhm: float | None
    ellipticity: float | None
    sharpness: float | None
    psf_support_pixels: float | None
    footprint_pixels: float | None
    class_count: int
    class_rejected_count: int
    class_quality_count: int

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class FeatureHardNegativeAuditResult:
    """类别内 hard-negative 审计的机器可读结果。"""

    catalog_path: str
    metric: str
    top_n: int
    excluded_classes: tuple[str, ...]
    summaries: tuple[FeatureHardNegativeSummary, ...]
    rows: tuple[FeatureHardNegativeRow, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "catalog_path": self.catalog_path,
            "metric": self.metric,
            "top_n": self.top_n,
            "excluded_classes": list(self.excluded_classes),
            "summaries": [summary.as_dict() for summary in self.summaries],
            "rows": [row.as_dict() for row in self.rows],
            "interpretation_guardrails": [
                "quality_passed=False 是当前算法质量层的结果，不是物理伪影真值。",
                "类别内排名只表示高显著性与拒绝规则的冲突，不能解释为恒星概率。",
                "高 SNR 候选仍需值域、独占像素、去混叠、PSF 留出和跨帧身份复核。",
                "feature_class 是按旗标和形态生成的算法审计标签，不是物理类别。",
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


def _read_rows(catalog_path: str | Path) -> tuple[Path, list[dict[str, str]]]:
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
        "feature_class",
        "feature_class_label",
        "quality_passed",
        "flags",
    }
    missing = sorted(required - set(rows[0]))
    if missing:
        raise ValueError(f"源级 CSV 缺少列：{', '.join(missing)}")
    return path, rows


def run_feature_hard_negative_audit(
    catalog_path: str | Path,
    *,
    top_n: int = 5,
    metric: str = "flux_snr",
    exclude_classes: Iterable[str] = ("other_rejected",),
) -> FeatureHardNegativeAuditResult:
    """按特征类别列出高显著性但未通过质量层的候选。

    ``metric`` 默认使用 ``flux_snr``，与当前质量层的暗星筛选口径一致；
    ``filter_snr`` 和 ``peak`` 仅用于敏感性对照，不会改变质量判定。
    """

    if top_n <= 0:
        raise ValueError("top_n 必须为正整数")
    if metric not in HARD_NEGATIVE_METRIC_FIELDS:
        raise ValueError(
            f"不支持的 hard-negative 排名指标：{metric!r}；"
            f"可选项为 {', '.join(HARD_NEGATIVE_METRIC_FIELDS)}"
        )
    excluded = tuple(dict.fromkeys(str(value) for value in exclude_classes))
    path, rows = _read_rows(catalog_path)

    by_class: dict[str, list[dict[str, str]]] = defaultdict(list)
    seen_ids: set[int] = set()
    for row in rows:
        try:
            detection_id = int(row["detection_id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"源级 CSV 存在非法 detection_id：{row.get('detection_id')!r}") from exc
        if detection_id in seen_ids:
            raise ValueError(f"源级 CSV 存在重复 detection_id：{detection_id}")
        seen_ids.add(detection_id)
        feature_class = row.get("feature_class", "")
        if feature_class not in excluded:
            by_class[feature_class].append(row)

    summaries: list[FeatureHardNegativeSummary] = []
    hard_negative_rows: list[FeatureHardNegativeRow] = []
    for feature_class in sorted(by_class):
        class_rows = by_class[feature_class]
        rejected = [row for row in class_rows if not _is_quality(row)]
        ranked = sorted(
            (
                (value, int(row["detection_id"]), row)
                for row in rejected
                if (value := _finite_float(row, metric)) is not None
            ),
            key=lambda item: (-item[0], item[1]),
        )
        quality_count = len(class_rows) - len(rejected)
        top_value = ranked[0][0] if ranked else None
        top_id = ranked[0][1] if ranked else None
        summaries.append(
            FeatureHardNegativeSummary(
                feature_class=feature_class,
                feature_class_label=class_rows[0].get("feature_class_label", feature_class),
                class_count=len(class_rows),
                class_quality_count=quality_count,
                class_rejected_count=len(rejected),
                class_rejected_fraction=len(rejected) / len(class_rows) if class_rows else 0.0,
                metric=metric,
                top_metric_value=top_value,
                top_detection_id=top_id,
            )
        )
        for rank, (metric_value, detection_id, row) in enumerate(ranked[:top_n], start=1):
            flux_snr = _finite_float(row, "flux_snr")
            filter_snr = _finite_float(row, "filter_snr")
            hard_negative_rows.append(
                FeatureHardNegativeRow(
                    detection_id=detection_id,
                    rank_within_class=rank,
                    feature_class=feature_class,
                    feature_class_label=row.get("feature_class_label", feature_class),
                    flags=row.get("flags", ""),
                    metric=metric,
                    metric_value=metric_value,
                    flux_snr=flux_snr,
                    filter_snr=filter_snr,
                    filter_flux_snr_ratio=(
                        filter_snr / flux_snr
                        if flux_snr is not None and filter_snr is not None and abs(flux_snr) > 1e-12
                        else None
                    ),
                    peak=_finite_float(row, "peak"),
                    fwhm=_finite_float(row, "fwhm"),
                    ellipticity=_finite_float(row, "ellipticity"),
                    sharpness=_finite_float(row, "sharpness"),
                    psf_support_pixels=_finite_float(row, "psf_support_pixels"),
                    footprint_pixels=_finite_float(row, "footprint_pixels"),
                    class_count=len(class_rows),
                    class_rejected_count=len(rejected),
                    class_quality_count=quality_count,
                )
            )
    return FeatureHardNegativeAuditResult(
        catalog_path=str(path),
        metric=metric,
        top_n=top_n,
        excluded_classes=excluded,
        summaries=tuple(summaries),
        rows=tuple(hard_negative_rows),
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


def write_feature_hard_negative_artifacts(
    result: FeatureHardNegativeAuditResult,
    out_dir: str | Path,
) -> Path:
    """写出类别摘要、hard-negative 明细和 JSON。"""

    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "feature_hard_negative.json").write_text(
        json.dumps(result.as_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_csv(output / "feature_hard_negative_summary.csv", result.summaries)
    _write_csv(output / "feature_hard_negative_rows.csv", result.rows)
    return output


__all__ = [
    "HARD_NEGATIVE_DETAIL_FIELDS",
    "HARD_NEGATIVE_METRIC_FIELDS",
    "FeatureHardNegativeAuditResult",
    "FeatureHardNegativeRow",
    "FeatureHardNegativeSummary",
    "run_feature_hard_negative_audit",
    "write_feature_hard_negative_artifacts",
]
