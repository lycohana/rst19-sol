"""汇总类别证据轴之间的同步程度。

本模块只读取 ``feature_evidence_matrix.csv``，不重读 FITS、不重跑检测，
也不把多个指标加权成恒星概率。它把两个已有的 ``12/15`` 计数放在同一
个分母下：候选位置持久数，以及注册坐标邻域的质量响应持久数。派生量
``quality_response_given_candidate_persistent_fraction`` 只表示“位置已经
反复出现的候选中，有多少同时伴随质量邻域响应”，且后者可能来自其它
特征类别；因此它是 detector-level 证据轴关系，不是逐源 precision、FDR
或物理恒星身份。
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Mapping


@dataclass(frozen=True, slots=True)
class FeatureEvidenceAxesRow:
    """一个首要特征类别在候选、质量邻域和 PSF 轴上的摘要。"""

    feature_class: str
    feature_class_label: str
    candidate_count: int
    candidate_persistent_count: int
    quality_response_count: int
    candidate_persistence_fraction: float
    quality_response_fraction: float
    quality_response_given_candidate_persistent_fraction: float | None
    spatial_local_correlation_median: float | None
    spatial_local_residual_median: float | None
    evidence_axis_pattern: str
    evidence_axis_pattern_label: str
    interpretation: str
    caution: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


# Keep the CSV schema explicit and stable.  Defining it after the dataclass
# avoids relying on dict insertion order in callers and tests.
AXIS_SUMMARY_FIELDS = tuple(field.name for field in fields(FeatureEvidenceAxesRow))


@dataclass(frozen=True, slots=True)
class FeatureEvidenceAxesResult:
    """证据轴关系的完整结果。"""

    matrix_path: str
    thresholds: dict[str, float]
    rows: tuple[FeatureEvidenceAxesRow, ...]
    conclusion: str

    def as_dict(self) -> dict[str, object]:
        return {
            "matrix_path": self.matrix_path,
            "thresholds": dict(self.thresholds),
            "rows": [row.as_dict() for row in self.rows],
            "conclusion": self.conclusion,
            "interpretation_boundary": (
                "all persistence and quality-response values are detector-level; "
                "quality response may come from another feature class in the same "
                "registered neighborhood; no star probability, precision, FDR, or truth"
            ),
        }


_REQUIRED_COLUMNS = {
    "feature_class",
    "feature_class_label",
    "candidate_count",
    "candidate_presence_ge_required_count",
    "quality_response_ge_required_count",
    "spatial_local_correlation_median",
    "spatial_local_residual_median",
}

_PATTERN_TEXT: dict[str, tuple[str, str]] = {
    "aligned_quality_psf": (
        "候选、质量邻域与局部 PSF 同向",
        "位置持久、质量邻域响应充分，且局部 PSF 相关度达到当前审计线；可进入身份核验队列。",
    ),
    "persistent_without_quality": (
        "位置持久但质量不同步",
        "位置响应达到持久线，但质量邻域响应很少；应优先检查值域、支持、混叠或局部噪声。",
    ),
    "location_persistent_quality_sparse": (
        "位置持久、质量部分同步",
        "位置响应较稳定，但质量邻域只部分同步；应结合有效孔径、边缘/掩膜和结构证据分流。",
    ),
    "sparse_or_structure_sensitive": (
        "位置稀疏或结构敏感",
        "位置持久线本身较少，或响应主要受线几何/值域等专项机制控制；不宜用通用持久率解释。",
    ),
    "no_sample": (
        "无样本",
        "当前矩阵没有该类别候选；缺失样本不支持任何物理结论。",
    ),
}

_CAUTION = (
    "Q|P=质量邻域持久数/候选位置持久数，仅为当前注册与规则口径下的描述性比值；"
    "质量邻域响应可能属于其它类别，不能解释为逐源质量通过率或真星概率。"
)


def _required_int(row: Mapping[str, str], key: str) -> int:
    value = str(row.get(key, "")).strip()
    if not value:
        raise ValueError(f"feature evidence matrix has empty {key}")
    try:
        parsed = int(float(value))
    except ValueError as exc:
        raise ValueError(f"feature evidence matrix has invalid {key}={value!r}") from exc
    if parsed < 0:
        raise ValueError(f"feature evidence matrix has negative {key}={parsed}")
    return parsed


def _optional_float(row: Mapping[str, str], key: str) -> float | None:
    value = str(row.get(key, "")).strip()
    if not value or value.lower() in {"none", "nan", "null"}:
        return None
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"feature evidence matrix has invalid {key}={value!r}") from exc


def _classify_pattern(
    *,
    candidate_count: int,
    candidate_persistence: float,
    quality_response: float,
    spatial_correlation: float | None,
) -> str:
    if candidate_count == 0:
        return "no_sample"
    if (
        candidate_persistence >= 0.80
        and quality_response >= 0.80
        and spatial_correlation is not None
        and spatial_correlation >= 0.80
    ):
        return "aligned_quality_psf"
    if candidate_persistence >= 0.50 and quality_response <= 0.05:
        return "persistent_without_quality"
    if candidate_persistence >= 0.50:
        return "location_persistent_quality_sparse"
    return "sparse_or_structure_sensitive"


def run_feature_evidence_axes(matrix_path: str | Path) -> FeatureEvidenceAxesResult:
    """从已有证据矩阵计算类别级证据轴关系。"""

    path = Path(matrix_path)
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        columns = set(reader.fieldnames or ())
        missing = sorted(_REQUIRED_COLUMNS - columns)
        if missing:
            raise ValueError(f"feature evidence matrix is missing required columns: {', '.join(missing)}")
        records = list(reader)
    if not records:
        raise ValueError("feature evidence matrix is empty")

    rows: list[FeatureEvidenceAxesRow] = []
    seen: set[str] = set()
    for record in records:
        feature_class = str(record.get("feature_class", "")).strip()
        if not feature_class:
            raise ValueError("feature evidence matrix has an empty feature_class")
        if feature_class in seen:
            raise ValueError(f"feature evidence matrix contains duplicate feature_class={feature_class}")
        seen.add(feature_class)

        candidate_count = _required_int(record, "candidate_count")
        candidate_persistent_count = _required_int(record, "candidate_presence_ge_required_count")
        quality_response_count = _required_int(record, "quality_response_ge_required_count")
        if candidate_persistent_count > candidate_count:
            raise ValueError(
                f"{feature_class}: candidate persistent count exceeds candidate count "
                f"({candidate_persistent_count}>{candidate_count})"
            )
        if quality_response_count > candidate_count:
            raise ValueError(
                f"{feature_class}: quality response count exceeds candidate count "
                f"({quality_response_count}>{candidate_count})"
            )

        candidate_persistence = candidate_persistent_count / candidate_count if candidate_count else 0.0
        quality_response = quality_response_count / candidate_count if candidate_count else 0.0
        quality_given_candidate = (
            quality_response_count / candidate_persistent_count if candidate_persistent_count else None
        )
        spatial_correlation = _optional_float(record, "spatial_local_correlation_median")
        spatial_residual = _optional_float(record, "spatial_local_residual_median")
        pattern = _classify_pattern(
            candidate_count=candidate_count,
            candidate_persistence=candidate_persistence,
            quality_response=quality_response,
            spatial_correlation=spatial_correlation,
        )
        pattern_label, interpretation = _PATTERN_TEXT[pattern]
        rows.append(
            FeatureEvidenceAxesRow(
                feature_class=feature_class,
                feature_class_label=str(record.get("feature_class_label", "")).strip(),
                candidate_count=candidate_count,
                candidate_persistent_count=candidate_persistent_count,
                quality_response_count=quality_response_count,
                candidate_persistence_fraction=candidate_persistence,
                quality_response_fraction=quality_response,
                quality_response_given_candidate_persistent_fraction=quality_given_candidate,
                spatial_local_correlation_median=spatial_correlation,
                spatial_local_residual_median=spatial_residual,
                evidence_axis_pattern=pattern,
                evidence_axis_pattern_label=pattern_label,
                interpretation=interpretation,
                caution=_CAUTION,
            )
        )

    thresholds = {
        "aligned_candidate_persistence_min": 0.80,
        "aligned_quality_response_min": 0.80,
        "aligned_spatial_correlation_min": 0.80,
        "persistent_candidate_persistence_min": 0.50,
        "persistent_quality_response_max": 0.05,
    }
    conclusion = (
        "类别证据轴关系已汇总：紧凑质量类是候选位置、质量邻域和局部 PSF 同向的对照；"
        "拥挤、尖峰、弱背景和形状类可出现位置持久但质量不同步；边缘/掩膜类处于部分同步；"
        "线状与范围异常类更适合走专项机制审计。Q|P 只描述 detector-level 证据同步，"
        "不提供逐源 precision、FDR、恒星概率或物理类别真值。"
    )
    return FeatureEvidenceAxesResult(
        matrix_path=str(path),
        thresholds=thresholds,
        rows=tuple(rows),
        conclusion=conclusion,
    )


def write_feature_evidence_axes_artifacts(
    result: FeatureEvidenceAxesResult,
    output_dir: str | Path,
) -> Path:
    """写出证据轴关系 CSV/JSON。"""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "feature_evidence_axes.csv"
    json_path = output / "feature_evidence_axes.json"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=AXIS_SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(row.as_dict() for row in result.rows)
    with json_path.open("w", encoding="utf-8") as stream:
        json.dump(result.as_dict(), stream, ensure_ascii=False, indent=2)
    return output


__all__ = [
    "AXIS_SUMMARY_FIELDS",
    "FeatureEvidenceAxesResult",
    "FeatureEvidenceAxesRow",
    "run_feature_evidence_axes",
    "write_feature_evidence_axes_artifacts",
]
