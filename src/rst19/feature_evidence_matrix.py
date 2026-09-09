"""合并类别形态、15 帧持久性和留一 PSF 的证据矩阵。

这个模块不重新检测 FITS，也不把多个指标加权成一个“真星分数”。它只
把已经生成的三张研究表按 ``feature_class`` 对齐，便于论文和答辩同时
展示候选数量、质量层邻域响应、跨类响应和局部 PSF 形状。输出中的
``pattern`` 是保守的证据组合标签，不是物理类别、恒星概率或伪影真值。
``recommended_audit_stage`` 是确定性的复核路由，同样不是分类器或概率。
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Mapping, Sequence

import numpy as np


_FEATURE_ORDER: tuple[tuple[str, str], ...] = (
    ("range_anomaly", "数据有效性/范围异常邻域"),
    ("linear_artifact", "线状/拖影候选"),
    ("masked_or_edge", "边缘/掩膜候选"),
    ("crowded_blend", "拥挤/未分辨近邻"),
    ("spike_or_support", "尖峰/PSF 支持不足"),
    ("weak_or_background", "弱通量/背景不确定"),
    ("shape_outlier", "形状异常"),
    ("compact_quality", "质量通过的紧凑候选"),
    ("other_rejected", "其他拒绝候选"),
)

_SECONDARY_FEATURES: tuple[tuple[str, str], ...] = (
    ("range_anomaly", "数据有效性/范围异常邻域"),
    ("linear_artifact", "线状/拖影候选"),
    ("masked_or_edge", "边缘/掩膜候选"),
    ("crowded_blend", "拥挤/未分辨近邻"),
    ("spike_or_support", "尖峰/PSF 支持不足"),
    ("weak_or_background", "弱通量/背景不确定"),
    ("shape_outlier", "形状异常"),
)

_AUDIT_ROUTES: dict[str, tuple[str, str, str, str]] = {
    "range_anomaly": (
        "value_domain",
        "原始值域/编码核验",
        "candidate_only",
        "仅保留候选，不直接计星",
    ),
    "linear_artifact": (
        "line_geometry",
        "线几何与跨帧运动核验",
        "candidate_only",
        "仅保留候选，不直接计星",
    ),
    "masked_or_edge": (
        "aperture_validity",
        "有效孔径、边缘与掩膜核验",
        "candidate_only",
        "仅保留候选，不直接计星",
    ),
    "crowded_blend": (
        "joint_psf_deblend",
        "联合单/双 PSF 去混叠",
        "candidate_only",
        "仅保留候选，不直接计星",
    ),
    "spike_or_support": (
        "psf_support",
        "核心支持与高频结构核验",
        "candidate_only",
        "仅保留候选，不直接计星",
    ),
    "weak_or_background": (
        "local_noise_injection",
        "分区噪声与注入回收核验",
        "candidate_only",
        "仅保留候选，不直接计星",
    ),
    "shape_outlier": (
        "shape_psf",
        "FWHM、椭圆率与径向 PSF 核验",
        "candidate_only",
        "仅保留候选，不直接计星",
    ),
    "compact_quality": (
        "catalog_wcs",
        "星表/WCS 与留出注入核验",
        "identity_review_queue",
        "仅可进入身份核验队列",
    ),
    "other_rejected": (
        "manual_review",
        "规则缺口与人工复核",
        "candidate_only",
        "仅保留候选，不直接计星",
    ),
}


@dataclass(frozen=True, slots=True)
class FeatureEvidenceRow:
    """一个首要特征类别的三层证据摘要。"""

    feature_class: str
    feature_class_label: str
    candidate_count: int | None
    quality_count: int | None
    quality_fraction: float | None
    high_flux_snr_rejection_fraction: float | None
    dominant_secondary_feature: str | None
    dominant_secondary_feature_label: str | None
    dominant_secondary_fraction: float | None
    candidate_presence_ge_required_count: int | None
    candidate_persistence_fraction: float | None
    # These two fields come from feature_cross_audit's quality response around
    # the anchor. They are deliberately not named "quality persistence": the
    # response can come from a different feature class in the same neighborhood.
    quality_response_ge_required_count: int | None
    quality_response_fraction: float | None
    candidate_any_response_fraction: float | None
    candidate_same_class_response_fraction: float | None
    quality_any_response_fraction: float | None
    quality_same_class_response_fraction: float | None
    leaveout_sample_count: int | None
    leaveout_valid_count: int | None
    leaveout_correlation_median: float | None
    leaveout_correlation_ge_threshold_fraction: float | None
    leaveout_residual_median: float | None
    spatial_local_template_available_cell_count: int | None
    spatial_local_template_source_count_median: float | None
    spatial_local_valid_count: int | None
    spatial_local_fallback_count: int | None
    spatial_paired_count: int | None
    spatial_local_correlation_median: float | None
    spatial_local_residual_median: float | None
    spatial_local_correlation_improved_fraction: float | None
    spatial_local_residual_reduced_fraction: float | None
    spatial_median_correlation_delta: float | None
    spatial_median_residual_delta: float | None
    candidate_persistence_wilson95_low: float | None
    candidate_persistence_wilson95_high: float | None
    quality_fraction_wilson95_low: float | None
    quality_fraction_wilson95_high: float | None
    quality_response_wilson95_low: float | None
    quality_response_wilson95_high: float | None
    evidence_pattern: str
    note: str
    recommended_audit_stage: str
    recommended_audit_stage_label: str
    counting_policy: str
    counting_policy_label: str

    def as_dict(self) -> dict[str, object]:
        return {
            "feature_class": self.feature_class,
            "feature_class_label": self.feature_class_label,
            "candidate_count": self.candidate_count,
            "quality_count": self.quality_count,
            "quality_fraction": self.quality_fraction,
            "high_flux_snr_rejection_fraction": self.high_flux_snr_rejection_fraction,
            "dominant_secondary_feature": self.dominant_secondary_feature,
            "dominant_secondary_feature_label": self.dominant_secondary_feature_label,
            "dominant_secondary_fraction": self.dominant_secondary_fraction,
            "candidate_presence_ge_required_count": self.candidate_presence_ge_required_count,
            "candidate_persistence_fraction": self.candidate_persistence_fraction,
            "quality_response_ge_required_count": self.quality_response_ge_required_count,
            "quality_response_fraction": self.quality_response_fraction,
            "candidate_any_response_fraction": self.candidate_any_response_fraction,
            "candidate_same_class_response_fraction": self.candidate_same_class_response_fraction,
            "quality_any_response_fraction": self.quality_any_response_fraction,
            "quality_same_class_response_fraction": self.quality_same_class_response_fraction,
            "leaveout_sample_count": self.leaveout_sample_count,
            "leaveout_valid_count": self.leaveout_valid_count,
            "leaveout_correlation_median": self.leaveout_correlation_median,
            "leaveout_correlation_ge_threshold_fraction": self.leaveout_correlation_ge_threshold_fraction,
            "leaveout_residual_median": self.leaveout_residual_median,
            "spatial_local_template_available_cell_count": self.spatial_local_template_available_cell_count,
            "spatial_local_template_source_count_median": self.spatial_local_template_source_count_median,
            "spatial_local_valid_count": self.spatial_local_valid_count,
            "spatial_local_fallback_count": self.spatial_local_fallback_count,
            "spatial_paired_count": self.spatial_paired_count,
            "spatial_local_correlation_median": self.spatial_local_correlation_median,
            "spatial_local_residual_median": self.spatial_local_residual_median,
            "spatial_local_correlation_improved_fraction": self.spatial_local_correlation_improved_fraction,
            "spatial_local_residual_reduced_fraction": self.spatial_local_residual_reduced_fraction,
            "spatial_median_correlation_delta": self.spatial_median_correlation_delta,
            "spatial_median_residual_delta": self.spatial_median_residual_delta,
            "candidate_persistence_wilson95_low": self.candidate_persistence_wilson95_low,
            "candidate_persistence_wilson95_high": self.candidate_persistence_wilson95_high,
            "quality_fraction_wilson95_low": self.quality_fraction_wilson95_low,
            "quality_fraction_wilson95_high": self.quality_fraction_wilson95_high,
            "quality_response_wilson95_low": self.quality_response_wilson95_low,
            "quality_response_wilson95_high": self.quality_response_wilson95_high,
            "evidence_pattern": self.evidence_pattern,
            "note": self.note,
            "recommended_audit_stage": self.recommended_audit_stage,
            "recommended_audit_stage_label": self.recommended_audit_stage_label,
            "counting_policy": self.counting_policy,
            "counting_policy_label": self.counting_policy_label,
        }


@dataclass(frozen=True, slots=True)
class FeatureEvidenceMatrixResult:
    """类别证据矩阵及其输入口径。"""

    source_morphology_path: str
    feature_cross_path: str
    psf_leaveout_path: str
    psf_spatial_path: str | None
    correlation_threshold: float
    rows: tuple[FeatureEvidenceRow, ...]
    parameters: dict[str, object]
    conclusion: str

    def as_dict(self) -> dict[str, object]:
        return {
            "source_morphology_path": self.source_morphology_path,
            "feature_cross_path": self.feature_cross_path,
            "psf_leaveout_path": self.psf_leaveout_path,
            "psf_spatial_path": self.psf_spatial_path,
            "correlation_threshold": self.correlation_threshold,
            "rows": [row.as_dict() for row in self.rows],
            "parameters": self.parameters,
            "conclusion": self.conclusion,
        }


def _read_csv(path: str | Path) -> tuple[dict[str, str], ...]:
    csv_path = Path(path)
    with csv_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or "feature_class" not in reader.fieldnames:
            raise ValueError(f"{csv_path} must contain feature_class")
        return tuple(dict(row) for row in reader)


def _index_rows(rows: Sequence[Mapping[str, str]], path: str | Path) -> dict[str, Mapping[str, str]]:
    indexed: dict[str, Mapping[str, str]] = {}
    for row in rows:
        feature_class = str(row.get("feature_class", "")).strip()
        if not feature_class:
            continue
        if feature_class in indexed:
            raise ValueError(f"{path} contains duplicate feature_class={feature_class}")
        indexed[feature_class] = row
    return indexed


def _optional_int(row: Mapping[str, str] | None, key: str) -> int | None:
    if row is None:
        return None
    value = str(row.get(key, "")).strip()
    if not value:
        return None
    try:
        return int(float(value))
    except ValueError as exc:
        raise ValueError(f"invalid integer {key}={value!r}") from exc


def _optional_float(row: Mapping[str, str] | None, key: str) -> float | None:
    if row is None:
        return None
    value = str(row.get(key, "")).strip()
    if not value:
        return None
    try:
        converted = float(value)
    except ValueError as exc:
        raise ValueError(f"invalid number {key}={value!r}") from exc
    return converted if np.isfinite(converted) else None


def _wilson95_interval(
    successes: int | None,
    trials: int | None,
) -> tuple[float | None, float | None]:
    """Return a descriptive Wilson 95% interval for an observed proportion.

    The interval is attached to an observed anchor count only. It is not a
    confidence interval for physical truth because candidates in one image
    share background, registration and detector systematics.
    """

    if successes is None or trials is None:
        return None, None
    if trials < 0 or successes < 0 or successes > trials:
        raise ValueError(f"invalid proportion counts: successes={successes}, trials={trials}")
    if trials == 0:
        return None, None
    z = NormalDist().inv_cdf(0.975)
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    center = (proportion + z * z / (2.0 * trials)) / denominator
    half_width = (
        z
        * np.sqrt(
            proportion * (1.0 - proportion) / trials
            + z * z / (4.0 * trials * trials)
        )
        / denominator
    )
    lower = 0.0 if successes == 0 else max(0.0, center - half_width)
    upper = 1.0 if successes == trials else min(1.0, center + half_width)
    return lower, upper


def _label_for(
    feature_class: str,
    default_label: str,
    *rows: Mapping[str, str] | None,
) -> str:
    for row in rows:
        if row is None:
            continue
        label = str(row.get("feature_class_label", "")).strip()
        if label:
            return label
    return default_label


def _dominant_secondary(
    feature_class: str,
    morphology_row: Mapping[str, str] | None,
) -> tuple[str | None, str | None, float | None]:
    """读取主类别内部的次级旗标重叠，排除主类别自身。"""

    if morphology_row is None:
        return None, None, None
    values: list[tuple[float, int, str, str]] = []
    for order, (secondary, label) in enumerate(_SECONDARY_FEATURES):
        if secondary == feature_class:
            continue
        value = _optional_float(morphology_row, f"{secondary}_fraction")
        if value is not None and value > 0.0:
            values.append((value, -order, secondary, label))
    if not values:
        return None, None, None
    value, _order, secondary, label = max(values)
    return secondary, label, value


def _pattern(
    feature_class: str,
    *,
    quality_fraction: float | None,
    candidate_persistence: float | None,
    quality_response: float | None,
    correlation: float | None,
    correlation_threshold: float,
) -> tuple[str, str]:
    """给出只基于组合规则的保守解释，不输出物理真值。"""

    if feature_class == "other_rejected" and quality_fraction is None:
        return "无非空样本", "当前表没有该类别的候选，不能从缺失数据推出物理结论。"
    if feature_class == "range_anomaly":
        return "值域优先审计", "范围/编码风险必须先于时序和 PSF 指标解释。"
    if (
        quality_fraction is not None
        and quality_fraction >= 0.8
        and quality_response is not None
        and quality_response >= 0.5
        and correlation is not None
        and correlation >= correlation_threshold
    ):
        return "质量/时序/PSF 三层一致", "三层证据在当前抽样中相互支持，但仍需星表或注入真值。"
    if (
        candidate_persistence is not None
        and candidate_persistence >= 0.5
        and (
            quality_fraction is None
            or quality_fraction < 0.1
            or (quality_response is not None and quality_response < 0.1)
        )
        and (correlation is None or correlation < 0.7)
    ):
        return "候选持续但质量/PSF 不支持", "跨帧重复不能单独升级为恒星，应优先复核噪声、结构、掩膜或混合。"
    if correlation is not None and correlation < 0.7:
        return "形状/结构优先复核", "局部裁剪不像标准孤立 PSF，需结合几何、值域和邻峰证据。"
    return "证据未闭环", "当前三层指标不足以完成身份判断，应做留出注入或星表/WCS 核验。"


def _audit_route(feature_class: str) -> tuple[str, str, str, str]:
    """返回类别的下一步复核路由，不把路由解释成物理类别或概率。"""

    return _AUDIT_ROUTES.get(feature_class, _AUDIT_ROUTES["other_rejected"])


def build_feature_evidence_matrix(
    source_morphology_path: str | Path,
    feature_cross_path: str | Path,
    psf_leaveout_path: str | Path,
    *,
    correlation_threshold: float = 0.8,
    psf_spatial_path: str | Path | None = None,
) -> FeatureEvidenceMatrixResult:
    """按类别合并形态、15 帧交叉响应和全局/局部留一 PSF 研究表。"""

    if not 0.0 < float(correlation_threshold) <= 1.0:
        raise ValueError("correlation_threshold must be between 0 and 1")
    morphology_rows = _read_csv(source_morphology_path)
    cross_rows = _read_csv(feature_cross_path)
    psf_rows = _read_csv(psf_leaveout_path)
    spatial_rows = _read_csv(psf_spatial_path) if psf_spatial_path is not None else ()
    morphology = _index_rows(morphology_rows, source_morphology_path)
    cross = _index_rows(cross_rows, feature_cross_path)
    psf = _index_rows(psf_rows, psf_leaveout_path)
    spatial = _index_rows(spatial_rows, psf_spatial_path) if psf_spatial_path is not None else {}

    output_rows: list[FeatureEvidenceRow] = []
    for feature_class, default_label in _FEATURE_ORDER:
        morphology_row = morphology.get(feature_class)
        cross_row = cross.get(feature_class)
        psf_row = psf.get(feature_class)
        spatial_row = spatial.get(feature_class)
        quality_fraction = _optional_float(morphology_row, "quality_fraction")
        dominant_secondary, dominant_secondary_label, dominant_secondary_fraction = _dominant_secondary(
            feature_class, morphology_row
        )
        candidate_persistence = _optional_float(cross_row, "candidate_presence_fraction")
        # The input column is named quality_presence_fraction for historical
        # compatibility. It measures a quality-layer response in the anchor's
        # neighborhood, not persistence of the same class.
        quality_response = _optional_float(cross_row, "quality_presence_fraction")
        correlation = _optional_float(psf_row, "leaveout_correlation_median")
        candidate_count = _optional_int(morphology_row, "candidate_count")
        quality_count = _optional_int(morphology_row, "quality_count")
        candidate_presence_count = _optional_int(cross_row, "candidate_presence_ge_required_count")
        anchor_candidate_count = _optional_int(cross_row, "anchor_candidate_count")
        quality_response_count = _optional_int(cross_row, "quality_presence_ge_required_count")
        candidate_persistence_wilson95_low, candidate_persistence_wilson95_high = _wilson95_interval(
            candidate_presence_count, anchor_candidate_count
        )
        quality_fraction_wilson95_low, quality_fraction_wilson95_high = _wilson95_interval(
            quality_count, candidate_count
        )
        quality_response_wilson95_low, quality_response_wilson95_high = _wilson95_interval(
            quality_response_count, anchor_candidate_count
        )
        pattern, note = _pattern(
            feature_class,
            quality_fraction=quality_fraction,
            candidate_persistence=candidate_persistence,
            quality_response=quality_response,
            correlation=correlation,
            correlation_threshold=float(correlation_threshold),
        )
        audit_stage, audit_stage_label, counting_policy, counting_policy_label = _audit_route(feature_class)
        output_rows.append(
            FeatureEvidenceRow(
                feature_class=feature_class,
                feature_class_label=_label_for(feature_class, default_label, morphology_row, cross_row, psf_row),
                candidate_count=candidate_count,
                quality_count=quality_count,
                quality_fraction=quality_fraction,
                high_flux_snr_rejection_fraction=_optional_float(
                    morphology_row, "high_flux_snr_rejection_fraction"
                ),
                dominant_secondary_feature=dominant_secondary,
                dominant_secondary_feature_label=dominant_secondary_label,
                dominant_secondary_fraction=dominant_secondary_fraction,
                candidate_presence_ge_required_count=_optional_int(
                    cross_row, "candidate_presence_ge_required_count"
                ),
                candidate_persistence_fraction=candidate_persistence,
                quality_response_ge_required_count=_optional_int(
                    cross_row, "quality_presence_ge_required_count"
                ),
                quality_response_fraction=quality_response,
                candidate_any_response_fraction=_optional_float(
                    cross_row, "candidate_any_response_fraction"
                ),
                candidate_same_class_response_fraction=_optional_float(
                    cross_row, "candidate_same_class_response_fraction"
                ),
                quality_any_response_fraction=_optional_float(
                    cross_row, "quality_any_response_fraction"
                ),
                quality_same_class_response_fraction=_optional_float(
                    cross_row, "quality_same_class_response_fraction"
                ),
                leaveout_sample_count=_optional_int(psf_row, "sample_count"),
                leaveout_valid_count=_optional_int(psf_row, "leaveout_valid_count"),
                leaveout_correlation_median=correlation,
                leaveout_correlation_ge_threshold_fraction=_optional_float(
                    psf_row, "leaveout_correlation_ge_threshold_fraction"
                ),
                leaveout_residual_median=_optional_float(psf_row, "leaveout_residual_median"),
                spatial_local_template_available_cell_count=_optional_int(
                    spatial_row, "local_template_available_cell_count"
                ),
                spatial_local_template_source_count_median=_optional_float(
                    spatial_row, "local_template_source_count_median"
                ),
                spatial_local_valid_count=_optional_int(spatial_row, "local_valid_count"),
                spatial_local_fallback_count=_optional_int(spatial_row, "local_fallback_count"),
                spatial_paired_count=_optional_int(spatial_row, "paired_correlation_count"),
                spatial_local_correlation_median=_optional_float(
                    spatial_row, "local_correlation_median"
                ),
                spatial_local_residual_median=_optional_float(spatial_row, "local_residual_median"),
                spatial_local_correlation_improved_fraction=_optional_float(
                    spatial_row, "local_correlation_improved_fraction"
                ),
                spatial_local_residual_reduced_fraction=_optional_float(
                    spatial_row, "local_residual_reduced_fraction"
                ),
                spatial_median_correlation_delta=_optional_float(
                    spatial_row, "median_correlation_delta_local_minus_global"
                ),
                spatial_median_residual_delta=_optional_float(
                    spatial_row, "median_residual_delta_local_minus_global"
                ),
                candidate_persistence_wilson95_low=candidate_persistence_wilson95_low,
                candidate_persistence_wilson95_high=candidate_persistence_wilson95_high,
                quality_fraction_wilson95_low=quality_fraction_wilson95_low,
                quality_fraction_wilson95_high=quality_fraction_wilson95_high,
                quality_response_wilson95_low=quality_response_wilson95_low,
                quality_response_wilson95_high=quality_response_wilson95_high,
                evidence_pattern=pattern,
                note=note,
                recommended_audit_stage=audit_stage,
                recommended_audit_stage_label=audit_stage_label,
                counting_policy=counting_policy,
                counting_policy_label=counting_policy_label,
            )
        )

    conclusion = (
        "该矩阵只把三类已有研究量按首要特征类别对齐：形态/质量层、15 帧候选与质量层邻域响应、"
        "以及逐源留一经验 PSF。当前数据中，候选层的跨帧重复在多个被拒类别中仍然较高，"
        "而质量层邻域响应和标准 PSF 形状并未同步出现；因此“可重复”不能单独等同于“真星”。"
        "若提供空间 PSF 表，矩阵还会保留局部模板可得率、局部相关度/残差及相对全局模板的变化；"
        "这些字段只检验空间校正是否改变解释，不把模板不足或局部改善直接转成真星结论。"
        "矩阵是证据组织工具，不是加权分类器、概率模型或物理真值表。"
        "recommended_audit_stage 只是确定性的下一步复核路由，counting_policy 只规定当前研究的计数边界，"
        "二者都不产生恒星概率。"
    )
    return FeatureEvidenceMatrixResult(
        source_morphology_path=str(Path(source_morphology_path)),
        feature_cross_path=str(Path(feature_cross_path)),
        psf_leaveout_path=str(Path(psf_leaveout_path)),
        psf_spatial_path=str(Path(psf_spatial_path)) if psf_spatial_path is not None else None,
        correlation_threshold=float(correlation_threshold),
        rows=tuple(output_rows),
        parameters={
            "feature_order": [feature_class for feature_class, _label in _FEATURE_ORDER],
            "temporal_required_presence": "沿用 feature_cross_audit.csv 的 required_presence，当前为 12/15 帧",
            "quality_response_definition": "feature_cross_audit.csv 的 quality_presence_* 是质量层在锚点邻域内的响应，不等价于同类质量通过或同类跨帧持久性",
            "same_class_response_definition": "feature_cross_audit.csv 的 *_same_class_response_fraction 只表示响应类别仍与锚点首要类别相同，不是真阳性率",
            "psf_definition": "逐源排除自身、最多 20 个隔离质量模板、留一相关度与相对残差",
            "spatial_psf_definition": (
                "可选 source_feature_psf_spatial.csv；局部模板按空间网格留出比较，"
                "local_valid/fallback 先于局部相关度解释"
            ),
            "spatial_psf_is_diagnostic_only": True,
            "proportion_interval_definition": (
                "95% Wilson interval from observed anchor counts; descriptive only, "
                "not a confidence interval for physical truth because shared image/systematic effects remain"
            ),
            "pattern_is_not_truth": True,
            "routing_definition": (
                "按 feature_class 确定下一步复核模块；不使用指标加权，不输出概率，不改写 detection 或 quality_count"
            ),
            "counting_policy_definition": (
                "只有 compact_quality 可进入星表/WCS 或留出注入身份核验队列；所有其他类别保留候选但不直接计星"
            ),
        },
        conclusion=conclusion,
    )


def write_feature_evidence_matrix_artifacts(
    result: FeatureEvidenceMatrixResult,
    out_dir: str | Path,
) -> Path:
    """写入类别证据矩阵 CSV/JSON。"""

    output = Path(out_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = [row.as_dict() for row in result.rows]
    fields = tuple(rows[0].keys()) if rows else ()
    with (output / "feature_evidence_matrix.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (output / "feature_evidence_matrix.json").write_text(
        json.dumps(result.as_dict(), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return output
