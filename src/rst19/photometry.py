"""检测源的相对/仪器星等和最暗可信源选择。

本模块刻意把三个容易混淆的量分开：

* ``m_inst``：由本机 ADU 测量得到的仪器星等；
* ``m_std``：经过标准星、零点和波段/颜色项标定后的表观星等；
* ``M_V``：还需要距离（或视差）和消光修正的绝对星等。

没有这些外部标定条件时，不能把 ``m_inst`` 改名成 ``m_V`` 或 ``M_V``。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Mapping, Sequence

import numpy as np

from .catalog import CatalogSource
from .detection import Detection
from .matching import CatalogMatch


DEFAULT_MAX_FIT_RMS_MAG = 0.25
DEFAULT_MAX_VALIDATION_RMS_MAG = 0.35
DEFAULT_MAX_RESIDUAL_MAD_MAG = 0.25
DEFAULT_MAX_VALIDATION_BIAS_MAG = 0.20
DEFAULT_MIN_VALIDATION_COUNT = 2
DEFAULT_MAX_SOURCE_RESIDUAL_MAG = 0.50


def _finite(value: float | None) -> bool:
    return value is not None and math.isfinite(float(value))


_UNKNOWN_PHOTOMETRIC_LABELS = frozenset(
    {"", "unknown", "none", "null", "n/a", "na", "undefined", "?", "-", "—"}
)


def _normalise_photometric_label(value: str | None) -> str | None:
    """Normalize a provenance label without guessing an omitted band/system."""

    if value is None:
        return None
    text = " ".join(str(value).strip().split())
    if text.casefold() in _UNKNOWN_PHOTOMETRIC_LABELS:
        return None
    return text.casefold()


def _photometric_metadata_declared(system: str | None, band: str | None) -> bool:
    return _normalise_photometric_label(system) is not None and _normalise_photometric_label(band) is not None


def _provenance_declared(value: str | None) -> bool:
    """Return whether a text provenance value carries an actual declaration."""

    return _normalise_photometric_label(value) is not None


def _photometric_metadata_matches(
    expected_system: str | None,
    expected_band: str | None,
    actual_system: str | None,
    actual_band: str | None,
) -> bool:
    """Require an explicit, exact system/band match.

    In particular, Gaia Vega ``G`` and Johnson ``V`` are different passbands.
    An omitted/``unknown`` label is not treated as a wildcard because that
    would allow a catalog value to cross the G/V boundary silently.
    """

    expected_system_normalized = _normalise_photometric_label(expected_system)
    expected_band_normalized = _normalise_photometric_label(expected_band)
    actual_system_normalized = _normalise_photometric_label(actual_system)
    actual_band_normalized = _normalise_photometric_label(actual_band)
    return (
        expected_system_normalized is not None
        and expected_band_normalized is not None
        and expected_system_normalized == actual_system_normalized
        and expected_band_normalized == actual_band_normalized
    )


def _absolute_unavailable(
    *,
    corrected_parallax_mas: float | None,
    distance_pc: float | None,
    extinction_mag: float | None,
    status: str,
    flags: tuple[str, ...],
    distance_source: str | None = None,
    distance_lower_pc: float | None = None,
    distance_upper_pc: float | None = None,
    extinction_band: str | None = None,
    extinction_system: str | None = None,
    extinction_source: str | None = None,
) -> "AbsoluteMagnitudeEstimate":
    """Build a non-numeric absolute-magnitude result for a failed quality gate."""

    return AbsoluteMagnitudeEstimate(
        value=None,
        error=None,
        distance_pc=distance_pc,
        corrected_parallax_mas=corrected_parallax_mas,
        extinction_mag=extinction_mag,
        status=status,
        flags=flags,
        distance_source=distance_source,
        distance_lower_pc=distance_lower_pc,
        distance_upper_pc=distance_upper_pc,
        extinction_band=extinction_band,
        extinction_system=extinction_system,
        extinction_source=extinction_source,
    )


@dataclass(frozen=True, slots=True)
class FaintestSource:
    """当前图像中最暗的可信检测源。

    没有零点时，`instrumental_magnitude` 只表示
    `-2.5 log10(flux_rate)` 的仪器星等，不能直接解释为 Gaia V、Johnson V
    或 `Mv`。
    """

    detection_id: int
    x: float
    y: float
    flux: float
    snr: float
    instrumental_magnitude: float
    calibrated_magnitude: float | None
    flags: tuple[str, ...]
    flux_snr: float | None = None
    flux_rate: float | None = None
    instrumental_magnitude_error: float | None = None
    photometric_system: str | None = None
    photometric_band: str | None = None
    calibration_status: str = "INSTRUMENTAL"
    calibrated_magnitude_error: float | None = None
    absolute_magnitude: "AbsoluteMagnitudeEstimate | None" = None
    # 记录“最暗”是在什么集合中求出的。尤其不能把仅覆盖部分
    # 检测源的星表标定结果误写成整幅图像的全局最暗星。
    selection_scope: str = "QUALITY_DETECTIONS"
    eligible_candidate_count: int = 0
    calibrated_candidate_count: int = 0

    def as_dict(self) -> dict[str, object]:
        return {
            "detection_id": self.detection_id,
            "x": self.x,
            "y": self.y,
            "flux": self.flux,
            "snr": self.snr,
            "flux_snr": self.flux_snr,
            "flux_rate": self.flux_rate,
            "instrumental_magnitude": self.instrumental_magnitude,
            "calibrated_magnitude": self.calibrated_magnitude,
            "flags": list(self.flags),
            "instrumental_magnitude_error": self.instrumental_magnitude_error,
            "photometric_system": self.photometric_system,
            "photometric_band": self.photometric_band,
            "calibration_status": self.calibration_status,
            "calibrated_magnitude_error": self.calibrated_magnitude_error,
            "absolute_magnitude": self.absolute_magnitude.as_dict() if self.absolute_magnitude else None,
            "selection_scope": self.selection_scope,
            "eligible_candidate_count": self.eligible_candidate_count,
            "calibrated_candidate_count": self.calibrated_candidate_count,
        }


@dataclass(frozen=True, slots=True)
class PhotometricCalibration:
    """由多颗参考星拟合得到的经验光度标定。

    ``coefficients`` 按 ``[zero_point, color_term, color_term_2]`` 顺序
    保存；``color_order`` 决定实际使用前几个系数。该对象同时保存拟合
    证据，避免只把一个零点数字传到 GUI 后失去来源和质量边界。
    """

    photometric_system: str
    photometric_band: str
    color_name: str | None
    color_order: int
    coefficients: tuple[float, ...]
    calibrator_count: int
    inlier_count: int
    validation_count: int
    fit_rms_mag: float | None
    validation_rms_mag: float | None
    residual_mad_mag: float | None
    color_min: float | None
    color_max: float | None
    status: str
    flags: tuple[str, ...] = ()
    catalog_filter_counts: tuple[tuple[str, int], ...] = ()
    validation_bias_mag: float | None = None
    max_fit_rms_mag: float | None = None
    max_validation_rms_mag: float | None = None
    max_residual_mad_mag: float | None = None
    max_validation_bias_mag: float | None = None
    min_validation_count: int = 0
    max_source_residual_mag: float | None = None
    calibration_sample_roles: tuple[tuple[str, str], ...] = ()

    @property
    def zero_point(self) -> float | None:
        return self.coefficients[0] if self.coefficients else None

    @property
    def color_coefficient(self) -> float | None:
        return self.coefficients[1] if len(self.coefficients) > 1 else None

    def apply(self, instrumental: float, *, color: float | None = None) -> float | None:
        """把仪器星等转换到该标定声明的表观光度系统。"""

        if self.status not in {"VALID", "VALID_NO_HOLDOUT"} or not self.coefficients:
            return None
        if not _photometric_metadata_declared(self.photometric_system, self.photometric_band):
            return None
        if not math.isfinite(float(instrumental)):
            return None
        if self.color_order and color is None:
            return None
        values = [1.0]
        if self.color_order:
            if not math.isfinite(float(color)):
                return None
            if self.color_min is None or self.color_max is None:
                return None
            if float(color) < float(self.color_min) or float(color) > float(self.color_max):
                return None
            values.append(float(color))
        if self.color_order >= 2:
            values.append(float(color) ** 2)
        coefficients = np.asarray(self.coefficients[: self.color_order + 1], dtype=np.float64)
        if coefficients.size != len(values) or not np.isfinite(coefficients).all():
            return None
        return float(instrumental) + float(np.dot(np.asarray(values, dtype=np.float64), coefficients))

    def as_dict(self) -> dict[str, object]:
        return {
            "photometric_system": self.photometric_system,
            "photometric_band": self.photometric_band,
            "color_name": self.color_name,
            "color_order": self.color_order,
            "coefficients": list(self.coefficients),
            "zero_point": self.zero_point,
            "color_coefficient": self.color_coefficient,
            "calibrator_count": self.calibrator_count,
            "inlier_count": self.inlier_count,
            "validation_count": self.validation_count,
            "fit_rms_mag": self.fit_rms_mag,
            "validation_rms_mag": self.validation_rms_mag,
            "residual_mad_mag": self.residual_mad_mag,
            "color_min": self.color_min,
            "color_max": self.color_max,
            "status": self.status,
            "flags": list(self.flags),
            "catalog_filter_counts": [
                {"reason": reason, "count": count}
                for reason, count in self.catalog_filter_counts
            ],
            "validation_bias_mag": self.validation_bias_mag,
            "max_fit_rms_mag": self.max_fit_rms_mag,
            "max_validation_rms_mag": self.max_validation_rms_mag,
            "max_residual_mad_mag": self.max_residual_mad_mag,
            "max_validation_bias_mag": self.max_validation_bias_mag,
            "min_validation_count": self.min_validation_count,
            "max_source_residual_mag": self.max_source_residual_mag,
            "calibration_sample_roles": [
                {"source_id": source_id, "role": role}
                for source_id, role in self.calibration_sample_roles
            ],
        }


@dataclass(frozen=True, slots=True)
class AbsoluteMagnitudeEstimate:
    """带质量状态的绝对星等估计。"""

    value: float | None
    error: float | None
    distance_pc: float | None
    corrected_parallax_mas: float | None
    extinction_mag: float | None
    status: str
    flags: tuple[str, ...] = ()
    distance_source: str | None = None
    distance_lower_pc: float | None = None
    distance_upper_pc: float | None = None
    # Extinction provenance is optional for the generic APIs.  ``None`` means
    # that the caller did not declare a band/system; it must never be guessed
    # from the numeric value alone.  Catalog-backed results copy the explicit
    # fields from CatalogSource so the calculation remains auditable after
    # serialization.
    extinction_band: str | None = None
    extinction_system: str | None = None
    extinction_source: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "value": self.value,
            "error": self.error,
            "distance_pc": self.distance_pc,
            "corrected_parallax_mas": self.corrected_parallax_mas,
            "extinction_mag": self.extinction_mag,
            "status": self.status,
            "flags": list(self.flags),
            "distance_source": self.distance_source,
            "distance_lower_pc": self.distance_lower_pc,
            "distance_upper_pc": self.distance_upper_pc,
            "distance_interval_status": self.distance_interval_status,
            "distance_error_pc": self.distance_error_pc,
            "distance_error_lower_pc": self.distance_error_lower_pc,
            "distance_error_upper_pc": self.distance_error_upper_pc,
            "extinction_band": self.extinction_band,
            "extinction_system": self.extinction_system,
            "extinction_source": self.extinction_source,
            "error_status": self.error_status,
            "is_strict": self.is_strict,
            "is_strict_with_uncertainty": self.is_strict_with_uncertainty,
        }

    @property
    def distance_interval_status(self) -> str:
        """Describe the distance uncertainty carried by this result.

        The status is derived from the existing flags and bounds so older
        cache payloads remain readable.  A central ``distance_pc`` alone is
        deliberately not treated as an interval.
        """

        flags = set(self.flags)
        if "DISTANCE_INTERVAL_INVALID" in flags:
            return "INVALID"
        if "DISTANCE_INTERVAL_DERIVED_FROM_PARALLAX_ERROR" in flags:
            return "DERIVED_FROM_PARALLAX_ERROR"
        if "DISTANCE_INTERVAL_USED" in flags:
            return "PROVIDED"
        if "DISTANCE_INTERVAL_ONE_SIDED" in flags:
            return "ONE_SIDED"
        if "DISTANCE_INTERVAL_REQUIRED" in flags:
            return "REQUIRED"
        if "DISTANCE_ERROR_NOT_PROVIDED" in flags:
            return "NOT_PROVIDED"
        if self.distance_lower_pc is not None or self.distance_upper_pc is not None:
            return "PRESENT_UNCLASSIFIED"
        return "NOT_AVAILABLE"

    @property
    def distance_error_lower_pc(self) -> float | None:
        """Lower-side distance deviation from the central estimate."""

        if self.distance_pc is None or self.distance_lower_pc is None:
            return None
        if not all(_finite(value) for value in (self.distance_pc, self.distance_lower_pc)):
            return None
        deviation = float(self.distance_pc) - float(self.distance_lower_pc)
        return deviation if deviation >= 0 else None

    @property
    def distance_error_upper_pc(self) -> float | None:
        """Upper-side distance deviation from the central estimate."""

        if self.distance_pc is None or self.distance_upper_pc is None:
            return None
        if not all(_finite(value) for value in (self.distance_pc, self.distance_upper_pc)):
            return None
        deviation = float(self.distance_upper_pc) - float(self.distance_pc)
        return deviation if deviation >= 0 else None

    @property
    def distance_error_pc(self) -> float | None:
        """Symmetric distance error when both bounds are available.

        For a one-sided diagnostic bound, return that side's deviation.  The
        structured estimate itself still remains non-strict unless a complete
        interval passed its quality gate.
        """

        lower_error = self.distance_error_lower_pc
        upper_error = self.distance_error_upper_pc
        if lower_error is not None and upper_error is not None:
            return 0.5 * (lower_error + upper_error)
        return lower_error if lower_error is not None else upper_error

    @property
    def error_status(self) -> str:
        """State whether ``error`` is complete, partial, or unavailable."""

        incomplete_flags = {
            "APPARENT_MAGNITUDE_ERROR_NOT_PROVIDED",
            "EXTINCTION_ERROR_NOT_PROVIDED",
            "DISTANCE_ERROR_NOT_PROVIDED",
            "DISTANCE_INTERVAL_REQUIRED",
            "PARALLAX_ERROR_REQUIRED",
            "PARALLAX_ZERO_POINT_ERROR_NOT_PROVIDED",
        }
        if incomplete_flags.intersection(self.flags):
            return "INCOMPLETE"
        return "AVAILABLE" if self.error is not None else "NOT_AVAILABLE"

    @property
    def is_strict(self) -> bool:
        """Whether the result is safe to consume as a strict numeric M.

        ``value`` is intentionally not the only gate: a value can be retained
        by a legacy payload while its provenance is incomplete.  A strict
        result needs a declared extinction semantics, a declared distance
        source, and a complete two-sided/derived distance interval.
        """

        return bool(
            self.value is not None
            and self.status in {"VALID", "VALID_MODEL_DISTANCE"}
            and _provenance_declared(self.distance_source)
            and _photometric_metadata_declared(self.extinction_system, self.extinction_band)
            and self.distance_interval_status in {"PROVIDED", "DERIVED_FROM_PARALLAX_ERROR"}
            and not {"EXTINCTION_SEMANTICS_REQUIRED", "DISTANCE_SOURCE_REQUIRED"}.intersection(self.flags)
        )

    @property
    def is_strict_with_uncertainty(self) -> bool:
        """Whether a strict numeric M also carries a complete error budget.

        ``is_strict`` answers whether the value has a declared physical
        interpretation and distance interval.  A model distance can satisfy
        that gate while the image magnitude or extinction uncertainty is
        absent, so this stricter property is the gate for quoting an error
        bar or ranking results by formal uncertainty.
        """

        return bool(self.is_strict and self.error_status == "AVAILABLE" and self.error is not None)


@dataclass(frozen=True, slots=True)
class SourcePhotometry:
    """单个检测源的完整星等链路和质量状态。

    这张表把 ``m_inst``、星表参考值、标定后的表观星等和绝对星等
    分开保存。没有匹配、没有颜色或视差质量不足时，数值保持 ``None``，
    并通过 ``status``/``flags`` 说明原因，而不是用一个看起来完整的
    数字掩盖缺失的物理条件。
    """

    detection_id: int
    source_id: str | None
    instrumental_magnitude: float | None
    instrumental_magnitude_error: float | None
    calibrated_magnitude: float | None
    calibrated_magnitude_error: float | None
    catalog_magnitude: float | None
    catalog_magnitude_error: float | None
    color: float | None
    color_name: str | None
    photometric_system: str | None
    photometric_band: str | None
    absolute_magnitude: AbsoluteMagnitudeEstimate | None
    status: str
    magnitude_source: str | None = None
    flags: tuple[str, ...] = ()
    photometric_residual_mag: float | None = None
    photometric_residual_limit_mag: float | None = None
    photometric_consistent: bool | None = None
    photometric_outlier_reason: str | None = None
    calibration_sample_role: str | None = None
    # Gaia DR3 GSP-Phot's model absolute magnitude is an external catalogue
    # datum.  Keep it separate from ``absolute_magnitude``, which is computed
    # here from the image-calibrated apparent magnitude, distance and
    # extinction.  The percentile bounds are retained as published rather
    # than collapsed into a symmetric sigma.
    catalog_mg_gspphot: float | None = None
    catalog_mg_gspphot_lower: float | None = None
    catalog_mg_gspphot_upper: float | None = None
    catalog_mg_gspphot_source: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "detection_id": self.detection_id,
            "source_id": self.source_id,
            "instrumental_magnitude": self.instrumental_magnitude,
            "instrumental_magnitude_error": self.instrumental_magnitude_error,
            "calibrated_magnitude": self.calibrated_magnitude,
            "calibrated_magnitude_error": self.calibrated_magnitude_error,
            "catalog_magnitude": self.catalog_magnitude,
            "catalog_magnitude_error": self.catalog_magnitude_error,
            "color": self.color,
            "color_name": self.color_name,
            "photometric_system": self.photometric_system,
            "photometric_band": self.photometric_band,
            "magnitude_source": self.magnitude_source,
            "absolute_magnitude": self.absolute_magnitude.as_dict() if self.absolute_magnitude else None,
            "status": self.status,
            "flags": list(self.flags),
            "photometric_residual_mag": self.photometric_residual_mag,
            "photometric_residual_limit_mag": self.photometric_residual_limit_mag,
            "photometric_consistent": self.photometric_consistent,
            "photometric_outlier_reason": self.photometric_outlier_reason,
            "calibration_sample_role": self.calibration_sample_role,
            "catalog_mg_gspphot": self.catalog_mg_gspphot,
            "catalog_mg_gspphot_lower": self.catalog_mg_gspphot_lower,
            "catalog_mg_gspphot_upper": self.catalog_mg_gspphot_upper,
            "catalog_mg_gspphot_source": self.catalog_mg_gspphot_source,
        }


def instrumental_magnitude(flux: float, *, exposure_s: float = 1.0) -> float | None:
    """将正通量率转换为未加零点的仪器星等。

    ``flux`` 使用 ADU，``exposure_s`` 用于把积分通量归一化为 ADU/s。
    默认 1 秒保留旧版 API 的同图相对排序语义。
    """

    if not math.isfinite(flux) or flux <= 0 or not math.isfinite(exposure_s) or exposure_s <= 0:
        return None
    return -2.5 * math.log10(flux / exposure_s)


def calibrated_apparent_magnitude(
    instrumental: float,
    *,
    zero_point: float,
    color: float | None = None,
    color_coefficient: float = 0.0,
    extinction_mag: float = 0.0,
) -> float | None:
    """把仪器星等转换为一个已声明波段的表观星等。

    ``zero_point``、颜色系数和消光必须来自同一设备/观测条件下的
    标准星拟合；本函数只执行公式，不声称输入已经完成科学标定。
    ``color`` 缺失时只允许颜色项系数为 0，避免静默把缺失颜色当成 0。
    返回值仍是表观星等，绝不是绝对星等。
    """

    if not math.isfinite(float(instrumental)):
        return None
    if not math.isfinite(float(zero_point)):
        raise ValueError("zero_point must be finite")
    if not math.isfinite(float(color_coefficient)):
        raise ValueError("color_coefficient must be finite")
    if not math.isfinite(float(extinction_mag)):
        raise ValueError("extinction_mag must be finite")
    if color is None:
        if color_coefficient != 0.0:
            raise ValueError("color is required when color_coefficient is non-zero")
        color_term = 0.0
    else:
        if not math.isfinite(float(color)):
            raise ValueError("color must be finite")
        color_term = float(color_coefficient) * float(color)
    return float(instrumental) + float(zero_point) + color_term + float(extinction_mag)


def _invalid_calibration(
    *,
    photometric_system: str,
    photometric_band: str,
    color_name: str | None,
    color_order: int,
    calibrator_count: int,
    flags: tuple[str, ...],
    status: str,
    catalog_filter_counts: tuple[tuple[str, int], ...] = (),
) -> PhotometricCalibration:
    return PhotometricCalibration(
        photometric_system=photometric_system,
        photometric_band=photometric_band,
        color_name=color_name,
        color_order=color_order,
        coefficients=(),
        calibrator_count=calibrator_count,
        inlier_count=0,
        validation_count=0,
        fit_rms_mag=None,
        validation_rms_mag=None,
        residual_mad_mag=None,
        color_min=None,
        color_max=None,
        status=status,
        flags=flags,
        catalog_filter_counts=catalog_filter_counts,
    )


def _catalog_quality_reasons(
    source: CatalogSource,
    *,
    min_catalog_flux_snr: float | None,
    max_catalog_ruwe: float | None,
    min_catalog_visibility_periods: int | None,
    reject_duplicated_sources: bool,
    reject_variable_sources: bool,
) -> tuple[str, ...]:
    """返回不宜进入光度零点拟合的目录质量原因。

    这些条件只针对明确提供了相应 Gaia 质量字段的行；字段缺失不会被
    静默解释为失败，从而保持旧的手写星表兼容性。真实 Gaia 行则会在
    进入零点拟合前排除明确的重复源、变量源和低质量天体。`BP/RP`
    excess factor 暂不使用固定阈值硬切，因官方建议结合颜色使用校正
    后的 excess factor；它仍由 `CatalogSource` 保留供后续审计。
    """

    reasons: list[str] = []
    if reject_duplicated_sources and source.duplicated_source is True:
        reasons.append("DUPLICATED_SOURCE")
    if reject_variable_sources and source.phot_variable_flag is not None:
        if str(source.phot_variable_flag).strip().upper() == "VARIABLE":
            reasons.append("VARIABLE_SOURCE")
    if (
        min_catalog_flux_snr is not None
        and source.phot_g_mean_flux_over_error is not None
        and source.phot_g_mean_flux_over_error < min_catalog_flux_snr
    ):
        reasons.append("LOW_CATALOG_FLUX_SNR")
    if max_catalog_ruwe is not None and source.ruwe is not None and source.ruwe > max_catalog_ruwe:
        reasons.append("HIGH_RUWE")
    if (
        min_catalog_visibility_periods is not None
        and source.visibility_periods_used is not None
        and source.visibility_periods_used < min_catalog_visibility_periods
    ):
        reasons.append("LOW_VISIBILITY_PERIODS")
    return tuple(reasons)


def _sorted_filter_counts(counts: dict[str, int]) -> tuple[tuple[str, int], ...]:
    return tuple(sorted((reason, int(count)) for reason, count in counts.items() if count > 0))


def _calibration_design(colors: np.ndarray, color_order: int) -> np.ndarray:
    columns = [np.ones(colors.size, dtype=np.float64)]
    if color_order >= 1:
        columns.append(colors)
    if color_order >= 2:
        columns.append(colors**2)
    return np.column_stack(columns)


def _fit_calibration_coefficients(
    instrumental: np.ndarray,
    catalog_magnitude: np.ndarray,
    colors: np.ndarray,
    color_order: int,
    errors: np.ndarray | None = None,
) -> np.ndarray | None:
    design = _calibration_design(colors, color_order)
    if design.shape[0] < design.shape[1] or np.linalg.matrix_rank(design) < design.shape[1]:
        return None
    target = catalog_magnitude - instrumental
    if errors is not None:
        # A small floor prevents a quoted zero catalog error from becoming an
        # infinite weight.  The floor is only a numerical guard; the returned
        # RMS/MAD remain in magnitudes and are not replaced by a formal
        # covariance estimate.
        safe_errors = np.maximum(np.asarray(errors, dtype=np.float64), 0.005)
        weights = 1.0 / safe_errors
        design = design * weights[:, None]
        target = target * weights
    coefficients, _residuals, _rank, _singular_values = np.linalg.lstsq(design, target, rcond=None)
    return np.asarray(coefficients, dtype=np.float64)


def fit_photometric_calibration(
    matches: Sequence[CatalogMatch],
    detections: Sequence[Detection],
    catalog: Sequence[CatalogSource],
    *,
    exposure_s: float = 1.0,
    photometric_system: str = "unknown",
    photometric_band: str = "unknown",
    color_name: str | None = "color",
    color_order: int = 1,
    min_snr: float = 5.0,
    min_calibrators: int = 3,
    max_match_residual_px: float | None = 3.0,
    min_catalog_flux_snr: float | None = 20.0,
    max_catalog_ruwe: float | None = 1.4,
    min_catalog_visibility_periods: int | None = 8,
    reject_duplicated_sources: bool = True,
    reject_variable_sources: bool = True,
    sigma_clip: float = 3.5,
    max_iterations: int = 5,
    validation_fraction: float = 0.2,
    max_fit_rms_mag: float | None = DEFAULT_MAX_FIT_RMS_MAG,
    max_validation_rms_mag: float | None = DEFAULT_MAX_VALIDATION_RMS_MAG,
    max_residual_mad_mag: float | None = DEFAULT_MAX_RESIDUAL_MAD_MAG,
    max_validation_bias_mag: float | None = DEFAULT_MAX_VALIDATION_BIAS_MAG,
    min_validation_count: int = DEFAULT_MIN_VALIDATION_COUNT,
    max_source_residual_mag: float | None = DEFAULT_MAX_SOURCE_RESIDUAL_MAG,
) -> PhotometricCalibration:
    """从多颗匹配参考星拟合经验光度零点和颜色项。

    ``matches`` 只提供身份和位置；通量、颜色、波段、视差等属性分别
    从 ``detections`` 和 ``catalog`` 读取。函数不会把单颗星的差值当成
    标定结果，而是进行确定性的留出验证和 MAD 迭代剔除。

    ``color_order=0`` 可用于只有零点、没有颜色资料的诊断场景；正式的
    宽谱标定建议至少提供颜色项，并在结果中报告颜色范围。

    对真实 Gaia 行，默认还会排除明确标记为重复源/变量源、低 G 通量
    SNR、高 RUWE 或过少 visibility periods 的参考星。质量字段缺失时
    保留该行，但会由星表审计区分“未知”与“通过”。
    """

    if not math.isfinite(float(exposure_s)) or exposure_s <= 0:
        raise ValueError("exposure_s must be positive")
    if not math.isfinite(float(min_snr)) or min_snr <= 0:
        raise ValueError("min_snr must be positive")
    if color_order not in {0, 1, 2}:
        raise ValueError("color_order must be 0, 1 or 2")
    if min_calibrators < color_order + 2:
        raise ValueError("min_calibrators must leave enough degrees of freedom")
    if sigma_clip <= 0 or max_iterations < 1:
        raise ValueError("sigma_clip must be positive and max_iterations must be positive")
    if not 0.0 <= validation_fraction < 0.5:
        raise ValueError("validation_fraction must be in [0, 0.5)")
    for name, value in (
        ("max_fit_rms_mag", max_fit_rms_mag),
        ("max_validation_rms_mag", max_validation_rms_mag),
        ("max_residual_mad_mag", max_residual_mad_mag),
        ("max_validation_bias_mag", max_validation_bias_mag),
        ("max_source_residual_mag", max_source_residual_mag),
    ):
        if value is not None and (
            not math.isfinite(float(value)) or float(value) <= 0
        ):
            raise ValueError(f"{name} must be positive or None")
    if (
        isinstance(min_validation_count, bool)
        or not isinstance(min_validation_count, int)
        or min_validation_count < 0
    ):
        raise ValueError("min_validation_count must be a non-negative integer")
    if min_catalog_flux_snr is not None and (
        not math.isfinite(float(min_catalog_flux_snr)) or float(min_catalog_flux_snr) <= 0
    ):
        raise ValueError("min_catalog_flux_snr must be positive or None")
    if max_catalog_ruwe is not None and (
        not math.isfinite(float(max_catalog_ruwe)) or float(max_catalog_ruwe) <= 0
    ):
        raise ValueError("max_catalog_ruwe must be positive or None")
    if min_catalog_visibility_periods is not None and (
        isinstance(min_catalog_visibility_periods, bool)
        or not isinstance(min_catalog_visibility_periods, int)
        or min_catalog_visibility_periods < 0
    ):
        raise ValueError("min_catalog_visibility_periods must be a non-negative integer or None")
    if not isinstance(reject_duplicated_sources, bool) or not isinstance(reject_variable_sources, bool):
        raise ValueError("catalog source rejection options must be boolean")

    # A fit with an omitted system or passband is still just a numerical
    # offset.  Refuse to call it a standard-band calibration; this is the
    # boundary that prevents a Gaia G value from being relabelled as V.
    if not _photometric_metadata_declared(photometric_system, photometric_band):
        return _invalid_calibration(
            photometric_system=photometric_system,
            photometric_band=photometric_band,
            color_name=color_name,
            color_order=color_order,
            calibrator_count=0,
            flags=("PHOTOMETRIC_METADATA_REQUIRED", "INSUFFICIENT_CALIBRATORS"),
            # Keep the historical insufficient status for an unlabelled
            # compatibility call, while guaranteeing that no coefficients
            # can be applied.
            status="INSUFFICIENT_CALIBRATORS",
        )

    detections_by_id = {int(source.detection_id): source for source in detections}
    catalog_by_id = {str(source.source_id): source for source in catalog}
    bad_flags = {"EDGE", "MASKED", "SATURATED"}
    rows: list[tuple[str, float, float, float, float]] = []
    catalog_filter_counts: dict[str, int] = {}
    missing_color = False
    metadata_mismatch = False
    color_metadata_mismatch = False
    seen_detection_ids: set[int] = set()
    seen_source_ids: set[str] = set()
    duplicate_match = False
    for match in matches:
        match_detection_id = int(match.detection_id)
        match_source_id = str(match.source_id)
        if match_detection_id in seen_detection_ids or match_source_id in seen_source_ids:
            duplicate_match = True
            continue
        seen_detection_ids.add(match_detection_id)
        seen_source_ids.add(match_source_id)
        detection = detections_by_id.get(match_detection_id)
        source = catalog_by_id.get(match_source_id)
        if detection is None or source is None:
            continue
        quality_reasons = _catalog_quality_reasons(
            source,
            min_catalog_flux_snr=min_catalog_flux_snr,
            max_catalog_ruwe=max_catalog_ruwe,
            min_catalog_visibility_periods=min_catalog_visibility_periods,
            reject_duplicated_sources=reject_duplicated_sources,
            reject_variable_sources=reject_variable_sources,
        )
        if quality_reasons:
            for reason in quality_reasons:
                catalog_filter_counts[reason] = catalog_filter_counts.get(reason, 0) + 1
            continue
        if not _photometric_metadata_matches(
            photometric_system,
            photometric_band,
            source.photometric_system,
            source.photometric_band,
        ):
            metadata_mismatch = True
            continue
        if color_order and (
            _normalise_photometric_label(color_name) is None
            or _normalise_photometric_label(source.color_name) != _normalise_photometric_label(color_name)
        ):
            color_metadata_mismatch = True
            continue
        signal_snr = detection.flux_snr if detection.flux_snr is not None else detection.snr
        if (
            not detection.quality_passed
            or not math.isfinite(float(signal_snr))
            or signal_snr < min_snr
            or not math.isfinite(float(detection.flux))
            or detection.flux <= 0
            or bad_flags.intersection(detection.flags)
        ):
            continue
        if max_match_residual_px is not None and (
            not math.isfinite(float(match.residual_px)) or match.residual_px > max_match_residual_px
        ):
            continue
        if source.magnitude is None and match.catalog_magnitude is None:
            continue
        catalog_magnitude = source.magnitude if source.magnitude is not None else match.catalog_magnitude
        if catalog_magnitude is None or not math.isfinite(float(catalog_magnitude)):
            continue
        if color_order:
            if source.color is None or not math.isfinite(float(source.color)):
                missing_color = True
                continue
            color = float(source.color)
        else:
            color = 0.0
        instrumental = instrumental_magnitude(float(detection.flux), exposure_s=exposure_s)
        if instrumental is None:
            continue
        measurement_error = 1.0857362047581296 / float(signal_snr)
        catalog_error = source.magnitude_error
        if catalog_error is None:
            catalog_error = match.catalog_magnitude_error
        total_error = math.sqrt(measurement_error**2 + float(catalog_error or 0.0) ** 2)
        rows.append((str(source.source_id), instrumental, float(catalog_magnitude), color, total_error))

    rows.sort(key=lambda row: row[0])
    if (metadata_mismatch or color_metadata_mismatch) and len(rows) >= min_calibrators:
        return _invalid_calibration(
            photometric_system=photometric_system,
            photometric_band=photometric_band,
            color_name=color_name,
            color_order=color_order,
            calibrator_count=len(rows),
            flags=(
                ("PHOTOMETRIC_METADATA_MISMATCH",) if metadata_mismatch else ()
            )
            + (("COLOR_METADATA_MISMATCH",) if color_metadata_mismatch else ()),
            status=("PHOTOMETRIC_METADATA_MISMATCH" if metadata_mismatch else "COLOR_METADATA_MISMATCH"),
            catalog_filter_counts=_sorted_filter_counts(catalog_filter_counts),
        )
    if duplicate_match and len(rows) >= min_calibrators:
        return _invalid_calibration(
            photometric_system=photometric_system,
            photometric_band=photometric_band,
            color_name=color_name,
            color_order=color_order,
            calibrator_count=len(rows),
            flags=("DUPLICATE_MATCHES",),
            status="DUPLICATE_MATCHES",
            catalog_filter_counts=_sorted_filter_counts(catalog_filter_counts),
        )
    if len(rows) < min_calibrators:
        flags = ("INSUFFICIENT_CALIBRATORS",)
        if missing_color and color_order:
            flags += ("MISSING_COLOR",)
        if metadata_mismatch:
            flags += ("PHOTOMETRIC_METADATA_MISMATCH",)
        if color_metadata_mismatch:
            flags += ("COLOR_METADATA_MISMATCH",)
        if duplicate_match:
            flags += ("DUPLICATE_MATCHES",)
        return _invalid_calibration(
            photometric_system=photometric_system,
            photometric_band=photometric_band,
            color_name=color_name,
            color_order=color_order,
            calibrator_count=len(rows),
            flags=flags,
            status="INSUFFICIENT_CALIBRATORS",
            catalog_filter_counts=_sorted_filter_counts(catalog_filter_counts),
        )

    row_array = np.asarray(rows, dtype=object)
    values_instrumental = np.asarray(row_array[:, 1], dtype=np.float64)
    values_catalog = np.asarray(row_array[:, 2], dtype=np.float64)
    values_color = np.asarray(row_array[:, 3], dtype=np.float64)
    values_error = np.asarray(row_array[:, 4], dtype=np.float64)

    validation_indices: set[int] = set()
    minimum_for_holdout = max(
        min_calibrators * 2,
        min_calibrators + max(1, min_validation_count),
    )
    if (
        validation_fraction > 0
        and len(rows) >= minimum_for_holdout
        and len(rows) - min_calibrators >= max(1, min_validation_count)
    ):
        validation_count = max(
            max(1, min_validation_count),
            int(round(len(rows) * validation_fraction)),
        )
        validation_count = min(validation_count, len(rows) - min_calibrators)
        evenly_spaced = np.linspace(0, len(rows) - 1, validation_count, dtype=int)
        validation_indices = {int(index) for index in evenly_spaced}

    training_indices = np.asarray(
        [index for index in range(len(rows)) if index not in validation_indices],
        dtype=np.int64,
    )
    validation_array = np.asarray(sorted(validation_indices), dtype=np.int64)
    if training_indices.size < min_calibrators:
        training_indices = np.arange(len(rows), dtype=np.int64)
        validation_array = np.asarray((), dtype=np.int64)

    train_instrumental = values_instrumental[training_indices]
    train_catalog = values_catalog[training_indices]
    train_color = values_color[training_indices]
    train_error = values_error[training_indices]
    active = np.ones(training_indices.size, dtype=bool)
    clipped = False
    coefficients: np.ndarray | None = None
    for _iteration in range(max_iterations):
        coefficients = _fit_calibration_coefficients(
            train_instrumental[active],
            train_catalog[active],
            train_color[active],
            color_order,
            train_error[active],
        )
        if coefficients is None:
            return _invalid_calibration(
                photometric_system=photometric_system,
                photometric_band=photometric_band,
                color_name=color_name,
                color_order=color_order,
                calibrator_count=len(rows),
                flags=("INSUFFICIENT_COLOR_RANGE",),
                status="INSUFFICIENT_COLOR_RANGE",
                catalog_filter_counts=_sorted_filter_counts(catalog_filter_counts),
            )
        design = _calibration_design(train_color, color_order)
        residuals = train_catalog - train_instrumental - design @ coefficients
        center = float(np.median(residuals[active]))
        mad = float(np.median(np.abs(residuals[active] - center)))
        scale = max(1.4826 * mad, 1e-6)
        new_active = np.abs(residuals - center) <= float(sigma_clip) * scale
        if int(new_active.sum()) < min_calibrators:
            break
        if np.array_equal(new_active, active):
            break
        clipped = clipped or not np.array_equal(new_active, active)
        active = new_active

    coefficients = _fit_calibration_coefficients(
        train_instrumental[active],
        train_catalog[active],
        train_color[active],
        color_order,
        train_error[active],
    )
    if coefficients is None or int(active.sum()) < min_calibrators:
        return _invalid_calibration(
            photometric_system=photometric_system,
            photometric_band=photometric_band,
            color_name=color_name,
            color_order=color_order,
            calibrator_count=len(rows),
            flags=("INSUFFICIENT_INLIERS",),
            status="INSUFFICIENT_INLIERS",
            catalog_filter_counts=_sorted_filter_counts(catalog_filter_counts),
        )

    train_design = _calibration_design(train_color[active], color_order)
    train_residuals = train_catalog[active] - train_instrumental[active] - train_design @ coefficients
    fit_rms = float(np.sqrt(np.mean(train_residuals**2)))
    residual_mad = float(1.4826 * np.median(np.abs(train_residuals - np.median(train_residuals))))
    validation_rms = None
    validation_bias = None
    validation_residuals = np.asarray((), dtype=np.float64)
    if validation_array.size:
        validation_design = _calibration_design(values_color[validation_array], color_order)
        validation_residuals = (
            values_catalog[validation_array]
            - values_instrumental[validation_array]
            - validation_design @ coefficients
        )
        validation_rms = float(np.sqrt(np.mean(validation_residuals**2)))
        validation_bias = float(np.mean(validation_residuals))

    flags: list[str] = []
    if clipped:
        flags.append("OUTLIERS_CLIPPED")
    if not validation_array.size:
        flags.append("NO_HOLDOUT_VALIDATION")
    gate_failures: list[str] = []
    if max_fit_rms_mag is not None and fit_rms > float(max_fit_rms_mag):
        gate_failures.append("FIT_RMS_EXCEEDS_LIMIT")
    if max_residual_mad_mag is not None and residual_mad > float(max_residual_mad_mag):
        gate_failures.append("RESIDUAL_MAD_EXCEEDS_LIMIT")
    if validation_fraction > 0 and min_validation_count > 0:
        if validation_array.size < min_validation_count:
            gate_failures.append("INSUFFICIENT_HOLDOUT_VALIDATION")
        else:
            if (
                max_validation_rms_mag is not None
                and validation_rms is not None
                and validation_rms > float(max_validation_rms_mag)
            ):
                gate_failures.append("VALIDATION_RMS_EXCEEDS_LIMIT")
            if (
                max_validation_bias_mag is not None
                and validation_bias is not None
                and abs(validation_bias) > float(max_validation_bias_mag)
            ):
                gate_failures.append("VALIDATION_BIAS_EXCEEDS_LIMIT")
    if gate_failures:
        flags.extend(gate_failures)
        status = "PHOTOMETRY_REJECTED_QUALITY_GATE"
    else:
        status = "VALID" if validation_array.size else "VALID_NO_HOLDOUT"
    active_by_row_index = {
        int(row_index): bool(active[position])
        for position, row_index in enumerate(training_indices)
    }
    calibration_sample_roles = tuple(
        (
            str(row[0]),
            (
                "validation"
                if index in validation_indices
                else (
                    "training_inlier"
                    if active_by_row_index.get(index, False)
                    else "training_outlier"
                )
            ),
        )
        for index, row in enumerate(rows)
    )
    return PhotometricCalibration(
        photometric_system=photometric_system,
        photometric_band=photometric_band,
        color_name=color_name,
        color_order=color_order,
        coefficients=tuple(float(value) for value in coefficients),
        calibrator_count=len(rows),
        inlier_count=int(active.sum()),
        validation_count=int(validation_array.size),
        fit_rms_mag=fit_rms,
        validation_rms_mag=validation_rms,
        residual_mad_mag=residual_mad,
        # The holdout rows are part of the observed calibration domain too;
        # using only the training subset would falsely classify a valid
        # holdout-colour source as an extrapolation at application time.
        color_min=float(values_color.min()) if color_order else None,
        color_max=float(values_color.max()) if color_order else None,
        status=status,
        flags=tuple(flags),
        catalog_filter_counts=_sorted_filter_counts(catalog_filter_counts),
        validation_bias_mag=validation_bias,
        max_fit_rms_mag=max_fit_rms_mag,
        max_validation_rms_mag=max_validation_rms_mag,
        max_residual_mad_mag=max_residual_mad_mag,
        max_validation_bias_mag=max_validation_bias_mag,
        min_validation_count=min_validation_count,
        max_source_residual_mag=max_source_residual_mag,
        calibration_sample_roles=calibration_sample_roles,
    )


def calibrated_magnitude_error(
    instrumental_error: float | None,
    *,
    calibration: PhotometricCalibration,
    color_error: float | None = None,
) -> float | None:
    """合并仪器测光、颜色项和标定残差的近似误差。"""

    if instrumental_error is None or not math.isfinite(float(instrumental_error)):
        return None
    variance = float(instrumental_error) ** 2
    if calibration.residual_mad_mag is not None:
        variance += float(calibration.residual_mad_mag) ** 2
    if calibration.color_order and color_error is not None and calibration.color_coefficient is not None:
        if math.isfinite(float(color_error)):
            variance += (float(calibration.color_coefficient) * float(color_error)) ** 2
    return math.sqrt(max(variance, 0.0))


def absolute_magnitude_from_apparent(
    apparent_magnitude: float,
    *,
    distance_pc: float,
    extinction_mag: float = 0.0,
) -> float | None:
    """由表观星等、距离和消光计算绝对星等。

    定义为 ``M = m - 5 log10(d / 10 pc) - A``。这里的 ``m`` 必须已经
    属于明确的标准波段（例如 V），而不是 ``m_inst``。距离可以来自
    可靠视差或独立测量；检测到多少颗星不能提供这个距离。这个旧的
    浮点接口只做兼容性算术，不携带质量/来源状态；需要严格语义时使用
    ``absolute_magnitude_estimate_from_distance`` 或其视差版本。
    """

    if not math.isfinite(float(apparent_magnitude)):
        return None
    if not math.isfinite(float(distance_pc)) or distance_pc <= 0:
        raise ValueError("distance_pc must be finite and positive")
    if not math.isfinite(float(extinction_mag)):
        raise ValueError("extinction_mag must be finite")
    return float(apparent_magnitude) - 5.0 * math.log10(float(distance_pc) / 10.0) - float(extinction_mag)


def absolute_magnitude_from_parallax(
    apparent_magnitude: float,
    *,
    parallax_mas: float,
    extinction_mag: float = 0.0,
) -> float | None:
    """由毫角秒视差换算绝对星等。

    仅接受正视差；负视差或零视差不能被当作距离使用，应由调用方保留
    为“绝对星等不可计算”而不是制造一个数值。这个旧的浮点接口不做
    视差质量或消光来源审计；严格路径使用结构化 estimate 接口。
    """

    if not math.isfinite(float(parallax_mas)) or parallax_mas <= 0:
        raise ValueError("parallax_mas must be finite and positive")
    distance_pc = 1000.0 / float(parallax_mas)
    return absolute_magnitude_from_apparent(
        apparent_magnitude,
        distance_pc=distance_pc,
        extinction_mag=extinction_mag,
    )


def _distance_interval_from_parallax(
    corrected_parallax_mas: float,
    parallax_error_mas: float,
) -> tuple[float | None, float | None]:
    """Convert a positive parallax +/- error into a finite distance interval."""

    lower_parallax = float(corrected_parallax_mas) + float(parallax_error_mas)
    upper_parallax = float(corrected_parallax_mas) - float(parallax_error_mas)
    if lower_parallax <= 0 or upper_parallax <= 0:
        return None, None
    return 1000.0 / lower_parallax, 1000.0 / upper_parallax


def absolute_magnitude_estimate_from_parallax(
    apparent_magnitude: float,
    *,
    parallax_mas: float,
    parallax_error_mas: float | None = None,
    extinction_mag: float | None = None,
    apparent_magnitude_error: float | None = None,
    extinction_error_mag: float | None = None,
    parallax_zero_point_mas: float = 0.0,
    max_fractional_parallax_error: float = 0.2,
    extinction_band: str | None = None,
    extinction_system: str | None = None,
    extinction_source: str | None = None,
) -> AbsoluteMagnitudeEstimate:
    """在质量门控下由表观星等和视差估计绝对星等。

    这是低视差误差场景下的可解释近似，不是对低信噪比视差做简单倒数。
    当视差误差超过 ``max_fractional_parallax_error`` 时，函数拒绝生成
    一个看似精确的绝对星等，调用方应改用带先验的距离后验。结构化
    结果还要求消光波段/系统已声明；仅有数值 A 不能证明它属于当前表观
    星等的波段。正视差和误差会派生一个有限的距离区间并写入结果。
    """

    if not math.isfinite(float(apparent_magnitude)):
        raise ValueError("apparent_magnitude must be finite")
    if not math.isfinite(float(parallax_mas)):
        raise ValueError("parallax_mas must be finite")
    if not math.isfinite(float(parallax_zero_point_mas)):
        raise ValueError("parallax_zero_point_mas must be finite")
    if not math.isfinite(float(max_fractional_parallax_error)) or max_fractional_parallax_error <= 0:
        raise ValueError("max_fractional_parallax_error must be positive")
    for name, value in (
        ("apparent_magnitude_error", apparent_magnitude_error),
        ("extinction_error_mag", extinction_error_mag),
    ):
        if value is not None and (not math.isfinite(float(value)) or value < 0):
            raise ValueError(f"{name} must be finite and non-negative")
    if extinction_mag is not None:
        if not math.isfinite(float(extinction_mag)):
            raise ValueError("extinction_mag must be finite when provided")
        if float(extinction_mag) < 0:
            return _absolute_unavailable(
                corrected_parallax_mas=None,
                distance_pc=None,
                extinction_mag=extinction_mag,
                status="INVALID_EXTINCTION",
                flags=("NEGATIVE_EXTINCTION",),
                extinction_band=extinction_band,
                extinction_system=extinction_system,
                extinction_source=extinction_source,
            )
    corrected_parallax = float(parallax_mas) - float(parallax_zero_point_mas)
    if corrected_parallax <= 0:
        return _absolute_unavailable(
            corrected_parallax_mas=corrected_parallax,
            distance_pc=None,
            extinction_mag=extinction_mag,
            status="INVALID_PARALLAX",
            flags=("NON_POSITIVE_CORRECTED_PARALLAX",),
            extinction_band=extinction_band,
            extinction_system=extinction_system,
            extinction_source=extinction_source,
        )

    distance_pc = 1000.0 / corrected_parallax
    if parallax_error_mas is None:
        flags = ["PARALLAX_ERROR_REQUIRED", "DISTANCE_ERROR_NOT_PROVIDED"]
        if extinction_mag is None:
            flags.append("EXTINCTION_NOT_PROVIDED")
        return _absolute_unavailable(
            corrected_parallax_mas=corrected_parallax,
            distance_pc=distance_pc,
            extinction_mag=extinction_mag,
            status="NO_PARALLAX_ERROR",
            flags=tuple(flags),
            extinction_band=extinction_band,
            extinction_system=extinction_system,
            extinction_source=extinction_source,
        )
    if not math.isfinite(float(parallax_error_mas)) or parallax_error_mas < 0:
        raise ValueError("parallax_error_mas must be finite and non-negative")
    parallax_error = float(parallax_error_mas)
    fractional_error = parallax_error / corrected_parallax
    distance_lower_pc, distance_upper_pc = _distance_interval_from_parallax(
        corrected_parallax,
        parallax_error,
    )
    distance_interval_flags = (
        ("DISTANCE_INTERVAL_DERIVED_FROM_PARALLAX_ERROR",)
        if distance_lower_pc is not None and distance_upper_pc is not None
        else ()
    )
    if fractional_error > max_fractional_parallax_error or distance_upper_pc is None:
        flags = ["USE_DISTANCE_POSTERIOR"]
        if extinction_mag is None:
            flags.append("EXTINCTION_NOT_PROVIDED")
        return _absolute_unavailable(
            corrected_parallax_mas=corrected_parallax,
            distance_pc=distance_pc,
            extinction_mag=extinction_mag,
            status="LOW_PARALLAX_SNR",
            flags=tuple(flags) + distance_interval_flags,
            distance_lower_pc=distance_lower_pc,
            distance_upper_pc=distance_upper_pc,
            extinction_band=extinction_band,
            extinction_system=extinction_system,
            extinction_source=extinction_source,
        )

    # A distance estimate without an extinction estimate is not a strict
    # absolute magnitude.  Keep the diagnostic distance, but never retain a
    # numeric M obtained by silently substituting A=0.
    if extinction_mag is None:
        return _absolute_unavailable(
            corrected_parallax_mas=corrected_parallax,
            distance_pc=distance_pc,
            extinction_mag=None,
            status="VALID_NO_EXTINCTION",
            flags=("EXTINCTION_NOT_PROVIDED",) + distance_interval_flags,
            distance_lower_pc=distance_lower_pc,
            distance_upper_pc=distance_upper_pc,
            extinction_band=extinction_band,
            extinction_system=extinction_system,
            extinction_source=extinction_source,
        )

    if not _photometric_metadata_declared(extinction_system, extinction_band):
        return _absolute_unavailable(
            corrected_parallax_mas=corrected_parallax,
            distance_pc=distance_pc,
            extinction_mag=extinction_mag,
            status="EXTINCTION_SEMANTICS_REQUIRED",
            flags=("EXTINCTION_SEMANTICS_REQUIRED",) + distance_interval_flags,
            distance_lower_pc=distance_lower_pc,
            distance_upper_pc=distance_upper_pc,
            extinction_band=extinction_band,
            extinction_system=extinction_system,
            extinction_source=extinction_source,
        )

    value = float(apparent_magnitude) - 5.0 * math.log10(distance_pc / 10.0) - float(extinction_mag)
    flags: list[str] = list(distance_interval_flags)
    variance = (5.0 / math.log(10.0) * parallax_error / corrected_parallax) ** 2
    errors_complete = True
    if apparent_magnitude_error is None:
        flags.append("APPARENT_MAGNITUDE_ERROR_NOT_PROVIDED")
        errors_complete = False
    else:
        variance += float(apparent_magnitude_error) ** 2
    if extinction_error_mag is None:
        flags.append("EXTINCTION_ERROR_NOT_PROVIDED")
        errors_complete = False
    else:
        variance += float(extinction_error_mag) ** 2
    if parallax_zero_point_mas:
        flags.append("PARALLAX_ZERO_POINT_CORRECTED")
    return AbsoluteMagnitudeEstimate(
        value=value,
        error=math.sqrt(variance) if errors_complete and variance > 0 else None,
        distance_pc=distance_pc,
        corrected_parallax_mas=corrected_parallax,
        extinction_mag=extinction_mag,
        status="VALID" if extinction_mag is not None else "VALID_NO_EXTINCTION",
        flags=tuple(flags),
        distance_source="parallax",
        distance_lower_pc=distance_lower_pc,
        distance_upper_pc=distance_upper_pc,
        extinction_band=extinction_band,
        extinction_system=extinction_system,
        extinction_source=extinction_source,
    )


def absolute_magnitude_estimate_from_distance(
    apparent_magnitude: float,
    *,
    distance_pc: float,
    distance_lower_pc: float | None = None,
    distance_upper_pc: float | None = None,
    extinction_mag: float | None = None,
    apparent_magnitude_error: float | None = None,
    extinction_error_mag: float | None = None,
    distance_source: str | None = None,
    extinction_band: str | None = None,
    extinction_system: str | None = None,
    extinction_source: str | None = None,
) -> AbsoluteMagnitudeEstimate:
    """Use an external/model distance posterior to estimate absolute magnitude.

    This path is intentionally separate from the parallax path.  Gaia
    GSP-Phot, for example, publishes a model distance and 16th/84th-percentile
    bounds; it is useful when a direct parallax is absent or has poor S/N, but
    it must remain visibly model-based.  A declared ``distance_source``, a
    complete lower/upper interval, and declared extinction system/band are
    required before a numeric value is returned.  The central value is used
    only when the apparent band and extinction band are already declared by
    the caller. ``distance_lower_pc``/``distance_upper_pc`` are interpreted as
    an approximate 16th/84th interval and are propagated in log-distance
    space; a missing or one-sided interval remains a diagnostic-only result.
    """

    if not math.isfinite(float(apparent_magnitude)):
        raise ValueError("apparent_magnitude must be finite")
    if not math.isfinite(float(distance_pc)) or distance_pc <= 0:
        raise ValueError("distance_pc must be finite and positive")
    distance_value = float(distance_pc)
    lower = None
    upper = None
    for name, value in (
        ("distance_lower_pc", distance_lower_pc),
        ("distance_upper_pc", distance_upper_pc),
    ):
        if value is not None and (not math.isfinite(float(value)) or float(value) <= 0):
            raise ValueError(f"{name} must be finite and positive when provided")
    if distance_lower_pc is not None:
        lower = float(distance_lower_pc)
    if distance_upper_pc is not None:
        upper = float(distance_upper_pc)
    if lower is not None and upper is not None:
        if lower > upper or not lower <= distance_value <= upper:
            return _absolute_unavailable(
                corrected_parallax_mas=None,
                distance_pc=distance_value,
                extinction_mag=extinction_mag,
                status="INVALID_DISTANCE_INTERVAL",
                flags=("DISTANCE_INTERVAL_INVALID",),
                distance_source=distance_source,
                distance_lower_pc=lower,
                distance_upper_pc=upper,
                extinction_band=extinction_band,
                extinction_system=extinction_system,
                extinction_source=extinction_source,
            )
    elif (lower is not None and lower > distance_value) or (upper is not None and upper < distance_value):
        return _absolute_unavailable(
            corrected_parallax_mas=None,
            distance_pc=distance_value,
            extinction_mag=extinction_mag,
            status="INVALID_DISTANCE_INTERVAL",
            flags=("DISTANCE_INTERVAL_INVALID",),
            distance_source=distance_source,
            distance_lower_pc=lower,
            distance_upper_pc=upper,
            extinction_band=extinction_band,
            extinction_system=extinction_system,
            extinction_source=extinction_source,
        )
    if lower is not None and upper is not None:
        interval_flags = ("DISTANCE_INTERVAL_USED",)
    elif lower is not None or upper is not None:
        interval_flags = ("DISTANCE_INTERVAL_ONE_SIDED", "DISTANCE_INTERVAL_REQUIRED")
    else:
        interval_flags = ("DISTANCE_ERROR_NOT_PROVIDED", "DISTANCE_INTERVAL_REQUIRED")
    for name, value in (
        ("apparent_magnitude_error", apparent_magnitude_error),
        ("extinction_error_mag", extinction_error_mag),
    ):
        if value is not None and (not math.isfinite(float(value)) or float(value) < 0):
            raise ValueError(f"{name} must be finite and non-negative")
    if not _provenance_declared(distance_source):
        return _absolute_unavailable(
            corrected_parallax_mas=None,
            distance_pc=distance_value,
            extinction_mag=extinction_mag,
            status="MODEL_DISTANCE_SOURCE_REQUIRED",
            flags=("DISTANCE_SOURCE_REQUIRED", *interval_flags),
            distance_source=distance_source,
            distance_lower_pc=lower,
            distance_upper_pc=upper,
            extinction_band=extinction_band,
            extinction_system=extinction_system,
            extinction_source=extinction_source,
        )
    if extinction_mag is None:
        flags = ["DISTANCE_MODEL_USED", "EXTINCTION_NOT_PROVIDED", *interval_flags]
        return _absolute_unavailable(
            corrected_parallax_mas=None,
            distance_pc=distance_value,
            extinction_mag=None,
            status="MODEL_DISTANCE_NO_EXTINCTION",
            flags=tuple(flags),
            distance_source=distance_source,
            distance_lower_pc=lower,
            distance_upper_pc=upper,
            extinction_band=extinction_band,
            extinction_system=extinction_system,
            extinction_source=extinction_source,
        )
    if not math.isfinite(float(extinction_mag)):
        raise ValueError("extinction_mag must be finite when provided")
    if float(extinction_mag) < 0:
        return _absolute_unavailable(
            corrected_parallax_mas=None,
            distance_pc=distance_value,
            extinction_mag=float(extinction_mag),
            status="INVALID_EXTINCTION",
            flags=("NEGATIVE_EXTINCTION", "DISTANCE_MODEL_USED", *interval_flags),
            distance_source=distance_source,
            distance_lower_pc=lower,
            distance_upper_pc=upper,
            extinction_band=extinction_band,
            extinction_system=extinction_system,
            extinction_source=extinction_source,
        )
    if not _photometric_metadata_declared(extinction_system, extinction_band):
        return _absolute_unavailable(
            corrected_parallax_mas=None,
            distance_pc=distance_value,
            extinction_mag=float(extinction_mag),
            status="EXTINCTION_SEMANTICS_REQUIRED",
            flags=("DISTANCE_MODEL_USED", "EXTINCTION_SEMANTICS_REQUIRED", *interval_flags),
            distance_source=distance_source,
            distance_lower_pc=lower,
            distance_upper_pc=upper,
            extinction_band=extinction_band,
            extinction_system=extinction_system,
            extinction_source=extinction_source,
        )

    if lower is None or upper is None:
        return _absolute_unavailable(
            corrected_parallax_mas=None,
            distance_pc=distance_value,
            extinction_mag=float(extinction_mag),
            status="MODEL_DISTANCE_INTERVAL_REQUIRED",
            flags=("DISTANCE_MODEL_USED", *interval_flags),
            distance_source=distance_source,
            distance_lower_pc=lower,
            distance_upper_pc=upper,
            extinction_band=extinction_band,
            extinction_system=extinction_system,
            extinction_source=extinction_source,
        )

    value = float(apparent_magnitude) - 5.0 * math.log10(distance_value / 10.0) - float(extinction_mag)
    variance = 0.0
    flags = ["DISTANCE_MODEL_USED"]
    distance_log_error = 0.5 * math.log(upper / lower)
    flags.append("DISTANCE_INTERVAL_USED")
    variance += (5.0 / math.log(10.0) * distance_log_error) ** 2
    errors_complete = True
    if apparent_magnitude_error is None:
        flags.append("APPARENT_MAGNITUDE_ERROR_NOT_PROVIDED")
        errors_complete = False
    else:
        variance += float(apparent_magnitude_error) ** 2
    if extinction_error_mag is None:
        flags.append("EXTINCTION_ERROR_NOT_PROVIDED")
        errors_complete = False
    else:
        variance += float(extinction_error_mag) ** 2
    return AbsoluteMagnitudeEstimate(
        value=value,
        error=math.sqrt(variance) if errors_complete and variance > 0 else None,
        distance_pc=distance_value,
        corrected_parallax_mas=None,
        extinction_mag=float(extinction_mag),
        status="VALID_MODEL_DISTANCE",
        flags=tuple(flags),
        distance_source=distance_source,
        distance_lower_pc=lower,
        distance_upper_pc=upper,
        extinction_band=extinction_band,
        extinction_system=extinction_system,
        extinction_source=extinction_source,
    )


def _catalog_extinction_gate(source: CatalogSource) -> tuple[str, tuple[str, ...]] | None:
    """Return an explicit failure for catalog extinction with unsafe semantics.

    The numeric extinction is provenance-bearing input, so it must agree with
    the source's declared photometric system and band before it can enter the
    strict catalog-backed path. ``CatalogSource.extinction_compatibility`` is
    the single source of truth for aliases such as Gaia Vega/Gaia G.
    """

    if source.extinction_mag is None:
        return None
    compatibility = source.extinction_compatibility()
    if compatibility == "compatible":
        return None
    if compatibility == "mismatch":
        return "EXTINCTION_BAND_MISMATCH", ("EXTINCTION_BAND_MISMATCH",)
    # ``unknown`` and an unexpected value are both unsafe: accepting either
    # would turn an unlabelled A0/A_V/custom-band value into a guessed A_G.
    return "EXTINCTION_SEMANTICS_REQUIRED", ("EXTINCTION_SEMANTICS_REQUIRED",)


def absolute_magnitude_from_catalog(
    source: CatalogSource,
    *,
    apparent_magnitude: float | None = None,
    apparent_magnitude_error: float | None = None,
    required_photometric_system: str | None = None,
    required_photometric_band: str | None = None,
    parallax_zero_point_mas: float = 0.0,
    max_fractional_parallax_error: float = 0.2,
) -> AbsoluteMagnitudeEstimate:
    """用 ``CatalogSource`` 的距离元数据估计绝对星等。

    ``source.magnitude`` 是目录中声明的表观星等；当调用方已经用本机
    测光标定得到更合适的表观星等时，可通过 ``apparent_magnitude`` 覆盖。
    缺少视差、视差质量不够或没有消光资料时都保留明确状态，避免把
    ``1000 / parallax`` 的形式计算误当成可靠距离。提供
    ``required_photometric_system``/``required_photometric_band`` 时，源的
    目录元数据必须逐字（忽略大小写和空白差异）匹配；这条严格路径用于
    防止把 Gaia G 的距离信息挂到 V 波段的表观星等上。
    """

    extinction_band = getattr(source, "extinction_band", None)
    extinction_system = getattr(source, "extinction_system", None)
    extinction_source = getattr(source, "extinction_source", None)

    if (required_photometric_system is None) != (required_photometric_band is None):
        return _absolute_unavailable(
            corrected_parallax_mas=None,
            distance_pc=None,
            extinction_mag=source.extinction_mag,
            status="PHOTOMETRIC_METADATA_REQUIRED",
            flags=("PHOTOMETRIC_SYSTEM_AND_BAND_REQUIRED",),
            extinction_band=extinction_band,
            extinction_system=extinction_system,
            extinction_source=extinction_source,
        )
    if required_photometric_system is not None and not _photometric_metadata_matches(
        required_photometric_system,
        required_photometric_band,
        source.photometric_system,
        source.photometric_band,
    ):
        return _absolute_unavailable(
            corrected_parallax_mas=None,
            distance_pc=None,
            extinction_mag=source.extinction_mag,
            status="PHOTOMETRIC_METADATA_MISMATCH",
            flags=("CATALOG_BAND_DOES_NOT_MATCH_CALIBRATION",),
            extinction_band=extinction_band,
            extinction_system=extinction_system,
            extinction_source=extinction_source,
        )

    # A structured catalog result is an auditable M candidate even when the
    # caller uses the historical no-argument form.  Do not let that form
    # bypass the extinction gate: a numeric A with unknown semantics could be
    # A_G, A_V, A_0, or a custom-band value, and none may silently become A in
    # the source's apparent band.
    extinction_failure = _catalog_extinction_gate(source)
    if extinction_failure is not None:
        status, flags = extinction_failure
        return _absolute_unavailable(
            corrected_parallax_mas=None,
            distance_pc=None,
            extinction_mag=source.extinction_mag,
            status=status,
            flags=flags,
            extinction_band=extinction_band,
            extinction_system=extinction_system,
            extinction_source=extinction_source,
        )

    magnitude_from_catalog = apparent_magnitude is None
    magnitude = source.magnitude if magnitude_from_catalog else apparent_magnitude
    if magnitude is None:
        return AbsoluteMagnitudeEstimate(
            value=None,
            error=None,
            distance_pc=None,
            corrected_parallax_mas=None,
            extinction_mag=source.extinction_mag,
            status="NO_APPARENT_MAGNITUDE",
            flags=("NO_CATALOG_MAGNITUDE",),
            extinction_band=extinction_band,
            extinction_system=extinction_system,
            extinction_source=extinction_source,
        )
    magnitude_error = (
        source.magnitude_error
        if magnitude_from_catalog and apparent_magnitude_error is None
        else apparent_magnitude_error
    )

    def model_distance_estimate(*, fallback_from_parallax: bool = False) -> AbsoluteMagnitudeEstimate:
        if source.distance_pc is None:
            return AbsoluteMagnitudeEstimate(
                value=None,
                error=None,
                distance_pc=None,
                corrected_parallax_mas=None,
                extinction_mag=source.extinction_mag,
                status="NO_PARALLAX",
                flags=("NO_DISTANCE",),
                extinction_band=extinction_band,
                extinction_system=extinction_system,
                extinction_source=extinction_source,
            )
        if source.distance_source is None:
            return _absolute_unavailable(
                corrected_parallax_mas=None,
                distance_pc=source.distance_pc,
                extinction_mag=source.extinction_mag,
                status="MODEL_DISTANCE_SOURCE_REQUIRED",
                flags=("DISTANCE_SOURCE_REQUIRED",),
                distance_lower_pc=source.distance_lower_pc,
                distance_upper_pc=source.distance_upper_pc,
                extinction_band=extinction_band,
                extinction_system=extinction_system,
                extinction_source=extinction_source,
            )
        estimate = absolute_magnitude_estimate_from_distance(
            float(magnitude),
            distance_pc=source.distance_pc,
            distance_lower_pc=source.distance_lower_pc,
            distance_upper_pc=source.distance_upper_pc,
            extinction_mag=source.extinction_mag,
            apparent_magnitude_error=magnitude_error,
            extinction_error_mag=source.extinction_error_mag,
            distance_source=source.distance_source,
            extinction_band=extinction_band,
            extinction_system=extinction_system,
            extinction_source=extinction_source,
        )
        if fallback_from_parallax:
            estimate = replace(
                estimate,
                flags=tuple(
                    dict.fromkeys(("PARALLAX_QUALITY_FALLBACK_TO_MODEL_DISTANCE", *estimate.flags))
                ),
            )
        return estimate

    if source.parallax_mas is not None:
        parallax_estimate = absolute_magnitude_estimate_from_parallax(
            float(magnitude),
            parallax_mas=source.parallax_mas,
            parallax_error_mas=source.parallax_error_mas,
            extinction_mag=source.extinction_mag,
            apparent_magnitude_error=magnitude_error,
            extinction_error_mag=source.extinction_error_mag,
            parallax_zero_point_mas=parallax_zero_point_mas,
            max_fractional_parallax_error=max_fractional_parallax_error,
            extinction_band=extinction_band,
            extinction_system=extinction_system,
            extinction_source=extinction_source,
        )
        if parallax_estimate.value is not None or source.distance_pc is None:
            return replace(
                parallax_estimate,
                distance_source=parallax_estimate.distance_source or "parallax",
            )
        return model_distance_estimate(fallback_from_parallax=True)

    return model_distance_estimate()


def inferred_zero_point_for_comparison(
    instrumental: float,
    reference_apparent_magnitude: float,
) -> float | None:
    """返回把两个数强行对齐所需的零点差，仅用于诊断比较。

    这个值不是标定结果。若用户把外部示例 ``13.56`` 与本地 ``m_inst``
    比较，可用它展示两者差了一个约 19 mag 的零点/口径，但不能因此把
    示例数写回所有源。
    """

    if not _finite(instrumental) or not _finite(reference_apparent_magnitude):
        return None
    return float(reference_apparent_magnitude) - float(instrumental)


def find_faintest_source(
    sources: Sequence[Detection],
    *,
    min_snr: float = 5.0,
    zero_point: float | None = None,
    exposure_s: float = 1.0,
    photometric_calibration: PhotometricCalibration | None = None,
    source_colors: Mapping[int, float] | None = None,
    source_absolute_magnitudes: Mapping[int, AbsoluteMagnitudeEstimate] | None = None,
    source_photometry: Mapping[int, SourcePhotometry] | None = None,
) -> FaintestSource | None:
    """从检测源中选择最暗可信源。

    默认排除边缘、掩膜和饱和源，避免把不完整光斑或异常像素当成最暗星。
    `zero_point` 只在有外部标定时提供；它是加到仪器星等上的零点偏移。
    """

    if not math.isfinite(exposure_s) or exposure_s <= 0:
        raise ValueError("exposure_s must be positive")
    candidates = []
    for source in sources:
        signal_snr = source.flux_snr if source.flux_snr is not None else source.snr
        if not math.isfinite(signal_snr) or signal_snr < min_snr or source.flux <= 0:
            continue
        if not source.quality_passed:
            continue
        if {"EDGE", "MASKED", "SATURATED"}.intersection(source.flags):
            continue
        magnitude = instrumental_magnitude(source.flux, exposure_s=exposure_s)
        if magnitude is not None:
            selected_color = source_colors.get(source.detection_id) if source_colors is not None else None
            calibrated_candidate = magnitude + zero_point if zero_point is not None else None
            if photometric_calibration is not None:
                calibrated_candidate = photometric_calibration.apply(magnitude, color=selected_color)
                if source_photometry is not None:
                    photometry_row = source_photometry.get(int(source.detection_id))
                    # A numerical m_cal is not enough for the final ranking:
                    # when the row has a reference magnitude, its residual
                    # must pass the per-source photometric consistency gate.
                    # Missing/unknown rows are kept out of the calibrated
                    # subset and make the selection scope explicitly partial.
                    if (
                        photometry_row is None
                        or photometry_row.status != "CALIBRATED"
                        or photometry_row.photometric_consistent is not True
                    ):
                        calibrated_candidate = None
            candidates.append((magnitude, calibrated_candidate, source))
    if not candidates:
        return None
    calibrated_candidates = [candidate for candidate in candidates if candidate[1] is not None]
    if calibrated_candidates:
        magnitude, calibrated_magnitude, source = max(
            calibrated_candidates,
            key=lambda item: (float(item[1]), item[2].detection_id),
        )
        if photometric_calibration is not None:
            selection_scope = (
                "CALIBRATED_MATCHES"
                if len(calibrated_candidates) == len(candidates)
                else "CALIBRATED_MATCHES_PARTIAL"
            )
        else:
            selection_scope = "ZERO_POINT_ADJUSTED"
    else:
        magnitude, calibrated_magnitude, source = max(
            candidates,
            key=lambda item: (item[0], item[2].detection_id),
        )
        # A calibration object may exist while no eligible detection has a
        # usable colour/domain value.  Keep the instrumental fallback for
        # continuity, but expose it explicitly instead of silently calling it
        # the faintest calibrated source.
        selection_scope = (
            "QUALITY_DETECTIONS_NO_USABLE_CALIBRATION"
            if photometric_calibration is not None
            else "QUALITY_DETECTIONS"
        )
    magnitude = float(magnitude)
    # Do not reuse the loop-local rate here.  The selected source is the
    # faintest one after filtering, and its rate must be derived from that
    # same source; otherwise the exported/UI value can belong to the last
    # source visited in the loop.
    selected_flux_rate = source.flux / exposure_s
    selected_snr = source.flux_snr if source.flux_snr is not None else source.snr
    magnitude_error = 1.0857362047581296 / selected_snr if selected_snr > 0 else None
    photometric_system = None
    photometric_band = None
    calibration_status = "INSTRUMENTAL"
    calibrated_error = None
    if photometric_calibration is not None:
        photometric_system = photometric_calibration.photometric_system
        photometric_band = photometric_calibration.photometric_band
        calibration_status = photometric_calibration.status
        if calibrated_magnitude is not None:
            calibrated_error = calibrated_magnitude_error(
                magnitude_error,
                calibration=photometric_calibration,
            )
    return FaintestSource(
        detection_id=source.detection_id,
        x=source.x,
        y=source.y,
        flux=source.flux,
        snr=source.snr,
        instrumental_magnitude=magnitude,
        calibrated_magnitude=calibrated_magnitude,
        flags=source.flags,
        flux_snr=source.flux_snr,
        flux_rate=selected_flux_rate,
        instrumental_magnitude_error=magnitude_error,
        photometric_system=photometric_system,
        photometric_band=photometric_band,
        calibration_status=calibration_status,
        calibrated_magnitude_error=calibrated_error,
        selection_scope=selection_scope,
        eligible_candidate_count=len(candidates),
        calibrated_candidate_count=len(calibrated_candidates),
        absolute_magnitude=(
            source_absolute_magnitudes.get(source.detection_id)
            if source_absolute_magnitudes is not None
            else None
        ),
    )


def build_source_photometry(
    sources: Sequence[Detection],
    *,
    matches: Sequence[CatalogMatch] = (),
    catalog: Sequence[CatalogSource] = (),
    exposure_s: float = 1.0,
    photometric_calibration: PhotometricCalibration | None = None,
    parallax_zero_point_mas: float = 0.0,
    max_fractional_parallax_error: float = 0.2,
) -> tuple[SourcePhotometry, ...]:
    """为检测结果生成逐源星等表。

    该函数是结果层的单一入口：GUI、CLI 和缓存都使用同一套状态语义。
    ``sources`` 可以包含被质量筛选拒绝的候选；这些行保留 ``REJECTED``
    状态，便于解释为什么某个亮点没有被当成可用星，而不会把它悄悄
    丢掉后让用户误以为算法没有看到它。
    """

    if not math.isfinite(float(exposure_s)) or exposure_s <= 0:
        raise ValueError("exposure_s must be positive")
    if not math.isfinite(float(parallax_zero_point_mas)):
        raise ValueError("parallax_zero_point_mas must be finite")
    if not math.isfinite(float(max_fractional_parallax_error)) or max_fractional_parallax_error <= 0:
        raise ValueError("max_fractional_parallax_error must be positive")

    catalog_by_id = {str(source.source_id): source for source in catalog}
    match_by_detection = {int(match.detection_id): match for match in matches}
    calibration_roles = (
        dict(photometric_calibration.calibration_sample_roles)
        if photometric_calibration is not None
        else {}
    )
    result: list[SourcePhotometry] = []
    for detection in sources:
        signal_snr = detection.flux_snr if detection.flux_snr is not None else detection.snr
        instrumental = instrumental_magnitude(detection.flux, exposure_s=exposure_s)
        instrumental_error = (
            1.0857362047581296 / float(signal_snr)
            if instrumental is not None and math.isfinite(float(signal_snr)) and signal_snr > 0
            else None
        )
        match = match_by_detection.get(int(detection.detection_id))
        catalog_source = catalog_by_id.get(str(match.source_id)) if match is not None else None
        catalog_magnitude = None
        catalog_magnitude_error = None
        color = None
        color_name = None
        photometric_system = None
        photometric_band = None
        magnitude_source = None
        catalog_mg_gspphot = None
        catalog_mg_gspphot_lower = None
        catalog_mg_gspphot_upper = None
        catalog_mg_gspphot_source = None
        photometric_residual = None
        photometric_residual_limit = None
        photometric_consistent = None
        photometric_outlier_reason = None
        calibration_sample_role = None
        if catalog_source is not None:
            catalog_magnitude = catalog_source.magnitude
            catalog_magnitude_error = catalog_source.magnitude_error
            color = catalog_source.color
            color_name = catalog_source.color_name
            photometric_system = catalog_source.photometric_system
            photometric_band = catalog_source.photometric_band
            magnitude_source = catalog_source.magnitude_source
            catalog_mg_gspphot = catalog_source.mg_gspphot
            catalog_mg_gspphot_lower = catalog_source.mg_gspphot_lower
            catalog_mg_gspphot_upper = catalog_source.mg_gspphot_upper
            catalog_mg_gspphot_source = catalog_source.mg_gspphot_source
        if match is not None:
            if catalog_magnitude is None:
                catalog_magnitude = match.catalog_magnitude
            if catalog_magnitude_error is None:
                catalog_magnitude_error = match.catalog_magnitude_error
            if color is None:
                color = match.catalog_color
            color_name = color_name or match.catalog_color_name
            photometric_system = photometric_system or match.photometric_system
            photometric_band = photometric_band or match.photometric_band
        if match is not None and photometric_calibration is not None:
            calibration_sample_role = calibration_roles.get(str(match.source_id))

        flags = list(str(flag) for flag in detection.flags)
        if catalog_magnitude is not None and catalog_magnitude_error is None:
            flags.append("CATALOG_MAGNITUDE_ERROR_NOT_PROVIDED")
        calibrated = None
        calibrated_error = None
        absolute = None
        if not detection.quality_passed:
            status = "REJECTED_QUALITY"
            flags.append("QUALITY_REJECTED")
        elif instrumental is None:
            status = "NO_POSITIVE_FLUX"
            flags.append("NO_POSITIVE_FLUX")
        elif match is None:
            status = "INSTRUMENTAL_ONLY"
            flags.append("NO_CATALOG_MATCH")
        elif photometric_calibration is None:
            status = "CATALOG_MATCH_NO_CALIBRATION"
            flags.append("NO_PHOTOMETRIC_CALIBRATION")
        else:
            reference_system = catalog_source.photometric_system if catalog_source is not None else photometric_system
            reference_band = catalog_source.photometric_band if catalog_source is not None else photometric_band
            reference_color_name = catalog_source.color_name if catalog_source is not None else color_name
            metadata_matches = _photometric_metadata_matches(
                photometric_calibration.photometric_system,
                photometric_calibration.photometric_band,
                reference_system,
                reference_band,
            )
            color_metadata_matches = (
                not photometric_calibration.color_order
                or (
                    _normalise_photometric_label(photometric_calibration.color_name) is not None
                    and _normalise_photometric_label(reference_color_name) is not None
                    and _normalise_photometric_label(photometric_calibration.color_name)
                    == _normalise_photometric_label(reference_color_name)
                )
            )
            calibrated = (
                photometric_calibration.apply(instrumental, color=color)
                if metadata_matches and color_metadata_matches
                else None
            )
            if not metadata_matches:
                status = "CALIBRATION_METADATA_MISMATCH"
                flags.append("CALIBRATION_NOT_APPLIED")
            elif not color_metadata_matches:
                status = "CALIBRATION_COLOR_METADATA_MISMATCH"
                flags.append("CALIBRATION_NOT_APPLIED")
            elif calibrated is None:
                status = (
                    "CALIBRATION_MISSING_COLOR"
                    if photometric_calibration.color_order and color is None
                    else "CALIBRATION_INVALID"
                )
                flags.append("CALIBRATION_NOT_APPLIED")
            else:
                # Once the calibration has been applied, its system and band
                # are the provenance of m_cal.  Do not copy a catalog label
                # that could describe a different passband.
                photometric_system = photometric_calibration.photometric_system
                photometric_band = photometric_calibration.photometric_band
                calibrated_error = calibrated_magnitude_error(
                    instrumental_error,
                    calibration=photometric_calibration,
                )
                photometric_residual_limit = photometric_calibration.max_source_residual_mag
                if catalog_magnitude is not None and math.isfinite(float(catalog_magnitude)):
                    photometric_residual = float(catalog_magnitude) - float(calibrated)
                    photometric_consistent = (
                        photometric_residual_limit is None
                        or abs(photometric_residual) <= photometric_residual_limit
                    )
                    if not photometric_consistent:
                        status = "CATALOG_INCONSISTENT"
                        photometric_outlier_reason = "RESIDUAL_EXCEEDS_LIMIT"
                        flags.extend(("PHOTOMETRIC_OUTLIER", "CALIBRATION_NOT_APPLIED_TO_SELECTION"))
                    else:
                        status = "CALIBRATED"
                else:
                    status = "CALIBRATED"
                if catalog_source is not None and photometric_consistent is not False:
                    absolute = absolute_magnitude_from_catalog(
                        catalog_source,
                        apparent_magnitude=calibrated,
                        apparent_magnitude_error=calibrated_error,
                        required_photometric_system=photometric_calibration.photometric_system,
                        required_photometric_band=photometric_calibration.photometric_band,
                        parallax_zero_point_mas=parallax_zero_point_mas,
                        max_fractional_parallax_error=max_fractional_parallax_error,
                    )
                    flags.extend(absolute.flags)
        result.append(
            SourcePhotometry(
                detection_id=int(detection.detection_id),
                source_id=str(match.source_id) if match is not None else None,
                instrumental_magnitude=instrumental,
                instrumental_magnitude_error=instrumental_error,
                calibrated_magnitude=calibrated,
                calibrated_magnitude_error=calibrated_error,
                catalog_magnitude=catalog_magnitude,
                catalog_magnitude_error=catalog_magnitude_error,
                color=color,
                color_name=color_name,
                photometric_system=photometric_system,
                photometric_band=photometric_band,
                absolute_magnitude=absolute,
                status=status,
                magnitude_source=magnitude_source,
                flags=tuple(dict.fromkeys(flags)),
                photometric_residual_mag=photometric_residual,
                photometric_residual_limit_mag=photometric_residual_limit,
                photometric_consistent=photometric_consistent,
                photometric_outlier_reason=photometric_outlier_reason,
                calibration_sample_role=calibration_sample_role,
                catalog_mg_gspphot=catalog_mg_gspphot,
                catalog_mg_gspphot_lower=catalog_mg_gspphot_lower,
                catalog_mg_gspphot_upper=catalog_mg_gspphot_upper,
                catalog_mg_gspphot_source=catalog_mg_gspphot_source,
            )
        )
    return tuple(result)
