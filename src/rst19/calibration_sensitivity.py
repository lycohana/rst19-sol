"""光度颜色模型敏感性审计。

相机只给出 450--750 nm 的宽谱范围，而不是一条已测量的系统响应曲线。
因此，把 Gaia G 的一次颜色项拟合结果当成“相机绝对星等”是不充分的。
本模块在同一组匹配和质量门控上重复拟合零阶、一次和二次颜色模型，输出
目标 ``m_cal`` 以及最暗源排序对模型阶数的敏感性。

这不是 Gaia XP 合成测光，也不是后验概率。它只回答一个更窄但可复核的
问题：在当前参考星、当前颜色范围和当前图像测光下，合理的低阶颜色模型
改变了多少结果。没有两个以上有效模型时，结果只能作为诊断，不能宣称
“模型稳定”。
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
import math
from typing import Any

from .catalog import CatalogSource
from .detection import Detection
from .matching import CatalogMatch
from .photometry import PhotometricCalibration, fit_photometric_calibration, instrumental_magnitude


DEFAULT_COLOR_MODEL_ORDERS = (0, 1, 2)
VALID_CALIBRATION_STATUSES = frozenset({"VALID", "VALID_NO_HOLDOUT"})
MODEL_NAMES = {
    0: "zero_point_only",
    1: "linear_color_term",
    2: "quadratic_color_term",
}
MODEL_LABELS = {
    0: "零阶零点",
    1: "一次颜色项",
    2: "二次颜色项",
}


def _finite(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _json_float(value: float | None) -> float | None:
    return None if value is None or not math.isfinite(float(value)) else float(value)


def _model_dict(
    order: int,
    calibration: PhotometricCalibration,
) -> dict[str, Any]:
    return {
        "name": MODEL_NAMES[order],
        "label": MODEL_LABELS[order],
        "color_order": order,
        "status": calibration.status,
        "coefficients": [float(value) for value in calibration.coefficients],
        "calibrator_count": int(calibration.calibrator_count),
        "inlier_count": int(calibration.inlier_count),
        "validation_count": int(calibration.validation_count),
        "fit_rms_mag": _json_float(calibration.fit_rms_mag),
        "validation_rms_mag": _json_float(calibration.validation_rms_mag),
        "residual_mad_mag": _json_float(calibration.residual_mad_mag),
        "color_min": _json_float(calibration.color_min),
        "color_max": _json_float(calibration.color_max),
        "flags": list(calibration.flags),
        "usable": calibration.status in VALID_CALIBRATION_STATUSES,
    }


def _matched_measurements(
    matches: Sequence[CatalogMatch],
    detections: Sequence[Detection],
    catalog: Sequence[CatalogSource],
    *,
    exposure_s: float,
    min_snr: float,
) -> list[dict[str, Any]]:
    detections_by_id = {int(source.detection_id): source for source in detections}
    catalog_by_id = {str(source.source_id): source for source in catalog}
    rows: list[dict[str, Any]] = []
    seen_detection_ids: set[int] = set()
    seen_source_ids: set[str] = set()
    for match in matches:
        detection_id = int(match.detection_id)
        source_id = str(match.source_id)
        if detection_id in seen_detection_ids or source_id in seen_source_ids:
            continue
        detection = detections_by_id.get(detection_id)
        source = catalog_by_id.get(source_id)
        if detection is None or source is None:
            continue
        signal_snr = detection.flux_snr if detection.flux_snr is not None else detection.snr
        if (
            not detection.quality_passed
            or not _finite(signal_snr)
            or float(signal_snr) < float(min_snr)
            or not _finite(detection.flux)
            or float(detection.flux) <= 0.0
            or {"EDGE", "MASKED", "SATURATED"}.intersection(detection.flags)
        ):
            continue
        instrumental = instrumental_magnitude(float(detection.flux), exposure_s=exposure_s)
        if instrumental is None:
            continue
        color = source.color if source.color is not None else match.catalog_color
        rows.append(
            {
                "detection_id": detection_id,
                "source_id": source_id,
                "x": float(detection.x),
                "y": float(detection.y),
                "instrumental_magnitude": float(instrumental),
                "color": float(color) if color is not None and _finite(color) else None,
                "flux_snr": float(signal_snr),
            }
        )
        seen_detection_ids.add(detection_id)
        seen_source_ids.add(source_id)
    rows.sort(key=lambda row: (int(row["detection_id"]), str(row["source_id"])))
    return rows


def _rank_faintest(rows: Sequence[Mapping[str, Any]], model_name: str) -> dict[str, Any] | None:
    valid = [
        row
        for row in rows
        if isinstance(row.get("models"), Mapping)
        and _finite(row["models"].get(model_name))
    ]
    if not valid:
        return None
    winner = max(valid, key=lambda row: (float(row["models"][model_name]), -int(row["detection_id"])))
    return {
        "model": model_name,
        "detection_id": int(winner["detection_id"]),
        "source_id": str(winner["source_id"]),
        "magnitude": float(winner["models"][model_name]),
        "eligible_count": len(valid),
    }


def build_calibration_model_sensitivity(
    matches: Sequence[CatalogMatch],
    detections: Sequence[Detection],
    catalog: Sequence[CatalogSource],
    *,
    exposure_s: float = 1.0,
    photometric_system: str = "unknown",
    photometric_band: str = "unknown",
    color_name: str | None = "BP-RP",
    model_orders: Sequence[int] = DEFAULT_COLOR_MODEL_ORDERS,
    min_snr: float = 5.0,
    min_calibrators: int = 6,
) -> dict[str, Any]:
    """比较低阶颜色模型对 ``m_cal`` 和最暗源排序的影响。

    ``model_orders`` 只允许 0、1、2，因为这三个模型对应现有的可解释
    光度方程。函数不会生成缺失颜色、目录值或响应曲线；没有数据的模型
    记为不可用。输出中的 ``winner_agreement_fraction`` 是有效模型之间的
    确定性计数比例，不是统计置信度或目标是真星的概率。
    """

    if not _finite(exposure_s) or float(exposure_s) <= 0.0:
        raise ValueError("exposure_s must be positive")
    if not _finite(min_snr) or float(min_snr) <= 0.0:
        raise ValueError("min_snr must be positive")
    if isinstance(min_calibrators, bool) or not isinstance(min_calibrators, int) or min_calibrators < 2:
        raise ValueError("min_calibrators must be an integer >= 2")
    orders = tuple(dict.fromkeys(int(order) for order in model_orders))
    if not orders or any(order not in MODEL_NAMES for order in orders):
        raise ValueError("model_orders must contain only 0, 1 or 2")

    # Fit exactly the same data/quality policy used by the ordinary
    # photometric calibration.  A failed variant remains in the audit so the
    # missing model is visible rather than silently dropping the reason.
    variants: list[tuple[int, PhotometricCalibration | None]] = []
    variant_errors: dict[int, str] = {}
    for order in orders:
        try:
            calibration = fit_photometric_calibration(
                matches,
                detections,
                catalog,
                exposure_s=float(exposure_s),
                photometric_system=photometric_system,
                photometric_band=photometric_band,
                color_name=color_name,
                color_order=order,
                min_snr=float(min_snr),
                min_calibrators=max(min_calibrators, order + 2),
            )
        except (OSError, RuntimeError, ValueError) as exc:
            # Keep a machine-readable failure without making a successful
            # primary calibration unusable.
            variants.append((order, None))
            variant_errors[order] = str(exc)
            continue
        variants.append((order, calibration))

    measurements = _matched_measurements(
        matches,
        detections,
        catalog,
        exposure_s=float(exposure_s),
        min_snr=float(min_snr),
    )
    source_rows: list[dict[str, Any]] = []
    for row in measurements:
        values: dict[str, float] = {}
        for order, calibration in variants:
            if calibration is None or calibration.status not in VALID_CALIBRATION_STATUSES:
                continue
            value = calibration.apply(row["instrumental_magnitude"], color=row["color"])
            if value is not None and math.isfinite(float(value)):
                values[MODEL_NAMES[order]] = float(value)
        numeric = list(values.values())
        source_rows.append(
            {
                **row,
                "models": values,
                "valid_model_count": len(values),
                "model_min_mag": min(numeric) if numeric else None,
                "model_max_mag": max(numeric) if numeric else None,
                "model_median_mag": (
                    sorted(numeric)[len(numeric) // 2]
                    if numeric and len(numeric) % 2
                    else (
                        0.5 * (sorted(numeric)[len(numeric) // 2 - 1] + sorted(numeric)[len(numeric) // 2])
                        if numeric
                        else None
                    )
                ),
                "model_spread_mag": max(numeric) - min(numeric) if len(numeric) >= 2 else None,
            }
        )

    rankings = [
        rank
        for order, calibration in variants
        if calibration is not None
        and calibration.status in VALID_CALIBRATION_STATUSES
        for rank in [_rank_faintest(source_rows, MODEL_NAMES[order])]
        if rank is not None
    ]
    winner_counts = Counter(str(rank["source_id"]) for rank in rankings)
    winner_source_id = None
    winner_count = 0
    if winner_counts:
        winner_source_id, winner_count = sorted(
            winner_counts.items(), key=lambda item: (-item[1], item[0])
        )[0]
    valid_model_count = len(rankings)
    agreement_fraction = (
        float(winner_count) / float(valid_model_count)
        if valid_model_count
        else None
    )
    usable_variants: list[dict[str, Any]] = []
    for order, calibration in variants:
        if calibration is None:
            usable_variants.append(
                {
                    "name": MODEL_NAMES[order],
                    "label": MODEL_LABELS[order],
                    "color_order": order,
                    "status": "ERROR",
                    "coefficients": [],
                    "calibrator_count": 0,
                    "inlier_count": 0,
                    "validation_count": 0,
                    "fit_rms_mag": None,
                    "validation_rms_mag": None,
                    "residual_mad_mag": None,
                    "color_min": None,
                    "color_max": None,
                    "flags": ["MODEL_FIT_ERROR"],
                    "usable": False,
                    "failure_reason": variant_errors.get(order, "unknown model fit error"),
                }
            )
        else:
            usable_variants.append(_model_dict(order, calibration))
    for model in usable_variants:
        if not model["usable"] and not model.get("failure_reason"):
            model["failure_reason"] = ";".join(model["flags"]) or model["status"]
    usable_count = sum(1 for model in usable_variants if model["usable"])
    status = (
        "MODEL_STABLE"
        if usable_count >= 2 and agreement_fraction is not None and agreement_fraction >= 2.0 / 3.0
        else "MODEL_SENSITIVE"
        if usable_count >= 2
        else "DIAGNOSTIC_ONLY"
    )
    return {
        "schema_version": 1,
        "status": status,
        "method": "low_order_color_model_sensitivity",
        "photometric_system": photometric_system,
        "photometric_band": photometric_band,
        "color_name": color_name,
        "models": usable_variants,
        "model_count_requested": len(orders),
        "model_count_usable": usable_count,
        "matched_measurement_count": len(measurements),
        "source_count_with_any_model": sum(1 for row in source_rows if row["valid_model_count"] > 0),
        "sources": source_rows,
        "faintest_ranking": {
            "per_model": rankings,
            "winner_source_id": winner_source_id,
            "winner_count": winner_count,
            "valid_model_count": valid_model_count,
            "winner_agreement_fraction": agreement_fraction,
            "interpretation": (
                "有效低阶颜色模型中最暗候选的确定性一致比例；不是后验概率、完备率或真星概率。"
            ),
        },
        "claim_boundary": {
            "can_claim": [
                "当前匹配参考星和颜色范围内的低阶光度模型敏感性",
                "模型阶数改变时逐源 m_cal 的范围",
                "最暗候选排序在有效模型之间是否一致",
            ],
            "cannot_claim": [
                "Gaia G 等于开运一号真实标准波段星等",
                "未知相机响应曲线已被恢复",
                "模型一致比例是统计置信度或真星概率",
                "没有距离和消光时的绝对星等 M",
            ],
        },
        "notes": [
            "零阶/一次/二次模型共享同一批检测、匹配和质量门控，但可用参考星数会因颜色缺失而不同。",
            "该实验量化颜色模型误差，不替代实验室响应曲线、标准星观测或 Gaia XP 合成测光。",
        ],
    }


__all__ = [
    "DEFAULT_COLOR_MODEL_ORDERS",
    "MODEL_LABELS",
    "MODEL_NAMES",
    "VALID_CALIBRATION_STATUSES",
    "build_calibration_model_sensitivity",
]
