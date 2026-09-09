"""近邻双框的局部异常值修复反事实审计。

``pair_response_attribution`` 比较的是掩膜后的响应；掩膜会减少有效像素，
因此还需要一个互补控制：在内存副本中用局部稳健邻域值替换重复码或极端
负值，再重新运行同一 detector。该操作不是数据校准，也不是物理真值重建，
只用于回答“响应分裂是否依赖某类局部值域缺口”。原始 FITS 永远不被写回。
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .detection import Detection, DetectionResult, detect_sources
from .fits import FitsFrame, read_fits


PAIR_REPAIR_VARIANTS = (
    "raw",
    "repair_repeated_code",
    "repair_negative_anomaly",
    "repair_both",
)


@dataclass(frozen=True, slots=True)
class PairRepairReplacementRow:
    """一个被替换的局部特殊像素。"""

    variant: str
    pixel_class: str
    x: int
    y: int
    original_value_adu: float
    replacement_value_adu: float

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PairRepairCandidateRow:
    """修复变体在目标附近重新提出的一个候选。"""

    variant: str
    detection_id: int
    x: float
    y: float
    peak_x: float | None
    peak_y: float | None
    peak: float
    flux_snr: float | None
    filter_snr: float | None
    quality_passed: bool
    flags: str
    distance_to_primary_px: float
    distance_to_secondary_px: float
    nearest_target: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PairRepairSummaryRow:
    """每个修复变体的全局和目标窗口计数。"""

    variant: str
    repaired_pixel_count: int
    candidate_count: int
    returned_count: int
    quality_count: int
    target_window_candidate_count: int
    target_window_quality_count: int
    primary_window_candidate_count: int
    secondary_window_candidate_count: int

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PairRepairCounterfactualResult:
    """局部异常值修复反事实结果。"""

    source_path: str
    primary_detection_id: int
    secondary_detection_id: int
    primary_target_x: float
    primary_target_y: float
    secondary_target_x: float
    secondary_target_y: float
    patch_x0: int
    patch_y0: int
    patch_x1: int
    patch_y1: int
    repeated_code_values: tuple[int, ...]
    negative_anomaly_threshold_adu: float
    replacement_radius_px: int
    target_match_radius_px: float
    summaries: tuple[PairRepairSummaryRow, ...]
    candidates: tuple[PairRepairCandidateRow, ...]
    replacements: tuple[PairRepairReplacementRow, ...]
    parameters: dict[str, object]
    conclusion: str

    def as_dict(self) -> dict[str, object]:
        return {
            "source_path": self.source_path,
            "primary_detection_id": self.primary_detection_id,
            "secondary_detection_id": self.secondary_detection_id,
            "primary_target": [self.primary_target_x, self.primary_target_y],
            "secondary_target": [self.secondary_target_x, self.secondary_target_y],
            "patch_bounds": [self.patch_x0, self.patch_y0, self.patch_x1, self.patch_y1],
            "repeated_code_values": list(self.repeated_code_values),
            "negative_anomaly_threshold_adu": self.negative_anomaly_threshold_adu,
            "replacement_radius_px": self.replacement_radius_px,
            "target_match_radius_px": self.target_match_radius_px,
            "summaries": [row.as_dict() for row in self.summaries],
            "candidates": [row.as_dict() for row in self.candidates],
            "replacements": [row.as_dict() for row in self.replacements],
            "parameters": dict(self.parameters),
            "conclusion": self.conclusion,
            "interpretation_guardrails": [
                "邻域补值是反事实敏感性控制，不是坏像素校准或物理真值重建。",
                "候选数量变化不等于物理星数变化；修复后的 detector 仍需 PSF/WCS/注入核验。",
                "局部窗口重跑的 detection_id 只在变体内部有意义，不能与首帧全图 ID 直接比较。",
                "补值结果只能说明当前响应对该类值域的敏感性，不提供噪点概率或伪影率。",
            ],
        }


def _finite_positive(value: float, name: str) -> float:
    converted = float(value)
    if not math.isfinite(converted) or converted <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return converted


def _peak_coordinate(source: Detection, axis: str) -> int:
    value = getattr(source, f"peak_{axis}")
    if value is None or not math.isfinite(float(value)):
        value = getattr(source, axis)
    return int(round(float(value)))


def _repair_pixels(
    patch: np.ndarray,
    selector: np.ndarray,
    *,
    exclusion: np.ndarray,
    radius: int,
    patch_x0: int,
    patch_y0: int,
    pixel_class: str,
) -> tuple[np.ndarray, tuple[PairRepairReplacementRow, ...]]:
    """用原始 patch 中的局部中位数同时替换 selector 像素。"""

    if selector.shape != patch.shape or exclusion.shape != patch.shape:
        raise ValueError("selector and exclusion must match patch shape")
    if int(radius) < 1:
        raise ValueError("radius must be positive")
    output = patch.copy()
    finite_ordinary = np.isfinite(patch) & ~exclusion
    fallback_values = patch[finite_ordinary]
    fallback = float(np.median(fallback_values)) if fallback_values.size else 0.0
    rows: list[PairRepairReplacementRow] = []
    for y, x in np.argwhere(selector):
        y = int(y)
        x = int(x)
        y0 = max(0, y - int(radius))
        y1 = min(patch.shape[0], y + int(radius) + 1)
        x0 = max(0, x - int(radius))
        x1 = min(patch.shape[1], x + int(radius) + 1)
        neighborhood = patch[y0:y1, x0:x1]
        neighborhood_exclusion = exclusion[y0:y1, x0:x1]
        values = neighborhood[np.isfinite(neighborhood) & ~neighborhood_exclusion]
        replacement = float(np.median(values)) if values.size else fallback
        original = float(patch[y, x])
        output[y, x] = replacement
        rows.append(
            PairRepairReplacementRow(
                variant="",
                pixel_class=pixel_class,
                x=int(x + patch_x0),
                y=int(y + patch_y0),
                original_value_adu=original,
                replacement_value_adu=replacement,
            )
        )
    return output, tuple(rows)


def _candidate_rows(
    variant: str,
    detection: DetectionResult,
    *,
    patch_x0: int,
    patch_y0: int,
    primary_target: tuple[float, float],
    secondary_target: tuple[float, float],
    target_match_radius_px: float,
) -> tuple[PairRepairCandidateRow, ...]:
    rows: list[PairRepairCandidateRow] = []
    for source in detection.sources:
        x = float(source.x + patch_x0)
        y = float(source.y + patch_y0)
        primary_distance = math.hypot(x - primary_target[0], y - primary_target[1])
        secondary_distance = math.hypot(x - secondary_target[0], y - secondary_target[1])
        nearest_distance = min(primary_distance, secondary_distance)
        if nearest_distance > target_match_radius_px:
            continue
        nearest_target = "primary" if primary_distance <= secondary_distance else "secondary"
        peak_x = None if source.peak_x is None else float(source.peak_x + patch_x0)
        peak_y = None if source.peak_y is None else float(source.peak_y + patch_y0)
        rows.append(
            PairRepairCandidateRow(
                variant=variant,
                detection_id=int(source.detection_id),
                x=x,
                y=y,
                peak_x=peak_x,
                peak_y=peak_y,
                peak=float(source.peak),
                flux_snr=None if source.flux_snr is None else float(source.flux_snr),
                filter_snr=None if source.filter_snr is None else float(source.filter_snr),
                quality_passed=bool(source.quality_passed),
                flags="|".join(source.flags),
                distance_to_primary_px=float(primary_distance),
                distance_to_secondary_px=float(secondary_distance),
                nearest_target=nearest_target,
            )
        )
    return tuple(sorted(rows, key=lambda row: (row.nearest_target, row.detection_id)))


def _variant_mask(
    variant: str,
    *,
    repeated_code_mask: np.ndarray,
    negative_mask: np.ndarray,
) -> tuple[np.ndarray, str | None]:
    if variant == "raw":
        return np.zeros_like(repeated_code_mask, dtype=bool), None
    if variant == "repair_repeated_code":
        return repeated_code_mask, "repeated_code"
    if variant == "repair_negative_anomaly":
        return negative_mask, "negative_anomaly"
    if variant == "repair_both":
        return repeated_code_mask | negative_mask, "both"
    raise ValueError(f"unknown repair variant: {variant}")


def run_pair_repair_counterfactual(
    frame: str | Path | FitsFrame,
    primary: Detection,
    secondary: Detection,
    *,
    patch_padding_px: int = 128,
    replacement_radius_px: int = 1,
    target_match_radius_px: float = 4.0,
    repeated_code_values: Sequence[int] | None = None,
    negative_anomaly_threshold_adu: float = -1000.0,
    threshold_sigma: float = 4.0,
    min_distance: int = 4,
    aperture_radius: int = 4,
    psf_fwhm: float = 2.0,
    background_box_size: int = 128,
    min_flux_snr: float = 5.0,
    min_psf_support_pixels: int = 3,
    proposal_mode: str = "hybrid",
) -> PairRepairCounterfactualResult:
    """对局部异常值做邻域补值，并重新运行当前 detector。"""

    if int(primary.detection_id) == int(secondary.detection_id):
        raise ValueError("primary and secondary detections must differ")
    if int(patch_padding_px) < 1:
        raise ValueError("patch_padding_px must be positive")
    if int(replacement_radius_px) < 1:
        raise ValueError("replacement_radius_px must be positive")
    target_radius = _finite_positive(target_match_radius_px, "target_match_radius_px")
    negative_threshold = float(negative_anomaly_threshold_adu)
    if not math.isfinite(negative_threshold):
        raise ValueError("negative_anomaly_threshold_adu must be finite")

    fits_frame = frame if isinstance(frame, FitsFrame) else read_fits(frame)
    values = np.asarray(fits_frame.data, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"expected a 2-D FITS image, got shape {values.shape}")
    height, width = values.shape
    primary_target = (float(primary.x), float(primary.y))
    secondary_target = (float(secondary.x), float(secondary.y))
    primary_peak_x = _peak_coordinate(primary, "x")
    primary_peak_y = _peak_coordinate(primary, "y")
    secondary_peak_x = _peak_coordinate(secondary, "x")
    secondary_peak_y = _peak_coordinate(secondary, "y")
    x0 = max(0, min(primary_peak_x, secondary_peak_x) - int(patch_padding_px))
    y0 = max(0, min(primary_peak_y, secondary_peak_y) - int(patch_padding_px))
    x1 = min(width, max(primary_peak_x, secondary_peak_x) + int(patch_padding_px) + 1)
    y1 = min(height, max(primary_peak_y, secondary_peak_y) + int(patch_padding_px) + 1)
    patch = values[y0:y1, x0:x1]
    finite = np.isfinite(patch)
    if not finite.any():
        raise ValueError("pair patch contains no finite pixels")

    repeated = set(int(value) for value in (repeated_code_values or ()))
    repeated.update(int(value) for value in primary.repeated_code_values)
    repeated.update(int(value) for value in secondary.repeated_code_values)
    repeated_codes = frozenset(repeated)
    repeated_mask = (
        np.isin(patch, np.asarray(tuple(sorted(repeated_codes)), dtype=np.float64))
        if repeated_codes
        else np.zeros_like(patch, dtype=bool)
    )
    negative_mask = finite & (patch <= negative_threshold)
    special_mask = (~finite) | repeated_mask | negative_mask
    summaries: list[PairRepairSummaryRow] = []
    candidate_rows: list[PairRepairCandidateRow] = []
    replacement_rows: list[PairRepairReplacementRow] = []
    for variant in PAIR_REPAIR_VARIANTS:
        selector, pixel_class = _variant_mask(
            variant,
            repeated_code_mask=repeated_mask,
            negative_mask=negative_mask,
        )
        if variant == "raw":
            variant_image = patch.copy()
            variant_replacements: tuple[PairRepairReplacementRow, ...] = ()
        else:
            variant_image, raw_replacements = _repair_pixels(
                patch,
                selector,
                exclusion=special_mask,
                radius=int(replacement_radius_px),
                patch_x0=x0,
                patch_y0=y0,
                pixel_class=str(pixel_class),
            )
            variant_replacements = tuple(
                PairRepairReplacementRow(
                    variant=variant,
                    pixel_class=row.pixel_class,
                    x=row.x,
                    y=row.y,
                    original_value_adu=row.original_value_adu,
                    replacement_value_adu=row.replacement_value_adu,
                )
                for row in raw_replacements
            )
        detection = detect_sources(
            variant_image,
            threshold_sigma=float(threshold_sigma),
            min_distance=int(min_distance),
            aperture_radius=int(aperture_radius),
            psf_fwhm=float(psf_fwhm),
            background_box_size=int(background_box_size),
            min_flux_snr=float(min_flux_snr),
            min_psf_support_pixels=int(min_psf_support_pixels),
            proposal_mode=str(proposal_mode),
            reject_linear_artifacts=True,
        )
        current_candidates = _candidate_rows(
            variant,
            detection,
            patch_x0=x0,
            patch_y0=y0,
            primary_target=primary_target,
            secondary_target=secondary_target,
            target_match_radius_px=target_radius,
        )
        candidate_rows.extend(current_candidates)
        replacement_rows.extend(variant_replacements)
        summaries.append(
            PairRepairSummaryRow(
                variant=variant,
                repaired_pixel_count=len(variant_replacements),
                candidate_count=int(detection.candidate_count),
                returned_count=int(detection.returned_count),
                quality_count=int(detection.star_count),
                target_window_candidate_count=len(current_candidates),
                target_window_quality_count=sum(row.quality_passed for row in current_candidates),
                primary_window_candidate_count=sum(row.nearest_target == "primary" for row in current_candidates),
                secondary_window_candidate_count=sum(row.nearest_target == "secondary" for row in current_candidates),
            )
        )

    summary_by_variant = {row.variant: row for row in summaries}
    raw_summary = summary_by_variant["raw"]
    negative_summary = summary_by_variant["repair_negative_anomaly"]
    both_summary = summary_by_variant["repair_both"]
    conclusion = (
        "局部异常值修复反事实完成："
        f"raw 目标窗口候选 {raw_summary.target_window_candidate_count} 个，"
        f"修复负异常后 {negative_summary.target_window_candidate_count} 个，"
        f"同时修复两类后 {both_summary.target_window_candidate_count} 个。"
        "若只修复重复码仍保留双响应、而修复负异常使目标窗口收缩为单主响应，"
        "则负值缺口是当前双峰分裂的主要形状触发，重复码更像值域有效性/锚点问题；"
        "该方向仍是局部 detector-level 反事实，不是硬件机理或物理星数结论。"
    )
    return PairRepairCounterfactualResult(
        source_path=str(fits_frame.path),
        primary_detection_id=int(primary.detection_id),
        secondary_detection_id=int(secondary.detection_id),
        primary_target_x=primary_target[0],
        primary_target_y=primary_target[1],
        secondary_target_x=secondary_target[0],
        secondary_target_y=secondary_target[1],
        patch_x0=int(x0),
        patch_y0=int(y0),
        patch_x1=int(x1),
        patch_y1=int(y1),
        repeated_code_values=tuple(sorted(repeated_codes)),
        negative_anomaly_threshold_adu=negative_threshold,
        replacement_radius_px=int(replacement_radius_px),
        target_match_radius_px=target_radius,
        summaries=tuple(summaries),
        candidates=tuple(candidate_rows),
        replacements=tuple(replacement_rows),
        parameters={
            "threshold_sigma": float(threshold_sigma),
            "min_distance": int(min_distance),
            "aperture_radius": int(aperture_radius),
            "psf_fwhm": float(psf_fwhm),
            "background_box_size": int(background_box_size),
            "min_flux_snr": float(min_flux_snr),
            "min_psf_support_pixels": int(min_psf_support_pixels),
            "proposal_mode": str(proposal_mode),
            "replacement_rule": (
                "simultaneous local median within square neighborhood after excluding "
                "all known special pixels"
            ),
            "target_window_rule": (
                "measured centroid distance to either target <= target_match_radius_px"
            ),
            "interpretation_boundary": (
                "local repair counterfactual; no calibration, truth, FDR, precision, "
                "or physical artifact rate"
            ),
        },
        conclusion=conclusion,
    )


def _write_rows(path: Path, rows: Iterable[object], row_type: type[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[field.name for field in fields(row_type)])
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def write_pair_repair_counterfactual_artifacts(
    result: PairRepairCounterfactualResult,
    out_dir: str | Path,
) -> Path:
    """写出修复变体汇总、目标候选和像素替换表。"""

    output = Path(out_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    _write_rows(output / "pair_repair_summaries.csv", result.summaries, PairRepairSummaryRow)
    _write_rows(output / "pair_repair_candidates.csv", result.candidates, PairRepairCandidateRow)
    _write_rows(output / "pair_repair_replacements.csv", result.replacements, PairRepairReplacementRow)
    (output / "pair_repair_counterfactual.json").write_text(
        json.dumps(result.as_dict(), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return output


__all__ = [
    "PAIR_REPAIR_VARIANTS",
    "PairRepairCandidateRow",
    "PairRepairCounterfactualResult",
    "PairRepairReplacementRow",
    "PairRepairSummaryRow",
    "run_pair_repair_counterfactual",
    "write_pair_repair_counterfactual_artifacts",
]
