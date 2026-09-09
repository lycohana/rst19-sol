"""近邻双框的匹配响应反事实归因。

本模块只回答一个局部、可复核的问题：同一原始亮斑附近的两个宽筛响应，
在保留原始值域的前提下，分别对重复工程码和极端负值做留出后，局部
Gaussian 响应图如何改变。它不重新定义质量门，也不输出物理恒星概率。

与 ``pair_pixel_topology`` 的区别是：像素拓扑看原图的正连通块和原始
局部极大值；本模块直接看检测器使用的 Gaussian 平滑响应，并把离散核
支持内的响应分解为普通像素、重复码和负异常三类。这样可以把“一个
亮斑被滤波器拆成两个响应”的机制说清楚，同时保留掩膜后的候选有效性。
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from scipy import ndimage

from .detection import Detection
from .fits import FitsFrame, read_fits


PAIR_RESPONSE_MASK_MODES = (
    "raw",
    "repeated_code_masked",
    "negative_anomaly_masked",
    "both_masked",
)
_PIXEL_CLASSES = ("ordinary", "repeated_code", "negative_anomaly")
_FWHM_TO_SIGMA = 2.35482


@dataclass(frozen=True, slots=True)
class PairResponseTargetRow:
    """一个目标峰在不同局部值域留出下的响应。"""

    mask_mode: str
    target_label: str
    detection_id: int
    peak_x: int
    peak_y: int
    raw_value_adu: float
    valid_at_target: bool
    gaussian_response_adu: float
    valid_weight: float
    response_local_maximum: bool
    local_maximum_rank: int | None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PairResponseMaximumRow:
    """局部 Gaussian 响应图中的一个有效局部峰。"""

    mask_mode: str
    rank: int
    x: int
    y: int
    raw_value_adu: float
    gaussian_response_adu: float
    valid_weight: float
    is_primary_target: bool
    is_secondary_target: bool

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PairResponseContributionRow:
    """离散 Gaussian 支持内某类原始像素对响应的带符号贡献。"""

    target_label: str
    detection_id: int
    pixel_class: str
    pixel_count: int
    kernel_weight_fraction: float
    signed_response_contribution_adu: float

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PairResponseAttributionResult:
    """近邻双框匹配响应归因结果。"""

    source_path: str
    primary_detection_id: int
    secondary_detection_id: int
    primary_peak_x: int
    primary_peak_y: int
    secondary_peak_x: int
    secondary_peak_y: int
    patch_x0: int
    patch_y0: int
    patch_x1: int
    patch_y1: int
    common_background_adu: float
    common_noise_adu: float
    psf_fwhm_px: float
    repeated_code_values: tuple[int, ...]
    negative_anomaly_threshold_adu: float
    target_rows: tuple[PairResponseTargetRow, ...]
    maximum_rows: tuple[PairResponseMaximumRow, ...]
    contribution_rows: tuple[PairResponseContributionRow, ...]
    parameters: dict[str, object]
    conclusion: str

    def as_dict(self) -> dict[str, object]:
        return {
            "source_path": self.source_path,
            "primary_detection_id": self.primary_detection_id,
            "secondary_detection_id": self.secondary_detection_id,
            "primary_peak": [self.primary_peak_x, self.primary_peak_y],
            "secondary_peak": [self.secondary_peak_x, self.secondary_peak_y],
            "patch_bounds": [self.patch_x0, self.patch_y0, self.patch_x1, self.patch_y1],
            "common_background_adu": self.common_background_adu,
            "common_noise_adu": self.common_noise_adu,
            "psf_fwhm_px": self.psf_fwhm_px,
            "repeated_code_values": list(self.repeated_code_values),
            "negative_anomaly_threshold_adu": self.negative_anomaly_threshold_adu,
            "target_rows": [row.as_dict() for row in self.target_rows],
            "maximum_rows": [row.as_dict() for row in self.maximum_rows],
            "contribution_rows": [row.as_dict() for row in self.contribution_rows],
            "parameters": dict(self.parameters),
            "conclusion": self.conclusion,
            "interpretation_guardrails": [
                "Gaussian 响应是 detector-level 匹配响应，不是物理源数或星等真值。",
                "屏蔽后某个响应消失只说明该响应依赖当前值域/候选有效性，不单独证明噪点。",
                "像素贡献是带符号的局部核归因，不是因果概率或伪影率。",
                "普通像素、重复码和负异常的分类依赖当前阈值与原始存储语义。",
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


def _response_and_weight(
    residual: np.ndarray,
    excluded: np.ndarray,
    *,
    sigma: float,
) -> tuple[np.ndarray, np.ndarray]:
    """复现检测器的掩膜归一化 Gaussian 响应。"""

    valid = ~np.asarray(excluded, dtype=bool)
    weighted = ndimage.gaussian_filter(
        residual * valid,
        sigma=sigma,
        mode="nearest",
    )
    weights = ndimage.gaussian_filter(
        valid.astype(np.float64),
        sigma=sigma,
        mode="nearest",
    )
    response = np.divide(
        weighted,
        weights,
        out=np.zeros_like(weighted),
        where=weights > 0.5,
    )
    return response, weights


def _local_maximum_rank(
    response: np.ndarray,
    valid: np.ndarray,
    *,
    neighborhood_px: int,
) -> tuple[np.ndarray, dict[tuple[int, int], int]]:
    local_maximum = valid & (
        response == ndimage.maximum_filter(response, size=neighborhood_px, mode="nearest")
    )
    points = np.argwhere(local_maximum)
    ranked = sorted(
        ((int(y), int(x), float(response[y, x])) for y, x in points),
        key=lambda item: item[2],
        reverse=True,
    )
    ranks = {(y, x): rank for rank, (y, x, _value) in enumerate(ranked, start=1)}
    return local_maximum, ranks


def _pixel_class(value: float, *, repeated_codes: frozenset[int], negative_threshold: float) -> str:
    if int(value) in repeated_codes and float(value).is_integer():
        return "repeated_code"
    if value <= negative_threshold:
        return "negative_anomaly"
    return "ordinary"


def _contribution_rows(
    patch: np.ndarray,
    *,
    patch_x0: int,
    patch_y0: int,
    targets: Sequence[tuple[str, Detection, int, int]],
    background: float,
    sigma: float,
    repeated_codes: frozenset[int],
    negative_threshold: float,
) -> tuple[PairResponseContributionRow, ...]:
    """按离散 4σ Gaussian 核计算带符号像素归因。"""

    support_radius = max(1, int(math.ceil(4.0 * sigma)))
    rows: list[PairResponseContributionRow] = []
    height, width = patch.shape
    for label, source, peak_x, peak_y in targets:
        local_x = peak_x - patch_x0
        local_y = peak_y - patch_y0
        contributions = {
            name: {"count": 0, "weight": 0.0, "value": 0.0}
            for name in _PIXEL_CLASSES
        }
        total_weight = 0.0
        for dy in range(-support_radius, support_radius + 1):
            for dx in range(-support_radius, support_radius + 1):
                x = local_x + dx
                y = local_y + dy
                if not (0 <= x < width and 0 <= y < height):
                    continue
                weight = math.exp(-0.5 * (dx * dx + dy * dy) / (sigma * sigma))
                value = float(patch[y, x])
                category = _pixel_class(
                    value,
                    repeated_codes=repeated_codes,
                    negative_threshold=negative_threshold,
                )
                contributions[category]["count"] += 1
                contributions[category]["weight"] += weight
                contributions[category]["value"] += weight * (value - background)
                total_weight += weight
        if total_weight <= 0:
            raise ValueError("Gaussian support contains no pixels")
        for category in _PIXEL_CLASSES:
            data = contributions[category]
            rows.append(
                PairResponseContributionRow(
                    target_label=label,
                    detection_id=int(source.detection_id),
                    pixel_class=category,
                    pixel_count=int(data["count"]),
                    kernel_weight_fraction=float(data["weight"] / total_weight),
                    signed_response_contribution_adu=float(data["value"] / total_weight),
                )
            )
    return tuple(rows)


def run_pair_response_attribution(
    frame: str | Path | FitsFrame,
    primary: Detection,
    secondary: Detection,
    *,
    patch_padding_px: int = 10,
    psf_fwhm: float = 2.0,
    repeated_code_values: Sequence[int] | None = None,
    negative_anomaly_threshold_adu: float = -1000.0,
    maximum_neighborhood_px: int = 3,
    maximum_count: int = 12,
) -> PairResponseAttributionResult:
    """比较 raw、重复码留出和负异常留出下的局部匹配响应。"""

    if int(primary.detection_id) == int(secondary.detection_id):
        raise ValueError("primary and secondary detections must differ")
    if int(patch_padding_px) < 1:
        raise ValueError("patch_padding_px must be positive")
    if int(maximum_neighborhood_px) < 3 or int(maximum_neighborhood_px) % 2 == 0:
        raise ValueError("maximum_neighborhood_px must be an odd integer >= 3")
    if int(maximum_count) < 1:
        raise ValueError("maximum_count must be positive")
    fwhm = _finite_positive(psf_fwhm, "psf_fwhm")
    sigma = fwhm / _FWHM_TO_SIGMA
    negative_threshold = float(negative_anomaly_threshold_adu)
    if not math.isfinite(negative_threshold):
        raise ValueError("negative_anomaly_threshold_adu must be finite")

    fits_frame = frame if isinstance(frame, FitsFrame) else read_fits(frame)
    values = np.asarray(fits_frame.data, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"expected a 2-D FITS image, got shape {values.shape}")
    height, width = values.shape

    primary_x = _peak_coordinate(primary, "x")
    primary_y = _peak_coordinate(primary, "y")
    secondary_x = _peak_coordinate(secondary, "x")
    secondary_y = _peak_coordinate(secondary, "y")
    if (primary_x, primary_y) == (secondary_x, secondary_y):
        raise ValueError("primary and secondary peak coordinates must differ")
    x0 = max(0, min(primary_x, secondary_x) - int(patch_padding_px))
    y0 = max(0, min(primary_y, secondary_y) - int(patch_padding_px))
    x1 = min(width, max(primary_x, secondary_x) + int(patch_padding_px) + 1)
    y1 = min(height, max(primary_y, secondary_y) + int(patch_padding_px) + 1)
    patch = values[y0:y1, x0:x1]
    finite = np.isfinite(patch)
    if not np.any(finite):
        raise ValueError("pair patch contains no finite pixels")

    backgrounds = np.asarray((float(primary.background), float(secondary.background)), dtype=np.float64)
    noises = np.asarray((float(primary.noise), float(secondary.noise)), dtype=np.float64)
    if not np.isfinite(backgrounds).all() or not np.isfinite(noises).all() or np.any(noises <= 0):
        raise ValueError("pair detections must contain finite positive noise and finite background")
    common_background = float(np.median(backgrounds))
    common_noise = float(np.median(noises))
    repeated = set(int(value) for value in (repeated_code_values or ()))
    repeated.update(int(value) for value in primary.repeated_code_values)
    repeated.update(int(value) for value in secondary.repeated_code_values)
    repeated_codes = frozenset(repeated)

    code_mask = (
        np.isin(patch, np.asarray(tuple(sorted(repeated_codes)), dtype=np.float64))
        if repeated_codes
        else np.zeros_like(patch, dtype=bool)
    )
    negative_mask = finite & (patch <= negative_threshold)
    base_invalid = ~finite
    masks = {
        "raw": base_invalid,
        "repeated_code_masked": base_invalid | code_mask,
        "negative_anomaly_masked": base_invalid | negative_mask,
        "both_masked": base_invalid | code_mask | negative_mask,
    }
    residual = np.where(finite, patch - common_background, 0.0)
    target_specs = (
        ("primary", primary, primary_x, primary_y),
        ("secondary", secondary, secondary_x, secondary_y),
    )
    target_rows: list[PairResponseTargetRow] = []
    maximum_rows: list[PairResponseMaximumRow] = []
    target_coordinates = {
        "primary": (primary_x, primary_y),
        "secondary": (secondary_x, secondary_y),
    }
    for mode in PAIR_RESPONSE_MASK_MODES:
        response, weights = _response_and_weight(residual, masks[mode], sigma=sigma)
        valid = ~masks[mode]
        local_maximum, ranks = _local_maximum_rank(
            response,
            valid,
            neighborhood_px=int(maximum_neighborhood_px),
        )
        local_points = sorted(
            (
                float(response[y, x]),
                int(x),
                int(y),
            )
            for y, x in np.argwhere(local_maximum)
        )
        local_points.reverse()
        for rank, (response_value, local_x, local_y) in enumerate(local_points[: int(maximum_count)], start=1):
            absolute_x = local_x + x0
            absolute_y = local_y + y0
            maximum_rows.append(
                PairResponseMaximumRow(
                    mask_mode=mode,
                    rank=rank,
                    x=absolute_x,
                    y=absolute_y,
                    raw_value_adu=float(patch[local_y, local_x]),
                    gaussian_response_adu=response_value,
                    valid_weight=float(weights[local_y, local_x]),
                    is_primary_target=(absolute_x, absolute_y) == target_coordinates["primary"],
                    is_secondary_target=(absolute_x, absolute_y) == target_coordinates["secondary"],
                )
            )
        for label, source, peak_x, peak_y in target_specs:
            local_x = peak_x - x0
            local_y = peak_y - y0
            if not (0 <= local_x < patch.shape[1] and 0 <= local_y < patch.shape[0]):
                raise ValueError(f"target {label} lies outside pair patch")
            target_rows.append(
                PairResponseTargetRow(
                    mask_mode=mode,
                    target_label=label,
                    detection_id=int(source.detection_id),
                    peak_x=peak_x,
                    peak_y=peak_y,
                    raw_value_adu=float(patch[local_y, local_x]),
                    valid_at_target=bool(valid[local_y, local_x]),
                    gaussian_response_adu=float(response[local_y, local_x]),
                    valid_weight=float(weights[local_y, local_x]),
                    response_local_maximum=bool(local_maximum[local_y, local_x]),
                    local_maximum_rank=ranks.get((local_y, local_x)),
                )
            )

    contribution_rows = _contribution_rows(
        patch,
        patch_x0=x0,
        patch_y0=y0,
        targets=target_specs,
        background=common_background,
        sigma=sigma,
        repeated_codes=repeated_codes,
        negative_threshold=negative_threshold,
    )
    raw_targets = {row.target_label: row for row in target_rows if row.mask_mode == "raw"}
    code_targets = {row.target_label: row for row in target_rows if row.mask_mode == "repeated_code_masked"}
    negative_targets = {row.target_label: row for row in target_rows if row.mask_mode == "negative_anomaly_masked"}
    conclusion = (
        "局部 Gaussian 响应归因完成：raw、重复码留出和负异常留出分别改变了目标位置的有效性与局部极大值结构。"
        f"主/副 raw 响应为 {raw_targets['primary'].gaussian_response_adu:.3f}/"
        f"{raw_targets['secondary'].gaussian_response_adu:.3f} ADU；"
        f"重复码留出后主位置有效={code_targets['primary'].valid_at_target}，"
        f"负异常留出后副位置局部极大={negative_targets['secondary'].response_local_maximum}。"
        "这支持‘共享亮斑被值域缺口和匹配滤波重塑为两个响应’的优先解释，"
        "但不单独给出物理源数或噪点概率。"
    )
    return PairResponseAttributionResult(
        source_path=str(fits_frame.path),
        primary_detection_id=int(primary.detection_id),
        secondary_detection_id=int(secondary.detection_id),
        primary_peak_x=primary_x,
        primary_peak_y=primary_y,
        secondary_peak_x=secondary_x,
        secondary_peak_y=secondary_y,
        patch_x0=int(x0),
        patch_y0=int(y0),
        patch_x1=int(x1),
        patch_y1=int(y1),
        common_background_adu=common_background,
        common_noise_adu=common_noise,
        psf_fwhm_px=fwhm,
        repeated_code_values=tuple(sorted(repeated_codes)),
        negative_anomaly_threshold_adu=negative_threshold,
        target_rows=tuple(target_rows),
        maximum_rows=tuple(maximum_rows),
        contribution_rows=contribution_rows,
        parameters={
            "patch_padding_px": int(patch_padding_px),
            "gaussian_sigma_px": sigma,
            "maximum_neighborhood_px": int(maximum_neighborhood_px),
            "maximum_count": int(maximum_count),
            "mask_modes": list(PAIR_RESPONSE_MASK_MODES),
            "response_definition": (
                "masked Gaussian convolution of (raw - common background), "
                "normalized by valid Gaussian weight"
            ),
            "contribution_definition": (
                "discrete Gaussian weights within 4 sigma; signed raw ADU excess "
                "by mutually exclusive pixel class"
            ),
            "interpretation_boundary": (
                "local response attribution; not source truth, FDR, precision, "
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


def write_pair_response_attribution_artifacts(
    result: PairResponseAttributionResult,
    out_dir: str | Path,
) -> Path:
    """写出目标响应、局部极大值和像素归因表。"""

    output = Path(out_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    _write_rows(output / "pair_response_targets.csv", result.target_rows, PairResponseTargetRow)
    _write_rows(output / "pair_response_maxima.csv", result.maximum_rows, PairResponseMaximumRow)
    _write_rows(output / "pair_response_contributions.csv", result.contribution_rows, PairResponseContributionRow)
    (output / "pair_response_attribution.json").write_text(
        json.dumps(result.as_dict(), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return output


__all__ = [
    "PAIR_RESPONSE_MASK_MODES",
    "PairResponseAttributionResult",
    "PairResponseContributionRow",
    "PairResponseMaximumRow",
    "PairResponseTargetRow",
    "run_pair_response_attribution",
    "write_pair_response_attribution_artifacts",
]
