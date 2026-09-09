"""近邻双框的原始像素连通性与局部极大值审计。

该模块只回答一个局部问题：两个宽筛候选在原始像素中是否属于同一块
正响应结构，以及候选的整数峰是否真的是原始像素局部极大值。它不把
连通区域数当作物理源数；真实近邻 PSF 也可能在有限阈值下连成一块，
因此结果只用于解释候选分裂、共享结构和特殊值域污染。
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy import ndimage

from .detection import Detection
from .fits import FitsFrame, read_fits


@dataclass(frozen=True, slots=True)
class PairPixelTopologyComponentRow:
    """一个阈值和像素有效性口径下的连通区域结果。"""

    mode: str
    sigma_level: float
    threshold_adu: float
    positive_pixel_count: int
    component_count: int
    primary_component_id: int
    secondary_component_id: int
    primary_component_size: int
    secondary_component_size: int
    same_nonzero_component: bool
    primary_peak_value_adu: float
    secondary_peak_value_adu: float
    primary_peak_is_positive: bool
    secondary_peak_is_positive: bool
    special_pixel_count: int

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PairPixelTopologyMaximumRow:
    """原始像素局部极大值与候选峰的关系。"""

    neighborhood_px: int
    threshold_adu: float
    local_maximum_count: int
    primary_peak_is_raw_local_maximum: bool
    secondary_peak_is_raw_local_maximum: bool
    primary_nearest_raw_maximum_px: float | None
    secondary_nearest_raw_maximum_px: float | None
    top_raw_maxima_json: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PairPixelTopologyAuditResult:
    """原始像素拓扑审计结果。"""

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
    repeated_code_values: tuple[int, ...]
    special_negative_threshold_adu: float
    brightest_raw_x: int
    brightest_raw_y: int
    brightest_raw_value_adu: float
    component_rows: tuple[PairPixelTopologyComponentRow, ...]
    maximum_rows: tuple[PairPixelTopologyMaximumRow, ...]
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
            "repeated_code_values": list(self.repeated_code_values),
            "special_negative_threshold_adu": self.special_negative_threshold_adu,
            "brightest_raw": {
                "x": self.brightest_raw_x,
                "y": self.brightest_raw_y,
                "value_adu": self.brightest_raw_value_adu,
            },
            "component_rows": [row.as_dict() for row in self.component_rows],
            "maximum_rows": [row.as_dict() for row in self.maximum_rows],
            "parameters": dict(self.parameters),
            "conclusion": self.conclusion,
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


def _component_id_and_size(labels: np.ndarray, x: int, y: int) -> tuple[int, int]:
    px = int(x)
    py = int(y)
    if not (0 <= py < labels.shape[0] and 0 <= px < labels.shape[1]):
        return 0, 0
    component_id = int(labels[py, px])
    if component_id <= 0:
        return 0, 0
    counts = np.bincount(labels.reshape(-1))
    size = int(counts[component_id]) if component_id < len(counts) else 0
    return component_id, size


def _point_in_patch(x: int, y: int, x0: int, y0: int) -> tuple[int, int]:
    return int(x - x0), int(y - y0)


def _nearest_distance(
    point: tuple[int, int],
    candidates: np.ndarray,
) -> float | None:
    if candidates.size == 0:
        return None
    dx = candidates[:, 1].astype(np.float64) - float(point[0])
    dy = candidates[:, 0].astype(np.float64) - float(point[1])
    return float(np.min(np.hypot(dx, dy)))


def run_pair_pixel_topology_audit(
    frame: str | Path | FitsFrame,
    primary: Detection,
    secondary: Detection,
    *,
    patch_padding_px: int = 10,
    sigma_levels: Sequence[float] = (3.0, 5.0, 8.0, 10.0),
    raw_maximum_neighborhoods_px: Sequence[int] = (3, 5, 7),
    raw_maximum_sigma_level: float = 3.0,
    repeated_code_values: Sequence[int] | None = None,
    special_negative_threshold_adu: float = -1000.0,
) -> PairPixelTopologyAuditResult:
    """在 pair 局部窗口内比较原始像素连通块和候选峰。"""

    if int(primary.detection_id) == int(secondary.detection_id):
        raise ValueError("primary and secondary detections must differ")
    if patch_padding_px < 1:
        raise ValueError("patch_padding_px must be positive")
    levels = tuple(sorted({float(level) for level in sigma_levels}))
    if not levels or any(not math.isfinite(level) or level <= 0 for level in levels):
        raise ValueError("sigma_levels must contain finite positive values")
    neighborhoods = tuple(sorted({int(size) for size in raw_maximum_neighborhoods_px}))
    if not neighborhoods or any(size < 3 or size % 2 == 0 for size in neighborhoods):
        raise ValueError("raw_maximum_neighborhoods_px must contain odd integers >= 3")
    maximum_sigma = _finite_positive(raw_maximum_sigma_level, "raw_maximum_sigma_level")
    negative_threshold = float(special_negative_threshold_adu)
    if not math.isfinite(negative_threshold):
        raise ValueError("special_negative_threshold_adu must be finite")

    if isinstance(frame, FitsFrame):
        fits_frame = frame
    else:
        fits_frame = read_fits(frame)
    values = np.asarray(fits_frame.data, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"expected a 2-D FITS image, got shape {values.shape}")

    primary_x = _peak_coordinate(primary, "x")
    primary_y = _peak_coordinate(primary, "y")
    secondary_x = _peak_coordinate(secondary, "x")
    secondary_y = _peak_coordinate(secondary, "y")
    if (primary_x, primary_y) == (secondary_x, secondary_y):
        raise ValueError("primary and secondary peak coordinates must differ")
    height, width = values.shape
    x0 = max(0, min(primary_x, secondary_x) - int(patch_padding_px))
    y0 = max(0, min(primary_y, secondary_y) - int(patch_padding_px))
    x1 = min(width, max(primary_x, secondary_x) + int(patch_padding_px) + 1)
    y1 = min(height, max(primary_y, secondary_y) + int(patch_padding_px) + 1)
    patch = values[y0:y1, x0:x1]
    finite = np.isfinite(patch)
    if not np.any(finite):
        raise ValueError("pair patch contains no finite pixels")

    backgrounds = np.asarray([float(primary.background), float(secondary.background)], dtype=np.float64)
    noises = np.asarray([float(primary.noise), float(secondary.noise)], dtype=np.float64)
    if not np.isfinite(backgrounds).all() or not np.isfinite(noises).all() or np.any(noises <= 0):
        raise ValueError("pair detections must contain finite positive noise and finite background")
    common_background = float(np.median(backgrounds))
    common_noise = float(np.median(noises))
    repeated = set(int(value) for value in (repeated_code_values or ()))
    repeated.update(int(value) for value in primary.repeated_code_values)
    repeated.update(int(value) for value in secondary.repeated_code_values)
    repeated_tuple = tuple(sorted(repeated))
    special_mask = (~finite) | np.isin(patch, np.asarray(repeated_tuple, dtype=np.float64))
    special_mask |= finite & (patch <= negative_threshold)

    primary_point = _point_in_patch(primary_x, primary_y, x0, y0)
    secondary_point = _point_in_patch(secondary_x, secondary_y, x0, y0)
    primary_peak_value = float(patch[primary_point[1], primary_point[0]])
    secondary_peak_value = float(patch[secondary_point[1], secondary_point[0]])
    component_rows: list[PairPixelTopologyComponentRow] = []
    structure = np.ones((3, 3), dtype=bool)
    for mode, excluded in (("raw", np.zeros_like(patch, dtype=bool)), ("special_excluded", special_mask)):
        for sigma_level in levels:
            threshold = common_background + float(sigma_level) * common_noise
            positive = finite & ~excluded & ((patch - common_background) >= threshold - common_background)
            labels, count = ndimage.label(positive, structure=structure)
            primary_id, primary_size = _component_id_and_size(labels, *primary_point)
            secondary_id, secondary_size = _component_id_and_size(labels, *secondary_point)
            component_rows.append(
                PairPixelTopologyComponentRow(
                    mode=mode,
                    sigma_level=float(sigma_level),
                    threshold_adu=float(threshold),
                    positive_pixel_count=int(np.count_nonzero(positive)),
                    component_count=int(count),
                    primary_component_id=primary_id,
                    secondary_component_id=secondary_id,
                    primary_component_size=primary_size,
                    secondary_component_size=secondary_size,
                    same_nonzero_component=bool(
                        primary_id > 0 and primary_id == secondary_id
                    ),
                    primary_peak_value_adu=primary_peak_value,
                    secondary_peak_value_adu=secondary_peak_value,
                    primary_peak_is_positive=bool(positive[primary_point[1], primary_point[0]]),
                    secondary_peak_is_positive=bool(positive[secondary_point[1], secondary_point[0]]),
                    special_pixel_count=int(np.count_nonzero(excluded)),
                )
            )

    maximum_threshold = common_background + maximum_sigma * common_noise
    finite_values = np.where(finite, patch, -np.inf)
    maximum_rows: list[PairPixelTopologyMaximumRow] = []
    for neighborhood in neighborhoods:
        local_maximum = finite & (finite_values == ndimage.maximum_filter(
            finite_values,
            size=int(neighborhood),
            mode="nearest",
        )) & (finite_values >= maximum_threshold)
        maximum_points = np.argwhere(local_maximum)
        ranked = sorted(
            (
                {
                    "x": int(x + x0),
                    "y": int(y + y0),
                    "value_adu": float(patch[y, x]),
                }
                for y, x in maximum_points
            ),
            key=lambda item: float(item["value_adu"]),
            reverse=True,
        )
        maximum_rows.append(
            PairPixelTopologyMaximumRow(
                neighborhood_px=int(neighborhood),
                threshold_adu=float(maximum_threshold),
                local_maximum_count=int(len(maximum_points)),
                primary_peak_is_raw_local_maximum=bool(local_maximum[primary_point[1], primary_point[0]]),
                secondary_peak_is_raw_local_maximum=bool(local_maximum[secondary_point[1], secondary_point[0]]),
                primary_nearest_raw_maximum_px=_nearest_distance(primary_point, maximum_points),
                secondary_nearest_raw_maximum_px=_nearest_distance(secondary_point, maximum_points),
                top_raw_maxima_json=json.dumps(ranked[:12], ensure_ascii=False, separators=(",", ":")),
            )
        )

    raw_rows = [row for row in component_rows if row.mode == "raw"]
    same_raw_count = sum(row.same_nonzero_component for row in raw_rows)
    primary_local_max_count = sum(row.primary_peak_is_raw_local_maximum for row in maximum_rows)
    secondary_local_max_count = sum(row.secondary_peak_is_raw_local_maximum for row in maximum_rows)
    brightest_y, brightest_x = np.unravel_index(int(np.nanargmax(np.where(finite, patch, np.nan))), patch.shape)
    conclusion = (
        f"局部像素拓扑审计完成：raw 在 {same_raw_count}/{len(raw_rows)} 个阈值下将两个候选放在同一"
        f"个非零正响应连通块；原始局部极大值口径下，主/副候选分别只有 "
        f"{primary_local_max_count}/{len(maximum_rows)}、{secondary_local_max_count}/{len(maximum_rows)} "
        "次是局部极大值。该结果支持共享亮斑/结构混合或匹配响应重定位的优先解释，"
        "但不单独排除真实近邻 PSF，因为真实近邻也可能在阈值下连成一块。"
    )
    return PairPixelTopologyAuditResult(
        source_path=str(fits_frame.path),
        primary_detection_id=int(primary.detection_id),
        secondary_detection_id=int(secondary.detection_id),
        primary_peak_x=int(primary_x),
        primary_peak_y=int(primary_y),
        secondary_peak_x=int(secondary_x),
        secondary_peak_y=int(secondary_y),
        patch_x0=int(x0),
        patch_y0=int(y0),
        patch_x1=int(x1),
        patch_y1=int(y1),
        common_background_adu=common_background,
        common_noise_adu=common_noise,
        repeated_code_values=repeated_tuple,
        special_negative_threshold_adu=negative_threshold,
        brightest_raw_x=int(brightest_x + x0),
        brightest_raw_y=int(brightest_y + y0),
        brightest_raw_value_adu=float(patch[brightest_y, brightest_x]),
        component_rows=tuple(component_rows),
        maximum_rows=tuple(maximum_rows),
        parameters={
            "patch_padding_px": int(patch_padding_px),
            "connectivity": 8,
            "sigma_levels": list(levels),
            "raw_maximum_neighborhoods_px": list(neighborhoods),
            "raw_maximum_sigma_level": maximum_sigma,
            "threshold_definition": "median(source backgrounds) + sigma * median(source noises)",
            "special_value_definition": "repeated code values or raw value <= special_negative_threshold_adu",
            "interpretation_boundary": "pixel topology diagnostic, not source truth or star probability",
        },
        conclusion=conclusion,
    )


def write_pair_pixel_topology_artifacts(
    result: PairPixelTopologyAuditResult,
    out_dir: str | Path,
) -> Path:
    """写局部像素拓扑 CSV、JSON 和可视化。"""

    output = Path(out_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    component_fields = list(result.component_rows[0].as_dict()) if result.component_rows else ["mode"]
    with (output / "pair_pixel_topology_components.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=component_fields)
        writer.writeheader()
        writer.writerows(row.as_dict() for row in result.component_rows)
    maximum_fields = list(result.maximum_rows[0].as_dict()) if result.maximum_rows else ["neighborhood_px"]
    with (output / "pair_pixel_topology_maxima.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=maximum_fields)
        writer.writeheader()
        writer.writerows(row.as_dict() for row in result.maximum_rows)
    (output / "pair_pixel_topology.json").write_text(
        json.dumps(result.as_dict(), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    frame = read_fits(result.source_path)
    patch = np.asarray(frame.data[result.patch_y0:result.patch_y1, result.patch_x0:result.patch_x1], dtype=np.float64)
    common_threshold = result.common_background_adu + 3.0 * result.common_noise_adu
    raw_mask = np.isfinite(patch) & (patch >= common_threshold)
    excluded = np.isin(patch, np.asarray(result.repeated_code_values, dtype=np.float64)) | (
        np.isfinite(patch) & (patch <= result.special_negative_threshold_adu)
    )
    excluded_mask = raw_mask & ~excluded
    _write_topology_svg(
        output / "pair_pixel_topology.svg",
        patch,
        excluded_mask,
        result,
    )
    return output


def _write_topology_svg(
    path: Path,
    patch: np.ndarray,
    excluded_mask: np.ndarray,
    result: PairPixelTopologyAuditResult,
) -> None:
    """用标准 SVG 写轻量可视化，避免把 matplotlib 作为运行依赖。"""

    finite = patch[np.isfinite(patch)]
    if finite.size == 0:
        raise ValueError("cannot render an empty topology patch")
    lower, upper = np.percentile(finite, [1.0, 99.0])
    if not math.isfinite(float(lower)) or not math.isfinite(float(upper)) or upper <= lower:
        lower = float(np.min(finite))
        upper = float(np.max(finite))
    upper = max(float(upper), float(lower) + 1.0)
    rows, columns = patch.shape
    cell = 18
    gap = 48
    panel_width = columns * cell
    panel_height = rows * cell
    total_width = panel_width * 2 + gap * 3
    total_height = panel_height + 70
    primary = (result.primary_peak_x - result.patch_x0, result.primary_peak_y - result.patch_y0)
    secondary = (result.secondary_peak_x - result.patch_x0, result.secondary_peak_y - result.patch_y0)
    brightest = (result.brightest_raw_x - result.patch_x0, result.brightest_raw_y - result.patch_y0)

    def color(value: float) -> str:
        clipped = min(1.0, max(0.0, (float(value) - lower) / (upper - lower)))
        channel = int(round(24.0 + 220.0 * clipped))
        return f"rgb({channel},{channel},{channel})"

    def panel(origin_x: int, image: np.ndarray, title: str, positive_only: bool) -> list[str]:
        lines = [f'<text x="{origin_x}" y="18" fill="#d8dee9" font-size="13">{title}</text>']
        for row in range(rows):
            for column in range(columns):
                value = float(image[row, column])
                visible = math.isfinite(value) and (not positive_only or bool(excluded_mask[row, column]))
                fill = color(value) if visible else "#121923"
                lines.append(
                    f'<rect x="{origin_x + column * cell}" y="{30 + row * cell}" '
                    f'width="{cell}" height="{cell}" fill="{fill}" stroke="#283342" stroke-width="0.35"/>'
                )
        for x, y, stroke, label in (
            (primary[0], primary[1], "#f0b54d", "P"),
            (secondary[0], secondary[1], "#61d7d0", "S"),
        ):
            if 0 <= x < columns and 0 <= y < rows:
                cx = origin_x + x * cell + cell / 2.0
                cy = 30 + y * cell + cell / 2.0
                lines.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="7" fill="none" stroke="{stroke}" stroke-width="2"/>')
                lines.append(f'<text x="{cx + 8:.1f}" y="{cy + 4:.1f}" fill="{stroke}" font-size="10">{label}</text>')
        if 0 <= brightest[0] < columns and 0 <= brightest[1] < rows:
            cx = origin_x + brightest[0] * cell + cell / 2.0
            cy = 30 + brightest[1] * cell + cell / 2.0
            lines.append(f'<path d="M {cx - 6:.1f} {cy:.1f} L {cx + 6:.1f} {cy:.1f} M {cx:.1f} {cy - 6:.1f} L {cx:.1f} {cy + 6:.1f}" stroke="#ff5b5b" stroke-width="1.5"/>')
        return lines

    origin_a = gap
    origin_b = gap * 2 + panel_width
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{total_width}" height="{total_height}" viewBox="0 0 {total_width} {total_height}">',
        '<rect width="100%" height="100%" fill="#0d131c"/>',
        f'<text x="{gap}" y="{total_height - 18}" fill="#9aa7b8" font-size="11">P=primary peak · S=secondary peak · red cross=brightest raw pixel · patch origin=({result.patch_x0},{result.patch_y0})</text>',
    ]
    lines.extend(panel(origin_a, patch, "raw local patch", False))
    lines.extend(panel(origin_b, patch, "3σ after special-value exclusion", True))
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")
