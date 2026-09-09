"""审计候选峰与原始像素局部峰的一致性。

宽筛检测通常在匹配滤波响应图上寻找局部极大值，随后再回到原始图像做
测光和形态测量。两种图上的极大值不是同一个对象：滤波会把邻近像素加权
求和，异常值、混合源和 PSF 失配都可能让响应图在原始像素峰之外产生候选。

本模块只做一个可复核的 detector-level 对照：把现有 ``source_catalog.csv``
中的候选峰坐标放回原始 FITS，在 3x3、5x5、7x7 窗口内检查它是否为原始
像素的局部极大值，以及邻域中更高像素的差值。它不把原始局部极大值当作
真星充分条件，也不改变检测缓存、质量层或物理恒星计数。
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from scipy import ndimage

from .fits import auxiliary_mask, read_fits


DEFAULT_LOCAL_MAX_RADII: tuple[int, ...] = (1, 2, 3)


@dataclass(frozen=True, slots=True)
class SourcePeakConsistencyRow:
    """一个候选在原始像素局部峰审计中的逐源记录。"""

    detection_id: int
    feature_class: str
    feature_class_label: str
    quality_passed: bool
    peak_x: int
    peak_y: int
    raw_peak_adu: float
    raw_valid: bool
    raw_local_maximum_r1: bool
    raw_unique_local_maximum_r1: bool
    raw_local_max_adu_r1: float
    raw_local_max_gap_adu_r1: float
    raw_local_maximum_r2: bool
    raw_unique_local_maximum_r2: bool
    raw_local_max_adu_r2: float
    raw_local_max_gap_adu_r2: float
    raw_local_maximum_r3: bool
    raw_unique_local_maximum_r3: bool
    raw_local_max_adu_r3: float
    raw_local_max_gap_adu_r3: float

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


SOURCE_PEAK_CONSISTENCY_FIELDS = tuple(field.name for field in fields(SourcePeakConsistencyRow))


@dataclass(frozen=True, slots=True)
class SourcePeakConsistencyClassSummary:
    """一个特征类别的局部峰一致性汇总。"""

    feature_class: str
    feature_class_label: str
    candidate_count: int
    quality_count: int
    raw_valid_count: int
    raw_local_maximum_count_r1: int
    raw_local_maximum_fraction_r1: float | None
    raw_local_maximum_count_r2: int
    raw_local_maximum_fraction_r2: float | None
    raw_local_maximum_count_r3: int
    raw_local_maximum_fraction_r3: float | None
    raw_unique_local_maximum_count_r1: int
    raw_unique_local_maximum_count_r2: int
    raw_unique_local_maximum_count_r3: int
    nonlocal_peak_count_r1: int
    nonlocal_peak_gap_median_adu_r1: float | None
    nonlocal_peak_gap_p90_adu_r1: float | None
    nonlocal_peak_count_r2: int
    nonlocal_peak_gap_median_adu_r2: float | None
    nonlocal_peak_gap_p90_adu_r2: float | None
    nonlocal_peak_count_r3: int
    nonlocal_peak_gap_median_adu_r3: float | None
    nonlocal_peak_gap_p90_adu_r3: float | None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


SOURCE_PEAK_CONSISTENCY_CLASS_FIELDS = tuple(
    field.name for field in fields(SourcePeakConsistencyClassSummary)
)


@dataclass(frozen=True, slots=True)
class SourcePeakConsistencyTargetRow:
    """指定目标在不同局部窗口中的最大像素位置和差值。"""

    detection_id: int
    feature_class: str
    feature_class_label: str
    quality_passed: bool
    peak_x: int
    peak_y: int
    radius_px: int
    local_max_x: int
    local_max_y: int
    raw_peak_adu: float
    local_max_adu: float
    local_max_gap_adu: float
    local_max_offset_px: float
    raw_local_maximum: bool
    raw_unique_local_maximum: bool

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


SOURCE_PEAK_CONSISTENCY_TARGET_FIELDS = tuple(
    field.name for field in fields(SourcePeakConsistencyTargetRow)
)


@dataclass(frozen=True, slots=True)
class SourcePeakConsistencyResult:
    """峰一致性审计的完整结果。"""

    catalog_path: str
    fits_path: str
    image_shape: tuple[int, int]
    local_max_radii_px: tuple[int, ...]
    rows: tuple[SourcePeakConsistencyRow, ...]
    class_summaries: tuple[SourcePeakConsistencyClassSummary, ...]
    targets: tuple[SourcePeakConsistencyTargetRow, ...]
    conclusion: str

    def as_dict(self) -> dict[str, object]:
        return {
            "catalog_path": self.catalog_path,
            "fits_path": self.fits_path,
            "image_shape": list(self.image_shape),
            "local_max_radii_px": list(self.local_max_radii_px),
            "rows": [row.as_dict() for row in self.rows],
            "class_summaries": [row.as_dict() for row in self.class_summaries],
            "targets": [row.as_dict() for row in self.targets],
            "conclusion": self.conclusion,
            "interpretation_boundary": (
                "raw 局部极大值只是像素拓扑诊断，不是恒星的充分条件；滤波候选可能因 PSF 混叠、探测器值域、"
                "背景结构或模型失配而偏离原始像素峰；不推导 precision、FDR、恒星概率或物理源数"
            ),
        }


_REQUIRED_COLUMNS = {
    "detection_id",
    "feature_class",
    "feature_class_label",
    "quality_passed",
    "peak_x",
    "peak_y",
}


def _required_int(row: Mapping[str, str], key: str, row_number: int) -> int:
    value = str(row.get(key, "")).strip()
    if not value:
        raise ValueError(f"catalog row {row_number}: empty {key}")
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(f"catalog row {row_number}: invalid {key}={value!r}") from exc
    if not np.isfinite(parsed) or not parsed.is_integer():
        raise ValueError(f"catalog row {row_number}: {key} must be an integer, got {value!r}")
    return int(parsed)


def _required_float(row: Mapping[str, str], key: str, row_number: int) -> float:
    value = str(row.get(key, "")).strip()
    if not value:
        raise ValueError(f"catalog row {row_number}: empty {key}")
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(f"catalog row {row_number}: invalid {key}={value!r}") from exc
    if not np.isfinite(parsed):
        raise ValueError(f"catalog row {row_number}: {key} must be finite, got {value!r}")
    return parsed


def _required_bool(row: Mapping[str, str], key: str, row_number: int) -> bool:
    value = str(row.get(key, "")).strip().lower()
    if value in {"true", "1", "yes"}:
        return True
    if value in {"false", "0", "no"}:
        return False
    raise ValueError(f"catalog row {row_number}: invalid {key}={value!r}")


def _load_catalog(path: Path) -> tuple[dict[str, object], ...]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        columns = set(reader.fieldnames or ())
        missing = sorted(_REQUIRED_COLUMNS - columns)
        if missing:
            raise ValueError(f"source catalog is missing required columns: {', '.join(missing)}")
        records: list[dict[str, object]] = []
        seen_ids: set[int] = set()
        for row_number, raw in enumerate(reader, start=2):
            detection_id = _required_int(raw, "detection_id", row_number)
            if detection_id in seen_ids:
                raise ValueError(f"source catalog contains duplicate detection_id={detection_id}")
            seen_ids.add(detection_id)
            records.append(
                {
                    "detection_id": detection_id,
                    "feature_class": str(raw.get("feature_class", "")).strip(),
                    "feature_class_label": str(raw.get("feature_class_label", "")).strip(),
                    "quality_passed": _required_bool(raw, "quality_passed", row_number),
                    "peak_x": _required_int(raw, "peak_x", row_number),
                    "peak_y": _required_int(raw, "peak_y", row_number),
                }
            )
            if not records[-1]["feature_class"]:
                raise ValueError(f"catalog row {row_number}: empty feature_class")
    if not records:
        raise ValueError("source catalog is empty")
    return tuple(records)


def _validate_radii(radii: Sequence[int]) -> tuple[int, ...]:
    parsed = tuple(int(radius) for radius in radii)
    if parsed != DEFAULT_LOCAL_MAX_RADII:
        raise ValueError(
            "source peak consistency uses the stable radius set (1, 2, 3); "
            f"got {parsed!r}"
        )
    return parsed


def _masked_image(image: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """为最大值滤波准备不会把辅助区当成亮点的数组。"""

    finite = np.isfinite(image)
    valid = np.asarray(valid, dtype=bool) & finite
    if np.issubdtype(image.dtype, np.integer):
        sentinel = np.iinfo(image.dtype).min
        return np.where(valid, image, sentinel)
    return np.where(valid, image, -np.inf)


def _local_max_location(
    image: np.ndarray,
    valid: np.ndarray,
    x: int,
    y: int,
    radius: int,
) -> tuple[int, int, float, bool, bool]:
    """返回一个目标窗口内的局部最大位置；并列时取窗口扫描顺序的首个值。"""

    y0 = max(0, y - radius)
    y1 = min(image.shape[0], y + radius + 1)
    x0 = max(0, x - radius)
    x1 = min(image.shape[1], x + radius + 1)
    patch = np.asarray(image[y0:y1, x0:x1])
    patch_valid = np.asarray(valid[y0:y1, x0:x1], dtype=bool) & np.isfinite(patch)
    if not bool(valid[y, x]) or not np.any(patch_valid):
        return x, y, float("nan"), False, False
    work = np.where(patch_valid, patch, -np.inf)
    flat_index = int(np.argmax(work))
    local_y, local_x = np.unravel_index(flat_index, work.shape)
    local_x = int(local_x + x0)
    local_y = int(local_y + y0)
    local_max = float(work[local_y - y0, local_x - x0])
    raw_value = float(image[y, x])
    equal_count = int(np.count_nonzero(patch_valid & (patch == local_max)))
    return (
        local_x,
        local_y,
        local_max,
        bool(np.isclose(raw_value, local_max, rtol=0.0, atol=0.0)),
        bool(np.isclose(raw_value, local_max, rtol=0.0, atol=0.0) and equal_count == 1),
    )


def _target_rows(
    records: Sequence[Mapping[str, object]],
    image: np.ndarray,
    valid: np.ndarray,
    local_maxima: Mapping[int, np.ndarray],
    unique_maxima: Mapping[int, np.ndarray],
    target_ids: Sequence[int],
) -> tuple[SourcePeakConsistencyTargetRow, ...]:
    by_id = {int(record["detection_id"]): record for record in records}
    missing = sorted(set(int(value) for value in target_ids) - set(by_id))
    if missing:
        raise ValueError(f"target detection_id not in source catalog: {', '.join(map(str, missing))}")
    rows: list[SourcePeakConsistencyTargetRow] = []
    for detection_id in dict.fromkeys(int(value) for value in target_ids):
        record = by_id[detection_id]
        x = int(record["peak_x"])
        y = int(record["peak_y"])
        raw_value = float(image[y, x])
        for radius in DEFAULT_LOCAL_MAX_RADII:
            local_x, local_y, local_max, is_max, is_unique = _local_max_location(
                image,
                valid,
                x,
                y,
                radius,
            )
            rows.append(
                SourcePeakConsistencyTargetRow(
                    detection_id=detection_id,
                    feature_class=str(record["feature_class"]),
                    feature_class_label=str(record["feature_class_label"]),
                    quality_passed=bool(record["quality_passed"]),
                    peak_x=x,
                    peak_y=y,
                    radius_px=radius,
                    local_max_x=local_x,
                    local_max_y=local_y,
                    raw_peak_adu=raw_value,
                    local_max_adu=local_max,
                    local_max_gap_adu=max(0.0, local_max - raw_value) if np.isfinite(local_max) else float("nan"),
                    local_max_offset_px=float(np.hypot(local_x - x, local_y - y)),
                    raw_local_maximum=bool(local_maxima[radius][y, x]),
                    raw_unique_local_maximum=bool(unique_maxima[radius][y, x]),
                )
            )
    return tuple(rows)


def run_source_peak_consistency(
    catalog_path: str | Path,
    fits_path: str | Path,
    *,
    target_ids: Sequence[int] = (),
    local_max_radii_px: Sequence[int] = DEFAULT_LOCAL_MAX_RADII,
) -> SourcePeakConsistencyResult:
    """把首帧候选放回原始 FITS，统计原始像素局部峰的一致性。"""

    radii = _validate_radii(local_max_radii_px)
    catalog = Path(catalog_path).resolve()
    fits = Path(fits_path).resolve()
    records = _load_catalog(catalog)
    frame = read_fits(fits)
    image = np.asarray(frame.data)
    if image.ndim != 2:
        raise ValueError(f"expected a 2-D FITS image, got shape {image.shape}")
    valid = ~auxiliary_mask(image.shape)
    valid &= np.isfinite(image)
    coordinates = [(int(record["peak_x"]), int(record["peak_y"])) for record in records]
    for record, (x, y) in zip(records, coordinates, strict=True):
        if not (0 <= x < image.shape[1] and 0 <= y < image.shape[0]):
            raise ValueError(
                f"detection_id={record['detection_id']} peak coordinate {(x, y)} "
                f"is outside image shape {image.shape}"
            )

    masked = _masked_image(image, valid)
    local_maxima: dict[int, np.ndarray] = {}
    unique_maxima: dict[int, np.ndarray] = {}
    max_values: dict[int, np.ndarray] = {}
    for radius in radii:
        size = 2 * radius + 1
        local_max = ndimage.maximum_filter(masked, size=size, mode="nearest")
        equal_max = valid & (image == local_max)
        equal_count = ndimage.convolve(
            equal_max.astype(np.int8),
            np.ones((size, size), dtype=np.int8),
            mode="nearest",
        )
        local_maxima[radius] = equal_max
        unique_maxima[radius] = equal_max & (equal_count == 1)
        max_values[radius] = local_max

    row_values: list[SourcePeakConsistencyRow] = []
    for record, (x, y) in zip(records, coordinates, strict=True):
        raw_value = float(image[y, x])
        row_kwargs: dict[str, object] = {
            "detection_id": int(record["detection_id"]),
            "feature_class": str(record["feature_class"]),
            "feature_class_label": str(record["feature_class_label"]),
            "quality_passed": bool(record["quality_passed"]),
            "peak_x": x,
            "peak_y": y,
            "raw_peak_adu": raw_value,
            "raw_valid": bool(valid[y, x]),
        }
        for radius in radii:
            max_value = float(max_values[radius][y, x])
            gap = max(0.0, max_value - raw_value) if np.isfinite(max_value) else float("nan")
            row_kwargs[f"raw_local_maximum_r{radius}"] = bool(local_maxima[radius][y, x])
            row_kwargs[f"raw_unique_local_maximum_r{radius}"] = bool(unique_maxima[radius][y, x])
            row_kwargs[f"raw_local_max_adu_r{radius}"] = max_value
            row_kwargs[f"raw_local_max_gap_adu_r{radius}"] = gap
        row_values.append(SourcePeakConsistencyRow(**row_kwargs))

    class_summaries: list[SourcePeakConsistencyClassSummary] = []
    classes = sorted({str(record["feature_class"]) for record in records})
    for feature_class in classes:
        indices = np.asarray(
            [index for index, record in enumerate(records) if str(record["feature_class"]) == feature_class],
            dtype=np.intp,
        )
        count = int(indices.size)
        quality_count = int(sum(bool(records[index]["quality_passed"]) for index in indices))
        valid_count = int(sum(bool(valid[coordinates[index][1], coordinates[index][0]]) for index in indices))
        label = next(str(record["feature_class_label"]) for record in records if str(record["feature_class"]) == feature_class)
        summary_kwargs: dict[str, object] = {
            "feature_class": feature_class,
            "feature_class_label": label,
            "candidate_count": count,
            "quality_count": quality_count,
            "raw_valid_count": valid_count,
        }
        for radius in radii:
            local_flags = np.asarray(
                [bool(local_maxima[radius][coordinates[index][1], coordinates[index][0]]) for index in indices],
                dtype=bool,
            )
            unique_flags = np.asarray(
                [bool(unique_maxima[radius][coordinates[index][1], coordinates[index][0]]) for index in indices],
                dtype=bool,
            )
            gaps = np.asarray(
                [
                    float(max_values[radius][coordinates[index][1], coordinates[index][0]])
                    - float(image[coordinates[index][1], coordinates[index][0]])
                    for index in indices
                ],
                dtype=np.float64,
            )
            gaps = np.maximum(gaps, 0.0)
            nonlocal_gaps = gaps[~local_flags]
            summary_kwargs[f"raw_local_maximum_count_r{radius}"] = int(np.count_nonzero(local_flags))
            summary_kwargs[f"raw_local_maximum_fraction_r{radius}"] = (
                float(np.count_nonzero(local_flags) / count) if count else None
            )
            summary_kwargs[f"raw_unique_local_maximum_count_r{radius}"] = int(np.count_nonzero(unique_flags))
            summary_kwargs[f"nonlocal_peak_count_r{radius}"] = int(nonlocal_gaps.size)
            summary_kwargs[f"nonlocal_peak_gap_median_adu_r{radius}"] = (
                float(np.median(nonlocal_gaps)) if nonlocal_gaps.size else None
            )
            summary_kwargs[f"nonlocal_peak_gap_p90_adu_r{radius}"] = (
                float(np.quantile(nonlocal_gaps, 0.90)) if nonlocal_gaps.size else None
            )
        class_summaries.append(SourcePeakConsistencyClassSummary(**summary_kwargs))

    targets = _target_rows(records, image, valid, local_maxima, unique_maxima, target_ids)
    target_text = ""
    if targets:
        target_text = "；".join(
            f"ID {target.detection_id} 在 r={target.radius_px} px 的原始局部极大值={target.raw_local_maximum}"
            for target in targets
        )
    conclusion = (
        "原始像素局部峰审计显示，匹配滤波候选与 raw 像素峰可以发生位置分歧；"
        "该分歧在范围异常、线状或拥挤结构中应作为专项复核信号，而不能用 raw 局部极大值"
        "单独决定真星。目标结果" + (f"：{target_text}。" if target_text else "未指定目标。")
    )
    return SourcePeakConsistencyResult(
        catalog_path=str(catalog),
        fits_path=str(fits),
        image_shape=tuple(int(value) for value in image.shape),
        local_max_radii_px=radii,
        rows=tuple(row_values),
        class_summaries=tuple(class_summaries),
        targets=targets,
        conclusion=conclusion,
    )


def write_source_peak_consistency_artifacts(
    result: SourcePeakConsistencyResult,
    output_dir: str | Path,
) -> Path:
    """写出逐源、类别和目标级 CSV/JSON 机器产物。"""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "source_peak_consistency.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=SOURCE_PEAK_CONSISTENCY_FIELDS)
        writer.writeheader()
        writer.writerows(row.as_dict() for row in result.rows)
    with (output / "source_peak_consistency_class_summary.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=SOURCE_PEAK_CONSISTENCY_CLASS_FIELDS)
        writer.writeheader()
        writer.writerows(row.as_dict() for row in result.class_summaries)
    with (output / "source_peak_consistency_targets.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=SOURCE_PEAK_CONSISTENCY_TARGET_FIELDS)
        writer.writeheader()
        writer.writerows(row.as_dict() for row in result.targets)
    with (output / "source_peak_consistency.json").open("w", encoding="utf-8") as stream:
        json.dump(result.as_dict(), stream, ensure_ascii=False, indent=2)
    return output


__all__ = [
    "DEFAULT_LOCAL_MAX_RADII",
    "SOURCE_PEAK_CONSISTENCY_CLASS_FIELDS",
    "SOURCE_PEAK_CONSISTENCY_FIELDS",
    "SOURCE_PEAK_CONSISTENCY_TARGET_FIELDS",
    "SourcePeakConsistencyClassSummary",
    "SourcePeakConsistencyResult",
    "SourcePeakConsistencyRow",
    "SourcePeakConsistencyTargetRow",
    "run_source_peak_consistency",
    "write_source_peak_consistency_artifacts",
]
