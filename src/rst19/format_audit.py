"""比赛 FITS 实际格式、值域和辅助遥测单位的只读审计。

该模块只回答一个边界问题：文件实际存了什么，格式说明又没有定义什么。
它不把负值、``-1`` 或 16 bit 极值自动改成坏像素/饱和，也不改变默认检测器
使用的有符号大端 ``BITPIX=16`` 读取方式。所有诊断统计均排除首行前 208
字节的比赛辅助数据区域。
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from .fits import auxiliary_mask, exposure_milliseconds, read_fits
from .models import FitsFrame


# These are deliberately reported as header facts, not interpreted as a
# complete WCS validator.  The current data may carry an auxiliary boresight
# in the first data row while still having none of these standard cards.
STANDARD_WCS_HEADER_CARDS = (
    "CRPIX1",
    "CRPIX2",
    "CRVAL1",
    "CRVAL2",
    "CTYPE1",
    "CTYPE2",
    "CD1_1",
    "CD1_2",
    "CD2_1",
    "CD2_2",
    "CDELT1",
    "CDELT2",
    "CROTA2",
    "PV1_0",
    "PV2_0",
    "A_ORDER",
    "B_ORDER",
)
WCS_MINIMAL_CORE_CARDS = ("CRPIX1", "CRPIX2", "CRVAL1", "CRVAL2", "CTYPE1", "CTYPE2")


def _optional_int(value: object) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed


def _optional_float(value: object) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _parse_timestamp(value: object) -> datetime | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _as_timestamp_text(value: object) -> str | None:
    parsed = _parse_timestamp(value)
    return parsed.isoformat() if parsed is not None else (str(value) if value is not None else None)


def _resolve_paths(input_path: str | Path) -> tuple[Path, ...]:
    path = Path(input_path)
    if path.is_file():
        if path.suffix.lower() != ".fits":
            raise ValueError(f"expected a .fits file, got {path}")
        return (path,)
    if path.is_dir():
        paths = tuple(sorted(path.glob("*.fits")))
        if not paths:
            raise ValueError(f"directory contains no FITS files: {path}")
        return paths
    raise FileNotFoundError(f"input path does not exist: {path}")


def _valid_pixels(frame: FitsFrame) -> np.ndarray:
    mask = auxiliary_mask(frame.data.shape)
    return np.asarray(frame.data[~mask])


def _unsigned_view(values: np.ndarray) -> np.ndarray:
    if values.dtype != np.dtype(">i2"):
        raise ValueError(f"unsigned diagnostic expects >i2 storage, got {values.dtype}")
    return np.frombuffer(values.tobytes(), dtype=np.dtype(">u2"))


@dataclass(frozen=True, slots=True)
class FormatAuditFrame:
    """单个 FITS 文件的实际存储和像素值域摘要。"""

    frame_index: int
    path: str
    file_size_bytes: int
    data_offset_bytes: int
    expected_file_size_bytes: int | None
    file_size_matches: bool | None
    bitpix: int | None
    naxis1: int | None
    naxis2: int | None
    shape: str
    shape_matches_header: bool | None
    dtype: str
    bscale: float | None
    bzero: float | None
    blank: float | None
    has_bscale: bool
    has_bzero: bool
    has_blank: bool
    timestamp: str | None
    exposure_ms: float | None
    auxiliary_complete: bool
    auxiliary_masked_pixel_count: int
    wcs_standard_card_count: int
    wcs_standard_cards_present: str
    wcs_standard_cards_missing: str
    wcs_minimal_core_present: bool
    valid_pixel_count: int
    signed_min: int | None
    signed_max: int | None
    negative_count: int
    zero_count: int
    exact_minus_one_count: int
    negative_extreme_count: int
    positive_extreme_count: int
    unsigned_diagnostic_min: int | None
    unsigned_diagnostic_max: int | None

    def as_dict(self) -> dict[str, object]:
        return {
            "frame_index": self.frame_index,
            "frame_number": self.frame_index + 1,
            "path": self.path,
            "file_size_bytes": self.file_size_bytes,
            "data_offset_bytes": self.data_offset_bytes,
            "expected_file_size_bytes": self.expected_file_size_bytes,
            "file_size_matches": self.file_size_matches,
            "bitpix": self.bitpix,
            "naxis1": self.naxis1,
            "naxis2": self.naxis2,
            "shape": self.shape,
            "shape_matches_header": self.shape_matches_header,
            "dtype": self.dtype,
            "bscale": self.bscale,
            "bzero": self.bzero,
            "blank": self.blank,
            "has_bscale": self.has_bscale,
            "has_bzero": self.has_bzero,
            "has_blank": self.has_blank,
            "timestamp": self.timestamp,
            "exposure_ms": self.exposure_ms,
            "auxiliary_complete": self.auxiliary_complete,
            "auxiliary_masked_pixel_count": self.auxiliary_masked_pixel_count,
            "wcs_standard_card_count": self.wcs_standard_card_count,
            "wcs_standard_cards_present": self.wcs_standard_cards_present,
            "wcs_standard_cards_missing": self.wcs_standard_cards_missing,
            "wcs_minimal_core_present": self.wcs_minimal_core_present,
            "valid_pixel_count": self.valid_pixel_count,
            "signed_min": self.signed_min,
            "signed_max": self.signed_max,
            "negative_count": self.negative_count,
            "zero_count": self.zero_count,
            "exact_minus_one_count": self.exact_minus_one_count,
            "negative_extreme_count": self.negative_extreme_count,
            "positive_extreme_count": self.positive_extreme_count,
            "unsigned_diagnostic_min": self.unsigned_diagnostic_min,
            "unsigned_diagnostic_max": self.unsigned_diagnostic_max,
        }


@dataclass(frozen=True, slots=True)
class FormatVelocityRow:
    """由位置差分与辅助速度字段得到的单位自洽性检查。"""

    frame_index: int
    path: str
    timestamp: str | None
    dt_seconds: float | None
    derivative_method: str
    j2000_position_velocity_error: float | None
    j2000_position_velocity_relative_error: float | None
    wgs84_position_velocity_error: float | None
    wgs84_position_velocity_relative_error: float | None

    def as_dict(self) -> dict[str, object]:
        return {
            "frame_index": self.frame_index,
            "frame_number": self.frame_index + 1,
            "path": self.path,
            "timestamp": self.timestamp,
            "dt_seconds": self.dt_seconds,
            "derivative_method": self.derivative_method,
            "j2000_position_velocity_error": self.j2000_position_velocity_error,
            "j2000_position_velocity_relative_error": self.j2000_position_velocity_relative_error,
            "wgs84_position_velocity_error": self.wgs84_position_velocity_error,
            "wgs84_position_velocity_relative_error": self.wgs84_position_velocity_relative_error,
        }


@dataclass(frozen=True, slots=True)
class FormatAuditResult:
    """格式审计的逐帧结果、单位对照和解释边界。"""

    input_path: str
    extreme_fraction: float
    frame_rows: tuple[FormatAuditFrame, ...]
    velocity_rows: tuple[FormatVelocityRow, ...]
    summary: Mapping[str, object]

    def as_dict(self) -> dict[str, object]:
        return {
            "input_path": self.input_path,
            "extreme_fraction": self.extreme_fraction,
            "frames": [row.as_dict() for row in self.frame_rows],
            "velocity": [row.as_dict() for row in self.velocity_rows],
            "summary": dict(self.summary),
            "interpretation_boundary": (
                "actual FITS storage and internal telemetry consistency; not a calibration of ADC signedness, "
                "BLANK semantics, saturation level, complete WCS, or physical source truth"
            ),
        }


def _frame_row(index: int, path: Path, frame: FitsFrame, extreme_fraction: float) -> FormatAuditFrame:
    width = _optional_int(frame.header.get("NAXIS1"))
    height = _optional_int(frame.header.get("NAXIS2"))
    bitpix = _optional_int(frame.header.get("BITPIX"))
    expected_data_bytes = (
        width * height * (abs(bitpix) // 8)
        if width is not None and height is not None and bitpix not in (None, 0)
        else None
    )
    expected_file_size = frame.data_offset + expected_data_bytes if expected_data_bytes is not None else None
    file_size = path.stat().st_size
    values = _valid_pixels(frame)
    signed = values.astype(np.int64, copy=False)
    negative_threshold = math.floor(np.iinfo(np.int16).min * extreme_fraction)
    positive_threshold = math.ceil(np.iinfo(np.int16).max * extreme_fraction)
    unsigned = _unsigned_view(values) if values.dtype == np.dtype(">i2") else np.asarray([], dtype=np.uint16)
    present_wcs_cards = tuple(card for card in STANDARD_WCS_HEADER_CARDS if card in frame.header)
    missing_wcs_cards = tuple(card for card in STANDARD_WCS_HEADER_CARDS if card not in frame.header)
    return FormatAuditFrame(
        frame_index=index,
        path=str(path.resolve()),
        file_size_bytes=file_size,
        data_offset_bytes=frame.data_offset,
        expected_file_size_bytes=expected_file_size,
        file_size_matches=(file_size == expected_file_size) if expected_file_size is not None else None,
        bitpix=bitpix,
        naxis1=width,
        naxis2=height,
        shape=f"{frame.height}x{frame.width}",
        shape_matches_header=(frame.width == width and frame.height == height)
        if width is not None and height is not None
        else None,
        dtype=str(frame.data.dtype),
        bscale=_optional_float(frame.header.get("BSCALE")),
        bzero=_optional_float(frame.header.get("BZERO")),
        blank=_optional_float(frame.header.get("BLANK")),
        has_bscale="BSCALE" in frame.header,
        has_bzero="BZERO" in frame.header,
        has_blank="BLANK" in frame.header,
        timestamp=_as_timestamp_text(frame.header.get("DATE-OBS")),
        exposure_ms=exposure_milliseconds(frame.header),
        auxiliary_complete=frame.auxiliary is not None,
        auxiliary_masked_pixel_count=min(frame.width, 104) if frame.height > 0 else 0,
        wcs_standard_card_count=len(present_wcs_cards),
        wcs_standard_cards_present="|".join(present_wcs_cards),
        wcs_standard_cards_missing="|".join(missing_wcs_cards),
        wcs_minimal_core_present=all(card in frame.header for card in WCS_MINIMAL_CORE_CARDS),
        valid_pixel_count=int(values.size),
        signed_min=int(signed.min()) if signed.size else None,
        signed_max=int(signed.max()) if signed.size else None,
        negative_count=int(np.count_nonzero(signed < 0)),
        zero_count=int(np.count_nonzero(signed == 0)),
        exact_minus_one_count=int(np.count_nonzero(signed == -1)),
        negative_extreme_count=int(np.count_nonzero(signed <= negative_threshold)),
        positive_extreme_count=int(np.count_nonzero(signed >= positive_threshold)),
        unsigned_diagnostic_min=int(unsigned.min()) if unsigned.size else None,
        unsigned_diagnostic_max=int(unsigned.max()) if unsigned.size else None,
    )


def _vector(auxiliary: Mapping[str, float], prefix: str, suffix: str = "") -> np.ndarray:
    return np.asarray([auxiliary[f"{prefix}_{axis}{suffix}"] for axis in "xyz"], dtype=np.float64)


def _relative_error(estimate: np.ndarray, reference: np.ndarray) -> tuple[float, float]:
    error = float(np.linalg.norm(estimate - reference))
    reference_norm = float(np.linalg.norm(reference))
    relative = error / reference_norm if reference_norm > np.finfo(np.float64).eps else math.nan
    return error, relative


def _velocity_rows(records: Sequence[tuple[Path, FitsFrame, datetime | None]]) -> tuple[FormatVelocityRow, ...]:
    if not records:
        return ()
    rows: list[FormatVelocityRow] = []
    for index, (path, frame, timestamp) in enumerate(records):
        auxiliary = frame.auxiliary.as_dict() if frame.auxiliary is not None else None
        if auxiliary is None or timestamp is None:
            rows.append(
                FormatVelocityRow(
                    frame_index=index,
                    path=str(path.resolve()),
                    timestamp=timestamp.isoformat() if timestamp is not None else None,
                    dt_seconds=None,
                    derivative_method="unavailable",
                    j2000_position_velocity_error=None,
                    j2000_position_velocity_relative_error=None,
                    wgs84_position_velocity_error=None,
                    wgs84_position_velocity_relative_error=None,
                )
            )
            continue

        if index == 0 and len(records) > 1:
            neighbor = records[1]
            dt = (neighbor[2] - timestamp).total_seconds() if neighbor[2] is not None else math.nan
            derivative_method = "forward_difference"
            left, right = frame.auxiliary.as_dict(), neighbor[1].auxiliary.as_dict()  # type: ignore[union-attr]
        elif index == len(records) - 1 and len(records) > 1:
            neighbor = records[index - 1]
            dt = (timestamp - neighbor[2]).total_seconds() if neighbor[2] is not None else math.nan
            derivative_method = "backward_difference"
            left, right = neighbor[1].auxiliary.as_dict(), frame.auxiliary.as_dict()  # type: ignore[union-attr]
        elif 0 < index < len(records) - 1 and records[index - 1][2] is not None and records[index + 1][2] is not None:
            dt = (records[index + 1][2] - records[index - 1][2]).total_seconds()  # type: ignore[operator]
            derivative_method = "centered_difference"
            left = records[index - 1][1].auxiliary.as_dict()  # type: ignore[union-attr]
            right = records[index + 1][1].auxiliary.as_dict()  # type: ignore[union-attr]
        else:
            dt = math.nan
            derivative_method = "unavailable"
            left = right = auxiliary

        if not math.isfinite(dt) or abs(dt) <= np.finfo(np.float64).eps or derivative_method == "unavailable":
            rows.append(
                FormatVelocityRow(
                    frame_index=index,
                    path=str(path.resolve()),
                    timestamp=timestamp.isoformat(),
                    dt_seconds=None,
                    derivative_method="unavailable",
                    j2000_position_velocity_error=None,
                    j2000_position_velocity_relative_error=None,
                    wgs84_position_velocity_error=None,
                    wgs84_position_velocity_relative_error=None,
                )
            )
            continue

        j2000_derivative = (_vector(right, "j2000") - _vector(left, "j2000")) / dt
        j2000_reference = _vector(auxiliary, "j2000", "v")
        wgs84_derivative = (_vector(right, "wgs84") - _vector(left, "wgs84")) / dt
        wgs84_reference = _vector(auxiliary, "wgs84", "v")
        j2000_error, j2000_relative = _relative_error(j2000_derivative, j2000_reference)
        wgs84_error, wgs84_relative = _relative_error(wgs84_derivative, wgs84_reference)
        rows.append(
            FormatVelocityRow(
                frame_index=index,
                path=str(path.resolve()),
                timestamp=timestamp.isoformat(),
                dt_seconds=float(abs(dt)),
                derivative_method=derivative_method,
                j2000_position_velocity_error=j2000_error,
                j2000_position_velocity_relative_error=j2000_relative,
                wgs84_position_velocity_error=wgs84_error,
                wgs84_position_velocity_relative_error=wgs84_relative,
            )
        )
    return tuple(rows)


def _median(values: Iterable[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(np.median(finite)) if finite else None


def _max(values: Iterable[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return max(finite) if finite else None


def _summarize(
    frame_rows: Sequence[FormatAuditFrame],
    velocity_rows: Sequence[FormatVelocityRow],
    *,
    extreme_fraction: float,
) -> dict[str, object]:
    timestamps = [parsed for parsed in (_parse_timestamp(row.timestamp) for row in frame_rows) if parsed is not None]
    bscale_present = sum(row.has_bscale for row in frame_rows)
    bzero_present = sum(row.has_bzero for row in frame_rows)
    blank_present = sum(row.has_blank for row in frame_rows)
    return {
        "frame_count": len(frame_rows),
        "file_size_bytes": sorted({row.file_size_bytes for row in frame_rows}),
        "shape_counts": {
            shape: sum(row.shape == shape for row in frame_rows)
            for shape in sorted({row.shape for row in frame_rows})
        },
        "bitpix_counts": {
            str(bitpix): sum(row.bitpix == bitpix for row in frame_rows)
            for bitpix in sorted({row.bitpix for row in frame_rows})
        },
        "dtype_counts": {
            dtype: sum(row.dtype == dtype for row in frame_rows)
            for dtype in sorted({row.dtype for row in frame_rows})
        },
        "all_file_sizes_match_header": all(row.file_size_matches is True for row in frame_rows),
        "all_shapes_match_header": all(row.shape_matches_header is True for row in frame_rows),
        "complete_auxiliary_frames": sum(row.auxiliary_complete for row in frame_rows),
        "wcs_standard_card_counts": {
            card: sum(card in row.wcs_standard_cards_present.split("|") for row in frame_rows)
            for card in STANDARD_WCS_HEADER_CARDS
        },
        "frames_with_any_standard_wcs_card": sum(row.wcs_standard_card_count > 0 for row in frame_rows),
        "frames_with_minimal_wcs_core": sum(row.wcs_minimal_core_present for row in frame_rows),
        "bscale_cards_present": bscale_present,
        "bzero_cards_present": bzero_present,
        "blank_cards_present": blank_present,
        "total_valid_pixels": sum(row.valid_pixel_count for row in frame_rows),
        "total_negative_pixels": sum(row.negative_count for row in frame_rows),
        "total_zero_pixels": sum(row.zero_count for row in frame_rows),
        "total_exact_minus_one_pixels": sum(row.exact_minus_one_count for row in frame_rows),
        "total_negative_extreme_pixels": sum(row.negative_extreme_count for row in frame_rows),
        "total_positive_extreme_pixels": sum(row.positive_extreme_count for row in frame_rows),
        "extreme_fraction": extreme_fraction,
        "timestamp_start": min(timestamps).isoformat() if timestamps else None,
        "timestamp_end": max(timestamps).isoformat() if timestamps else None,
        "timestamp_span_seconds": (max(timestamps) - min(timestamps)).total_seconds() if len(timestamps) >= 2 else None,
        "velocity_consistency": {
            "usable_frame_count": sum(row.dt_seconds is not None for row in velocity_rows),
            "j2000_relative_error_median": _median(
                row.j2000_position_velocity_relative_error for row in velocity_rows
            ),
            "j2000_relative_error_max": _max(
                row.j2000_position_velocity_relative_error for row in velocity_rows
            ),
            "wgs84_relative_error_median": _median(
                row.wgs84_position_velocity_relative_error for row in velocity_rows
            ),
            "wgs84_relative_error_max": _max(
                row.wgs84_position_velocity_relative_error for row in velocity_rows
            ),
            "unit_interpretation": (
                "位置字段标注为 m 且与 DATE-OBS 差分高度自洽时，速度字段的 m/s 是内部一致性推断；"
                "格式说明原文仍需主办方确认。"
            ),
        },
        "interpretation_notes": [
            "以合法 FITS 头、数据长度和 BITPIX 作为实际存储事实，不把格式说明中的尺寸文字写死。",
            "负值、精确 -1 和正负极值只做诊断统计；没有 BLANK、ADC、满阱或坏像素定义时不自动掩膜。",
            "uint16 视图只用于显示同一字节的替代解释，不改变默认有符号 >i2 读取。",
            "首行前 208 字节辅助数据已从像素值域统计中排除。",
            "标准 WCS 卡片只做存在性记录；辅助光轴/姿态字段位于首行数据区时，不把它们混同为完整 WCS。",
        ],
    }


def run_format_audit(
    input_path: str | Path,
    *,
    extreme_fraction: float = 0.9,
) -> FormatAuditResult:
    """审计一个 FITS 或目录，返回不改变默认读取语义的只读结果。"""

    if not 0.0 < float(extreme_fraction) <= 1.0:
        raise ValueError("extreme_fraction must be in (0, 1]")
    paths = _resolve_paths(input_path)
    records: list[tuple[Path, FitsFrame, datetime | None]] = []
    frame_rows: list[FormatAuditFrame] = []
    for path in paths:
        frame = read_fits(path, apply_scaling=False)
        timestamp = _parse_timestamp(frame.header.get("DATE-OBS"))
        records.append((path, frame, timestamp))
    records.sort(key=lambda item: (item[2] is None, item[2] or datetime.max.replace(tzinfo=timezone.utc), str(item[0])))
    for index, (path, frame, _) in enumerate(records):
        frame_rows.append(_frame_row(index, path, frame, float(extreme_fraction)))
    velocity_rows = _velocity_rows(records)
    summary = _summarize(frame_rows, velocity_rows, extreme_fraction=float(extreme_fraction))
    return FormatAuditResult(
        input_path=str(Path(input_path).resolve()),
        extreme_fraction=float(extreme_fraction),
        frame_rows=tuple(frame_rows),
        velocity_rows=velocity_rows,
        summary=summary,
    )


def write_format_audit_artifacts(result: FormatAuditResult, output_dir: str | Path) -> Path:
    """写出逐帧 CSV、位置—速度 CSV 和 JSON 汇总。"""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    frame_path = output / "format_audit_frames.csv"
    velocity_path = output / "format_audit_velocity.csv"
    json_path = output / "format_audit.json"
    frame_fields = ["frame_number", *FormatAuditFrame.__dataclass_fields__]
    velocity_fields = ["frame_number", *FormatVelocityRow.__dataclass_fields__]
    with frame_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=frame_fields)
        writer.writeheader()
        writer.writerows(row.as_dict() for row in result.frame_rows)
    with velocity_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=velocity_fields)
        writer.writeheader()
        writer.writerows(row.as_dict() for row in result.velocity_rows)
    with json_path.open("w", encoding="utf-8") as stream:
        json.dump(result.as_dict(), stream, ensure_ascii=False, indent=2)
    return output


__all__ = [
    "FormatAuditFrame",
    "FormatAuditResult",
    "FormatVelocityRow",
    "STANDARD_WCS_HEADER_CARDS",
    "WCS_MINIMAL_CORE_CARDS",
    "run_format_audit",
    "write_format_audit_artifacts",
]
