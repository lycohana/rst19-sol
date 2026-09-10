"""检测源的相对/仪器星等和最暗可信源选择。

本模块刻意把三个容易混淆的量分开：

* ``m_inst``：由本机 ADU 测量得到的仪器星等；
* ``m_std``：经过标准星、零点和波段/颜色项标定后的表观星等；
* ``M_V``：还需要距离（或视差）和消光修正的绝对星等。

没有这些外部标定条件时，不能把 ``m_inst`` 改名成 ``m_V`` 或 ``M_V``。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from .detection import Detection


def _finite(value: float | None) -> bool:
    return value is not None and math.isfinite(float(value))


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


def absolute_magnitude_from_apparent(
    apparent_magnitude: float,
    *,
    distance_pc: float,
    extinction_mag: float = 0.0,
) -> float | None:
    """由表观星等、距离和消光计算绝对星等。

    定义为 ``M = m - 5 log10(d / 10 pc) - A``。这里的 ``m`` 必须已经
    属于明确的标准波段（例如 V），而不是 ``m_inst``。距离可以来自
    可靠视差或独立测量；检测到多少颗星不能提供这个距离。
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
    为“绝对星等不可计算”而不是制造一个数值。
    """

    if not math.isfinite(float(parallax_mas)) or parallax_mas <= 0:
        raise ValueError("parallax_mas must be finite and positive")
    distance_pc = 1000.0 / float(parallax_mas)
    return absolute_magnitude_from_apparent(
        apparent_magnitude,
        distance_pc=distance_pc,
        extinction_mag=extinction_mag,
    )


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
            candidates.append((magnitude, source))
    if not candidates:
        return None
    magnitude, source = max(candidates, key=lambda item: (item[0], item[1].detection_id))
    # Do not reuse the loop-local rate here.  The selected source is the
    # faintest one after filtering, and its rate must be derived from that
    # same source; otherwise the exported/UI value can belong to the last
    # source visited in the loop.
    selected_flux_rate = source.flux / exposure_s
    return FaintestSource(
        detection_id=source.detection_id,
        x=source.x,
        y=source.y,
        flux=source.flux,
        snr=source.snr,
        instrumental_magnitude=magnitude,
        calibrated_magnitude=magnitude + zero_point if zero_point is not None else None,
        flags=source.flags,
        flux_snr=source.flux_snr,
        flux_rate=selected_flux_rate,
    )
