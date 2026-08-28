"""检测源的相对/仪器星等和最暗可信源选择。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from .detection import Detection


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
        flux_rate = source.flux / exposure_s
        magnitude = instrumental_magnitude(source.flux, exposure_s=exposure_s)
        if magnitude is not None:
            candidates.append((magnitude, source))
    if not candidates:
        return None
    magnitude, source = max(candidates, key=lambda item: (item[0], item[1].detection_id))
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
        flux_rate=flux_rate,
    )
