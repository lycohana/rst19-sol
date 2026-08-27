"""跨模块共用的数据模型。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np


@dataclass(frozen=True, slots=True)
class AuxiliaryData:
    """FITS 图像首行前 208 字节解码得到的 26 个辅助字段。"""

    values: tuple[float, ...]

    FIELD_NAMES = (
        "p_az",
        "p_el",
        "ra",
        "dec",
        "j2000_x",
        "j2000_y",
        "j2000_z",
        "j2000_xv",
        "j2000_yv",
        "j2000_zv",
        "q1",
        "q2",
        "q3",
        "q4",
        "roll",
        "pitch",
        "yaw",
        "roll_v",
        "pitch_v",
        "yaw_v",
        "wgs84_x",
        "wgs84_y",
        "wgs84_z",
        "wgs84_xv",
        "wgs84_yv",
        "wgs84_zv",
    )

    def __post_init__(self) -> None:
        if len(self.values) != len(self.FIELD_NAMES):
            raise ValueError(f"expected {len(self.FIELD_NAMES)} auxiliary values, got {len(self.values)}")

    @classmethod
    def from_values(cls, values: tuple[float, ...] | list[float]) -> "AuxiliaryData":
        return cls(tuple(float(value) for value in values))

    def as_dict(self) -> dict[str, float]:
        return dict(zip(self.FIELD_NAMES, self.values, strict=True))

    @property
    def ra_deg(self) -> float:
        return self.values[2]

    @property
    def dec_deg(self) -> float:
        return self.values[3]


@dataclass(frozen=True, slots=True)
class FitsFrame:
    """读取后的单帧 FITS 图像及其头部信息。"""

    path: Path
    header: Mapping[str, object]
    data: np.ndarray
    auxiliary: AuxiliaryData | None
    data_offset: int

    @property
    def height(self) -> int:
        return int(self.data.shape[0])

    @property
    def width(self) -> int:
        return int(self.data.shape[1])
