"""离线任务星表的轻量读取和历元传播。"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


def _optional_float(value: str | None, *, field: str, row_number: int) -> float | None:
    if value is None or not value.strip():
        return None
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(f"catalog row {row_number}: invalid {field}={value!r}") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"catalog row {row_number}: non-finite {field}={value!r}")
    return parsed


def _required_float(value: str | None, *, field: str, row_number: int) -> float:
    parsed = _optional_float(value, field=field, row_number=row_number)
    if parsed is None:
        raise ValueError(f"catalog row {row_number}: missing {field}")
    return parsed


def _first_value(row: dict[str, str], names: Iterable[str]) -> str | None:
    for name in names:
        value = row.get(name)
        if value is not None and value.strip():
            return value
    return None


@dataclass(frozen=True, slots=True)
class CatalogSource:
    """任务星表中的单个源。

    `pmra_mas_yr` 遵循 Gaia 常用定义，即包含 cos(dec) 的 RA 自行；
    传播时会除以 cos(dec)。没有自行数据时按静止源处理。
    """

    source_id: str
    ra_deg: float
    dec_deg: float
    magnitude: float | None = None
    color: float | None = None
    pmra_mas_yr: float = 0.0
    pmdec_mas_yr: float = 0.0
    ref_epoch: float = 2016.0

    def __post_init__(self) -> None:
        if not self.source_id:
            raise ValueError("catalog source_id cannot be empty")
        numeric_values = (
            self.ra_deg,
            self.dec_deg,
            self.magnitude,
            self.color,
            self.pmra_mas_yr,
            self.pmdec_mas_yr,
            self.ref_epoch,
        )
        if any(value is not None and not math.isfinite(float(value)) for value in numeric_values):
            raise ValueError(f"catalog source {self.source_id!r} contains a non-finite value")
        if not -90.0 <= self.dec_deg <= 90.0:
            raise ValueError(f"catalog declination out of range: {self.dec_deg}")
        if not -360.0 <= self.ra_deg <= 360.0:
            raise ValueError(f"catalog right ascension out of range: {self.ra_deg}")

    def at_epoch(self, epoch: float | None) -> "CatalogSource":
        """将自行线性传播到指定 Julian 年，返回新对象。"""

        if epoch is None or epoch == self.ref_epoch:
            return self
        years = float(epoch) - self.ref_epoch
        cos_dec = max(1e-8, abs(math.cos(math.radians(self.dec_deg))))
        ra = self.ra_deg + self.pmra_mas_yr * years / cos_dec / 3_600_000.0
        dec = self.dec_deg + self.pmdec_mas_yr * years / 3_600_000.0
        return CatalogSource(
            source_id=self.source_id,
            ra_deg=ra % 360.0,
            dec_deg=dec,
            magnitude=self.magnitude,
            color=self.color,
            pmra_mas_yr=self.pmra_mas_yr,
            pmdec_mas_yr=self.pmdec_mas_yr,
            ref_epoch=epoch,
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "source_id": self.source_id,
            "ra_deg": self.ra_deg,
            "dec_deg": self.dec_deg,
            "magnitude": self.magnitude,
            "color": self.color,
            "pmra_mas_yr": self.pmra_mas_yr,
            "pmdec_mas_yr": self.pmdec_mas_yr,
            "ref_epoch": self.ref_epoch,
        }


def load_catalog_csv(path: str | Path) -> tuple[CatalogSource, ...]:
    """读取离线 CSV 星表。

    必需列为 `source_id`、`ra_deg`/`ra`、`dec_deg`/`dec`；其余字段可选。
    兼容 `phot_g_mean_mag`、`pmra`、`pmdec` 和 `ref_epoch` 常见列名。
    """

    file_path = Path(path)
    with file_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError("catalog CSV has no header")
        sources: list[CatalogSource] = []
        source_ids: set[str] = set()
        for row_number, row in enumerate(reader, start=2):
            source_id = _first_value(row, ("source_id", "sourceid", "id"))
            if source_id is None:
                raise ValueError(f"catalog row {row_number}: missing source_id")
            source_id = source_id.strip()
            if source_id in source_ids:
                raise ValueError(f"catalog row {row_number}: duplicate source_id={source_id!r}")
            source_ids.add(source_id)
            ra = _required_float(_first_value(row, ("ra_deg", "ra", "RA")), field="ra_deg", row_number=row_number)
            dec = _required_float(_first_value(row, ("dec_deg", "dec", "DEC")), field="dec_deg", row_number=row_number)
            magnitude = _optional_float(
                _first_value(row, ("magnitude", "mag", "phot_g_mean_mag")),
                field="magnitude",
                row_number=row_number,
            )
            color = _optional_float(
                _first_value(row, ("color", "bp_rp")),
                field="color",
                row_number=row_number,
            )
            pmra = _optional_float(
                _first_value(row, ("pmra_mas_yr", "pmra")),
                field="pmra_mas_yr",
                row_number=row_number,
            )
            pmdec = _optional_float(
                _first_value(row, ("pmdec_mas_yr", "pmdec")),
                field="pmdec_mas_yr",
                row_number=row_number,
            )
            ref_epoch = _optional_float(
                _first_value(row, ("ref_epoch", "epoch")),
                field="ref_epoch",
                row_number=row_number,
            )
            sources.append(
                CatalogSource(
                    source_id=source_id,
                    ra_deg=ra,
                    dec_deg=dec,
                    magnitude=magnitude,
                    color=color,
                    pmra_mas_yr=pmra if pmra is not None else 0.0,
                    pmdec_mas_yr=pmdec if pmdec is not None else 0.0,
                    ref_epoch=ref_epoch if ref_epoch is not None else 2016.0,
                )
            )
    return tuple(sources)
