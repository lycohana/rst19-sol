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


def _optional_bool(value: str | None, *, field: str, row_number: int) -> bool | None:
    """读取 CSV 中常见的布尔表示；空值仍表示“该质量字段未知”。"""

    if value is None or not value.strip():
        return None
    normalised = value.strip().lower()
    if normalised in {"true", "t", "1", "yes", "y"}:
        return True
    if normalised in {"false", "f", "0", "no", "n"}:
        return False
    raise ValueError(f"catalog row {row_number}: invalid {field}={value!r}")


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
    # Photometric provenance is intentionally explicit.  The historical
    # ``magnitude``/``color`` fields remain as compatibility aliases, while
    # these fields prevent a Gaia G value from silently becoming a generic
    # or Johnson-V value later in the pipeline.
    catalog_name: str = "unknown"
    photometric_system: str = "unknown"
    photometric_band: str = "unknown"
    magnitude_error: float | None = None
    color_name: str | None = None
    parallax_mas: float | None = None
    parallax_error_mas: float | None = None
    extinction_mag: float | None = None
    extinction_error_mag: float | None = None
    # Gaia DR3 GSP-Phot metadata. These are model-based, single-star
    # estimates and remain separate from image-derived photometry.
    distance_pc: float | None = None
    distance_lower_pc: float | None = None
    distance_upper_pc: float | None = None
    distance_source: str | None = None
    # Optional Gaia quality metadata.  Missing values are intentionally kept
    # as ``None``: hand-written/offline catalogues predating this contract
    # remain usable, while downloaded Gaia rows can be filtered explicitly
    # before they enter a photometric zero-point fit.
    phot_g_mean_flux_over_error: float | None = None
    phot_bp_rp_excess_factor: float | None = None
    ruwe: float | None = None
    duplicated_source: bool | None = None
    visibility_periods_used: int | None = None
    phot_variable_flag: str | None = None

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
            self.magnitude_error,
            self.parallax_mas,
            self.parallax_error_mas,
            self.extinction_mag,
            self.extinction_error_mag,
            self.distance_pc,
            self.distance_lower_pc,
            self.distance_upper_pc,
            self.phot_g_mean_flux_over_error,
            self.phot_bp_rp_excess_factor,
            self.ruwe,
        )
        if any(value is not None and not math.isfinite(float(value)) for value in numeric_values):
            raise ValueError(f"catalog source {self.source_id!r} contains a non-finite value")
        for name, value in (
            ("magnitude_error", self.magnitude_error),
            ("parallax_error_mas", self.parallax_error_mas),
            ("extinction_error_mag", self.extinction_error_mag),
            ("distance_pc", self.distance_pc),
            ("distance_lower_pc", self.distance_lower_pc),
            ("distance_upper_pc", self.distance_upper_pc),
            ("phot_g_mean_flux_over_error", self.phot_g_mean_flux_over_error),
            ("phot_bp_rp_excess_factor", self.phot_bp_rp_excess_factor),
            ("ruwe", self.ruwe),
        ):
            if value is not None and float(value) < 0:
                raise ValueError(f"catalog source {self.source_id!r} has negative {name}")
        if self.visibility_periods_used is not None:
            if isinstance(self.visibility_periods_used, bool) or not isinstance(self.visibility_periods_used, int):
                raise ValueError("visibility_periods_used must be an integer or None")
            if self.visibility_periods_used < 0:
                raise ValueError("visibility_periods_used cannot be negative")
        if self.duplicated_source is not None and not isinstance(self.duplicated_source, bool):
            raise ValueError("duplicated_source must be a boolean or None")
        if self.phot_variable_flag is not None and not str(self.phot_variable_flag).strip():
            raise ValueError("phot_variable_flag cannot be empty when provided")
        for name, value in (
            ("distance_pc", self.distance_pc),
            ("distance_lower_pc", self.distance_lower_pc),
            ("distance_upper_pc", self.distance_upper_pc),
        ):
            if value is not None and float(value) <= 0:
                raise ValueError(f"catalog source {self.source_id!r} has non-positive {name}")
        if self.distance_lower_pc is not None and self.distance_upper_pc is not None:
            if float(self.distance_lower_pc) > float(self.distance_upper_pc):
                raise ValueError(f"catalog source {self.source_id!r} has inverted distance interval")
        if self.distance_source is not None and not str(self.distance_source).strip():
            raise ValueError("distance_source cannot be empty when provided")
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
            catalog_name=self.catalog_name,
            photometric_system=self.photometric_system,
            photometric_band=self.photometric_band,
            magnitude_error=self.magnitude_error,
            color_name=self.color_name,
            parallax_mas=self.parallax_mas,
            parallax_error_mas=self.parallax_error_mas,
            extinction_mag=self.extinction_mag,
            extinction_error_mag=self.extinction_error_mag,
            distance_pc=self.distance_pc,
            distance_lower_pc=self.distance_lower_pc,
            distance_upper_pc=self.distance_upper_pc,
            distance_source=self.distance_source,
            phot_g_mean_flux_over_error=self.phot_g_mean_flux_over_error,
            phot_bp_rp_excess_factor=self.phot_bp_rp_excess_factor,
            ruwe=self.ruwe,
            duplicated_source=self.duplicated_source,
            visibility_periods_used=self.visibility_periods_used,
            phot_variable_flag=self.phot_variable_flag,
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
            "catalog_name": self.catalog_name,
            "photometric_system": self.photometric_system,
            "photometric_band": self.photometric_band,
            "magnitude_error": self.magnitude_error,
            "color_name": self.color_name,
            "parallax_mas": self.parallax_mas,
            "parallax_error_mas": self.parallax_error_mas,
            "extinction_mag": self.extinction_mag,
            "extinction_error_mag": self.extinction_error_mag,
            "distance_pc": self.distance_pc,
            "distance_lower_pc": self.distance_lower_pc,
            "distance_upper_pc": self.distance_upper_pc,
            "distance_source": self.distance_source,
            "phot_g_mean_flux_over_error": self.phot_g_mean_flux_over_error,
            "phot_bp_rp_excess_factor": self.phot_bp_rp_excess_factor,
            "ruwe": self.ruwe,
            "duplicated_source": self.duplicated_source,
            "visibility_periods_used": self.visibility_periods_used,
            "phot_variable_flag": self.phot_variable_flag,
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
        fieldnames = {str(field).strip().lower() for field in reader.fieldnames}
        has_gaia_g = "phot_g_mean_mag" in fieldnames
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
                _first_value(row, ("magnitude", "apparent_magnitude", "mag", "phot_g_mean_mag")),
                field="magnitude",
                row_number=row_number,
            )
            magnitude_error = _optional_float(
                _first_value(
                    row,
                    (
                        "magnitude_error",
                        "mag_error",
                        "phot_g_mean_mag_error",
                        "phot_g_mean_mag_err",
                    ),
                ),
                field="magnitude_error",
                row_number=row_number,
            )
            color = _optional_float(
                _first_value(row, ("color", "bp_rp")),
                field="color",
                row_number=row_number,
            )
            color_name = _first_value(row, ("color_name", "color_band"))
            if color_name is None and "bp_rp" in fieldnames:
                color_name = "BP-RP"
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
            parallax_mas = _optional_float(
                _first_value(row, ("parallax_mas", "parallax", "plx")),
                field="parallax_mas",
                row_number=row_number,
            )
            parallax_error_mas = _optional_float(
                _first_value(row, ("parallax_error_mas", "parallax_error", "parallax_err")),
                field="parallax_error_mas",
                row_number=row_number,
            )
            extinction_mag = _optional_float(
                _first_value(
                    row,
                    ("extinction_mag", "ag_gspphot", "a_band", "a_g", "ag", "a_v", "av"),
                ),
                field="extinction_mag",
                row_number=row_number,
            )
            extinction_error_mag = _optional_float(
                _first_value(
                    row,
                    (
                        "extinction_error_mag",
                        "ag_gspphot_error",
                        "extinction_error",
                        "a_band_error",
                    ),
                ),
                field="extinction_error_mag",
                row_number=row_number,
            )
            distance_pc = _optional_float(
                _first_value(row, ("distance_pc", "distance_gspphot", "distance")),
                field="distance_pc",
                row_number=row_number,
            )
            distance_lower_pc = _optional_float(
                _first_value(row, ("distance_lower_pc", "distance_gspphot_lower", "distance_lower")),
                field="distance_lower_pc",
                row_number=row_number,
            )
            distance_upper_pc = _optional_float(
                _first_value(row, ("distance_upper_pc", "distance_gspphot_upper", "distance_upper")),
                field="distance_upper_pc",
                row_number=row_number,
            )
            distance_source = _first_value(row, ("distance_source", "distance_method"))
            phot_g_mean_flux_over_error = _optional_float(
                _first_value(row, ("phot_g_mean_flux_over_error", "g_flux_over_error", "g_snr")),
                field="phot_g_mean_flux_over_error",
                row_number=row_number,
            )
            phot_bp_rp_excess_factor = _optional_float(
                _first_value(row, ("phot_bp_rp_excess_factor", "bp_rp_excess_factor")),
                field="phot_bp_rp_excess_factor",
                row_number=row_number,
            )
            ruwe = _optional_float(_first_value(row, ("ruwe",)), field="ruwe", row_number=row_number)
            duplicated_source = _optional_bool(
                _first_value(row, ("duplicated_source", "duplicate_source")),
                field="duplicated_source",
                row_number=row_number,
            )
            visibility_raw = _optional_float(
                _first_value(row, ("visibility_periods_used", "visibility_periods")),
                field="visibility_periods_used",
                row_number=row_number,
            )
            visibility_periods_used = None if visibility_raw is None else int(visibility_raw)
            if visibility_raw is not None and visibility_raw != visibility_periods_used:
                raise ValueError(
                    f"catalog row {row_number}: visibility_periods_used must be an integer; got {visibility_raw!r}"
                )
            phot_variable_flag = _first_value(row, ("phot_variable_flag", "variable_flag"))
            catalog_name = _first_value(row, ("catalog_name", "catalog", "catalogue"))
            photometric_system = _first_value(row, ("photometric_system", "mag_system", "system"))
            photometric_band = _first_value(row, ("photometric_band", "band", "passband"))
            if has_gaia_g:
                catalog_name = catalog_name or "Gaia DR3"
                photometric_system = photometric_system or "Gaia Vega"
                photometric_band = photometric_band or "G"
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
                    catalog_name=catalog_name or "unknown",
                    photometric_system=photometric_system or "unknown",
                    photometric_band=photometric_band or "unknown",
                    magnitude_error=magnitude_error,
                    color_name=color_name,
                    parallax_mas=parallax_mas,
                    parallax_error_mas=parallax_error_mas,
                    extinction_mag=extinction_mag,
                    extinction_error_mag=extinction_error_mag,
                    distance_pc=distance_pc,
                    distance_lower_pc=distance_lower_pc,
                    distance_upper_pc=distance_upper_pc,
                    distance_source=distance_source,
                    phot_g_mean_flux_over_error=phot_g_mean_flux_over_error,
                    phot_bp_rp_excess_factor=phot_bp_rp_excess_factor,
                    ruwe=ruwe,
                    duplicated_source=duplicated_source,
                    visibility_periods_used=visibility_periods_used,
                    phot_variable_flag=phot_variable_flag,
                )
            )
    return tuple(sources)
