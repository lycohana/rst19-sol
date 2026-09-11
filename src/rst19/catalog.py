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


def _first_named_value(row: dict[str, str], names: Iterable[str]) -> tuple[str | None, str | None]:
    """返回第一个非空值及其规范化列名。

    CSV 表头在历史文件中并不总是小写。读取时进行大小写不敏感匹配，
    但只返回规范化后的列名，避免把表头的写法误当成字段的物理语义。
    """

    folded: dict[str, str | None] = {}
    for raw_name, value in row.items():
        if raw_name is None:
            continue
        folded.setdefault(str(raw_name).strip().lower(), value)
    for name in names:
        canonical_name = str(name).strip().lower()
        value = row.get(name)
        if value is None or not str(value).strip():
            value = folded.get(canonical_name)
        if value is not None and str(value).strip():
            return canonical_name, str(value).strip()
    return None, None


def _first_value(row: dict[str, str], names: Iterable[str]) -> str | None:
    """返回第一个非空值，并兼容大小写不同的历史 CSV 表头。"""

    return _first_named_value(row, names)[1]


def _canonical_extinction_band(value: str | None) -> str:
    if value is None or not str(value).strip():
        return "unknown"
    text = str(value).strip()
    key = "".join(character for character in text.casefold() if character.isalnum())
    if key in {"g", "gaiag", "gaiadr3g"}:
        return "G"
    if key in {"v", "johnsonv", "johnsoncousinsv"}:
        return "V"
    if key in {"a0", "azero", "a05414nm", "5414nm"}:
        return "A0(541.4 nm)"
    if key in {"unknown", "unk", "na", "n/a", "none"}:
        return "unknown"
    return text


def _canonical_extinction_system(value: str | None) -> str:
    if value is None or not str(value).strip():
        return "unknown"
    text = str(value).strip()
    key = "".join(character for character in text.casefold() if character.isalnum())
    if key in {"gaia", "gaiavega", "gaiadr3", "gaiadr3vega"}:
        return "Gaia"
    if key in {"johnson", "johnsonv", "johnsoncousins", "johnsoncousinsv"}:
        return "Johnson"
    if key in {"monochromatic", "monochromatic5414nm", "a0"}:
        return "monochromatic"
    if key in {"unknown", "unk", "na", "none"}:
        return "unknown"
    return text


_EXTINCTION_VALUE_FIELDS = (
    "ag_gspphot",
    "azero_gspphot",
    "a0_gspphot",
    "a0",
    "azero",
    "a_v",
    "av",
    "a_band",
    "a_g",
    "ag",
    "extinction_mag",
)


def _extinction_field_kind(field: str | None) -> str:
    if field == "ag_gspphot":
        return "gaia_g"
    if field in {"azero_gspphot", "a0_gspphot", "a0", "azero"}:
        return "a0"
    if field in {"a_v", "av"}:
        return "johnson_v"
    return "unknown"


def _inferred_extinction_semantics(field: str | None) -> tuple[str, str, str]:
    if field == "ag_gspphot":
        return "G", "Gaia", "Gaia DR3 GSP-Phot: ag_gspphot"
    if field in {"azero_gspphot", "a0_gspphot"}:
        return "A0(541.4 nm)", "monochromatic", f"Gaia DR3 GSP-Phot: {field}"
    if field in {"a0", "azero"}:
        return "A0(541.4 nm)", "monochromatic", f"CSV column: {field}"
    if field in {"a_v", "av"}:
        return "V", "Johnson", f"CSV column: {field}"
    if field is not None:
        return "unknown", "unknown", f"CSV column: {field}"
    return "unknown", "unknown", "unknown"


def _numeric_values_equal(left: str, right: str) -> bool:
    try:
        left_value = float(left)
        right_value = float(right)
    except (TypeError, ValueError):
        return left.strip() == right.strip()
    return math.isfinite(left_value) and math.isfinite(right_value) and math.isclose(
        left_value, right_value, rel_tol=1e-9, abs_tol=1e-12
    )


def _resolve_extinction_semantics(
    row: dict[str, str],
    *,
    row_number: int,
) -> tuple[str | None, str | None, str, str, str]:
    """解析消光值及其 provenance，拒绝互相冲突的已知列。

    ``extinction_mag``、``a_band``、``a_g`` 和 ``ag`` 都是兼容性别名，
    本身不能证明波段。只有字段名明确指向 Gaia GSP-Phot、A_V 或 A0
    时才推断语义；显式的 ``extinction_*`` 元数据可以为通用值补充语义。
    """

    present: list[tuple[str, str]] = []
    for field in _EXTINCTION_VALUE_FIELDS:
        value = _first_value(row, (field,))
        if value is not None:
            present.append((field, value))
    selected_field, selected_value = (present[0] if present else (None, None))
    explicit_band = _first_value(row, ("extinction_band", "ext_band", "extinction_passband"))
    explicit_system = _first_value(
        row,
        ("extinction_system", "extinction_photometric_system", "ext_system"),
    )
    explicit_source = _first_value(
        row,
        ("extinction_source", "extinction_provenance", "ext_source"),
    )

    known_kinds = {_extinction_field_kind(field) for field, _ in present}
    known_kinds.discard("unknown")
    if len(known_kinds) > 1:
        fields = ", ".join(field for field, _ in present)
        raise ValueError(
            f"catalog row {row_number}: conflicting extinction columns ({fields}); "
            "provide one extinction value with explicit semantics"
        )

    # A Gaia export contains both the source-specific ag_gspphot column and
    # the historical extinction_mag alias.  Permit that compatibility pair
    # only when their values agree; never silently choose between differing
    # values.
    if selected_field is not None and _extinction_field_kind(selected_field) != "unknown":
        selected_kind = _extinction_field_kind(selected_field)
        for field, value in present[1:]:
            kind = _extinction_field_kind(field)
            if kind == "unknown" and not _numeric_values_equal(selected_value or "", value):
                raise ValueError(
                    f"catalog row {row_number}: extinction column {field} conflicts with "
                    f"{selected_field}"
                )
            if kind not in {"unknown", selected_kind}:
                raise ValueError(
                    f"catalog row {row_number}: extinction column {field} conflicts with "
                    f"{selected_field}"
                )

    inferred_band, inferred_system, inferred_source = _inferred_extinction_semantics(selected_field)
    band = _canonical_extinction_band(explicit_band) if explicit_band is not None else inferred_band
    system = _canonical_extinction_system(explicit_system) if explicit_system is not None else inferred_system
    if explicit_band is not None and inferred_band != "unknown":
        if _canonical_extinction_band(explicit_band) != _canonical_extinction_band(inferred_band):
            raise ValueError(
                f"catalog row {row_number}: extinction_band={explicit_band!r} conflicts with {selected_field}"
            )
    if explicit_system is not None and inferred_system != "unknown":
        if _canonical_extinction_system(explicit_system) != _canonical_extinction_system(inferred_system):
            raise ValueError(
                f"catalog row {row_number}: extinction_system={explicit_system!r} conflicts with {selected_field}"
            )
    source = str(explicit_source).strip() if explicit_source is not None else inferred_source
    return selected_field, selected_value, band, system, source or "unknown"


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
    magnitude_source: str = "unknown"
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
    # Extinction provenance is separate from the historical numeric alias
    # ``extinction_mag``.  ``unknown`` is intentional: an old generic CSV
    # column must not be silently interpreted as Gaia G or Johnson V.
    extinction_band: str = "unknown"
    extinction_system: str = "unknown"
    extinction_source: str = "unknown"

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
        magnitude_source = "unknown" if self.magnitude_source is None else str(self.magnitude_source).strip()
        object.__setattr__(self, "magnitude_source", magnitude_source or "unknown")
        object.__setattr__(self, "extinction_band", _canonical_extinction_band(self.extinction_band))
        object.__setattr__(self, "extinction_system", _canonical_extinction_system(self.extinction_system))
        extinction_source = "unknown" if self.extinction_source is None else str(self.extinction_source).strip()
        object.__setattr__(self, "extinction_source", extinction_source or "unknown")
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
        if self.extinction_compatibility() == "mismatch":
            raise ValueError(
                f"catalog source {self.source_id!r} has extinction semantics "
                f"({self.extinction_system}/{self.extinction_band}) incompatible with "
                f"photometry ({self.photometric_system}/{self.photometric_band})"
            )

    def extinction_compatibility(self) -> str:
        """返回消光与源光度元数据的兼容性。

        返回值为 ``compatible``、``mismatch``、``unknown`` 或
        ``not_applicable``。未知语义不会被升级为兼容；这让旧 CSV 可以
        继续读取，同时为需要严格绝对星等的调用方提供安全门控。
        """

        if self.extinction_mag is None:
            return "not_applicable"
        extinction_band = _canonical_extinction_band(self.extinction_band)
        photometric_band = _canonical_extinction_band(self.photometric_band)
        if extinction_band == "unknown" or photometric_band == "unknown":
            return "unknown"
        if extinction_band != photometric_band:
            return "mismatch"
        extinction_system = _canonical_extinction_system(self.extinction_system)
        photometric_system = _canonical_extinction_system(self.photometric_system)
        if extinction_system == "unknown" or photometric_system == "unknown":
            return "unknown"
        return "compatible" if extinction_system == photometric_system else "mismatch"

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
            magnitude_source=self.magnitude_source,
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
            extinction_band=self.extinction_band,
            extinction_system=self.extinction_system,
            extinction_source=self.extinction_source,
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
            "magnitude_source": self.magnitude_source,
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
            "extinction_band": self.extinction_band,
            "extinction_system": self.extinction_system,
            "extinction_source": self.extinction_source,
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
            explicit_magnitude_source = _first_value(
                row,
                ("magnitude_source", "mag_source", "magnitude_provenance", "mag_provenance"),
            )
            gaia_magnitude_raw = _first_value(row, ("phot_g_mean_mag",))
            generic_magnitude_raw = _first_value(row, ("magnitude", "apparent_magnitude", "mag"))
            if gaia_magnitude_raw is not None and generic_magnitude_raw is not None:
                if not _numeric_values_equal(gaia_magnitude_raw, generic_magnitude_raw):
                    raise ValueError(
                        f"catalog row {row_number}: magnitude and phot_g_mean_mag conflict; "
                        "declare one magnitude source in a separate catalogue"
                    )
            source_text = explicit_magnitude_source.casefold() if explicit_magnitude_source else ""
            magnitude_is_gaia = gaia_magnitude_raw is not None and (
                explicit_magnitude_source is None
                or "gaia" in source_text
                or "phot_g" in source_text
            )
            if magnitude_is_gaia:
                magnitude_raw = gaia_magnitude_raw
                magnitude_source = explicit_magnitude_source or "Gaia DR3 phot_g_mean_mag"
            else:
                magnitude_raw = generic_magnitude_raw or gaia_magnitude_raw
                magnitude_source = explicit_magnitude_source or (
                    "CSV magnitude column" if generic_magnitude_raw is not None else "unknown"
                )
            magnitude = _optional_float(
                magnitude_raw,
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
            (
                extinction_field,
                extinction_value,
                extinction_band,
                extinction_system,
                extinction_source,
            ) = _resolve_extinction_semantics(row, row_number=row_number)
            extinction_mag = _optional_float(
                extinction_value,
                field=extinction_field or "extinction_mag",
                row_number=row_number,
            )
            extinction_error_mag = _optional_float(
                _first_value(
                    row,
                    (
                        "extinction_error_mag",
                        "ag_gspphot_error",
                        "azero_gspphot_error",
                        "a0_gspphot_error",
                        "a0_error",
                        "azero_error",
                        "extinction_error",
                        "a_band_error",
                        "a_v_error",
                        "av_error",
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
            if magnitude_is_gaia:
                catalog_name = catalog_name or "Gaia DR3"
                photometric_system = photometric_system or "Gaia Vega"
                photometric_band = photometric_band or "G"
            try:
                source = CatalogSource(
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
                    magnitude_source=magnitude_source,
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
                    extinction_band=extinction_band,
                    extinction_system=extinction_system,
                    extinction_source=extinction_source,
                )
            except ValueError as exc:
                raise ValueError(f"catalog row {row_number}: {exc}") from exc
            sources.append(source)
    return tuple(sources)
