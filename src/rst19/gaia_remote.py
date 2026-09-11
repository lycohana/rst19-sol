"""公共 Gaia DR3 TAP 接入。

这个模块只负责构造安全的 ADQL、访问 Gaia TAP 以及把 TAP 的 CSV/JSON
表格响应规整成可供 rst19 离线星表读取器消费的行字典。它刻意不导入
``rst19.catalog``，因此导入本模块不会触发网络访问，也不会把远程服务
耦合进已有的离线星表实现。

``radius`` 的单位是度，这是 ADQL ``CIRCLE`` 的单位。返回字典同时保留
Gaia 原始列和 ``CatalogSource`` 兼容别名；本模块不做光度拟合、零点
拟合或波段转换。
"""

from __future__ import annotations

import csv
import io
import json
import math
import socket
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any


DEFAULT_GAIA_TAP_SYNC_URL = "https://gea.esac.esa.int/tap-server/tap/sync"
GAIA_DR3_TABLE = "gaiadr3.gaia_source"

# Keep this list in the SELECT clause and in the response contract.  The
# columns are deliberately explicit so a later schema change cannot silently
# remove astrometric or quality information needed by downstream matching.
GAIA_REQUIRED_COLUMNS = (
    "source_id",
    "ra",
    "dec",
    "ref_epoch",
    "pmra",
    "pmdec",
    "phot_g_mean_mag",
    "phot_g_mean_flux",
    "phot_g_mean_flux_error",
    "phot_g_mean_flux_over_error",
    "phot_bp_mean_mag",
    "phot_rp_mean_mag",
    "phot_bp_rp_excess_factor",
    "parallax",
    "parallax_error",
    "ruwe",
    "duplicated_source",
    "visibility_periods_used",
    "phot_variable_flag",
    # Gaia DR3 copies these GSP-Phot estimates into gaia_source. They are
    # nullable per source, but keeping the columns enables an explicit
    # absolute-magnitude audit downstream.
    "distance_gspphot",
    "distance_gspphot_lower",
    "distance_gspphot_upper",
    "ag_gspphot",
    "ag_gspphot_lower",
    "ag_gspphot_upper",
    "ebpminrp_gspphot",
)

# These are the columns written by ``write_catalog_csv``.  The first group is
# understood by the existing offline catalog reader; the second group keeps
# the Gaia names and quality fields available for later audits.
CATALOG_COMPAT_COLUMNS = (
    "source_id",
    "ra_deg",
    "dec_deg",
    "magnitude",
    "magnitude_error",
    "color",
    "color_name",
    "pmra_mas_yr",
    "pmdec_mas_yr",
    "ref_epoch",
    "parallax_mas",
    "parallax_error_mas",
    "distance_pc",
    "distance_lower_pc",
    "distance_upper_pc",
    "distance_source",
    "extinction_mag",
    "extinction_error_mag",
    "extinction_band",
    "extinction_system",
    "extinction_source",
    "catalog_name",
    "photometric_system",
    "photometric_band",
    "ra",
    "dec",
    "pmra",
    "pmdec",
    "phot_g_mean_mag",
    "phot_g_mean_flux",
    "phot_g_mean_flux_error",
    "phot_g_mean_flux_over_error",
    "phot_bp_mean_mag",
    "phot_rp_mean_mag",
    "phot_bp_rp_excess_factor",
    "parallax",
    "parallax_error",
    "ruwe",
    "duplicated_source",
    "visibility_periods_used",
    "phot_variable_flag",
    "distance_gspphot",
    "distance_gspphot_lower",
    "distance_gspphot_upper",
    "ag_gspphot",
    "ag_gspphot_lower",
    "ag_gspphot_upper",
    "ebpminrp_gspphot",
)

_NUMERIC_RESPONSE_COLUMNS = (
    "ra",
    "dec",
    "ref_epoch",
    "pmra",
    "pmdec",
    "phot_g_mean_mag",
    "phot_g_mean_flux",
    "phot_g_mean_flux_error",
    "phot_g_mean_flux_over_error",
    "phot_bp_mean_mag",
    "phot_rp_mean_mag",
    "phot_bp_rp_excess_factor",
    "parallax",
    "parallax_error",
    "ruwe",
    "visibility_periods_used",
    "distance_gspphot",
    "distance_gspphot_lower",
    "distance_gspphot_upper",
    "ag_gspphot",
    "ag_gspphot_lower",
    "ag_gspphot_upper",
    "ebpminrp_gspphot",
    # Optional semantic extinction columns are accepted when a local CSV or
    # JSON proxy supplies them.  They are not required in the Gaia cone query
    # because the standard query uses ag_gspphot as the G-band value.
    "azero_gspphot",
    "azero_gspphot_lower",
    "azero_gspphot_upper",
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


class GaiaError(Exception):
    """所有 Gaia 接入错误的基类。"""


class GaiaInputError(GaiaError, ValueError):
    """查询参数或 TAP endpoint 不合法。"""


class GaiaResponseError(GaiaError, ValueError):
    """服务返回的内容不是可解析的 Gaia 表格。"""


class GaiaMissingColumnsError(GaiaResponseError):
    """TAP 表格响应缺少下游契约要求的列。"""

    def __init__(
        self,
        missing: Iterable[str],
        available: Iterable[str] = (),
        *,
        response_format: str = "table",
    ) -> None:
        self.missing = tuple(missing)
        self.available = tuple(available)
        self.response_format = response_format
        missing_text = ", ".join(self.missing) or "<unknown>"
        available_text = ", ".join(self.available) or "<none>"
        super().__init__(
            f"Gaia {response_format} response is missing required columns: "
            f"{missing_text}; available columns: {available_text}"
        )


class GaiaNetworkError(GaiaError, OSError):
    """连接 Gaia TAP 时发生的非 HTTP 网络错误。"""


class GaiaTimeoutError(GaiaNetworkError):
    """Gaia TAP 请求超时。"""


class GaiaHTTPError(GaiaNetworkError):
    """Gaia TAP 返回非 2xx HTTP 状态。"""

    def __init__(self, status_code: int, url: str, reason: object = "", body: str = "") -> None:
        self.status_code = int(status_code)
        self.url = url
        self.reason = str(reason)
        self.body = body
        message = f"Gaia TAP HTTP error {self.status_code} for {url}"
        if self.reason:
            message += f": {self.reason}"
        if body:
            message += f"; response body: {body}"
        super().__init__(message)


def _coerce_finite_float(value: object, *, name: str) -> float:
    if isinstance(value, bool):
        raise GaiaInputError(f"{name} must be a finite number, not a boolean")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise GaiaInputError(f"{name} must be a finite number; got {value!r}") from exc
    if not math.isfinite(parsed):
        raise GaiaInputError(f"{name} must be a finite number; got {value!r}")
    return parsed


def _format_adql_float(value: float) -> str:
    """以固定的 Python 数值格式输出，避免原始用户文本进入 ADQL。"""

    if value == 0.0:
        value = 0.0
    return format(value, ".15g")


def _validate_coordinates(ra: object, dec: object, radius: object) -> tuple[float, float, float]:
    ra_value = _coerce_finite_float(ra, name="ra")
    dec_value = _coerce_finite_float(dec, name="dec")
    radius_value = _coerce_finite_float(radius, name="radius")
    if not 0.0 <= ra_value < 360.0:
        raise GaiaInputError(f"ra must be in [0, 360) degrees; got {ra_value}")
    if not -90.0 <= dec_value <= 90.0:
        raise GaiaInputError(f"dec must be in [-90, 90] degrees; got {dec_value}")
    if not 0.0 < radius_value <= 180.0:
        raise GaiaInputError(f"radius must be greater than 0 and at most 180 degrees; got {radius_value}")
    return ra_value, dec_value, radius_value


def _validate_limit(limit: object | None) -> int | None:
    if limit is None:
        return None
    if isinstance(limit, bool):
        raise GaiaInputError("limit must be a positive integer, not a boolean")
    if isinstance(limit, float):
        if not math.isfinite(limit) or not limit.is_integer():
            raise GaiaInputError(f"limit must be a positive integer; got {limit!r}")
    try:
        parsed = int(limit)
    except (TypeError, ValueError, OverflowError) as exc:
        raise GaiaInputError(f"limit must be a positive integer; got {limit!r}") from exc
    if parsed < 1:
        raise GaiaInputError(f"limit must be a positive integer; got {limit!r}")
    return parsed


def _validate_timeout(timeout: object) -> float:
    timeout_value = _coerce_finite_float(timeout, name="timeout")
    if timeout_value <= 0.0:
        raise GaiaInputError(f"timeout must be greater than 0 seconds; got {timeout_value}")
    return timeout_value


def _validate_magnitude_limit(value: object | None, *, name: str) -> float | None:
    """Validate an optional Gaia G magnitude cut before it enters ADQL."""

    if value is None:
        return None
    parsed = _coerce_finite_float(value, name=name)
    if not -50.0 <= parsed <= 50.0:
        raise GaiaInputError(f"{name} must be in [-50, 50] magnitudes; got {parsed}")
    return parsed


def _normalise_response_format(response_format: object, *, allow_auto: bool = False) -> str:
    if response_format is None and allow_auto:
        return "auto"
    if not isinstance(response_format, str):
        raise GaiaInputError(f"response_format must be csv or json; got {response_format!r}")
    value = response_format.strip().lower()
    aliases = {
        "text/csv": "csv",
        "application/csv": "csv",
        "application/json": "json",
        "text/json": "json",
    }
    value = aliases.get(value, value)
    allowed = {"csv", "json"}
    if allow_auto:
        allowed.add("auto")
    if value not in allowed:
        choices = "csv or json" + (" or auto" if allow_auto else "")
        raise GaiaInputError(f"response_format must be {choices}; got {response_format!r}")
    return value


def build_gaia_adql(
    ra: object,
    dec: object,
    radius: object,
    *,
    limit: object | None = None,
    min_g_mag: object | None = None,
    max_g_mag: object | None = None,
) -> str:
    """构造 Gaia DR3 cone-search ADQL。

    参数均经过数值解析后再格式化，原始字符串不会直接拼进 ADQL；因此
    类似 ``"1); DROP TABLE ..."`` 的输入会在构造阶段失败。``radius``
    使用度，``limit`` 为可选的正整数 ``TOP`` 限制。
    """

    ra_value, dec_value, radius_value = _validate_coordinates(ra, dec, radius)
    limit_value = _validate_limit(limit)
    min_g_value = _validate_magnitude_limit(min_g_mag, name="min_g_mag")
    max_g_value = _validate_magnitude_limit(max_g_mag, name="max_g_mag")
    if min_g_value is not None and max_g_value is not None and min_g_value > max_g_value:
        raise GaiaInputError("min_g_mag cannot be greater than max_g_mag")
    top = f"TOP {limit_value} " if limit_value is not None else ""
    selected_columns = ",\n       ".join(GAIA_REQUIRED_COLUMNS)
    conditions = [
        "1 = CONTAINS(\n"
        "  POINT('ICRS', ra, dec),\n"
        f"  CIRCLE('ICRS', {_format_adql_float(ra_value)}, "
        f"{_format_adql_float(dec_value)}, {_format_adql_float(radius_value)})\n"
        ")"
    ]
    if min_g_value is not None:
        conditions.append(f"phot_g_mean_mag >= {_format_adql_float(min_g_value)}")
    if max_g_value is not None:
        conditions.append(f"phot_g_mean_mag <= {_format_adql_float(max_g_value)}")
    return (
        f"SELECT {top}{selected_columns}\n"
        f"FROM {GAIA_DR3_TABLE}\n"
        "WHERE " + "\n  AND ".join(conditions)
    )


# Descriptive aliases make the construction API discoverable without creating
# a second implementation or a second source of query semantics.
build_adql_query = build_gaia_adql
construct_gaia_adql = build_gaia_adql


def _validate_endpoint(endpoint: object) -> str:
    if not isinstance(endpoint, str) or not endpoint.strip():
        raise GaiaInputError("endpoint must be a non-empty HTTP(S) URL")
    endpoint_value = endpoint.strip()
    parsed = urllib.parse.urlsplit(endpoint_value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise GaiaInputError(f"endpoint must be an HTTP(S) URL; got {endpoint!r}")
    return endpoint_value


def build_tap_request_url(
    adql: str,
    endpoint: str = DEFAULT_GAIA_TAP_SYNC_URL,
    response_format: str = "csv",
    *,
    maxrec: object | None = None,
) -> str:
    """把 ADQL 编码成 Gaia TAP sync GET URL，不执行网络请求。

    ``MAXREC`` 是 TAP 层的返回行数上限；当调用方已经给出 ``TOP``
    限制时，把同一个值传给它可以避免服务端采用更小的默认上限而静默
    截断结果。它不改变 ADQL 本身，也不保证查询覆盖了所有可能的源。
    """

    if not isinstance(adql, str) or not adql.strip():
        raise GaiaInputError("adql must be a non-empty string")
    endpoint_value = _validate_endpoint(endpoint)
    format_value = _normalise_response_format(response_format)
    maxrec_value = _validate_limit(maxrec)
    params = {
        "REQUEST": "doQuery",
        "LANG": "ADQL",
        "FORMAT": format_value,
        "PHASE": "RUN",
        "QUERY": adql,
    }
    if maxrec_value is not None:
        params["MAXREC"] = str(maxrec_value)
    encoded = urllib.parse.urlencode(params, quote_via=urllib.parse.quote)
    separator = "&" if "?" in endpoint_value else "?"
    return f"{endpoint_value}{separator}{encoded}"


def _decode_text(payload: bytes | bytearray | str, *, context: str) -> str:
    if isinstance(payload, (bytes, bytearray)):
        try:
            return bytes(payload).decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise GaiaResponseError(f"Gaia {context} response is not valid UTF-8") from exc
    if isinstance(payload, str):
        return payload
    raise GaiaResponseError(f"Gaia {context} response must be bytes or text; got {type(payload).__name__}")


def _preview_bytes(payload: object, *, max_length: int = 500) -> str:
    if isinstance(payload, (bytes, bytearray)):
        text = bytes(payload).decode("utf-8", "replace")
    else:
        text = str(payload)
    return " ".join(text.strip().split())[:max_length]


def _read_error_body(error: object) -> str:
    reader = getattr(error, "read", None)
    if not callable(reader):
        return ""
    try:
        return _preview_bytes(reader())
    except Exception:
        return ""


def _is_timeout_reason(reason: object) -> bool:
    if isinstance(reason, (TimeoutError, socket.timeout)):
        return True
    return "timed out" in str(reason).lower() or "timeout" in str(reason).lower()


def download_gaia(
    ra: object,
    dec: object,
    radius: object,
    *,
    limit: object | None = None,
    timeout: object = 30.0,
    endpoint: str = DEFAULT_GAIA_TAP_SYNC_URL,
    response_format: str = "csv",
    min_g_mag: object | None = None,
    max_g_mag: object | None = None,
    opener: Callable[..., Any] | None = None,
) -> bytes:
    """显式执行一次 Gaia TAP 下载并返回原始响应字节。

    ``opener`` 仅用于测试或注入自定义 urllib opener；默认使用
    ``urllib.request.urlopen``。模块导入和查询构造都不会调用此函数。
    """

    format_value = _normalise_response_format(response_format)
    timeout_value = _validate_timeout(timeout)
    adql = build_gaia_adql(
        ra,
        dec,
        radius,
        limit=limit,
        min_g_mag=min_g_mag,
        max_g_mag=max_g_mag,
    )
    url = build_tap_request_url(
        adql,
        endpoint=endpoint,
        response_format=format_value,
        maxrec=limit,
    )
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "text/csv, text/plain;q=0.9" if format_value == "csv" else "application/json",
            "User-Agent": "rst19-gaia-remote/1.0",
        },
        method="GET",
    )
    open_url = opener or urllib.request.urlopen
    response: Any | None = None
    try:
        response = open_url(request, timeout=timeout_value)
        status = getattr(response, "status", None)
        if status is None:
            getcode = getattr(response, "getcode", None)
            status = getcode() if callable(getcode) else None
        if status is not None and int(status) >= 400:
            body = _read_error_body(response)
            raise GaiaHTTPError(int(status), url, getattr(response, "reason", ""), body)
        payload = response.read()
        if isinstance(payload, str):
            return payload.encode("utf-8")
        if not isinstance(payload, (bytes, bytearray)):
            raise GaiaResponseError(
                f"Gaia TAP response read() returned {type(payload).__name__}, expected bytes"
            )
        return bytes(payload)
    except GaiaNetworkError:
        raise
    except urllib.error.HTTPError as exc:
        body = _read_error_body(exc)
        raise GaiaHTTPError(exc.code, exc.geturl(), exc.reason, body) from exc
    except (socket.timeout, TimeoutError) as exc:
        raise GaiaTimeoutError(f"Gaia TAP request timed out after {timeout_value:g} seconds: {url}") from exc
    except urllib.error.URLError as exc:
        if _is_timeout_reason(exc.reason):
            raise GaiaTimeoutError(f"Gaia TAP request timed out after {timeout_value:g} seconds: {url}") from exc
        raise GaiaNetworkError(f"Gaia TAP network error for {url}: {exc.reason}") from exc
    except OSError as exc:
        if _is_timeout_reason(exc):
            raise GaiaTimeoutError(f"Gaia TAP request timed out after {timeout_value:g} seconds: {url}") from exc
        raise GaiaNetworkError(f"Gaia TAP network error for {url}: {exc}") from exc
    finally:
        close = getattr(response, "close", None)
        if callable(close):
            close()


def _normalise_column_name(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()


def _prepare_headers(headers: Iterable[object], *, response_format: str) -> tuple[str, ...]:
    prepared: list[str] = []
    seen: set[str] = set()
    for raw_header in headers:
        header = _normalise_column_name(raw_header)
        if not header:
            raise GaiaResponseError(f"Gaia {response_format} response contains an empty column name")
        if header in seen:
            raise GaiaResponseError(f"Gaia {response_format} response contains duplicate column: {header}")
        seen.add(header)
        prepared.append(header)
    missing = [column for column in GAIA_REQUIRED_COLUMNS if column not in seen]
    if missing:
        raise GaiaMissingColumnsError(missing, prepared, response_format=response_format)
    return tuple(prepared)


def _stringify_cell(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value).strip()


def _finite_cell_number(value: str, *, field: str, row_number: int, response_format: str) -> float | None:
    if not value:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise GaiaResponseError(
            f"Gaia {response_format} row {row_number}: invalid numeric {field}={value!r}"
        ) from exc
    if not math.isfinite(parsed):
        raise GaiaResponseError(
            f"Gaia {response_format} row {row_number}: non-finite numeric {field}={value!r}"
        )
    return parsed


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


def _first_response_value(row: Mapping[str, str], names: Iterable[str]) -> tuple[str | None, str | None]:
    for name in names:
        value = row.get(name, "")
        if value is not None and str(value).strip():
            return name, str(value).strip()
    return None, None


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
    if key in {"unknown", "unk", "na", "none"}:
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
        return "A0(541.4 nm)", "monochromatic", f"response column: {field}"
    if field in {"a_v", "av"}:
        return "V", "Johnson", f"response column: {field}"
    if field is not None:
        return "unknown", "unknown", f"response column: {field}"
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
    row: Mapping[str, str],
    *,
    row_number: int,
    response_format: str,
) -> tuple[str | None, str | None, str, str, str]:
    """保留 Gaia/代理响应中的消光波段、系统和来源。

    具体字段优先于历史 ``extinction_mag`` 别名；通用别名本身不会被
    推断为 Gaia G。若响应同时携带了互相冲突的已知消光字段，则拒绝
    该行，避免在 JSON/CSV 归一化时悄悄覆盖物理语义。
    """

    present = [
        (field, value)
        for field in _EXTINCTION_VALUE_FIELDS
        for actual_field, value in [_first_response_value(row, (field,))]
        if actual_field is not None and value is not None
    ]
    selected_field, selected_value = present[0] if present else (None, None)
    explicit_band = _first_response_value(
        row, ("extinction_band", "ext_band", "extinction_passband")
    )[1]
    explicit_system = _first_response_value(
        row, ("extinction_system", "extinction_photometric_system", "ext_system")
    )[1]
    explicit_source = _first_response_value(
        row, ("extinction_source", "extinction_provenance", "ext_source")
    )[1]

    known_kinds = {_extinction_field_kind(field) for field, _ in present}
    known_kinds.discard("unknown")
    if len(known_kinds) > 1:
        fields = ", ".join(field for field, _ in present)
        raise GaiaResponseError(
            f"Gaia {response_format} row {row_number}: conflicting extinction columns ({fields})"
        )
    if selected_field is not None and _extinction_field_kind(selected_field) != "unknown":
        selected_kind = _extinction_field_kind(selected_field)
        for field, value in present[1:]:
            kind = _extinction_field_kind(field)
            if kind == "unknown" and not _numeric_values_equal(selected_value or "", value):
                raise GaiaResponseError(
                    f"Gaia {response_format} row {row_number}: extinction column {field} conflicts with "
                    f"{selected_field}"
                )
            if kind not in {"unknown", selected_kind}:
                raise GaiaResponseError(
                    f"Gaia {response_format} row {row_number}: extinction column {field} conflicts with "
                    f"{selected_field}"
                )

    inferred_band, inferred_system, inferred_source = _inferred_extinction_semantics(selected_field)
    band = _canonical_extinction_band(explicit_band) if explicit_band is not None else inferred_band
    system = _canonical_extinction_system(explicit_system) if explicit_system is not None else inferred_system
    if explicit_band is not None and inferred_band != "unknown":
        if _canonical_extinction_band(explicit_band) != _canonical_extinction_band(inferred_band):
            raise GaiaResponseError(
                f"Gaia {response_format} row {row_number}: extinction_band={explicit_band!r} conflicts with "
                f"{selected_field}"
            )
    if explicit_system is not None and inferred_system != "unknown":
        if _canonical_extinction_system(explicit_system) != _canonical_extinction_system(inferred_system):
            raise GaiaResponseError(
                f"Gaia {response_format} row {row_number}: extinction_system={explicit_system!r} conflicts with "
                f"{selected_field}"
            )
    source = str(explicit_source).strip() if explicit_source is not None else inferred_source
    return selected_field, selected_value, band, system, source or "unknown"


def _extinction_error_fields(field: str | None) -> tuple[str, ...]:
    if field == "ag_gspphot":
        return ("ag_gspphot_error", "ag_error")
    if field in {"azero_gspphot", "a0_gspphot"}:
        return (f"{field}_error", "azero_error", "a0_error")
    if field in {"a0", "azero"}:
        return (f"{field}_error", "a0_error", "azero_error")
    if field in {"a_v", "av"}:
        return (f"{field}_error", "a_v_error", "av_error")
    if field is not None:
        return (f"{field}_error",)
    return ()


def _extinction_bound_fields(field: str | None) -> tuple[str | None, str | None]:
    if field == "ag_gspphot":
        return "ag_gspphot_lower", "ag_gspphot_upper"
    if field in {"azero_gspphot", "a0_gspphot"}:
        return f"{field}_lower", f"{field}_upper"
    if field in {"a0", "azero", "a_v", "av", "a_band", "a_g", "ag", "extinction_mag"}:
        return f"{field}_lower", f"{field}_upper"
    return None, None


def _normalise_rows(
    headers: Iterable[str],
    records: Iterable[Mapping[str, object]],
    *,
    response_format: str,
) -> tuple[dict[str, str], ...]:
    header_tuple = tuple(headers)
    # ``_prepare_headers`` is normally called by each format parser, but keep
    # this guard so the internal normalization helper remains safe to reuse.
    _prepare_headers(header_tuple, response_format=response_format)
    normalized_rows: list[dict[str, str]] = []
    source_ids: set[str] = set()
    for row_number, record in enumerate(records, start=2):
        if not isinstance(record, Mapping):
            raise GaiaResponseError(
                f"Gaia {response_format} row {row_number} is not an object/table row"
            )
        canonical_record: dict[str, str] = {}
        for raw_name, raw_value in record.items():
            name = _normalise_column_name(raw_name)
            if name:
                canonical_record[name] = _stringify_cell(raw_value)
        row = {header: canonical_record.get(header, "") for header in header_tuple}
        source_id = row.get("source_id", "").strip()
        if not source_id:
            raise GaiaResponseError(f"Gaia {response_format} row {row_number}: missing source_id")
        if source_id in source_ids:
            raise GaiaResponseError(
                f"Gaia {response_format} row {row_number}: duplicate source_id={source_id!r}"
            )
        source_ids.add(source_id)

        parsed_numbers: dict[str, float | None] = {}
        for field in _NUMERIC_RESPONSE_COLUMNS:
            parsed_numbers[field] = _finite_cell_number(
                row.get(field, ""),
                field=field,
                row_number=row_number,
                response_format=response_format,
            )
        ra_value = parsed_numbers["ra"]
        dec_value = parsed_numbers["dec"]
        if ra_value is None:
            raise GaiaResponseError(f"Gaia {response_format} row {row_number}: missing ra")
        if dec_value is None:
            raise GaiaResponseError(f"Gaia {response_format} row {row_number}: missing dec")
        if not 0.0 <= ra_value <= 360.0:
            raise GaiaResponseError(f"Gaia {response_format} row {row_number}: ra out of range: {ra_value}")
        if not -90.0 <= dec_value <= 90.0:
            raise GaiaResponseError(f"Gaia {response_format} row {row_number}: dec out of range: {dec_value}")

        # Keep every response column, then add explicit aliases used by the
        # existing catalog reader.  The aliases are deliberately derived from
        # Gaia G/BP/RP values and are not a fitted or transformed photometry
        # result.
        output = dict(row)
        output["source_id"] = source_id
        output["ra_deg"] = row.get("ra", "")
        output["dec_deg"] = row.get("dec", "")
        output["magnitude"] = row.get("phot_g_mean_mag", "")
        g_mag_error = row.get("phot_g_mean_mag_error", "")
        if not g_mag_error:
            g_flux = parsed_numbers["phot_g_mean_flux"]
            g_flux_error = parsed_numbers["phot_g_mean_flux_error"]
            if g_flux is not None and g_flux > 0 and g_flux_error is not None and g_flux_error >= 0:
                # Gaia publishes the G uncertainty in flux space.  This is a
                # local first-order approximation for the CSV compatibility
                # field, not an official symmetric magnitude uncertainty.
                g_mag_error = _format_adql_float((2.5 / math.log(10.0)) * g_flux_error / g_flux)
        output["magnitude_error"] = g_mag_error
        output["phot_g_mean_mag_error"] = g_mag_error
        output["pmra_mas_yr"] = row.get("pmra", "")
        output["pmdec_mas_yr"] = row.get("pmdec", "")
        output["parallax_mas"] = row.get("parallax", "")
        output["parallax_error_mas"] = row.get("parallax_error", "")
        output["distance_pc"] = row.get("distance_gspphot", "")
        output["distance_lower_pc"] = row.get("distance_gspphot_lower", "")
        output["distance_upper_pc"] = row.get("distance_gspphot_upper", "")
        output["distance_source"] = (
            "Gaia DR3 GSP-Phot" if row.get("distance_gspphot", "") else ""
        )
        (
            extinction_field,
            extinction_value,
            extinction_band,
            extinction_system,
            extinction_source,
        ) = _resolve_extinction_semantics(
            row,
            row_number=row_number,
            response_format=response_format,
        )
        output["extinction_mag"] = extinction_value or ""
        error_field, error_value = _first_response_value(
            row,
            ("extinction_error_mag", *_extinction_error_fields(extinction_field)),
        )
        if error_value is not None:
            _finite_cell_number(
                error_value,
                field=error_field or "extinction_error_mag",
                row_number=row_number,
                response_format=response_format,
            )
            output["extinction_error_mag"] = error_value
        else:
            lower_field, upper_field = _extinction_bound_fields(extinction_field)
            lower = parsed_numbers.get(lower_field) if lower_field is not None else None
            upper = parsed_numbers.get(upper_field) if upper_field is not None else None
            if lower is not None and upper is not None and upper >= lower:
                # Gaia publishes 16th/84th percentiles. This compact
                # half-width is a local uncertainty summary, not an official
                # Gaia column.
                output["extinction_error_mag"] = _format_adql_float(0.5 * (upper - lower))
            else:
                output["extinction_error_mag"] = ""
        output["extinction_band"] = extinction_band
        output["extinction_system"] = extinction_system
        output["extinction_source"] = extinction_source
        output["phot_g_mean_flux_over_error"] = row.get("phot_g_mean_flux_over_error", "")
        output["phot_bp_rp_excess_factor"] = row.get("phot_bp_rp_excess_factor", "")
        output["ruwe"] = row.get("ruwe", "")
        output["duplicated_source"] = row.get("duplicated_source", "")
        output["visibility_periods_used"] = row.get("visibility_periods_used", "")
        output["phot_variable_flag"] = row.get("phot_variable_flag", "")
        output["catalog_name"] = row.get("catalog_name") or "Gaia DR3"
        output["photometric_system"] = row.get("photometric_system") or "Gaia Vega"
        output["photometric_band"] = row.get("photometric_band") or "G"

        bp_value = parsed_numbers["phot_bp_mean_mag"]
        rp_value = parsed_numbers["phot_rp_mean_mag"]
        if bp_value is not None and rp_value is not None:
            output["color"] = _format_adql_float(bp_value - rp_value)
            output["color_name"] = row.get("color_name") or "BP-RP"
        else:
            output["color"] = ""
            output["color_name"] = row.get("color_name", "")
        normalized_rows.append(output)
    return tuple(normalized_rows)


def parse_gaia_csv(payload: bytes | bytearray | str) -> tuple[dict[str, str], ...]:
    """解析 Gaia TAP CSV，并返回 CatalogSource 兼容的行字典。"""

    text = _decode_text(payload, context="CSV")
    reader = csv.DictReader(io.StringIO(text, newline=""))
    if reader.fieldnames is None:
        raise GaiaResponseError("Gaia CSV response has no header")
    raw_headers = tuple(reader.fieldnames)
    headers = _prepare_headers(raw_headers, response_format="CSV")
    records: list[dict[str, object]] = []
    for row_number, raw_row in enumerate(reader, start=2):
        if None in raw_row:
            raise GaiaResponseError(f"Gaia CSV row {row_number} has more fields than its header")
        records.append(
            {
                _normalise_column_name(raw_header): raw_row.get(raw_header)
                for raw_header in raw_headers
            }
        )
    return _normalise_rows(headers, records, response_format="CSV")


def _metadata_columns(metadata: object) -> list[object]:
    if metadata is None:
        return []
    if isinstance(metadata, Mapping):
        if "name" in metadata:
            return [metadata["name"]]
        return list(metadata.keys())
    if isinstance(metadata, (list, tuple)):
        names: list[object] = []
        for item in metadata:
            if isinstance(item, Mapping):
                if "name" in item:
                    names.append(item["name"])
                elif "column" in item:
                    names.append(item["column"])
                else:
                    raise GaiaResponseError("Gaia JSON metadata item has no column name")
            else:
                names.append(item)
        return names
    raise GaiaResponseError("Gaia JSON metadata/columns must be a list or object")


def _json_table(payload: object) -> tuple[tuple[str, ...], list[dict[str, object]]]:
    if isinstance(payload, list):
        if not payload:
            raise GaiaResponseError("Gaia JSON response has no columns or rows")
        if not all(isinstance(item, Mapping) for item in payload):
            raise GaiaResponseError("Gaia JSON row arrays require metadata/columns")
        raw_headers: list[object] = []
        seen: set[str] = set()
        for item in payload:
            for key in item:
                name = _normalise_column_name(key)
                if name and name not in seen:
                    seen.add(name)
                    raw_headers.append(key)
        headers = _prepare_headers(raw_headers, response_format="JSON")
        return headers, [dict(item) for item in payload]

    if not isinstance(payload, Mapping):
        raise GaiaResponseError("Gaia JSON response must be an object or row list")

    if "error" in payload and not any(key in payload for key in ("data", "rows", "results")):
        raise GaiaResponseError(f"Gaia JSON response reports an error: {_preview_bytes(payload['error'])}")

    data_key = next((key for key in ("data", "rows", "results") if key in payload), None)
    if data_key is None:
        # Accept a single row object as a small convenience for fixtures and
        # local proxies, but reject arbitrary error/status objects below.
        if "source_id" in {_normalise_column_name(key) for key in payload}:
            raw_headers = list(payload.keys())
            headers = _prepare_headers(raw_headers, response_format="JSON")
            return headers, [dict(payload)]
        detail = payload.get("message") or payload.get("error") or "no data/rows/results field"
        raise GaiaResponseError(f"Gaia JSON response has no tabular data: {_preview_bytes(detail)}")

    data = payload[data_key]
    metadata_key = next((key for key in ("metadata", "columns", "fields", "schema") if key in payload), None)
    metadata_headers = _metadata_columns(payload[metadata_key]) if metadata_key is not None else []

    if not isinstance(data, list):
        raise GaiaResponseError(f"Gaia JSON {data_key} field must be a list")
    if not data:
        headers = _prepare_headers(metadata_headers, response_format="JSON")
        return headers, []

    if all(isinstance(item, Mapping) for item in data):
        raw_headers = list(metadata_headers)
        seen = {_normalise_column_name(item) for item in raw_headers}
        for item in data:
            for key in item:
                name = _normalise_column_name(key)
                if name and name not in seen:
                    seen.add(name)
                    raw_headers.append(key)
        headers = _prepare_headers(raw_headers, response_format="JSON")
        return headers, [dict(item) for item in data]

    if not metadata_headers or not all(isinstance(item, (list, tuple)) for item in data):
        raise GaiaResponseError("Gaia JSON array rows require metadata/columns")
    headers = _prepare_headers(metadata_headers, response_format="JSON")
    records: list[dict[str, object]] = []
    for row_number, values in enumerate(data, start=2):
        if len(values) != len(headers):
            raise GaiaResponseError(
                f"Gaia JSON row {row_number} has {len(values)} values but {len(headers)} columns"
            )
        records.append(dict(zip(headers, values)))
    return headers, records


def parse_gaia_json(
    payload: bytes | bytearray | str | Mapping[str, object] | list[object],
) -> tuple[dict[str, str], ...]:
    """解析常见 Gaia TAP JSON 表格表示，并返回兼容行字典。"""

    if isinstance(payload, (bytes, bytearray, str)):
        text = _decode_text(payload, context="JSON")
        try:
            decoded: object = json.loads(text)
        except json.JSONDecodeError as exc:
            raise GaiaResponseError(f"Gaia JSON response is invalid JSON: {exc.msg}") from exc
    else:
        decoded = payload
    headers, records = _json_table(decoded)
    return _normalise_rows(headers, records, response_format="JSON")


def parse_gaia_response(
    payload: bytes | bytearray | str | Mapping[str, object] | list[object],
    response_format: str | None = None,
) -> tuple[dict[str, str], ...]:
    """按显式格式或内容首字符解析 Gaia CSV/JSON 响应。"""

    format_value = _normalise_response_format(response_format, allow_auto=True)
    if format_value == "auto":
        if isinstance(payload, (Mapping, list)):
            format_value = "json"
        else:
            text = _decode_text(payload, context="response")
            first = text.lstrip()[:1]
            format_value = "json" if first in {"{", "["} else "csv"
    if format_value == "csv":
        return parse_gaia_csv(payload)  # type: ignore[arg-type]
    return parse_gaia_json(payload)  # type: ignore[arg-type]


def query_gaia(
    ra: object,
    dec: object,
    radius: object,
    *,
    limit: object | None = None,
    timeout: object = 30.0,
    endpoint: str = DEFAULT_GAIA_TAP_SYNC_URL,
    response_format: str = "csv",
    min_g_mag: object | None = None,
    max_g_mag: object | None = None,
    opener: Callable[..., Any] | None = None,
) -> tuple[dict[str, str], ...]:
    """执行 Gaia TAP cone query 并解析成 CatalogSource 兼容行。"""

    format_value = _normalise_response_format(response_format)
    payload = download_gaia(
        ra,
        dec,
        radius,
        limit=limit,
        timeout=timeout,
        endpoint=endpoint,
        response_format=format_value,
        min_g_mag=min_g_mag,
        max_g_mag=max_g_mag,
        opener=opener,
    )
    rows = parse_gaia_response(payload, response_format=format_value)
    limit_value = _validate_limit(limit)
    if limit_value is not None:
        return rows[:limit_value]
    return rows


query_gaia_catalog = query_gaia
download_gaia_response = download_gaia


def write_catalog_csv(rows: Iterable[Mapping[str, object]], path: str | Path) -> Path:
    """把查询结果写成可被现有离线星表读取器消费的 UTF-8 CSV。"""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CATALOG_COMPAT_COLUMNS, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row_number, row in enumerate(rows, start=1):
            if not isinstance(row, Mapping):
                raise GaiaResponseError(f"catalog output row {row_number} is not a mapping")
            writer.writerow({column: _stringify_cell(row.get(column, "")) for column in CATALOG_COMPAT_COLUMNS})
    return output_path


__all__ = [
    "CATALOG_COMPAT_COLUMNS",
    "DEFAULT_GAIA_TAP_SYNC_URL",
    "GAIA_DR3_TABLE",
    "GAIA_REQUIRED_COLUMNS",
    "GaiaError",
    "GaiaHTTPError",
    "GaiaInputError",
    "GaiaMissingColumnsError",
    "GaiaNetworkError",
    "GaiaResponseError",
    "GaiaTimeoutError",
    "build_adql_query",
    "build_gaia_adql",
    "build_tap_request_url",
    "construct_gaia_adql",
    "download_gaia",
    "download_gaia_response",
    "parse_gaia_csv",
    "parse_gaia_json",
    "parse_gaia_response",
    "query_gaia",
    "query_gaia_catalog",
    "write_catalog_csv",
]
