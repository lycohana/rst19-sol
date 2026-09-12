from __future__ import annotations

import ast
import queue
from pathlib import Path
from types import SimpleNamespace

from rst19 import gui


def _absolute(status: str, value: float | None) -> SimpleNamespace:
    return SimpleNamespace(status=status, value=value, error=0.12)


def _row(
    *,
    status: str = "CALIBRATED",
    absolute_status: str = "VALID_NO_EXTINCTION",
    absolute_value: float | None = 4.2,
) -> SimpleNamespace:
    return SimpleNamespace(
        status=status,
        photometric_system="Gaia Vega",
        photometric_band="G",
        calibrated_magnitude=13.4,
        calibrated_magnitude_error=0.08,
        absolute_magnitude=_absolute(absolute_status, absolute_value),
    )


def test_gui_calibrated_magnitude_requires_a_valid_declared_calibration() -> None:
    text = gui._gui_format_calibrated_magnitude(
        13.4,
        status="VALID",
        system="Gaia Vega",
        band="G",
        error=0.08,
    )

    assert "m_G" in text
    assert "Gaia Vega/G" in text
    assert "13.400" in text
    assert "± 0.080" in text

    uncalibrated = gui._gui_format_calibrated_magnitude(
        13.4,
        status="INSTRUMENTAL",
        system="Gaia Vega",
        band="G",
    )
    assert "m_user" in uncalibrated
    assert "标准波段" not in uncalibrated

    missing_provenance = gui._gui_format_calibrated_magnitude(
        13.4,
        status="VALID",
        system="unknown",
        band="",
        wcs_available=True,
    )
    assert missing_provenance == "m_cal = 不可用（未声明系统/波段）"

    missing_wcs = gui._gui_format_calibrated_magnitude(
        13.4,
        status="VALID",
        system="Gaia Vega",
        band="G",
        wcs_available=False,
    )
    assert missing_wcs == "m_cal = 不可用（无 WCS，未进行星表匹配）"


def test_gui_headline_prefers_verified_calibrated_magnitude() -> None:
    faintest = SimpleNamespace(
        instrumental_magnitude=19.1,
        calibrated_magnitude=12.7,
        calibration_status="VALID",
        photometric_system="Gaia Vega",
        photometric_band="G",
    )

    title, value, calibrated = gui._gui_primary_faintest_display(
        faintest,
        SimpleNamespace(status="VALID", photometric_system="Gaia Vega", photometric_band="G"),
        wcs_verified=True,
    )

    assert title == "m_G,cal · 已标定"
    assert value == "12.70"
    assert calibrated is True

    fallback_title, fallback_value, fallback_calibrated = gui._gui_primary_faintest_display(
        faintest,
        SimpleNamespace(status="VALID", photometric_system="Gaia Vega", photometric_band="G"),
        wcs_verified=False,
    )
    assert fallback_title == "m_inst（未定标）"
    assert fallback_value == "19.10"
    assert fallback_calibrated is False


def test_gui_absolute_magnitude_hides_value_without_extinction() -> None:
    provisional = gui._gui_format_absolute_magnitude(
        _absolute("VALID_NO_EXTINCTION", 4.2),
        band="G",
    )
    assert provisional == "M_G = 不可用（未提供消光修正）"
    assert "4.2" not in provisional

    strict = gui._gui_format_absolute_magnitude(
        _absolute("VALID", 4.2),
        system="Gaia Vega",
        band="G",
    )
    assert strict == "M_G = 4.200 ± 0.120"

    missing_provenance = gui._gui_format_absolute_magnitude(
        _absolute("VALID", 4.2),
        system="unknown",
        band="G",
    )
    assert missing_provenance == "M_G = 不可用（未声明系统/波段）"


def test_gui_labels_model_distance_absolute_magnitude_without_calling_it_parallax() -> None:
    model = SimpleNamespace(
        status="VALID_MODEL_DISTANCE",
        value=4.2,
        error=0.31,
        distance_source="Gaia DR3 GSP-Phot",
    )

    rendered = gui._gui_format_absolute_magnitude(
        model,
        system="Gaia Vega",
        band="G",
    )

    assert rendered == "M_G（模型距离：Gaia DR3 GSP-Phot） = 4.200 ± 0.310"

    compact = gui._gui_format_absolute_magnitude(
        model,
        system="Gaia Vega",
        band="G",
        compact=True,
    )
    assert compact == "4.20·模型"


def test_gui_photometry_state_exposes_system_band_and_absolute_boundary() -> None:
    row = _row()
    status = gui._gui_photometry_status_text(
        row,
        calibration=SimpleNamespace(
            status="VALID",
            photometric_system="Gaia Vega",
            photometric_band="G",
        ),
        wcs_available=True,
    )

    assert "系统/波段 Gaia Vega/G" in status
    assert "M 未消光" in status

    no_wcs = gui._gui_photometry_context_summary(
        SimpleNamespace(photometric_calibration=None, source_photometry=()),
        wcs_available=False,
    )
    assert "仅显示 m_inst" in no_wcs
    assert "m_cal/M 不可用" in no_wcs


def test_active_single_frame_analysis_does_not_implicitly_match_or_calibrate() -> None:
    source_path = Path(gui.__file__)
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    app_class = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "StarfieldApp")
    run_analysis = next(node for node in app_class.body if isinstance(node, ast.FunctionDef) and node.name == "run_analysis")
    calls = [
        node
        for node in ast.walk(run_analysis)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "analyze_frame"
    ]

    assert len(calls) == 1
    keyword_names = {keyword.arg for keyword in calls[0].keywords if keyword.arg is not None}
    assert "catalog" not in keyword_names
    assert "wcs" not in keyword_names
    assert "fit_photometry" not in keyword_names


def test_gui_has_no_implicit_network_imports() -> None:
    source_path = Path(gui.__file__)
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imported_modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.add(node.module.split(".")[0])

    assert imported_modules.isdisjoint({"requests", "urllib", "httpx", "aiohttp", "socket"})
    assert "gaia_remote" not in source_path.read_text(encoding="utf-8")


def test_auto_reuse_requires_a_complete_matching_detector_fingerprint() -> None:
    detector_parameters = {
        "threshold_sigma": 4.0,
        "min_distance": 4,
        "max_sources": -1,
        "psf_fwhm": 2.0,
        "min_flux_snr": 5.0,
        "proposal_mode": "hybrid",
        "reject_linear_artifacts": 1,
        "enable_local_deblend": 0,
    }
    analysis = SimpleNamespace(detection=SimpleNamespace(parameters=detector_parameters))
    detector_kwargs = {
        "threshold_sigma": 4.0,
        "min_distance": 4,
        "max_sources": None,
        "psf_fwhm": 2.0,
        "min_flux_snr": 5.0,
        "proposal_mode": "hybrid",
        "reject_linear_artifacts": True,
        "enable_local_deblend": False,
        "zero_point": None,
    }

    assert gui._analysis_matches_detector_parameters(analysis, detector_kwargs)

    for key, changed_value in (
        ("threshold_sigma", 5.0),
        ("min_distance", 5),
        ("max_sources", 1000),
        ("psf_fwhm", 3.0),
        ("min_flux_snr", 7.0),
        ("proposal_mode", "gaussian"),
        ("reject_linear_artifacts", False),
        ("enable_local_deblend", True),
    ):
        changed = dict(detector_kwargs)
        changed[key] = changed_value
        assert not gui._analysis_matches_detector_parameters(analysis, changed), key

    incomplete = SimpleNamespace(detection=SimpleNamespace(parameters={"threshold_sigma": 4.0}))
    assert not gui._analysis_matches_detector_parameters(incomplete, detector_kwargs)


def test_result_event_decoder_accepts_legacy_and_generation_aware_events() -> None:
    assert gui.StarfieldApp._decode_result_event(("analysis", 3, "payload")) == (
        "analysis",
        3,
        None,
        "payload",
    )
    assert gui.StarfieldApp._decode_result_event(("analysis", 3, 8, "payload")) == (
        "analysis",
        3,
        8,
        "payload",
    )


def test_stale_generation_aware_auto_result_never_calls_render_callback() -> None:
    class _Widget:
        def __init__(self) -> None:
            self.config_calls: list[dict[str, object]] = []

        def winfo_exists(self) -> bool:
            return True

        def config(self, **kwargs: object) -> None:
            self.config_calls.append(kwargs)

    window = _Widget()
    button = _Widget()
    status = _Widget()
    rendered: list[object] = []
    payload = (
        window,
        button,
        status,
        Path("frame.fits"),
        SimpleNamespace(),
        Path("catalog.csv"),
        SimpleNamespace(),
        lambda *_args: rendered.append(True),
    )
    app = object.__new__(gui.StarfieldApp)
    app.result_queue = queue.Queue()
    app.frame_token = 11
    app.cache_generation = 4
    app.after = lambda *_args: None
    app.result_queue.put(("catalog-auto", 11, 3, payload))

    app._poll_result()

    assert rendered == []
    assert status.config_calls[-1]["text"] == "任务已失效 · 当前帧或缓存已变化 · 请重新运行"
    assert button.config_calls[-1]["state"] == "normal"


def test_calibration_render_path_repaints_match_tree_with_verified_wcs() -> None:
    source = Path(gui.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    show_catalog = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_show_catalog_match"
    )
    render_calibration = next(
        node
        for node in ast.walk(show_catalog)
        if isinstance(node, ast.FunctionDef) and node.name == "render_calibration"
    )
    text = ast.get_source_segment(source, render_calibration) or ""

    assert "refined_analysis" in text
    assert "verified_wcs=calibration" in text
    assert "current_catalog_path" in text
    assert "render(" in text


def test_auto_photometry_wires_detector_fingerprint_and_cache_generation() -> None:
    source = Path(gui.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    show_catalog = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_show_catalog_match"
    )
    run_auto = next(
        node
        for node in ast.walk(show_catalog)
        if isinstance(node, ast.FunctionDef) and node.name == "run_auto_photometry"
    )
    text = ast.get_source_segment(source, run_auto) or ""

    assert "_analysis_matches_detector_parameters" in text
    assert "not self.manual_tuning_dirty" in text
    assert "cache_generation = self.cache_generation" in text
    assert "_queue_task_result" in text
    assert "if catalog_text and audit_path.is_file():" in text
    assert "endpoint=endpoint_value" in text


def test_active_gui_exposes_explicit_photometry_and_public_service_selection() -> None:
    source = Path(gui.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    methods = {
        node.name: ast.get_source_segment(source, node) or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name in {"_build_controls_starfield", "_show_catalog_match"}
    }
    assert '"星等标定 · Gaia", self._show_catalog_match' in methods["_build_controls_starfield"]
    assert "单张分析得到仪器星等" in methods["_build_controls_starfield"]
    assert "AIP_GAIA_TAP_SYNC_URL" in methods["_show_catalog_match"]
    assert "DEFAULT_GAIA_TAP_SYNC_URL" in methods["_show_catalog_match"]
