from __future__ import annotations

import ast
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from rst19 import gui


GUI_SOURCE = Path(gui.__file__).read_text(encoding="utf-8")
GUI_TREE = ast.parse(GUI_SOURCE)


def _method(name: str) -> ast.FunctionDef:
    app = next(
        node
        for node in GUI_TREE.body
        if isinstance(node, ast.ClassDef) and node.name == "StarfieldApp"
    )
    return next(
        node
        for node in app.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _call_names(method: ast.FunctionDef) -> set[str]:
    return {
        node.func.id
        for node in ast.walk(method)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }


def test_active_entry_is_python_tkinter_and_builds_the_starfield_layout() -> None:
    starfield = next(
        node
        for node in GUI_TREE.body
        if isinstance(node, ast.ClassDef) and node.name == "StarfieldApp"
    )
    assert any(
        isinstance(base, ast.Attribute)
        and isinstance(base.value, ast.Name)
        and base.value.id == "tk"
        and base.attr == "Tk"
        for base in starfield.bases
    )

    init = _method("__init__")
    init_calls = {
        node.func.attr
        for node in ast.walk(init)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "self"
    }
    assert "_build_layout_starfield" in init_calls
    assert "_load_frames" in init_calls

    main = next(
        node
        for node in GUI_TREE.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    main_calls = {
        node.func.attr
        for node in ast.walk(main)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "app"
    }
    assert "mainloop" in main_calls


def test_one_click_defaults_are_the_declared_research_defaults() -> None:
    assert gui.GUI_DEFAULT_THRESHOLD_SIGMA == 4.0
    assert gui.GUI_DEFAULT_MIN_DISTANCE == 4
    assert gui.GUI_DEFAULT_PSF_FWHM == 2.0
    assert gui.GUI_DEFAULT_MIN_FLUX_SNR == 5.0
    assert gui.GUI_DEFAULT_PROPOSAL_MODE == "hybrid"
    assert gui.MAX_PREVIEW_ZOOM == 15.0
    assert gui.PREVIEW_MAX_SIDE >= 4096

    controls = _method("_build_controls_starfield")
    controls_text = ast.get_source_segment(GUI_SOURCE, controls) or ""
    assert "一键分析当前帧" in controls_text
    assert "GUI_DEFAULT_THRESHOLD_SIGMA" in controls_text
    assert "GUI_DEFAULT_MIN_FLUX_SNR" in controls_text
    assert 'value=GUI_DEFAULT_PROPOSAL_MODE' in controls_text


def test_single_one_click_does_not_implicitly_match_or_calibrate() -> None:
    run_analysis = _method("run_analysis")
    analyze_calls = [
        node
        for node in ast.walk(run_analysis)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "analyze_frame"
    ]
    assert len(analyze_calls) == 1
    keyword_names = {
        keyword.arg
        for keyword in analyze_calls[0].keywords
        if keyword.arg is not None
    }
    assert {"catalog", "wcs", "fit_photometry"}.isdisjoint(keyword_names)
    assert "_set_task_mode(\"single\")" in (ast.get_source_segment(GUI_SOURCE, run_analysis) or "")


def test_zoom_is_runtime_clamped_to_fifteen_times() -> None:
    class _Canvas:
        def winfo_width(self) -> int:
            return 1000

        def winfo_height(self) -> int:
            return 700

    class _Status:
        def set(self, _value: str) -> None:
            pass

    app = object.__new__(gui.StarfieldApp)
    app.preview = Image.new("RGB", (4096, 4096))
    app.preview_zoom = 14.9
    app.preview_pan_x = 0.0
    app.preview_pan_y = 0.0
    app.canvas = _Canvas()
    app.status_var = _Status()
    app._view_transform = lambda: (1.0, 0.0, 0.0, 1.0)  # type: ignore[method-assign]
    app._draw_preview = lambda: None  # type: ignore[method-assign]

    gui.StarfieldApp._zoom_at(app, SimpleNamespace(x=500, y=350), 100.0)

    assert app.preview_zoom == gui.MAX_PREVIEW_ZOOM == 15.0


def test_viewer_has_hover_pan_zoom_and_explicit_overlay_layers() -> None:
    content = _method("_build_content_starfield")
    content_text = ast.get_source_segment(GUI_SOURCE, content) or ""
    assert 'self.overlay_mode_var = tk.StringVar(value="quality")' in content_text
    assert 'self.motion_overlay_var = tk.BooleanVar(value=True)' in content_text
    for mode in ("quality", "stable", "stack-faint", "candidates", "motion", "catalog"):
        assert f'("{mode}"' in content_text
    for binding in ("<MouseWheel>", "<ButtonPress-1>", "<B1-Motion>", "<Motion>"):
        assert binding in content_text

    draw_preview = _method("_draw_preview")
    draw_text = ast.get_source_segment(GUI_SOURCE, draw_preview) or ""
    assert "quality_sources" in draw_text
    assert "moving_points_for_frame" in draw_text
    assert "_draw_motion_trajectory_overlay" in draw_text


def test_single_and_sequence_paths_publish_visible_progress() -> None:
    single_text = ast.get_source_segment(GUI_SOURCE, _method("run_analysis")) or ""
    sequence_text = ast.get_source_segment(GUI_SOURCE, _method("run_sequence_analysis")) or ""
    progress_text = ast.get_source_segment(GUI_SOURCE, _method("_poll_result")) or ""
    render_text = ast.get_source_segment(GUI_SOURCE, _method("_render_running_progress")) or ""

    assert '"analysis-progress"' in single_text
    assert '"analysis-progress"' in sequence_text
    assert "detail_progress" in sequence_text
    assert 'kind == "analysis-progress"' in progress_text
    assert 'self._render_running_progress' in progress_text
    assert 'self.active_job_kind == "single"' in render_text
    assert 'self.active_job_kind == "sequence"' in render_text


def test_frame_switch_invalidates_old_preview_and_analysis_events() -> None:
    select_text = ast.get_source_segment(GUI_SOURCE, _method("_select_frame")) or ""
    poll_text = ast.get_source_segment(GUI_SOURCE, _method("_poll_result")) or ""
    assert "self.frame_token += 1" in select_text
    assert "self.analysis = None" in select_text
    assert "self.preview = None" in select_text
    assert "token = self.frame_token" in select_text
    assert "if token != self.frame_token" in poll_text
    assert 'kind == "preview"' in poll_text
    assert 'kind == "analysis"' in poll_text


def test_missing_wcs_or_provenance_never_becomes_a_standard_magnitude() -> None:
    valid = gui._gui_format_calibrated_magnitude(
        13.4,
        status="VALID",
        system="Gaia Vega",
        band="G",
        wcs_available=True,
    )
    assert "13.400" in valid and "m_G" in valid

    assert "不可用" in gui._gui_format_calibrated_magnitude(
        13.4,
        status="VALID",
        system="Gaia Vega",
        band="G",
        wcs_available=False,
    )
    assert "不可用" in gui._gui_format_calibrated_magnitude(
        13.4,
        status="VALID",
        system=None,
        band="",
        wcs_available=True,
    )

    no_extinction = gui._gui_format_absolute_magnitude(
        SimpleNamespace(status="VALID_NO_EXTINCTION", value=4.2, error=0.1),
        system="Gaia Vega",
        band="G",
        wcs_available=True,
    )
    assert "4.2" not in no_extinction
    assert "未提供消光修正" in no_extinction

    strict = gui._gui_format_absolute_magnitude(
        SimpleNamespace(status="VALID", value=4.2, error=0.1),
        system="Gaia Vega",
        band="G",
        wcs_available=True,
    )
    assert strict == "M_G = 4.200 ± 0.100"


@pytest.mark.skipif(
    os.name != "nt" and not os.environ.get("DISPLAY"),
    reason="Tk window smoke test requires a display",
)
def test_tkinter_entry_smoke_loads_the_fifteen_frame_dataset() -> None:
    import tkinter as tk

    root = tk.Tk()
    root.withdraw()
    try:
        app = gui.StarfieldApp(gui.DEFAULT_DATA_DIR)
        app.withdraw()
        app.update_idletasks()
        assert len(app.frames) == 15
        assert app.task_mode == "single"
        assert app.overlay_mode_var.get() == "quality"
        assert app.motion_overlay_var.get() is True
        assert app.mosaic_button.winfo_exists()
    finally:
        try:
            app.destroy()  # type: ignore[union-attr]
        except (NameError, tk.TclError):
            pass
        root.destroy()
