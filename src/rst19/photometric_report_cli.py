"""Audit a saved rst19 photometry JSON payload without opening the GUI."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .photometric_report import build_photometric_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="审计 rst19 星等 JSON：区分仪器、相对、表观和绝对星等证据"
    )
    parser.add_argument("input", type=Path, help="rst19 JSON 结果或包含 source_rows/frame_rows 的 JSON")
    parser.add_argument("--out", type=Path, help="可选审计报告 JSON；不提供时输出到 stdout")
    parser.add_argument(
        "--require-wcs",
        action="store_true",
        help="把 WCS 作为强制几何条件；没有 WCS 的源降级为 GEOMETRY_UNAVAILABLE",
    )
    parser.add_argument(
        "--no-require-wcs",
        action="store_true",
        help="不把 WCS 作为绝对星等审计的强制条件（仅适合已有可靠像素到天空映射的输入）",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="发现数值与状态矛盾时返回失败，而不是只在报告中记录 FALSE_VALID",
    )
    return parser


def _photometry_input(payload: object) -> tuple[object, object | None, bool]:
    """Extract photometry rows without expanding a large raw detection table.

    A normal ``rst19`` result also contains ``detection.sources``.  Those are
    detection records, not photometry rows, and can contain tens of thousands
    of entries.  The report is intentionally scoped to ``source_photometry``;
    if that section is absent, return an empty one-frame report instead of
    recursively walking the raw detector payload.  The automatic Gaia
    workflow wraps the same ``FrameAnalysis`` payload under ``analysis``;
    unwrap that documented envelope so its compact matched-source rows are
    audited instead of treating the wrapper itself as one malformed row.
    """

    if not isinstance(payload, dict):
        return payload, None, False

    # ``AutoPhotometricResult.as_dict()`` stores the actual frame result under
    # ``analysis`` and keeps the verified affine WCS at the outer level.  Do
    # not use the outer status as a calibration row: the source-level status,
    # nested calibration and WCS are the evidence the report gate needs.
    analysis_payload = payload
    is_auto_result = isinstance(payload.get("analysis"), dict) and "detection" in payload["analysis"]
    if is_auto_result:
        analysis_payload = payload["analysis"]

    if "detection" not in analysis_payload:
        return payload, None, False

    source_rows = analysis_payload.get("source_photometry")
    frame: dict[str, object] = {
        "frame_id": str(
            analysis_payload.get(
                "path",
                payload.get("frame_path", "frame"),
            )
        ),
        "sources": source_rows if isinstance(source_rows, list) else [],
    }
    if is_auto_result:
        # ``affine_wcs`` is present only when the geometric refinement was
        # produced.  A reference WCS alone is merely a prior and must not
        # satisfy ``--require-wcs``.
        if payload.get("affine_wcs") is not None:
            frame["wcs"] = payload["affine_wcs"]
    elif "wcs" in payload:
        frame["wcs"] = payload["wcs"]
    return [frame], analysis_payload.get("photometric_calibration"), not isinstance(source_rows, list) or not source_rows


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.require_wcs and args.no_require_wcs:
        print("photometric-report: --require-wcs and --no-require-wcs are mutually exclusive", file=sys.stderr)
        return 2
    try:
        payload = json.loads(args.input.read_text(encoding="utf-8"))
        require_wcs: bool | None = None
        if args.require_wcs:
            require_wcs = True
        elif args.no_require_wcs:
            require_wcs = False
        report_input, calibration, raw_only = _photometry_input(payload)
        report = build_photometric_report(
            report_input,
            calibration=calibration,
            require_wcs=require_wcs,
            strict=args.strict,
        )
        if raw_only:
            print(
                "photometric-report: input has no source_photometry rows; "
                "raw detection.sources were intentionally not treated as calibrated photometry",
                file=sys.stderr,
            )
        rendered = report.to_json() + "\n"
        if args.out is not None:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(rendered, encoding="utf-8")
        else:
            print(rendered, end="")
        return 0
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        print(f"photometric-report: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
