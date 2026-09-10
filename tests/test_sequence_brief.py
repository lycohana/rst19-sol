from types import SimpleNamespace

import pytest

from rst19.sequence import FrameSequenceSummary, MotionFeaturePoint, MotionFeatureTrack, SequenceResult
from rst19.sequence_brief import build_sequence_brief


def _result() -> SequenceResult:
    frames = tuple(
        FrameSequenceSummary(
            index,
            f"frame-{index}.fits",
            100,
            80,
            70,
            timestamp=f"2026-03-30T16:32:{5 + index * 2:02d}.000",
            exposure_ms=1500.0,
            width_px=4096,
            height_px=4096,
        )
        for index in range(3)
    )
    moving_points = tuple(
        MotionFeaturePoint(
            index,
            100.0 + index * 10.0,
            200.0 - index * 4.0,
            100.0 + index * 10.0,
            200.0 - index * 4.0,
            25.0,
            80,
            12.0,
            2.0,
            -21.8,
            (0, 0, 10, 10),
            False,
        )
        for index in range(3)
    )
    candidate = MotionFeatureTrack(
        8,
        "candidate",
        (moving_points[0],),
        0.0,
        0.0,
        None,
    )
    moving = MotionFeatureTrack(7, "moving", moving_points, 21.5, 10.75, 0.2)
    return SequenceResult(
        frames,
        ((0.0, 0.0), (0.1, 0.2), (0.2, 0.3)),
        (),
        4.0,
        2,
        2.0,
        0.75,
        motion_features=(candidate, moving),
    )


def test_sequence_brief_explains_motion_and_scientific_boundaries() -> None:
    brief = build_sequence_brief(_result())
    text = brief.text()

    assert "形成 1 条连续运动候选" in text
    assert "F01" in text and "F03" in text
    assert "图像平面速度" in text
    assert "不等于已经完成星表身份" in text
    assert "未生成" in text
    assert brief.targets[0].classification == "moving"


def test_sequence_brief_adds_mosaic_interpretation_without_calling_it_new_stars() -> None:
    mosaic = SimpleNamespace(
        output_shape=(4098, 4098),
        covered_pixel_count=16_777_216,
        overlap_pixel_count=16_777_000,
    )
    brief = build_sequence_brief(_result(), mosaic)

    assert "4098×4098" in brief.mosaic_note
    assert "不会自动增加恒星数" in brief.mosaic_note
    assert brief.as_dict()["evidence_refs"]


def test_sequence_brief_has_no_false_motion_claim_when_empty() -> None:
    result = _result()
    result = SequenceResult(
        result.frames,
        result.cumulative_shifts,
        result.tracks,
        result.link_radius_px,
        result.min_presence,
        result.motion_min_displacement_px,
        result.max_motion_fit_rms_px,
    )
    brief = build_sequence_brief(result)

    assert "没有形成运动候选" in brief.headline
    assert "不等于物理上绝对没有目标" in brief.text()
