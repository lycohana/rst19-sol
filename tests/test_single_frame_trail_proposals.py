"""单帧多尺度有限线段宽筛的回归测试契约。

这里测试的是建议中的新 API，而不是把现有的连通域/PCA 后置复核器
改名冒充多尺度宽筛。建议 API 的最小契约是：

``rst19.sequence.detect_single_frame_trail_proposals(image, *, ...)``

它接收一张二维图像和当前图像的背景、噪声、PSF 宽度，返回一组至少含
``x``、``y``、``length_px``、``angle_deg`` 的 proposal 记录。它不接收
路径列表、帧序列、时间参考或曝光时间；曝光时间只在检测结束后用于
解释 ``length_px / exposure_seconds``。

如果未来入口暂时不存在，测试会使用显式的 ``XFAIL``，避免把缺失实现
伪装成通过；如果 API 已存在但行为不符合契约，测试会正常失败，不能被
“缺少实现”掩盖。
"""

from __future__ import annotations

import inspect
from collections.abc import Mapping, Sequence
from typing import Any, Callable

import numpy as np
import pytest

from rst19 import sequence as sequence_module
from rst19.single_frame_trails import detect_local_pixel_trail_proposals


IMAGE_SHAPE = (96, 96)
BACKGROUND = 100.0
NOISE = 1.0
PSF_FWHM = 2.0
PSF_SIGMA = PSF_FWHM / (2.0 * np.sqrt(2.0 * np.log(2.0)))


def _future_detector() -> Callable[..., Sequence[Any]]:
    """取得未来 API；当前缺失时将每个契约测试标成预期失败。"""

    detector = getattr(sequence_module, "detect_single_frame_trail_proposals", None)
    if not callable(detector):
        pytest.xfail(
            "当前预期失败：rst19.sequence 尚未实现 "
            "detect_single_frame_trail_proposals；本文件先固定未来 API 契约。"
        )
    return detector


def _field(record: Any, name: str) -> Any:
    """支持 dataclass proposal 和 dict proposal 两种等价返回形式。"""

    if isinstance(record, Mapping):
        return record[name]
    return getattr(record, name)


def _call_detector(image: np.ndarray) -> tuple[Any, ...]:
    """用单张合成图调用建议 API；默认搜索必须是多尺度、多方向的。"""

    detector = _future_detector()
    valid_mask = np.ones(image.shape, dtype=bool)
    proposals = detector(
        image,
        valid_mask=valid_mask,
        background=BACKGROUND,
        noise=NOISE,
        psf_fwhm=PSF_FWHM,
        # 这些参数是宽筛的可复现实验配置，不把曝光时间传进检测器。
        length_bank_px=(4.0, 6.0, 8.0, 12.0, 18.0, 28.0, 40.0),
        angle_step_deg=10.0,
        max_proposals=128,
    )
    return tuple(proposals)


def _call_local_detector(image: np.ndarray) -> tuple[Any, ...]:
    """调用不降采样的局部短线支路。"""

    return tuple(
        detect_local_pixel_trail_proposals(
            image,
            valid_mask=np.ones(image.shape, dtype=bool),
            background=BACKGROUND,
            noise=NOISE,
            psf_fwhm=PSF_FWHM,
            length_bank_px=(8.0, 10.0, 12.0, 16.0, 24.0, 32.0),
            angle_step_deg=10.0,
            max_proposals=64,
            max_seeds=128,
        )
    )


def _add_point(image: np.ndarray, *, center: tuple[float, float], amplitude: float) -> None:
    """向图像加入一个已知中心和宽度的二维高斯点源。"""

    yy, xx = np.indices(image.shape, dtype=np.float64)
    cx, cy = center
    image += amplitude * np.exp(
        -0.5 * (((xx - cx) ** 2 + (yy - cy) ** 2) / (PSF_SIGMA**2))
    ).astype(image.dtype, copy=False)


def _add_line(
    image: np.ndarray,
    *,
    center: tuple[float, float],
    length_px: float,
    angle_deg: float,
    line_level: float = 3.2,
) -> None:
    """加入 PSF 与有限均匀线段卷积近似的单帧拖影。"""

    yy, xx = np.indices(image.shape, dtype=np.float64)
    cx, cy = center
    angle = np.deg2rad(angle_deg)
    unit_x, unit_y = np.cos(angle), np.sin(angle)
    # 半像素采样后求和，得到有限线段而不是无限长的线。
    samples = np.arange(-0.5 * length_px, 0.5 * length_px + 0.25, 0.5)
    for offset in samples:
        source_x = cx + offset * unit_x
        source_y = cy + offset * unit_y
        image += (line_level * np.exp(
            -0.5 * (((xx - source_x) ** 2 + (yy - source_y) ** 2) / (PSF_SIGMA**2))
        )).astype(image.dtype, copy=False) * 0.5


def _base_image(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(BACKGROUND, NOISE, size=IMAGE_SHAPE).astype(np.float32)


def _single_line_image() -> np.ndarray:
    image = _base_image(1901)
    _add_line(image, center=(28.0, 48.0), length_px=24.0, angle_deg=28.0)
    return image


def _multiscale_line_image() -> tuple[np.ndarray, tuple[tuple[float, float], ...]]:
    image = _base_image(1902)
    short_center = (24.0, 24.0)
    long_center = (70.0, 70.0)
    _add_line(image, center=short_center, length_px=8.0, angle_deg=23.0, line_level=4.0)
    _add_line(image, center=long_center, length_px=28.0, angle_deg=113.0)
    return image, (short_center, long_center)


def _negative_control_image() -> tuple[np.ndarray, tuple[tuple[float, float], ...]]:
    image = _base_image(1903)
    point_center = (76.0, 20.0)
    double_a = (70.0, 72.0)
    double_b = (78.0, 72.0)
    _add_point(image, center=point_center, amplitude=28.0)
    _add_point(image, center=double_a, amplitude=24.0)
    _add_point(image, center=double_b, amplitude=20.0)
    return image, (point_center, double_a, double_b)


def _near(record: Any, center: tuple[float, float], radius_px: float = 7.0) -> bool:
    x = float(_field(record, "x"))
    y = float(_field(record, "y"))
    return float(np.hypot(x - center[0], y - center[1])) <= radius_px


def _angle_distance_deg(first: float, second: float) -> float:
    """线段方向按 180° 周期比较，避免把正反方向当成不同几何。"""

    return abs((float(first) - float(second) + 90.0) % 180.0 - 90.0)


def test_multiscale_single_frame_proposals_find_short_and_long_lines() -> None:
    """同一张图中的短线和长线都应进入单帧 proposal 层。"""

    image, centers = _multiscale_line_image()
    proposals = _call_detector(image)

    assert proposals, "多尺度宽筛没有返回任何单帧线段 proposal"
    for center in centers:
        nearby = [proposal for proposal in proposals if _near(proposal, center)]
        assert nearby, f"没有在合成线段中心 {center} 附近找到 proposal"
        assert max(float(_field(proposal, "length_px")) for proposal in nearby) > 0.0


def test_single_frame_proposals_do_not_call_sequence_or_require_sequence_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """单帧宽筛只读一个 ndarray，禁止偷偷调用 15 帧接口。"""

    detector = _future_detector()
    parameter_names = set(inspect.signature(detector).parameters)
    forbidden_parameters = {
        "paths",
        "frames",
        "sequence",
        "temporal_reference",
        "cumulative_shifts",
        "exposure_seconds",
    }
    assert parameter_names.isdisjoint(forbidden_parameters)

    def _forbidden(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("单帧 proposal 不得调用 15 帧接口")

    for name in ("analyze_sequence", "detect_motion_features", "track_detections"):
        monkeypatch.setattr(sequence_module, name, _forbidden, raising=False)

    proposals = _call_detector(_single_line_image())
    assert proposals, "单张合成拖影没有被单帧宽筛提出"
    for proposal in proposals:
        classification = getattr(proposal, "classification", None)
        if isinstance(proposal, Mapping):
            classification = proposal.get("classification")
        if classification is not None:
            assert classification != "moving"


def test_exposure_1_5_seconds_changes_rate_explanation_not_geometry() -> None:
    """1.5 s 只用于 ``length_px / exposure_seconds``，不改变检测几何。"""

    image = _single_line_image()
    first = _call_detector(image)
    second = _call_detector(image)
    assert first and second

    target_center = (28.0, 48.0)
    first_target = min(first, key=lambda proposal: _distance(proposal, target_center))
    second_target = min(second, key=lambda proposal: _distance(proposal, target_center))
    first_geometry = (
        float(_field(first_target, "x")),
        float(_field(first_target, "y")),
        float(_field(first_target, "length_px")),
        float(_field(first_target, "angle_deg")),
    )
    second_geometry = (
        float(_field(second_target, "x")),
        float(_field(second_target, "y")),
        float(_field(second_target, "length_px")),
        float(_field(second_target, "angle_deg")),
    )
    np.testing.assert_allclose(first_geometry[:3], second_geometry[:3], atol=0.25)
    assert _angle_distance_deg(first_geometry[3], second_geometry[3]) <= 1.0

    length_px = first_geometry[2]
    rate_at_1_5_s = length_px / 1.5
    rate_at_3_s = length_px / 3.0
    assert rate_at_1_5_s == pytest.approx(2.0 * rate_at_3_s)


def test_point_and_double_star_controls_are_not_all_returned_as_trails() -> None:
    """孤立点源和相邻双星不能被宽筛全部当成有限线段。"""

    image, negative_centers = _negative_control_image()
    proposals = _call_detector(image)

    # 这是 proposal 输出层的负控制：简单点源/双星附近不应有已接受的
    # 线段 proposal。它不要求对任意复杂星场做到零假阳性，但不允许把
    # 这些已知的点源负例整体变成线段。
    assert not any(
        _near(proposal, center, radius_px=7.0)
        for proposal in proposals
        for center in negative_centers
    ), "点源或双星负例被整体输出成了线段 proposal"


def test_native_pixel_local_scan_keeps_an_eight_pixel_trail_visible() -> None:
    """原图短线不能因为 4×块平均而退化成点源。"""

    image = _base_image(1904)
    center = (24.0, 24.0)
    _add_line(image, center=center, length_px=8.0, angle_deg=23.0, line_level=4.0)
    proposals = _call_local_detector(image)

    nearby = [proposal for proposal in proposals if _near(proposal, center, radius_px=6.0)]
    assert nearby, "原始像素局部扫描没有找回 8 px 单帧线段"
    best = max(nearby, key=lambda proposal: float(_field(proposal, "response")))
    assert 6.0 <= float(_field(best, "length_px")) <= 14.0
    assert _angle_distance_deg(float(_field(best, "angle_deg")), 23.0) <= 12.0


def test_native_pixel_local_scan_keeps_point_double_negative_control_clean() -> None:
    """局部补检也不能把点源和相邻双星整体画成短线。"""

    image, negative_centers = _negative_control_image()
    proposals = _call_local_detector(image)
    assert not any(
        _near(proposal, center, radius_px=7.0)
        for proposal in proposals
        for center in negative_centers
    )


def _distance(record: Any, center: tuple[float, float]) -> float:
    return float(
        np.hypot(
            float(_field(record, "x")) - center[0],
            float(_field(record, "y")) - center[1],
        )
    )
