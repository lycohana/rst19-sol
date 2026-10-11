"""独立的单帧有限曝光线段宽筛原型。

这个模块只处理调用者传入的当前一张图像。它没有 FITS、序列、时间轴或
跨帧状态，也不导入 ``rst19.sequence``。它的任务是产生一个高召回的线段
proposal 集合，供后续的点源/双星/椭圆像/宇宙线模型做细筛；proposal 本身
不能被解释成已经确认的运动目标。

算法是一个面向大图的匹配滤波近似：

1. 用 ``(image - background) / noise`` 得到白化残差，并用给定的 PSF FWHM
   做高斯平滑；
2. 对若干方向旋转当前图像，使该方向的线段变成水平线；
3. 在旋转图上用一维均值滤波近似 ``PSF 与有限均匀线段的卷积模板``，对多
   个长度尺度取局部峰值；
4. 对粗筛峰值在原图坐标中重新采样，做小范围角度/长度复核；
5. 按中心距离、无向角度和长度比例做贪心非极大值抑制，去掉同一线段在
   相邻模板上的重复 proposal。

返回的 ``response`` 是基于白化噪声的近似匹配响应（近似 z 分数），不是
经过整幅图多重搜索假警率校准后的概率。``length`` 和 ``angle`` 是图像
平面量；角度采用列为 x、行为 y 的坐标约定，范围为 ``[0, 180)``，因此
单帧线段不区分相反的运动方向。``width`` 是传入的 ``psf_fwhm`` 先验。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
import math
from typing import Any

import numpy as np
from scipy import ndimage
from scipy.special import erf


__all__ = [
    "TrailProposal",
    "detect_local_pixel_trail_proposals",
    "detect_single_frame_trail_proposals",
]


# 这些是原型的固定搜索策略，不依赖任何序列信息。
_ANGLE_STEP_DEG = 10.0
_COARSE_RESPONSE_MIN = 3.5
_FINAL_RESPONSE_MIN = 5.0
_MIN_LINE_COVERAGE = 0.90
_MAX_PEAKS_PER_TEMPLATE = 24
_MAX_REFINEMENT_CANDIDATES = 2048
_MAX_OUTPUT_PROPOSALS = 256
_PEAK_SEPARATION_FACTOR = 1.5
_ANGLE_DEDUP_TOLERANCE_DEG = 12.0
_MIN_LENGTH_RATIO_FOR_DEDUP = 0.50
_MAX_LENGTH_RATIO_FOR_DEDUP = 2.00
# 小于约 3 个 PSF 的结构与单点/双点模型高度退化。宽筛可以扫描更短
# 的模板，但在 proposal 输出层先不把这种退化响应称为线段，避免孤立
# 星点在最短模板上制造大量“线”。
_MIN_ACCEPTED_LENGTH_IN_FWHM = 3.0
_GAUSSIAN_TRUNCATE = 4.0
# The large-image fallback used by the sequence module deliberately works on
# a reduced image.  That is useful for very long structures, but it destroys
# an 8--20 px trail before the local model gets to see it.  This second pass
# therefore stays at native pixel scale and only evaluates a bounded set of
# local seeds.  It is intentionally a single-frame API: no frame index,
# exposure time, registration, or temporal reference enters this search.
_LOCAL_SCAN_LENGTHS_PX = (8.0, 10.0, 12.0, 16.0, 20.0, 24.0, 32.0, 48.0, 64.0)
_LOCAL_SCAN_SEED_THRESHOLD = 2.25
_LOCAL_SCAN_RESPONSE_THRESHOLD = 4.0
_LOCAL_SCAN_MAX_SEEDS = 512
_LOCAL_SCAN_TOP_PER_LENGTH = 96
_LOCAL_SCAN_PEAK_SIZE = 5
_LOCAL_SCAN_CENTER_FRACTIONS = (-0.25, 0.0, 0.25)
_LOCAL_SCAN_TILE_SIZE_PX = 256
_LOCAL_SCAN_TILE_SEEDS = 2


@dataclass(frozen=True, slots=True)
class TrailProposal:
    """一个单帧线段宽筛 proposal。

    ``x``/``y`` 是线段中心的像素坐标；``length`` 和 ``width`` 的单位是
    像素；``angle`` 是无向角度（度）；``response`` 是近似白化匹配响应。
    该数据类只保存当前这一张图的结果，故可以直接作为后续去重或细筛的
    输入记录。
    """

    x: float
    y: float
    length: float
    width: float
    angle: float
    response: float

    @property
    def length_px(self) -> float:
        """契约层使用的长度字段别名。"""

        return self.length

    @property
    def width_px(self) -> float:
        """显式表示 ``width`` 的像素单位。"""

        return self.width

    @property
    def angle_deg(self) -> float:
        """契约层使用的角度字段别名。"""

        return self.angle

    def as_tuple(self) -> tuple[float, float, float, float, float, float]:
        """按 ``(x, y, length, width, angle, response)`` 返回轻量元组。"""

        return (self.x, self.y, self.length, self.width, self.angle, self.response)

    def as_dict(self) -> dict[str, float]:
        """返回适合 JSON/表格层使用的普通字典。"""

        return asdict(self)


def _as_float_image(value: Any, name: str) -> np.ndarray:
    """把图像类输入转换为二维 float64 数组，并拒绝错误维度。"""

    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} 必须是可以转换为数值数组的二维输入") from exc
    if array.ndim != 2:
        raise ValueError(f"{name} 必须是二维数组，实际维度为 {array.ndim}")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} 不能是空数组")
    return array


def _as_field(value: Any, name: str, shape: tuple[int, int]) -> np.ndarray:
    """接受与图像同形状的背景/噪声图，也接受一个标量。"""

    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} 必须是标量或与 image 同形状的数值数组") from exc
    if array.ndim == 0:
        return np.full(shape, float(array), dtype=np.float64)
    if array.ndim != 2 or array.shape != shape:
        raise ValueError(f"{name} 必须是标量或形状为 {shape} 的二维数组，实际形状为 {array.shape}")
    return array


def _as_valid_mask(value: Any, shape: tuple[int, int]) -> np.ndarray:
    """校验 valid mask 的维度，并转换为布尔数组。

    允许 bool、整数和 0/1 浮点 mask；拒绝字符串、复数及含有其他浮点值的
    输入，避免诸如 ``"false"`` 被意外转换成 True。
    """

    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise TypeError("valid_mask 必须是二维布尔/0-1 数组") from exc
    if raw.ndim != 2 or raw.shape != shape:
        raise ValueError(f"valid_mask 必须是形状为 {shape} 的二维数组，实际形状为 {raw.shape}")
    if raw.dtype.kind == "b":
        return raw.astype(bool, copy=False)
    if raw.dtype.kind in "iu":
        if not np.all((raw == 0) | (raw == 1)):
            raise ValueError("valid_mask 的整数值只能是 0 或 1")
        return raw.astype(bool, copy=False)
    if raw.dtype.kind == "f":
        if not np.all(np.isfinite(raw)) or not np.all((raw == 0.0) | (raw == 1.0)):
            raise ValueError("valid_mask 的浮点值只能是有限的 0 或 1")
        return raw.astype(bool, copy=False)
    raise TypeError("valid_mask 必须是布尔、整数或 0/1 浮点数组")


def _validate_inputs(
    image: Any,
    valid_mask: Any,
    background: Any,
    noise: Any,
    psf_fwhm: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
    """集中完成公共入口的输入边界检查。"""

    image_array = _as_float_image(image, "image")
    shape = image_array.shape
    if min(shape) < 8:
        raise ValueError(f"image 的最短边至少需要 8 像素，实际形状为 {shape}")

    mask = _as_valid_mask(valid_mask, shape)
    background_array = _as_field(background, "background", shape)
    noise_array = _as_field(noise, "noise", shape)

    try:
        fwhm = float(psf_fwhm)
    except (TypeError, ValueError) as exc:
        raise TypeError("psf_fwhm 必须是有限的正数") from exc
    if not math.isfinite(fwhm) or fwhm <= 0.0:
        raise ValueError("psf_fwhm 必须是有限的正数")
    if fwhm < 0.5:
        raise ValueError("psf_fwhm 不能小于 0.5 像素；否则当前离散 PSF 近似不适用")
    if fwhm > min(shape) / 2.0:
        raise ValueError("psf_fwhm 不能大于图像最短边的一半")

    finite = np.isfinite(image_array) & np.isfinite(background_array) & np.isfinite(noise_array)
    usable = mask & finite
    if not np.any(mask):
        raise ValueError("valid_mask 没有任何有效像素")
    if not np.all(finite[mask]):
        bad_count = int(np.count_nonzero(mask & ~finite))
        raise ValueError(f"valid_mask 标记的区域中有 {bad_count} 个非有限像素")
    if np.any(noise_array[usable] <= 0.0):
        raise ValueError("valid_mask 标记的区域内 noise 必须全部大于 0")
    if not np.any(usable):
        raise ValueError("没有同时满足 valid、有限且 noise>0 的像素")

    return image_array, mask, background_array, noise_array, fwhm, usable


def _length_bank(shape: tuple[int, int], fwhm: float) -> tuple[int, ...]:
    """建立覆盖短线到长线的几何长度库。

    对 FWHM 约 2 像素的开运一号图像，该库通常覆盖约 3 像素到 1024
    像素。长度上限受图像尺寸约束，避免把整幅图误当成一条线。
    """

    shortest = float(min(shape))
    min_length = max(2.0, 1.5 * fwhm)
    max_length = min(shortest - 1.0, max(1024.0, 256.0 * fwhm))
    if max_length < min_length:
        min_length = min(min_length, max_length)
    min_length = max(2.0, min_length)
    max_length = max(min_length, max_length)

    values = [min_length]
    while values[-1] < max_length:
        next_value = min(max_length, values[-1] * math.sqrt(2.0))
        if next_value <= values[-1]:
            break
        values.append(next_value)

    lengths: list[int] = []
    for value in values:
        integer_length = max(2, int(round(value)))
        if integer_length <= int(shortest - 1.0) and integer_length not in lengths:
            lengths.append(integer_length)
    if not lengths:
        raise ValueError("图像尺寸不足以建立有效的线段长度库")
    return tuple(lengths)


@lru_cache(maxsize=512)
def _line_noise_std(sigma: float, length: int) -> float:
    """估计高斯平滑后线段均值的噪声标准差。

    白化残差的单像素噪声近似为 1。高斯平滑让相邻采样相关，因此这里
    使用高斯自相关的解析近似，而不是错误地把所有线段像素当成独立样本。
    """

    radius = max(1, int(math.ceil(_GAUSSIAN_TRUNCATE * sigma)))
    axis = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-(axis[:, None] ** 2 + axis[None, :] ** 2) / (2.0 * sigma * sigma))
    kernel /= float(np.sum(kernel))
    gaussian_variance = float(np.sum(kernel * kernel))

    offsets = np.arange(length, dtype=np.float64)
    autocorrelation = np.exp(-(offsets * offsets) / (4.0 * sigma * sigma))
    weighted_sum = float(length * autocorrelation[0])
    if length > 1:
        weighted_sum += float(2.0 * np.sum((length - offsets[1:]) * autocorrelation[1:]))
    variance = gaussian_variance * weighted_sum / float(length * length)
    return math.sqrt(max(variance, np.finfo(np.float64).tiny))


def _rotated_to_original(
    x_rotated: np.ndarray,
    y_rotated: np.ndarray,
    angle_deg: float,
    center_x: float,
    center_y: float,
) -> tuple[np.ndarray, np.ndarray]:
    """把 scipy 正角度旋转后的坐标映回原图坐标。"""

    angle = math.radians(angle_deg)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    dx = x_rotated - center_x
    dy = y_rotated - center_y
    x_original = center_x + cosine * dx - sine * dy
    y_original = center_y + sine * dx + cosine * dy
    return x_original, y_original


def _top_local_peaks(
    response: np.ndarray,
    coverage: np.ndarray,
    threshold: float,
    max_peaks: int,
    peak_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """提取一个方向/长度响应图的少量局部峰值。"""

    local_maximum = ndimage.maximum_filter(
        response,
        size=peak_size,
        mode="constant",
        cval=-np.inf,
    )
    eligible = (
        np.isfinite(response)
        & (response >= threshold)
        & (response >= local_maximum)
        & (coverage >= _MIN_LINE_COVERAGE)
    )
    flat_indices = np.flatnonzero(eligible)
    if flat_indices.size == 0:
        empty = np.empty(0, dtype=np.float64)
        return empty, empty, empty

    flat_response = response.ravel()[flat_indices]
    if flat_indices.size > max_peaks:
        keep = np.argpartition(flat_response, -max_peaks)[-max_peaks:]
        flat_indices = flat_indices[keep]
        flat_response = response.ravel()[flat_indices]
    order = np.argsort(flat_response)[::-1]
    flat_indices = flat_indices[order]
    ys, xs = np.unravel_index(flat_indices, response.shape)
    return xs.astype(np.float64), ys.astype(np.float64), response[ys, xs].astype(np.float64)


def _sample_line_response(
    smoothed: np.ndarray,
    valid_float: np.ndarray,
    x: float,
    y: float,
    length: float,
    angle_deg: float,
    sigma: float,
) -> float:
    """在原图坐标对一条候选线做局部复核。

    这是旋转+滑动积分的局部连续化版本。它只读取传入的当前图像数组，
    不使用任何其他帧。``map_coordinates`` 的线性插值足以用于宽筛复核，
    细筛阶段仍应使用更完整的联合 PSF 拟合。
    """

    integer_length = max(2, int(round(length)))
    sample_count = integer_length
    offsets = np.linspace(-length / 2.0, length / 2.0, sample_count, dtype=np.float64)
    angle = math.radians(angle_deg)
    x_coordinates = x + offsets * math.cos(angle)
    y_coordinates = y + offsets * math.sin(angle)
    coordinates = np.vstack((y_coordinates, x_coordinates))

    values = ndimage.map_coordinates(
        smoothed,
        coordinates,
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )
    support = ndimage.map_coordinates(
        valid_float,
        coordinates,
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )
    support = np.clip(support, 0.0, 1.0)
    coverage = float(np.mean(support))
    if coverage < _MIN_LINE_COVERAGE:
        return -np.inf

    weight_sum = float(np.sum(support))
    if weight_sum <= 0.0:
        return -np.inf
    mean_signal = float(np.sum(values * support) / weight_sum)
    noise_std = _line_noise_std(sigma, integer_length)
    # 缺失的少量像素会降低有效信息量，故用 sqrt(coverage) 做保守修正。
    return mean_signal * math.sqrt(coverage) / noise_std


def _fit_local_templates(
    data: np.ndarray,
    templates: tuple[np.ndarray, ...],
) -> float:
    """返回局部背景加非负模板拟合的残差平方和。

    这是宽筛的轻量负例检查，不是最终的精测器。枚举最多两个模板的
    active set，避免把一个负振幅的“第二颗星”拿来人为改善线段模型。
    """

    values = np.asarray(data, dtype=np.float64).reshape(-1)
    arrays = tuple(np.asarray(template, dtype=np.float64).reshape(-1) for template in templates)
    if not arrays or any(template.size != values.size for template in arrays):
        raise ValueError("local template sizes must match")
    best = math.inf
    for active_mask in range(1 << len(arrays)):
        active = tuple(template for index, template in enumerate(arrays) if active_mask & (1 << index))
        design = np.column_stack((np.ones(values.size, dtype=np.float64), *active))
        try:
            coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)
        except np.linalg.LinAlgError:
            continue
        if active and np.any(coefficients[1:] < -1e-9):
            continue
        coefficients[1:] = np.maximum(coefficients[1:], 0.0)
        residual = values - design @ coefficients
        best = min(best, float(np.dot(residual, residual)))
    return best


def _positive_matched_score(data: np.ndarray, template: np.ndarray) -> float:
    """计算去除常数背景后的非负单模板匹配分数。"""

    values = np.asarray(data, dtype=np.float64).reshape(-1)
    basis = np.asarray(template, dtype=np.float64).reshape(-1)
    if values.size != basis.size or values.size == 0:
        return 0.0
    values = values - float(np.mean(values))
    basis = basis - float(np.mean(basis))
    norm = float(np.linalg.norm(basis))
    if norm <= 0.0 or not math.isfinite(norm):
        return 0.0
    correlation = float(np.dot(values, basis) / norm)
    return max(0.0, correlation)


def _positive_two_template_score(
    data: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
) -> float:
    """计算两个非负模板的联合匹配分数。"""

    values = np.asarray(data, dtype=np.float64).reshape(-1)
    first_basis = np.asarray(first, dtype=np.float64).reshape(-1)
    second_basis = np.asarray(second, dtype=np.float64).reshape(-1)
    if values.size == 0 or first_basis.size != values.size or second_basis.size != values.size:
        return 0.0
    values = values - float(np.mean(values))
    design = np.column_stack((first_basis - float(np.mean(first_basis)), second_basis - float(np.mean(second_basis))))
    gram = design.T @ design
    rhs = design.T @ values
    try:
        coefficients = np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError:
        coefficients = np.zeros(2, dtype=np.float64)
    if np.any(coefficients < 0.0):
        single_scores = (
            _positive_matched_score(values, first_basis),
            _positive_matched_score(values, second_basis),
        )
        return max(single_scores)
    improvement = float(np.dot(coefficients, rhs))
    return math.sqrt(max(0.0, improvement)) if math.isfinite(improvement) else 0.0


def _line_template_on_patch(
    local_x: np.ndarray,
    local_y: np.ndarray,
    *,
    center_x: float,
    center_y: float,
    length: float,
    angle_deg: float,
    sigma: float,
) -> np.ndarray:
    """生成局部 PSF 展宽有限线段模板，峰值归一化。"""

    angle = math.radians(float(angle_deg))
    cosine = math.cos(angle)
    sine = math.sin(angle)
    along = (local_x - center_x) * cosine + (local_y - center_y) * sine
    across = -(local_x - center_x) * sine + (local_y - center_y) * cosine
    half_length = 0.5 * max(0.5, float(length))
    edge_scale = math.sqrt(2.0) * max(float(sigma), 0.45)
    finite_segment = 0.5 * (
        erf((along + half_length) / edge_scale)
        - erf((along - half_length) / edge_scale)
    )
    template = finite_segment * np.exp(-0.5 * (across / max(float(sigma), 0.45)) ** 2)
    peak = float(np.max(template))
    return template / peak if peak > 0.0 and math.isfinite(peak) else template


def _passes_compact_source_control(
    whitened: np.ndarray,
    valid: np.ndarray,
    *,
    x: float,
    y: float,
    length: float,
    angle_deg: float,
    fwhm: float,
) -> bool:
    """排除局部点源/双星模型明显更合理的响应。

    线段匹配滤波的最短尺度必然会响应亮星，因此不能仅按响应阈值
    接受 proposal。这里在同一个当前帧 cutout 中比较有限线段、单点和
    双点三个非负模型；只有线段的残差在 BIC 风格的轻量惩罚后明显更小，
    才把该宽筛峰保留。它不使用时间信息，也不试图证明目标跨帧运动。
    """

    sigma = max(float(fwhm) / 2.354820045, 0.5)
    half_length = 0.5 * max(float(length), 2.0)
    radius = max(7.0, half_length + 4.0 * sigma + 2.0)
    height, width = whitened.shape
    x0 = max(0, int(math.floor(x - radius)))
    x1 = min(width, int(math.ceil(x + radius + 1.0)))
    y0 = max(0, int(math.floor(y - radius)))
    y1 = min(height, int(math.ceil(y + radius + 1.0)))
    if x1 - x0 < 7 or y1 - y0 < 7:
        return False
    patch_valid = np.asarray(valid[y0:y1, x0:x1], dtype=bool)
    if int(np.count_nonzero(patch_valid)) < 25:
        return False
    yy, xx = np.mgrid[y0:y1, x0:x1]
    local_x = np.asarray(xx[patch_valid], dtype=np.float64)
    local_y = np.asarray(yy[patch_valid], dtype=np.float64)
    local_data = np.asarray(whitened[y0:y1, x0:x1], dtype=np.float64)[patch_valid]
    line = _line_template_on_patch(
        local_x,
        local_y,
        center_x=x,
        center_y=y,
        length=length,
        angle_deg=angle_deg,
        sigma=sigma,
    )
    # 旋转插值会让粗筛峰的中心偏离真正点源约一个像素。若只在
    # proposal 中心放单点模板，偏移后的点源会被错误地解释成短线；
    # 用局部最亮像素补一个点源中心候选，仍然只读取当前 cutout。
    peak_index = int(np.argmax(local_data))
    compact_centers = ((float(x), float(y)), (float(local_x[peak_index]), float(local_y[peak_index])))
    point_templates = tuple(
        np.exp(
            -0.5
            * (((local_x - center_x) ** 2 + (local_y - center_y) ** 2) / (sigma * sigma))
        )
        for center_x, center_y in compact_centers
    )
    point_score = max(_positive_matched_score(local_data, template) for template in point_templates)
    line_score = _positive_matched_score(local_data, line)
    # 先做最便宜的单点负例门。真实点源在最短模板上的响应通常会很高，
    # 没必要为它再遍历双星方向库；这也是大图运行时间的主要控制点。
    if line_score <= point_score + 1.0:
        return False

    angle = math.radians(float(angle_deg))
    cosine = math.cos(angle)
    sine = math.sin(angle)
    # 双星方向不能被粗筛方向强制绑定。候选方向只用于优先级，负例
    # 控制额外检查一个小型无向方向库，避免横向双星被斜向线模板放过。
    double_score = 0.0
    for factor in (0.75, 1.0, 1.25):
        separation = max(1.0, float(length) * factor)
        half_separation = 0.5 * separation
        for test_angle_deg in np.arange(0.0, 180.0, 30.0):
            test_angle = math.radians(float(test_angle_deg))
            test_cosine = math.cos(test_angle)
            test_sine = math.sin(test_angle)
            templates = tuple(
                np.exp(
                    -0.5
                    * (
                        (local_x - (x + sign * half_separation * test_cosine)) ** 2
                        + (local_y - (y + sign * half_separation * test_sine)) ** 2
                    )
                    / (sigma * sigma)
                )
                for sign in (-1.0, 1.0)
            )
            double_score = max(double_score, _positive_two_template_score(local_data, templates[0], templates[1]))
    # 线模型和单点模型使用相同的一个振幅自由度；双点模型多一个
    # 振幅，给它一个轻微的复杂度补偿，避免把随机相关误当作线段胜出。
    double_score_adjusted = max(0.0, double_score - 0.5)
    return line_score > double_score_adjusted + 1.0


def detect_local_pixel_trail_proposals(
    image: Any,
    valid_mask: Any,
    background: Any,
    noise: Any,
    psf_fwhm: Any,
    *,
    length_bank_px: Any = None,
    angle_step_deg: Any = _ANGLE_STEP_DEG,
    max_proposals: Any = 128,
    max_seeds: Any = _LOCAL_SCAN_MAX_SEEDS,
    seed_threshold: Any = _LOCAL_SCAN_SEED_THRESHOLD,
    sample_step_px: Any = 1.0,
) -> tuple[TrailProposal, ...]:
    """在原始像素尺度寻找短小的单帧曝光线段。

    这是给大尺寸 FITS 使用的局部补检器。它先在当前图像的高斯白化残差
    上取有限数量的局部峰，再对每个峰的短长度、无向角度和三个轴向中心
    偏移做向量化线积分，最后只对少量最高响应候选运行点/双点/线段竞争。
    因而不会像整幅图逐角度旋转那样昂贵，也不会把 8--20 px 结构先压缩
    成 2--5 个降采样像素。

    这个入口只接收一张二维图像和它自己的背景、噪声、PSF 先验。它没有
    帧列表、曝光时间、配准位移或时间参考，返回值只是“当前曝光内的线形
    候选”，不能被解释为跨帧 ``moving``。

    ``response`` 仍是局部白化匹配响应，不是假警率校准后的概率。候选会
    经过单点/双点负例控制；这一步的目的是真实线段不要因单个像素不够高
    而漏掉，而不是把任意亮点都变成运动目标。

    ``sample_step_px`` 只控制粗扫阶段沿线的采样间距。序列宽筛可以用较
    粗的间距快速找候选，候选随后仍会回到原图做局部精查；单帧默认 1 px。
    """

    image_array, valid_mask_array, background_array, noise_array, fwhm, usable = _validate_inputs(
        image,
        valid_mask,
        background,
        noise,
        psf_fwhm,
    )

    try:
        angle_step = float(angle_step_deg)
    except (TypeError, ValueError) as exc:
        raise TypeError("angle_step_deg 必须是有限的正数") from exc
    if not math.isfinite(angle_step) or angle_step <= 0.0 or angle_step > 180.0:
        raise ValueError("angle_step_deg 必须满足 0 < angle_step_deg <= 180")

    if isinstance(max_proposals, (bool, np.bool_)):
        raise TypeError("max_proposals 必须是正整数")
    try:
        max_output = int(max_proposals)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError("max_proposals 必须是正整数") from exc
    if max_output <= 0 or float(max_proposals) != float(max_output):
        raise ValueError("max_proposals 必须是正整数")

    if isinstance(max_seeds, (bool, np.bool_)):
        raise TypeError("max_seeds 必须是正整数")
    try:
        seed_limit = int(max_seeds)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError("max_seeds 必须是正整数") from exc
    if seed_limit <= 0 or float(max_seeds) != float(seed_limit):
        raise ValueError("max_seeds 必须是正整数")
    seed_limit = min(seed_limit, _LOCAL_SCAN_MAX_SEEDS)

    try:
        seed_floor = float(seed_threshold)
    except (TypeError, ValueError) as exc:
        raise TypeError("seed_threshold 必须是有限的正数") from exc
    if not math.isfinite(seed_floor) or seed_floor <= 0.0:
        raise ValueError("seed_threshold 必须是有限的正数")

    try:
        sample_step = float(sample_step_px)
    except (TypeError, ValueError) as exc:
        raise TypeError("sample_step_px 必须是有限的正数") from exc
    if not math.isfinite(sample_step) or sample_step <= 0.0:
        raise ValueError("sample_step_px 必须是有限的正数")

    if length_bank_px is None:
        raw_lengths = np.asarray(_LOCAL_SCAN_LENGTHS_PX, dtype=np.float64)
    else:
        try:
            raw_lengths = np.asarray(length_bank_px, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise TypeError("length_bank_px 必须是一维正数序列") from exc
        if raw_lengths.ndim == 0:
            raw_lengths = raw_lengths.reshape(1)
    if raw_lengths.ndim != 1 or raw_lengths.size == 0:
        raise ValueError("length_bank_px 必须是非空的一维正数序列")
    if not np.all(np.isfinite(raw_lengths)) or np.any(raw_lengths <= 0.0):
        raise ValueError("length_bank_px 必须全部是有限的正数")
    max_allowed_length = float(min(image_array.shape) - 1)
    lengths: tuple[int, ...] = tuple(
        dict.fromkeys(
            max(2, int(round(float(value))))
            for value in raw_lengths
            if float(value) <= max_allowed_length
        )
    )
    if not lengths:
        raise ValueError(f"length_bank_px 中没有不超过图像尺寸的长度（上限 {max_allowed_length:g} px）")

    whitened = np.zeros_like(image_array, dtype=np.float64)
    whitened[usable] = (image_array[usable] - background_array[usable]) / noise_array[usable]
    valid_float = usable.astype(np.float64)
    sigma = max(fwhm / 2.354820045, 0.5)
    smoothed_signal = ndimage.gaussian_filter(
        whitened,
        sigma=sigma,
        mode="constant",
        cval=0.0,
        truncate=_GAUSSIAN_TRUNCATE,
    )
    smoothed_support = ndimage.gaussian_filter(
        valid_float,
        sigma=sigma,
        mode="constant",
        cval=0.0,
        truncate=_GAUSSIAN_TRUNCATE,
    )
    smoothed = np.zeros_like(smoothed_signal)
    np.divide(smoothed_signal, smoothed_support, out=smoothed, where=smoothed_support > 1e-6)

    local_maximum = ndimage.maximum_filter(
        smoothed,
        size=max(_LOCAL_SCAN_PEAK_SIZE, 2 * int(math.ceil(fwhm)) + 1),
        mode="constant",
        cval=-np.inf,
    )
    eligible = usable & np.isfinite(smoothed) & (smoothed >= seed_floor) & (smoothed >= local_maximum)
    seed_indices = np.flatnonzero(eligible)
    if seed_indices.size == 0:
        return ()
    if seed_indices.size > seed_limit:
        # 全局 top-k 会把种子集中在最亮恒星、热像素和最长亮线附近，
        # 使低峰值短拖影根本没有机会进入线段匹配。按 256 px 网格各取
        # 少量局部峰，再按分数截到总预算，给整幅图保留空间覆盖；这仍是
        # 当前帧内部的确定性宽筛，不引入任何序列/时间信息。
        height, width = image_array.shape
        tiled: list[int] = []
        tile_size = _LOCAL_SCAN_TILE_SIZE_PX
        per_tile = max(1, int(_LOCAL_SCAN_TILE_SEEDS))
        flat_scores = smoothed.ravel()
        for y0 in range(0, height, tile_size):
            y1 = min(height, y0 + tile_size)
            for x0 in range(0, width, tile_size):
                x1 = min(width, x0 + tile_size)
                tile_indices = np.flatnonzero(eligible[y0:y1, x0:x1])
                if tile_indices.size == 0:
                    continue
                tile_height = y1 - y0
                tile_width = x1 - x0
                local_y, local_x = np.unravel_index(
                    tile_indices,
                    (tile_height, tile_width),
                )
                tile_indices = (y0 + local_y) * width + (x0 + local_x)
                tile_scores = flat_scores[tile_indices]
                take = min(per_tile, tile_indices.size)
                if tile_indices.size > take:
                    keep = np.argpartition(tile_scores, -take)[-take:]
                    tile_indices = tile_indices[keep]
                tiled.extend(int(index) for index in tile_indices)
        seed_indices = np.asarray(tiled, dtype=np.int64)
        if seed_indices.size > seed_limit:
            seed_scores = flat_scores[seed_indices]
            keep = np.argpartition(seed_scores, -seed_limit)[-seed_limit:]
            seed_indices = seed_indices[keep]
    seed_scores = smoothed.ravel()[seed_indices]
    order = np.argsort(seed_scores)[::-1]
    seed_indices = seed_indices[order]
    seed_y, seed_x = np.unravel_index(seed_indices, image_array.shape)
    seed_x = seed_x.astype(np.float64, copy=False)
    seed_y = seed_y.astype(np.float64, copy=False)

    angles = np.arange(0.0, 180.0, angle_step, dtype=np.float64)
    if angles.size == 0:
        angles = np.asarray([0.0], dtype=np.float64)
    angles = angles[angles < 180.0]
    if angles.size == 0:
        angles = np.asarray([0.0], dtype=np.float64)
    cosines = np.cos(np.deg2rad(angles))
    sines = np.sin(np.deg2rad(angles))
    center_fractions = np.asarray(_LOCAL_SCAN_CENTER_FRACTIONS, dtype=np.float64)
    candidate_shape = (seed_x.size, center_fractions.size, angles.size)
    coarse_candidates: list[tuple[float, float, float, float, float]] = []
    top_per_length = min(_LOCAL_SCAN_TOP_PER_LENGTH, max(16, max_output * 2))

    # 每个长度单独处理，限制 map_coordinates 的临时数组大小；种子、方向
    # 和中心偏移三维则一次向量化，避免 Python 层逐候选调用插值器。
    for length in lengths:
        candidate_x = seed_x[:, None, None] + (
            float(length) * center_fractions[None, :, None] * cosines[None, None, :]
        )
        candidate_y = seed_y[:, None, None] + (
            float(length) * center_fractions[None, :, None] * sines[None, None, :]
        )
        sample_offsets = np.linspace(
            -0.5 * float(length),
            0.5 * float(length),
            max(5, int(math.ceil(float(length) / sample_step)) + 1),
            dtype=np.float64,
        )
        sample_x = candidate_x[..., None] + sample_offsets[None, None, None, :] * cosines[None, None, :, None]
        sample_y = candidate_y[..., None] + sample_offsets[None, None, None, :] * sines[None, None, :, None]
        flattened_coordinates = (sample_y.reshape(-1), sample_x.reshape(-1))
        sampled_signal = ndimage.map_coordinates(
            smoothed,
            flattened_coordinates,
            order=1,
            mode="constant",
            cval=0.0,
            prefilter=False,
        ).reshape(*candidate_shape, sample_offsets.size)
        sampled_support = ndimage.map_coordinates(
            valid_float,
            flattened_coordinates,
            order=1,
            mode="constant",
            cval=0.0,
            prefilter=False,
        ).reshape(*candidate_shape, sample_offsets.size)
        sampled_support = np.clip(sampled_support, 0.0, 1.0)
        coverage = np.mean(sampled_support, axis=-1)
        weight_sum = np.sum(sampled_support, axis=-1)
        signal_sum = np.sum(sampled_signal * sampled_support, axis=-1)
        mean_signal = np.zeros_like(signal_sum)
        np.divide(signal_sum, weight_sum, out=mean_signal, where=weight_sum > 1e-6)
        response = mean_signal * np.sqrt(np.clip(coverage, 0.0, 1.0)) / _line_noise_std(sigma, length)
        response[coverage < _MIN_LINE_COVERAGE] = -np.inf
        eligible_response = np.flatnonzero(np.isfinite(response) & (response >= _LOCAL_SCAN_RESPONSE_THRESHOLD))
        if eligible_response.size == 0:
            continue
        if eligible_response.size > top_per_length:
            values = response.ravel()[eligible_response]
            keep = np.argpartition(values, -top_per_length)[-top_per_length:]
            eligible_response = eligible_response[keep]
        eligible_response = eligible_response[np.argsort(response.ravel()[eligible_response])[::-1]]
        for flat_index in eligible_response:
            seed_index, center_index, angle_index = np.unravel_index(int(flat_index), candidate_shape)
            coarse_candidates.append(
                (
                    float(candidate_x[seed_index, center_index, angle_index]),
                    float(candidate_y[seed_index, center_index, angle_index]),
                    float(length),
                    float(angles[angle_index]),
                    float(response[seed_index, center_index, angle_index]),
                )
            )

    if not coarse_candidates:
        return ()
    coarse_candidates.sort(key=lambda item: item[4], reverse=True)
    refinement_limit = min(len(coarse_candidates), max(64, max_output * 4))
    refined: list[tuple[float, float, float, float, float]] = []
    for candidate in coarse_candidates[:refinement_limit]:
        x, y, length, angle, response = _refine_candidate(
            smoothed,
            valid_float,
            candidate,
            sigma,
            angle_step,
        )
        if not all(math.isfinite(float(value)) for value in (x, y, length, angle, response)):
            continue
        if length < max(4.0, _MIN_ACCEPTED_LENGTH_IN_FWHM * fwhm):
            continue
        if not _passes_compact_source_control(
            whitened,
            usable,
            x=x,
            y=y,
            length=length,
            angle_deg=angle,
            fwhm=fwhm,
        ):
            continue
        if response >= _LOCAL_SCAN_RESPONSE_THRESHOLD:
            refined.append((x, y, length, angle, response))
    if not refined:
        return ()
    return _deduplicate(refined, fwhm, max_output)


def _refine_candidate(
    smoothed: np.ndarray,
    valid_float: np.ndarray,
    candidate: tuple[float, float, float, float, float],
    sigma: float,
    angle_step_deg: float,
) -> tuple[float, float, float, float, float]:
    """在粗方向/长度附近做一次小网格复核，返回最佳参数。"""

    x, y, coarse_length, coarse_angle, coarse_response = candidate
    angle_offsets = (-angle_step_deg / 2.0, 0.0, angle_step_deg / 2.0)
    length_factors = (0.80, 1.00, 1.25)
    best = (coarse_response, coarse_length, coarse_angle)
    for angle_offset in angle_offsets:
        angle = (coarse_angle + angle_offset) % 180.0
        for length_factor in length_factors:
            length = max(2.0, coarse_length * length_factor)
            response = _sample_line_response(
                smoothed,
                valid_float,
                x,
                y,
                length,
                angle,
                sigma,
            )
            if response > best[0]:
                best = (response, length, angle)
    response, length, angle = best
    return x, y, length, angle, response


def _angle_distance_deg(first: float, second: float) -> float:
    """计算无向线角度的最小差值，范围为 [0, 90]。"""

    difference = abs((first - second) % 180.0)
    return min(difference, 180.0 - difference)


def _deduplicate(
    candidates: list[tuple[float, float, float, float, float]],
    fwhm: float,
    max_output: int,
) -> tuple[TrailProposal, ...]:
    """按响应从高到低做线段几何 NMS，保留最佳 proposal。

    仅按中心距离去重会在一条均匀长线的响应平台上留下许多相邻峰。本
    函数还比较两条无向线段的垂直距离、沿线投影间距和一维重叠比例，因而
    同一条线在相邻长度/方向模板中产生的多个峰只保留一个；两条平行但
    实际分开的线，只要垂直间距超过 PSF 尺度，仍可分别保留。
    """

    candidates.sort(key=lambda item: item[4], reverse=True)
    kept: list[TrailProposal] = []
    center_radius = max(2.5, _PEAK_SEPARATION_FACTOR * fwhm)
    for x, y, length, angle, response in candidates:
        duplicate = False
        for existing in kept:
            shorter_length = min(length, existing.length)
            angle_tolerance = max(
                _ANGLE_DEDUP_TOLERANCE_DEG,
                min(30.0, 240.0 * fwhm / max(shorter_length, 1.0)),
            )
            if _angle_distance_deg(angle, existing.angle) > angle_tolerance:
                continue

            angle_radians = math.radians(existing.angle)
            direction_x = math.cos(angle_radians)
            direction_y = math.sin(angle_radians)
            delta_x = x - existing.x
            delta_y = y - existing.y
            perpendicular_distance = abs(delta_x * direction_y - delta_y * direction_x)
            if perpendicular_distance > center_radius:
                continue

            parallel_distance = abs(delta_x * direction_x + delta_y * direction_y)
            overlap = (length + existing.length) / 2.0 - parallel_distance
            overlap_fraction = overlap / shorter_length if shorter_length > 0.0 else 0.0
            length_ratio = length / existing.length if existing.length > 0.0 else math.inf

            # 中心很近时仍保留长度比例门，避免把一个真正独立的极短源
            # 吞进超长线；线段沿线明显重叠时则允许跨尺度去重。
            centers_close = math.hypot(delta_x, delta_y) <= center_radius
            same_scale = _MIN_LENGTH_RATIO_FOR_DEDUP <= length_ratio <= _MAX_LENGTH_RATIO_FOR_DEDUP
            if (centers_close and same_scale) or overlap_fraction >= 0.25:
                duplicate = True
                break
        if duplicate:
            continue
        kept.append(
            TrailProposal(
                x=float(x),
                y=float(y),
                length=float(length),
                width=float(fwhm),
                angle=float(angle % 180.0),
                response=float(response),
            )
        )
        if len(kept) >= max_output:
            break
    return tuple(kept)


def detect_single_frame_trail_proposals(
    image: Any,
    valid_mask: Any,
    background: Any,
    noise: Any,
    psf_fwhm: Any,
    *,
    length_bank_px: Any = None,
    angle_step_deg: Any = _ANGLE_STEP_DEG,
    max_proposals: Any = _MAX_OUTPUT_PROPOSALS,
) -> tuple[TrailProposal, ...]:
    """从当前一张图像产生去重后的有限曝光线段 proposals。

    参数只有当前图像上下文：

    ``image``
        当前单帧二维 ADU/灰度数组。
    ``valid_mask``
        与图像同形状的 bool 或 0/1 数组；False 区域不参与白化和线积分。
    ``background``
        标量或与图像同形状的背景估计图。
    ``noise``
        标量或与图像同形状的正噪声标准差图。它不是跨帧统计量。
    ``psf_fwhm``
        当前图像的 PSF FWHM，单位为像素。
    ``length_bank_px``
        可选的线段长度库，单位为像素。省略时自动建立从短线到长线的
        几何长度库；它只描述当前图像内的空间搜索尺度。
    ``angle_step_deg``
        可选的无向方向步长，单位为度，范围为 ``(0, 180]``。
    ``max_proposals``
        返回的 proposal 数量上限，必须是正整数。

    返回
    ------
    tuple[TrailProposal, ...]
        按 ``response`` 从高到低排列、已做几何去重的 proposals。该宽筛
        有意保留点源/双星/固定线等可能混淆项，不能直接当作 ``moving``。

    复杂度
    --------
    设图像大小为 ``H×W``，方向数为 ``A``，长度尺度数为 ``S``，粗筛峰值
    数为 ``K``。旋转和一维滑动积分的主要复杂度约为
    ``O(HW + A*S*HW + K*R)``，其中 ``R`` 是局部复核采样总量；内存主要为
    ``O(HW + K)``，不会同时保存所有方向/长度的响应图。

    注意
    ------
    单帧线段只能给出无向角度 ``angle mod 180°``。1.5 秒曝光若要换算像面
    速度，应在调用者拿到拟合长度后另行计算 ``length / 1.5``；本原型不
    接收曝光序列，也不把曝光时间偷偷当成跨帧运动证据。
    """

    image_array, valid_mask_array, background_array, noise_array, fwhm, usable = _validate_inputs(
        image,
        valid_mask,
        background,
        noise,
        psf_fwhm,
    )

    try:
        angle_step = float(angle_step_deg)
    except (TypeError, ValueError) as exc:
        raise TypeError("angle_step_deg 必须是有限的正数") from exc
    if not math.isfinite(angle_step) or angle_step <= 0.0 or angle_step > 180.0:
        raise ValueError("angle_step_deg 必须满足 0 < angle_step_deg <= 180")

    if isinstance(max_proposals, (bool, np.bool_)):
        raise TypeError("max_proposals 必须是正整数")
    try:
        max_output = int(max_proposals)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError("max_proposals 必须是正整数") from exc
    if max_output <= 0 or float(max_proposals) != float(max_output):
        raise ValueError("max_proposals 必须是正整数")
    if max_output > 100_000:
        raise ValueError("max_proposals 不能大于 100000")

    if length_bank_px is None:
        lengths = _length_bank(image_array.shape, fwhm)
    else:
        try:
            raw_lengths = np.asarray(length_bank_px, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise TypeError("length_bank_px 必须是一维正数序列") from exc
        if raw_lengths.ndim == 0:
            raw_lengths = raw_lengths.reshape(1)
        if raw_lengths.ndim != 1 or raw_lengths.size == 0:
            raise ValueError("length_bank_px 必须是非空的一维正数序列")
        if not np.all(np.isfinite(raw_lengths)) or np.any(raw_lengths <= 0.0):
            raise ValueError("length_bank_px 必须全部是有限的正数")
        max_allowed_length = float(min(image_array.shape) - 1)
        if np.any(raw_lengths > max_allowed_length):
            raise ValueError(f"length_bank_px 不能大于图像最短边减一（{max_allowed_length:g} px）")
        normalised_lengths: list[int] = []
        for raw_length in raw_lengths:
            integer_length = max(2, int(round(float(raw_length))))
            if integer_length > int(max_allowed_length):
                raise ValueError("length_bank_px 中存在取整后超出图像尺寸的长度")
            if integer_length not in normalised_lengths:
                normalised_lengths.append(integer_length)
        lengths = tuple(normalised_lengths)
        if not lengths:
            raise ValueError("length_bank_px 取整后没有有效长度")

    # 白化只使用当前图像提供的背景/噪声。无效区置零，避免 NaN/Inf 进入
    # scipy 的滤波器；有效区的有限性和 noise>0 已在入口统一检查。
    whitened = np.zeros_like(image_array, dtype=np.float64)
    whitened[usable] = (image_array[usable] - background_array[usable]) / noise_array[usable]
    valid_float = usable.astype(np.float64)

    sigma = max(fwhm / 2.354820045, 0.5)
    smoothed_signal = ndimage.gaussian_filter(
        whitened,
        sigma=sigma,
        mode="constant",
        cval=0.0,
        truncate=_GAUSSIAN_TRUNCATE,
    )
    smoothed_support = ndimage.gaussian_filter(
        valid_float,
        sigma=sigma,
        mode="constant",
        cval=0.0,
        truncate=_GAUSSIAN_TRUNCATE,
    )
    # 在 mask 边界做归一化，避免有效区域靠近无效边缘时被人为压暗。
    smoothed = np.zeros_like(smoothed_signal)
    np.divide(smoothed_signal, smoothed_support, out=smoothed, where=smoothed_support > 1e-6)

    center_y = (image_array.shape[0] - 1) / 2.0
    center_x = (image_array.shape[1] - 1) / 2.0
    peak_size = max(3, 2 * int(math.ceil(fwhm)) + 1)
    raw_candidates: list[tuple[float, float, float, float, float]] = []
    angles = np.arange(0.0, 180.0, angle_step, dtype=np.float64)
    if angles.size == 0:
        angles = np.asarray([0.0], dtype=np.float64)
    if angles[-1] >= 180.0:
        angles = angles[angles < 180.0]
    peaks_per_template = max(8, min(_MAX_PEAKS_PER_TEMPLATE, max_output))

    # 逐方向处理：旋转图像和一个长度响应图只在当前循环中存在，不保存
    # H×W×A×S 的响应张量，适合先做真实大图原型性能测试。长度库先在
    # 像素位置上取最大响应，再做一次二维局部峰值筛选；原先每个长度都
    # 做一遍 maximum_filter，在大图上会把同一条线的重复平台反复扫描。
    for angle_deg in angles:
        rotated_signal = ndimage.rotate(
            smoothed,
            float(angle_deg),
            reshape=False,
            order=1,
            mode="constant",
            cval=0.0,
            prefilter=False,
        )
        rotated_valid = ndimage.rotate(
            valid_float,
            float(angle_deg),
            reshape=False,
            order=1,
            mode="constant",
            cval=0.0,
            prefilter=False,
        )
        rotated_valid = np.clip(rotated_valid, 0.0, 1.0)

        best_response = np.full(rotated_signal.shape, -np.inf, dtype=np.float64)
        best_coverage = np.zeros(rotated_signal.shape, dtype=np.float64)
        best_length = np.zeros(rotated_signal.shape, dtype=np.float64)
        for length in lengths:
            line_mean = ndimage.uniform_filter1d(
                rotated_signal,
                size=length,
                axis=1,
                mode="constant",
                cval=0.0,
            )
            coverage = ndimage.uniform_filter1d(
                rotated_valid,
                size=length,
                axis=1,
                mode="constant",
                cval=0.0,
            )
            # line_mean 是含零填充的均值；先还原有效像素均值，再按有效比例
            # 对显著性做保守修正。
            corrected_mean = np.zeros_like(line_mean)
            np.divide(line_mean, coverage, out=corrected_mean, where=coverage > 1e-6)
            noise_std = _line_noise_std(sigma, length)
            response = corrected_mean * np.sqrt(np.clip(coverage, 0.0, 1.0)) / noise_std
            response[coverage < _MIN_LINE_COVERAGE] = -np.inf

            better = response > best_response
            best_response[better] = response[better]
            best_coverage[better] = coverage[better]
            best_length[better] = float(length)

        xs, ys, responses = _top_local_peaks(
            best_response,
            best_coverage,
            _COARSE_RESPONSE_MIN,
            peaks_per_template,
            peak_size,
        )
        if xs.size == 0:
            continue
        original_x, original_y = _rotated_to_original(
            xs,
            ys,
            float(angle_deg),
            center_x,
            center_y,
        )
        for x, y, coarse_response, response_y, response_x in zip(
            original_x,
            original_y,
            responses,
            ys.astype(np.intp, copy=False),
            xs.astype(np.intp, copy=False),
        ):
            if not (math.isfinite(float(x)) and math.isfinite(float(y))):
                continue
            length = best_length[response_y, response_x]
            if length <= 0.0 or not math.isfinite(float(length)):
                continue
            raw_candidates.append(
                (float(x), float(y), float(length), float(angle_deg), float(coarse_response))
            )

    if not raw_candidates:
        return ()

    # 先限制局部复核的数量。每个方向/长度已经保留了局部峰值，这一步只
    # 是为了让最坏情况下的候选复核成本有上界，不使用任何跨帧信息。
    raw_candidates.sort(key=lambda item: item[4], reverse=True)
    refinement_candidates = raw_candidates[: min(_MAX_REFINEMENT_CANDIDATES, max(64, max_output * 4))]
    refined: list[tuple[float, float, float, float, float]] = []
    height, width = image_array.shape
    for candidate in refinement_candidates:
        x, y, length, angle, response = _refine_candidate(
            smoothed,
            valid_float,
            candidate,
            sigma,
            angle_step,
        )
        if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(response)):
            continue
        if length < max(4.0, _MIN_ACCEPTED_LENGTH_IN_FWHM * fwhm):
            continue
        if not _passes_compact_source_control(
            whitened,
            usable,
            x=x,
            y=y,
            length=length,
            angle_deg=angle,
            fwhm=fwhm,
        ):
            continue
        # 中心在图内且线段本身有足够的局部支持，才进入最终 NMS。
        if x < -0.5 or x > width - 0.5 or y < -0.5 or y > height - 0.5:
            continue
        if response >= _FINAL_RESPONSE_MIN:
            refined.append((x, y, length, angle, response))

    if not refined:
        return ()
    return _deduplicate(refined, fwhm, max_output)
