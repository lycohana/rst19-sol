"""15 帧注册镶嵌图生成。

这个模块和 ``sequence._registered_median_reference`` 有意分开：后者只在
参考帧坐标中生成一张同尺寸的提案图，用于弱星候选提取；这里生成的是
15 帧输入视场的联合坐标域（union footprint），因此输出可以比单帧更大。

科学口径：

* 所有帧先使用序列分析得到的累计平移注册到第 1 帧坐标系；
* 重叠像素默认采用一次迭代的稳健均值，压制只在少数帧出现的线状/瞬态
  值；``median`` 模式保留为更保守的显示对照；
* ``coverage`` 记录每个输出像素真正贡献了多少帧，不能用填充色伪造覆盖区；
* ``scatter`` 记录同一输出像素的帧间稳健离散度，帮助识别拼接边界和不稳定
  结构；
* 输出图像仍是原始 ADU 的重采样/融合结果，不是物理标定后的辐亮度或星表。

大图只保存为独立缓存/研究产物，不写回 FITS，也不改变星点检测结果。
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from scipy import ndimage

from .cache import _file_identity
from .fits import auxiliary_mask, read_fits


MOSAIC_CACHE_VERSION = 1
MOSAIC_DEFAULT_TILE_ROWS = 256
MOSAIC_DEFAULT_CLIP_SIGMA = 3.0


@dataclass(frozen=True, slots=True)
class MosaicResult:
    """注册 15 帧联合视场及其可审计元数据。"""

    image: np.ndarray
    coverage: np.ndarray
    scatter: np.ndarray
    source_shape: tuple[int, int]
    output_shape: tuple[int, int]
    common_origin_xy: tuple[float, float]
    frame_origins_xy: tuple[tuple[float, float], ...]
    cumulative_shifts: tuple[tuple[float, float], ...]
    frame_paths: tuple[str, ...]
    combine_mode: str
    clip_sigma: float
    interpolation_order: int
    masked_zero_pixels: bool
    masked_auxiliary_pixels: bool
    frame_valid_pixel_counts: tuple[int, ...]

    @property
    def frame_count(self) -> int:
        return len(self.cumulative_shifts)

    @property
    def covered_pixel_count(self) -> int:
        return int(np.count_nonzero(self.coverage > 0))

    @property
    def overlap_pixel_count(self) -> int:
        return int(np.count_nonzero(self.coverage >= 2))

    @property
    def max_coverage(self) -> int:
        return int(np.max(self.coverage)) if self.coverage.size else 0

    @property
    def mean_coverage(self) -> float:
        covered = self.coverage[self.coverage > 0]
        return float(np.mean(covered)) if covered.size else 0.0

    @property
    def covered_bbox_xy(self) -> tuple[int, int, int, int] | None:
        """返回有效 footprint 的紧包围盒 ``(x0, y0, x1, y1)``。"""

        ys, xs = np.where(self.coverage > 0)
        if xs.size == 0:
            return None
        return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

    @property
    def uncovered_pixel_count(self) -> int:
        return int(np.count_nonzero(self.coverage == 0))

    @property
    def union_area_ratio(self) -> float:
        source_area = max(1, int(self.source_shape[0]) * int(self.source_shape[1]))
        return float(self.covered_pixel_count / source_area)

    def as_dict(self) -> dict[str, object]:
        """返回不含大数组的 JSON 元数据。"""

        return {
            "mosaic_cache_version": MOSAIC_CACHE_VERSION,
            "frame_count": self.frame_count,
            "source_shape": list(self.source_shape),
            "output_shape": list(self.output_shape),
            "common_origin_xy": list(self.common_origin_xy),
            "frame_origins_xy": [list(origin) for origin in self.frame_origins_xy],
            "cumulative_shifts": [list(shift) for shift in self.cumulative_shifts],
            "frame_paths": list(self.frame_paths),
            "combine_mode": self.combine_mode,
            "clip_sigma": self.clip_sigma,
            "interpolation_order": self.interpolation_order,
            "masked_zero_pixels": self.masked_zero_pixels,
            "masked_auxiliary_pixels": self.masked_auxiliary_pixels,
            "frame_valid_pixel_counts": list(self.frame_valid_pixel_counts),
            "covered_pixel_count": self.covered_pixel_count,
            "overlap_pixel_count": self.overlap_pixel_count,
            "max_coverage": self.max_coverage,
            "mean_coverage": self.mean_coverage,
            "covered_bbox_xy": list(self.covered_bbox_xy) if self.covered_bbox_xy is not None else None,
            "uncovered_pixel_count": self.uncovered_pixel_count,
            "union_area_ratio": self.union_area_ratio,
        }


def _coerce_frame(item: Any) -> tuple[np.ndarray, str | None]:
    """读取路径/FitsFrame/数组，保留原始存储值的数值语义。"""

    if isinstance(item, (str, Path)):
        path = Path(item)
        frame = read_fits(path)
        return np.asarray(frame.data), str(path.resolve())
    data = getattr(item, "data", item)
    path = getattr(item, "path", None)
    return np.asarray(data), str(Path(path).resolve()) if path is not None else None


def _prepare_float_frame(
    data: np.ndarray,
    *,
    mask_zero_pixels: bool,
    mask_auxiliary_pixels: bool,
) -> np.ndarray:
    """把一帧转为融合用 float32，并只屏蔽已知无效位置。"""

    if data.ndim != 2:
        raise ValueError(f"mosaic expects 2-D frames, got shape {data.shape}")
    working = np.asarray(data, dtype=np.float32).copy()
    working[~np.isfinite(working)] = np.nan
    if mask_zero_pixels:
        working[working == 0.0] = np.nan
    if mask_auxiliary_pixels:
        working[auxiliary_mask(tuple(int(value) for value in working.shape))] = np.nan
    return working


def _mosaic_bounds(
    shape: tuple[int, int],
    shifts: np.ndarray,
) -> tuple[int, int, float, float, tuple[tuple[float, float], ...]]:
    """计算联合注册坐标域及每一帧左上角在输出网格中的位置。"""

    height, width = shape
    # 注册约定与 sequence.py 一致：common = raw - cumulative_shift。
    common_min_x = float(np.min(-shifts[:, 0]))
    common_max_x = float(np.max(float(width - 1) - shifts[:, 0]))
    common_min_y = float(np.min(-shifts[:, 1]))
    common_max_y = float(np.max(float(height - 1) - shifts[:, 1]))
    # 向外取整，给双线性插值的边界留一个完整像素，避免最后一行/列
    # 因浮点误差被裁掉。origin 是输出像素 (0,0) 对应的第 1 帧坐标。
    origin_x = float(np.floor(common_min_x))
    origin_y = float(np.floor(common_min_y))
    max_x = int(np.ceil(common_max_x))
    max_y = int(np.ceil(common_max_y))
    output_width = max(1, max_x - int(origin_x) + 1)
    output_height = max(1, max_y - int(origin_y) + 1)
    frame_origins = tuple(
        (
            float(-shift_x - origin_x),
            float(-shift_y - origin_y),
        )
        for shift_x, shift_y in shifts
    )
    return output_height, output_width, origin_x, origin_y, frame_origins


def _sample_frame_tile(
    frame: np.ndarray,
    *,
    shift_x: float,
    shift_y: float,
    origin_x: float,
    origin_y: float,
    y_start: int,
    y_stop: int,
    output_width: int,
    interpolation_order: int,
) -> np.ndarray:
    """从一帧原始 ADU 采样到联合注册坐标域的一行块。"""

    output_y = np.arange(y_start, y_stop, dtype=np.float32)[:, None] + np.float32(origin_y)
    output_x = np.arange(output_width, dtype=np.float32)[None, :] + np.float32(origin_x)
    source_y = output_y + np.float32(shift_y)
    source_x = output_x + np.float32(shift_x)
    coordinates = (
        np.broadcast_to(source_y, (y_stop - y_start, output_width)),
        np.broadcast_to(source_x, (y_stop - y_start, output_width)),
    )
    # 不能把 NaN 直接交给双线性插值：在图像边界或辅助字段附近，SciPy
    # 会让一个无效邻居污染整个插值点。先对有效值和有效权重分别重采样，
    # 再做归一化，既不填入虚构 ADU，也不把边界有效像素整行丢掉。
    valid = np.isfinite(frame)
    finite_frame = np.where(valid, frame, 0.0).astype(np.float32, copy=False)
    sampled_values = ndimage.map_coordinates(
        finite_frame,
        coordinates,
        order=int(interpolation_order),
        mode="constant",
        cval=0.0,
        prefilter=False,
    ).astype(np.float32, copy=False)
    sampled_weights = ndimage.map_coordinates(
        valid.astype(np.float32, copy=False),
        coordinates,
        order=int(interpolation_order),
        mode="constant",
        cval=0.0,
        prefilter=False,
    ).astype(np.float32, copy=False)
    with np.errstate(invalid="ignore", divide="ignore"):
        sampled = sampled_values / sampled_weights
    sampled[sampled_weights <= np.float32(1e-6)] = np.nan
    return sampled


def _combine_tile(
    samples: np.ndarray,
    *,
    mode: str,
    clip_sigma: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """融合一个 ``frame × tile`` 样本块并返回图像、覆盖数、离散度。"""

    valid = np.isfinite(samples)
    coverage = np.sum(valid, axis=0, dtype=np.uint16)
    output = np.full(coverage.shape, np.nan, dtype=np.float32)
    scatter = np.full(coverage.shape, np.nan, dtype=np.float32)
    if not np.any(valid):
        return output, coverage, scatter

    with warnings.catch_warnings(), np.errstate(invalid="ignore", divide="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        median = np.nanmedian(samples, axis=0)
        absolute_deviation = np.abs(samples - median[None, :, :])
        robust_scatter = 1.4826 * np.nanmedian(absolute_deviation, axis=0)
    scatter[:] = robust_scatter.astype(np.float32, copy=False)

    if mode == "median":
        output[:] = median.astype(np.float32, copy=False)
        return output, coverage, scatter

    # 稳健均值：先以中位数/MAD 找到少数帧的异常值，再对剩余原始 ADU
    # 做均值。覆盖数为 1/2 时不做过度裁剪，避免边缘被算法“修平”。
    scale = np.maximum(robust_scatter, np.float32(1e-3))
    keep = valid & (
        (np.abs(samples - median[None, :, :]) <= np.float32(clip_sigma) * scale[None, :, :])
        | (coverage[None, :, :] <= 2)
    )
    kept_count = np.sum(keep, axis=0, dtype=np.uint16)
    with np.errstate(invalid="ignore", divide="ignore"):
        summed = np.sum(np.where(keep, samples, 0.0), axis=0, dtype=np.float32)
        robust_mean = summed / kept_count
    # 极少数 MAD=0 或全部样本被裁剪的像素回退到中位数，不填充无覆盖区域。
    use_median = (kept_count == 0) & (coverage > 0)
    robust_mean[use_median] = median[use_median]
    output[:] = robust_mean.astype(np.float32, copy=False)
    return output, coverage, scatter


def build_registered_mosaic(
    frames: Sequence[Any],
    cumulative_shifts: Sequence[tuple[float, float]],
    *,
    combine_mode: str = "robust_mean",
    clip_sigma: float = MOSAIC_DEFAULT_CLIP_SIGMA,
    tile_rows: int = MOSAIC_DEFAULT_TILE_ROWS,
    interpolation_order: int = 1,
    mask_zero_pixels: bool = True,
    mask_auxiliary_pixels: bool = True,
    progress: Callable[[float, str], None] | None = None,
) -> MosaicResult:
    """生成真实的 15 帧注册联合视场大图。

    ``frames`` 可传 FITS 路径、``FitsFrame`` 或二维 NumPy 数组。输入数组会
    被复制为 float32 后再处理；函数不会修改用户传入的数组。
    """

    if combine_mode not in {"robust_mean", "median"}:
        raise ValueError("combine_mode must be 'robust_mean' or 'median'")
    if not frames:
        raise ValueError("at least one frame is required")
    if len(frames) != len(cumulative_shifts):
        raise ValueError("frames and cumulative_shifts must have the same length")
    if clip_sigma <= 0:
        raise ValueError("clip_sigma must be positive")
    if not 0 <= interpolation_order <= 3:
        raise ValueError("interpolation_order must be between 0 and 3")

    raw_frames: list[np.ndarray] = []
    frame_paths: list[str] = []
    if progress is not None:
        progress(1.0, "读取合成输入")
    for index, item in enumerate(frames, start=1):
        data, path = _coerce_frame(item)
        if data.ndim != 2:
            raise ValueError(f"frame {index} is not a 2-D image: {data.shape}")
        raw_frames.append(data)
        if path is not None:
            frame_paths.append(path)
        if progress is not None:
            progress(1.0 + 9.0 * index / len(frames), f"读取合成输入 {index}/{len(frames)}")

    shape = tuple(int(value) for value in raw_frames[0].shape)
    if any(tuple(int(value) for value in frame.shape) != shape for frame in raw_frames):
        raise ValueError("all mosaic frames must have the same image shape")
    shifts = np.asarray(cumulative_shifts, dtype=np.float64)
    if shifts.shape != (len(raw_frames), 2) or not np.all(np.isfinite(shifts)):
        raise ValueError("cumulative_shifts must be finite (frame_count, 2) values")
    if progress is not None:
        progress(11.0, "计算联合视场边界")
    output_height, output_width, origin_x, origin_y, frame_origins = _mosaic_bounds(shape, shifts)
    tile_height = max(16, int(tile_rows))
    tile_count = max(1, int(np.ceil(output_height / tile_height)))

    # 15 张比赛 FITS 通常是 int16；只在这里转成 float32，每一帧只保留一份
    # 工作数组。合成块本身最多是 15×tile_rows×output_width，内存有界。
    prepared_frames = [
        _prepare_float_frame(
            data,
            mask_zero_pixels=mask_zero_pixels,
            mask_auxiliary_pixels=mask_auxiliary_pixels,
        )
        for data in raw_frames
    ]
    del raw_frames

    mosaic = np.full((output_height, output_width), np.nan, dtype=np.float32)
    coverage_map = np.zeros((output_height, output_width), dtype=np.uint16)
    scatter_map = np.full((output_height, output_width), np.nan, dtype=np.float32)
    frame_valid_counts = np.zeros(len(prepared_frames), dtype=np.int64)
    x_axis = np.arange(output_width, dtype=np.float32)[None, :]
    del x_axis

    for tile_index, y_start in enumerate(range(0, output_height, tile_height), start=1):
        y_stop = min(output_height, y_start + tile_height)
        samples = np.full(
            (len(prepared_frames), y_stop - y_start, output_width),
            np.nan,
            dtype=np.float32,
        )
        for frame_index, (frame, shift) in enumerate(zip(prepared_frames, shifts, strict=True)):
            sampled = _sample_frame_tile(
                frame,
                shift_x=float(shift[0]),
                shift_y=float(shift[1]),
                origin_x=origin_x,
                origin_y=origin_y,
                y_start=y_start,
                y_stop=y_stop,
                output_width=output_width,
                interpolation_order=interpolation_order,
            )
            samples[frame_index] = sampled
            frame_valid_counts[frame_index] += int(np.count_nonzero(np.isfinite(sampled)))
        image_tile, coverage_tile, scatter_tile = _combine_tile(
            samples,
            mode=combine_mode,
            clip_sigma=float(clip_sigma),
        )
        mosaic[y_start:y_stop] = image_tile
        coverage_map[y_start:y_stop] = coverage_tile
        scatter_map[y_start:y_stop] = scatter_tile
        if progress is not None:
            progress(
                12.0 + 84.0 * tile_index / tile_count,
                f"注册融合 {tile_index}/{tile_count} 块",
            )

    result = MosaicResult(
        image=mosaic,
        coverage=coverage_map,
        scatter=scatter_map,
        source_shape=shape,
        output_shape=(output_height, output_width),
        common_origin_xy=(origin_x, origin_y),
        frame_origins_xy=frame_origins,
        cumulative_shifts=tuple((float(row[0]), float(row[1])) for row in shifts),
        frame_paths=tuple(frame_paths),
        combine_mode=combine_mode,
        clip_sigma=float(clip_sigma),
        interpolation_order=int(interpolation_order),
        masked_zero_pixels=bool(mask_zero_pixels),
        masked_auxiliary_pixels=bool(mask_auxiliary_pixels),
        frame_valid_pixel_counts=tuple(int(value) for value in frame_valid_counts),
    )
    if progress is not None:
        progress(100.0, "合成大图完成")
    return result


def mosaic_cache_key(
    frame_paths: Sequence[str | Path],
    cumulative_shifts: Sequence[tuple[float, float]],
    *,
    combine_mode: str = "robust_mean",
    clip_sigma: float = MOSAIC_DEFAULT_CLIP_SIGMA,
    interpolation_order: int = 1,
    mask_zero_pixels: bool = True,
    mask_auxiliary_pixels: bool = True,
) -> str:
    """用输入文件状态、配准平移和合成口径生成独立缓存键。"""

    files = []
    for item in frame_paths:
        path = Path(item).resolve()
        # 只用 size+mtime 会在文件被等长替换且恢复时间戳时误命中旧
        # 大图。复用普通缓存的内容摘要契约，确保显示层也绑定原始 FITS。
        files.append(_file_identity(path))
    payload = {
        "mosaic_cache_version": MOSAIC_CACHE_VERSION,
        "frames": files,
        "cumulative_shifts": [[float(x), float(y)] for x, y in cumulative_shifts],
        "combine_mode": combine_mode,
        "clip_sigma": float(clip_sigma),
        "interpolation_order": int(interpolation_order),
        "mask_zero_pixels": bool(mask_zero_pixels),
        "mask_auxiliary_pixels": bool(mask_auxiliary_pixels),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def mosaic_cache_path(cache_dir: Path, key: str) -> Path:
    return Path(cache_dir) / f"{key}.mosaic.npz"


def save_mosaic_cache(cache_dir: Path, key: str, result: MosaicResult) -> Path:
    """原子写入大图数组和元数据。"""

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = mosaic_cache_path(cache_dir, key)
    temporary_fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=cache_dir,
    )
    os.close(temporary_fd)
    temporary = Path(temporary_name)
    metadata = json.dumps(result.as_dict(), ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    try:
        with temporary.open("wb") as stream:
            np.savez(
                stream,
                image=result.image.astype(np.float32, copy=False),
                coverage=result.coverage.astype(np.uint16, copy=False),
                scatter=result.scatter.astype(np.float32, copy=False),
                metadata=np.asarray(metadata),
            )
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return path


def load_mosaic_cache(cache_dir: Path, key: str) -> MosaicResult | None:
    """读取大图缓存；损坏、旧版本或形状不一致时安全地按未命中处理。"""

    path = mosaic_cache_path(cache_dir, key)
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as payload:
            image = np.asarray(payload["image"], dtype=np.float32)
            coverage = np.asarray(payload["coverage"], dtype=np.uint16)
            scatter = np.asarray(payload["scatter"], dtype=np.float32)
            metadata = json.loads(str(np.asarray(payload["metadata"]).item()))
        if metadata.get("mosaic_cache_version") != MOSAIC_CACHE_VERSION:
            return None
        if image.ndim != 2 or coverage.shape != image.shape or scatter.shape != image.shape:
            return None
        source_shape = tuple(int(value) for value in metadata["source_shape"])
        output_shape = tuple(int(value) for value in metadata["output_shape"])
        if tuple(image.shape) != output_shape:
            return None
        return MosaicResult(
            image=image,
            coverage=coverage,
            scatter=scatter,
            source_shape=source_shape,
            output_shape=output_shape,
            common_origin_xy=tuple(float(value) for value in metadata["common_origin_xy"]),
            frame_origins_xy=tuple(tuple(float(value) for value in row) for row in metadata["frame_origins_xy"]),
            cumulative_shifts=tuple(tuple(float(value) for value in row) for row in metadata["cumulative_shifts"]),
            frame_paths=tuple(str(value) for value in metadata.get("frame_paths", [])),
            combine_mode=str(metadata["combine_mode"]),
            clip_sigma=float(metadata["clip_sigma"]),
            interpolation_order=int(metadata["interpolation_order"]),
            masked_zero_pixels=bool(metadata["masked_zero_pixels"]),
            masked_auxiliary_pixels=bool(metadata["masked_auxiliary_pixels"]),
            frame_valid_pixel_counts=tuple(int(value) for value in metadata.get("frame_valid_pixel_counts", [])),
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError, EOFError):
        return None


def write_mosaic_artifacts(output_dir: Path, result: MosaicResult, *, preview: Any | None = None) -> Path:
    """写出可复核的大图数组、覆盖图、离散度和元数据。"""

    from PIL import Image

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "registered_mosaic_adu.npy", result.image.astype(np.float32, copy=False))
    np.save(output / "registered_mosaic_coverage.npy", result.coverage.astype(np.uint16, copy=False))
    np.save(output / "registered_mosaic_scatter.npy", result.scatter.astype(np.float32, copy=False))
    (output / "registered_mosaic_metadata.json").write_text(
        json.dumps(result.as_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    footprint_mask = Image.fromarray(
        np.where(result.coverage > 0, 255, 0).astype(np.uint8),
        mode="L",
    )
    footprint_mask.save(output / "registered_mosaic_footprint.png")
    if preview is not None:
        preview.save(output / "registered_mosaic_preview.png")
    return output


def render_mosaic_preview(
    image: np.ndarray,
    *,
    mode: str = "enhanced",
    max_side: int = 4096,
    transparent_outside: bool = True,
) -> Any:
    """把 ADU 大图映射为观察用 PNG；不改变 ``image`` 的定量数据。

    无 coverage 的 NaN 区域默认使用透明 alpha，而不是黑色填充。这样
    GUI/PNG 会显示真实 footprint 的形状；NumPy/ADU 数组仍保持矩形存储，
    便于后续计算和复现。
    """

    from PIL import Image, ImageFilter

    values = np.asarray(image, dtype=np.float32)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("mosaic image contains no finite pixel")
    # 只用于显示拉伸的统计量不需要扫描排序全部 4098² 个像素。固定步长
    # 抽取最多约 2^18 个样本，保持确定性，同时避免切换显示层时长时间卡住 UI。
    preview_sample_limit = 1 << 18
    if finite.size > preview_sample_limit:
        sample_step = int(np.ceil(finite.size / preview_sample_limit))
        finite = finite[::sample_step]
    low_percentile, q25, center, q75, high_percentile = np.percentile(
        finite,
        [1.0, 25.0, 50.0, 75.0, 99.5],
    )
    robust_noise = max((float(q75) - float(q25)) / 1.3489795, np.finfo(np.float32).eps)
    if mode == "raw":
        low = float(low_percentile)
        high = max(float(high_percentile), low + 1.0)
        gamma = 1.0
        denoise_radius = 0.0
    elif mode == "noise":
        low = max(float(low_percentile), float(center) - 2.0 * robust_noise)
        high = max(float(center) + 7.0 * robust_noise, low + 1.0)
        gamma = 0.82
        denoise_radius = 0.0
    else:
        low = max(float(low_percentile), float(center) - 0.5 * robust_noise)
        high = max(float(high_percentile), float(center) + 12.0 * robust_noise)
        gamma = 1.15
        denoise_radius = 0.6
    normalized = np.clip((values - low) / (high - low), 0.0, 1.0)
    normalized = np.power(normalized, gamma)
    normalized[~np.isfinite(normalized)] = 0.0
    rendered = Image.fromarray(np.rint(normalized * 255.0).astype(np.uint8), mode="L").convert("RGB")
    # 合成图本身已经经过多帧稳健融合；在 4096² 原图上再做 Pillow 全图
    # GaussianBlur 会阻塞 Tk 主线程。只有较小的预览图才保留轻度平滑。
    if denoise_radius > 0 and max_side < 2048:
        rendered = rendered.filter(ImageFilter.GaussianBlur(denoise_radius))
    if transparent_outside:
        alpha = Image.fromarray(np.where(np.isfinite(values), 255, 0).astype(np.uint8), mode="L")
        rendered = rendered.convert("RGBA")
        rendered.putalpha(alpha)
    rendered.thumbnail((max(64, int(max_side)), max(64, int(max_side))), Image.Resampling.LANCZOS)
    return rendered
