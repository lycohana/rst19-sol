"""当前比赛 FITS 文件的最小、可审计读取器。

项目数据为单个 4096x4096、16 bit 主图像，且图像第一行前 208 字节包含
比赛方定义的辅助字段。这里保留原始存储值，不自动应用 BSCALE/BZERO，
避免在尚未确认物理量定义前改变检测输入。
"""

from __future__ import annotations

import math
import re
import struct
from pathlib import Path
from typing import BinaryIO

import numpy as np

from .models import AuxiliaryData, FitsFrame

FITS_BLOCK_SIZE = 2880
AUXILIARY_BYTES = 208
AUXILIARY_VALUE_COUNT = 26


class FitsError(ValueError):
    """FITS 文件结构或项目数据约定不满足时抛出。"""


def _parse_card_value(raw_value: str) -> object:
    value = raw_value.split("/", 1)[0].strip()
    if not value:
        return None
    if value.startswith("'"):
        end = value.rfind("'")
        return value[1:end].replace("''", "'") if end > 0 else value[1:]
    if value in {"T", "F"}:
        return value == "T"
    numeric = value.replace("D", "E").replace("d", "e")
    try:
        if any(marker in numeric for marker in (".", "e", "E")):
            return float(numeric)
        return int(numeric)
    except ValueError:
        # 当前比赛文件的少数数值卡片在标准 20 字符值区中包含了两个
        # 连续数值，例如 `0000          01500`。保留标准解析优先级，
        # 对这种已知格式选择首个非零数值，避免把 EXPOSURE 读成字符串。
        tokens = re.findall(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?", numeric)
        if not tokens:
            return value
        selected = next((token for token in tokens if float(token) != 0.0), tokens[0])
        return float(selected) if any(marker in selected for marker in (".", "e", "E")) else int(selected)


def _read_header(stream: BinaryIO) -> tuple[dict[str, object], int]:
    cards: list[str] = []
    header_bytes = 0
    found_end = False

    while not found_end:
        block = stream.read(FITS_BLOCK_SIZE)
        if len(block) != FITS_BLOCK_SIZE:
            raise FitsError("FITS header is truncated before a complete 2880-byte block")
        header_bytes += len(block)
        decoded = block.decode("ascii", errors="replace")
        for offset in range(0, FITS_BLOCK_SIZE, 80):
            card = decoded[offset : offset + 80]
            cards.append(card)
            if card[:8].strip() == "END":
                found_end = True
                break

    header: dict[str, object] = {}
    for card in cards:
        key = card[:8].strip()
        if key == "END":
            break
        if key and card[8] == "=":
            header[key] = _parse_card_value(card[10:80])
    return header, header_bytes


def decode_auxiliary(raw: bytes) -> AuxiliaryData:
    """解码比赛格式的 208 字节辅助数据。

    比赛说明给出的编码方式是：每个 16 bit 存储单元先交换两个字节，
    再将整体按小端 26 个 double 解包。此函数不猜测字段含义，只做长度、
    有限性和字段数量检查。
    """

    if len(raw) != AUXILIARY_BYTES:
        raise FitsError(f"expected {AUXILIARY_BYTES} auxiliary bytes, got {len(raw)}")
    swapped = bytearray(raw)
    for index in range(0, len(swapped), 2):
        swapped[index], swapped[index + 1] = swapped[index + 1], swapped[index]
    values = struct.unpack("<26d", swapped)
    if not all(math.isfinite(value) for value in values):
        raise FitsError("auxiliary data contains NaN or infinity")
    return AuxiliaryData.from_values(values)


def auxiliary_mask(shape: tuple[int, int]) -> np.ndarray:
    """返回屏蔽图像第一行前 104 个辅助字段位置的布尔掩膜。"""

    if len(shape) != 2 or shape[0] < 1 or shape[1] < 1:
        raise ValueError(f"expected a non-empty 2-D image shape, got {shape}")
    mask = np.zeros(shape, dtype=bool)
    mask[0, : min(shape[1], AUXILIARY_BYTES // 2)] = True
    return mask


_BITPIX_DTYPES: dict[int, np.dtype] = {
    8: np.dtype(">u1"),
    16: np.dtype(">i2"),
    32: np.dtype(">i4"),
    64: np.dtype(">i8"),
    -32: np.dtype(">f4"),
    -64: np.dtype(">f8"),
}


def read_fits(path: str | Path, *, apply_scaling: bool = False) -> FitsFrame:
    """读取一个主 HDU 为二维图像的 FITS 文件。

    `apply_scaling=False` 返回 FITS 的原始存储值；若明确需要物理值，
    可以设置为 True，此时按头部 BSCALE/BZERO 应用线性缩放。
    """

    file_path = Path(path)
    if not file_path.is_file():
        raise FitsError(f"FITS file does not exist: {file_path}")

    with file_path.open("rb") as stream:
        header, data_offset = _read_header(stream)
        if header.get("SIMPLE") is not True:
            raise FitsError("primary HDU is missing SIMPLE=T")
        if header.get("NAXIS") != 2:
            raise FitsError(f"expected a 2-D primary image, got NAXIS={header.get('NAXIS')!r}")

        bitpix = header.get("BITPIX")
        if not isinstance(bitpix, int) or bitpix not in _BITPIX_DTYPES:
            raise FitsError(f"unsupported BITPIX={bitpix!r}")
        try:
            width = int(header["NAXIS1"])
            height = int(header["NAXIS2"])
        except (KeyError, TypeError, ValueError) as exc:
            raise FitsError("FITS header must contain integer NAXIS1 and NAXIS2") from exc
        if width <= 0 or height <= 0:
            raise FitsError(f"invalid image shape: {(height, width)}")

        dtype = _BITPIX_DTYPES[bitpix]
        expected_bytes = width * height * dtype.itemsize
        raw = stream.read(expected_bytes)
        if len(raw) != expected_bytes:
            raise FitsError(f"FITS image data is truncated: expected {expected_bytes}, got {len(raw)}")

    data = np.frombuffer(raw, dtype=dtype).reshape((height, width)).copy()
    if apply_scaling:
        bscale = float(header.get("BSCALE", 1.0))
        bzero = float(header.get("BZERO", 0.0))
        data = data.astype(np.float64) * bscale + bzero

    auxiliary = decode_auxiliary(raw[:AUXILIARY_BYTES]) if len(raw) >= AUXILIARY_BYTES else None
    return FitsFrame(
        path=file_path,
        header=header,
        data=data,
        auxiliary=auxiliary,
        data_offset=data_offset,
    )
