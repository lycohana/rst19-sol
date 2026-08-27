# FITS 读取与校验

状态：**方法调研；当前文件已完成基础抽检**

## 1. 为什么先处理 FITS

FITS 不是普通图片格式，而是由 Header/Data Units 组成的科学数据格式。NASA 的 FITS Primer 说明，每个 HDU 的头和数据单元按 2880 字节块组织，头部由固定长度的 80 字节关键字卡片组成；`BITPIX` 和 `NAXIS*` 决定图像数据类型与尺寸。

当前数据的算法结果会同时依赖像素数组和头部/首行辅助信息。如果把 FITS 的大端 16 bit 数据、`NAXIS1/NAXIS2` 轴顺序或首行辅助数据处理错，后续星点计数、通量和运动速度都会失真。

## 2. 当前文件的正确读取假设

根据本地核验和比赛方格式说明，当前第一版实现可以采用以下假设，但每次读取都应通过校验而不是静默相信：

- 主 HDU 是二维图像，`NAXIS=2`。
- `BITPIX=16`，对应 16 bit 有符号整型存储；FITS 图像数组在 NumPy 中通常表现为大端 `int16`。
- `NAXIS1=4096` 是横向像素数，`NAXIS2=4096` 是纵向像素数；NumPy 数组形状应为 `(4096, 4096)`，访问顺序是 `[y, x]`。
- FITS 头部占前 2880 字节，图像数据随后开始。
- 图像数据第一行前 208 字节由格式说明保留给 26 个 `double` 辅助字段，即前 104 个 16 bit 位置不能直接计入普通图像像素。
- 当前样本文件没有证明存在可直接使用的 WCS；头部的 `ra`、`dec` 和姿态信息不能单独替代像素到天球坐标的完整映射。

Astropy 文档还提醒，若存在 `BSCALE/BZERO`，读取 `.data` 可能触发物理值缩放；因此应显式检查缩放字段，并在需要保留原始存储值时谨慎使用 `do_not_scale_image_data=True`。

## 3. 推荐读取流程

### 3.1 头部和数组

后续代码可以以 Astropy 为主读取器，先检查结构，再访问数据。示例只展示流程，不代表最终接口：

```python
from pathlib import Path

import numpy as np
from astropy.io import fits


def read_image(path: str | Path):
    path = Path(path)
    with fits.open(path, memmap=True, do_not_scale_image_data=True) as hdul:
        hdul.verify("exception")
        hdu = hdul[0]
        header = hdu.header.copy()
        data = np.asarray(hdu.data)

        if data.ndim != 2:
            raise ValueError(f"expected 2-D primary image, got {data.shape}")
        if data.shape != (header["NAXIS2"], header["NAXIS1"]):
            raise ValueError("array shape does not match NAXIS1/NAXIS2")
        if header["BITPIX"] != 16:
            raise ValueError(f"unexpected BITPIX={header['BITPIX']}")

        return header, data
```

Astropy 的 `memmap` 适合逐帧或分块访问大型本地图像；不要一次把 15 张图像复制成 `float64` 数组。单张原始图像约 32 MiB，15 张约 480 MiB；若做差分、背景图和临时浮点数组，峰值内存会更高。

### 3.2 文件级校验

每个文件至少检查：

1. 文件名和扩展名是否符合数据集约定。
2. `SIMPLE`、`NAXIS`、`BITPIX`、`NAXIS1`、`NAXIS2` 是否存在且类型正确。
3. 文件长度是否符合 `header_blocks × 2880 + image_bytes`。
4. `DATE-OBS` 是否能解析为 UTC，且按时间升序排列。
5. `EXPOSURE` 是否为正，并统一单位为秒或毫秒，避免混用。
6. `BSCALE`、`BZERO`、NaN/Inf、负值和接近极值的像素是否需要单独标记。
7. 首行 208 字节辅助数据是否能解包成 26 个有限 `double`。
8. `p_az/p_el` 是否与 `AZIMUTH/ELEVATIO` 一致，四元数范数是否接近 1，位置/速度在相邻帧之间是否连续。

### 3.3 辅助数据解码

比赛方给出的 Python 方式是取图像数据起始处的 208 字节，将 16 bit 字节交换后以小端格式解包为 26 个 `double`。建议实现时保留原始 208 字节和解码后的结构化记录，便于出现异常时追溯：

```python
import struct


def decode_auxiliary(raw_image_bytes: bytes) -> tuple[float, ...]:
    first_row_aux = raw_image_bytes[:208]
    swapped = bytearray(first_row_aux)
    for i in range(0, len(swapped), 2):
        swapped[i], swapped[i + 1] = swapped[i + 1], swapped[i]
    return struct.unpack("<26d", swapped)
```

这段逻辑必须用 15 个文件逐一验证，并与 C++ 版本和头部字段交叉比对。不要在没有数值范围检查的情况下把任意 208 字节解释成姿态或轨道数据。

## 4. 图像值处理原则

- 先保存原始整型数组或只读映射，再生成用于算法的浮点视图。
- 把首行前 104 个位置做成显式 mask；是否需要把整行排除，要用比赛格式和可视化结果进一步确认。
- 先统计每帧的 min/max、分位数、背景中位数、MAD/RMS、饱和率和坏值比例，再决定显示拉伸和检测阈值。
- 不要把原始像素最小值直接当成最暗星；负值可能是背景扣除、字节序或异常像素的结果。
- 所有裁剪、归一化和背景扣除都要记录参数，保证比赛结果可复现。

## 5. 参考资料与方法边界

- [NASA FITS Primer](https://fits.gsfc.nasa.gov/fits_primer.html)：FITS 的 HDU、2880 字节块、80 字节头卡片和图像数据类型。
- [FITS Standard Page](https://fits.gsfc.nasa.gov/fits_standard.html)：FITS 官方标准入口。
- [Astropy FITS File Handling](https://docs.astropy.org/en/stable/io/fits/)：HDU、头部、验证和读写接口。
- [Astropy Image Data](https://docs.astropy.org/en/stable/io/fits/usage/image.html)：`BITPIX` 类型映射、数组轴顺序、缩放和分块访问。

这些资料能说明格式和读取工具的行为，不能证明当前比赛文件的业务约定；业务约定仍以本地核验和比赛方原始说明为准。
