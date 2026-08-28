# rst19 实现模块

这是 rst19-sol 的第一版 Python 算法实现，负责把资料中的星表路线落成可测试的本地 Python 基线。

## 定位

模块当前覆盖：

- 当前比赛 FITS 主 HDU 的读取、结构校验和 26 个辅助字段解码；
- 屏蔽首行辅助区域后的鲁棒背景/噪声估计；
- 局部背景/RMS、Gaussian PSF 匹配滤波、全量候选审计、孔径通量误差、三种 SNR 和点源质量分层；
- 基于切平面先验 WCS 的星表一对一匹配；
- CSV 离线任务星表读取、自行传播和 JSON CLI 输出。
- Python Tkinter 桌面工作台：选择 FITS、调整参数，在“可信星点 / 全部候选 / 运动候选”图层间切换，并查看最暗源标注。
- 15 帧质量源的全局平移配准、唯一轨迹关联和 `static`/`moving`/`transient` 分类基线。

模块当前不声称已经完成：

- 完全盲的 plate solving 或 Astrometry.net 索引生成；
- 经过真实标定的绝对星等、`Mv` 转换和检测完备率；当前只输出仪器星等 `m_inst`；
- 官方逐星/运动目标真值、注入完备率和误检率验证；当前 `quality_count` 是明确规则下的可信点源数，不自动等于物理恒星真值。

## 安装

在仓库根目录执行：

```powershell
python -m pip install -e ".[dev]"
```

核心依赖是 NumPy、SciPy 和 Pillow（用于本地预览 PNG）。当前 FITS 读取器不要求联网或 Astropy，后续若使用标准 WCS/SIP 工具可以再增加可选依赖。

## 快速使用

只检测一帧：

```powershell
rst19 doc/00-项目资料/原始数据/20260330163205413_9901.fits `
  --threshold-sigma 4 `
  --min-distance 3 `
  --json-out output/frame-01.json
```

检测默认在 Gaussian PSF 匹配响应上采用 4σ 候选阈值并优先保留候选；不传 `--max-sources` 就不截断候选源。`--max-sources` 仍可作为显式的性能/导出限制，但带有限制时最暗源只会在返回的检测源中选择。结果同时保留 `candidate_count`、`returned_count`、`quality_count` 和拒绝标志，不把外部示例数量写入算法。

分析结果中的 `faintest_detected` 是通过局部通量 SNR、正通量、点源形状以及边缘/掩膜/饱和质量筛选后的最暗可信候选源。其 `instrumental_magnitude` 按 `m_inst = -2.5 log10(flux_rate)` 计算；只有提供经过验证的 `--zero-point` 时才会附带 `calibrated_magnitude`，不能在未标定时直接称为 Gaia V 或 `Mv`。`snr` 是峰值 SNR，`flux_snr` 是孔径通量 SNR，`filter_snr` 是匹配滤波 SNR，三者语义不同。

分析 15 帧：

```powershell
rst19-sequence doc/00-项目资料/原始数据 `
  --threshold-sigma 4 --min-distance 3 --aperture-radius 4 `
  --psf-fwhm 3 --min-flux-snr 5 --min-presence 12 `
  --motion-min-displacement-px 2 --max-motion-fit-rms-px 0.75
```

使用离线 CSV 星表进行先验匹配：

```powershell
rst19 frame.fits `
  --catalog catalog.csv `
  --center-ra 129.533548 `
  --center-dec -1.845372 `
  --pixel-scale-arcsec 10 `
  --rotation-deg 0 `
  --parity 1
```

CSV 至少需要以下列：

```text
source_id,ra_deg,dec_deg,magnitude,pmra,pmdec,ref_epoch
```

`--pixel-scale-arcsec`、旋转角和 parity 是相机参数假设，不是当前数据已经核验的事实；应通过稳定匹配星、残差和留出星验证后再固定。

启动 Python 桌面界面：

```powershell
rst19-gui
```

界面默认读取 `doc/00-项目资料/原始数据/`，只在本机处理；可以通过 `--data-dir` 指定其他获授权的 FITS 目录。最大源数输入框留空表示全量检测，绿色环表示最暗可信源。检测默认会把跨越多个 PSF 宽度的长线标记为 `LINE_ARTIFACT`：它们仍保留在“全部候选”审计层，但不会进入可信星点和运动轨迹。

界面启动和切换帧时只载入图像预览，不会提前计算检测结果。点击“分析当前帧”后，结果会按 FITS 文件路径、文件修改状态和检测参数写入项目根目录的 `.rst19-cache/` 压缩缓存；相同输入再次分析时直接复用。点击“分析 15 帧动目标”后，程序对整个目录做平移配准和轨迹关联，完成后自动切换到“运动候选”层；该层只绘制分类为 `moving` 的当前帧轨迹点，不把整幅质量星场误画成动目标。切换帧时右上角的图层选择会保持不变，旧线程结果会因帧令牌失效而丢弃，过期任务也不会把旧结果写入当前显示状态。点击“清空检测缓存”会删除当前 gzip 缓存和旧版本 JSON 产物，即使任务正在运行也可以执行；本次任务完成后不会把清理前的结果写回缓存，原始 FITS 不会被删除。

预览支持鼠标滚轮缩放和左键拖拽平移。在“可信星点 / 全部候选”层将鼠标移到候选点附近时，底部信息栏和图像标注会显示检测 ID、坐标、峰值、通量、通量误差、三类 SNR、FWHM/椭圆率、仪器星等和质量标记；在“运动候选”层悬停蓝色环会显示轨迹 ID、当前帧坐标、出现帧数、总位移、速度和拟合 RMS。底部信息栏宽度固定跟随 viewer，长文本自动换行，不会把右侧结果面板挤出窗口。

## 测试

```powershell
python -m pytest
```

## 相关资料

- [星表路线深度研究](../doc/02-星图识别/星表路线深度研究.md)
- [检测器实验记录](../doc/02-星图识别/检测器实验记录.md)
- [FITS 读取与校验](../doc/01-数据解析/FITS读取与校验.md)
- [星点识别方法](../doc/02-星图识别/星点识别方法.md)
- [论文与答辩材料](../doc/05-论文答辩/论文与答辩材料.md)
