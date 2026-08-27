# rst19 实现模块

这是 rst19-sol 的第一版 Python 算法实现，负责把资料中的星表路线落成可测试的离线基线。

## 定位

模块当前覆盖：

- 当前比赛 FITS 主 HDU 的读取、结构校验和 26 个辅助字段解码；
- 屏蔽首行辅助区域后的鲁棒背景/噪声估计；
- 局部峰值星点候选检测、简单通量、质心、SNR 和形状摘要；
- 基于切平面先验 WCS 的星表一对一匹配；
- CSV 离线任务星表读取、自行传播和 JSON CLI 输出。

模块当前不声称已经完成：

- 完全盲的 plate solving 或 Astrometry.net 索引生成；
- 经过真实标定的绝对星等、`Mv` 转换和检测完备率；
- 15 帧运动目标轨迹关联和比赛最终真值验证。

## 安装

在仓库根目录执行：

```powershell
python -m pip install -e ".[dev]"
```

核心依赖是 NumPy 和 SciPy。当前 FITS 读取器不要求联网或 Astropy，后续若使用标准 WCS/SIP 工具可以再增加可选依赖。

## 快速使用

只检测一帧：

```powershell
rst19 doc/00-项目资料/原始数据/20260330163205413_9901.fits `
  --threshold-sigma 5 `
  --min-distance 3 `
  --max-sources 20000 `
  --json-out output/frame-01.json
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

## 测试

```powershell
python -m pytest
```

## 相关资料

- [星表路线深度研究](../doc/02-星图识别/星表路线深度研究.md)
- [FITS 读取与校验](../doc/01-数据解析/FITS读取与校验.md)
- [星点识别方法](../doc/02-星图识别/星点识别方法.md)
