# 来源与证据说明

编写和核对日期：2026-10-10。正文实验对象为原始 15 帧比赛数据；补充 16 帧 PIC/POS 单独讨论，不混入主统计。

## 本地直接核对

首帧缓存：`.rst19-cache/fb1332214b8c0c8664c02c55e6993a5bfeac8c83b5a7e378c0bf9bd8f53e62ab.json.gz`。

序列缓存：`.rst19-cache/3cbeaf34b7381421ab13d688048b6eaa1a084da0c91cf0862800759e9e52a955.sequence.json.gz`。

合成数组：`.rst19-cache/2c5cab1b6609450e5a2d4eaf9d61cad7148500fb7b333fa4d6ef177d1c472455.mosaic.npz`。

完整文件哈希在 `evidence.json`；原始输入哈希在 `input_manifest.csv`。原图裁剪仅展示数据，没有修改 FITS。图中灰度拉伸与定量值分开。

## 项目文档与代码

| 材料 | 论文使用方式 | 验证层级 |
| --- | --- | --- |
| `doc/00-项目资料/天基图像数据格式说明.docx` | 辅助区域、UTC 曝光中心与曝光单位 | 读取原始 XML 文本；未重新排版文档 |
| `doc/00-项目资料/赛题与数据总览.md` | 数据背景和设备边界 | 文档事实，与 FITS／缓存交叉核对 |
| `doc/00-项目资料/开运一号相机参数与尺度先验.md` | 像元、焦距、视场尺度先验 | 外部设备资料；非实际 WCS 标定 |
| `doc/05-论文答辩/竞赛论文核心短稿.md` | 分层方法和双响应案例 | 方法重写；部分数值有现存缓存，模型敏感性为历史记录 |
| `doc/05-论文答辩/论文与答辩材料.md` | 方法、误差和限制 | 从扩展底稿选择，不把不同版本合并 |
| `doc/05-论文答辩/report-source.md` | 数值与外部文献追溯 | 来源账本不独立证明实验正确性 |
| `doc/04-创新分析/创新交付审计.md` | 三布局条件化注入，3 条件 × 3 强度 × 24 个源 | 当前缺少原机读产物，明确为历史记录；本轮仅重算区间 |
| `doc/04-创新分析/15帧关系与证据说明.md` | 时间关系与历史版本差异 | 与现存缓存不同的结果不放入主表 |
| `doc/04-创新分析/15帧注册合成大图.md` | ADU／coverage／scatter 方法 | 当前数字采用现存 NPZ，旧位移不混用 |
| `doc/00-项目资料/定位数据测试checkpoint-20260913.md` | 补充 PIC/POS 无法配对的限制 | 文档记录，不作为主数据真值 |
| `src/rst19/detection.py` | 净通量误差、PSF 与局部模型定义 | 代码事实；增益未知时缺少完整 Poisson 项 |
| `src/rst19/experiments.py` | 首要类别优先级 | 生成脚本按当前函数规则汇总缓存 |
| `src/rst19/cache.py` | 缓存版本 21／71 | 当前字段一致性；不等于本轮算法重跑 |

## 外部参考资料

参考文献采用原始论文或官方资料。检索日期 2026-10-10；正文只借用各方法的条件和职责，不转用其他数据集性能。

- Bertin & Arnouts (1996), SExtractor：[期刊 PDF](https://aas.aanda.org/articles/aas/pdf/1996/08/ds1060.pdf)。当前直接打开返回 403，搜索可返回原期刊摘要；条目信息另与已有研究记录一致。
- Stetson (1987), DAOPHOT：[原始论文](https://articles.adsabs.harvard.edu/pdf/1987PASP...99..191S)。原始 PDF 可访问。
- Lang & Hogg (2020), Principled point-source detection：[arXiv](https://arxiv.org/abs/2012.15836)。核对作者、题名与匹配滤波最优性前提。
- Vio & Andreani (2016)：[arXiv](https://arxiv.org/abs/1602.02392)。核对题名、作者、A&A 589 A20 和未知位置峰搜索的误警问题。
- Hedges et al. (2021), Linearized Field Deblending：[arXiv](https://arxiv.org/abs/2106.08411)。核对 PSF 去混叠与源间退化条件。
- Lang et al. (2010), Astrometry.net：[arXiv](https://arxiv.org/abs/0910.2233)。核对 WCS／几何模式匹配职责。
- Nir et al. (2018), Streak Detection：[arXiv](https://arxiv.org/abs/1806.04204)。核对线状模板与不同线形污染类型。
- Zackay et al. (2016), Proper image subtraction：[arXiv](https://arxiv.org/abs/1601.02655)。正文未宣称项目实现为最优差分算法。
- FITS Working Group：[标准入口](https://fits.gsfc.nasa.gov/fits_standard.html)。2016 年通过、2018 年语言修订版；本文引用后者。
- ESA/Gaia/DPAC：[DR3 passbands](https://www.cosmos.esa.int/web/gaia/dr3-passbands)。支持 G/BP/RP 系统区别；不把 Gaia 响应当作本相机响应。
- Chiu, Desai & Liu (2016), ComEst：[arXiv](https://arxiv.org/abs/1605.02650)。核对真实背景注入的用途；本文未移植该软件或声称同等性能。

## 缺失与保留

历史注入原布局、模板与控制文件没有出现在当前 `tmp` 中。本文不补造原产物、不虚构实验运行信息，也不将文档日期等同于当前缓存生成日期。现存缓存没有提供完整设备标定与逐星真值，因此相关字段和投稿前任务显式保留。
