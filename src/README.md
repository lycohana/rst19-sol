# rst19 实现模块

这是 rst19-sol 的第一版 Python 算法实现，负责把资料中的星表路线落成可测试的本地 Python 基线。

## 定位

模块当前覆盖：

- 当前比赛 FITS 主 HDU 的读取、结构校验和 26 个辅助字段解码；
- 屏蔽首行辅助区域后的鲁棒背景/噪声估计；
- 局部背景/RMS、Gaussian PSF 匹配滤波、3×3 PSF 支持反尖峰审计、全量候选审计、孔径通量误差、三种 SNR 和点源质量分层；
- 基于切平面先验 WCS 的全局/贪心可对照星表一对一匹配；
- CSV 离线任务星表读取、自行传播和 JSON CLI 输出。
- Python Tkinter 桌面工作台：选择 FITS、调整参数，在“可信星点 / 全部候选 / 运动候选”图层间切换，并查看最暗源的 `m_inst`、标定表观星等和绝对星等质量状态。
- 15 帧质量源的全局平移配准、唯一轨迹关联和 `static`/`moving`/`transient` 点轨迹基线；另有独立的线状候选检测与跨帧关联。
- 单帧长线候选检测和 `rst19-innovation` 证据导出：按真实 `DATE-OBS` 生成逐帧/逐轨迹 JSON、CSV 和 PNG，区分单帧候选与跨帧 `moving`。
- `rst19-innovation-package` 条件化创新交付包：从真实背景分层注入表生成候选/质量召回、质量筛选损失、Wilson 区间和可答辩边界，不读取 FITS、不把背景检测数当误检率。

模块当前不声称已经完成：

- 完全盲的 plate solving 或 Astrometry.net 索引生成；
- 本设备实验室响应曲线、官方逐星真值和检测完备率；代码已实现星表驱动的 `m_inst → m_cal → M` 结果链，但实际绝对星等仍取决于用户提供的目录、波段转换、视差质量和消光资料；
- 官方逐星/运动目标真值、基于逐星标签的 precision/误检率验证；当前 `quality_count` 是明确规则下的可信点源数，不自动等于物理恒星真值。真实首帧已完成 Gaussian 和修正后全局经验 PSF 的首轮背景注入，以及边缘/拥挤/特殊值域的分层机制审计，但尚未完成空间变 PSF 和大样本分层完备率。

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
  --threshold-sigma 4 --min-distance 4 `
  --psf-fwhm 3 --min-flux-snr 5 --min-psf-support-pixels 3 `
  --json-out output/frame-01.json
```

核心/CLI 检测默认在 Gaussian PSF 匹配响应上采用 4σ 候选阈值并优先保留候选；不传 `--max-sources` 就不截断候选源。GUI 入口另采用已经过真实 FITS 对照的 `hybrid`（Gaussian + 双尺度 DoG）平衡宽筛和 `FWHM=2 px`，不改变 CLI 的可复现实验默认。`--max-sources` 仍可作为显式的性能/导出限制，但带有限制时最暗源只会在返回的检测源中选择。结果同时保留 `candidate_count`、`returned_count`、`quality_count` 和拒绝标志，不把外部示例数量写入算法。

分析结果中的 `faintest_detected` 是通过局部通量 SNR、正通量、点源形状以及边缘/掩膜/饱和质量筛选后的最暗可信候选源。其 `instrumental_magnitude` 按 `m_inst = -2.5 log10(flux_rate)` 计算；提供星表并启用 `--fit-photometry` 后，`source_photometry` 会逐源输出 `calibrated_magnitude`、误差、光度系统/波段和 `absolute_magnitude` 质量状态。没有匹配、颜色或可靠视差时数值保持 `null`，不把它直接称为 Gaia V 或 `Mv`。`snr` 是峰值 SNR，`flux_snr` 是孔径通量 SNR，`filter_snr` 是匹配滤波 SNR，三者语义不同。

需要公共参考目录时，显式运行 `python -m rst19.gaia_remote_cli --ra ... --dec ... --radius ... --out ...` 获取 Gaia DR3 子表；默认单帧/15 帧分析不联网，GUI 只有用户明确点击“在线获取 Gaia DR3”时才发起网络查询。Gaia 行中的 `phot_g_mean_flux_over_error`、`ruwe`、`duplicated_source`、`visibility_periods_used` 和 `phot_variable_flag` 会进入 `CatalogSource`，光度拟合默认排除明确重复/变量、低 G 通量 SNR、高 RUWE 和过少 visibility periods 的参考星，并把排除原因写入 `catalog_filter_counts`。15 帧相对光度可用 `python -m rst19.relative_photometry_cli` 或 `rst19-sequence --relative-photometry`，输出的帧零点和源亮度是相对量。`rst19-photometric-report` 用于审计 JSON 中的仪器/相对/表观/绝对星等证据层级。

GUI 的“在线获取 Gaia DR3”和“自动板解 + 测光”会默认额外保留 GSP-Phot 的
`M_G` 中位数及 16/84 百分位区间。界面将其显示为“Gaia模型 M_G”，并与
本地图像重算的 `M` 分开；这组目录模型值只作外部绝对星等证据，不会改变最暗源的
表观星等排序。

分析 15 帧：

```powershell
rst19-sequence doc/00-项目资料/原始数据 `
  --threshold-sigma 4 --min-distance 4 --aperture-radius 4 `
  --psf-fwhm 3 --min-flux-snr 5 --min-psf-support-pixels 3 --min-presence 12 `
  --motion-min-displacement-px 2 --max-motion-fit-rms-px 0.75
```

序列默认使用有界工作集：每帧保留完整 `candidate_count`，但按匹配滤波 `filter_snr` 取前 6000 个源进行源级测量和点轨迹关联，并用 `background_box_size=256`、共享局部背景/RMS 图减少重复统计。序列网格背景使用两轮稳健裁剪，单图仍使用四轮裁剪与环形局部精修；背景网格的线性插值直接输出目标尺寸，并缓存重复孔径几何，避免 4096² 临时数组和数千次小数组分配。对当前 16 位 FITS，序列还启用 `use_float32` 像素中间阵列；它能精确表示输入 ADU，只降低 4096² 阵列的内存带宽，单图科学测光仍默认使用 float64。默认使用 4 个 worker 并行检测不同帧，结果按原始帧序恢复；`--workers 1/2/3` 可降低峰值内存，`--workers 4` 是当前机器的吞吐档位。`returned_count`、`quality_count` 和点轨迹数量是工作集口径；单图检测仍默认不截断且保留精细局部背景。需要全量序列工作集时传 `--source-limit 0`，需要对照实验时可传其他正整数。序列还提供 `static/persistent/transient` 分层和原图时序小窗审计，持续候选默认门槛为 `8/15`，严格静态默认门槛为 `12/15`；这两个序列计数都不是官方逐星真值。

在已有序列配准结果上生成 15 帧注册合成大图：

```powershell
rst19-mosaic doc/00-项目资料/原始数据 `
  --sequence-json tmp/sequence-current-fast-verified.json `
  --out-dir tmp/registered-mosaic `
  --mode robust_mean --clip-sigma 3 --tile-rows 256
```

`rst19-mosaic` 不会把 15 张图直接平均，也不会在没有 `cumulative_shifts` 时伪造配准；它读取原始 ADU，按共同 detector 坐标的联合 footprint 重采样，重叠区执行 MAD 裁剪稳健融合，并额外输出 `coverage` 覆盖数和 `scatter` 帧间离散度。`registered_mosaic_adu.npy` 是定量数组，`registered_mosaic_preview.png` 是带透明无覆盖区的观察图，`registered_mosaic_footprint.png` 是独立覆盖掩膜；因此它与“叠加暗星”（弱源恢复层）和“稳定星场”（跨帧点源层）是不同证据层。当前真实 15 帧回归输出 `4098×4098`，最大覆盖 `15`、平均覆盖约 `14.989`；当前实现只做平移注册，不宣称完成旋转、畸变或完整 WCS 镶嵌。详细口径见 [15 帧注册合成大图](../doc/04-创新分析/15帧注册合成大图.md)。
序列默认使用有界工作集：每帧保留完整 `candidate_count`，但按匹配滤波 `filter_snr` 取前 6000 个源进行源级测量和点轨迹关联，并用 `background_box_size=256`、共享局部背景/RMS 图减少重复统计。序列网格背景使用两轮稳健裁剪，单图仍使用四轮裁剪与环形局部精修；背景网格的线性插值直接输出目标尺寸，并缓存重复孔径几何，避免 4096² 临时数组和数千次小数组分配。对当前 16 位 FITS，序列还启用 `use_float32` 像素中间阵列；它能精确表示输入 ADU，只降低 4096² 阵列的内存带宽，单图科学测光仍默认使用 float64。默认使用 4 个 worker 并行检测不同帧，结果按原始帧序恢复；`--workers 1/2/3` 可降低峰值内存，`--workers 4` 是当前机器的吞吐档位。`returned_count`、`quality_count` 和点轨迹数量是工作集口径；单图检测仍默认不截断且保留精细局部背景。需要全量序列工作集时传 `--source-limit 0`，需要对照实验时可传其他正整数。序列还提供 `static/persistent/transient` 分层和原图时序小窗审计，持续候选默认门槛为 `8/15`，严格静态默认门槛为 `12/15`；这两个序列计数都不是官方逐星真值。

### GUI 默认分析口径（2026-09-02）

`rst19-gui`/`rst19-ui` 的单帧和 15 帧按钮共用当前默认：`threshold=4σ`、`min_distance=4 px`、`PSF FWHM=2 px`、通量 `SNR≥5`、`proposal_mode=hybrid`、15 帧弱星提案为 `median`、参考图 `SNR≥15`、普通候选共识 `15σ`、逐帧补测 `7.5σ`、时序 PSF 相关 `≥0.80`，并在预测点 `±1 px` 内做局部峰重定位。时序多尺度、局部去混叠和 15 帧全量关联默认关闭；它们仍是可显式开启的审计/性能选项。该默认偏向在可控耗时内减少固定 `FWHM=3 px` 和单一 Gaussian 对拥挤/失配源的漏检，但 `persistent`/`quality` 仍是质量定义下的候选层，不等同官方恒星真值。真实对照数字和选择理由见 [检测器实验记录](../doc/02-星图识别/检测器实验记录.md) 的 2026-09-02 小节。

单帧质量层默认要求候选峰中心 3×3 核心至少 3 个像素超过 `max(2×局部噪声, 0.1×峰值超额)`；不满足时仍保留候选并记录 `INSUFFICIENT_PSF_SUPPORT`，避免把孤立单像素脉冲当成点源。该约束不改变原始 ADU、通量或星等，只影响质量层。当前快速 15 帧序列还对时间补提案在预测点 `±1 px` 内做局部峰重定位，再执行 PSF 相关和跨帧细筛；同一真实数据的最新口径为严格静态 `5,276`、持续候选 `15,524`、稳定/持续合计 `20,800`。早期全量工作集的 `9,938 / 6,249 / 16,187` 仍保留在实验记录中作为历史对照，不能与当前结果混写；持续候选仍需星表/人工复核。

当前质量层还会审计有符号整型 FITS 中的“候选峰本身落在全幅异常重复高位码上 + 孔径含异常负值”模式，记录为 `CODE_PATTERN`。它只把受污染候选降级，不删除宽筛候选；仅“孔径内含重复码”不足以拒绝，否则真实亮星的饱和/溢出出血列会把 `3991/3992` 等重复码和 `-20628` 这类负值同时带进孔径而被误拒。近邻联合门控修补后的首帧 `hybrid + FWHM=2` 为 `84,594` 个候选、`29,260` 个质量源、`12` 个 `CODE_PATTERN`；`29,263` 是修补前质量基线。这个规则不是相机满阱、`BLANK` 或坏像素的替代标定；详细局部复核见 [检测器实验记录 8.49](../doc/02-星图识别/检测器实验记录.md) 和 [真实性研究 10.32](../doc/02-星图识别/亮点真实性与伪影判别研究.md)。

15 帧序列另有一个独立的**叠加暗星恢复层**（`evidence_level="stack_faint"`）：它先把注册后的 15 帧取时间中值（或稳健均值）得到降噪参考图，在参考图自身噪声上以 `4σ`、`flux_snr≥5` 重新检测，再回到每帧原始图做 `±1 px` 局部峰强制测光、3×3 支持、形状和边缘/掩膜审计，要求达到 `persistent` 出现帧数且逐帧放宽 `flux_snr≥3`。这样能把单帧 `flux_snr<5` 但跨帧持久的真实暗星恢复出来，同时不给单帧质量层灌进噪声。结果单独计为 `stack_faint_count`，GUI 提供“叠加暗星 · 待复核”图层，`rst19-sequence` 默认开启，可用 `--no-stack-faint` 关闭。该层仍是低置信候选，不是官方逐星真值；方法依据与对照见 [检测器实验记录 8.42](../doc/02-星图识别/检测器实验记录.md)。

从序列 JSON 导出创新分析证据：

```powershell
rst19-innovation tmp/sequence-final.json --out-dir tmp/innovation

# 从已有真实背景分层注入表生成创新交付包
rst19-innovation-package tmp/stratified-real-background-code-pattern-current-n8/stratified_real_background_injection.json `
  --out-dir $env:TEMP/rst19-innovation-package
```

导出结果包括逐帧背景稳健统计、候选/质量计数、累计平移、辅助姿态字段，以及逐条线状轨迹的出现帧、持续时间、X/Y 速度、速度/方向 95% 区间、长度、SNR、拟合 RMS 和短期外推逐轴区间；同时输出逐帧线状点表、原图裁剪和接触表，便于人工复核。三点及以上轨迹以真实 `DATE-OBS` 做常速度 OLS，并使用小样本 Student-t 与 delta method；两点轨迹不伪造区间，单帧候选不伪造速度。没有标准 WCS 和像元角尺度时，报告只保留像素平面量。GUI 的证据窗口另有“创新摘要”“图证”“星点实验”和“单图长线”页：“创新摘要”集中显示时间、配准、运动量和预测边界，“图证”按需并列生成注册中值拼图、正/负残差差分图、线状候选裁剪接触表和论文式跨帧轨迹图；“图证”还可以按需打开星点时序审计，查看严格静态、持续候选和瞬态轨迹的原图小窗。星点实验展示候选层与质量层的召回曲线及真实背景辅助统计，单图长线页则逐帧显示独立长线候选的数量、几何属性和两张回归曲线。

对真实首帧做阈值敏感性扫描：

```powershell
rst19-sweep doc/00-项目资料/原始数据/20260330163205413_9901.fits `
  --thresholds 4 5 6 8 10 12 15 20 --min-distance 4 `
  --psf-fwhm 3 --background-box-size 128 --min-flux-snr 5 `
  --out-dir tmp/detection-sweep
```

命令会输出 `detection_threshold_sweep.csv` 和 `detection_threshold_sweep.png`，记录候选峰、返回数、质量通过数、拒绝数、线状审计数和噪声基线；GUI 的“阈值扫描”页使用同一实验逻辑。该实验用于解释参数敏感性，不把某个阈值下的数量直接当作真实恒星数。

固定候选阈值、扫描匹配滤波 PSF 尺度：

```powershell
rst19-sweep doc/00-项目资料/原始数据/20260330163205413_9901.fits `
  --psf-fwhms 1.5 2 2.5 3 4 --thresholds 4 --min-distance 4 `
  --psf-fwhm 3 --background-box-size 128 --min-flux-snr 5 `
  --out-dir tmp/detection-psf-sweep
```

输出 `detection_psf_sweep.csv` 和 `detection_psf_sweep.png`；扫描只改变 Gaussian 匹配滤波尺度。不能只按候选数量选择 FWHM，应与实测 PSF、真实背景注入召回率和人工抽检一起判断。

在真实首帧背景上做注入-回收：

rst19-injection --real-fits doc/00-项目资料/原始数据/20260330163205413_9901.fits --out-dir tmp/real-background-final --trials 2 --sources 16

该模式只在内存副本上叠加 Gaussian PSF，并显式继承原始 int16 基线的边界/零值掩膜；输出候选/质量召回率和背景检测辅助量。没有逐星标签时，不能据此报告 precision 或误检率。

真实背景分层注入审计：

```powershell
rst19-injection --real-fits doc/00-项目资料/原始数据/20260330163205413_9901.fits `
  --stratified-real `
  --peak-levels 24 56 128 --trials 1 --sources 8 --psf-fwhm 2 `
  --proposal-mode hybrid --psf-model empirical `
  --strata blank high_background edge crowded special_code line `
  --out-dir tmp/stratified-real-background-code-pattern-current-n8
```

该模式把低纹理/高局部噪声、边缘、拥挤邻域和特殊值域分开，记录候选层召回、质量层召回、局部背景/RMS 和基线邻源歧义；示例使用每格 8 个位置，便于比较条件趋势。`crowded`、`special_code` 和可选的 `line` 是困难条件控制，不是逐星真值；`line` 建议单独输出到独立目录，避免与普通星点分层混报。完整说明见 [检测器实验记录](../doc/02-星图识别/检测器实验记录.md) 的真实背景分层注入章节。

近邻双峰的强弱比控制：

```powershell
rst19-injection --pair-flux-ratio-audit --real-fits doc/00-项目资料/原始数据/20260330163205413_9901.fits `
  --total-peak-levels 256 --secondary-ratios 0.5 0.25 0.125 `
  --trials 1 --pairs 8 --psf-fwhm 2 --min-distance 4 `
  --min-psf-support-pixels 3 --out-dir tmp/pair-flux-ratio-256-gui-default-n8
```

该实验固定 `primary+secondary` 总峰值，在同一真实背景和同一位置上改变 `secondary/primary`，同时报告单源候选/质量召回、成对候选/质量解析率和中间峰合并比例。它专门用于区分“两个局部峰”与“两个独立点源”，不替代星表身份、实测 PSF 或正式完备率；结果见 [检测器实验记录](../doc/02-星图识别/检测器实验记录.md) 8.31 节。

扩展实验把总峰值扫描为 `256/512/1024 ADU`，Gaussian 在 `r=0.125` 下的成对质量解析率为 `0/0.25/1.0`；经验 PSF 在 `512 ADU`、`r=0.25/0.125` 下为 `0.25/0`。经验核对照同时记录 20 个模板源、中位 FWHM `2.497 px` 和核和 `3.647`，用于 PSF 形状敏感性审计，不直接替代 GUI 默认或星表身份；详见实验记录 8.32 节。

对截图的三个候选位置执行固定位置多源 PSF 审计：

```powershell
python -m rst19.multipsf_cli `
  doc/00-项目资料/原始数据 `
  --positions "1269,3465;1274,3467;1272,3453" `
  --psf-fwhm 2 `
  --out-dir tmp/local-multipsf-audit-screenshot
```

该研究工具在同一局部窗口和共享背景下比较 `K=1/2/3` 个固定中心，使用非负分量、BIC 和分量 SNR 输出模型敏感性，并分别保存 `raw`、范围码屏蔽和精确 `-1` 屏蔽结果。raw 15 帧只有 F07/F12 的 `K=2` BIC 更优，但新增分量 SNR 约为 `4.38/4.00`，没有一帧同时通过 `ΔBIC≥10`、`SNR≥5`；`K=3` 没有越线。它解释候选层为何会出现两个框，但不把两个局部响应直接计为两颗恒星；逐帧机器表和限制见 [`检测器实验记录`](../doc/02-星图识别/检测器实验记录.md) 8.37。

要检查固定中心误差与自由度过拟合，可扫描有界自由位置半径：

```powershell
python -m rst19.free_multipsf_cli `
  doc/00-项目资料/原始数据 `
  --radius-sweep --mask-modes raw `
  --position-radii 0.25 0.5 0.75 1.0 1.25 `
  --positions "1269,3465;1274,3467;1272,3453" `
  --psf-fwhm 2 `
  --out-dir tmp/local-free-multipsf-radius-sweep-screenshot
```

raw `K=2` 的确认数依次为 `0/15、0/15、0/15、1/15、1/15`，固定位置为 `0/15`；半径越宽，位置触边和局部结构吸收风险越需要单独记录。这一研究工具不接入 GUI 默认计数，完整解释见 [`亮点真实性与伪影判别研究`](../doc/02-星图识别/亮点真实性与伪影判别研究.md) 10.24。

第二组截图的局部双框可用同一工具复核：

```powershell
python -m rst19.multipsf_cli `
  doc/00-项目资料/原始数据 `
  --positions "2439,4021;2435,4022" --psf-fwhm 2 `
  --out-dir tmp/local-multipsf-audit-screenshot-pair2
```

该组 raw `K=2` 在 15 帧的 `ΔBIC` 全部为负，不能作为两颗独立恒星。检测器会将同孔径的重复高位码/异常负值记录为 `CODE_PATTERN`；候选仍可在候选视图查看，质量视图只显示通过当前质量规则的源。

对当前 pair 再运行有限自由位置审计：

```powershell
python -m rst19.free_multipsf_cli `
  doc/00-项目资料/原始数据 `
  --positions "2439,4021;2435,4022" --psf-fwhm 2 `
  --position-radius 1.25 --mask-modes raw range_masked sentinel_masked `
  --shifts-json tmp/sequence-feature-persistence-code-pattern-current-v3/sequence_feature_persistence.json `
  --out-dir tmp/local-free-multipsf-audit-current-pair-v1
```

当前 raw 结果为：K=2 的 BIC 在 `9/15` 帧有改善，但 `ΔBIC≥10` 只有 `1/15`，第二分量 SNR 最大约 `2.41`，且 `11/15` 帧位置触及 `±1.25 px` 边界。因此局部模型改善只能作为混合结构敏感性证据，不能写成第二颗星确认；详见 [`检测器实验记录`](../doc/02-星图识别/检测器实验记录.md) 8.61。

经验 PSF 形状对照命令：

```powershell
rst19-empirical-pair-audit doc/00-项目资料/原始数据 `
  --source-catalog tmp/source-quality-audit-code-pattern-current-v3/source_catalog.csv `
  --primary-id 82931 --secondary-id 82934 `
  --coordinate-mode centroid `
  --shifts-json tmp/sequence-feature-persistence-code-pattern-current-v3/sequence_feature_persistence.json `
  --out-dir tmp/empirical-pair-psf-audit-code-pattern-current-v1
```

该研究命令只在模板帧中从质量通过且隔离的源估计经验 PSF，再对两个指定 detector 位置逐帧计算余弦相关、相对残差和中心能量占比；局部隔离模板不足时显式记录 `None`。它不参与默认检测、质量门控、GUI 或缓存，相关度也不代表恒星概率、星表身份或独立去混叠通量。

对 `82931/82934` 的局部模板窗口扫描显示，`18--512 px` 范围内虽有 `6,081--17,132` 个候选，但满足 `21 px` 隔离条件的源始终只有 `2` 个；`1024 px` 才得到 `6` 个宽范围模板源。该结果说明 `local None` 是拥挤区的模板适用域缺口，不是物理否定；宽范围 fallback 的主/副相关度中位数为 `0.553/0.480`，机器记录位于 `tmp/local-psf-availability-pair-82931-82934-current-v1/`。

pair 强制测光共变审计：

```powershell
rst19-pair-covariance-audit `
  tmp/forced-stability-audit-code-pattern-current-v8-pair-peak/forced_stability_frame_metrics.csv `
  --primary-id 82931 --secondary-id 82934 `
  --out-dir tmp/pair-flux-covariance-audit-code-pattern-current-v1
```

它比较固定/局部 `flux_snr` 的跨帧共变，并用同表其它完整源构造控制 pair；另按 pair 总响应中位数输出组内相关，防止共同状态切换被误写成两颗独立星。输出是共享响应/孔径/值域的研究诊断，不是双星概率、正式 p 值或星表身份。

当前 v3 还输出 `primary_fraction=primary/(primary+secondary)` 的中位数、范围、`1.4826×MAD`、低/高 pair 总响应状态下的分配中位数和差值，以及 `29,639--29,640` 个控制 pair 的绝对分配变化中位数、P95 和探索性上尾。该比例使用强制测光 SNR，只描述两个 detector response 如何重新分配，不是校准通量比例；小样本分层和控制上尾也不构成显著性检验。复现已有结果可将输出目录改为 `tmp/pair-flux-covariance-audit-code-pattern-current-v3`。

经验 PSF 的单源位置控制：

```powershell
rst19-empirical-pair-fit doc/00-项目资料/原始数据 `
  --source-catalog tmp/source-quality-audit-code-pattern-current-v3/source_catalog.csv `
  --primary-id 82931 --secondary-id 82934 `
  --coordinate-mode centroid `
  --mask-modes raw,range_masked,sentinel_masked,range_sentinel_masked `
  --shifts-json tmp/sequence-feature-persistence-code-pattern-current-v3/sequence_feature_persistence.json `
  --out-dir tmp/empirical-pair-fit-code-pattern-current-centroid-v2
```

它在同一窗口/掩膜中比较固定质心 K=1/K=2，再用双源中点周围的单源位置网格控制锚定偏差；真实 pair 的质心固定双源 raw 线为 `11/15`，网格控制为 `0/15`，峰值锚点固定与控制均为 `0/15`。该结果用于解释局部模型自由度，不接入默认检测或星数。

为检验局部模板缺口是否改变上述结论，研究审计又把 `1024 px` 范围内筛出的 6 个隔离源直接作为经验 PSF 重跑。宽范围模板的固定质心双源 raw 线仍为 `11/15`，最佳单源位置控制仍为 `0/15`，首帧固定双源 `ΔBIC=-6.31`、第二分量 SNR `0`；因此模板不足会改变拟合幅度，但不能把两个宽筛响应升级为两颗物理恒星。该复核不进入默认检测、GUI 或缓存，产物为 `tmp/empirical-pair-fit-local-wide-1024-pair-82931-82934-current-v1/`，完整解释见 [`检测器实验记录`](../doc/02-星图识别/检测器实验记录.md) 8.91。

正式的有界自由位置经验 PSF 对照：

```powershell
rst19-empirical-pair-free doc/00-项目资料/原始数据 `
  --source-catalog tmp/source-quality-audit-code-pattern-current-v3/source_catalog.csv `
  --primary-id 82931 --secondary-id 82934 `
  --mask-modes raw,repeated_code_masked,range_repeated_code_masked,range_sentinel_repeated_code_masked `
  --shifts-json tmp/sequence-feature-persistence-code-pattern-current-v3/sequence_feature_persistence.json `
  --single-position-radius-px 2 --double-position-radius-px 3 `
  --out-dir tmp/empirical-pair-free-code-pattern-current-v3
```

该研究命令在同一经验 PSF、窗口、平面背景和掩膜下连续优化 K=1/K=2 的位置与幅度，并允许 K=2 退化为同一位置。当前 pair 的 raw 联合证据线（`ΔBIC≥10`、第二分量 SNR `≥5`、拟合间距 `≥1 px`、收敛且不触边）为 `0/15`；raw 拟合间距中位约 `0 px`、最大约 `1.05 px`。值域掩膜组合仍无通过帧；同一实现的 `6 px` 双源注入回归控制可以通过联合线，说明结果不是算法完全不会拟合双源。该工具只写 `tmp/empirical-pair-free-code-pattern-current-v3/`，不接入默认检测、质量门、GUI 或缓存，详细解释见 [`检测器实验记录`](../doc/02-星图识别/检测器实验记录.md) 8.92。

需要排查工程重复码时，可使用 `repeated_code_masked`、`range_repeated_code_masked` 等模式并传入 `--repeated-code-values 3990,3991,3992,3993`；这只是值域敏感性对照，不是设备满阱标定。

源目录最近邻群体控制可用 rst19-source-separation-audit。它分开统计测量质心与整数峰的最近邻距离，并支持用 --target-id 回查指定候选；当前 82931/82934 的质心最近邻距为 2.738 px，位于 84,594 个候选最近邻距离的 0.0165% 分位，而峰坐标最近邻距 4.123 px 位于 5.70% 分位。该统计用于说明峰竞争与质心收缩的几何背景，不是双星概率或星表身份。产物为 tmp/source-separation-audit-code-pattern-current-v1/。

几何条件化的近邻机制背景可用 `rst19-pair-mechanism-context`：在测量质心距 `≤3.5 px` 且整数峰距 `3.5--4.8 px` 的无序候选 pair 中，统计两端 `feature_class` 是否一致、是否同时通过质量层，并回查目标 pair。当前窗口共 `11` 对，其中同类别 `7` 对、跨类别 `4` 对，双质量通过 `0/11`；`82931/82934` 属于 `range_anomaly + crowded_blend` 跨类别组合。该结果只描述 detector-level 机制背景，不是噪点率、双星概率或星表身份；目标位置污染注入还要把 baseline、injected 和 new 分开。产物为 `tmp/pair-mechanism-context-code-pattern-current-v1/`，实现为 `src/rst19/pair_mechanism_context.py`。
质心上限放宽到 `4.0 px` 时会得到 `840` 对、双质量通过 `149`，所以严格窗口的 `0/11` 不能外推成全图比例；窗口改变后必须重新定义比较总体。

窗口敏感性可用 `rst19-pair-mechanism-sensitivity` 固定复现：它输出 `3.0/3.5/4.0 px` 质心上限和两组峰距扩展的 CSV/JSON，当前对应 `8/11/840` 对、双质量 `0/0/149`。该模块只读源表，不重跑 FITS、不写检测缓存；产物为 `tmp/pair-mechanism-sensitivity-code-pattern-current-v1/`。

若要避免把局部峰数直接解释成恒星数，可使用 `rst19-source-group-audit` 建立“父源组—子候选”层：它只读取已有 `source_catalog.csv`，按测量质心距离和 `max(2.5 px, 1.5×PSF FWHM)` 建立连通组，再依据质量通过、值域/结构旗标和双 PSF 独立证据，将组标为 `isolated`、`unresolved_group` 或 `independent_group_candidate`。组代表仅用于显示和排序，不能覆盖子候选、落选原因或物理星数；一对一像素分配、跨帧稳定性和星表身份仍是后续拆分前的必要核验。当前 v3 的 `84,594` 个候选形成 `84,586` 个组，其中 `8` 个多成员组全部为 `unresolved_group`，`0` 个达到独立候选线；`82931/82934` 是两成员未分辨组，质心距 `2.738 px`。该解释层不修改检测、质量层、GUI 或缓存，产物为 `tmp/source-group-audit-code-pattern-current-v1/`，测试为 `tests/test_source_group_audit.py`。

父源组半径敏感性可用 `rst19-source-group-sensitivity`：固定同一源表和 PSF/质量证据线，扫描 `2.5/3.0/3.5/4.0 px` 并输出统一 CSV/JSON。当前多成员组数为 `4/8/12/831`，独立候选组为 `0/0/0/25`；目标在 `2.5 px` 暂时分开，在 `3.0 px` 及以上均为 `unresolved_group`。该扫描只说明解释层对半径的敏感性，不是光学分辨率、恒星数量或双星概率；产物为 `tmp/source-group-radius-sensitivity-code-pattern-current-v1/`，测试为 `tests/test_source_group_sensitivity.py`。

```powershell
rst19-source-group-sensitivity `
  tmp/source-quality-audit-code-pattern-current-v3/source_catalog.csv `
  --target-id 82931 --target-id 82934 --psf-fwhm 2 --workers 4 `
  --out-dir tmp/source-group-radius-sensitivity-code-pattern-current-v1
```

近邻 pair 的独立像素支持留出可用 `rst19-pair-support-audit`：它在每帧目标局部裁剪上重新执行当前宽筛，随后用最大基数/最小残差一对一分配，并把总孔径、共同孔径、非共同孔径和最近目标二分区的有效像素/正残差通量分开记录。它还可屏蔽全局极值、重复码、局部极端负值和共同孔径；15 帧截图 pair 的 raw 双分配为 `4/15`、双质量为 `0/15`，重复码或局部异常屏蔽后的双分配均为 `0/15`。v2 另写出 `pair_support_summary.csv`，保留候选分配与质量通过的不同分母；结果只用于混叠/值域敏感性审计，不直接修改质量层、GUI 或缓存。产物为 `tmp/pair-support-audit-code-pattern-current-v2/`。

源级特征依赖审计可用 `rst19-feature-correlation`：它只读取 `source_catalog.csv`，计算峰值、通量/滤波 SNR、FWHM、椭圆率、sharpness、足迹、PSF 支持和质心偏移之间的 Spearman 相关性，并输出每个首要类别的中位数及类别内相关性。当前全量候选中 `peak—filter_snr=0.9355`、`flux_snr—filter_snr=0.8711`、`filter_snr—FWHM=-0.8444`；这用于防止把相关显著性量重复计权，不是因果分析、真星概率或质量规则改写。产物为 `tmp/feature-correlation-audit-code-pattern-current-v1/feature_correlations.csv`、`feature_class_medians.csv`、`feature_class_correlations.csv` 和 `feature_correlation_audit.json`，测试为 `tests/test_feature_correlation_audit.py`。

孔径半径 `3/4/5 px` 的敏感性对照中，raw 双分配均为 `4/15`，双质量均为 `0/15`，重复码和局部异常屏蔽后的双分配均为 `0/15`；对应产物为 `tmp/pair-support-audit-code-pattern-current-r3/` 和 `current-r5/`。共享比例随孔径变化，因此它是几何测光诊断量，不是孔径无关的伪影概率。

原始像素拓扑审计可用 `rst19-pair-pixel-topology`。它在指定 pair 的未降噪局部窗口内，以多个阈值统计 8 邻域正值连通块，再以多个邻域半径寻找原始局部极大值；同时输出重复码/特殊负值屏蔽对照。当前 `82931/82934` 在 `3/5/8/10σ` 下均属于同一正值连通块，但 `3/5/7 px` 局部峰审计中两者均不是局部极大值，首帧最亮像素为 `(2438,4022)=13028`。因此它是“共享响应结构/峰重定位”的像素级证据，不是噪点真值判定；不能单独排除真实混叠恒星，也不应把两个宽筛框直接计为两颗星。实现为 `src/rst19/pair_pixel_topology.py`，测试为 `tests/test_pair_pixel_topology.py`，产物为 `tmp/pair-pixel-topology-pair-82931-82934-current-v1/`。

另有一个小样本类别控制 pilot：8 个几何相近的 `compact_quality` pair 中，7/8 对的两个候选同时为 `3 px` raw 局部极大值；目标 pair 为 0/1，且目标质心距 2.738 px 小于控制最小 3.54 px。它只用于描述性上下文，不是 precision 或物理身份估计；产物为 `tmp/pair-pixel-topology-controls-current-v1/`。

按特征类别做经验 PSF 留一法交叉核验：

```powershell
rst19-feature-psf-leaveout `
  doc/00-项目资料/原始数据/20260330163205413_9901.fits `
  --source-catalog tmp/source-quality-audit-code-pattern-current-v3/source_catalog.csv `
  --per-class 16 --max-template-sources 20 `
  --out-dir tmp/feature-psf-leaveout-code-pattern-current-v1
```

它逐源排除自身后重建经验核，并同时输出类别汇总和逐源成对指标，用来检查类别 PSF 差异是否被模板泄漏夸大；结果不参与默认检测、质量门控、GUI 或缓存，也不等于恒星概率。

把宽筛、质量层、15 帧响应和留一 PSF 结果合并成类别证据矩阵：

```powershell
rst19-feature-evidence `
  --source-morphology tmp/source-quality-audit-code-pattern-current-v3/source_feature_morphology.csv `
  --feature-cross tmp/feature-cross-audit-code-pattern-current-v3/feature_cross_audit.csv `
  --psf-leaveout tmp/feature-psf-leaveout-code-pattern-current-v1/feature_psf_leaveout.csv `
  --psf-spatial tmp/source-feature-psf-grid2-per16-code-pattern-current-v4/source_feature_psf_spatial.csv `
  --out-dir tmp/feature-evidence-matrix-code-pattern-current-v3-routing
```

输出 `feature_evidence_matrix.csv/json`。其中候选持久比例、同类质量比例、质量层邻域响应和候选/质量同类响应保持分列；邻域响应不是同类质量持久性。可选的 `--psf-spatial` 还会合并局部模板可得率、局部相关度/残差以及相对全局模板的变化，用来检验空间 PSF 校正是否改变类别解释；模板不足时只记录校准缺口。当前矩阵显示多个被拒类别仍可在 `12/15` 帧附近重复出现，但 PSF/质量层不支持，说明跨帧重复不能单独升级为恒星；该工具只组织研究证据，不修改默认检测、GUI 或缓存。

如果要把“位置持久”与“质量响应同步”进一步压缩为类别级审计，可运行：

```powershell
rst19-feature-evidence-axes `
  tmp/feature-evidence-matrix-code-pattern-current-v3-routing/feature_evidence_matrix.csv `
  --out-dir tmp/feature-evidence-axes-code-pattern-current-v1
```

输出 `feature_evidence_axes.csv`、`feature_evidence_axes_sensitivity.csv` 和 `feature_evidence_axes.json`，其中 `Q|P` 是质量邻域持久数除以候选位置持久数；质量邻域可能属于其它类别，所以它不是逐源质量通过率或恒星概率。当前紧凑质量类 `Q|P=86.1%`，拥挤/尖峰/弱背景/形状类仅 `0.5%--5.0%`，边缘/掩膜类 `22.1%`。敏感性表覆盖基线、较严格 PSF 线、较宽候选线和两组质量差异线，用来区分稳健类别规律与工程阈值边界。该只读汇总用于论文证据分流，不重读 FITS、不修改默认检测、GUI 或缓存。

候选峰与 raw 像素局部峰的一致性审计：

```powershell
rst19-source-peak-consistency `
  tmp/source-quality-audit-code-pattern-current-v3/source_catalog.csv `
  doc/00-项目资料/原始数据/20260330163205413_9901.fits `
  --target-id 82931 --target-id 82934 `
  --out-dir tmp/source-peak-consistency-code-pattern-current-v1
```

`source_peak_consistency.py` 只读取现有源表和原始 FITS，在 `3×3/5×5/7×7` 窗口输出逐源 CSV、类别汇总 CSV、目标明细 CSV 与 JSON。候选来自匹配滤波响应，因此 raw 像素峰与响应峰可以不重合；输出的局部峰标志、最大值差值和类别比例是 detector-level 拓扑诊断，不是恒星概率、precision、FDR 或物理源数。该模块不重新检测、不修改质量层、GUI 或缓存。

`source_proposal_peak_audit.py` 再把 `proposal_methods` 与 raw 峰表按 `detection_id` 连接，按 `feature_class × method_group` 输出来源子组的质量、raw 局部峰率、非局部差值和目标相对占比。它用于发现类别总平均掩盖的来源尾部，例如目标 `82934` 是拥挤类仅 `2/905` 的 `gaussian_only` 子组。来源组合共享同一原图，不是独立投票；模块只读 CSV，不重跑 FITS、不修改 detector、质量层、GUI 或缓存。

`source_proposal_temporal_audit.py` 再把上述来源组合接到同一次 15 帧实验：非紧凑类别逐源连接 `sequence_feature_diagnostic_sources.csv`，紧凑质量类别按 `sequence_feature_source_subgroup_persistence.csv` 聚合。它并列输出候选位置 `≥12/15`、同诊断子组 `≥12/15` 和质量邻域 `≥12/15`，严格区分“来源大组持久”与“目标逐源持久”。当前拥挤 `dog_only` 为 `509/902、196/902、6/902`，目标 `82934` 的 `gaussian_only` 来源组仅 `2` 个且目标自身为 `4/15、3/15、0/15`；`82931` 自身为 `4/15、4/15、0/15`。紧凑类没有逐源时序时保持空值，不把不可用写成零；所有计数仍是注册邻域 detector-level 诊断，不是星表身份、precision、FDR 或物理恒星数。命令为 `rst19-source-proposal-temporal-audit`，产物为 `tmp/source-proposal-temporal-audit-code-pattern-current-v1/`，测试为 `tests/test_source_proposal_temporal_audit.py`。

类别规则签名的非参数对照可用 `rst19-feature-effect-size`：它以 `compact_quality` 为参考，输出各落选类别在 flux/filter SNR、峰值、FWHM、椭圆率、sharpness、足迹、PSF 支持、质心偏移和值域计数上的 AUC/Cliff's delta。该结果只描述当前标签规则的分布差异，不是物理恒星分类器；产物为 `tmp/feature-effect-size-code-pattern-current-v1/`。

真实污染结构中的双源注入审计可用 `rst19-contaminated-pair-injection`。它把已知 Gaussian 双源叠加到目标 pair 中点、紧凑质量源和线状候选附近，扫描 `2.738/4.123 px` 分离和 `0.143/1` 副/主峰比，并将 baseline、injected、new 三层命中分别输出。当前 pilot 用于说明污染结构会改变双源回收，不是当前 FITS 的 precision、伪影概率或物理恒星数；产物为 `tmp/contaminated-pair-injection-code-pattern-current-v1/`，不接入默认检测、质量层、GUI 或缓存。

跨类别污染双源复核可用 `rst19-contaminated-pair-class`：

```powershell
rst19-contaminated-pair-class `
  doc/00-项目资料/原始数据/20260330163205413_9901.fits `
  --source-catalog tmp/source-quality-audit-code-pattern-current-v3/source_catalog.csv `
  --separations-px 2.738 4.123 --ratios 0.143 1 `
  --total-peak-adu 4096 --out-dir tmp/contaminated-pair-class-audit-code-pattern-current-v1
```

它从 8 个互斥首要特征类别中各选一个中位 `filter_snr` 锚点，输出类别选择、逐条件 baseline/injected/new 三层以及 `contaminated_pair_class_summary.csv`。当前机制 pilot 在 `2.738 px` 下候选/质量双命中为 `0/16`、`0/16`，在 `4.123 px` 下为 `11/16`、`9/16`；线状和范围异常类仍为 `0/4`。类别回收只用于区分间距—PSF 几何与局部结构质量门，不是类别召回率、precision、FDR 或物理恒星概率。

多锚点研究可增加 `--per-class 3 --analysis-scope local_roi --roi-half-size-px 192`。该口径在每个类别的 3 个中位锚点周围运行同一 detector，完成 `96` 个局部条件；输出中的 `analysis_scope`、`roi_bounds` 和 injected 计数范围明确标注 ROI 局部边界。当前结果为短距候选/质量 `0/48、0/48`，长距 `33/48、28/48`；它不能替代全幅候选总数，也不接入 GUI 默认或缓存。

逐条件结果还保存注入端点的匹配 ID、匹配距离和质量原因，并写出 `contaminated_pair_quality_reason_summary.csv`。原因标签来自实际 detector flags，`NO_CANDIDATE` 表示宽筛未命中，`QUALITY_PASS` 表示通过质量层；该输出用于机制审计，不改变检测器判断。

当前版本还为每个类别输出确定性的 `recommended_audit_stage` 和 `counting_policy`：例如拥挤类进入联合 PSF 去混叠，范围异常进入原始值域核验，弱源进入分区噪声/注入核验，只有 `compact_quality` 进入星表/WCS 或留出注入的身份核验队列。它们只是下一步审计路由和计数边界，不是加权分数、恒星概率或物理真值；最新带路由字段的复现产物可写入 `tmp/feature-evidence-matrix-code-pattern-current-v3-routing/`。

多次重复特征机制控制：

```powershell
rst19-feature-audit-replicates `
  --trials 64 --seed 19019 `
  --out-dir tmp/feature-audit-replicates-code-pattern-current-v2
```

该命令在不同随机背景上重复已知源注入和阴性结构场景；正样本按注入真值统计候选/质量回收，阴性样本只统计邻域泄漏。当前 `64` 次是扩展版，首轮 `16` 次保留为历史 pilot；它用于检验“宽筛能找到、细筛按机制拒绝”的稳定性，不把合成回收率写成真实 FITS 的 precision、完备率或伪影概率。

真实首帧的符号反相阴性对照：

```powershell
rst19-signed-null-audit `
  doc/00-项目资料/原始数据/20260330163205413_9901.fits `
  --threshold-sigma 4 --min-distance 4 --aperture-radius 4 `
  --psf-fwhm 2 --proposal-mode hybrid --min-flux-snr 5 `
  --min-psf-support-pixels 3 `
  --out-dir tmp/signed-null-audit-code-pattern-current-v1
```

该命令以稳健背景 `B` 构造 `I_mirror=2B-I`，在正向和反相图上使用同一套候选/质量参数，输出 `signed_null_summary.json`、SNR 分档表和特征类别表。它只估计近似对称的背景/噪声尾部泄漏，不是 FDR、p 值、precision 或真实伪影概率；对于有符号整型 FITS，报告还会单独标出“原始极端负码经反相转移”的候选，避免把这类值域结构误认为噪声或恒星。命令只写研究汇总，不写全量源目录、不修改 GUI 默认值或检测缓存。

15 帧聚合入口为 `rst19-signed-null-sequence doc/00-项目资料/原始数据 --out-dir tmp/signed-null-sequence-code-pattern-current-v3`。它逐帧复用同一阴性对照，并额外输出 `signed_null_sequence_quality_sources.csv` 与 `signed_null_sequence_recurrences.csv`，把反相质量源拆成极端负码转移、原始负值/固定哨兵异常、普通弱/宽响应三层；源表的互斥 `raw_evidence_layer` 实际分布为 `19/13/3`，同时保留负值、极值和 `-1` 的非互斥计数。`≤1 px` 复现簇只表示 detector-level 固定坐标重复，不是物理身份匹配。当前 pooled 结果为反相 `36,170/35`、排除转移后质量 `16`，不改变默认检测、GUI 或缓存。

反相质量源的同帧正向近邻控制使用 rst19-signed-null-overlap，复用当前 GUI 的 hybrid + 4σ + FWHM=2 px + flux SNR≥5 + PSF support≥3，只在原始 detector 坐标查询 ≤1/2/4 px 候选和质量邻域，不做跨帧注册、不写正向全量目录。当前结果为候选 0/35、23/35、32/35，质量源 0/35；它用于证明“局部候选近邻不等于正常星”，不替代星表/WCS 或注入真值。正式 CLI 产物为 tmp/signed-null-positive-overlap-code-pattern-current-v2/。

使用用户显式提供的 CSV 星表进行先验匹配（仓库不内置离线星表）：

```powershell
rst19 frame.fits `
  --catalog catalog.csv `
  --center-ra 129.533548 `
  --center-dec -1.845372 `
  --pixel-scale-arcsec 10 `
  --rotation-deg 0 `
  --parity 1
```

主界面的“打开星表核验”会复用这套 CSV/WCS 参数并显示唯一匹配、位置残差和目录星等；默认匹配在半径门控形成的稀疏候选连通分量内使用全局最大基数、最小总残差的一对一分配，并保留 `assignment_mode="greedy"` 作为历史贪心对照；运行匹配后可点击“根据匹配拟合 WCS”，用至少 6 个有效、非共线匹配点估计视场内的局部仿射模型，展示像元角尺度、旋转、parity、内点 RMS 和各向异性，并把结果用于创新摘要的视场切平面角速度。它仍以先验 WCS 为入口，不是全天空盲解算；未填写可靠像元角尺度时不要运行匹配。CSV 至少需要以下列：

```text
source_id,ra_deg,dec_deg,magnitude,magnitude_source,pmra,pmdec,ref_epoch
```

正式星等实验建议将公共目录拆成两层：亮星定标层（例如 `G=5--13.5`，避开饱和并提供零点/颜色项）和深星覆盖层（例如 `G=8--18`，只用于覆盖检测极限）。两层分别用 `rst19-gaia-frame` 或 `rst19-gaia-tiled` 查询后，可用下面的命令按 `source_id` 去重、补齐字段并继承完整性证据：

```powershell
rst19-gaia-merge tmp/gaia-calibration.csv tmp/gaia-deep.csv `
  --out tmp/gaia-merged.csv
```

命令会在 `gaia-merged.csv.meta.json` 中记录输入文件 SHA-256、行数、重复源、关键字段冲突和天空覆盖；任一输入缺少 `complete=true`、覆盖不一致或存在关键字段冲突时，输出仍可供诊断，但明确标记为 `complete=false`，并以退出码 `3` 结束。只有完整性审计通过的合并表，才可以作为“全场覆盖”或正式最暗星结论的目录依据。

`--pixel-scale-arcsec`、旋转角和 parity 是相机参数假设，不是当前数据已经核验的事实；应通过稳定匹配星、残差和留出星验证后再固定。

格式说明中的 26 个辅助字段还可以先提供逐帧的光轴中心先验。对本地 15 张真实 FITS 运行：

```powershell
rst19-aux-audit doc/00-项目资料/原始数据 --out-dir tmp/auxiliary-attitude-audit
```

该命令把四元数按向量在前、标量在后的 Hamilton 约定转换为主动旋转，并以相机机体系 `-Y` 轴计算 RA/Dec；实际 15 帧中四元数模长均为 1，计算光轴与辅助 RA/Dec 的最大差为约 `1.1e-10` 角秒，`p_az/p_el` 与 FITS 头匹配 `15/15`。这证明辅助数据可以用于按帧收窄星表候选搜索和检查坐标约定，但当前头文件仍缺少像元尺度、完整轴向矩阵、畸变/SIP 等信息，所以它是“光轴/姿态先验”，不是完整 WCS，也不能单独给出逐星身份或正式星数。输出为 `auxiliary_boresight_audit.csv` 和 `auxiliary_boresight_audit.json`。

若要把格式说明中的文字与实际 FITS 文件分开核验，可运行：

```powershell
rst19-format-audit doc/00-项目资料/原始数据 --out-dir tmp/format-audit-code-pattern-current-v1
```

该命令只读检查 15 帧的文件长度、`NAXIS1/NAXIS2`、`BITPIX`、存储 dtype、`BSCALE/BZERO/BLANK`、首行 208 字节辅助区域、标准 WCS 卡片、负值/`-1`/正负极值，并输出用 `DATE-OBS` 对位置—速度字段的差分自洽误差。当前结果为 `15/15` 个 `4096×4096`、`BITPIX=16`、大端 `>i2`，文件长度全部匹配；`BLANK` 为 `0/15`，精确 `-1` 合计 `2,185`，标准 WCS 卡片为 `0/15` 帧存在、最小 WCS 核心为 `0/15`。这只是格式和值域审计，不确认 ADC signedness、坏像素/饱和语义或完整 WCS；首行辅助区的光轴/姿态字段不能替代标准 WCS。产物为 `tmp/format-audit-code-pattern-current-v1/format_audit_frames.csv`、`format_audit_velocity.csv` 和 `format_audit.json`。

对整组 15 帧做逐帧验证：

```powershell
rst19-wcs-validation doc/00-项目资料/原始数据 catalog.csv `
  --center-ra 129.533548 --center-dec -1.845372 `
  --pixel-scale 10 --rotation 0 --parity 1 `
  --out-dir tmp/wcs-validation
```

该命令对每一帧独立运行同一套检测和先验一对一匹配，再拟合局部二维仿射 WCS；输出 `wcs_frame_validation.csv`、`wcs_validation_report.json`、残差曲线和尺度曲线。`validated` 只表示该帧有足够的唯一匹配并通过留一残差计算，不能替代全天空盲解算、官方逐星真值或人工抽检。主界面的星表核验窗口在完成当前帧匹配后也提供“验证 15 帧 WCS”，后台完成后用 Tkinter 表格和曲线显示逐帧结果。

启动 Python 桌面界面：

```powershell
rst19-gui
```

等价的 UI 别名是 `rst19-ui`。界面中的最大源数输入框留空表示单图全量，15 帧此时自动使用每帧前 6000 个匹配滤波高 SNR 源作为配准工作集；候选总数仍完整显示，工作集内的 `returned_count`/质量数/点轨迹不会冒充全量恒星数。序列快速路径先由首帧建立共享局部 background/RMS 图，同尺寸帧复用；每帧仍独立计算全局统计和有效像素掩膜，尺寸不一致时自动回退。每个 256 px 背景块最多确定性抽样 4,096 点，并在无效像素稀疏时用单遍 Gaussian 卷积；线性伪迹连通域审计和源质量规则仍保留，结果会记录 `background_model_mode` 等快速口径。检测参数区、F01…F15 帧状态带、右侧证据卡和底部状态栏会同时显示当前 `Fxx/总帧数`、百分比、源级测量阶段和线状残差筛选阶段；过密的后台事件会合并，避免界面落后于实际计算。

界面默认读取 `doc/00-项目资料/原始数据/`，只在本机处理；可以通过 `--data-dir` 指定其他获授权的 FITS 目录。最大源数输入框留空表示全量检测，绿色环表示最暗可信源。检测默认会把跨越多个 PSF 宽度的长线标记为 `LINE_ARTIFACT`：它们仍保留在“全部候选”审计层，但不会进入可信星点和点源运动轨迹；点击“分析当前帧”时会另外提取单帧长线候选，显示在运动候选层并标成橙色线，不能仅凭单帧称为已确认运动目标。完成 15 帧分析后，“15 帧证据”窗口提供逐帧表、线状轨迹表、逐帧筛选漏斗的“线状诊断”页、集中展示时序/配准/速度及 95% 区间/方向及区间/短期外推逐轴区间的“创新摘要”页、图表、辅助遥测表、“遥测轨迹”相对首帧 XY 投影图、按需生成的注册中值拼图/正负残差差分图、按需运行的星点注入实验和“单图长线”逐帧审计页；若星表局部 WCS 已校准，创新摘要还会显示视场切平面角速度，否则继续显示“待标定”；辅助遥测表会显示位置 `m`、速度 `m/s?`、位置—速度自洽误差，以及末帧状态向后 5 个中位帧间隔的常速度位置外推，待主办方确认格式说明中的速度单位。图像轨迹图只把跨帧 `moving` 组画成轨迹，显示全部质心、OLS 线、外推段和逐轴轨迹均值区间；单帧长线保持独立。遥测轨迹图只绘制相对首帧的 J2000/WGS84 XY 投影，并用琥珀色标出预测段，不替代轨道传播。

界面启动和切换帧时只载入图像预览，不会提前计算检测结果。点击“分析当前帧”后，程序先用轻量背景/连通域基线寻找长线候选，橙色候选会先进入运动层；随后继续全量星点、SNR、形状和最暗源星等精测，因此单帧形状不会被误称为 `moving`，精测完成后会用同一候选口径收敛结果。结果会按 FITS 文件路径、文件修改状态和检测参数写入项目根目录的 `.rst19-cache/` 压缩缓存；相同输入再次分析时直接复用。点击“分析 15 帧动目标”后，程序对整个目录做平移配准和轨迹关联，完成后默认切换到“稳定星场”层；该层只绘制当前帧实际存在的 `static/persistent` 点源。需要查看运动证据时可手动切换到“运动候选”层：洋红线表示跨帧拟合通过的线状 `moving` 目标，橙色线表示单帧或尚未达到跨帧证据门槛的线状 `candidate`，蓝色环表示严格 `moving` 点轨迹。切换帧时右上角的图层选择会保持不变，旧线程结果会因帧令牌失效而丢弃，过期任务也不会把旧结果写入当前显示状态。点击“清空检测缓存”会删除当前 gzip 缓存和旧版本 JSON 产物，即使任务正在运行也可以执行；本次任务完成后不会把清理前的结果写回缓存，原始 FITS 不会被删除。

序列结果的“稳定星场”层只显示跨帧持续的 `static/persistent` 点源，分别用青绿色/蓝色区分严格静态与低置信持续候选；悬停可查看出现帧数、通量 SNR、配准位移和拟合 RMS。15 帧证据窗口的“图证”页提供“星点时序审计”按钮，后台从原始 FITS 导出 `source_track_contact_sheet.png`、`source_track_audit.csv` 和汇总 JSON，灰框代表缺帧，不把颜色或抽样图当作真值。单帧源表同时保存 `peak_x/peak_y`、PSF 加权质心 `x/y` 和 `centroid_shift_px`，便于检查弱源标记是否被噪声拉偏。

命令 `rst19-trails <FITS目录>` 可以逐张运行单图长线审计，输出候选数量、最长线长度/宽度/方向、残差 SNR、触边状态和包围盒的 CSV 与曲线；GUI 的“单图长线”页按当前检测参数按需运行同一审计。它只回答“单张图中是否存在长线形状候选”，不跨帧判定 `moving`；15 帧速度和方向仍以序列证据为准。

命令 `rst19-sources <FITS文件>` 会在一次检测完成后导出全量 `source_catalog.csv`、质量/拒绝标记汇总、互斥首要特征类别 `source_feature_summary.csv`、提议器来源交叉表 `source_feature_method_summary.csv`、`source_feature_morphology.csv`、可重叠机制子类 `source_feature_subclass_summary.csv`、按类别 4×4 粗网格统计的 `source_feature_spatial.csv`、按类别原图抽样的 `source_cutout_contact_sheet.png`/`source_cutout_manifest.csv`、按类别经验 PSF 相似度的 `source_feature_psf_similarity.csv`、全局/局部经验 PSF 留出对照 `source_feature_psf_spatial.csv`、类别图、两张 SNR 图、重叠质量标志图以及 16×16 返回源密度/质量通过率空间图和 `source_spatial_grid.csv`。方法来源表记录 Gaussian/DoG 的交集、`gaussian_only`、`dog_only` 和无 Gaussian 质量数，用于区分提议器敏感性与质量层证据，但不产生真星概率。形态表按类别记录峰值/通量/滤波 SNR、FWHM、椭圆率、sharpness、PSF 支持、质心偏移、空间分位数和拒绝原因比例；子类表按完整 flag token 拆开重复码、极端负值、饱和、边界、部分掩膜、线状、未分辨近邻和尖峰，明确这些行允许重叠、不能相加；空间表用于检查线状、边缘、固定结构是否集中在局部区域；PSF 相似度表记录确定性原图抽样的经验核相关系数、相对残差和中心能量占比；空间 PSF 表用留出样本构造局部模板，并用整幅候选表做隔离检查，模板不足时只记录回退，不改写 `quality_passed`。高 SNR 拒绝计数、空间集中性、方法交集、`corr≥0.8` 和局部模板可得率都只是诊断切片，不是物理伪影真值，也不会自动改写质量层。它保留 `peak SNR`、孔径 `flux SNR` 与匹配滤波 `filter SNR` 三个不同量，并明确记录当前背景、噪声、阈值、PSF、质量参数和空间诊断边界；`feature_class` 只是按固定优先级整理的算法审计标签，原始 `flags` 仍允许重叠。GUI 检测参数区的“研究工具 → 导出星点研究表”复用当前已完成的检测结果，不会因为导出而再次检测，导出完成后直接打开 Tkinter 的“论文表图 / 指标与口径”窗口。

空间 PSF 诊断默认使用 `4×4` 网格、每类 8 个抽样源，以保持日常导出速度；进行类别研究时可以显式扩大抽样：

```powershell
rst19-sources <FITS文件> --psf-fwhm 2 `
  --spatial-psf-per-class 16 --spatial-psf-grid-size 2 `
  --out-dir tmp/source-feature-psf-grid2-per16-code-pattern-current-v4
```

这两个参数只影响 `source_feature_psf_spatial.csv` 的诊断抽样，不改变候选数、质量规则或 GUI 默认结果。

若要把空间差异与信号强度分开，可在真实首帧上运行空间分区留出注入：

```powershell
rst19-spatial-injection `
  doc/00-项目资料/原始数据/20260330163205413_9901.fits `
  --grid-size 2 --peak-levels 256 512 1024 `
  --trials 1 --sources-per-cell 1 --psf-fwhm 2 `
  --proposal-mode hybrid --psf-model gaussian `
  --signal-normalization integrated_excess `
  --out-dir tmp/spatial-injection-code-pattern-current-2x2-gaussian-t1-high-paired-v2
```

该命令在每个 detector 单元中从当前候选、特殊值和局部亮结构之外选相对空白位置，记录已知注入源的候选/质量回收率、局部背景噪声和实际峰值超额。`integrated_excess` 固定离散注入小窗的总积分信号，避免不同 PSF 核和把峰值实验混在一起。回收率分母是已知注入源，不是当前 FITS 的真实恒星数、precision 或完备率；统一 Gaussian 核的 pilot 也不能替代各分区实测 PSF。当前首帧 `2×2` pilot 显示 `256/512/1024 ADU` 的候选回收均为 `4/4`，质量回收为 `0/4、4/4、4/4`，支持“质量层跃迁首先受通量 SNR 控制”。配对版还保存同 dtype 无注入控制：某单元的候选增量为 `+16`，但质量只增 `+1`，新增项集中在 `filter_snr≈4` 边界，说明宽筛全图响应噪声重估会产生非局部候选膨胀，不能把它解释为空间特殊星点。每档仅 4 个样本，不能外推成正式空间完备率；低信号 `56/128 ADU` 对照保存在 `tmp/spatial-injection-code-pattern-current-2x2-gaussian-t1/`，配对版保存在 `tmp/spatial-injection-code-pattern-current-2x2-gaussian-t1-high-paired-v2/`。

经验 PSF 的孔径敏感性审计可复核“亮点是否因测光口径而改变”：

```powershell
rst19-aperture-sensitivity `
  doc/00-项目资料/原始数据/20260330163205413_9901.fits `
  --aperture-radii 3 4 5 6 --reference-aperture-radius 4 `
  --control-signal-adu 512 --grid-size 2 --psf-fwhm 2 `
  --background-box-size 128 --min-flux-snr 5 `
  --min-psf-support-pixels 3 --min-distance 4 `
  --threshold-sigma 4 --proposal-mode hybrid `
  --empirical-psf-radius 7 --empirical-psf-sources 64 `
  --seed 19019 --out-dir tmp/aperture-sensitivity-code-pattern-current-v1
```

当前首帧产物中四档候选回收均为 `4/4`，质量回收为 `4/4、2/4、0/4、0/4`；`flux SNR` 中位数为 `6.428、5.168、3.940、3.644`。这只是固定四点、固定经验模板的测光边界实验，不是完备率，不修改默认孔径、质量规则或缓存。机器产物为 `tmp/aperture-sensitivity-code-pattern-current-v1/aperture_sensitivity_audit.csv/json`。

各类特征的参数敏感性和同坐标类别转移：

```powershell
rst19-feature-parameter-sensitivity `
  doc/00-项目资料/原始数据/20260330163205413_9901.fits `
  --threshold-levels 4,5,6,8 --flux-snr-levels 3,5,7,9 `
  --out-dir tmp/feature-parameter-sensitivity-code-pattern-current-v1
```

固定同一首帧分别扫描候选阈值和质量层 `flux SNR`，输出运行汇总、按类别默认坐标匹配率和类别转移矩阵。当前首帧 `flux SNR=3/5/7/9` 的候选数都为 `84,594`，质量数为 `41,110/29,260/22,438/18,296`；`flux SNR=9` 时 `9,614` 个默认紧凑质量位置转为弱/背景，`flux SNR=3` 时 `10,280` 个默认弱/背景位置转入紧凑类。该结果用于证明质量门会重标记同一候选，不能把质量数变化直接叫恒星数量变化；候选层、质量层和物理身份仍需分开。该研究工具不写检测缓存、不修改 GUI 默认参数。

三组序列 JSON 完成后，可用 `rst19-feature-sequence-compare --run NAME=JSON ...` 做只读比较。该命令校验唯一帧数、`required_presence` 和关联半径一致，输出首帧候选/质量数量、各类候选/质量持久性和类别转移；它不重跑 FITS，也不会把每帧按类别展开的行数当作帧数。当前 `SNR=3/5/9` 的 `compact_quality` 质量持久率为 `0.776/0.815/0.864`，截图 `82931/82934` 在三组中仍为异常值域/未分辨近邻候选，而不是两个稳定质量源。产物目录示例为 `tmp/sequence-feature-parameter-comparison-current-v1/`。

若要检查指定候选在其所属类别中是否属于异常高显著性 hard negative，可运行：

```powershell
rst19-feature-class-context `
  tmp/source-quality-audit-code-pattern-current-v3/source_catalog.csv `
  --target-id 82931 --target-id 82934 `
  --out-dir tmp/feature-class-context-pair-82931-82934-current-v1
```

该命令只读取源级 CSV，输出同类 `p10/median/p90/empirical percentile`，不重跑 FITS、不输出恒星概率。当前 `82934` 在 `crowded_blend` 类中 `flux SNR/filter SNR/PSF 支持` 约为 `98.2%/99.9%/99.6%` 分位，但仍因 `UNRESOLVED_BLEND` 和共享几何不能计星；这正是“高 SNR 不等于独立身份”的 hard negative 证据。

若要对所有类别筛查高显著性落选候选，可运行：

```powershell
rst19-feature-hard-negative `
  tmp/source-quality-audit-code-pattern-current-v3/source_catalog.csv `
  --top-n 5 --metric flux_snr `
  --out-dir tmp/feature-hard-negative-code-pattern-current-v1
```

命令只读取 `source_catalog.csv`，按 `quality_passed=False` 且类别内 `flux_snr` 降序输出每类 top 5，另外保留 `filter_snr/peak/FWHM/椭圆率/sharpness/PSF支持/足迹/flags` 和 `filter_flux_snr_ratio`。该比值只是孔径通量 SNR 与匹配滤波 SNR 的分歧诊断，不是新的综合分数。工具用于发现“显著性与质量拒绝规则冲突”的 hard negative，不输出恒星概率；`other_rejected` 默认不纳入，可用 `--include-other-rejected` 显式加入。当前产物为 `tmp/feature-hard-negative-code-pattern-current-v1/`。

若要检查“单个旗标是否可容错、两个旗标叠加后是否必然落选”，可运行：

```powershell
rst19-feature-flag-interaction `
  tmp/source-quality-audit-code-pattern-current-v3/source_catalog.csv `
  --high-snr-threshold 10 `
  --out-dir tmp/feature-flag-interaction-code-pattern-current-v1
```

该命令只读取源级 CSV，分别输出旗标 presence、完整旗标集合恰好相等的 `exact_set`、质量通过数和高 SNR 落选数。当前 v3 中 `PARTIAL_MASKED` 出现在 `11,490` 行，其中恰好只有该旗标的 `2,777` 行全部通过；但 `LOW_FLUX_SNR+PARTIAL_MASKED` 的 `7,703` 行和 `INSUFFICIENT_PSF_SUPPORT+PARTIAL_MASKED` 的 `5,084` 行质量通过数均为 `0`。这说明当前质量门是条件组合逻辑，而不是“看到一个掩膜就全部删除”；这些数字仍是 detector-level 规则行为，不是物理真星率或伪影概率。产物为 `tmp/feature-flag-interaction-code-pattern-current-v1/feature_flag_interactions.csv/json`，实现为 `src/rst19/feature_flag_interaction.py`，测试为 `tests/test_feature_flag_interaction.py`。

若要解释不同类别为什么落选，可运行 `rst19-feature-gate-margin`：

```powershell
rst19-feature-gate-margin `
  tmp/source-quality-audit-code-pattern-current-v3/source_catalog.csv `
  --out-dir tmp/feature-gate-margin-code-pattern-current-v1
```

它只读取 `source_catalog.csv`，按 `feature_class` 输出 `flux_snr`、PSF 支撑、FWHM、椭圆率、sharpness 和足迹相对于当前质量门的有符号余量，并保留负余量比例。正值只代表该字段未越过数值门，负值只代表该字段越界；不同单位不相加，也不生成恒星概率。当前结果显示弱/背景类由 `flux_snr` 主导、尖峰类由 PSF 支撑主导，而线状类虽有很高 SNR 仍由 `LINE_ARTIFACT` 结构旗标分流。实现为 `src/rst19/feature_gate_margin.py`，测试为 `tests/test_feature_gate_margin.py`，产物为 `tmp/feature-gate-margin-code-pattern-current-v1/`。

若要查看每个候选的观测规则路径，可运行 `rst19-feature-gate-route`。该命令只读源级 CSV，输出逐候选路径表和按类别汇总表，将拒绝候选分成 `rejected_numeric_only`、`rejected_structural_only` 和 `rejected_numeric_and_structural`；数值同义旗标不会在相应数值字段已越界时重复计入结构旗标。当前 pair `82931/82934` 均为 `rejected_structural_only`，但这仍是 detector-rule 分解，不是反事实因果消融、噪点概率或物理恒星身份。产物为 `tmp/feature-gate-route-code-pattern-current-v1/`，实现为 `src/rst19/feature_gate_route.py`，测试为 `tests/test_feature_gate_route.py`。

若要把质量门路径与 15 帧逐源响应继续交叉，可运行 `rst19-feature-gate-temporal-route`。该命令只读取路径 CSV、非紧凑诊断源 CSV 和同一次序列 JSON，按 `detection_id` 连接候选 `presence`、同诊断子组 `presence` 和邻域质量 `presence`；缺少逐源行保留空值，被拒候选缺行则报错。当前 `82931/82934` 为 `4/15、4/15、0/15` 与 `4/15、3/15、0/15`，说明候选重复不等于稳定质量身份；输出不提供噪点概率、precision、完备率或物理恒星数。产物为 `tmp/feature-gate-temporal-route-code-pattern-current-v1/`，实现为 `src/rst19/feature_gate_temporal_route.py`，测试为 `tests/test_feature_gate_temporal_route.py`。
该命令还写出 `feature_gate_temporal_subgroup_summary.csv`，将 `masked_partial`、`blend_unresolved`、`weak_low_flux_snr` 等诊断子组的规则路径、候选持久、同机制持久和邻域质量持久并列。当前 `masked_partial` 候选为 `9,048/11,289`、同机制仅 `9/11,289`，`blend_unresolved` 为 `509/905`、`196/905`；这只是 detector-level 的复核路由，不是噪点率或物理恒星身份。
追加 `--target-id` 可写出 `feature_gate_temporal_targets.csv`，报告目标在所属子组中的含并列值经验 `le/ge` 位置；当前 `82934` 的候选 `4/15` 在 `blend_unresolved` 的低持久端（`le/ge=9.4%/94.7%`）。它只用于复核排序，不是噪点概率或显著性检验。

若要把类别代表带入 15 帧原始图复核，可在 hard-negative 命令后运行 `rst19-forced-stability`，显式追加代表 ID，并使用同一 `sequence_feature_persistence.json` 的累计平移。该复核输出固定位置/局部峰的逐帧通量 SNR、支持、形态、重定位和 `quality_like`；`quality_like` 不是源级 `quality_passed` 的覆盖，也不是恒星概率。当前代表产物为 `tmp/feature-hard-negative-forced-stability-current-v1/`。

若要继续检查“某类是否集中在边缘/单个 detector 单元，以及 hard-negative 周围到底有哪些候选”，可运行：

```powershell
rst19-feature-spatial-context `
  tmp/source-quality-audit-code-pattern-current-v3/source_catalog.csv `
  --target-id 82931 --target-id 82934 `
  --top-n 5 --grid-size 4 --edge-margin-px 16 `
  --high-flux-snr 10 `
  --out-dir tmp/feature-spatial-context-code-pattern-current-v1
```

该命令只读取源级表，输出每类的网格熵、最大单格占比、边缘比例、高 SNR 落选热点，以及 top hard-negative/显式目标的最近邻、`5/10/20 px` 邻域和类别组成。它把“线状集中”“拥挤全场分散”“截图 pair 是否在边缘”等问题分开；空间集中和近邻都只是 detector-level 线索，不是伪影真值、双星概率或星表身份。当前结果显示 `linear_artifact` 的 `96/96` 个候选在同一 `4×4` 单元，而 `crowded_blend` 覆盖 `16/16` 单元；`82931/82934` 的边缘距离约 `73.8/73.2 px`，最近候选距离 `2.738 px`，两者都落在 `r3c2`，因此截图不是边缘截断，却确实是一个局部近邻/混合问题。产物为 `tmp/feature-spatial-context-code-pattern-current-v1/`。

需要做特征分层控制实验时运行：

```powershell
rst19-injection --feature-audit --out-dir tmp/feature-audit
```

该命令对孤立/弱源、宽/窄 PSF、3/6 px 双源、单像素尖峰、掩膜、边缘、饱和和长线阴性样本输出 `feature_audit.csv`、`feature_audit.json` 与回收率曲线。正样本记录已知坐标的候选/质量回收率，阴性样本记录特征邻域的候选/质量泄漏；它们是合成机制控制，不是当前 15 张 FITS 的真实伪影比例。

对截图式近邻双峰同时控制强弱比和 PSF 归一化：

```powershell
rst19-injection --pair-flux-ratio-audit `
  --real-fits doc/00-项目资料/原始数据/20260330163205413_9901.fits `
  --total-peak-levels 1867 --secondary-ratios 0.25 0.125 `
  --pair-signal-normalization integrated_excess `
  --trials 2 --pairs 8 --psf-fwhm 2 --min-distance 4 `
  --min-psf-support-pixels 3 --psf-model gaussian `
  --out-dir tmp/pair-integrated-1867-gaussian-gui-default-t2-n8
```

`peak_excess` 固定两源峰值超额之和；`integrated_excess` 固定离散注入小窗的积分超额，避免经验核和 Gaussian 的核和差异混入比较。输出同时给出单源召回、成对一对一解析率和成对质量解析率；后两者不能把一个中间峰重复计数。当前 `2×8` 对/条件的结果只作为 PSF/强弱比机制证据，不修改 GUI 默认，也不等同真实双星比例或正式完备率。

对 `crowded_blend` 继续做多源组审计：

```powershell
rst19-injection --crowded-blend-audit `
  --real-fits doc/00-项目资料/原始数据/20260330163205413_9901.fits `
  --blend-sizes 2 3 --blend-separations 3 5.4 6 `
  --blend-total-levels 512 --trials 1 --blend-groups 4 `
  --psf-fwhm 2 --proposal-mode hybrid `
  --blend-signal-normalization integrated_excess `
  --out-dir tmp/crowded-blend-audit-integrated-gui-default
```

该命令按组记录单源召回、组级一对一完整解析、合并率和组邻域额外候选率；`merged` 不等于误检，`extra` 也不等于 precision。它用于区分 PSF 重叠造成的合并和弱副源造成的质量层失败，不改变 GUI 默认结果。将总信号改为 `1024` 并替换输出目录可复现第二档对照。

对真实 15 帧按首帧特征类别做跨帧持久性审计：

```powershell
rst19-feature-sequence doc/00-项目资料/原始数据 `
  --out-dir tmp/sequence-feature-persistence-gui-default
```

默认使用完整源级返回、质量源配准、`1 px` 首帧邻域和 `12/15` 持续门槛，输出 `sequence_feature_persistence.csv`、逐帧类别表、`sequence_feature_temporal_profile.csv`、`sequence_feature_class_transition.csv`、`sequence_feature_method_frame_summary.csv`、`sequence_feature_method_profile.csv`、`sequence_feature_source_subgroup_persistence.csv`、`sequence_feature_diagnostic_subgroup_persistence.csv`、JSON 和曲线。方法逐帧表把每个首要类别拆成三路共同支持、Gaussian-only、DoG-only、部分组合、无 Gaussian 和无提议器；方法聚合表再给出 15 帧总数及比例，用于检查来源分歧是否稳定，但它描述的是 detector-level 提议，不是物理目标身份。来源子组表只对首帧 `compact_quality` 的来源子组做逐源持久性统计，进一步区分 DoG-only 近邻双 PSF 记录与非近邻尺度响应；新增诊断子组表对其他首要类别按边界/掩膜、尖峰/支持、弱背景、拥挤、线状、范围和形状拆成互斥子组，并同时报告位置重复与同机制重复，后者要求响应帧仍属于同一诊断子组。持久性表把候选邻域与质量邻域分开；时间剖面表另外报告每类在 15 帧中的候选占比 CV、质量计数和活跃帧数；类别转移表再把“任意类别响应”和“同类响应”拆开，检查邻域命中是否只是检测器类别切换。注意 `compact_quality` 本身按 `quality_passed=True` 定义，因此该表的质量比例不是独立真阳性率；所有表都不是星表/WCS 身份匹配，不能替代星表、注入回收或人工真值。传入 `--max-sources 6000` 可做快速工作集抽测，但结果不能与全量质量率混用。

对上述首帧源表和 15 帧持久性 JSON 做分层强制测光控制：

```powershell
rst19-forced-stability `
  tmp/source-quality-audit-code-pattern-current-v3/source_catalog.csv `
  tmp/sequence-feature-persistence-code-pattern-current-v3/sequence_feature_persistence.json `
  --base-dir . --anchor-coordinate peak `
  --local-peak-search-radius 1 `
  --out-dir tmp/forced-stability-audit-code-pattern-current-v8-peak
```

该命令只在预测注册位置回到未降噪原图测量，不重跑候选、不写检测缓存；它输出每类最多 32 个锚点的固定位置逐帧通量 SNR、支持像素、FWHM、椭圆率和质量样式出现帧数。默认还在预测位置周围最多 `±1 px` 用三尺度 Gaussian 匹配响应选局部位置，并输出 `local_peak_*` 与 `relocalization_*`；固定值是主控，局部最大值有多位置选择偏差，只用于混合/配准诊断。传 `--local-peak-search-radius 0` 可关闭该层。`peak` 是与序列提案一致的默认锚点，`--anchor-coordinate centroid` 只用于中心敏感性对照，不能把任一输出解释为星表身份或真星率。完整限制见 [`检测器实验记录`](../02-星图识别/检测器实验记录.md) 8.58/8.59。
若要把某个截图或反例追加到分层样本，可重复传入 `--include-detection-id <ID>`；例如 `82931`/`82934` 的 pair 控制会保留原有分位点样本并额外输出这两个源。
当显式追加两个或更多 ID 时，还会写出 `forced_stability_pair_frame_metrics.csv` 和 `forced_stability_pair_summary.csv`，记录注册后局部相对间距、收缩比例和向内偏移比例；这些字段只用于共享结构风险审计，不是双星或星表身份判据。

将单帧特征汇总与 15 帧持久性汇总连接成跨类别反例报告：

```powershell
rst19-feature-cross `
  tmp/source-feature-audit-gui-default/source_feature_summary.csv `
  tmp/sequence-feature-persistence-gui-default/sequence_feature_persistence.csv `
  --class-transition-csv tmp/sequence-feature-persistence-gui-default/sequence_feature_class_transition.csv `
  --out-dir tmp/feature-cross-audit-gui-default
```

该命令输出 `feature_cross_audit.csv`、JSON 和候选/质量持久性对照图，自动标记“高 SNR 但质量层拒绝”、候选持久性与质量持久性之间的差距，以及类别转移表中的任意响应/同类响应率和最大跨类流向。`--class-transition-csv` 是可选的，但提供后必须是完整的候选/质量 `类别×类别` 矩阵；命令只连接已生成的审计表，不重新检测 FITS。质量邻域命中不是星表身份，SNR 和同类响应率也不是物理真值。

针对局部“识别到两颗”的近邻响应，可运行固定 detector 坐标审计：

```powershell
rst19-pair-audit doc/00-项目资料/原始数据 `
  --primary-x 1269 --primary-y 3465 `
  --secondary-x 1274 --secondary-y 3467 `
  --out-dir tmp/source-pair-audit-screenshot
```

命令输出逐帧源级候选/质量状态、原始孔径中的重复值、精确 `-1`、极端正负码和单/双 PSF `ΔBIC`；固定 detector 坐标只用于伪影/编码结构审计，不是注册坐标或星表匹配。默认坐标对应本次截图复核，其他近邻应显式传入坐标；结果不能把两个局部峰直接写成两颗物理恒星。

若要继续拆解“两个框”在匹配滤波中由哪些像素支撑，可运行局部响应反事实归因：

```powershell
rst19-pair-response-attribution `
  doc/00-项目资料/原始数据/20260330163205413_9901.fits `
  --source-catalog tmp/source-quality-audit-code-pattern-current-v3/source_catalog.csv `
  --primary-id 82931 --secondary-id 82934 `
  --psf-fwhm 2 --negative-anomaly-threshold-adu -1000 `
  --out-dir tmp/pair-response-attribution-code-pattern-current-v1
```

该命令在同一局部窗口比较 raw、重复码屏蔽、极端负值屏蔽和联合屏蔽的 Gaussian 响应，并输出目标响应、局部响应峰及普通像素/重复码/负异常的带符号核归因。它用于解释候选分裂机制，不改默认质量层、GUI 或缓存；响应贡献不是噪点概率、FDR、物理星数或伪影率。

如果要区分“屏蔽特殊像素后消失”与“特殊像素缺口改变了响应形状”，可运行局部补值反事实：

```powershell
rst19-pair-repair-counterfactual `
  doc/00-项目资料/原始数据/20260330163205413_9901.fits `
  --source-catalog tmp/source-quality-audit-code-pattern-current-v3/source_catalog.csv `
  --primary-id 82931 --secondary-id 82934 `
  --patch-padding-px 128 --replacement-radius-px 1 `
  --target-match-radius-px 4 --negative-anomaly-threshold-adu -1000 `
  --threshold-sigma 4 --min-distance 4 --aperture-radius 4 `
  --psf-fwhm 2 --background-box-size 128 --min-flux-snr 5 `
  --min-psf-support-pixels 3 --proposal-mode hybrid `
  --out-dir tmp/pair-repair-counterfactual-code-pattern-current-v1
```

命令只在内存副本中用局部中位数替换重复码或极端负值，然后重跑同一 detector；原始 FITS 不会被覆盖。输出 `pair_repair_summaries.csv`、`pair_repair_candidates.csv`、`pair_repair_replacements.csv` 和 JSON 解释。补值是 detector-level 敏感性控制，不是坏像素标定、天空值重建或物理星数结论；目标 pair 的当前结果为 raw `2` 个目标窗口候选，仅修复重复码仍 `2` 个，修复负异常后 `1` 个。

若要把首帧目标按 15 帧的累计平移带入注册 detector 坐标，可复用序列分析输出的 `cumulative_shifts`：

```powershell
rst19-pair-audit doc/00-项目资料/原始数据 `
  --primary-x 1269 --primary-y 3465 `
  --secondary-x 1274 --secondary-y 3467 `
  --shifts-json tmp/sequence-current-shared-bg-verified.json `
  --out-dir tmp/source-pair-audit-registered
```

`--shifts-json` 只做 detector 平面平移，不是 WCS/星表身份匹配；它用于避免把真实静态星误判为“固定坏点”，同时仍需用原始像素、PSF、星表和人工注入完成最终确认。

对 15 帧做 detector 固定码和低变化值审计：

```powershell
rst19-temporal-codes doc/00-项目资料/原始数据 `
  --low-variation-span-adu 2 `
  --code-min-adu 3990 --code-max-adu 3993 `
  --sentinel-value -1 `
  --focus-x 1274 --focus-y 3466 `
  --sentinel-focus-x 1271 --sentinel-focus-y 3465 `
  --out-dir tmp/temporal-code-audit
```

该命令只比较原始 FITS 的跨帧 detector 坐标，输出 `temporal_code_audit.json`、`temporal_code_focus_series.csv` 和 `temporal_code_components.csv`；焦点时序表保留每一帧关注坐标的原始值，便于复核固定正码、`-1` 或其它异常值是否与候选共位。它不修改像素、不改变候选/质量层，也不能把低变化簇直接定义为坏点或恒星。

预览支持鼠标滚轮缩放（最大 `15×`）和左键拖拽平移。在“可信星点 / 全部候选”层将鼠标移到候选点附近时，底部信息栏和图像标注会显示检测 ID、坐标、峰值、通量、通量误差、三类 SNR、FWHM/椭圆率、仪器星等和质量标记；在“运动候选”层悬停洋红线、橙色线或蓝色环会显示轨迹 ID、当前帧坐标、残差 SNR、线长/线宽/方向、出现帧数、总位移、速度和拟合 RMS。完成单帧检测后，从检测参数区的“研究工具”菜单选择“导出星点研究表”，即可从当前结果生成全量源 CSV、质量标记汇总和两张 SNR 图，不会再次运行检测。检测参数区固定显示序列进度条和 F01…F15 帧状态带，右侧证据卡同步显示总体百分比、已完成帧数、当前帧号和阶段；底部状态栏保留短的确定性进度条，因此在 1120×720 最小窗口仍有可见反馈。单图长轨迹筛选、15 帧共享背景准备、逐帧检测、时间中值参考、残差和轨迹拟合都会持续推进。悬浮信息与右侧说明按各自容器动态换行，窗口变矮时次要源表自动收起，完整表格仍可从“研究工具”打开，不会再由长文本或固定表格列宽挤走右侧结果面板。

预览支持鼠标滚轮缩放（最大 `15×`）和左键拖拽平移。GUI 保留当前 4096² 传感器空间分辨率，15× 放大不会再把 1600² 缩略图二次放大；显示层使用稳健 IQR 噪声估计设置背景底、gamma 1.15 和 0.6 px 轻度高斯去噪，避免背景被抬成整幅灰噪声，同时保留弱源与亮源层次；这些只改变 PNG 预览，不改变 FITS ADU、SNR、测光或星等。质量/候选标记在缩小总览时使用单像素证据点，放大后才扩成小方框，避免数万标记覆盖原图。检测参数区的“人工调参”提供候选阈值和可信通量 SNR 滑块，拖动只修改待运行参数，点击“应用并分析”后才启动当前帧重算；完成后可选择“保留弱星 / 减少伪点 / 当前平衡”，把已实际运行的参数、计数和拒绝标志追加到 `tmp/manual-threshold-feedback.jsonl`，供后续离线调参读取，不宣称在线自动训练。鼠标移到候选点附近时，底部信息栏和图像标注仍会显示检测 ID、坐标、峰值、通量、通量误差、三类 SNR、FWHM/椭圆率、仪器星等和质量标记。

## 显示层与悬停证据

主图提供三种互不改变检测结果的观察层：`增强显示 · 轻度降噪` 适合总览，`原始显示 · 未降噪` 直接查看未做平滑的 FITS 像素，`增亮噪声 · 看弱点` 用更宽的背景范围展示噪声和弱点。三种图像在切换帧时一起生成并缓存，切换显示层不会重新检测，也不会修改 FITS、SNR、测光或星等。

在“可信星点”层可检查通过质量筛选的源；切换到“全部候选”后，将鼠标移到任意候选点，可在底部悬浮详情中查看通量、通量误差、peak/flux/filter 三类 SNR、FWHM、椭圆率、PSF 支持像素、质心偏移、仪器星等、判定和落选原因。图内短标签使用 ASCII 字段（例如 `FSNR 6.2 | REJECT/MASKED`），并优先使用 Windows 中文字体绘制，避免截图中出现 `□□` 或 `��`。

落选原因由检测器实际写入的质量标记翻译而来，包括边缘、坏点/掩膜、饱和、线状伪迹、背景不确定、非正通量、低 SNR、PSF 支持不足、形状异常和小面积足迹；这只是当前参数下的质量判定，不把“亮点”直接等同于恒星。

## 测试

当前全量回归为 `492` 项，覆盖分层抽样、峰/质心锚点切换、指定源追加、汇总统计、局部匹配响应选择、pair 相对几何、经验 PSF 留一法、类别证据矩阵及其确定性复核路由、pair 独立像素支持留出、原始像素拓扑审计、空间分区留出注入、经验 PSF 孔径敏感性、类别参数敏感性转移矩阵、序列参数比较器、类别内经验分位审计、类别内高显著性落选审计、hard-negative 空间条件化审计、固定码焦点时序导出、污染局部背景双源注入、跨类别污染双源复核、端点级质量原因记录、旗标交互审计、质量门余量审计、质量门逐候选路径审计、质量门路径—跨帧响应交叉审计、局部 ROI 坐标恢复和计数范围回归、两种序列 shift JSON 兼容口径记录以及注册合成大图的 footprint、覆盖数、掩膜、透明预览、缓存和导出回归，并覆盖公共 Gaia 查询解析、可选 GSP-Phot `M_G` 查询/目录往返/结果层传递、Gaia 大视场分块完整性审计、星等质量门、GUI 最终入口审计、星对几何 WCS 候选、创新报告证据摘要、Bailer--Jones 距离别名与来源推断和条件化创新交付包。

```powershell
python -m pytest
```

## 相关资料

- [星表路线深度研究](../doc/02-星图识别/星表路线深度研究.md)
- [检测器实验记录](../doc/02-星图识别/检测器实验记录.md)
- [FITS 读取与校验](../doc/01-数据解析/FITS读取与校验.md)
- [星点识别方法](../doc/02-星图识别/星点识别方法.md)
- [论文与答辩材料](../doc/05-论文答辩/论文与答辩材料.md)
- [15 帧注册合成大图](../doc/04-创新分析/15帧注册合成大图.md)

界面行为更新（2026-08-31）：15 帧分析完成后主图默认显示“稳定星场”，按当前 FITS 路径映射并绘制当前帧实际存在的 `static/persistent` 点源；“运动候选”仍保留为单独图层，需要查看运动证据时手动切换。稳定层在缩小总览时保留可见的实心中心，并在状态栏报告当前帧点数。
