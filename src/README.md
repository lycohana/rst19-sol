# rst19 实现模块

这是 rst19-sol 的第一版 Python 算法实现，负责把资料中的星表路线落成可测试的本地 Python 基线。

## 定位

模块当前覆盖：

- 当前比赛 FITS 主 HDU 的读取、结构校验和 26 个辅助字段解码；
- 屏蔽首行辅助区域后的鲁棒背景/噪声估计；
- 局部背景/RMS、Gaussian PSF 匹配滤波、3×3 PSF 支持反尖峰审计、全量候选审计、孔径通量误差、三种 SNR 和点源质量分层；
- 基于切平面先验 WCS 的星表一对一匹配；
- CSV 离线任务星表读取、自行传播和 JSON CLI 输出。
- Python Tkinter 桌面工作台：选择 FITS、调整参数，在“可信星点 / 全部候选 / 运动候选”图层间切换，并查看最暗源标注。
- 15 帧质量源的全局平移配准、唯一轨迹关联和 `static`/`moving`/`transient` 点轨迹基线；另有独立的线状候选检测与跨帧关联。
- 单帧长线候选检测和 `rst19-innovation` 证据导出：按真实 `DATE-OBS` 生成逐帧/逐轨迹 JSON、CSV 和 PNG，区分单帧候选与跨帧 `moving`。

模块当前不声称已经完成：

- 完全盲的 plate solving 或 Astrometry.net 索引生成；
- 经过真实标定的绝对星等、`Mv` 转换和检测完备率；当前只输出仪器星等 `m_inst`；
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

分析结果中的 `faintest_detected` 是通过局部通量 SNR、正通量、点源形状以及边缘/掩膜/饱和质量筛选后的最暗可信候选源。其 `instrumental_magnitude` 按 `m_inst = -2.5 log10(flux_rate)` 计算；只有提供经过验证的 `--zero-point` 时才会附带 `calibrated_magnitude`，不能在未标定时直接称为 Gaia V 或 `Mv`。`snr` 是峰值 SNR，`flux_snr` 是孔径通量 SNR，`filter_snr` 是匹配滤波 SNR，三者语义不同。

分析 15 帧：

```powershell
rst19-sequence doc/00-项目资料/原始数据 `
  --threshold-sigma 4 --min-distance 4 --aperture-radius 4 `
  --psf-fwhm 3 --min-flux-snr 5 --min-psf-support-pixels 3 --min-presence 12 `
  --motion-min-displacement-px 2 --max-motion-fit-rms-px 0.75
```

序列默认使用有界工作集：每帧保留完整 `candidate_count`，但按匹配滤波 `filter_snr` 取前 6000 个源进行源级测量和点轨迹关联，并用 `background_box_size=256`、共享局部背景/RMS 图减少重复统计。序列网格背景使用两轮稳健裁剪，单图仍使用四轮裁剪与环形局部精修；背景网格的线性插值直接输出目标尺寸，并缓存重复孔径几何，避免 4096² 临时数组和数千次小数组分配。对当前 16 位 FITS，序列还启用 `use_float32` 像素中间阵列；它能精确表示输入 ADU，只降低 4096² 阵列的内存带宽，单图科学测光仍默认使用 float64。默认使用 4 个 worker 并行检测不同帧，结果按原始帧序恢复；`--workers 1/2/3` 可降低峰值内存，`--workers 4` 是当前机器的吞吐档位。`returned_count`、`quality_count` 和点轨迹数量是工作集口径；单图检测仍默认不截断且保留精细局部背景。需要全量序列工作集时传 `--source-limit 0`，需要对照实验时可传其他正整数。序列还提供 `static/persistent/transient` 分层和原图时序小窗审计，持续候选默认门槛为 `8/15`，严格静态默认门槛为 `12/15`；这两个序列计数都不是官方逐星真值。
序列默认使用有界工作集：每帧保留完整 `candidate_count`，但按匹配滤波 `filter_snr` 取前 6000 个源进行源级测量和点轨迹关联，并用 `background_box_size=256`、共享局部背景/RMS 图减少重复统计。序列网格背景使用两轮稳健裁剪，单图仍使用四轮裁剪与环形局部精修；背景网格的线性插值直接输出目标尺寸，并缓存重复孔径几何，避免 4096² 临时数组和数千次小数组分配。对当前 16 位 FITS，序列还启用 `use_float32` 像素中间阵列；它能精确表示输入 ADU，只降低 4096² 阵列的内存带宽，单图科学测光仍默认使用 float64。默认使用 4 个 worker 并行检测不同帧，结果按原始帧序恢复；`--workers 1/2/3` 可降低峰值内存，`--workers 4` 是当前机器的吞吐档位。`returned_count`、`quality_count` 和点轨迹数量是工作集口径；单图检测仍默认不截断且保留精细局部背景。需要全量序列工作集时传 `--source-limit 0`，需要对照实验时可传其他正整数。序列还提供 `static/persistent/transient` 分层和原图时序小窗审计，持续候选默认门槛为 `8/15`，严格静态默认门槛为 `12/15`；这两个序列计数都不是官方逐星真值。

### GUI 默认分析口径（2026-09-02）

`rst19-gui`/`rst19-ui` 的单帧和 15 帧按钮共用当前默认：`threshold=4σ`、`min_distance=4 px`、`PSF FWHM=2 px`、通量 `SNR≥5`、`proposal_mode=hybrid`、15 帧弱星提案为 `median`、参考图 `SNR≥15`、普通候选共识 `15σ`、逐帧补测 `7.5σ`、时序 PSF 相关 `≥0.80`，并在预测点 `±1 px` 内做局部峰重定位。时序多尺度、局部去混叠和 15 帧全量关联默认关闭；它们仍是可显式开启的审计/性能选项。该默认偏向在可控耗时内减少固定 `FWHM=3 px` 和单一 Gaussian 对拥挤/失配源的漏检，但 `persistent`/`quality` 仍是质量定义下的候选层，不等同官方恒星真值。真实对照数字和选择理由见 [检测器实验记录](../doc/02-星图识别/检测器实验记录.md) 的 2026-09-02 小节。

单帧质量层默认要求候选峰中心 3×3 核心至少 3 个像素超过 `max(2×局部噪声, 0.1×峰值超额)`；不满足时仍保留候选并记录 `INSUFFICIENT_PSF_SUPPORT`，避免把孤立单像素脉冲当成点源。该约束不改变原始 ADU、通量或星等，只影响质量层。当前快速 15 帧序列还对时间补提案在预测点 `±1 px` 内做局部峰重定位，再执行 PSF 相关和跨帧细筛；同一真实数据的最新口径为严格静态 `5,276`、持续候选 `15,524`、稳定/持续合计 `20,800`。早期全量工作集的 `9,938 / 6,249 / 16,187` 仍保留在实验记录中作为历史对照，不能与当前结果混写；持续候选仍需星表/人工复核。

当前质量层还会审计有符号整型 FITS 中的“候选峰本身落在全幅异常重复高位码上 + 孔径含异常负值”模式，记录为 `CODE_PATTERN`。它只把受污染候选降级，不删除宽筛候选；仅“孔径内含重复码”不足以拒绝，否则真实亮星的饱和/溢出出血列会把 `3991/3992` 等重复码和 `-20628` 这类负值同时带进孔径而被误拒。当前首帧 `hybrid + FWHM=2` 为 `84,594` 个候选、`29,263` 个质量源、`12` 个 `CODE_PATTERN`。这个规则不是相机满阱、`BLANK` 或坏像素的替代标定；详细局部复核见 [检测器实验记录 8.38](../doc/02-星图识别/检测器实验记录.md) 和 [真实性研究 10.25](../doc/02-星图识别/亮点真实性与伪影判别研究.md)。

从序列 JSON 导出创新分析证据：

```powershell
rst19-innovation tmp/sequence-final.json --out-dir tmp/innovation
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
  --stratified-real --strata blank high_background edge crowded special_code `
  --peak-levels 24 56 128 --trials 1 --sources 8 --psf-fwhm 2 `
  --proposal-mode hybrid --psf-model empirical `
  --out-dir tmp/stratified-real-background-gui-default-n8
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

主界面的“打开星表核验”会复用这套 CSV/WCS 参数并显示唯一匹配、位置残差和目录星等；运行匹配后可点击“根据匹配拟合 WCS”，用至少 6 个有效、非共线匹配点估计视场内的局部仿射模型，展示像元角尺度、旋转、parity、内点 RMS 和各向异性，并把结果用于创新摘要的视场切平面角速度。它仍以先验 WCS 为入口，不是全天空盲解算；未填写可靠像元角尺度时不要运行匹配。CSV 至少需要以下列：

```text
source_id,ra_deg,dec_deg,magnitude,pmra,pmdec,ref_epoch
```

`--pixel-scale-arcsec`、旋转角和 parity 是相机参数假设，不是当前数据已经核验的事实；应通过稳定匹配星、残差和留出星验证后再固定。

格式说明中的 26 个辅助字段还可以先提供逐帧的光轴中心先验。对本地 15 张真实 FITS 运行：

```powershell
rst19-aux-audit doc/00-项目资料/原始数据 --out-dir tmp/auxiliary-attitude-audit
```

该命令把四元数按向量在前、标量在后的 Hamilton 约定转换为主动旋转，并以相机机体系 `-Y` 轴计算 RA/Dec；实际 15 帧中四元数模长均为 1，计算光轴与辅助 RA/Dec 的最大差为约 `1.1e-10` 角秒，`p_az/p_el` 与 FITS 头匹配 `15/15`。这证明辅助数据可以用于按帧收窄星表候选搜索和检查坐标约定，但当前头文件仍缺少像元尺度、完整轴向矩阵、畸变/SIP 等信息，所以它是“光轴/姿态先验”，不是完整 WCS，也不能单独给出逐星身份或正式星数。输出为 `auxiliary_boresight_audit.csv` 和 `auxiliary_boresight_audit.json`。

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

界面启动和切换帧时只载入图像预览，不会提前计算检测结果。点击“分析当前帧”后，程序先用轻量背景/连通域基线寻找长线候选，橙色候选会先进入运动层；随后继续全量星点、SNR、形状和最暗源星等精测，因此单帧形状不会被误称为 `moving`，精测完成后会用同一候选口径收敛结果。结果会按 FITS 文件路径、文件修改状态和检测参数写入项目根目录的 `.rst19-cache/` 压缩缓存；相同输入再次分析时直接复用。点击“分析 15 帧动目标”后，程序对整个目录做平移配准和轨迹关联，完成后自动切换到“运动候选”层；该层不绘制静态星场：洋红线表示跨帧拟合通过的线状 `moving` 目标，橙色线表示单帧或尚未达到跨帧证据门槛的线状 `candidate`，蓝色环表示严格 `moving` 点轨迹。切换帧时右上角的图层选择会保持不变，旧线程结果会因帧令牌失效而丢弃，过期任务也不会把旧结果写入当前显示状态。点击“清空检测缓存”会删除当前 gzip 缓存和旧版本 JSON 产物，即使任务正在运行也可以执行；本次任务完成后不会把清理前的结果写回缓存，原始 FITS 不会被删除。

序列结果的“稳定星场”层只显示跨帧持续的 `static/persistent` 点源，分别用青绿色/蓝色区分严格静态与低置信持续候选；悬停可查看出现帧数、通量 SNR、配准位移和拟合 RMS。15 帧证据窗口的“图证”页提供“星点时序审计”按钮，后台从原始 FITS 导出 `source_track_contact_sheet.png`、`source_track_audit.csv` 和汇总 JSON，灰框代表缺帧，不把颜色或抽样图当作真值。单帧源表同时保存 `peak_x/peak_y`、PSF 加权质心 `x/y` 和 `centroid_shift_px`，便于检查弱源标记是否被噪声拉偏。

命令 `rst19-trails <FITS目录>` 可以逐张运行单图长线审计，输出候选数量、最长线长度/宽度/方向、残差 SNR、触边状态和包围盒的 CSV 与曲线；GUI 的“单图长线”页按当前检测参数按需运行同一审计。它只回答“单张图中是否存在长线形状候选”，不跨帧判定 `moving`；15 帧速度和方向仍以序列证据为准。

命令 `rst19-sources <FITS文件>` 会在一次检测完成后导出全量 `source_catalog.csv`、质量/拒绝标记汇总、互斥首要特征类别 `source_feature_summary.csv`、`source_feature_morphology.csv`、按类别 4×4 粗网格统计的 `source_feature_spatial.csv`、按类别原图抽样的 `source_cutout_contact_sheet.png`/`source_cutout_manifest.csv`、按类别经验 PSF 相似度的 `source_feature_psf_similarity.csv`、全局/局部经验 PSF 留出对照 `source_feature_psf_spatial.csv`、类别图、两张 SNR 图、重叠质量标志图以及 16×16 返回源密度/质量通过率空间图和 `source_spatial_grid.csv`。形态表按类别记录峰值/通量/滤波 SNR、FWHM、椭圆率、sharpness、PSF 支持、质心偏移、空间分位数和拒绝原因比例；空间表用于检查线状、边缘、固定结构是否集中在局部区域；PSF 相似度表记录确定性原图抽样的经验核相关系数、相对残差和中心能量占比；空间 PSF 表用留出样本构造局部模板，并用整幅候选表做隔离检查，模板不足时只记录回退，不改写 `quality_passed`。高 SNR 拒绝计数、空间集中性、`corr≥0.8` 和局部模板可得率都只是诊断切片，不是物理伪影真值，也不会自动改写质量层。它保留 `peak SNR`、孔径 `flux SNR` 与匹配滤波 `filter SNR` 三个不同量，并明确记录当前背景、噪声、阈值、PSF、质量参数和空间诊断边界；`feature_class` 只是按固定优先级整理的算法审计标签，原始 `flags` 仍允许重叠。GUI 检测参数区的“研究工具 → 导出星点研究表”复用当前已完成的检测结果，不会因为导出而再次检测，导出完成后直接打开 Tkinter 的“论文表图 / 指标与口径”窗口。

空间 PSF 诊断默认使用 `4×4` 网格、每类 8 个抽样源，以保持日常导出速度；进行类别研究时可以显式扩大抽样：

```powershell
rst19-sources <FITS文件> --psf-fwhm 2 `
  --spatial-psf-per-class 16 --spatial-psf-grid-size 2 `
  --out-dir tmp/source-feature-psf-grid2-per16-gui-default
```

这两个参数只影响 `source_feature_psf_spatial.csv` 的诊断抽样，不改变候选数、质量规则或 GUI 默认结果。

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

默认使用完整源级返回、质量源配准、`1 px` 首帧邻域和 `12/15` 持续门槛，输出 `sequence_feature_persistence.csv`、逐帧类别表、`sequence_feature_temporal_profile.csv`、`sequence_feature_class_transition.csv`、JSON 和曲线。持久性表把候选邻域与质量邻域分开；时间剖面表另外报告每类在 15 帧中的候选占比 CV、质量计数和活跃帧数；类别转移表再把“任意类别响应”和“同类响应”拆开，检查邻域命中是否只是检测器类别切换。注意 `compact_quality` 本身按 `quality_passed=True` 定义，因此该表的质量比例不是独立真阳性率；所有表都不是星表/WCS 身份匹配，不能替代星表、注入回收或人工真值。传入 `--max-sources 6000` 可做快速工作集抽测，但结果不能与全量质量率混用。

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

该命令只比较原始 FITS 的跨帧 detector 坐标，输出 `temporal_code_audit.json` 和 `temporal_code_components.csv`；它不修改像素、不改变候选/质量层，也不能把低变化簇直接定义为坏点或恒星。

预览支持鼠标滚轮缩放（最大 `15×`）和左键拖拽平移。在“可信星点 / 全部候选”层将鼠标移到候选点附近时，底部信息栏和图像标注会显示检测 ID、坐标、峰值、通量、通量误差、三类 SNR、FWHM/椭圆率、仪器星等和质量标记；在“运动候选”层悬停洋红线、橙色线或蓝色环会显示轨迹 ID、当前帧坐标、残差 SNR、线长/线宽/方向、出现帧数、总位移、速度和拟合 RMS。完成单帧检测后，从检测参数区的“研究工具”菜单选择“导出星点研究表”，即可从当前结果生成全量源 CSV、质量标记汇总和两张 SNR 图，不会再次运行检测。检测参数区固定显示序列进度条和 F01…F15 帧状态带，右侧证据卡同步显示总体百分比、已完成帧数、当前帧号和阶段；底部状态栏保留 220 px 确定性进度条，因此在 1120×720 最小窗口仍有可见反馈。单图长轨迹筛选、15 帧共享背景准备、逐帧检测、时间中值参考、残差和轨迹拟合都会持续推进。悬浮信息与右侧说明按各自容器动态换行，窗口变矮时次要源表自动收起，完整表格仍可从“研究工具”打开，不会再由长文本或固定表格列宽挤走右侧结果面板。

预览支持鼠标滚轮缩放（最大 `15×`）和左键拖拽平移。GUI 保留当前 4096² 传感器空间分辨率，15× 放大不会再把 1600² 缩略图二次放大；显示层使用稳健 IQR 噪声估计设置背景底、gamma 1.15 和 0.6 px 轻度高斯去噪，避免背景被抬成整幅灰噪声，同时保留弱源与亮源层次；这些只改变 PNG 预览，不改变 FITS ADU、SNR、测光或星等。质量/候选标记在缩小总览时使用单像素证据点，放大后才扩成小方框，避免数万标记覆盖原图。检测参数区的“人工调参”提供候选阈值和可信通量 SNR 滑块，拖动只修改待运行参数，点击“应用并分析”后才启动当前帧重算；完成后可选择“保留弱星 / 减少伪点 / 当前平衡”，把已实际运行的参数、计数和拒绝标志追加到 `tmp/manual-threshold-feedback.jsonl`，供后续离线调参读取，不宣称在线自动训练。鼠标移到候选点附近时，底部信息栏和图像标注仍会显示检测 ID、坐标、峰值、通量、通量误差、三类 SNR、FWHM/椭圆率、仪器星等和质量标记。

## 显示层与悬停证据

主图提供三种互不改变检测结果的观察层：`增强显示 · 轻度降噪` 适合总览，`原始显示 · 未降噪` 直接查看未做平滑的 FITS 像素，`增亮噪声 · 看弱点` 用更宽的背景范围展示噪声和弱点。三种图像在切换帧时一起生成并缓存，切换显示层不会重新检测，也不会修改 FITS、SNR、测光或星等。

在“可信星点”层可检查通过质量筛选的源；切换到“全部候选”后，将鼠标移到任意候选点，可在底部悬浮详情中查看通量、通量误差、peak/flux/filter 三类 SNR、FWHM、椭圆率、PSF 支持像素、质心偏移、仪器星等、判定和落选原因。图内短标签使用 ASCII 字段（例如 `FSNR 6.2 | REJECT/MASKED`），并优先使用 Windows 中文字体绘制，避免截图中出现 `□□` 或 `��`。

落选原因由检测器实际写入的质量标记翻译而来，包括边缘、坏点/掩膜、饱和、线状伪迹、背景不确定、非正通量、低 SNR、PSF 支持不足、形状异常和小面积足迹；这只是当前参数下的质量判定，不把“亮点”直接等同于恒星。

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

界面行为更新（2026-08-31）：15 帧分析完成后主图默认显示“稳定星场”，按当前 FITS 路径映射并绘制当前帧实际存在的 `static/persistent` 点源；“运动候选”仍保留为单独图层，需要查看运动证据时手动切换。稳定层在缩小总览时保留可见的实心中心，并在状态栏报告当前帧点数。
