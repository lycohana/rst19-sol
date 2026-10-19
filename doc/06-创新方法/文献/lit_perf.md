# 性能基准与工程可行性文献（G–K）

检索/核验日期：2026-10-19。所有条目均经 API 元数据核验（OpenAlex / arXiv / Crossref / crates.io）并做过 HTTP 可达性探测。
标记约定：OK = 探测返回 200；WARN = 出版方/站点拒绝自动抓取（403/405/timeout），DOI 与元数据已核验、浏览器可开；所有"定量结果"要么引自可核验摘要/全文，要么明确标注为推算或教科书恒等式。

---

## G. 检测阈值的统计标定

### [G1] Vio, R.; Andreani, P. (2016) — On the Correct Estimate of the Probability of False Detection of the Matched Filter in Weak-Signal Detection Problems
- 出处：A&A 589, A20
- 链接：https://arxiv.org/abs/1602.02392 [OK] ｜ DOI https://doi.org/10.1051/0004-6361/201527128 （出版方 WARN）
- 一句话思想：匹配滤波（MF）的"标准"虚警率算法默认噪声峰值服从高斯，而 MF 只在信号位置已知时最优；一旦位置/数目未知（图像检测的常态），高斯尾部会严重低估虚警概率，正确做法是用噪声峰值幅度 PDF（PAM）来定义阈值。
- 定量结果：给出 1D/2D/3D 通用的"峰值 PDF 求 PFA"方法，并在真实 ALMA 2D 干涉图上演示：原本"高显著"的峰在新算法下不能以可信置信度声明为源（摘要无单一数字）[核到摘要 + 正文首段]。
- 用途：论证"nσ 不是可移植阈值"的核心引文；把阈值从单点像素统计改写为随机场峰值统计。
- 局限：面向高斯、各向同性、平滑随机场；非高斯/强相关场景需扩展。

### [G2] Vio, R.; Verges, C.; Andreani, P. (2017) — ... II. Specific probability of false alarm
- 出处：A&A 604, A115
- 链接：https://arxiv.org/abs/1705.03285 [OK]
- 一句话思想：PFA 只是"整幅图里冒出假峰的概率"，回答不了"这一个峰有多可信"；提出特定虚警概率 SPFA，把全局阈值换算成单次检测的可靠度。
- 定量结果：在 ALMA / ATCA 干涉图上对若干候选点源给出逐源检测可靠度（方法+实测，摘要无单一数字）[核到摘要]。
- 用途：支持"报告每个检测的可靠度而非一个全局 nσ"这条方法学设计。

### [G3] Vio, R.; Andreani, P.; Biggs, A.; Hayatsu, N. (2019) — ... III. Peak distribution method versus the Gumbel distribution method
- 出处：A&A 627, A103
- 链接：https://arxiv.org/abs/1907.01465 [OK] ｜ DOI https://doi.org/10.1051/0004-6361/201834854 （出版方 WARN，正文 HTML 亦 403）
- 一句话思想：相关高斯随机场（GRF）的极值统计有两条路线——峰值幅度 PDF（PAM）与 Gumbel 极值分布（GDM）；两者数值结果几乎一致，但 PAM 更灵活且能自检适用条件（Pavesi et al. 2018 走的是 Gumbel 路线）。
- 定量结果：数值实验 + ALMA 干涉图实测，两法结果 "almost identical"；PAM 适用条件可检验 [核到正文摘段]。
- 用途："Gumbel 在图像检测里怎么用"的最直接可引条目，同时给出"选哪种极值模型"的判断依据。

### [G4] Zackay, B.; Ofek, E. O.; Gal-Yam, A. (2016) — Proper Image Subtraction: Optimal Transient Detection, Photometry, and Hypothesis Testing
- 出处：ApJ 830, 27
- 链接：https://doi.org/10.3847/0004-637X/830/1/27 [OK] ｜ https://arxiv.org/abs/1601.02655 [OK]
- 一句话思想：在背景噪声主导极限下推导相关噪声中瞬变检测/测光的最优统计量（闭式、数值稳定），并证明差分像残差为不相关白噪声、可作为后续任意统计检验的充分统计量。
- 定量结果：给出可计算的检测显著性与通量（含误差）的闭式表达；差分像噪声白化后可按高斯白噪声解读（文中定理，非单一数字）[核到摘要]。
- 用途：论证"相关噪声下必须先白化、再谈 nσ"；也是频域匹配滤波的理论依据。

### [G5] Bertin, E.; Arnouts, S. (1996) — SExtractor: Software for source extraction
- 出处：A&AS 117, 393
- 链接：项目页 https://www.astromatic.net/software/sextractor [OK] ｜ DOI https://doi.org/10.1051/aas:1996164 （出版方 WARN）
- 一句话思想：以"阈值 σ × 最小连通面积"双参数（DETECT_THRESH / DETECT_MINAREA）+ 多阈值反卷积去 blend 的工程范式，成为光学图像检测的事实标准。
- 定量结果：论文给出阈值化与去 blend 的参数体系及默认值（具体默认数值请以发行版文档为准，本条未逐字核对）[元数据已核验]。
- 用途：说明"nσ + 面积"这类工程阈值约定的来源；所有现代实现的对照基线。

### [G6] King, I. R. (1983) — Accuracy of measurement of star images on a pixel array
- 出处：PASP 95, 163
- 链接：https://doi.org/10.1086/131139 [OK]
- 一句话思想：以最小二乘拟合已知 PSF 为框架，给出等效噪声面积（equivalent noise area）经验规则——测光噪声不取决于星像覆盖多少像素，而取决于与 PSF 和像素尺度相关的等效面积。
- 定量结果：摘要明确给出 equivalent noise area 的近似经验公式 + 更严格结果的表格，并把像素尺寸相对像尺寸的影响显式纳入 [核到摘要]。
- 用途：关键。它是"相关噪声/重采样后光圈噪声该乘多大系数"的可引公式来源，也是把 nσ 与相关长度联系起来的经典入口。

### [G7] Adler, R. J.; Taylor, J. E. (2007) — Random Fields and Geometry
- 出处：Springer Monographs in Mathematics
- 链接：https://doi.org/10.1007/978-0-387-48116-6 [OK]
- 一句话思想：光滑随机场高阈值的期望峰数 / 欧拉示性数理论——超过 ν 的极大值个数随阈值呈 exp(-ν²/2) 衰减，但前置系数由搜索域体积与相关长度（导数协方差）决定。
- 定量结果：给出 E[峰数 > ν] 的通用 Euler characteristic 展开式，是"nσ 应写成 面积/相关长度²"的数学根据 [教科书结论，本条为公式来源引用，未逐字核对页码]。
- 用途：方法论证中解释"同一 nσ、不同相关长度 ⇒ 不同虚警数"的理论落点。

补充（标准数学恒等式，非引自上述论文，供量级判断）：
- 独立高斯像素单点尾部概率 p = 0.5 * erfc(ν/√2)：3σ -> 1.35e-3，5σ -> 2.87e-7，6σ -> 9.9e-10。
- 整幅虚警数 N_FA ≈ N_eff * p，N_eff ~ 搜索面积 /(2π σ_corr²)（σ_corr 为噪声相关长度）。
- 由此：4096² 全白噪声下 5σ 期望假峰 ≈ 16.8e6 * 2.87e-7 ≈ 4.8 个；若相关长度 3 px，N_eff 降约 28 倍、同一 5σ 只剩 ≈ 0.17 个假峰。反过来说，把阈值按"独立像素"标定的实现，在相关噪声下虚警被低估同样倍数。以上为本报告自算，不是文献数值。

---

## H. 速度基准

### [H1] Hu, L. et al. (2022) — Image Subtraction in Fourier Space (SFFT)
- 出处：ApJ 936, 157
- 链接：https://doi.org/10.3847/1538-4357/ac7394 [OK] ｜ https://arxiv.org/abs/2109.09334 [OK] ｜ 可读全文 https://ar5iv.labs.arxiv.org/html/2109.09334 [OK]
- 一句话思想：用 δ 函数基在频域分解差分核，使减影完全可并行（利于 GPU），并容纳 PSF / 光度 / 天光的空间变化。
- 定量结果（核到正文）：4K 图像（DECam CCD，2046x4094）单次减影 SFFT 约 4 s（server I）/ 约 1 s（server II）；PyTorchDIA 约 30 s / 约 10 s，摘要口径为"约一个数量级"提速；SFFT 耗时对核尺寸不敏感，HOTPANTS / PyTorchDIA 明显敏感。一次 DECam 曝光 = 62 个 2046x4094 CCD。
- 用途：直接给出"4K 量级 + FFT 路线"的真实耗时锚点。

### [H2] Hu, L. et al. (2026) — A GPU-Accelerated Transient Detection Pipeline for DECam Time-Domain Surveys
- 出处：arXiv:2603.08593
- 链接：https://arxiv.org/abs/2603.08593 [OK]
- 一句话思想：SFFT 差分 + CNN real-bogus + 候选过滤串成端到端 GPU 流水线，直接产出告警与光变曲线。
- 定量结果（核到摘要）：每张 DECam 曝光端到端约 50 s（已标定像 -> 告警），用 modest 算力；real-bogus 完备度约 99%、伪源剔除约 96%。按 62 x 2046 x 4094 约 5.2e8 px 反推约 10 Mpix/s（此比值本报告推算）。
- 用途：目前能查到的最接近"4096² 量级整链耗时"的公开数字。

### [H3] Masci, F. J. et al. (2019) — The Zwicky Transient Facility: Data Processing, Products, and Archive
- 出处：PASP 131, 018003
- 链接：https://doi.org/10.1088/1538-3873/aae8ac [OK] ｜ https://arxiv.org/abs/1902.01872 [OK]
- 一句话思想：ZTF 科学数据系统（实时差分 + 机器学习筛选 + 归档）的完整工程描述，是巡天级检测吞吐的公开基线。
- 定量结果（核到摘要）：600 Mpix 相机、扫描速率约 3760 deg²/hr；告警包典型 13 分钟内（95 分位）可用；S/N >= 10 时天测精度 45-85 mas（每轴 RMS）；亮星测光重复性 8-25 mmag；相对 Pan-STARRS1 定标优于 2%。
- 用途：给出"每夜千万级源 + 分钟级延迟"的现实约束。

### [H4] Jurić, M. et al. (2015) — The LSST Data Management System
- 出处：ADASS XXV（arXiv:1512.07914）
- 链接：https://arxiv.org/abs/1512.07914 [OK]
- 一句话思想：LSST DM 的总体架构（实时告警流 + 多次数据发布），强调通用、高性能、可扩展的处理栈。
- 定量结果（核到摘要）：平均 15 TB/夜 原始数据；10 年运行期内 11 次数据发布。
- 用途：给出"处理管线必须在什么数据量级上跑"的官方口径。

### [H5] Ivezić, Ž. et al. (2019) — LSST: From Science Drivers to Reference Design and Anticipated Data Products
- 出处：ApJ 873, 111
- 链接：https://doi.org/10.3847/1538-4357/ab042c [OK]
- 一句话思想：LSST 系统总设计文档，含相机规模、巡天节奏与数据产品量级的权威数字。
- 定量结果（核到 arXiv 版摘要）：3.2 Gpix 相机、9.6 deg² 视场；单滤光片 3 夜可成像约 10,000 deg²；单次曝光 r 波段 5σ 点源深度约 24.5，coadd 约 27.5；最终约 32 万亿次测量 / 400 亿天体。
- 用途：把"4096² 单片"放进"每天上千片"的量级里做工程设计。

### [H6] Barbary, K. (2016) — SEP: Source Extractor as a library
- 出处：JOSS 1, 58
- 链接：https://doi.org/10.21105/joss.00058 [OK]
- 一句话思想：把 SExtractor 的检测/测光算法抽成 C 库（+ Python 绑定），让检测能嵌进 Python 管线而不必跑外部进程。
- 定量结果：JOSS 摘要未给性能数字（仅声明算法与 SExtractor 一致）——本条无定量，仅作现代实现参照 [核到摘要]。
- 用途：说明"复用算法但去掉 I/O 与进程开销"这条工程路线。

### [H7] FFTW benchmark（benchFFT）页面
- 链接：http://www.fftw.org/speed/ [OK]
- 一句话思想：公开的 FFT 实现横评，给出跨机器 mflops 曲线与测量口径。
- 定量结果（核到页面）：性能口径定义为 mflops = 5 * N * log2(N) / (单次 FFT 微秒数) / 2（实数 FFT）；页面按 CPU/编译器给出实测曲线（页面为图表，本条只引用定义式）。
- 用途：判断"FFT 卷积在 4K 尺度是否值得"时的可复算基准口径（配 H1 的实测 1-4 s 使用）。

---

## I. 注入-回收（injection-recovery）

### [I1] Kessler, R. et al. (2015) — The Difference Imaging Pipeline for the Transient Search in the Dark Energy Survey
- 出处：AJ 150, 172
- 链接：https://doi.org/10.1088/0004-6256/150/6/172 [OK] ｜ https://arxiv.org/abs/1507.05137 [OK]
- 一句话思想：差分成像检测链的标准做法——把假 SN Ia 叠加进真实图像再跑同一条检测链，用回收率标定效率，并以"50% 完备度深度"作为报告口径。
- 定量结果（核到摘要）：浅场单历元 50% 完备度深度 23.5、深场 24.5；SN Ia 检测效率降到 1/2 处分别在 z 约 0.7 / 约 1.1；每 deg² 每次观测 130 个检测，其中仅 25% 为伪源。
- 用途：最干净的"完备度曲线 + 50% 深度"可引范例。

### [I2] Bernardinelli, P. H. et al. (2024) — The DECam Ecliptic Exploration Project (DEEP). III. Survey Characterization and Simulation Methods
- 出处：AJ（DOI 10.3847/1538-3881/ad1527）
- 链接：https://doi.org/10.3847/1538-3881/ad1527 [OK]
- 一句话思想：把合成天体种群注入图像、再用与真实检测完全相同的处理链回收，同时标定"按星等"与"按移动速率"的完备度。
- 定量结果（核到摘要）：各指向组峰值完备度平均 80%；在 m = 26.22 处降到峰值的 25%（即 20%）。
- 用途：演示"完备度不能只报一个数，要给曲线 + 转折点"的现代报告规范。

### [I3] LSST Dark Energy Science Collaboration (2021) — Effects of overlapping sources on cosmic shear estimation: statistical sensitivity and pixel-noise bias
- 出处：JCAP（arXiv:2103.02078）
- 链接：https://arxiv.org/abs/2103.02078 [OK]
- 一句话思想：用 Fisher 形式化框架定量评估 blending（源重叠）的影响——不只是噪声上升，更是源与源之间的统计相关。
- 定量结果（核到摘要）：全深度 LSST 像中约 62% 星系其像素内至少 1% 通量来自重叠源；blending 使有效数密度 N_eff 下降约 18%；r 波段 N_eff 上限 39.4 gal/arcmin²。
- 用途：论证"blending 会系统性改变完备度与有效样本量"的定量依据。

### [I4] Vera C. Rubin Observatory Team (2026) — The Vera C. Rubin Observatory Data Preview 1
- 出处：arXiv:2603.23786
- 链接：https://arxiv.org/abs/2603.23786 [OK]
- 一句话思想：Rubin 首次数据发布（LSSTComCam，1792 次曝光），同时公布各波段 5σ 点源深度口径与合成源注入做法。
- 定量结果（核到摘要）：约 15 deg²、7 个场；PSF FWHM 中位数 1.14 角秒（最锐 0.58 角秒）；最深场 coadd 的 5σ 点源深度 u=24.55, g=26.18, r=25.96, i=25.71, z=25.07, y=23.1；约 230 万天体；数据量约 3.5 TB。
- 用途：当下最新的"深度定义 + 注入检验"官方范式（配 Rubin 官方 notebook：Injecting Synthetic Sources Into Single-Visit Images）。

### [I5] CADC / WIRWolf 文档：Depth profile（人工星注入与完备度深度）
- 链接：https://www.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/en/wirwolf/docs/depthprof.html [OK]
- 一句话思想：把"完备度深度"的操作化定义写成流程文档：按星等阶梯注入人工星 -> 用生产检测器回收 -> 拟合回收率曲线 -> 取 50% 处为深度。
- 定量结果：文档化流程（非论文，无单一数字）[OK 可访问]。
- 用途：写方法章节与做 CI 回归基线时的标准操作规程引用。

---

## J. 亚像元定位精度

### [J1] Winick, K. A. (1986) — Cramér-Rao lower bounds on the performance of charge-coupled-device optical position estimators
- 出处：JOSA A 3, 1809
- 链接：https://doi.org/10.1364/josaa.3.001809 [WARN：探测 timeout，出版方反爬；DOI 与元数据已核验]
- 一句话思想：在方像素 CCD + 高斯光斑 + 泊松散粒噪声模型下，给出位置估计的二维 Cramér-Rao 下界与最大似然估计量，说明采样比决定精度上限。
- 定量结果（核到摘要）：一维情形使 CRB 最小的像素/像尺寸比 Δx/σs 在 1 到 2 之间；二维情形在 1.5 到 2.5 之间（在很宽 S/N 范围内成立）。
- 用途："质心 vs 拟合"精度讨论的数学天花板来源，也解释为何欠采样 + 整数峰会掉精度。

### [J2] Hechenblaikner, G. (2023) — Unbiased centroiding of point targets close to the Cramér-Rao limit
- 出处：JOSA A（DOI 10.1364/josaa.507588）；开放版 arXiv:2307.09557
- 链接：https://arxiv.org/abs/2307.09557 [OK] ｜ https://doi.org/10.1364/josaa.507588 [WARN：出版方反爬]
- 一句话思想：质心（CoG）类估计量的系统偏差主要来自两处：PSF 采样过粗、ROI 边界截断信号；提出几乎无系统偏差的无偏估计量并给出方差解析式。
- 定量结果（核到摘要）：推导并 Monte-Carlo 验证"光子散粒噪声 + 像素噪声 + 残余系统误差"三项方差分解；与迭代加权 CoG、阈值 CoG、迭代最小二乘、2D 高斯回归比较精度与计算速度；低光子数下无偏估计量优势最大。
- 用途：直接支撑"质心类实现自带系统偏差，必须做偏差校正与 ROI 选择"这条设计约束。

### [J3] Stone, R. C. (1989) — A comparison of digital centering algorithms
- 出处：AJ 97, 1227
- 链接：https://doi.org/10.1086/115066 [WARN：探测 405，出版方反爬；DOI 与元数据已核验]
- 一句话思想：在合成星像上系统比较五类一维定心算法，结论是拟合类（高斯拟合）与中值法好、矩法与导数搜索法差；背景亮时"阈值截断后的矩法"最好。
- 定量结果（核到摘要）：覆盖宽观测条件的对比与各方法优缺点（摘要无单一数值），并讨论 USNO 7-inch 子午仪实时处理用法。
- 用途：说明"质心（矩法）与拟合不是等价选择"，以及背景/阈值对定心的强依赖。

### [J4] Anderson, J.; King, I. R. (2000) — Toward High-Precision Astrometry with WFPC2. I. Deriving an Accurate Point-Spread Function
- 出处：PASP 112, 1360
- 链接：https://doi.org/10.1086/316632 [OK]
- 一句话思想：HST 高精度天体测量的前提是把 PSF 建到像素相位级别；PSF 建模误差与像素相位效应是定心系统误差的主因。
- 定量结果：论文给出 PSF 模型与像素相位校正后的精度量级（本条仅核到元数据/标题，未逐字核对数值）[元数据已核验]。
- 用途："整数峰 vs 质心"这类实现差异为何会引入与像素相位相关的系统偏差——经典出处。

### [J5] Stetson, P. B. (1987) — DAOPHOT: A computer program for crowded-field stellar photometry
- 出处：PASP 99, 191
- 链接：https://doi.org/10.1086/131977 [OK]
- 一句话思想：拥挤场测光的经典范式：先建 PSF 模型，再用 PSF 拟合同时解出位置与流量（ALLSTAR），并配套人工星注入（ADDSTAR）做回收评估。
- 定量结果：给出 PSF 拟合与孔径测光在拥挤场的误差行为框架（本条仅核到元数据；ADDSTAR 注入流程由该文确立）[元数据已核验]。
- 用途：注入-回收与 PSF 拟合两条线索的共同源头，可与 I1 / I2 / J2 串起来。

---

## K. Rust / 系统级实现

（版本与下载量均于 2026-10-19 取自 crates.io API；docs.rs 与 GitHub 页面均返回 200）

### [K1] rustfft 6.4.1（32.66M 下载，2025-09-18 更新）
- 链接：https://docs.rs/rustfft/latest/rustfft/ [OK] ｜ https://github.com/ejmahler/RustFFT [OK] ｜ https://crates.io/crates/rustfft [OK]
- 一句话思想：纯 Rust 高性能 FFT，支持任意长度、运行时算法选择（FftPlanner）与 SIMD/多线程。
- 定量结果：版本与下载量如上；跨库对比口径请用 H7 的 mflops 公式自测（本条不给未核验倍数）。
- 用途：Rust 侧频域卷积/匹配滤波的首选底座。

### [K2] realfft 3.5.0（18.70M 下载）
- 链接：https://github.com/HEnquist/realfft [OK] ｜ https://docs.rs/realfft [OK]
- 一句话思想：基于 rustfft 的实数->复数 FFT 封装，省掉一半冗余计算——图像卷积恰好只吃实数输入。
- 定量结果：版本与下载量如上；封装复用 rustfft 的 SIMD 后端 [crates.io API]。

### [K3] ndarray 0.17.2（134.26M 下载）
- 链接：https://docs.rs/ndarray [OK] ｜ https://github.com/rust-ndarray/ndarray [OK]
- 一句话思想：Rust 的 n 维数组 + 视图/切片，等价于 Rust 侧 numpy 的 ndarray 层，是天文学 2D/3D 图像栈的自然容器。
- 定量结果：版本与下载量如上；支持 rayon 并行迭代与 BLAS 后端 [crates.io API]。

### [K4] fitsio 0.21.10（140k 下载）
- 链接：https://docs.rs/fitsio [OK] ｜ https://github.com/simonrw/rust-fitsio [OK]
- 一句话思想：Rust 的 FITS 读写（绑定 cfitsio），覆盖图像 HDU 与表 HDU。
- 定量结果：版本与下载量如上；下载量比通用 crate 低约 3 个数量级——Rust 天文生态仍很小，这本身就是工程风险的输入 [crates.io API]。

### [K5] rayon 1.12.0（607.00M 下载）
- 链接：https://docs.rs/rayon [OK] ｜ https://github.com/rayon-rs/rayon [OK]
- 一句话思想：数据并行的工作窃取线程池，par_iter 即可把 4096² 的像素循环铺满多核。
- 定量结果：版本与下载量如上；按行/按 tile 切分是最省事的并行策略 [crates.io API]。

### [K6] imageproc 0.27.0（14.12M 下载）
- 链接：https://docs.rs/imageproc [OK] ｜ https://github.com/image-rs/imageproc [OK]
- 一句话思想：通用图像处理（滤波、形态学、连通域/轮廓），可承担检测后的形态学清洗与连通域统计。
- 定量结果：版本与下载量如上；提供 connected_components 的 4/8 连通标注 [crates.io API]。
- 局限：面向通用图像，不含天文标定/测光。

### [K7] wide 1.7.1（83.19M 下载）
- 链接：https://docs.rs/wide [OK] ｜ https://github.com/Lokathor/wide [OK]
- 一句话思想：可移植 SIMD 封装（f32x8 等），让逐像素比较/累加在 x86 与 ARM 上都吃到向量化。
- 定量结果：版本与下载量如上；典型用于 8 路并行阈值比较与加权和 [crates.io API]。

### [K8] light-curve-feature 0.20.0（180k 下载）
- 链接：https://crates.io/crates/light-curve-feature [OK] ｜ https://github.com/light-curve/light-curve-feature
- 一句话思想：Rust 实现的时序特征提取库，说明时域天文社区确实在用 Rust 承担计算密集的特征层。
- 定量结果：版本与下载量如上 [crates.io API]。
- 用途：作为"Rust 承担哪一层"的现实证据：特征计算层可 Rust 化，I/O 与编排层仍留在 Python。

### [K9] astro（astro-rust）2.0.0（196k 下载）
- 链接：https://crates.io/crates/astro [OK] ｜ https://github.com/saurvs/astro-rust
- 一句话思想：Rust 的天文基础算法集合（坐标/时间/星历等），可作 Rust 管线的基础算法层。
- 定量结果：版本与下载量如上 [crates.io API]。
- 局限：不含图像检测与测光；成熟度远低于 Astropy。

### [K10] FFTW benchmark 页面（工程对照，同 H7）
- 链接：http://www.fftw.org/speed/ [OK]
- 定量结果：mflops = 5*N*log2(N)/(微秒数)/2 的口径 + 各平台实测曲线。
- 用途：评估 rustfft 与 FFTW 时用同一口径在自己的 4096² 实数图上实测，而不是引二手倍数。

### [K11] 负面结果：Rubin/LSST 官方栈中没有 Rust 组件
- 检索方式：crates.io 搜索（lsst / rubin / astronomy / fits）、arXiv 标题检索。
- 结论：Rubin/LSST 科学管线（Science Pipelines / Prompt Processing）以 C++ + Python 为主；未找到官方 Rust 组件或 crate。Rust 侧最接近的是 K4 这类 I/O 与 K8 这类特征计算库。
- 意义：若新方法用 Rust 实现，与 Rubin 生态的接口只能是 FITS 文件 + 命令行/服务层，不要期待官方 Rust 绑定。

---

## 未找到 / 检索边界（诚实记录）

1. 没有检索到"SExtractor 在 4096² 图像上的单图 CPU 耗时"的同行评议数字（查了 OpenAlex / arXiv / Crossref / 网页）。可用替代锚点：H1 的 4K 减影 1-4 s、H2 的整链 50 s/曝光。
2. 没有检索到把"整数峰 vs 质心"当独立课题的系统性偏差论文；该问题在文献中以像素相位误差 / PSF 采样偏差形式出现（J2、J4），工程上需自证（注入已知亚像元相位 -> 量偏差曲线）。
3. Gumbel 用于图像检测的直接文献以 G3 为核心（并述及 Pavesi et al. 2018 路线）；Pavesi 2018 的 DOI 与可访问链接本次未能核验，故不列入条目。
4. 部分出版方（A&A、IOP、OSA、UChicago Press）对自动抓取返回 403/405，相关条目已标 WARN 并同时给出 arXiv / 项目页等可访问镜像。
