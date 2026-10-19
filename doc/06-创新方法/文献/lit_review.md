# 天基星图快速高精度星点/拖影检测 — 真实文献综述

> 生成日期：2026-10-19　｜　共 34 条，全部带可访问链接。
>
> 核验方式：DOI 经 doi.org 解析为 302 并做 BibTeX 内容协商/OpenAlex 元数据比对；arXiv 经官方 API 取元数据；官方文档经 HTTP 200 校验。未列入任何无法解析链接的条目。
> 凡摘要/文档未给出数值者，条目内明确写“无数值可引”，不做任何数值推断。

## A. 经典星点检测与测光管线

- **Stetson, P. B. (1987).** DAOPHOT: A Computer Program for Crowded-Field Stellar Photometry. *PASP 99, 191*. [链接](https://doi.org/10.1086/131977)
  - 核心思想：FIND 用局部背景+N×σ 阈值找峰 → PHOT 做合成孔径测光 → 用亮星迭代出经验 PSF → GROUP/NSTAR 做最小二乘 PSF 拟合测光。
  - 定量结果：摘要无数值。已知局限（摘要自述）：crowded field 下经验 PSF 必须逐帧重建，且作者明确指出程序存在 shortcomings 待改进。
- **Bertin, E.; Arnouts, S. (1996).** SExtractor: Software for source extraction. *A&AS 117, 393*. [链接](https://doi.org/10.1051/aas:1996164)
  - 核心思想：背景/σ 图 + 阈值分割 + 多阈值 deblending + 神经网络星/系分类，把检测做成可在大图上无人值守跑完的管线。
  - 定量结果：摘要无数值（无完备度/虚警率）。定性结论：多数图像上可用神经网络得到“very reliable”星/系分离。
- **Barbary, K. (2016).** SEP: Source Extractor as a library. *JOSS 1, 58*. [链接](https://doi.org/10.21105/joss.00058)
  - 核心思想：把 SExtractor 的检测/去blend/测量算法重写为可嵌入 Python 的 C 库，使同一阈值策略能以库函数方式被调用。
  - 定量结果：JOSS 短文，无数值。
- **Astropy/Photutils team (Bradley et al.) (2025).** Photutils — Point-like Source Detection（DAOStarFinder / find_peaks）官方文档. *photutils.readthedocs.io（官方文档，持续更新）*. [链接](https://photutils.readthedocs.io/en/stable/user_guide/detection.html)
  - 核心思想：在背景扣除 + 高斯核平滑后的图像上做 DAOFIND 式局部极大值检测，阈值由用户给（文档示例：threshold = median + 5.0×std，box_size=11）。
  - 定量结果：文档示例的可核验数值：阈值 median+5σ；box_size=11 时两峰在 x/y 方向至少相距 6 px 才会被分辨（即近邻峰会被合并/漏检）。
- **Bertin, E. (2011).** PSFEx: Point Spread Function Extractor（官方文档；会议文集 ASP Conf. Ser. 442, 435）. *ASP Conf. Ser. 442, 435（ADS: 2011ASPC..442..435B）；文档 psfex.readthedocs.io*. [链接](https://psfex.readthedocs.io/en/latest/)
  - 核心思想：把 PSF 写成“像素基/Moffat/高斯基函数 × 随视场位置变化的系数”的线性模型，用星像拟合出全场空间变化的 PSF，供 PSF 测光使用。
  - 定量结果：文档/会议文集无基准数值（无数值可引）。
- **Annunziatella, M.; Mercurio, A.; Brescia, M.; Cavuoti, S.; Longo, G. (2013).** Inside Catalogs: A Comparison of Source Extraction Software. *PASP 125, 68*. [链接](https://doi.org/10.1086/669333)
  - 核心思想：在两类模拟场（均匀分布 / 中心过密）上对 SExtractor+PSFEx 与 DAOPHOT+ALLSTAR 做同条件对比。
  - 定量结果：结论：SExtractor 生成的目录比 DAOPHOT 更深；SPREAD_MODEL + 神经网络把星/系分类推到极限星等；DAOPHOT/ALLSTAR 在点源测光与 PSF 测光上最优但目录对暗源不完备。摘要未给具体百分比。

## B. 现代 / 机器学习检测

- **Hausen, R.; Robertson, B. E. (2020).** Morpheus: A Deep Learning Framework for the Pixel-level Analysis of Astronomical Image Data. *ApJS 248, 20*. [链接](https://doi.org/10.3847/1538-4365/ab8868)
  - 核心思想：用语义分割网络把“检测+分割+形态分类”统一成逐像素分类，靠真实源的形态信息抑制假阳性。
  - 定量结果：在 CANDELS/GOODS-S 上对已知 3D-HST 源（H<26 AB）达到“high completeness”；摘要未给具体百分比/虚警率。
- **Tanoglidis, D.; Ćiprijanović, A.; Drlica-Wagner, A. et al. (2021).** DeepShadows: Separating low surface brightness galaxies from artifacts using deep learning. *Astronomy and Computing 35, 100469*. [链接](https://doi.org/10.1016/j.ascom.2021.100469)
  - 核心思想：用 CNN 把低表面亮度星系与同形态的伪影（鬼影、反射、串扰）分开，替代纯形态切选。
  - 定量结果：摘要未取到（数据库无摘要），无数值可引。
- **Zhang, K.; Bloom, J. S. (2020).** deepCR: Cosmic Ray Rejection with Deep Learning. *ApJ 889, 24*. [链接](https://doi.org/10.3847/1538-4357/ab3fa6)
  - 核心思想：用 CNN 做像素级宇宙线掩模 + 图像修补，是“像素级二分类检测”在真实管线里最成熟的速度/完备度对照案例。
  - 定量结果：假阳性率 0.5% 时检出率≈100%（河外场、球状星团场），分辨星系场 91%；相对多核 CPU 版 LACosmic 快 6.5×，单 GPU 快 90×；修补 MSE 比最好的非神经方法低 2.5×/5×/20×。
- **Guan, J.; Cheng, H.; Wu, Y.; Tian, C.; Qi, J.-Y. (2025).** Multi-target tracking for star sensor based on CenterTrack deep learning model. *Scientific Reports 15*. [链接](https://doi.org/10.1038/s41598-025-21077-4)
  - 核心思想：把 CenterTrack 类多目标跟踪网络改造到星敏感器序列图像上，用整段序列的聚合特征做弱小目标检测+跟踪。
  - 定量结果：相对原始 CenterTrack：假阳性率降约 60%，真实目标漏检率降 20%，ID 切换降约 50%；相比传统算法仅用 2 帧即可捕获新出现目标。
- **Yan, Y.; Liu, C.; Li, J.; Wang, F. (2026).** Detection and Classification of Astronomical Sources with Astro-RetinaNet in Crowded Stellar Fields. *ApJS*. [链接](https://doi.org/10.3847/1538-4365/ae4a24)
  - 核心思想：用 RetinaNet 一阶段检测器在密集星场中同时做源检测与分类（星/系/伪影），代表“检测即回归”的现代路线。
  - 定量结果：摘要未取到，无数值可引。

## C. 统计检测理论

- **Vio, R.; Andreani, P. (2021).** Everything you always wanted to know about matched filters (but were afraid to ask). *arXiv:2107.09378*. [链接](https://arxiv.org/abs/2107.09378)
  - 核心思想：系统梳理匹配滤波在“已知平滑信号 + 平稳高斯噪声”下检测弱信号的最优性，并逐条回答离散化、非高斯噪声等实际折损问题。
  - 定量结果：综述性文章，无数值基准；核心结论是离散化与非高斯噪声会显著偏离理想 MF 性能。
- **Bardeen, J. M.; Bond, J. R.; Kaiser, N.; Szalay, A. S. (1986).** The statistics of peaks of Gaussian random fields. *ApJ 304, 15*. [链接](https://doi.org/10.1086/164143)
  - 核心思想：给出高斯随机场中“超过阈值 ν 的峰”的期望数密度与峰形分布解析式，是把阈值换算成期望虚警个数的经典工具。
  - 定量结果：解析结果（以峰高 ν=δ/σ 与谱矩为参数），摘要无数值表；实用性在于可直接对给定 ν 算出整幅图像的期望峰数。
- **Bond, J. R.; Efstathiou, G. (1987).** The statistics of cosmic background radiation fluctuations. *MNRAS 226, 655*. [链接](https://doi.org/10.1093/mnras/226.3.655)
  - 核心思想：用 GRF 的期望上穿率/超越集理论，把“阈值—面积—期望虚警数”闭式联系起来，是天文里阈值标定的源头文献之一。
  - 定量结果：解析式（含谱指数依赖），摘要无数值表。
- **Benjamini, Y.; Hochberg, Y. (1995).** Controlling the False Discovery Rate: A Practical and Powerful Approach to Multiple Testing. *JRSS-B 57, 289*. [链接](https://doi.org/10.1111/j.2517-6161.1995.tb02031.x)
  - 核心思想：BH 升步法：把 m 个 p 值排序后按 q·k/m 阈值截断，使期望 FDR ≤ q·m0/m，从而在“检出数”而非“单点虚警”层面控制误差。
  - 定量结果：定理保证 E[FDR] ≤ (m0/m)·q（独立或 PRDS 依赖下），是全部 FDR 天文应用的基线。
- **Cheng, D.; Cammarota, V.; Fantaye, Y.; Marinucci, D.; Schwartzman, A. (2019).** Multiple testing of local maxima for detection of peaks on the (celestial) sphere. *Bernoulli 26(1)*. [链接](https://doi.org/10.3150/18-bej1068)
  - 核心思想：只在“局部极大值”处做检验（拓扑多重检验），配合 BH 控制 FDR，给出球面上点源检测的渐近误差控制。
  - 定量结果：渐近结论：FDR 受控、且随信号强度与 needlet 频率增大功效相合（摘要未给有限样本数值）。
- **Carrón Duque, J.; Buzzelli, A.; Fantaye, Y.; Marinucci, D.; Schwartzman, A.; Vittorio, N. (2019).** Point source detection and false discovery rate control on CMB maps. *Astronomy and Computing 28, 100310*. [链接](https://doi.org/10.1016/j.ascom.2019.100310)
  - 核心思想：把 FDR 控制直接接到 CMB 图点源检测流程里，用 p 值化检测统计量替代固定 σ 阈值。
  - 定量结果：摘要未取到，无数值可引。
- **Hu, M.; Sun, H.; Harness, A. D.; Kasdin, N. J. (2021).** Bernoulli generalized likelihood ratio test for signal detection from photon counting images. *JATIS 7(2), 028006*. [链接](https://doi.org/10.1117/1.jatis.7.2.028006)
  - 核心思想：直接对单帧光子计数图像建 Bernoulli 似然并做 GLRT，同时给出目标与背景强度的最大似然估计，可在线判定何时停止观测。
  - 定量结果：摘要定性结论：优于“共加图+高斯假设”的旧 GLRT 与两种基于 SNR 的检测算法；并给出可在线停止的置信判据（无数值表）。
- **Zackay, B.; Ofek, E. O.; Gal-Yam, A. (2016).** Proper Image Subtraction — Optimal Transient Detection, Photometry, and Hypothesis Testing. *ApJ 830, 27*. [链接](https://doi.org/10.3847/0004-637x/830/1/27)
  - 核心思想：把图像相减写成匹配滤波统计量，并给出该统计量的分布与假设检验框架——检测、测光、显著性三合一。
  - 定量结果：摘要以推导为主；给出检测统计量的解析分布，使阈值可直接换算成虚警概率（无数值表）。

## D. 拖影 / 卫星条纹检测

- **Nir, G.; Zackay, B.; Ofek, E. O. (2018).** Optimal and Efficient Streak Detection in Astronomical Images. *AJ 156, 229*. [链接](https://doi.org/10.3847/1538-3881/aaddff)
  - 核心思想：条纹的最优统计量是“被 PSF 展宽的线段模板”的匹配滤波；用快速 Radon 变换在(角度,长度)空间一次搜完，并扩展到任意短线段。
  - 定量结果：暴力 Radon O(N³) → 快速 Radon O(N²log N)，快 N/log N（典型图像 2–3 个数量级）；模拟中恢复理论 SNR；并指出 1k×1k 图中 2σ 级条纹产生的 5σ 虚警可比背景涨落多好几倍。
- **Waszczak, A.; Prince, T. A.; Laher, R. et al. (2017).** Small Near-Earth Asteroids in the Palomar Transient Factory Survey: A Real-Time Streak-detection System. *PASP 129, 034402*. [链接](https://doi.org/10.1088/1538-3873/129/973/034402)
  - 核心思想：在 PTF（1.2 m，7.3 deg²）实时管线里专门做高角速度（视觉呈条纹）小行星的条纹检测子系统。
  - 定量结果：摘要未给完备度/虚警数值（仅给出巡天配置 1.2 m / 7.3 deg²）。
- **Virtanen, J.; Poikonen, J.; Säntti, T.; Komulainen, T.; Torppa, J.; Granvik, M. (2016).** Streak detection and analysis pipeline for space-debris optical images (StreakDet). *Advances in Space Research 57, 1607*. [链接](https://doi.org/10.1016/j.asr.2015.09.024)
  - 核心思想：面向空间碎片光学图像的全流程条纹检测器：线段检测 + 假警抑制 + 后续测光/天测分析。
  - 定量结果：摘要未取到（数据库无摘要），无数值可引。
- **Mróz, P.; Otarola, A. C.; Prince, T. A.; Dekany, R.; Duev, D. A.; Graham, M. J. et al. (2022).** Impact of the SpaceX Starlink Satellites on the Zwicky Transient Facility Survey Observations. *ApJL 924, L30*. [链接](https://doi.org/10.3847/2041-8213/ac470a)
  - 核心思想：用 ZTF 存档统计 Starlink 条纹的出现率随时间/天光条件的演化，量化星座对巡天的影响。
  - 定量结果：2019-11～2021-09 共找到 5301 条 Starlink 条纹；暮光图像被条纹污染比例从 2019 末 <0.5% 升至 2021-08 的 18%；Starlink 规模达 1 万颗时暮光图像几乎全部受影响。
- **Kruk, S.; García-Martín, P.; Popescu, M.; Aussel, B.; Dillmann, S.; Perks, M. E. et al. (2023).** The impact of satellite trails on Hubble Space Telescope observations. *Nature Astronomy 7, 262*. [链接](https://doi.org/10.1038/s41550-023-01903-3)
  - 核心思想：用公民科学 + 深度学习扫描 HST 2002–2021 全档案，首次测出 LEO 轨道上卫星条纹的污染率。
  - 定量结果：2.7% 的单帧曝光（典型 11 min）被卫星穿越；比例随时间上升，并依赖视场大小、曝光时间、滤光片与指向。
- **Hainaut, O. R.; Williams, A. P. (2020).** Impact of satellite constellations on astronomical observations with ESO telescopes in the visible and infrared domains. *A&A 636, A121*. [链接](https://doi.org/10.1051/0004-6361/202037501)
  - 核心思想：用几何概率方法对 18 个在建星座（>2.6 万颗卫星）估算 ESO 望远镜在可见/近红外/热红外的受影响比例。
  - 定量结果：覆盖 18 个星座、>26,000 颗卫星；给出保守的数量级估计（摘要未列单值表格）。
- **Tyson, T.; Ivezić, Ž.; Bradshaw, A.; Rawls, M.; Xin, B.; Yoachim, P. et al. (2020).** Mitigation of LEO Satellite Brightness and Trail Effects on the Rubin Observatory LSST. *AJ 160, 226*. [链接](https://doi.org/10.3847/1538-3881/abba3e)
  - 核心思想：从指向规避、传感器亮条纹实验、条纹伪影校正算法三方面评估 Rubin/LSST 的卫星条纹缓解能力。
  - 定量结果：Starlink v0.9 g≈4.5，DarkSat g≈6.1，未来目标 g≈7；即使变暗，条纹仍达 SNR≈100；8.4 m 镜 + 550 km 卫星时条纹 FWHM≈3″（离焦展宽，避免饱和）；情景含 48,000 颗 LEO 卫星。
- **Rachith, E.; Hellmich, S.; Fiszbin, V.; Irureta-Goyena, B. Y.; Price, A.; Kneib, J.-P. (2026).** Streak detection in the VST/OmegaCAM archive using deep learning. *arXiv:2606.30286*. [链接](https://arxiv.org/abs/2606.30286)
  - 核心思想：HT-LCNN（Hough 变换查表式 CNN）在原始图像上做初检 + VGG6 分类器复核，构成两级 CNN 条纹检测管线。
  - 定量结果：摘要未给完备度/虚警率（仅有管线结构描述）。

## E. 多帧 / 轨迹检测（track-before-detect）

- **Shao, M.; Nemati, B.; Zhai, C.; Turyshev, S. G.; Sandhu, J.; Hallinan, G. et al. (2014).** Finding Very Small Near-Earth Asteroids Using Synthetic Tracking. *ApJ 782, 1*. [链接](https://doi.org/10.1088/0004-637x/782/1/1)
  - 核心思想：高速短曝光“冻结”目标运动 + 盲搜速度矢量 + 移位叠加，等价于把望远镜长时间跟踪该目标，从而把单帧不可检信号抬到可检。
  - 定量结果：摘要定性：单帧 SNR 不足的目标经速度搜索+叠加后可检；并改善快速移动目标的天测精度（无具体数值）。
- **Davey, S. J.; Rutten, M. G.; Cheung, B. (2008).** A Comparison of Detection Performance for Several Track-before-Detect Algorithms. *EURASIP J. Adv. Signal Process. 2008, 428036*. [链接](https://doi.org/10.1155/2008/428036)
  - 核心思想：在同一数据上横向比较离散网格 Bayes、动态规划(DP)、粒子滤波、H-PMHT 四类 TBD 的检测性能与计算/存储开销。
  - 定量结果：以检测性能 vs 计算资源两条轴对比（摘要未给具体 Pd/Pfa 数值表）。
- **Dao, P. D.; Schmidt, V.; Schlaegel, W.; Rast, R.; Gregory, S.; Dentamaro, A. V. (2014).** Track-Before-Detect Algorithm for Faint Moving Objects based on Random Sampling and Consensus. *AMOS Conf. (Advanced Maui Optical and Space Surveillance Technologies)*. [链接](https://doi.org/10.64861/anxo2089)
  - 核心思想：用 RANSAC 利用“准匀速 + 恒星杂波可预测”的先验，在 TBD 框架下以远低于 Viterbi/Bayes 的算力换取低虚警。
  - 定量结果：摘要定性：相比 Viterbi/Bayes 类 TBD 大幅降低算力与内存需求（无具体数值）。
- **Liu, S.; Chen, X.; Zeng, T.; Zheng, L. (2013).** New analytical approach to detection threshold of a dynamic programming track-before-detect algorithm. *IET Radar, Sonar & Navigation 7(6)*. [链接](https://doi.org/10.1049/iet-rsn.2012.0172)
  - 核心思想：用广义帕累托分布（超阈值 POT 模型）拟合 DP-TBD 统计量的尾部，解析算出虚警概率从而给出 CFAR 阈值。
  - 定量结果：仿真显示该法给出的虚警概率估计比旧极值理论(EVT)方法更准（摘要未给具体数量级差）。

## F. PSF 估计与建模

- **Anderson, J.; King, I. R. (2000).** Toward High-Precision Astrometry with WFPC2. I. Deriving an Accurate Point-Spread Function. *PASP 112, 1360*. [链接](https://doi.org/10.1086/316632)
  - 核心思想：用抖动(dither)把欠采样的星像投到更细网格上迭代拟合，得到“有效 PSF”(ePSF)，突破像素采样限制。
  - 定量结果：摘要无数值表；方法学结论是把 PSF 建到亚像素网格后可用于高精度天体测量。
- **Jee, M. J.; Blakeslee, J. P.; Sirianni, M.; Martel, A. R.; White, R. L.; Ford, H. C. (2007).** Principal Component Analysis of the Time- and Position-dependent Point-Spread Function of the Advanced Camera for Surveys. *PASP 119, 1203*. [链接](https://doi.org/10.1086/524849)
  - 核心思想：对 ACS 的 PSF 做 PCA 分解，用少数主成分重构随位置与时间变化的 PSF 场。
  - 定量结果：摘要未取到（数据库无摘要），无数值可引；方法结论：少数主成分即可描述主要变化。
- **Liaudat, T. I.; Starck, J.-L.; Kilbinger, M.; Frugier, P.-A. (2023).** Rethinking data-driven point spread function modeling with a differentiable optical model (WaveDiff). *Inverse Problems 39, 035001*. [链接](https://doi.org/10.1088/1361-6420/acb664)
  - 核心思想：在数据驱动 PSF 建模中插入可微的光学前向模型，把建模空间从像素搬到波前，从而用欠采样、含噪、宽波段的星像推断超分辨且随波长/位置变化的 PSF。
  - 定量结果：摘要定性：相比现有数据驱动模型额外获得色差(chromatic)建模能力（无具体误差数值）。

## 可用于本文方法设计的 5 条关键结论

1. 阈值不能是“像素数”，必须是归一化的检测统计量：DAOPHOT/SExtractor/DAOStarFinder 都用局部背景+kσ 的单点阈值，近邻峰还会被 box 参数合并（photutils 文档：box_size=11 时相距<6 px 的峰不可分），密集场与拖影场景下虚警不可控；匹配滤波 SNR（Vio & Andreani 2021）才是可标定的量。

2. 点源与拖影应统一为同一个统计量族：Nir+2018 证明最优条纹统计量是“被 PSF 展宽的线段模板”的匹配滤波，点源只是长度→0 的极限；用快速 Radon/FFT 在(角度,长度)空间一次搜索可把 O(N³) 降到 O(N²log N)（快 N/log N，典型 2–3 个数量级）。

3. 阈值标定要从“单点虚警”升级到“多重检验”：GRF 超越数/峰密度（Bardeen+1986；Bond & Efstathiou 1987）给出与图像面积、阈值一致的期望虚警个数；BH-FDR（1995）与其球面局部极大值版本（Cheng+2019）把误差控制到“检出集合”层面——这正好对应“角度×长度×多帧”这种组合搜索带来的自由度爆炸。

4. 多帧是灵敏度的主要增益来源，代价是组合爆炸：合成跟踪（Shao+2014）与 TBD（Davey+2008；Dao+2014）把单帧不可检目标抬到可检，但速度空间搜索使有效检验次数暴涨；TBD 侧已有 GPD 尾部拟合给出 CFAR 阈值（Liu+2013），可直接借来对轨迹级统计量定阈。

5. PSF 失配是“精度”而非“速度”的瓶颈：ePSF（Anderson & King 2000）、PCA 时空 PSF（Jee+2007）、可微光学模型 WaveDiff（Liaudat+2023）都说明模板必须随位置/时间/波长更新；天基场景还叠加离焦展宽（Rubin 8.4 m 下 550 km 卫星条纹 FWHM≈3″，Tyson+2020），点源与条纹的有效 PSF 各向异性必须分别建模。

## 现有方法共同的缺口（3 条）

1. 缺少统一口径的“速度—完备度—虚警率”三元基准：条纹类工作报告的是发生率或管线结构（Nir+2018 报复杂度、Waszczak+2017 报系统配置、Virtanen+2016 无摘要数值），CNN 类报告的是特定数据集上的 FPR/检出率（deepCR 0.5% FPR→≈100%），两者不可直接比较，也没有“同等算力预算下”的对照。

2. 卫星条纹文献几乎只量化“发生率”（暮光 <0.5%→18%，HST 2.7%），很少给出“条纹存在时点源检测完备度/测光偏差的传递函数”，即在条纹背景下的检测阈值该抬多少、抬了以后丢多少真源，没有定量结论。

3. 天基/星上实时检测是明显空白：ML 侧只报 GPU 加速比（deepCR 90×），传统侧只报算法复杂度，缺少“星上算力/功耗受限 + 单帧 + 同时存在点源与拖影”的统一基准；星敏感器文献（Guan+2025）虽报 FPR↓60%，但任务是序列跟踪而非单帧检测+拖影判别。
