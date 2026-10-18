# 门限审计主表（rst19 单帧检测→质量门→测光→天体测量→覆盖守卫）

审计对象：`backend/src/` 下 detection / measurement / filtering / deblend / psf / statistics / source_classification / source_artifacts / empirical_photometry / photometric_calibration / astrometry_blind / catalog_coverage / gaia_tiled 及其 CLI 入口。
只读审计：未修改任何 .rs/.md/.json/.fits。行号均按当前工作树逐一核对。

**统计**：条目 112 条，可分离的具体数值条件约 171 个。
类型：结构常数 30；可配默认 33；实测值 7；经验调参 42。
出处：有理由 31；只声明数值 29；无出处 52。
适用范围：标注「只对本图/本相机」的共 47 条。

类型说明：【结构常数】=算法/数学定义；【可配默认】=CLI/结构体默认值；【实测值】=由本数据测得后写死；【经验调参】=为通过本图而选且代码/文档无依据。
出处列的 `sk` 标记：reason=有理由，value_only=只声明数值无理由，none=无出处。


## DET · 检测层（40 条）

| ID | 名称 / 值（单位） | 位置 | 类型 | 出处（sk） | 适用范围 | 改动该值会改变什么 |
|---|---|---|---|---|---|---|
| DET-01 | **threshold_sigma** = 4.0 σ | `backend/src/detection.rs:48` | 可配默认 | 数值见 deliverables/paper/main.tex:109「候选阈值 4σ」；理由未给（value_only） | 通用（高斯噪声显著性；仅当噪声近似高斯时成立） | 候选数、阈附近完备度、近邻对数、pair 审计样本量 |
| DET-02 | **min_distance** = 3（默认）/ 4（rst19、rst19-sources 生效） px | `backend/src/detection.rs:49；覆盖 backend/src/source_audit.rs:51` | 可配默认 | main.tex:109 写「最小峰距 4 px」；代码默认 3，仅两处入口覆盖；3 与 4 的差别无任何解释（value_only） | 只对 PSF FWHM≈2–3 px 的这类相机（4 px≈1.3–2×FWHM） **[image-specific]** | 3–4 px 间距的近邻是否被 NMS 合并 → 候选数、直接近邻/pair 结论 |
| DET-03 | **max_sources（元数据 -1 哨兵）** = None / −1 count | `backend/src/detection.rs:50；哨兵 :702` | 可配默认 | 无出处（工程上限）（none） | 通用 | 截断在质量判定之前（:638），quality_count 随之变小 |
| DET-04 | **background_box_size** = 128 px | `backend/src/detection.rs:51；校验 ≥16 :202` | 可配默认 | main.tex:109 提到「局部背景网格 128 px」，未给理由（value_only） | 只对 4096² 大视场、背景平缓的这类帧 **[image-specific]** | 局部 bg/noise map → 位置相关有效阈值 → 背景梯度处与边缘的近阈候选 |
| DET-05 | **background_sample_limit** = 100000 px | `backend/src/detection.rs:52` | 可配默认 | 无出处（抽样上限，statistics.rs:70-75 的 stride 抽样）（none） | 通用 | sigma-clip 抽样噪声 → noise 末位 |
| DET-06 | **statistics_sample_limit** = 1000000 px | `backend/src/detection.rs:346,355,403,412；元数据 :702` | 结构常数 | 效果=全帧（4096²=1.68e7 时约 1/17 抽样）（reason） | 通用 | global background/noise 的抽样误差 |
| DET-07 | **全局 sigma-clip** = 3.0σ × 4 次迭代 σ/iter | `backend/src/detection.rs:346,355,403,412；实现 backend/src/statistics.rs:77-119` | 结构常数 | 天文惯例（DAOPHOT/SExtractor 同类 3σ）；本仓库无注释说明（value_only） | 通用 | bg/noise → threshold = bg+4σ |
| DET-08 | **局部背景迭代次数** = 4（refine=true）/ 2 iter | `backend/src/detection.rs:448-452；开关 backend/src/measurement.rs:112` | 可配默认 | 无出处（none） | 通用 | 局部 bg map 收敛 → 逐块 noise → 边缘候选 |
| DET-09 | **mask_zero_pixels 自动规则** = None → 整型帧且 \|background\|>1 时置 true bool | `backend/src/detection.rs:154-155,269-270` | 结构常数 | 数据依赖规则（仅整型帧有 0 值哨兵语义）（reason） | 只对整型（u8/i16/i32/i64）帧 **[image-specific]** | 掩膜覆盖 → MASKED/PARTIAL_MASKED 比例 → quality_count |
| DET-10 | **饱和推断与掩膜** = None → max−32 ADU（i16→32735）；掩膜 value ≥max−32 或 ≤min+32 ADU | `backend/src/detection.rs:156,164,271,280-282` | 经验调参 | **无出处**：32 ADU 既非满阱标定也非头文件字段（none） | 只对 int16 满量程 32767 的这类相机 **[image-specific]** | SATURATED 旗标与掩膜 → quality_count；与 peak_ceiling 构成两套不同的饱和判据 |
| DET-11 | **负溢出下限** = −0.9×i16max = −29491 ADU | `backend/src/detection.rs:417-421` | 经验调参 | **无出处**（0.9 系数无说明）（none） | 只对该相机 int16 值域 **[image-specific]** | NEGATIVE_OVERFLOW → quality_count |
| DET-12 | **重复码值域审计** = 最小计数 128；floor = max(500, bg+20σ) count/ADU | `backend/src/detection.rs:123,413` | 实测值 | 现象有出处（DESIGN.md:366 记录 3990–3993 与 CODE_PATTERN 修正）；128/20σ/500 本身无推导（value_only） | 只对本图（该探测器的异常重复高位码） **[image-specific]** | CODE_PATTERN 旗标 → quality_count（DESIGN.md 记 159→12） |
| DET-13 | **DoG 原始残差峰门** = dog_min_peak_sigma = 2.0 σ | `backend/src/detection.rs:58；判据 :532` | 可配默认 | main.tex:155「原始残差正峰门槛 2」；推导未给（value_only） | 通用（作用于标准化残差） | DoG 提案是否接受 → 候选并集与 proposal_methods |
| DET-14 | **DoG 响应阈值** = max(6.0, threshold_sigma+2.0) σ | `backend/src/detection.rs:511-513` | 经验调参 | 只给结果：main.tex:155「DoG 响应阈值 6」；+2σ 偏移无出处（value_only） | 通用形式，6 与 4σ 的差值为本流程选定 | DoG 提案数 → 候选并集 → 去混/配对输入 |
| DET-15 | **starlet 响应阈值** = max(7.0, threshold_sigma+3.0) σ | `backend/src/detection.rs:514-516` | 经验调参 | **无出处**：文档未给 7 或 +3σ 的依据（none） | 通用形式，取值为本流程选定 | starlet 提案数 → 候选并集 |
| DET-16 | **DoG 尺度因子** = [0.75, 1.60] ×σ | `backend/src/detection.rs:60；使用 :535-545` | 经验调参 | **无出处**（非等比：1.6/0.75≈2.13）（none） | 只对 FWHM≈2–3 px（0.75σ 在 FWHM 2 px 时 σ=0.64 px，已近采样极限） **[image-specific]** | DoG 覆盖尺度 → 窄/宽提案数 |
| DET-17 | **starlet 提案尺度** = 1.7 / 3.5 px | `backend/src/detection.rs:582-583` | 经验调参 | **无出处**（1.7≈2^0.77、3.5≈2^1.81，不成整数倍频）（none） | 只对 FWHM≈2–3 px **[image-specific]** | starlet_s1/s2 提案数 |
| DET-18 | **UNRESOLVED_BLEND 距离门** = dog_blend_radius_factor 2.5 → 2.5×psf_fwhm（默认 7.5 px） px | `backend/src/detection.rs:59,423；判据 backend/src/measurement.rs:460-466` | 经验调参 | **无出处**（none） | 只对 FWHM≈2–3 px **[image-specific]** | UNRESOLVED_BLEND → quality_count；决定「非 gaussian 提案是否算混叠」 |
| DET-19 | **deblend ΔBIC 下限** = 10.0 ΔBIC | `backend/src/detection.rs:63；判据 backend/src/psf.rs:224,291` | 结构常数 | 统计惯例（Kass&Raftery：ΔBIC>10 为 very strong），代码未注明（value_only） | 通用（模型比较） | 双源确认 → UNRESOLVED_BLEND/deblend 提案保留 |
| DET-20 | **deblend 次分量 SNR** = 5.0 σ | `backend/src/detection.rs:64；判据 backend/src/psf.rs:225,292` | 可配默认 | 无出处（与 min_flux_snr 同值，未说明是否有意对齐）（none） | 通用 | 次源是否认定 → 双源结论 |
| DET-21 | **deblend 主源 SNR** = 12.0 σ | `backend/src/detection.rs:65；判据 backend/src/deblend.rs:80` | 经验调参 | **无出处**（12σ 与 4σ 主阈值无换算关系）（none） | 本流程选定 | 哪些源允许被拆 → 拥挤场候选数 |
| DET-22 | **deblend 残余显著性** = 4.0 σ | `backend/src/detection.rs:66；判据 backend/src/deblend.rs:130,208,252` | 可配默认 | 无出处（取 4=主阈值，未说明）（none） | 通用 | 残余峰是否成为 deblend 提案 |
| DET-23 | **deblend 搜索半径** = 2.0×psf_fwhm（≥3 px；像素窗 ceil(radius) 且 ≥4） px | `backend/src/detection.rs:67；backend/src/deblend.rs:50-51` | 经验调参 | **无出处**（none） | 只对 FWHM≈2–3 px **[image-specific]** | deblend 搜索范围 → 拥挤场新提案数 |
| DET-24 | **deblend 最小间距** = max(1.5, 0.8×psf_fwhm) px | `backend/src/deblend.rs:52` | 经验调参 | **无出处**（FWHM 2 px 时 0.8×FWHM=1.6 略高于 1.5 地板）（none） | 只对 FWHM≈2–3 px **[image-specific]** | deblend 提案与主源的最小可分辨间距 → 近邻结论 |
| DET-25 | **deblend 原始图像约束** = raw−background ≥ 2.0×noise σ | `backend/src/deblend.rs:253` | 经验调参 | **无出处**（与 4σ 残余门语义重叠）（none） | 通用 | deblend 提案是否接受 |
| DET-26 | **拟合最小样本** = 24 像素（deblend 可用样本、PSF pair 拟合同值） px | `backend/src/deblend.rs:178；backend/src/psf.rs:102` | 结构常数 | 最小二乘自由度下限（4–5 参数+余量），代码未注明（reason） | 通用 | 近边缘/掩膜区是否还能做双源判定 |
| DET-27 | **deblend 提案去重** = 半径 max(0.5×min_distance,1.5)；与已接受提案硬编码 2.25 px²（=1.5 px） px | `backend/src/deblend.rs:273,280` | 经验调参 | **无出处**（2.25 写死，与 min_distance 脱钩）（none） | 只与 min_distance 取值配套 **[image-specific]** | deblend 提案被合并/丢弃比例 |
| DET-28 | **PSF 双源 guard 半径** = max(2.5, 1.5×psf_fwhm) px | `backend/src/psf.rs:170` | 经验调参 | **无出处**（none） | 只对 FWHM≈2–3 px **[image-specific]** | 被检对数目 → UNRESOLVED_BLEND 判定数 |
| DET-29 | **PSF 拟合 patch 余量** = margin=max(ceil(3.5σ),4)；每边 ≥5；样本 ≥24 px | `backend/src/psf.rs:64,75,102` | 结构常数 | 3.5σ≈截断到 ~99.95% 能量（与 4σ 同族）（reason） | 通用 | 短边处双源判定是否可得 |
| DET-30 | **PSF σ 换算** = σ = FWHM/2.35482（下限 0.5 px） px | `backend/src/detection.rs:477；measurement.rs:255,313-315；filtering.rs:304；deblend.rs:49；psf.rs:63` | 结构常数 | main.tex:153「Gaussian 尺度以 FWHM=2.35482σ_P 转换」；2.35482=2√(2ln2)（reason） | 通用 | 所有 σ 表达判据（NMS 半径、质心权重、拟合模板） |
| DET-31 | **Gaussian 核截断** = radius=floor(4σ+0.5) σ | `backend/src/filtering.rs:56` | 结构常数 | 4σ 截断惯例（解析可得 99.99% 能量）（reason） | 通用 | 滤波核边缘截断误差 → filter_snr 系统项 |
| DET-32 | **极大值/NMS 半径** = ceil(2.0×σ)（≥1 px），再按 min_distance 抑制 px | `backend/src/filtering.rs:304-307,321；合并 :332-399` | 结构常数 | 2σ 窗口=匹配滤波有效半宽（reason） | 通用 | 候选峰是否保留（双重抑制） |
| DET-33 | **掩膜稀疏快速路径** = 无效像素 ≤2% 时走单遍卷积 fraction | `backend/src/filtering.rs:79,129` | 经验调参 | **无出处**（性能折中；两路径归一化不同）（none） | 通用 | 掩膜密集场下 filter_snr 的归一化口径被静默切换 |
| DET-34 | **噪声归一化下限** = (noise/median_noise).max(0.25) ratio | `backend/src/filtering.rs:185` | 经验调参 | **无出处**（none） | 通用 | 低噪声像元的 filter_snr 最多被放大 4×，直接改弱候选排序与阈值等效 |
| DET-35 | **线状伪影支持阈值** = threshold = max(detection_sigma, 4.0) σ | `backend/src/filtering.rs:413` | 经验调参 | **无出处**（4.0 地板与主阈值 4 同值，但参数化后不再跟随）（none） | 通用 | 线掩膜 → LINE_ARTIFACT → quality_count |
| DET-36 | **线状连通域最小像素** = max(round(2.5×FWHM²), 24) px | `backend/src/filtering.rs:433` | 经验调参 | **无出处**；FWHM=2 时 2.5×4=10，24 的地板完全压过参数化项（none） | 只对 FWHM≈2–3 px 且依赖本图拖影形态 **[image-specific]** | 多少拖影被判为线 → LINE_ARTIFACT 计数 |
| DET-37 | **线状主轴长度** = 4×√major ≥ max(16, 5×psf_fwhm) px | `backend/src/filtering.rs:481` | 经验调参 | **无出处**（5×FWHM=10 时 16 px 地板生效）（none） | 只对 FWHM≈2–3 px **[image-specific]** | <16 px 的短线不可能被判为线 |
| DET-38 | **线状轴比** = √(major/minor) ≥ 4.0 ratio | `backend/src/filtering.rs:482` | 经验调参 | **无出处**（4:1 为常见「明显拉长」经验值）（none） | 通用 | 线/点判定的核心形状门 |
| DET-39 | **线掩膜膨胀** = dilation=max(round(FWHM/2),1) 次 4-邻域膨胀 iter | `backend/src/filtering.rs:489-511` | 经验调参 | **无出处**（none） | 只与 FWHM 相关 **[image-specific]** | LINE_ARTIFACT 牵连邻域 → 被误拒的好源数 |
| DET-40 | **遥测块掩膜** = 行/列前 104 px px | `backend/src/fits.rs:112-120（:118）` | 实测值 | 格式/相机常数（辅助数据块）（reason） | 只对这一台相机（4096²+104 px 辅助块） **[image-specific]** | 边缘有效区、EDGE 判定与空间统计取景 |

## QUA · 质量门（18 条）

| ID | 名称 / 值（单位） | 位置 | 类型 | 出处（sk） | 适用范围 | 改动该值会改变什么 |
|---|---|---|---|---|---|---|
| QUA-01 | **aperture_radius** = 4 px | `backend/src/measurement.rs:100；校验 ≥1 detection.rs:199` | 可配默认 | main.tex:109「孔径半径 4 px」；未给理由（value_only） | 只对 FWHM≈2 px（4 px≈1.7×FWHM） **[image-specific]** | flux/flux_snr → 定标星集合 → ZP 及误差；亦改形态测量权重范围 |
| QUA-02 | **测光 patch 半宽** = outer = radius+4 px | `backend/src/measurement.rs:171` | 结构常数 | 外框=孔径+4（容纳环带）（reason） | 通用 | 环带可用性与边缘判定可用样本 |
| QUA-03 | **背景环带内/外半径** = inner=radius+2，outer=radius+4 px | `backend/src/measurement.rs:196` | 经验调参 | **无出处**（环带宽度固定 2 px，不随 FWHM 变化）（none） | 只对 FWHM≈2 px（FWHM 3 px 时环带吃进 PSF 翼） **[image-specific]** | 局部 background/noise → flux_snr → LOW_FLUX_SNR |
| QUA-04 | **环带最小可用像素** = 16 px | `backend/src/measurement.rs:201,427-429` | 经验调参 | **无出处**（none） | 通用 | 环带不足即 BACKGROUND_UNCERTAIN（拒绝） |
| QUA-05 | **环带 sigma-clip** = 3.0σ × 4 次，样本上限 100000 σ/iter | `backend/src/measurement.rs:208,212` | 结构常数 | 同 DET-07 惯例（value_only） | 通用 | 局部 noise → flux_error → flux_snr |
| QUA-06 | **min_flux_snr** = 5.0 σ | `backend/src/measurement.rs:102；判据 :433` | 可配默认 | main.tex:109「通量 SNR 至少 5」；无推导（value_only） | 通用（5σ 惯例） | 质量源数、定标星数 → ZP；也改 pair/覆盖统计分母 |
| QUA-07 | **FWHM 上下限** = min 0.8 / max 12.0（NARROW/BROAD） px | `backend/src/measurement.rs:103-104；判据 :437-442；校验 detection.rs:207` | 经验调参 | **无出处**；相对实际运行 FWHM 2 px 为 −60%/+500%，门极松（几乎不拒）。注：全仓库不存在「1.3–3.2」的 FWHM 门（none） | 形式通用，区间宽度针对本图 PSF（实际几乎只拦极端值） **[image-specific]** | NARROW/BROAD → quality_count；松门意味着形状门实际未生效 |
| QUA-08 | **max_ellipticity** = 0.65 e=(fx−fy)/max | `backend/src/measurement.rs:105；判据 :443；校验 detection.rs:209` | 经验调参 | **无出处**；0.65 ⇔ 轴比 0.21（很松）。注：代码中不存在「椭率 ≤0.4」门（none） | 形式通用；取值为本图放宽 **[image-specific]** | ELONGATED → quality_count |
| QUA-09 | **sharpness 上下限** = min 0.005 / max 0.85 峰值/孔径通量 | `backend/src/measurement.rs:106-107；判据 :449-453` | 经验调参 | **无出处**（0.005 几乎等于不设下界）（none） | 本流程选定 | DIFFUSE/SPIKE → quality_count |
| QUA-10 | **min_footprint_pixels** = 2（footprint=孔径内残差 ≥1σ 的像元数） px | `backend/src/measurement.rs:108,333-337,454-456` | 经验调参 | **无出处**；判据里的 1σ 地板同样无出处（none） | 通用 | SMALL_FOOTPRINT → quality_count |
| QUA-11 | **min_psf_support_pixels** = 3（校验范围 1..=9） px | `backend/src/measurement.rs:109；判据 :338,344,457-459；校验 detection.rs:213` | 可配默认 | **无出处**；文档里「≥4 支持像素」只是诊断口径（source_artifacts.rs:512/517），不是门限（none） | 只对 FWHM 2 px 的 3×3 支持窗（FWHM 更大时门失效） **[image-specific]** | INSUFFICIENT_PSF_SUPPORT → quality_count，**最敏感的一条**（3→4 显著减少质量源） |
| QUA-12 | **部分掩膜容错** = 有效孔径像素 ≥ max(ceil(0.8×几何像素), 9) px | `backend/src/measurement.rs:382,388-392；开关 detection.rs:416` | 经验调参 | **无出处**（0.8 与 9 均无推导）（none） | 只对含 0 值掩膜的整型帧（本图） **[image-specific]** | PARTIAL_MASKED↔MASKED → quality_count |
| QUA-13 | **EDGE 判定** = patch 触边界即 EDGE bool | `backend/src/measurement.rs:370-372` | 结构常数 | 几何定义（patch=radius+4）（reason） | 通用 | 边缘源一律被拒（不区分截断比例） |
| QUA-14 | **质心加权** = 权重半径 min(radius,max(2.5σ,2))；权重=max(r−0.5σ,0)·exp(−0.5(d/σ)²) px/σ | `backend/src/measurement.rs:256,264-266` | 经验调参 | **无出处**（2.5σ 窗口与 0.5σ 噪声扣除均为经验）（none） | 通用 | x/y 质心 → WCS 残差、匹配率、配对结论 |
| QUA-15 | **值域异常下限** = range_limit = min(−1000, background−50×noise) ADU | `backend/src/measurement.rs:360-365` | 经验调参 | **无出处**（50σ 与 −1000 ADU 地板均无推导）（none） | 只对本图值域（ADU 量级与噪声水平） **[image-specific]** | CODE_PATTERN 的第二条件（须有 ≥1 异常负像元） |
| QUA-16 | **饱和哨兵** = saturation_level = −1.0 表示未设定 ADU | `backend/src/detection.rs:704` | 结构常数 | 哨兵约定（同 gain −1.0，:702）（reason） | 通用 | 下游若当真实值会全帧 SATURATED |
| QUA-17 | **QUALITY_REJECT_FLAGS** = 18 个旗标任一出现即拒 set | `backend/src/measurement.rs:5-24；判定 :25-29` | 结构常数 | 定义式（决定质量门语义边界）（reason） | 通用 | quality_count 的定义；增删旗标直接改所有下游计数 |
| QUA-18 | **质量门语义** = 旗标集为空即通过（无加权/无优先级） bool | `backend/src/measurement.rs:25-29,467` | 结构常数 | 一票否决设计（reason） | 通用 | 任一门变严 ⇒ 质量源 1:1 减少 |

## PHO · 测光层（22 条）

| ID | 名称 / 值（单位） | 位置 | 类型 | 出处（sk） | 适用范围 | 改动该值会改变什么 |
|---|---|---|---|---|---|---|
| PHO-01 | **match_radius（参考星配对）** = 1.2 px | `backend/src/empirical_photometry.rs:67` | 可配默认 | 帮助文本 empirical_photometry.rs:2733「default 1.2」；无推导（value_only） | 只对 WCS 残差 ~0.3–1 px 的本图解 **[image-specific]** | 配对数 → ZP 样本；过小会系统性丢掉暗端 |
| PHO-02 | **bootstrap 半径/迭代** = 4.0 px / 3 次 px/iter | `backend/src/empirical_photometry.rs:68-69（校验≥match :335）` | 可配默认 | 无出处（none） | 只对本图（依赖初始 WCS 与像素尺度） **[image-specific]** | 初始仿射解是否收敛 |
| PHO-03 | **peak_ceiling** = 3990 DN | `backend/src/empirical_photometry.rs:70；判据 :785；范围声明 :2478` | 实测值 | 数据侧有出处：本帧异常重复码 3990–3993（DESIGN.md:366,420）；代码自述「not proven physical full-well saturation」（:2478）（value_only） | 只对本图/该探测器（换探测器须重测） **[image-specific]** | 饱和剔除的配对样本 → ZP（改小则亮星被剔、ZP 偏暗且样本变少） |
| PHO-04 | **测光孔径/环带** = aperture 4.0；annulus 6.0–10.0 px | `backend/src/empirical_photometry.rs:71-73（校验 :337-338）` | 可配默认 | 与 QUA-01/03 同源（4 px 孔径+2/4 px 环带），无独立推导（value_only） | 只对 FWHM≈2 px **[image-specific]** | flux → g_emp；环带 → noise → 误差权重 → ZP 误差 |
| PHO-05 | **稳健裁剪** = 3.0σ × 4 次；样本 <8 放弃 σ/iter | `backend/src/empirical_photometry.rs:74-75,650-675（:651）` | 可配默认 | 惯例；8 的下限无出处（value_only） | 通用 | 内点集 → ZP/C 系数与 in_sample_scatter |
| PHO-06 | **kfold** = 5（校验 ≥2；CV 需 n ≥ kfold+order+7） folds | `backend/src/empirical_photometry.rs:76；校验 :343；门槛 :1493` | 可配默认 | 无出处；+7 的自由度余量为经验调参（value_only） | 通用 | σ_out 稳定性；n 不足则 CV=None（报告缺 holdout） |
| PHO-07 | **block_grid** = 8（校验 ≤32） grid | `backend/src/empirical_photometry.rs:77；校验 :344-345` | 可配默认 | 无出处（空间留一块 out 的分辨率）（none） | 只对 4096² 帧（8×8→每块 512 px） **[image-specific]** | 空间分块 CV 与留一法结论 |
| PHO-08 | **max_iter / min_pairs** = 12 / 20（校验 min_pairs ≥8） iter/count | `backend/src/empirical_photometry.rs:78-79；校验 :346` | 可配默认 | 无出处（none） | 通用 | 稳健拟合是否收敛；样本过少时是否出解 |
| PHO-09 | **随机种子** = 19 seed | `backend/src/empirical_photometry.rs:84` | 实测值 | 固定种子（呼应项目名）用于可复现（reason） | 通用（可复现性） | k 折分配 → σ_out 抖动（不应改结论） |
| PHO-10 | **noise_inflation** = 1.0（自述 unmeasured） ratio | `backend/src/empirical_photometry.rs:85；校验 ≥1 :340；用法 :769` | 可配默认 | 帮助文本「default 1, unmeasured」（value_only） | 通用 | flux_error 整体缩放 → ZP 权重 |
| PHO-11 | **参考误差地板** = PHOTOMETRIC_SIGMA_FLOOR_MAG = 0.05 mag | `backend/src/empirical_photometry.rs:14；声明 :2480` | 经验调参 | **无出处**（0.05 mag 为惯例量级，本图未标定）（value_only） | 通用形式，取值依赖参考星表（Gaia G） | 权重 → ZP 误差与 C 显著性 |
| PHO-12 | **仿射稳健裁剪窗** = clamp(median(d)+3×MADσ, 0.08, 3.0) px，6 次迭代 px/iter | `backend/src/empirical_photometry.rs:1219-1231` | 经验调参 | 有注释无出处：:1229-1230「…A MAD-only cutoff repeatedly trims good stars」；0.08/3.0/6 无推导（value_only） | 只对 WCS 残差 ~0.3–1 px 的本图 **[image-specific]** | 保留配对数 → ZP；过紧截断长尾（真失配） |
| PHO-13 | **颜色项裁决带** = σ_out 比 <0.9 有实质改善；≤1.1 无实质变化 ratio | `backend/src/empirical_photometry.rs:2542-2549` | 经验调参 | **无出处**（±10% 经验带）；同处声明「not a significance test」（:2526）（value_only） | 通用形式，带宽针对本图散点量级 | 是否采用颜色项 → g_emp 色改正结论 |
| PHO-14 | **测光样本前置门** = quality_passed 必须 true（除非 --include-rejected-sources） bool | `backend/src/empirical_photometry.rs:832-841` | 结构常数 | 上游质量门传导（不新增阈值）（reason） | 通用 | 测光样本=质量源；门变严即样本变少 |
| PHO-15 | **星等换算** = −2.5×log10(flux)；1.0857×flux_err/flux mag | `backend/src/empirical_photometry.rs:1986-1992,2263-2268；backend/src/photometry.rs:56` | 结构常数 | Pogson 定义（1.0857=2.5/ln10）（reason） | 通用 | 所有星等量单位一致性 |
| PHO-16 | **定标最低星数** = min_calibrators 3（且须 ≥color_order+2） count | `backend/src/photometric_calibration.rs:194；校验 :220` | 可配默认 | 无出处（3 星拟 2 参数，自由度 1）（none） | 通用 | ZP 是否给出；3 星时不确定度极大 |
| PHO-17 | **测光配对残差上限** = max_match_residual_px = 3.0 px | `backend/src/photometric_calibration.rs:195；判据 :429-433` | 可配默认 | 无出处（none） | 只对 8″/px 像元（3 px≈24″ 的极宽容差） **[image-specific]** | 进入定标的配对数与污染率 |
| PHO-18 | **参考星表质量门** = catalog flux SNR ≥20；RUWE ≤1.4；visibility_periods ≥8 σ/ratio/count | `backend/src/photometric_calibration.rs:196-198；判据 :374-388` | 可配默认 | RUWE 1.4 与 visibility 8 属 Gaia 社区惯例（仓库未注明）（value_only） | 只对 Gaia 参考星表（不是对本图） | 可用定标星数 → ZP 与颜色项 |
| PHO-19 | **定标拟合门** = 3.5σ×5 次；fit RMS ≤0.25；MAD ≤0.25；holdout RMS ≤0.35；holdout bias ≤0.20；单源残差 ≤0.50；holdout ≥2 星；validation_fraction 0.2 mag/σ | `backend/src/photometric_calibration.rs:199-209；门实现 :640-671；holdout 规则 :504-531` | 可配默认 | **全部无出处**（四个 mag 门槛与 3.5σ 均为经验值；3.5σ/5 次也与 detector 的 3.0σ/4 次不一致）（none） | 通用形式，量级针对本图散点（~0.1–0.3 mag） **[image-specific]** | PHOTOMETRY_REJECTED_QUALITY_GATE 是否触发 → 星等是否可用 |
| PHO-20 | **误差→星等换算** = 1.0857362047581296/SNR mag | `backend/src/photometric_calibration.rs:444；source_photometry.rs:109,369；sequence_pipeline.rs:790` | 结构常数 | 2.5/ln10（reason） | 通用 | 权重与星等误差 |
| PHO-21 | **MAD→σ 系数** = 1.4826 系数 | `backend/src/statistics.rs:91,115；photometric_calibration.rs:570,613；empirical_photometry.rs:404` | 结构常数 | 正态下 1/Φ⁻¹(0.75)（reason） | 通用 | 所有稳健尺度的定义 |
| PHO-22 | **自动测光流程默认** = pixel_scale 8.5″；plate tol 0.04；min matches 8；min coverage 0.02；max RMS 2.0 px；LOO RMS 3.0 px；photometry match 1.0 px mixed | `backend/src/photometric_workflow.rs:25-31` | 可配默认 | 板解参数无出处；1.0 px 与 PHO-01 的 1.2 px 不一致，**两套配对半径无统一说明**（none） | 只对该相机像素尺度与 4096² 帧 **[image-specific]** | 是否接受板解 → 是否给星等；配对半径差异改配对数 |

## AST · 天体测量层（18 条）

| ID | 名称 / 值（单位） | 位置 | 类型 | 出处（sk） | 适用范围 | 改动该值会改变什么 |
|---|---|---|---|---|---|---|
| AST-01 | **盲搜索栅格与层数** = bins 1024（→4 px/bin）；search_bins=max(bins/8,32)；层数=trailing_zeros(bins/search_bins)，每层 n×2、(roll,scale) 步长÷4、窗口 ±4 步 px/bin, levels | `backend/src/astrometry_blind.rs:320-325；层数 :1412；每层 :1476-1485` | 可配默认 | 结构式设计（帮助 :41-42）；注释 :13-15 说「±1 previous step window」，实现却是 ±4 步（**注释与实现不一致**）（value_only） | 只对 4096 px 帧（1024 bin→4 px；8192 px 帧变 8 px/bin） **[image-specific]** | 搜索分辨率与耗时；最细 4 px 决定平移精度上限（3×frame/bins=12 px 接受限与之相关） |
| AST-02 | **plate scale 搜索范围** = scale_min 7.6 / scale_max 10.6 arcsec/px | `backend/src/astrometry_blind.rs:394-395（帮助 :43-44）` | 实测值 | 中心值来自板解 8.16（empirical_photometry.rs:59）与自标定真值 8.44（:412）；±14% 的窗口宽度无出处（value_only） | 只对这一台相机 **[image-specific]** | 真值落在窗外则永不命中；窗口越宽代价越高 |
| AST-03 | **roll 步长** = zeta=1.0 → raw=zeta×(frame/n)/r_max，r_max=0.5·frame·√2，再 2π/ceil(2π/raw) rad | `backend/src/astrometry_blind.rs:396；公式 :1190-1195` | 结构常数 | 几何式（一个 bin 的弧长=一个 roll 步），量化到 2π 整数等分（reason） | 通用形式（随 frame/n 缩放） | roll 覆盖率（步长>PSF 宽度可能漏解） |
| AST-04 | **scale 步长** = zeta×(frame/n)×s_ref/r_max arcsec/px | `backend/src/astrometry_blind.rs:1197-1200` | 结构常数 | 同一几何式（径向一个 bin 的伸缩量）（reason） | 通用形式 | scale 分辨率与节点数 |
| AST-05 | **每层保留数** = keep=[32,10,3]（可 --keep 覆盖） count | `backend/src/astrometry_blind.rs:333（帮助 :48）` | 经验调参 | **无出处**（none） | 通用 | 阶梯修剪可能丢掉真解（召回率） |
| AST-06 | **去重容差** = roll 0.75×step；scale 0.006（粗）/0.004（细）；平移 2.0×frame/n（粗）/3.0×frame/n（细） mixed | `backend/src/astrometry_blind.rs:1374-1379,1525-1530` | 经验调参 | **无出处**（0.75/0.006/0.004 无推导）（none） | 通用形式，取值为本流程选定 | 候选去重 → 保留峰数与最终 peak/σ |
| AST-07 | **粗扫统计采样** = sample_sigma=100 trials | `backend/src/astrometry_blind.rs:411（帮助 :49）` | 可配默认 | 无出处（估计每 trial 的 map σ）（none） | 通用 | peak/σ 归一化是否无偏 |
| AST-08 | **自标定注入设置** = scale 8.44″/px；jitter 1.5 px；completeness 0.35；FP 0.30/星；seed 20260330 mixed | `backend/src/astrometry_blind.rs:412-416（帮助 :52-56）` | 实测值 | 8.44 与板解 8.16 同源（实测）；jitter/completeness/FP **无出处**（value_only） | 只对本图（视场、星密度、板解） **[image-specific]** | 自标定能否复原 → METHOD_FAILED 一票否决整条链 |
| AST-09 | **对照规模** = control_trials 1（× 3 对照族）；resolution full（与目标同阶梯） trials | `backend/src/astrometry_blind.rs:419-428；完备性判定 :2588-2591` | 可配默认 | 无出处；但同预算声明有代码级强制（null_sweeps.len()==3×control_trials）（value_only） | 通用 | gate 中的 control 项；trials=1 时 p99 退化为 max（:3122 注释） |
| AST-10 | **判决门** = real peak/σ ≥ max(8.0, 3×max(control_max_peak_σ, control_p99_peak_σ)) σ | `backend/src/astrometry_blind.rs:2560-2564；规则文本 :2626` | 经验调参 | **无出处**：3 倍系数与 8σ 地板均无推导（8 又与自标定门 8.0 同值，:3034）（none） | 通用形式；control 项是本图实测，8σ 是固定地板 **[image-specific]** | VALIDATED/UNVALIDATED 的唯一主判据 → 是否输出 WCS/星等 |
| AST-11 | **match audit 半径与对照偏移** = 半径 [1,2,3,5,8] px；内区 frame/4；对照平移 (37, 53) px px | `backend/src/astrometry_blind.rs:2170,2189,2493；注释 :2111-2115` | 经验调参 | 半径梯度无出处；(37,53) 偏移无出处（注释只说 fixed offset）（none） | 只与 4096 px 帧/4 px 像元相关 **[image-specific]** | on2/off2 → audit_ok |
| AST-12 | **match audit 判据** = 2 px 匹配数 ≥50 且 on/off ≥3.0 count/ratio | `backend/src/astrometry_blind.rs:2586` | 经验调参 | **无出处**（50 与 3.0 均无推导）（none） | 只对 4096² 帧+本星表密度（50 与本帧星表星数绑定） **[image-specific]** | audit_ok → validated；与 peak/σ 门并列的第二道独立判据 |
| AST-13 | **自标定接受限** = peak ≥8.0；\|Δroll\| ≤1.5×roll_step0；\|Δscale\| ≤0.02；\|Δt\| ≤3×frame/bins；parity 相同 mixed | `backend/src/astrometry_blind.rs:3007,3034-3038` | 经验调参 | 1.5×/0.02/3× 三个系数**无出处**；8.0 与判决门同值（none） | 只对 4 px/bin 最细栅格（3×frame/bins=12 px） **[image-specific]** | baseline_ok → 否则 METHOD_FAILED，整链无输出 |
| AST-14 | **selftest 收紧容差** = \|Δroll\| ≤0.06°；\|Δscale\| ≤0.003 deg/frac | `backend/src/astrometry_blind.rs:3491-3492` | 经验调参 | **无出处**（比 AST-13 严约 10 倍）（none） | 纯合成几何（与真实数据无关） | --selftest 是否通过 |
| AST-15 | **log-polar 网格** = r_min 1、r_max=n0/2、dlr=ln(r_max/r_min)/(nl−1)、角向 2π/nl；白化 eps=0.1×mean\|C\| bin/rad | `backend/src/astrometry_blind.rs:1621-1633；白化 :1702；报告 :1869-1871,1917-1923` | 结构常数 | 对数极坐标定义（结构性）；eps 0.1 为经验（reason） | 通用形式 | scale 比值与 roll 分辨率（radial_log_step 直接写进报告） |
| AST-16 | **输入门** = peak_ceiling 20000 DN；catalog G∈[3,13]；frame 4096 px DN/mag/px | `backend/src/astrometry_blind.rs:393,407-408,398（帮助 :33-37）` | 可配默认 | G 3–13 属 Gaia 饱和/信噪比惯例；20000 DN 与 PHO-03 的 3990 相差 5 倍——**两处饱和天花板不一致且无说明**（value_only） | 只对本图/该星表 **[image-specific]** | 参与搜索的点数 → 自标定与真实搜索统计量 |
| AST-17 | **guided 窗口** = roll ±3.0°；scale ±1%（与 logpolar 互斥） deg/frac | `backend/src/astrometry_blind.rs:377-378（帮助 :72-73）` | 可配默认 | 无出处（none） | 仅作先验窗口（工具显式标 VALIDATED_GUIDED，不算盲解 :2596,2612） | 有先验时的收敛速度与「是否仍算盲解」的结论 |
| AST-18 | **null p99 定义** = 对各 null sweep 取第 99 百分位（trials=1 时退化为 max） σ | `backend/src/astrometry_blind.rs:3122-3123,3278；用法 :2555-2559` | 结构常数 | 注释已明写退化（:3122）（reason） | 通用 | gate 的 control 项 |

## COV · 覆盖守卫（7 条）

| ID | 名称 / 值（单位） | 位置 | 类型 | 出处（sk） | 适用范围 | 改动该值会改变什么 |
|---|---|---|---|---|---|---|
| COV-01 | **Full 覆盖判据** = 分离角 ≤0.35° deg | `backend/src/catalog_coverage.rs:7；判定 :197；注释 :6,189` | 经验调参 | 注释只说「accepted as a full-coverage solution」；0.35° 与本相机 9.78° 视场（3.6%）无换算说明（value_only） | 形式通用，数值针对本相机视场与 2° tile **[image-specific]** | Full/Partial/NotCovered → NotCovered 直接 blocks_run |
| COV-02 | **Partial 覆盖判据** = 分离角 ≤0.5×星表半径 deg | `backend/src/catalog_coverage.rs:199（注释 :189）` | 结构常数 | 几何式（隐含 FOV ≤ 星表半径的假设，本图 9.78° 视场远大于 2° tile → 假设不成立）（reason） | 通用形式，但在本相机下几何前提失效 **[image-specific]** | Partial 与 NotCovered 的分界 → 是否放行 |
| COV-03 | **覆盖半径 99 分位** = RADIUS_PERCENTILE=0.99 fraction | `backend/src/catalog_coverage.rs:9；使用 :178；注释 :8` | 可配默认 | 注释给出理由：「ignores a few outlier rows without inflating the circle」（reason） | 通用 | radius_deg → COV-01/02 的阈值 |
| COV-04 | **中心估计 1% 修剪** = trim=floor(0.01×N)，midrange(trim…N−trim) fraction | `backend/src/catalog_coverage.rs:70-78（注释 :68-69）；RA 环内 :91-99` | 经验调参 | 注释解释「为什么修剪」（中位值会被密度梯度偏置），但 1% 无推导（value_only） | 通用 | 中心 RA/Dec → 分离角 → 覆盖状态 |
| COV-05 | **Gaia tile 步长因子** = cone 0.8×tile_radius；box 1.3×tile_radius ×radius | `backend/src/gaia_tiled.rs:11,16；注释 :12-15` | 结构常数 | **有完整推导**：切平面单元最远点=对角线一半（1.3/√2=0.92）<1，故整格仍在一个 tile 内（reason） | 通用（几何） | tile 数与查询数 → 星表完整性 |
| COV-06 | **box rotation margin** = factor=cosθ+sinθ（θ=\|roll mod 90\|，校验 0…45°） ×half-width | `backend/src/gaia_tiled.rs:232-235；注释 :225-231；校验 :1156；字段 :100,179` | 结构常数 | **有完整推导**：旋转正方形外接轴对齐框半宽=w(cosθ+sinθ)（reason） | 通用式；θ 是**实测 roll**（本图 129.85°→mod 90=39.85°→1.41），故该守卫对本图姿态生效 **[image-specific]** | box 查询范围：margin 少算即丢探测器四角真星 → 覆盖误判 |
| COV-07 | **相机视场与星等范围** = FOV 9.78°×9.78°；G∈[5.0,13.5]；tile 半径 2.0°；tile 上限 50000；深度 2；查询上限 2000；超时 30 s deg/mag | `backend/src/public_catalog.rs:16-24（同值 gaia_frame.rs:10-23）` | 实测值 | 分别为本相机实测视场与可用星等；无文档推导（value_only） | 只对这一台相机 **[image-specific]** | 星表查询范围 → 覆盖判定与完备度审计基准 |

## OTH · 其它影响判据的常数（7 条）

| ID | 名称 / 值（单位） | 位置 | 类型 | 出处（sk） | 适用范围 | 改动该值会改变什么 |
|---|---|---|---|---|---|---|
| OTH-01 | **−1 哨兵** = max_sources −1；gain −1.0；saturation_level −1.0 sentinel | `backend/src/detection.rs:702,704` | 结构常数 | JSON 元数据编码约定（reason） | 通用 | 下游若当数值用会得到荒谬判据（−1 ADU 饱和=全帧饱和） |
| OTH-02 | **阈值扫描集合** = [4,5,6,8,10,12,15,20] σ σ | `backend/src/detection_sweep.rs:22` | 结构常数 | 扫描用（非判据）；4 与 5 恰为主阈值与 min_flux_snr（reason） | 通用 | 不改单帧结论，只改敏感性分析覆盖 |
| OTH-03 | **经验 PSF 审计常数** = 支持半径 7 px；模板上限 64；相关系数门 0.8 px/count/ratio | `backend/src/source_psf.rs:19-21` | 经验调参 | **无出处**（none） | 只对 FWHM≈2 px（7≈3.5×FWHM） **[image-specific]** | 经验 PSF 形状/相关度诊断（不改质量门，但改「PSF 合规」证据） |
| OTH-04 | **支持像素诊断计数** = support ≥4 与 ≥5 的计数（不参与判定） px | `backend/src/source_artifacts.rs:471,476-487；口径文本 :512,517` | 结构常数 | 文档明确「诊断口径…不改变 quality_count」「不因接近外部数量而自动采用」（reason） | 通用 | 仅报告；**极易被误读为门限 4**（实际门限是 QUA-11 的 3） |
| OTH-05 | **空间统计网格** = 16×16（变异系数）；spatial PSF per_class 8、grid 4 grid | `backend/src/source_artifacts.rs:601；backend/src/source_audit.rs:56-57` | 可配默认 | 无出处（描述性统计分辨率）（none） | 只对 4096² 帧 **[image-specific]** | 仅报告的空间均匀性指标 |
| OTH-06 | **高 SNR 切片阈值** = high_flux_snr_threshold（调用方给定，测试用 10.0） σ | `backend/src/source_classification.rs:263,872,882` | 可配默认 | 注释 :258「remains a diagnostic; it does not alter quality flags」（reason） | 通用 | 仅分类统计的高 SNR 子集 |
| OTH-07 | **常见旗标取前 N 名** = 6（形态表）/4（子类与特征表） count | `backend/src/source_classification.rs:341,531,676` | 结构常数 | 输出格式约定（reason） | 通用 | 仅报告列宽 |

## 经验调参逐条判定（依据 / 独立验证 / 等价物理改写）

共 42 条【经验调参】，其中 36 条连数值都无出处。

| ID | 名称 | 依据（代码/文档是否给出） | 独立于本图的验证 | 按物理量重写的等价条件 |
|---|---|---|---|---|
| DET-10 | 饱和推断与掩膜（None → max−32 ADU（i16→32735）；掩膜 value ≥max−32 或 ≤min+32） | 无出处，推断为对 int16 值域留白（32767−32） | 无：没有暗/平场或满阱标定产物 | saturation_level 应取 FITS 头 SATURATE/DATAMAX 或平场标定；缺失时输出「未标定」而非隐式 32735 |
| DET-11 | 负溢出下限（−0.9×i16max = −29491） | 无出处；0.9×满量程=「接近整数下溢」的经验近似 | 无（DESIGN.md 只把它当值域风险描述） | 按 ADC 有符号约定由头文件给出「≤ −(满量程−偏移)」 |
| DET-14 | DoG 响应阈值（max(6.0, threshold_sigma+2.0)） | 无出处：+2σ 无灵敏度或假阳性推导 | 无（没有 DoG 阈值扫描） | 按与主阈值「相同的每像素假阳性率」折算 DoG 阈值（需除以实测 DoG 噪声比） |
| DET-15 | starlet 响应阈值（max(7.0, threshold_sigma+3.0)） | 无出处，推断为让 starlet 比 DoG 更保守 | 无 | 同 DET-14，按等效假阳性率折算或用统一定义的响应显著性 |
| DET-16 | DoG 尺度因子（[0.75, 1.60]） | 无出处 | 无 | 写成 [0.5,2.0]×σ_PSF 的对数网格，下限受采样定理约束（σ≥0.6 px） |
| DET-17 | starlet 提案尺度（1.7 / 3.5） | 无出处 | 无 | B3-spline starlet 的物理尺度应写成 2^k×σ_PSF |
| DET-18 | UNRESOLVED_BLEND 距离门（dog_blend_radius_factor 2.5 → 2.5×psf_fwhm（默认 7.5 px）） | 无出处；2.5×FWHM 是常见的「互相污染半径」直觉值 | 无注入回收给出该半径的最优值 | 改为「两源 PSF 重叠积分 > 阈值」或「间距 <k×FWHM 且峰比 <r」的双条件，k 由注入标定 |
| DET-21 | deblend 主源 SNR（12.0） | 无出处，推断为避免在 4–12σ 做不可靠拆分 | 无 | 可由「ΔBIC 增益 ≥10 且次源 SNR ≥5」单条件取代；主源 SNR 只作预筛 |
| DET-23 | deblend 搜索半径（2.0×psf_fwhm（≥3 px；像素窗 ceil(radius) 且 ≥4）） | 无出处；2×FWHM=「同级污染」经验半径 | 无 | 搜索半径=PSF 有效拟合半径，由注入回收标定，并随 FWHM 参数化 |
| DET-24 | deblend 最小间距（max(1.5, 0.8×psf_fwhm)） | 无出处 | 无（无注入双源的分辨极限实测） | 由注入回收的 50% 分辨极限（≈1.0×FWHM）给出，并保留该曲线 |
| DET-25 | deblend 原始图像约束（raw−background ≥ 2.0×noise） | 无出处 | 无 | 与 DET-22 合并为单一显著性判据，避免 double-counting |
| DET-27 | deblend 提案去重（半径 max(0.5×min_distance,1.5)；与已接受提案硬编码 2.25 px²（=1.5 px）） | 无出处；硬编码 2.25 与参数脱钩 | 无 | 统一为 k×PSF_FWHM 的单一去重半径 |
| DET-28 | PSF 双源 guard 半径（max(2.5, 1.5×psf_fwhm)） | 无出处 | 无 | 1.5×FWHM≈两源 PSF 重叠 50% 的距离，可写成重叠判据 |
| DET-33 | 掩膜稀疏快速路径（无效像素 ≤2% 时走单遍卷积） | 无出处；2% 是性能阈值却改变算法路径 | 无：无两路径差异量化 | 对同一帧强制走归一化两遍路径，或输出路径标记并按路径分别统计 |
| DET-34 | 噪声归一化下限（(noise/median_noise).max(0.25)） | 无出处；0.25=允许放大 4 倍 | 无 | 按噪声图 χ² 上界（99% 分位）设下限，或直接用未归一化 σ 显著性 |
| DET-35 | 线状伪影支持阈值（threshold = max(detection_sigma, 4.0)） | 无出处 | 无 | 线状结构用独立的形态显著性阈值，与主阈值解耦 |
| DET-36 | 线状连通域最小像素（max(round(2.5×FWHM²), 24)） | 无出处；24 是硬地板使 FWHM 标度失效 | 无（DESIGN.md 仅记 96/96 候选落在同一单元，无阈值扫描） | 最小像素 = π(1.5×FWHM)²（≈2 个 PSF 面积），去掉 24 地板 |
| DET-37 | 线状主轴长度（4×√major ≥ max(16, 5×psf_fwhm)） | 无出处 | 无 | 主轴 ≥3×FWHM（点源/拖影经典判据），去掉 16 px 地板 |
| DET-38 | 线状轴比（√(major/minor) ≥ 4.0） | 无出处 | 无 | 按二阶矩椭率 e>0.88（对应轴比 4:1），或与单点 PSF 模板的拟合优度差 |
| DET-39 | 线掩膜膨胀（dilation=max(round(FWHM/2),1) 次 4-邻域膨胀） | 无出处 | 无 | 膨胀半径=1×FWHM（覆盖 PSF 溢出）并写明 |
| QUA-03 | 背景环带内/外半径（inner=radius+2，outer=radius+4） | 无出处；环带宽度 2 px 与 PSF 翼宽无参数化关系 | 无 | 环带=[1.5,3.0]×FWHM 且环带面积 ≥ 一定像素数 |
| QUA-04 | 环带最小可用像素（16） | 无出处（16 px 太少，σ 相对误差≈18%） | 无 | 要求环带像素 ≥ 满足 noise 相对误差目标的数量（如 ≥100 px→7%） |
| QUA-07 | FWHM 上下限（min 0.8 / max 12.0（NARROW/BROAD）） | 无出处；0.8 与 12.0 无物理/统计推导 | 无 FWHM 分布与拒绝率的对照（feature_gate_margin 只报余量） | FWHM ∈ [0.7,1.4]×PSF_FWHM（≈±30%），且 PSF_FWHM 改为实测中位 FWHM |
| QUA-08 | max_ellipticity（0.65） | 无出处，推断为避免与线状判定重叠而放宽 | 无 | 上限=max(0.15, PSF 椭率 p99+0.1)；0.4 ⇔ 轴比 0.43（可作参考锚点） |
| QUA-09 | sharpness 上下限（min 0.005 / max 0.85） | 无出处 | 无 | 上界写成 1/有效像元数的函数，或改用与 PSF 模板的拟合残差 |
| QUA-10 | min_footprint_pixels（2（footprint=孔径内残差 ≥1σ 的像元数）） | 无出处 | 无 | ≥覆盖 50% 能量的像元数（PSF 积分给出） |
| QUA-12 | 部分掩膜容错（有效孔径像素 ≥ max(ceil(0.8×几何像素), 9)） | 无出处 | 无 | 容错下限写成「缺失面积 <x% ⇒ 通量改正不确定度 <y mag」，由注入给出 |
| QUA-14 | 质心加权（权重半径 min(radius,max(2.5σ,2))；权重=max(r−0.5σ,0)·exp(−0.5(d/σ)²)） | 无出处 | 无（DESIGN.md 只记 centroid_method=psf_weighted_noise_floor，无最优性验证） | 窗口=匹配滤波有效半宽（≈1.5×FWHM），噪声扣除取 σ 的固定分位 |
| QUA-15 | 值域异常下限（range_limit = min(−1000, background−50×noise)） | 无出处 | 无 | 写成「≤−(k×满量程) 或 ≤bg−kσ，k 由 ADC 位宽与读出噪声决定」 |
| PHO-11 | 参考误差地板（PHOTOMETRIC_SIGMA_FLOOR_MAG = 0.05） | 无出处；与 Gaia G 在 G≈13–14 的内部误差量级相符 | 部分：报告同时给 sigma_scope 与 residual_scaled_standard_error（:2492-2493）提示系统项未计入 | 直接使用参考星表 mG 误差列；缺列时输出「权重未标定」 |
| PHO-12 | 仿射稳健裁剪窗（clamp(median(d)+3×MADσ, 0.08, 3.0) px，6 次迭代） | 3×MAD+median 的合理性有注释；上下界无出处 | 间接：注释显示这是对「纯 MAD 截断」失败模式的修正 | 窗=k×WCS 残差 σ（k 由期望失配率给出），上下界取残差分布实测分位 |
| PHO-13 | 颜色项裁决带（σ_out 比 <0.9 有实质改善；≤1.1 无实质变化） | 无出处 ±10% | 部分：比较同一批样本同一折的 holdout RMSE（配对比较，:2534-2535），但无置信区间 | 用配对 bootstrap/F 检验给出 ΔRMSE 区间，判据写成「区间不含 0」 |
| AST-05 | 每层保留数（keep=[32,10,3]（可 --keep 覆盖）） | 无出处 | 无（无「保留数 vs 命中率」扫描） | 按误检预算（漏检概率 <ε 的峰数）或 significance 阈值替代固定个数 |
| AST-06 | 去重容差（roll 0.75×step；scale 0.006（粗）/0.004（细）；平移 2.0×frame/n（粗）/3.0×frame/n（细）） | 无出处 | 无 | 容差=1 个 bin 对应的参数偏差（纯几何），去掉 0.006/0.004 |
| AST-10 | 判决门（real peak/σ ≥ max(8.0, 3×max(control_max_peak_σ, control_p99_peak_σ))） | 3× 与 8 无推导；同一常数兼作自标定成功门，二者是否应相等未讨论 | **有强独立验证**：null sweeps、invariance、translate 三类对照族+自标定，gate 用对照实测最大值 | 写成「给定 N_trials 下的家族式假阳性率 α」的等效阈值（α=0.01 时期望最大 σ≈√(2 ln(N/α))），8σ 由 N、α 推出 |
| AST-11 | match audit 半径与对照偏移（半径 [1,2,3,5,8] px；内区 frame/4；对照平移 (37, 53) px） | (37,53) 既非整数 bin 也非特征尺度，无推导 | 有结构性对照：同一 audit 在平移后重算（真零假设对照） | 对照偏移应 ≥ 若干倍星表相关长度并随机多次取均值 |
| AST-12 | match audit 判据（2 px 匹配数 ≥50 且 on/off ≥3.0） | 无出处；50 明显与本帧星表星数（几千）绑定 | 有对照侧 off2，比值判据已零假设归一；但 50 未归一 | 匹配数按帧内星表星数归一（如 ≥5%），比值 ≥ 随机期望的 χ² 上界 |
| AST-13 | 自标定接受限（peak ≥8.0；\|Δroll\| ≤1.5×roll_step0；\|Δscale\| ≤0.02；\|Δt\| ≤3×frame/bins；parity 相同） | 容差与栅格成正比（可辩护），系数无推导 | 自标定本身是对本图的独立检验，但无「容差 vs 假失败率」标定 | 容差=k×对应栅格分辨率，k 由 selftest 重复试验的分布给出 |
| AST-14 | selftest 收紧容差（\|Δroll\| ≤0.06°；\|Δscale\| ≤0.003） | 无出处 | 不适用（自身即验证器） | 由 FFT 栅格量化误差解析给出 |
| COV-01 | Full 覆盖判据（分离角 ≤0.35°） | 无出处；0.35° 既非 tile 半径整数分之一也非视场整比例 | 有单元测试（验证行为不验证取值合理性） | 改为「星表圆心到视轴 ≤k×视场半宽且四角都在星表半径内」，k 由边缘缺星率标定 |
| COV-04 | 中心估计 1% 修剪（trim=floor(0.01×N)，midrange(trim…N−trim)） | 1% 无出处 | 有单元测试 trimmed_center_ignores_an_outlier_row（100 行插 1 离群） | 修剪比例由星表离群率自适应（如 3σ 迭代） |
| OTH-03 | 经验 PSF 审计常数（支持半径 7 px；模板上限 64；相关系数门 0.8） | 7 与 0.8 无推导 | 无 | 支持半径 ≥3×FWHM；相关门按模板数与噪声给零假设分布 |

## 与交付文档不一致之处（必须显式标注）

| # | 文档/简报说法 | 代码实际 | 后果 |
|---|---|---|---|
| 1 | 椭率 ≤ 0.4 | 全仓库不存在 0.4 的椭率门；`max_ellipticity` 默认 **0.65**（measurement.rs:105），交付运行亦为 0.65（deliverables/paper/evidence.json:33） | 「椭率 ≤0.4」若写进论文即与代码/证据不符 |
| 2 | FWHM 1.3–3.2 px | 全仓库不存在 1.3–3.2 的区间；门是 **0.8–12.0 px**（measurement.rs:103-104） | 门极松（−60%/+500%），形状门实际几乎不拒 |
| 3 | 最小峰距 4 px | `DetectionOptions::default().min_distance = 3`（detection.rs:49），仅 `rst19`/`rst19-sources` 覆盖为 4（source_audit.rs:51） | 走库默认的调用方（序列等）实际是 3 |
| 4 | PSF FWHM 2 px | 默认 `psf_fwhm = 3.0`（measurement.rs:113）；交付运行显式传 2.0（evidence.json:26） | 默认路径与论文路径的 NMS/孔径/线状判据全部不同 |
| 5 | 「≥4 支持像素」 | 门是 **3**（measurement.rs:109）；≥4/≥5 只是诊断计数（source_artifacts.rs:512,517） | 把诊断口径当门限是统计口径错位 |
| 6 | 饱和天花板 | 测光用 **3990 DN**（empirical_photometry.rs:70），天体测量用 **20000 DN**（astrometry_blind.rs:393），detector 掩膜用 **32735**（detection.rs:156） | 三套互不一致且无说明 |
| 7 | 天体测量阶梯 | 注释写「±1 previous step window」（astrometry_blind.rs:13-15），实现是 **±4 步**（:1482-1485） | 注释与实现矛盾 |
