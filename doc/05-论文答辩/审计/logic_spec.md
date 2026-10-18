# RST19 实现逻辑规格（可引用版）

审计范围：backend/src/（Rust 实现，非 Python；Python 版仅作为命名血缘）。所有行号取自当前工作树，逐条核对过源码。
审计方式：只读；未修改任何源码。数据侧抽样核对使用了 tmp/rust-sources-prod/source_catalog.csv。

阅读约定：“门限数值”只在与语义不可分割时才出现；本文件的主体是**逻辑与语义**。

---

## L1 检测链的真实顺序

### L1-1 从 FITS 到候选列表的调用链

**结论**：入口是 pipeline（CLI 在 cli.rs，GUI/桌面在 main.rs），检测是单趟函数式管线：掩膜 -> 全局统计 -> 局部背景/噪声 -> 线性伪迹掩膜 -> 匹配滤波 -> 标准化 -> 峰值提案 -> 合并 -> 可选局部去混叠 -> 截断 -> 逐源测光 -> 单/双 PSF 证据。关键函数（按执行顺序）：

1. pipeline::analyze_frame_with_progress —— 组织阶段与进度（pipeline.rs:205）
2. fits::auxiliary_mask —— 遥测区掩膜（fits.rs:112）
3. detection::detect_sources / detect_typed —— 精度分发 + 主流程（detection.rs:291 / 378）
4. detection::working_mask_typed —— 构造 effective 掩膜（detection.rs:257）
5. statistics::sigma_clipped_stats —— 全局与每-box 稳健估计（statistics.rs:77）
6. statistics::local_background_rms + resize_grid —— 背景/噪声图（statistics.rs:186 / 120）
7. filtering::linear_artifact_mask / linear_support_mask —— 线状伪迹像素掩膜（filtering.rs:400 / 422）
8. filtering::masked_gaussian_response —— 掩膜归一化高斯（filtering.rs:67）
9. filtering::standardize_response —— 噪声加权标准化（filtering.rs:156）
10. filtering::response_proposals + merge_proposals —— 峰值提案与多方法合并（filtering.rs:292 / 332）
11. deblend::local_deblend_proposals + append_deblend_proposals —— 可选局部去混叠（deblend.rs:26 / 264）
12. measurement::source_from_peak —— 孔径测光与质量旗标（measurement.rs:130）
13. psf::resolve_supplemental_blends_typed / guard_unresolved_gaussian_pairs_typed —— 单/双 PSF 证据（psf.rs:253 / 159）

**证据**：

    backend/src/cli.rs:245                    let analysis = pipeline::analyze_frame_with_progress(
    backend/src/pipeline.rs:217               let mask = fits::auxiliary_mask(frame.height, frame.width)?;
    backend/src/pipeline.rs:219               let detection = detection::detect_sources(
    backend/src/detection.rs:404              let (effective, saturation, used_zero) = working_mask_typed(
    backend/src/detection.rs:412              let (background, noise) = T::stats(&numeric, Some(&effective), 3.0, 4, 1_000_000)?;
    backend/src/detection.rs:440              T::background(
    backend/src/detection.rs:464              let line = if options.reject_linear_artifacts {
    backend/src/detection.rs:478              let (filtered, filter_mode) = T::gaussian(
    backend/src/detection.rs:486              let (filter_snr, _, filter_noise) =
    backend/src/detection.rs:597              let mut selected = filtering::merge_proposals(&gaussian, &extras, options.min_distance);
    backend/src/detection.rs:637              if let Some(limit) = options.max_sources {
    backend/src/detection.rs:652              measurement::source_from_peak(
    backend/src/detection.rs:674              let (mut sources, pair_tested, pair_guarded, pair_radius) =

**边界**：detect_sources 只做 f32/f64 静态分发（detection.rs:299-303），两种精度跑同一套算法；掩膜、背景、提案、测光全部在 f64 接口上返回 f64 数值。

### L1-2 effective 掩膜的构造与来源

**结论**：effective 是四类来源的逻辑或，逐像素构造，顺序固定：用户/遥测掩膜、非有限值、（可选）零值、（饱和判定）。各来源与默认开关：

- 用户/遥测掩膜：pipeline 无条件传入 fits::auxiliary_mask；该掩膜只置位**第 0 行前 min(width,104) 个像素**（104=208 字节遥测区的 16-bit 口径），不是 104x104 方块。
- 非有限值：永远屏蔽，无开关。
- 零值：mask_zero_pixels 默认 None，即“整数类型且 |全局背景|>1”时启用；浮点数据默认不启用。
- 饱和：measurement.saturation_level 为 None 时，整数类型用 max-32 与 min+32 **双向**屏蔽极端值；给定阈值时只按 value >= 阈值 屏蔽。
- allow_partial_zero_mask 默认 None，运行时取 used_zero（即零值掩膜是否启用），决定 PARTIAL_MASKED 是否可能取代 MASKED。

**证据**：

    backend/src/fits.rs:118                   mask[..width.min(104)].fill(true);
    backend/src/detection.rs:155              .unwrap_or(limits.is_some() && background.is_finite() && background.abs() > 1.0);
    backend/src/detection.rs:160-162          provided.is_some_and(|m| m[i]) || !value.is_finite() || (zeros && value == 0.0)
    backend/src/detection.rs:164              limits.is_some_and(|(min, max, _)| value >= max - 32.0 || value <= min + 32.0)
    backend/src/detection.rs:53-54            mask_zero_pixels: None, allow_partial_zero_mask: None,
    backend/src/detection.rs:416              measurement.allow_partial_zero_mask = options.allow_partial_zero_mask.unwrap_or(used_zero);

**边界**：全局背景/噪声先用 base_mask（仅非有限值+用户掩膜）估计，再用 effective 重估（detection.rs:397-412），所以掩膜本身会影响背景水平；这是一次固定的两步，不迭代。

### L1-3 背景图与噪声图的生成

**结论**：按 background_box_size 划分互不重叠的 box（默认 128 px，行列方向各自 div_ceil），每 box 内用 sigma-clipped 中位数+MAD（3σ，迭代 4 次或 2 次，取决于 refine_local_background；抽样上限 background_sample_limit=100000）；box 估计失败、非有限或噪声<=0 时整块回退到调用方传入的全局 (background, noise)。随后用**双线性**把 gh x gw 网格放大到全图；缩放采用 round_ties_even 的采样约定，越界部分填最后一个采样值，噪声图最后统一 max(EPSILON)。

**证据**：

    backend/src/detection.rs:51-52            background_box_size: 128, background_sample_limit: 100_000,
    backend/src/statistics.rs:213-214         let gh = height.div_ceil(box_size); let gw = width.div_ceil(box_size);
    backend/src/statistics.rs:230-231         if !stats.0.is_finite() || !stats.1.is_finite() || stats.1 <= 0.0 { stats = global; }
    backend/src/statistics.rs:238-239         resize_grid(&centers, gh, gw, height, width)?, resize_grid(&noises, gh, gw, height, width)?
    backend/src/detection.rs:364-368          iterations: if options.measurement.refine_local_background { 4 } else { 2 },
    backend/src/detection.rs:431              background_mode = "shared_sequence_pilot";
    backend/src/detection.rs:438              background_mode = "per_frame_local_shape_fallback";

**边界**：序列快速路径可注入首帧模型（background_mode=shared_sequence_pilot），尺寸不符时退回逐帧（detection.rs:431-439）；单帧路径永远是 per_frame_local。

### L1-4 三条提案路径的启用条件

**结论**：proposal_mode 只接受 gaussian / hybrid / ensemble（大小写与空白归一）。库默认 gaussian（DoG 与 starlet 都不跑），CLI 额外注入 hybrid。语义是包含关系：

- gaussian：残差 -> 高斯（sigma=FWHM/2.35482）-> 标准化 -> 峰值提案。
- hybrid：gaussian + 两条 DoG（窄 0.75、宽 1.60 倍 sigma；差分为 other-filtered 或 filtered-other），DoG 提案受“原残差必须是 3x3 局部最大且 >= dog_min_peak_sigma*noise”的 gate 约束。
- ensemble：hybrid + 两层 starlet（w1、w2，3x3 B3-spline a trous），同样受 starlet_min_peak_sigma 的局部极大 gate。
- 两条 DoG 阈值与 starlet 阈值未显式给定时按 threshold_sigma 派生（6或4+2；7或4+3）。

**证据**：

    backend/src/detection.rs:56               proposal_mode: "gaussian".into(),
    backend/src/cli.rs:216                    fields.entry("proposal_mode").or_insert(json!("hybrid"));
    backend/src/detection.rs:513              .unwrap_or(6.0f64.max(options.threshold_sigma + 2.0));
    backend/src/detection.rs:525              if mode == "hybrid" || mode == "ensemble" {
    backend/src/detection.rs:569              if mode == "ensemble" {
    backend/src/detection.rs:530-532          valid[i] && *r == maxima.as_ref().unwrap()[i] && *r >= T::from_f64(options.dog_min_peak_sigma) * noise_map[i]

**边界**：三条路径的提案最终都进 merge_proposals，同一 (x,y) 在 min_distance 内合并为一条，methods/scales 记录来源；因此 methods 是“证据集合”，不是“选中算法”。

### L1-5 去混叠的触发、输出与合并

**结论**：代码里有**两套**不同的“去混叠”，不要混为一谈：

(A) 局部残差去混叠 local_deblend_proposals：默认**关闭**（enable_local_deblend=false），且 fast_sequence 时强制跳过。仅在开启时，对 score>=deblend_primary_snr_min 的 gaussian 提案逐个做局部最小二乘（设计矩阵 [1, gx/N, gy/N, Gaussian]），在残差图上找 3x3 局部极大、距主峰 >= 1.5(或 0.8*FWHM)、<= 2*FWHM、残差 >= min_residual_sigma*noise 且 image-background >= 2*noise 的最佳点；输出 method="deblend_local" 的 Proposal。合并时按 score 降序，与已接受提案冲突阈值是先 0.5*min_distance 后再硬编码 1.5 px（2.25 px^2），通过者追加并重排。
(B) 单/双 PSF 证据：总是运行。先在“非 gaussian 且距最近 gaussian 提案 < 2.5*FWHM”的源上加 UNRESOLVED_BLEND，再对 gaussian 源做 pair 检验；ΔBIC 与分量 SNR 未同时达标的副源被强制 quality_passed=false 并记录 deblend_delta_bic / deblend_component_snr；resolve_supplemental_blends 会用最近 gaussian 邻点重算一次，达标就移除 UNRESOLVED_BLEND 并重算 quality。

**证据**：

    backend/src/detection.rs:598             let deblend_proposals = if options.enable_local_deblend && !options.fast_sequence {
    backend/src/deblend.rs:80                 .filter(|p| p.score >= options.min_primary_snr)
    backend/src/deblend.rs:252-253           && value >= gate.threshold * noise && ctx.image.value(*index) - ctx.background.value(*index) >= 2.0 * noise
    backend/src/deblend.rs:273-280           let radius = (0.5 * min_distance as f64).max(1.5).floor().max(1.0); ... < 2.25
    backend/src/psf.rs:236                   s.quality_passed = false;
    backend/src/psf.rs:294-296               source.flags.retain(|f| f != "UNRESOLVED_BLEND"); source.quality_passed = quality_passes(&source.flags);

**边界**：deblend_delta_bic_min / deblend_component_snr_min 这两个名字带 deblend 的参数**不参与**(A)，只喂给 (B)（detection.rs:661-665 -> psf.rs:144-148）。

### L1-6 max_sources 截断的位置与排序

**结论**：截断发生在“合并（含可选去混叠）之后、逐源测光之前”，按**提案 score 降序**截断；score 是同一位置各方法 score 的最大值（merge 时 score = max），不是通量、不是 SNR、不是质量。被截掉的提案不会进入测光，也不会出现在 rejected_count 里，只体现为 candidate_count > sources.len()。

**证据**：

    backend/src/detection.rs:637-638         if let Some(limit) = options.max_sources { selected.truncate(limit); }
    backend/src/filtering.rs:370             c.score = c.score.max(p.score);
    backend/src/filtering.rs:397             accepted.sort_by(|a, b| b.score.total_cmp(&a.score));
    backend/src/detection.rs:95              "rejected_count":self.sources.len()-self.quality_count,"truncated":self.candidate_count>self.sources.len()

**边界**：max_sources 默认 None（不截断，detection.rs:50）；validate 拒绝 Some(0)。

---

## L2 质量标记的真实语义

### L2-7 quality_passed 的合成条件

**结论**：quality_passed 由**黑名单**合成：只要 flags 中出现 QUALITY_REJECT_FLAGS 里 18 个字符串之一即为 false，否则 true。最终赋值点有三处：(1) measurement::source_from_peak 内 `let passed = quality_passes(&flags)`；(2) psf::guard_unresolved_gaussian_pairs_typed 直接写 false；(3) psf::resolve_supplemental_blends_typed 用 quality_passes 重算。18 个拒绝旗标（含伪影类）：EDGE、MASKED、LINE_ARTIFACT、BACKGROUND_UNCERTAIN、SATURATED、NON_POSITIVE_FLUX、LOW_FLUX_SNR、NO_SHAPE、NARROW、BROAD、ELONGATED、DIFFUSE、SPIKE、SMALL_FOOTPRINT、INSUFFICIENT_PSF_SUPPORT、UNRESOLVED_BLEND、NEGATIVE_OVERFLOW、CODE_PATTERN。注意 PARTIAL_MASKED **不在**名单里。

**证据**：

    backend/src/measurement.rs:5-6            pub const QUALITY_REJECT_FLAGS: [&str; 18] = [ "EDGE",
    backend/src/measurement.rs:25-28          pub fn quality_passes(flags: &[String]) -> bool { !flags.iter().any(|flag| QUALITY_REJECT_FLAGS.contains(&flag.as_str())) }
    backend/src/measurement.rs:467            let passed = quality_passes(&flags);
    backend/src/psf.rs:236                    s.quality_passed = false;

**边界**：没有“质量分数”，没有软门限；quality 是二值的旗标合取取反。

### L2-8 rejected 与 quality_passed=false 是否同一集合

**结论**：在检测结果字典里它们**是**同一集合：rejected_count = sources.len() - quality_count，quality_count 就是 quality_passed 为真的计数。但有两处不同：(1) 该数只统计“已测光源”，被 max_sources 截掉的提案既不算 rejected 也不算 quality（只记 truncated=true）；(2) 下游模块另有自己的“拒绝”口径——pipeline 做星表匹配时只用 quality_passed 子集，astrometry 额外用 peak<peak_ceiling 再筛一层，所以“被拒”在不同环节不是同一批源。

**证据**：

    backend/src/detection.rs:95              "rejected_count":self.sources.len()-self.quality_count
    backend/src/detection.rs:701             let quality_count = sources.iter().filter(|s| s.quality_passed).count();
    backend/src/pipeline.rs:90                .filter(|s| s.quality_passed)
    backend/src/astrometry_blind.rs:502-512  if !truthy(&r[cq]) { rep.rejected_quality += 1; ... } if !(peak < cfg.peak_ceiling) { rep.rejected_peak += 1; ... }

**边界**：源表 CSV 同时保留了原始可重叠 flags 与互斥首要类别（source_classification），后者是报告层派生，不回流影响 quality。

### L2-9 三类饱和判据：位置、阈值来源、是否并集

**结论**：分两个层次，不要混用：

检测层（每源一个 SATURATED 旗标，只看“孔径几何内任一像素”）： 阈值来自 measurement.saturation_level；用户不给时由 integer_limits 推出 max-32（i16 即 32735）。它**不是**三类判据的并集，只有一类。负值方向另有 NEGATIVE_OVERFLOW，阈值来自 negative_overflow_limit，用户不给时对有符号整数推 -0.9*max（i16 即 -29490）。零值不进旗标，进的是掩膜（mask_zero_pixels）。

测光层（empirical_photometry 的 Photometry）：saturated() 是三项**并集**——sat_negative（signed16 原值 <0，按回绕/异常码处理并置 NaN；或窗口内触发 wrapped_at）、sat_peak_ceiling（peak >= peak_ceiling，默认 3990）、sat_zero_clip（coverage>0.5 的孔径像素值为 0）。

第三处“peak ceiling”在盲解算输入侧：astrometry_blind 用 peak < peak_ceiling（默认 20000）直接丢弃源的整行。

**证据**：

    backend/src/measurement.rs:400-406        if options.saturation_level.is_some_and(|s| patch.iter().zip(&geometry).any(|(v, g)| *g && v.is_finite() && *v >= scalar(s))) { flags.push("SATURATED") }
    backend/src/measurement.rs:408-414        options.negative_overflow_limit.is_some_and(|limit| a 内任一像素 *v <= limit) -> "NEGATIVE_OVERFLOW"
    backend/src/detection.rs:420              .map(|(_, max, _)| -0.9 * max)
    backend/src/empirical_photometry.rs:691-693 fn saturated(&self) -> bool { self.sat_negative || self.sat_peak_ceiling || self.sat_zero_clip }
    backend/src/empirical_photometry.rs:752-754 if coverage > 0.5 && value == 0.0 { sat_zero_clip = true; }
    backend/src/empirical_photometry.rs:785    sat_peak_ceiling: peak.is_finite() && peak >= cfg.peak_ceiling,
    backend/src/astrometry_blind.rs:510       if !(peak < cfg.peak_ceiling) { rep.rejected_peak += 1; continue; }

**边界**：两处 peak_ceiling 默认值相差 5 倍（3990 vs 20000），同一颗源可以在一个模块饱和、在另一个模块正常。

### L2-10 flags 全集与每个旗标的触发条件

**结论**：Detection.flags 只由两个文件写入（measurement.rs 与 psf.rs），可能出现的字符串共 19 个：

| 旗标 | 触发条件（源码位置） | 是否拒绝 |
|---|---|---|
| EDGE | 测光 patch 触到图像边界（measurement.rs:370-372） | 是 |
| MASKED | 几何孔径内存在掩膜像素，且不满足“中心有效+全部是精确 0+aperture>=max(ceil(0.8*几何像素),9)”（373-399） | 是 |
| PARTIAL_MASKED | 同上且 allow_partial_zero_mask && center_valid && zero_only && aperture_pixels>=minimum（387-393） | **否** |
| SATURATED | 几何孔径内任一像素 >= saturation_level（400-407） | 是 |
| NEGATIVE_OVERFLOW | 几何孔径内任一像素 <= negative_overflow_limit（408-415） | 是 |
| CODE_PATTERN | 峰值取整后属于全幅重复高位码 **且** 孔径内 range_count>0（416-418；repeated 由 detection.rs:413 以 count>=128 统计） | 是 |
| LINE_ARTIFACT | 几何孔径内任一像素落在线状伪迹掩膜上（419-426） | 是 |
| BACKGROUND_UNCERTAIN | 环带可用像素 <16（427-429） | 是 |
| NON_POSITIVE_FLUX | 孔径内残差和 <= 0（430-432） | 是 |
| LOW_FLUX_SNR | flux_snr 非有限或 < min_flux_snr（433-435） | 是 |
| NARROW / BROAD / ELONGATED | fwhm < min / > max / 椭圆率 > max（436-445） | 是 |
| NO_SHAPE | positive_sum<=0 导致 fwhm/椭圆率全部为 None（446-448） | 是 |
| DIFFUSE | sharpness 缺省或 < min_sharpness（449-450） | 是 |
| SPIKE | sharpness > max_sharpness（451-453） | 是 |
| SMALL_FOOTPRINT | 孔径内 >=noise 的像素数 < min_footprint_pixels（454-456） | 是 |
| INSUFFICIENT_PSF_SUPPORT | 峰值 3x3 邻域内 >=max(2*noise,0.1*峰值-背景) 的像素数 < min_psf_support_pixels（333-349, 457-459） | 是 |
| UNRESOLVED_BLEND | 该源不是 gaussian 方法，且距最近 gaussian 提案 < dog_blend_radius_px(=2.5*FWHM)（460-466）；psf.rs:233-235 也会补写 | 是 |

**证据**：

    backend/src/measurement.rs:425            flags.push("LINE_ARTIFACT".into());
    backend/src/measurement.rs:416-417         if peak_repeated && range_count > 0 { flags.push("CODE_PATTERN".into()); }
    backend/src/psf.rs:233-235                 if !s.flags.iter().any(|f| f == "UNRESOLVED_BLEND") { s.flags.push("UNRESOLVED_BLEND".into()); }

**边界**：题面里的 LINEAR_ARTIFACT / REPEATED_CODE / RANGE_ANOMALY **不是** flags 字符串：前两者在代码里是 LINE_ARTIFACT 与 CODE_PATTERN；RANGE_ANOMALY 只是 source_classification 的互斥首要类别，plate_solver 把它当旗标用（见 discrepancies D1）。

### L2-11 线状伪影判据与标注步骤

**结论**：判据在**像素域**，单位是连通像元集，不是“段（segment）”对象。步骤：残差/噪声 >= max(threshold_sigma,4.0) 得到 support；8 连通分量；分量像元数 >= round(2.5*FWHM^2) 且 >=24 才继续；对分量做二阶矩得到 major/minor；**长度判据** 4*sqrt(major) >= max(16, 5*FWHM)，**细长判据** sqrt(major/minor) >= 4.0；两条同时满足才把整个分量置为线掩膜；最后按 round(FWHM/2) 次 4 邻域膨胀。源级 LINE_ARTIFACT 由测光阶段检查“几何孔径内是否存在掩膜像素”得到。linear_artifact_fraction 的定义是“组内成员中含 LINE_ARTIFACT 旗标的比例”，不是像元比例、也不是长度加权。

**证据**：

    backend/src/filtering.rs:413             let threshold = detection_sigma.max(4.0);
    backend/src/filtering.rs:433             let min_pixels = (2.5 * psf_fwhm.powi(2)).round_ties_even().max(24.0) as usize;
    backend/src/filtering.rs:481-482         if 4.0 * major.max(0.0).sqrt() >= 16.0f64.max(5.0 * psf_fwhm) && (major / minor.max(f64::EPSILON)).sqrt() >= 4.0 {
    backend/src/filtering.rs:489             let dilation = (psf_fwhm / 2.0).round_ties_even().max(1.0) as usize;
    backend/src/detection.rs:464-465         let line = if options.reject_linear_artifacts { Some(T::lines(&residual, &noise_map, &valid, height, width, options.threshold_sigma, measurement.psf_fwhm)?) }
    backend/src/measurement.rs:423           any(|(p, g)| *g && line[p[1] as usize * width + p[0] as usize])
    backend/src/source_classification.rs:379 linear_artifact_fraction: flag_fraction(&members, &["LINE_ARTIFACT"]),

**边界**：reject_linear_artifacts 默认 true（detection.rs:55）；关掉后 line=None，LINE_ARTIFACT 永不出现，且去混叠的 line_mask 也为空。数组规模：tmp/rust-sources-prod/source_catalog.csv 共 66469 数据行，其中 97 行含 LINE_ARTIFACT（实测 grep -c = 97）；DESIGN.md:404 记录的“96/96”是更早快照。

---

## L3 测光与配对

### L3-12 孔径中心与环带半径的推导

**结论**：检测层（measurement::source_from_peak）的孔径中心是**提案的整数峰**（xp,yp），不是质心；质心只用于报告位置 x,y。patch 半宽 outer = aperture_radius+4；孔径判据是 distance <= aperture_radius 且像素未被掩膜；环带是 aperture_radius+2 <= distance <= outer，且未掩膜。测光层（empirical_photometry::measure）的孔径中心是源表里给定的浮点位置（检测表 x,y，即质心），环带半径**不由 aperture_radius 推**，而是独立配置项 6.0/10.0。

**证据**：

    backend/src/measurement.rs:154            let [xp, yp] = peak;
    backend/src/measurement.rs:170-171        let radius = options.aperture_radius; let outer = radius.checked_add(4)...
    backend/src/measurement.rs:189            let geometric = distance <= radius as f64;
    backend/src/measurement.rs:196            if distance >= (radius + 2) as f64 && distance <= outer as f64 && !masked {
    backend/src/measurement.rs:489-490        peak_x: Some(xp as f64), peak_y: Some(yp as f64),
    backend/src/empirical_photometry.rs:72-73 annulus_inner: 6.0, annulus_outer: 10.0,
    backend/src/empirical_photometry.rs:1982 let Some(photometry) = measure(&image, detection.x, detection.y, &cfg)

**边界**：x,y 的算法是“PSF 加权 + 半个噪声底扣除”，权重 = max(r-0.5*noise,0)*exp(-0.5*(d/sigma)^2)，且只在 d <= min(aperture_radius, max(2.5*sigma,2.0)) 内求和；权重和 <= EPSILON 时回退用正残差权重。

### L3-13 flux_error 的完整表达式与分支

**结论**：检测层 flux_error = sqrt(source_variance + background_variance)，其中

- source_variance = (gain 存在 ? max(flux,0)/gain : 0) + aperture_pixels * read_noise_adu^2
- background_variance = aperture_pixels * noise^2 + aperture_pixels^2 / background_pixels * noise^2
- background_pixels = 环带有效像元数（无环带时取 max(1)）
- 总和先 max(EPSILON) 再开方；flux_snr = flux / flux_error。

分支：环带有效像元 >=16 **且** refine_local_background=true -> background/noise 来自环带 sigma-clipped 统计（3σ、4 次、上限 100000）；否则直接取峰值像素处的 background_map/noise_map。gain 缺席时源项为 0（纯背景受限），read_noise_adu 默认 0，因此默认情况下 flux_error 只剩背景项。

**证据**：

    backend/src/measurement.rs:237-238        .unwrap_or(0.0) + aperture_pixels as f64 * options.read_noise_adu.powi(2);
    backend/src/measurement.rs:239-240        let background_variance = aperture_pixels as f64 * noise.powi(2) + (aperture_pixels as f64).powi(2) / background_pixels as f64 * noise.powi(2);
    backend/src/measurement.rs:241-244        let flux_error = (source_variance + background_variance).max(f64::EPSILON).sqrt(); let flux_snr = flux / flux_error;
    backend/src/measurement.rs:201-202        let annulus_usable = annulus_values.len() >= 16; let (background, local_noise) = if annulus_usable && options.refine_local_background {

**边界**：测光层另有一套（含 pi/2 中位数方差因子、noise_inflation、加权像元覆盖）：variance = ni^2*Σw^2*σ^2 + (π/2)*A^2*σ^2/B + (gain ? Σw^2(v-b)/g : 0)（empirical_photometry.rs:769-776）。两套误差定义不共享公式，论文里不能互称。

### L3-14 配对流程：种子、迭代、二次项、双向最近邻、稳健剔除

**结论**：以测光配对 empirical_photometry 为准（这是唯一带二次项的配对链）：

1. 种子 = seed_affine(cfg)：直接用配置的 scale/rotation/parity/crpix 建立初始 3 项仿射，不做星对搜索。
2. 迭代调度是**固定表**（不是收敛驱动）：bootstrap_iters(3) 轮 terms=3 + 半径 bootstrap_radius；再 bootstrap_iters 轮 terms=6（同一宽半径，先拟合畸变再收紧）；两轮 terms=6、半径 1.7*match_radius；最后 max_iter(12) 轮 terms=6、半径 match_radius。
3. 每轮：用当前仿射预测全部参考星位置 -> 双向最近邻匹配 -> 若配对数 < min_pairs 立即终止 -> robust_affine 重新拟合。
4. 收敛判据：仅当 `shift < 1e-6 && radius <= match_radius && terms == 6` 才提前 break（shift 是两次仿射系数的差值范数，terms 改变时视为无穷）。
5. 收尾再做一次 match_radius 下的匹配 + terms=6 稳健拟合 + 再匹配。
6. 稳健剔除在 robust_affine 内：最多 6 轮，窗口 = clamp(median(残差)+3*MAD, 0.08, 3.0)，保留数 < terms+4 或不再变化即停。
7. 双向最近邻 = 正向找最近检测、反向找最近参考，只有互为最近且落在 radius 内才算一对（unique assignment）。

**证据**：

    backend/src/empirical_photometry.rs:1872  schedule.push((3, cfg.bootstrap_radius.max(cfg.match_radius)));
    backend/src/empirical_photometry.rs:1884  schedule.push((6, cfg.match_radius));
    backend/src/empirical_photometry.rs:1915  if shift < 1e-6 && radius <= cfg.match_radius + 1e-9 && terms == 6 { break; }
    backend/src/empirical_photometry.rs:1900  if current.len() < cfg.min_pairs { break; }
    backend/src/empirical_photometry.rs:1231  let window = (median(&distances).unwrap_or(0.0) + 3.0 * sigma).clamp(0.08, 3.0);
    backend/src/empirical_photometry.rs:1160-1162 if back as usize != index { continue; }
    backend/src/empirical_photometry.rs:1188-1191 let basis: Vec<f64> = if terms == 3 { vec![1.0, xi, eta] } else { vec![1.0, xi, eta, xi * xi, xi * eta, eta * eta] };

**边界**：WCS 求解走的是**另一条**配对链 plate_solver.rs（星对假设 -> 单向最近邻 + 贪心唯一化 -> 仅 3 项仿射 + MAD 截断），没有二次项；它的二次项只在 empirical_photometry 里作为诊断量出现（2181-2187：affine_rms 与 quadratic_rms 对比，不参与匹配）。

### L3-15 photometric_inlier 与 saturated 在拟合中的作用

**结论**：(1) saturated 是**硬剔除**：进入标定样本前就 continue，既不降权也不标记为 inlier=false；它对测光定标的影响是减少样本，而不是加权。(2) photometric_inlier 是稳健裁剪的结果位，权重是 1/sigma_total^2（sigma_total = sqrt(参考星误差^2 + 仪器误差^2)，下限 PHOTOMETRIC_SIGMA_FLOOR_MAG），即加权最小二乘 + 迭代裁剪；裁剪窗口 = max(clip_sigma(3.0)*MAD, 0.03 mag)，初始种子窗口由 mode_estimate(0.1 mag bin) 给出（order>0 用 1.2 mag，否则 0.5 mag），保留数 < order+8 或集合不变即停。(3) 位置仿射拟合**不加权**。

**证据**：

    backend/src/empirical_photometry.rs:2030-2032 if row.photometry.saturated() { continue; }
    backend/src/empirical_photometry.rs:2054            weight: 1.0 / (sigma_total * sigma_total),
    backend/src/empirical_photometry.rs:1297-1301     normal[i][j] += sample.weight * basis[i] * basis[j]; rhs[i] += sample.weight * basis[i] * (...)
    backend/src/empirical_photometry.rs:1420-1425     let window = (clip_sigma * sigma).max(0.03); ... filter(|(_, r)| (*r - center).abs() <= window)
    backend/src/empirical_photometry.rs:2098-2099     if fit.inliers[position] { rows[sample_row[*sample_index]].inlier = true; }

**边界**：photometric_inlier 只写进配对 CSV（empirical_photometry.rs:2375），不回流到源表、不改变 quality_passed。

---

## L4 天体测量与判决

### L4-16 盲搜阶梯：bins、roll 步长、scale 步长、保留种子、细化窗口

**结论**：默认 bins=1024，search_bins=bins/8=128（都必须是 2 的幂），keep=[32,10,3]，sample_sigma=100（抽样统计噪声底），roll_zeta=1.0。

- 粗搜：地图边长 n=search_bins=128 bins；roll 步长 = zeta*(frame/n)/r_max，再向下取整成 2π 的整数分之一；scale 步长 = zeta*(frame/n)*s_ref/r_max（s_ref 为 scale 区间中点）；roll 覆盖整圆 2π、scale 在 [scale_min, scale_max] 上按步长成阶梯；两种 parity 各一套，共 2*n_roll*n_scale 次试验；每批保留 keep[0]=32 个种子。
- 细化：max_levels = (bins/search_bins).trailing_zeros() = 3 层；每层 n 翻倍（上限 bins），roll/scale 步长各缩 4 倍，窗口为每个种子 ±4 个新步长（等价于 ±1 个上一级步长，共 9x9 个候选）；每层去重容差 roll 0.75*新步长、scale 0.4%、位移 3*frame/n；保留 keep[level]（10、3、1）。
- 亚像素：hit 的 t_px 用 bins 内的 sub_dx/sub_dy 内插（hit_from_eval）。

**证据**：

    backend/src/astrometry_blind.rs:320-324   let bins = cmd.usize_or("bins", 1024)?; ... (bins / 8).max(32)
    backend/src/astrometry_blind.rs:333       None => vec![32, 10, 3],
    backend/src/astrometry_blind.rs:1192      let raw = zeta * (frame / n as f64) / r_max;
    backend/src/astrometry_blind.rs:1199      zeta * (frame / n as f64) * s_ref / r_max
    backend/src/astrometry_blind.rs:1412      let max_levels = (cfg.bins / cfg.search_bins).trailing_zeros() as usize;
    backend/src/astrometry_blind.rs:1476-1485 n = (n * 2).min(cfg.bins); let sub = 4.0; ... for i in -4..=4 { roll = sd.roll + i*roll_step_new } for j in -4..=4 { scale = sd.scale + j*scale_step_new }
    backend/src/astrometry_blind.rs:1524-1530 let kept = keep.get(level).copied().unwrap_or(1); dedup(&mut hits, roll_step_new*0.75, 0.004, 3.0*cfg.frame/n as f64)

**边界**：--guided 只搜先验窗口（roll ±roll_window=3°、scale ±1%），输出状态为 VALIDATED_GUIDED，并在 verdict 里明写“不是盲解”。log-polar 引擎复用同一判决路径，只是 (roll,scale) 初值来源不同。

### L4-17 VALIDATED 的判定表达式

**结论**：原文是一个合取：

    validated = passed_task_gate && audit_ok && baseline_ok && !cfg.skip_selfcal && controls_complete

各项定义：

- passed_task_gate = !skip_real && best_real.is_some() && real_ps >= gate；real_ps 是最佳真实命中的 peak_sigma =(max-median)/MAD_sigma。
- gate = control_upper 有限时 max(8, 3*max(control_upper, null_p99))，否则 8；control_upper = 所有 kind=null 控制族 peak_sigma 的最大值（translate/mirror 属 invariance，被显式排除在零假设上界之外）。
- audit_ok = on2 >= 50 且 on2/off2 >= 3，其中 on2/off2 是 on-target 与 (37,53) px 平移对照在 2 px 内的匹配数（audit_positions）。
- baseline_ok 来自自标定 baseline 行是否 success（不成功时整轮直接 METHOD_FAILED 退出，连 verdict 都不写）。
- controls_complete = 未跳过控制 && control_trials>0 && null 扫描条数 == 3*control_trials && control_upper 有限。
- 状态字符串：validated 且 guided -> VALIDATED_GUIDED，validated -> VALIDATED，否则 UNVALIDATED 且 exit_code=1。

**证据**：

    backend/src/astrometry_blind.rs:2560-2564 let gate = if control_upper.is_finite() { (3.0 * control_upper.max(control_p99)).max(8.0) } else { 8.0 };
    backend/src/astrometry_blind.rs:2586      let audit_ok = on2 >= 50.0 && audit_ratio >= 3.0;
    backend/src/astrometry_blind.rs:2587      let passed_task_gate = !cfg.skip_real && best_real.is_some() && real_ps >= gate;
    backend/src/astrometry_blind.rs:2593      passed_task_gate && audit_ok && baseline_ok && !cfg.skip_selfcal && controls_complete;
    backend/src/astrometry_blind.rs:2594-2599 let status = if validated { if cfg.guided { "VALIDATED_GUIDED" } else { "VALIDATED" } } else { "UNVALIDATED" };

**边界**：verdict 里另写了 significance_scope：这是“稳健相关图对比 + 有限匹配搜索控制”，不是高斯 sigma、也不是标定过的全局虚警率（astrometry_blind.rs:2653）。

### L4-18 自标定三种扰动的施加方式

**结论**：先把星表按真值 (roll, scale, parity, tx, ty) 投影到探测器坐标，再逐点施加三种扰动：

1. 抖动：对保留下来的点加高斯噪声 (sigma = jitter_px，默认 1.5 px)，逐轴独立；抖动后若出画幅则丢弃。
2. 完备度：投影后先按 completeness（默认 0.35）做伯努利保留，未保留即跳过——所以“完备度”是保留概率，不是加噪。
3. 假阳性：合成完成后追加 round(fp_frac * 已保留点数) 个画幅内均匀随机点（fp_frac 默认 0.30）。

真值抽样：盲模式下 roll 在 [0,2π) 均匀、scale 固定 selfcal_scale（默认 8.44，matrix/extend 模式可覆盖）、tx/ty 在中心 ±400 px 内均匀；guided 模式下 roll/scale 改在先验窗口内抽。判定成功需同时满足：peak_sigma>=8、|Δroll| <= 1.5*粗步长、|Δscale/scale| <= 2%、|Δt| <= 3*frame/bins、parity 正确。

**证据**：

    backend/src/astrometry_blind.rs:2070-2074 if rng.uniform() > completeness { continue; } let x = x + if jitter > 0.0 { rng.normal() * jitter } else { 0.0 };
    backend/src/astrometry_blind.rs:2091-2094 let n_fp = (fp_frac * out.len() as f64).round() as usize; for _ in 0..n_fp { out.push((rng.uniform() * frame, rng.uniform() * frame)); }
    backend/src/astrometry_blind.rs:2957   spec.scale.unwrap_or(cfg.selfcal_scale)
    backend/src/astrometry_blind.rs:2962   rng.uniform() * 2.0 * std::f64::consts::PI
    backend/src/astrometry_blind.rs:2969-2970 let tx = cfg.frame / 2.0 + (rng.uniform() - 0.5) * 800.0;
    backend/src/astrometry_blind.rs:3034-3038 let ok = h.peak_sigma >= 8.0 && droll.abs() <= roll_tol && dscale.abs() <= 0.02 && dt <= 3.0 * cfg.frame / cfg.bins as f64 && h.trial.parity == truth.parity;

**边界**：--selfcal-mode matrix/extend 是**敏感性矩阵**（改变 jitter/completeness/fp/尺度半步/roll 半步），baseline 行永远在 i=0 且决定 baseline_ok；matrix 只增加行，不改变 baseline 定义。

---

## L5 覆盖守卫

### L5-19 catalog_bounds 与 evaluate 的判定式

**结论**：

- 中心：RA 先减去首个值转成 [-180,180) 的偏移（跨 0/360 安全），再取“截尾中点”：排序后 trim = min(floor(0.01*n), last/2)，center = 0.5*(v[trim] + v[last-trim])；RA 再回绕到 [0,360)。Dec 用同一函数，但在副本上排序，避免打乱 RA/Dec 行配对。
- 半径：对每个源算到中心的 haversine 角距，取 99% 分位（nearest-rank：index = ceil(0.99*n).clamp(1,last+1)-1），并要求有限且 >0。
- 判定：sep <= 0.35° -> Full；否则 sep <= 0.5*radius -> Partial；否则 NotCovered。只有 NotCovered 会 blocks_run。

**证据**：

    backend/src/catalog_coverage.rs:7         const FULL_SEPARATION_DEG: f64 = 0.35;
    backend/src/catalog_coverage.rs:9         const RADIUS_PERCENTILE: f64 = 0.99;
    backend/src/catalog_coverage.rs:76-77   let trim = ((0.01 * values.len() as f64).floor() as usize).min(last / 2); Some(0.5 * (values[trim] + values[last - trim]))
    backend/src/catalog_coverage.rs:86      let index = ((fraction * values.len() as f64).ceil() as usize).clamp(1, last + 1) - 1;
    backend/src/catalog_coverage.rs:178     let radius_deg = percentile(&mut separations, RADIUS_PERCENTILE)
    backend/src/catalog_coverage.rs:197-203 let status = if sep_deg <= FULL_SEPARATION_DEG { Full } else if sep_deg <= 0.5 * bounds.radius_deg { Partial } else { NotCovered };

**边界**：中心是 1–99% 分位的中点（不是中位数、不是均值），半径是 99 分位（不是最大角距）；两者共同的效果是“忽略少量离群行而不放大圆”。单源星表时 trim=0，中心即该点，半径=0 -> 触发“覆盖半径无效”错误。
