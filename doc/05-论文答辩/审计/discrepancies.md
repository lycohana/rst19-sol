# 实现与文档/直觉不一致清单（含证据）

判定口径：以 backend/src 的当前代码为准；文档指 DESIGN.md / README.md / dist 内 PROJECT-README.md / doc/ 下的论文稿。每条给出代码位置、被挑战的说法、以及影响。

---

## D1 plate_solver 的坏旗标清单里有一个永远不会出现的旗标

**代码**：backend/src/plate_solver.rs:27-35

    const BAD_SOLVER_FLAGS: [&str; 7] = ["EDGE", "MASKED", "SATURATED", "LINE_ARTIFACT", "CODE_PATTERN", "RANGE_ANOMALY", "UNRESOLVED_BLEND"];

**事实**：全仓库只有 measurement.rs 与 psf.rs 会向 Detection.flags 写入字符串，两者的产出并集不含 RANGE_ANOMALY（RANGE_ANOMALY 是 source_classification 的互斥类别名，见 source_classification.rs:11-21、25-27）。grep RANGE_ANOMALY 在 backend/src 只有 plate_solver.rs:33 一处。

**影响**：星位求解器的质量筛选事实上是 6 条，而不是 7 条；论文若照抄 7 类坏旗标会引用一个不存在的门。

---

## D2 名字里带 deblend 的两个参数并不控制去混叠

**代码**：backend/src/detection.rs:661-665（PairGuardOptions 取 deblend_delta_bic_min / deblend_component_snr_min）

    let pair_options = PairGuardOptions { psf_fwhm: ..., delta_bic_min: options.deblend_delta_bic_min, component_snr_min: options.deblend_component_snr_min };

**事实**：局部去混叠 deblend::local_deblend_proposals 只消费 min_primary_snr / min_residual_sigma / search_radius_factor（detection.rs:611-616、deblend.rs:20-25）。BIC 与分量 SNR 只用于 PSF 单/双源证据（psf.rs）。

**影响**：调 deblend_delta_bic_min 不会改变任何去混叠提案，只会改变 UNRESOLVED_BLEND 的判罚；反过来关掉 enable_local_deblend 也不会关掉这套 BIC 门。

---

## D3 同一个峰值上限概念有两套默认值，相差 5 倍

**代码**：backend/src/empirical_photometry.rs:70（peak_ceiling: 3990.0）与 backend/src/astrometry_blind.rs:393（peak_ceiling: 20_000.0）

    peak_ceiling: 3990.0,          // photometry
    peak_ceiling: cmd.f64_or("peak-ceiling", 20_000.0)?,   // astrometry

**影响**：峰值 5000 ADU 的源在测光审计里判为 saturated 并被剔出定标样本，但在盲解算里完全正常；两处没有共享常量，也没有交叉校验。论文若写饱和剔除阈值为 X 必须指明是哪一个模块。
---

## D4 flags 名字与论文/题面里的名字不一致（三处）

| 论文/题面写法 | 代码真实字符串 | 证据 |
|---|---|---|
| LINEAR_ARTIFACT | LINE_ARTIFACT | measurement.rs:8、425 |
| REPEATED_CODE | CODE_PATTERN（且必须同时有值域异常像素） | measurement.rs:23、416-417；DESIGN.md:360 |
| RANGE_ANOMALY（当作逐源旗标） | 不是旗标，是互斥首要类别 range_anomaly（由 SATURATED/NEGATIVE_OVERFLOW/CODE_PATTERN 归组） | source_classification.rs:25-27 |

**影响**：按名字去 CSV 里检索会得到 0 行；把 range_anomaly 当旗标统计会把三类旗标压成一类。

---

## D5 遥测区掩膜在两处是两种几何形状

**代码 A**（检测与测光）：backend/src/fits.rs:118

    mask[..width.min(104)].fill(true);

即只屏蔽第 0 行的前 104 个像素（一维条带）；empirical_photometry 同样处理（empirical_photometry.rs:593-599，注释写明 y=0, x=0..103 是永久遥测窗）。

**代码 B**（盲解算）：backend/src/astrometry_blind.rs:517 与 astrometry_blind.rs:2086

    if cfg.mask_telemetry && x < 104.0 && y < 104.0 { continue; }

即一个 104x104 的角块。

**影响**：同一个遥测窗在两个模块面积相差 104 倍；若论文用 B 的几何描述检测器的掩膜，会与实际屏蔽的像素集合不符。

---

## D6 proposal_mode 的默认值有三层，且互不相同

**代码**：detection.rs:56（库默认 gaussian）、cli.rs:216（CLI 注入 hybrid）、DESIGN.md:306（称单帧和 15 帧 GUI 入口统一使用 proposal_mode=hybrid）

    proposal_mode: "gaussian".into(),                       // DetectionOptions::default
    fields.entry("proposal_mode").or_insert(json!("hybrid")); // CLI

**事实**：Rust 层没有任何代码把 hybrid 设为默认；GUI 是否 hybrid 完全取决于前端下发的 options（main.rs:1248-1250 直接反序列化 request.options）。

**影响**：（1）直接用库或 GUI 未传参时跑的是纯 Gaussian，DoG/starlet 提案数为 0；（2）论文说默认 hybrid 只对 CLI 成立，且没有代码级保证。

---

## D7 与 D6 同源：FWHM 与 min_distance 的默认值也和文档不符

**代码**：measurement.rs:113（psf_fwhm: 3.0）、detection.rs:49（min_distance: 3）；对照 DESIGN.md:306（PSF FWHM=2 px、min_distance=4 px）。另外 main.rs 内部对审计类命令的 psf_fwhm 回退值是 2.0（main.rs:82、196、236），而 psf 证据接口的回退值是 3.0（main.rs:1596）——同一个文件里两种回退。

**影响**：FWHM 直接决定高斯 sigma、DoG 尺度、线状伪迹长度门（5*FWHM）、UNRESOLVED_BLEND 半径（2.5*FWHM），2 与 3 的差别会传导到整条链，不能当成无关紧要的参数差。

---

## D8 极负值的掩膜门比旗标门更宽松

**代码**：detection.rs:164（掩膜：value <= min+32）与 detection.rs:417-421（旗标：value <= -0.9*max）

    limits.is_some_and(|(min, max, _)| value >= max - 32.0 || value <= min + 32.0)
    .map(|(_, max, _)| -0.9 * max)

**事实**：i16 下掩膜切在 -32736，NEGATIVE_OVERFLOW 旗标切在 -29490。落在 [-32736, -29490] 的像素被掩膜屏蔽但不会触发旗标；反之任何被旗标判定的源一定有一个未被掩膜的极负像素——两条规则不同源、不同阈值。

**影响**：不能把掩膜与溢出旗标当作同一判据的两种表述；注入实验里改掩膜口径会改变旗标统计而不改变掩膜统计（DESIGN.md:312 提到这一点，但未点明阈值不一致）。
---

## D9 文档里的线状伪迹计数已过期

**文档**：DESIGN.md:404、504 与 dist/*/PROJECT-README.md:289 记 linear_artifact 96/96 同类同格。

**实测**：tmp/rust-sources-prod/source_catalog.csv 共 66469 数据行，含 LINE_ARTIFACT 的行数为 97（grep -c 结果）。

**影响**：引用 96 会与当前产物不一致；这不是代码变更，而是数据/快照漂移。

---

## D10 keep_zero_pixels 在浮点帧上是空操作

**代码**：cli.rs:228-230（keep_zero_pixels=true -> mask_zero_pixels=false）与 detection.rs:154-155（mask_zero_pixels=None 时仅在整数类型且 |背景|>1 才屏蔽零值）。

**影响**：对 float32/float64 帧，零值本来就不被屏蔽，用户勾选保留零值看不到任何变化；对 int16 帧则确实翻转行为。文档把该开关描述成通用开关，会让复现者误判。

---

## D11 detection_id 的顺序与行顺序不一致

**代码**：detection.rs:686-700

    id_order.sort_by(|left, right| a.peak_y.unwrap_or(a.y).total_cmp(&b.peak_y.unwrap_or(b.y)) ...
    sources.sort_by(|a, b| a.y.total_cmp(&b.y).then(a.x.total_cmp(&b.x)));

**事实**：id 按整数峰排序赋值，行按质心排序输出。

**影响**：任何第 i 行就是 detection_id=i 的假设（含部分 CSV 消费者、部分审计脚本的 join）都会错位；这也解释了为什么同一源在两次运行中 id 稳定但行号不一定稳定。

---

## D12 四个都在叫 SNR 的列，定义互不相同

**代码**：measurement.rs:244（flux_snr = flux/flux_error）、measurement.rs:332（snr = (peak-background)/noise）、detection.rs:486-487 + filtering.rs:156-189（filter_snr = 标准化响应，除以 scale*(noise/median_noise).max(0.25)）、detection.rs:494（proposal_snr = 提案合并时的方法得分）。

**事实**：filter_snr 的除数带噪声比修正并被截断在 0.25，即高噪声区的显著性最多被压低 4 倍后不再继续惩罚；proposal_snr 只是峰值响应，与孔径通量无关。

**影响**：论文里的 4σ 检测实际发生在 filter_snr 上，而质量层的 SNR>=5 发生在 flux_snr 上，两者不可互换、也不可互相换算。

---

## D13 默认管线里没有局部去混叠

**代码**：detection.rs:68（enable_local_deblend: false）、detection.rs:598（且 fast_sequence 时强制跳过）。

**影响**：默认链的去混叠只有 PSF 单/双源证据（flag/降级），没有任何子源分裂提案。把 local deblend 写进默认流程的描述属于过度陈述；DESIGN.md:306 的局部去混叠默认关闭与代码一致，但 README 的流程图式描述容易让人误以为一直在跑。

---

## D14 结果字典里的 background/noise/threshold 是全局量，不是逐像素门

**代码**：detection.rs:412（全局 stats）、detection.rs:727（threshold = background + threshold_sigma*noise）。

**事实**：提案与测光实际使用的是 background_map/noise_map 与标准化响应。

**影响**：把 threshold 当每个像素的门会得出错误的完备度/虚警推算；它只是全局口径的展示值。

---

## D15 rejected_count 在三个层级有三种含义

**代码**：detection.rs:95（检测字典：len(sources)-quality_count）、astrometry_blind.rs:502-512（按 quality_passed 与 peak_ceiling 分别计数 rejected_quality / rejected_peak）、source_artifacts.rs:144-157（subclass 表的 quality_count/rejected_count 是分组内计数）。

**影响**：跨文档比较拒绝率时必须指明层级；三者分母不同。

---

## D16 一个开关两种效果：refine_local_background

**代码**：measurement.rs:90 同时控制（a）背景图迭代次数 4 或 2（detection.rs:364-368、448-452），（b）测光是否使用环带局部背景（measurement.rs:202）。

**影响**：把它当只影响背景图的调参，会在测光层同时改变 flux_error 的分支，进而改变 flux_snr 与质量层；消融实验若不报告这一耦合，结论不可比。

---

## 附：本次核对未发现不一致、但容易误读的三点

1. CODE_PATTERN 需要两个条件同时成立（峰值属于重复高位码 且 孔径内有 <= min(-1000, background-50*noise) 的像素），与 DESIGN.md:360 的描述一致。
2. PARTIAL_MASKED 不是拒绝旗标，但它并非无条件放行：当孔径有效像素 < max(ceil(0.8*几何像素),9) 时会降级为 MASKED（measurement.rs:380-399），仍然拒绝。
3. catalog_bounds 的中心是 1-99% 截尾中点而非中位数，半径是 99 分位而非最大角距，两者都不等于星表中心/半径的常见定义，是刻意设计（catalog_coverage.rs:68-69 注释）。
