# rst19 论文研究来源与证据账本

> 这是论文短稿的来源账本，不是正文。它把外部方法依据、项目实测事实和当前未知量分开，便于答辩时追溯，避免把文献中的条件性结论写成普适真值。

## 研究主张

| 主张 | 直接来源 | 在 rst19 中的对应证据 | 不能推出 |
| --- | --- | --- | --- |
| 局部峰是候选响应，不等于物理源数 | [DAOPHOT](https://articles.adsabs.harvard.org/pdf/1987PASP...99..191S)、[StarFinder](https://arxiv.org/abs/astro-ph/0004101)、[StarNet](https://www.jmlr.org/papers/v24/21-0169.html) | 单源/双源 PSF、候选/质量分层 | 两个峰一定是两颗星 |
| 可解析性由 PSF、分离、亮度比和方向共同决定 | [Penoyre, 2026](https://doi.org/10.1093/rasti/rzaf062) | 5.4 px 等强与不等强双源注入 | 5.4 px 是普适分辨率阈值 |
| 完备率应由已知真值注入回收，而不是由峰数反推 | [ComEst](https://arxiv.org/abs/1605.02650) | 真实背景分层注入、成对一对一回收 | 注入召回率等于真实图像 precision |
| 高 SNR 不能排除结构性异常 | [L.A.Cosmic](https://arxiv.org/abs/astro-ph/0108003)、[Rhoads](https://arxiv.org/abs/astro-ph/0002041) | 线状、范围异常、尖峰类别的高 SNR 反例 | 亮点必然是宇宙线或必然是真星 |
| PSF 估计需考虑模板选择与空间变化 | [PSFEx](https://www.astromatic.net/wp-content/uploads/psfex_article.pdf)、[Wohlberg & Wozniak](https://arxiv.org/abs/2101.01268) | 20 个模板源、全局/局部 PSF 抽检 | 当前全局经验核已代表所有 detector 位置 |
| 近邻多峰的源数需要比较多个 PSF 模型，而不能由局部峰数决定 | [Kass & Raftery 1995](https://doi.org/10.1080/01621459.1995.10476572)、[DAOPHOT](https://articles.adsabs.harvard.edu/pdf/1987PASP...99..191S)、[StarFinder](https://arxiv.org/abs/astro-ph/0004101)、[Hedges et al. 2021](https://arxiv.org/abs/2106.08411) | 固定位置 `K=1/2/3` 的非负 Gaussian PSF、BIC、分量 SNR 和掩膜敏感性审计 | `ΔBIC` 越大就等于物理上已确认的多颗恒星；自由位置/空间变 PSF 仍未完成 |
| 固定中心误差与自由度过拟合必须用有界位置敏感性分开审计 | [DAOPHOT](https://articles.adsabs.harvard.org/pdf/1987PASP...99..191S)、[StarFinder](https://arxiv.org/abs/astro-ph/0004101)、[Penoyre 2026](https://doi.org/10.1093/rasti/rzaf062) | `r=0.25/0.5/0.75/1.0/1.25 px` 自由位置扫描、位置触边和优化收敛字段 | 某个搜索半径下通过就代表半径无关的源身份；有界 Gaussian 仍不是完整联合似然 |
| 主办方路线也采用亮星估计 PSF、增强暗目标和时序跟踪 | [StarImage_SimuIden](https://github.com/Aoyeww/StarImage_SimuIden) 及其页面列出的 [IEEE Sensors Journal 论文](https://doi.org/10.1109/JSEN.2024.3350089)、[IEEE TIM 论文](https://doi.org/10.1109/TIM.2025.3573015) | rst19 的宽筛—细筛、经验 PSF、序列分支 | 主办方实现输出就是本数据真值 |
| 全幅重复高位码与局部异常负值应先作为数据有效性模式审计 | [HST/ACS detector considerations](https://hst-docs.stsci.edu/acsdhb/chapter-4-acs-data-processing-considerations/4-3-dark-current-hot-pixels-and-cosmic-rays)、[2MASS reliability](https://www.ipac.caltech.edu/2mass/releases/allsky/doc/seca5_1.html) | F01 `3990--3993` 全幅频次、局部负值共现、`CODE_PATTERN` 质量门 | 这些统计模式已经等价于坏像素、ADC 饱和或普通白噪声真值 |
| 类别跨帧持久性必须同时看候选层和质量层 | [ComEst](https://arxiv.org/abs/1605.02650)、[Zackay et al.](https://arxiv.org/abs/1601.02655) | 最新 15 帧 `12/15` 邻域审计：范围异常 `33/159` 候选持久、质量 `0`；紧凑质量 `25,028/26,376` 候选持久、`21,543` 质量邻域 | `12/15` 响应就是逐星真阳性或真实恒星完备率 |

## 项目中已经测得的事实

1. 当前 GUI 默认首帧一次运行返回 `84,594` 个宽筛候选、`29,263` 个质量规则通过源；此前的 `29,271` 是只加入极端负码审计的中间口径，`29,153` 是 `CODE_PATTERN` 初版（仅“孔径内含重复码”即拒）的历史口径，这三个数字都不是星表真值。
2. 截图局部两个响应相距约 `5.39 px`；副响应有 `NEGATIVE_OVERFLOW`、固定 `-1` 邻域和重复正值结构，原始双 PSF `ΔBIC=-2.59`、次分量 SNR 约 `1.55`，经验 PSF 对照也未达到双源确认线。注册版 F07 只有在屏蔽范围码或精确 `-1` 后形式上越过工程线，原始分量 SNR 为 `4.511` 且该帧带 `MASKED|SATURATED`，因此只记为观测模型敏感性。
3. 同一真实背景的 Gaussian 双源控制显示，固定 `r=0.125` 时总峰值 `256/512/1024 ADU` 的成对质量解析率为 `0/0.25/1.0`；这说明弱副源先在质量层失去证据，再在宽筛层消失。
4. 经验 PSF 模板池为 `20` 个源，中位 FWHM `2.4966 px`、归一化核和 `3.6465`。在 detector FWHM `2 px` 的形状敏感性控制中，`512 ADU`、`r=0.25/0.125` 的成对质量解析率为 `0.25/0`；匹配尺度 `2.5 px` 探针仍为 `0.25/0`，同尺度 Gaussian 为 `1.0/0.5`，但两次布局随 detector FWHM 改变，不能当作严格逐点排名。
5. 真实首帧的 `-1` 固定坐标审计得到 `74` 个跨 15 帧稳定坐标、总精确 `-1` 观测 `2,185`；但格式说明没有把 `-1` 定义为坏像素或无效码，因此正文只称“固定无效码候选”。
6. 为控制 PSF 核和混杂，固定总离散积分超额 `1867 ADU`、同一布局、`2 trials × 8 对` 后，Gaussian/经验 PSF 在 `r=0.25` 的成对质量解析率为 `0.812/0.625`，在 `r=0.125` 为 `0.188/0.125`；候选层分别为 `1.000/0.938` 与 `0.688/0.812`。这支持“PSF 形状影响细筛，候选层和质量层可反向排序”，但不是最终 PSF 排名。
7. 在同一首帧把空间 PSF 诊断扩大到 `2×2` 网格、每类 `16` 个样本后，紧凑质量类全局/局部相关中位数均约 `0.860`，线状类 `0.412→0.405`，拥挤类 `0.121→0.111`；局部模板有效数分别为 `10/16`、`16/16`、`11/16`。这支持“紧凑源没有可见的系统性空间改善，线状结构不是点源，拥挤问题主要是混合/去混叠”，但仍不是空间变 PSF 的最终实现。
8. 在同一真实背景注入 2 源/3 源拥挤组并扩展到每条件 `2 trials × 8` 组后，`3 px` 相邻组的候选合并率为 `0.75--1.0`、组级解析率为 `0`；`5.4/6 px` 候选组级解析率为 `1.0`，但质量级完整解析对每源信号和组源数仍敏感。两档总积分信号为 `512/1024 ADU`；2 源在 1024 ADU 的质量解析为 `15/16`、`14/16`，3 源为 `1/16`、`0/16`，因此仍只是机制证据。
9. 经验 PSF 的 4 组/条件控制中，`3 px` 组级解析仍为 `0`，`5.4 px` 候选组级解析为 `1.0`，而 `6 px` 候选组级解析在 2 源/3 源条件下出现 `0.75/0.25`；因此间距结果受 PSF 形状与每源信号影响，不能写成普适分辨率。
10. 对截图三个位置做固定位置 `K=1/2/3` 审计时，raw 原图仅 F07、F12 的 `K=2` BIC 比 `K=1` 更优，但新增分量 SNR 分别为 `4.38/4.00`，没有一帧同时达到 `ΔBIC≥10`、`SNR≥5`；`K=3` 没有越线。屏蔽范围码后只有 F07 形式上越线（`ΔBIC=19.98`、新增分量 SNR `5.14`），但该帧带 `MASKED|SATURATED`，因此属于模型敏感性，不是原始真值确认。

11. 有界自由位置 raw `K=2` 半径扫描在 `r=0.25/0.5/0.75/1.0/1.25 px` 下分别确认 `0/15、0/15、0/15、1/15、1/15`；F07 只有在允许约 `1 px` 移动后才达到当前局部确认线，窄半径时新增中心触碰边界。固定位置 `0/15` 与自由位置 `1/15` 的差异记录为模型自由度敏感性，不写成双星比例。
12. 第二组截图的首帧两个提案约为 `(2439,4021)` 与 `(2435,4022)`；峰坐标间距 `4.12 px`、测量质心间距约 `2.74 px`。在加入 `CODE_PATTERN` 前，两者的峰值/通量 SNR/FWHM/支持像素约为 `3992/204.61/2.59/9` 与 `569/115.51/2.44/8`，因此旧质量规则会把它们同时保留。
13. 第二组截图固定位置 raw `K=2` 的 15 帧 `ΔBIC` 全部为负（约 `-4.40` 到 `-2.98`），自由位置半径 `0.25/0.5/0.75/1.0/1.25 px` 的确认数全部为 `0/15`；同时屏蔽固定 `-1` 与正负范围异常的对照仍为 `0/15`（`ΔBIC` 约 `-4.40` 到 `-0.95`）。新增 `CODE_PATTERN` 后，F01 候选数仍为 `84,594`，质量数从门控前 `29,295` 变为 `29,263`；两个提案保留在候选审计层但不再进入质量层。
14. 最新 `CODE_PATTERN`（峰值落在重复码上才拒绝）版本的首帧范围异常候选为 `44`（`NEGATIVE_OVERFLOW 36`、`CODE_PATTERN 12`、`SATURATED 1` 的并集；初版“孔径内含重复码即拒”口径为 `159`）。此前的 15 帧特征持久性审计仍按初版口径生成（范围异常候选持久 `33`、质量持久 `0`，紧凑质量 `26,376/25,028/21,543`，弱源 `21,246/16,128/404`，尖峰 `23,824/17,521/87`，拥挤 `902/509/6`，线状 `96/22/16`），需按修正后的 `CODE_PATTERN` 重跑后才能作为当前 15 帧主证据。

对应机器产物：

- `tmp/pair-flux-ratio-multilevel-gui-default/pair_flux_ratio_audit.csv`
- `tmp/pair-flux-ratio-512-empirical-gui-default/pair_flux_ratio_audit.csv`
- `tmp/pair-flux-ratio-512-empirical-matched25-gui-default/pair_flux_ratio_audit.csv`
- `tmp/pair-flux-ratio-512-gaussian25-gui-default/pair_flux_ratio_audit.csv`
- `tmp/pair-integrated-1867-gaussian-gui-default-t2-n8/pair_flux_ratio_audit.csv`
- `tmp/pair-integrated-1867-empirical-gui-default-t2-n8/pair_flux_ratio_audit.csv`
- `tmp/source-feature-morphology-gui-default/source_feature_morphology.csv`
- `tmp/source-feature-morphology-gui-default/source_feature_psf_similarity.csv`
- `tmp/source-feature-psf-grid2-per16-gui-default/source_feature_psf_spatial.csv`
- `tmp/crowded-blend-audit-integrated-gui-default/crowded_blend_audit.csv`
- `tmp/crowded-blend-audit-integrated1024-gui-default/crowded_blend_audit.csv`
- `tmp/crowded-blend-audit-integrated-t2-g8-gui-default/crowded_blend_audit.csv`
- `tmp/crowded-blend-audit-integrated-empirical-t1-g4-gui-default/crowded_blend_audit.csv`
- `tmp/sequence-feature-persistence-gui-default/sequence_feature_persistence.csv`
- `tmp/local-multipsf-audit-screenshot/local_multipsf_audit.csv`
- `tmp/local-multipsf-audit-screenshot/local_multipsf_audit.json`
- `tmp/local-multipsf-audit-screenshot/local_multipsf_bic.png`
- `tmp/local-free-multipsf-audit-screenshot/local_free_multipsf_audit.csv`
- `tmp/local-free-multipsf-audit-screenshot/local_free_multipsf_audit.json`
- `tmp/local-free-multipsf-audit-screenshot/local_free_multipsf_bic.png`
- `tmp/local-free-multipsf-radius-sweep-screenshot/local_free_multipsf_radius_sweep.csv`
- `tmp/local-free-multipsf-radius-sweep-screenshot/local_free_multipsf_radius_sweep.json`
- `tmp/local-free-multipsf-radius-sweep-screenshot/local_free_multipsf_radius_sweep.png`
- `tmp/local-multipsf-audit-screenshot-pair2/local_multipsf_audit.csv`
- `tmp/local-multipsf-audit-screenshot-pair2/local_multipsf_audit.json`
- `tmp/local-multipsf-audit-screenshot-pair2/local_multipsf_bic.png`
- `tmp/local-multipsf-audit-screenshot-pair2-range-sentinel/local_multipsf_audit.csv`
- `tmp/local-multipsf-audit-screenshot-pair2-range-sentinel/local_multipsf_audit.json`
- `tmp/local-free-multipsf-radius-sweep-screenshot-pair2/local_free_multipsf_radius_sweep.csv`
- `tmp/local-free-multipsf-radius-sweep-screenshot-pair2/local_free_multipsf_radius_sweep.json`
- `tmp/local-free-multipsf-radius-sweep-screenshot-pair2/local_free_multipsf_radius_sweep.png`
- `tmp/source-quality-audit-code-pattern/source_quality_summary.csv`
- `tmp/source-quality-audit-code-pattern/source_catalog.csv`
- `tmp/sequence-feature-persistence-code-pattern/sequence_feature_persistence.csv`
- `tmp/sequence-feature-persistence-code-pattern/sequence_feature_temporal_profile.csv`
- `tmp/sequence-feature-persistence-code-pattern/sequence_feature_class_transition.csv`
- `tmp/feature-cross-audit-code-pattern/feature_cross_audit.csv`

## 正文采用的判定层级

```text
宽筛候选响应
  -> 原始像素质量审计
  -> PSF/形态与异常值域诊断
  -> 跨帧注册坐标响应
  -> WCS/星表唯一匹配与分层注入校准
```

其中 `quality_passed` 只表示当前图像质量规则通过，`12/15` 只表示注册坐标邻域响应持久，`pair_resolution_recall` 只表示已知注入双源是否被两个不同候选一对一恢复。只有在 WCS、星表、设备值域和阴性样本得到验证后，才可以把这些中间量提升为物理身份或正式完备率。

## 未决事项

- FITS signedness、`BLANK`、坏像素和满阱定义；
- 增益、读出噪声和完整 Poisson 误差；
- detector 分区留出 PSF 与更多 trial；当前空间 PSF 类别诊断仍是 `2×2/16` 的候选抽样，多源拥挤控制虽已扩展为每条件 `16` 组，但仍需更多方向/位置，等积分双源对照仍只有 `2×8` 对/条件；
- 真实运动目标逐帧真值和完整 WCS；
- 星表版本、滤波系统、颜色项和唯一匹配残差。
- 当前固定位置 `K` 模型仍假设全局 Gaussian PSF、已知中心和共享背景；需用分区留出 PSF、自由位置多源拟合和官方星表/WCS 复核后，才能决定该局部是否包含多个物理源。
- 当前自由位置模型只在初始候选周围做有限半径优化；F07 的 `1/15` 结果随半径变化，仍需多起点、分区 PSF、完整噪声似然和注册天空坐标复核。
- `CODE_PATTERN` 是当前 FITS 的可解释数据质量门，不是主办方已确认的坏像素/饱和定义；需取得 signedness、`BLANK`、ADC 满阱、增益、读出噪声和坏像素资料后再校准。
