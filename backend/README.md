# Rust 运算层迁移

目标是把全部运算实现移到 Rust，并从生产运行包中移除 Python。
本目录为原生核心库 `rst19-core` 与隔离算法进程 `rst19-engine`，运行时不嵌入 Python、不通过 FFI 或子进程调用 Python。

迁移尚未完成。完整后端模块、函数、源码指纹和 78 个工具入口列在 `migration.json`。
已开始原生实现 FITS、辅助遥测、曝光单位、稳健背景统计、TAN 坐标和四元数几何。
检测链中的 Gaussian/Starlet 掩膜归一化滤波、滑动极大值、候选 NMS、多通道合并、连通域 PCA 线结构审计和 KD-tree 空间检索也已建立原生实现。float64 与 float32 检测已通过当前固定样例和原生进程的合成图／ROI 对照；完全重合的 PSF 模板现在被明确视作不可辨识双源，Python 与 Rust 都返回空证据，避免奇异线性求解依赖 BLAS 分支。
float64 的 Gaussian / hybrid / ensemble 已串成完整检测链，包含孔径测光、PSF 加权定位、质量规则、双 PSF/BIC 证据、近邻 Gaussian 门控和可选局部残差去混叠。共享背景模型核验二维形状，不同形状会回退逐帧背景。
`rst19-engine` 已支持 `detection.array`、`detection.fits`、`measurement.source` 和 `psf.pair` JSONL 操作。全部 78 个原 CLI 工具的完整兼容层与剩余桌面工具路由仍待迁移；单帧 `rst19` 主入口已接入 Rust worker。
星表链已原生实现源模型、CSV 读取、自行历元传播、TAN／完整仿射投影及一对一匹配，支持 `catalog.load`、`catalog.propagate` 和 `matching.catalog`；匹配入口可直接读取 CSV。分配核心支持 `assignment.solve` 和 `matching.assign`，包括矩形最小代价求解、稀疏连通分量全局分配及历史贪心分配。固定样例与原生进程对照覆盖来源、消光语义、质量字段、空值、冲突及返回排序。
先验 WCS 的局部仿射细化已迁入 Rust：wcs.affine.fit 执行几何条件检查、MAD 离群剔除和匹配点留一验证；wcs.validate.matches 汇总逐帧结果，wcs.validate.sequence 直接读取 FITS 并调用原生检测／匹配。6 个单元场景和双帧 FITS 全链路均与 Python 基线逐字段通过；它使用已有先验 WCS 与星表匹配，不是全天空盲解算。
辅助姿态审计的计算与命令行入口已迁入 Rust：`attitude.audit.frames` 检查内存中的辅助字段，`attitude.audit.paths` 直接读 FITS；独立 `rst19-aux-audit` CLI 接受单个 FITS 或目录并生成 CSV/JSON，发布包内附带此原生二进制。缺失辅助块、RA 跨零、替代相机轴和 FITS 路径均有 Python 对照；CSV 与 Python 输出逐字节一致，JSON 逐字段一致。该审计只检查辅助字段约定，不提供完整 WCS 或物理指向结论。
源级坐标几何审计已迁入 Rust：`audit.source_geometry`、桌面 `rst19-source-geometry-audit` worker 路由和独立 CLI 共用同一实现。它复刻 NumPy 1.26.4 的 SeedSequence、PCG64 和无放回抽样，以 PCA 轴比、方向及垂直残差描述类别坐标；Python 对照覆盖默认与指定类别、64 位种子、错误输入及 CSV/JSON 数值产物。对照结果的数值容差为 `2e-12`；Windows 包含原生 CLI，并在禁用外部 Python 搜索后完成 worker/CLI 验收。该审计仍只描述 detector 坐标，不证明运动或伪影真值。
点源序列的若干关联层已迁入 Rust：`sequence.track_detections` 串联高 SNR 互为最近邻平移注册、稀疏连通分量一对一轨迹关联、仿射残差 RANSAC 与稳健细化，并按观测时间分类；`sequence.fast_point_movers` 补充已有质量点源中的高速常速度轨迹；`sequence.fast_candidate_associate` 将已筛选的匹配滤波候选按实际时间轴做三点常速度种子、单帧缺测追踪、拟合门和重复种子去重；`sequence.candidate_consensus` 实现重复候选的几何锚定、互为最近邻关联、时间中值 SNR 通道和质量轨迹去重，并可按帧核验原图有效孔径、局部背景、3×3 支持、二阶矩形状、PSF 模板相关和线伪迹重叠。新操作 `sequence.reference.median` 与 `sequence.reference.coadd` 构建注册中值和 MAD 裁剪稳健叠加参考图；`sequence.temporal_reference_candidates` 从参考图提出时间候选，再逐帧搜索并计算多尺度强制 PSF 响应；`sequence.line_profile_metrics` 复现线状残差沿主轴的连续支撑度与剖面填充度；`sequence.trail_shape_evidence` 给出单帧线形候选的拒绝原因、局部模型支持和跨帧形状准入；`sequence.motion_feature_track_gates` 复现跨帧形状/长度/角度一致性、运动轴多数门、固定伪影兼容性和边缘截断门；`sequence.motion_feature_continuity_gates` 检查最长连续观测段和受限内部缺帧回接；`sequence.motion_feature_exposure_evidence` 将 PSF 去卷积后的曝光线长度与帧间位移按实际曝光/时间间隔比较；`sequence.motion_feature_confirmation_evidence` 提供序列长度对应的最低确认帧数和稳定拒绝原因代码；`sequence.motion_proposal_thresholds` 根据已验证的 1.5 秒曝光元数据调整候选线长与短拖影连通面积下限。 `sequence.motion_time_axis` 校验相对帧时间，非法或缺失输入回退到帧序号，并计算中位参考间隔。 `sequence.observation_time_axis` 将 FITS `DATE-OBS` 解析为 UTC 相对秒；无效或非递增时间戳组整体回退到帧序号。 `sequence.motion_feature_pair_geometry` 计算候选线段的方向、预计步长、平移端点与边缘权重软代价，用于配对排序而非硬门控。 `sequence.fit_collinear_motion_geometry` 对线状候选中心做 PCA，返回运动跨度、垂直 RMS、主轴角和按参考帧间隔折算的速度。 `sequence.best_collinear_motion_feature_group` 从未关联长线中选择跨帧、单调推进且共线的一组候选。`sequence.stitch_unassociated_single_frame_points` 复核后回接孤立单帧候选；`sequence.link_motion_evidence_tracks` 去重关联点源轨迹与曝光线轨迹；`sequence.rescue_collinear_motion_feature_groups` 从未关联长线恢复互不重叠的共线候选组；`sequence.coherent_raw_motion_point_indices` 连接相邻帧原始线候选。单帧原始像素检测由 `trails.local_pixel_proposals` 提供。上述层均有 Python 基线对照。完整多帧 `analyze_sequence` 编排、多尺度/raw-candidate 帧候选装配及候选原图强制测光仍待迁移；仿射采样仍使用 SplitMix64，未复现 NumPy PCG64 样本序列。所有这些数量仍是图像平面算法证据，不代表物理身份。
`sequence.stitch_unassociated_single_frame_points` 将通过线模型复核的单帧候选按常速度预测、时间间隔、运动轴方向、线长比例和伪影重叠门回接到既有轨迹，并按帧号恢复输出顺序。该操作只补充图像平面候选轨迹证据，不确认物理身份；多尺度候选装配与高层序列编排仍待迁移。
`sequence.link_motion_evidence_tracks` 将移动点源轨迹和曝光线轨迹按共同观测帧、运动方向、位移比例和垂直距离做一对一证据关联；链接只用于去重和审计，不删去任一轨迹或确认物理身份。
`sequence.rescue_collinear_motion_feature_groups` 在常规中心关联失败时，从长单帧线候选中迭代恢复互不重叠的共线运动组；结果仍是图像平面候选，需要后续轨迹门复核。
`sequence.coherent_raw_motion_point_indices` 将相邻帧原始线候选按实测时间、曝光时长、位移方向和线形支持连成短链；少数弱形态帧可以由强点桥接，但整条链至少需要两个强形态点。它只输出候选帧/点索引，后续关联与运动确认仍需独立运行；完整多尺度候选装配和多帧编排尚未迁入 Rust。
`sequence.motion_feature_frame_components` 将序列拖影检测的单帧阈值支撑图分成连通候选，支持一像素缺口分组，并返回 PCA 长宽/方向、沿线连续度、边缘和线伪影重叠审计。它只提供几何候选；原图点/双源模型复核与跨帧运动确认仍由后续阶段完成。
`trails.local_pixel_proposals` 已将单帧原始像素线候选检测迁入 Rust：保留掩膜归一化 Gaussian 平滑、局部峰值预算、方向/长度扫描、局部复核、点源/双星负例控制和线段几何去重。输出仍只是单帧线形候选，不代表跨帧运动；完整序列候选装配仍待迁移。

测光层已原生实现仪器／表观星等算术，带状态的视差／模型距离／星表绝对星等，以及加权零点和颜色项拟合、离群点剔除、确定性留出验证、目录质量门控、样本角色和标定应用。支持 `photometry.evaluate`、`photometry.calibration.fit`、`photometry.calibration.apply`、`photometry.calibration.error`。绝对星等保留消光语义、距离区间和误差完整性门控。
逐源光度表与最暗源选择支持 `photometry.table`、`photometry.faintest`；保留目录模型星等、图像推导绝对星等、残差拒绝和部分标定集合的区别。`pipeline.analyze` 直接读取 FITS，串联检测、匹配、标定、逐源光度表和最暗源；原生库支持不重复检测的再标定。
新增 `rst19-native` 单帧 CLI，兼容原 `rst19` 参数入口、默认 hybrid、标准输出与 `--json-out`。已通过合成输入和有返回上限的真实首帧对照；原 CLI 帮助文本布局与全部解析边界仍需扩展核验。
`rst19-pair-mechanism-context` 已原生计算候选源的质心/整数峰距离、无序近邻对、类别组合与双质量通过背景，并生成与 Python 字节一致的三份 CSV。`rst19-pair-mechanism-sensitivity` 在预先声明的几何窗口内重复该审计，保存窗口参数、目标 pair 状态及对照总体计数；窗口变化不解释成物理概率。两项工具均支持 Rust JSONL engine 与桌面 worker，覆盖 BOM、可选旗标列、目标去重、几何边界和自定义窗口。

`rst19-pair-pixel-topology` 对局部原始 ADU patch 计算 raw 与特殊像素排除后的 8 连通分量、峰值成员关系及滑动局部极大值；结果 JSON、CSV 和 SVG 与 Python 基线逐项/逐字节核对。该审计描述像素拓扑，不单独确认物理源身份；Rust JSONL engine 与桌面 worker 均已接入。

`rst19-aperture-peak-ratio` 按 `feature_class` 与提议器来源组统计有限 `flux / peak` 的中位数和高分位数，并在请求候选所在类别中给出经验百分位。三份 CSV 与 Python 输出逐字节一致；这些数值受孔径、背景和 PSF 影响，只是描述性审计，不是真星概率或误报率。

`rst19-feature-correlation` 对请求字段逐对计算 Spearman 相关、按首次出现顺序计算类别中位数与类内相关；保留并列秩、缺失值成对剔除、强相关阈值和 UTF-8 CSV/JSON 输出。它只描述当前源表中的字段依赖，不表示因果或分类准确率。

`rst19-feature-transition-flow` 直接读取已有的类别转移 CSV，按层级和首帧类别汇总任意响应、同类响应与最大跨类流向，并检查各响应行的锚点元数据一致性。它不重新读取 FITS 或运行检测器，CSV/JSON 产物对照 Python 基线。

`rst19-feature-gate-route` 按候选拆分数值质量门越界与结构旗标，去重已由数值门解释的同名旗标，再按特征类别汇总拒绝路径。它复用默认质量门阈值，只描述检测器规则，不构成反事实或物理类别结论。

`rst19-feature-hard-negative` 在每个算法类别内按通量 SNR、滤波 SNR 或峰值排序质量层落选候选，保留类别计数和形态字段，并可选择纳入 `other_rejected`。这些记录是供后续复核的样本入口，不是物理恒星或误报真值。

`rst19-gaia-merge` 的去重、字段补齐、关键字段冲突检查和输入 sidecar 完整性继承已由 Rust 引擎、桌面 worker 与独立原生 CLI 实现。只有输入 CSV 哈希与行数仍匹配、各层天空覆盖一致且没有关键字段冲突时，合并结果才标记为 `complete=true`；诊断性合并仍会输出，但会保留 `complete=false` 和原因列表。

`rst19-feature-evidence-axes` 只读取已有 feature evidence matrix，按类别汇总候选持久性、质量邻域响应与局部 PSF 相关度，并检查五组预设阈值的模式敏感性。Q|P 是 detector-level 描述量，不是逐源精度、FDR 或恒星概率。

`rst19-direct-neighbor-audit` 保留源表里每条质心半径内的直接近邻边，不做传递合并；按类别组合汇总距离、质量状态和双 PSF 工程证据，并可单独查询目标 pair。独立候选仍需真实 PSF、注入、星表/WCS 或人工真值确认。

`rst19-feature-cross-axis-audit` 严格按 `detection_id` 对齐源目录、raw 局部摘要、raw 峰审计和逐帧响应，输出逐源证据并置、类别中位数、模式计数及输入 SHA-256。模式只描述共享检测器中的证据组合，不是分类器、精度、FDR、伪影率或恒星概率。

`rst19-source-group-audit` 在 Rust 中按质心半径建立传递近邻组，输出组成员、代表候选、质量/结构旗标、类别统计和目标回查。`independent_group_candidate` 仍是工程候选标签，不代表物理星组已确认。

`rst19-source-group-sensitivity` 对同一源表按用户给定半径重复分组，保留半径顺序并汇总全图组数和目标去向；可并行执行多个半径。半径扫描描述 detector-level 分流变化，不代表光学分辨率或双星概率。

`rst19-source-separation-audit` 分别按质心与峰坐标计算每个候选的最近其它候选，汇总半径端点比例、质量通过端点比例、类别分位数和指定目标邻居。近邻统计只描述候选几何，不构成无序源对计数或物理身份判断。

`rst19-source-peak-consistency` 直接读取 FITS，在辅助掩膜下按 nearest-edge 局部最大值规则检查目录峰坐标，并统计三种半径中的局部峰数量、差值、类别分位数和指定目标附近最大值。CSV 与 JSON 对照 Python 基线；这些结果描述图像像素与候选目录的一致性，不确认物理源身份。

`rst19-source-proposal-peak-audit` 按 `detection_id` 连接源目录与 raw 峰一致性 CSV，再按 `feature_class × method_group` 汇总质量通过率、局部峰比例、非局部峰差值和 SNR 中位数，并可保留指定目标的输入顺序。它只读取已有 CSV；提议器来源彼此相关，不是独立投票，也不表示物理身份。

`rst19-source-proposal-temporal-audit` 将来源目录与非紧凑候选的逐源时序、紧凑候选的 proposal subgroup 聚合连接起来，分别生成逐源行、类别×来源汇总和有序目标结果。紧凑来源只有子组聚合，指定目标的空逐源时序值表示证据不可用，不是零响应。

人工阈值反馈以版本化 UTF-8 JSONL 保存；`feedback.append` 与 `feedback.load` 由 Rust 引擎和桌面 worker 实现，保留旧记录默认值和损坏行定位。反馈仅用于人工复核和调参记录，不会在线训练或自动改变检测器。

桌面主入口 `rst19` 的单帧检测、星表匹配与光度流程，以及 `rst19-aux-audit`、`rst19-gaia-merge`、`rst19-feature-evidence-axes`、`rst19-direct-neighbor-audit`、`rst19-feature-cross-axis-audit`、`rst19-source-group-audit`、`rst19-source-group-sensitivity`、`rst19-source-separation-audit`、`rst19-source-peak-consistency`、`rst19-source-proposal-peak-audit`、`rst19-source-proposal-temporal-audit`、`rst19-source-geometry-audit`、`rst19-feature-flag-profile`、`rst19-feature-flag-interaction`、`rst19-feature-gate-margin`、`rst19-feature-gate-route`、`rst19-feature-hard-negative`、`rst19-feature-effect-size`、`rst19-feature-class-context`、`rst19-pair-mechanism-context`、`rst19-pair-mechanism-sensitivity`、`rst19-pair-pixel-topology`、`rst19-aperture-peak-ratio`、`rst19-feature-correlation` 和 `rst19-feature-transition-flow`，已由原生 Rust worker 执行；其余尚未迁移的研究工具仍使用 Python worker。尚未支持的工具不会被静默替换为近似实现。

```powershell
cargo test --locked --manifest-path backend/Cargo.toml
cargo build --release --locked --manifest-path backend/Cargo.toml --bins
cargo build --release --locked --manifest-path desktop/Cargo.toml --bins
$worker = "desktop/target/release/rst19-native-worker.exe"
$engine = "backend/target/release/rst19-engine.exe"
$verificationDir = "tmp/native-feature-cross-axis-audit-v1"
python backend/verify_feature_cross_axis_audit.py $worker $engine --out $verificationDir
python backend/verify_source_group_audit.py $worker $engine --out tmp/native-source-group-audit-v1
python backend/verify_source_group_sensitivity.py $worker $engine --out tmp/native-source-group-sensitivity-v1
python backend/verify_source_separation_audit.py $worker $engine --out tmp/native-source-separation-audit-v1
python backend/verify_source_peak_consistency.py $worker $engine --out tmp/native-source-peak-consistency-v1
python backend/verify_source_proposal_peak_audit.py $worker $engine --out tmp/native-source-proposal-peak-audit-v1
python backend/verify_source_proposal_temporal_audit.py $worker $engine --out tmp/native-source-proposal-temporal-audit-v1
python backend/verify_source_geometry_audit.py $worker $engine --cli backend/target/release/rst19-source-geometry-audit.exe --out tmp/native-source-geometry-audit-v1
python backend/verify_feedback.py $worker $engine --out tmp/native-feedback-v1
$verificationDir = "tmp/native-sequence-chain-verification-$((Get-Date).ToString('yyyyMMdd-HHmmss'))"
python backend/verify_sequence_chain.py backend/target/release/rst19-engine.exe --out $verificationDir
python backend/verify_fits.py backend/target/release/rst19-engine.exe --out tmp/native-fits-verification --data-dir "doc/00-项目资料/原始数据"
```

`generate_reference.py` 仅用于捕获 Python 基线，生成固定的 Rust 对照测试样例，不在生产运算链中运行。
对照覆盖曝光优先级、MAD/标准差回退、采样、旧版网格插值坐标、RA 跨零、旋转、奇偶性与四元数归一化。
`verify_fits.py` 覆盖六种 FITS 存储类型、显式 BSCALE/BZERO、64 位整数原始值、遥测识别、异常文件与可选真实数据。

只有全部算法、工具输出和界面路由完成对照验证后，才移除原有 Python 运行时并将迁移标记为完成。
