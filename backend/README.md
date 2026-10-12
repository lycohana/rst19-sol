# Rust 运算层迁移

目标是把全部运算实现移到 Rust，并从生产运行包中移除 Python。
本目录为原生核心库 `rst19-core` 与隔离算法进程 `rst19-engine`，运行时不嵌入 Python、不通过 FFI 或子进程调用 Python。

迁移尚未完成。完整后端模块、函数、源码指纹和 78 个工具入口列在 `migration.json`。
已开始原生实现 FITS、辅助遥测、曝光单位、稳健背景统计、TAN 坐标和四元数几何。
检测链中的 Gaussian/Starlet 掩膜归一化滤波、滑动极大值、候选 NMS、多通道合并、连通域 PCA 线结构审计和 KD-tree 空间检索也已建立原生实现。float64 与 float32 检测已通过当前固定样例和原生进程的合成图／ROI 对照；完全重合的 PSF 模板现在被明确视作不可辨识双源，Python 与 Rust 都返回空证据，避免奇异线性求解依赖 BLAS 分支。
float64 的 Gaussian / hybrid / ensemble 已串成完整检测链，包含孔径测光、PSF 加权定位、质量规则、双 PSF/BIC 证据、近邻 Gaussian 门控和可选局部残差去混叠。共享背景模型核验二维形状，不同形状会回退逐帧背景。
`rst19-engine` 已支持 `detection.array`、`detection.fits`、`measurement.source` 和 `psf.pair` JSONL 操作。78 个原 CLI 契约与桌面路由仍待迁移。
星表链已原生实现源模型、CSV 读取、自行历元传播、TAN／完整仿射投影及一对一匹配，支持 `catalog.load`、`catalog.propagate` 和 `matching.catalog`；匹配入口可直接读取 CSV。分配核心支持 `assignment.solve` 和 `matching.assign`，包括矩形最小代价求解、稀疏连通分量全局分配及历史贪心分配。固定样例与原生进程对照覆盖来源、消光语义、质量字段、空值、冲突及返回排序。
先验 WCS 的局部仿射细化已迁入 Rust：wcs.affine.fit 执行几何条件检查、MAD 离群剔除和匹配点留一验证；wcs.validate.matches 汇总逐帧结果，wcs.validate.sequence 直接读取 FITS 并调用原生检测／匹配。6 个单元场景和双帧 FITS 全链路均与 Python 基线逐字段通过；它使用已有先验 WCS 与星表匹配，不是全天空盲解算。
辅助姿态审计的计算与输出已迁入 Rust：`attitude.audit.frames` 检查内存中的辅助字段，`attitude.audit.paths` 直接读 FITS；独立 `rst19-aux-audit` CLI 接受单个 FITS 或目录并生成 CSV/JSON。缺失辅助块、RA 跨零、替代相机轴和 FITS 路径均有 Python 对照；CSV 与 Python 输出逐字节一致，JSON 逐字段一致。该审计只检查辅助字段约定，不提供完整 WCS 或物理指向结论。
点源序列的若干关联层已迁入 Rust：`sequence.track_detections` 串联高 SNR 互为最近邻平移注册、稀疏连通分量一对一轨迹关联、仿射残差 RANSAC 与稳健细化，并按观测时间分类；`sequence.fast_point_movers` 补充已有质量点源中的高速常速度轨迹；`sequence.candidate_consensus` 实现重复候选的几何锚定、互为最近邻关联、时间中值 SNR 通道和质量轨迹去重。三者都有 Python 基线对照。候选共识的原图 PSF 支持、坏像素/线伪迹门控和完整多帧分析器尚未迁入；候选峰高速源、拖影特征也待迁移。仿射采样仍使用 SplitMix64，未复现 NumPy PCG64 样本序列。所有这些数量仍是图像平面算法证据，不代表物理身份。
测光层已原生实现仪器／表观星等算术，带状态的视差／模型距离／星表绝对星等，以及加权零点和颜色项拟合、离群点剔除、确定性留出验证、目录质量门控、样本角色和标定应用。支持 `photometry.evaluate`、`photometry.calibration.fit`、`photometry.calibration.apply`、`photometry.calibration.error`。绝对星等保留消光语义、距离区间和误差完整性门控。
逐源光度表与最暗源选择支持 `photometry.table`、`photometry.faintest`；保留目录模型星等、图像推导绝对星等、残差拒绝和部分标定集合的区别。`pipeline.analyze` 直接读取 FITS，串联检测、匹配、标定、逐源光度表和最暗源；原生库支持不重复检测的再标定。
新增 `rst19-native` 单帧 CLI，兼容原 `rst19` 参数入口、默认 hybrid、标准输出与 `--json-out`。已通过合成输入和有返回上限的真实首帧对照；尚未替换桌面路由或全部 Python 工具。原 CLI 帮助文本布局与全部解析边界仍需扩展核验。
现有桌面工作台继续使用迁移前的 Python 算法进程；尚未支持的工具不会被静默替换为近似实现。

```powershell
cargo test --locked --manifest-path backend/Cargo.toml
cargo build --release --locked --manifest-path backend/Cargo.toml --bins
python backend/verify_fits.py backend/target/release/rst19-engine.exe --out tmp/native-fits-verification --data-dir "doc/00-项目资料/原始数据"
```

`generate_reference.py` 仅用于捕获 Python 基线，生成固定的 Rust 对照测试样例，不在生产运算链中运行。
对照覆盖曝光优先级、MAD/标准差回退、采样、旧版网格插值坐标、RA 跨零、旋转、奇偶性与四元数归一化。
`verify_fits.py` 覆盖六种 FITS 存储类型、显式 BSCALE/BZERO、64 位整数原始值、遥测识别、异常文件与可选真实数据。

只有全部算法、工具输出和界面路由完成对照验证后，才移除原有 Python 运行时并将迁移标记为完成。
