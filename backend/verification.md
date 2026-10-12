# 原生运算层验证记录

日期：2026-10-12。迁移前 Git 基线：`f834d91b0483b5e0b66f5b80ac3795a10a8616a3`。

## 当前工作区快照的验证状态

本节覆盖提交前的最新代码；后文记录的是此前验证，不能视为当前全部测试通过。

- `cargo test --locked --manifest-path backend/Cargo.toml` 当前失败：138 个固定检测案例通过，但 float32 阶段测试的 `SNR[0]` 存在 1 ULP 差异（位模式 3143924160 与 3143924159）。
- 单独运行 `measurement_parity`：32 个 float64 / float32 源测量案例及近邻质量门控通过；扩展样例后，完全重合 PSF 的奇异拟合案例失败，Rust 返回 BIC `-4.9628446302598945`，Python 基线返回空值。该失败保留，尚未修复。
- 额外 float32 进程级合成图对照也存在滤波噪声差异：Rust `0.4615626931190491`，Python `0.46156272292137146`。未放宽容差或删除失败测试。
- float32 背景、噪声、残差、Gaussian 与 Starlet 阶段在当前扩展固定样例中逐位通过；SNR 阶段仍待定位。
- float32 路径已经实现；后文“仍返回未迁移错误”、16 个源测量案例和 57 个检测案例的描述属于较早版本，当前数量分别为 32 和 138。
- 后端全量迁移未完成，桌面仍调用原 Python worker；尚无纯 Rust 发布包。

## 此前已验证记录

- 较早的样例集运行 `cargo test --locked --manifest-path backend/Cargo.toml`：12 个测试通过；最新扩展样例集的失败见上文。
- `cargo clippy --manifest-path backend/Cargo.toml --all-targets -- -D warnings`：通过。
- `cargo build --release --locked --manifest-path backend/Cargo.toml`：通过，生成原生 `rst19-engine`。
- `verify_fits.py`：六种存储类型、六组显式缩放、两个辅助字段识别案例、四个异常文件案例通过。
- 同一验证器读取本地全部 15 个原始 FITS：头部、图像尺寸、数据类型、数据偏移、辅助字段、曝光时间与原始像素抽样均与 Python 结果一致。验证前后文件 SHA-256 不变。
- `BITPIX=64` 使用整型存储，对超过 `2^53` 的原始像素值不经过浮点数转换；显式要求 BSCALE/BZERO 时才转为 float64。

## 数值对照覆盖

固定 Python/SciPy 基线样例保存在 `tests/fixtures/reference.json`，同时记录相关 Python 源码 SHA-256。

- 稳健背景统计：常量、随机背景、异常值、掩膜和确定性抽样。
- 旧版背景网格：单行、单列、缩小、放大、裁剪及右下角补齐；局部背景与 RMS 图。
- TAN：RA 跨零、不同赤纬、旋转、奇偶性和像素参考点；正反变换。
- 四元数：归一化、旋转矩阵、相机本体 `-Y` 光轴；非法输入。
- Gaussian 与 Starlet：nearest 边界、稀疏／密集掩膜和两种快速路径；逐像素对照。
- 极大值与 NMS：近邻边界、平台峰、图像边缘和多个距离；候选顺序对照。
- 通道合并：Gaussian 测量锚点、来源、尺度、最近 Gaussian 距离和稳定排序。
- 线结构：八邻域连通域、PCA 长宽比、长度／面积规则及十字结构膨胀；掩膜逐像素对照。
- KD-tree：最近邻、半径查询、空集及非有限输入，与穷举距离对照。
- 源级测量：16 个固定案例，覆盖弱源、PSF 加权定位、孔径／环形背景、增益／读噪、零值部分掩膜、NaN 邻域、饱和、极端负码、CODE_PATTERN、线结构和边缘。
- 双 PSF：10 个固定单／双源案例，覆盖重合与近重合位置、负分量、BIC、协方差与次分量幅度 SNR；两组 Gaussian 近邻质量门控案例。
- 完整 float64 检测：57 个固定案例，覆盖 Gaussian / hybrid / ensemble、int16 / float64、伪迹、截断、快速滤波、局部残差去混叠、边缘／掩膜回退、共享背景和同像素数不同形状的背景回退。
- float64 输入即便请求 `use_float32=True`，按 Python 基线继续走 float64；整数／float32 输入真正启用 float32 的计算路径仍明确返回未迁移错误。

## 原生进程和真实图像计算

`verify_detection.py` 已通过三个合成图案例和首帧 `(1250,160,512,512)` ROI 的三种模式逐字段对照。ROI 显式设置 `max_sources=300`，保留完整候选计数与截断语义，不能当作全幅结果。

随后执行首帧 4096×4096 的完整 float64 hybrid 检测，不设置源数量上限。第一次全幅对照发现 1 个质量源与 32 个 PSF 证据差异，根因是等距 Gaussian 邻峰选择不同；没有修改科学阈值或忽略差异。

新增独立的 compact/median KD-tree 查询实现，沿用 SciPy 的首个访问近邻规则。1,500 个固定查询，以及真实结果中 58,411 个 Gaussian 候选上的 4,907 次补充峰查询，对照均无差异。相关算法参考 SciPy 的 [tree build](https://raw.githubusercontent.com/scipy/scipy/v1.17.1/scipy/spatial/ckdtree/src/build.cxx) 与 [nearest query](https://raw.githubusercontent.com/scipy/scipy/v1.17.1/scipy/spatial/ckdtree/src/query.cxx)。

修正后通过 `verify_full_snapshot.py` 对同一原始 FITS 重跑：

- 候选／返回数均为 `66,469`，质量通过数 `27,553`，全部源字段和参数记录通过对照。
- 原生进程直接读取原始 FITS，没有经 Python 传递像素；本次含进程启动与 JSON 输出约 `10.16 s`。保存的 Python 基线运算约 `59.39 s`；这些是本机本次实测，不代表所有算法或平台的性能。
- Python 基线源码 SHA-256：`b790c1451e76906e8a2b1f01bc1bc06aec399229ecd3e5f971d34e6f30846e6c`。
- FITS 前后 SHA-256：`6d6926a0546a5e2d28c6e76762bf72127ff3f0e37b589993eb381881bfe0acc6`。
- 保存的 Python 基线与原生输出各有压缩 JSON，验证器同时检查源码、数据与基线指纹；报告位于本地 `tmp/native-detection-full-corrected/verification.json`。

## 尚未完成

现有桌面仍使用原有 Python worker，完整生产运算链尚未切换；`migration.json` 中 `complete` 与 `python_runtime_removed` 均为 false。

数值对照当前覆盖 float64 检测路径；float32 快速序列路径、星表与测光标定、序列运动、合成、全部审计／实验及 78 个 CLI 契约仍需继续移植和对照。完整生产桌面尚未切换，未生成纯 Rust 发布包。

15 帧 FITS 读取验证证明读取、遥测和抽样像素契约；新增全幅计算验证只证明首帧 hybrid 检测与现有 Python 算法一致，不证明完整 15 帧运动分类或物理天体真值。

本轮详细真实文件报告位于本地 `tmp/native-fits-verification-final/verification.json`，未把原始 FITS 或本地验证产物加入源码目录。
