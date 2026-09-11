# rst19-sol

面向太空天基图像星图识别、星图分析与创新分析赛项的方案仓库。

当前仓库已完成比赛规则、数据格式和本地数据集边界整理，并开始落地可离线运行的 Python 基线。本文档和 `doc/` 只把已经从比赛材料、实际 FITS 文件或可复现实验中确认的信息写成事实，不把规划内容写成已完成能力。

详细的资料分类、方法调研、验证清单和参考链接见 [`doc/README.md`](./doc/README.md)；第一次使用软件请先看 [`软件使用说明`](./doc/软件使用说明.md)。

## 赛题概览

比赛采用排位赛制，依据所有参赛队伍的最终得分进行总排序，并按排位划分奖项。参赛数据为大赛提供的 15 张太空天基图像，数据来源为开运一号卫星。

基础比赛内容包括：

- **星图识别**：使用指定的一张图片完成星点识别、星等识别中的一项或全部。
  - 星点识别：从图片中找出全部星点，并计算星点总数。
  - 星等识别：找出图片中亮度最低的星，并标注该星的星等。
  - 参赛方案需要提交问题思路和算法；同时完成两个方向可获得更完整的基础赛项覆盖。
- **星图分析**：从 15 张图片中识别多个运动目标，并提交问题思路和算法。

完成基础比赛内容后进入创新板块。参赛队可结合照片信息和团队能力，提取、计算和分析其他有用数据。

## 当前资料

| 内容 | 当前状态 |
| --- | --- |
| FITS 图像 | 本地已提供 15 张，首次提交不上传原始二进制数据 |
| 数据格式说明 | 已归档至 [`doc/00-项目资料/天基图像数据格式说明.docx`](./doc/00-项目资料/天基图像数据格式说明.docx) |
| 星点识别算法 | 已实现 Gaussian、双尺度 DoG、两尺度 à trous/Starlet 多通道宽筛、候选来源追踪、双 PSF/BIC 去混叠与统一质量细筛；GUI 当前默认使用 `hybrid` 平衡口径，单图不限制候选源数量，见 [`检测器实验记录`](./doc/02-星图识别/检测器实验记录.md) |
| 星等识别算法 | 已实现 `m_inst`、显式 Gaia DR3 TAP 公共星表接入、自动星对板解/仿射 WCS、参考星质量筛选、多参考星零点/颜色项标定、15 帧相对光度标尺、逐源误差与状态、以及优先使用高质量视差、必要时回退到带来源 GSP-Phot 距离的 `M` 估计；首帧已有真实链路通过，完整场景 WCS、同设备响应和最暗源全场覆盖仍待验证 |
| 运动目标识别算法 | 已实现 15 帧平移配准、普通点轨迹关联、高速点源“完整候选宽筛 → 原始 ADU/PSF 细筛”、原图线状候选提取与跨帧拟合；严格 `moving` 目标和待复核线分层显示，真值仍待星表/注入验证 |
| 创新分析指标 | 已建立 15 帧关系/轨迹证据导出器、相对光度标尺和星等可观测性/false-valid 质量报告；创新结论仍待真值和注入实验验证 |
| 15 帧合成大图 | 已实现注册联合视场、稳健 ADU 融合、coverage/scatter 审计、15×查看和 PNG/NPY/JSON 导出；当前只做平移注册，不宣称完整 WCS 镶嵌 |
| 实验结果与提交材料 | 已建立首帧实验记录、特殊亮点真实性研究、参赛用核心短稿、扩展证据稿和详版答辩材料，见 [`检测器实验记录`](./doc/02-星图识别/检测器实验记录.md)、[`亮点真实性与伪影判别研究`](./doc/02-星图识别/亮点真实性与伪影判别研究.md)、[`竞赛论文核心短稿`](./doc/05-论文答辩/竞赛论文核心短稿.md)、[`竞赛论文短稿（扩展证据版）`](./doc/05-论文答辩/竞赛论文短稿.md) 与 [`论文与答辩材料（详版）`](./doc/05-论文答辩/论文与答辩材料.md) |

## 评分规则整理

以下内容按目前收到的比赛规则整理。基础赛项和创新赛项当前列出的评分项分别为 60 分和 30 分；如比赛方后续发布补充规则，以官方版本为准。

### 基础赛项：60 分

| 分项 | 评分项 | 分值 | 评分标准摘要 |
| --- | --- | ---: | --- |
| 星图识别 | 科学性 | 15 | 问题清晰、逻辑完整、星点/星等数据准确、无原理性冲突；同时完成两个方向为 15 分，只完成一个方向为 14 分；存在科学性问题为 12 分；问题不清晰时，专家每提出一次原理性质疑扣 1 分 |
| 星图识别 | 创新性 | 10 | 独创性强且领域内无相同作品为 10 分；有 1-2 件类似作品但有明显优化为 8 分；有 3 件以上类似作品时，专家每提出一次质疑扣 2 分 |
| 星图分析 | 科学性 | 20 | 问题清晰、逻辑完整、数据准确、无原理性冲突为 20 分；存在科学性问题为 15 分；问题不清晰时，专家每提出一次原理性质疑扣 1 分 |
| 星图分析 | 创新性 | 15 | 独创性强且领域内无相同作品为 15 分；有 1-2 件类似作品但有明显优化为 12 分；有 3 件以上类似作品时，专家每提出一次质疑扣 2 分 |

### 创新赛项：30 分

| 评分项 | 分值 | 评分标准摘要 |
| --- | ---: | --- |
| 创新分析：创新性 | 15 | 得出 4 个及以上创新有效分析数据为 15 分；3 个为 12 分；2 个为 9 分；1 个为 6 分 |
| 创新分析：实用性 | 10 | 分析所得数据对现实场景有实际意义为 10 分；一次质疑且无法解答扣 1 分 |
| 创新分析：科学性 | 5 | 问题清晰、逻辑完整、无原理性冲突为 5 分；存在科学性问题为 3 分；问题不清晰时，专家每提出一次原理性质疑扣 1 分 |

## 数据说明

### 数据集概况

`doc/00-项目资料/原始数据/` 中的本地数据经文件和 FITS 头检查得到：

- 文件数量：15 个 FITS 文件。
- 单文件大小：33,557,312 字节，约 32 MiB。
- 数据总大小：503,359,680 字节，约 480 MiB。
- 图像位数：`BITPIX=16`。
- 图像尺寸：`NAXIS1=4096`、`NAXIS2=4096`。
- 曝光时间：`EXPOSURE=1500 ms`。
- 方位与俯仰：样本文件头为 `AZIMUTH=270`、`ELEVATIO=0`。
- 观测时间：由 `DATE-OBS` 给出，当前 15 张图像形成连续时间序列。

### 15 张图的关系

这 15 张图属于同一段连续观测，不是 15 个独立场景：尺寸、曝光和成像视场一致，`DATE-OBS` 只相隔约 1.5 秒，总跨度约 21.014 秒。它们应先通过静态星场估计帧间配准，再在配准坐标中分析源的持续性、位移和形态。逐帧候选点数量不能相加后当作星数，单帧亮线也不能直接当作运动目标；当前仓库把“静态星点”“单帧长线/异常候选”“跨帧运动候选”和“注册合成大图”分开记录。合成大图保留联合 footprint、coverage 和 scatter，不用生成式填充扩展视场。完整证据见 [`15 帧关系与证据说明`](./doc/04-创新分析/15帧关系与证据说明.md) 和 [`15 帧注册合成大图`](./doc/04-创新分析/15帧注册合成大图.md)。

文件名当前为：

```text
20260330163205413_9901.fits
20260330163206886_9901.fits
20260330163208415_9901.fits
20260330163209888_9901.fits
20260330163211417_9901.fits
20260330163212890_9901.fits
20260330163214419_9901.fits
20260330163215892_9901.fits
20260330163217421_9901.fits
20260330163218894_9901.fits
20260330163220423_9901.fits
20260330163221896_9901.fits
20260330163223425_9901.fits
20260330163224898_9901.fits
20260330163226427_9901.fits
```

原始 FITS 文件已按资料分类收纳到 `doc/00-项目资料/原始数据/`，但未纳入 Git 提交，原因是单个数据集约 480 MiB，且原始比赛数据的公开和再分发权限尚未在本仓库中声明。`.gitignore` 已对 FITS 文件进行保护；后续如需通过 Git LFS、Release 附件或其他数据存储发布，应先确认比赛方授权和分发方式。

### FITS 文件头字段

| 字段 | 含义 |
| --- | --- |
| `BITPIX` | 图像位数 |
| `NAXIS1` | 横向像素数 |
| `NAXIS2` | 纵向像素数 |
| `SITELONG` | 观测点经度；天基图像不使用 |
| `SITELATI` | 观测点纬度；天基图像不使用 |
| `SITEALTI` | 观测点海拔；天基图像不使用 |
| `AZIMUTH` | 方位 |
| `ELEVATIO` | 俯仰 |
| `DATE-OBS` | UTC 观测时间，即曝光中心时刻 |
| `EXPOSURE` | 曝光时间，单位为 ms |

FITS 头部占前 2880 字节，图像数据从偏移 2880 字节开始。资料说明中写作“4096×4098×16bit”，但当前实际文件的 `NAXIS2` 均为 4096，且文件大小与 4096×4096×16 bit 图像一致；实现时应读取并校验实际 FITS 头，不应仅依赖文字说明中的尺寸。

### 补充的开运一号相机参数

用户随后补充了《开运01星相机参数》截图：像元尺寸 `9 μm`、阵列 `4096×4096`、位深 `16 bit`、入瞳口径 `120 mm`、焦距 `220 mm`、谱段 `450–750 nm`、标称视场 `9.78°×9.78°`。其中阵列尺寸和位深与当前 FITS 实测一致；其余是外部设备资料，不是 FITS 头中已经验证的 WCS 卡片。完整推导和边界见 [`开运一号相机参数与尺度先验`](./doc/00-项目资料/开运一号相机参数与尺度先验.md)，机器可读记录见 [`开运一号相机参数.json`](./doc/00-项目资料/开运一号相机参数.json)。

按 `9 μm + 220 mm` 计算的像元角尺度约为 `8.438 arcsec/pixel`；按标称视场计算约为 `8.596 arcsec/pixel`。两者相差约 `1.87%`，因此它们目前只能作为 WCS 先验敏感性范围，不能自动替代星表匹配后的尺度、旋转、parity 和畸变验证。`450–750 nm` 也只是宽谱响应范围，不能直接把仪器星等改名为 V/G 星等。

### 首行辅助数据

按数据格式说明，图像数据的第一行前 208 字节为辅助数据，每个字段为 8 字节 `double`，共 26 个字段。辅助数据字段如下：

| 顺序 | 字段 | 含义 |
| ---: | --- | --- |
| 1-2 | `p_az`, `p_el` | 方位、俯仰指向，与 FITS 头中的 `AZIMUTH`、`ELEVATIO` 对应 |
| 3-4 | `ra`, `dec` | 卫星光轴指向的赤经、赤纬 |
| 5-7 | `j2000_x`, `j2000_y`, `j2000_z` | J2000 坐标系下位置，单位 m |
| 8-10 | `j2000_xv`, `j2000_yv`, `j2000_zv` | J2000 坐标系下速度，`m/s?`；由位置/时间差分自洽支持，格式说明速度行原文待确认 |
| 11-14 | `q1`, `q2`, `q3`, `q4` | 卫星本体相对于 J2000 的四元数；矢量在前、标量在后，表示 J2000 到本体坐标系的旋转 |
| 15-17 | `roll`, `pitch`, `yaw` | 横滚角、俯仰角、偏航角，单位 deg |
| 18-20 | `roll_v`, `pitch_v`, `yaw_v` | 横滚角速度、俯仰角速度、偏航角速度，单位 deg/s |
| 21-23 | `wgs84_x`, `wgs84_y`, `wgs84_z` | WGS84 坐标系下位置，单位 m |
| 24-26 | `wgs84_xv`, `wgs84_yv`, `wgs84_zv` | WGS84 坐标系下速度，`m/s?`；由位置/时间差分自洽支持，格式说明速度行原文待确认 |

数据说明附录提供了 Python 和 C++ 的读取示例。示例包含 16 位字节交换和小端 `double` 解包步骤，正式实现前需要用已知字段、数值范围和跨语言结果进行校验，避免把字节序问题带入姿态、位置或速度分析。

本轮新增辅助姿态一致性审计：按照格式说明的相机本体 `-Y` 指向和 `(q1,q2,q3,q4)` 四元数约定，将姿态推导的光轴与每帧辅助 `ra/dec` 对照。15 帧四元数范数偏离 1 不超过 `2.3×10^-16`，推导光轴与辅助 `ra/dec` 的最大角残差约 `1.1×10^-10` 角秒，且 `p_az/p_el` 与头部 `AZIMUTH/ELEVATIO` 全部一致。这个结果只证明辅助字段之间的几何自洽；它还没有给出图像 x/y 轴方向、像元角尺度、旋转和畸变，因此不能直接当作完整 WCS。审计命令为：

```powershell
rst19-aux-audit doc/00-项目资料/原始数据 --out-dir tmp/auxiliary-attitude-audit
```

输出 `auxiliary_boresight_audit.csv/json`，并把逐帧光轴先验和姿态范围留给后续星表匹配使用。

格式说明与实际 FITS 的只读核验可运行：

```powershell
rst19-format-audit doc/00-项目资料/原始数据 --out-dir tmp/format-audit-code-pattern-current-v1
```

该审计确认实际文件的尺寸、文件长度、`BITPIX`/字节序、缩放卡片、`BLANK` 卡片、首行辅助区域和负值/极值统计，并用 `DATE-OBS` 检查位置—速度内部自洽性；它不把 `-1` 或极值自动判为坏像素/饱和，也不改变默认检测读取。当前 15 帧均为合法 `4096×4096`、大端 `>i2`，`BLANK` 为 `0/15`，精确 `-1` 合计 `2,185`。输出为 `tmp/format-audit-code-pattern-current-v1/format_audit_frames.csv`、`format_audit_velocity.csv` 和 `format_audit.json`。

## 当前实现路线

当前已实现 FITS 读取、首行辅助数据解码、局部背景/RMS、Gaussian/DoG/Starlet 多提案宽筛、候选来源追踪、双 PSF/BIC 去混叠、统一点源质量细筛、重复高位码/异常负值数据有效性审计、线状结构审计、最暗可信源筛选、用户提供 CSV 星表加载、切平面 WCS、先验 WCS 下的全局一对一匹配、匹配后的局部仿射 WCS 校准、15 帧平移配准/普通轨迹关联、高速点源完整候选宽筛与原始 ADU 细筛、15 帧注册中值/稳健叠加暗星恢复层和 Python 桌面工作台。Starlet 当前是有完整负结果记录的实验模式，不替代 GUI 的 `hybrid` 默认口径；完整比赛能力仍按以下顺序推进：

1. **数据校验**：批量读取 FITS 头，检查尺寸、位深、曝光时间、时间顺序和辅助数据解码结果；当前已具备单帧读取校验基线。
2. **星点识别**：GUI 当前默认以 `hybrid`（Gaussian + 双尺度 DoG）在 `4σ` 取得候选，以 `PSF FWHM=2 px`、通量 SNR/形状/掩膜规则形成质量层；单图不设置候选源数量上限。已完成第一轮参数扫描、按需注入实验和星表核验 UI，仍需可靠星表/WCS 与人工抽检，才可冻结比赛星点计数。
3. **星等识别**：无星表时输出最暗可信检测源及 `m_inst = -2.5 log10(flux_rate)`；可通过 `rst19-gaia-frame` 或 GUI“研究工具 → 星表核验 → 在线获取 Gaia DR3”显式生成当前 FITS 的公共参考子表，再在先验 WCS/匹配通过时按 Gaia 质量字段筛选参考星，拟合声明波段的 `m_cal`、逐源误差/状态，以及通过视差质量门控的 `M`。大视场可用 `rst19-gaia-tiled` 分块查询并审计截断；15 帧还可选拟合相对光度标尺。实际物理解释仍受本设备响应、波段转换、消光和星表质量限制。结果审计可用 `rst19-photometric-report`，避免把仪器星等冒充绝对星等。
4. **运动目标识别**：已实现质量点源的相邻帧平移估计、配准坐标唯一关联、针对超过普通 `4 px` 帧间位移的高速点源“宽筛候选峰 → 三帧常速度种子 → 15 帧关联 → 原始 ADU flux SNR/PSF/线掩膜细筛”，以及注册中值差分基线和静态/运动/瞬态分类；对被点源质量规则拒绝的细长高残差结构建立独立线状候选轨迹。当前高速候选关联取每帧匹配滤波响应前 `20,000` 条作为性能工作集，完整 `candidate_count` 仍保留，最终 `flux_snr` 不使用匹配滤波分数替代；星表窗口可用唯一匹配点拟合局部仿射 WCS，并把图像平面 `px/s` 换算成视场切平面 `arcsec/s`，但这仍不是盲解算、完整 WCS 或目标轨道速度。
5. **创新分析**：在检测结果可靠后，围绕目标运动特征、图像质量、观测几何、时间序列和帧间相对光度标尺等方向设计可解释、可复核的分析指标。相对标尺和星等质量报告的命令说明见 [`星等估计与标定`](./doc/02-星图识别/星等估计与标定.md)。

研究工具还提供 `rst19-feature-sequence` 的真实 15 帧特征持久性审计，以及 `rst19-pair-audit` 的固定 detector 坐标近邻双源审计；前者同时输出首帧锚点持久性、`sequence_feature_temporal_profile.csv` 的逐类时间剖面、`sequence_feature_class_transition.csv` 的类别条件跨帧转移、`sequence_feature_method_frame_summary.csv` / `sequence_feature_method_profile.csv` 的逐帧与 15 帧提议器来源交叉、`sequence_feature_source_subgroup_persistence.csv` 的紧凑质量来源子组逐源持久性，以及 `sequence_feature_diagnostic_subgroup_persistence.csv` 的非紧凑特征“位置重复/同机制重复”对照。后者直接复核截图局部的固定/注册坐标、原始值域与单/双 PSF 证据，检测器还会将“重复高位码 + 异常负值”记录为 `CODE_PATTERN` 并保留候选审计；`rst19-temporal-codes` 另输出关注坐标的逐帧原始值表，区分“固定/低变化 detector 结构”与“每帧重新测得的星光响应”。二者输出原始证据和质量层结果，但都不会把候选数直接包装成物理恒星真值。

若要进一步检查“跨帧稳定响应是否可能只是固定结构”，可在首帧源表和序列持久性 JSON 已生成后运行分层强制测光：

```powershell
rst19-forced-stability `
  tmp/source-quality-audit-code-pattern-current-v3/source_catalog.csv `
  tmp/sequence-feature-persistence-code-pattern-current-v3/sequence_feature_persistence.json `
  --base-dir . --anchor-coordinate peak `
  --local-peak-search-radius 1 `
  --out-dir tmp/forced-stability-audit-code-pattern-current-v8-peak
```

该命令按首要特征类别抽取最多 `32` 个源，沿累计平移回到 15 帧未降噪 FITS，在半径 `4 px` 孔径上输出逐源逐帧通量 SNR、3×3 支持、FWHM、椭圆率和质量样式帧数。默认 `peak` 与序列提案坐标一致；传 `--anchor-coordinate centroid` 可做测量质心敏感性对照。默认还在预测位置周围最多 `±1 px` 用三尺度 Gaussian 匹配响应做局部重定位，并同时输出固定/局部测量和 `relocalization_*` 残差；局部最大值有多重试验偏差，只是去混叠诊断，不是第二次星点发现。传 `--local-peak-search-radius 0` 可关闭重定位。它不重跑候选、不修改 `quality_passed`、不写入检测缓存，结果不能当作真星率；解释和限制见 [`检测器实验记录 8.58/8.59`](./doc/02-星图识别/检测器实验记录.md)。

要复核“各类特征是否只是某个门槛下的标签”，运行首帧参数敏感性审计：

```powershell
rst19-feature-parameter-sensitivity `
  doc/00-项目资料/原始数据/20260330163205413_9901.fits `
  --threshold-levels 4,5,6,8 --flux-snr-levels 3,5,7,9 `
  --out-dir tmp/feature-parameter-sensitivity-code-pattern-current-v1
```

该命令固定同一 FITS，分别扫描候选阈值与质量层 `flux SNR`，输出 `feature_parameter_runs.csv`、按类别匹配的 `feature_parameter_profiles.csv`、默认类别到新类别的 `feature_parameter_transitions.csv` 和 JSON。当前首帧的候选池对 `flux SNR=3/5/7/9` 均为 `84,594`，但质量数为 `41,110/29,260/22,438/18,296`；`flux SNR=9` 时有 `9,614` 个默认紧凑位置转入弱/背景类，降到 `3` 时有 `10,280` 个默认弱/背景位置转入紧凑类。这是质量标签的可逆转移，不是物理恒星增删；实现只写研究产物，不改 GUI 默认值、质量规则或缓存。

若三组序列审计已经完成，可只读取已有 JSON 做 15 帧参数对照，不重复检测：

```powershell
rst19-feature-sequence-compare `
  --run SNR3=tmp/sequence-feature-persistence-parameter-flux3-current-v1/sequence_feature_persistence.json `
  --run SNR5=tmp/sequence-feature-persistence-code-pattern-current-v4/sequence_feature_persistence.json `
  --run SNR9=tmp/sequence-feature-persistence-parameter-flux9-current-v1/sequence_feature_persistence.json `
  --out-dir tmp/sequence-feature-parameter-comparison-current-v1
```

比较器会校验三组是否都是唯一 `15` 帧、相同 `≥12/15` 持久性门槛和相同关联半径，并分别输出候选持久性、质量持久性和类别转移；它不会把每帧按类别展开的 `135` 行误当成 `135` 帧，也不会修改 GUI 默认参数或检测缓存。当前三组候选数均为 `84,594`，`compact_quality` 的质量持久率为 `0.776/0.815/0.864`；这表示门槛改变质量子集稳定性，不表示物理恒星真值率。产物为 `tmp/sequence-feature-parameter-comparison-current-v1/`。

对指定候选做类别内经验分位审计：

```powershell
rst19-feature-class-context `
  tmp/source-quality-audit-code-pattern-current-v3/source_catalog.csv `
  --target-id 82931 --target-id 82934 `
  --out-dir tmp/feature-class-context-pair-82931-82934-current-v1
```

该命令把候选放回所属 `feature_class` 的 SNR、FWHM、形状、PSF 支持和峰值分布，输出同类 `p10/median/p90/empirical percentile`。它只做相对上下文审计，不重新检测、不输出恒星概率；当前 `82934` 在拥挤类中虽然位于高显著性端，仍因 `UNRESOLVED_BLEND`、共同孔径和跨帧独立性不足而保留为候选。

若要把“各类中 SNR 最高但仍被拒”的 hard negative 系统列出，可运行：

```powershell
rst19-feature-hard-negative `
  tmp/source-quality-audit-code-pattern-current-v3/source_catalog.csv `
  --top-n 5 --metric flux_snr `
  --out-dir tmp/feature-hard-negative-code-pattern-current-v1
```

该命令默认排除 `other_rejected` 汇总桶，按 8 个有效类别输出每类 top 5 落选候选、flags、FWHM、椭圆率、PSF 支持、足迹和 `filter_flux_snr_ratio`。当前最高落选 `flux SNR` 在不同类别分别达到 `1863.2/936.3/612.4/540.6/133.9`（线状/范围异常/边缘掩膜/拥挤/尖峰支持不足），但这些是不同机制的高显著性响应，不是噪点率或恒星概率；比值字段只表示孔径通量与匹配滤波的分歧，不是综合真星分数。产物为 `tmp/feature-hard-negative-code-pattern-current-v1/`，实现只读源级 CSV，不修改默认检测、GUI 或缓存。

若要检查质量旗标的条件交互，可运行：

```powershell
rst19-feature-flag-interaction `
  tmp/source-quality-audit-code-pattern-current-v3/source_catalog.csv `
  --high-snr-threshold 10 `
  --out-dir tmp/feature-flag-interaction-code-pattern-current-v1
```

该命令把旗标分成 presence 和 `exact_set` 两种口径。当前 `PARTIAL_MASKED` 单独出现的 `2,777` 行全部通过质量层，但与 `LOW_FLUX_SNR` 或 `INSUFFICIENT_PSF_SUPPORT` 叠加的 `7,703/5,084` 行均未通过；这些是当前质量规则的条件行为，不是物理真星率或伪影概率。产物为 `tmp/feature-flag-interaction-code-pattern-current-v1/`，不重新读取 FITS、不修改 GUI 或检测缓存。

若要量化每类候选距离当前数值质量门还有多远，可运行：

```powershell
rst19-feature-gate-margin `
  tmp/source-quality-audit-code-pattern-current-v3/source_catalog.csv `
  --out-dir tmp/feature-gate-margin-code-pattern-current-v1
```

该命令输出 `flux_snr`、PSF 支撑、FWHM、椭圆率、sharpness 和足迹的有符号余量：正值表示字段在门内，负值表示越界。当前弱/背景类的 `flux_snr` 越界为 `21,250/21,250`，尖峰类 PSF 支撑越界为 `23,823/23,824`；线状类 `flux_snr` 中位余量仍为 `+828.9`，所以高 SNR 不能替代线状结构审计。余量不跨量纲求和，不是新的真星分数；该命令只读已有源表，不重算 FITS、不修改 GUI 或缓存。

若要进一步查看每个候选具体走了哪条质量规则路径，可运行：

```powershell
rst19-feature-gate-route `
  tmp/source-quality-audit-code-pattern-current-v3/source_catalog.csv `
  --out-dir tmp/feature-gate-route-code-pattern-current-v1
```

它逐行输出数值门越界字段、结构旗标和 `quality_passed`，并把拒绝候选分成数值型、结构型和混合型；当前 `82931/82934` 都是结构型拒绝，分别对应 `CODE_PATTERN/UNRESOLVED_BLEND`。这是当前 detector-rule 的观测路径，不是关闭某个规则后的反事实实验，也不是噪点率或物理恒星概率；产物为 `tmp/feature-gate-route-code-pattern-current-v1/`。

若要继续把“质量门路径”和“15 帧是否反复出现响应”放在一起检查，可运行：

```powershell
rst19-feature-gate-temporal-route `
  tmp/feature-gate-route-code-pattern-current-v1/feature_gate_route_sources.csv `
  tmp/sequence-feature-persistence-diagnostic-sources-v6/sequence_feature_diagnostic_sources.csv `
  tmp/sequence-feature-persistence-code-pattern-current-v4/sequence_feature_persistence.json `
  --target-id 82931 --target-id 82934 --target-id 44132 `
  --out-dir tmp/feature-gate-temporal-route-code-pattern-current-v1
```

该命令只读已有 CSV/JSON，不重算 FITS。它区分候选出现、同诊断子组出现和邻域质量响应；缺少逐源诊断行时保留空值，不把它写成 `0`。当前 `82931/82934` 都是 `4/15` 候选、`0/15` 邻域质量响应，`82934` 同机制为 `3/15`；这加强“两个宽筛响应尚未形成稳定质量身份”的解释，但不等于普通噪声证明或物理恒星计数。产物为 `tmp/feature-gate-temporal-route-code-pattern-current-v1/`，实现为 `src/rst19/feature_gate_temporal_route.py`，测试为 `tests/test_feature_gate_temporal_route.py`。
它同时输出 `feature_gate_temporal_subgroup_summary.csv`，按诊断子组比较候选位置持久与同机制持久；当前 `masked_partial` 为 `9,048/11,289` 对 `9/11,289`，`blend_unresolved` 为 `509/905` 对 `196/905`。这个“机制差额”用于决定复核路由，不是噪点概率、precision、完备率或物理恒星数。
若追加 `--target-id`，还会输出 `feature_gate_temporal_targets.csv`，给出目标在所属子组中的含并列值经验位置；当前 `82934` 的候选 `4/15` 位于 `blend_unresolved` 子组的低持久端（`le/ge=9.4%/94.7%`）。该经验位置只用于复核排序，不是概率或显著性检验。

若要把类别空间分布和截图 pair 的局部邻域一起纳入筛查，可运行：

```powershell
rst19-feature-spatial-context `
  tmp/source-quality-audit-code-pattern-current-v3/source_catalog.csv `
  --target-id 82931 --target-id 82934 `
  --top-n 5 --grid-size 4 --edge-margin-px 16 `
  --out-dir tmp/feature-spatial-context-code-pattern-current-v1
```

它输出类别网格熵、最大单格占比、边缘比例、高 SNR 落选热点，以及代表候选的最近邻和 `5/10/20 px` 邻域类别。当前 `linear_artifact` 为 `96/96` 同格，`crowded_blend` 覆盖 `16/16` 格；`82931/82934` 距边缘约 `73.8/73.2 px`、最近候选距 `2.738 px`，因此不是边缘层，而是局部近邻/混合待核验。空间集中只决定下一步实验，不是伪影真值、双星概率或星表身份；该命令不重新检测、不修改默认质量层或缓存。

将每类最高 SNR 的代表带回 15 帧原始图做固定/局部强制测光：

```powershell
rst19-forced-stability `
  tmp/source-quality-audit-code-pattern-current-v3/source_catalog.csv `
  tmp/sequence-feature-persistence-code-pattern-current-v4/sequence_feature_persistence.json `
  --max-per-class 1 `
  --include-detection-id 64963 --include-detection-id 4469 `
  --include-detection-id 42 --include-detection-id 39547 `
  --include-detection-id 25028 --include-detection-id 23168 `
  --include-detection-id 66157 `
  --out-dir tmp/feature-hard-negative-forced-stability-current-v1
```

该命令只做逐帧原图测量，不改源级质量标志或检测缓存。`quality_like` 只是当前测量窗口通过样式门的诊断，不覆盖原始 `quality_passed=False`；当前结果显示线状最高单帧 SNR 可在注册位置降到约 `5.4`，范围异常仍可稳定很亮，拥挤类则频繁局部重定位，说明“亮、稳定、独立”仍是三个不同问题。
若要保证具体截图反例进入同一分层审计，可重复传入 `--include-detection-id <ID>`；该 ID 会追加到分位点样本，不会替换原有抽样。
若同时显式追加两个或更多 ID，结果还会写出 `forced_stability_pair_frame_metrics.csv` 和 `forced_stability_pair_summary.csv`，记录注册后局部相对间距、收缩比例和向内偏移比例；它们只用于解释共享结构风险，不是双星或星表身份判据。

若要继续复核某一对候选是否具有正常点扩散函数（PSF）形状，可先用 `rst19-sources` 生成首帧全量源表，再运行经验 PSF 对照：

```powershell
rst19-empirical-pair-audit doc/00-项目资料/原始数据 `
  --source-catalog tmp/source-quality-audit-code-pattern-current-v3/source_catalog.csv `
  --primary-id 82931 --secondary-id 82934 `
  --coordinate-mode centroid `
  --shifts-json tmp/sequence-feature-persistence-code-pattern-current-v3/sequence_feature_persistence.json `
  --out-dir tmp/empirical-pair-psf-audit-code-pattern-current-v1
```

该命令只从模板帧的质量通过、未饱和、未掩膜和隔离源构造经验 PSF，在 15 帧原始 FITS 的注册预测位置计算全局/局部相关度、相对残差和中心能量占比；局部模板不足时保留为空，不把“没有局部模板”解释成“不是星”。它不改默认检测数量、`quality_passed`、GUI 或检测缓存。相关度是形状诊断，不是恒星概率、星表身份或去混叠后的独立通量。

对 `82931/82934` 的局部适用域扫描还显示：在 `18--512 px` 窗口内虽然有 `6,081--17,132` 个候选，但满足 `21 px` 隔离条件的模板源始终只有 `2` 个；扩大到 `1024 px` 才得到 `6` 个宽范围模板源。该结果把 `local_psf=None` 解释为拥挤区校准缺口，而不是噪点证明；宽范围 fallback 的主/副相关度中位数为 `0.553/0.480`，仅作敏感性证据。产物为 `tmp/local-psf-availability-pair-82931-82934-current-v1/local_psf_availability.csv/json`。

对同一 pair 的强制测光共变做控制审计：

```powershell
rst19-pair-covariance-audit `
  tmp/forced-stability-audit-code-pattern-current-v8-pair-peak/forced_stability_frame_metrics.csv `
  --primary-id 82931 --secondary-id 82934 `
  --out-dir tmp/pair-flux-covariance-audit-code-pattern-current-v1
```

该命令比较固定/局部 `flux_snr` 的跨帧 Pearson、按其它源逐帧中位数归一后的相关性，并用同表其它完整源的两两组合给出探索性控制分布；它还报告按 pair 总响应中位数分层的组内相关，避免把共同状态切换误读为两条独立星光曲线。输出只用于共享孔径/局部结构/值域状态诊断，不是双星概率、正式 p 值或星表身份。

控制经验 PSF 的坐标锚点敏感性：

```powershell
rst19-empirical-pair-fit doc/00-项目资料/原始数据 `
  --source-catalog tmp/source-quality-audit-code-pattern-current-v3/source_catalog.csv `
  --primary-id 82931 --secondary-id 82934 `
  --coordinate-mode centroid `
  --mask-modes raw,range_masked,sentinel_masked,range_sentinel_masked `
  --shifts-json tmp/sequence-feature-persistence-code-pattern-current-v3/sequence_feature_persistence.json `
  --out-dir tmp/empirical-pair-fit-code-pattern-current-centroid-v2
```

该研究命令在同一局部窗口、同一掩膜和同一经验 PSF 中比较固定位置 K=1/K=2，并在双源中点周围搜索最佳单源位置，专门检查固定质心是否给双源模型不公平优势。质心锚点的固定双源 raw 结果为 `11/15` 帧达到当前证据线，但最佳单源网格控制为 `0/15`；峰值锚点固定与控制均为 `0/15`。网格结果是选择偏差控制，不是双星概率或正式自由位置 BIC；该模块只写审计产物，不改默认检测、GUI 或缓存。

为检查局部模板缺口是否改变结论，另用 `1024 px` 范围内的 `6` 个隔离源重建经验 PSF；固定质心双源仍为 `11/15`，最佳单源位置控制仍为 `0/15`，首帧固定双源 `ΔBIC=-6.31`、第二分量 SNR `0`。这只说明模板选择会改变拟合幅度，不能把两个宽筛响应升级为两颗恒星；结果保存在 `tmp/empirical-pair-fit-local-wide-1024-pair-82931-82934-current-v1/`。

若要单独检查工程重复码，可把 `--mask-modes` 增加为 `raw,repeated_code_masked,range_repeated_code_masked,sentinel_repeated_code_masked`，并显式传入 `--repeated-code-values 3990,3991,3992,3993`；重复码屏蔽后的结果仍需和坐标锚点、PSF 和星表证据合并解释。

若要检查截图 pair 的近邻距离是否只是全图候选的普遍几何现象，可运行源目录群体控制：

~~~powershell
rst19-source-separation-audit tmp/source-quality-audit-code-pattern-current-v3/source_catalog.csv --target-id 82931 --target-id 82934 --out-dir tmp/source-separation-audit-code-pattern-current-v1
~~~

它分别统计测量质心和整数峰的最近邻距离，并把指定 ID 放回全量候选分布；当前 pair 的质心最近邻距为 2.738 px，处在全量 84,594 个候选最近邻距离的 0.0165% 分位，而峰坐标最近邻距 4.123 px 处在 5.70% 分位。输出是群体几何控制，不是双星概率、星表身份或真值。

如果还要检查两个框在同一几何背景下是否属于同一种检测机制，可运行：

~~~powershell
python -m rst19.pair_mechanism_context_cli `
  tmp/source-quality-audit-code-pattern-current-v3/source_catalog.csv `
  --target-id 82931 --target-id 82934 `
  --out-dir tmp/pair-mechanism-context-code-pattern-current-v1
~~~

默认筛选质心距 `≤3.5 px`、峰距 `3.5--4.8 px` 的候选对。当前得到 `11` 对，其中同类别 `7` 对、跨类别 `4` 对，双质量通过 `0/11`；目标 pair 是 `range_anomaly + crowded_blend`。这是候选目录的机制背景，不是噪点率、双星概率或物理真值；窗口敏感性显示质心上限放宽至 `4.0 px` 会变成 `840` 对，因此 `0/11` 不能外推为全图比例。目标位置污染注入还必须区分 baseline、injected 和 new，不能把已有候选重复算成新增双源。产物为 `tmp/pair-mechanism-context-code-pattern-current-v1/`。

要保存这组窗口敏感性对照，可运行：

~~~powershell
python -m rst19.pair_mechanism_sensitivity_cli `
  tmp/source-quality-audit-code-pattern-current-v3/source_catalog.csv `
  --target-id 82931 --target-id 82934 `
  --out-dir tmp/pair-mechanism-sensitivity-code-pattern-current-v1
~~~

该命令固定输出五组窗口：`3.0/3.5/4.0 px` 质心上限和两组峰距扩展；结果只用于窗口选择控制，不是噪点率、双星概率或物理真值。产物为 `tmp/pair-mechanism-sensitivity-code-pattern-current-v1/pair_mechanism_sensitivity.csv/json`。

如果要把“两个局部峰”转换成更接近对象层的复核单位，可运行父源组审计：

```powershell
rst19-source-group-audit `
  tmp/source-quality-audit-code-pattern-current-v3/source_catalog.csv `
  --target-id 82931 --target-id 82934 `
  --psf-fwhm 2 `
  --out-dir tmp/source-group-audit-code-pattern-current-v1
```

它只读取源级 CSV，默认用 `max(2.5 px, 1.5×PSF FWHM)` 的测量质心距离建立连通组，并把组标为 `isolated`、`unresolved_group` 或 `independent_group_candidate`。组内代表只用于显示/排序，不是恒星数；当前 CLI 的独立候选线检查质量、值域/结构旗标和双 PSF 证据，一对一像素分配、跨帧稳定性和星表身份仍是把组拆成可计数源前的后续核验。当前 v3 得到 `84,586` 个组，其中 `8` 个多成员组、`8` 个未分辨组、`0` 个独立候选组；`82931|82934` 是其中一个 `unresolved_group`。输出为 `source_groups.csv`、`source_group_class_summary.csv`、`source_group_targets.csv` 和 JSON，不修改检测缓存、默认质量层或 GUI。

组半径敏感性可用 `rst19-source-group-sensitivity`：在同一源表上并行扫描 `2.5/3.0/3.5/4.0 px`，输出统一的半径汇总和目标定位。当前全量结果的多成员组数为 `4/8/12/831`，独立候选组为 `0/0/0/25`；目标在 `2.5 px` 时分别为两个 `isolated` 候选，在 `3.0/3.5/4.0 px` 时均为同一 `unresolved_group`。这说明 `3 px` 是结合当前 PSF 的保守复核半径，不是相机的物理分辨率；半径改变会改变比较总体，不能据此推导全图星数或双星概率。产物为 `tmp/source-group-radius-sensitivity-code-pattern-current-v1/`。

```powershell
rst19-source-group-sensitivity `
  tmp/source-quality-audit-code-pattern-current-v3/source_catalog.csv `
  --target-id 82931 --target-id 82934 --psf-fwhm 2 --workers 4 `
  --out-dir tmp/source-group-radius-sensitivity-code-pattern-current-v1
```

若要继续核查一对近邻候选是否各自拥有独立像素支持，可运行局部留出审计：

```powershell
rst19-pair-support-audit `
  doc/00-项目资料/原始数据 `
  --source-catalog tmp/source-quality-audit-code-pattern-current-v3/source_catalog.csv `
  --primary-id 82931 --secondary-id 82934 `
  --shifts-json tmp/current_sequence_v3.json `
  --out-dir tmp/pair-support-audit-code-pattern-current-v2
```

该命令在每帧局部裁剪上重跑当前宽筛，用一对一分配防止一个候选被双计数，并把共同/非共同孔径、重复码和局部异常留出结果分开保存。除逐帧 `pair_support_audit.csv/json` 外，当前 v2 还生成 `pair_support_summary.csv`，明确分开统计双候选分配与双质量通过：raw 为 `4/15` 与 `0/15`，重复码或局部极端负值屏蔽后的双候选均为 `0/15`；该结果是混叠和值域敏感性诊断，不是双星概率、物理真值或默认质量层改写。

源级特征相关性审计可用 `rst19-feature-correlation`。它对 `peak`、`flux_snr`、`filter_snr`、FWHM、椭圆率、sharpness、足迹、PSF 支持和质心偏移计算 Spearman 相关性，并输出每个 `feature_class` 的特征中位数及类别内相关性。当前 `peak—filter_snr=0.9355`、`flux_snr—filter_snr=0.8711`、`filter_snr—FWHM=-0.8444`，因此多个显著性量不能相加成“真星分数”；该工具只做规则依赖审计，不输出恒星概率。产物为 `tmp/feature-correlation-audit-code-pattern-current-v1/`，其中包含 `feature_correlations.csv`、`feature_class_medians.csv`、`feature_class_correlations.csv` 和 JSON 汇总。

同一审计还在孔径半径 `3/4/5 px` 下做敏感性对照：raw 双分配均为 `4/15`，双质量通过均为 `0/15`，重复码和局部异常屏蔽后的双分配均为 `0/15`；产物分别位于 `tmp/pair-support-audit-code-pattern-current-r3/`、`current-v1/` 和 `current-r5/`。

若要回答“两个候选是否真的对应两个独立亮点”，可继续做原始像素拓扑审计：

```powershell
rst19-pair-pixel-topology `
  doc/00-项目资料/原始数据/20260330163205413_9901.fits `
  --source-catalog tmp/source-quality-audit-code-pattern-current-v3/source_catalog.csv `
  --primary-id 82931 --secondary-id 82934 `
  --out-dir tmp/pair-pixel-topology-pair-82931-82934-current-v1
```

该工具只在目标局部窗口内读取未降噪原始像素，分别统计多个阈值下的 8 邻域正值连通块和原始局部极大值，并把重复工程码/特殊负值屏蔽作为敏感性对照。当前 pair 在 `3/5/8/10σ` 下均落入同一正值连通块；但以 `3/5/7 px` 邻域寻找局部极大值时，两颗候选均不是局部峰。首帧最亮像素位于 `(2438,4022)`，而非两个候选的整数峰；这支持“同一响应结构被宽筛分成两个候选”的优先解释，但不等价于噪点证明，也不能排除真实近邻恒星在当前采样/PSF 下被混叠。产物包含 `pair_pixel_topology_components.csv`、`pair_pixel_topology_maxima.csv`、`pair_pixel_topology.json` 和可直接打开的 `pair_pixel_topology.svg`；该审计不改默认检测、质量层、GUI 或缓存。

若要检查“各类候选的 PSF 差异是否只是模板包含自身造成的”，可运行逐源留一法交叉核验：

```powershell
rst19-feature-psf-leaveout `
  doc/00-项目资料/原始数据/20260330163205413_9901.fits `
  --source-catalog tmp/source-quality-audit-code-pattern-current-v3/source_catalog.csv `
  --per-class 16 --max-template-sources 20 `
  --out-dir tmp/feature-psf-leaveout-code-pattern-current-v1
```

该命令对九个互斥首要特征类别按 `flux_snr` 等距抽样，每个样本从经验 PSF 模板候选池排除自身，同时仍用全幅候选表检查邻峰；输出 `feature_psf_leaveout.csv`、逐源表和 JSON。当前 8 个非空类别均建立了中位 `20` 个模板；`compact_quality` 留一相关度中位数为 `0.862`，`crowded_blend` 为 `0.156`，`linear_artifact` 为 `0.412`，`shape_outlier` 为 `0.270`，与全局模板结果在 `127` 个有效成对样本上没有存储精度内变化。这支持“类别形状差异不是模板自相似单独造成”，但仍只是形状诊断，不是恒星概率或物理真值。

把宽筛、质量层、15 帧响应和留一 PSF 结果合并成类别证据矩阵：

```powershell
rst19-feature-evidence `
  --source-morphology tmp/source-quality-audit-code-pattern-current-v3/source_feature_morphology.csv `
  --feature-cross tmp/feature-cross-audit-code-pattern-current-v3/feature_cross_audit.csv `
  --psf-leaveout tmp/feature-psf-leaveout-code-pattern-current-v1/feature_psf_leaveout.csv `
  --psf-spatial tmp/source-feature-psf-grid2-per16-code-pattern-current-v4/source_feature_psf_spatial.csv `
  --out-dir tmp/feature-evidence-matrix-code-pattern-current-v3-routing
```

输出 `feature_evidence_matrix.csv/json`。其中候选持久比例、同类质量比例、质量层邻域响应和候选/质量同类响应保持分列；邻域响应不是同类质量持久性。可选的 `--psf-spatial` 还会合并局部模板可得率、局部相关度/残差以及相对全局模板的变化，用来检验空间 PSF 校正是否改变类别解释；模板不足时只记录校准缺口。当前矩阵显示多个被拒类别仍可在 `12/15` 帧附近重复出现，但 PSF/质量层不支持，说明跨帧重复不能单独升级为恒星；该工具只组织研究证据，不修改默认检测、GUI 或缓存。

类别级证据轴同步汇总可在上述矩阵上运行：

```powershell
rst19-feature-evidence-axes `
  tmp/feature-evidence-matrix-code-pattern-current-v3-routing/feature_evidence_matrix.csv `
  --out-dir tmp/feature-evidence-axes-code-pattern-current-v1
```

它输出 `feature_evidence_axes.csv`、`feature_evidence_axes_sensitivity.csv` 和 JSON，其中包含 `Q|P=质量邻域持久数/候选位置持久数` 及四种 detector-level 证据模式。当前紧凑质量类为 `86.1%`，拥挤/尖峰/弱背景/形状类为 `0.5%--5.0%`，边缘/掩膜类为 `22.1%`；五组敏感性配置显示拥挤、尖峰、弱背景的不同步模式较稳，而形状/紧凑类存在工程阈值边界。该比值不等于逐源质量通过率、precision、FDR 或恒星概率，也不修改默认检测、GUI 或缓存。

检查“匹配滤波候选峰是否也是原始像素局部峰”，可运行：

```powershell
rst19-source-peak-consistency `
  tmp/source-quality-audit-code-pattern-current-v3/source_catalog.csv `
  doc/00-项目资料/原始数据/20260330163205413_9901.fits `
  --target-id 82931 --target-id 82934 `
  --out-dir tmp/source-peak-consistency-code-pattern-current-v1
```

该只读审计在 `3×3/5×5/7×7` raw 窗口中输出逐源峰一致性、类别汇总和目标明细。当前 `82931/82934` 在三个窗口都不是 raw 局部峰，且 `7×7` 共同指向 `(2438,4022)=13028 ADU`；这支持共享亮结构/值域异常导致的 detector response 重定位，但不把 raw 局部峰当作恒星充分条件，也不直接输出 precision、FDR 或物理源数。产物为 `tmp/source-peak-consistency-code-pattern-current-v1/`，实现为 `src/rst19/source_peak_consistency.py`，测试为 `tests/test_source_peak_consistency.py`。

若要进一步检查“类别平均是否掩盖提议器来源差异”，可运行：

```powershell
rst19-source-proposal-peak-audit `
  tmp/source-quality-audit-code-pattern-current-v3/source_catalog.csv `
  tmp/source-peak-consistency-code-pattern-current-v1/source_peak_consistency.csv `
  --target-id 82931 --target-id 82934 `
  --out-dir tmp/source-proposal-peak-audit-code-pattern-current-v1
```

该命令按 `feature_class × method_group` 汇总 `all_three / gaussian_only / dog_only / partial` 的质量数、来源比例和 `3×3` raw 局部峰率。当前拥挤类总体 raw 峰率为 `902/905`，但目标 `82934` 所在 `gaussian_only` 子组为 `0/2`；`82931` 所在 `range_anomaly/partial` 子组为 `0/18`。来源组合共享同一原图，不是独立投票；结果只用于复核排序，不输出恒星概率、precision、FDR 或物理源数。产物为 `tmp/source-proposal-peak-audit-code-pattern-current-v1/`，实现为 `src/rst19/source_proposal_peak_audit.py`，测试为 `tests/test_source_proposal_peak_audit.py`。

若要继续检查“来源子组是否在 15 帧中稳定重复”，可运行：

```powershell
rst19-source-proposal-temporal-audit `
  tmp/source-quality-audit-code-pattern-current-v3/source_catalog.csv `
  tmp/sequence-feature-persistence-diagnostic-sources-v6/sequence_feature_diagnostic_sources.csv `
  tmp/sequence-feature-persistence-diagnostic-sources-v6/sequence_feature_source_subgroup_persistence.csv `
  --target-id 82931 --target-id 82934 `
  --out-dir tmp/source-proposal-temporal-audit-code-pattern-current-v1
```

该命令把 `candidate_presence≥12/15`、同诊断子组持久和 `quality_presence≥12/15` 按 `feature_class × method_group` 对照；非紧凑候选逐源精确，`compact_quality` 仅按来源子组聚合，缺少逐源字段不会填成零。当前 `crowded_blend/dog_only` 为 `509/902、196/902、6/902`，目标 `82934` 所在 `gaussian_only` 为 `0/2` 达到候选持久线，目标自身为 `4/15、3/15、0/15`；`82931` 自身为 `4/15、4/15、0/15`。这些是注册邻域 detector-level 计数，不是星表身份、噪点概率或恒星数。产物为 `tmp/source-proposal-temporal-audit-code-pattern-current-v1/`，实现为 `src/rst19/source_proposal_temporal_audit.py`，测试为 `tests/test_source_proposal_temporal_audit.py`。

如需量化不同特征类别的“规则签名”而不是把它们混成一个 SNR 分数，可运行：

```powershell
rst19-feature-effect-size `
  tmp/source-quality-audit-code-pattern-current-v3/source_catalog.csv `
  --reference-class compact_quality `
  --out-dir tmp/feature-effect-size-code-pattern-current-v1
```

该命令对 `filter_snr`、flux SNR、峰值、FWHM、椭圆率、sharpness、足迹、PSF 支持、质心偏移和值域计数输出非参数 AUC、Cliff's delta 和稳健中位差。AUC 固定表示参考类数值大于比较类的概率；它只审计当前 detector-rule signature，不能解释为分类器精度、precision、FDR 或恒星概率。当前结果显示线状候选可以很亮但 FWHM/sharpness 不像点源，尖峰类的二维支持不足，范围异常应先查值域；产物为 `tmp/feature-effect-size-code-pattern-current-v1/`。

若要单独检查线状候选是否在探测器坐标上形成细长集合，可运行：

```powershell
rst19-source-geometry-audit `
  tmp/source-quality-audit-code-pattern-current-v3/source_catalog.csv `
  --target-class linear_artifact `
  --control-class compact_quality `
  --trials 5000 `
  --seed 1909 `
  --out-dir tmp/source-geometry-audit-code-pattern-current-v1
```

该审计只读取源表坐标，用 PCA 轴比和垂直残差对线状类与等样本量点源类做可重复对照；它支持“线状类需要独立线/轨迹分支”的工程判断，但不证明运动目标、固定条带或伪影身份，也不是 p 值、FDR 或恒星概率。当前结果为 `linear_artifact` 轴比 `40.07`，`compact_quality` 对照最大 `1.51`；详细解释见 [`亮点真实性与伪影判别研究`](doc/02-星图识别/亮点真实性与伪影判别研究.md) 10.106。

若要把原始 FITS 抽样中的局部剖面和值域证据按首要类别汇总，可运行：

```powershell
rst19-feature-raw-evidence-audit `
  tmp/source-quality-audit-code-pattern-current-v3/source_catalog.csv `
  tmp/raw-source-diagnostics-stratified-v2/stratified_raw_source_summary.csv `
  --out-dir tmp/feature-raw-evidence-audit-code-pattern-current-v2
```

该命令只连接已经生成的 CSV，不重读 FITS；它输出核心能量占比、局部噪声、零值/负值/重复码和 15 帧支持的类别分布，并逐字段标注与类别规则的重叠。当前样本中，尖峰类核心占比高但二维支持低，拥挤类局部噪声高且通量 SNR 持久弱，范围异常类重复码持续；这些是机制分流证据，不是独立 precision、FDR 或恒星概率。必须使用当前 v3 源表；旧 GUI 源表会把 8 个范围异常样本误归为紧凑类。详细解释见 [`检测器实验记录`](doc/02-星图识别/检测器实验记录.md) 8.121。

若要进一步审计各特征类在 15 帧原始固定坐标孔径中的响应波动，可运行：

```powershell
rst19-feature-temporal-consistency-audit `
  tmp/source-quality-audit-code-pattern-current-v3/source_catalog.csv `
  tmp/raw-source-diagnostics-stratified-v2/stratified_raw_source_frame_metrics.csv `
  --out-dir tmp/feature-temporal-consistency-audit-code-pattern-current-v2
```

它输出逐源/逐类别的覆盖率、稳健相对 MAD、正值比例、`flux SNR≥5` 比例、符号翻转和重复码/异常值比例。默认从 `source_catalog.csv` 取类别，并在逐帧表带有不同类别时拒绝静默连接；pair 专用表的类别属于其本地产物，必须显式限定 ID 和来源：

```powershell
rst19-feature-temporal-consistency-audit `
  tmp/source-quality-audit-code-pattern-current-v3/source_catalog.csv `
  tmp/raw-source-diagnostics-v1/raw_source_frame_metrics.csv `
  --feature-class-source frame `
  --detection-id 82931 `
  --detection-id 82934 `
  --out-dir tmp/feature-temporal-consistency-pair-82931-82934-current-v2
```

pair 结果中的 `15/15` 是固定孔径强制响应，不等价于注册候选 `15/15` 帧出现；重复码、负异常和共享孔径仍必须单独解释。该审计只读 CSV，不改默认 detector、质量层、GUI 或缓存。

若要把 raw 局部峰、二维支持、值域和逐帧波动放在同一张类别交叉表中，可运行：

```powershell
rst19-feature-cross-axis-audit `
  tmp/source-quality-audit-code-pattern-current-v3/source_catalog.csv `
  tmp/raw-source-diagnostics-stratified-v2/stratified_raw_source_summary.csv `
  tmp/source-peak-consistency-code-pattern-current-v1/source_peak_consistency.csv `
  tmp/feature-temporal-consistency-audit-code-pattern-current-v2/feature_temporal_consistency_sources.csv `
  --out-dir tmp/feature-cross-axis-audit-code-pattern-current-v2
```

该命令要求四张表的抽样 `detection_id` 严格对齐，并额外校验 peak 表的 `feature_class`/`quality_passed` 与 canonical 源表一致；输出逐源证据组合和类别汇总。例如 raw 局部峰为真不代表有二维支持，低波动也不代表值域干净。它是规则重叠/证据冲突审计，不输出新的质量分数或恒星概率，也不修改默认检测与 GUI。旧 v1 因源表版本不一致已撤回。

若要检查这些源级字段是否在重复表达同一响应，可运行：

```powershell
rst19-feature-correlation `
  tmp/source-quality-audit-code-pattern-current-v3/source_catalog.csv `
  --out-dir tmp/feature-correlation-audit-code-pattern-current-v1
```

该命令输出字段两两 Spearman 相关性和按类别的中位数；相关性只用于识别重复显著性量与机制分流，不是因果关系、真星概率或新的质量门槛。

若要检查“同一个双源在不同污染结构中是否仍能被拆开”，可运行真实局部背景注入 pilot：

```powershell
rst19-contaminated-pair-injection `
  doc/00-项目资料/原始数据/20260330163205413_9901.fits `
  --anchor target_pair,2437,4021.5,4,-1 `
  --anchor compact_quality_22381,1525.8136,1226.8184,1,0 `
  --anchor linear_artifact_4469,1522.9654,264.9937,1,0 `
  --separations-px 2.738 4.123 --ratios 0.143 1 `
  --total-peak-adu 4096 --out-dir tmp/contaminated-pair-injection-code-pattern-current-v1
```

该命令把 baseline、injected 和相对 baseline 的 new 命中分开；当前 pilot 显示 `2.738 px` 在紧凑/线状污染结构中不能稳定形成两个质量源，而 `4.123 px` 才可能分开。它是机制控制，不是当前 FITS 的 precision、伪影率或物理恒星数；产物为 `tmp/contaminated-pair-injection-code-pattern-current-v1/`，不改默认检测、质量层、GUI 或缓存。

若要把该控制扩展到各类特征候选，可运行每类一个中位 `filter_snr` 锚点的复核：

```powershell
rst19-contaminated-pair-class `
  doc/00-项目资料/原始数据/20260330163205413_9901.fits `
  --source-catalog tmp/source-quality-audit-code-pattern-current-v3/source_catalog.csv `
  --separations-px 2.738 4.123 --ratios 0.143 1 `
  --total-peak-adu 4096 --out-dir tmp/contaminated-pair-class-audit-code-pattern-current-v1
```

该命令从 8 个互斥首要特征类别各选一个中位锚点，并额外写出类别选择表、逐条件注入表和 `contaminated_pair_class_summary.csv`。当前 pilot 在 `2.738 px` 下候选/质量双命中为 `0/16`、`0/16`，在 `4.123 px` 下为 `11/16`、`9/16`；线状与范围异常类仍为 `0/4`。它用于区分“间距相对 PSF 的几何可解析性”和“局部结构对质量门的影响”，不是类别召回率、precision、FDR 或物理恒星数。

若要扩展到每类 3 个锚点，使用局部 ROI 研究口径，避免对每个条件重复处理整张 4096×4096 图像：

```powershell
rst19-contaminated-pair-class `
  doc/00-项目资料/原始数据/20260330163205413_9901.fits `
  --source-catalog tmp/source-quality-audit-code-pattern-current-v3/source_catalog.csv `
  --per-class 3 --analysis-scope local_roi --roi-half-size-px 192 `
  --separations-px 2.738 4.123 --ratios 0.143 1 `
  --total-peak-adu 4096 `
  --out-dir tmp/contaminated-pair-class-audit-code-pattern-current-v2-local-per3
```

该口径会完成 `96` 个局部条件，并在结果中标明 `analysis_scope=local_roi`、ROI 边界和局部计数；当前结果为短距候选/质量 `0/48、0/48`，长距 `33/48、28/48`。它只能用于局部机制回收，不应与全图候选总数或全图 FDR 混合。

当前还会在逐条件结果中保存注入端点的检测 ID、匹配距离和质量原因，并额外写出 `contaminated_pair_quality_reason_summary.csv`。它可以区分 `NO_CANDIDATE`、`UNRESOLVED_BLEND`、`LINE_ARTIFACT`、`NEGATIVE_OVERFLOW` 和 `QUALITY_PASS`，用于解释失败机制，不改变默认检测路径。

把特征类别注入从一次实验扩展为多次随机背景重复，分开统计已知源回收和阴性结构泄漏：

```powershell
rst19-feature-audit-replicates `
  --trials 64 --seed 19019 `
  --out-dir tmp/feature-audit-replicates-code-pattern-current-v2
```

该命令复用同一检测参数，在不同随机背景上重复孤立/弱源/宽窄 PSF/双源/尖峰/掩膜/边缘/饱和/长线场景；正样本输出候选与质量回收，阴性样本只输出邻域候选/质量泄漏，不伪造 truth 分母。当前 `64` 次是扩展版，首轮 `16` 次保留为历史 pilot；结果仍显示弱、窄、边缘、掩膜和饱和场景可以被宽筛找到但质量层拒绝，3 px 双源只部分一对一解析，长线阴性有明显候选泄漏；这用于校准分流顺序，不是当前 FITS 的真实 precision、完备率或恒星概率。

真实首帧的符号反相阴性对照：

```powershell
rst19-signed-null-audit `
  doc/00-项目资料/原始数据/20260330163205413_9901.fits `
  --threshold-sigma 4 --min-distance 4 --aperture-radius 4 `
  --psf-fwhm 2 --proposal-mode hybrid --min-flux-snr 5 `
  --min-psf-support-pixels 3 `
  --out-dir tmp/signed-null-audit-code-pattern-current-v1
```

它在稳健背景 `B` 上构造 `I_mirror=2B-I`，保持正向/反相检测器参数一致，并输出正负候选、质量层、SNR 分档和特征类别对照。当前首帧原始结果为 `84,594/29,260` 与 `2,309/1`（候选/质量），但反相唯一质量源对应原图极端负码的值域转移；因此 `2.7295%` 只能称 signed-tail leakage 诊断，不能称 FDR、p 值、precision 或真实伪影率。命令只写 `signed_null_summary.json`、`signed_null_thresholds.csv` 和 `signed_null_feature_counts.csv`，不写源目录、不修改 GUI 默认值或缓存。

15 帧聚合对照可用 `rst19-signed-null-sequence`：pooled 反相候选/质量为 `36,170/35`，其中 `19` 个质量源落在原始极端负码扩张区，排除后剩 `16` 个；逐源回查显示 `13/16` 仍含原始负值异常，另有 `3/16` 为无负值证据的弱/宽响应。按质心 `≤1 px` 跨帧连接得到 `6` 个复现簇，输出源级和复现簇 CSV；v3 源表还提供 `raw_evidence_layer`，实际分布为 `19/13/3`。该分层是值域/候选诊断，不是恒星总数、伪影率或物理身份。详见 [`检测器实验记录 8.72/8.73`](./doc/02-星图识别/检测器实验记录.md)。
若要继续检查反相质量源在同一帧原始 detector 坐标附近是否有正向对应，可运行：

~~~powershell
rst19-signed-null-overlap doc/00-项目资料/原始数据 --reverse-source-csv tmp/signed-null-sequence-code-pattern-current-v3/signed_null_sequence_quality_sources.csv --threshold-sigma 4 --min-distance 4 --aperture-radius 4 --psf-fwhm 2 --proposal-mode hybrid --min-flux-snr 5 --min-psf-support-pixels 3 --out-dir tmp/signed-null-positive-overlap-code-pattern-current-v2
~~~

该审计输出 35 行反相源交叉表；当前正向候选在 ≤1/2/4 px 内为 0/35、23/35、32/35，正向质量源 ≤4 px 为 0/35。这是 detector-level 局部近邻控制，不是星表身份匹配、precision 或真星率。

6. **比赛材料**：沉淀算法流程图、关键参数、实验对照、误差分析、创新数据和演示结果。

## 本地使用

克隆仓库并安装实现模块：

```powershell
git clone https://github.com/lycohana/rst19-sol.git
Set-Location rst19-sol
python -m pip install -e ".[dev]"
```

只检测一帧并输出 JSON：

```powershell
rst19 doc/00-项目资料/原始数据/20260330163205413_9901.fits `
  --threshold-sigma 4 --min-distance 4 `
  --psf-fwhm 3 --min-flux-snr 5 --min-psf-support-pixels 3 `
  --json-out output/frame-01.json
```

不提供 `--max-sources` 时会保留全部经过匹配滤波和非极大值抑制的候选源；只有在明确需要限制内存或导出规模时才传入正整数。输出同时包含候选数、质量通过数、局部背景/RMS、峰值/通量/滤波 SNR、通量误差和形状标志。需要使用已有零点时，可增加 `--zero-point <数值>`，否则输出中的星等字段只表示仪器星等。

分析 15 帧时使用：

```powershell
rst19-sequence doc/00-项目资料/原始数据 `
  --threshold-sigma 4 --min-distance 4 --psf-fwhm 3 `
  --min-flux-snr 5 --min-psf-support-pixels 3 `
  --min-presence 12 --motion-min-displacement-px 2 `
  --max-motion-fit-rms-px 0.75 `
  --json-out tmp/sequence-final.json
```

需要检查星点在 15 帧中是否真的保持点状时，可追加原图时序审计：

```powershell
rst19-sequence doc/00-项目资料/原始数据 `
  --source-limit 0 --workers 4 `
  --threshold-sigma 4 --min-distance 4 --aperture-radius 4 `
  --psf-fwhm 3 --min-flux-snr 5 --min-psf-support-pixels 3 `
  --json-out tmp/sequence-current-psf-support-full.json `
  --audit-out-dir tmp/sequence-source-audit-psf-support
```

审计输出 `source_track_contact_sheet.png`、`source_track_audit.csv` 和 `source_track_audit_summary.json`，把严格静态、持续候选、1 帧瞬态和低持续性瞬态分层展示。严格静态默认要求约 `12/15` 帧，持续候选默认要求约 `8/15` 帧；二者不能与单帧 `quality_count` 混写，也都不是官方逐星真值。GUI 的“15 帧证据”窗口中提供同一功能的“星点时序审计”按钮。

当前单帧质量层另有一条防尖峰约束：候选峰中心 3×3 核心至少 3 个像素需超过 `max(2×局部噪声, 0.1×峰值超额)`，否则保留在候选审计层并标记 `INSUFFICIENT_PSF_SUPPORT`，不进入可信星点。该规则不改变原始 ADU、孔径通量或星等计算；它只防止“一个很高的单像素 + 一圈随机正噪声”通过旧的 1σ footprint。当前快速序列基线还对时间补提案在预测点 `±1 px` 内做局部峰重定位，再执行 PSF 相关和 15 帧审计；真实 15 帧 `Gaussian / median / 参考 SNR 15 / PSF 相关 0.80` 复测得到严格静态 `5,276`、持续候选 `15,524`、稳定/持续合计 `20,800`。早期 `9,938 / 6,249 / 16,187` 是另一套全量工作集和旧时序口径的历史结果，不能与当前数字混写；所有数量仍不是官方逐星真值。

15 帧默认采用“全量候选审计 + 有界配准工作集”：每帧仍完整记录匹配滤波后的 `candidate_count`，但只对按匹配滤波 SNR 排名前 6000 的源做源级测量和点轨迹关联；序列使用 `background_box_size=256`，先由首帧建立共享局部背景/RMS 图，后续同尺寸帧复用它，避免重复执行 256 个局部网格的 sigma-clipping。每帧仍独立计算全局背景、坏点/饱和掩膜和 PSF 检测；尺寸不一致时自动回退到逐帧背景模型。当前数据为 16 位 FITS，序列路径另外启用精确表示输入 ADU 的 `float32` 中间阵列，降低 4096² 像素计算的内存带宽；单图 `rst19`/“分析当前帧”仍默认全量、精细测光并使用 float64。序列快速路径对每个背景块最多抽样 4,096 个像素，并在无效像素稀疏时使用单遍卷积；线性伪迹审计、通量 SNR、点源形状、边缘和饱和检查仍保留，结果中记录 `background_sample_limit`、`background_model_mode`、`fast_sequence` 和实际滤波口径。当前默认由 4 个 worker 并行检测不同帧，输出仍按帧号排序；内存紧张时传 `--workers 1/2/3`，也可以传 `--workers 4` 作为当前机器的吞吐档位。`returned_count`、`quality_count` 和点轨迹数量因此是工作集口径，不是候选总数；需要进行全量 15 帧配准时显式传 `--source-limit 0`，代价是明显更慢；也可用 `--source-limit 12000` 在召回率和耗时之间做对照。

序列性能口径：默认按每帧前 6000 个高 SNR 源做配准工作集，但完整记录候选峰总数；当前默认 4 个 worker。序列路径使用两轮网格稳健裁剪、每块最多 4,096 点的确定性背景抽样、直接目标尺寸的背景插值、稀疏掩膜单遍卷积、缓存孔径几何，以及整数 FITS 的精确 `float32` 中间阵列，减少重复的 4096² 临时数组分配。时间补提案默认使用单一基准 PSF；可选 `--temporal-multiscale` 做窄/基准/宽尺度召回实验，但它会明显增加纹理候选，只能经同一套原图细筛后审计。单图仍保持全量候选、四轮背景裁剪、归一化卷积和环形局部精测；`--workers 1/2/3/4` 可按内存调节 15 帧吞吐，`--float64` 可做精度对照。

### GUI 当前默认分析口径（2026-09-02）

单张和 15 张入口共用下面这套默认参数；参数区仍允许用户显式切换到 Gaussian 基线、Starlet 实验、局部去混叠或 15 帧全量关联。这里的“最优”是基于当前真实 FITS 抽测得到的召回/稳定性/耗时折中，不是把结果数量调到某个预设数字，也不等同于官方逐星真值。

| 范围 | 当前默认值 | 作用 |
| --- | --- | --- |
| 单帧宽筛 | `hybrid` = Gaussian + 双尺度 DoG | 先扩大拥挤/失配候选，再回到原图统一细筛 |
| PSF 尺度 | `FWHM=2 px` | 接近首帧亮且孤立质量源的实测中位 FWHM，减少固定 `3 px` 失配 |
| 候选与质量 | `threshold=4σ`、`min_distance=4 px`、通量 `SNR≥5` | 保留高召回候选，同时用通量、形态、PSF 支持和伪影 flags 分层 |
| 15 帧弱星提案 | `median` | 比 `both` 更快且当前实测稳定/持续输出更高；只作提案，仍逐帧回原图复核 |
| 15 帧参考门槛 | `15σ`；逐帧补测 `7.5σ`；普通候选共识 `15σ` | 不把降低参考门槛后的弱纹理直接当作正式星点 |
| 时序细筛 | PSF 相关 `≥0.80`、预测点 `±1 px` 局部峰重定位 | 修正注册取整误差，不扩大到远邻搜索 |
| 默认性能开关 | 时序多尺度关闭、局部去混叠关闭、15 帧全量关联关闭 | 避免分钟级逐候选拟合和多尺度纹理膨胀；需要时手动开启审计 |

当前同一批真实 15 帧的对照为：Gaussian + `FWHM=2` 得到 `4,998` 条严格静态、`25,137` 条持续候选、合计 `30,135`；hybrid + `FWHM=2` 得到 `4,439`、`25,858`、合计 `30,297`。hybrid 的单帧质量层也比同 FWHM 的 Gaussian 多 `302` 个，但严格静态少 `559` 条，说明它是偏召回的平衡默认，而不是“所有输出都更可信”。因此界面必须继续区分严格静态、持续候选、候选总数和质量源，不能把 `30,297` 直接写成恒星总数。局部去混叠在一次 `FWHM=3` 控制实验中仅增加 `255` 个质量源，却额外产生约 `10,817` 个候选并显著拖慢运行，故不随默认 hybrid 自动开启；Starlet、时序多尺度、`both` 和参考 `12σ` 继续保留为研究/召回审计模式。

本机同一真实 15 帧、`min_distance=3`、工作集 6000、4 worker 的最新复测约 `43.8 s`；结果记录 `background_model_mode=shared_sequence_pilot`。该数字是当前机器的工程实测，不是准确率或跨设备承诺；需要逐帧背景对照时应使用非快速口径并单独记录。

根据序列 JSON 生成逐帧和逐轨迹的创新分析证据表及 PNG 图表：

```powershell
rst19-innovation tmp/sequence-final.json --out-dir tmp/innovation
```

证据报告会把单帧长线候选与跨帧 `moving` 轨迹分开，并使用 `DATE-OBS` 的真实时间计算 pixel/s；三点及以上轨迹还输出基于小样本 OLS 的速度/方向 95% 区间和外推逐轴区间，两点轨迹不伪造区间，单帧候选不伪造速度。没有像元角尺度和标准 WCS 时不会输出角秒或真实天体速度。通过星表窗口的唯一匹配和局部仿射校准后，创新摘要才会额外显示视场切平面 `arcsec/s`，并同时保留匹配内点数、像素残差和各向异性；这不等同于完整 WCS 或轨道速度。辅助遥测页还会以位置 `m` 和 `DATE-OBS` 检查速度字段的 `m/s?` 自洽性，并给出末帧状态向后 5 个中位帧间隔的常速度位置外推；该预测会标注单位假设，不把它与图像平面速度混用。单帧“分析当前帧”也会运行同一套长线几何筛选，橙色线只表示待复核候选。GUI 的“15 帧证据”窗口提供逐帧表、线状轨迹表、逐帧“线状诊断”筛选表/曲线、集中展示时序/配准/速度/方向/预测及统计边界的“创新摘要”页、Canvas 图表、辅助遥测表、“遥测轨迹”相对首帧 XY 投影图、按需生成的注册中值/正负残差图证和线状裁剪接触表、注入-回收曲线、真实首帧阈值扫描，以及按需运行的“单图长线”逐帧审计页。遥测轨迹图将观测序列与琥珀色外推段分开显示，仍保留 `m?`/`m/s?` 假设边界。

逐张审计单图长线候选（不跨帧、不把单帧形状写成运动真值）：

```powershell
rst19-trails doc/00-项目资料/原始数据 --out-dir tmp/single-frame-trails
```

星表核验窗口在当前帧匹配后还提供“验证 15 帧 WCS”：逐帧执行先验 WCS 下的全局一对一匹配和局部二维仿射拟合，显示每帧的匹配数、内点数、留一 RMS、像元角尺度和残差曲线，并导出 `wcs_frame_validation.csv`、`wcs_validation_report.json` 及 PNG。星表可以是用户提供的获授权离线 CSV，也可以通过窗口内的 **在线获取 Gaia DR3** 显式生成；两种方式都仍需要先验 WCS，`validated` 不是盲解算或官方逐星真值。当前帧还可点击 **自动板解 + 测光**，执行星对几何候选、仿射细化和 Gaia G 经验光度拟合；只有状态为 `CALIBRATED` 时才提升主卡片的表观星等显示。

输出 `single_frame_trails.csv`、候选数量曲线和最长线长度曲线；其中 `trail_count`、长度、宽度、方向、残差 SNR、触边状态和包围盒可以与 15 帧关联结果并列复核。

对已经完成检测的单帧输出全量源级研究表（不重新解释为真实恒星数）：

```powershell
rst19-sources doc/00-项目资料/原始数据/20260330163205413_9901.fits --out-dir tmp/source-audit
```

输出 `source_catalog.csv`、`source_quality_summary.csv`、`source_feature_summary.csv`、`source_feature_method_summary.csv`、`source_feature_morphology.csv`、`source_feature_subclass_summary.csv`、`source_feature_spatial.csv`、`source_feature_psf_similarity.csv`、`source_feature_psf_spatial.csv`、SNR 排名曲线和通量 SNR 分布图；方法来源表按类别记录 Gaussian/DoG 提议器的交集、`gaussian_only`、`dog_only`、部分组合和无 Gaussian 质量数，用于区分算法敏感性与物理可信度。形态表按类别保存 peak/flux/filter SNR、FWHM、椭圆率、sharpness、PSF 支持、质心偏移、空间分位数和拒绝原因比例，子类表按完整 flag token 分开边界、部分掩膜、重复码、极端负值、饱和、线状、未分辨近邻和尖峰，但明确允许重叠，空间表保存类别在 4×4 粗网格中的候选/质量数和高 SNR 拒绝数，PSF 表保存原始抽样裁剪与经验核的相关系数、相对残差和中心能量占比，空间 PSF 表进一步比较全局核与留出局部模板，并报告模板不可用的回退数，便于解释候选峰、质量源和被拒绝源之间的数量差异。高 SNR 拒绝计数、空间集中性、方法交集、PSF 相关系数和局部模板可得率都只是诊断切片，不是物理伪影真值。

在独立的合成注入实验中，可以运行：

```powershell
rst19-injection --out-dir tmp/injection --trials 4 --sources 24
```

这项实验只用于测量当前检测器在简化 Gaussian PSF/背景模型下的召回趋势，不是 15 张真实图像的星点真值。运行参数、随机种子和结果表见 [`创新指标候选`](./doc/04-创新分析/创新指标候选.md)。

在真实首帧背景上运行注入-回收实验：

rst19-injection --real-fits doc/00-项目资料/原始数据/20260330163205413_9901.fits --out-dir tmp/real-background-final --trials 2 --sources 16 --psf-model gaussian

真实背景实验只在内存副本中注入 Gaussian PSF，输出候选层/质量层召回率、基线数量、注入区域外背景检测数和净数量变化。它比均匀合成背景更接近当前数据，但没有逐星真值时不能从背景检测数推出 precision 或误检率。

将 psf-model 改为 empirical 可从首帧完整候选表中筛选质量通过且隔离的亮源，提取实测 PSF，再在相同真实背景上注入；样本不足时程序会明确报错，不会悄悄把实测 PSF 当成 Gaussian。经验核是模型敏感性审计，不等于星表真值。

进一步按真实背景条件分层做注入审计：

```powershell
rst19-injection --real-fits doc/00-项目资料/原始数据/20260330163205413_9901.fits `
  --stratified-real `
  --peak-levels 24 56 128 --trials 1 --sources 8 --psf-fwhm 2 `
  --proposal-mode hybrid --psf-model empirical `
  --strata blank high_background edge crowded special_code line `
  --out-dir tmp/stratified-real-background-code-pattern-current-n8
```

该模式分别测试候选稀疏的低/高局部噪声背景、边缘截断、亮源邻域和特殊值域邻域，示例每格使用 8 个位置，输出 `stratified_real_background_injection.csv/json/png`。`crowded` 与 `special_code` 会保留基线邻源并记录 `ambiguous_injection_count`，只能解释为困难条件下的可检出性，不是孤立星完备率；另可将 `--strata line` 单独运行以审计亮线邻域。分层实验仍不能替代官方星表、WCS、实测 PSF 留出验证或人工真值。

若先要区分“空间区域失效”和“注入信号不足”，可运行 2×2 空间分区留出注入：

```powershell
rst19-spatial-injection `
  doc/00-项目资料/原始数据/20260330163205413_9901.fits `
  --grid-size 2 --peak-levels 256 512 1024 `
  --trials 1 --sources-per-cell 1 --psf-fwhm 2 `
  --proposal-mode hybrid --psf-model gaussian `
  --signal-normalization integrated_excess `
  --out-dir tmp/spatial-injection-code-pattern-current-2x2-gaussian-t1-high-paired-v2
```

当前首帧 pilot 的候选回收在三档均为 `4/4`，质量回收为 `0/4、4/4、4/4`；`256 ADU` 的四个注入源 `flux SNR=2.35--3.57`，而 `512 ADU` 升至 `5.89--7.52`。同 dtype 无注入配对控制显示，三个单元的注入候选增量约为 `+1`，另一个单元虽增加 `+16`，质量却只增加 `+1`；新增远处项集中在 `filter_snr≈4`，伴随全图响应噪声约 `0.045%` 的重估变化。因此这项 pilot 支持“质量层跃迁首先受通量 SNR/PSF 支持控制”，同时揭示宽筛的全图归一化耦合，不能把候选膨胀当成空间特殊星点。每档只有 4 个注入源且使用统一 Gaussian 核，结果不是空间完备率、precision 或真实恒星数；详细记录见 [`检测器实验记录`](./doc/02-星图识别/检测器实验记录.md) 8.79，低信号 `56/128 ADU` 对照位于 `tmp/spatial-injection-code-pattern-current-2x2-gaussian-t1/`，配对版产物位于 `tmp/spatial-injection-code-pattern-current-2x2-gaussian-t1-high-paired-v2/`。

针对经验 PSF 与孔径口径的耦合，可复现实验：

```powershell
rst19-aperture-sensitivity `
  doc/00-项目资料/原始数据/20260330163205413_9901.fits `
  --aperture-radii 3 4 5 6 --reference-aperture-radius 4 `
  --control-signal-adu 512 --grid-size 2 --psf-fwhm 2 `
  --background-box-size 128 --min-flux-snr 5 `
  --min-psf-support-pixels 3 --min-distance 4 `
  --threshold-sigma 4 --proposal-mode hybrid `
  --empirical-psf-radius 7 --empirical-psf-sources 64 `
  --seed 19019 --out-dir tmp/aperture-sensitivity-code-pattern-current-v1
```

首帧四档候选回收均为 `4/4`，质量回收为 `4/4、2/4、0/4、0/4`；四点 `flux SNR` 中位数为 `6.428、5.168、3.940、3.644`。该结果只说明当前经验模板下的测光边界，不能外推为真实完备率，也不直接改变 GUI 默认参数；产物位于 `tmp/aperture-sensitivity-code-pattern-current-v1/`，详细解释见 [`检测器实验记录`](./doc/02-星图识别/检测器实验记录.md) 8.80。

针对截图式近邻双峰，固定总峰值、只扫描弱源相对强度：

```powershell
rst19-injection --pair-flux-ratio-audit `
  --real-fits doc/00-项目资料/原始数据/20260330163205413_9901.fits `
  --total-peak-levels 256 --secondary-ratios 0.5 0.25 0.125 `
  --trials 1 --pairs 8 --psf-fwhm 2 --min-distance 4 `
  --min-psf-support-pixels 3 `
  --out-dir tmp/pair-flux-ratio-256-gui-default-n8
```

该实验把单源召回与成对解析率分开：一对真值必须由两个不同候选分别命中才算解析，不能把一个中间峰重复计数。首帧真实背景结果为：`r=0.5` 的候选/质量成对解析率为 `1/0.625`，`r=0.25` 为 `1/0`，`r=0.125` 为 `0.375/0`。这说明弱源先失去质量层证据，再失去宽筛候选证据；它用于解释“双框”现象，不用于估计真实双星比例。完整字段和限制见 [`检测器实验记录`](./doc/02-星图识别/检测器实验记录.md) 的 8.31 节。

多级强度与经验 PSF 对照进一步显示：Gaussian `r=0.125` 的成对质量解析率在总峰值 `256/512/1024 ADU` 下为 `0/0.25/1.0`；经验核在 `512 ADU`、`r=0.25/0.125` 下为 `0.25/0`。后者仍是 detector `FWHM=2 px` 与实测核中位 `2.497 px` 的形状敏感性控制，不是两种 PSF 的最终排名。结果见 [`检测器实验记录`](./doc/02-星图识别/检测器实验记录.md) 8.32。

为避免不同 PSF 的离散核和把峰值实验混杂，可用 `--pair-signal-normalization integrated_excess` 固定注入小窗积分超额。相同真实背景、同一布局、`2 trials × 8 对`、总积分超额 `1867 ADU` 的对照中，Gaussian/经验 PSF 的成对质量解析率在 `r=0.25` 为 `0.812/0.625`，在 `r=0.125` 为 `0.188/0.125`；候选层在弱比 `r=0.125` 反而为 `0.688/0.812`。这说明 PSF 形状的影响主要体现在质量确认，且候选层与质量层不能混为一个分数；结果和复现命令见 [`检测器实验记录`](./doc/02-星图识别/检测器实验记录.md) 8.33。

拥挤场再用真实首帧做 2/3 源等间距组审计，并扩展到每条件 `2 trials × 8` 组：`3 px` 时候选组级解析率仍为 `0`、合并率为 `0.75--1.0`；`5.4/6 px` 时候选可分开，但在总积分 `512 ADU` 下质量级完整解析为 `0/16`。总积分提高到 `1024 ADU` 后，2 源 `5.4/6 px` 的质量解析为 `15/16`、`14/16`，3 源仅 `1/16`、`0/16`。这里的 `group_resolution`、`merged`、`extra` 是三种不同诊断，均不等于 precision；它说明“看到多个峰”不能直接换算成多个独立恒星。完整多源表和限制见 [`检测器实验记录`](./doc/02-星图识别/检测器实验记录.md) 8.35。

对截图的三个候选位置还可以做固定位置 `K=1/2/3` 多源 PSF 审计：

```powershell
python -m rst19.multipsf_cli `
  doc/00-项目资料/原始数据 `
  --positions "1269,3465;1274,3467;1272,3453" `
  --psf-fwhm 2 `
  --out-dir tmp/local-multipsf-audit-screenshot
```

这项研究工具在同一局部窗口和背景平面下比较 `K=1/2/3`，同时输出 raw、范围码屏蔽和精确 `-1` 屏蔽三种口径。raw 15 帧中只有 F07/F12 的 `K=2` BIC 更优，但新增分量 SNR 约为 `4.38/4.00`，没有一帧同时达到 `ΔBIC≥10`、`SNR≥5`；`K=3` 没有越线。范围码屏蔽后仅 F07 形式上越线，但该帧仍为 `MASKED|SATURATED`，所以不能把两个局部峰直接写成两颗恒星。完整解释和逐帧表见 [`检测器实验记录`](./doc/02-星图识别/检测器实验记录.md) 8.37 与 [`亮点真实性与伪影判别研究`](./doc/02-星图识别/亮点真实性与伪影判别研究.md) 10.23。

为检查固定中心误差和自由度过拟合，再运行有界自由位置及半径敏感性审计：

```powershell
python -m rst19.free_multipsf_cli `
  doc/00-项目资料/原始数据 `
  --radius-sweep --mask-modes raw `
  --position-radii 0.25 0.5 0.75 1.0 1.25 `
  --positions "1269,3465;1274,3467;1272,3453" `
  --psf-fwhm 2 `
  --out-dir tmp/local-free-multipsf-radius-sweep-screenshot
```

raw `K=2` 在上述五个半径下的确认数为 `0/15、0/15、0/15、1/15、1/15`，固定位置为 `0/15`。F07 只有允许约 `1 px` 中心移动后才出现内部双源拟合，因此该结果用于暴露模型自由度敏感性，不作为双星比例或 GUI 默认计数。固定位置逐帧结果见 [`检测器实验记录`](./doc/02-星图识别/检测器实验记录.md) 8.37，自由位置解释见 [`亮点真实性与伪影判别研究`](./doc/02-星图识别/亮点真实性与伪影判别研究.md) 10.24。

固定候选阈值、扫描匹配滤波 PSF 尺度：

```powershell
rst19-sweep doc/00-项目资料/原始数据/20260330163205413_9901.fits `
  --psf-fwhms 1.5 2 2.5 3 4 --thresholds 4 --min-distance 4 `
  --psf-fwhm 3 --background-box-size 128 --min-flux-snr 5 `
  --out-dir tmp/detection-psf-sweep
```

该扫描只改变匹配滤波器的 FWHM，不改变 FITS 原图，也不把数量变化直接解释为真实恒星数；实测 PSF、注入回收和人工抽检应联合决定最终参数。GUI 的“阈值扫描”页提供同一入口。

模块设计、参数边界和当前限制见 [`src/README.md`](./src/README.md) 与 [`src/DESIGN.md`](./src/DESIGN.md)。

运行单元测试（测试文件位于 [`tests/`](./tests/)）：

```powershell
python -m pytest
```

启动 Python 桌面界面：

```powershell
rst19-gui
```

也可以使用同一个 Python/Tkinter 入口：

```powershell
rst19-ui
```

面向非开发用户的按钮、图层、缓存和“稳定星场/叠加暗星”解释见 [`doc/软件使用说明.md`](./doc/软件使用说明.md)。

单帧和 15 帧分析时，底部状态栏会显示确定性进度：单帧按读取、局部背景/PSF 匹配、源级测量、逐步长轨迹筛选和预览生成推进；15 帧显示当前 `Fxx/总帧数`，并继续报告共享背景准备、时间中值参考、逐帧残差、线状候选关联和运动轨迹拟合，而不是在“线状筛选”阶段停住。检测参数区保留大进度条，并新增 F01…F15 帧状态带；右侧 `15-FRAME EVIDENCE` 卡同步显示百分比、已完成帧数和当前阶段，底部状态栏保留短状态。进度只由当前任务更新，切换帧或清理缓存后，旧线程不会覆盖当前进度。

界面中的最大源数输入框留空表示单图全量，15 帧此时自动使用每帧前 6000 个匹配滤波高 SNR 源作为配准工作集；候选总数仍完整显示，工作集内的 `returned_count`/质量数/点轨迹不会冒充全量恒星数。15 帧进度同时固定显示在检测参数区和底部状态栏，包含总体百分比、已完成帧数、当前帧号与阶段；过密的后台事件会合并，避免界面落后于实际计算。
界面使用 Tkinter 在本机运行，可选择 15 张本地 FITS、调整候选阈值、最小间距、PSF FWHM、通量 SNR 和显式返回上限；“分析 15 帧动目标”完成后切换到不含静态星场的运动层：洋红线为算法判定的跨帧 `moving` 线状候选，橙色线为单帧/待复核长线，蓝色环为严格 `moving` 点轨迹，青绿色虚线为最后观测点向后 5 帧的图像平面匀速外推。单帧质量层用绿色环标出最暗可信源及其 `m_inst`；长线候选保留在全部候选审计层，但默认不进入质量层。序列结果另提供“稳定星场”层：青绿色环为严格静态轨迹，蓝色环为持续候选，只有跨帧持续点源才进入该层。启动和切换帧只加载预览，不提前算检测；点击“分析当前帧”后先做轻量长线预检，橙线会先出现，后台再完成全量星点/SNR/星等精测；右上角图层选择不会因切帧回到默认项；单帧检测和 15 帧序列都会复用根目录 `.rst19-cache/` 中按文件状态和全部检测/关联参数生成的 gzip 缓存，清空缓存会同时失效两类结果。界面中的源表只展示通量 SNR 排名前 40 行以保持交互流畅，候选数、质量数和审计返回数分别显示；完成单帧检测后可从检测参数区“研究工具”菜单导出星点研究表，得到全量源级 CSV、质量标记汇总、互斥类别摘要、按类别的原图裁剪接触表 `source_cutout_contact_sheet.png`/`source_cutout_manifest.csv`、按类别 4×4 空间分布表 `source_feature_spatial.csv`、重叠质量标志图、16×16 空间密度/通过率图和 `source_spatial_grid.csv`，并在 Tkinter 窗口直接查看论文表图与指标口径。15 帧证据窗口的“图证”页提供“星点时序审计”按钮，后台从原始 FITS 生成严格静态/持续候选/瞬态的接触表和轨迹清单；鼠标滚轮可缩放至最大 `15×`、左键可平移，悬停候选点或运动轨迹会显示对应的测光/轨迹信息。主界面的“研究工具”菜单中的“打开星表核验”允许用户显式选择 CSV 星表并填写先验 WCS，显示唯一匹配、位置残差和目录星等；运行后可点击“根据匹配拟合 WCS”，用至少 6 个有效、非共线匹配点进行局部仿射拟合和 MAD 离群剔除，展示尺度、旋转、parity、内点 RMS、留一残差与各向异性，并可导出校准 JSON。该按钮不会执行全天空盲解算；匹配点不足、几何退化或残差不稳定时保持待标定。星表匹配层用绿色环区分目录预测位置，不把匹配数冒充恒星总数。“15 帧证据”窗口中的“创新摘要”页集中展示序列关系、固定星场配准、图像平面速度/方向及小样本 95% 区间、短期外推坐标及逐轴区间和辅助遥测边界；若当前已有局部 WCS 校准，还会为跨帧线状候选显示视场切平面角速度。“图证”页按需并列显示注册中值拼图、正/负残差差分图、逐帧线状裁剪和含全部质心/OLS/外推区间的图像平面轨迹，“星点实验”页提供理想背景与真实 FITS 背景两种按需注入-回收实验；真实实验显示阶段进度并在后台运行，清空检测缓存不会删除 FITS 原图。

当前主界面另外把“单帧识别”和“15 帧分析”拆成两个工作区：单帧区只解释当前 FITS 的可信源、最暗候选和落选原因；15 帧区才显示 F01–F15 进度、稳定星场、运动候选和通俗简报。简报中的 `moving` 是图像证据候选，不等于星表真值；速度/方向没有 WCS 时只写 `px/s` 和图像坐标角度。最暗源卡片同时显示 `m_inst`、有星表时声明波段的 `m_cal` 和通过视差门控时的 `M`，不会把未标定的 `-5.43` 冒充为物理绝对星等；若拿外部 `13.56` 比较，只显示诊断性零点差，不自动改写测光结果。

界面设计按 [qiaomu-design](https://github.com/joeseesun/qiaomu-design) 的反模板打磨原则适配为纯 Python/Tkinter：纸张色、深靛蓝、琥珀和青绿色构成观测站实验台，采用数据优先的非对称布局和可滚动证据表；不使用 Web UI、紫色渐变、玻璃拟态、Inter 或无语义装饰卡片。qiaomu-design 在这里作为设计读取和 Audit/Polish/Harden 约束，不作为运行时前端依赖。

界面使用 Tkinter 在本机运行。参数区提供候选阈值和可信通量 SNR 的人工调参滑块，拖动不会连续重算，点击“应用并分析”后才运行当前帧；完成后可选择“保留弱星 / 减少伪点 / 当前平衡”，把已实际运行的阈值、计数和拒绝标志记录到 `tmp/manual-threshold-feedback.jsonl`，作为后续离线调参样本而不是在线自动训练。GUI 预览保留当前 4096² 传感器空间分辨率，15× 放大不会把 1600² 缩略图二次放大；显示层用稳健 IQR 噪声估计设置背景底，采用 gamma 1.15 和 0.6 px 轻度高斯去噪，避免把背景抬成整幅灰噪声，同时保留弱源与亮源层次。缩小总览时源标记退化为单像素证据点，放大后才扩成小方框，避免数万候选覆盖原图。上述处理只改变观察预览，不改变原始 ADU、SNR、测光和星等。

由于原始 FITS 数据未随仓库提交，请将获授权的数据文件放置在 `doc/00-项目资料/原始数据/` 或后续配置指定的数据目录；不要把 API Key、访问 Token 或未获授权的原始数据提交到仓库。

## 仓库状态

当前阶段：**星点检测、长轨迹候选、15 帧证据与可复核的星表/光度校准基线**。

- 已完成：根 README、`doc/` 资料索引、数据格式说明归档、比赛规则整理、原始数据边界说明、Python FITS/检测/测光/序列模块、Tkinter 桌面工作台、可清理单帧/序列检测缓存、源级星点研究表与 SNR 图、当前全量 417 项单元测试、格式层机器复核、单帧长线候选、单图长线逐帧审计、15 帧证据窗口中的按需单图长线表和曲线、集中式创新摘要页、逐帧线状筛选诊断表和曲线、长线伪迹对照、注册时间中值差分、首帧实测、真实首帧阈值敏感性扫描、真实首帧 PSF-FWHM 扫描、真实首帧按类别参数敏感性与类别转移矩阵、15 帧序列证据导出、15 帧序列参数比较器、类别内经验分位 hard-negative 审计、类别内高显著性落选 hard-negative 扫描、hard-negative 空间条件化审计、固定码关注坐标逐帧值域导出、严格静态/持续候选分层、原图星点时序接触表、PSF 加权定位质心与候选峰坐标审计、3×3 PSF 支持反尖峰审计、按特征类别的时间剖面和类别条件跨帧转移审计、按真实背景条件分层的注入-回收审计、固定总峰值超额的双源强弱比与经验 PSF 对照、固定离散积分信号的 PSF 公平对照、2×2/每类16个样本的空间 PSF 类别留出诊断、逐源留一法经验 PSF 类别交叉核验、类别三层证据矩阵及确定性复核路由、类别持续率—PSF 形状解耦审计、真实背景 2/3 源拥挤组的合并—解析—质量分层审计及经验 PSF 敏感性控制、经验 PSF 孔径敏感性审计、82931/82934 局部 PSF 适用域扫描、截图局部固定位置 `K=1/2/3` 多源 PSF/BIC 审计、有界自由位置和半径敏感性审计、逐源逐帧分层强制测光及峰/质心锚点敏感性对照、固定—局部匹配响应和重定位残差审计、显式 pair 相对间距收缩审计、82931/82934 独立像素支持留出审计、原始像素拓扑审计、各类特征规则签名效应量、旗标交互审计、质量门余量审计、质量门逐候选路径审计、质量门路径—跨帧响应交叉审计、污染局部背景双源注入审计、跨类别污染双源复核、跨类别 ROI 多锚点复核、端点级质量原因审计、速度/方向/外推的小样本 95% 区间、按需图证导出、理想背景与真实 FITS 背景注入-回收实验、用户提供星表的 UI 核验入口、匹配后局部仿射 WCS 校准、逐帧 WCS 验证命令和 UI 入口、辅助遥测位置—速度自洽检查和末帧常速度位置外推及相对首帧轨迹图、单帧和 15 帧分析进度显示、序列快速背景/卷积路径与右侧实时进度卡、噪声抑制观察预览、可拖动人工阈值和本地反馈记录、64 次重复特征机制注入审计（首轮16次为 pilot）、首帧最近邻间距群体控制、2×2 空间分区留出注入信号边界 pilot、15 帧注册合成大图及其 coverage/scatter/透明预览/缓存导出回归、Gaia 远程查询与大视场分块审计、星对几何 WCS 候选接口和星等证据层级审计、GSP-Phot 模型距离回退和绝对星等来源审计。
- 本轮增量审计：新增几何条件化近邻机制背景命令 `rst19-pair-mechanism-context` 及窗口敏感性命令 `rst19-pair-mechanism-sensitivity`，并完成 `82931/82934` 的可复核窗口对照；它们与宽筛—细筛论文证据链配套，但不改变默认检测或缓存。
- 本轮增量审计：新增只读父源组—子候选审计 `rst19-source-group-audit` 及半径敏感性审计 `rst19-source-group-sensitivity`，把局部峰先组织为可复核源组；当前 v3 的 `84,594` 个候选形成 `84,586` 个组，`8` 个多成员组全部未分辨，目标 `82931/82934` 在 `3 px` 及以上归入同一 `unresolved_group`。该层不修改默认检测、质量层、GUI 或缓存。
- 未完成：官方逐星真值、基于逐星真值的 precision/误检率、空间变实测 PSF 与大样本边缘/拥挤分层完备率、同设备绝对零点/波段响应标定、完全盲解算、仿射/WCS 在全部 15 帧上的留出验证、创新分析结论和正式比赛提交冻结。公共 Gaia 子表接入、相对光度拟合和结果质量审计已实现，但不替代这些实验室/任务级标定证据。

- 本轮增量审计：局部补值反事实在替换半径 1/2/3 px 下均将目标 pair 的候选窗口从 2 个收缩为 1 个（仅修复负异常或联合修复）；类别级门限扫描明确了稳定紧凑类与拥挤、尖峰、弱背景、边缘类的不同敏感性。结果已同步至两版论文和来源账本，不改变 GUI 默认检测或缓存。
- 本轮增量审计：注册坐标与 detector-fixed 零平移 null 的类别对照显示，候选层持久率会随坐标模型显著变化，而质量邻域基本稳定；该结果已写入核心短稿和来源账本，不能把“稳定星场”直接当作 WCS 真值清单。
## 稳定星场显示更新

2026-08-31：15 帧分析完成后，主图默认进入“稳定星场”层，直接显示当前帧实际存在的 `static/persistent` 点源；如需检查线状或点状运动证据，再点击右上角“运动候选”。稳定层不会为缺帧位置补画恒星，状态栏会显示当前帧实际可见点数。该行为更新了早期版本“完成后默认进入运动层”的说明，检测与轨迹分类口径不变。

## License

本仓库暂未声明软件许可证。原始比赛数据的版权、授权和再分发条件也尚未在本仓库中确认；在获得明确许可前，请勿将 FITS 原始数据上传到公开位置。
