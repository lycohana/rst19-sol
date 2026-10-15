# RST19 原生运算核心

`rst19-core` 实现 FITS 读取、原始整数码、辅助遥测、稳健背景、Gaussian /
DoG / Starlet 提案、孔径测光、质量规则、PSF/BIC 去混叠、WCS/星表、相对
与绝对星等证据、序列配准与轨迹、线状运动、注入实验、合成图和报告导出。
所有生产运算直接执行 Rust，没有解释器或跨语言回退。

`desktop/src/bin/rst19-native-worker.rs` 内嵌 71 项工具目录，为原生桌面
提供 JSONL 事件和可取消的隔离任务。`rst19-engine` 提供细粒度算法 JSONL
操作；各 `rst19-*` binary 提供独立 CLI。构建与包核验同样使用 Rust。

## 构建、测试和发布

```powershell
cargo build -j4 --locked --release --manifest-path backend/Cargo.toml --bins
cargo test -j4 --locked --manifest-path backend/Cargo.toml
cargo run --locked --manifest-path backend/Cargo.toml --bin rst19-package -- build
```

独立 CLI 位于 `backend/target/release/`。开发桌面需要同时构建
`desktop/target` 下的 worker，具体命令见 [桌面说明](../desktop/README.md)。
三平台 CI 对同一套 Rust 代码进行测试、构建与包核验。

## 序列与运动

`sequence_pipeline` 使用同一批原始 FITS 完成共享背景、检测、平移配准、
质量点源关联、快速点源补充、候选共识、时间参考与叠加暗星恢复、相对测光、
固定码审计、运动残差提案、原图模型复核、跨帧关联、断段回接和预测 ROI
恢复，随后装配公开序列结果。原图复核、候选关联和最终运动门相互分开。

`sequence_cli` 保留研究命令参数并导出原图序列源审计。`--workers` 接受
并记录调用意图，当前原生逐帧检测编排按序执行；它不改变阈值或科学结果。
缺失观测不作为真实检测加入轨迹。候选轨迹与物理天体身份不等价。

## 注入与证据导出

`injection` 覆盖合成噪声、十一类特征控制、真实背景、背景分层、同像素
提案比较、双源强弱比与拥挤组。冻结 NPZ 文件由 Rust 写出，保留输入
哈希、真值坐标或场景条件和有效参数；真实 FITS 始终不被注入修改。
源回收率、组解析率、合并响应和背景数量分别报告。

`source_pair_audit` 保留原始整数的唯一值/重复码统计、极端范围码、
孔径共享通量及原始/范围掩膜/哨兵掩膜 PSF 比较。两个目标未同时被独立
候选找到时不报告成对证据，避免回退坐标造成虚假双源结论。

`innovation` 与其测光/绘图模块导出运动拟合、平均轨迹置信区间、遥测
差分与单位待核验状态、注册差分图、原图候选切片、源表和严格测光证据。
各消融报告可在同一 HTML 展示 CSV、冻结输入、哈希清单与场景图。

## 缓存和科学边界

原生缓存采用独立命名空间与版本，使用原始文件内容 SHA-256、输入顺序、
科学参数和相关星表/WCS/背景参考绑定结果。损坏或不完整缓存按未命中处理；
原子写入保留完整候选工作集。Mosaic 缓存保留 float32 位模式和 uint16
覆盖计数。旧版缓存字节/键格式不作为原生兼容承诺。

冻结旧版输出作为离线数据进入 Rust 数值回归，测试不执行旧版语言。
`tests/fixtures/legacy_sources/*.txt` 仅用于旧结果的来源指纹校验；不编译、
解释或分发。每项数值容差与已执行的验证层记录在 [verification.md](verification.md)。
图表字体、布局和图像编码不宣称与旧版逐像素相同。

算法候选、质量通过源、注入回收和物理身份始终分开；相对测光、未声明
光度系统/波段的数值以及缺乏距离/消光来源的绝对星等保持诊断状态。
Windows、macOS 与 Linux 的实际包/窗口验收范围以验证记录和 CI 结果为准。
