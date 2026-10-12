# Rust 运算层迁移

目标是把全部运算实现移到 Rust，并从生产运行包中移除 Python。
本目录为原生核心库 `rst19-core` 与隔离算法进程 `rst19-engine`，运行时不嵌入 Python、不通过 FFI 或子进程调用 Python。

迁移尚未完成。完整后端模块、函数、源码指纹和 78 个工具入口列在 `migration.json`。
已开始原生实现 FITS、辅助遥测、曝光单位、稳健背景统计、TAN 坐标和四元数几何。
检测链中的 Gaussian/Starlet 掩膜归一化滤波、滑动极大值、候选 NMS、多通道合并、连通域 PCA 线结构审计和 KD-tree 空间检索也已建立原生实现。float64 检测已通过对照；float32 已实现独立存储与计算路径，但仍存在数值验证失败，详见 `verification.md`。
float64 的 Gaussian / hybrid / ensemble 已串成完整检测链，包含孔径测光、PSF 加权定位、质量规则、双 PSF/BIC 证据、近邻 Gaussian 门控和可选局部残差去混叠。共享背景模型核验二维形状，不同形状会回退逐帧背景。
`rst19-engine` 已支持 `detection.array`、`detection.fits`、`measurement.source` 和 `psf.pair` JSONL 操作。78 个原 CLI 契约与桌面路由仍待迁移。
现有桌面工作台继续使用迁移前的 Python 算法进程；尚未支持的工具不会被静默替换为近似实现。

```powershell
cargo test --locked --manifest-path backend/Cargo.toml
cargo build --release --locked --manifest-path backend/Cargo.toml
python backend/verify_fits.py backend/target/release/rst19-engine.exe --out tmp/native-fits-verification --data-dir "doc/00-项目资料/原始数据"
```

`generate_reference.py` 仅用于捕获 Python 基线，生成固定的 Rust 对照测试样例，不在生产运算链中运行。
对照覆盖曝光优先级、MAD/标准差回退、采样、旧版网格插值坐标、RA 跨零、旋转、奇偶性与四元数归一化。
`verify_fits.py` 覆盖六种 FITS 存储类型、显式 BSCALE/BZERO、64 位整数原始值、遥测识别、异常文件与可选真实数据。

只有全部算法、工具输出和界面路由完成对照验证后，才移除原有 Python 运行时并将迁移标记为完成。
