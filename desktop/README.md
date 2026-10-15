# RST19 Rust 原生桌面

界面使用 Rust + egui/eframe 原生绘制。71 项工具全部由 `rst19-native-worker`
执行，检测、测光、星表、序列、运动、注入实验、报告和图像合成均调用
`rst19-core`。生产调用链没有解释器、Python 回退、PyInstaller 或构建虚拟环境。

## 使用发布包

解压完整 ZIP，Windows 双击 `rst19-desktop.exe`，macOS/Linux 运行
`./rst19-desktop`。保留同目录的 `runtime/` 和 `assets/`。包内含原生运算
程序、可再分发的静态 Noto 简体中文字体及许可证、环境记录和逐文件
SHA-256 清单。原始比赛 FITS 与个人星表由用户选择，不随包分发。

工具面板支持搜索、参数输入、原始 FITS 预览、源表、序列轨迹叠加、任务
取消、日志和产物查看。序列结果把稳定点源、点状运动和线状运动证据分开
保存，缺帧位置不补画检测。孔径重叠与双 PSF 证据用于复核候选，不确认
物理恒星身份。相对测光不自动升级为已标定绝对星等。

单帧星点检测同时提取当前曝光的线形候选，并将长度、轴角、宽度、模型
比较与显示证据保存在结果 JSON 的 `single_frame_trails` 中。观测视场的
“曝光线形 / 运动轨迹”开关控制叠加：橙色实线表示局部线段模型和形态通过，
橙色虚线表示待复核，`ARTIFACT?` 标识超长线与线状伪影冲突。总览中短线
使用可见的方向指示，放大后按原始像素长度绘制；悬停显示原始测量。
单帧线形保留候选身份，不输出跨帧运动方向，切换文件不会借用其它帧的线形。

所有工具也可通过统一原生命令调用：

```powershell
runtime/rst19-native-worker.exe --catalog
runtime/rst19-native-worker.exe --tool rst19-sequence "D:/FITS" --json-out "D:/results/sequence.json"
runtime/rst19-native-worker.exe --tool rst19-innovation "D:/results/sequence.json" --out-dir "D:/results/report"
runtime/rst19-native-worker.exe --tool rst19-injection --trials 1 --sources 2 --peak-levels 128 --out-dir "D:/results/injection"
```

完整包还包含独立的 `rst19-*` CLI；紧凑包通过统一 worker 提供同一份
71 项工具目录。Linux/macOS 去掉 `.exe` 后缀。

## 从源码运行

开发构建需要 Rust 和对应平台的链接工具链。Windows 使用 MSVC，macOS
使用 Xcode Command Line Tools；Linux 构建图形依赖：

```sh
sudo apt-get install build-essential pkg-config libgtk-3-dev libxkbcommon-dev libwayland-dev libgl1-mesa-dev
```

在仓库根目录执行：

```powershell
cargo build -j4 --locked --manifest-path desktop/Cargo.toml --bin rst19-native-worker
cargo run --locked --manifest-path desktop/Cargo.toml --bin rst19-desktop
```

桌面在开发目录发现同目录或 `desktop/target` 下的 Rust worker。可用
`RST19_NATIVE_WORKER` 指定原生 worker 文件，`RST19_PROJECT_ROOT` 指定
开发工作区。构建和运行不修改系统 PATH。

## 构建和核验

```powershell
cargo run --locked --manifest-path backend/Cargo.toml --bin rst19-package -- build
```

Rust 构建程序编译桌面、worker、引擎和独立 CLI，下载固定版本的静态字体，
写入环境与哈希清单，运行包内 worker 的实际调用验证，再生成新目录和 ZIP。
输出位于 `dist/RST19-平台-架构-时间/`；每次使用新目录，保留已有发布包。
`build --compact` 可只分发桌面、worker 和引擎。

```powershell
cargo test -j4 --locked --manifest-path backend/Cargo.toml
cargo test -j4 --locked --manifest-path desktop/Cargo.toml
cargo run --locked --manifest-path backend/Cargo.toml --bin rst19-package -- verify "dist/RST19-平台-架构-时间" --out "tmp/package-verification"
desktop/target/release/rst19-desktop.exe --data-dir "D:/FITS" --smoke-screenshot "tmp/rust-desktop.png"
```

包核验检查文件哈希、71 项原生目录、单帧、序列、报告、注入、双源审计和
运动提案消融的实际输出，并确认输入 FITS 未改变。像素/数值回归使用冻结
旧版参考数据或明确标注的源代码边界案例；图表布局可以改变，不宣称与旧版
逐像素相同。

三平台 CI 使用同一套 Rust 测试、构建和包核验流程，没有 setup-python。
各平台的实际构建结果以 CI 和 `backend/verification.md` 为准；macOS
签名、公证和真实桌面窗口验收需要对应宿主环境。
