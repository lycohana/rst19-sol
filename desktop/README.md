# RST19 Rust 原生桌面

前端使用 Rust + egui/eframe 原生绘制。单帧 `rst19` 检测、星表匹配、光度分析、`rst19-aux-audit`、`rst19-gaia-merge`、`rst19-feature-evidence-axes`、`rst19-direct-neighbor-audit`、`rst19-feature-cross-axis-audit`、`rst19-source-group-audit`、`rst19-feature-flag-profile`、`rst19-feature-flag-interaction`、`rst19-feature-gate-margin`、`rst19-feature-gate-route`、`rst19-feature-hard-negative`、`rst19-feature-effect-size`、`rst19-feature-class-context`、`rst19-pair-mechanism-context`、`rst19-pair-mechanism-sensitivity`、`rst19-pair-pixel-topology`、`rst19-aperture-peak-ratio`、`rst19-feature-correlation` 和 `rst19-feature-transition-flow` 已通过独立 Rust worker 运行；尚未迁移的研究工具仍由 Python worker 执行。
桌面 Rust worker 还提供人工阈值反馈 JSONL 的追加与读取操作，作为人工复核记录保存。
完整发布包暂时同时包含 Rust 原生 worker 与 Python/NumPy/SciPy 环境，以及中文字体；使用者无需安装 Python、Rust、Node.js 或浏览器。

## 直接使用

解压整个发布 ZIP，双击 `rst19-desktop.exe`（Windows）或运行 `./rst19-desktop`（macOS/Linux）。保留同目录的 `runtime/` 与 `assets/`。
第一次启动选择原始 FITS 目录。原始比赛数据和星表不随软件分发；选择本地数据即可运行离线检测。
结果默认保存到用户目录的 `RST19-results/run-时间戳/`，不需要向安装目录写入。

界面左侧选择帧与研究工具，右侧填写参数，中间查看图像、源表、结构化证据、日志与产物。
读取预览后可滚轮缩放、拖动平移、点击叠加源查看完整字段；质量通过源与被拒候选分别显示。
完整研究工具表由现有命令的 argparse 定义生成，覆盖单帧、序列、合成、星表、测光、注入、参数扫描与各类科学审计。
鼠标悬停参数可查看说明；多值或重复参数填写 JSON 数组，例如 `["路径 1", "路径 2"]`，多组参数填写嵌套数组。
留空可选参数沿用算法默认值，输出目录默认分配到每次任务的独立目录；也可以指定输出路径。
下载公共星表需要网络，原有下载参数和科学质量门仍由算法模块控制。

单帧叠加绑定分析文件路径，切换帧后不会把上一帧的检测源画到新图。
序列 `moving` 点使用该帧原始 `x/y` 绘制；序列 JSON 和全部线状运动证据可在结构化证据页或导出的 JSON 中查看。
取消会终止独立算法进程，已经生成的部分产物保留；未完成任务不会显示为成功。
源表和结构化证据保留算法原始字段，包含质量、标定、星等、速度、拟合残差与未通过原因；检测数量与轨迹分类仍需科学验证。

## 开发

安装 Rust 工具链及 Python 3.11，在仓库根目录执行：

```powershell
python -m pip install -e .
cargo build --manifest-path desktop/Cargo.toml --bins
cargo run --manifest-path desktop/Cargo.toml
```

开发时优先使用仓库 `.venv`，也可设置 `RST19_PYTHON` 为指定解释器。
`RST19_PROJECT_ROOT` 可指定源码目录，`RST19_DESKTOP` 可为 Python 启动入口指定原生程序路径。
`rst19-gui` / `rst19-ui` 启动 Rust 桌面；`rst19-tk` 保留旧 Tkinter 工作台供迁移对照。

## 完整环境打包

```powershell
python desktop/build.py
```

构建脚本创建专用 `.build-venv`，安装固定版本的科学与打包依赖，构建 Rust release，
构建 Rust 桌面与原生计算 worker，使用 PyInstaller onedir 打包尚未迁移的 Python 工具环境，下载可再分发的 Noto 中文字体与许可证，
验证全部算法工具能在冻结环境中导入，再输出 `dist/RST19-平台-架构-时间/` 与 ZIP。
包中包含运行环境版本与逐文件 SHA-256 清单。每次构建使用新目录，不清理旧包或用户数据。

Windows 构建需要 MSVC 工具链；macOS 需要 Xcode Command Line Tools；Ubuntu 构建需要：

```sh
sudo apt-get install build-essential pkg-config libgtk-3-dev libxkbcommon-dev libwayland-dev libgl1-mesa-dev
python3 desktop/build.py
```

macOS 与 Linux 必须在目标平台构建 Python 原生扩展，不能把 Windows 的 runtime 复制过去。
CI 提供三平台构建与冻结环境检查；macOS 包尚未签名或公证。

## 验证

```powershell
cargo test --manifest-path desktop/Cargo.toml
python -m pytest tests/test_desktop_worker.py tests/test_fits.py -p no:cacheprovider --basetemp tmp/desktop-pytest
python desktop/verify_package.py "dist/RST19-平台-架构-时间" --out "tmp/package-verification"
# 从真实原生窗口请求截图并自动退出：
desktop/target/release/rst19-desktop.exe --data-dir "doc/00-项目资料/原始数据" --smoke-screenshot "tmp/rust-desktop.png"
```

截图验证的是界面实际启动、环境就绪和 FITS 预览，不等同于全部科学流程验收。
添加 `--smoke-analysis` 可让截图检查通过 Rust worker 执行当前目录首帧检测并绘制结果叠加；建议使用小型合成 FITS。
`verify_package.py` 禁用外部 Python 搜索路径，检查冻结环境的工具导入、Rust 单帧分析／辅助姿态审计／旗标统计／旗标交互审计／质量门余量审计／效应量审计／类别上下文审计／字段相关性审计／近邻机制上下文、窗口敏感性、像素拓扑、通量/峰值比、跨证据轴和源组审计、人工反馈 JSONL／预览路由、原图不变与包内 SHA-256。
平台运行与完整 15 帧分析需要各自的实际验证记录。

库文档：[eframe](https://docs.rs/eframe/0.31.1/eframe/)、[PyInstaller 打包选项](https://pyinstaller.org/en/stable/usage.html)。


Windows 发布包另附 runtime/rst19-gaia-merge.exe，macOS/Linux 发布包会附带同名无扩展名二进制；该命令直接运行 Rust 合并逻辑，完整审计返回 0，不完整审计返回 3，错误返回 2。
