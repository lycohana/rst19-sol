# 天基星图研究论文工程

题目：**天基星图的分层源检测、测光与多帧运动分析方法**。

这是中文单栏研究论文，包含中英文摘要、方法推导、结果图表、参考文献和证据附录。作者：李钧耀；单位：广东科技学院。未提供通信方式，首页暂不显示通信作者字段。正文不把算法质量源数、仪器星等、运动候选或注入回收率解释为未经验证的物理真值。

## 文件

- `main.tex`：正文、公式、图表交叉引用与参考文献，使用 XeLaTeX 编译。
- `figures/`：7 组 PDF 矢量／混合图及对应 PNG；流程图在 TeX 内由 TikZ 绘制。
- `build.ps1`：三轮编译与引用／缺字检查。
- `prepare_assets.py`：从指定现存缓存和原始 FITS 重建结果图件，**不运行检测器，不写项目缓存**。
- `qa_pdf.py`：PDF 全页渲染、缩略图和文本边界检查。
- `evidence.json`：缓存 SHA-256、参数、结果摘要及核对范围。
- `input_manifest.csv`：15 个原始 FITS 的重新核验哈希、时间和结构。
- `source_classes.csv`、`motion_cache.csv`：从现存缓存重新汇总的源类别和运动结果。
- `injection_documented.csv`：历史注入记录转录及重新计算的 Wilson 区间，**不是本次重跑结果**。
- `sources.md`：项目内部证据和外部参考资料的使用说明。
- `verification.md`：最终构建与视觉检查记录。

最终 PDF 在项目根目录 `output/pdf/rst19-starfield-paper.pdf`；工程压缩包在 `output/pdf/rst19-paper-source.zip`。

## 编译

在项目根目录 PowerShell 执行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\deliverables\paper\build.ps1
```

需要 Windows 字体（宋体、黑体、Times New Roman、Consolas）及 MiKTeX／TeX Live 的 XeLaTeX、ctex、TikZ、placeins 等宏包。工程已经包含图件，**重新编译论文不需要原始 FITS**。日志在 `tmp/paper-build/`。

若解压到独立目录，可在 `main.tex` 所在目录执行：

```powershell
xelatex -interaction=nonstopmode -halt-on-error main.tex
xelatex -interaction=nonstopmode -halt-on-error main.tex
xelatex -interaction=nonstopmode -halt-on-error main.tex
```

跨平台使用时需把 `fontset=windows` 和字体配置替换为当地可用的 CJK 字体。

## 重建图件与检查 PDF

Python 依赖：`numpy`、`matplotlib`、`Pillow`、`pymupdf`。脚本以自身位置推导项目根目录，不依赖启动目录。

```powershell
python .\deliverables\paper\prepare_assets.py
powershell -NoProfile -ExecutionPolicy Bypass -File .\deliverables\paper\build.ps1
python .\deliverables\paper\qa_pdf.py
```

重建图件需要当前项目的 15 个原始 FITS 和脚本指定的三个缓存文件。脚本按序列记录核对每个 FITS 的 SHA-256；不匹配时停止，避免引用错误数据。原始数据与大型缓存没有放入工程压缩包。

## 内容边界

本轮确认的是原始文件与缓存身份、缓存统计和原图可视化的一致性，不是重新运行算法的准确率验证。三布局注入和部分双响应模型实验目前仅能追溯到 Markdown 记录，原机读产物缺失，正文及附录已标明。

当前序列缓存 71 与旧稿结果不同：3000 源工作集，点状 moving=0、线状 moving=4，其中一条有伪影重叠复核标记、一条含 5 个预测窗口恢复点。论文采用现存缓存口径，并未把它们升级为已确认天体。最暗质量候选只有未标定仪器星等。

正式投稿前需补齐通信方式及目标期刊声明、原注入产物或复跑、WCS／标准测光与独立运动真值。现有源码和 Markdown 修改未由本论文工程改写。
