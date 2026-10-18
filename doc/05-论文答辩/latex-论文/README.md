# 论文稿（LaTeX）数据来源

编译：`xelatex main.tex`（两遍，MiKTeX 25.12 + ctexart）。成品 `main.pdf`（11 页）。

| 图/表 | 数据文件 | 来源（可复算） |
|---|---|---|
| 图 1 峰值分布（含 3992 DN 台阶） | figs/peak_hist.csv, figs/peak_zoom.csv | `tmp/rust-sources-prod/source_catalog.csv`（第 1 帧，66,469 行） |
| 图 2 FWHM 分布 | figs/fwhm_hist.csv | 同上，按 `quality_passed` 分列 |
| 图 3 信噪比分布 | figs/snr_hist.csv | 同上，质量通过子集 |
| 图 4/5 零点与残差随 G | figs/zp_vs_g.csv, figs/res_vs_g.csv | `tmp/rust-pairs/deep-20261017T142347Z/pairs.csv`（20,906 未饱和内点） |
| 图 6 零点随半径 | figs/zp_vs_r.csv | 同上 |
| 图 7 帧间散度 vs 光子预期 | figs/delta_vs_snr.csv | `tmp/rust-pairs/f15_summary/delta_vs_snr.csv`（105 个帧对） |
| 图 8 拖影标注图 | figs/trail_annotated.png | `tmp/paper/motion/`（`measure_trail.py` 等四个脚本） |
| 表 1 仪器层事实 | 正文 | FITS 头 + 首行遥测解码 |
| 表 2 饱和判据影响 | 正文 | 两种 `--peak-ceiling` 下的 photometric-pairs 运行 |
| 表 3 误差预算 | 正文 | 见 `doc/交付验收-最终包-1792273471314.md` 与各专题文档 |

生成脚本：`tmp/paper/mk_figs.py`、`tmp/paper/mk_zoom.py`（图数据），`tmp/paper/motion/*.py`（拖影）。

## 不可声称项（与正文一致）

- $g_{\rm emp}$ 不是 Gaia $G$、不是 $V$；波段声明为 450–750 nm 标称、响应未知。
- 无增益/平场/总响应曲线，绝对精度不可验证。
- 拖影的通量为下界，峰值只给量级（饱和码与 4 倍读数并存的仪器异常）。
- 单帧不能定运动方向、真实速率与天体类别。