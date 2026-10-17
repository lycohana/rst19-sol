# E2 · 生产误差模型只读审计（第 1 帧，detection_id = 63263）

> 独立交叉验证工具，非生产实现。本文只**读**生产代码，未修改任何 `backend/src` 文件。
> 数值全部由 `tools/xcheck/xcheck_frame1.py` 的 E2 段重跑得到，证据 JSON：
> `tmp/xcheck-frame1/20261017T100431650_2C449A5B/e2_snr_table.json`。
> 本轮未做板解、未下载星表、未做光度标定。

## 1. 逐字摘录（`backend/src/measurement.rs`）

环带纳入条件（第 196–198 行）：

```rust
            if distance >= (radius + 2) as f64 && distance <= outer as f64 && !masked {
                annulus_values.push(value);
            }
```

背景/噪声选择（第 201–220 行）：

```rust
    let annulus_usable = annulus_values.len() >= 16;
    let (background, local_noise) = if annulus_usable && options.refine_local_background {
        ...
            statistics::sigma_clipped_stats(&annulus_values, None, 3.0, 4, 100_000)?
    } else {
        (
            ctx.background_map.value(yp * width + xp),
            ctx.noise_map.value(yp * width + xp),
        )
    };
    let noise = local_noise.max(f64::EPSILON);
```

误差模型（第 232–244 行，逐字）：

```rust
    let aperture_pixels = aperture.iter().filter(|v| **v).count();
    let background_pixels = annulus_values.len().max(1);
    let source_variance = options
        .gain_e_per_adu
        .map(|g| flux.max(0.0) / g)
        .unwrap_or(0.0)
        + aperture_pixels as f64 * options.read_noise_adu.powi(2);
    let background_variance = aperture_pixels as f64 * noise.powi(2)
        + (aperture_pixels as f64).powi(2) / background_pixels as f64 * noise.powi(2);
    let flux_error = (source_variance + background_variance)
        .max(f64::EPSILON)
        .sqrt();
    let flux_snr = flux / flux_error;
```

几何判据（第 189、195 行）：`let geometric = distance <= radius as f64;`，
`aperture.push(geometric && !masked);` —— **无分数像元权重**，孔径以候选整数峰为中心。

噪声估计器（`backend/src/statistics.rs` 第 88–118 行，节选）：

```rust
        let mut scale = 1.4826 * median(&deviations)?;
        ...
        if keep.len() == selected.len() {
            return Ok((center, scale.max(stddev(&selected)).max(f64::EPSILON)));
        }
```
即 `sigma = max(1.4826*MAD, std)`。

## 2. 三个具体判断

| 问题 | 判断 | 依据 |
|---|---|---|
| 环带 sigma-clip RMS 是否已含读出噪声 | **已含** | 环带取自**原始 DN**，未做任何噪声扣除；其 RMS 必然含暗电流、天光与读出噪声。 |
| 再加 `read_noise` 项是否重复计入 | **是，确认重复** | `source_variance` 的 `N_ap*RN^2` 与 `background_variance` 的 `N_ap*noise^2` 相加，而 `noise` 已含 RN。gain 与 RN 都给时误差被**高估**（偏保守但物理上错）。 |
| 缺 gain 时源 Poisson 项 | **被完全丢弃** | `.unwrap_or(0.0)`；`gain_e_per_adu = -1` 是缺省哨兵，不是物理增益。 |
| 背景估计误差用哪个 N | **clip 前的原始环带像元数** | `background_pixels = annulus_values.len()` 在 clipping 之前求值；clipping 后有效样本变少，该分母偏大 ⇒ 误差被**低估**。 |
| 几何 / 加权 | 整数像元判据、无分数权重、孔径中心为候选整数峰 | 第 189/195 行；质心另行计算但未用于孔径几何。 |

## 3. 对 detection_id = 63263 的纯算术

输入（保存记录）：flux = 219 ADU，`flux_snr` = 5.110702614293284，t = 1.5 s，r_ap = 4，环带 6–8 px。
按代码语义独立计数：**N_ap = 49，N_ann = 88**。
由 SNR 反解 **s = 4.9062 ADU/pixel**，flux_error = 42.851 ADU。

| gain (e-/ADU) | RN (ADU) | SNR（按现有代码） | σ_mag = 1.0857/SNR | 与记录 0.21244 之差 |
|---:|---:|---:|---:|---:|
| UNKNOWN | 0 | 5.111 | 0.2124 | 0.0000（构造性自检） |
| UNKNOWN | 2 | 4.858 | 0.2235 | +0.0110 |
| UNKNOWN | 5 | 3.958 | 0.2743 | +0.0619 |
| 0.5 | 0 | 4.592 | 0.2364 | +0.0240 |
| 0.5 | 5 | 3.702 | 0.2933 | +0.0809 |
| 1.0 | 0 | 4.831 | 0.2248 | +0.0123 |
| 1.0 | 5 | 3.824 | 0.2839 | +0.0714 |
| 2.0 | 0 | 4.965 | 0.2187 | +0.0062 |
| 4.0 | 0 | 5.036 | 0.2156 | +0.0031 |

按“不重复计入 RN”的写法，RN 项消失，结果等于 RN = 0 行（表中 `snr_no_double_count` 列）。
**gain/RN 的不确定性对这颗源只值 0.003–0.08 mag，远小于第 4、5 节的问题。**

## 4. 关键发现：这条误差条无法由图像复现

独立复刻生产语义（整数中心 (2930, 3915)，几何孔径 49 px，环带 6–8 px 共 88 px，
噪声取 `max(1.4826*MAD, std)`）：

- **通量复现成功**：flux = **219.00 ADU**（记录 219）；相邻整数中心为 219 / 220 / 225。
- **误差条复现失败**：环带 std = **7.2698**，MAD-σ = **5.9304**，IQR/1.349 = **5.189**，
  p16–84 半宽 = **6.0987**，3σ-clip 后 std = **7.2698**；128 px 盒 MAD-σ = **7.413**。
  生产隐式使用的 **4.9062** 比上述最小值 5.189 还低 5.5%，比文档给出的
  `max(1.4826*MAD, std)` = 7.2698 低 **33%**。
- 用可复现的 σ = 7.2698 代入生产的同一公式：flux_error = **63.50 ADU**，SNR = **3.449**，
  σ_mag = **0.315**（记录 0.212）。

**为什么重要**：流程用 `min_flux_snr = 5` 做质量筛选。若 σ 取可复现值，这颗源**进不了**
“质量通过”集合 —— “最暗质量源”这条记录对噪声估计器高度敏感。

**尚未确定（UNKNOWN）**：差异来源。候选原因：(a) 生产端 `ctx.mask` 排除了部分环带像元
（该 mask 未随结果保存）；(b) 实际候选整数中心与我推断的不同；(c) 走了 `noise_map` 分支。
128 px 盒噪声 7.413 也不等于 4.9062，故 (c) 单独不能解释。
需要在生产端对该 source 输出逐项 debug（中心、mask 命中、N_ap、N_ann、background、noise 来源）后定案。

## 5. 更本质的问题：模型本身低估孔径通量误差

在 2000 个“无源”位置直接测**生产估计量自身**的 RMS
（flux = Σ_aperture(I − median(annulus 6–8 px))，local σ = 7.745 ADU）：

| r_ap | N_ap | 实测 RMS（robust） | 模型 sqrt(N·s²+N²/N_ann·s²) | 比值 |
|---:|---:|---:|---:|---:|
| 2 | 13 | 33.36 ADU | 29.92 | 1.12 |
| 3 | 29 | 53.37 ADU | 48.09 | 1.11 |
| 4 | **49** | **80.06 ADU** | 67.65 | **1.18** |
| 6 | 113 | 145.29 ADU | 124.43 | 1.17 |

白噪声假设下模型**系统性低估 11–18%**（相关背景结构，κ ≈ 1.11–1.18）。
合起来：r = 4 处**野值位置的典型孔径通量误差是 80 ADU**，而这条记录写的是 **42.85 ADU（1.87× 偏小）**。
对应 σ_mag：**0.397**（80 ADU） vs **0.212**（记录） vs **0.145**（匹配滤波 7.51σ，见 E6）。

## 6. 对“m_inst = −5.41 的最暗质量源”的直接含义

- 它是一条**真实、PSF 形状良好**的检测：匹配滤波 SNR = **7.51**，FWHM = **3.40 px**，
  椭率 = **0.035**，位置与记录相差 **0.26 px**（E6）。
- 但它的**通量**不确定度按孔径统计是 **80 ADU（1σ）**，即 **38%**；σ_mag ≈ **0.40**，不是 0.212。
- 因此：**“已检测”成立，“通量可信到 0.21 mag”不成立。**

## 7. 修复方向（不改生产代码即可先做）

1. 生产端补逐源 debug 记录：center、mask 命中、N_ap、N_ann、clip 后 N_eff、background、noise 来源标记、σ。
2. 误差模型改成**分量结构体**而不是一个标量，让重复计入在输出里可见。
3. 用 E1 的做法把 κ（1.11–1.18）作为可标定常数引入，而不是假设白噪声。
4. RN / gain：要么都不给并把 Poisson、RN 标 UNKNOWN，要么给明确单位（e-）并在 ADU 域换算；
   当前 `read_noise_adu` 与环带 σ 同时生效必然重复。
