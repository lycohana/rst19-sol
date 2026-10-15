//! Original-image conditions used to select reproducible injection sites.
use crate::{
    detection::DetectionResult, measurement::Detection, models::FitsFrame,
    source_geometry_audit::NumpyPcg64, spatial_injection, statistics, Result,
};
use serde::Serialize;

#[derive(Clone, Debug, Serialize)]
pub(crate) struct Site {
    pub x: f64,
    pub y: f64,
    pub local_background_adu: f64,
    pub local_noise_adu: f64,
    pub upper_excess_adu: f64,
    pub special_pixel_fraction: f64,
    pub baseline_neighbor_count: usize,
    pub nearest_baseline_source_px: Option<f64>,
}
pub(crate) fn site(
    frame: &FitsFrame,
    sources: &[Detection],
    special: &[bool],
    x: f64,
    y: f64,
    radius: usize,
) -> Option<Site> {
    if !x.is_finite()
        || !y.is_finite()
        || x < 0.
        || y < 0.
        || x >= frame.width as f64
        || y >= frame.height as f64
    {
        return None;
    }
    let (background, noise, upper) =
        spatial_injection::local_patch_stats(&frame.data, frame.height, frame.width, x, y)?;
    let cx = x.round_ties_even() as usize;
    let cy = y.round_ties_even() as usize;
    let (x0, x1) = (
        cx.saturating_sub(radius),
        (cx + radius + 1).min(frame.width),
    );
    let (y0, y1) = (
        cy.saturating_sub(radius),
        (cy + radius + 1).min(frame.height),
    );
    let mut count = 0;
    for yy in y0..y1 {
        for xx in x0..x1 {
            count += usize::from(special[yy * frame.width + xx]);
        }
    }
    let distances = sources
        .iter()
        .map(|s| (s.x - x).hypot(s.y - y))
        .filter(|d| *d <= radius as f64)
        .collect::<Vec<_>>();
    Some(Site {
        x,
        y,
        local_background_adu: background,
        local_noise_adu: noise,
        upper_excess_adu: upper,
        special_pixel_fraction: count as f64 / ((x1 - x0) * (y1 - y0)).max(1) as f64,
        baseline_neighbor_count: distances.len(),
        nearest_baseline_source_px: distances.into_iter().min_by(f64::total_cmp),
    })
}
pub(crate) fn select_blank(
    frame: &FitsFrame,
    baseline: &DetectionResult,
    rng: &mut NumpyPcg64,
    count: usize,
    margin: usize,
    exclusion: f64,
) -> Result<Vec<Site>> {
    if count == 0 || frame.height <= 2 * margin || frame.width <= 2 * margin {
        return Err("image is too small for injection margins or count is zero".into());
    }
    let mut selected = Vec::<Site>::new();
    let special = vec![false; frame.data.len()];
    for _ in 0..5000usize.max(count.saturating_mul(1500)) {
        if selected.len() == count {
            break;
        }
        let x = rng.uniform(margin as f64, (frame.width - margin) as f64);
        let y = rng.uniform(margin as f64, (frame.height - margin) as f64);
        if selected
            .iter()
            .any(|p| (p.x - x).hypot(p.y - y) < exclusion)
            || baseline
                .sources
                .iter()
                .any(|s| (s.x - x).hypot(s.y - y) < exclusion)
        {
            continue;
        }
        let Some(mut candidate) = site(frame, &baseline.sources, &special, x, y, 8) else {
            continue;
        };
        if candidate.upper_excess_adu
            > (3. * candidate.local_noise_adu)
                .max(3. * baseline.noise)
                .max(f64::EPSILON)
        {
            continue;
        }
        candidate.baseline_neighbor_count = 0;
        candidate.nearest_baseline_source_px = None;
        selected.push(candidate);
    }
    if selected.len() != count {
        return Err(format!(
            "无法在真实背景找到 {count} 个合规位置，仅找到 {} 个",
            selected.len()
        ));
    }
    Ok(selected)
}
fn greedy(pool: Vec<Site>, count: usize, distance: f64) -> Vec<Site> {
    let mut result = Vec::<Site>::new();
    for p in pool {
        if result
            .iter()
            .all(|v| (p.x - v.x).hypot(p.y - v.y) >= distance)
        {
            result.push(p);
            if result.len() == count {
                break;
            }
        }
    }
    result
}
pub(crate) fn select_stratum(
    frame: &FitsFrame,
    baseline: &DetectionResult,
    special: &[bool],
    rng: &mut NumpyPcg64,
    count: usize,
    stratum: &str,
    psf: f64,
    aperture: usize,
) -> Result<Vec<Site>> {
    let radius = 8usize.max((2. * psf).ceil() as usize);
    let recovery = 3f64.max(psf);
    let distance = 10f64.max(2.5 * psf);
    let margin = (radius + 2).max((4. * psf / 2.354820045).ceil() as usize + 2);
    let inside = |x: f64, y: f64, m: usize| {
        x >= m as f64
            && y >= m as f64
            && x < (frame.width as f64 - m as f64)
            && y < (frame.height as f64 - m as f64)
    };
    if count == 0
        || !matches!(
            stratum,
            "blank" | "high_background" | "edge" | "crowded" | "special_code" | "line"
        )
    {
        return Err("unknown injection stratum or zero count".into());
    }
    if frame.height <= 2 * margin || frame.width <= 2 * margin {
        return Err("image too small for stratified injection margins".into());
    }
    let mut pool = Vec::new();
    match stratum {
        "blank" | "high_background" => {
            let target = 256usize.max(count.saturating_mul(80));
            for _ in 0..20_000usize.max(target.saturating_mul(160)) {
                if pool.len() >= target {
                    break;
                }
                let x = rng.uniform(margin as f64, (frame.width - margin) as f64);
                let y = rng.uniform(margin as f64, (frame.height - margin) as f64);
                if baseline
                    .sources
                    .iter()
                    .any(|s| (s.x - x).hypot(s.y - y) < radius as f64)
                {
                    continue;
                }
                let Some(mut p) = site(frame, &baseline.sources, special, x, y, radius) else {
                    continue;
                };
                if p.special_pixel_fraction > 0.
                    || p.upper_excess_adu > (4. * p.local_noise_adu).max(4. * baseline.noise)
                {
                    continue;
                }
                p.baseline_neighbor_count = 0;
                p.nearest_baseline_source_px = None;
                pool.push(p);
            }
            if pool.len() < count.max(8) {
                return Err(format!("真实背景 {stratum} 层候选不足：{}", pool.len()));
            }
            let cut = statistics::percentile(
                &pool.iter().map(|p| p.local_noise_adu).collect::<Vec<_>>(),
                if stratum == "blank" { 50. } else { 75. },
            )?;
            pool.retain(|p| {
                if stratum == "blank" {
                    p.local_noise_adu <= cut
                } else {
                    p.local_noise_adu >= cut
                }
            });
            pool.sort_by(|a, b| {
                let order = a
                    .local_noise_adu
                    .total_cmp(&b.local_noise_adu)
                    .then_with(|| a.upper_excess_adu.total_cmp(&b.upper_excess_adu));
                (if stratum == "blank" {
                    order
                } else {
                    order.reverse()
                })
                .then_with(|| a.y.total_cmp(&b.y))
                .then_with(|| a.x.total_cmp(&b.x))
            });
        }
        "edge" => {
            let band = (aperture + 4).max((2. * psf).ceil() as usize);
            if frame.height <= 2 * band + 4 {
                return Err("image too short for edge injection".into());
            }
            for _ in 0..20_000usize.max(count.saturating_mul(2000)) {
                let side = rng.bounded(1);
                let x = if side == 0 {
                    rng.uniform(1.5, 1.6f64.max(band as f64 - 0.25))
                } else {
                    rng.uniform(
                        frame.width as f64 - band as f64 + 0.25,
                        frame.width as f64 - 1.5,
                    )
                };
                let y = rng.uniform(
                    12usize.max(band + 2) as f64,
                    (frame.height - band - 2) as f64,
                );
                if !inside(x, y, 1) {
                    continue;
                }
                let Some(p) = site(frame, &baseline.sources, special, x, y, radius) else {
                    continue;
                };
                if p.nearest_baseline_source_px
                    .is_some_and(|d| d < radius as f64)
                {
                    continue;
                }
                pool.push(p);
            }
        }
        "crowded" | "line" => {
            let mut eligible = baseline
                .sources
                .iter()
                .filter(|s| {
                    if stratum == "line" {
                        s.flags.iter().any(|f| f == "LINE_ARTIFACT")
                    } else {
                        s.quality_passed
                            && inside(s.x, s.y, margin)
                            && !s.flags.iter().any(|f| {
                                matches!(
                                    f.as_str(),
                                    "EDGE" | "MASKED" | "SATURATED" | "LINE_ARTIFACT"
                                )
                            })
                    }
                })
                .collect::<Vec<_>>();
            if stratum == "crowded" {
                eligible.sort_by(|a, b| {
                    b.flux_snr
                        .unwrap_or(b.snr)
                        .total_cmp(&a.flux_snr.unwrap_or(a.snr))
                });
                eligible.truncate(256usize.max(count * 32));
            }
            if !eligible.is_empty() {
                for _ in 0..20_000usize.max(count.saturating_mul(3000)) {
                    let anchor = eligible[rng.bounded(eligible.len() - 1)];
                    let lo = (recovery + 1.).max(1.4 * psf);
                    let hi = if stratum == "crowded" {
                        (lo + 1.).max(3.5 * psf)
                    } else {
                        3f64.max(3. * psf)
                    };
                    let distance = rng.uniform(lo, hi);
                    let angle = rng.uniform(0., std::f64::consts::TAU);
                    let x = anchor.x + distance * angle.cos();
                    let y = anchor.y + distance * angle.sin();
                    if !inside(x, y, margin) {
                        continue;
                    }
                    let Some(p) = site(frame, &baseline.sources, special, x, y, radius) else {
                        continue;
                    };
                    if p.baseline_neighbor_count == 0
                        || p.nearest_baseline_source_px.is_some_and(|d| d <= recovery)
                    {
                        continue;
                    }
                    pool.push(p);
                }
            }
        }
        "special_code" => {
            let mut dilated = special.to_vec();
            for _ in 0..(psf.ceil() as usize).max(1) {
                let previous = dilated.clone();
                for y in 0..frame.height {
                    for x in 0..frame.width {
                        let k = y * frame.width + x;
                        dilated[k] = previous[k]
                            || (x > 0 && previous[k - 1])
                            || (x + 1 < frame.width && previous[k + 1])
                            || (y > 0 && previous[k - frame.width])
                            || (y + 1 < frame.height && previous[k + frame.width]);
                    }
                }
            }
            let coordinates = dilated
                .iter()
                .enumerate()
                .filter_map(|(i, v)| v.then_some(i))
                .collect::<Vec<_>>();
            for i in rng.permutation(coordinates.len()) {
                let k = coordinates[i];
                let x = (k % frame.width) as f64;
                let y = (k / frame.width) as f64;
                if !inside(x, y, margin) {
                    continue;
                }
                let Some(p) = site(frame, &baseline.sources, special, x, y, radius) else {
                    continue;
                };
                if p.special_pixel_fraction > 0. {
                    pool.push(p);
                }
            }
        }
        _ => unreachable!(),
    }
    let selected = greedy(pool, count, distance);
    if selected.len() != count {
        return Err(format!(
            "真实背景 {stratum} 层无法得到 {count} 个分离位置，仅找到 {} 个；该层未执行",
            selected.len()
        ));
    }
    Ok(selected)
}
