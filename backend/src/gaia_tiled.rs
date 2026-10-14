//! Native planning, adaptive querying, and completeness audit for Gaia cones.

use crate::gaia_remote;
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet, VecDeque};
use std::fs;
use std::path::PathBuf;

const ANGLE_EPSILON_DEG: f64 = 1.0e-9;
const GRID_STEP_FACTOR: f64 = 0.8;
type Row = HashMap<String, String>;

#[derive(Debug, Clone, PartialEq)]
pub struct GaiaTile {
    pub tile_id: String,
    pub ra_deg: f64,
    pub dec_deg: f64,
    pub radius_deg: f64,
    pub strategy: String,
    pub depth: usize,
    pub parent_id: Option<String>,
}

impl GaiaTile {
    fn value(&self) -> Value {
        json!({
            "tile_id":self.tile_id,"ra_deg":self.ra_deg,"dec_deg":self.dec_deg,
            "radius_deg":self.radius_deg,"strategy":self.strategy,"depth":self.depth,
            "parent_id":self.parent_id
        })
    }
}

#[derive(Debug, Clone)]
struct TileRecord {
    sequence: usize,
    tile: GaiaTile,
    requested_limit: Option<usize>,
    returned_rows: usize,
    in_scope_rows: usize,
    new_rows: usize,
    duplicate_rows: usize,
    outside_scope_rows: usize,
    truncation_status: String,
    status: String,
    adql: String,
    error: Option<String>,
}

impl TileRecord {
    fn possibly_truncated(&self) -> bool {
        matches!(
            self.truncation_status.as_str(),
            "saturated" | "unknown_limit"
        )
    }
    fn value(&self) -> Value {
        let mut value = self.tile.value();
        let object = value
            .as_object_mut()
            .expect("tile serialization is an object");
        object.insert("sequence".into(), json!(self.sequence));
        object.insert("requested_limit".into(), json!(self.requested_limit));
        object.insert("returned_rows".into(), json!(self.returned_rows));
        object.insert("in_scope_rows".into(), json!(self.in_scope_rows));
        object.insert("new_rows".into(), json!(self.new_rows));
        object.insert("duplicate_rows".into(), json!(self.duplicate_rows));
        object.insert("outside_scope_rows".into(), json!(self.outside_scope_rows));
        object.insert("truncation_status".into(), json!(self.truncation_status));
        object.insert(
            "possibly_truncated".into(),
            json!(self.possibly_truncated()),
        );
        object.insert("status".into(), json!(self.status));
        object.insert("adql".into(), json!(self.adql));
        object.insert("error".into(), json!(self.error));
        value
    }
}

#[derive(Debug, Clone)]
struct GaiaTiledResult {
    rows: Vec<Row>,
    records: Vec<TileRecord>,
    center_ra_deg: f64,
    center_dec_deg: f64,
    search_radius_deg: f64,
    strategy: String,
    tile_radius_deg: f64,
    tile_limit: Option<usize>,
    complete: bool,
    duplicate_count: usize,
    outside_scope_count: usize,
    unresolved_tile_ids: Vec<String>,
    errors: Vec<String>,
    initial_tile_count: usize,
}

impl GaiaTiledResult {
    fn truncated_tile_ids(&self) -> Vec<String> {
        self.records
            .iter()
            .filter(|record| record.possibly_truncated())
            .map(|record| record.tile.tile_id.clone())
            .collect()
    }
    fn value(&self) -> Value {
        json!({
            "rows":self.rows,
            "tile_records":self.records.iter().map(TileRecord::value).collect::<Vec<_>>(),
            "center_ra_deg":self.center_ra_deg,"center_dec_deg":self.center_dec_deg,
            "search_radius_deg":self.search_radius_deg,"strategy":self.strategy,
            "tile_radius_deg":self.tile_radius_deg,"tile_limit":self.tile_limit,
            "complete":self.complete,"duplicate_count":self.duplicate_count,
            "outside_scope_count":self.outside_scope_count,
            "unresolved_tile_ids":self.unresolved_tile_ids,
            "truncated_tile_ids":self.truncated_tile_ids(),"errors":self.errors,
            "initial_tile_count":self.initial_tile_count,"queried_tile_count":self.records.len()
        })
    }
}

#[derive(Debug)]
struct TiledError {
    message: String,
    partial: Option<Box<GaiaTiledResult>>,
}

#[derive(Debug, Clone)]
struct Options {
    ra: f64,
    dec: f64,
    radius: f64,
    out: Option<PathBuf>,
    tile_radius: f64,
    limit: Option<usize>,
    strategy: String,
    min_g_mag: Option<f64>,
    max_g_mag: Option<f64>,
    include_model: bool,
    max_depth: usize,
    max_queries: usize,
    saturated_policy: String,
    timeout: f64,
    endpoint: String,
    dry_run: bool,
}

fn finite(raw: &str, name: &str) -> Result<f64, String> {
    let value = raw
        .parse::<f64>()
        .map_err(|_| format!("{name} must be a finite number; got {raw:?}"))?;
    if !value.is_finite() {
        return Err(format!("{name} must be a finite number; got {raw:?}"));
    }
    Ok(value)
}
fn coordinate_ra(value: f64, name: &str) -> Result<f64, String> {
    if !value.is_finite() || !(0.0..=360.0).contains(&value) {
        return Err(format!("{name} must be in [0, 360] degrees; got {value}"));
    }
    Ok(value.rem_euclid(360.0))
}
fn coordinate_dec(value: f64, name: &str) -> Result<f64, String> {
    if !value.is_finite() || !(-90.0..=90.0).contains(&value) {
        return Err(format!("{name} must be in [-90, 90] degrees; got {value}"));
    }
    Ok(value)
}
fn validate_radius(value: f64, name: &str) -> Result<f64, String> {
    if !value.is_finite() || !(value > 0.0 && value <= 180.0) {
        return Err(format!(
            "{name} must be greater than 0 and at most 180 degrees; got {value}"
        ));
    }
    Ok(value)
}
fn normalize_choice(value: &str, name: &str, allowed: &[&str]) -> Result<String, String> {
    let normalized = value.trim().to_ascii_lowercase().replace('-', "_");
    if allowed.contains(&normalized.as_str()) {
        Ok(normalized)
    } else {
        Err(format!(
            "{name} must be {}; got {value:?}",
            allowed.join(", ")
        ))
    }
}

pub fn angular_distance_deg(ra1: f64, dec1: f64, ra2: f64, dec2: f64) -> f64 {
    let (ra1, ra2, dec1, dec2) = (
        ra1.to_radians(),
        ra2.to_radians(),
        dec1.to_radians(),
        dec2.to_radians(),
    );
    let delta_dec = dec2 - dec1;
    let delta_ra = (ra2 - ra1 + std::f64::consts::PI).rem_euclid(2.0 * std::f64::consts::PI)
        - std::f64::consts::PI;
    let h =
        (delta_dec / 2.0).sin().powi(2) + dec1.cos() * dec2.cos() * (delta_ra / 2.0).sin().powi(2);
    2.0 * h
        .clamp(0.0, 1.0)
        .sqrt()
        .atan2((1.0 - h.clamp(0.0, 1.0)).sqrt())
        .to_degrees()
}

fn grid_values(low: f64, high: f64, step: f64) -> Vec<f64> {
    let span = (high - low).max(0.0);
    let count = ((span / step).ceil() as usize).max(1);
    let actual = span / count as f64;
    (0..=count)
        .map(|index| low + actual * index as f64)
        .collect()
}

fn longitude_half_width(center_dec: f64, row_dec: f64, cap: f64) -> Option<f64> {
    if cap >= std::f64::consts::PI - 1.0e-12 {
        return Some(std::f64::consts::PI);
    }
    let denominator = center_dec.cos() * row_dec.cos();
    let cosine_limit = cap.cos();
    if denominator.abs() <= 1.0e-14 {
        if center_dec.sin() * row_dec.sin() > cosine_limit + 1.0e-12 {
            None
        } else {
            Some(std::f64::consts::PI)
        }
    } else {
        let needed = (cosine_limit - center_dec.sin() * row_dec.sin()) / denominator;
        if needed > 1.0 + 1.0e-12 {
            None
        } else if needed < -1.0 - 1.0e-12 {
            Some(std::f64::consts::PI)
        } else {
            Some(needed.clamp(-1.0, 1.0).acos())
        }
    }
}

fn longitude_step(row_dec: f64, tile_radius: f64) -> f64 {
    let target = (tile_radius * GRID_STEP_FACTOR).max(1.0e-6_f64.to_radians());
    let cosine = row_dec.cos().abs();
    if cosine <= 1.0e-14 {
        2.0 * std::f64::consts::PI
    } else {
        (target / cosine).min(2.0 * std::f64::consts::PI)
    }
}

fn plan_small_circle(
    ra: f64,
    dec: f64,
    search: f64,
    tile_radius: f64,
) -> Result<Vec<GaiaTile>, String> {
    if search > tile_radius + ANGLE_EPSILON_DEG {
        return Err("small_circle strategy requires search_radius_deg <= tile_radius_deg; use grid for a larger field".into());
    }
    Ok(vec![GaiaTile {
        tile_id: "tile-00000".into(),
        ra_deg: ra,
        dec_deg: dec,
        radius_deg: search,
        strategy: "small_circle".into(),
        depth: 0,
        parent_id: None,
    }])
}

fn plan_grid(ra: f64, dec: f64, search: f64, tile_radius: f64) -> Result<Vec<GaiaTile>, String> {
    if search <= tile_radius + ANGLE_EPSILON_DEG {
        return Ok(vec![GaiaTile {
            tile_id: "tile-00000".into(),
            ra_deg: ra,
            dec_deg: dec,
            radius_deg: search,
            strategy: "grid".into(),
            depth: 0,
            parent_id: None,
        }]);
    }
    let extended = (search + tile_radius).min(180.0);
    let extended_rad = extended.to_radians();
    let center_dec = dec.to_radians();
    let lat_low = (-90.0_f64).max(dec - extended);
    let lat_high = 90.0_f64.min(dec + extended);
    let lat_step = (tile_radius * GRID_STEP_FACTOR).max(1.0e-6).to_radians();
    let mut centers: Vec<(f64, f64)> = Vec::new();
    let mut seen = HashSet::<(i64, i64)>::new();
    for row_dec_deg in grid_values(lat_low, lat_high, lat_step.to_degrees()) {
        let row_dec = row_dec_deg.to_radians();
        let Some(half_width) = longitude_half_width(center_dec, row_dec, extended_rad) else {
            continue;
        };
        let step = longitude_step(row_dec, tile_radius.to_radians());
        let ras: Vec<f64> = if half_width >= std::f64::consts::PI - 1e-12 {
            let count = ((2.0 * std::f64::consts::PI / step).ceil() as usize).max(1);
            (0..count)
                .map(|index| ra + 360.0 * index as f64 / count as f64)
                .collect()
        } else {
            let width = 2.0 * half_width;
            let count = ((width / step).ceil() as usize).max(1);
            (0..=count)
                .map(|index| {
                    (ra.to_radians() - half_width + width * index as f64 / count as f64)
                        .to_degrees()
                })
                .collect()
        };
        for raw_ra in ras {
            let mut tile_ra = raw_ra.rem_euclid(360.0);
            if (row_dec_deg.abs() - 90.0).abs() <= 1e-10 {
                tile_ra = ra;
            }
            let key = (
                (tile_ra * 1e9).round_ties_even() as i64,
                (row_dec_deg * 1e9).round_ties_even() as i64,
            );
            if seen.contains(&key) {
                continue;
            }
            if angular_distance_deg(ra, dec, tile_ra, row_dec_deg) <= extended + ANGLE_EPSILON_DEG {
                seen.insert(key);
                centers.push((tile_ra, row_dec_deg));
            }
        }
    }
    if centers.is_empty() {
        return Err("grid planner generated no tile centers".into());
    }
    centers.sort_by(|a, b| {
        angular_distance_deg(ra, dec, a.0, a.1)
            .total_cmp(&angular_distance_deg(ra, dec, b.0, b.1))
            .then(a.1.total_cmp(&b.1))
            .then(a.0.total_cmp(&b.0))
    });
    Ok(centers
        .into_iter()
        .enumerate()
        .map(|(index, (tile_ra, tile_dec))| GaiaTile {
            tile_id: format!("tile-{index:05}"),
            ra_deg: tile_ra,
            dec_deg: tile_dec,
            radius_deg: tile_radius,
            strategy: "grid".into(),
            depth: 0,
            parent_id: None,
        })
        .collect())
}

pub fn plan_gaia_tiles(
    ra: f64,
    dec: f64,
    search: f64,
    tile_radius: f64,
    strategy: &str,
) -> Result<Vec<GaiaTile>, String> {
    let ra = coordinate_ra(ra, "center_ra_deg")?;
    let dec = coordinate_dec(dec, "center_dec_deg")?;
    let search = validate_radius(search, "search_radius_deg")?;
    let tile_radius = validate_radius(tile_radius, "tile_radius_deg")?;
    let strategy = normalize_choice(strategy, "strategy", &["auto", "small_circle", "grid"])?;
    if strategy == "small_circle" || (strategy == "auto" && search <= tile_radius) {
        plan_small_circle(ra, dec, search, tile_radius)
    } else {
        plan_grid(ra, dec, search, tile_radius)
    }
}

fn source_id(row: &Row, row_number: usize) -> Result<String, String> {
    row.get("source_id")
        .or_else(|| row.get("SOURCE_ID"))
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty())
        .ok_or_else(|| format!("row {row_number}: missing source_id"))
}
fn row_coords(row: &Row, row_number: usize) -> Result<(f64, f64), String> {
    let ra = row
        .get("ra_deg")
        .or_else(|| row.get("ra"))
        .or_else(|| row.get("RA"))
        .ok_or_else(|| format!("row {row_number}: missing ra_deg/ra"))?
        .parse::<f64>()
        .map_err(|_| format!("row {row_number}: invalid ra_deg/ra"))?;
    let dec = row
        .get("dec_deg")
        .or_else(|| row.get("dec"))
        .or_else(|| row.get("DEC"))
        .ok_or_else(|| format!("row {row_number}: missing dec_deg/dec"))?
        .parse::<f64>()
        .map_err(|_| format!("row {row_number}: invalid dec_deg/dec"))?;
    Ok((
        coordinate_ra(ra, &format!("row {row_number} ra"))?,
        coordinate_dec(dec, &format!("row {row_number} dec"))?,
    ))
}

fn tile_adql(tile: &GaiaTile, options: &Options) -> Result<String, String> {
    gaia_remote::build_adql(
        tile.ra_deg,
        tile.dec_deg,
        tile.radius_deg,
        options.limit,
        options.min_g_mag,
        options.max_g_mag,
        options.include_model,
    )
}

fn subdivide(tile: &GaiaTile) -> Result<Vec<GaiaTile>, String> {
    let child_radius = tile.radius_deg * 0.5;
    if child_radius <= 1e-6 {
        return Ok(Vec::new());
    }
    let children = plan_grid(tile.ra_deg, tile.dec_deg, tile.radius_deg, child_radius)?;
    Ok(children
        .into_iter()
        .enumerate()
        .map(|(index, child)| GaiaTile {
            tile_id: format!("{}.{index:04}", tile.tile_id),
            ra_deg: child.ra_deg,
            dec_deg: child.dec_deg,
            radius_deg: child.radius_deg,
            strategy: "adaptive_grid".into(),
            depth: tile.depth + 1,
            parent_id: Some(tile.tile_id.clone()),
        })
        .collect())
}

fn source_sort_key(id: &str) -> (u8, u128, String) {
    match id.parse::<u128>() {
        Ok(number) => (0, number, id.to_owned()),
        Err(_) => (1, 0, id.to_owned()),
    }
}

#[allow(clippy::too_many_arguments)]
fn assemble(
    rows: HashMap<String, Row>,
    records: Vec<TileRecord>,
    options: &Options,
    strategy: &str,
    initial_count: usize,
    unresolved: Vec<String>,
    errors: Vec<String>,
    duplicate_count: usize,
    outside_count: usize,
) -> GaiaTiledResult {
    let mut ids = rows.keys().cloned().collect::<Vec<_>>();
    ids.sort_by_key(|id| source_sort_key(id));
    let rows = ids
        .into_iter()
        .filter_map(|id| rows.get(&id).cloned())
        .collect();
    let mut unresolved_unique = Vec::new();
    let mut seen = HashSet::new();
    for id in unresolved {
        if seen.insert(id.clone()) {
            unresolved_unique.push(id);
        }
    }
    let complete = unresolved_unique.is_empty() && errors.is_empty();
    GaiaTiledResult {
        rows,
        records,
        center_ra_deg: options.ra,
        center_dec_deg: options.dec,
        search_radius_deg: options.radius,
        strategy: strategy.into(),
        tile_radius_deg: options.tile_radius,
        tile_limit: options.limit,
        complete,
        duplicate_count,
        outside_scope_count: outside_count,
        unresolved_tile_ids: unresolved_unique,
        errors,
        initial_tile_count: initial_count,
    }
}

fn run_with_query<F>(
    options: &Options,
    mut query: F,
    raise_on_error: bool,
) -> Result<GaiaTiledResult, TiledError>
where
    F: FnMut(&GaiaTile, &Options) -> Result<Vec<Row>, String>,
{
    let strategy = normalize_choice(
        &options.strategy,
        "strategy",
        &["auto", "small_circle", "grid"],
    )
    .map_err(|message| TiledError {
        message,
        partial: None,
    })?;
    let initial = plan_gaia_tiles(
        options.ra,
        options.dec,
        options.radius,
        options.tile_radius,
        &strategy,
    )
    .map_err(|message| TiledError {
        message,
        partial: None,
    })?;
    let initial_count = initial.len();
    let mut pending = VecDeque::from(initial);
    let mut records = Vec::new();
    let mut rows_by_id = HashMap::<String, Row>::new();
    let mut unresolved = Vec::new();
    let mut errors = Vec::new();
    let mut duplicate_count = 0;
    let mut outside_count = 0;
    while let Some(tile) = pending.pop_front() {
        if records.len() >= options.max_queries {
            let message = format!(
                "maximum tile query count {} reached before {}",
                options.max_queries, tile.tile_id
            );
            unresolved.push(tile.tile_id);
            errors.push(message.clone());
            let partial = assemble(
                rows_by_id,
                records,
                options,
                &strategy,
                initial_count,
                unresolved,
                errors,
                duplicate_count,
                outside_count,
            );
            if raise_on_error {
                return Err(TiledError {
                    message,
                    partial: Some(Box::new(partial)),
                });
            }
            return Ok(partial);
        }
        let adql = tile_adql(&tile, options).unwrap_or_default();
        match query(&tile, options) {
            Err(error) => {
                let error_text = error;
                errors.push(format!("{}: {error_text}", tile.tile_id));
                unresolved.push(tile.tile_id.clone());
                records.push(TileRecord {
                    sequence: records.len(),
                    tile: tile.clone(),
                    requested_limit: options.limit,
                    returned_rows: 0,
                    in_scope_rows: 0,
                    new_rows: 0,
                    duplicate_rows: 0,
                    outside_scope_rows: 0,
                    truncation_status: "query_error".into(),
                    status: "error".into(),
                    adql,
                    error: Some(error_text),
                });
                if raise_on_error {
                    let message = format!("tile {} query failed", tile.tile_id);
                    let partial = assemble(
                        rows_by_id,
                        records,
                        options,
                        &strategy,
                        initial_count,
                        unresolved,
                        errors,
                        duplicate_count,
                        outside_count,
                    );
                    return Err(TiledError {
                        message,
                        partial: Some(Box::new(partial)),
                    });
                }
                continue;
            }
            Ok(raw_rows) => {
                let returned = raw_rows.len();
                let mut local = HashMap::<String, Row>::new();
                let mut local_duplicates = 0;
                let mut outside = 0;
                for (index, row) in raw_rows.into_iter().enumerate() {
                    let row_id = match source_id(&row, index + 1) {
                        Ok(id) => id,
                        Err(error) => {
                            errors.push(format!("{}: {error}", tile.tile_id));
                            unresolved.push(tile.tile_id.clone());
                            continue;
                        }
                    };
                    let (row_ra, row_dec) = match row_coords(&row, index + 1) {
                        Ok(coords) => coords,
                        Err(error) => {
                            errors.push(format!("{}: {error}", tile.tile_id));
                            unresolved.push(tile.tile_id.clone());
                            continue;
                        }
                    };
                    if angular_distance_deg(options.ra, options.dec, row_ra, row_dec)
                        > options.radius + ANGLE_EPSILON_DEG
                    {
                        outside += 1;
                        continue;
                    }
                    if local.contains_key(&row_id) {
                        local_duplicates += 1;
                        continue;
                    }
                    local.insert(row_id, row);
                }
                let mut new_rows = 0;
                let mut cross_duplicates = 0;
                for (id, row) in &local {
                    if rows_by_id.contains_key(id) {
                        cross_duplicates += 1;
                    } else {
                        rows_by_id.insert(id.clone(), row.clone());
                        new_rows += 1;
                    }
                }
                let duplicates = local_duplicates + cross_duplicates;
                duplicate_count += duplicates;
                outside_count += outside;
                let saturated = match options.limit {
                    None => true,
                    Some(limit) => returned >= limit,
                };
                let truncation = match options.limit {
                    None => "unknown_limit",
                    Some(_) if saturated => "saturated",
                    Some(_) => "not_saturated",
                }
                .to_owned();
                let mut status = "ok".to_owned();
                if saturated
                    && options.limit.is_some()
                    && options.saturated_policy == "subdivide"
                    && tile.depth < options.max_depth
                {
                    match subdivide(&tile) {
                        Ok(children) if !children.is_empty() => {
                            for child in children.into_iter().rev() {
                                pending.push_front(child);
                            }
                            status = "saturated_subdivided".into();
                        }
                        _ => {
                            unresolved.push(tile.tile_id.clone());
                            status = "saturated_unresolved".into();
                        }
                    }
                } else if saturated {
                    unresolved.push(tile.tile_id.clone());
                    status = "saturated_unresolved".into();
                }
                records.push(TileRecord {
                    sequence: records.len(),
                    tile: tile.clone(),
                    requested_limit: options.limit,
                    returned_rows: returned,
                    in_scope_rows: local.len(),
                    new_rows,
                    duplicate_rows: duplicates,
                    outside_scope_rows: outside,
                    truncation_status: truncation,
                    status,
                    adql,
                    error: None,
                });
                if saturated && options.saturated_policy == "raise" {
                    let message = if let Some(limit) = options.limit {
                        format!(
                            "tile {} returned {returned} rows at limit {limit}",
                            tile.tile_id
                        )
                    } else {
                        format!("tile {} has unknown remote limit", tile.tile_id)
                    };
                    let partial = assemble(
                        rows_by_id,
                        records,
                        options,
                        &strategy,
                        initial_count,
                        unresolved,
                        errors,
                        duplicate_count,
                        outside_count,
                    );
                    return Err(TiledError {
                        message,
                        partial: Some(Box::new(partial)),
                    });
                }
            }
        }
    }
    Ok(assemble(
        rows_by_id,
        records,
        options,
        &strategy,
        initial_count,
        unresolved,
        errors,
        duplicate_count,
        outside_count,
    ))
}

fn parse_options(args: &[String]) -> Result<Options, String> {
    let mut values = HashMap::<String, String>::new();
    let mut flags = HashSet::<String>::new();
    let mut index = 0;
    while index < args.len() {
        let arg = &args[index];
        if arg == "--help" || arg == "-h" {
            return Err("__help__".into());
        }
        if ["--dry-run", "--no-limit", "--include-gspphot-model"].contains(&arg.as_str()) {
            flags.insert(arg.clone());
            index += 1;
            continue;
        }
        if !arg.starts_with("--") {
            return Err(format!("unexpected argument: {arg}"));
        }
        let (name, inline) = arg
            .split_once('=')
            .map(|(name, value)| (name, Some(value.to_owned())))
            .unwrap_or((arg.as_str(), None));
        if ![
            "--ra",
            "--dec",
            "--radius",
            "--out",
            "--tile-radius",
            "--limit",
            "--strategy",
            "--min-g-mag",
            "--max-g-mag",
            "--max-depth",
            "--max-queries",
            "--saturated-policy",
            "--timeout",
            "--endpoint",
        ]
        .contains(&name)
        {
            return Err(format!("unrecognized argument: {name}"));
        }
        let value = if let Some(value) = inline {
            value
        } else {
            index += 1;
            args.get(index)
                .cloned()
                .ok_or_else(|| format!("{name} requires a value"))?
        };
        values.insert(name.into(), value);
        index += 1;
    }
    let required = |key: &str| values.get(key).ok_or_else(|| format!("{key} is required"));
    let ra = coordinate_ra(finite(required("--ra")?, "ra")?, "ra")?;
    let dec = coordinate_dec(finite(required("--dec")?, "dec")?, "dec")?;
    let radius = validate_radius(finite(required("--radius")?, "radius")?, "radius")?;
    let tile_radius = validate_radius(
        finite(
            values
                .get("--tile-radius")
                .map(String::as_str)
                .unwrap_or("1.0"),
            "tile_radius",
        )?,
        "tile_radius",
    )?;
    let limit = if flags.contains("--no-limit") {
        None
    } else {
        Some(
            values
                .get("--limit")
                .map(|v| {
                    v.parse::<usize>()
                        .map_err(|_| "limit must be a positive integer".to_owned())
                })
                .transpose()?
                .unwrap_or(5000),
        )
    };
    if limit == Some(0) {
        return Err("limit must be a positive integer".into());
    }
    let strategy = normalize_choice(
        values
            .get("--strategy")
            .map(String::as_str)
            .unwrap_or("auto"),
        "strategy",
        &["auto", "small_circle", "grid"],
    )?;
    let min_g_mag = values
        .get("--min-g-mag")
        .map(|v| finite(v, "min_g_mag"))
        .transpose()?;
    let max_g_mag = values
        .get("--max-g-mag")
        .map(|v| finite(v, "max_g_mag"))
        .transpose()?;
    for (name, v) in [("min_g_mag", min_g_mag), ("max_g_mag", max_g_mag)] {
        if v.is_some_and(|m| !(-50.0..=50.0).contains(&m)) {
            return Err(format!("{name} must be in [-50, 50] magnitudes"));
        }
    }
    if min_g_mag.zip(max_g_mag).is_some_and(|(a, b)| a > b) {
        return Err("min_g_mag cannot be greater than max_g_mag".into());
    }
    let max_depth = values
        .get("--max-depth")
        .map(|v| {
            v.parse::<usize>()
                .map_err(|_| "max-depth must be a non-negative integer".to_owned())
        })
        .transpose()?
        .unwrap_or(2);
    let max_queries = values
        .get("--max-queries")
        .map(|v| {
            v.parse::<usize>()
                .map_err(|_| "max-queries must be a positive integer".to_owned())
        })
        .transpose()?
        .unwrap_or(10000);
    if max_queries == 0 {
        return Err("max-queries must be a positive integer".into());
    }
    let saturated_policy = normalize_choice(
        values
            .get("--saturated-policy")
            .map(String::as_str)
            .unwrap_or("subdivide"),
        "saturated-policy",
        &["subdivide", "record", "raise"],
    )?;
    let timeout = finite(
        values.get("--timeout").map(String::as_str).unwrap_or("30"),
        "timeout",
    )?;
    if timeout <= 0.0 {
        return Err("timeout must be greater than 0 seconds".into());
    }
    let endpoint = values
        .get("--endpoint")
        .map(|v| v.trim().to_owned())
        .unwrap_or_else(|| gaia_remote::DEFAULT_GAIA_TAP_SYNC_URL.into());
    let parsed =
        reqwest::Url::parse(&endpoint).map_err(|_| "endpoint must be an HTTP(S) URL".to_owned())?;
    if !matches!(parsed.scheme(), "http" | "https") || parsed.host_str().is_none() {
        return Err("endpoint must be an HTTP(S) URL".into());
    }
    Ok(Options {
        ra,
        dec,
        radius,
        out: values.get("--out").map(PathBuf::from),
        tile_radius,
        limit,
        strategy,
        min_g_mag,
        max_g_mag,
        include_model: flags.contains("--include-gspphot-model"),
        max_depth,
        max_queries,
        saturated_policy,
        timeout,
        endpoint,
        dry_run: flags.contains("--dry-run"),
    })
}

fn run_query(options: &Options, raise_on_error: bool) -> Result<GaiaTiledResult, TiledError> {
    run_with_query(
        options,
        |tile, options| {
            gaia_remote::query_rows(
                tile.ra_deg,
                tile.dec_deg,
                tile.radius_deg,
                options.limit,
                options.min_g_mag,
                options.max_g_mag,
                options.include_model,
                options.timeout,
                &options.endpoint,
            )
        },
        raise_on_error,
    )
}

fn metadata(options: &Options, result: &GaiaTiledResult) -> Value {
    json!({"catalog":"Gaia DR3","table":if options.include_model{"gaiadr3.gaia_source LEFT OUTER JOIN gaiadr3.astrophysical_parameters"}else{"gaiadr3.gaia_source"},
    "query_center_ra_deg":options.ra,"query_center_dec_deg":options.dec,"query_radius_deg":options.radius,"tile_radius_deg":options.tile_radius,"tile_limit":options.limit,
    "min_g_mag":options.min_g_mag,"max_g_mag":options.max_g_mag,"max_subdivide_depth":options.max_depth,"max_queries":options.max_queries,"saturated_policy":options.saturated_policy,"endpoint":options.endpoint,
    "row_count":result.rows.len(),"complete":result.complete,"initial_tile_count":result.initial_tile_count,"queried_tile_count":result.records.len(),"duplicate_count":result.duplicate_count,"outside_scope_count":result.outside_scope_count,
    "unresolved_tile_ids":result.unresolved_tile_ids,"truncated_tile_ids":result.truncated_tile_ids(),"errors":result.errors,"tile_records":result.records.iter().map(TileRecord::value).collect::<Vec<_>>(),
    "magnitude_error_source":"local first-order approximation from phot_g_mean_flux_error; Gaia does not publish a symmetric phot_g_mean_mag_error column",
    "gspphot_fields":if options.include_model{"mg_gspphot, mg_gspphot_lower and mg_gspphot_upper are included from gaiadr3.astrophysical_parameters"}else{"mg_gspphot is not requested"}})
}

fn write_artifacts(
    result: &GaiaTiledResult,
    options: &Options,
) -> Result<(PathBuf, PathBuf), String> {
    let output = options
        .out
        .as_ref()
        .ok_or("--out is required unless --dry-run is used")?;
    gaia_remote::write_catalog(&result.rows, output)?;
    let metadata_path = PathBuf::from(format!("{}.tiled.meta.json", output.to_string_lossy()));
    let mut bytes =
        serde_json::to_vec_pretty(&metadata(options, result)).map_err(|e| e.to_string())?;
    bytes.push(b'\n');
    fs::write(&metadata_path, bytes).map_err(|e| e.to_string())?;
    Ok((output.clone(), metadata_path))
}

fn dry_run_value(options: &Options) -> Result<Value, String> {
    let tiles = plan_gaia_tiles(
        options.ra,
        options.dec,
        options.radius,
        options.tile_radius,
        &options.strategy,
    )?;
    Ok(
        json!({"strategy":options.strategy,"tile_radius_deg":options.tile_radius,"initial_tile_count":tiles.len(),"first_tile":tiles.first().map(GaiaTile::value),"last_tile":tiles.last().map(GaiaTile::value),"network":false}),
    )
}

fn help() -> &'static str {
    "按天空坐标分块查询公共 Gaia DR3 星表并导出完整性审计结果\n\nUSAGE: rst19-gaia-tiled --ra DEG --dec DEG --radius DEG [OPTIONS]\n\nOptions:\n  --out PATH\n  --tile-radius DEG  [default: 1]\n  --limit N          [default: 5000]\n  --no-limit\n  --strategy auto|small_circle|grid\n  --min-g-mag MAG\n  --max-g-mag MAG\n  --include-gspphot-model\n  --max-depth N      [default: 2]\n  --max-queries N    [default: 10000]\n  --saturated-policy subdivide|record|raise\n  --timeout SECONDS  [default: 30]\n  --endpoint URL\n  --dry-run\n"
}

pub fn run_cli(args: &[String]) -> i32 {
    let options = match parse_options(args) {
        Ok(options) => options,
        Err(error) if error == "__help__" => {
            print!("{}", help());
            return 0;
        }
        Err(error) => {
            eprintln!("gaia-tiled: {error}");
            return 2;
        }
    };
    if options.dry_run {
        match dry_run_value(&options) {
            Ok(value) => println!(
                "{}",
                serde_json::to_string_pretty(&value).unwrap_or_default()
            ),
            Err(error) => {
                eprintln!("gaia-tiled: {error}");
                return 2;
            }
        }
        return 0;
    }
    if options.out.is_none() {
        eprintln!("gaia-tiled: --out is required unless --dry-run is used");
        return 2;
    }
    match run_query(&options, false) {
        Ok(result) => match write_artifacts(&result, &options) {
            Ok((csv, meta)) if result.complete => {
                eprintln!(
                    "Gaia DR3 tiled: wrote {} rows to {} ({} tiles; audit: {})",
                    result.rows.len(),
                    csv.display(),
                    result.records.len(),
                    meta.display()
                );
                0
            }
            Ok((csv, meta)) => {
                eprintln!("gaia-tiled: incomplete result wrote {} rows to {}; see audit {} for unresolved/truncated blocks",result.rows.len(),csv.display(),meta.display());
                3
            }
            Err(error) => {
                eprintln!("gaia-tiled: {error}");
                2
            }
        },
        Err(error) => {
            let code = 2;
            if let Some(result) = error.partial {
                if let Ok((csv, meta)) = write_artifacts(&result, &options) {
                    eprintln!("gaia-tiled: query failed; partial result wrote {} rows to {}; audit: {}; {}",result.rows.len(),csv.display(),meta.display(),error.message);
                } else {
                    eprintln!("gaia-tiled: {}", error.message);
                }
            } else {
                eprintln!("gaia-tiled: {}", error.message);
            }
            code
        }
    }
}

pub fn run_desktop(args: &[String]) -> Result<Value, String> {
    let options = parse_options(args).map_err(|e| {
        if e == "__help__" {
            help().to_owned()
        } else {
            e
        }
    })?;
    if options.dry_run {
        return dry_run_value(&options);
    }
    let result = match run_query(&options, false) {
        Ok(result) => result,
        Err(error) => *error.partial.ok_or(error.message)?,
    };
    let (csv, metadata) = write_artifacts(&result, &options)?;
    Ok(
        json!({"csv":csv.to_string_lossy(),"metadata":metadata.to_string_lossy(),"result":result.value()}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_json_matches_reference(actual: &Value, expected: &Value, path: &str) {
        match (actual, expected) {
            (Value::Number(actual), Value::Number(expected)) => {
                let actual = actual.as_f64().expect("actual JSON number converts to f64");
                let expected = expected
                    .as_f64()
                    .expect("reference JSON number converts to f64");
                let tolerance = 1.0e-12;
                assert!(
                    (actual - expected).abs() <= tolerance,
                    "{path}: Rust {actual:?} != Python reference {expected:?}"
                );
            }
            (Value::Array(actual), Value::Array(expected)) => {
                assert_eq!(actual.len(), expected.len(), "{path}: array lengths differ");
                for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
                    assert_json_matches_reference(actual, expected, &format!("{path}[{index}]"));
                }
            }
            (Value::Object(actual), Value::Object(expected)) => {
                assert_eq!(actual.len(), expected.len(), "{path}: object fields differ");
                for (key, expected) in expected {
                    let actual = actual
                        .get(key)
                        .unwrap_or_else(|| panic!("{path}.{key}: Rust output is missing field"));
                    assert_json_matches_reference(actual, expected, &format!("{path}.{key}"));
                }
            }
            _ => assert_eq!(actual, expected, "{path}"),
        }
    }

    fn row(id: &str, ra: f64, dec: f64) -> Row {
        HashMap::from([
            ("source_id".into(), id.into()),
            ("ra".into(), ra.to_string()),
            ("dec".into(), dec.to_string()),
            ("phot_g_mean_mag".into(), "12".into()),
        ])
    }

    fn reference_row(id: &str, ra: &str, dec: &str, magnitude: &str) -> Row {
        HashMap::from([
            ("source_id".into(), id.into()),
            ("ra_deg".into(), ra.into()),
            ("dec_deg".into(), dec.into()),
            ("phot_g_mean_mag".into(), magnitude.into()),
        ])
    }

    fn minimal_reference_row(id: &str, ra: &str, dec: &str) -> Row {
        HashMap::from([
            ("source_id".into(), id.into()),
            ("ra_deg".into(), ra.into()),
            ("dec_deg".into(), dec.into()),
        ])
    }

    fn raw_gaia_response_fixture_row() -> Row {
        let mut reader = csv::ReaderBuilder::new().from_reader(
            include_bytes!("../tests/fixtures/gaia_remote_parity/response.csv").as_slice(),
        );
        let headers = reader
            .headers()
            .expect("frozen Gaia response has a CSV header")
            .clone();
        let record = reader
            .records()
            .next()
            .expect("frozen Gaia response has one data row")
            .expect("frozen Gaia response row is valid CSV");
        headers
            .iter()
            .zip(record.iter())
            .map(|(name, value)| (name.to_owned(), value.to_owned()))
            .collect()
    }

    fn options() -> Options {
        Options {
            ra: 359.8,
            dec: 70.0,
            radius: 0.2,
            out: None,
            tile_radius: 0.2,
            limit: Some(2),
            strategy: "auto".into(),
            min_g_mag: None,
            max_g_mag: None,
            include_model: false,
            max_depth: 2,
            max_queries: 100,
            saturated_policy: "subdivide".into(),
            timeout: 30.0,
            endpoint: "http://127.0.0.1".into(),
            dry_run: false,
        }
    }

    #[test]
    fn planner_and_adaptive_query_match_frozen_python_reference() {
        let reference: Value = serde_json::from_str(include_str!(
            "../tests/fixtures/gaia_tiled_parity/reference.json"
        ))
        .expect("frozen Python reference JSON is valid");
        assert_eq!(
            reference["provenance"]["source_sha256"],
            "ec7508e0e05bc2445ded5172f789ddd82c135589466f9181b832947cee2e2a00"
        );
        let planning = &reference["planning"];
        let planning_cases = [
            ("small_circle_ra_360", 360.0, -1.0, 0.25, 0.5, "auto"),
            ("grid_ra_seam_high_dec", 359.8, 70.0, 2.0, 0.5, "grid"),
            ("grid_north_pole", 37.0, 88.0, 1.5, 0.6, "grid"),
        ];
        for (name, ra, dec, radius, tile_radius, strategy) in planning_cases {
            let actual = plan_gaia_tiles(ra, dec, radius, tile_radius, strategy)
                .expect("frozen planning case remains valid")
                .iter()
                .map(GaiaTile::value)
                .collect::<Vec<_>>();
            assert_json_matches_reference(
                &json!(actual),
                &planning[name],
                &format!("planning.{name}"),
            );
        }

        let mut adaptive_options = options();
        adaptive_options.ra = 359.8;
        adaptive_options.dec = 70.0;
        adaptive_options.radius = 0.2;
        adaptive_options.tile_radius = 0.2;
        adaptive_options.limit = Some(4);
        adaptive_options.strategy = "small_circle".into();
        adaptive_options.max_depth = 1;
        adaptive_options.max_queries = 100;
        let mut calls = 0usize;
        let result = run_with_query(
            &adaptive_options,
            |tile, _| {
                calls += 1;
                if tile.depth == 0 {
                    Ok(vec![
                        reference_row("10", "359.8", "70.0", "12"),
                        reference_row("10", "359.8", "70.0", "99"),
                        reference_row("20", "359.85", "70.02", "13"),
                        reference_row("30", "5.0", "0.0", "14"),
                    ])
                } else {
                    Ok(vec![reference_row("10", "359.8", "70.0", "12")])
                }
            },
            false,
        )
        .expect("frozen adaptive query completes");
        assert_eq!(calls, reference["query_result"]["queried_tile_count"]);
        assert_json_matches_reference(&result.value(), &reference["query_result"], "query_result");
        let mut cli_options = options();
        cli_options.ra = 45.5;
        cli_options.dec = -12.25;
        cli_options.radius = 0.1;
        cli_options.tile_radius = 1.0;
        cli_options.limit = Some(10);
        cli_options.strategy = "auto".into();
        cli_options.max_queries = 10_000;
        cli_options.endpoint = gaia_remote::DEFAULT_GAIA_TAP_SYNC_URL.into();
        let cli_result = run_with_query(
            &cli_options,
            |_, _| Ok(vec![raw_gaia_response_fixture_row()]),
            false,
        )
        .expect("CLI fixture query completes with a raw Gaia CSV row");
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock is after Unix epoch")
            .as_nanos();
        let csv_path = std::env::temp_dir().join(format!(
            "rst19-gaia-tiled-parity-{}-{nonce}.csv",
            std::process::id()
        ));
        cli_options.out = Some(csv_path.clone());
        let (written_csv, written_metadata) = write_artifacts(&cli_result, &cli_options)
            .expect("native CLI artifact writer accepts the frozen Gaia response");
        assert_eq!(
            fs::read(&written_csv).expect("native CSV artifact can be read"),
            include_bytes!("../tests/fixtures/gaia_tiled_parity/catalog.csv")
        );
        let actual_metadata: Value = serde_json::from_slice(
            &fs::read(&written_metadata).expect("native metadata artifact can be read"),
        )
        .expect("native metadata artifact is JSON");
        let expected_metadata: Value = serde_json::from_str(include_str!(
            "../tests/fixtures/gaia_tiled_parity/metadata.json"
        ))
        .expect("frozen Python CLI metadata is JSON");
        assert_json_matches_reference(&actual_metadata, &expected_metadata, "cli_metadata");
        fs::remove_file(written_csv).expect("remove native CSV test artifact");
        fs::remove_file(written_metadata).expect("remove native metadata test artifact");

        let mut unknown_options = options();
        unknown_options.ra = 2.0;
        unknown_options.dec = 3.0;
        unknown_options.radius = 0.05;
        unknown_options.tile_radius = 0.1;
        unknown_options.limit = None;
        unknown_options.strategy = "small_circle".into();
        unknown_options.max_depth = 0;
        unknown_options.max_queries = 10;
        unknown_options.saturated_policy = "record".into();
        let unknown_result = run_with_query(
            &unknown_options,
            |_, _| Ok(vec![minimal_reference_row("9", "2.0", "3.0")]),
            false,
        )
        .expect("unknown-limit query returns its auditable partial result");
        assert_json_matches_reference(
            &unknown_result.value(),
            &reference["unknown_limit_result"],
            "unknown_limit_result",
        );

        let mut error_options = unknown_options.clone();
        error_options.limit = Some(10);
        let error_result = run_with_query(
            &error_options,
            |_, _| Err("synthetic TAP failure".into()),
            false,
        )
        .expect("non-raising query mode returns its auditable error result");
        // Python prefixes the exception class to query errors; the Rust
        // transport keeps the source message. Normalize only that wrapper text.
        let mut normalized_error_reference = reference["query_error_result"].clone();
        normalized_error_reference["tile_records"][0]["error"] = json!("synthetic TAP failure");
        normalized_error_reference["errors"][0] = json!("tile-00000: synthetic TAP failure");
        assert_json_matches_reference(
            &error_result.value(),
            &normalized_error_reference,
            "query_error_result",
        );
    }

    #[test]
    fn small_circle_and_ra_wrap_plans_are_stable() {
        let tiles = plan_gaia_tiles(360.0, 89.0, 0.1, 0.5, "auto").unwrap();
        assert_eq!(tiles.len(), 1);
        assert_eq!(tiles[0].ra_deg, 0.0);
        let grid = plan_gaia_tiles(359.8, 70.0, 2.0, 0.5, "grid").unwrap();
        assert!(grid.len() > 1);
        assert!(grid
            .iter()
            .all(|tile| tile.ra_deg >= 0.0 && tile.ra_deg < 360.0));
        assert!(grid.windows(2).all(|pair| angular_distance_deg(
            359.8,
            70.0,
            pair[0].ra_deg,
            pair[0].dec_deg
        ) <= angular_distance_deg(
            359.8,
            70.0,
            pair[1].ra_deg,
            pair[1].dec_deg
        ) + 1e-12));
    }
    #[test]
    fn saturated_parent_subdivides_and_tracks_duplicates() {
        let options = options();
        let mut calls = 0;
        let result = run_with_query(
            &options,
            |tile, _| {
                calls += 1;
                if tile.depth == 0 {
                    Ok(vec![row("1", 359.8, 70.0), row("2", 359.81, 70.01)])
                } else {
                    Ok(vec![row("1", 359.8, 70.0)])
                }
            },
            false,
        )
        .unwrap();
        assert!(calls > 1);
        assert!(result.complete);
        assert_eq!(result.rows.len(), 2);
        assert_eq!(result.duplicate_count, calls - 1);
        assert_eq!(result.records[0].status, "saturated_subdivided");
    }
    #[test]
    fn unknown_remote_limit_is_never_reported_complete() {
        let mut options = options();
        options.limit = None;
        let result =
            run_with_query(&options, |_, _| Ok(vec![row("1", 359.8, 70.0)]), false).unwrap();
        assert!(!result.complete);
        assert_eq!(result.records[0].truncation_status, "unknown_limit");
        assert_eq!(result.truncated_tile_ids(), vec!["tile-00000"]);
    }
    #[test]
    fn dry_run_does_not_make_network_request() {
        let mut options = options();
        options.radius = 2.0;
        options.tile_radius = 0.5;
        options.strategy = "grid".into();
        options.dry_run = true;
        let value = dry_run_value(&options).unwrap();
        assert_eq!(value["network"], false);
        let reference: Value = serde_json::from_str(include_str!(
            "../tests/fixtures/gaia_tiled_parity/reference.json"
        ))
        .unwrap();
        assert_json_matches_reference(&value, &reference["dry_run_result"], "dry_run_result");
    }
}
