use rst19_core::{
    catalog::CatalogSource,
    coordinates::TangentPlaneWcs,
    matching::CatalogMatch,
    sequence::{track_detections, TrackSequenceOptions},
    sequence_assignment::{global_frame_assignments, TrackPosition},
    sequence_motion::{
        derive_motion_displacement_floor, fit_track_motion, MotionMetric, TrackObservation,
    },
    sequence_registration::estimate_translation,
    wcs_fit::{fit_affine_wcs_from_matches, validate_frame_matches, AffineFitOptions},
};
use serde_json::Value;

fn fixture() -> Value {
    serde_json::from_str(include_str!("fixtures/wcs_sequence.json")).unwrap()
}

fn compare(actual: &Value, expected: &Value, path: &str) {
    match (actual, expected) {
        (Value::Number(a), Value::Number(b)) => {
            let a = a.as_f64().unwrap();
            let b = b.as_f64().unwrap();
            assert!(
                (a - b).abs() <= 1e-7 + 1e-10 * b.abs(),
                "{path}: actual={a}, expected={b}"
            );
        }
        (Value::Array(a), Value::Array(b)) => {
            assert_eq!(a.len(), b.len(), "{path}");
            for (index, (a, b)) in a.iter().zip(b).enumerate() {
                compare(a, b, &format!("{path}[{index}]"));
            }
        }
        (Value::Object(a), Value::Object(b)) => {
            assert_eq!(a.len(), b.len(), "{path}");
            for (key, b) in b {
                compare(&a[key], b, &format!("{path}.{key}"));
            }
        }
        _ => assert_eq!(actual, expected, "{path}"),
    }
}

#[test]
fn affine_wcs_fit_and_rejection_match_python_reference() {
    let fixture = fixture();
    for case in fixture["fit_cases"].as_array().unwrap() {
        let matches: Vec<CatalogMatch> = serde_json::from_value(case["matches"].clone()).unwrap();
        let catalog: Vec<CatalogSource> = serde_json::from_value(case["catalog"].clone()).unwrap();
        let reference: TangentPlaneWcs =
            serde_json::from_value(case["reference_wcs"].clone()).unwrap();
        let options: AffineFitOptions =
            serde_json::from_value(case["fit_options"].clone()).unwrap();
        let actual = fit_affine_wcs_from_matches(
            &matches,
            &catalog,
            &reference,
            case["epoch"].as_f64(),
            options,
        );
        if case["expected"].is_null() {
            assert_eq!(
                actual.unwrap_err(),
                case["error"].as_str().unwrap(),
                "{}",
                case["label"]
            );
        } else {
            compare(
                &actual.unwrap().as_dict(),
                &case["expected"],
                case["label"].as_str().unwrap(),
            );
        }
    }
}

#[test]
fn per_frame_wcs_validation_report_matches_python_reference() {
    let case = fixture()["validation_case"].clone();
    let matches: Vec<Vec<CatalogMatch>> =
        serde_json::from_value(case["frame_matches"].clone()).unwrap();
    let catalog: Vec<CatalogSource> = serde_json::from_value(case["catalog"].clone()).unwrap();
    let reference: TangentPlaneWcs = serde_json::from_value(case["reference_wcs"].clone()).unwrap();
    let paths: Vec<String> = serde_json::from_value(case["frame_paths"].clone()).unwrap();
    let options: AffineFitOptions = serde_json::from_value(case["fit_options"].clone()).unwrap();
    let actual = validate_frame_matches(
        &matches,
        &catalog,
        &reference,
        Some(&paths),
        case["epoch"].as_f64(),
        options,
    )
    .unwrap();
    compare(&actual, &case["expected"], "validation");
}

#[test]
fn time_aware_track_fit_matches_python_reference() {
    let fixture = fixture();
    for case in fixture["motion_cases"].as_array().unwrap() {
        let points: Vec<TrackObservation> = serde_json::from_value(case["points"].clone()).unwrap();
        let times: Option<Vec<f64>> = if case["frame_times"].is_null() {
            None
        } else {
            Some(serde_json::from_value(case["frame_times"].clone()).unwrap())
        };
        let actual = fit_track_motion(&points, times.as_deref())
            .unwrap()
            .as_dict();
        compare(&actual, &case["expected"], case["label"].as_str().unwrap());
    }
}

#[test]
fn robust_registration_motion_floor_matches_python_reference() {
    for case in fixture()["floor_cases"].as_array().unwrap() {
        let metrics: Vec<MotionMetric> = serde_json::from_value(case["metrics"].clone()).unwrap();
        let actual = derive_motion_displacement_floor(
            &metrics,
            case["required_presence"].as_u64().unwrap() as usize,
            case["user_floor"].as_f64().unwrap(),
            case["max_motion_fit_rms_px"].as_f64().unwrap(),
            case["affine_frame_count"].as_u64().unwrap() as usize,
        )
        .unwrap();
        compare(
            &serde_json::json!(actual),
            &case["expected"],
            case["label"].as_str().unwrap(),
        );
    }
}

#[test]
fn robust_frame_translation_matches_python_reference() {
    for case in fixture()["translation_cases"].as_array().unwrap() {
        let previous: Vec<rst19_core::measurement::Detection> =
            serde_json::from_value(case["previous"].clone()).unwrap();
        let current: Vec<rst19_core::measurement::Detection> =
            serde_json::from_value(case["current"].clone()).unwrap();
        let actual =
            estimate_translation(&previous, &current, case["radius_px"].as_f64().unwrap()).unwrap();
        compare(
            &serde_json::json!({"translation_px":actual}),
            &case["expected"],
            case["label"].as_str().unwrap(),
        );
    }
}

#[test]
fn sparse_global_frame_assignment_matches_python_reference() {
    for case in fixture()["assignment_cases"].as_array().unwrap() {
        let tracks: Vec<(i64, f64, f64)> =
            serde_json::from_value(case["track_positions"].clone()).unwrap();
        let tracks: Vec<_> = tracks
            .iter()
            .map(|(track_id, x, y)| TrackPosition {
                track_id: *track_id,
                x: *x,
                y: *y,
            })
            .collect();
        let points: Vec<[f64; 2]> = serde_json::from_value(case["current_points"].clone()).unwrap();
        let allowed_pairs: Option<Vec<(i64, usize)>> = if case["allowed_pairs"].is_null() {
            None
        } else {
            Some(serde_json::from_value(case["allowed_pairs"].clone()).unwrap())
        };
        let cost_rows: Option<Vec<(i64, usize, f64)>> = if case["pair_costs"].is_null() {
            None
        } else {
            Some(serde_json::from_value(case["pair_costs"].clone()).unwrap())
        };
        let allowed = |track_id, source_index, _distance| {
            allowed_pairs
                .as_ref()
                .is_none_or(|pairs| pairs.contains(&(track_id, source_index)))
        };
        let cost = |track_id, source_index, _distance| {
            cost_rows
                .as_ref()
                .and_then(|rows| {
                    rows.iter()
                        .find(|row| row.0 == track_id && row.1 == source_index)
                })
                .map(|row| row.2)
                .unwrap_or(0.0)
        };
        let result = global_frame_assignments(
            &tracks,
            &points,
            case["link_radius_px"].as_f64().unwrap(),
            allowed_pairs
                .as_ref()
                .map(|_| &allowed as &dyn Fn(i64, usize, f64) -> bool),
            cost_rows
                .as_ref()
                .map(|_| &cost as &dyn Fn(i64, usize, f64) -> f64),
        )
        .unwrap();
        compare(
            &serde_json::json!({"assignments":result.iter().map(|row|Value::Array(vec![
                serde_json::json!(row.distance_px),
                serde_json::json!(row.track_id),
                serde_json::json!(row.source_index),
            ])).collect::<Vec<_>>()}),
            &case["expected"],
            case["label"].as_str().unwrap(),
        );
    }
}

#[test]
fn end_to_end_sequence_tracking_matches_python_reference() {
    for case in fixture()["tracking_cases"].as_array().unwrap() {
        let frames: Vec<Vec<rst19_core::measurement::Detection>> =
            serde_json::from_value(case["frames"].clone()).unwrap();
        let options: TrackSequenceOptions =
            serde_json::from_value(case["options"].clone()).unwrap();
        let actual = track_detections(&frames, &options).unwrap();
        compare(&actual, &case["expected"], case["label"].as_str().unwrap());
    }
}
