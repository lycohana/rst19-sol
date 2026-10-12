use rst19_core::filtering;
use rst19_core::{
    coordinates::{auxiliary_boresight_radec, quaternion_to_active_matrix, TangentPlaneWcs},
    fits,
    models::AuxiliaryData,
    statistics,
};
use serde_json::Value;

fn fixture() -> Value {
    serde_json::from_str(include_str!("fixtures/reference.json")).unwrap()
}
fn nums(v: &Value) -> Vec<f64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|n| n.as_f64().unwrap())
        .collect()
}
fn approx(actual: f64, expected: f64) {
    let tolerance = 1e-10 + expected.abs() * 5e-13;
    assert!(
        (actual - expected).abs() <= tolerance,
        "actual={actual:.17} expected={expected:.17} tolerance={tolerance}"
    );
}
fn compare(actual: &[f64], expected: &Value) {
    let expected = nums(expected);
    assert_eq!(actual.len(), expected.len());
    for (a, b) in actual.iter().zip(expected) {
        approx(*a, b);
    }
}

#[test]
fn cards_and_exposure_units_match_python() {
    let f = fixture();
    for case in f["cards"].as_array().unwrap() {
        assert_eq!(
            fits::parse_card_value(case["input"].as_str().unwrap()),
            case["expected"]
        );
    }
    for case in f["exposures"].as_array().unwrap() {
        let header = case["header"].as_object().unwrap();
        approx(
            fits::exposure_seconds(header, 1.0).unwrap(),
            case["seconds"].as_f64().unwrap(),
        );
        match case["milliseconds"].as_f64() {
            Some(expected) => approx(
                fits::exposure_milliseconds(header).unwrap().unwrap(),
                expected,
            ),
            None => assert!(fits::exposure_milliseconds(header).unwrap().is_none()),
        }
    }
    assert!(fits::exposure_seconds(&serde_json::Map::new(), 0.0).is_err());
}

#[test]
fn clipped_statistics_and_interpolation_match_python() {
    let f = fixture();
    for case in f["statistics"].as_array().unwrap() {
        let values = nums(&case["values"]);
        let mask: Vec<_> = case["mask"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_bool().unwrap())
            .collect();
        let (center, noise) = statistics::sigma_clipped_stats(
            &values,
            Some(&mask),
            3.0,
            4,
            case["sample_limit"].as_u64().unwrap() as usize,
        )
        .unwrap();
        approx(center, case["center"].as_f64().unwrap());
        approx(noise, case["noise"].as_f64().unwrap());
    }
    for case in f["grids"].as_array().unwrap() {
        let shape = nums(&case["shape"]);
        let target = nums(&case["target"]);
        let result = statistics::resize_grid(
            &nums(&case["values"]),
            shape[0] as usize,
            shape[1] as usize,
            target[0] as usize,
            target[1] as usize,
        )
        .unwrap();
        compare(&result, &case["expected"]);
    }
    for case in f["backgrounds"].as_array().unwrap() {
        let shape = nums(&case["shape"]);
        let mask: Vec<_> = case["mask"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_bool().unwrap())
            .collect();
        let (b, n) = statistics::local_background_rms(
            &nums(&case["values"]),
            &mask,
            shape[0] as usize,
            shape[1] as usize,
            statistics::BackgroundOptions {
                box_size: 16,
                fallback: None,
                iterations: 4,
                sample_limit: 100_000,
            },
        )
        .unwrap();
        compare(&b, &case["background"]);
        compare(&n, &case["noise"]);
    }
}

#[test]
fn tan_projection_and_quaternion_geometry_match_python() {
    let f = fixture();
    for case in f["projections"].as_array().unwrap() {
        let wcs: TangentPlaneWcs = serde_json::from_value(case["wcs"].clone()).unwrap();
        let wcs = wcs.validated().unwrap();
        for (i, point) in case["world"].as_array().unwrap().iter().enumerate() {
            let p = nums(point);
            let (x, y) = wcs.world_to_pixel(p[0], p[1]);
            compare(&[x, y], &case["world_to_pixel"][i]);
            let (e, n) = wcs.world_to_tangent_arcsec(p[0], p[1]);
            compare(&[e, n], &case["tangent"][i]);
        }
        for (i, point) in case["pixels"].as_array().unwrap().iter().enumerate() {
            let p = nums(point);
            let (ra, dec) = wcs.pixel_to_world(p[0], p[1]);
            compare(&[ra, dec], &case["pixel_to_world"][i]);
        }
    }
    for case in f["quaternions"].as_array().unwrap() {
        let q = nums(&case["q"]);
        let matrix = quaternion_to_active_matrix(&q).unwrap();
        for (i, row) in matrix.iter().enumerate() {
            compare(row, &case["matrix"][i]);
        }
        let mut aux = AuxiliaryData { values: [0.0; 26] };
        aux.values[10..14].copy_from_slice(&q);
        let (ra, dec) = auxiliary_boresight_radec(&aux, &[0.0, -1.0, 0.0]).unwrap();
        compare(&[ra, dec], &case["boresight"]);
    }
    assert!(quaternion_to_active_matrix(&[0.0; 4]).is_err());
    assert!(quaternion_to_active_matrix(&[1.0, 2.0, 3.0]).is_err());
}

#[test]
fn masked_filters_maxima_and_line_masks_match_scipy() {
    let f = fixture();
    for case in f["filters"].as_array().unwrap() {
        let shape = nums(&case["shape"]);
        let values = nums(&case["values"]);
        let valid: Vec<_> = case["valid"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_bool().unwrap())
            .collect();
        let (h, w) = (shape[0] as usize, shape[1] as usize);
        let fast = case["fast"].as_bool().unwrap();
        let (g, mode) =
            filtering::masked_gaussian_response(&values, &valid, h, w, 1.3, fast).unwrap();
        compare(&g, &case["gaussian"]);
        assert_eq!(mode, case["gaussian_mode"].as_str().unwrap());
        let (s, mode) = filtering::masked_starlet_smooth(&values, &valid, h, w, 2, fast).unwrap();
        compare(&s, &case["starlet"]);
        assert_eq!(mode, case["starlet_mode"].as_str().unwrap());
        compare(
            &filtering::maximum_filter(&values, h, w, 3).unwrap(),
            &case["maximum"],
        );
        let line = filtering::linear_artifact_mask(
            &values,
            &vec![3.0; values.len()],
            &valid,
            h,
            w,
            4.0,
            3.0,
        )
        .unwrap();
        assert_eq!(serde_json::to_value(line).unwrap(), case["line_mask"]);
    }
}

#[test]
fn peak_window_nms_and_multichannel_merging_match_python() {
    let f = fixture();
    for case in f["proposals"].as_array().unwrap() {
        let shape = nums(&case["shape"]);
        let values = nums(&case["values"]);
        let options = filtering::ProposalOptions {
            threshold_sigma: 4.0,
            min_distance: case["distance"].as_u64().unwrap() as usize,
            scale_fwhm: 3.0,
            method: "gaussian".into(),
        };
        let proposals = filtering::response_proposals(
            &values,
            &vec![true; values.len()],
            shape[0] as usize,
            shape[1] as usize,
            &options,
            None,
        )
        .unwrap();
        assert_eq!(serde_json::to_value(proposals).unwrap(), case["expected"]);
    }
    let primary: Vec<filtering::Proposal> =
        serde_json::from_value(f["merged"]["primary"].clone()).unwrap();
    let extras: Vec<filtering::Proposal> =
        serde_json::from_value(f["merged"]["extras"].clone()).unwrap();
    let merged = serde_json::to_value(filtering::merge_proposals(&primary, &extras, 4)).unwrap();
    assert_eq!(merged, f["merged"]["expected"]);
}
