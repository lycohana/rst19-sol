use rst19_core::{
    models::{FitsFrame, PixelData},
    source_pair_audit::{self, Options},
};
use serde_json::json;
use std::path::PathBuf;

fn frame(data: PixelData, width: usize, height: usize) -> FitsFrame {
    FitsFrame {
        path: PathBuf::from("synthetic-code-audit.fits"),
        header: Default::default(),
        data,
        auxiliary: None,
        data_offset: 0,
        width,
        height,
    }
}

#[test]
fn integer_codes_remain_distinct_above_float64_precision_and_mode_ties_choose_smaller() {
    let a = (1i64 << 53) + 1;
    let b = a + 1;
    let f = frame(PixelData::I64(vec![a, b, a, b, -1]), 5, 1);
    let stats = source_pair_audit::raw_region_statistics(&f, [2., 0.], 3);
    assert_eq!(stats["unique"], 3);
    assert_eq!(stats["mode_count"], 2);
    assert_eq!(stats["fixed_minus_one"], 1);
    assert_eq!(stats["mode"], json!(a as f64));
}

#[test]
fn original_aperture_overlap_preserves_background_subtraction_and_missing_flux() {
    let f = frame(PixelData::F64(vec![10.; 25]), 5, 5);
    let stats = source_pair_audit::overlap_statistics(&f, [1., 2.], [3., 2.], 1, Some(8.)).unwrap();
    assert_eq!(stats["secondary_pixels"], 5);
    assert_eq!(stats["shared_pixels"], 1);
    assert_eq!(stats["secondary_net_sum"], 10.);
    assert_eq!(stats["shared_net_sum"], 2.);
    assert_eq!(stats["shared_net_fraction"], 0.2);
    let absent =
        source_pair_audit::overlap_statistics(&f, [30., 30.], [32., 30.], 1, None).unwrap();
    assert_eq!(absent["secondary_pixels"], 0);
    assert!(absent["shared_net_fraction"].is_null());
}

#[test]
fn missing_independent_targets_never_promote_fallback_coordinates_to_pair_evidence() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/contaminated_pair_injection_parity.fits");
    let options = Options {
        primary: [-50., -50.],
        secondary: [-45., -50.],
        ..Default::default()
    };
    let result = source_pair_audit::run(&[path.clone()], &options).unwrap();
    let row = &result["rows"][0];
    assert_eq!(row.as_object().unwrap().len(), 56);
    assert!(row["primary_detection_id"].is_null());
    assert!(row["secondary_detection_id"].is_null());
    for key in [
        "pair_delta_bic_raw",
        "pair_component_snr_raw",
        "pair_delta_bic_range_masked",
        "pair_component_snr_sentinel_masked",
    ] {
        assert!(row[key].is_null());
    }
    let mut invalid = options.clone();
    invalid.shifts = Some(vec![]);
    assert!(source_pair_audit::run(&[path], &invalid).is_err());
}
