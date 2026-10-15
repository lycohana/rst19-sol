use rst19_core::{
    detection::DetectionOptions,
    feature_audit_replicates,
    injection::{self, Mode, Options},
    measurement::Detection,
};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{collections::BTreeSet, fs, path::PathBuf};

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}
fn output(name: &str) -> PathBuf {
    root()
        .join("target/test-artifacts")
        .join(format!("injection-{name}-{}", std::process::id()))
}
fn cleanup(path: &PathBuf) {
    assert!(path
        .canonicalize()
        .unwrap()
        .starts_with(root().join("target/test-artifacts").canonicalize().unwrap()));
    fs::remove_dir_all(path).unwrap();
}
fn contract(row: &Value, class: &str) {
    let bytes = fs::read(root().join("tests/fixtures/injection_contract.json")).unwrap();
    let bytes = bytes.strip_prefix(&[0xef, 0xbb, 0xbf]).unwrap_or(&bytes);
    let expected: Value = serde_json::from_slice(bytes).unwrap();
    let fields = expected["rows"][class]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap())
        .collect::<BTreeSet<_>>();
    let actual = row
        .as_object()
        .unwrap()
        .keys()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    assert_eq!(actual, fields, "legacy row contract {class}");
}

#[test]
fn injection_parser_keeps_mode_defaults_array_options_and_rejects_conflicting_experiments() {
    let args = |values: &[&str]| values.iter().map(|v| v.to_string()).collect::<Vec<_>>();
    let synthetic = injection::parse_args(&[]).unwrap();
    assert_eq!(
        synthetic.peaks,
        vec![8., 12., 16., 24., 36., 56., 84., 128.]
    );
    let pair = injection::parse_args(&args(&[
        "--real-fits",
        "a.fits",
        "--pair-flux-ratio-audit",
        "--total-peak-levels",
        "100",
        "200",
        "--secondary-ratios",
        "1",
        "0.25",
    ]))
    .unwrap();
    assert_eq!(pair.mode, Mode::PairRatio);
    assert_eq!(pair.peaks, vec![100., 200.]);
    assert_eq!(pair.ratios, vec![1., 0.25]);
    let crowded =
        injection::parse_args(&args(&["--real-fits=a.fits", "--crowded-blend-audit"])).unwrap();
    assert_eq!(crowded.normalization, "integrated_excess");
    for bad in [
        &["--compare-proposals"][..],
        &[
            "--real-fits",
            "a.fits",
            "--stratified-real",
            "--compare-proposals",
        ],
        &["--peak-levels", "-1"],
        &["--secondary-ratios", "1.2"],
        &["--strata", "blank", "blank"],
    ] {
        assert!(injection::parse_args(&args(bad)).is_err());
    }
}

#[test]
fn one_candidate_can_hit_only_one_injected_endpoint() {
    let value = json!({"detection_id":1,"x":10.,"y":10.,"peak":100.,"flux":100.,"background":10.,"noise":1.,"snr":10.,"flags":[],"quality_passed":true,"proposal_methods":[],"proposal_scales":[],"repeated_code_values":[]});
    let source: Detection = serde_json::from_value(value).unwrap();
    assert_eq!(
        injection::match_flags(&[[9., 10.], [11., 10.]], &[source], 3.),
        vec![true, false]
    );
}

#[test]
fn single_feature_controls_match_frozen_legacy_one_trial_aggregates() {
    let mut options = DetectionOptions::default();
    options.min_distance = 4;
    options.background_box_size = 32;
    options.proposal_mode = "hybrid".into();
    options.measurement.psf_fwhm = 2.;
    let rows = feature_audit_replicates::run_feature_audit(&options, 19019).unwrap();
    let expected: Value = serde_json::from_slice(
        &fs::read(root().join("tests/fixtures/feature_audit_replicates_parity/reference.json"))
            .unwrap(),
    )
    .unwrap();
    assert_eq!(rows.len(), 11);
    for row in rows {
        let actual = serde_json::to_value(&row).unwrap();
        contract(&actual, "FeatureAuditRow");
        let reference = expected["rows"]
            .as_array()
            .unwrap()
            .iter()
            .find(|v| v["scenario"] == row.scenario)
            .unwrap();
        for key in [
            "feature_class",
            "scenario",
            "control_type",
            "truth_count",
            "candidate_true_hits",
            "quality_true_hits",
            "candidate_recall",
            "quality_recall",
            "note",
        ] {
            assert_eq!(actual[key], reference[key], "{}.{}", row.scenario, key);
        }
        // Replicate aggregates count trials with each flag, while the single audit counts detections.
        let flag_names = |value: &Value| {
            value
                .as_str()
                .unwrap_or("")
                .split('|')
                .filter(|v| !v.is_empty())
                .map(|v| v.split(':').next().unwrap().to_owned())
                .collect::<BTreeSet<_>>()
        };
        assert_eq!(
            flag_names(&actual["nearby_flags"]),
            flag_names(&reference["nearby_flags"])
        );
        for (key, mean) in [
            ("nearby_candidate_count", "mean_nearby_candidate_count"),
            ("nearby_quality_count", "mean_nearby_quality_count"),
            ("candidate_count", "mean_candidate_count"),
            ("quality_count", "mean_quality_count"),
        ] {
            assert_eq!(
                actual[key].as_f64(),
                reference[mean].as_f64(),
                "{}.{}",
                row.scenario,
                key
            );
        }
    }
}

#[test]
fn synthetic_trial_exports_frozen_pixels_truth_coordinates_and_recovers_strong_sources() {
    let path = output("synthetic");
    let options = Options {
        out_dir: path.clone(),
        trials: 1,
        sources: 2,
        peaks: vec![128.],
        ..Default::default()
    };
    let result = injection::run(&options).unwrap();
    contract(&result["rows"][0], "InjectionRecoveryRow");
    assert_eq!(result["rows"][0]["candidate_recovered_count"], 2);
    assert_eq!(result["rows"][0]["quality_recovered_count"], 2);
    let manifest: Value =
        serde_json::from_slice(&fs::read(path.join("manifest.json")).unwrap()).unwrap();
    assert_eq!(
        manifest["frozen_inputs"][0]["truth_positions_px"]
            .as_array()
            .unwrap()
            .len(),
        2
    );
    assert!(path.join("injection_recovery.csv").is_file());
    assert!(path.join("injection_recovery.png").is_file());
    cleanup(&path);
}

#[test]
fn real_modes_preserve_legacy_row_contracts_and_do_not_modify_original_fits() {
    let source = root().join("tests/fixtures/contaminated_pair_injection_parity.fits");
    let before = Sha256::digest(fs::read(&source).unwrap());
    for (mode, class, name) in [
        (Mode::Real, "RealBackgroundInjectionRow", "real"),
        (
            Mode::Stratified,
            "StratifiedRealBackgroundInjectionRow",
            "stratified",
        ),
        (Mode::Comparison, "ProposalModeComparisonRow", "comparison"),
        (Mode::PairRatio, "PairFluxRatioAuditRow", "pair"),
        (Mode::Crowded, "CrowdedBlendAuditRow", "crowded"),
    ] {
        let path = output(name);
        let options = Options {
            mode,
            real_fits: Some(source.clone()),
            out_dir: path.clone(),
            trials: 1,
            sources: 1,
            pairs: 1,
            groups: 1,
            peaks: vec![128.],
            ratios: vec![0.5],
            blend_sizes: vec![2],
            blend_separations: vec![6.],
            blend_levels: vec![128.],
            strata: vec!["blank".into()],
            ..Default::default()
        };
        let result = injection::run(&options).unwrap();
        assert_eq!(result["row_count"], 1);
        contract(&result["rows"][0], class);
        assert!(path.join("manifest.json").is_file());
        cleanup(&path);
    }
    assert_eq!(Sha256::digest(fs::read(source).unwrap()), before);
}
