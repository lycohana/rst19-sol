use rst19_core::innovation_photometry::{
    absolute_magnitude_evidence, photometric_evidence_section,
};
use serde_json::{json, Value};

fn strict_row() -> Value {
    json!({"source_id":"strict-source","status":"CALIBRATED","m_cal":14.5,"photometric_system":"Gaia Vega","photometric_band":"G",
        "absolute_magnitude":{"value":4.2,"status":"VALID_MODEL_DISTANCE","is_strict":true,"distance_source":"Gaia DR3 GSP-Phot","distance_pc":100.0,"distance_lower_pc":95.0,"distance_upper_pc":106.0,"distance_interval_status":"PROVIDED","extinction_mag":0.2,"extinction_band":"G","extinction_system":"Gaia","extinction_source":"Gaia DR3 GSP-Phot: ag_gspphot"}})
}

#[test]
fn strict_and_diagnostic_values_remain_separate_and_invalid_bounds_reject_strictness() {
    let mut row = strict_row();
    let parsed = absolute_magnitude_evidence(&row);
    assert_eq!(parsed["strict_value"], 4.2);
    assert_eq!(parsed["label"], "M_G");
    assert_eq!(parsed["system"], "Gaia");
    assert!(parsed["strict_reasons"].as_array().unwrap().is_empty());
    let section = photometric_evidence_section(&json!({"source_photometry":[row]}));
    assert_eq!(section["strict_absolute_magnitude"]["count"], 1);
    assert_eq!(
        section["summary"]["reported_strict_absolute_magnitude_count"],
        1
    );
    row["absolute_magnitude"]["distance_lower_pc"] = json!(105.0);
    let parsed = absolute_magnitude_evidence(&row);
    assert_eq!(parsed["diagnostic_value"], 4.2);
    assert!(parsed["strict_value"].is_null());
    assert!(parsed["strict_reasons"]
        .as_array()
        .unwrap()
        .contains(&json!("DISTANCE_INTERVAL_STATUS_INVALID")));
}

#[test]
fn legacy_absolute_claim_and_relative_scale_do_not_become_strict_absolute_evidence() {
    let section = photometric_evidence_section(
        &json!({"photometric_quality_report":{"level_counts":{"ABSOLUTE_ELIGIBLE":5},"absolute_magnitude_value_count":2},
        "relative_photometry":{"status":"VALID","reference_sample_count":4,"frame_zero_points":{"0":0.0,"1":0.1},"relative_magnitudes":{"star":1.2}}}),
    );
    assert_eq!(section["status"], "AVAILABLE");
    assert_eq!(section["summary"]["reported_absolute_eligible_count"], 5);
    assert_eq!(
        section["summary"]["reported_absolute_magnitude_diagnostic_count"],
        2
    );
    assert_eq!(section["strict_absolute_magnitude"]["count"], 0);
    assert_eq!(
        section["relative_photometry"]["absolute_magnitude_claim"],
        "NOT_DERIVED_FROM_RELATIVE_SCALE"
    );
    assert_eq!(
        photometric_evidence_section(&json!({}))["status"],
        "NOT_PRESENT"
    );
}

#[test]
fn extinction_band_mismatch_and_false_valid_gate_block_formal_absolute_counts() {
    let mut row = strict_row();
    row["absolute_magnitude"]["extinction_band"] = json!("V");
    let parsed = absolute_magnitude_evidence(&row);
    assert_eq!(parsed["gate_status"], "DIAGNOSTIC_ONLY");
    assert!(parsed["strict_reasons"]
        .as_array()
        .unwrap()
        .contains(&json!("EXTINCTION_BAND_MISMATCH")));
    let section = photometric_evidence_section(
        &json!({"photometric_quality_report":{"rows":[strict_row()],"false_valid_gate":{"passed":false}}}),
    );
    assert_eq!(section["status"], "PRESENT_BUT_REJECTED");
    assert_eq!(
        section["summary"]["reported_strict_absolute_magnitude_evidence_count"],
        1
    );
    assert_eq!(
        section["summary"]["reported_strict_absolute_magnitude_count"],
        0
    );
}
