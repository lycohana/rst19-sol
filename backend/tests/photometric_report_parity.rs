use rst19_core::photometric_report::report_from_cli_payload;
use serde_json::Value;

#[test]
fn report_matches_frozen_python_reference_for_all_observability_levels() {
    let input: Value = serde_json::from_str(include_str!(
        "fixtures/photometric_report_four_levels_input.json"
    ))
    .unwrap();
    let expected: Value = serde_json::from_str(include_str!(
        "fixtures/photometric_report_four_levels_expected.json"
    ))
    .unwrap();
    let actual = report_from_cli_payload(&input, None, false).unwrap();
    assert_eq!(actual, expected);
}
