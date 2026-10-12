use rst19_core::assignment::{assign_candidates, linear_sum_assignment, CandidateEdge};
use serde_json::Value;

#[test]
fn rectangular_assignment_matches_scipy_including_ties_and_transposition() {
    let fixture: Value = serde_json::from_str(include_str!("fixtures/assignment.json")).unwrap();
    for case in fixture["matrices"].as_array().unwrap() {
        let cost: Vec<f64> = serde_json::from_value(case["cost"].clone()).unwrap();
        let expected: Vec<(usize, usize)> =
            serde_json::from_value(case["expected"].clone()).unwrap();
        let actual = linear_sum_assignment(
            &cost,
            case["rows"].as_u64().unwrap() as usize,
            case["columns"].as_u64().unwrap() as usize,
        )
        .unwrap();
        assert_eq!(actual, expected, "{case}");
    }
    assert!(linear_sum_assignment(&[f64::INFINITY], 1, 1).is_err());
    assert!(linear_sum_assignment(&[f64::NAN], 1, 1).is_err());
    assert!(linear_sum_assignment(&[0.], 2, 2).is_err());
}

#[test]
fn sparse_global_and_greedy_assignments_match_python() {
    let fixture: Value = serde_json::from_str(include_str!("fixtures/assignment.json")).unwrap();
    for (i, case) in fixture["graphs"].as_array().unwrap().iter().enumerate() {
        let edges: Vec<CandidateEdge> = serde_json::from_value(case["edges"].clone()).unwrap();
        let expected: Vec<CandidateEdge> =
            serde_json::from_value(case["expected"].clone()).unwrap();
        let actual = assign_candidates(
            &edges,
            case["detection_count"].as_u64().unwrap() as usize,
            case["catalog_count"].as_u64().unwrap() as usize,
            case["radius"].as_f64().unwrap(),
            case["mode"].as_str().unwrap(),
        )
        .unwrap();
        assert_eq!(actual, expected, "graph {i}");
    }
    assert!(assign_candidates(&[CandidateEdge(1., 1, 0)], 1, 1, 3., "global").is_err());
}
