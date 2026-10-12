use rst19_core::compatible_spatial::CompatibleKdTree;
use serde_json::Value;
#[test]
fn compact_median_tree_and_first_visited_ties_match_scipy() {
    let fixtures: Value = serde_json::from_str(include_str!("fixtures/spatial.json")).unwrap();
    for case in fixtures.as_array().unwrap() {
        let points: Vec<_> = case["points"]
            .as_array()
            .unwrap()
            .iter()
            .map(|p| [p[0].as_f64().unwrap(), p[1].as_f64().unwrap()])
            .collect();
        let tree = CompatibleKdTree::new(&points).unwrap();
        for (i, q) in case["queries"].as_array().unwrap().iter().enumerate() {
            let (index, distance) = tree
                .nearest([q[0].as_f64().unwrap(), q[1].as_f64().unwrap()])
                .unwrap();
            assert_eq!(index, case["expected"][i][0].as_u64().unwrap() as usize);
            assert!((distance - case["expected"][i][1].as_f64().unwrap()).abs() < 1e-10);
        }
    }
    assert!(CompatibleKdTree::new(&[])
        .unwrap()
        .nearest([0.0, 0.0])
        .is_none());
    assert!(CompatibleKdTree::new(&[[f64::NAN, 0.0]]).is_err());
}
