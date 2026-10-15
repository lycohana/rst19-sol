use rst19_core::ablation_report;
use std::{fs, path::PathBuf};
#[test]
fn combined_report_escapes_measurements_preserves_existing_content_and_refreshes_once() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("target/test-artifacts")
        .join(format!("ablation-html-{}", std::process::id()));
    fs::create_dir_all(root.join("evidence/point")).unwrap();
    let report = root.join("index.html");
    fs::write(
        &report,
        "<!doctype html><html><head></head><body><p>Existing science</p></body></html>",
    )
    .unwrap();
    fs::write(
        root.join("evidence/point/summary.csv"),
        "version,recall\nA,0.5\n<script>,1.0\n",
    )
    .unwrap();
    ablation_report::decorate(&report).unwrap();
    ablation_report::decorate(&report).unwrap();
    let html = fs::read_to_string(&report).unwrap();
    assert_eq!(
        html.matches("<!-- RST19-NATIVE-EVIDENCE-START -->").count(),
        1
    );
    assert!(html.contains("Existing science"));
    assert!(html.contains("&lt;script&gt;"));
    assert!(html.contains("evidence/point/summary.csv"));
    assert!(html.contains("<td>0.5</td>"));
    assert!(root.canonicalize().unwrap().starts_with(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("target/test-artifacts")
            .canonicalize()
            .unwrap()
    ));
    fs::remove_dir_all(root).unwrap();
}
