use rst19_core::sequence_source_audit::write_sequence_source_audit;
use serde_json::{json, Value};
use std::{fs, path::PathBuf};

#[test]
fn deterministic_quantile_samples_include_missing_frame_cutouts_and_manifest() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let frame = root.join("tests/fixtures/contaminated_pair_injection_parity.fits");
    let output = root
        .join("target/test-artifacts")
        .join(format!("sequence-source-audit-{}", std::process::id()));
    let track = |id, snr, class, indices: &[usize]| {
        json!({
            "track_id":id,"classification":class,"displacement_px":0.0,"fit_rms_px":0.0,
            "points":indices.iter().map(|index|json!({"frame_index":index,"x":32.0,"y":32.0,"aligned_x":32.0,"aligned_y":32.0,"flux_snr":snr})).collect::<Vec<_>>()
        })
    };
    let sequence = json!({"frames":[{"path":frame},{"path":frame},{"path":frame}],
        "cumulative_shifts":[[0.0,0.0],[1.0,1.0],[0.0,0.0]],"persistent_min_presence":3,
        "tracks":[track(9,30.0,"static",&[0,1,2]),track(4,10.0,"static",&[0,1,2]),track(7,20.0,"static",&[0,1,2]),track(11,5.0,"transient",&[0,2])]});
    let written = write_sequence_source_audit(&sequence, &output, 2, 10, 2).unwrap();
    let csv = fs::read(written.join("source_track_audit.csv")).unwrap();
    assert_eq!(&csv[..3], &[0xef, 0xbb, 0xbf]);
    let mut reader = csv::Reader::from_reader(&csv[3..]);
    let records = reader.records().map(Result::unwrap).collect::<Vec<_>>();
    assert_eq!(records.len(), 3);
    assert_eq!(&records[0][2], "4");
    assert_eq!(&records[1][2], "9");
    assert_eq!(&records[2][5], "1|3");
    let summary: Value =
        serde_json::from_slice(&fs::read(written.join("source_track_audit_summary.json")).unwrap())
            .unwrap();
    assert_eq!(summary["strict_static_count"], 3);
    assert_eq!(summary["sampled_track_count"], 3);
    let decoder = png::Decoder::new(std::io::BufReader::new(
        fs::File::open(written.join("source_track_contact_sheet.png")).unwrap(),
    ));
    let reader = decoder.read_info().unwrap();
    assert_eq!(reader.info().width, 126);
    assert_eq!(reader.info().height, 251);
    assert_eq!(reader.info().color_type, png::ColorType::Rgb);
    let decoder = png::Decoder::new(std::io::BufReader::new(
        fs::File::open(
            written.join("source_track_cutouts/transient_near_threshold_track00011.png"),
        )
        .unwrap(),
    ));
    let mut reader = decoder.read_info().unwrap();
    let mut pixels = vec![0; reader.output_buffer_size().unwrap()];
    let info = reader.next_frame(&mut pixels).unwrap();
    let missing_border = (25 * info.width as usize + 42) * 3;
    assert_eq!(
        &pixels[missing_border..missing_border + 3],
        &[105, 115, 134]
    );
    let target = fs::canonicalize(root.join("target/test-artifacts")).unwrap();
    assert!(written.starts_with(target));
    fs::remove_dir_all(written).unwrap();
}
