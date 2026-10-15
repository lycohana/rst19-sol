use rst19_core::{
    cache, fits,
    pipeline::{self, FrameOptions},
};
use serde_json::{json, Value};
use std::{
    fs,
    path::PathBuf,
    sync::atomic::{AtomicUsize, Ordering},
};

static NEXT: AtomicUsize = AtomicUsize::new(0);
struct TestDirectory {
    path: PathBuf,
    base: PathBuf,
}
impl TestDirectory {
    fn new() -> Self {
        let base = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/test-artifacts");
        fs::create_dir_all(&base).unwrap();
        let base = fs::canonicalize(base).unwrap();
        let path = base.join(format!(
            "cache-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(path.join("input")).unwrap();
        Self { path, base }
    }
    fn fixture(&self, name: &str) -> PathBuf {
        let path = self.path.join("input").join(name);
        fs::copy(
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("tests/fixtures/contaminated_pair_injection_parity.fits"),
            &path,
        )
        .unwrap();
        path
    }
}
impl Drop for TestDirectory {
    fn drop(&mut self) {
        if let Ok(path) = fs::canonicalize(&self.path) {
            assert!(path.starts_with(&self.base));
            fs::remove_dir_all(path).unwrap();
        }
    }
}

#[test]
fn content_digest_invalidates_same_size_same_timestamp_files_and_catalog_context() {
    let directory = TestDirectory::new();
    let input = directory.path.join("input/source.fits");
    fs::write(&input, b"alpha").unwrap();
    let before = cache::file_identity(&input).unwrap();
    let modified = fs::metadata(&input).unwrap().modified().unwrap();
    fs::write(&input, b"bravo").unwrap();
    fs::OpenOptions::new()
        .write(true)
        .open(&input)
        .unwrap()
        .set_times(fs::FileTimes::new().set_modified(modified))
        .unwrap();
    let after = cache::file_identity(&input).unwrap();
    assert_eq!(before["size"], after["size"]);
    assert_eq!(before["mtime_ns"], after["mtime_ns"]);
    assert_ne!(before["sha256"], after["sha256"]);
    let catalog = directory.path.join("catalog.csv");
    fs::write(&catalog, b"one").unwrap();
    let parameters = json!({"psf_fwhm":2.0,"threshold_sigma":4.0});
    let context = json!({"catalog_path":catalog,"wcs":{"rotation":1.0},"photometric_band":"G"});
    let first = cache::cache_key(&input, &parameters, &context).unwrap();
    fs::write(&catalog, b"two").unwrap();
    let second = cache::cache_key(&input, &parameters, &context).unwrap();
    assert_ne!(first, second);
    let mut changed = context;
    changed["wcs"]["rotation"] = json!(2.0);
    assert_ne!(
        second,
        cache::cache_key(&input, &parameters, &changed).unwrap()
    );
}

#[test]
fn frame_cache_roundtrip_preserves_measurements_and_full_candidate_working_set() {
    let directory = TestDirectory::new();
    let path = directory.fixture("frame.fits");
    let frame = fits::read_fits(&path, false).unwrap();
    let options: FrameOptions =
        serde_json::from_value(json!({"max_sources":100,"background_box_size":64})).unwrap();
    let mut analysis = pipeline::analyze_frame(frame, None, None, &options, None).unwrap();
    analysis.detection.candidate_peaks = vec![[17.0, 19.0, 5.0]];
    analysis.detection.line_artifact_coordinates = vec![[7, 8]];
    let key = cache::cache_key(
        &path,
        &json!({"psf_fwhm":3.0}),
        &json!({"photometric_band":"G"}),
    )
    .unwrap();
    let cache_dir = directory.path.join("cache");
    let saved = cache::save_analysis(&cache_dir, &key, &analysis).unwrap();
    let loaded =
        cache::load_analysis(&cache_dir, &key, fits::read_fits(&path, false).unwrap()).unwrap();
    assert_eq!(loaded.as_dict(), analysis.as_dict());
    assert_eq!(
        loaded.detection.candidate_peaks,
        analysis.detection.candidate_peaks
    );
    assert_eq!(
        loaded.detection.line_artifact_coordinates,
        analysis.detection.line_artifact_coordinates
    );
    // Replacing the same cache is atomic, including on Windows.
    cache::save_analysis(&cache_dir, &key, &analysis).unwrap();
    let mut decoder = flate2::read::GzDecoder::new(fs::File::open(&saved).unwrap());
    let mut payload: Value = serde_json::from_reader(&mut decoder).unwrap();
    payload["analysis"]
        .as_object_mut()
        .unwrap()
        .remove("matching");
    let mut encoder = flate2::write::GzEncoder::new(
        fs::File::create(&saved).unwrap(),
        flate2::Compression::default(),
    );
    serde_json::to_writer(&mut encoder, &payload).unwrap();
    encoder.finish().unwrap();
    assert!(
        cache::load_analysis(&cache_dir, &key, fits::read_fits(&path, false).unwrap()).is_none()
    );
    fs::write(saved, b"broken gzip").unwrap();
    assert!(
        cache::load_analysis(&cache_dir, &key, fits::read_fits(&path, false).unwrap()).is_none()
    );
}

#[test]
fn sequence_identity_preserves_order_and_detects_added_fits_without_hashing_notes() {
    let directory = TestDirectory::new();
    let first = directory.fixture("one.fits");
    let second = directory.fixture("two.fits");
    let paths = vec![first.clone(), second.clone()];
    let parameters = json!({"motion_line_link_radius_px":64.0,"relative_photometry":true});
    let key = cache::sequence_cache_key(&paths, &parameters, &Value::Null).unwrap();
    assert_ne!(
        key,
        cache::sequence_cache_key(&[second.clone(), first.clone()], &parameters, &Value::Null)
            .unwrap()
    );
    let sequence = json!({"frames":[{"path":first},{"path":second}],"tracks":[],"motion_features":[],"relative_photometry":null,"detector_parameters":parameters});
    let cache_dir = directory.path.join("cache");
    cache::save_sequence_result(&cache_dir, &key, &sequence).unwrap();
    assert_eq!(
        cache::load_sequence_result(&cache_dir, &key, &paths).unwrap(),
        sequence
    );
    assert!(
        cache::load_sequence_result(&cache_dir, &key, &[paths[1].clone(), paths[0].clone()])
            .is_none()
    );
    directory.fixture("additional.FTS");
    assert!(cache::load_sequence_result(&cache_dir, &key, &paths).is_none());
    let identity = cache::sequence_input_identity(&paths).unwrap();
    assert_eq!(
        identity["directories"][0]["files"]
            .as_array()
            .unwrap()
            .len(),
        3
    );
    fs::write(directory.path.join("input/notes.txt"), b"unrelated").unwrap();
    let identity = cache::sequence_input_identity(&paths).unwrap();
    assert_eq!(
        identity["directories"][0]["files"]
            .as_array()
            .unwrap()
            .len(),
        3
    );
}

#[test]
fn native_cache_cleanup_retains_original_inputs_and_legacy_payloads() {
    let directory = TestDirectory::new();
    let cache_dir = directory.path.join("cache");
    fs::create_dir_all(&cache_dir).unwrap();
    for name in [
        "key.native.json.gz",
        "key.native.sequence.json.gz",
        "legacy.json.gz",
        "original.fits",
    ] {
        fs::write(cache_dir.join(name), b"fixture").unwrap();
    }
    assert_eq!(cache::clear_cache(&cache_dir).unwrap(), 2);
    assert!(cache_dir.join("legacy.json.gz").is_file());
    assert!(cache_dir.join("original.fits").is_file());
}

#[test]
fn sequence_frame_cache_reuses_detection_and_invalidates_all_shared_pilot_dependents() {
    let directory = TestDirectory::new();
    let first = directory.fixture("pilot.fits");
    let second = directory.fixture("dependent.fits");
    let request = json!({"paths":[first,second],"cache_dir":directory.path.join("cache"),
        "fast_point_motion":false,"stack_faint_recovery":false,
        "frame_options":{"max_sources":100,"background_box_size":64}});
    let initial = rst19_core::sequence_pipeline::handle_request(&request).unwrap();
    assert_eq!(initial["frame_cache_hits"], 0);
    let repeated = rst19_core::sequence_pipeline::handle_request(&request).unwrap();
    assert_eq!(repeated["frame_cache_hits"], 2);
    assert_eq!(initial["sequence_result"], repeated["sequence_result"]);
    let mut bytes = fs::read(&first).unwrap();
    let last = bytes.len() - 1;
    bytes[last] ^= 1; // FITS padding changes identity without changing the science pixels.
    let modified = fs::metadata(&first).unwrap().modified().unwrap();
    fs::write(&first, bytes).unwrap();
    fs::OpenOptions::new()
        .write(true)
        .open(&first)
        .unwrap()
        .set_times(fs::FileTimes::new().set_modified(modified))
        .unwrap();
    let invalidated = rst19_core::sequence_pipeline::handle_request(&request).unwrap();
    assert_eq!(invalidated["frame_cache_hits"], 0);
    assert_eq!(initial["sequence_result"], invalidated["sequence_result"]);
}
