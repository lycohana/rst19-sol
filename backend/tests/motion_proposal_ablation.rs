use flate2::read::DeflateDecoder;
use rst19_core::{frozen_arrays, motion_proposal_ablation};
use std::{fs, io::Read, path::PathBuf};

#[test]
fn frozen_npz_contains_valid_deflated_numpy_arrays_with_shape_and_exact_float_bits() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/test-artifacts");
    fs::create_dir_all(&root).unwrap();
    let path = root.join(format!("native-frozen-{}.npz", std::process::id()));
    let pixels = vec![1.25f32, -2.5, 0., f32::from_bits(0x7fc00001)];
    let hash =
        frozen_arrays::write_npz_f32(&path, &[("control_frames", &pixels, &[1, 2, 2])]).unwrap();
    assert_eq!(hash.len(), 64);
    let bytes = fs::read(&path).unwrap();
    assert_eq!(&bytes[0..4], b"PK\x03\x04");
    let len = u32::from_le_bytes(bytes[18..22].try_into().unwrap()) as usize;
    let name_len = u16::from_le_bytes(bytes[26..28].try_into().unwrap()) as usize;
    assert_eq!(&bytes[30..30 + name_len], b"control_frames.npy");
    let mut decoded = Vec::new();
    DeflateDecoder::new(&bytes[30 + name_len..30 + name_len + len])
        .read_to_end(&mut decoded)
        .unwrap();
    assert_eq!(&decoded[0..8], b"\x93NUMPY\x01\x00");
    let header_len = u16::from_le_bytes(decoded[8..10].try_into().unwrap()) as usize;
    assert!(std::str::from_utf8(&decoded[10..10 + header_len])
        .unwrap()
        .contains("(1, 2, 2)"));
    assert_eq!(
        &decoded[10 + header_len..],
        pixels
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>()
    );
    let central = 30 + name_len + len;
    assert_eq!(&bytes[central..central + 4], b"PK\x01\x02");
    assert_eq!(
        u32::from_le_bytes(bytes[central + 20..central + 24].try_into().unwrap()),
        len as u32
    );
    assert!(path
        .canonicalize()
        .unwrap()
        .starts_with(root.canonicalize().unwrap()));
    fs::remove_file(path).unwrap();
}

#[test]
fn moving_trail_versions_share_frozen_noise_and_preserve_the_exposure_kernel_definition() {
    let (control, injected) = motion_proposal_ablation::frozen_inputs(19019);
    assert_eq!(control.len(), 4 * 96 * 96);
    assert_eq!(motion_proposal_ablation::frozen_inputs(19019).0, control);
    for i in 0..4 {
        let center = (40 * 96 + 22 + 10 * i) + i * 96 * 96;
        let delta = injected[center] - control[center];
        assert!(
            (delta - 12.7736).abs() < 0.01,
            "sampled central increment {delta}"
        );
    }
    let (base, base_audits) = motion_proposal_ablation::run_version(&injected, false).unwrap();
    let (multiscale, multi_audits) =
        motion_proposal_ablation::run_version(&injected, true).unwrap();
    assert_eq!(base_audits.len(), 4);
    assert_eq!(multi_audits.len(), 4);
    assert!(base.iter().all(|t| t["classification"] != "moving"));
    assert!(
        multiscale
            .iter()
            .any(|t| t["classification"] == "moving" && t["presence"] == 4),
        "native low peak recovery: {multiscale:?}"
    );
    let (negative, _) = motion_proposal_ablation::run_version(&control, true).unwrap();
    assert!(negative.iter().all(|t| t["classification"] != "moving"));
}
