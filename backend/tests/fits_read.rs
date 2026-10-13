use rst19_core::{fits, models::PixelData};
use serde_json::json;
use std::{
    fs,
    path::PathBuf,
    sync::atomic::{AtomicUsize, Ordering},
};

static NEXT_FILE: AtomicUsize = AtomicUsize::new(0);

fn temporary_fits(bytes: &[u8]) -> PathBuf {
    let sequence = NEXT_FILE.fetch_add(1, Ordering::Relaxed);
    let path = std::env::temp_dir().join(format!(
        "rst19-fits-read-{}-{sequence}.fits",
        std::process::id()
    ));
    fs::write(&path, bytes).unwrap();
    path
}

fn card(key: &str, value: &str) -> [u8; 80] {
    let mut result = [b' '; 80];
    let text = format!("{key:<8}= {value:>20}");
    result[..text.len()].copy_from_slice(text.as_bytes());
    result
}

fn fits_file(
    bitpix: i64,
    width: usize,
    height: usize,
    data: &[u8],
    extras: &[(&str, &str)],
) -> Vec<u8> {
    let mut header = Vec::new();
    for (key, value) in [
        ("SIMPLE", "T".to_owned()),
        ("BITPIX", bitpix.to_string()),
        ("NAXIS", "2".to_owned()),
        ("NAXIS1", width.to_string()),
        ("NAXIS2", height.to_string()),
    ] {
        header.extend_from_slice(&card(key, &value));
    }
    for (key, value) in extras {
        header.extend_from_slice(&card(key, value));
    }
    header.extend_from_slice(&{
        let mut end = [b' '; 80];
        end[..3].copy_from_slice(b"END");
        end
    });
    header.resize(header.len().div_ceil(2880) * 2880, b' ');
    header.extend_from_slice(data);
    header
}

#[test]
fn reads_all_supported_primary_pixel_types_without_rounding_integer_storage() {
    let cases: Vec<(i64, Vec<u8>)> = vec![
        (8, vec![0, 1, 128, 255]),
        (
            16,
            [-2_i16, 0, 5, 12]
                .into_iter()
                .flat_map(i16::to_be_bytes)
                .collect(),
        ),
        (
            32,
            [-200_000_i32, 0, 5, 120_000]
                .into_iter()
                .flat_map(i32::to_be_bytes)
                .collect(),
        ),
        (
            64,
            [i64::MIN + 1, 0, 5, i64::MAX - 1]
                .into_iter()
                .flat_map(i64::to_be_bytes)
                .collect(),
        ),
        (
            -32,
            [-2.5_f32, 0.0, 5.25, 12.0]
                .into_iter()
                .flat_map(f32::to_be_bytes)
                .collect(),
        ),
        (
            -64,
            [-2.5_f64, 0.0, 5.25, 12.0]
                .into_iter()
                .flat_map(f64::to_be_bytes)
                .collect(),
        ),
    ];

    for (bitpix, data) in cases {
        let path = temporary_fits(&fits_file(bitpix, 2, 2, &data, &[]));
        let frame = fits::read_fits(&path, false).unwrap();
        assert_eq!((frame.height, frame.width, frame.data_offset), (2, 2, 2880));
        match (bitpix, frame.data) {
            (8, PixelData::U8(values)) => assert_eq!(values, [0, 1, 128, 255]),
            (16, PixelData::I16(values)) => assert_eq!(values, [-2, 0, 5, 12]),
            (32, PixelData::I32(values)) => assert_eq!(values, [-200_000, 0, 5, 120_000]),
            (64, PixelData::I64(values)) => {
                assert_eq!(values, [i64::MIN + 1, 0, 5, i64::MAX - 1]);
            }
            (-32, PixelData::F32(values)) => assert_eq!(values, [-2.5, 0.0, 5.25, 12.0]),
            (-64, PixelData::F64(values)) => assert_eq!(values, [-2.5, 0.0, 5.25, 12.0]),
            (other, data) => panic!("BITPIX={other} returned unexpected storage: {data:?}"),
        }
        fs::remove_file(path).unwrap();
    }
}

#[test]
fn scaling_and_auxiliary_telemetry_follow_the_python_fits_contract() {
    let mut pixels = vec![0_u8; 104 * 2 * 2];
    let mut values = [0.0_f64; 26];
    values[0..4].copy_from_slice(&[10.0, 20.0, 30.0, 40.0]);
    values[10] = 1.0;
    let mut auxiliary: Vec<u8> = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    for pair in auxiliary.as_chunks_mut::<2>().0 {
        pair.swap(0, 1);
    }
    pixels[..auxiliary.len()].copy_from_slice(&auxiliary);
    pixels[auxiliary.len() + 1] = 2;
    let bytes = fits_file(
        16,
        104,
        2,
        &pixels,
        &[
            ("AZIMUTH", "10.0"),
            ("ELEVATIO", "20.0"),
            ("BSCALE", "2.5"),
            ("BZERO", "1.0"),
        ],
    );
    let path = temporary_fits(&bytes);
    let raw = fits::read_fits(&path, false).unwrap();
    assert_eq!(raw.auxiliary.as_ref().unwrap().ra_deg(), 30.0);
    assert_eq!(raw.auxiliary.as_ref().unwrap().dec_deg(), 40.0);
    assert_eq!(raw.data.dtype(), ">i2");
    let scaled = fits::read_fits(&path, true).unwrap();
    assert_eq!(scaled.data.dtype(), ">f8");
    assert_eq!(scaled.data.value(104), 6.0);
    assert_eq!(scaled.header["BSCALE"], json!(2.5));
    fs::remove_file(path).unwrap();
}

#[test]
fn rejects_truncated_fits_image_data() {
    let bytes = fits_file(16, 2, 2, &[0, 1], &[]);
    let path = temporary_fits(&bytes);
    let error = fits::read_fits(&path, false).unwrap_err();
    assert!(error.contains("FITS image data is truncated"));
    fs::remove_file(path).unwrap();
}
