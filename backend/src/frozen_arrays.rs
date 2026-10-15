//! NumPy-compatible frozen float32 scientific inputs, written entirely in Rust.
use crate::Result;
use flate2::{write::DeflateEncoder, Compression};
use sha2::{Digest, Sha256};
use std::{fs, io::Write, path::Path};

pub fn array_sha256(values: &[f32], shape: &[usize]) -> String {
    let mut digest = Sha256::new();
    digest.update(b"<f4");
    digest.update(serde_json::to_vec(shape).expect("integer shape"));
    for value in values {
        digest.update(value.to_le_bytes());
    }
    format!("{:x}", digest.finalize())
}
pub fn npy_f32(values: &[f32], shape: &[usize]) -> Result<Vec<u8>> {
    if shape.is_empty()
        || shape.iter().try_fold(1usize, |n, d| n.checked_mul(*d)) != Some(values.len())
    {
        return Err("frozen array dimensions do not match pixels".into());
    }
    let dimensions = shape
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join(", ");
    let mut header = format!(
        "{{'descr': '<f4', 'fortran_order': False, 'shape': ({dimensions}{}), }}",
        if shape.len() == 1 { "," } else { "" }
    );
    let padding = (64 - (10 + header.len() + 1) % 64) % 64;
    header.extend(std::iter::repeat_n(' ', padding));
    header.push('\n');
    let mut bytes = b"\x93NUMPY\x01\x00".to_vec();
    bytes.extend((header.len() as u16).to_le_bytes());
    bytes.extend(header.bytes());
    for v in values {
        bytes.extend(v.to_le_bytes());
    }
    Ok(bytes)
}
fn crc32(bytes: &[u8]) -> u32 {
    let mut crc = !0u32;
    for b in bytes {
        crc ^= *b as u32;
        for _ in 0..8 {
            crc = (crc >> 1) ^ if crc & 1 != 0 { 0xedb88320 } else { 0 };
        }
    }
    !crc
}
fn u16le(out: &mut Vec<u8>, v: u16) {
    out.extend(v.to_le_bytes());
}
fn u32le(out: &mut Vec<u8>, v: u32) {
    out.extend(v.to_le_bytes());
}
pub fn write_npz_f32(path: &Path, arrays: &[(&str, &[f32], &[usize])]) -> Result<String> {
    if arrays.len() > u16::MAX as usize {
        return Err("too many frozen arrays".into());
    }
    let mut archive = Vec::new();
    let mut central = Vec::new();
    for (name, values, shape) in arrays {
        if name.contains(['/', '\\']) || name.is_empty() {
            return Err("invalid frozen array name".into());
        }
        let name = format!("{name}.npy");
        let bytes = npy_f32(values, shape)?;
        let mut encoder = DeflateEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(&bytes).map_err(|e| e.to_string())?;
        let compressed = encoder.finish().map_err(|e| e.to_string())?;
        if bytes.len() > u32::MAX as usize || archive.len() > u32::MAX as usize {
            return Err("frozen archive exceeds ZIP32 limits".into());
        }
        let offset = archive.len() as u32;
        let crc = crc32(&bytes);
        u32le(&mut archive, 0x04034b50);
        u16le(&mut archive, 20);
        u16le(&mut archive, 0);
        u16le(&mut archive, 8);
        u16le(&mut archive, 0);
        u16le(&mut archive, 33);
        u32le(&mut archive, crc);
        u32le(&mut archive, compressed.len() as u32);
        u32le(&mut archive, bytes.len() as u32);
        u16le(&mut archive, name.len() as u16);
        u16le(&mut archive, 0);
        archive.extend(name.bytes());
        archive.extend(compressed);
        u32le(&mut central, 0x02014b50);
        u16le(&mut central, 20);
        u16le(&mut central, 20);
        u16le(&mut central, 0);
        u16le(&mut central, 8);
        u16le(&mut central, 0);
        u16le(&mut central, 33);
        u32le(&mut central, crc);
        u32le(
            &mut central,
            archive.len() as u32 - offset - 30 - name.len() as u32,
        );
        u32le(&mut central, bytes.len() as u32);
        u16le(&mut central, name.len() as u16);
        for _ in 0..4 {
            u16le(&mut central, 0);
        }
        u32le(&mut central, 0);
        u32le(&mut central, offset);
        central.extend(name.bytes());
    }
    let offset = archive.len() as u32;
    let size = central.len() as u32;
    archive.extend(central);
    u32le(&mut archive, 0x06054b50);
    u16le(&mut archive, 0);
    u16le(&mut archive, 0);
    u16le(&mut archive, arrays.len() as u16);
    u16le(&mut archive, arrays.len() as u16);
    u32le(&mut archive, size);
    u32le(&mut archive, offset);
    u16le(&mut archive, 0);
    fs::write(path, &archive).map_err(|e| e.to_string())?;
    Ok(format!("{:x}", Sha256::digest(&archive)))
}
