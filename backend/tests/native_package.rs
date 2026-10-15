use flate2::read::DeflateDecoder;
use rst19_core::native_package;
use std::{fs, io::Read, path::PathBuf};

#[test]
fn native_package_zip_preserves_paths_content_and_executable_permissions() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("target/test-artifacts")
        .join(format!("native-package-{}", std::process::id()));
    let package = root.join("RST19-test");
    fs::create_dir_all(package.join("runtime")).unwrap();
    fs::write(
        package.join("runtime/tool.exe"),
        b"native executable test data",
    )
    .unwrap();
    let archive = root.join("package.zip");
    native_package::archive(&package, &archive).unwrap();
    let bytes = fs::read(archive).unwrap();
    let name_len = u16::from_le_bytes(bytes[26..28].try_into().unwrap()) as usize;
    let size = u32::from_le_bytes(bytes[18..22].try_into().unwrap()) as usize;
    assert_eq!(&bytes[30..30 + name_len], b"RST19-test/runtime/tool.exe");
    let mut actual = Vec::new();
    DeflateDecoder::new(&bytes[30 + name_len..30 + name_len + size])
        .read_to_end(&mut actual)
        .unwrap();
    assert_eq!(actual, b"native executable test data");
    let central = 30 + name_len + size;
    assert_eq!(&bytes[central..central + 4], b"PK\x01\x02");
    assert_eq!(
        (u32::from_le_bytes(bytes[central + 38..central + 42].try_into().unwrap()) >> 16)
            & 0o170000,
        0o100000
    );
    assert!(root.canonicalize().unwrap().starts_with(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("target/test-artifacts")
            .canonicalize()
            .unwrap()
    ));
    fs::remove_dir_all(root).unwrap();
}
