fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if let Err(error) = rst19_core::signed_null::run_cli(&args) {
        eprintln!("rst19-signed-null-audit: {error}");
        std::process::exit(2);
    }
}
