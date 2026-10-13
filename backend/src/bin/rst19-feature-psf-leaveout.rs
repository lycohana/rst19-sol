fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match rst19_core::feature_psf_leaveout::run_cli(&args) {
        Ok(code) => std::process::exit(code),
        Err(error) if error == "help" => std::process::exit(0),
        Err(error) => {
            eprintln!("rst19-feature-psf-leaveout: 审计失败：{error}");
            std::process::exit(2);
        }
    }
}
