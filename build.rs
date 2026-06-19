fn main() {
    let out = std::process::Command::new("python")
        .args(["-c", "import torch; print(torch.__path__[0])"])
        .output()
        .expect("failed to locate torch");
    let path = String::from_utf8(out.stdout).unwrap();
    let path = path.trim();
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}/lib", path);

    let os = std::env::var("CARGO_CFG_TARGET_OS").expect("Unable to get TARGET_OS");
    match os.as_str() {
        "linux" | "windows" => {
            if let Some(lib_path) = std::env::var_os("DEP_TCH_LIBTORCH_LIB") {
                println!("cargo:rustc-link-arg=-Wl,-rpath={}", lib_path.to_string_lossy());
            }
            println!("cargo:rustc-link-arg=-Wl,--no-as-needed");
            println!("cargo:rustc-link-arg=-Wl,-ltorch");
            println!("cargo:rustc-link-arg=-Wl,-ltorch_cuda");
        }
        _ => {}
    };
}
