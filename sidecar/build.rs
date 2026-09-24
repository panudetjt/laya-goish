fn main() {
    // WSL2: the CUDA driver shim lives in /usr/lib/wsl/lib (not a default
    // linker path). Prefer it when present; plain -lcuda otherwise.
    let wsl = std::path::Path::new("/usr/lib/wsl/lib/libcuda.so").exists();
    if wsl {
        println!("cargo:rustc-link-search=native=/usr/lib/wsl/lib");
        println!("cargo:rustc-link-lib=dylib=cuda");
        println!("cargo:rustc-link-arg=-Wl,-rpath,/usr/lib/wsl/lib");
    } else {
        println!("cargo:rustc-link-lib=dylib=cuda");
    }
    // igpu-probe dlopens libze_loader/libOpenCL at run time; -ldl works on
    // every glibc (a stub library on >= 2.34, the real thing before).
    println!("cargo:rustc-link-lib=dylib=dl");
}
