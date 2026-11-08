use std::{env, path::PathBuf, process::Command};

fn main() {
    let manifest = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let source = manifest.join("ggml");
    let out = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let build = out.join("ggml-build");
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();

    let backends = ["METAL", "OPENCL", "VULKAN", "CUDA"];
    let enabled: Vec<_> = backends
        .iter()
        .filter(|backend| env::var_os(format!("CARGO_FEATURE_{}", backend)).is_some())
        .collect();
    assert!(
        enabled.len() <= 1,
        "select at most one accelerator backend feature: metal, opencl, vulkan, cuda"
    );

    if enabled.contains(&&"METAL") {
        assert_eq!(target_os, "macos", "the Metal backend requires macOS");
    }

    let mut configure = Command::new("cmake");
    configure
        .arg("-S")
        .arg(&source)
        .arg("-B")
        .arg(&build)
        .arg("-DCMAKE_BUILD_TYPE=Release")
        .arg("-DGGML_BUILD_TESTS=OFF")
        .arg("-DGGML_BUILD_EXAMPLES=OFF")
        .arg(format!("-DGGML_METAL={}", enabled.contains(&&"METAL")))
        .arg(format!("-DGGML_OPENCL={}", enabled.contains(&&"OPENCL")))
        .arg(format!("-DGGML_VULKAN={}", enabled.contains(&&"VULKAN")))
        .arg(format!("-DGGML_CUDA={}", enabled.contains(&&"CUDA")))
        .arg(format!("-DGGML_ACCELERATE={}", target_os == "macos"));

    let status = configure
        .status()
        .expect("failed to run cmake; install CMake and a C/C++ toolchain");
    assert!(status.success(), "GGML CMake configuration failed");

    let status = Command::new("cmake")
        .arg("--build")
        .arg(&build)
        .arg("--config")
        .arg("Release")
        .arg("--parallel")
        .status()
        .expect("failed to build GGML");
    assert!(status.success(), "GGML CMake build failed");

    let source_build = build.join("src");
    println!("cargo:rustc-link-search=native={}", source_build.display());
    for backend_dir in ["ggml-metal", "ggml-opencl", "ggml-vulkan", "ggml-cuda"] {
        println!(
            "cargo:rustc-link-search=native={}",
            source_build.join(backend_dir).display()
        );
    }

    for lib in ["ggml", "ggml-base", "ggml-cpu"] {
        println!("cargo:rustc-link-lib=dylib={lib}");
    }
    if let Some(backend) = enabled.first() {
        let name = match **backend {
            "METAL" => "ggml-metal",
            "OPENCL" => "ggml-opencl",
            "VULKAN" => "ggml-vulkan",
            "CUDA" => "ggml-cuda",
            _ => unreachable!(),
        };
        println!("cargo:rustc-link-lib=dylib={name}");
    }
    if enabled.contains(&&"METAL") {
        for framework in [
            "Accelerate",
            "Metal",
            "Foundation",
            "MetalKit",
            "MetalPerformanceShaders",
        ] {
            println!("cargo:rustc-link-lib=framework={framework}");
        }
        println!("cargo:rustc-link-lib=c++");
    }
    println!("cargo:rerun-if-changed=ggml");
}
