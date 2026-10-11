//! Builds llama.cpp from vendor/llama.cpp (a submodule at the tag rummage pins) as static libraries, and generates
//! bindings for llama.h and mtmd.h. GPU: Metal on macOS; `--features cuda` or `--features vulkan` elsewhere; else CPU.

use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..").canonicalize().unwrap();
    let source = root.join("vendor/llama.cpp");
    if !source.join("CMakeLists.txt").exists() {
        panic!("vendor/llama.cpp is empty: run `git submodule update --init`");
    }
    apply_patches(&root, &source);
    println!("cargo:rerun-if-changed={}", root.join("patches").display());

    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap();
    let macos = target_os == "macos";
    let cuda = std::env::var("CARGO_FEATURE_CUDA").is_ok();
    let vulkan = std::env::var("CARGO_FEATURE_VULKAN").is_ok();
    let on_off = |on: bool| if on { "ON" } else { "OFF" };

    // Only the libraries: no tools, server or tests, nothing downloaded at build time.
    let mut config = cmake::Config::new(&source);
    config
        .profile("Release")
        .define("BUILD_SHARED_LIBS", "OFF")
        .define("LLAMA_BUILD_COMMON", "OFF")
        .define("LLAMA_BUILD_TOOLS", "OFF")
        .define("LLAMA_BUILD_SERVER", "OFF")
        .define("LLAMA_BUILD_EXAMPLES", "OFF")
        .define("LLAMA_BUILD_TESTS", "OFF")
        .define("LLAMA_BUILD_APP", "OFF")
        .define("LLAMA_BUILD_UI", "OFF")
        .define("LLAMA_BUILD_MTMD", "ON")
        .define("LLAMA_OPENSSL", "OFF")
        .define("LLAMA_CURL", "OFF")
        .define("GGML_NATIVE", "OFF")
        .define("GGML_OPENMP", "OFF")
        .define("GGML_METAL", on_off(macos))
        .define("GGML_METAL_EMBED_LIBRARY", on_off(macos))
        .define("GGML_CUDA", on_off(cuda))
        .define("GGML_VULKAN", on_off(vulkan))
        .build_target("mtmd");
    if macos {
        config.define("CMAKE_OSX_DEPLOYMENT_TARGET", "13.0");
    }
    let out = config.build();
    let build = out.join("build");

    // Every static library the build made, mtmd and llama first, then the ggml backends, then ggml's core, so the
    // order also works for linkers that resolve symbols in a single pass (GNU ld).
    let mut libraries = Vec::new();
    collect_static_libraries(&build, &mut libraries);
    let rank = |name: &str| match name {
        "mtmd" => 0,
        "llama" => 1,
        "ggml" => 3,
        "ggml-base" => 4,
        n if n.starts_with("ggml-") => 2,
        _ => 5,
    };
    libraries.sort_by_key(|(_, name)| rank(name));
    for (dir, name) in &libraries {
        println!("cargo:rustc-link-search=native={}", dir.display());
        println!("cargo:rustc-link-lib=static={name}");
    }
    if macos {
        println!("cargo:rustc-link-lib=c++");
        for framework in ["Foundation", "Metal", "MetalKit", "Accelerate"] {
            println!("cargo:rustc-link-lib=framework={framework}");
        }
    } else {
        println!("cargo:rustc-link-lib=stdc++");
        println!("cargo:rustc-link-lib=m");
    }
    if cuda {
        for lib in ["cudart", "cublas", "cublasLt", "cuda"] {
            println!("cargo:rustc-link-lib={lib}");
        }
    }
    if vulkan {
        println!("cargo:rustc-link-lib=vulkan");
    }

    let bindings = bindgen::Builder::default()
        .header(source.join("include/llama.h").to_string_lossy())
        .header(source.join("tools/mtmd/mtmd.h").to_string_lossy())
        .header(source.join("tools/mtmd/mtmd-helper.h").to_string_lossy())
        .clang_arg(format!("-I{}", source.join("include").display()))
        .clang_arg(format!("-I{}", source.join("ggml/include").display()))
        .allowlist_function("llama_.*|mtmd_.*|ggml_backend_load_all")
        .allowlist_type("llama_.*|mtmd_.*|ggml_log_.*")
        .allowlist_var("LLAMA_.*|MTMD_.*|GGML_LOG_.*")
        .prepend_enum_name(false)
        .generate()
        .expect("bindgen failed on llama.h / mtmd.h");
    bindings.write_to_file(PathBuf::from(std::env::var("OUT_DIR").unwrap()).join("bindings.rs")).unwrap();
}

/// Applies each patches/*.patch to the submodule unless it's already there (`git apply --check` fails then).
fn apply_patches(root: &Path, source: &Path) {
    let mut patches: Vec<_> = std::fs::read_dir(root.join("patches"))
        .map(|dir| {
            dir.filter_map(|e| e.ok())
                .map(|e| e.path())
                .filter(|p| p.extension().is_some_and(|x| x == "patch"))
                .collect()
        })
        .unwrap_or_default();
    patches.sort();
    for patch in patches {
        let check = Command::new("git").arg("-C").arg(source).args(["apply", "--check"]).arg(&patch).output();
        if check.is_ok_and(|o| o.status.success()) {
            let status = Command::new("git").arg("-C").arg(source).arg("apply").arg(&patch).status().unwrap();
            assert!(status.success(), "couldn't apply {}", patch.display());
        }
    }
}

fn collect_static_libraries(dir: &Path, found: &mut Vec<(PathBuf, String)>) {
    let Ok(entries) = std::fs::read_dir(dir) else { return };
    for entry in entries.filter_map(|e| e.ok()) {
        let path = entry.path();
        if path.is_dir() {
            collect_static_libraries(&path, found);
        } else if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
            if let Some(stem) = name.strip_prefix("lib").and_then(|n| n.strip_suffix(".a")) {
                found.push((dir.to_path_buf(), stem.to_string()));
            }
        }
    }
}
