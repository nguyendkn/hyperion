//! build.rs for helios-hal — wires FFI linkage per enabled backend.
//!
//! Per the architecture decision (FFI direct, never subprocess), each
//! `smNN` feature links its native libraries here at build time:
//!
//! - `sm90`  → TensorRT-LLM (C++) + CUDA Driver. We compile a small C++ shim
//!   via `cc` and generate Rust bindings to it with `bindgen`.
//! - `sm120` → CUDA Driver (via pre-built `cuda-driver-sys` bindings). No
//!   bindgen; native CUDA Graph capture lives in the driver, kernels come
//!   from PTX embedded in the package.
//!
//! This file is intentionally a skeleton: the `cfg(feature)` blocks below
//! carry the linkage plan as commented-out `println!("cargo:rustc-link-lib=…")`
//! directives and TODO markers. Activate them once the corresponding C++ shim
//! and `wrapper.h` are in place under `backends/<arch>/ffi/`. Until then a
//! default (`stub`) build compiles with zero native linkage.
//!
//! Why the indirection (shim + bindgen vs binding TensorRT straight): a thin
//! C++ shim gives us a stable C ABI surface that survives TensorRT-LLM
//! version churn, and lets `bindgen` emit one clean set of Rust FFI types
//! rather than mirroring a C++ class hierarchy.

fn main() {
    // Re-run if the user changes which backends are compiled in.
    println!("cargo:rustc-check-cfg=cfg(feature, values(\"sm90\",\"sm120\",\"multi\"))");

    #[cfg(feature = "sm90")]
    wire_sm90();

    #[cfg(feature = "sm120")]
    wire_sm120();

    // `stub` and `multi` need no native linkage here — `multi` just enables the
    // two feature blocks above; `stub` needs nothing.
}

/// Link TensorRT-LLM shim + CUDA Driver for the Hopper SM90 backend.
#[allow(dead_code)]
fn wire_sm90() {
    // TODO once backends/hopper_sm90/ffi/wrapper.h + shim.cpp exist:
    //
    // let out_dir = std::env::var("OUT_DIR").unwrap();
    //
    // cc::Build::new()
    //     .cpp(true)
    //     .file("src/backends/hopper_sm90/ffi/shim.cpp")
    //     .include("src/backends/hopper_sm90/ffi")
    //     .compile("hopper_shim");
    //
    // println!("cargo:rustc-link-lib=dylib=nvinfer");
    // println!("cargo:rustc-link-lib=dylib=cudart");
    //
    // bindgen::Builder::default()
    //     .header("src/backends/hopper_sm90/ffi/wrapper.h")
    //     .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
    //     .generate()
    //     .expect("bindgen failed for hopper shim")
    //     .write_to_file(std::path::Path::new(&out_dir).join("hopper_bindings.rs"))
    //     .expect("write bindings failed");
    //
    // println!("cargo:rerun-if-changed=src/backends/hopper_sm90/ffi/");
}

/// Note native-link requirements for the Blackwell SM120 backend. `cuda-driver-sys`
/// provides pre-built Rust bindings already, so unlike SM90 we don't run
/// bindgen here; we just emit env-watch reruns so a CUDA env change rebuilds.
#[allow(dead_code)]
fn wire_sm120() {
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=CUDA_LIB_DIR");
}
