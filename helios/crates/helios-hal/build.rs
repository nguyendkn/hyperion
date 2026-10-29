//! build.rs for helios-hal. Nothing links against CUDA (the driver and cuBLAS are loaded at run
//! time), so this script only declares the feature names the code checks with `cfg`.

fn main() {
    println!("cargo:rustc-check-cfg=cfg(feature, values(\"cuda\",\"sm90\",\"sm120\",\"multi\"))");
}
