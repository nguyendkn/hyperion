//! build.rs for helios-hal. The CUDA Driver is linked by `cuda-driver-sys`; this
//! script only makes a CUDA environment change rebuild the crate. A default
//! (`stub`) build has no native linkage.

fn main() {
    println!("cargo:rustc-check-cfg=cfg(feature, values(\"cuda\",\"sm90\",\"sm120\",\"multi\"))");
    #[cfg(feature = "cuda")]
    {
        println!("cargo:rerun-if-env-changed=CUDA_PATH");
        println!("cargo:rerun-if-env-changed=CUDA_LIB_DIR");
    }
}
