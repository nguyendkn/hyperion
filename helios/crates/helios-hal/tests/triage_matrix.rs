//! Triage matrix: arch x tensor_format x backend, no GPU needed. Each case is
//! a named test so a failure says exactly which cell of the matrix broke.

use helios_hal::{supported_formats, triage, ProbedBackend, TriageError, TriageOutcome};
use hyperion_specification::v1::{
    Activation, AttentionKind, GenerationConfig, Manifest, ModelConfig, RopeStyle, TensorFormat, TokenizerConfig, CONTRACT_REVISION,
};

fn manifest(target_arch: &str, tensor_format: TensorFormat) -> Manifest {
    Manifest {
        model_id: "m".into(),
        target_arch: target_arch.into(),
        tensor_format,
        model: ModelConfig {
            num_layers: 2,
            attention: AttentionKind::Mha,
            num_attention_heads: 4,
            num_kv_heads: 4,
            head_dim: 16,
            vocab_size: 10,
            max_seq_len: 16,
            hidden_size: 64,
            intermediate_size: 128,
            rope_theta: 10000.0,
            rope_style: RopeStyle::RotateHalf,
            rms_norm_eps: 1e-6,
            tie_word_embeddings: false,
            qkv_bias: true,
            qk_norm: false,
            activation: Activation::Silu,
            attention_scale: None,
        },
        contract_revision: CONTRACT_REVISION,
        generation: GenerationConfig { eos_token_ids: vec![0], sampling_defaults: None },
        chat_template: None,
        qualified_checkpoint: None,
        tokenizer: TokenizerConfig::default(),
        layers: vec![],
        payloads: Vec::new(),
        hardware_recipe: Default::default(),
        signature: None,
    }
}

/// Backend written for `backend` running on a device that reports `device`.
fn probed(backend: &'static str, device: &str) -> ProbedBackend {
    ProbedBackend { backend_arch: backend, device_arch: device.into() }
}

fn accept(arch: &'static str, format: TensorFormat) {
    let decision = triage(&manifest(arch, format), &probed(arch, arch)).expect("should be accepted");
    assert_eq!(decision.outcome, TriageOutcome::Exact);
}

#[test]
fn sm90_accepts_bf16() {
    accept("sm_90", TensorFormat::Bf16);
}

#[test]
fn sm90_accepts_fp8() {
    accept("sm_90", TensorFormat::Fp8);
}

#[test]
fn sm90_refuses_fp4() {
    let err = triage(&manifest("sm_90", TensorFormat::Fp4), &probed("sm_90", "sm_90")).unwrap_err();
    assert!(matches!(err, TriageError::FormatUnsupported { ref format, ref arch, .. } if format == "fp4" && arch == "sm_90"));
}

#[test]
fn sm120_accepts_every_format() {
    for f in [TensorFormat::Bf16, TensorFormat::Fp8, TensorFormat::Fp4] {
        accept("sm_120", f);
    }
}

#[test]
fn stub_accepts_every_format() {
    for f in [TensorFormat::Bf16, TensorFormat::Fp8, TensorFormat::Fp4] {
        accept("sm_0", f);
    }
}

#[test]
fn package_for_sm120_on_sm90_device_is_refused() {
    let err = triage(&manifest("sm_120", TensorFormat::Bf16), &probed("sm_90", "sm_90")).unwrap_err();
    assert_eq!(err, TriageError::ArchMismatch { target: "sm_120".into(), device: "sm_90".into() });
}

#[test]
fn package_for_sm90_on_sm120_device_is_refused_no_downgrade() {
    let err = triage(&manifest("sm_90", TensorFormat::Bf16), &probed("sm_120", "sm_120")).unwrap_err();
    assert_eq!(err, TriageError::ArchMismatch { target: "sm_90".into(), device: "sm_120".into() });
}

#[test]
fn sm120_package_on_sm121_device_is_refused_exact_match_only() {
    let err = triage(&manifest("sm_120", TensorFormat::Bf16), &probed("sm_121", "sm_121")).unwrap_err();
    assert!(matches!(err, TriageError::ArchMismatch { .. }));
}

/// Regression from the H100 run: the Blackwell backend was constructed on an
/// sm_90 device and a sm_90 package was called "Exact".
#[test]
fn blackwell_backend_on_hopper_device_is_refused() {
    let err = triage(&manifest("sm_90", TensorFormat::Bf16), &probed("sm_120", "sm_90")).unwrap_err();
    assert_eq!(err, TriageError::BackendArchMismatch { backend: "sm_120".into(), device: "sm_90".into() });
}

#[test]
fn hopper_backend_on_blackwell_device_is_refused() {
    let err = triage(&manifest("sm_120", TensorFormat::Bf16), &probed("sm_90", "sm_120")).unwrap_err();
    assert!(matches!(err, TriageError::BackendArchMismatch { .. }));
}

#[test]
fn stub_never_runs_a_real_package() {
    for arch in ["sm_90", "sm_120"] {
        let err = triage(&manifest(arch, TensorFormat::Bf16), &probed("sm_0", "sm_0")).unwrap_err();
        assert_eq!(err, TriageError::NoGpuBackend { target: arch.into() }, "{arch}");
        assert!(err.to_string().contains("rebuild helios-cli"), "{err}");
    }
}

#[test]
fn unknown_arch_has_no_formats_and_is_refused() {
    assert!(supported_formats("sm_75").is_empty());
    let err = triage(&manifest("sm_75", TensorFormat::Bf16), &probed("sm_75", "sm_75")).unwrap_err();
    assert!(matches!(err, TriageError::FormatUnsupported { ref supported, .. } if supported.contains("unknown arch")));
}

#[test]
fn malformed_capability_strings_are_rejected() {
    for bad in ["", "90", "sm_", "sm_9x", "SM_90", "sm90"] {
        let err = triage(&manifest(bad, TensorFormat::Bf16), &probed("sm_90", "sm_90")).unwrap_err();
        assert_eq!(err, TriageError::BadCapability(bad.into()), "{bad:?}");
    }
}

#[test]
fn refusals_say_what_to_do() {
    let arch = triage(&manifest("sm_120", TensorFormat::Bf16), &probed("sm_90", "sm_90")).unwrap_err().to_string();
    assert!(arch.contains("sm_120") && arch.contains("sm_90") && arch.contains("hyperion-compile --hw sm_90"), "{arch}");
    let fmt = triage(&manifest("sm_90", TensorFormat::Fp4), &probed("sm_90", "sm_90")).unwrap_err().to_string();
    assert!(fmt.contains("fp4") && fmt.contains("bf16, fp8") && fmt.contains("recompile"), "{fmt}");
}
