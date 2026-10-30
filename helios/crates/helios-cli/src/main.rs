//! `helios-cli` — the single-binary entrypoint. Implements the runtime
//! dynamic triage flow from the architecture doc:
//!
//! 1. read + verify the `.hyperion` package signature,
//! 2. parse the manifest to learn `target_arch`,
//! 3. via HAL, probe the real device's compute capability,
//! 4. triage (exact arch + tensor_format match), pick the backend,
//! 5. (later) load weights, allocate KV — gated on the `helios-pkg` codec,
//! 6. spawn the scheduler, wire the tokenizer from the manifest, serve Axum.
//!
//! Steps 1–2 are stubbed (we just deserialize the manifest) until the package
//! codec exists; 3–4 and 6 are wired against the HAL factory and core state
//! machine. The engine stays at `Ready` until weights are really loaded.

use std::path::{Component, Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{anyhow, Result};
use clap::Parser;
use tokio_util::sync::CancellationToken;
use tracing_subscriber::EnvFilter;

/// How long the CLI waits for `axum::serve` to drain in-flight HTTP requests
/// after the shutdown signal. A hung client could otherwise block shutdown
/// indefinitely; this bounds the whole drain phase. Fat LTO release builds
/// can afford a bit more — keep it modest for the dev slice.
const HTTP_DRAIN_TIMEOUT: Duration = Duration::from_secs(10);

/// How long the CLI waits for the scheduler task to observe its cancellation
/// token and exit on its own after the server has stopped serving. Only if
/// this elapses do we fall back to `AbortHandle::abort`, which can cut a
/// device call mid-flight (use-after-free risk) — so the timeout is the
/// last resort, not the default path.
const SCHED_EXIT_TIMEOUT: Duration = Duration::from_secs(15);

/// After aborting the scheduler task, how long to wait for confirmation that
/// it actually stopped. A task blocked in a synchronous device call only
/// stops once that call returns; without confirmation we must not tear the
/// backend down underneath it.
const ABORT_CONFIRM_TIMEOUT: Duration = Duration::from_secs(2);

/// Launch the Helios runtime against a single `.hyperion` package.
#[derive(Parser, Debug)]
#[command(name = "helios-cli", version, about = "Run a local inference engine from a .hyperion package")]
struct Cli {
    /// Path to the `.hyperion` model package.
    #[arg(short, long)]
    model: PathBuf,

    /// Port for the OpenAI-compatible HTTP API.
    #[arg(short, long, default_value = "8080")]
    port: u16,

    /// Address to bind. Loopback by default: there is no authentication yet,
    /// so exposing the API on every interface must be an explicit choice.
    #[arg(long, default_value = "127.0.0.1")]
    bind: std::net::IpAddr,

    /// Longest sequence (prompt + generation) the runtime will serve. Defaults
    /// to the model's `max_seq_len`; a lower value bounds KV-cache memory.
    /// Values above the model's maximum are rejected.
    #[arg(long)]
    max_seq_len: Option<u32>,
}

/// Bytes of KV cache one sequence of `tokens` needs: K and V, per layer, per
/// KV head, `head_dim` wide, 2 bytes per element (bf16 cache).
fn kv_bytes_per_sequence(m: &hyperion_specification::v1::ModelConfig, tokens: u32) -> u64 {
    2 * u64::from(m.num_layers) * u64::from(m.num_kv_heads) * u64::from(m.head_dim) * 2 * u64::from(tokens)
}

/// Resolve the serving limit and refuse impossible configurations up front,
/// with the setting that fixes them, before any socket is opened.
fn resolve_max_seq_len(
    requested: Option<u32>,
    model: &hyperion_specification::v1::ModelConfig,
    total_vram_bytes: u64,
) -> Result<u32> {
    let limit = requested.unwrap_or(model.max_seq_len);
    if limit == 0 || limit > model.max_seq_len {
        return Err(anyhow!(
            "--max-seq-len {limit} is outside 1..={max} (the model's max_seq_len): choose a value in that range",
            max = model.max_seq_len
        ));
    }
    let need = kv_bytes_per_sequence(model, limit);
    // A backend that reports no VRAM (stub) cannot be checked; any real
    // device reports its size. One sequence must at least fit on the device.
    if total_vram_bytes > 0 && need > total_vram_bytes {
        let per_token = kv_bytes_per_sequence(model, 1).max(1);
        return Err(anyhow!(
            "KV cache for one {limit}-token sequence needs {need} bytes but the device has {total_vram_bytes}:              lower --max-seq-len (at most {fit} tokens could fit even with no weights)",
            fit = total_vram_bytes / per_token
        ));
    }
    tracing::info!(max_seq_len = limit, kv_bytes_per_sequence = need, "serving limit resolved");
    Ok(limit)
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")))
        // Escape codes only on a terminal: piped logs (tests, files, k8s) stay plain text.
        .with_ansi(std::io::IsTerminal::is_terminal(&std::io::stdout()))
        .init();

    let cli = Cli::parse();

    // 1–2. Load + verify the manifest. TODO: real codec (signature, encryption).
    let manifest = load_manifest(&cli.model)?;
    tracing::info!(model = %manifest.model_id, target = %manifest.target_arch, "manifest loaded");

    // 3–4. Probe device via HAL factory and triage. The backend is shared
    //    (via `Arc`) between the scheduler task and the CLI's shutdown path:
    //    the scheduler drives prefill/decode while it runs, and after the
    //    server stops the CLI calls `shutdown()` to tear down device
    //    allocations. `triage.outcome` is bound for logging only.
    let created = helios_hal::create_backend(&manifest)?;
    let (actual_cap, triage) = (created.device_arch, created.triage);
    let max_seq_len = resolve_max_seq_len(cli.max_seq_len, &manifest.model, created.hardware.total_vram_bytes)?;
    let backend: Arc<helios_hal::DynBackend> = Arc::from(created.backend);
    tracing::info!(actual = %actual_cap, outcome = ?triage.outcome, "triage done");

    // 5. Bring up the engine state machine and load the weights. `Ready` means
    //    the device is probed and triage passed; `Loaded` means the weights are
    //    really in device memory, so it is set only after `load_model_weights`
    //    succeeds. A failed load exits non-zero at startup (fail fast) and the
    //    backend is dropped, which releases the device context.
    let sm = helios_core::StateMachine::new();
    sm.mark_ready()?;
    backend
        .load_model_weights(&cli.model, &manifest)
        .map_err(|e| anyhow!("loading model weights failed: {e:#}"))?;
    sm.mark_loaded()?;
    tracing::info!(status = ?sm.status(), "weights loaded");

    // 6. Build the ingress channel and spawn the scheduler. The receiver
    //    moves into the scheduler task; the sender moves into `AppState`.
    //    A *supervisor* task awaits the scheduler's `JoinHandle` (in turn
    //    observing the scheduler's `Result`) and moves the engine to
    //    `Stopped` when it resolves — this covers `Ok`, `Err`, and panic
    //    (a panic surfaces as a `JoinError` on `await`). Relying on a line
    //    after `run()` inside the scheduler task would miss the panic case,
    //    since that line never runs.
    //
    //    `sched_cancel` is the cooperative shutdown signal: the CLI trips it
    //    (below, step 9) after the server stops serving, then waits
    //    `SCHED_EXIT_TIMEOUT` for `run()` to exit at a safe point (it checks
    //    `is_cancelled()` between batch iterations — never mid-call). Only if
    //    that timeout elapses do we `abort()` the scheduler task itself (via
    //    its `AbortHandle`; aborting the supervisor would only detach it) and
    //    wait to confirm it stopped. Abort can cut a device call mid-flight,
    //    so it is a last resort, and an unconfirmed stop skips backend
    //    teardown.
    let (ingress_tx, ingress_rx) = helios_core::scheduler::ingress_channel();
    let sched_cancel = CancellationToken::new();
    let sched_cancel_for_task = sched_cancel.clone();
    let scheduler =
        helios_core::Scheduler::new(backend.clone(), sm.clone(), ingress_rx);
    let sched_handle = tokio::spawn(async move {
        scheduler.run(sched_cancel_for_task).await
    });
    let sched_abort = sched_handle.abort_handle();

    let sm_for_sup = sm.clone();
    let mut sup_handle = tokio::spawn(async move {
        match sched_handle.await {
            Ok(Ok(())) => tracing::info!("scheduler task exited cleanly"),
            Ok(Err(e)) => tracing::error!(error = %e, "scheduler task failed"),
            Err(je) => tracing::error!(error = %je, "scheduler task panicked"),
        }
        // Any exit path → engine must be Stopped so subsequent API requests
        // are rejected consistently. `Stopped -> Stopped` is idempotent, so
        // a racing CLI shutdown is safe.
        let _ = sm_for_sup.shutdown();
    });

    // 7. Resolve the tokenizer asset from the manifest. A package whose
    //    manifest names a tokenizer must actually carry one — a missing
    //    file is a broken package and we refuse to boot (architecture
    //    rule: fail loudly, don't paper over with a per-request 501).
    //    `resolve_in_package` keeps `asset_path` inside the package dir:
    //    absolute paths and `..` components are rejected so the manifest
    //    cannot point at files outside the package. Empty `asset_path` is
    //    the only way to ship a tokenizer-less package; then we boot and
    //    the handler returns 501 per request.
    let tokenizer = if manifest.tokenizer.asset_path.as_os_str().is_empty() {
        None
    } else {
        let tok_path = resolve_in_package(&cli.model, &manifest.tokenizer.asset_path)?;
        if !tok_path.exists() {
            return Err(anyhow!(
                "manifest names tokenizer asset {p} but it does not exist",
                p = tok_path.display()
            ));
        }
        tracing::info!(tok = %tok_path.display(), "tokenizer asset located");
        let mut prompt_tokenizer = helios_core::tokenizer::PromptTokenizer::new(
            tok_path.to_string_lossy().into_owned(),
            Some(max_seq_len as usize),
        );
        // The model's own chat template, carried (and signed) in the manifest. One that does not
        // compile is a broken package: refuse to boot instead of serving prompts in the wrong format.
        if let Some(spec) = &manifest.chat_template {
            let template = helios_core::chat_template::ChatTemplate::new(&spec.template, spec.bos_token.clone(), spec.eos_token.clone())
                .map_err(|e| anyhow!("manifest chat_template: {e:#}"))?;
            tracing::info!("chat template loaded from the manifest");
            prompt_tokenizer = prompt_tokenizer.with_chat_template(template);
        }
        Some(prompt_tokenizer)
    };

    // 8. Assemble AppState and serve.
    let app_state = helios_core::api::AppState {
        state: Arc::new(sm.clone()),
        tokenizer,
        ingress_tx: Some(ingress_tx),
        inflight: helios_core::inflight::InflightTracker::new(),
    };
    let inflight = app_state.inflight.clone();
    let app = helios_core::api::router().with_state(app_state);
    let listener = tokio::net::TcpListener::bind((cli.bind, cli.port))
        .await
        .map_err(|e| anyhow!("bind {addr}:{port} failed: {e}", addr = cli.bind, port = cli.port))?;
    tracing::info!(addr = %listener.local_addr()?, "helios API up");

    // 9. Serve until a shutdown signal, a scheduler exit or a server failure.
    //    Sequence once one of those fires:
    //    a. engine -> `Draining` (API rejects new work),
    //    b. axum stops accepting and drains in-flight HTTP, bounded by
    //       `HTTP_DRAIN_TIMEOUT` counted from this point, not from startup,
    //    c. wait for in-flight requests (`InflightTracker`, same bound), then trip `sched_cancel`, wait up to `SCHED_EXIT_TIMEOUT`; on timeout
    //       abort the scheduler task and confirm it stopped,
    //    d. `backend.shutdown()` only if the scheduler is confirmed stopped;
    //       otherwise skip it and let process exit reclaim device memory,
    //    e. engine -> `Stopped`. A scheduler or server failure exits non-zero.
    let stop = CancellationToken::new();
    let stop_for_serve = stop.clone();
    let mut serve_task = tokio::spawn(async move {
        axum::serve(listener, app)
            .with_graceful_shutdown(async move { stop_for_serve.cancelled().await })
            .await
    });

    let mut failed = false;
    let mut serve_done = false;
    let mut sup_done = false;
    tokio::select! {
        _ = shutdown_signal() => tracing::info!("shutdown signal received"),
        _ = &mut sup_handle => {
            sup_done = true;
            failed = true;
            tracing::error!("scheduler exited before shutdown was requested; shutting down");
        }
        res = &mut serve_task => {
            serve_done = true;
            failed = true;
            tracing::error!(result = ?res, "http server stopped before shutdown was requested");
        }
    }

    match sm.drain_for_shutdown() {
        Ok(()) => tracing::info!(status = ?sm.status(), "engine draining"),
        Err(e) => tracing::warn!(error = %e, "engine drain failed; continuing shutdown"),
    }
    stop.cancel();
    if !serve_done {
        match tokio::time::timeout(HTTP_DRAIN_TIMEOUT, &mut serve_task).await {
            Ok(Ok(Ok(()))) => tracing::info!("http server drained"),
            Ok(Ok(Err(e))) => tracing::error!(error = %e, "axum serve returned an error"),
            Ok(Err(je)) => tracing::error!(error = %je, "http server task panicked"),
            Err(_) => {
                tracing::warn!(timeout = ?HTTP_DRAIN_TIMEOUT, "http drain timed out; aborting server");
                serve_task.abort();
            }
        }
    }

    // Streams run in tasks of their own that axum's drain does not see; wait for every request
    // the handlers counted, within the same bound, before the scheduler is told to stop.
    if inflight.wait_idle(HTTP_DRAIN_TIMEOUT).await {
        tracing::info!("no requests in flight");
    } else {
        tracing::warn!(remaining = inflight.current(), timeout = ?HTTP_DRAIN_TIMEOUT, "requests still in flight at the drain deadline; cancelling them");
    }

    sched_cancel.cancel();
    let mut scheduler_stopped = sup_done;
    if !sup_done {
        if tokio::time::timeout(SCHED_EXIT_TIMEOUT, &mut sup_handle).await.is_ok() {
            scheduler_stopped = true;
        } else {
            tracing::warn!(timeout = ?SCHED_EXIT_TIMEOUT, "scheduler did not exit in time; aborting it");
            sched_abort.abort();
            scheduler_stopped =
                tokio::time::timeout(ABORT_CONFIRM_TIMEOUT, &mut sup_handle).await.is_ok();
        }
    }

    if scheduler_stopped {
        tracing::info!("scheduler stopped");
        match backend.shutdown() {
            Ok(()) => tracing::info!("backend shut down"),
            Err(e) => tracing::error!(error = %e, "backend shutdown failed"),
        }
    } else {
        failed = true;
        tracing::error!("scheduler stop not confirmed; skipping backend shutdown");
    }
    let _ = sm.shutdown();
    tracing::info!(status = ?sm.status(), "engine stopped");
    if failed {
        return Err(anyhow!("helios stopped abnormally (see log)"));
    }
    Ok(())
}

/// Wait for Ctrl-C / SIGTERM to trigger `axum::serve`'s graceful shutdown.
async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("installing ctrl-c handler should not fail");
    };
    #[cfg(unix)]
    let terminate = async {
        use tokio::signal::unix::{signal, SignalKind};
        signal(SignalKind::terminate())
            .expect("installing SIGTERM handler should not fail")
            .recv()
            .await;
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();
    tokio::select! {
        _ = ctrl_c => {}
        _ = terminate => {}
    }
}

/// Deserialize a manifest from the package. Placeholder until the real
/// `.hyperion` codec exists — reads a JSON sidecar next to the package path.
fn load_manifest(path: &std::path::Path) -> Result<hyperion_specification::v1::Manifest> {
    let sidecar = path.with_extension("manifest.json");
    let bytes = std::fs::read(&sidecar)
        .map_err(|e| anyhow!("read manifest {p}: {e}", p = sidecar.display()))?;
    let wrapped = hyperion_specification::Manifest::from_json_bytes(&bytes).map_err(|e| anyhow!("manifest {p}: {e}", p = sidecar.display()))?;
    match wrapped {
        hyperion_specification::Manifest::V1(m) => Ok(m),
    }
}

/// Resolve a manifest's package-relative `asset_path` against the package's
/// own directory and guarantee the result stays *inside* that directory.
///
/// - `package_path` is the `.hyperion` file (or whatever the codec hands us
///   today); its parent is the package root.
/// - `asset_path` may be relative (`"tokenizer.json"`, `"assets/vocab.json"`).
/// - We reject absolute paths and any `..` component, so a hostile or buggy
///   manifest cannot escape the package and point at arbitrary host files.
/// - `CurDir` (`.`) components are collapsed by the join and are harmless.
///
/// Returns a path under the package dir. We do not `canonicalize` because
/// the asset need not exist yet at this layer; the caller checks existence.
fn resolve_in_package(package_path: &Path, asset_path: &Path) -> Result<PathBuf> {
    if asset_path.is_absolute() {
        return Err(anyhow!(
            "tokenizer asset_path {p} is absolute; must be relative to the package",
            p = asset_path.display()
        ));
    }
    if asset_path.components().any(|c| matches!(c, Component::ParentDir)) {
        return Err(anyhow!(
            "tokenizer asset_path {p} contains '..'; must stay inside the package",
            p = asset_path.display()
        ));
    }
    // `parent()` is None only for paths like "foo" with no separator. The
    // CLI requires a real package file path, so hitting None is misuse.
    let package_dir = package_path
        .parent()
        .ok_or_else(|| anyhow!("package path {p} has no parent dir", p = package_path.display()))?;
    Ok(package_dir.join(asset_path))
}

#[cfg(test)]
mod tests {
    use hyperion_specification::v1::{Activation, AttentionKind, ModelConfig, RopeStyle};

    use super::{kv_bytes_per_sequence, resolve_max_seq_len};

    /// Qwen2-7B as published: 28 layers, 4 KV heads, head_dim 128, 131072 ctx.
    fn qwen2_7b() -> ModelConfig {
        ModelConfig {
            num_layers: 28,
            attention: AttentionKind::Gqa,
            num_attention_heads: 28,
            num_kv_heads: 4,
            head_dim: 128,
            vocab_size: 152_064,
            max_seq_len: 131_072,
            hidden_size: 3584,
            intermediate_size: 18944,
            rope_theta: 10000.0,
            rope_style: RopeStyle::RotateHalf,
            rms_norm_eps: 1e-6,
            tie_word_embeddings: false,
            qkv_bias: true,
            qk_norm: false,
            activation: Activation::Silu,
            attention_scale: None,
        }
    }

    #[test]
    fn kv_bytes_per_token_matches_the_hand_calculation() {
        // 28 layers x 4 kv heads x 128 dim x 2 (K and V) x 2 bytes = 57_344
        assert_eq!(kv_bytes_per_sequence(&qwen2_7b(), 1), 57_344);
        assert_eq!(kv_bytes_per_sequence(&qwen2_7b(), 131_072), 57_344 * 131_072);
    }

    #[test]
    fn defaults_to_the_model_limit() {
        assert_eq!(resolve_max_seq_len(None, &qwen2_7b(), 80 << 30).unwrap(), 131_072);
    }

    #[test]
    fn a_lower_runtime_limit_is_used() {
        assert_eq!(resolve_max_seq_len(Some(8192), &qwen2_7b(), 80 << 30).unwrap(), 8192);
    }

    #[test]
    fn zero_and_above_model_limit_are_refused() {
        for bad in [0, 131_073] {
            let msg = resolve_max_seq_len(Some(bad), &qwen2_7b(), 80 << 30).unwrap_err().to_string();
            assert!(msg.contains("1..=131072"), "{msg}");
        }
    }

    #[test]
    fn a_sequence_that_cannot_fit_on_the_device_is_refused_with_the_fix() {
        // 1 GiB of VRAM cannot hold 131072 tokens (~7.5 GB of KV).
        let msg = resolve_max_seq_len(None, &qwen2_7b(), 1 << 30).unwrap_err().to_string();
        assert!(msg.contains("lower --max-seq-len") && msg.contains("131072-token"), "{msg}");
        // ...but a short enough limit is accepted on the same device.
        assert_eq!(resolve_max_seq_len(Some(4096), &qwen2_7b(), 1 << 30).unwrap(), 4096);
    }

    #[test]
    fn a_backend_reporting_no_vram_is_not_capacity_checked() {
        assert_eq!(resolve_max_seq_len(None, &qwen2_7b(), 0).unwrap(), 131_072);
    }
}
