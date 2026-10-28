//! End-to-end over the real `helios-cli` binary with the stub backend: start,
//! probe, triage, serve, refuse. No GPU needed. Fixtures resolve through
//! `CARGO_MANIFEST_DIR`, the binary through `CARGO_BIN_EXE_helios-cli`.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(name)
}

/// A package dir under the target dir holding `model.manifest.json`, with
/// `edit` applied to the fixture's JSON first.
fn package(tag: &str, edit: impl FnOnce(&mut serde_json::Value)) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("pkg-{tag}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let mut json: serde_json::Value = serde_json::from_slice(&std::fs::read(fixture("stub-manifest.json")).unwrap()).unwrap();
    edit(&mut json);
    std::fs::write(dir.join("model.manifest.json"), serde_json::to_vec_pretty(&json).unwrap()).unwrap();
    dir.join("model.hyperion")
}

/// Owns a spawned server and kills it on drop, so a failing assertion cannot
/// leak a process that keeps `cargo test`'s stdout pipe open forever.
struct Server(Child);

impl std::ops::Deref for Server {
    type Target = Child;
    fn deref(&self) -> &Child {
        &self.0
    }
}

impl std::ops::DerefMut for Server {
    fn deref_mut(&mut self) -> &mut Child {
        &mut self.0
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn spawn(model: &PathBuf, extra: &[&str]) -> Server {
    Server(spawn_child(model, extra))
}

fn spawn_child(model: &PathBuf, extra: &[&str]) -> Child {
    Command::new(env!("CARGO_BIN_EXE_helios-cli"))
        .args(["--model", model.to_str().unwrap(), "--port", "0"])
        .args(extra)
        .env_remove("RUST_LOG")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap()
}

/// Read stdout until the "API up" line; returns the bound `ip:port`.
fn wait_for_addr(child: &mut Child) -> String {
    let out = child.stdout.take().unwrap();
    let deadline = Instant::now() + Duration::from_secs(20);
    for line in BufReader::new(out).lines() {
        let line = line.unwrap();
        if let Some(i) = line.find("addr=") {
            let rest = &line[i + 5..];
            let end = rest.find(|c: char| !(c.is_ascii_digit() || c == '.' || c == ':')).unwrap_or(rest.len());
            return rest[..end].to_string();
        }
        assert!(Instant::now() < deadline, "server did not come up");
    }
    panic!("server exited before binding");
}

fn http(addr: &str, method: &str, path: &str, body: &str) -> u16 {
    http_full(addr, method, path, body).0
}

fn http_full(addr: &str, method: &str, path: &str, body: &str) -> (u16, String) {
    let mut s = TcpStream::connect(addr).unwrap();
    let req = format!(
        "{method} {path} HTTP/1.1\r\nHost: t\r\nConnection: close\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    );
    s.write_all(req.as_bytes()).unwrap();
    let mut raw = String::new();
    s.read_to_string(&mut raw).unwrap();
    let code = raw.split_whitespace().nth(1).unwrap().parse().unwrap();
    (code, raw.split("\r\n\r\n").nth(1).unwrap_or("").to_string())
}

/// Run to completion and return (success, stdout + stderr).
fn run_to_exit(model: &PathBuf, extra: &[&str]) -> (bool, String) {
    let out = spawn_child(model, extra).wait_with_output().unwrap();
    let text = format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
    (out.status.success(), text)
}

#[test]
fn serves_on_loopback_with_health_and_v1_routes() {
    let mut child = spawn(&package("ok", |_| {}), &[]);
    let addr = wait_for_addr(&mut child);
    assert!(addr.starts_with("127.0.0.1:"), "default bind must be loopback, got {addr}");
    assert_eq!(http(&addr, "GET", "/health", ""), 200);
    let body = r#"{"model":"stub","messages":[{"role":"user","content":"hi"}]}"#;
    // The package ships no tokenizer: the documented capability error.
    assert_eq!(http(&addr, "POST", "/v1/chat/completions", body), 501);
    assert_eq!(http(&addr, "POST", "/chat/completions", body), 404);
}

#[test]
fn package_for_another_arch_is_refused_without_a_panic() {
    let (ok, text) = run_to_exit(&package("arch", |j| j["target_arch"] = "sm_90".into()), &[]);
    assert!(!ok);
    assert!(text.contains("no compiled-in backend can run this package"), "{text}");
    assert!(text.contains("sm_90") && text.contains("rebuild helios-cli"), "{text}");
    assert!(!text.contains("panicked"), "{text}");
}

#[test]
fn max_seq_len_above_the_model_limit_is_refused_before_the_socket_opens() {
    let (ok, text) = run_to_exit(&package("cap", |_| {}), &["--max-seq-len", "65"]);
    assert!(!ok);
    assert!(text.contains("--max-seq-len 65") && text.contains("1..=64"), "{text}");
    assert!(!text.contains("API up"), "{text}");
}

#[test]
fn max_seq_len_within_the_limit_is_accepted() {
    let mut child = spawn(&package("capok", |_| {}), &["--max-seq-len", "32"]);
    let addr = wait_for_addr(&mut child);
    assert_eq!(http(&addr, "GET", "/health", ""), 200);
}

#[test]
fn health_reports_loaded_once_the_weights_are_loaded() {
    let mut child = spawn(&package("loaded", |_| {}), &[]);
    let addr = wait_for_addr(&mut child);
    let (code, body) = http_full(&addr, "GET", "/health", "");
    assert_eq!(code, 200);
    assert!(body.contains("Loaded"), "engine must be Loaded after load_model_weights: {body}");
}
