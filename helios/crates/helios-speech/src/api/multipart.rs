//! `multipart/form-data` parser per RFC 2046 section 5.1.1, in-repo so no dependency is added (S-A5).
//! A delimiter is `CRLF "--" boundary`, followed by `"--"` (the close delimiter) or by optional spaces/tabs and CRLF. It is a delimiter wherever
//! it occurs: the sender chose a boundary absent from the data. A longer token that merely starts with the boundary (`--boundaryX`) is not one.
//! Limits are checked before bytes are copied: `check_content_length` before the body is read at all, the rest while walking the parts.
use super::error::ApiError;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Part {
    pub name: String,
    pub filename: Option<String>,
    pub data: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Limits {
    pub max_body: usize,
    /// Largest single part (the audio file); default 25 MB like OpenAI's endpoint, a server flag changes it.
    pub max_file: usize,
    pub max_parts: usize,
    /// Bytes of one part's header block.
    pub max_header: usize,
}

impl Default for Limits {
    fn default() -> Self {
        Self { max_body: 26 << 20, max_file: 25 << 20, max_parts: 32, max_header: 8 << 10 }
    }
}

fn bad(m: &str) -> ApiError {
    ApiError::bad_request("invalid_multipart", None, format!("malformed multipart body: {m}"))
}

fn too_large(what: &str, limit: usize) -> ApiError {
    ApiError { status: 413, message: format!("{what} exceeds the limit of {limit} bytes"), param: None, code: "request_too_large" }
}

/// Call with the `Content-Length` before reading the body.
pub fn check_content_length(len: u64, l: &Limits) -> Result<(), ApiError> {
    if len > l.max_body as u64 { Err(too_large("request body", l.max_body)) } else { Ok(()) }
}

pub fn boundary_of(content_type: &str) -> Result<String, ApiError> {
    let mut it = content_type.split(';');
    if !it.next().unwrap_or("").trim().eq_ignore_ascii_case("multipart/form-data") {
        return Err(ApiError::bad_request("invalid_content_type", None, "Content-Type must be multipart/form-data"));
    }
    let b = it.filter_map(|p| p.trim().strip_prefix("boundary=")).map(|b| b.trim_matches('"').to_string()).find(|b| !b.is_empty()).ok_or_else(|| bad("no boundary"))?;
    if b.len() > 70 {
        return Err(bad("boundary longer than 70 characters"));
    }
    Ok(b)
}

fn find(hay: &[u8], needle: &[u8], from: usize) -> Option<usize> {
    if needle.is_empty() || hay.len() < needle.len() || from > hay.len() - needle.len() {
        return None;
    }
    (from..=hay.len() - needle.len()).find(|&i| &hay[i..i + needle.len()] == needle)
}

/// What follows `dash-boundary` at `at`: `Some((next_index, is_close))` when it is a real delimiter.
fn after_delimiter(body: &[u8], at: usize) -> Option<(usize, bool)> {
    if body[at..].starts_with(b"--") {
        return Some((at + 2, true));
    }
    let mut i = at;
    while i < body.len() && (body[i] == b' ' || body[i] == b'\t') {
        i += 1;
    }
    body[i..].starts_with(b"\r\n").then_some((i + 2, false))
}

fn param(header: &str, key: &str) -> Option<String> {
    // match ` key="` or `;key="` so that `name` does not match inside `filename`
    let pat = format!("{key}=\"");
    let mut from = 0;
    while let Some(p) = header[from..].find(&pat) {
        let at = from + p;
        let prev = header[..at].chars().next_back();
        if matches!(prev, Some(' ' | ';' | '\t')) {
            let start = at + pat.len();
            return Some(header[start..start + header[start..].find('"')?].to_string());
        }
        from = at + 1;
    }
    None
}

pub fn parse(content_type: &str, body: &[u8]) -> Result<Vec<Part>, ApiError> {
    parse_with(content_type, body, &Limits::default())
}

pub fn parse_with(content_type: &str, body: &[u8], limits: &Limits) -> Result<Vec<Part>, ApiError> {
    check_content_length(body.len() as u64, limits)?;
    let dash = format!("--{}", boundary_of(content_type)?).into_bytes();
    let mut sep = b"\r\n".to_vec();
    sep.extend_from_slice(&dash);
    // The opening delimiter has no CRLF before it when it is the first bytes of the body; otherwise it follows a preamble line.
    let mut pos = None;
    let mut from = 0;
    while let Some(i) = find(body, &dash, from) {
        if i == 0 || body[..i].ends_with(b"\r\n") {
            if let Some((next, close)) = after_delimiter(body, i + dash.len()) {
                if close { return Ok(Vec::new()) }
                pos = Some(next);
                break;
            }
        }
        from = i + 1;
    }
    let mut pos = pos.ok_or_else(|| bad("opening delimiter not found"))?;
    let mut parts = Vec::new();
    loop {
        if parts.len() >= limits.max_parts {
            return Err(too_large("number of parts (limit)", limits.max_parts));
        }
        let head_end = find(body, b"\r\n\r\n", pos).ok_or_else(|| bad("part headers not terminated"))?;
        if head_end - pos > limits.max_header {
            return Err(ApiError::bad_request("invalid_multipart", None, format!("a part header block is longer than {} bytes", limits.max_header)));
        }
        let headers = std::str::from_utf8(&body[pos..head_end]).map_err(|_| bad("part headers are not UTF-8"))?;
        let disp = headers.lines().find(|l| l.to_ascii_lowercase().starts_with("content-disposition:")).ok_or_else(|| bad("part without Content-Disposition"))?;
        let name = param(disp, "name").ok_or_else(|| bad("part without a name"))?;
        let data_start = head_end + 4;
        let mut from = data_start;
        let (data_end, next, close) = loop {
            let i = find(body, &sep, from).ok_or_else(|| bad("missing delimiter after a part (truncated body or no close delimiter)"))?;
            if let Some((next, close)) = after_delimiter(body, i + sep.len()) {
                break (i, next, close);
            }
            from = i + 1;
        };
        if data_end - data_start > limits.max_file {
            return Err(too_large("a part", limits.max_file));
        }
        parts.push(Part { name, filename: param(disp, "filename"), data: body[data_start..data_end].to_vec() });
        if close {
            return Ok(parts);
        }
        pos = next;
    }
}
