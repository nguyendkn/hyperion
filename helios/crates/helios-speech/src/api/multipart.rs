//! Minimal `multipart/form-data` parser (RFC 7578), in-repo so no dependency is added (S-A5). Handles quoted boundaries, `name`/`filename`
//! in `Content-Disposition`, binary bodies. Refuses a missing boundary, an unterminated body and header sections that do not parse.
use super::error::ApiError;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Part {
    pub name: String,
    pub filename: Option<String>,
    pub data: Vec<u8>,
}

fn bad(m: &str) -> ApiError {
    ApiError::bad_request("invalid_multipart", None, format!("malformed multipart body: {m}"))
}

pub fn boundary_of(content_type: &str) -> Result<String, ApiError> {
    let mut it = content_type.split(';');
    if !it.next().unwrap_or("").trim().eq_ignore_ascii_case("multipart/form-data") {
        return Err(ApiError::bad_request("invalid_content_type", None, "Content-Type must be multipart/form-data"));
    }
    it.filter_map(|p| p.trim().strip_prefix("boundary=")).map(|b| b.trim_matches('"').to_string()).find(|b| !b.is_empty()).ok_or_else(|| bad("no boundary"))
}

fn find(hay: &[u8], needle: &[u8], from: usize) -> Option<usize> {
    if needle.is_empty() || hay.len() < needle.len() {
        return None;
    }
    (from..=hay.len() - needle.len()).find(|&i| &hay[i..i + needle.len()] == needle)
}

fn param(header: &str, key: &str) -> Option<String> {
    let pat = format!("{key}=\"");
    let start = header.find(&pat)? + pat.len();
    // a filename may contain `;`, so read to the closing quote
    Some(header[start..start + header[start..].find('"')?].to_string())
}

pub fn parse(content_type: &str, body: &[u8]) -> Result<Vec<Part>, ApiError> {
    let delim = format!("--{}", boundary_of(content_type)?).into_bytes();
    let mut pos = find(body, &delim, 0).ok_or_else(|| bad("opening boundary not found"))? + delim.len();
    let mut parts = Vec::new();
    loop {
        if body[pos..].starts_with(b"--") {
            return Ok(parts);
        }
        if !body[pos..].starts_with(b"\r\n") {
            return Err(bad("boundary line not followed by CRLF"));
        }
        pos += 2;
        let head_end = find(body, b"\r\n\r\n", pos).ok_or_else(|| bad("part headers not terminated"))?;
        let headers = std::str::from_utf8(&body[pos..head_end]).map_err(|_| bad("part headers are not UTF-8"))?;
        let disp = headers.lines().find(|l| l.to_ascii_lowercase().starts_with("content-disposition:")).ok_or_else(|| bad("part without Content-Disposition"))?;
        let name = param(disp, "name").ok_or_else(|| bad("part without a name"))?;
        let data_start = head_end + 4;
        let mut next = vec![b'\r', b'\n'];
        next.extend_from_slice(&delim);
        // a delimiter counts only when followed by CRLF or `--`; the same bytes inside binary data are data
        let mut from = data_start;
        let data_end = loop {
            let i = find(body, &next, from).ok_or_else(|| bad("part not terminated by a boundary"))?;
            let after = &body[i + next.len()..];
            if after.starts_with(b"\r\n") || after.starts_with(b"--") {
                break i;
            }
            from = i + 1;
        };
        parts.push(Part { name, filename: param(disp, "filename"), data: body[data_start..data_end].to_vec() });
        pos = data_end + next.len();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn body(file: &[u8]) -> Vec<u8> {
        let mut b = b"--XyZ\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\nqwen3-asr\r\n--XyZ\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a;b.wav\"\r\nContent-Type: audio/wav\r\n\r\n".to_vec();
        b.extend_from_slice(file);
        b.extend_from_slice(b"\r\n--XyZ--\r\n");
        b
    }
    #[test]
    fn parses_fields_and_binary_file_containing_crlf_and_boundary_lookalikes() {
        let file = b"RIFF\r\n--XyZ-not-a-boundary\r\n\x00\xff";
        let parts = parse("multipart/form-data; boundary=XyZ", &body(file)).unwrap();
        assert_eq!(parts.len(), 2);
        assert_eq!((parts[0].name.as_str(), parts[0].data.as_slice()), ("model", &b"qwen3-asr"[..]));
        assert_eq!((parts[1].filename.as_deref(), parts[1].data.as_slice()), (Some("a;b.wav"), &file[..]));
    }
    #[test]
    fn quoted_boundary_and_refusals() {
        assert!(parse("multipart/form-data; boundary=\"XyZ\"", &body(b"x")).is_ok());
        assert!(boundary_of("application/json").is_err());
        assert!(boundary_of("multipart/form-data").is_err());
        let mut cut = body(b"abc");
        cut.truncate(cut.len() - 12);
        assert!(parse("multipart/form-data; boundary=XyZ", &cut).is_err());
        assert!(parse("multipart/form-data; boundary=XyZ", b"garbage").is_err());
    }
}
