//! Randomized and adversarial tests of the multipart parser (RFC 2046). No dependency: a fixed-seed xorshift generator, so a failure reproduces.
use helios_speech::api::multipart::{boundary_of, check_content_length, parse, parse_with, Limits, Part};

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
    fn bytes(&mut self, n: usize) -> Vec<u8> {
        // biased toward the bytes that matter: CR, LF, dash, and a boundary-looking alphabet
        const SPICY: &[u8] = b"\r\n--\r\n-- \t";
        (0..n).map(|_| if self.below(3) == 0 { SPICY[self.below(SPICY.len())] } else { self.next() as u8 }).collect()
    }
}

fn contains(hay: &[u8], needle: &[u8]) -> bool {
    needle.len() <= hay.len() && hay.windows(needle.len()).any(|w| w == needle)
}

fn encode(boundary: &str, parts: &[Part], padding: &str) -> Vec<u8> {
    let mut out = b"preamble line\r\n".to_vec();
    for p in parts {
        out.extend_from_slice(format!("--{boundary}{padding}\r\nContent-Disposition: form-data; name=\"{}\"", p.name).as_bytes());
        if let Some(f) = &p.filename {
            out.extend_from_slice(format!("; filename=\"{f}\"").as_bytes());
        }
        out.extend_from_slice(b"\r\n\r\n");
        out.extend_from_slice(&p.data);
        out.extend_from_slice(b"\r\n");
    }
    out.extend_from_slice(format!("--{boundary}--\r\nepilogue").as_bytes());
    out
}

fn random_case(r: &mut Rng) -> (String, Vec<Part>) {
    let parts: Vec<Part> = (0..1 + r.below(5))
        .map(|i| Part { name: format!("f{i}"), filename: (r.below(2) == 0).then(|| format!("a{i}.wav")), data: { let n = r.below(600); r.bytes(n) } })
        .collect();
    loop {
        const ALPHA: &[u8] = b"abcXYZ019'()+_,-./:=?";
        let b: String = (0..1 + r.below(30)).map(|_| ALPHA[r.below(ALPHA.len())] as char).collect();
        // the sender picks a boundary whose delimiter occurs in no payload (RFC 2046); retry otherwise
        if !b.ends_with(' ') && !parts.iter().any(|p| contains(&p.data, format!("--{b}").as_bytes())) {
            return (b, parts);
        }
    }
}

#[test]
fn random_payloads_round_trip_byte_exact() {
    let mut r = Rng(0x9E3779B97F4A7C15);
    for case in 0..3000 {
        let (b, parts) = random_case(&mut r);
        let padding = ["", " ", "\t  "][r.below(3)];
        let got = parse(&format!("multipart/form-data; boundary={b}"), &encode(&b, &parts, padding)).unwrap_or_else(|e| panic!("case {case} boundary {b:?}: {e}"));
        assert_eq!(got, parts, "case {case} boundary {b:?}");
    }
}

#[test]
fn lookalikes_inside_data_are_data_but_real_delimiters_split_even_in_binary() {
    let ct = "multipart/form-data; boundary=BB";
    let data = b"x\r\n--BBX not a delimiter\r\n--BB- nor this\r\n--B nor this\r\n--BBB".to_vec(); // all prefixes or partial matches, none followed by CRLF or two dashes
    let parts = vec![Part { name: "a".into(), filename: None, data: data.clone() }];
    assert_eq!(parse(ct, &encode("BB", &parts, "")).unwrap(), parts);
    // a payload that really contains the delimiter ends the part there, whatever the bytes look like
    let evil = b"head\r\n--BB\r\nContent-Disposition: form-data; name=\"injected\"\r\n\r\nv".to_vec();
    let got = parse(ct, &encode("BB", &[Part { name: "a".into(), filename: None, data: evil }], "")).unwrap();
    assert!(got.iter().any(|p| p.name == "injected"), "a real delimiter must split");
}

#[test]
fn every_truncation_is_an_error_or_a_clean_parse_never_a_panic() {
    let mut r = Rng(7);
    let (b, parts) = random_case(&mut r);
    let full = encode(&b, &parts, "");
    let ct = format!("multipart/form-data; boundary={b}");
    let close = full.windows(b.len() + 4).rposition(|w| w == format!("--{b}--").as_bytes()).unwrap();
    for cut in 0..full.len() {
        let res = parse(&ct, &full[..cut]);
        if cut < close + b.len() + 4 {
            assert!(res.is_err(), "cut {cut} before the close delimiter must fail, got {res:?}");
        }
    }
    assert!(parse(&ct, &full).is_ok());
}

#[test]
fn missing_close_delimiter_and_garbage() {
    let ct = "multipart/form-data; boundary=BB";
    let no_close = b"--BB\r\nContent-Disposition: form-data; name=\"a\"\r\n\r\nv\r\n";
    assert!(parse(ct, no_close).is_err());
    assert!(parse(ct, b"").is_err() && parse(ct, b"--BBX\r\n").is_err());
    assert!(parse("multipart/form-data", b"--BB--").is_err());
    assert!(boundary_of(&format!("multipart/form-data; boundary={}", "x".repeat(71))).is_err());
}

#[test]
fn limits_apply_before_and_while_reading() {
    let l = Limits { max_body: 400, max_file: 50, max_parts: 2, max_header: 80 };
    assert_eq!(check_content_length(401, &l).unwrap_err().status, 413);
    assert!(check_content_length(400, &l).is_ok());
    let ct = "multipart/form-data; boundary=BB";
    let part = |n: &str, d: &[u8]| Part { name: n.into(), filename: None, data: d.to_vec() };
    assert!(parse_with(ct, &encode("BB", &[part("a", &[1; 50])], ""), &l).is_ok());
    assert_eq!(parse_with(ct, &encode("BB", &[part("a", &[1; 51])], ""), &l).unwrap_err().status, 413, "file size");
    assert_eq!(parse_with(ct, &encode("BB", &[part("a", b"1"), part("b", b"2"), part("c", b"3")], ""), &l).unwrap_err().status, 413, "part count");
    let huge_header = format!("--BB\r\nContent-Disposition: form-data; name=\"a\"\r\nX-Pad: {}\r\n\r\nv\r\n--BB--", "p".repeat(200));
    assert_eq!(parse_with(ct, huge_header.as_bytes(), &Limits { max_body: 4096, ..l.clone() }).unwrap_err().status, 400, "header size");
    assert_eq!(parse_with(ct, &vec![b'x'; 401], &l).unwrap_err().status, 413, "body size");
}
