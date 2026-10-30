//! SHA-256 through the RustCrypto `sha2` crate: one implementation, hardware-accelerated where the CPU has SHA instructions. The wrapper
//! keeps the three shapes the project needs: incremental hashing, a hex digest of a byte slice, and a streaming digest of a reader (a
//! 15 GB weights file is read in blocks, never loaded whole). The NIST vectors below and Python `hashlib`'s digests in
//! `tests/manifest.rs` guard the wrapper.

use std::io::{self, Read};

use sha2::Digest;

/// Incremental SHA-256.
#[derive(Default)]
pub struct Sha256(sha2::Sha256);

impl Sha256 {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn update(&mut self, data: &[u8]) {
        self.0.update(data);
    }

    pub fn finalize(self) -> [u8; 32] {
        self.0.finalize().into()
    }

    /// The digest as 64 lowercase hex digits.
    pub fn finalize_hex(self) -> String {
        to_hex(&self.finalize())
    }
}

/// Lowercase hex of bytes.
pub fn to_hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// Lowercase hex of the SHA-256 of `data`.
pub fn sha256_hex(data: &[u8]) -> String {
    to_hex(&sha2::Sha256::digest(data))
}

/// Lowercase hex of the SHA-256 of everything `reader` yields, read in blocks of 1 MiB.
pub fn sha256_reader(mut reader: impl Read) -> io::Result<String> {
    let mut hasher = Sha256::new();
    let mut block = vec![0u8; 1 << 20];
    loop {
        match reader.read(&mut block)? {
            0 => return Ok(hasher.finalize_hex()),
            n => hasher.update(&block[..n]),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nist_vectors() {
        assert_eq!(sha256_hex(b""), "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
        assert_eq!(sha256_hex(b"abc"), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
        assert_eq!(
            sha256_hex(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
        );
    }

    #[test]
    fn one_million_a_in_pieces_and_from_a_reader_agree() {
        let data = vec![b'a'; 1_000_000];
        let expected = "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0";
        assert_eq!(sha256_hex(&data), expected);
        let mut h = Sha256::new();
        for chunk in data.chunks(997) {
            h.update(chunk);
        }
        assert_eq!(h.finalize_hex(), expected);
        assert_eq!(sha256_reader(&data[..]).unwrap(), expected, "a reader that returns short reads");
        assert_eq!(sha256_reader(io::empty()).unwrap(), sha256_hex(b""));
    }
}
