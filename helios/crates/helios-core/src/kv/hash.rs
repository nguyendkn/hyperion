//! Chained prefix hashes (ADR 0008). A block's hash covers its tokens and its parent's hash; the parent of the first block is the 32-byte
//! cache identity of the model (ADR 0010), so equal hashes mean equal tokens run by the same weights the same way on the same hardware,
//! and prefixes of two models can never alias. The hash block size is part of every hash.

use hyperion_specification::sha256::Sha256;

/// Domain separation: the same bytes hashed for another purpose give another digest.
const DOMAIN: &[u8] = b"hyperion.kv-block.v1";

/// A 32-byte digest: the identity at the root of a chain, or the hash of a block.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BlockHash(pub [u8; 32]);

impl BlockHash {
    /// The root of every chain of a model: its `cache_identity()` (64 lowercase hex digits). `None` when it is not one.
    pub fn from_identity(hex: &str) -> Option<Self> {
        if hex.len() != 64 || !hex.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)) {
            return None;
        }
        let mut bytes = [0u8; 32];
        for (i, byte) in bytes.iter_mut().enumerate() {
            *byte = u8::from_str_radix(&hex[2 * i..2 * i + 2], 16).ok()?;
        }
        Some(Self(bytes))
    }

    pub fn short(&self) -> String {
        self.0[..4].iter().map(|b| format!("{b:02x}")).collect()
    }
}

/// The hash of the block of `tokens` that follows `parent`.
pub fn block_hash(parent: &BlockHash, block_tokens: usize, tokens: &[u32]) -> BlockHash {
    let mut h = Sha256::new();
    h.update(DOMAIN);
    h.update(&parent.0);
    h.update(&(block_tokens as u32).to_le_bytes());
    for token in tokens {
        h.update(&token.to_le_bytes());
    }
    BlockHash(h.finalize())
}

/// The hashes of every complete block of `tokens`, in order (a trailing partial block has none).
pub fn chain(root: &BlockHash, block_tokens: usize, tokens: &[u32]) -> Vec<BlockHash> {
    let mut parent = *root;
    tokens
        .chunks_exact(block_tokens)
        .map(|block| {
            parent = block_hash(&parent, block_tokens, block);
            parent
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn root(byte: u8) -> BlockHash {
        BlockHash([byte; 32])
    }

    #[test]
    fn a_chain_depends_on_every_earlier_token_and_on_the_root() {
        let tokens: Vec<u32> = (0..256).collect();
        let a = chain(&root(1), 64, &tokens);
        assert_eq!(a.len(), 4);
        let mut changed = tokens.clone();
        changed[3] += 1;
        let b = chain(&root(1), 64, &changed);
        assert!(a.iter().zip(&b).all(|(x, y)| x != y), "a change in block 0 changes every later hash");
        let other_model = chain(&root(2), 64, &tokens);
        assert!(a.iter().zip(&other_model).all(|(x, y)| x != y), "the same tokens under another identity never share a hash");
        assert_eq!(chain(&root(1), 64, &tokens[..200]).len(), 3, "a partial block has no hash");
        assert_eq!(chain(&root(1), 64, &tokens[..200]), a[..3]);
    }

    #[test]
    fn the_block_size_is_part_of_the_hash() {
        let tokens: Vec<u32> = (0..128).collect();
        assert_ne!(chain(&root(1), 64, &tokens)[1], chain(&root(1), 128, &tokens)[0]);
    }

    #[test]
    fn an_identity_is_64_lowercase_hex_digits() {
        assert!(BlockHash::from_identity(&"ab".repeat(32)).is_some());
        assert_eq!(BlockHash::from_identity(&"ab".repeat(32)).unwrap().0[0], 0xab);
        assert!(BlockHash::from_identity("ab").is_none());
        assert!(BlockHash::from_identity(&"AB".repeat(32)).is_none());
    }
}
