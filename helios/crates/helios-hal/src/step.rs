//! The scheduler's output, one `StepPlan` per step (ADR 0007), and what a step returns, with a compact versioned binary encoding.
//!
//! The plan names only what changed: the device holds each slot's page table and sampling state, so the plan carries page-table deltas and,
//! once per request, a slot initialisation. A row whose next input is the token the device just sampled says `FromDevice` instead of an id,
//! so a greedy chain never waits for the host. A row has `n >= 1` tokens (a decode row 1, a prefill chunk row up to the chunk capacity).
//!
//! Encoding: little-endian, fixed-width fields, `"HSTP"` magic, a format version, the step number, a body length and a plan hash (the first 8
//! bytes of the body's SHA-256, so a rank that applies a corrupted or diverging plan is told). A decoder refuses an unknown version, a bad
//! magic, a length or hash that does not match, and trailing bytes, each with its own error. `tests/data/step-plan-v1.bin` pins the format.

use hyperion_specification::sha256::Sha256;
use thiserror::Error;

pub const MAGIC: [u8; 4] = *b"HSTP";
pub const VERSION: u16 = 1;
/// A page-table delta that releases the entry instead of assigning it.
pub const RELEASED: u32 = u32::MAX;

/// Which captured graph runs, and how many rows are active in it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Capacity {
    /// A decode bucket of `bucket` rows (1, 2, 4, ...).
    Decode { bucket: u16 },
    /// A prefill chunk graph of `rows` rows (a power of two, 128 to 4096).
    Prefill { rows: u32 },
}

/// Where a row's input tokens come from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Input {
    /// Token ids from the host: a prompt or a chunk of it.
    Tokens(Vec<u32>),
    /// The token the device sampled for this slot in the previous step.
    FromDevice,
}

/// What a row returns.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Want {
    /// Nothing (an intermediate prefill chunk).
    Nothing,
    /// The sampled token id of its last token.
    Token,
    /// The logits of its last token (verification).
    Logits,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Row {
    /// The sequence's slot on the device.
    pub slot: u32,
    pub input: Input,
    /// Number of tokens of this row (equals the ids of `Input::Tokens`; 1 for `FromDevice`).
    pub n_tokens: u32,
    /// Position of the row's first token.
    pub position: u32,
    /// Pages in the slot's table once this step's deltas are applied.
    pub table_len: u32,
    pub want: Want,
}

/// A page-table change: slot `slot`, table index `index` now points at physical page `page` (or [`RELEASED`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PageDelta {
    pub slot: u32,
    pub index: u32,
    pub page: u32,
}

/// Per-request sampling state, set on the device when the request is admitted.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SamplingInit {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: u32,
    pub repetition_penalty: f32,
    pub seed: u64,
}

impl SamplingInit {
    pub const GREEDY: SamplingInit = SamplingInit { temperature: 0.0, top_p: 1.0, top_k: 0, repetition_penalty: 1.0, seed: 0 };
}

#[derive(Debug, Clone, PartialEq)]
pub struct SlotInit {
    pub slot: u32,
    pub sampling: SamplingInit,
    pub max_new_tokens: u32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct StepPlan {
    pub step: u64,
    pub capacity: Capacity,
    pub rows: Vec<Row>,
    pub deltas: Vec<PageDelta>,
    /// Slots admitted in this step.
    pub inits: Vec<SlotInit>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RowOutput {
    pub slot: u32,
    /// The token ids the row produced (empty for `Want::Nothing`/`Logits`).
    pub tokens: Vec<u32>,
    pub logits: Option<Vec<f32>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct StepOutput {
    /// The step this is the output of (not necessarily the latest submitted: outputs may lag one step).
    pub step: u64,
    pub rows: Vec<RowOutput>,
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum PlanError {
    #[error("not a step plan: bad magic")]
    BadMagic,
    #[error("step plan format version {found} is not supported (this build reads {supported})")]
    UnsupportedVersion { found: u16, supported: u16 },
    #[error("the plan is truncated: it ends inside {0}")]
    Truncated(&'static str),
    #[error("the plan has {0} bytes after its end")]
    TrailingBytes(usize),
    #[error("the plan hash does not match its body: the plan was corrupted or two ranks diverged")]
    HashMismatch,
    #[error("invalid plan: {0}")]
    Invalid(String),
}

struct Writer(Vec<u8>);

impl Writer {
    fn u8(&mut self, v: u8) {
        self.0.push(v);
    }
    fn u16(&mut self, v: u16) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn u32(&mut self, v: u32) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn u64(&mut self, v: u64) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn f32(&mut self, v: f32) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
}

struct Reader<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize, what: &'static str) -> Result<&'a [u8], PlanError> {
        let end = self.at.checked_add(n).filter(|&e| e <= self.bytes.len()).ok_or(PlanError::Truncated(what))?;
        let slice = &self.bytes[self.at..end];
        self.at = end;
        Ok(slice)
    }
    fn u8(&mut self, what: &'static str) -> Result<u8, PlanError> {
        Ok(self.take(1, what)?[0])
    }
    fn u16(&mut self, what: &'static str) -> Result<u16, PlanError> {
        Ok(u16::from_le_bytes(self.take(2, what)?.try_into().expect("2 bytes")))
    }
    fn u32(&mut self, what: &'static str) -> Result<u32, PlanError> {
        Ok(u32::from_le_bytes(self.take(4, what)?.try_into().expect("4 bytes")))
    }
    fn u64(&mut self, what: &'static str) -> Result<u64, PlanError> {
        Ok(u64::from_le_bytes(self.take(8, what)?.try_into().expect("8 bytes")))
    }
    fn f32(&mut self, what: &'static str) -> Result<f32, PlanError> {
        Ok(f32::from_le_bytes(self.take(4, what)?.try_into().expect("4 bytes")))
    }
}

fn body_hash(body: &[u8]) -> u64 {
    let mut h = Sha256::new();
    h.update(body);
    let digest = h.finalize();
    u64::from_le_bytes(digest[..8].try_into().expect("8 bytes"))
}

impl StepPlan {
    /// Check what can be checked without a device: every row's token count agrees with its input, `FromDevice` rows have one token, a decode
    /// capacity holds the rows, and a prefill capacity holds the row's tokens.
    pub fn validate(&self) -> Result<(), PlanError> {
        for row in &self.rows {
            match &row.input {
                Input::Tokens(ids) if ids.len() != row.n_tokens as usize || ids.is_empty() => {
                    return Err(PlanError::Invalid(format!("row of slot {} has {} token ids but n_tokens {}", row.slot, ids.len(), row.n_tokens)))
                }
                Input::FromDevice if row.n_tokens != 1 => return Err(PlanError::Invalid(format!("row of slot {} takes its input from the device but has {} tokens", row.slot, row.n_tokens))),
                _ => {}
            }
        }
        match self.capacity {
            Capacity::Decode { bucket } if self.rows.len() > bucket as usize => Err(PlanError::Invalid(format!("{} rows do not fit a decode bucket of {bucket}", self.rows.len()))),
            Capacity::Prefill { rows } if self.rows.iter().any(|r| r.n_tokens > rows) => Err(PlanError::Invalid(format!("a row has more tokens than the {rows}-row prefill graph"))),
            _ => Ok(()),
        }
    }

    /// The versioned binary form.
    pub fn encode(&self) -> Vec<u8> {
        let mut body = Writer(Vec::new());
        match self.capacity {
            Capacity::Decode { bucket } => {
                body.u8(0);
                body.u32(u32::from(bucket));
            }
            Capacity::Prefill { rows } => {
                body.u8(1);
                body.u32(rows);
            }
        }
        body.u32(self.rows.len() as u32);
        for r in &self.rows {
            body.u32(r.slot);
            body.u32(r.n_tokens);
            body.u32(r.position);
            body.u32(r.table_len);
            body.u8(match r.want {
                Want::Nothing => 0,
                Want::Token => 1,
                Want::Logits => 2,
            });
            match &r.input {
                Input::FromDevice => body.u8(0),
                Input::Tokens(ids) => {
                    body.u8(1);
                    body.u32(ids.len() as u32);
                    for id in ids {
                        body.u32(*id);
                    }
                }
            }
        }
        body.u32(self.deltas.len() as u32);
        for d in &self.deltas {
            body.u32(d.slot);
            body.u32(d.index);
            body.u32(d.page);
        }
        body.u32(self.inits.len() as u32);
        for i in &self.inits {
            body.u32(i.slot);
            body.f32(i.sampling.temperature);
            body.f32(i.sampling.top_p);
            body.u32(i.sampling.top_k);
            body.f32(i.sampling.repetition_penalty);
            body.u64(i.sampling.seed);
            body.u32(i.max_new_tokens);
        }
        let body = body.0;
        let mut out = Writer(Vec::with_capacity(body.len() + 26));
        out.0.extend_from_slice(&MAGIC);
        out.u16(VERSION);
        out.u64(self.step);
        out.u32(body.len() as u32);
        out.u64(body_hash(&body));
        out.0.extend_from_slice(&body);
        out.0
    }

    /// Decode and verify a plan.
    pub fn decode(bytes: &[u8]) -> Result<Self, PlanError> {
        let mut r = Reader { bytes, at: 0 };
        if r.take(4, "the magic").map_err(|_| PlanError::BadMagic)? != MAGIC {
            return Err(PlanError::BadMagic);
        }
        let version = r.u16("the version")?;
        if version != VERSION {
            return Err(PlanError::UnsupportedVersion { found: version, supported: VERSION });
        }
        let step = r.u64("the step number")?;
        let length = r.u32("the body length")? as usize;
        let hash = r.u64("the plan hash")?;
        let body = r.take(length, "the body")?;
        if r.at != bytes.len() {
            return Err(PlanError::TrailingBytes(bytes.len() - r.at));
        }
        if body_hash(body) != hash {
            return Err(PlanError::HashMismatch);
        }
        let mut b = Reader { bytes: body, at: 0 };
        let capacity = match b.u8("the capacity kind")? {
            0 => Capacity::Decode { bucket: u16::try_from(b.u32("the bucket")?).map_err(|_| PlanError::Invalid("decode bucket above 65535".into()))? },
            1 => Capacity::Prefill { rows: b.u32("the prefill rows")? },
            other => return Err(PlanError::Invalid(format!("unknown capacity kind {other}"))),
        };
        let n_rows = b.u32("the row count")? as usize;
        let mut rows = Vec::with_capacity(n_rows.min(4096));
        for _ in 0..n_rows {
            let (slot, n_tokens, position, table_len) = (b.u32("a slot")?, b.u32("n_tokens")?, b.u32("a position")?, b.u32("a table length")?);
            let want = match b.u8("a want")? {
                0 => Want::Nothing,
                1 => Want::Token,
                2 => Want::Logits,
                other => return Err(PlanError::Invalid(format!("unknown want {other}"))),
            };
            let input = match b.u8("an input kind")? {
                0 => Input::FromDevice,
                1 => {
                    let n = b.u32("a token count")? as usize;
                    let mut ids = Vec::with_capacity(n.min(1 << 16));
                    for _ in 0..n {
                        ids.push(b.u32("a token id")?);
                    }
                    Input::Tokens(ids)
                }
                other => return Err(PlanError::Invalid(format!("unknown input kind {other}"))),
            };
            rows.push(Row { slot, input, n_tokens, position, table_len, want });
        }
        let n_deltas = b.u32("the delta count")? as usize;
        let mut deltas = Vec::with_capacity(n_deltas.min(1 << 16));
        for _ in 0..n_deltas {
            deltas.push(PageDelta { slot: b.u32("a delta slot")?, index: b.u32("a delta index")?, page: b.u32("a delta page")? });
        }
        let n_inits = b.u32("the init count")? as usize;
        let mut inits = Vec::with_capacity(n_inits.min(1 << 12));
        for _ in 0..n_inits {
            inits.push(SlotInit {
                slot: b.u32("an init slot")?,
                sampling: SamplingInit { temperature: b.f32("a temperature")?, top_p: b.f32("a top-p")?, top_k: b.u32("a top-k")?, repetition_penalty: b.f32("a repetition penalty")?, seed: b.u64("a seed")? },
                max_new_tokens: b.u32("a max length")?,
            });
        }
        if b.at != body.len() {
            return Err(PlanError::TrailingBytes(body.len() - b.at));
        }
        Ok(Self { step, capacity, rows, deltas, inits })
    }

    /// The plan hash a rank compares with the scheduler's.
    pub fn hash(&self) -> u64 {
        let encoded = self.encode();
        u64::from_le_bytes(encoded[18..26].try_into().expect("8 bytes"))
    }
}

/// The page table of a slot rebuilt from deltas (`RELEASED` entries are `None`), for tests and for backends that mirror the device table.
pub fn apply_deltas(table: &mut Vec<Option<u32>>, slot: u32, deltas: &[PageDelta]) {
    for d in deltas.iter().filter(|d| d.slot == slot) {
        let index = d.index as usize;
        if table.len() <= index {
            table.resize(index + 1, None);
        }
        table[index] = (d.page != RELEASED).then_some(d.page);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> StepPlan {
        StepPlan {
            step: 41,
            capacity: Capacity::Decode { bucket: 4 },
            rows: vec![
                Row { slot: 0, input: Input::FromDevice, n_tokens: 1, position: 1023, table_len: 16, want: Want::Token },
                Row { slot: 3, input: Input::Tokens(vec![151644, 872, 198]), n_tokens: 3, position: 0, table_len: 1, want: Want::Logits },
            ],
            deltas: vec![PageDelta { slot: 0, index: 15, page: 902 }, PageDelta { slot: 3, index: 0, page: 7 }, PageDelta { slot: 1, index: 2, page: RELEASED }],
            inits: vec![SlotInit { slot: 3, sampling: SamplingInit { temperature: 0.7, top_p: 0.8, top_k: 20, repetition_penalty: 1.05, seed: 42 }, max_new_tokens: 2048 }],
        }
    }

    #[test]
    fn a_plan_round_trips_and_its_hash_is_the_hash_in_the_header() {
        let plan = sample();
        plan.validate().unwrap();
        let bytes = plan.encode();
        assert_eq!(&bytes[..4], b"HSTP");
        assert_eq!(StepPlan::decode(&bytes).unwrap(), plan);
        assert_eq!(plan.hash(), StepPlan::decode(&bytes).unwrap().hash());
        let mut other = plan.clone();
        other.deltas[0].page += 1;
        assert_ne!(other.hash(), plan.hash(), "a different plan has a different hash");
        let mut later = plan.clone();
        later.step += 1;
        assert_eq!(later.hash(), plan.hash(), "the hash covers the body; the step number is in the header");
    }

    #[test]
    fn the_encoding_is_pinned_by_a_golden_file() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/step-plan-v1.bin");
        let bytes = sample().encode();
        if std::env::var_os("UPDATE_FIXTURES").is_some() {
            std::fs::write(&path, &bytes).unwrap();
        }
        assert_eq!(bytes, std::fs::read(&path).expect("golden file exists (UPDATE_FIXTURES=1 writes it)"), "the plan encoding changed: bump VERSION and the golden file");
    }

    #[test]
    fn a_corrupted_truncated_or_unknown_plan_is_refused_with_its_own_error() {
        let bytes = sample().encode();
        let mut bad = bytes.clone();
        bad[0] = b'X';
        assert_eq!(StepPlan::decode(&bad), Err(PlanError::BadMagic));
        let mut future = bytes.clone();
        future[4] = 9;
        assert_eq!(StepPlan::decode(&future), Err(PlanError::UnsupportedVersion { found: 9, supported: VERSION }));
        let mut flipped = bytes.clone();
        *flipped.last_mut().unwrap() ^= 1;
        assert_eq!(StepPlan::decode(&flipped), Err(PlanError::HashMismatch));
        assert!(matches!(StepPlan::decode(&bytes[..bytes.len() - 3]), Err(PlanError::Truncated(_))));
        let mut trailing = bytes.clone();
        trailing.push(0);
        assert_eq!(StepPlan::decode(&trailing), Err(PlanError::TrailingBytes(1)));
        assert_eq!(StepPlan::decode(b"HS"), Err(PlanError::BadMagic));
    }

    #[test]
    fn validation_catches_a_row_that_disagrees_with_its_input_or_its_graph() {
        let mut p = sample();
        p.rows[1].n_tokens = 2;
        assert!(p.validate().is_err());
        let mut p = sample();
        p.rows[0].n_tokens = 2;
        assert!(p.validate().is_err(), "a FromDevice row is one token");
        let mut p = sample();
        p.capacity = Capacity::Decode { bucket: 1 };
        assert!(p.validate().is_err(), "two rows do not fit a bucket of one");
        let mut p = sample();
        p.capacity = Capacity::Prefill { rows: 2 };
        assert!(p.validate().is_err(), "a three-token row does not fit a two-row graph");
    }

    #[test]
    fn deltas_rebuild_the_page_table_a_scheduler_holds() {
        // The scheduler's table of slot 0 after three steps equals the table rebuilt from the deltas it sent.
        let mut device: Vec<Option<u32>> = Vec::new();
        let steps = [vec![PageDelta { slot: 0, index: 0, page: 5 }, PageDelta { slot: 0, index: 1, page: 9 }], vec![PageDelta { slot: 0, index: 2, page: 4 }, PageDelta { slot: 1, index: 0, page: 8 }], vec![PageDelta { slot: 0, index: 1, page: RELEASED }]];
        for deltas in &steps {
            apply_deltas(&mut device, 0, deltas);
        }
        assert_eq!(device, vec![Some(5), None, Some(4)]);
    }
}
