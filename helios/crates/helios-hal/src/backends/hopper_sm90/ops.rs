//! Single ops on the sm90 backend, each a custom kernel from `kernels/elementwise.cu`.
//! They take and return host slices (upload, launch, synchronise, download) so a test can
//! compare them with `crate::reference` one op at a time. The assembled forward pass will
//! keep activations on the device instead; these are verification entry points.

use std::ffi::c_void;

use anyhow::{anyhow, Result};

use super::backend::HopperBackend;
use crate::cuda::{CudaDevice, Kernel, KernelModule};

/// PTX generated from `kernels/elementwise.cu` (rebuild with `scripts/build-ptx.sh`).
pub(super) const ELEMENTWISE_PTX: &str = include_str!("../../../kernels/elementwise.ptx");

const BLOCK: u32 = 256;

/// One kernel argument, in the order of the kernel's signature.
pub(super) enum Arg<'a> {
    /// Host bytes, uploaded to a fresh device buffer.
    Host(&'a [u8]),
    /// An existing device pointer (a loaded weight).
    Dev(u64),
    /// The output buffer (allocated by the runner, returned to the caller).
    Out,
    I32(i32),
}

fn bytes_of(values: &[u16]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn bf16_from(raw: &[u8]) -> Vec<u16> {
    raw.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect()
}

impl HopperBackend {
    /// Load (once) the PTX module stored under `key` and look up `name`.
    pub(super) fn kernel(&self, dev: &CudaDevice, key: &'static str, ptx: &str, name: &str) -> Result<Kernel> {
        let mut modules = self.modules.lock().expect("modules mutex poisoned");
        if !modules.contains_key(key) {
            modules.insert(key, KernelModule::load(dev, ptx)?);
        }
        modules[key].function(name)
    }

    /// Upload the host arguments, launch `name` over `grid` blocks of `block` threads, wait,
    /// and return the `Out` buffer's `out_bytes` bytes. Every device buffer is freed on all paths.
    pub(super) fn run_kernel(
        &self,
        key: &'static str,
        ptx: &str,
        name: &str,
        args: &[Arg],
        out_bytes: usize,
        grid: u32,
        block: u32,
        shared: u32,
    ) -> Result<Vec<u8>> {
        let mut dev = self.device.lock().expect("device mutex poisoned");
        let kernel = self.kernel(&dev, key, ptx, name)?;

        // Pass 1: allocate and upload; values are final before any pointer to them is taken.
        let mut allocated: Vec<u64> = Vec::new();
        let mut ptr_values: Vec<u64> = Vec::new();
        let mut int_values: Vec<i32> = Vec::new();
        let mut order: Vec<(bool, usize)> = Vec::new(); // (is_pointer, index)
        let mut out_ptr = 0u64;
        let setup = (|| -> Result<()> {
            for arg in args {
                match arg {
                    Arg::Host(bytes) => {
                        let p = dev.alloc(bytes.len().max(1))?;
                        allocated.push(p);
                        dev.htod(p, bytes)?;
                        order.push((true, ptr_values.len()));
                        ptr_values.push(p);
                    }
                    Arg::Dev(p) => {
                        order.push((true, ptr_values.len()));
                        ptr_values.push(*p);
                    }
                    Arg::Out => {
                        out_ptr = dev.alloc(out_bytes.max(1))?;
                        allocated.push(out_ptr);
                        order.push((true, ptr_values.len()));
                        ptr_values.push(out_ptr);
                    }
                    Arg::I32(v) => {
                        order.push((false, int_values.len()));
                        int_values.push(*v);
                    }
                }
            }
            Ok(())
        })();

        // Pass 2: launch.
        let result = setup.and_then(|()| {
            let mut params: Vec<*mut c_void> = order
                .iter()
                .map(|&(is_ptr, i)| {
                    if is_ptr {
                        &ptr_values[i] as *const u64 as *mut c_void
                    } else {
                        &int_values[i] as *const i32 as *mut c_void
                    }
                })
                .collect();
            // Safety: `args` mirror the kernel's signature (callers in this module).
            unsafe { kernel.launch(&dev, (grid, 1, 1), (block, 1, 1), shared, &mut params)? };
            dev.synchronize()?;
            let mut raw = vec![0u8; out_bytes];
            dev.dtoh(&mut raw, out_ptr)?;
            Ok(raw)
        });
        for p in allocated {
            let _ = dev.free(p);
        }
        result
    }

    fn weight(&self, name: &str) -> Result<(u64, usize)> {
        self.weights
            .lock()
            .expect("weights mutex poisoned")
            .as_ref()
            .and_then(|w| w.pointer(name))
            .ok_or_else(|| anyhow!("weight {name} is not loaded"))
    }

    /// Gather rows of a loaded bf16 embedding table `[vocab, hidden]` for `ids`.
    pub fn embedding_bf16(&self, table: &str, ids: &[u32], hidden: usize) -> Result<Vec<u16>> {
        let (ptr, len) = self.weight(table)?;
        if hidden == 0 || len % (hidden * 2) != 0 {
            return Err(anyhow!("table {table} holds {len} bytes, not a whole number of bf16 rows of {hidden}"));
        }
        let vocab = len / (hidden * 2);
        if let Some(bad) = ids.iter().find(|&&i| i as usize >= vocab) {
            return Err(anyhow!("token id {bad} is outside the vocabulary of {vocab}"));
        }
        let id_bytes: Vec<u8> = ids.iter().flat_map(|v| v.to_le_bytes()).collect();
        let raw = self.run_kernel(
            "elementwise",
            ELEMENTWISE_PTX,
            "embedding_gather_bf16",
            &[Arg::Host(&id_bytes), Arg::Dev(ptr), Arg::Out, Arg::I32(i32::try_from(hidden)?)],
            ids.len() * hidden * 2,
            u32::try_from(ids.len())?,
            BLOCK,
            0,
        )?;
        Ok(bf16_from(&raw))
    }

    fn binary(&self, name: &str, a: &[u16], b: &[u16]) -> Result<Vec<u16>> {
        if a.len() != b.len() {
            return Err(anyhow!("operands differ in length: {} vs {}", a.len(), b.len()));
        }
        let n = i32::try_from(a.len())?;
        let (ab, bb) = (bytes_of(a), bytes_of(b));
        let raw = self.run_kernel(
            "elementwise",
            ELEMENTWISE_PTX,
            name,
            &[Arg::Host(&ab), Arg::Host(&bb), Arg::Out, Arg::I32(n)],
            a.len() * 2,
            u32::try_from(a.len().div_ceil(BLOCK as usize))?,
            BLOCK,
            0,
        )?;
        Ok(bf16_from(&raw))
    }

    /// `a + b` elementwise (residual connection).
    pub fn add_bf16(&self, a: &[u16], b: &[u16]) -> Result<Vec<u16>> {
        self.binary("add_bf16", a, b)
    }

    /// `bf16(silu(gate)) * up` elementwise (Qwen2's MLP gate).
    pub fn swiglu_bf16(&self, gate: &[u16], up: &[u16]) -> Result<Vec<u16>> {
        self.binary("swiglu_bf16", gate, up)
    }

    /// Rotary embedding of `x` (`[tokens, heads, head_dim]`) at `positions`, using
    /// `[max_pos, head_dim]` cos/sin tables.
    pub fn rope_bf16(
        &self,
        x: &[u16],
        positions: &[u32],
        heads: usize,
        head_dim: usize,
        cos: &[u16],
        sin: &[u16],
    ) -> Result<Vec<u16>> {
        if head_dim == 0 || head_dim % 2 != 0 || x.len() != positions.len() * heads * head_dim {
            return Err(anyhow!("x must be [tokens, heads, head_dim] with an even head_dim"));
        }
        if cos.len() != sin.len() || cos.len() % head_dim != 0 {
            return Err(anyhow!("cos/sin tables must both be [max_pos, head_dim]"));
        }
        let max_pos = cos.len() / head_dim;
        if let Some(bad) = positions.iter().find(|&&p| p as usize >= max_pos) {
            return Err(anyhow!("position {bad} is outside the table of {max_pos} positions"));
        }
        let (xb, cb, sb) = (bytes_of(x), bytes_of(cos), bytes_of(sin));
        let pb: Vec<u8> = positions.iter().flat_map(|v| v.to_le_bytes()).collect();
        let raw = self.run_kernel(
            "elementwise",
            ELEMENTWISE_PTX,
            "rope_bf16",
            &[
                Arg::Host(&xb),
                Arg::Host(&cb),
                Arg::Host(&sb),
                Arg::Host(&pb),
                Arg::Out,
                Arg::I32(i32::try_from(heads)?),
                Arg::I32(i32::try_from(head_dim)?),
            ],
            x.len() * 2,
            u32::try_from(positions.len() * heads)?,
            64,
            0,
        )?;
        Ok(bf16_from(&raw))
    }
}
