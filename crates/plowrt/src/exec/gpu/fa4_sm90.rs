//! Host stub of FlashAttention-4's SM90 forward and combine (vLLM 0.28.0 `vllm_flash_attn.cute`,
//! CuTe DSL 4.6.2), for the entries `scripts/fa4_aot/build.py` exports: what the export's own host
//! code computes per launch (kernel parameters, tensor maps, grid), rebuilt here so the packet
//! carries cubins only. The container records the upstream it was built from and exemplar
//! launches captured from that host code; `Fa4::load` refuses an object whose upstream or stub
//! version differs, or whose exemplars this stub does not reproduce.
//!
//! Entries: causal varlen forward over 64-row KV pages, whole-KV (`whole`) or split-KV (`split`:
//! f32 partials + LSE), and the `combine` that folds the partials. hd 512 (head-major cache,
//! TMA loads of Q / K / V, TMA store of O; GQA packed at 2 KV heads) and hd 256 (row-interleaved
//! cache, row-predicated loads, no tensor maps; GQA packed when heads > KV heads).

use crate::device::cuda::aot::{Entry, Image, Kernel, Launch};
use crate::device::cuda::{CudaBackend, CudaStream};
use crate::error::{Result, RuntimeError};

pub(super) const HOST_STUB: &str = "fa4_sm90_fwd";
pub(super) const HOST_STUB_VERSION: u32 = 1;
/// The upstream whose host logic this stub transcribes.
const UPSTREAM: [(&str, &str); 3] = [
    ("nvidia_cutlass_dsl", "4.6.2"),
    ("vllm", "0.28.0"),
    (
        "vllm_flash_attn_cute_sha256",
        "70bfc1f1938e73d2361f080dace4448478845eb8d699627b6752a160bda2ca22",
    ),
];
const PAGE: u32 = 64;
/// The forward's L2 swizzle budget (bytes) over `2 x (hd + hd_v) x tile_n`.
const L2_BUDGET: u32 = 52_428_800;
/// The varlen O store map offsets its base by this many rows (a virtual batch dimension).
const O_ROWS_SHIFT: u64 = 1 << 30;
/// A tensor map's global address field: 57 bits, 16-byte aligned.
const TMAP_ADDRESS_MASK: u64 = 0x01ff_ffff_ffff_fff0;
const COMBINE_TILE_M: u32 = 8;
const COMBINE_K_BLOCK: u32 = 128;
// cuda.h enums.
const TMAP_BF16: u32 = 9;
const TMAP_F16: u32 = 6;
const SWIZZLE_128B: u32 = 3;
const L2_PROMOTION_128B: u32 = 2;

/// The exported geometry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Geometry {
    pub(super) head_dim: u32,
    pub(super) heads: u32,
    pub(super) f16: bool,
}

impl Geometry {
    /// hd 512 reads Q / K / V and writes O through tensor maps; hd 256 does not.
    fn tma(&self) -> bool {
        self.head_dim == 512
    }
    fn pack(&self, kv_heads: u32) -> bool {
        if self.tma() {
            kv_heads == 2
        } else {
            self.heads > kv_heads
        }
    }
    fn tile_m(&self) -> u32 {
        if self.tma() {
            64
        } else {
            128
        }
    }
    fn tile_n(&self) -> u32 {
        if self.tma() {
            64
        } else {
            80
        }
    }
}

/// One forward or combine call's operands (device addresses, extents in elements).
#[derive(Clone, Copy, Debug)]
pub(super) struct Call {
    pub(super) q: u64,
    pub(super) k: u64,
    pub(super) v: u64,
    /// Whole: bf16 output; split: f32 partials `[splits][total_q][heads][hd]`. Combine: the
    /// partials it reads.
    pub(super) o: u64,
    /// Split: LSE `[splits][heads][total_q]`. Combine: the partial LSE it reads.
    pub(super) lse: u64,
    /// Combine: its output `[total_q][heads][hd]`.
    pub(super) out: u64,
    pub(super) cu_q: u64,
    pub(super) used_k: u64,
    pub(super) page_table: u64,
    pub(super) total_q: u32,
    pub(super) requests: u32,
    pub(super) width: u32,
    pub(super) pages: u32,
    pub(super) head_stride: u64,
    pub(super) splits: u32,
    pub(super) scale: f32,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Role {
    Whole,
    Split,
    Combine,
}

struct Prepared {
    kernel: Kernel,
    role: Role,
    kv_heads: u32,
    smem: u32,
    block: [u32; 3],
}

/// The loaded entries of one container.
pub(super) struct Fa4 {
    geometry: Geometry,
    entries: Vec<Prepared>,
}

impl Fa4 {
    /// Loads every kernel entry of `image` and checks it against this stub (module doc).
    pub(super) fn load(be: &CudaBackend, image: &Image, geometry: Geometry) -> Result<Fa4> {
        let m = &image.manifest;
        let refuse = |why: String| RuntimeError::Rejected(format!("FA4 object: {why}"));
        if m.host_stub != HOST_STUB || m.host_stub_version != HOST_STUB_VERSION {
            return Err(refuse(format!(
                "host stub {} v{}, runtime has {HOST_STUB} v{HOST_STUB_VERSION}",
                m.host_stub, m.host_stub_version
            )));
        }
        let upstream: Vec<(&str, &str)> = m
            .upstream
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect();
        if upstream != UPSTREAM {
            return Err(refuse(format!(
                "built from {upstream:?}, the runtime's stub transcribes {UPSTREAM:?}"
            )));
        }
        let mut entries = Vec::new();
        for (e, role) in kernel_entries(&m.entries).map_err(refuse)? {
            let kernel = Kernel::load(be, image, e)?;
            let first = &e.exemplars[0];
            let p = Prepared {
                kernel,
                role,
                kv_heads: e.kv_heads,
                smem: first.smem,
                block: first.block,
            };
            check_exemplars(be, geometry, &p, e)
                .map_err(|why| refuse(format!("{}: {why}", e.name)))?;
            entries.push(p);
        }
        Ok(Fa4 { geometry, entries })
    }

    pub(super) fn unload(&mut self, be: &CudaBackend) -> Result<()> {
        for p in self.entries.drain(..) {
            be.module_unload(&p.kernel.module)?;
        }
        Ok(())
    }

    fn entry(&self, role: Role, kv_heads: u32) -> Result<&Prepared> {
        self.entries
            .iter()
            .find(|p| p.role == role && (role == Role::Combine || p.kv_heads == kv_heads))
            .ok_or_else(|| {
                RuntimeError::Rejected(format!(
                    "FA4 object: no {role:?} entry at {kv_heads} KV heads"
                ))
            })
    }

    /// The forward (split when `c.splits > 1`), then the combine.
    pub(super) fn run(
        &self,
        be: &CudaBackend,
        kv_heads: u32,
        c: &Call,
        stream: &CudaStream,
    ) -> Result<()> {
        let role = if c.splits > 1 {
            Role::Split
        } else {
            Role::Whole
        };
        let p = self.entry(role, kv_heads)?;
        let mut l = forward(be, self.geometry, p, c)?;
        p.kernel.launch(be, &mut l, stream)?;
        if role == Role::Split {
            let p = self.entry(Role::Combine, 0)?;
            let mut l = combine(self.geometry, p, c);
            p.kernel.launch(be, &mut l, stream)?;
        }
        Ok(())
    }
}

/// The name of the container's one adapter entry; `LibAttention::load` hash- and ABI-checks it.
pub(super) const ADAPTERS: &str = "kv_adapters";

/// The kernel entries with their roles. Every entry is a kernel this stub loads (hash, layout and
/// exemplar checked) or the one `adapters` entry named [`ADAPTERS`]; anything else is refused.
fn kernel_entries(entries: &[Entry]) -> std::result::Result<Vec<(&Entry, Role)>, String> {
    let mut names = std::collections::BTreeSet::new();
    let mut adapters = 0;
    let mut kernels = Vec::new();
    for e in entries {
        if !names.insert(e.name.as_str()) {
            return Err(format!("duplicate entry {}", e.name));
        }
        let role = match (e.role.as_str(), e.name.as_str()) {
            ("adapters", ADAPTERS) => {
                adapters += 1;
                continue;
            }
            ("whole", _) => Role::Whole,
            ("split", _) => Role::Split,
            ("combine", _) => Role::Combine,
            (r, name) => return Err(format!("entry {name} role {r}")),
        };
        kernels.push((e, role));
    }
    if adapters != 1 {
        return Err(format!("no {ADAPTERS} entry"));
    }
    Ok(kernels)
}

/// The exemplar operands `capture_main.c` passes (fake addresses; nothing is dereferenced).
fn exemplar_call(args: &[i64]) -> Option<Call> {
    let &[total_q, requests, width, pages, head_stride, splits] = args else {
        return None;
    };
    Some(Call {
        q: 0x7f00_0000_0000,
        k: 0x7e00_0000_0000,
        v: 0x7d00_0000_0000,
        o: 0x7c00_0000_0000,
        lse: 0x7b00_0000_0000,
        out: 0x7f00_0000_0000,
        cu_q: 0x7a00_0000_0000,
        used_k: 0x7900_0000_0000,
        page_table: 0x7800_0000_0000,
        total_q: total_q as u32,
        requests: requests as u32,
        width: width as u32,
        pages: pages as u32,
        head_stride: head_stride as u64,
        splits: splits as u32,
        scale: 1.0,
    })
}

fn check_exemplars(
    be: &CudaBackend,
    g: Geometry,
    p: &Prepared,
    e: &Entry,
) -> std::result::Result<(), String> {
    for x in &e.exemplars {
        let c = exemplar_call(&x.args).ok_or("exemplar arity")?;
        let mut l = match p.role {
            Role::Combine => combine(g, p, &c),
            _ => forward(be, g, p, &c).map_err(|e| e.to_string())?,
        };
        l.matches(x)
            .map_err(|why| format!("exemplar {:?}: {why}", x.args))?;
    }
    Ok(())
}

fn rows(g: Geometry, kv_heads: u32, total_q: u32) -> u32 {
    if g.pack(kv_heads) {
        total_q * (g.heads / kv_heads)
    } else {
        total_q
    }
}

fn forward(be: &CudaBackend, g: Geometry, p: &Prepared, c: &Call) -> Result<Launch> {
    let kvh = p.kv_heads;
    let split = p.role == Role::Split;
    // A whole-KV entry has no split extent: its scheduler runs one split.
    let (d, h, t, s) = (
        g.head_dim,
        g.heads,
        c.total_q,
        if split { c.splits } else { 1 },
    );
    let pack = g.pack(kvh);
    let qhpk = h / kvh;
    // Query heads per tile row group: KV heads when packed, else every query head.
    let x = if pack { kvh } else { h };
    let packed = rows(g, kvh, t);
    let mut l = Launch::new(&p.kernel.layout);
    l.grid = [
        (packed + c.requests * (g.tile_m() - 1)) / g.tile_m() * x,
        s,
        1,
    ];
    l.block = p.block;
    l.smem = p.smem;
    let elem = 2u64;
    let row_bytes = u64::from(h * d) * elem;
    let (os, lse_strides): (u64, Vec<u64>) = (
        u64::from(t) * u64::from(h * d),
        if pack {
            vec![u64::from(t), u64::from(t * qhpk), u64::from(t * h)]
        } else {
            vec![u64::from(t), u64::from(t * h)]
        },
    );
    let lse_shapes = [t, x, s];
    // Parameters up to the shared tail.
    let tail;
    if g.tma() {
        l.coords(0, &[t, d, x]);
        l.coords(1, &[PAGE, d, kvh, c.pages]);
        l.coords(2, &[PAGE, d, kvh, c.pages]);
        let base = if split {
            l.memref(3, c.o, &[t, d, x, s], &[os]);
            l.memref(4, c.lse, &lse_shapes, &lse_strides);
            5
        } else {
            l.coords(3, &[d, x]);
            4
        };
        l.memref(base, c.cu_q, &[c.requests + 1], &[]);
        l.memref(base + 1, c.used_k, &[c.requests], &[]);
        l.memref32(base + 2, c.page_table, &[c.requests, c.width], &[c.width]);
        let dtype = if g.f16 { TMAP_F16 } else { TMAP_BF16 };
        let tmap = |base: u64, dims: &[u64], strides: &[u64], boxes: &[u32]| {
            let mut m = be.encode_tmap_tiled(
                dtype,
                base,
                dims,
                strides,
                boxes,
                &vec![1; dims.len()],
                0,
                SWIZZLE_128B,
                L2_PROMOTION_128B,
                0,
            )
            .map_err(|e| {
                RuntimeError::Rejected(format!(
                    "FA4 tensor map base {base:#x} dims {dims:?} strides {strides:?} boxes {boxes:?}: {e}"
                ))
            })?;
            upstream_tmap_flags(&mut m, elem, dims, strides);
            Ok::<_, RuntimeError>(m)
        };
        let (d64, h64, t64) = (u64::from(d), u64::from(h), u64::from(t));
        let q = if pack {
            tmap(
                c.q,
                &[d64, h64, t64],
                &[d64 * elem, row_bytes],
                &[64, qhpk, 64 / qhpk],
            )?
        } else {
            tmap(
                c.q,
                &[d64, t64, h64],
                &[row_bytes, d64 * elem],
                &[64, 64, 1],
            )?
        };
        let kv = |ptr: u64| {
            tmap(
                ptr,
                &[d64, u64::from(PAGE), u64::from(kvh), u64::from(c.pages)],
                &[
                    d64 * elem,
                    c.head_stride * elem,
                    u64::from(PAGE) * d64 * elem,
                ],
                &[64, PAGE, 1, 1],
            )
        };
        let at = base + 3;
        l.tmap(at, &q);
        l.tmap(at + 1, &kv(c.k)?);
        l.tmap(at + 2, &kv(c.v)?);
        tail = if split {
            at + 3
        } else {
            // The varlen O view starts 2^30 rows before `o` (upstream's domain offset). Below
            // 2^44 that base wraps, which the driver refuses but the hardware's 57-bit address
            // arithmetic accepts, as upstream encodes it: encode at `o`, then store the wrapped
            // base.
            let o_base = c.o.wrapping_sub(O_ROWS_SHIFT * row_bytes) & TMAP_ADDRESS_MASK;
            let mut o = if pack {
                tmap(
                    c.o,
                    &[d64, h64, O_ROWS_SHIFT, (1 << 31) - 1],
                    &[d64 * elem, row_bytes, row_bytes],
                    &[64, qhpk, 64 / qhpk, 1],
                )?
            } else {
                tmap(
                    c.o,
                    &[d64, O_ROWS_SHIFT, h64, (1 << 31) - 1],
                    &[row_bytes, d64 * elem, row_bytes],
                    &[64, 64, 1, 1],
                )?
            };
            o[..8].copy_from_slice(&o_base.to_le_bytes());
            l.tmap(at + 3, &o);
            at + 4
        };
    } else {
        l.memref(0, c.q, &[t, d, x], &[]);
        l.memref(1, c.k, &[PAGE, d, kvh, c.pages], &[]);
        l.memref(2, c.v, &[PAGE, d, kvh, c.pages], &[]);
        if split {
            l.memref(3, c.o, &[t, d, x, s], &[os]);
            l.memref(4, c.lse, &lse_shapes, &lse_strides);
        } else {
            l.memref(3, c.o, &[t, d, x], &[]);
        }
        let base = if split { 5 } else { 4 };
        l.memref(base, c.cu_q, &[c.requests + 1], &[]);
        l.memref(base + 1, c.used_k, &[c.requests], &[]);
        l.memref32(base + 2, c.page_table, &[c.requests, c.width], &[c.width]);
        tail = base + 3;
    }
    // softmax scale in log2 units, the two MMA atoms' accumulate flags, then the tile
    // scheduler's arguments.
    l.put(tail, 0, &(c.scale * std::f32::consts::LOG2_E).to_le_bytes());
    l.put(tail + 1, 0, &[0]);
    l.put(tail + 2, 0, &[0]);
    l.u32(tail + 3, 0, x);
    l.u32(tail + 4, 0, c.requests);
    l.u32(tail + 5, 0, packed);
    l.u32(tail + 6, 0, s);
    l.u32(tail + 7, 0, L2_BUDGET / (2 * (2 * d) * g.tile_n()));
    l.memref(tail + 8, c.cu_q, &[c.requests + 1], &[]);
    l.u32(tail + 9, 0, s);
    Ok(l)
}

/// The upstream host code encodes its tensor maps itself, not through the driver, and sets two
/// flags the driver's encoder leaves clear: byte 8 bit 1 always, and byte 10 bit 5 when the last
/// element's byte offset is at least 128 KiB. Reproducing them keeps the descriptors byte-equal
/// to the upstream route's; the exemplar check covers both cases of the second.
fn upstream_tmap_flags(m: &mut [u8; 128], elem: u64, dims: &[u64], strides: &[u64]) {
    m[8] |= 0x02;
    let last = (dims[0] - 1) * elem
        + dims[1..]
            .iter()
            .zip(strides)
            .map(|(&n, &s)| (n - 1) * s)
            .sum::<u64>();
    if last >= 0x2_0000 {
        m[10] |= 0x20;
    }
}

/// cutlass `FastDivmod`: divisor, multiplier, then the two shifts as bytes.
fn divmod(l: &mut Launch, index: usize, d: u32) {
    let log2 = 32 - (d - 1).leading_zeros().min(32);
    let multiplier = ((1u64 << 32) * ((1u64 << log2) - u64::from(d)) / u64::from(d) + 1) as u32;
    l.u32(index, 0, d);
    l.u32(index, 4, multiplier);
    l.put(index, 8, &[log2.min(1) as u8, log2.saturating_sub(1) as u8]);
    l.pad(index, 10, 2);
}

fn combine(g: Geometry, p: &Prepared, c: &Call) -> Launch {
    let (d, h, t, s) = (g.head_dim, g.heads, c.total_q, c.splits);
    let mut l = Launch::new(&p.kernel.layout);
    l.grid = [
        (t * h).div_ceil(COMBINE_TILE_M),
        d / COMBINE_K_BLOCK,
        c.requests,
    ];
    l.block = p.block;
    l.smem = p.smem;
    l.memref(0, c.o, &[t, d, s, h], &[u64::from(t) * u64::from(h * d)]);
    l.memref(1, c.lse, &[t, s, h], &[u64::from(t * h), u64::from(t)]);
    l.memref(2, c.out, &[t, d, h], &[]);
    l.memref(3, c.cu_q, &[c.requests + 1], &[]);
    divmod(&mut l, 4, t);
    divmod(&mut l, 5, h);
    l
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(name: &str, role: &str) -> Entry {
        Entry {
            name: name.into(),
            role: role.into(),
            kv_heads: 1,
            symbol: String::new(),
            params: Vec::new(),
            sha256: String::new(),
            offset: 0,
            bytes: 0,
            exemplars: Vec::new(),
        }
    }

    #[test]
    fn every_entry_is_a_kernel_or_the_one_adapter_object() {
        let ok = [
            entry("w", "whole"),
            entry("s", "split"),
            entry("c", "combine"),
            entry(ADAPTERS, "adapters"),
        ];
        let roles: Vec<Role> = kernel_entries(&ok).unwrap().iter().map(|e| e.1).collect();
        assert_eq!(roles, [Role::Whole, Role::Split, Role::Combine]);
        // An adapters-role entry under another name would bypass the adapter checks.
        assert!(kernel_entries(&[
            entry("w", "whole"),
            entry("extra", "adapters"),
            entry(ADAPTERS, "adapters")
        ])
        .is_err());
        assert!(kernel_entries(&[
            entry("w", "whole"),
            entry(ADAPTERS, "adapters"),
            entry(ADAPTERS, "adapters")
        ])
        .is_err());
        assert!(kernel_entries(&[entry("w", "whole")]).is_err());
        assert!(kernel_entries(&[entry(ADAPTERS, "whole")]).is_err());
        assert!(kernel_entries(&[
            entry("w", "whole"),
            entry("w", "split"),
            entry(ADAPTERS, "adapters")
        ])
        .is_err());
        assert!(kernel_entries(&[entry("x", "decode"), entry(ADAPTERS, "adapters")]).is_err());
    }
}
