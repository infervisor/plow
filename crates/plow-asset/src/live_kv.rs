use crate::program::{Packet, Program};
use crate::splitk::ProjectionAccess;
use packet::dev::{DevOp, ROPE_PAIR_HALF, TENSOR_NONE16};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

pub const SECTION: &str = "live_kv";
type Result<T> = std::result::Result<T, String>;
fn require(ok: bool, reason: &str) -> Result<()> {
    if ok {
        Ok(())
    } else {
        Err(format!("live KV manifest: {reason}"))
    }
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Cache {
    pub pair: [u16; 2],
    pub heads: u32,
    pub hd: u32,
    pub stride: u32,
    pub window: u32,
    pub mask: u32,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Map {
    pub handle: u32,
    pub pair: [u16; 2],
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    pub version: u32,
    pub n_cu: u32,
    pub batch: u32,
    pub max_ctx: u32,
    pub position: u16,
    pub kv_length: u16,
    pub caches: Vec<Cache>,
    pub maps: Vec<Map>,
    pub programs: Vec<String>,
    pub splitk: Vec<ProjectionAccess>,
}

pub fn program_digest(p: &Program<'_>) -> String {
    let mut h = Sha256::new();
    h.update(b"plow-live-kv-program-v1");
    for v in [
        p.rows,
        p.packed_prefill_only as u32,
        p.n_counter,
        p.l2_domains,
    ] {
        h.update(v.to_le_bytes());
    }
    h.update((p.insts.len() as u64).to_le_bytes());
    for d in p.insts {
        h.update(d.op.to_le_bytes());
        h.update(d.blocks.to_le_bytes());
        for x in d.fj {
            h.update(x.to_le_bytes());
        }
        for x in d.t {
            h.update(x.to_le_bytes());
        }
        for x in d.i {
            h.update(x.to_le_bytes());
        }
    }
    for table in [p.stream, p.gq_stream] {
        h.update((table.len() as u64).to_le_bytes());
        for e in table {
            for x in [e.inst, e.slice, e.wait_ofs, e.succ_ofs] {
                h.update(x.to_le_bytes());
            }
            for x in [e.wait_len, e.succ_len, e.flags, e.seg] {
                h.update(x.to_le_bytes());
            }
        }
    }
    for table in [p.stream_ofs, p.stream_len, p.succs, p.gq_seg_ofs] {
        h.update((table.len() as u64).to_le_bytes());
        for x in table {
            h.update(x.to_le_bytes());
        }
    }
    h.update((p.waits.len() as u64).to_le_bytes());
    for w in p.waits {
        h.update(w.id.to_le_bytes());
        h.update(w.threshold.to_le_bytes());
    }
    format!("{:x}", h.finalize())
}

// Compiler-only extraction. Runtime validation below uses the declared geometry.
pub fn emit(packet: &Packet<'_>) -> Result<Manifest> {
    let widest = packet.programs.last().ok_or("missing decode program")?;
    let handle = |name| {
        packet
            .tensors
            .iter()
            .position(|t| t.name == name)
            .and_then(|i| u16::try_from(i).ok())
            .filter(|&i| i != TENSOR_NONE16)
            .ok_or("missing runtime tensor")
    };
    let position = handle("in.pos")?;
    let max_ctx = u32::try_from(packet.tensors[position as usize].bytes / 4)
        .map_err(|_| "context overflow")?;
    let caches: Vec<_> = widest
        .insts
        .iter()
        .filter(|d| d.op == DevOp::FlashDecode as u16)
        .map(|d| Cache {
            pair: [d.t[3], d.t[4]],
            heads: d.i[2],
            hd: d.i[6],
            stride: d.i[3],
            window: d.i[4],
            mask: d.i[7],
        })
        .collect();
    let mut maps = Vec::new();
    for g in packet
        .generated
        .iter()
        .filter(|g| g.kind == packet::rope::GEN_TMAP_KV_PAIR)
    {
        maps.push(Map {
            handle: g.tensor,
            pair: [
                u16::try_from(g.aux).map_err(|_| "map handle overflow")?,
                u16::try_from(g.scale).map_err(|_| "map handle overflow")?,
            ],
        });
    }
    let splitk = crate::splitk::validate(packet)?.map_or_else(Vec::new, |p| p.access);
    let m = Manifest {
        version: 1,
        n_cu: packet.n_cu,
        batch: widest.rows,
        max_ctx,
        position,
        kv_length: handle("in.kvlen")?,
        caches,
        maps,
        programs: packet.programs.iter().map(program_digest).collect(),
        splitk,
    };
    m.validate(packet)?;
    Ok(m)
}

impl Manifest {
    pub fn validate(&self, packet: &Packet<'_>) -> Result<()> {
        require(
            self.version == 1 && self.n_cu == packet.n_cu && self.n_cu > 0 && !packet.tp,
            "version, grid or TP mismatch",
        )?;
        require(
            packet.prefill_count < packet.programs.len()
                && self.batch > 0
                && self.max_ctx >= self.batch
                && packet.programs.last().is_some_and(|p| p.rows == self.batch),
            "invalid batch/context",
        )?;
        require(
            self.programs.len() == packet.programs.len()
                && self
                    .programs
                    .iter()
                    .zip(packet.programs)
                    .all(|(digest, p)| *digest == program_digest(p)),
            "program identity mismatch",
        )?;
        let proof = crate::splitk::validate(packet)?;
        require(
            proof.as_ref().map_or(&[][..], |p| p.access.as_slice()) == self.splitk,
            "splitK access proof mismatch",
        )?;
        let tensor = |h: u16| {
            packet
                .tensors
                .get(h as usize)
                .filter(|_| h != TENSOR_NONE16)
                .ok_or("invalid tensor handle")
        };
        require(
            self.position != self.kv_length
                && tensor(self.position)?.name == "in.pos"
                && tensor(self.position)?.bytes == self.max_ctx as u64 * 4
                && tensor(self.kv_length)?.name == "in.kvlen"
                && tensor(self.kv_length)?.bytes == self.batch as u64 * 4,
            "runtime position/length extent",
        )?;
        let mut caches = BTreeMap::new();
        let mut handles = BTreeSet::new();
        for c in &self.caches {
            require(
                c.pair[0] != c.pair[1]
                    && !c.pair.contains(&self.position)
                    && !c.pair.contains(&self.kv_length)
                    && c.heads > 0
                    && matches!(c.hd, 64 | 256 | 512)
                    && c.stride > 0
                    && if c.window == 0 {
                        c.stride == self.max_ctx && c.mask == u32::MAX
                    } else {
                        c.stride.is_power_of_two() && c.window <= c.stride && c.mask == c.stride - 1
                    },
                "cache geometry",
            )?;
            let bytes = (self.batch as u64)
                .checked_mul(c.heads as u64)
                .and_then(|v| v.checked_mul(c.stride as u64))
                .and_then(|v| v.checked_mul(c.hd as u64 * 2))
                .ok_or("cache extent overflow")?;
            for &h in &c.pair {
                require(
                    handles.insert(h) && tensor(h)?.bytes == bytes && !tensor(h)?.initialized,
                    "cache alias/extent/initialization",
                )?;
            }
            require(caches.insert(c.pair, c).is_none(), "duplicate cache")?;
        }
        require(!caches.is_empty(), "no declared cache")?;
        let mut patched = BTreeSet::new();
        let widest = packet.programs.last().ok_or("missing widest program")?;
        for &pc in packet.kv_row_insts {
            let d = widest
                .insts
                .get(pc as usize)
                .ok_or("KV row patch out of range")?;
            require(
                patched.insert(pc)
                    && d.op == DevOp::HeadNormRope as u16
                    && handles.contains(&d.t[0]),
                "KV row patch must target a declared cache writer",
            )?;
        }
        let mut maps = BTreeMap::new();
        for m in &self.maps {
            require(
                caches.contains_key(&m.pair)
                    && m.handle < TENSOR_NONE16 as u32
                    && !handles.contains(&(m.handle as u16))
                    && maps.insert(m.handle, m.pair).is_none(),
                "map alias or pair",
            )?;
        }
        let mut generated_maps = BTreeMap::new();
        for g in packet.generated {
            require(
                g.kind <= packet::rope::GEN_TMAP_KV_PAIR,
                "unknown generator access contract",
            )?;
            require(
                !handles.iter().any(|&h| u32::from(h) == g.tensor),
                "generated cache contents",
            )?;
            if g.kind == packet::rope::GEN_TMAP_KV_PAIR {
                let pair = *maps.get(&g.tensor).ok_or("undeclared KV map")?;
                let c = caches[&pair];
                require(
                    g.aux == u32::from(pair[0])
                        && g.scale == u32::from(pair[1])
                        && g.ctx == c.stride
                        && g.hd == c.hd
                        && g.frac == c.heads as f64
                        && c.stride % 32 == 0
                        && tensor(g.tensor as u16)?.bytes == 256
                        && generated_maps.insert(g.tensor, pair).is_none(),
                    "KV map geometry",
                )?;
            } else if maps.contains_key(&g.tensor) {
                return Err("non-KV generator aliases a KV map".into());
            } else if matches!(
                g.kind,
                packet::rope::GEN_TMAP_BF16 | packet::rope::GEN_TMAP_E4M3
            ) {
                require(
                    !handles.iter().any(|&h| u32::from(h) == g.aux),
                    "unaudited indirect cache access",
                )?;
            }
        }
        require(generated_maps == maps, "missing generated KV map")?;
        for (pi, p) in packet.programs.iter().enumerate() {
            let prefill = pi < packet.prefill_count;
            require(
                !p.packed_prefill_only && p.rows > 0 && p.rows <= self.max_ctx,
                "program row contract",
            )?;
            let mut reads = BTreeSet::new();
            let mut writes = BTreeSet::new();
            for (pc, d) in p.insts.iter().enumerate() {
                let op = DevOp::from_u16(d.op).ok_or("unknown opcode access contract")?;
                direct_operands(op)?;
                if matches!(op, DevOp::FlashDecode | DevOp::FlashPrefill) {
                    let pair = [d.t[3], d.t[4]];
                    let c = caches.get(&pair).ok_or("undeclared attention cache pair")?;
                    let qh = d.i[if prefill { 2 } else { 1 }];
                    require(qh > 0 && qh % c.heads == 0, "invalid query grouping")?;
                    let valid = if prefill {
                        op == DevOp::FlashPrefill
                            && d.t[6] == TENSOR_NONE16
                            && (d.t[7] == TENSOR_NONE16
                                || maps.get(&u32::from(d.t[7])) == Some(&pair))
                            && d.i[0] == p.rows
                            && d.i[1] == p.rows
                            && d.i[3] == c.heads
                            && d.i[4] == 0
                            && d.i[5] == c.window
                            && d.i[6] == c.hd
                            && d.fj[1] == c.stride
                            && d.fj[2] == c.mask
                    } else {
                        op == DevOp::FlashDecode
                            && d.t[5] == self.kv_length
                            && d.t[7] == TENSOR_NONE16
                            && crate::mixed_step::validate_decode_slot_tensor(
                                d.t[6],
                                p.rows,
                                tensor(d.t[6]).ok().map(|slot_map| {
                                    crate::mixed_step::TensorContract {
                                        name: slot_map.name,
                                        bytes: slot_map.bytes,
                                        initialized: slot_map.initialized,
                                    }
                                }),
                            )
                            .is_ok()
                            && d.i[0] == p.rows
                            && (d.i[2], d.i[6], d.i[3], d.i[4], d.i[7])
                                == (c.heads, c.hd, c.stride, c.window, c.mask)
                    };
                    require(valid && reads.insert(pair), "attention reader contract")?;
                }
                for (slot, &h) in d.t.iter().enumerate() {
                    require(
                        h == TENSOR_NONE16 || (h as usize) < packet.tensors.len(),
                        "operand handle out of range",
                    )?;
                    require(
                        !maps.contains_key(&u32::from(h))
                            || (op == DevOp::FlashPrefill && slot == 7),
                        "map operand access",
                    )?;
                    if !handles.contains(&h) {
                        continue;
                    }
                    match op {
                        DevOp::FlashDecode | DevOp::FlashPrefill if slot == 3 || slot == 4 => {}
                        DevOp::HeadNormRope if slot == 0 => {
                            let c = self.caches.iter().find(|c| c.pair.contains(&h)).unwrap();
                            let legacy = pi + 1 == packet.programs.len()
                                && p.rows == 1
                                && packet.kv_row_insts.contains(&(pc as u32));
                            require(
                                d.i[0] == p.rows
                                    && d.i[1] == c.heads
                                    && d.i[2] == c.hd
                                    && matches!(
                                        (c.hd, d.i[5]),
                                        (64, ROPE_PAIR_HALF) | (256 | 512, 0)
                                    )
                                    && d.i[3] == 0
                                    && d.fj[1] == c.stride
                                    && d.fj[2] == c.mask
                                    && d.t[5] == self.position
                                    && d.t[7] == TENSOR_NONE16
                                    && (if prefill {
                                        d.i[6] == 0
                                    } else {
                                        d.i[6] == p.rows || (legacy && d.i[6] == 0)
                                    })
                                    && writes.insert(h),
                                "cache writer contract",
                            )?;
                        }
                        _ => return Err("unsupported cache operand access".into()),
                    }
                }
            }
            require(
                reads.len() == caches.len() && writes.len() == handles.len(),
                "cache access coverage",
            )?;
        }
        Ok(())
    }
}

fn direct_operands(op: DevOp) -> Result<()> {
    require(
        matches!(
            op,
            DevOp::Nop
                | DevOp::RmsNorm
                | DevOp::RowRms
                | DevOp::HeadNormRope
                | DevOp::Residual
                | DevOp::Glu
                | DevOp::Gemm
                | DevOp::GemmNorm
                | DevOp::Gemv
                | DevOp::FlashPrefill
                | DevOp::FlashDecode
                | DevOp::FlashMerge
                | DevOp::NormResidual
                | DevOp::NormResidualNorm
                | DevOp::AddNorm
                | DevOp::GemmGlu
                | DevOp::GemvGlu
                | DevOp::GemvQkv
                | DevOp::Embed
                | DevOp::Argmax
                | DevOp::ArgmaxFin
                | DevOp::SoftCap
                | DevOp::ZeroF32
                | DevOp::GemmSplitK
                | DevOp::CastF32Bf16
        ),
        "opcode has no audited direct-operand access contract",
    )
}

#[cfg(test)]
#[path = "live_kv_tests.rs"]
mod tests;
