//! Generated-kernel catalog: role objects AOT-built by `scripts/gen_kernels/build_catalog.py`
//! from its checked-in tuning table (`tuning/nvidia/sm_90a/h100-sxm5/gen_kernels.json`), keyed
//! by op signature, never by model. `PLOW_EMIT_GEN_KERNELS=<entry,...>` binds every packet op
//! matching an entry's signature to that entry's role; without it the packet is unchanged.

use crate::attention_prefill_role::{self, Selection};
use packet::dev::{DevInst, DevOp};
use packet::devbuild::{Model, SectionData};
use plow_asset::segment_roles::{
    AttentionCapability, GeneratedAbi, GENERATED_FIRST, GENERATED_FLASH_PREFILL_ABI,
};
use std::collections::BTreeSet;
use std::path::Path;

pub(crate) struct Entry {
    pub name: &'static str,
    pub role: u8,
    pub file: &'static str,
    /// Causal flash prefill over one head width.
    pub head_dim: u32,
    /// Sliding window (`FlashPrefill` i[5]); 0 = global attention.
    pub window: u32,
    /// Accepts a KV ring (masked row index); otherwise the op must index KV linearly.
    pub ring_kv: bool,
    /// Narrowest prefill rung bound: the table's smallest shape class.
    pub min_rows: u32,
}

/// Mirrors the build_catalog.py entries; a new entry takes the next generated role ID.
pub(crate) const CATALOG: [Entry; 2] = [
    Entry {
        name: "attn_pf_hd512",
        role: GENERATED_FIRST,
        file: "gen_sm90a_attn_pf_hd512.cubin",
        head_dim: 512,
        window: 0,
        ring_kv: false,
        min_rows: 1024,
    },
    Entry {
        name: "attn_pf_hd256_sliding",
        role: GENERATED_FIRST + 1,
        file: "gen_sm90a_attn_pf_hd256_sliding.cubin",
        head_dim: 256,
        window: 1024,
        ring_kv: true,
        min_rows: 1024,
    },
];

/// The packet entry; plowrt also requires the `_direct` entry it launches with packed requests.
pub(crate) const ENTRY_SYMBOL: &str = "plow_gen_flash_prefill";

pub(crate) fn parse(list: &str) -> Result<Vec<&'static Entry>, String> {
    let mut out: Vec<&'static Entry> = Vec::new();
    for name in list.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        let entry = CATALOG.iter().find(|e| e.name == name).ok_or_else(|| {
            let known: Vec<_> = CATALOG.iter().map(|e| e.name).collect();
            format!(
                "unknown generated-kernel entry {name:?} (catalog: {})",
                known.join(", ")
            )
        })?;
        if !out.iter().any(|e| e.name == name) {
            out.push(entry);
        }
    }
    Ok(out)
}

impl Entry {
    /// The packet op signature: causal one-split fused flash prefill at this head width and
    /// window. Heads, KV heads and the ring size are runtime operands.
    pub(crate) fn matches(&self, op: &DevInst) -> bool {
        op.op == DevOp::FlashPrefill as u16
            && op.i[6] == self.head_dim
            && op.i[5] == self.window
            && op.i[7] == 1
            && (self.ring_kv || op.j[1] == u32::MAX)
    }

    fn object(&self, directory: &Path, profile: &str, gpu: &str) -> Result<Selection, String> {
        let path = directory.join(self.file);
        let image = std::fs::read(&path).map_err(|error| format!("{}: {error}", path.display()))?;
        let info = plow_asset::cubin::inspect(&image)
            .ok_or_else(|| format!("{} is not a valid cubin", path.display()))?;
        let global = |name| plow_asset::cubin::global_u32(&image, name);
        let (Some(block), Some(smem), Some(query_tile), Some(kv_tile), Some(warps)) = (
            global("plow_gen_block"),
            global("plow_gen_arena_bytes"),
            global("plow_attention_query_tile"),
            global("plow_attention_kv_tile"),
            global("plow_attention_warps"),
        ) else {
            return Err(format!("{} lacks generated-role geometry", path.display()));
        };
        if profile != "sm90a"
            || info.sm != 90
            || !info.entries.iter().any(|entry| entry == ENTRY_SYMBOL)
            || global("plow_gen_flash_prefill_abi") != Some(1)
            || global("plow_attention_head_dim") != Some(self.head_dim)
            || global("plow_pf_request_abi") != Some(2)
        {
            return Err(format!(
                "{} is not a generated {} flash-prefill object",
                path.display(),
                self.name
            ));
        }
        attention_prefill_role::validate_hardware_resources(gpu, profile, block, warps, smem)?;
        let abi = GeneratedAbi {
            family: GENERATED_FLASH_PREFILL_ABI.into(),
            entry: self.name.into(),
            block,
            smem,
        };
        let attention = AttentionCapability {
            profile: "sm90a".into(),
            dtype: "bf16".into(),
            head_dim: self.head_dim,
            query_tile,
            kv_tile,
            warps,
        };
        Ok(Selection::generated(
            self.file.into(),
            &image,
            attention_prefill_role::Generated {
                role: self.role,
                abi: abi.format(),
                attention,
                window: self.window,
                ring_kv: self.ring_kv,
            },
        ))
    }
}

/// Bind each listed entry's matching prefill ops (rungs of at least `min_rows`) to its role.
/// Runs before the hand-written attention roles, which then keep the remaining rungs.
#[allow(clippy::too_many_arguments)]
pub(crate) fn apply_output_objects(
    model: &mut Model,
    sections: &mut Vec<SectionData>,
    profile: &str,
    output: &Path,
    gpu: &str,
    packed: bool,
    list: &str,
    hd512_px4_role: bool,
) -> Result<(), String> {
    let profile = if profile == "sm_90a" {
        "sm90a"
    } else {
        profile
    };
    let entries = parse(list)?;
    if entries.is_empty() {
        return Ok(());
    }
    if !packed {
        return Err("generated flash-prefill roles require packed prefill metadata".into());
    }
    if hd512_px4_role && entries.iter().any(|e| e.head_dim == 512) {
        return Err("PLOW_EMIT_GEN_KERNELS hd512 and the px4 BQ64 role both claim HD512".into());
    }
    let directory = output.parent().unwrap_or_else(|| Path::new("."));
    let prefill_count = packet::devbuild::decode_rung_lo(&model.prog_t);
    for entry in entries {
        let selection = entry.object(directory, profile, gpu)?;
        let programs: BTreeSet<usize> = (0..prefill_count)
            .filter(|&index| {
                model.prog_t[index] >= entry.min_rows
                    && model.progs[index].insts.iter().any(|op| entry.matches(op))
            })
            .collect();
        if programs.is_empty() {
            return Err(format!(
                "packet has no prefill op matching generated entry {}",
                entry.name
            ));
        }
        attention_prefill_role::apply(model, sections, &selection, profile, Some(&programs))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_rejects_unknown_entries_and_dedups() {
        assert_eq!(parse("attn_pf_hd512, attn_pf_hd512,").unwrap().len(), 1);
        assert!(parse("").unwrap().is_empty());
        assert!(matches!(parse("attn_pf_hd999"), Err(e) if e.contains("attn_pf_hd512")));
    }

    #[test]
    fn catalog_roles_are_distinct_generated_ids() {
        let mut roles = BTreeSet::new();
        for entry in &CATALOG {
            assert!(plow_asset::segment_roles::is_generated(entry.role));
            assert!(roles.insert(entry.role));
        }
    }
}
