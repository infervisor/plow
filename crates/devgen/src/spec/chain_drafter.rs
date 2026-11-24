//! The checkpoint-MTP chain drafter (`qwen3_5_mtp`, `nemotron_h_mtp`): one program per
//! `(rows, k)` running `k` chained draft steps for `rows` sequences on the decode interpreter.
//! Step `j` embeds `tokens[j]`, joins it with the carried hidden (the target's final-norm row at
//! step 0, gathered by `hidden_row`; then the previous step's output norm), runs the family's MTP
//! layer against its own cache (`kv.mtp.{k,v}`, stored at `position` of `slot`), and writes the
//! greedy draft to `tokens[j + 1]`.
//!
//! This module owns what the families share: the step inputs (`in.spec.*`), the drafter cache,
//! the step's embed / gather / attention / greedy-head instructions, and the `spec.json` +
//! aux-program sections. The layer in the middle is the family's.
use packet::dev::DevOp;
use packet::devbuild::{Builder, Program, SectionData, TensorDecl, SECT_METADATA};
use packet::typed_abi::{FlashDecodeArgs, FlashMergeArgs, TypedOp};
use plow_asset::spec_draft;

use crate::Model;

/// The step inputs a draft program reads (`in.spec.*`).
pub(crate) struct Inputs {
    pub tokens: Vec<u32>,
    pub hidden_row: u32,
    pub slot: u32,
    /// Per step: (position, kv length).
    pub steps: Vec<(u32, u32)>,
}

/// The handles a draft program shares with the target programs.
pub(crate) struct Shared {
    pub hidden: u32,
    pub hidden_rows: u32,
    pub kv: (u32, u32),
}

impl Shared {
    pub(crate) fn find(m: &Model, family: &str) -> Self {
        Shared {
            hidden: handle(m, family, "act.hn"),
            // A decode or verify launch's rows: the widest rung's.
            hidden_rows: *m.prog_t.last().expect("decode rungs"),
            kv: (handle(m, family, "kv.mtp.k"), handle(m, family, "kv.mtp.v")),
        }
    }
}

/// The handle of the target tensor `name`.
pub(crate) fn handle(m: &Model, family: &str, name: &str) -> u32 {
    m.tensors
        .iter()
        .position(|t| t.name == name)
        .unwrap_or_else(|| panic!("{family}: target lacks {name}")) as u32
}

/// The drafter's cache: `[slots][kv heads][ctx][hd]` of `elem` bytes each (`kv_dim` = kv heads x
/// hd; 2 = bf16, 1 = E4M3).
pub(crate) fn declare_kv(b: &mut Builder, kv_dim: u64, slots: u32, ctx: u32, elem: u64) -> (u32, u32) {
    let bytes = u64::from(slots) * kv_dim * u64::from(ctx) * elem;
    (b.tensor("kv.mtp.k", bytes), b.tensor("kv.mtp.v", bytes))
}

/// A draft program's builder over the packet's tensor table, with its step inputs declared.
pub(crate) fn builder(
    tensors: Vec<TensorDecl>,
    n_cu: u32,
    rows: u32,
    kmax: u32,
) -> (Builder, Inputs) {
    let mut b = Builder::new(n_cu);
    b.adopt_tensors(tensors);
    b.set_tensor_dedup(true);
    b.force_uniseg();
    let i32s = |b: &mut Builder, name: String| b.tensor(&name, u64::from(rows) * 4);
    let inputs = Inputs {
        tokens: (0..=kmax)
            .map(|j| i32s(&mut b, format!("in.spec.tokens.{j}")))
            .collect(),
        hidden_row: i32s(&mut b, "in.spec.hidden_row".into()),
        slot: i32s(&mut b, "in.spec.slot".into()),
        steps: (0..kmax)
            .map(|j| {
                let suffix = if j == 0 {
                    String::new()
                } else {
                    format!(".{j}")
                };
                (
                    i32s(&mut b, format!("in.spec.position{suffix}")),
                    i32s(&mut b, format!("in.spec.kv_len{suffix}")),
                )
            })
            .collect(),
    };
    (b, inputs)
}

/// Rows of one draft step: the blocks of its row-parallel ops and its row count.
#[derive(Clone)]
pub(crate) struct Step {
    pub rows: u32,
    pub hidden: u32,
    pub rblocks: Vec<u32>,
}

impl Step {
    pub(crate) fn new(rows: u32, hidden: u32, n_cu: u32) -> Self {
        Step {
            rows,
            hidden,
            rblocks: (0..rows.min(n_cu)).collect(),
        }
    }

    /// `out = table[ids]` after the previous step's draft landed (`prev`).
    pub(crate) fn embed(
        &self,
        b: &mut Builder,
        prev: Option<u32>,
        out: u32,
        table: u32,
        ids: u32,
    ) -> u32 {
        let (rows, h) = (self.rows, self.hidden);
        b.emit(DevOp::Embed, self.rblocks.clone(), prev.as_slice(), |d| {
            d.t[..3].copy_from_slice(&[out, table, ids]);
            d.i[..2].copy_from_slice(&[rows, h]);
            d.f[0] = 1.0;
        })
    }

    /// Step 0's carried hidden: row `hidden_row[r]` of the target's final-norm rows.
    pub(crate) fn gather_hidden(
        &self,
        b: &mut Builder,
        out: u32,
        s: &Shared,
        hidden_row: u32,
    ) -> u32 {
        let (rows, h) = (self.rows, self.hidden);
        b.emit(DevOp::RowGather, self.rblocks.clone(), &[], |d| {
            d.t[..3].copy_from_slice(&[out, s.hidden, hidden_row]);
            d.i[..3].copy_from_slice(&[rows, h, s.hidden_rows]);
        })
    }

    /// Decode attention over the drafter cache at `slot`: split flash + merge into `at`.
    /// `kv_scale`: an E4M3 cache's per-tensor `{k_scale, v_scale}` (FlashDecode t7), else none.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn attention(
        &self,
        b: &mut Builder,
        dep: u32,
        geom: (u32, u32, u32),
        part: (u32, u32, u32),
        q: u32,
        s: &Shared,
        kv_len: u32,
        slot: u32,
        ctx: u32,
        at: u32,
        kv_scale: u32,
    ) -> u32 {
        let (heads, kvh, hd) = geom;
        let (opart, mlpart, ns) = part;
        let n_cu = b.n_cu();
        let all = b.all();
        let c_fa = FlashDecodeArgs {
            Opart: opart,
            mlpart,
            Q: q,
            K: s.kv.0,
            V: s.kv.1,
            kv_len,
            decode_slot: slot,
            kv_scale,
            n_batch: self.rows,
            n_head: heads,
            n_kv_head: kvh,
            kv_stride: ctx,
            window: 0,
            nsplit: ns,
            hd,
            kv_mask: u32::MAX,
            scale: 1.0 / (hd as f32).sqrt(),
            ..Default::default()
        }
        .emit(b, all, &[dep]);
        FlashMergeArgs {
            O: at,
            Opart: opart,
            mlpart,
            n_batch: self.rows,
            n_head: heads,
            nsplit: ns,
            hd,
            ..Default::default()
        }
        .emit(b, (0..heads.min(n_cu)).collect(), &[c_fa])
    }

    /// The greedy draft of `logits` into `out_ids`; the program's next step waits on it.
    pub(crate) fn greedy(
        &self,
        b: &mut Builder,
        dep: u32,
        amax: u32,
        logits: u32,
        vocab: u32,
        out_ids: u32,
    ) -> u32 {
        let rows = self.rows;
        let amax_blocks = crate::AMAX_BLOCKS.min(b.n_cu());
        let c_am = b.emit(DevOp::Argmax, (0..amax_blocks).collect(), &[dep], |d| {
            d.t[..2].copy_from_slice(&[amax, logits]);
            d.i[0] = vocab;
            d.i[1] = rows;
        });
        b.emit(DevOp::ArgmaxFin, vec![0], &[c_am], |d| {
            d.t[..2].copy_from_slice(&[out_ids, amax]);
            d.i[0] = amax_blocks;
            d.i[1] = rows;
        })
    }
}

/// Every `(rows, k)` draft program (rows from `PLOW_SPEC_ROWS` up to the decode slots, k up to
/// the verify depth) appended to `m`'s tensor table, plus `spec.json`. `program` builds one over
/// the table it is handed and returns the extended table.
pub(crate) fn sections(
    m: &mut Model,
    family: &str,
    shared: &Shared,
    shard: String,
    mut program: impl FnMut(Vec<TensorDecl>, u32, u32, u32) -> (Program, Vec<TensorDecl>, Inputs),
) -> Vec<SectionData> {
    let kmax = super::verify_depth();
    let slots = *m.prog_t.last().expect("decode rungs");
    let mut programs = Vec::new();
    let mut table = Vec::new();
    let mut inputs = None;
    for rows in crate::mtp::rows_list().into_iter().filter(|&r| r <= slots) {
        for k in 1..=kmax {
            let (p, tensors, inp) = program(std::mem::take(&mut m.tensors), rows, k, kmax);
            m.tensors = tensors;
            table.push(spec_draft::DraftProgram {
                rows,
                k,
                index: programs.len() as u32,
            });
            programs.push(p);
            inputs = Some(inp);
        }
    }
    let inp =
        inputs.unwrap_or_else(|| panic!("{family}: no PLOW_SPEC_ROWS entry fits {slots} slots"));
    // Input tensors are sized at the first (narrowest) program's rows; the widest must fit.
    let widest = table.iter().map(|p| p.rows).max().unwrap_or(1);
    for &t in inp.tokens.iter().chain([&inp.hidden_row, &inp.slot]) {
        assert!(
            m.tensors[t as usize].bytes >= u64::from(widest) * 4,
            "{family}: {} holds fewer than {widest} rows",
            m.tensors[t as usize].name
        );
    }
    let packed: Vec<Vec<_>> = programs
        .iter()
        .map(|p| p.insts.iter().map(|d| d.pack()).collect())
        .collect();
    let views: Vec<plow_asset::program::Program<'_>> = programs
        .iter()
        .zip(&packed)
        .zip(&table)
        .map(|((p, insts), e)| plow_asset::program::Program {
            rows: e.rows,
            role: packet::devbuild::ProgramRole::DecodeRung { rows: e.rows },
            n_counter: p.n_counter,
            insts,
            stream: &p.stream,
            stream_ofs: &p.stream_ofs,
            stream_len: &p.stream_len,
            waits: &p.waits,
            succs: &p.succs,
            gq_stream: &p.gq_stream,
            gq_seg_ofs: &p.gq_seg_ofs,
            l2_domains: p.l2_domains,
        })
        .collect();
    let bytes = plow_asset::aux_program::encode(m.n_cu, m.tensors.len(), &views)
        .unwrap_or_else(|e| panic!("{family}: draft programs: {e}"));
    let h16 = |h: u32| u16::try_from(h).expect("tensor handle");
    let manifest = spec_draft::Manifest {
        version: spec_draft::VERSION,
        family: family.into(),
        kmax,
        programs: table,
        tensors: spec_draft::DraftTensors {
            tokens: inp.tokens.iter().map(|&t| h16(t)).collect(),
            hidden_row: Some(h16(inp.hidden_row)),
            position: h16(inp.steps[0].0),
            kv_len: h16(inp.steps[0].1),
            slot: h16(inp.slot),
            hidden: Some(h16(shared.hidden)),
            hidden_rows: shared.hidden_rows,
            step_position: inp.steps[1..].iter().map(|s| h16(s.0)).collect(),
            step_kv_len: inp.steps[1..].iter().map(|s| h16(s.1)).collect(),
            own_kv: vec![h16(shared.kv.0), h16(shared.kv.1)],
        },
        block: None,
        // The drafter is part of the target checkpoint, which the recipe pins.
        shard,
        shard_sha256: String::new(),
    };
    manifest
        .validate(programs.len(), m.tensors.len())
        .unwrap_or_else(|e| panic!("{family}: {e}"));
    eprintln!(
        "  spec draft ({family}): {} programs (rows {:?}, k 1..={kmax})",
        programs.len(),
        manifest
            .programs
            .iter()
            .map(|p| p.rows)
            .collect::<std::collections::BTreeSet<_>>()
    );
    vec![
        SectionData {
            kind: SECT_METADATA,
            name: spec_draft::SECTION.into(),
            data: serde_json::to_vec(&manifest).expect("spec.json"),
        },
        SectionData {
            kind: SECT_METADATA,
            name: spec_draft::PROGRAMS_SECTION.into(),
            data: bytes,
        },
    ]
}
