use packet::dev::{DevInst64, StreamEnt, Wait};
use packet::rope::GenTensor;

#[derive(Clone, Copy)]
pub struct Tensor<'a> {
    pub name: &'a str,
    pub bytes: u64,
    pub initialized: bool,
}

#[derive(Clone, Copy)]
pub struct Program<'a> {
    pub rows: u32,
    pub packed_prefill_only: bool,
    pub n_counter: u32,
    pub insts: &'a [DevInst64],
    pub stream: &'a [StreamEnt],
    pub stream_ofs: &'a [u32],
    pub stream_len: &'a [u32],
    pub waits: &'a [Wait],
    pub succs: &'a [u32],
    pub gq_stream: &'a [StreamEnt],
    pub gq_seg_ofs: &'a [u32],
    pub l2_domains: u32,
}

pub struct Packet<'a> {
    pub n_cu: u32,
    pub tp: bool,
    pub prefill_count: usize,
    pub tensors: &'a [Tensor<'a>],
    pub programs: &'a [Program<'a>],
    pub generated: &'a [GenTensor],
    pub kv_row_insts: &'a [u32],
}

pub fn with_model<T>(model: &packet::devbuild::Model, f: impl FnOnce(&Packet<'_>) -> T) -> T {
    let packed: Vec<Vec<_>> = model
        .progs
        .iter()
        .map(|p| p.insts.iter().map(|d| d.pack()).collect())
        .collect();
    let tensors: Vec<_> = model
        .tensors
        .iter()
        .map(|t| Tensor {
            name: &t.name,
            bytes: t.bytes,
            initialized: t.init.is_some(),
        })
        .collect();
    let programs: Vec<_> = model
        .progs
        .iter()
        .zip(&model.prog_t)
        .zip(&packed)
        .map(|((p, &rows), insts)| Program {
            rows,
            packed_prefill_only: false,
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
    f(&Packet {
        n_cu: model.n_cu,
        tp: model.progs.iter().any(|p| {
            p.hier_base != 0
                || p.insts.iter().any(|d| {
                    packet::dev::DevOp::from_u16(d.op)
                        .is_some_and(|op| matches!(op, packet::dev::DevOp::XReduce))
                })
        }),
        prefill_count: packet::devbuild::decode_rung_lo(&model.prog_t),
        tensors: &tensors,
        programs: &programs,
        generated: &model.gen,
        kv_row_insts: &model.kv_row_insts,
    })
}
