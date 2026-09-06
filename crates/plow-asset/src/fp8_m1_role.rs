use crate::program::{Packet, Program};
use packet::dev::{DevOp, TENSOR_NONE16};
use std::collections::BTreeSet;

pub const ABI: &str = "fp8_gemm_m1_tma_v1";
pub const ENTRY: &str = "_Z18interp_sm90a_fp8m111PlowProgram";
pub const ARENA: u32 = 205840;
type Result<T> = std::result::Result<T, String>;
fn require(ok: bool, why: &str) -> Result<()> {
    if ok {
        Ok(())
    } else {
        Err(format!("FP8 M1 role: {why}"))
    }
}

pub fn options(multistep: u32, lt: bool, stamped: bool) -> Result<()> {
    require(
        multistep == 0 && !lt && !stamped,
        "stamped, multistep and Lt combinations are unsupported",
    )
}

pub fn capability(
    profile: &str,
    values: [Option<u32>; 7],
    promote: u32,
    capacity: Option<u32>,
    grid: u32,
) -> Result<()> {
    require(
        profile == "sm90a"
            && promote <= 1
            && values
                == [
                    Some(1),
                    Some(1),
                    Some(256),
                    Some(ARENA),
                    Some(17408),
                    Some(16),
                    Some(promote),
                ],
        "incompatible compiled capability",
    )?;
    require(
        grid > 0 && capacity.is_some_and(|n| n >= grid),
        "insufficient cooperative capacity",
    )
}

fn writes(op: u16) -> Result<&'static [usize]> {
    use DevOp::*;
    Ok(match DevOp::from_u16(op).ok_or("unknown opcode")? {
        Nop => &[],
        QuantFp8 => &[0, 2],
        RmsNorm => &[0, 3, 4],
        AddNorm | NormResidualNorm | FlashDecode | QwenQGateSplit => &[0, 1],
        GemvQkv => &[0, 3, 5],
        QwenGdnConv | QwenGdnConvPrefill => &[0, 3],
        QwenGdnStep => &[0, 6],
        QwenGdnQkvPrep => &[0, 1, 2],
        QwenGdnGatePrep => &[0, 1],
        QwenGdnPrefill => &[0, 7],
        GemmFp8 | GemmMedFp8 | GemmSmallFp8 | GemmGluFp8 | Gemm | GemmMed | GemmSmall | GemmGlu
        | Gemv | GemvGlu | GemvFp8 | QwenRmsNorm | QwenGatedNorm | HeadNormRope | FlashPrefill
        | FlashMerge | NormResidual | Embed | Residual | Glu | SoftCap | Argmax | ArgmaxFin => &[0],
        _ => return Err("unaudited direct operand writer".into()),
    })
}

fn dependency_ancestors(dependencies: &[BTreeSet<usize>], pc: usize) -> Vec<bool> {
    let mut ancestors = vec![false; dependencies.len()];
    let mut pending: Vec<_> = dependencies[pc].iter().copied().collect();
    while let Some(parent) = pending.pop() {
        if !ancestors[parent] {
            ancestors[parent] = true;
            pending.extend(dependencies[parent].iter().copied());
        }
    }
    ancestors
}

pub fn validate(packet: &Packet<'_>, program: usize, pc: usize) -> Result<()> {
    require(
        !packet.tp
            && packet.n_cu > 0
            && packet.n_cu <= u16::MAX as u32
            && packet.programs.len() == packet.prefill_count + 1
            && program == packet.prefill_count,
        "one M1 decode program required",
    )?;
    let g = packet.programs.get(program).ok_or("program index")?;
    require(
        g.rows == 1 && !g.packed_prefill_only && g.l2_domains == 0,
        "decode geometry",
    )?;
    let d = g.insts.get(pc).ok_or("instruction index")?;
    // The first numerical record covers only this shape; unknown cells retain baseline.
    require(
        d.op == DevOp::GemmFp8 as u16
            && d.i == [1, 10240, 5120, 0, 0, 0, 0, d.i[7]]
            && d.fj == [0; 3]
            && u32::from(d.blocks) == packet.n_cu
            && d.t[5..] == [TENSOR_NONE16; 3],
        "unqualified shape or operands",
    )?;
    let n = u64::from(d.i[1]);
    let k = u64::from(d.i[2]);
    require(
        d.i[7] > 0 && d.i[7] < u32::from(TENSOR_NONE16),
        "missing map",
    )?;
    let handles = [d.t[0], d.t[1], d.t[2], d.t[3], d.t[4], d.i[7] as u16];
    require(
        handles.iter().copied().collect::<BTreeSet<_>>().len() == 6,
        "aliased operands",
    )?;
    for (slot, (&h, bytes)) in handles
        .iter()
        .zip([2 * n, k, n * k, 4, 4 * n, 128])
        .enumerate()
    {
        let t = packet.tensors.get(h as usize).ok_or("tensor handle")?;
        require(
            t.bytes >= bytes && !t.initialized,
            "tensor extent or initializer",
        )?;
        require(
            if slot == 2 || slot == 4 {
                packet::names::is_checkpoint_weight(t.name) && t.bytes == bytes
            } else if slot == 5 {
                t.name.starts_with("tmap.") && t.bytes == 128
            } else {
                t.name.starts_with("act.")
            },
            "operand storage class",
        )?;
    }
    let maps: Vec<_> = packet
        .generated
        .iter()
        .filter(|r| r.tensor == d.i[7])
        .collect();
    require(maps.len() == 1, "map generator coverage")?;
    let m = maps[0];
    require(
        m.kind == packet::rope::GEN_TMAP_E4M3
            && m.aux == u32::from(d.t[2])
            && m.ctx == d.i[1]
            && m.hd == d.i[2]
            && m.scale == 64
            && [m.theta, m.frac, m.factor, m.low, m.high, m.orig]
                .iter()
                .all(|v| v.to_bits() == 0),
        "weight map geometry or dtype",
    )?;
    for r in packet.generated {
        require(
            !handles[..5].iter().any(|&h| u32::from(h) == r.tensor),
            "generated operand alias",
        )?;
        if r.kind == packet::rope::GEN_TMAP_KV_PAIR {
            require(
                !handles
                    .iter()
                    .any(|&h| u32::from(h) == r.aux || u32::from(h) == r.scale),
                "KV map alias",
            )?;
        }
    }
    let mut stream = g.stream.to_vec();
    let mut queue = g.gq_stream.to_vec();
    for e in stream.iter_mut().chain(&mut queue) {
        e.seg = 0;
    }
    let windows = [0, queue.len() as u32];
    let normalized = Program {
        stream: &stream,
        gq_stream: &queue,
        gq_seg_ofs: &windows,
        ..*g
    };
    let deps = crate::splitk::dependencies(&normalized)?;
    require(
        g.gq_stream.windows(2).all(|w| w[0].inst <= w[1].inst),
        "reordered instruction windows",
    )?;
    let pc_ancestors = dependency_ancestors(&deps, pc);
    let mut pc_descendants = vec![false; deps.len()];
    pc_descendants[pc] = true;
    for j in pc + 1..deps.len() {
        pc_descendants[j] = deps[j].iter().any(|&parent| pc_descendants[parent]);
    }
    let mut quant = None;
    for (j, inst) in g.insts.iter().enumerate() {
        if inst.op == DevOp::QuantFp8 as u16
            && inst.t[0] == d.t[1]
            && inst.t[2] == d.t[3]
            && pc_ancestors[j]
        {
            quant = Some(j);
        }
        if inst.t.iter().any(|h| handles.contains(h)) {
            let output = writes(inst.op)?;
            for (slot, &h) in inst
                .t
                .iter()
                .enumerate()
                .filter(|(_, h)| handles.contains(h))
            {
                require(h != handles[5], "descriptor used as ordinary tensor")?;
                if output.contains(&slot) {
                    require(
                        h != d.t[2] && h != d.t[4],
                        "immutable weight or scale writer",
                    )?;
                    require(
                        j == pc
                            || if j < pc {
                                pc_ancestors[j]
                            } else {
                                pc_descendants[j]
                            },
                        "unordered activation/output lifetime",
                    )?;
                }
            }
        }
    }
    let q = quant.ok_or("missing ordered activation quantizer")?;
    let a = &g.insts[q];
    require(
        a.i == [1, d.i[2], 0, 0, 0, 0, 0, 0]
            && a.fj == [0; 3]
            && a.t[3..].iter().all(|&t| t == TENSOR_NONE16),
        "activation quantizer dtype/shape",
    )?;
    let input = packet
        .tensors
        .get(a.t[1] as usize)
        .ok_or("quantizer input")?;
    require(
        input.name.starts_with("act.") && input.bytes >= 2 * k && !handles.contains(&a.t[1]),
        "BF16 quantizer input",
    )?;
    let mut norm_writer = None;
    for (j, inst) in g.insts.iter().enumerate().take(pc) {
        if inst
            .t
            .iter()
            .any(|&h| h == d.t[1] || h == d.t[3] || h == a.t[1])
        {
            let output = writes(inst.op)?;
            for &slot in output {
                if inst.t[slot] == d.t[1] || inst.t[slot] == d.t[3] {
                    require(j <= q, "activation overwritten after quantizer")?;
                }
                if inst.t[slot] == a.t[1] && j < q {
                    norm_writer = Some(j);
                }
            }
        }
    }
    let nw = norm_writer.ok_or("missing BF16 quantizer-input producer")?;
    let quant_ancestors = dependency_ancestors(&deps, q);
    require(
        quant_ancestors[nw]
            && matches!(
                DevOp::from_u16(g.insts[nw].op),
                Some(
                    DevOp::QwenRmsNorm
                        | DevOp::RmsNorm
                        | DevOp::Gemm
                        | DevOp::GemmMed
                        | DevOp::GemmSmall
                        | DevOp::Gemv
                        | DevOp::Residual
                )
            ),
        "quantizer input producer dtype or ordering",
    )?;
    for &site in packet.kv_row_insts {
        require(
            site as usize != pc && site as usize != q,
            "KV row patch alias",
        )?;
    }
    // The immutable operands may be shared by prefill, but never written by any program.
    for other in packet.programs {
        for inst in other.insts {
            if inst
                .t
                .iter()
                .any(|&h| h == d.t[2] || h == d.t[4] || h == handles[5])
            {
                let output = writes(inst.op)?;
                require(
                    inst.t.iter().enumerate().all(|(s, &h)| {
                        h != handles[5] && (!(h == d.t[2] || h == d.t[4]) || !output.contains(&s))
                    }),
                    "cross-program immutable operand write",
                )?;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture() -> packet::devbuild::Model {
        use packet::dev::{DevOp, TENSOR_NONE};
        use packet::devbuild::{Builder, Model};
        let mut b = Builder::new(132);
        b.force_uniseg();
        let out = b.tensor("act.out", 20480);
        let a = b.tensor("act.fp8", 5120);
        let w = b.tensor("fp8/weight", 10240 * 5120);
        let scale = b.tensor("act.scale", 4);
        let ws = b.tensor("fp8/weight_scale", 10240 * 4);
        let norm = b.tensor("act.norm", 10240);
        let input = b.tensor("act.input", 10240);
        let map = b.tensor_gen(
            "tmap.weight",
            128,
            packet::rope::GenTensor::tmap_e4m3(w, 10240, 5120, 64),
        );
        let z = b.emit(DevOp::Nop, vec![0], &[], |_| {});
        let n = b.emit(DevOp::QwenRmsNorm, vec![0], &[z], |d| {
            d.t[0] = norm;
            d.t[1] = input;
            d.i[0] = 1;
            d.i[1] = 5120;
        });
        let q = b.emit(DevOp::QuantFp8, vec![0], &[n], |d| {
            d.t = [
                a,
                norm,
                scale,
                TENSOR_NONE,
                TENSOR_NONE,
                TENSOR_NONE,
                TENSOR_NONE,
                TENSOR_NONE,
            ];
            d.i[0] = 1;
            d.i[1] = 5120;
        });
        let g = b.emit(DevOp::GemmFp8, b.all(), &[q], |d| {
            d.t[..5].copy_from_slice(&[out, a, w, scale, ws]);
            d.i = [1, 10240, 5120, 0, 0, 0, 0, map];
        });
        b.emit(DevOp::Nop, vec![0], &[g], |_| {});
        let gen = b.gen_tensors();
        let p = b.finish();
        Model {
            n_cu: 132,
            target: 0,
            tensors: p.tensors.clone(),
            progs: vec![p],
            kv_row_insts: vec![],
            prog_t: vec![1],
            gen,
        }
    }

    #[test]
    fn exact_packet_and_unsafe_mutations() {
        let run =
            |m: &packet::devbuild::Model| crate::program::with_model(m, |p| validate(p, 0, 3));
        assert!(run(&fixture()).is_ok());
        let mutations: &[fn(&mut packet::devbuild::Model)] = &[
            |m| m.prog_t[0] = 2,
            |m| m.progs[0].insts[3].i[0] = 2,
            |m| m.progs[0].insts[3].i[1] = 10241,
            |m| m.progs[0].insts[3].i[2] = 5121,
            |m| m.progs[0].insts[3].i[4] = 1,
            |m| m.progs[0].insts[3].blocks = 131,
            |m| m.progs[0].insts[3].t[0] = m.progs[0].insts[3].t[1],
            |m| m.tensors[2].bytes -= 1,
            |m| m.tensors[4].name = "act.fake_scale".into(),
            |m| m.gen[0].kind = packet::rope::GEN_TMAP_BF16,
            |m| m.gen[0].aux = 0,
            |m| m.gen[0].scale = 128,
            |m| m.gen[0].ctx = 10239,
            |m| m.gen.push(m.gen[0]),
            |m| m.kv_row_insts.push(3),
            |m| m.progs[0].insts[2].op = DevOp::Nop as u16,
            |m| m.progs[0].gq_stream.swap(0, 3),
            |m| m.progs[0].stream[0].flags = packet::dev::SE_XCTR,
            |m| m.progs[0].waits.last_mut().unwrap().threshold = 1,
            |m| m.progs[0].insts[4].op = DevOp::Residual as u16,
        ];
        for (i, mutate) in mutations.iter().enumerate() {
            let mut m = fixture();
            mutate(&mut m);
            if i == 19 {
                m.progs[0].insts[4].t[0] = 2;
            }
            assert!(run(&m).is_err(), "mutation{i}");
        }
    }
    #[test]
    fn compiled_capability_and_capacity() {
        assert!(options(0, false, false).is_ok());
        for args in [(1, false, false), (0, true, false), (0, false, true)] {
            assert!(options(args.0, args.1, args.2).is_err());
        }
        let v = [
            Some(1),
            Some(1),
            Some(256),
            Some(ARENA),
            Some(17408),
            Some(16),
            Some(1),
        ];
        assert!(capability("sm90a", v, 1, Some(264), 132).is_ok());
        for i in 0..v.len() {
            let mut bad = v;
            bad[i] = None;
            assert!(capability("sm90a", bad, 1, Some(132), 132).is_err());
        }
        for cap in [None, Some(0), Some(131)] {
            assert!(capability("sm90a", v, 1, cap, 132).is_err());
        }
        assert!(capability("sm120", v, 1, Some(132), 132).is_err());
        assert!(capability("sm90a", v, 0, Some(132), 132).is_err());
        assert!(capability("sm90a", v, 1, Some(132), 0).is_err());
    }
}
