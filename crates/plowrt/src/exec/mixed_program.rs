use std::collections::{BTreeMap, BTreeSet};

use packet::dev::{DevInst64, DevOp, TENSOR_NONE, TENSOR_NONE16};
use packet::devbuild::Builder;
use plow_asset::aux_program::Program;

use crate::asset::devblob::{DevBlob, DevProg};
use crate::{Result, RuntimeError};

pub(crate) struct TensorSpec {
    pub handle: u16,
    pub name: String,
    pub bytes: u64,
}

pub(crate) struct MixedProgramSpec {
    pub program: Program,
    pub decode_rows: u32,
    pub decode_slot: u16,
}

pub(crate) struct SynthesizedMixed {
    pub programs: Vec<MixedProgramSpec>,
    pub tensors: Vec<TensorSpec>,
}

fn reject(message: impl Into<String>) -> RuntimeError {
    RuntimeError::Rejected(format!("runtime mixed program: {}", message.into()))
}

fn op(inst: &DevInst64) -> Result<DevOp> {
    DevOp::from_u16(inst.op).ok_or_else(|| reject("unknown instruction"))
}

fn gemm(op: DevOp) -> bool {
    matches!(
        op,
        DevOp::Gemm | DevOp::GemmSmall | DevOp::GemmMed | DevOp::GemmWide | DevOp::GemmC5
    )
}

fn split_norm_residual_norm(inst: &DevInst64, rows: u32) -> Result<[DevInst64; 2]> {
    if inst.i[0] != rows
        || inst.i[1] == 0
        || inst.i[2..] != [0; 6]
        || inst.fj[2] != 0
        || inst.t[6..] != [TENSOR_NONE16; 2]
        || inst.t[..4].contains(&TENSOR_NONE16)
        || inst.t[0] == inst.t[1]
        || inst.t[4..6]
            .iter()
            .any(|g| *g != TENSOR_NONE16 && inst.t[..2].contains(g))
    {
        return Err(reject("unsupported sandwich norm operands"));
    }
    // Keep the BF16 residual store and its dependent reload. Both mixed consumers
    // already implement the ordinary logical reduction width and runtime rows.
    let mut residual = DevInst64 {
        op: DevOp::NormResidual as u16,
        t: [TENSOR_NONE16; 8],
        ..Default::default()
    };
    residual.t[..4].copy_from_slice(&inst.t[1..5]);
    residual.i[..2].copy_from_slice(&inst.i[..2]);
    residual.fj = inst.fj;
    let mut norm = DevInst64 {
        op: DevOp::RmsNorm as u16,
        t: [TENSOR_NONE16; 8],
        ..Default::default()
    };
    norm.t[..3].copy_from_slice(&[inst.t[0], inst.t[1], inst.t[5]]);
    norm.i[..2].copy_from_slice(&inst.i[..2]);
    norm.fj[0] = inst.fj[0];
    Ok([residual, norm])
}

fn bytes(factors: &[u32]) -> Result<u64> {
    factors.iter().try_fold(1u64, |size, &factor| {
        size.checked_mul(factor as u64)
            .ok_or_else(|| reject("tensor size overflow"))
    })
}

fn tensor(blob: &DevBlob, name: &str) -> Result<u16> {
    let index = blob
        .tensors
        .iter()
        .position(|t| t.name == name)
        .ok_or_else(|| reject(format!("missing {name}")))?;
    u16::try_from(index)
        .ok()
        .filter(|&h| h != TENSOR_NONE16)
        .ok_or_else(|| reject("tensor handle overflow"))
}

fn reserve(
    blob: &DevBlob,
    specs: &mut BTreeMap<u16, TensorSpec>,
    handle: u16,
    required: u64,
) -> Result<()> {
    let original = blob
        .tensors
        .get(handle as usize)
        .ok_or_else(|| reject("tensor handle out of bounds"))?;
    if original.init.is_some() || !original.name.starts_with("act.") {
        return Err(reject(
            "scratch expansion requires an uninitialized act.* tensor",
        ));
    }
    if original.bytes >= required && !specs.contains_key(&handle) {
        return Ok(());
    }
    let spec = specs.entry(handle).or_insert_with(|| TensorSpec {
        handle,
        name: original.name.clone(),
        bytes: original.bytes,
    });
    spec.bytes = spec.bytes.max(required);
    Ok(())
}

fn decode_attention(program: &DevProg) -> Result<BTreeMap<(u16, u16), (DevInst64, DevInst64)>> {
    let mut result = BTreeMap::new();
    for (index, inst) in program.insts.iter().enumerate() {
        if inst.op != DevOp::FlashDecode as u16 {
            continue;
        }
        if inst.i[1] & 0xffff0000 != 0 || inst.t[6] != TENSOR_NONE16 || inst.t[7] != TENSOR_NONE16 {
            return Err(reject("folded decode attention is unsupported"));
        }
        let merge = program
            .insts
            .get(index + 1)
            .filter(|next| {
                next.op == DevOp::FlashMerge as u16
                    && next.t[1] == inst.t[0]
                    && next.t[2] == inst.t[1]
                    && next.i[..4] == [program.t, inst.i[1], inst.i[5], inst.i[6]]
            })
            .ok_or_else(|| reject("decode attention requires an adjacent merge"))?;
        if result
            .insert((inst.t[3], inst.t[4]), (*inst, *merge))
            .is_some()
        {
            return Err(reject("ambiguous decode KV operands"));
        }
    }
    if result.is_empty() {
        return Err(reject("no dense decode attention"));
    }
    Ok(result)
}

pub(crate) fn synthesize(blob: &DevBlob, physical_batch: usize) -> Result<SynthesizedMixed> {
    if blob.tp.is_some()
        || physical_batch < 2
        || physical_batch > u32::MAX as usize
        || blob.n_cu == 0
        || blob.n_cu > u16::MAX as u32
    {
        return Err(reject(
            "requires single-GPU dense BF16 with at least two slots",
        ));
    }
    let decode = blob
        .progs
        .iter()
        .filter(|p| {
            !p.packed_prefill_only
                && p.insts.iter().any(|i| i.op == DevOp::FlashDecode as u16)
                && !p.insts.iter().any(|i| i.op == DevOp::FlashPrefill as u16)
        })
        .min_by_key(|p| p.t)
        .ok_or_else(|| reject("no ordinary decode program"))?;
    let attention = decode_attention(decode)?;
    let prefills: Vec<_> = blob
        .progs
        .iter()
        .filter(|p| {
            !p.packed_prefill_only
                && p.t > 1
                && p.insts.iter().any(|i| i.op == DevOp::FlashPrefill as u16)
        })
        .collect();
    if prefills.is_empty() {
        return Err(reject("no ordinary dense prefill program"));
    }
    let max_rows = prefills.iter().map(|p| p.t).max().unwrap();
    let ids = tensor(blob, "in.ids")?;
    let pos = tensor(blob, "in.pos")?;
    let kvlen = tensor(blob, "in.kvlen")?;
    let decode_capacity = blob
        .progs
        .iter()
        .filter(|p| {
            !p.packed_prefill_only
                && p.insts.iter().any(|i| i.op == DevOp::FlashDecode as u16)
                && !p.insts.iter().any(|i| i.op == DevOp::FlashPrefill as u16)
        })
        .map(|p| p.t)
        .max()
        .unwrap_or(0);
    if blob.tensors[kvlen as usize].bytes != bytes(&[physical_batch as u32, 4])?
        || decode_capacity != physical_batch as u32
    {
        return Err(reject(
            "physical batch disagrees with ordinary decode metadata",
        ));
    }
    for (decode, _) in attention.values() {
        let required = bytes(&[
            physical_batch as u32,
            decode.i[2],
            decode.i[3],
            decode.i[6],
            2,
        ])?;
        if required == 0
            || [decode.t[3], decode.t[4]].into_iter().any(|handle| {
                !blob.tensors.get(handle as usize).is_some_and(|tensor| {
                    tensor.name.starts_with("kv.")
                        && tensor.init.is_none()
                        && tensor.bytes >= required
                })
            })
        {
            return Err(reject("KV allocation does not cover physical batch"));
        }
    }
    let mut specs = BTreeMap::new();
    for handle in [ids, pos, kvlen] {
        let original = &blob.tensors[handle as usize];
        if original.init.is_some() {
            return Err(reject("initialized runtime input"));
        }
        specs.insert(
            handle,
            TensorSpec {
                handle,
                name: original.name.clone(),
                bytes: original.bytes.max(bytes(&[max_rows, 4])?),
            },
        );
    }
    let next = u16::try_from(blob.tensors.len()).map_err(|_| reject("tensor handle overflow"))?;
    if next > TENSOR_NONE16 - 3 {
        return Err(reject("tensor handle overflow"));
    }
    let decode_slot = next;
    let pf_partial = next + 1;
    let pf_ml = next + 2;
    for (handle, name, size) in [
        (
            decode_slot,
            plow_asset::mixed_step::DECODE_SLOT_TENSOR,
            bytes(&[physical_batch as u32, 4])?,
        ),
        (pf_partial, "act.mixed_prefill_opart", 4),
        (pf_ml, "act.mixed_prefill_mlpart", 4),
    ] {
        if blob.tensors.iter().any(|t| t.name == name) {
            return Err(reject("runtime tensor name already exists"));
        }
        specs.insert(
            handle,
            TensorSpec {
                handle,
                name: name.into(),
                bytes: size,
            },
        );
    }
    let mut programs = Vec::new();
    let mut seen_rows = BTreeSet::new();
    for source in prefills {
        if !seen_rows.insert(source.t) {
            return Err(reject("duplicate ordinary prefill capacity"));
        }
        let dcap = (physical_batch as u32 - 1).min(source.t - 1);
        let mut instructions = Vec::new();
        let mut paired = BTreeSet::new();
        let mut index = 0;
        while index < source.insts.len() {
            let mut inst = source.insts[index];
            let code = op(&inst)?;
            if gemm(code)
                && source
                    .insts
                    .get(index + 2)
                    .is_some_and(|i| i.op == DevOp::Glu as u16)
            {
                let up = source.insts[index + 1];
                let glu = source.insts[index + 2];
                if !gemm(op(&up)?)
                    || inst.i[..3] != up.i[..3]
                    || inst.i[0] != source.t
                    || inst.i[3..6] != [0; 3]
                    || up.i[3..6] != [0; 3]
                    || inst.t[1] != up.t[1]
                    || glu.t[1] != inst.t[0]
                    || glu.t[2] != up.t[0]
                    || Some(glu.i[0]) != source.t.checked_mul(inst.i[1])
                    || glu.i[1] > 1
                {
                    return Err(reject("unsupported gate/up/GLU pattern"));
                }
                inst.op = DevOp::GemmGlu as u16;
                inst.t[0] = glu.t[0];
                inst.t[5] = up.t[2];
                inst.i[5] = glu.i[1];
                inst.fj = [0; 3];
                instructions.push(inst);
                index += 3;
                continue;
            }
            match code {
                DevOp::NormResidualNorm => {
                    instructions.extend(split_norm_residual_norm(&inst, source.t)?);
                    index += 1;
                    continue;
                }
                DevOp::Embed | DevOp::RmsNorm | DevOp::NormResidual | DevOp::GemmGlu => {
                    if inst.i[0] != source.t {
                        return Err(reject("body row count mismatch"));
                    }
                }
                DevOp::HeadNormRope => {
                    if inst.i[0] != source.t || inst.t[5] != pos {
                        return Err(reject("head norm row binding"));
                    }
                    if inst.fj[1] != 0 {
                        inst.t[6] = decode_slot;
                    }
                }
                DevOp::FlashPrefill => {
                    let key = (inst.t[3], inst.t[4]);
                    let &(mut dec, mut merge) = attention
                        .get(&key)
                        .ok_or_else(|| reject("prefill/decode KV mismatch"))?;
                    if !paired.insert(key)
                        || inst.i[0] != source.t
                        || !matches!(inst.i[6], 256 | 512)
                        || inst.i[2] != dec.i[1]
                        || inst.i[3] != dec.i[2]
                        || inst.i[5] != dec.i[4]
                        || inst.i[6] != dec.i[6]
                        || inst.fj[0] != dec.fj[0]
                        || inst.fj[1] != dec.i[3]
                        || inst.fj[2] != dec.i[7]
                        || inst.t[2] != dec.t[2]
                        || dec.t[5] != kvlen
                        || inst.i[7] == 0
                    {
                        return Err(reject("prefill/decode attention geometry mismatch"));
                    }
                    let out = if inst.i[7] == 1 && inst.t[5] != TENSOR_NONE16 {
                        inst.t[5]
                    } else {
                        let next = source
                            .insts
                            .get(index + 1)
                            .filter(|next| {
                                next.op == DevOp::FlashMerge as u16
                                    && next.t[1] == inst.t[0]
                                    && next.t[2] == inst.t[1]
                                    && next.i[..4] == [source.t, inst.i[2], inst.i[7], inst.i[6]]
                            })
                            .ok_or_else(|| reject("prefill attention requires its merge"))?;
                        index += 1;
                        next.t[0]
                    };
                    if merge.t[0] != out {
                        return Err(reject("attention output binding mismatch"));
                    }
                    dec.i[0] = dcap;
                    dec.t[6] = decode_slot;
                    merge.i[0] = dcap;
                    reserve(
                        blob,
                        &mut specs,
                        dec.t[0],
                        bytes(&[dcap, dec.i[1], dec.i[5], dec.i[6], 4])?,
                    )?;
                    reserve(
                        blob,
                        &mut specs,
                        dec.t[1],
                        bytes(&[dcap, dec.i[1], dec.i[5], 2, 4])?,
                    )?;
                    instructions.extend([dec, merge]);
                    inst.t[0] = pf_partial;
                    inst.t[1] = pf_ml;
                    inst.t[5] = if inst.i[7] == 1 { out } else { TENSOR_NONE16 };
                    for (handle, size) in [
                        (
                            pf_partial,
                            bytes(&[source.t, inst.i[2], inst.i[7], inst.i[6], 4])?,
                        ),
                        (pf_ml, bytes(&[source.t, inst.i[2], inst.i[7], 2, 4])?),
                    ] {
                        let spec = specs.get_mut(&handle).unwrap();
                        spec.bytes = spec.bytes.max(size);
                    }
                    instructions.push(inst);
                    if inst.i[7] > 1 {
                        merge.t[1] = pf_partial;
                        merge.t[2] = pf_ml;
                        merge.i = [source.t, inst.i[2], inst.i[7], inst.i[6], dcap, 0, 0, 0];
                        instructions.push(merge);
                    }
                    index += 1;
                    continue;
                }
                DevOp::Gemv => {
                    if inst.i[0] != 1 || inst.i[4] != source.t - 1 || inst.i[3] != 0 {
                        return Err(reject("unsupported prefill GEMV"));
                    }
                    inst.op = DevOp::Gemm as u16;
                    inst.i[0] = dcap;
                    inst.i[4] = 0;
                    reserve(blob, &mut specs, inst.t[0], bytes(&[dcap, inst.i[1], 2])?)?;
                }
                code if gemm(code) => {
                    if inst.i[3] != 0 || inst.i[5] != 0 {
                        return Err(reject("folded/banded GEMM unsupported"));
                    }
                    if inst.i[0] == 1 && inst.i[4] == source.t - 1 {
                        inst.i[0] = dcap;
                        inst.i[4] = 0;
                        reserve(blob, &mut specs, inst.t[0], bytes(&[dcap, inst.i[1], 2])?)?;
                    } else if inst.i[0] != source.t || inst.i[4] != 0 {
                        return Err(reject("GEMM row role mismatch"));
                    }
                    inst.op = DevOp::Gemm as u16;
                }
                DevOp::SoftCap => {
                    inst.i[0] = inst.i[0]
                        .checked_mul(dcap)
                        .ok_or_else(|| reject("logit size overflow"))?;
                    inst.i[1] = dcap;
                }
                DevOp::Argmax => {
                    inst.i[1] = dcap;
                    reserve(blob, &mut specs, inst.t[0], bytes(&[dcap, blob.n_cu, 8])?)?;
                }
                DevOp::ArgmaxFin => {
                    inst.i[0] = blob.n_cu;
                    inst.i[1] = dcap;
                }
                _ => return Err(reject(format!("unsupported prefill opcode {}", inst.op))),
            }
            instructions.push(inst);
            index += 1;
        }
        if paired.len() != attention.len() {
            return Err(reject("prefill/decode layer count mismatch"));
        }
        let mut builder = Builder::new(blob.n_cu);
        builder.force_uniseg();
        builder.set_fuse_materialized_residual_inputs(false);
        for (handle, t) in blob.tensors.iter().enumerate() {
            builder.tensor(
                &t.name,
                specs.get(&(handle as u16)).map_or(t.bytes, |s| s.bytes),
            );
        }
        for spec in specs
            .values()
            .filter(|s| s.handle as usize >= blob.tensors.len())
        {
            builder.tensor(&spec.name, spec.bytes);
        }
        let mut previous = None;
        for inst in instructions {
            let deps: Vec<_> = previous.into_iter().collect();
            previous = Some(
                builder.emit(op(&inst)?, (0..blob.n_cu).collect(), &deps, |d| {
                    d.t = inst.t.map(|h| {
                        if h == TENSOR_NONE16 {
                            TENSOR_NONE
                        } else {
                            h as u32
                        }
                    });
                    d.i = inst.i;
                    d.f = [f32::from_bits(inst.fj[0]), 0.0];
                    d.j = [inst.fj[1], inst.fj[2]];
                }),
            );
        }
        let built = builder.finish();
        let program = Program {
            rows: source.t,
            n_counter: built.n_counter,
            insts: built.insts.iter().map(|i| i.pack()).collect(),
            stream: built.stream,
            stream_ofs: built.stream_ofs,
            stream_len: built.stream_len,
            waits: built.waits,
            succs: built.succs,
            gq_stream: built.gq_stream,
            gq_seg_ofs: built.gq_seg_ofs,
        };
        programs.push(MixedProgramSpec {
            program,
            decode_rows: dcap,
            decode_slot,
        });
    }
    let mut contracts: Vec<_> = blob
        .tensors
        .iter()
        .enumerate()
        .map(|(handle, t)| plow_asset::mixed_step::TensorContract {
            name: &t.name,
            bytes: specs.get(&(handle as u16)).map_or(t.bytes, |s| s.bytes),
            initialized: t.init.is_some(),
        })
        .collect();
    for spec in specs
        .values()
        .filter(|s| s.handle as usize >= blob.tensors.len())
    {
        contracts.push(plow_asset::mixed_step::TensorContract {
            name: &spec.name,
            bytes: spec.bytes,
            initialized: false,
        });
    }
    for spec in &programs {
        plow_asset::mixed_step::dense_amd_capacity_consumer_contract(
            &spec.program,
            spec.decode_rows,
            &contracts,
        )
        .map_err(reject)?;
        plow_asset::aux_program::Section {
            n_cu: blob.n_cu,
            programs: vec![spec.program.clone()],
        }
        .validate(contracts.len())
        .map_err(reject)?;
    }
    Ok(SynthesizedMixed {
        programs,
        tensors: specs.into_values().collect(),
    })
}

#[cfg(test)]
#[path = "mixed_program_tests.rs"]
mod tests;
