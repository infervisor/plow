//! Lower a scheduled tile graph to the runtime [`packet::Program`] (Pass G's
//! ABI form). Each scheduled task becomes one [`packet::Inst`]: a kernel opcode,
//! an opcode-specific body (only the fields that op needs), and its
//! wait/successor counters. The ABI itself lives in the standalone `packet` crate.

use crate::expand::{TaskGraph, TaskId, TaskKind};
use crate::passes::{Schedule, Scope};
use crate::resource::ResourceId;
use packet::{Body, Counter, Inst, Opcode, Program, ResourceKind, SLOT_NONE};
use rewrite::{Compute, ConstraintSet, GraphNode, OpKind, TileGraph};

/// Build the runtime instruction stream from a scheduled graph.
///
/// `bucket_id` and `plan_gen` are embedded in the stream header so the runtime
/// can identify which shape bucket / generation this program belongs to.
pub fn emit_program(
    g: &TileGraph,
    cons: &ConstraintSet,
    tasks: &TaskGraph,
    sched: &Schedule,
) -> Program {
    emit_program_with_meta(g, cons, tasks, sched, 0, 0)
}

/// Like [`emit_program`] but allows setting stream-level metadata (bucket id,
/// plan generation) that the runtime uses for dispatch and invalidation.
pub fn emit_program_with_meta(
    g: &TileGraph,
    cons: &ConstraintSet,
    tasks: &TaskGraph,
    sched: &Schedule,
    bucket_id: u16,
    plan_gen: u16,
) -> Program {
    // Per-task wait / successor counter ids (from Pass G's high-level packets).
    let mut wait: Vec<Vec<u32>> = vec![Vec::new(); tasks.tasks.len()];
    let mut succ: Vec<Vec<u32>> = wait.clone();
    for pkts in sched.packets.values() {
        for p in pkts {
            wait[p.task] = p.wait.iter().map(|&c| c as u32).collect();
            succ[p.task] = p.successors.iter().map(|&c| c as u32).collect();
        }
    }

    // Compile-time assertion: wait/succ must fit in u16 (Header field width).
    for (t, (w, s)) in wait.iter().zip(succ.iter()).enumerate() {
        assert!(
            w.len() <= u16::MAX as usize,
            "task {t}: wait counter list ({}) exceeds u16 max (65535) — \
             enable tree-relay counters or reduce tile count",
            w.len()
        );
        assert!(
            s.len() <= u16::MAX as usize,
            "task {t}: succ counter list ({}) exceeds u16 max (65535) — \
             enable tree-relay counters or reduce tile count",
            s.len()
        );
    }

    let order = issue_order(sched);

    // HBM address map: each named tensor → a logical slot id the runtime resolves
    // to a base+offset via the address-map artifact. DMA records carry this slot
    // in the `tensor` field (was always TENSOR_NONE before the allocator existed).
    let amap = crate::memory::plan_from_schedule(tasks, sched, cons);
    let tensor_slot = |t: TaskId| {
        tasks.tasks[t]
            .tensor
            .as_deref()
            .and_then(|name| amap.get(name))
            .map(|e| e.slot)
            .unwrap_or(u32::MAX)
    };
    // The buffer's data-type tag (`BufKind` as u8) for DMA records, resolved
    // from its address-map class. `KIND_UNSPECIFIED` when the tensor has no map
    // entry (e.g. TENSOR_NONE transfers).
    let tensor_kind = |t: TaskId| -> u8 {
        tasks.tasks[t]
            .tensor
            .as_deref()
            .and_then(|name| amap.get(name))
            .map(|e| e.class.default_kind() as u8)
            .unwrap_or(packet::KIND_UNSPECIFIED)
    };

    let slot = |t: TaskId| {
        sched
            .sram_slots
            .get(&t)
            .and_then(|s| s.first())
            .map(|&p| p as u16)
            .unwrap_or(SLOT_NONE)
    };
    let tmem = |t: TaskId| {
        sched
            .tmem_slots
            .get(&t)
            .and_then(|s| s.first())
            .map(|&c| c as u16)
            .unwrap_or(SLOT_NONE)
    };

    let mut insts = Vec::with_capacity(order.len());
    for t in order {
        let task = &tasks.tasks[t];
        let (resource, unit, index) = encode_resource(sched.placement[&t]);
        let body = body_for(
            g,
            cons,
            task,
            slot(t),
            tmem(t),
            tensor_slot(t),
            tensor_kind(t),
        );
        insts.push(Inst {
            resource,
            unit,
            index,
            body,
            wait: wait[t].clone(),
            succ: succ[t].clone(),
        });
    }

    let counters = sched
        .counters
        .iter()
        .map(|c| Counter {
            id: c.id as u32,
            threshold: c.threshold,
            scope: match c.scope {
                Scope::IntraSm => 0,
                Scope::IntraGpu => 1,
                Scope::CrossUnit => 2,
            },
            _pad: [0; 3],
        })
        .collect();

    Program {
        insts,
        counters,
        bucket_id,
        plan_gen,
        flags: 0,
    }
}

/// Task ids in the exact order [`emit_program`] lays them in the stream
/// (scheduled start, then task id). `program.insts[i]` is the instruction for
/// `issue_order(sched)[i]` — the verifier uses this to map records back to tasks.
pub fn issue_order(sched: &Schedule) -> Vec<TaskId> {
    let mut order: Vec<(TaskId, u64)> = sched
        .placement
        .keys()
        .map(|&t| (t, sched.starts.get(t).copied().unwrap_or(0)))
        .collect();
    order.sort_by_key(|&(t, s)| (s, t));
    order.into_iter().map(|(t, _)| t).collect()
}

/// Pack a [`ResourceId`] into the ABI `(kind, unit:u8, index:u16)` fields. The
/// widths are ample at current scales (≤256 units, ≤65 535 SMs/engines per unit);
/// the debug asserts catch any future topology that would silently truncate.
fn encode_resource(r: ResourceId) -> (ResourceKind, u8, u16) {
    debug_assert!(
        resource_fits(r),
        "resource id {r:?} exceeds the packet ABI field widths"
    );
    match r {
        ResourceId::Sm(u, s) => (ResourceKind::Sm, u as u8, s as u16),
        ResourceId::Dma(u, e) => (ResourceKind::Dma, u as u8, e as u16),
        ResourceId::Dpu(i) => (ResourceKind::Dpu, 0, i as u16),
        ResourceId::Host(i) => (ResourceKind::Host, 0, i as u16),
    }
}

/// Whether `r`'s indices fit the ABI field widths (unit in `u8`, slot in `u16`).
fn resource_fits(r: ResourceId) -> bool {
    match r {
        ResourceId::Sm(u, s) => u <= u8::MAX as usize && s <= u16::MAX as usize,
        ResourceId::Dma(u, e) => u <= u8::MAX as usize && e <= u16::MAX as usize,
        ResourceId::Dpu(i) | ResourceId::Host(i) => i <= u16::MAX as usize,
    }
}

fn body_for(
    g: &TileGraph,
    cons: &ConstraintSet,
    task: &crate::expand::Task,
    out: u16,
    tmem: u16,
    tensor: u32,
    tensor_kind: u8,
) -> Body {
    // The ABI `bytes` field is u32 by design (a single tile/transfer never moves
    // ≥4 GiB). Catch any op that would exceed it in debug builds rather than
    // silently truncating to a wrong-size DMA at runtime.
    debug_assert!(
        task.bytes <= u32::MAX as u64,
        "task '{}' transfer {} B exceeds the 4 GiB packet `bytes` field — tile it finer",
        task.op,
        task.bytes
    );
    let bytes = task.bytes.min(u32::MAX as u64) as u32;
    let c0 = task.coord.first().copied().unwrap_or(0).max(0) as u32;
    let c1 = task.coord.get(1).copied().unwrap_or(0).max(0) as u32;
    match task.kind {
        TaskKind::DmaIn if task.cross_unit => Body::Rdma {
            bytes,
            src_unit: 0,
            dst_unit: task.unit as u8,
        },
        // A load reads the moved buffer, a store writes it.
        TaskKind::DmaIn => Body::Dma {
            load: true,
            bytes,
            slot: out,
            tensor,
            kind: tensor_kind,
            access: packet::ACCESS_READ,
        },
        TaskKind::DmaOut => Body::Dma {
            load: false,
            bytes,
            slot: out,
            tensor,
            kind: tensor_kind,
            access: packet::ACCESS_WRITE,
        },
        TaskKind::Host => Body::Host,
        TaskKind::Compute => {
            let kind = cons.op_io.get(&task.node).map(|d| d.kind);
            let tile = match &g.nodes[task.node] {
                GraphNode::Compute { kind, .. } => Some(*kind),
                _ => None,
            };
            match (kind, tile) {
                (Some(OpKind::Gemm(s)), Some(Compute::Gemm(tl))) => Body::Gemm {
                    coord: [c0, c1],
                    m: s.m as u32,
                    n: s.n as u32,
                    k: s.k as u32,
                    bm: tl.bm as u16,
                    bn: tl.bn as u16,
                    bk: tl.bk as u16,
                    out,
                    tmem,
                    variant: gemm_variant_for(cons, task.node),
                },
                (Some(OpKind::Flash(a)), Some(Compute::Flash(tl))) => Body::Flash {
                    coord: [c0, c1],
                    seq_q: a.seq_q as u32,
                    seq_kv: a.seq_kv as u32,
                    head_dim: a.head_dim as u16,
                    bq: tl.bq as u16,
                    bkv: tl.bkv as u16,
                    heads: a.heads as u16,
                    kv_heads: a.kv_heads as u16,
                    window: a.sliding_window as u32,
                    out,
                    tmem,
                    variant: if a.sliding_window > 0 {
                        Opcode::VARIANT_FLASH_SLIDING_BF16
                    } else if a.causal {
                        Opcode::VARIANT_FLASH_CAUSAL_BF16
                    } else {
                        Opcode::VARIANT_GOLDEN
                    },
                },
                (Some(OpKind::Row(s)), _) => Body::Row {
                    reduce: s.reduce,
                    coord: c0,
                    rows: s.rows as u32,
                    feat: s.feat as u32,
                    operands: s.operands.clamp(0, 255) as u8,
                    br: row_block(g, task.node),
                    out,
                    variant: if s.reduce {
                        Opcode::VARIANT_ROW_RMS_BF16
                    } else {
                        Opcode::VARIANT_BF16
                    },
                    args: [0; 4],
                },
                (Some(OpKind::Model(m)), _) => Body::Row {
                    reduce: matches!(
                        m.kind,
                        rewrite::ModelOpKind::RmsNorm | rewrite::ModelOpKind::RmsNormZeroCentered
                    ),
                    coord: c0,
                    rows: m.rows as u32,
                    feat: m.feat as u32,
                    operands: m.operands.clamp(0, 255) as u8,
                    args: m.args,
                    br: row_block(g, task.node),
                    out,
                    variant: Opcode::VARIANT_MODEL_BASE + m.kind as u8,
                },
                // Layout: a strided descriptor (transpose/broadcast/slice) when the
                // bridge produced one, else a contiguous copy into `out`.
                (Some(OpKind::Layout(s)), _) if s.kind != 0 => Body::Layout {
                    kind: s.kind,
                    rank: s.rank,
                    elem_size: s.elem_size,
                    out,
                    shape: s.shape,
                    in_stride: s.in_stride,
                    out_stride: s.out_stride,
                    in_base: s.in_base,
                    out_base: s.out_base,
                },
                // kind-0 layouts and internal joins: contiguous byte copy.
                _ => Body::layout_copy(out, bytes),
            }
        }
    }
}

/// Select the GEMM kernel variant byte based on the op's per-operand dtype info
/// stored in the constraint set's `OpDesc`.
fn gemm_variant_for(cons: &ConstraintSet, node: usize) -> u8 {
    let desc = match cons.op_io.get(&node) {
        Some(d) => d,
        None => return Opcode::VARIANT_BF16,
    };
    // Block-quant weights always select the W4A8 dequant-fused kernel.
    if desc.block_quant {
        return Opcode::VARIANT_W4A8;
    }
    // MX FP4: native Blackwell 4-bit tensor-core path.
    if desc.native_fp4 {
        return Opcode::VARIANT_FP4;
    }
    // Standard (non-block-quant) dtype selection by element size.
    match (desc.weight_elem, desc.activation_elem) {
        (2, 2) => Opcode::VARIANT_BF16, // standard bf16×bf16
        (1, 2) => Opcode::VARIANT_FP8,  // FP8 weights, bf16 activations
        (1, 1) => Opcode::VARIANT_FP8,  // FP8×FP8
        _ => Opcode::VARIANT_BF16,
    }
}

/// The row tile's block size, if the node carries a row tile.
fn row_block(g: &TileGraph, node: usize) -> u16 {
    match &g.nodes[node] {
        GraphNode::Compute {
            kind: Compute::Row(t),
            ..
        } => t.br as u16,
        _ => 0,
    }
}
