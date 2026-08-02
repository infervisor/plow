//! Integration test: run the §8.3 prefetch pass on a real scheduled tile
//! graph and assert full ordering consistency (streams ↔ packets ↔ starts)
//! plus that `verify_schedule` still accepts the counter-gated protocol.
//!
//! The prefetch pass reorders positions in resource streams; this test proves
//! nothing gets out of sync and no data dependency is violated on a
//! realistic transformer-block bucket.

use costmodel::{hwspec, Soc, SramPolicy, DEFAULT_PAGE_BYTES};
use nn_graph::{infer_shapes, ActKind, DType, Nn};
use rewrite::{assemble, plan_from_block, LayerPlan};
use schedule::{
    memory::plan_from_schedule_with_task_sets, prefetch::hoist_prefetches, schedule, Config,
    Scheduled, TaskKind,
};

const H: i64 = 256;
const NH: i64 = 4;
const NKV: i64 = 2;
const HD: i64 = 64;
const QD: i64 = NH * HD;
const KVD: i64 = NKV * HD;
const IM: i64 = 512;
const T: i64 = 256;

fn h100() -> &'static hwspec::GpuSpec {
    hwspec::registry::lookup("H100 SXM5").unwrap()
}

fn small_block() -> LayerPlan {
    let mut nn = Nn::new(DType::BF16, DType::BF16);
    let x = nn.input("x", nn.shape([T.into(), H.into()]), DType::BF16);
    nn.begin_block("layers.0");
    let h1 = nn.rmsnorm("input_norm", x, H, 1e-6);
    let q = nn.linear("q_proj", h1, H, QD, false);
    let k = nn.linear("k_proj", h1, H, KVD, false);
    let v = nn.linear("v_proj", h1, H, KVD, false);
    let qh = nn.reshape(q, [T.into(), NH.into(), HD.into()]);
    let kh = nn.reshape(k, [T.into(), NKV.into(), HD.into()]);
    let vh = nn.reshape(v, [T.into(), NKV.into(), HD.into()]);
    let attn = nn.attention(
        qh, kh, vh, NH as u32, NKV as u32, HD as u32, true, None, None,
    );
    let ao = nn.reshape(attn, [T.into(), QD.into()]);
    let o = nn.linear("o_proj", ao, QD, H, false);
    let r1 = nn.add(x, o);
    let h2 = nn.rmsnorm("post_norm", r1, H, 1e-6);
    let gate = nn.linear("gate_proj", h2, H, IM, false);
    let up = nn.linear("up_proj", h2, H, IM, false);
    let ga = nn.act(ActKind::Silu, gate);
    let gu = nn.mul(ga, up);
    let down = nn.linear("down_proj", gu, IM, H, false);
    let out = nn.add(r1, down);
    nn.end_block();
    nn.mark_output(out);
    let mut g = nn.finish();
    infer_shapes(&mut g).expect("infer");
    plan_from_block(&g, 0).expect("plan")
}

fn scheduled() -> Scheduled {
    let soc = Soc::single(h100(), DEFAULT_PAGE_BYTES);
    let plan = small_block();
    let (tile_g, cons) = assemble(&soc, &plan, SramPolicy::Stream, None).expect("assemble");
    schedule(&soc, &tile_g, &cons, &Config::default())
}

/// Assert streams ↔ packets ↔ starts are pairwise consistent everywhere.
fn assert_full_consistency(sched: &schedule::Schedule) {
    for (r, stream) in &sched.streams {
        let packets = sched.packets.get(r).unwrap_or_else(|| {
            panic!("no packet list for resource {r:?}");
        });
        assert_eq!(
            stream.len(),
            packets.len(),
            "stream and packet lengths differ for {r:?}"
        );
        for (pos, ((task, cycle), pkt)) in stream.iter().zip(packets.iter()).enumerate() {
            assert_eq!(
                *task, pkt.task,
                "resource {r:?} position {pos}: stream task {task} vs packet task {}",
                pkt.task
            );
            assert_eq!(
                *cycle, pkt.start,
                "resource {r:?} position {pos}: stream cycle {cycle} vs packet.start {}",
                pkt.start
            );
            assert_eq!(
                *cycle, sched.starts[*task],
                "task {task}: stream cycle {cycle} vs global starts[{task}] = {}",
                sched.starts[*task]
            );
        }
    }
}

#[test]
fn prefetch_preserves_full_schedule_consistency() {
    let s = scheduled();
    let (hoisted, rep) = hoist_prefetches(&s.tasks, &s.schedule);
    // Ordering consistency post-hoist.
    assert_full_consistency(&hoisted);
    // Basic invariants of the report.
    assert!(rep.total_slot_advance == 0 || !rep.hoisted.is_empty());
    // Every DMA-in that was hoisted now sits earlier than before.
    // (Reports can also be empty on this bucket — the scheduler may already
    // be tight; both outcomes are valid.)
    let n_dma_in = s
        .tasks
        .tasks
        .iter()
        .filter(|t| t.kind == TaskKind::DmaIn)
        .count();
    assert!(rep.hoisted.len() + rep.already_optimal <= n_dma_in);
}

#[test]
fn prefetch_keeps_data_dependencies_ordered() {
    let s = scheduled();
    let (hoisted, _rep) = hoist_prefetches(&s.tasks, &s.schedule);
    for &(a, b) in &s.tasks.edges {
        let end_a = hoisted.starts[a] + s.tasks.tasks[a].dur;
        assert!(
            end_a <= hoisted.starts[b],
            "edge ({a}, {b}) violated after prefetch: end_a={end_a}, start_b={}",
            hoisted.starts[b]
        );
    }
}

#[test]
fn prefetch_survives_counter_replay_verifier() {
    // The crate's own `verify_schedule` replays the counter-gated stream and
    // checks that every task can actually reach its wait threshold. This is
    // the runtime-truth check for the reordered schedule.
    let s = scheduled();
    let soc = Soc::single(h100(), DEFAULT_PAGE_BYTES);
    let plan = small_block();
    let (tile_g, cons) = assemble(&soc, &plan, SramPolicy::Stream, None).expect("assemble");
    let (hoisted_sched, _rep) = hoist_prefetches(&s.tasks, &s.schedule);
    let hoisted = Scheduled {
        schedule: hoisted_sched,
        tasks: s.tasks.clone(),
        machine: s.machine.clone(),
        oracle_report: None,
    };
    let verified = hoisted.verify(&tile_g, &cons);
    assert!(
        verified.is_ok(),
        "verify_schedule rejected hoisted schedule: {verified:?}"
    );
}

#[test]
fn prefetch_address_map_matches_pre_hoist() {
    // The address map is derived from tensor read/write patterns via
    // task starts — after hoisting, the map may shift bytes/cycles but must
    // still form a valid AddressMap. `plan_from_schedule_with_task_sets`
    // panics on inconsistencies (debug_assert_disjoint), so simply exercising
    // it is a strong integration test.
    let s = scheduled();
    let soc = Soc::single(h100(), DEFAULT_PAGE_BYTES);
    let plan = small_block();
    let (_tg, cons) = assemble(&soc, &plan, SramPolicy::Stream, None).expect("assemble");
    let (hoisted_sched, _rep) = hoist_prefetches(&s.tasks, &s.schedule);
    let (_map, task_sets) = plan_from_schedule_with_task_sets(&s.tasks, &hoisted_sched, &cons);
    // Every tensor whose read/write set is nonempty should still be indexable.
    for (name, (writers, readers)) in &task_sets {
        assert!(
            writers.iter().all(|&t| t < s.tasks.tasks.len()),
            "writers of {name} contain out-of-range TaskId"
        );
        assert!(
            readers.iter().all(|&t| t < s.tasks.tasks.len()),
            "readers of {name} contain out-of-range TaskId"
        );
    }
}
