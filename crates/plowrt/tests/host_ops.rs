//! §P Host token ops: a counter-gated SAMPLE packet runs on the HostExecutor the
//! instant its producer completes, writes the argmax token, and bumps its
//! successor — all inside the same counter-gated walk as the device packets.

use packet::{Body, Counter, Inst, Opcode, Program, ResourceKind, SLOT_NONE};
use plowrt::device::cpu::run_streams;
use plowrt::exec::counters::CounterPool;
use plowrt::exec::host::HostExecutor;

/// A tiny decode tail: a compute packet (the logits producer) increments
/// counter 0; a host SAMPLE packet waits on it and produces a token; its own
/// counter 1 unblocks a downstream consumer.
fn sample_program() -> Program {
    Program {
        insts: vec![
            // logits producer (stand-in for the final GEMM's output store)
            Inst {
                resource: ResourceKind::Sm,
                unit: 0,
                index: 0,
                body: Body::Host, // (compute body irrelevant here)
                wait: vec![],
                succ: vec![0],
            },
            // host SAMPLE — gated on the logits counter
            Inst {
                resource: ResourceKind::Host,
                unit: 0,
                index: 0,
                body: Body::Token {
                    in_slot: SLOT_NONE,
                    out_slot: SLOT_NONE,
                    kind: Opcode::TOKEN_SAMPLE_GREEDY,
                    vocab: 8,
                    arg: 0,
                },
                wait: vec![0],
                succ: vec![1],
            },
            // downstream consumer of the sampled token
            Inst {
                resource: ResourceKind::Sm,
                unit: 0,
                index: 1,
                body: Body::Host,
                wait: vec![1],
                succ: vec![],
            },
        ],
        counters: vec![
            Counter { id: 0, threshold: 1, scope: 1, _pad: [0; 3] },
            Counter { id: 1, threshold: 1, scope: 2, _pad: [0; 3] },
        ],
        bucket_id: 0,
        plan_gen: 0,
        flags: 0,
    }
}

#[test]
fn gated_sample_produces_argmax_token() {
    let prog = sample_program();
    let pool = CounterPool::from_counters(&prog.counters);

    let mut host = HostExecutor::new();
    // argmax of this row is index 5.
    host.set_logits(vec![0.1, 0.2, 0.0, 0.3, 0.1, 0.9, 0.4, 0.2]);
    // greedy
    host.params.temperature = 0.0;

    let stats = run_streams(&prog, &pool, &mut host);

    assert!(stats.completed, "the whole schedule (incl. host packet) completes");
    assert_eq!(host.ran, 1, "the SAMPLE host op ran exactly once");
    assert_eq!(host.tokens, vec![5], "greedy argmax token");
    // The sample packet bumped its successor, unblocking the consumer.
    assert!(pool.satisfied(1));
}

#[test]
fn tokenize_packet_encodes_input_text() {
    // A head TOKENIZE packet gates a downstream compute; the HostExecutor
    // encodes its input text into token ids when the packet fires.
    let prog = Program {
        insts: vec![
            Inst {
                resource: ResourceKind::Host,
                unit: 0,
                index: 0,
                body: Body::Token {
                    in_slot: SLOT_NONE,
                    out_slot: SLOT_NONE,
                    kind: Opcode::TOKEN_TOKENIZE,
                    vocab: 0,
                    arg: 0,
                },
                wait: vec![],
                succ: vec![0],
            },
            Inst {
                resource: ResourceKind::Sm,
                unit: 0,
                index: 0,
                body: Body::Host,
                wait: vec![0],
                succ: vec![],
            },
        ],
        counters: vec![Counter { id: 0, threshold: 1, scope: 2, _pad: [0; 3] }],
        bucket_id: 0,
        plan_gen: 0,
        flags: 0,
    };
    let pool = CounterPool::from_counters(&prog.counters);
    let mut host = HostExecutor::new();
    host.input_text = Some("hi".to_string());

    let stats = run_streams(&prog, &pool, &mut host);
    assert!(stats.completed);
    assert_eq!(host.token_ids, vec![u32::from(b'h'), u32::from(b'i')]);
}

#[test]
fn sample_waits_for_its_producer() {
    // If the producer never increments counter 0, the host SAMPLE can't fire and
    // the schedule deadlocks — proving the host packet is genuinely gated.
    let mut prog = sample_program();
    prog.insts[0].succ.clear(); // drop the logits producer's increment

    let pool = CounterPool::from_counters(&prog.counters);
    let mut host = HostExecutor::new();
    host.set_logits(vec![1.0, 2.0, 3.0]);

    let stats = run_streams(&prog, &pool, &mut host);
    assert!(!stats.completed, "gated SAMPLE cannot fire without its producer");
    assert_eq!(host.ran, 0, "SAMPLE did not run");
}
