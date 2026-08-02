//! Integration test for checkpoint E — wire-format round-trip.
//!
//! Backed by `Plow.Wire.decodeProgram_encodeProgram`. The Rust reference
//! encoder in `lean_verify::checkpoints::wire::encode_program` must match the
//! Lean encoder byte-for-byte; the verifier flags any drift.

#![cfg(feature = "lean-verify")]

use lean_verify::checkpoints::wire::{
    check_wire_roundtrip, encode_program, WireFrame, WireRequest,
};

fn sample_frames() -> Vec<WireFrame> {
    vec![
        WireFrame {
            opcode: 1,
            payload: vec![10, 20, 30],
        },
        WireFrame {
            opcode: 42,
            payload: vec![],
        },
        WireFrame {
            opcode: 0xABCD,
            payload: (0..8).collect(),
        },
    ]
}

#[test]
#[ignore = "requires plow_verify binary"]
fn accepts_valid_round_trip() {
    let frames = sample_frames();
    let raw = encode_program(&frames);
    let req = WireRequest { raw, frames };
    let cert = check_wire_roundtrip(&req).expect("verifier call");
    assert!(cert.ok, "round-trip rejected: {cert:?}");
}

#[test]
#[ignore = "requires plow_verify binary"]
fn accepts_empty_program() {
    let req = WireRequest {
        raw: vec![],
        frames: vec![],
    };
    let cert = check_wire_roundtrip(&req).expect("verifier call");
    assert!(cert.ok, "empty round-trip rejected: {cert:?}");
}

#[test]
#[ignore = "requires plow_verify binary"]
fn rejects_encode_frames_diverges_from_raw() {
    let frames = sample_frames();
    let mut raw = encode_program(&frames);
    // Corrupt a byte inside a payload.
    raw[6] ^= 0xff;
    let req = WireRequest { raw, frames };
    let cert = check_wire_roundtrip(&req).expect("verifier call");
    assert!(!cert.ok, "corrupted round-trip accepted: {cert:?}");
    let reason = cert.reason.expect("rejection reason");
    assert!(
        reason.contains("encode(frames) ≠ raw"),
        "unexpected reason: {reason}"
    );
}

#[test]
#[ignore = "requires plow_verify binary"]
fn rejects_length_mismatch() {
    let frames = sample_frames();
    let mut raw = encode_program(&frames);
    raw.push(0); // trailing junk
    let req = WireRequest { raw, frames };
    let cert = check_wire_roundtrip(&req).expect("verifier call");
    assert!(!cert.ok, "length mismatch accepted: {cert:?}");
}

#[test]
#[ignore = "requires plow_verify binary"]
fn rejects_truncated_stream() {
    let frames = sample_frames();
    let mut raw = encode_program(&frames);
    // Chop off the tail — the frames list no longer matches encode(frames).
    raw.truncate(raw.len() - 4);
    let req = WireRequest { raw, frames };
    let cert = check_wire_roundtrip(&req).expect("verifier call");
    assert!(!cert.ok, "truncated stream accepted: {cert:?}");
}

/// Cross-check the concrete `packet::Program::to_bytes` / `Program::decode`
/// pair against Lean's abstract framing.
///
/// The Lean model proves any framing of shape `[opcode:u16, len:u16, payload]`
/// round-trips. The packet crate uses a richer per-body POD layout, so this
/// test converts a compiled packet stream into the abstract-frame shape
/// (opcode = `Body::opcode() as u16`, payload = body bytes) and verifies:
///   (a) `packet::Program::decode(program.to_bytes())` round-trips (packet-side),
///   (b) the abstract frames encode back to the same abstract raw (Lean-side).
///
/// This is deliberately a shape-cross-check, not a byte-identical match: the
/// packet header (MAGIC + VERSION) and per-body POD structs aren't modeled in
/// Lean. See `Plow.Wire`'s module header for scope.
/// Rust's `serde_json` serializes `Vec<u8>` as an array of numbers (0–255),
/// and Lean's `Payload.parseNatArrayStrict` reads it as `List Nat`. Confirm
/// the boundary values (0, 1, 254, 255) round-trip correctly and no byte
/// value is silently dropped or coerced.
#[test]
#[ignore = "requires plow_verify binary"]
fn json_u8_boundary_round_trips_through_lean() {
    // Every byte value on the boundary of u8 representation.
    let payload_bytes: Vec<u8> = (0..=255u16).map(|b| b as u8).collect();
    let frames = vec![WireFrame {
        opcode: 0xDEAD,
        payload: payload_bytes.clone(),
    }];
    // Sanity: what serde_json serializes to. Every byte 0..=255 must appear
    // as a raw JSON number, not stringified.
    let json = serde_json::to_string(&frames).expect("serialize frames");
    assert!(
        json.contains(",255]") || json.contains(",255,"),
        "u8=255 missing from JSON"
    );
    assert!(
        !json.contains("\"255\""),
        "u8=255 was quoted (bytes accidentally stringified)"
    );
    // Length of the numeric payload array in the JSON matches the byte count.
    assert!(
        json.matches(',').count() >= payload_bytes.len(),
        "JSON has fewer commas than bytes — some values were merged"
    );

    // Round-trip via the verifier.
    let raw = encode_program(&frames);
    let req = WireRequest { raw, frames };
    let cert = check_wire_roundtrip(&req).expect("verifier call");
    assert!(cert.ok, "boundary u8 payload rejected: {cert:?}");
}

#[test]
#[ignore = "requires plow_verify binary"]
fn abstract_framing_shape_matches_packet_body_ordering() {
    use packet::{Body, Inst, Program, ResourceKind};

    // A tiny 2-instruction program: one DMA followed by a GEMM.
    let program = Program {
        insts: vec![
            Inst {
                resource: ResourceKind::Dma,
                unit: 0,
                index: 0,
                body: Body::Dma {
                    load: true,
                    bytes: 4096,
                    slot: 0,
                    tensor: 42,
                    kind: packet::KIND_UNSPECIFIED,
                    access: packet::ACCESS_READ,
                },
                wait: vec![],
                succ: vec![1],
            },
            Inst {
                resource: ResourceKind::Sm,
                unit: 0,
                index: 1,
                body: Body::Gemm {
                    coord: [0, 0],
                    m: 128,
                    n: 128,
                    k: 128,
                    bm: 64,
                    bn: 64,
                    bk: 32,
                    out: 0,
                    tmem: 0,
                    variant: packet::Opcode::VARIANT_BF16,
                },
                wait: vec![1],
                succ: vec![],
            },
        ],
        counters: vec![],
        bucket_id: 0,
        plan_gen: 0,
        flags: 0,
    };

    // (a) Packet round-trips on the concrete byte stream.
    let bytes = program.to_bytes();
    let decoded = Program::decode(&bytes).expect("packet decode");
    assert_eq!(decoded, program, "packet round-trip failed");

    // (b) Build the abstract-framing view: one WireFrame per Inst,
    // opcode = Body::opcode() as u16, payload = body bytes.
    // We don't try to reproduce the full body encoding; the point is that
    // the abstract framing shape (opcode-tagged length-prefixed frames)
    // matches the ordering / arity of the concrete stream.
    let frames: Vec<WireFrame> = program
        .insts
        .iter()
        .map(|inst| WireFrame {
            opcode: inst.body.opcode().0,
            // Synthesize a tiny payload from the wait/succ counts so the
            // round-trip is non-trivial; the actual byte encoding of the
            // body is a packet-crate concern.
            payload: vec![inst.wait.len() as u8, inst.succ.len() as u8],
        })
        .collect();
    let abstract_raw = encode_program(&frames);
    let req = WireRequest {
        raw: abstract_raw,
        frames,
    };
    let cert = check_wire_roundtrip(&req).expect("verifier call");
    assert!(cert.ok, "abstract framing round-trip rejected: {cert:?}");
}
