//! Checkpoint E — wire-format round-trip.
//!
//! Backed by `Plow.Wire.decodeProgram_encodeProgram`. Caller submits a raw
//! byte stream + its decoded view; verifier confirms both directions match.

use serde::{Deserialize, Serialize};

use crate::{call, Certificate, VerifyError};

/// One decoded frame — opcode tag + payload bytes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WireFrame {
    pub opcode: u16,
    pub payload: Vec<u8>,
}

/// Full payload for checkpoint E.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WireRequest {
    /// Raw serialized byte stream.
    pub raw: Vec<u8>,
    /// Caller's decoded view of the stream.
    pub frames: Vec<WireFrame>,
}

/// Verify the wire round-trip.
pub fn check_wire_roundtrip(req: &WireRequest) -> Result<Certificate, VerifyError> {
    let payload = serde_json::to_value(req).map_err(VerifyError::SerializeRequest)?;
    call("E", payload)
}

/// Simple reference encoder matching `Plow.Wire.encodeFrame`: opcode (u16 BE)
/// + payload_len (u16 BE) + payload bytes. Callers can build their own — this
/// is convenient for tests.
pub fn encode_frame(frame: &WireFrame) -> Vec<u8> {
    let mut out = Vec::with_capacity(4 + frame.payload.len());
    out.push((frame.opcode >> 8) as u8);
    out.push((frame.opcode & 0xff) as u8);
    let len = frame.payload.len() as u16;
    out.push((len >> 8) as u8);
    out.push((len & 0xff) as u8);
    out.extend_from_slice(&frame.payload);
    out
}

/// Encode a full program using the reference encoder.
pub fn encode_program(frames: &[WireFrame]) -> Vec<u8> {
    let mut out = Vec::new();
    for f in frames {
        out.extend(encode_frame(f));
    }
    out
}
