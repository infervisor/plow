//! Silero VAD v5 from the safetensors `scripts/asr/silero_export.py` writes to a
//! `vad.frame.v1` packet (contract: plow_asset::speech_contract).

use std::collections::BTreeMap;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 3 {
        return Err("usage: asr_silero_vad_compile MODEL_SAFETENSORS OUTPUT_PACKET".into());
    }
    let tensors = read_f32_safetensors(&std::fs::read(&args[1])?)?;
    let mut packets = devgen::vad::lower(1)?;
    packets.embed_weights(|name| tensors.get(name).cloned().ok_or_else(|| format!("{name} is missing")))?;
    let section = packets.pipeline_section()?;
    std::fs::write(&args[2], packets.model.to_blob_v6(&[section]))?;
    Ok(())
}

fn read_f32_safetensors(bytes: &[u8]) -> Result<BTreeMap<String, Vec<u8>>, String> {
    let length = bytes.get(..8).ok_or("truncated safetensors")?;
    let length = u64::from_le_bytes(length.try_into().unwrap()) as usize;
    let header = bytes.get(8..8 + length).ok_or("truncated safetensors header")?;
    let data = &bytes[8 + length..];
    let header: BTreeMap<String, serde_json::Value> = serde_json::from_slice(header).map_err(|e| e.to_string())?;
    let mut out = BTreeMap::new();
    for (name, entry) in header.into_iter().filter(|(name, _)| name != "__metadata__") {
        if entry["dtype"] != "F32" {
            return Err(format!("{name} is not F32"));
        }
        let offsets = entry["data_offsets"].as_array().ok_or("missing data_offsets")?;
        let range = |i: usize| offsets.get(i).and_then(serde_json::Value::as_u64).map(|v| v as usize);
        let (start, end) = range(0).zip(range(1)).ok_or("invalid data_offsets")?;
        out.insert(name, data.get(start..end).ok_or("tensor outside the file")?.to_vec());
    }
    Ok(out)
}
