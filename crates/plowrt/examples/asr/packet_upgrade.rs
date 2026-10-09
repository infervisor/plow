//! Upgrade a contract-0 ASR/VAD packet to the current speech contract
//! (`plow_asset::speech_contract`, `docs/runtime/asr-packet-contract.md`) by rewriting its
//! metadata only: programs, tensors, weights and objects stay byte for byte, so the packet's
//! numerics and object pairing are unchanged. A fresh emit writes the same metadata.
//!
//!   asr_packet_upgrade vad IN.pkt OUT.pkt              Silero `vad.silero.v1` -> `vad.frame.v1`
//!   asr_packet_upgrade rnnt IN.pkt MODEL.gguf OUT.pkt  vocabulary/detokenizer/language from the GGUF
//!   asr_packet_upgrade audio-lm IN.pkt OUT.pkt         Qwen3-ASR decoder (`model.pkt`) host policy

mod nemotron_packet_support;

use plow_asset::packet_pipeline::{PacketPipelines, SECTION};
use plow_asset::speech_contract as contract;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let usage = "usage: asr_packet_upgrade vad IN OUT | rnnt IN GGUF OUT | audio-lm IN OUT";
    let (kind, input, output) = match args.get(1).map(String::as_str) {
        Some("vad" | "audio-lm") if args.len() == 4 => (args[1].as_str(), &args[2], &args[3]),
        Some("rnnt") if args.len() == 5 => ("rnnt", &args[2], &args[4]),
        _ => return Err(usage.into()),
    };
    let blob = std::fs::read(input)?;
    let parsed = plowrt::asset::devblob::DevBlob::parse(&blob)?;
    let raw = parsed.reserved_metadata(&blob, SECTION)?.ok_or("packet has no pipeline section")?;
    let mut pipelines: PacketPipelines = serde_json::from_slice(raw)?;
    let mut sections = Vec::new();
    let mut upgraded = 0;
    for p in &mut pipelines.pipelines {
        if p.parameters.contains_key(contract::CONTRACT) {
            continue;
        }
        match (kind, p.driver.as_str()) {
            ("vad", contract::VAD_DRIVER_V0) => {
                p.driver = contract::VAD_DRIVER.into();
                p.parameters.insert(contract::CONTRACT.into(), contract::VAD_CONTRACT);
                p.parameters.insert("executor".into(), contract::EXECUTOR_HOST);
                p.parameters.insert("state_banks".into(), 2);
                devgen::vad::POLICY.to_parameters(&mut p.parameters);
            }
            ("rnnt", "rnnt.greedy.v1") => {
                let gguf = plowrt::asset::gguf::GgufFile::open(std::path::Path::new(&args[3]))?;
                let prompted = p.parameters.get("prompt_count").copied().unwrap_or(0) > 0;
                let index = p.parameters.get("prompt_index").copied().unwrap_or(0) as usize;
                let output = nemotron_packet_support::token_output(&gguf, prompted.then_some(index))?;
                output.apply(&mut p.parameters, &mut p.strings);
                sections.push(packet::devbuild::SectionData {
                    kind: packet::devbuild::SECT_METADATA,
                    name: contract::VOCABULARY_SECTION.into(),
                    data: output.vocabulary_section(),
                });
            }
            ("audio-lm", "causal.v1") if p.parameters.get("overlay_rows").is_some_and(|&r| r > 0) => {
                devgen::asr::qwen::AUDIO_LM_POLICY.to_parameters(&mut p.parameters);
            }
            _ => continue,
        }
        upgraded += 1;
    }
    if upgraded == 0 {
        return Err(format!("{input}: no contract-0 {kind} pipeline to upgrade").into());
    }
    pipelines.validate(parsed.progs.len(), |name| parsed.tensors.iter().find(|t| t.name == name).map(|t| t.bytes as u64))?;
    sections.push(packet::devbuild::SectionData {
        kind: packet::devbuild::SECT_METADATA,
        name: SECTION.into(),
        data: serde_json::to_vec(&pipelines)?,
    });
    std::fs::write(output, packet::devbuild::replace_metadata_sections(&blob, &sections)?)?;
    println!("{}", serde_json::json!({"upgraded": upgraded, "kind": kind, "output": output}));
    Ok(())
}
