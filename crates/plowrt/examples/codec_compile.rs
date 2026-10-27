//! A codec-LM codec export (`fish_dac`, `qwen3_tts_12hz` or `mimi_dac` `config.json` +
//! `model.safetensors`) to a standalone `codec.pkt`, for `codec_check` parity runs before the codec
//! rides an LM emit: `codec_compile CODEC_DIR GPU N_CU MAX_ROWS OUT.pkt` (MAX_ROWS 0 = every
//! capacity).

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 6 {
        return Err("usage: codec_compile CODEC_DIR GPU N_CU MAX_ROWS OUT.pkt".into());
    }
    let n_cu: u32 = args[3].parse()?;
    let max_rows: u32 = args[4].parse()?;
    let (model, section) = devgen::fish_codec::lower_fish_dac(
        std::path::Path::new(&args[1]),
        n_cu,
        packet::devbuild::gpu_fingerprint(&args[2]),
        false,
        (max_rows > 0).then_some(max_rows),
    )?;
    devgen::write_sidecar_packet(std::path::Path::new(&args[5]), &model, &[section]);
    Ok(())
}
