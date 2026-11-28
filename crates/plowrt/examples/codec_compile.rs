//! A codec-LM codec export (`fish_dac`, `qwen3_tts_12hz` or `mimi_dac` `config.json` +
//! `model.safetensors`) to a standalone `codec.pkt`, for `codec_check` parity runs before the codec
//! rides an LM emit: `codec_compile CODEC_DIR GPU N_CU MAX_ROWS OUT.pkt` (MAX_ROWS 0 = every
//! capacity). The emit environment applies as in `plowc` (`PLOW_EMIT_SNAC_SPLIT_BF16`,
//! `PLOW_EMIT_CODEC_CONV_ARMS`), so the same environment rebuilds an emitted `codec.pkt`.
//! `--block L --batch B --frames N` emits one stateful Mimi transformer layer as
//! `codec.layer.block` / `forward.0`, with raw stream controls and state available by tensor name.

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 6
        && !(args.len() == 12
            && args[6] == "--block"
            && args[8] == "--batch"
            && args[10] == "--frames")
    {
        return Err("usage: codec_compile CODEC_DIR GPU N_CU MAX_ROWS OUT.pkt [--block L --batch B --frames N]".into());
    }
    let n_cu: u32 = args[3].parse()?;
    let max_rows: u32 = args[4].parse()?;
    let (model, section) = if args.len() == 12 {
        devgen::fish_codec::lower_mimi_layer_block(
            std::path::Path::new(&args[1]),
            n_cu,
            packet::devbuild::gpu_fingerprint(&args[2]),
            devgen::emit_config::active().snac_split_bf16,
            (max_rows > 0).then_some(max_rows),
            args[7].parse()?,
            args[9].parse()?,
            args[11].parse()?,
        )?
    } else {
        devgen::fish_codec::lower_fish_dac(
            std::path::Path::new(&args[1]),
            n_cu,
            packet::devbuild::gpu_fingerprint(&args[2]),
            devgen::emit_config::active().snac_split_bf16,
            (max_rows > 0).then_some(max_rows),
        )?
    };
    devgen::write_sidecar_packet(std::path::Path::new(&args[5]), &model, &[section]);
    Ok(())
}
