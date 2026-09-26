W = '/root/plow/.claude/worktrees/tts-veena-chatterbox/'


def patch(path, pairs):
    s = open(W + path).read()
    for a, b in pairs:
        assert s.count(a) == 1, (path, s.count(a), a[:70])
        s = s.replace(a, b)
    open(W + path, 'w').write(s)


patch('runtime/CMakeLists.txt', [
    ("""        list(APPEND _cubin_outs "${_snac_out}")
    endif()
""", """        list(APPEND _cubin_outs "${_snac_out}")
    endif()

    # Chatterbox speech packets (`tts.t3_cfg.v1`): the native S3Gen stage (tokens -> 24 kHz PCM).
    option(PLOW_TTS_S3GEN "Also build codec/libplow_s3gen.so for a Chatterbox T3 speech packet" OFF)
    if(PLOW_TTS_S3GEN AND PLOW_SM90A_CUBIN)
        set(_s3gen_src "${CMAKE_CURRENT_SOURCE_DIR}/nvidia/s3gen/s3gen.cu")
        set(_s3gen_out "${PLOW_CUBIN_DIR}/codec/libplow_s3gen.so")
        add_custom_command(OUTPUT "${_s3gen_out}"
            COMMAND env PLOW_NVCC=${PLOW_CUBIN_NVCC}
                    bash "${CMAKE_CURRENT_SOURCE_DIR}/nvidia/s3gen/build.sh" "${_s3gen_out}"
            DEPENDS "${_s3gen_src}" "${CMAKE_CURRENT_SOURCE_DIR}/nvidia/s3gen/build.sh"
            COMMENT "nvcc codec/libplow_s3gen.so"
            VERBATIM)
        list(APPEND _cubin_outs "${_s3gen_out}")
    endif()
"""),
])
patch('crates/plowc/src/main.rs', [
    ("build_cubin_from_manifest(&pkt, &cli.arch, cli.segmented, cli.emit_cfg.tts_profile.is_some())?;",
     """let t3 = cli.hf_dir.as_deref().is_some_and(|dir| {
            std::fs::read(dir.join("config.json"))
                .ok()
                .and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok())
                .is_some_and(|v| v.get("chatterbox_t3").is_some())
        });
        build_cubin_from_manifest(&pkt, &cli.arch, cli.segmented, cli.emit_cfg.tts_profile.is_some(), t3)?;"""),
    ("""    speech: bool,
) -> Result<(), Box<dyn std::error::Error>> {""", """    speech: bool,
    t3: bool,
) -> Result<(), Box<dyn std::error::Error>> {"""),
    ("""    args.push(format!("-DPLOW_TTS_SNAC={}", if speech { "ON" } else { "OFF" }));""",
     """    args.push(format!("-DPLOW_TTS_SNAC={}", if speech { "ON" } else { "OFF" }));
    args.push(format!("-DPLOW_TTS_S3GEN={}", if t3 { "ON" } else { "OFF" }));"""),
])
patch('crates/devgen/src/knob_spec.rs', [
    ('    KnobSpec::new("def.PLOW_TTS_SNAC", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),\n',
     '    KnobSpec::new("def.PLOW_TTS_SNAC", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),\n'
     '    KnobSpec::new("def.PLOW_TTS_S3GEN", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),\n'),
])
print("ok")
