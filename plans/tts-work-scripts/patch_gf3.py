W = '/root/plow/.claude/worktrees/tts-veena-chatterbox/'


def patch(path, pairs):
    s = open(W + path).read()
    for a, b in pairs:
        assert s.count(a) == 1, (path, s.count(a), a[:70])
        s = s.replace(a, b)
    open(W + path, 'w').write(s)


patch('crates/devgen/src/knob_spec.rs', [
    ('    KnobSpec::new("def.PLOW_NV_FA_WPR128", None, Layer::ObjectDefine, Domain::Str, UNSET, OPT_IN),\n', ''),
])
# Emitter: an odd group fuses whole (GF = gqa) when it fits a work item, instead of GF = 1.
patch('crates/devgen/src/lib.rs', [
    ("""            // An odd group (Llama-3.2-3B / Veena: 24 heads over 8 KV heads = 3) takes no fusion;
            // the manifest pairs the object's hd128 arm at PLOW_NV_FA_GF=1 for the same shapes.
            let g = fa_gf_full().min(gqa);
            if gqa % g == 0 {
                g
            } else {
                1
            }""",
     """            // A group the configured GF does not divide (Llama-3.2-3B / Veena: 24 heads over 8
            // KV heads = 3) fuses WHOLE: one work item reads each KV row once for all its heads.
            // The manifest pairs the object's hd128 arm at the same PLOW_NV_FA_GF.
            let g = fa_gf_full().min(gqa);
            if gqa % g == 0 {
                g
            } else {
                odd_group_gf(gqa)
            }"""),
    ("""pub(crate) fn attention_decode_ns(""", """/// GF for a GQA group the power-of-two GFs do not divide: the whole group when it fits one work
/// item (the decode kernel allows GF <= warps), else no fusion.
pub(crate) fn odd_group_gf(gqa: u32) -> u32 {
    if gqa <= 8 {
        gqa
    } else {
        1
    }
}

pub(crate) fn attention_decode_ns("""),
])
patch('crates/devgen/src/manifest.rs', [
    ("""        req.push(format!("PLOW_NV_FA_GF={}", if s.gqa % 2 == 0 { 2 } else { 1 }));""",
     """        let gf = if s.gqa % 2 == 0 { 2 } else { crate::odd_group_gf(s.gqa) };
        req.push(format!("PLOW_NV_FA_GF={gf}"));"""),
])
print("ok")
