W = '/root/plow/.claude/worktrees/tts-veena-chatterbox/'


def patch(path, pairs, count=1):
    s = open(W + path).read()
    for a, b in pairs:
        n = s.count(a)
        assert n == (count if isinstance(count, int) else count[a]), (path, n, a[:70])
        s = s.replace(a, b)
    open(W + path, 'w').write(s)


# Cfg field + constructor defaults
s = open(W + 'crates/devgen/src/config.rs').read()
s = s.replace("""    pub(crate) encoder_overlay_rows: u32,
""", """    pub(crate) encoder_overlay_rows: u32,
    // Learned speech-position rows added to the DECODE token embedding (Chatterbox T3:
    // speech_emb[tok] + speech_pos_emb[pos - in.pos_base]). Zero means a plain Embed.
    pub(crate) speech_pos_rows: u32,
""", 1)
assert s.count("        encoder_overlay_rows: 0,\n") == 5
s = s.replace("        encoder_overlay_rows: 0,\n", "        encoder_overlay_rows: 0,\n        speech_pos_rows: 0,\n")
# chatterbox_t3: a Llama body whose prefill rows are host embeddings (overlay) and whose decode
# embedding carries learned positions. The prep script writes the block into config.json.
old = """    if arch == Arch::ModernBert {
        return cfg_modernbert(&v);
    }
    cfg_llama_qwen(&v, arch)
}"""
new = """    if arch == Arch::ModernBert {
        return cfg_modernbert(&v);
    }
    let mut c = cfg_llama_qwen(&v, arch);
    if let Some(t3) = v.get("chatterbox_t3") {
        c.encoder_overlay_rows = t3["overlay_rows"].as_u64().expect("chatterbox_t3.overlay_rows") as u32;
        c.speech_pos_rows = t3["speech_pos_rows"].as_u64().expect("chatterbox_t3.speech_pos_rows") as u32;
        assert!(c.encoder_overlay_rows > 0 && c.speech_pos_rows > 0);
    }
    c
}"""
assert s.count(old) == 1
s = s.replace(old, new)
open(W + 'crates/devgen/src/config.rs', 'w').write(s)

patch('crates/devgen/src/lib.rs', [
    ("""    encoder_overlay_index: u32,
    pos: u32,""", """    encoder_overlay_index: u32,
    // Chatterbox T3 decode: per-slot speech start and the learned speech-position table.
    pos_base: u32,
    speech_pos: u32,
    pos: u32,"""),
    ("""        pos: b.tensor("in.pos", ctx as u64 * I32),
        // BATCH>1 (serving pending #4): one KV length per sequence. dbatch==1 => I32, identical.""",
     """        pos_base: if c.speech_pos_rows > 0 {
            b.tensor("in.pos_base", dbatch as u64 * I32)
        } else {
            TENSOR_NONE
        },
        speech_pos: if c.speech_pos_rows > 0 {
            b.tensor(
                &format!("{}speech_pos_emb.weight", c.prefix),
                u64::from(c.speech_pos_rows) * u64::from(c.hidden) * BF16,
            )
        } else {
            TENSOR_NONE
        },
        pos: b.tensor("in.pos", ctx as u64 * I32),
        // BATCH>1 (serving pending #4): one KV length per sequence. dbatch==1 => I32, identical."""),
    ("""    } else {
        b.emit(DevOp::Embed, rows.clone(), &[], |d| {
            d.t[0] = n.x;
            d.t[1] = n.emb;
            d.t[2] = n.ids;
            d.i[0] = t;
            d.i[1] = c.hidden;
            d.f[0] = escale;
        })
    };""", """    } else if decode && c.speech_pos_rows > 0 {
        b.emit(DevOp::EmbedPosBf16, rows.clone(), &[], |d| {
            d.t[..6].copy_from_slice(&[n.x, n.emb, n.ids, n.speech_pos, n.pos, n.pos_base]);
            d.i[..4].copy_from_slice(&[t, c.hidden, c.vocab, c.speech_pos_rows]);
        })
    } else {
        b.emit(DevOp::Embed, rows.clone(), &[], |d| {
            d.t[0] = n.x;
            d.t[1] = n.emb;
            d.t[2] = n.ids;
            d.i[0] = t;
            d.i[1] = c.hidden;
            d.f[0] = escale;
        })
    };"""),
])
print("ok")
