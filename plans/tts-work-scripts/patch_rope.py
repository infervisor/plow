p = '/root/plow/.claude/worktrees/tts-veena-chatterbox/crates/devgen/src/lib.rs'
s = open(p).read()
a = """                    d.t[5] = n.pos;
                    d.i[0] = t;
                    d.i[1] = heads;
                    d.i[2] = hd;
                    d.i[3] = 0;
                    d.i[4] = qk_skip;
                    d.f[0] = c.eps;
                },"""
b = """                    d.t[5] = n.pos;
                    d.i[0] = t;
                    d.i[1] = heads;
                    d.i[2] = hd;
                    d.i[3] = 0;
                    d.i[4] = qk_skip;
                    d.i[5] = rope_pair;
                    d.f[0] = c.eps;
                },"""
assert s.count(a) == 1
s = s.replace(a, b)
a = """                d.t[6] = n.kcs[l];
                d.i[0] = t;
                d.i[1] = kvh;
                d.i[2] = hd;
                d.i[3] = 0;
                d.i[4] = qk_skip;
                d.f[0] = c.eps;"""
b = """                d.t[6] = n.kcs[l];
                d.i[0] = t;
                d.i[1] = kvh;
                d.i[2] = hd;
                d.i[3] = 0;
                d.i[4] = qk_skip;
                d.i[5] = rope_pair;
                d.f[0] = c.eps;"""
assert s.count(a) == 1
s = s.replace(a, b)
a = """        let c_qn = if fuse_hnr {
            0 // no packet: the fold computes q's norm+rope in flash's staging"""
b = """        // Every model this dense emitter serves (Gemma, Llama, Qwen, Chatterbox T3) rotates
        // NeoX-style (rotate_half). HEADNORM_ROPE's legacy pairing is GPT-J interleaved at hd 64
        // (GLM/Kimi k_rope, emitted elsewhere), so hd 64 must force the half split.
        let rope_pair = if hd == 64 { packet::dev::ROPE_PAIR_HALF } else { 0 };
        let c_qn = if fuse_hnr {
            0 // no packet: the fold computes q's norm+rope in flash's staging"""
assert s.count(a) == 1
s = s.replace(a, b)
open(p, 'w').write(s)
print("ok")
