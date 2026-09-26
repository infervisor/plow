p='/root/plow/.claude/worktrees/tts-veena-chatterbox/crates/devgen/src/lib.rs'
s=open(p).read()
a='''    check_cpu_or_metal_opcode_coverage(
        &m,
        arch == "metal3" || (arch.is_empty() && gpu.is_empty()),
    );'''
b='''    check_cpu_or_metal_opcode_coverage(
        &m,
        arch == "metal3" || (arch.is_empty() && gpu.is_empty()),
        arch.starts_with("sm_"),
    );'''
assert s.count(a)==1; s=s.replace(a,b)
a='''fn check_cpu_or_metal_opcode_coverage(m: &Model, supported: bool) {
    if supported {
        return;
    }'''
b='''fn check_cpu_or_metal_opcode_coverage(m: &Model, supported: bool, cuda: bool) {
    if supported {
        return;
    }
    // The CUDA interpreter carries the embedding-overlay handoff (op 179) too.
    let cuda_ok = |op: DevOp| cuda && op == DevOp::EmbedOverlayBf16;'''
assert s.count(a)==1; s=s.replace(a,b)
a='''        .filter(|op| {
            m.progs'''
b='''        .filter(|op| !cuda_ok(**op))
        .filter(|op| {
            m.progs'''
assert s.count(a)==1; s=s.replace(a,b)
open(p,'w').write(s); print("ok")
