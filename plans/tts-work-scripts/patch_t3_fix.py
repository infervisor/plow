p = '/root/plow/.claude/worktrees/tts-veena-chatterbox/crates/plowrt/src/tts/t3.rs'
s = open(p).read()
s = s.replace("Ok(bytemuck::pod_collect_to_vec(t.data()))", "Ok(le_f32s(t.data()))")
s = s.replace("let rows: Vec<f32> = bytemuck::pod_collect_to_vec(&raw);", "let rows: Vec<f32> = le_f32s(&raw);")
s = s.replace("pub fn load(assets: &Path, device: usize) -> Result<Self> {", "pub fn load(assets: &Path, device: u8) -> Result<Self> {")
s = s.replace("""/// Host tables for prefill rows""", """/// Little-endian f32s from bytes of any alignment (mmap'd safetensors data need not be aligned).
fn le_f32s(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

/// Host tables for prefill rows""")
open(p, 'w').write(s)
print("ok")
