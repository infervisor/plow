p = '/root/plow/.claude/worktrees/tts-veena-chatterbox/crates/plowrt/src/tts/chatterbox.rs'
s = open(p).read()
start = s.index('        std::thread::Builder::new()\n            .name("plow-tts-s3gen".into())')
end = s.index('        std::thread::Builder::new()\n            .name("plow-tts-t3".into())')
s3gen_block = s[start:end]
s = s[:start] + s[end:]
anchor = '        ready_rx.recv().map_err(|e| RuntimeError::Device(e.to_string()))??;\n        Ok(ChatterboxWorker'
assert s.count(anchor) == 1
s = s.replace(anchor, '''        ready_rx.recv().map_err(|e| RuntimeError::Device(e.to_string()))??;
        // After the engine: its backend loads the real driver by path. The stage's static CUDA
        // runtime then resolves `libcuda.so.1` to that library instead of searching (which can
        // land on a toolkit stub: "driver version is insufficient").
''' + s3gen_block + '''        Ok(ChatterboxWorker''')
open(p, 'w').write(s)
print("ok")
