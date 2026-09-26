p = '/root/plow/.claude/worktrees/tts-veena-chatterbox/crates/plowrt/src/exec/gpu.rs'
s = open(p).read()


def rep(a, b, n=1):
    global s
    assert s.count(a) == n, (s.count(a), a[:80])
    s = s.replace(a, b)


rep("""    /// `[batch][vocab]` f32 softmax-weight scratch the kernel reuses each pass.
    d_escratch: DeviceMem,""", """    /// `[batch][vocab]` f32 softmax-weight scratch the kernel reuses each pass.
    d_escratch: DeviceMem,
    /// Block width the object was built for (`plow_sample_threads`; absent on older objects => 256).
    threads: u32,""")
rep("""        let f = be.get_function(&module, &kname)?;
        let slab = (4 + SAMPLE_RNG_STEPS) * batch * 4;""", """        let f = be.get_function(&module, &kname)?;
        let threads = be.module_global_u32(&module, "plow_sample_threads")?.unwrap_or(256);
        let slab = (4 + SAMPLE_RNG_STEPS) * batch * 4;""")
rep("""            d_escratch,
            batch,
        }))""", """            d_escratch,
            threads,
            batch,
        }))""")
rep("""            let (sf, sdp, ses) = (smp.f, smp.d_params.base, smp.d_escratch.base);""",
    """            let (sf, sdp, ses, sthreads) = (smp.f, smp.d_params.base, smp.d_escratch.base, smp.threads);""")
rep("""                .launch_kernel(sf, bsz as u32, 256, 0, &mut a, Some(&self.stream))?;
        }

        // Token readback""", """                .launch_kernel(sf, bsz as u32, sthreads, 0, &mut a, Some(&self.stream))?;
        }

        // Token readback""")
rep("""            Some((smp.f, smp.d_params.base, smp.d_escratch.base))""",
    """            Some((smp.f, smp.d_params.base, smp.d_escratch.base, smp.threads))""")
rep("""            if let Some((sf, sdp, ses)) = sampler_args {""", """            if let Some((sf, sdp, ses, sthreads)) = sampler_args {""")
rep("""                    .launch_kernel(sf, bsz as u32, 256, 0, &mut a, Some(&self.stream))?;""",
    """                    .launch_kernel(sf, bsz as u32, sthreads, 0, &mut a, Some(&self.stream))?;""")
open(p, 'w').write(s)
print("ok")
