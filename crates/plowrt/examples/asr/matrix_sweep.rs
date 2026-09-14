#[cfg(all(feature = "metal", target_os = "macos"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use objc2_foundation::NSString;
    use objc2_metal::*;
    use serde_json::json;
    use std::ptr::NonNull;

    let args: Vec<_> = std::env::args().collect();
    if args.len() > 3 || args.get(2).is_some_and(|s| s != "--reverse") {
        return Err("usage: asr_matrix_sweep [REPEATS [--reverse]]".into());
    }
    let repeats = args
        .get(1)
        .map(|s| s.parse::<usize>())
        .transpose()?
        .unwrap_or(10);
    if repeats < 2 {
        return Err("at least two repeats required".into());
    }
    let device = MTLCreateSystemDefaultDevice().ok_or("Metal device")?;
    let options = MTLCompileOptions::new();
    options.setMathMode(MTLMathMode::Safe);
    options.setLanguageVersion(MTLLanguageVersion::Version3_2);
    let library = device.newLibraryWithSource_options_error(
        &NSString::from_str(include_str!("../../../../runtime/apple/asr.metal")),
        Some(&options),
    )?;
    let mut pipelines = Vec::new();
    for name in ["asr_linear_direct", "asr_linear_tile64"] {
        let function = library
            .newFunctionWithName(&NSString::from_str(name))
            .ok_or("kernel")?;
        let pipeline = device.newComputePipelineStateWithFunction_error(&function)?;
        if pipeline.threadExecutionWidth() != 32
            || pipeline.maxTotalThreadsPerThreadgroup() < 256
            || pipeline.staticThreadgroupMemoryLength() > device.maxThreadgroupMemoryLength()
        {
            return Err(format!("unsupported pipeline: {name}").into());
        }
        println!(
            "{}",
            json!({"kind":"pipeline", "device":device.name().to_string(),
            "kernel":name, "scratch_bytes":pipeline.staticThreadgroupMemoryLength(),
            "max_threads":pipeline.maxTotalThreadsPerThreadgroup()})
        );
        pipelines.push(pipeline);
    }
    let queue = device.newCommandQueue().ok_or("queue")?;
    let buffer = |count: usize, seed: usize| {
        let buf = device
            .newBufferWithLength_options(count * 4, MTLResourceOptions::StorageModeShared)
            .expect("buffer");
        let values =
            unsafe { std::slice::from_raw_parts_mut(buf.contents().as_ptr().cast::<f32>(), count) };
        for (i, value) in values.iter_mut().enumerate() {
            *value = ((i * seed % 127) as f32 - 63.0) / 64.0;
        }
        buf
    };
    let mut classes = [
        (1024usize, 1024usize),
        (1024, 4096),
        (4096, 1024),
        (480, 4320),
    ];
    let mut row_counts = [
        65usize, 127, 128, 129, 191, 192, 193, 255, 256, 257, 319, 320, 321, 383, 384, 385, 448,
        449, 511, 512, 513, 639, 640, 641, 768, 769, 800,
    ];
    if args.len() == 3 {
        classes.reverse();
        row_counts.reverse();
    }
    for (n, k) in classes {
        let w = buffer(n * k, 17);
        let bias = buffer(n, 31);
        for m in row_counts {
            let x = buffer(m * k, 13);
            let outputs = [buffer(m * n + 32, 1), buffer(m * n + 32, 1)];
            for out in &outputs {
                unsafe {
                    std::slice::from_raw_parts_mut(
                        out.contents().as_ptr().cast::<f32>(),
                        m * n + 32,
                    )
                }
                .fill(12345.0);
            }
            let params = [m as u32, n as u32, k as u32, 1, 0];
            for iteration in 0..repeats + 2 {
                for variant in [iteration % 2, 1 - iteration % 2] {
                    let cb = queue.commandBuffer().ok_or("command buffer")?;
                    let enc = cb.computeCommandEncoder().ok_or("encoder")?;
                    enc.setComputePipelineState(&pipelines[variant]);
                    unsafe {
                        for (i, b) in [&x, &w, &bias, &outputs[variant]].iter().enumerate() {
                            enc.setBuffer_offset_atIndex(Some(b), 0, i);
                        }
                        enc.setBytes_length_atIndex(NonNull::from(&params).cast(), 20, 4);
                    }
                    let rows = if variant == 0 { 32 } else { 64 };
                    // Warmups validate an extra group; timed grids match production exactly.
                    enc.dispatchThreads_threadsPerThreadgroup(
                        MTLSize {
                            width: (m.div_ceil(rows) * n.div_ceil(64) + usize::from(iteration < 2))
                                * 256,
                            height: 1,
                            depth: 1,
                        },
                        MTLSize {
                            width: 256,
                            height: 1,
                            depth: 1,
                        },
                    );
                    enc.endEncoding();
                    cb.commit();
                    cb.waitUntilCompleted();
                    if cb.status() != MTLCommandBufferStatus::Completed {
                        return Err(format!("dispatch failed: {:?}", cb.error()).into());
                    }
                    let gpu_us = (cb.GPUEndTime() - cb.GPUStartTime()) * 1e6;
                    if !gpu_us.is_finite() || gpu_us <= 0.0 {
                        return Err("invalid GPU timestamps".into());
                    }
                    if iteration >= 2 {
                        println!(
                            "{}",
                            json!({"kind":"sample", "m":m,"n":n,"k":k,
                            "tile_rows":rows,"iteration":iteration - 2,"gpu_us":gpu_us})
                        );
                    }
                }
                let read = |variant: usize| unsafe {
                    std::slice::from_raw_parts(
                        outputs[variant].contents().as_ptr().cast::<u32>(),
                        m * n + 32,
                    )
                };
                if read(0) != read(1) || read(0)[m * n..].iter().any(|v| *v != 12345.0f32.to_bits())
                {
                    return Err(format!("parity/guard failure: {m}x{n}x{k}").into());
                }
            }
            println!("{}", json!({"kind":"validated", "m":m,"n":n,"k":k}));
        }
    }
    Ok(())
}

#[cfg(not(all(feature = "metal", target_os = "macos")))]
fn main() {
    eprintln!("requires macOS and --features metal");
    std::process::exit(1);
}
