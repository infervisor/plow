use super::mlp_weights::Matrix;
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_foundation::NSString;
use objc2_metal::*;
use std::ffi::c_void;
use std::ptr::NonNull;

pub type Buffer = Retained<ProtocolObject<dyn MTLBuffer>>;
pub type Command = Retained<ProtocolObject<dyn MTLCommandBuffer>>;
type Pipeline = Retained<ProtocolObject<dyn MTLComputePipelineState>>;

pub struct Metal {
    pub device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    gate: Pipeline,
    down: Pipeline,
    finish: Pipeline,
    consume: Pipeline,
    pub x: Buffer,
    pub residual: Buffer,
    pub ane: Buffer,
    pub output: Buffer,
    pub consumed: Buffer,
    pub h: usize,
}

pub fn read<T: Copy>(b: &Buffer, count: usize) -> &[T] {
    assert!(count * std::mem::size_of::<T>() <= b.length());
    // SAFETY: shared buffers are aligned; the caller only reads after joining the GPU.
    unsafe { std::slice::from_raw_parts(b.contents().as_ptr() as *const T, count) }
}

pub fn write<T: Copy>(b: &Buffer, offset: usize, src: &[T]) {
    let bytes = std::mem::size_of_val(src);
    assert!(offset + bytes <= b.length());
    // SAFETY: caller owns this region and has joined earlier GPU users.
    unsafe {
        std::ptr::copy_nonoverlapping(
            src.as_ptr() as *const u8,
            (b.contents().as_ptr() as *mut u8).add(offset),
            bytes,
        )
    };
}

impl Metal {
    pub fn new(rows: usize, h: usize) -> Self {
        let device = MTLCreateSystemDefaultDevice().expect("Metal device");
        let queue = device.newCommandQueue().expect("Metal queue");
        let options = MTLCompileOptions::new();
        options.setMathMode(MTLMathMode::Safe);
        options.setLanguageVersion(MTLLanguageVersion::Version3_2);
        let source = format!(
            "{}\n{}",
            include_str!("../../../../runtime/apple/interp.metal"),
            include_str!("../../../../runtime/apple/mlp_channel.metal")
        );
        let lib = device
            .newLibraryWithSource_options_error(&NSString::from_str(&source), Some(&options))
            .expect("probe Metal compile");
        let pipeline = |name| {
            device
                .newComputePipelineStateWithFunction_error(
                    &lib.newFunctionWithName(&NSString::from_str(name))
                        .expect(name),
                )
                .expect("pipeline")
        };
        let (gate, down, finish) = (
            pipeline("mlp_gate"),
            pipeline("mlp_down"),
            pipeline("mlp_finish"),
        );
        let consume = pipeline("mlp_consume");
        let buffer = |bytes| {
            device
                .newBufferWithLength_options(bytes, MTLResourceOptions::StorageModeShared)
                .expect("buffer")
        };
        let (x, residual, ane, output) = (
            buffer(rows * h * 2),
            buffer(rows * h * 2),
            buffer(rows * h * 4),
            buffer(rows * h * 2),
        );
        let consumed = buffer(rows * h * 2);
        Self {
            device,
            queue,
            gate,
            down,
            finish,
            consume,
            x,
            residual,
            ane,
            output,
            consumed,
            h,
        }
    }

    pub fn buffer(&self, bytes: usize) -> Buffer {
        self.device
            .newBufferWithLength_options(bytes.max(4), MTLResourceOptions::StorageModeShared)
            .expect("buffer")
    }

    fn encode(
        &self,
        cb: &Command,
        pipeline: &Pipeline,
        buffers: &[(&Buffer, usize)],
        params: &[u32],
        groups: usize,
        threads: usize,
    ) {
        let enc = cb.computeCommandEncoder().expect("encoder");
        enc.setComputePipelineState(pipeline);
        unsafe {
            for (i, &(b, offset)) in buffers.iter().enumerate() {
                enc.setBuffer_offset_atIndex(Some(b), offset, i);
            }
            enc.setBytes_length_atIndex(
                NonNull::new(params.as_ptr() as *mut c_void).unwrap(),
                std::mem::size_of_val(params),
                buffers.len(),
            );
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: groups,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: threads,
                height: 1,
                depth: 1,
            },
        );
        enc.endEncoding();
    }

    fn finish_into(&self, cb: &Command, gpu: &Buffer, m: usize, has_ane: bool, bf16: bool) {
        self.encode(
            cb,
            &self.finish,
            &[
                (gpu, 0),
                (&self.ane, 0),
                (&self.residual, 0),
                (&self.output, 0),
            ],
            &[(m * self.h) as u32, u32::from(has_ane), u32::from(bf16), 0],
            (m * self.h).div_ceil(256),
            256,
        );
        self.consume_into(cb, m);
    }

    fn consume_into(&self, cb: &Command, m: usize) {
        self.encode(
            cb,
            &self.consume,
            &[(&self.output, 0), (&self.consumed, 0)],
            &[m as u32, self.h as u32],
            16,
            1024,
        );
    }

    pub fn consume(&self, m: usize) -> Command {
        let cb = self.queue.commandBuffer().expect("command buffer");
        self.consume_into(&cb, m);
        cb.commit();
        cb
    }

    pub fn finish(&self, gpu: &Buffer, m: usize) -> Command {
        let cb = self.queue.commandBuffer().expect("command buffer");
        self.finish_into(&cb, gpu, m, true, false);
        cb.commit();
        cb
    }
}

pub struct Mlp {
    weights: Vec<Buffer>,
    scales: Vec<Buffer>,
    pub intermediate: Buffer,
    pub output: Buffer,
    pub channels: usize,
    encoding: u32,
    pub weight_bytes: usize,
}

impl Mlp {
    pub fn new(gpu: &Metal, weights: &[Matrix; 3], rows: usize) -> Self {
        let load = |data: &[u8]| {
            let b = gpu.buffer(data.len());
            write(&b, 0, data);
            b
        };
        let weight_bytes = weights.iter().map(|w| w.data.len() + w.scales.len()).sum();
        Self {
            weights: weights.iter().map(|w| load(&w.data)).collect(),
            scales: weights.iter().map(|w| load(&w.scales)).collect(),
            intermediate: gpu.buffer(rows * weights[0].n * 2),
            output: gpu.buffer(rows * gpu.h * 4),
            channels: weights[0].n,
            encoding: weights[0].encoding,
            weight_bytes,
        }
    }

    pub fn submit(&self, gpu: &Metal, rows: usize, finish: bool, bf16: bool) -> Command {
        let cb = gpu.queue.commandBuffer().expect("command buffer");
        let p = [
            rows as u32,
            gpu.h as u32,
            self.channels as u32,
            self.encoding,
        ];
        gpu.encode(
            &cb,
            &gpu.gate,
            &[
                (&gpu.x, 0),
                (&self.weights[0], 0),
                (&self.weights[1], 0),
                (&self.scales[0], 0),
                (&self.scales[1], 0),
                (&self.intermediate, 0),
            ],
            &p,
            16,
            1024,
        );
        let p = [p[0], p[1], p[2], p[3] | if bf16 { 4 } else { 0 }];
        gpu.encode(
            &cb,
            &gpu.down,
            &[
                (&self.intermediate, 0),
                (&self.weights[2], 0),
                (&self.scales[2], 0),
                (&self.output, 0),
            ],
            &p,
            16,
            1024,
        );
        if finish {
            gpu.finish_into(&cb, &self.output, rows, false, bf16);
        }
        cb.commit();
        cb
    }
}

pub fn join(cb: &Command) -> f64 {
    cb.waitUntilCompleted();
    assert_eq!(
        cb.status(),
        MTLCommandBufferStatus::Completed,
        "{:?}",
        cb.error()
    );
    (cb.GPUEndTime() - cb.GPUStartTime()) * 1e3
}
