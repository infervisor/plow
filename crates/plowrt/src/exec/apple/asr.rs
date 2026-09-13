use super::*;
use crate::asr::frontend::{MelFeatures, MEL_BINS};
use crate::asset::checkpoint::Checkpoint;
use objc2_metal::MTLResource;
use std::collections::HashMap;

#[derive(Clone, Copy, PartialEq)]
enum ConvKernel {
    Scalar,
    Tiled,
    Packed,
}

#[derive(Clone, Copy, PartialEq)]
enum AudioCapture {
    None,
    Convolution,
    Projection,
    Positioned,
    FirstLayerNorm,
    FirstQuery,
    FirstKey,
    FirstValue,
    FirstAttention,
    FirstLayerOutput,
    TransformerOutput,
}

struct CapturedAudio {
    buffer: Buf,
    rows: usize,
    columns: usize,
}

pub(crate) struct ReadyAudio {
    buffer: Buf,
    rows: usize,
    cols: usize,
}

impl ReadyAudio {
    pub(crate) fn shape(&self) -> [usize; 2] {
        [self.rows, self.cols]
    }

    pub(crate) fn read(&self) -> Vec<f32> {
        unsafe {
            std::slice::from_raw_parts(
                self.buffer.contents().as_ptr().cast::<f32>(),
                self.rows * self.cols,
            )
        }
        .to_vec()
    }
}

pub struct QwenAudioEncoder {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    kernels: HashMap<&'static str, Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    weights: HashMap<String, Buf>,
    packed_linear_weights: HashMap<String, Buf>,
    bf16_linear_weights: bool,
    simd_attention: bool,
    direct_epilogue: bool,
    tile64: bool,
    tile64_selective: bool,
    direct_epilogue_validated: Option<bool>,
    tile64_validated: Option<bool>,
    layers: usize,
    hidden: usize,
    inter: usize,
    output: usize,
    profile_kernels: bool,
    tiled_linear: bool,
    wide_linear: bool,
    large_linear: bool,
    conv_kernel: ConvKernel,
}

// Metal resources are thread-safe; the owning ASR engine serializes submissions.
unsafe impl Send for QwenAudioEncoder {}

impl QwenAudioEncoder {
    pub fn load(checkpoint: &Path) -> Result<Self> {
        let mut this = Self::load_empty(checkpoint)?;
        let ckpt = Checkpoint::open(checkpoint)?;
        for (name, shape, _) in this.weight_specs() {
            this.weight(&ckpt, &name, &shape)?;
        }
        Ok(this)
    }

    pub(crate) fn load_from_packet(
        checkpoint: &Path,
        packet: &crate::exec::packet_runtime::ForwardPacket,
    ) -> Result<Self> {
        if packet.optional_parameter("qwen_audio_graph_v1") != Some(1) {
            return Err(err(
                "ASR packet",
                "missing Qwen audio specialization contract",
            ));
        }
        let mut this = Self::load_empty(checkpoint)?;
        for (name, shape, bf16) in this.weight_specs() {
            this.packet_weight(packet, &name, &shape, bf16)?;
        }
        Ok(this)
    }

    fn load_empty(checkpoint: &Path) -> Result<Self> {
        let cfg: serde_json::Value = serde_json::from_slice(
            &std::fs::read(checkpoint.join("config.json")).map_err(|e| err("ASR config", e))?,
        )
        .map_err(|e| err("ASR config", e))?;
        if cfg["model_type"] != "qwen3_asr" {
            return Err(err("ASR", "unsupported model family"));
        }
        let a = &cfg["thinker_config"]["audio_config"];
        for (key, expected) in [
            ("d_model", 1024),
            ("encoder_attention_heads", 16),
            ("downsample_hidden_size", 480),
            ("num_mel_bins", 128),
            ("n_window", 50),
            ("n_window_infer", 800),
        ] {
            if a[key].as_u64() != Some(expected) {
                return Err(err("ASR config", format!("unsupported {key}")));
            }
        }
        if a["activation_function"] != "gelu" {
            return Err(err("ASR", "unsupported activation"));
        }
        let size = |key: &str| {
            a[key]
                .as_u64()
                .filter(|n| *n > 0 && *n <= 16384)
                .map(|n| n as usize)
                .ok_or_else(|| err("ASR config", key))
        };
        let device =
            MTLCreateSystemDefaultDevice().ok_or_else(|| err("ASR", "Metal unavailable"))?;
        let queue = device
            .newCommandQueue()
            .ok_or_else(|| err("ASR", "command queue"))?;
        let options = MTLCompileOptions::new();
        options.setMathMode(MTLMathMode::Safe);
        options.setLanguageVersion(MTLLanguageVersion::Version3_2);
        let library = device
            .newLibraryWithSource_options_error(
                &NSString::from_str(include_str!("../../../../../runtime/apple/asr.metal")),
                Some(&options),
            )
            .map_err(|e| err("ASR shader", e))?;
        let mut kernels = HashMap::new();
        for name in [
            "asr_conv",
            "asr_conv_tiled",
            "asr_conv_implicit",
            "asr_unfold",
            "asr_unpack_conv",
            "asr_pack",
            "asr_linear",
            "asr_linear_tiled",
            "asr_linear_wide",
            "asr_linear_large",
            "asr_linear_direct",
            "asr_linear_tile64",
            "asr_norm",
            "asr_norm_staged",
            "asr_add",
            "asr_attention",
            "asr_attention_simd",
            "asr_splice",
        ] {
            let function = library
                .newFunctionWithName(&NSString::from_str(name))
                .ok_or_else(|| err("ASR kernel", name))?;
            let pso = match device.newComputePipelineStateWithFunction_error(&function) {
                Ok(pso) => pso,
                Err(_)
                    if matches!(
                        name,
                        "asr_linear_wide"
                            | "asr_linear_large"
                            | "asr_linear_direct"
                            | "asr_linear_tile64"
                            | "asr_attention_simd"
                            | "asr_norm_staged"
                            | "asr_conv_implicit"
                    ) =>
                {
                    continue
                }
                Err(e) => return Err(err(name, e)),
            };
            if matches!(
                name,
                "asr_linear_wide"
                    | "asr_linear_large"
                    | "asr_linear_direct"
                    | "asr_linear_tile64"
                    | "asr_conv_implicit"
            ) && pso.maxTotalThreadsPerThreadgroup()
                < if name == "asr_linear_wide" { 128 } else { 256 }
            {
                continue;
            }
            if pso.threadExecutionWidth() != 32
                || pso.maxTotalThreadsPerThreadgroup() < 32
                || pso.staticThreadgroupMemoryLength() > device.maxThreadgroupMemoryLength()
            {
                if matches!(
                    name,
                    "asr_linear_wide"
                        | "asr_linear_large"
                        | "asr_linear_direct"
                        | "asr_linear_tile64"
                        | "asr_attention_simd"
                        | "asr_norm_staged"
                        | "asr_conv_implicit"
                ) {
                    continue;
                }
                return Err(err(name, "unsupported SIMD/threadgroup limits"));
            }
            kernels.insert(name, pso);
        }
        Ok(Self {
            device,
            queue,
            kernels,
            weights: HashMap::new(),
            packed_linear_weights: HashMap::new(),
            bf16_linear_weights: false,
            simd_attention: false,
            direct_epilogue: false,
            tile64: false,
            tile64_selective: false,
            direct_epilogue_validated: None,
            tile64_validated: None,
            layers: size("encoder_layers")?,
            hidden: size("d_model")?,
            inter: size("encoder_ffn_dim")?,
            output: size("output_dim")?,
            profile_kernels: false,
            tiled_linear: false,
            wide_linear: false,
            large_linear: false,
            conv_kernel: ConvKernel::Scalar,
        })
    }

    fn weight_specs(&self) -> Vec<(String, Vec<usize>, bool)> {
        let mut specs = Vec::new();
        for i in 1..=3 {
            specs.push((
                format!("conv2d{i}.weight"),
                vec![480, if i == 1 { 1 } else { 480 }, 3, 3],
                false,
            ));
            specs.push((format!("conv2d{i}.bias"), vec![480], false));
        }
        let h = self.hidden;
        specs.push(("conv_out.weight".into(), vec![h, 7680], true));
        for l in 0..self.layers {
            for norm in ["self_attn_layer_norm", "final_layer_norm"] {
                for suffix in ["weight", "bias"] {
                    specs.push((format!("layers.{l}.{norm}.{suffix}"), vec![h], false));
                }
            }
            for name in ["q_proj", "k_proj", "v_proj", "out_proj"] {
                let prefix = format!("layers.{l}.self_attn.{name}");
                specs.push((format!("{prefix}.weight"), vec![h, h], true));
                specs.push((format!("{prefix}.bias"), vec![h], false));
            }
            for (name, n, k) in [("fc1", self.inter, h), ("fc2", h, self.inter)] {
                let prefix = format!("layers.{l}.{name}");
                specs.push((format!("{prefix}.weight"), vec![n, k], true));
                specs.push((format!("{prefix}.bias"), vec![n], false));
            }
        }
        specs.push(("ln_post.weight".into(), vec![h], false));
        specs.push(("ln_post.bias".into(), vec![h], false));
        for (name, n) in [("proj1", h), ("proj2", self.output)] {
            specs.push((format!("{name}.weight"), vec![n, h], true));
            specs.push((format!("{name}.bias"), vec![n], false));
        }
        specs
    }

    pub fn set_kernel_profiling(&mut self, enabled: bool) {
        self.profile_kernels = enabled;
    }

    pub fn set_simd_attention(&mut self, enabled: bool) -> Result<()> {
        if enabled && !self.supports_simd_attention() {
            return Err(err("ASR", "SIMD attention pipeline unavailable"));
        }
        self.simd_attention = enabled;
        Ok(())
    }

    pub fn supports_simd_attention(&self) -> bool {
        self.kernels.contains_key("asr_attention_simd")
    }

    pub fn set_direct_epilogue(&mut self, enabled: bool) -> Result<()> {
        if enabled && !self.validate_direct_epilogue()? {
            return Err(err(
                "ASR",
                "direct epilogue unavailable or failed validation",
            ));
        }
        self.direct_epilogue = enabled;
        Ok(())
    }

    pub fn set_tile64(&mut self, enabled: bool) -> Result<()> {
        if enabled && !self.validate_tile64()? {
            return Err(err("ASR", "64x64 tile unavailable or failed validation"));
        }
        self.tile64 = enabled;
        self.tile64_selective = false;
        Ok(())
    }

    pub fn set_tile64_selective(&mut self, enabled: bool) -> Result<()> {
        self.set_tile64(enabled)?;
        self.tile64_selective = enabled;
        Ok(())
    }

    pub fn validate_direct_epilogue(&mut self) -> Result<bool> {
        if let Some(valid) = self.direct_epilogue_validated {
            return Ok(valid);
        }
        self.direct_epilogue_validated = Some(false);
        let valid = self.validate_matrix_epilogue("asr_linear_direct", 32)?;
        self.direct_epilogue_validated = Some(valid);
        Ok(valid)
    }

    pub fn validate_tile64(&mut self) -> Result<bool> {
        if let Some(valid) = self.tile64_validated {
            return Ok(valid);
        }
        self.tile64_validated = Some(false);
        let valid = self.validate_matrix_epilogue("asr_linear_tile64", 64)?;
        self.tile64_validated = Some(valid);
        Ok(valid)
    }

    fn validate_matrix_epilogue(&self, kernel: &str, tile_rows: usize) -> Result<bool> {
        if !self.kernels.contains_key(kernel) {
            return Ok(false);
        }
        let (m, n, k) = (tile_rows + 1, 65usize, tile_rows + 3);
        let x = self.buffer(m * k)?;
        let w = self.buffer(n * k)?;
        let baseline = self.buffer(m * n + 32)?;
        let candidate = self.buffer(m * n + 32)?;
        unsafe {
            let values = std::slice::from_raw_parts_mut(x.contents().as_ptr().cast::<f32>(), m * k);
            for (i, value) in values.iter_mut().enumerate() {
                *value = if i / k == i % k { 1.0 } else { 0.0 };
            }
        }
        // Separate coordinate codes avoid BF16 collisions masking a fragment permutation.
        for rows in [true, false] {
            unsafe {
                let weights =
                    std::slice::from_raw_parts_mut(w.contents().as_ptr().cast::<f32>(), n * k);
                for (i, value) in weights.iter_mut().enumerate() {
                    *value = if rows {
                        (i % k + 1) as f32 / 64.0
                    } else {
                        (i / k + 1) as f32 / 128.0
                    };
                }
                for output in [&baseline, &candidate] {
                    std::slice::from_raw_parts_mut(
                        output.contents().as_ptr().cast::<f32>(),
                        m * n + 32,
                    )
                    .fill(12345.0);
                }
            }
            let cb = self
                .queue
                .commandBuffer()
                .ok_or_else(|| err("ASR", "epilogue probe command buffer"))?;
            let params = [m as u32, n as u32, k as u32, 0, 0];
            self.dispatch(
                &cb,
                "asr_linear_tiled",
                [&x, &w, &w],
                &baseline,
                params,
                m.div_ceil(8) * n.div_ceil(8) * 32,
            )?;
            self.dispatch(
                &cb,
                kernel,
                [&x, &w, &w],
                &candidate,
                params,
                (m.div_ceil(tile_rows) * n.div_ceil(64) + 1) * 256,
            )?;
            cb.commit();
            cb.waitUntilCompleted();
            if cb.status() != MTLCommandBufferStatus::Completed {
                return Err(err("ASR epilogue probe", format!("{:?}", cb.error())));
            }
            for output in [&baseline, &candidate] {
                let values = unsafe {
                    std::slice::from_raw_parts(output.contents().as_ptr().cast::<f32>(), m * n + 32)
                };
                for (i, &value) in values.iter().enumerate() {
                    let expected = if i >= m * n {
                        12345.0
                    } else if rows {
                        (i / n + 1) as f32 / 64.0
                    } else {
                        (i % n + 1) as f32 / 128.0
                    };
                    if value.to_bits() != expected.to_bits() {
                        return Ok(false);
                    }
                }
            }
        }
        Ok(true)
    }

    pub fn set_bf16_linear_weights(&mut self, enabled: bool) -> Result<()> {
        if enabled && !self.kernels.contains_key("asr_linear_large_bf16") {
            let options = MTLCompileOptions::new();
            options.setMathMode(MTLMathMode::Safe);
            options.setLanguageVersion(MTLLanguageVersion::Version3_2);
            let library = self
                .device
                .newLibraryWithSource_options_error(
                    &NSString::from_str(include_str!("../../../../../runtime/apple/asr.metal")),
                    Some(&options),
                )
                .map_err(|e| err("ASR shader", e))?;
            let function = library
                .newFunctionWithName(&NSString::from_str("asr_linear_large_bf16"))
                .ok_or_else(|| err("ASR", "BF16 weight kernel unavailable"))?;
            let pso = self
                .device
                .newComputePipelineStateWithFunction_error(&function)
                .map_err(|e| err("ASR BF16 weights", e))?;
            if pso.threadExecutionWidth() != 32
                || pso.maxTotalThreadsPerThreadgroup() < 256
                || pso.staticThreadgroupMemoryLength() > self.device.maxThreadgroupMemoryLength()
            {
                return Err(err("ASR", "BF16 weight tile unsupported"));
            }
            self.kernels.insert("asr_linear_large_bf16", pso);
        }
        if enabled && self.packed_linear_weights.is_empty() {
            let mut names = vec![
                "conv_out.weight".to_owned(),
                "proj1.weight".to_owned(),
                "proj2.weight".to_owned(),
            ];
            for layer in 0..self.layers {
                for suffix in [
                    "self_attn.q_proj",
                    "self_attn.k_proj",
                    "self_attn.v_proj",
                    "self_attn.out_proj",
                    "fc1",
                    "fc2",
                ] {
                    names.push(format!("layers.{layer}.{suffix}.weight"));
                }
            }
            let mut packed = HashMap::new();
            for name in names {
                let source = &self.weights[&name];
                let count = source.length() / 4;
                let buffer = self.buffer(count.div_ceil(2))?;
                unsafe {
                    let src =
                        std::slice::from_raw_parts(source.contents().as_ptr().cast::<f32>(), count);
                    let dst = std::slice::from_raw_parts_mut(
                        buffer.contents().as_ptr().cast::<u16>(),
                        count,
                    );
                    for (dst, src) in dst.iter_mut().zip(src) {
                        *dst = (src.to_bits() >> 16) as u16;
                    }
                }
                packed.insert(name, buffer);
            }
            self.packed_linear_weights = packed;
        }
        self.bf16_linear_weights = enabled;
        Ok(())
    }

    pub fn set_tiled_linear(&mut self, enabled: bool) {
        self.tiled_linear = enabled;
    }

    pub fn set_wide_linear(&mut self, enabled: bool) -> Result<()> {
        if enabled && !self.supports_wide_linear() {
            return Err(err("ASR", "wide tile pipeline unavailable on this device"));
        }
        self.wide_linear = enabled;
        Ok(())
    }

    pub fn supports_wide_linear(&self) -> bool {
        self.kernels.contains_key("asr_linear_wide")
    }

    pub fn set_large_linear(&mut self, enabled: bool) -> Result<()> {
        if enabled && !self.supports_large_linear() {
            return Err(err("ASR", "large tile pipeline unavailable on this device"));
        }
        self.large_linear = enabled;
        Ok(())
    }

    pub fn supports_large_linear(&self) -> bool {
        self.kernels.contains_key("asr_linear_large")
    }

    fn tiled_dispatch(&self, m: usize, n: usize) -> (&'static str, usize) {
        if self.tile64
            && (!self.tile64_selective
                || (n == 1024 && ((193..=256).contains(&m) || (385..=448).contains(&m)))
                || (m == 800 && n == 480))
        {
            ("asr_linear_tile64", m.div_ceil(64) * n.div_ceil(64) * 256)
        } else if self.direct_epilogue {
            ("asr_linear_direct", m.div_ceil(32) * n.div_ceil(64) * 256)
        } else if self.large_linear {
            ("asr_linear_large", m.div_ceil(32) * n.div_ceil(64) * 256)
        } else if self.wide_linear {
            ("asr_linear_wide", m.div_ceil(16) * n.div_ceil(32) * 128)
        } else {
            ("asr_linear_tiled", m.div_ceil(8) * n.div_ceil(8) * 32)
        }
    }

    pub fn set_tiled_conv(&mut self, enabled: bool) {
        self.conv_kernel = if enabled {
            ConvKernel::Tiled
        } else {
            ConvKernel::Scalar
        };
    }

    pub fn set_packed_conv(&mut self, enabled: bool) {
        self.conv_kernel = if enabled {
            ConvKernel::Packed
        } else {
            ConvKernel::Tiled
        };
    }

    fn buffer(&self, count: usize) -> Result<Buf> {
        self.device
            .newBufferWithLength_options(count * 4, MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| err("ASR allocation", count))
    }

    fn weight(&mut self, ckpt: &Checkpoint, name: &str, shape: &[usize]) -> Result<()> {
        let full = format!("thinker.audio_tower.{name}");
        let (bytes, actual) = ckpt
            .tensor_ex(&full)
            .ok_or_else(|| err("missing ASR weight", &full))?;
        if actual != shape || ckpt.dtype(&full) != Some(safetensors::Dtype::BF16) {
            return Err(err(
                "ASR weight",
                format!("{full}: expected BF16 {shape:?}, found {actual:?}"),
            ));
        }
        let buf = self.buffer(bytes.len() / 2)?;
        let target = unsafe {
            std::slice::from_raw_parts_mut(buf.contents().as_ptr().cast::<f32>(), bytes.len() / 2)
        };
        for (dst, src) in target.iter_mut().zip(bytes.chunks_exact(2)) {
            *dst = f32::from_bits((u16::from_le_bytes([src[0], src[1]]) as u32) << 16);
        }
        self.weights.insert(name.into(), buf);
        Ok(())
    }

    fn packet_weight(
        &mut self,
        packet: &crate::exec::packet_runtime::ForwardPacket,
        name: &str,
        shape: &[usize],
        bf16: bool,
    ) -> Result<()> {
        let full = format!("thinker.audio_tower.{name}");
        let count = shape
            .iter()
            .try_fold(1usize, |count, &dim| count.checked_mul(dim))
            .ok_or_else(|| err("ASR packet weight", format!("{full}: shape overflows")))?;
        let expected = count
            .checked_mul(if bf16 { 2 } else { 4 })
            .ok_or_else(|| err("ASR packet weight", format!("{full}: byte size overflows")))?;
        let bytes = packet.read_tensor(&full)?;
        if expected != bytes.len() {
            return Err(err(
                "ASR packet weight",
                format!("{full}: expected {expected} bytes, found {}", bytes.len()),
            ));
        }
        let buffer = self.buffer(count)?;
        let target = unsafe {
            std::slice::from_raw_parts_mut(buffer.contents().as_ptr().cast::<f32>(), count)
        };
        if bf16 {
            for (target, source) in target.iter_mut().zip(bytes.chunks_exact(2)) {
                *target = f32::from_bits((u16::from_le_bytes([source[0], source[1]]) as u32) << 16);
            }
        } else {
            for (target, source) in target.iter_mut().zip(bytes.chunks_exact(4)) {
                *target = f32::from_le_bytes([source[0], source[1], source[2], source[3]]);
            }
        }
        self.weights.insert(name.into(), buffer);
        Ok(())
    }

    fn dispatch(
        &self,
        cb: &ProtocolObject<dyn MTLCommandBuffer>,
        name: &str,
        inputs: [&Buf; 3],
        output: &Buf,
        params: [u32; 5],
        count: usize,
    ) -> Result<()> {
        self.dispatch_offsets(cb, name, inputs, output, params, count, [0; 4])
    }

    fn dispatch_offsets(
        &self,
        cb: &ProtocolObject<dyn MTLCommandBuffer>,
        name: &str,
        inputs: [&Buf; 3],
        output: &Buf,
        params: [u32; 5],
        count: usize,
        offsets: [usize; 4],
    ) -> Result<()> {
        let isolated = if self.profile_kernels {
            Some(
                self.queue
                    .commandBuffer()
                    .ok_or_else(|| err("ASR", "profile command buffer"))?,
            )
        } else {
            None
        };
        let cb = isolated.as_deref().unwrap_or(cb);
        let enc = cb
            .computeCommandEncoder()
            .ok_or_else(|| err("ASR", "encoder"))?;
        enc.setComputePipelineState(&self.kernels[name]);
        unsafe {
            for (i, b) in inputs.iter().enumerate() {
                enc.setBuffer_offset_atIndex(Some(b), offsets[i], i);
            }
            enc.setBuffer_offset_atIndex(Some(output), offsets[3], 3);
            enc.setBytes_length_atIndex(NonNull::from(&params).cast(), 20, 4);
        }
        enc.dispatchThreads_threadsPerThreadgroup(
            MTLSize {
                width: count,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: match name {
                    "asr_linear_large"
                    | "asr_linear_large_bf16"
                    | "asr_linear_direct"
                    | "asr_linear_tile64"
                    | "asr_conv_implicit" => 256,
                    "asr_linear_wide" => 128,
                    _ => 32,
                },
                height: 1,
                depth: 1,
            },
        );
        enc.endEncoding();
        if isolated.is_some() {
            cb.commit();
            cb.waitUntilCompleted();
            if cb.status() != MTLCommandBufferStatus::Completed {
                return Err(err("ASR profile dispatch", format!("{:?}", cb.error())));
            }
            eprintln!(
                "asr_kernel={name} params={params:?} gpu_us={:.3}",
                (cb.GPUEndTime() - cb.GPUStartTime()) * 1e6
            );
        }
        Ok(())
    }

    fn linear(
        &self,
        cb: &ProtocolObject<dyn MTLCommandBuffer>,
        x: &Buf,
        name: &str,
        m: usize,
        n: usize,
        k: usize,
        gelu: bool,
    ) -> Result<Buf> {
        let y = self.buffer(m * n)?;
        self.linear_into(cb, x, name, m, n, k, gelu, &y)?;
        Ok(y)
    }

    fn linear_into(
        &self,
        cb: &ProtocolObject<dyn MTLCommandBuffer>,
        x: &Buf,
        name: &str,
        m: usize,
        n: usize,
        k: usize,
        gelu: bool,
        y: &Buf,
    ) -> Result<()> {
        let w = if self.bf16_linear_weights {
            &self.packed_linear_weights[&format!("{name}.weight")]
        } else {
            &self.weights[&format!("{name}.weight")]
        };
        let bias = self.weights.get(&format!("{name}.bias"));
        let (kernel, count) = if self.bf16_linear_weights {
            (
                "asr_linear_large_bf16",
                m.div_ceil(32) * n.div_ceil(64) * 256,
            )
        } else if self.tiled_linear {
            self.tiled_dispatch(m, n)
        } else {
            ("asr_linear", m * n)
        };
        self.dispatch(
            cb,
            kernel,
            [x, w, bias.unwrap_or(w)],
            y,
            [
                m as u32,
                n as u32,
                k as u32,
                bias.is_some() as u32,
                gelu as u32,
            ],
            count,
        )
    }

    fn norm(
        &self,
        cb: &ProtocolObject<dyn MTLCommandBuffer>,
        x: &Buf,
        name: &str,
        m: usize,
    ) -> Result<Buf> {
        let y = self.buffer(m * self.hidden)?;
        self.norm_into(cb, x, name, m, &y)?;
        Ok(y)
    }

    fn norm_into(
        &self,
        cb: &ProtocolObject<dyn MTLCommandBuffer>,
        x: &Buf,
        name: &str,
        m: usize,
        y: &Buf,
    ) -> Result<()> {
        let staged =
            self.hidden == 1024 && m <= 256 && self.kernels.contains_key("asr_norm_staged");
        self.dispatch(
            cb,
            if staged {
                "asr_norm_staged"
            } else {
                "asr_norm"
            },
            [
                x,
                &self.weights[&format!("{name}.weight")],
                &self.weights[&format!("{name}.bias")],
            ],
            y,
            [m as u32, self.hidden as u32, 0, 0, 0],
            if staged { m * 32 } else { m },
        )
    }

    pub fn encode(&self, mel: &MelFeatures) -> Result<Vec<f32>> {
        Ok(self.encode_device(mel)?.read())
    }

    pub fn convolution_output(&self, mel: &MelFeatures) -> Result<(Vec<f32>, [usize; 4])> {
        let chunks = mel.frames.div_ceil(100);
        let time = mel.frames.min(100).div_ceil(8);
        let (_, captured) = self.encode_device_capture(mel, AudioCapture::Convolution)?;
        let captured = captured.ok_or_else(|| err("ASR convolution", "capture failed"))?;
        let shape = [chunks, 480, 16, time];
        let elements = shape.into_iter().product();
        let values = unsafe {
            std::slice::from_raw_parts(captured.buffer.contents().as_ptr().cast::<f32>(), elements)
        }
        .to_vec();
        Ok((values, shape))
    }

    pub fn convolution_projection_output(
        &self,
        mel: &MelFeatures,
    ) -> Result<(Vec<f32>, [usize; 2])> {
        let (_, captured) = self.encode_device_capture(mel, AudioCapture::Projection)?;
        let captured = captured.ok_or_else(|| err("ASR projection", "capture failed"))?;
        let shape = [captured.rows, captured.columns];
        let values = unsafe {
            std::slice::from_raw_parts(
                captured.buffer.contents().as_ptr().cast::<f32>(),
                captured.rows * captured.columns,
            )
        }
        .to_vec();
        Ok((values, shape))
    }

    pub fn positioned_convolution_output(
        &self,
        mel: &MelFeatures,
    ) -> Result<(Vec<f32>, [usize; 2])> {
        let (_, captured) = self.encode_device_capture(mel, AudioCapture::Positioned)?;
        let captured = captured.ok_or_else(|| err("ASR position", "capture failed"))?;
        let shape = [captured.rows, captured.columns];
        let values = unsafe {
            std::slice::from_raw_parts(
                captured.buffer.contents().as_ptr().cast::<f32>(),
                captured.rows * captured.columns,
            )
        }
        .to_vec();
        Ok((values, shape))
    }

    pub fn first_layer_norm_output(&self, mel: &MelFeatures) -> Result<(Vec<f32>, [usize; 2])> {
        let (_, captured) = self.encode_device_capture(mel, AudioCapture::FirstLayerNorm)?;
        let captured = captured.ok_or_else(|| err("ASR layer norm", "capture failed"))?;
        let shape = [captured.rows, captured.columns];
        let values = unsafe {
            std::slice::from_raw_parts(
                captured.buffer.contents().as_ptr().cast::<f32>(),
                captured.rows * captured.columns,
            )
        }
        .to_vec();
        Ok((values, shape))
    }

    pub fn first_layer_qkv_output(&self, mel: &MelFeatures) -> Result<([Vec<f32>; 3], [usize; 2])> {
        let mut outputs = Vec::with_capacity(3);
        let mut shape = None;
        for capture in [
            AudioCapture::FirstQuery,
            AudioCapture::FirstKey,
            AudioCapture::FirstValue,
        ] {
            let (_, captured) = self.encode_device_capture(mel, capture)?;
            let captured = captured.ok_or_else(|| err("ASR QKV", "capture failed"))?;
            let current = [captured.rows, captured.columns];
            if shape.is_some_and(|prior| prior != current) {
                return Err(err("ASR QKV", "capture shapes differ"));
            }
            shape = Some(current);
            outputs.push(
                unsafe {
                    std::slice::from_raw_parts(
                        captured.buffer.contents().as_ptr().cast::<f32>(),
                        captured.rows * captured.columns,
                    )
                }
                .to_vec(),
            );
        }
        Ok((outputs.try_into().unwrap(), shape.unwrap()))
    }

    pub fn first_layer_attention_output(
        &self,
        mel: &MelFeatures,
    ) -> Result<(Vec<f32>, [usize; 2])> {
        let (_, captured) = self.encode_device_capture(mel, AudioCapture::FirstAttention)?;
        let captured = captured.ok_or_else(|| err("ASR attention", "capture failed"))?;
        let values = unsafe {
            std::slice::from_raw_parts(
                captured.buffer.contents().as_ptr().cast::<f32>(),
                captured.rows * captured.columns,
            )
        }
        .to_vec();
        Ok((values, [captured.rows, captured.columns]))
    }

    pub fn first_layer_output(&self, mel: &MelFeatures) -> Result<(Vec<f32>, [usize; 2])> {
        let (_, captured) = self.encode_device_capture(mel, AudioCapture::FirstLayerOutput)?;
        let captured = captured.ok_or_else(|| err("ASR layer", "capture failed"))?;
        let values = unsafe {
            std::slice::from_raw_parts(
                captured.buffer.contents().as_ptr().cast::<f32>(),
                captured.rows * captured.columns,
            )
        }
        .to_vec();
        Ok((values, [captured.rows, captured.columns]))
    }

    pub fn transformer_output(&self, mel: &MelFeatures) -> Result<(Vec<f32>, [usize; 2])> {
        let (_, captured) = self.encode_device_capture(mel, AudioCapture::TransformerOutput)?;
        let captured = captured.ok_or_else(|| err("ASR transformer", "capture failed"))?;
        let values = unsafe {
            std::slice::from_raw_parts(
                captured.buffer.contents().as_ptr().cast::<f32>(),
                captured.rows * captured.columns,
            )
        }
        .to_vec();
        Ok((values, [captured.rows, captured.columns]))
    }

    fn finish_capture(
        cb: &ProtocolObject<dyn MTLCommandBuffer>,
        buffer: Buf,
        rows: usize,
        columns: usize,
    ) -> Result<(ReadyAudio, Option<CapturedAudio>)> {
        cb.commit();
        cb.waitUntilCompleted();
        if cb.status() != MTLCommandBufferStatus::Completed {
            return Err(err("ASR dispatch", format!("{:?}", cb.error())));
        }
        Ok((
            ReadyAudio {
                buffer: buffer.clone(),
                rows,
                cols: columns,
            },
            Some(CapturedAudio {
                buffer,
                rows,
                columns,
            }),
        ))
    }

    pub(crate) fn encode_device(&self, mel: &MelFeatures) -> Result<ReadyAudio> {
        Ok(self.encode_device_capture(mel, AudioCapture::None)?.0)
    }

    fn encode_device_capture(
        &self,
        mel: &MelFeatures,
        capture: AudioCapture,
    ) -> Result<(ReadyAudio, Option<CapturedAudio>)> {
        if !(50..=3000).contains(&mel.frames) || mel.values.len() != MEL_BINS * mel.frames {
            return Err(err("ASR", "invalid feature shape"));
        }
        let chunks = mel.frames.div_ceil(100);
        let t = mel.frames.min(100);
        let initial = self.buffer(chunks * MEL_BINS * t)?;
        let dst = unsafe {
            std::slice::from_raw_parts_mut(
                initial.contents().as_ptr().cast::<f32>(),
                chunks * MEL_BINS * t,
            )
        };
        dst.fill(0.0);
        for b in 0..chunks {
            for m in 0..MEL_BINS {
                for j in 0..t {
                    if b * 100 + j < mel.frames {
                        dst[(b * MEL_BINS + m) * t + j] =
                            bf(mel.values[m * mel.frames + b * 100 + j]);
                    }
                }
            }
        }
        let cb = self
            .queue
            .commandBuffer()
            .ok_or_else(|| err("ASR", "command buffer"))?;
        let mut x = initial;
        let (mut f, mut time, mut ci) = (128usize, t, 1usize);
        for i in 1..=3 {
            let (fo, to) = (f.div_ceil(2), time.div_ceil(2));
            let y = self.buffer(chunks * 480 * fo * to)?;
            if self.conv_kernel == ConvKernel::Packed
                && self.direct_epilogue
                && self.kernels.contains_key("asr_conv_implicit")
            {
                self.dispatch(
                    &cb,
                    "asr_conv_implicit",
                    [
                        &x,
                        &self.weights[&format!("conv2d{i}.weight")],
                        &self.weights[&format!("conv2d{i}.bias")],
                    ],
                    &y,
                    [ci as u32, 480, f as u32, time as u32, chunks as u32],
                    chunks * (fo * to).div_ceil(32) * 480usize.div_ceil(64) * 256,
                )?;
            } else if self.conv_kernel == ConvKernel::Packed && ci > 1 {
                let params = [ci as u32, 480, f as u32, time as u32, 1];
                let m = fo * to;
                let k = ci * 9;
                let patches = self.buffer(m * k)?;
                let projected = self.buffer(m * 480)?;
                let w = &self.weights[&format!("conv2d{i}.weight")];
                let bias = &self.weights[&format!("conv2d{i}.bias")];
                let (kernel, count) = self.tiled_dispatch(m, 480);
                for batch in 0..chunks {
                    self.dispatch_offsets(
                        &cb,
                        "asr_unfold",
                        [&x, &x, &x],
                        &patches,
                        params,
                        m * k,
                        [batch * ci * f * time * 4, 0, 0, 0],
                    )?;
                    self.dispatch(
                        &cb,
                        kernel,
                        [&patches, w, bias],
                        &projected,
                        [m as u32, 480, k as u32, 1, 1],
                        count,
                    )?;
                    self.dispatch_offsets(
                        &cb,
                        "asr_unpack_conv",
                        [&projected, &projected, &projected],
                        &y,
                        params,
                        m * 480,
                        [0, 0, 0, batch * m * 480 * 4],
                    )?;
                }
            } else {
                self.dispatch(
                    &cb,
                    if self.conv_kernel != ConvKernel::Scalar {
                        "asr_conv_tiled"
                    } else {
                        "asr_conv"
                    },
                    [
                        &x,
                        &self.weights[&format!("conv2d{i}.weight")],
                        &self.weights[&format!("conv2d{i}.bias")],
                    ],
                    &y,
                    [ci as u32, 480, f as u32, time as u32, chunks as u32],
                    if self.conv_kernel != ConvKernel::Scalar {
                        chunks * (fo * to).div_ceil(8) * 480usize.div_ceil(8) * 32
                    } else {
                        chunks * 480 * fo * to
                    },
                )?;
            }
            x = y;
            f = fo;
            time = to;
            ci = 480;
        }
        let mut captured = (capture == AudioCapture::Convolution).then(|| CapturedAudio {
            buffer: x.clone(),
            rows: chunks * 480 * f,
            columns: time,
        });
        let rows = crate::asr::qwen_audio_rows(mel.frames);
        let packed = self.buffer(rows * 7680)?;
        self.dispatch(
            &cb,
            "asr_pack",
            [&x, &x, &x],
            &packed,
            [rows as u32, 7680, time as u32, 16, 0],
            rows * 7680,
        )?;
        x = self.linear(&cb, &packed, "conv_out", rows, self.hidden, 7680, false)?;
        if capture == AudioCapture::Projection {
            captured = Some(CapturedAudio {
                buffer: x.clone(),
                rows,
                columns: self.hidden,
            });
        }
        let positioned = self.buffer(rows * self.hidden)?;
        self.dispatch(
            &cb,
            "asr_add",
            [&x, &x, &x],
            &positioned,
            [rows as u32, self.hidden as u32, time as u32, 0, 0],
            rows * self.hidden,
        )?;
        x = positioned;
        if capture == AudioCapture::Positioned {
            return Self::finish_capture(&cb, x, rows, self.hidden);
        }
        // Sequential encoders track scratch read/write hazards.
        // Scratch is request-local; projected audio owns a separate buffer.
        let norm = self.buffer(rows * self.hidden)?;
        let q = self.buffer(rows * self.hidden)?;
        let k = self.buffer(rows * self.hidden)?;
        let v = self.buffer(rows * self.hidden)?;
        let attn = self.buffer(rows * self.hidden)?;
        let out = self.buffer(rows * self.hidden)?;
        let residual = self.buffer(rows * self.hidden)?;
        let up = self.buffer(rows * self.inter)?;
        for l in 0..self.layers {
            self.norm_into(
                &cb,
                &x,
                &format!("layers.{l}.self_attn_layer_norm"),
                rows,
                &norm,
            )?;
            if l == 0 && capture == AudioCapture::FirstLayerNorm {
                return Self::finish_capture(&cb, norm, rows, self.hidden);
            }
            self.linear_into(
                &cb,
                &norm,
                &format!("layers.{l}.self_attn.q_proj"),
                rows,
                self.hidden,
                self.hidden,
                false,
                &q,
            )?;
            if l == 0 && capture == AudioCapture::FirstQuery {
                return Self::finish_capture(&cb, q, rows, self.hidden);
            }
            self.linear_into(
                &cb,
                &norm,
                &format!("layers.{l}.self_attn.k_proj"),
                rows,
                self.hidden,
                self.hidden,
                false,
                &k,
            )?;
            if l == 0 && capture == AudioCapture::FirstKey {
                return Self::finish_capture(&cb, k, rows, self.hidden);
            }
            self.linear_into(
                &cb,
                &norm,
                &format!("layers.{l}.self_attn.v_proj"),
                rows,
                self.hidden,
                self.hidden,
                false,
                &v,
            )?;
            if l == 0 && capture == AudioCapture::FirstValue {
                return Self::finish_capture(&cb, v, rows, self.hidden);
            }
            self.dispatch(
                &cb,
                if self.simd_attention {
                    "asr_attention_simd"
                } else {
                    "asr_attention"
                },
                [&q, &k, &v],
                &attn,
                [rows as u32, self.hidden as u32, 64, (time * 8) as u32, 0],
                rows * 16 * if self.simd_attention { 32 } else { 1 },
            )?;
            if l == 0 && capture == AudioCapture::FirstAttention {
                return Self::finish_capture(&cb, attn, rows, self.hidden);
            }
            self.linear_into(
                &cb,
                &attn,
                &format!("layers.{l}.self_attn.out_proj"),
                rows,
                self.hidden,
                self.hidden,
                false,
                &out,
            )?;
            self.dispatch(
                &cb,
                "asr_add",
                [&x, &out, &out],
                &residual,
                [rows as u32, self.hidden as u32, 0, 0, 0],
                rows * self.hidden,
            )?;
            self.norm_into(
                &cb,
                &residual,
                &format!("layers.{l}.final_layer_norm"),
                rows,
                &norm,
            )?;
            self.linear_into(
                &cb,
                &norm,
                &format!("layers.{l}.fc1"),
                rows,
                self.inter,
                self.hidden,
                true,
                &up,
            )?;
            self.linear_into(
                &cb,
                &up,
                &format!("layers.{l}.fc2"),
                rows,
                self.hidden,
                self.inter,
                false,
                &out,
            )?;
            self.dispatch(
                &cb,
                "asr_add",
                [&residual, &out, &out],
                &x,
                [rows as u32, self.hidden as u32, 0, 0, 0],
                rows * self.hidden,
            )?;
            if l == 0 && capture == AudioCapture::FirstLayerOutput {
                return Self::finish_capture(&cb, x, rows, self.hidden);
            }
        }
        if capture == AudioCapture::TransformerOutput {
            return Self::finish_capture(&cb, x, rows, self.hidden);
        }
        x = self.norm(&cb, &x, "ln_post", rows)?;
        x = self.linear(&cb, &x, "proj1", rows, self.hidden, self.hidden, true)?;
        x = self.linear(&cb, &x, "proj2", rows, self.output, self.hidden, false)?;
        cb.commit();
        cb.waitUntilCompleted();
        if cb.status() != MTLCommandBufferStatus::Completed {
            return Err(err("ASR dispatch", format!("{:?}", cb.error())));
        }
        Ok((
            ReadyAudio {
                buffer: x,
                rows,
                cols: self.output,
            },
            captured,
        ))
    }

    pub(crate) fn splice_into(
        &self,
        audio: &ReadyAudio,
        table: &Buf,
        vocab: usize,
        output: &Buf,
        bindings: &[[u32; 2]],
    ) -> Result<()> {
        let cb = self
            .queue
            .commandBuffer()
            .ok_or_else(|| err("ASR splice", "command buffer"))?;
        self.encode_splice(&cb, audio, table, vocab, output, bindings)?;
        cb.commit();
        cb.waitUntilCompleted();
        if cb.status() != MTLCommandBufferStatus::Completed {
            return Err(err("ASR splice", format!("{:?}", cb.error())));
        }
        Ok(())
    }

    pub(crate) fn encode_splice(
        &self,
        cb: &ProtocolObject<dyn MTLCommandBuffer>,
        audio: &ReadyAudio,
        table: &Buf,
        vocab: usize,
        output: &Buf,
        bindings: &[[u32; 2]],
    ) -> Result<()> {
        let hidden = audio.cols;
        if hidden == 0
            || bindings.is_empty()
            || bindings
                .len()
                .checked_mul(hidden)
                .and_then(|n| n.checked_mul(2))
                .is_none_or(|bytes| bytes > output.length())
            || table.device().registryID() != self.device.registryID()
            || output.device().registryID() != self.device.registryID()
            || audio.buffer.device().registryID() != self.device.registryID()
        {
            return Err(err("ASR splice", "invalid shape or device"));
        }
        if vocab
            .checked_mul(hidden)
            .and_then(|n| n.checked_mul(2))
            .is_none_or(|bytes| bytes > table.length())
        {
            return Err(err("ASR splice", "embedding table too small"));
        }
        if bindings.iter().any(|&[id, row]| {
            id as usize >= vocab || (row != u32::MAX && row as usize >= audio.rows)
        }) {
            return Err(err("ASR splice", "token or audio row out of bounds"));
        }
        let rows = shared_from(&self.device, bytes_of(bindings))?;
        self.dispatch(
            cb,
            "asr_splice",
            [&audio.buffer, table, &rows],
            output,
            [bindings.len() as u32, hidden as u32, 0, 0, 0],
            bindings.len() * hidden,
        )?;
        Ok(())
    }
}

fn bf(v: f32) -> f32 {
    let u = v.to_bits();
    f32::from_bits(u.wrapping_add(0x7fff + ((u >> 16) & 1)) & 0xffff0000)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metal_asr_linear_matches_known_values() {
        let device = MTLCreateSystemDefaultDevice().expect("Metal device");
        let options = MTLCompileOptions::new();
        options.setMathMode(MTLMathMode::Safe);
        options.setLanguageVersion(MTLLanguageVersion::Version3_2);
        let library = device
            .newLibraryWithSource_options_error(
                &NSString::from_str(include_str!("../../../../../runtime/apple/asr.metal")),
                Some(&options),
            )
            .expect("ASR shader compiles");
        let mut kernels = HashMap::new();
        for name in [
            "asr_conv",
            "asr_conv_tiled",
            "asr_conv_implicit",
            "asr_unfold",
            "asr_unpack_conv",
            "asr_pack",
            "asr_linear",
            "asr_linear_tiled",
            "asr_linear_wide",
            "asr_linear_large",
            "asr_linear_large_bf16",
            "asr_linear_direct",
            "asr_linear_tile64",
            "asr_attention_simd",
            "asr_norm",
            "asr_norm_staged",
            "asr_add",
            "asr_attention",
            "asr_splice",
        ] {
            let function = library
                .newFunctionWithName(&NSString::from_str(name))
                .unwrap();
            kernels.insert(
                name,
                device
                    .newComputePipelineStateWithFunction_error(&function)
                    .unwrap(),
            );
        }
        let mut engine = QwenAudioEncoder {
            queue: device.newCommandQueue().unwrap(),
            device,
            kernels,
            weights: HashMap::new(),
            packed_linear_weights: HashMap::new(),
            bf16_linear_weights: false,
            simd_attention: false,
            direct_epilogue: false,
            tile64: false,
            tile64_selective: false,
            direct_epilogue_validated: None,
            tile64_validated: None,
            layers: 1,
            hidden: 4,
            inter: 8,
            output: 4,
            profile_kernels: false,
            tiled_linear: false,
            wide_linear: false,
            large_linear: false,
            conv_kernel: ConvKernel::Scalar,
        };
        let upload = |data: &[f32]| {
            let buf = engine.buffer(data.len()).unwrap();
            unsafe {
                std::ptr::copy_nonoverlapping(
                    data.as_ptr(),
                    buf.contents().as_ptr().cast::<f32>(),
                    data.len(),
                );
            }
            buf
        };
        let x = upload(&[1., 2., 3., 4.]);
        for hidden in [1usize, 7, 32, 65] {
            let values: Vec<_> = (0..3 * hidden)
                .map(|i| f32::from_bits(0x3f008000 + (i as u32 * 65537)))
                .collect();
            let audio = ReadyAudio {
                buffer: upload(&values),
                rows: 3,
                cols: hidden,
            };
            let words: Vec<u16> = (0..5 * hidden).map(|i| 0x3f00 + i as u16).collect();
            let table = shared_from(&engine.device, bytes_of(&words)).unwrap();
            let bindings = [[4, u32::MAX], [0, 2], [3, 0], [1, u32::MAX]];
            let expected: Vec<_> = bindings
                .iter()
                .flat_map(|&[id, row]| {
                    let values = &values;
                    let words = &words;
                    (0..hidden).map(move |col| {
                        if row == u32::MAX {
                            words[id as usize * hidden + col]
                        } else {
                            (bf(values[row as usize * hidden + col]).to_bits() >> 16) as u16
                        }
                    })
                })
                .collect();
            let output = shared_from(
                &engine.device,
                bytes_of(&vec![0xdead_u16; expected.len() + 32]),
            )
            .unwrap();
            engine
                .splice_into(&audio, &table, 5, &output, &bindings)
                .unwrap();
            let actual = unsafe {
                std::slice::from_raw_parts(
                    output.contents().as_ptr().cast::<u16>(),
                    expected.len() + 32,
                )
            };
            assert_eq!(&actual[..expected.len()], expected);
            assert!(actual[expected.len()..].iter().all(|&v| v == 0xdead));
            assert!(engine
                .splice_into(&audio, &table, 5, &output, &[[5, u32::MAX]])
                .is_err());
            assert!(engine
                .splice_into(&audio, &table, 5, &output, &[[0, 3]])
                .is_err());
            let too_small = engine
                .device
                .newBufferWithLength_options(1, MTLResourceOptions::StorageModeShared)
                .unwrap();
            assert!(engine
                .splice_into(&audio, &table, 5, &too_small, &bindings)
                .is_err());
        }
        let w = upload(&[1., 0., 0., 1.]);
        let b = upload(&[0.5, -0.5]);
        let y = engine.buffer(4).unwrap();
        let cb = engine.queue.commandBuffer().unwrap();
        engine
            .dispatch(&cb, "asr_linear", [&x, &w, &b], &y, [2, 2, 2, 1, 0], 4)
            .unwrap();
        cb.commit();
        cb.waitUntilCompleted();
        assert_eq!(cb.status(), MTLCommandBufferStatus::Completed);
        let out = unsafe { std::slice::from_raw_parts(y.contents().as_ptr().cast::<f32>(), 4) };
        assert_eq!(out, &[1.5, 1.5, 3.5, 3.5]);

        let conv_x = upload(&[1.0; 18]);
        let conv_w = upload(&[1.0; 36]);
        let conv_b = upload(&[0.0; 2]);
        let conv_y = engine.buffer(8).unwrap();
        let pack_x = upload(&(0..16).map(|i| i as f32).collect::<Vec<_>>());
        let pack_y = engine.buffer(12).unwrap();
        let q = upload(&[0.0; 10]);
        let v = upload(&[0.0, 2.0, 2.0, 4.0, 10.0, 12.0, 12.0, 14.0, 20.0, 22.0]);
        let attn = engine.buffer(10).unwrap();
        let cb = engine.queue.commandBuffer().unwrap();
        engine
            .dispatch(
                &cb,
                "asr_conv",
                [&conv_x, &conv_w, &conv_b],
                &conv_y,
                [2, 2, 3, 3, 1],
                8,
            )
            .unwrap();
        engine
            .dispatch(
                &cb,
                "asr_pack",
                [&pack_x, &pack_x, &pack_x],
                &pack_y,
                [3, 4, 2, 2, 0],
                12,
            )
            .unwrap();
        engine
            .dispatch(
                &cb,
                "asr_attention",
                [&q, &q, &v],
                &attn,
                [5, 2, 2, 2, 0],
                5,
            )
            .unwrap();
        cb.commit();
        cb.waitUntilCompleted();
        assert_eq!(cb.status(), MTLCommandBufferStatus::Completed);
        let read = |b: &Buf, n| unsafe {
            std::slice::from_raw_parts(b.contents().as_ptr().cast::<f32>(), n).to_vec()
        };
        assert_eq!(
            read(&conv_y, 8),
            vec![8.0; 8],
            "padded stride-two convolution"
        );
        assert_eq!(
            read(&pack_y, 12),
            [0., 2., 4., 6., 1., 3., 5., 7., 8., 10., 12., 14.],
            "time-major packing with truncated tail"
        );
        assert_eq!(
            read(&attn, 10),
            [1., 3., 1., 3., 11., 13., 11., 13., 20., 22.],
            "attention cannot cross a window boundary"
        );
        for (batches, ci, co, f, t) in [
            (1usize, 1usize, 1usize, 1usize, 1usize),
            (2, 2, 13, 3, 5),
            (2, 9, 8, 4, 6),
            (1, 480, 9, 5, 7),
        ] {
            let x = upload(
                &(0..batches * ci * f * t)
                    .map(|i| ((i % 17) as f32 - 8.0) / 16.0)
                    .collect::<Vec<_>>(),
            );
            let w = upload(
                &(0..co * ci * 9)
                    .map(|i| ((i % 13) as f32 - 6.0) / 16.0)
                    .collect::<Vec<_>>(),
            );
            let b = upload(&(0..co).map(|i| (i as f32 - 5.0) / 16.0).collect::<Vec<_>>());
            let spatial = f.div_ceil(2) * t.div_ceil(2);
            let count = batches * co * spatial;
            let scalar = upload(&vec![12345.0; count + 32]);
            let tiled = upload(&vec![12345.0; count + 32]);
            let params = [ci as u32, co as u32, f as u32, t as u32, batches as u32];
            let cb = engine.queue.commandBuffer().unwrap();
            engine
                .dispatch(&cb, "asr_conv", [&x, &w, &b], &scalar, params, count)
                .unwrap();
            engine
                .dispatch(
                    &cb,
                    "asr_conv_tiled",
                    [&x, &w, &b],
                    &tiled,
                    params,
                    batches * spatial.div_ceil(8) * co.div_ceil(8) * 32,
                )
                .unwrap();
            let m = batches * spatial;
            let k = ci * 9;
            let patches = upload(&vec![12345.0; m * k + 32]);
            let projected = engine.buffer(m * co).unwrap();
            let packed = upload(&vec![12345.0; count + 32]);
            engine
                .dispatch(&cb, "asr_unfold", [&x, &x, &x], &patches, params, m * k)
                .unwrap();
            engine
                .dispatch(
                    &cb,
                    "asr_linear_tiled",
                    [&patches, &w, &b],
                    &projected,
                    [m as u32, co as u32, k as u32, 1, 1],
                    m.div_ceil(8) * co.div_ceil(8) * 32,
                )
                .unwrap();
            engine
                .dispatch(
                    &cb,
                    "asr_unpack_conv",
                    [&projected, &projected, &projected],
                    &packed,
                    params,
                    count,
                )
                .unwrap();
            let panel_patches = upload(&vec![12345.0; spatial * k + 32]);
            let panel_projected = upload(&vec![12345.0; spatial * co + 32]);
            let panel_output = upload(&vec![12345.0; count + 32]);
            for batch in 0..batches {
                let params = [ci as u32, co as u32, f as u32, t as u32, 1];
                engine
                    .dispatch_offsets(
                        &cb,
                        "asr_unfold",
                        [&x, &x, &x],
                        &panel_patches,
                        params,
                        spatial * k,
                        [batch * ci * f * t * 4, 0, 0, 0],
                    )
                    .unwrap();
                engine
                    .dispatch(
                        &cb,
                        "asr_linear_tiled",
                        [&panel_patches, &w, &b],
                        &panel_projected,
                        [spatial as u32, co as u32, k as u32, 1, 1],
                        spatial.div_ceil(8) * co.div_ceil(8) * 32,
                    )
                    .unwrap();
                engine
                    .dispatch_offsets(
                        &cb,
                        "asr_unpack_conv",
                        [&panel_projected, &panel_projected, &panel_projected],
                        &panel_output,
                        params,
                        spatial * co,
                        [0, 0, 0, batch * spatial * co * 4],
                    )
                    .unwrap();
            }
            cb.commit();
            cb.waitUntilCompleted();
            assert_eq!(cb.status(), MTLCommandBufferStatus::Completed);
            assert_eq!(
                read(&scalar, count + 32),
                read(&tiled, count + 32),
                "convolution batch/padding/tail/bias/GELU parity {params:?}"
            );
            assert_eq!(
                read(&scalar, count + 32),
                read(&packed, count + 32),
                "packed convolution parity {params:?}"
            );
            assert_eq!(
                read(&scalar, count + 32),
                read(&panel_output, count + 32),
                "panel reuse and batch offset parity {params:?}"
            );
            assert!(read(&panel_patches, spatial * k + 32)[spatial * k..]
                .iter()
                .all(|&v| v == 12345.0));
            assert!(read(&panel_projected, spatial * co + 32)[spatial * co..]
                .iter()
                .all(|&v| v == 12345.0));
            assert!(read(&patches, m * k + 32)[m * k..]
                .iter()
                .all(|&v| v == 12345.0));
            assert!(read(&tiled, count + 32)[count..]
                .iter()
                .all(|&v| v == 12345.0));
        }
        for (rows, heads, hd, window) in [
            (1usize, 1usize, 1usize, 1usize),
            (7, 2, 17, 3),
            (33, 3, 33, 32),
            (105, 2, 64, 104),
            (208, 16, 64, 104),
        ] {
            let h = heads * hd;
            let values = |salt: usize| {
                (0..rows * h)
                    .map(|i| (((i * salt + 17) % 257) as f32 - 128.0) / 64.0)
                    .collect::<Vec<_>>()
            };
            let q = upload(&values(13));
            let k = upload(&values(31));
            let v = upload(&values(47));
            let baseline = upload(&vec![12345.0; rows * h + 32]);
            let candidate = upload(&vec![12345.0; rows * h + 32]);
            let params = [rows as u32, h as u32, hd as u32, window as u32, 0];
            let cb = engine.queue.commandBuffer().unwrap();
            engine
                .dispatch(
                    &cb,
                    "asr_attention",
                    [&q, &k, &v],
                    &baseline,
                    params,
                    rows * heads,
                )
                .unwrap();
            engine
                .dispatch(
                    &cb,
                    "asr_attention_simd",
                    [&q, &k, &v],
                    &candidate,
                    params,
                    (rows * heads + 1) * 32,
                )
                .unwrap();
            cb.commit();
            cb.waitUntilCompleted();
            assert_eq!(cb.status(), MTLCommandBufferStatus::Completed);
            assert_eq!(
                read(&baseline, rows * h + 32),
                read(&candidate, rows * h + 32),
                "attention SIMD parity {params:?}"
            );
            assert!(read(&candidate, rows * h + 32)[rows * h..]
                .iter()
                .all(|&v| v == 12345.0));
        }
        for (m, n, k) in [
            (1usize, 1usize, 1usize),
            (7, 13, 11),
            (8, 16, 24),
            (17, 9, 33),
            (16, 32, 32),
            (17, 33, 35),
            (33, 65, 35),
            (65, 65, 67),
            (13, 480, 4320),
        ] {
            let x = upload(
                &(0..m * k)
                    .map(|i| ((i % 17) as f32 - 8.0) / 16.0)
                    .collect::<Vec<_>>(),
            );
            let w = upload(
                &(0..n * k)
                    .map(|i| ((i % 13) as f32 - 6.0) / 16.0)
                    .collect::<Vec<_>>(),
            );
            let packed_w = engine.buffer((n * k).div_ceil(2)).unwrap();
            unsafe {
                let src = std::slice::from_raw_parts(w.contents().as_ptr().cast::<f32>(), n * k);
                let dst = std::slice::from_raw_parts_mut(
                    packed_w.contents().as_ptr().cast::<u16>(),
                    n * k,
                );
                for (dst, src) in dst.iter_mut().zip(src) {
                    *dst = (src.to_bits() >> 16) as u16;
                }
            }
            let b = upload(&(0..n).map(|i| i as f32 / 16.0).collect::<Vec<_>>());
            for bias in [0, 1] {
                for gelu in [0, 1] {
                    let scalar = upload(&vec![12345.0; m * n + 32]);
                    let tiled = upload(&vec![12345.0; m * n + 32]);
                    let wide = upload(&vec![12345.0; m * n + 32]);
                    let large = upload(&vec![12345.0; m * n + 32]);
                    let packed = upload(&vec![12345.0; m * n + 32]);
                    let direct = upload(&vec![12345.0; m * n + 32]);
                    let tile64 = upload(&vec![12345.0; m * n + 32]);
                    let params = [m as u32, n as u32, k as u32, bias, gelu];
                    let cb = engine.queue.commandBuffer().unwrap();
                    engine
                        .dispatch(&cb, "asr_linear", [&x, &w, &b], &scalar, params, m * n)
                        .unwrap();
                    engine
                        .dispatch(
                            &cb,
                            "asr_linear_tiled",
                            [&x, &w, &b],
                            &tiled,
                            params,
                            m.div_ceil(8) * n.div_ceil(8) * 32,
                        )
                        .unwrap();
                    engine
                        .dispatch(
                            &cb,
                            "asr_linear_wide",
                            [&x, &w, &b],
                            &wide,
                            params,
                            m.div_ceil(16) * n.div_ceil(32) * 128,
                        )
                        .unwrap();
                    engine
                        .dispatch(
                            &cb,
                            "asr_linear_large",
                            [&x, &w, &b],
                            &large,
                            params,
                            m.div_ceil(32) * n.div_ceil(64) * 256,
                        )
                        .unwrap();
                    engine
                        .dispatch(
                            &cb,
                            "asr_linear_large_bf16",
                            [&x, &packed_w, &b],
                            &packed,
                            params,
                            m.div_ceil(32) * n.div_ceil(64) * 256,
                        )
                        .unwrap();
                    engine
                        .dispatch(
                            &cb,
                            "asr_linear_direct",
                            [&x, &w, &b],
                            &direct,
                            params,
                            m.div_ceil(32) * n.div_ceil(64) * 256,
                        )
                        .unwrap();
                    engine
                        .dispatch(
                            &cb,
                            "asr_linear_tile64",
                            [&x, &w, &b],
                            &tile64,
                            params,
                            (m.div_ceil(64) * n.div_ceil(64) + 1) * 256,
                        )
                        .unwrap();
                    cb.commit();
                    cb.waitUntilCompleted();
                    assert_eq!(cb.status(), MTLCommandBufferStatus::Completed);
                    assert_eq!(
                        read(&direct, m * n + 32),
                        read(&tile64, m * n + 32),
                        "64x64 parity {params:?}"
                    );
                    assert_eq!(
                        read(&large, m * n + 32),
                        read(&direct, m * n + 32),
                        "direct epilogue parity {params:?}"
                    );
                    assert_eq!(
                        read(&large, m * n + 32),
                        read(&packed, m * n + 32),
                        "BF16 storage parity {params:?}"
                    );
                    assert_eq!(
                        read(&scalar, m * n + 32),
                        read(&tiled, m * n + 32),
                        "tile/tail/bias/GELU parity {params:?}"
                    );
                    assert_eq!(
                        read(&tiled, m * n + 32),
                        read(&wide, m * n + 32),
                        "wide tile/tail/bias/GELU parity {params:?}"
                    );
                    assert_eq!(
                        read(&wide, m * n + 32),
                        read(&large, m * n + 32),
                        "large tile parity {params:?}"
                    );
                }
            }
        }
        assert!(engine.validate_direct_epilogue().unwrap());
        engine.set_direct_epilogue(true).unwrap();
        assert!(engine.direct_epilogue);
        engine.set_direct_epilogue(false).unwrap();
        engine.kernels.remove("asr_linear_direct");
        engine.direct_epilogue_validated = None;
        assert!(!engine.validate_direct_epilogue().unwrap());
        assert!(engine.set_direct_epilogue(true).is_err());
        assert!(!engine.direct_epilogue);
        assert!(engine.validate_tile64().unwrap());
        engine.set_tile64_selective(true).unwrap();
        assert!(engine.tile64 && engine.tile64_selective);
        engine.set_tile64(false).unwrap();
        engine.kernels.remove("asr_linear_tile64");
        engine.tile64_validated = None;
        assert!(!engine.validate_tile64().unwrap());
        assert!(engine.set_tile64(true).is_err());
        assert!(!engine.tile64);
    }
}
