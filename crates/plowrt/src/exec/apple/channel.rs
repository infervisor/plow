use super::*;
use crate::exec::ane::{AneNet, PreparedNet};
use crate::exec::ane_mlp::{bf, mlp_spec, Matrix};
use plow_asset::hetero::{Plan, WeightEncoding};
use plow_asset::hetero_channel::{Span, Weight};

pub(super) const MSL: &str = include_str!("../../../../../runtime/apple/mlp_channel.metal");
type Pipeline = Retained<ProtocolObject<dyn MTLComputePipelineState>>;
type Command = Retained<ProtocolObject<dyn MTLCommandBuffer>>;

#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct Stats {
    pub mlps: usize,
    pub ane_calls: usize,
    pub fallbacks: usize,
    pub disabled: bool,
    pub load_ms: f64,
    pub packed_gpu_bytes: usize,
    pub scratch_bytes: usize,
    pub prediction_ms: f64,
    pub gpu_device_ms: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn completion_publication_preserves_duplicate_successors() {
        let mut b = packet::devbuild::Builder::new(2);
        let x = b.tensor("x", 32);
        let a = b.emit(packet::dev::DevOp::Residual, vec![0, 1], &[], |d| {
            d.t[..3].copy_from_slice(&[x, x, x]);
            d.i[0] = 16;
            d.f[0] = 1.0;
        });
        b.emit(packet::dev::DevOp::Residual, vec![0, 1], &[a], |d| {
            d.t[..3].copy_from_slice(&[x, x, x]);
            d.i[0] = 16;
            d.f[0] = 1.0;
        });
        let model = packet::devbuild::Model {
            n_cu: 2,
            target: 0,
            tensors: b.tensors(),
            progs: vec![b.finish()],
            prog_t: vec![128],
            kv_row_insts: vec![],
            gen: vec![],
        };
        let mut blob = crate::asset::devblob::DevBlob::parse(&model.to_blob()).unwrap();
        let p = &mut blob.progs[0];
        let span = Span {
            layer: 0,
            insts: [0, 2],
            input: "x".into(),
            residual: "r".into(),
            intermediate: "z".into(),
            down_output: "y".into(),
        };
        p.n_counter = p.n_counter.max(1);
        p.succs = vec![0, 0, 0];
        for e in &mut p.stream {
            e.succ_ofs = 0;
            e.succ_len = 3;
        }
        assert_eq!(
            completion_bumps(p, &span).unwrap(),
            vec![(0, p.stream.len() as u32 * 3)]
        );
        p.succs[1] = p.n_counter;
        assert!(completion_bumps(p, &span).is_err());
    }

    #[test]
    fn packing_cache_identity_includes_shape_encoding_and_scales() {
        let w = Matrix {
            n: 32,
            k: 32,
            encoding: 1,
            data: vec![0; 1024],
            scales: vec![0; 128],
        };
        let mut weights = [w.clone(), w.clone(), w];
        let original = cache_key(&weights, 128, b"os1");
        assert_ne!(original, cache_key(&weights, 64, b"os1"));
        assert_ne!(original, cache_key(&weights, 128, b"os2"));
        weights[2].scales[0] = 1;
        assert_ne!(original, cache_key(&weights, 128, b"os1"));
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Failure {
    None,
    BeforeSubmit,
    AfterSubmit,
    AfterJoin,
}

struct Layer {
    span: Span,
    input: usize,
    residual: usize,
    weights: [Buf; 3],
    scales: [Buf; 3],
    io: PreparedNet,
    bumps: Vec<(u32, u32)>,
}

pub struct Channel {
    prog: usize,
    rows: u32,
    min_rows: u32,
    hidden: usize,
    channels: usize,
    encoding: u32,
    layers: Vec<Layer>,
    intermediate: Buf,
    gpu_partial: Buf,
    ane_partial: Buf,
    gate: Pipeline,
    down: Pipeline,
    finish: Pipeline,
    pub stats: Stats,
    pub(super) clen: u32,
    failure: Failure,
}

struct Submitted(Command);
impl Submitted {
    fn join(&self) -> Result<()> {
        self.0.waitUntilCompleted();
        if self.0.status() != MTLCommandBufferStatus::Completed {
            return Err(err("channel command", format!("{:?}", self.0.error())));
        }
        Ok(())
    }
}
impl Drop for Submitted {
    fn drop(&mut self) {
        self.0.waitUntilCompleted();
    }
}

fn completion_bumps(prog: &crate::asset::devblob::DevProg, span: &Span) -> Result<Vec<(u32, u32)>> {
    let mut bumps = std::collections::BTreeMap::<u32, u32>::new();
    for e in prog
        .stream
        .iter()
        .filter(|e| span.insts[0] <= e.inst && e.inst < span.insts[1])
    {
        let lo = e.succ_ofs as usize;
        let hi = lo
            .checked_add(e.succ_len as usize)
            .ok_or_else(|| err("channel counters", "overflow"))?;
        for &counter in prog
            .succs
            .get(lo..hi)
            .ok_or_else(|| err("channel counters", "successor extent"))?
        {
            if counter >= prog.n_counter {
                return Err(err("channel counters", "counter extent"));
            }
            let n = bumps.entry(counter).or_default();
            *n = n
                .checked_add(1)
                .ok_or_else(|| err("channel counters", "count overflow"))?;
        }
    }
    Ok(bumps.into_iter().collect())
}

fn packed(engine: &MetalEngine, weight: &Weight, encoding: u32) -> (Matrix, Matrix) {
    let bytes = |name: &str| {
        engine
            .tensor_bytes(engine.model.names.iter().position(|n| n == name).unwrap())
            .to_vec()
    };
    let source = Matrix {
        n: weight.rows as usize,
        k: weight.cols as usize,
        encoding,
        data: bytes(&weight.tensor),
        scales: weight.scale.as_deref().map(bytes).unwrap_or_default(),
    };
    let slice = |s: &plow_asset::hetero_channel::Slice| {
        source.slice(
            s.rows[0] as usize..s.rows[1] as usize,
            s.cols[0] as usize..s.cols[1] as usize,
        )
    };
    (slice(&weight.gpu), slice(&weight.ane))
}

fn cache_key(weights: &[Matrix; 3], rows: u32, os: &[u8]) -> String {
    use plow_asset::mixed_step::payload_sha256 as hash;
    let mut identity = format!(
        "channel-mlp-v1-{rows}-{}-{}",
        hash(os),
        hash(include_bytes!("../ane_mlp.rs"))
    );
    identity.push_str(&hash(include_bytes!("../ane.rs")));
    for w in weights {
        identity.push_str(&format!(
            "-{}-{}-{}-{}-{}",
            w.n,
            w.k,
            w.encoding,
            hash(&w.data),
            hash(&w.scales)
        ));
    }
    hash(identity.as_bytes())
}

fn require_placement(inspector: &Path, compiled: &Path, layers: usize) -> Result<()> {
    let output = std::process::Command::new(inspector)
        .arg(compiled)
        .output()
        .map_err(|e| err("channel placement inspector", e))?;
    let text = String::from_utf8_lossy(&output.stdout);
    if !output.status.success()
        || text.lines().count() != layers
        || !text
            .lines()
            .all(|s| s.ends_with("preferred=MLNeuralEngineComputeDevice"))
    {
        return Err(err(
            "channel placement",
            format!(
                "graph not wholly ANE: {text} {}",
                String::from_utf8_lossy(&output.stderr)
            ),
        ));
    }
    Ok(())
}

impl Channel {
    pub(super) fn load(
        engine: &MetalEngine,
        lib: &ProtocolObject<dyn MTLLibrary>,
        blob: &Path,
    ) -> Result<Self> {
        let start = Instant::now();
        let bytes = std::fs::read(blob.with_file_name(plow_asset::hetero::FILE))
            .map_err(|e| err("channel sidecar", e))?;
        let Plan::Channel(plan) =
            plow_asset::hetero::parse(&bytes).map_err(|e| err("channel sidecar", e))?
        else {
            return Err(err("channel sidecar", "v3 channel plan required"));
        };
        engine
            .model
            .blob
            .with_packet_view(|p| plan.validate(p))
            .map_err(|e| err("channel plan", e))?;
        if plan.programs.len() != 1 {
            return Err(err("channel plan", "one 128-row program required"));
        }
        let config = &crate::config::RuntimeConfig::get().apple;
        let count = config.ane_mlp_layers.unwrap_or(plan.layers.len());
        if count == 0 || count > plan.layers.len() {
            return Err(err("channel layers", "out of range"));
        }
        let inspector = config.ane_mlp_placement.as_ref().ok_or_else(|| {
            err(
                "channel placement",
                "set PLOW_ANE_MLP_PLACEMENT to the built ane_placement probe",
            )
        })?;
        let cache = config
            .ane_mlp_cache
            .clone()
            .unwrap_or_else(|| blob.with_file_name("ane-channel-cache"));
        let os = std::process::Command::new("/usr/bin/sw_vers")
            .output()
            .map_err(|e| err("channel OS identity", e))?;
        if !os.status.success() {
            return Err(err("channel OS identity", "sw_vers failed"));
        }
        let device = &*engine._device;
        let buffer = |size| {
            device
                .newBufferWithLength_options(size, MTLResourceOptions::StorageModeShared)
                .ok_or_else(|| err("channel buffer", "allocation failed"))
        };
        let pipeline = |name| {
            let f = lib
                .newFunctionWithName(&NSString::from_str(name))
                .ok_or_else(|| err("channel kernel", name))?;
            device
                .newComputePipelineStateWithFunction_error(&f)
                .map_err(|e| err("channel pipeline", e))
        };
        let pp = &plan.programs[0];
        let (h, g, rows) = (
            plan.hidden as usize,
            (plan.inter - plan.ane_channels) as usize,
            pp.rows as usize,
        );
        let encoding = match plan.weight_encoding {
            WeightEncoding::Bf16 => 0,
            WeightEncoding::Fp8 => 1,
            WeightEncoding::Mxfp4 => 2,
        };
        let mut result = Self {
            prog: pp.prog as usize,
            rows: pp.rows,
            min_rows: pp.min_rows,
            hidden: h,
            channels: g,
            encoding,
            layers: Vec::with_capacity(count),
            intermediate: buffer(rows * g * 2)?,
            gpu_partial: buffer(rows * h * 4)?,
            ane_partial: buffer(rows * h * 4)?,
            gate: pipeline("mlp_gate")?,
            down: pipeline("mlp_down")?,
            finish: pipeline("mlp_finish")?,
            stats: Stats {
                scratch_bytes: rows * (g * 2 + h * 8) + count * rows * h * 8,
                ..Stats::default()
            },
            clen: 0,
            failure: match config.ane_mlp_fail.as_deref() {
                None => Failure::None,
                Some("before_submit") => Failure::BeforeSubmit,
                Some("after_submit") => Failure::AfterSubmit,
                Some("after_join") => Failure::AfterJoin,
                _ => return Err(err("channel failure injection", "unknown stage")),
            },
        };
        for (weights, span) in plan.layers.iter().zip(&pp.spans).take(count) {
            let pairs =
                [&weights.gate, &weights.up, &weights.down].map(|w| packed(engine, w, encoding));
            let [(gg, ag), (gu, au), (gd, ad)] = pairs;
            let gpu = [gg, gu, gd];
            let ane = [ag, au, ad];
            let name = cache_key(&ane, pp.call_rows, &os.stdout);
            let spec = std::panic::catch_unwind(|| {
                mlp_spec(pp.call_rows as usize, &ane[0], &ane[1], &ane[2], 0)
            })
            .map_err(|_| err("channel weights", "invalid FP16 weights"))?;
            let layers = spec.layers.len();
            let io = (spec.inputs.clone(), spec.outputs.clone());
            let net = AneNet::new(
                &cache,
                &name,
                || spec,
                io,
                objc2_core_ml::MLComputeUnits::CPUAndNeuralEngine,
            )?;
            require_placement(&inspector, &cache.join(format!("{name}.mlmodelc")), layers)?;
            let io = net.prepare_io(pp.call_rows as usize)?;
            let upload = |w: &Matrix| shared_from(device, &w.data);
            let scale = |w: &Matrix| {
                shared_from(
                    device,
                    if w.scales.is_empty() {
                        &[0; 4]
                    } else {
                        &w.scales
                    },
                )
            };
            result.stats.packed_gpu_bytes += gpu
                .iter()
                .map(|w| w.data.len() + w.scales.len())
                .sum::<usize>();
            result.layers.push(Layer {
                span: span.clone(),
                input: engine
                    .model
                    .names
                    .iter()
                    .position(|n| n == &span.input)
                    .unwrap(),
                residual: engine
                    .model
                    .names
                    .iter()
                    .position(|n| n == &span.residual)
                    .unwrap(),
                weights: [upload(&gpu[0])?, upload(&gpu[1])?, upload(&gpu[2])?],
                scales: [scale(&gpu[0])?, scale(&gpu[1])?, scale(&gpu[2])?],
                io,
                bumps: completion_bumps(&engine.model.blob.progs[result.prog], span)?,
            });
        }
        result.stats.load_ms = start.elapsed().as_secs_f64() * 1e3;
        tracing::info!(
            layers = count,
            channels = plan.ane_channels,
            load_ms = result.stats.load_ms,
            "experimental channel MLP ready; automatic policy remains disabled"
        );
        Ok(result)
    }

    pub(super) fn eligible(&self, prog: usize) -> bool {
        !self.stats.disabled
            && prog == self.prog
            && self.min_rows <= self.clen
            && self.clen <= self.rows
    }

    fn encode(
        &self,
        cb: &Command,
        pipeline: &Pipeline,
        buffers: &[&Buf],
        params: &[u32],
        groups: usize,
        threads: usize,
    ) -> Result<()> {
        let enc = cb
            .computeCommandEncoder()
            .ok_or_else(|| err("channel encoder", "allocation failed"))?;
        enc.setComputePipelineState(pipeline);
        unsafe {
            for (i, b) in buffers.iter().enumerate() {
                enc.setBuffer_offset_atIndex(Some(b), 0, i);
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
        Ok(())
    }

    fn submit(&self, engine: &MetalEngine, layer: usize) -> Result<Submitted> {
        let l = &self.layers[layer];
        let cb = engine
            .queue
            .commandBuffer()
            .ok_or_else(|| err("channel command", "allocation failed"))?;
        let p = [
            self.clen,
            self.hidden as u32,
            self.channels as u32,
            self.encoding,
        ];
        self.encode(
            &cb,
            &self.gate,
            &[
                &engine.bufs[l.input],
                &l.weights[0],
                &l.weights[1],
                &l.scales[0],
                &l.scales[1],
                &self.intermediate,
            ],
            &p,
            16,
            THREADS,
        )?;
        self.encode(
            &cb,
            &self.down,
            &[
                &self.intermediate,
                &l.weights[2],
                &l.scales[2],
                &self.gpu_partial,
            ],
            &p,
            16,
            THREADS,
        )?;
        cb.commit();
        Ok(Submitted(cb))
    }

    fn run_layer(&mut self, engine: &mut MetalEngine, index: usize) -> Result<bool> {
        if self.failure == Failure::BeforeSubmit {
            return Ok(false);
        }
        let cb = self.submit(engine, index)?;
        let l = &mut self.layers[index];
        let count = self.clen as usize * self.hidden;
        let mut timing = crate::exec::ane::RunTimings::default();
        let predict = if self.failure == Failure::AfterSubmit {
            Err(err("channel prediction", "injected after GPU submit"))
        } else {
            // The producer boundary has joined. Neither lane writes input or residual.
            let x = unsafe {
                std::slice::from_raw_parts(engine.host_ptr(l.input) as *const u16, count)
            };
            let input = l.io.input_mut(0);
            for (dst, &src) in input[..count].iter_mut().zip(x) {
                *dst = bf(src);
            }
            input[count..].fill(0.0);
            self.stats.ane_calls += 1;
            objc2::rc::autoreleasepool(|_| l.io.run(engine.profile.as_ref().map(|_| &mut timing)))
        };
        // Even a prediction failure must join GPU work before buffers can be reused.
        cb.join()?;
        if engine.profile.is_some() {
            self.stats.prediction_ms += timing.prediction_ms;
            self.stats.gpu_device_ms += (cb.0.GPUEndTime() - cb.0.GPUStartTime()).max(0.0) * 1e3;
        }
        if let Err(e) = predict {
            tracing::warn!(error = %e, "channel MLP falling back");
            return Ok(false);
        }
        if self.failure == Failure::AfterJoin {
            return Ok(false);
        }
        let output = l.io.output(0);
        let gpu = unsafe {
            std::slice::from_raw_parts(self.gpu_partial.contents().as_ptr() as *const f32, count)
        };
        if !output[..count].iter().chain(gpu).all(|v| v.is_finite()) {
            return Ok(false);
        }
        unsafe {
            std::ptr::copy_nonoverlapping(
                output.as_ptr(),
                self.ane_partial.contents().as_ptr() as *mut f32,
                count,
            );
        }
        let residual = l.residual;
        let finish = engine
            .queue
            .commandBuffer()
            .ok_or_else(|| err("channel reduction", "allocation failed"))?;
        self.encode(
            &finish,
            &self.finish,
            &[
                &self.gpu_partial,
                &self.ane_partial,
                &engine.bufs[residual],
                &engine.bufs[residual],
            ],
            &[count as u32, 1, 0, 0],
            count.div_ceil(256),
            256,
        )?;
        finish.commit();
        let finish = Submitted(finish);
        finish.join()?;
        if engine.profile.is_some() {
            self.stats.gpu_device_ms +=
                (finish.0.GPUEndTime() - finish.0.GPUStartTime()).max(0.0) * 1e3;
        }
        // This includes internal fine/coarse successor multiplicities, not one bump per op.
        unsafe {
            let counters = engine.progs[self.prog].ctr.contents().as_ptr() as *mut u32;
            for &(counter, n) in &self.layers[index].bumps {
                *counters.add(counter as usize) += n;
            }
        }
        self.stats.mlps += 1;
        Ok(true)
    }

    pub(super) fn run(&mut self, engine: &mut MetalEngine) -> Result<()> {
        let mut lo = 0;
        let p = self.prog;
        for index in 0..self.layers.len() {
            let [start, end] = self.layers[index].span.insts;
            engine.dispatch_range(p, lo, start)?;
            engine.check_fault(p)?;
            if !self.run_layer(engine, index)? {
                self.stats.fallbacks += 1;
                self.stats.disabled = true;
                // No partial or counter was published: execute the complete original MLP and suffix.
                engine.dispatch_range(p, start, engine.progs[p].insts_host.len() as u32)?;
                return engine.check_fault(p);
            }
            lo = end;
        }
        engine.dispatch_range(p, lo, engine.progs[p].insts_host.len() as u32)?;
        engine.check_fault(p)
    }
}
