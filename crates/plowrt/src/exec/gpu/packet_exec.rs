//! A self-contained packet (weights embedded, no language-model step protocol) on the CUDA
//! interpreter: the `PacketRuntime` for forward pipelines such as the Qwen3-ASR audio encoder.
//! Every program runs as one cooperative launch per global-queue segment of the speech object
//! (`interp_sm90a_speech.cubin`, the interpreter with the FP32 speech arms).

use std::path::Path;
use std::sync::Arc;

use packet::dev::{DevProgram, CTR_STRIDE};

use super::{pod_bytes, BLOCK};
use crate::asset::devblob::DevBlob;
use crate::device::cuda::{CudaBackend, CudaEvent, CudaStream, GraphExec, KernelFn};
use crate::device::{Backend, DeviceMem, Module};
use crate::exec::packet_runtime::{check_copy, check_transfer, PacketRuntime, PacketTensor};
use crate::{Result, RuntimeError};

const OBJECT: &str = "interp_sm90a_speech.cubin";
const SYMBOL: &str = "_Z19interp_sm90a_speech11PlowProgram";

struct Program {
    kernarg: DevProgram,
    segments: usize,
    counters: u64,
    counter_bytes: usize,
    _tables: Vec<DeviceMem>,
}

pub struct CudaPacketRuntime {
    be: Arc<CudaBackend>,
    stream: CudaStream,
    events: (CudaEvent, CudaEvent),
    module: Module,
    function: KernelFn,
    grid: u32,
    smem: u32,
    names: Vec<String>,
    tensors: Vec<DeviceMem>,
    _table: DeviceMem,
    /// Every program's counters and cursors, in program order: a run of consecutive programs
    /// zeroes its counters with one memset.
    _counters: DeviceMem,
    programs: Vec<Program>,
    /// One CUDA graph per program sequence (every pointer is fixed at load, so a capture stays
    /// valid); `None` once capture failed for that sequence.
    graphs: std::collections::HashMap<Vec<usize>, Option<GraphExec>>,
    last_us: f64,
}

// SAFETY: the runtime owns its stream, module and allocations and is used by one thread at a time
// (`PacketRuntime: Send`, every method takes the runtime by reference).
unsafe impl Send for CudaPacketRuntime {}

impl CudaPacketRuntime {
    pub fn load(path: &Path, device: u8) -> Result<Self> {
        let be = Arc::new(CudaBackend::new(device)?);
        Self::load_on(be, path)
    }

    pub fn load_on(be: Arc<CudaBackend>, path: &Path) -> Result<Self> {
        let raw = std::fs::read(path).map_err(|source| RuntimeError::Io { path: path.to_path_buf(), source })?;
        let blob = DevBlob::parse(&raw)?;
        if !blob.gen.is_empty() || blob.tp.is_some() {
            return Err(RuntimeError::Rejected(format!("{}: generated tensors or TP are not supported by the CUDA packet runtime", path.display())));
        }
        let dir = path.parent().unwrap_or(Path::new("."));
        let object = dir.join(OBJECT);
        let image = std::fs::read(&object).map_err(|source| RuntimeError::Io { path: object.clone(), source })?;
        let module = be.module_load(&image)?;
        let function = be.get_function(&module, SYMBOL)?;
        let smem = be.module_global_u32(&module, "plow_arena_bytes_speech")?.unwrap_or(49152);
        if let Some(limit) = crate::config::RuntimeConfig::debug_max_inst() {
            if be.module_global_set_u32(&module, "plow_debug_max_inst_speech", limit)? {
                tracing::warn!(limit, "packet runtime: instructions at or past the limit are skipped");
            }
        }
        be.set_max_dynamic_smem(function, smem)?;
        let capacity = be.occupancy_blocks_per_sm(function, BLOCK, smem as usize)? * be.sm_count();
        if blob.n_cu == 0 || blob.n_cu > capacity {
            return Err(RuntimeError::Rejected(format!("{}: n_cu {} exceeds cooperative capacity {capacity}", path.display(), blob.n_cu)));
        }

        let mut tensors = Vec::with_capacity(blob.tensors.len());
        for t in &blob.tensors {
            let mem = be.alloc(0, t.bytes.max(4))?;
            match &t.init {
                Some(r) => be.upload(&mem, 0, &blob.init[r.clone()])?,
                None => be.memset_d8(mem.base, 0, t.bytes as usize)?,
            }
            tensors.push(mem);
        }
        let ptrs: Vec<u64> = tensors.iter().map(|m| m.base).collect();
        let table = be.alloc(0, (ptrs.len() * 8).max(8) as u64)?;
        be.upload(&table, 0, pod_bytes(&ptrs))?;

        let regions: Vec<(usize, usize)> = blob
            .progs
            .iter()
            .map(|p| {
                let segments = p.gq_seg_ofs.len().saturating_sub(1);
                ((p.n_counter as usize * CTR_STRIDE as usize * 4).max(4), segments * CTR_STRIDE as usize * 4)
            })
            .collect();
        let slab_bytes: usize = regions.iter().map(|&(c, k)| (c + k).next_multiple_of(256)).sum();
        let counter_slab = be.alloc(0, slab_bytes.max(4) as u64)?;
        let mut slab_offset = 0u64;
        let mut programs = Vec::with_capacity(blob.progs.len());
        for (p, &(counter_only, cursor_bytes)) in blob.progs.iter().zip(&regions) {
            if p.gq_seg_ofs.len() < 2 || p.l2_domains != 0 {
                return Err(RuntimeError::Rejected(format!("{}: program needs an unplaced global-queue stream", path.display())));
            }
            let segments = p.gq_seg_ofs.len() - 1;
            let upload = |bytes: &[u8]| -> Result<DeviceMem> {
                let mem = be.alloc(0, bytes.len().max(4) as u64)?;
                if !bytes.is_empty() {
                    be.upload(&mem, 0, bytes)?;
                }
                Ok(mem)
            };
            let d_inst = upload(pod_bytes(&p.insts))?;
            let d_stream = upload(pod_bytes(&p.stream))?;
            let d_sofs = upload(pod_bytes(&p.stream_ofs))?;
            let d_slen = upload(pod_bytes(&p.stream_len))?;
            let d_waits = upload(pod_bytes(&p.waits))?;
            let d_succs = upload(pod_bytes(&p.succs))?;
            let d_gq_stream = upload(pod_bytes(&p.gq_stream))?;
            let d_gq_seg = upload(pod_bytes(&p.gq_seg_ofs))?;
            let counters = counter_slab.base + slab_offset;
            let cursors = counters + counter_only as u64;
            slab_offset += (counter_only + cursor_bytes).next_multiple_of(256) as u64;
            let kernarg = DevProgram {
                insts: d_inst.base,
                stream: d_stream.base,
                stream_ofs: d_sofs.base,
                stream_len: d_slen.base,
                waits: d_waits.base,
                succs: d_succs.base,
                counters,
                tensors: table.base,
                trace: 0,
                cur_seg: 0,
                l2_domains: 0,
                hier_base: 0,
                n_seg: 1,
                gq_stream: d_gq_stream.base,
                gq_seg_ofs: d_gq_seg.base,
                gq_cursor: cursors,
                xctr: 0,
                peer_scratch: 0,
                rank: 0,
                n_gpu: 1,
                seg_ofs: 0,
                prefill_spans: 0,
                prefill_parked: 0,
                n_prefill_spans: 0,
                n_prefill_rows: 0,
                token_batch: 0,
            };
            programs.push(Program {
                kernarg,
                segments,
                counters,
                counter_bytes: counter_only + cursor_bytes,
                _tables: vec![d_inst, d_stream, d_sofs, d_slen, d_waits, d_succs, d_gq_stream, d_gq_seg],
            });
        }
        let stream = be.stream_create()?;
        let events = (be.event_create(true)?, be.event_create(true)?);
        tracing::info!(packet = %path.display(), programs = programs.len(), tensors = tensors.len(), grid = blob.n_cu, smem, "cuda packet runtime loaded");
        Ok(CudaPacketRuntime {
            be,
            stream,
            events,
            module,
            function,
            grid: blob.n_cu,
            smem,
            names: blob.tensors.iter().map(|t| t.name.clone()).collect(),
            tensors,
            _table: table,
            _counters: counter_slab,
            programs,
            graphs: std::collections::HashMap::new(),
            last_us: 0.0,
        })
    }

    /// Skip every instruction at or past `limit` (per-instruction cost by caps).
    pub fn set_debug_max_inst(&self, limit: u32) -> Result<bool> {
        self.be.module_global_set_u32(&self.module, "plow_debug_max_inst_speech", limit)
    }

    pub fn backend(&self) -> &Arc<CudaBackend> {
        &self.be
    }

    /// Device address of a tensor, for device-to-device handoff to another engine on the same GPU.
    pub fn device_ptr(&self, tensor: PacketTensor) -> Option<u64> {
        self.tensors.get(tensor.handle).map(|m| m.base)
    }

    fn enqueue(&self, programs: &[usize]) -> Result<()> {
        let consecutive = programs.windows(2).all(|w| w[1] == w[0] + 1);
        if let (true, Some(&first), Some(&last)) = (consecutive, programs.first(), programs.last()) {
            let (first, last) = (&self.programs[first], &self.programs[last]);
            let bytes = (last.counters + last.counter_bytes as u64 - first.counters) as usize;
            self.be.memset_d8_async(first.counters, 0, bytes, &self.stream)?;
        }
        for &program in programs {
            let p = &self.programs[program];
            if !consecutive {
                self.be.memset_d8_async(p.counters, 0, p.counter_bytes, &self.stream)?;
            }
            for seg in 0..p.segments {
                let mut arg = p.kernarg;
                arg.gq_seg_ofs += (seg * 4) as u64;
                arg.gq_cursor += (seg * CTR_STRIDE as usize * 4) as u64;
                let mut params = [&mut arg as *mut DevProgram as *mut std::ffi::c_void];
                self.be.launch_cooperative(self.function, self.grid, BLOCK, self.smem, &mut params, Some(&self.stream))?;
            }
        }
        Ok(())
    }

    fn mem(&self, t: PacketTensor) -> Result<&DeviceMem> {
        self.tensors.get(t.handle).ok_or_else(|| RuntimeError::Device(format!("packet tensor handle {} is missing", t.handle)))
    }
}

impl PacketRuntime for CudaPacketRuntime {
    fn tensor(&self, name: &str) -> Option<PacketTensor> {
        let handle = self.names.iter().position(|n| n == name)?;
        Some(PacketTensor { handle, bytes: self.tensors[handle].len as usize })
    }

    fn write_tensor(&mut self, tensor: PacketTensor, bytes: &[u8]) -> Result<()> {
        let mem = self.mem(tensor)?;
        check_transfer(tensor, tensor.handle, mem.len as usize, bytes.len())?;
        self.be.upload(mem, 0, bytes)
    }

    fn read_tensor(&self, tensor: PacketTensor, bytes: &mut [u8]) -> Result<()> {
        let mem = self.mem(tensor)?;
        check_transfer(tensor, tensor.handle, mem.len as usize, bytes.len())?;
        self.be.download(mem, 0, bytes)
    }

    fn write_tensor_at(&mut self, tensor: PacketTensor, offset: usize, bytes: &[u8]) -> Result<()> {
        let mem = self.mem(tensor)?;
        check_transfer(tensor, tensor.handle, mem.len as usize, mem.len as usize)?;
        self.be.upload(mem, offset as u64, bytes)
    }

    fn read_tensor_at(&self, tensor: PacketTensor, offset: usize, bytes: &mut [u8]) -> Result<()> {
        let mem = self.mem(tensor)?;
        check_transfer(tensor, tensor.handle, mem.len as usize, mem.len as usize)?;
        self.be.download(mem, offset as u64, bytes)
    }

    fn copy_tensor(
        &mut self,
        source: PacketTensor,
        source_offset: usize,
        target: PacketTensor,
        target_offset: usize,
        bytes: usize,
    ) -> Result<()> {
        let (s, t) = (self.mem(source)?, self.mem(target)?);
        check_copy(source, s.len as usize, source_offset, target, t.len as usize, target_offset, bytes)?;
        self.be.memcpy_dtod(t.base + target_offset as u64, s.base + source_offset as u64, bytes as u64)
    }

    fn run(&mut self, program: usize) -> Result<()> {
        self.run_sequence(&[program])
    }

    /// Every program is enqueued on the one stream (stream order is the dependency), as one
    /// CUDA graph after the first run of a sequence; one sync.
    fn run_sequence(&mut self, programs: &[usize]) -> Result<()> {
        if let Some(&bad) = programs.iter().find(|&&p| p >= self.programs.len()) {
            return Err(RuntimeError::Rejected(format!("packet program {bad} is missing")));
        }
        if !self.graphs.contains_key(programs) {
            let graph = match self.be.graph_capture(&self.stream, || self.enqueue(programs)) {
                Ok(g) => Some(g),
                Err(error) => {
                    tracing::warn!(%error, programs = programs.len(), "packet sequence graph capture failed; launching directly");
                    None
                }
            };
            self.graphs.insert(programs.to_vec(), graph);
        }
        self.be.event_record(&self.events.0, &self.stream)?;
        match self.graphs.get(programs).and_then(Option::as_ref) {
            Some(graph) => self.be.graph_launch(graph, &self.stream)?,
            None => self.enqueue(programs)?,
        }
        self.be.event_record(&self.events.1, &self.stream)?;
        self.be.stream_synchronize(&self.stream)?;
        self.last_us = f64::from(self.be.event_elapsed_ms(&self.events.0, &self.events.1)?) * 1e3;
        Ok(())
    }

    fn last_run_us(&self) -> f64 {
        self.last_us
    }
}
