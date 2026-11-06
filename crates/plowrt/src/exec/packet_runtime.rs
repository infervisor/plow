//! Backend-neutral interface for executing a compiled packet asset.

use std::collections::BTreeMap;
use std::path::Path;

use plow_asset::packet_pipeline::{PacketPipeline, PacketPipelines, SECTION};

use crate::{Result, RuntimeError};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PacketTensor {
    pub handle: usize,
    pub bytes: usize,
}

enum BufferStorage {
    Host(Vec<u8>),
    #[cfg(feature = "cuda")]
    Cuda {
        memory: std::sync::Arc<crate::device::DeviceMem>,
        backend: std::sync::Arc<crate::device::cuda::CudaBackend>,
        _resident: std::sync::Arc<()>,
    },
}

/// Immutable bytes whose producing work has completed, independent of the source runtime.
#[derive(Clone)]
pub struct PacketBuffer {
    storage: std::sync::Arc<BufferStorage>,
    offset: usize,
    bytes: usize,
}

impl std::fmt::Debug for PacketBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PacketBuffer")
            .field("bytes", &self.bytes)
            .finish()
    }
}

impl From<Vec<u8>> for PacketBuffer {
    fn from(bytes: Vec<u8>) -> Self {
        let len = bytes.len();
        Self {
            storage: std::sync::Arc::new(BufferStorage::Host(bytes)),
            offset: 0,
            bytes: len,
        }
    }
}

impl PacketBuffer {
    pub fn len(&self) -> usize {
        self.bytes
    }

    pub fn is_empty(&self) -> bool {
        self.bytes == 0
    }

    pub fn slice(&self, range: std::ops::Range<usize>) -> Result<Self> {
        if range.start > range.end || range.end > self.bytes {
            return Err(RuntimeError::Device(
                "packet buffer slice is outside its bounds".into(),
            ));
        }
        Ok(Self {
            storage: self.storage.clone(),
            offset: self.offset + range.start,
            bytes: range.end - range.start,
        })
    }

    pub fn read(&self, output: &mut [u8]) -> Result<()> {
        if output.len() != self.bytes {
            return Err(RuntimeError::Device(
                "packet buffer read size changed".into(),
            ));
        }
        if output.is_empty() {
            return Ok(());
        }
        match self.storage.as_ref() {
            BufferStorage::Host(data) => {
                output.copy_from_slice(&data[self.offset..self.offset + self.bytes])
            }
            #[cfg(feature = "cuda")]
            BufferStorage::Cuda {
                memory, backend, ..
            } => {
                use crate::device::Backend;
                backend.download(memory, self.offset as u64, output)?;
            }
        }
        Ok(())
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn completed_cuda(
        memory: crate::device::DeviceMem,
        backend: std::sync::Arc<crate::device::cuda::CudaBackend>,
        bytes: usize,
        resident: std::sync::Arc<()>,
    ) -> Self {
        assert!(memory.len >= bytes as u64);
        Self {
            storage: std::sync::Arc::new(BufferStorage::Cuda {
                memory: std::sync::Arc::new(memory),
                backend,
                _resident: resident,
            }),
            offset: 0,
            bytes,
        }
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn cuda_source(&self) -> Option<(&crate::device::cuda::CudaBackend, u64)> {
        match self.storage.as_ref() {
            BufferStorage::Cuda {
                memory, backend, ..
            } => Some((backend, memory.base + self.offset as u64)),
            _ => None,
        }
    }
}

#[cfg(feature = "cuda")]
pub(crate) struct PacketBufferPool {
    slots: Vec<PacketBuffer>,
    next: usize,
}

#[cfg(feature = "cuda")]
pub(crate) struct PendingPacketBuffer {
    buffer: Option<PacketBuffer>,
    source: Option<std::sync::Arc<dyn Send + Sync>>,
}

#[cfg(feature = "cuda")]
impl PacketBufferPool {
    pub(crate) fn cuda(
        backend: std::sync::Arc<crate::device::cuda::CudaBackend>,
        slot_bytes: usize,
        slots: usize,
    ) -> Result<Self> {
        use crate::device::Backend;
        let bytes = slot_bytes
            .checked_mul(slots)
            .filter(|&n| n > 0)
            .ok_or_else(|| RuntimeError::Device("packet buffer pool size overflow".into()))?;
        let memory = std::sync::Arc::new(backend.alloc(0, bytes as u64)?);
        let resident = std::sync::Arc::new(());
        Ok(Self {
            slots: (0..slots)
                .map(|i| PacketBuffer {
                    storage: std::sync::Arc::new(BufferStorage::Cuda {
                        memory: memory.clone(),
                        backend: backend.clone(),
                        _resident: resident.clone(),
                    }),
                    offset: i * slot_bytes,
                    bytes: slot_bytes,
                })
                .collect(),
            next: 0,
        })
    }

    pub(crate) fn acquire(
        &mut self,
        bytes: usize,
        source: std::sync::Arc<dyn Send + Sync>,
    ) -> Option<PendingPacketBuffer> {
        for _ in 0..self.slots.len() {
            let i = self.next;
            self.next = (i + 1) % self.slots.len();
            let slot = &self.slots[i];
            if bytes <= slot.bytes && std::sync::Arc::strong_count(&slot.storage) == 1 {
                let mut buffer = slot.clone();
                buffer.bytes = bytes;
                return Some(PendingPacketBuffer {
                    buffer: Some(buffer),
                    source: Some(source),
                });
            }
        }
        None
    }
}

#[cfg(feature = "cuda")]
impl PendingPacketBuffer {
    pub(crate) fn address(&self) -> u64 {
        self.buffer.as_ref().unwrap().cuda_source().unwrap().1
    }

    /// Safety: all work addressing this slot and its source has completed successfully.
    pub(crate) unsafe fn retire(mut self) -> PacketBuffer {
        self.source.take();
        self.buffer.take().unwrap()
    }
}

#[cfg(feature = "cuda")]
impl Drop for PendingPacketBuffer {
    fn drop(&mut self) {
        // Dropping an unretired copy cannot authorize freeing either DMA endpoint.
        if let Some(buffer) = self.buffer.take() {
            std::mem::forget(buffer);
            std::mem::forget(self.source.take());
        }
    }
}

pub(crate) fn check_region(tensor: PacketTensor, offset: usize, bytes: usize) -> Result<()> {
    if offset
        .checked_add(bytes)
        .is_none_or(|end| end > tensor.bytes)
    {
        return Err(RuntimeError::Device(
            "packet tensor transfer is outside its bounds".into(),
        ));
    }
    Ok(())
}

#[derive(Clone, Copy, Debug)]
pub enum PacketData<'a> {
    Host(&'a [u8]),
    Buffer(&'a PacketBuffer),
    Zero(usize),
}

impl PacketData<'_> {
    pub fn len(&self) -> usize {
        match self {
            Self::Host(bytes) => bytes.len(),
            Self::Buffer(buffer) => buffer.len(),
            Self::Zero(bytes) => *bytes,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub(crate) fn read(&self, output: &mut [u8]) -> Result<()> {
        if output.len() != self.len() {
            return Err(RuntimeError::Device("packet input size changed".into()));
        }
        match self {
            Self::Host(bytes) => output.copy_from_slice(bytes),
            Self::Buffer(buffer) => buffer.read(output)?,
            Self::Zero(_) => output.fill(0),
        }
        Ok(())
    }
}

pub struct PacketWrite<'a> {
    pub tensor: PacketTensor,
    pub offset: usize,
    pub data: PacketData<'a>,
}

impl PacketWrite<'_> {
    pub(crate) fn check(&self) -> Result<()> {
        check_region(self.tensor, self.offset, self.data.len())
    }
}

/// Runtime surface used by packetized pipelines.
///
/// Model import, graph lowering and tensor-name conventions stay outside this interface. The
/// executor only resolves tensors from the compiled asset and dispatches a numbered program.
pub trait PacketRuntime: Send {
    fn begin_execution(&mut self) -> Result<()> {
        Ok(())
    }
    fn end_execution(&mut self) -> Result<()> {
        Ok(())
    }
    fn tensor(&self, name: &str) -> Option<PacketTensor>;
    fn write_tensor(&mut self, tensor: PacketTensor, bytes: &[u8]) -> Result<()>;
    /// Reserve transfer staging for full writes of these tensors before serving.
    fn prepare_writes(&mut self, _tensors: &[PacketTensor]) -> Result<()> {
        Ok(())
    }
    /// Ordered writes, completed before return. Overlapping writes retain slice order.
    fn write_tensors(&mut self, writes: &[PacketWrite<'_>]) -> Result<()> {
        for w in writes {
            w.check()?;
        }
        for w in writes {
            let owned;
            let bytes = if let PacketData::Host(bytes) = w.data {
                bytes
            } else {
                owned = {
                    let mut bytes = vec![0; w.data.len()];
                    w.data.read(&mut bytes)?;
                    bytes
                };
                &owned
            };
            if w.offset == 0 && bytes.len() == w.tensor.bytes {
                self.write_tensor(w.tensor, bytes)?;
            } else {
                self.write_tensor_at(w.tensor, w.offset, bytes)?;
            }
        }
        Ok(())
    }
    fn read_tensor(&self, tensor: PacketTensor, bytes: &mut [u8]) -> Result<()>;
    fn snapshot_tensor(
        &mut self,
        tensor: PacketTensor,
        offset: usize,
        bytes: usize,
    ) -> Result<PacketBuffer> {
        check_region(tensor, offset, bytes)?;
        let mut data = vec![0; bytes];
        self.read_tensor_at(tensor, offset, &mut data)?;
        Ok(data.into())
    }
    fn write_tensor_from(
        &mut self,
        tensor: PacketTensor,
        offset: usize,
        source: &PacketBuffer,
    ) -> Result<()> {
        self.write_tensors(&[PacketWrite {
            tensor,
            offset,
            data: PacketData::Buffer(source),
        }])
    }
    /// Write `bytes` at byte `offset` of `tensor`; the rest keeps its contents.
    fn write_tensor_at(&mut self, tensor: PacketTensor, offset: usize, bytes: &[u8]) -> Result<()> {
        let mut all = vec![0; tensor.bytes];
        self.read_tensor(tensor, &mut all)?;
        all.get_mut(offset..offset.saturating_add(bytes.len()))
            .ok_or_else(|| {
                RuntimeError::Device("packet tensor write is outside its bounds".into())
            })?
            .copy_from_slice(bytes);
        self.write_tensor(tensor, &all)
    }
    /// Read `bytes.len()` bytes from byte `offset` of `tensor`.
    fn read_tensor_at(&self, tensor: PacketTensor, offset: usize, bytes: &mut [u8]) -> Result<()> {
        let mut all = vec![0; tensor.bytes];
        self.read_tensor(tensor, &mut all)?;
        bytes.copy_from_slice(
            all.get(offset..offset.saturating_add(bytes.len()))
                .ok_or_else(|| {
                    RuntimeError::Device("packet tensor read is outside its bounds".into())
                })?,
        );
        Ok(())
    }
    fn copy_tensor(
        &mut self,
        source: PacketTensor,
        source_offset: usize,
        target: PacketTensor,
        target_offset: usize,
        bytes: usize,
    ) -> Result<()>;
    /// A zeroed device buffer outside the packet (per-session state copies); `copy_tensor` moves
    /// bytes between it and packet tensors. Unsupported by default.
    fn create_tensor(&mut self, bytes: usize) -> Result<PacketTensor> {
        let _ = bytes;
        Err(RuntimeError::Rejected(
            "this packet backend cannot create tensors".into(),
        ))
    }
    fn run(&mut self, program: usize) -> Result<()>;
    fn run_sequence(&mut self, programs: &[usize]) -> Result<()> {
        for &program in programs {
            self.run(program)?;
        }
        Ok(())
    }
    fn last_run_us(&self) -> f64;
}

pub struct PacketAsset {
    pipelines: PacketPipelines,
    /// The speech contract's own metadata sections (`asr_vocabulary.json`), by name.
    sections: std::collections::BTreeMap<String, Vec<u8>>,
}

pub struct BoundPacketPipeline {
    name: String,
    driver: String,
    programs: BTreeMap<String, usize>,
    tensors: BTreeMap<String, PacketTensor>,
    parameters: BTreeMap<String, u64>,
    strings: BTreeMap<String, String>,
}

pub struct LoadedPacketRuntime {
    pub backend: &'static str,
    pub runtime: Box<dyn PacketRuntime>,
}

pub struct ForwardPacket {
    backend: &'static str,
    runtime: Box<dyn PacketRuntime>,
    pipeline: BoundPacketPipeline,
    programs: Vec<usize>,
    capacities: Vec<(u32, Vec<usize>)>,
    input: PacketTensor,
    output: PacketTensor,
}

impl ForwardPacket {
    pub fn load(path: &Path, pipeline_name: &str, backend: &str) -> Result<Self> {
        Self::load_on(path, pipeline_name, backend, 0)
    }

    /// As [`Self::load`], on CUDA device `device` (other backends ignore it).
    pub fn load_on(path: &Path, pipeline_name: &str, backend: &str, device: u8) -> Result<Self> {
        let asset = PacketAsset::load(path)?;
        let mut loaded = load_packet_runtime_on(path, backend, device)?;
        let pipeline = asset.bind(pipeline_name, loaded.runtime.as_ref())?;
        if pipeline.driver() != "forward.v1" {
            return Err(RuntimeError::Rejected(format!(
                "packet pipeline {pipeline_name:?} uses driver {:?}",
                pipeline.driver()
            )));
        }
        let programs = pipeline.program_sequence("forward")?;
        let capacities = pipeline.program_capacity_sequences("forward")?;
        let input = pipeline.tensor("input")?;
        let output = pipeline.tensor("output")?;
        loaded.runtime.end_execution()?;
        Ok(Self {
            backend: loaded.backend,
            runtime: loaded.runtime,
            pipeline,
            programs,
            capacities,
            input,
            output,
        })
    }

    pub fn pipeline(&self) -> &BoundPacketPipeline {
        &self.pipeline
    }

    pub fn runtime(&self) -> &dyn PacketRuntime {
        self.runtime.as_ref()
    }

    pub fn runtime_mut(&mut self) -> &mut dyn PacketRuntime {
        self.runtime.as_mut()
    }

    pub fn backend(&self) -> &'static str {
        self.backend
    }

    pub fn input_bytes(&self) -> usize {
        self.input.bytes
    }

    pub fn output_bytes(&self) -> usize {
        self.output.bytes
    }

    pub fn parameter(&self, name: &str) -> Result<u64> {
        self.pipeline.parameter(name)
    }

    pub fn optional_parameter(&self, name: &str) -> Option<u64> {
        self.pipeline.optional_parameter(name)
    }

    pub fn read_tensor(&self, name: &str) -> Result<Vec<u8>> {
        let tensor = self
            .runtime
            .tensor(name)
            .ok_or_else(|| RuntimeError::Rejected(format!("packet tensor {name:?} is missing")))?;
        let mut bytes = vec![0; tensor.bytes];
        self.runtime.read_tensor(tensor, &mut bytes)?;
        Ok(bytes)
    }

    pub fn write(&mut self, role: &str, bytes: &[u8]) -> Result<()> {
        let tensor = self.pipeline.tensor(role)?;
        self.runtime.write_tensor(tensor, bytes)
    }

    pub fn run(&mut self, input: &[u8], output: &mut [u8]) -> Result<()> {
        self.run_programs(input, output, None)
    }

    pub fn run_for_capacity(
        &mut self,
        capacity: u32,
        input: &[u8],
        output: &mut [u8],
    ) -> Result<()> {
        self.run_programs(input, output, Some(capacity))
    }

    fn run_programs(
        &mut self,
        input: &[u8],
        output: &mut [u8],
        capacity: Option<u32>,
    ) -> Result<()> {
        if input.len() != self.input.bytes || output.len() != self.output.bytes {
            return Err(RuntimeError::Rejected(format!(
                "forward packet buffers are {}/{}, expected {}/{} bytes",
                input.len(),
                output.len(),
                self.input.bytes,
                self.output.bytes
            )));
        }
        self.runtime.begin_execution()?;
        let result = (|| {
            self.runtime.write_tensor(self.input, input)?;
            let programs = if let Some(requested) = capacity {
                self.capacities
                    .iter()
                    .filter(|(capacity, _)| *capacity >= requested)
                    .min_by_key(|(capacity, _)| *capacity)
                    .map(|(_, programs)| programs.as_slice())
                    .ok_or_else(|| {
                        RuntimeError::Rejected(format!(
                            "forward packet has no capacity covering {requested}"
                        ))
                    })?
            } else {
                &self.programs
            };
            self.runtime.run_sequence(programs)?;
            self.runtime.read_tensor(self.output, output)
        })();
        let ended = self.runtime.end_execution();
        result.and(ended)
    }
}

pub fn load_packet_runtime(path: &Path, requested: &str) -> Result<LoadedPacketRuntime> {
    load_packet_runtime_on(path, requested, 0)
}

/// As [`load_packet_runtime`], with a CUDA runtime on device `device` (other backends ignore it).
pub fn load_packet_runtime_on(
    path: &Path,
    requested: &str,
    device: u8,
) -> Result<LoadedPacketRuntime> {
    let _ = (path, device);
    if requested == "auto" {
        #[cfg(all(feature = "metal", target_os = "macos"))]
        {
            return Ok(LoadedPacketRuntime {
                backend: "metal",
                runtime: Box::new(crate::exec::apple::MetalEngine::load_packet(path)?),
            });
        }
        #[cfg(all(feature = "cpu", not(all(feature = "metal", target_os = "macos"))))]
        {
            return load_packet_runtime_on(path, "cpu", device);
        }
        #[cfg(all(
            feature = "cuda",
            not(feature = "cpu"),
            not(all(feature = "metal", target_os = "macos"))
        ))]
        {
            return load_packet_runtime_on(path, "cuda", device);
        }
        #[cfg(not(any(
            feature = "cpu",
            feature = "cuda",
            all(feature = "metal", target_os = "macos")
        )))]
        {
            return Err(RuntimeError::Rejected(
                "no packet execution backend is compiled in".into(),
            ));
        }
    }
    match requested {
        #[cfg(feature = "cpu")]
        "cpu" => {
            let options = crate::exec::cpu::engine::CpuEngineOpts::default();
            Ok(LoadedPacketRuntime {
                backend: "cpu",
                runtime: Box::new(crate::exec::cpu::engine::CpuEngine::load_packet(
                    path, &options,
                )?),
            })
        }
        #[cfg(feature = "cuda")]
        "cuda" => Ok(LoadedPacketRuntime {
            backend: "cuda",
            runtime: Box::new(crate::exec::gpu::packet_exec::CudaPacketRuntime::load(
                path, device,
            )?),
        }),
        #[cfg(all(feature = "metal", target_os = "macos"))]
        "metal" => Ok(LoadedPacketRuntime {
            backend: "metal",
            runtime: Box::new(crate::exec::apple::MetalEngine::load_packet(path)?),
        }),
        _ => Err(RuntimeError::Rejected(format!(
            "packet backend {requested:?} is unavailable"
        ))),
    }
}

/// The path of the sidecar `key` names in the pipelines of `model_pkt`, beside it.
pub fn stage_packet(model_pkt: &Path, key: &str, legacy: &str) -> Result<std::path::PathBuf> {
    Ok(model_pkt.with_file_name(PacketAsset::load(model_pkt)?.stage_file(key, legacy)?))
}

/// The pipeline string naming the clone stage's reference encoder packet.
pub const CLONE_ENCODER: &str = "clone.encoder";
/// The pipeline string naming a speech pipeline's text-encoder sub-bundle.
pub const TEXT_ENCODER: &str = "text.encoder";
/// The voice-conditioning packet a speech bundle binds when its pipeline names none.
pub const VOICE_COND_PACKET: &str = "voice_cond.pkt";

fn beside_packet(key: &str, name: &str) -> Result<String> {
    if !name.is_empty() && !name.contains(['/', '\\']) && name != "." && name != ".." {
        Ok(name.to_string())
    } else {
        Err(RuntimeError::Rejected(format!(
            "{key} {name:?} is not a file name beside the packet"
        )))
    }
}

impl PacketAsset {
    #[cfg(test)]
    pub(crate) fn from_pipelines(
        pipelines: Vec<plow_asset::packet_pipeline::PacketPipeline>,
    ) -> Self {
        Self {
            pipelines: PacketPipelines {
                version: 1,
                pipelines,
            },
            sections: Default::default(),
        }
    }

    pub fn load(path: &Path) -> Result<Self> {
        let image = crate::asset::packet_image::PacketImage::open(path)?;
        Self::from_parsed(&image.raw, &image.blob)
    }

    /// `None` for a packet without a pipeline section (a plain language-model packet).
    pub fn load_if_present(path: &Path) -> Result<Option<Self>> {
        let image = crate::asset::packet_image::PacketImage::open(path)?;
        if image.blob.reserved_metadata(&image.raw, SECTION)?.is_none() {
            return Ok(None);
        }
        Self::from_parsed(&image.raw, &image.blob).map(Some)
    }

    pub fn from_bytes(image: &[u8]) -> Result<Self> {
        Self::from_parsed(
            image,
            &crate::asset::devblob::DevBlob::parse_l2(image, true)?,
        )
    }

    fn from_parsed(image: &[u8], blob: &crate::asset::devblob::DevBlob) -> Result<Self> {
        let raw = blob
            .reserved_metadata(image, SECTION)?
            .ok_or_else(|| RuntimeError::Rejected(format!("packet is missing {SECTION}")))?;
        if raw.len() > 1024 * 1024 {
            return Err(RuntimeError::Rejected(format!("{SECTION} is too large")));
        }
        let pipelines: PacketPipelines = serde_json::from_slice(raw)
            .map_err(|error| RuntimeError::Rejected(format!("invalid {SECTION}: {error}")))?;
        pipelines
            .validate(blob.progs.len(), |name| {
                blob.tensors
                    .iter()
                    .find(|tensor| tensor.name == name)
                    .map(|tensor| tensor.bytes)
            })
            .map_err(RuntimeError::Rejected)?;
        let mut sections = std::collections::BTreeMap::new();
        for name in [plow_asset::speech_contract::VOCABULARY_SECTION] {
            if let Some(data) = blob.reserved_metadata(image, name)? {
                if data.len() > 4 * 1024 * 1024 {
                    return Err(RuntimeError::Rejected(format!("{name} is too large")));
                }
                sections.insert(name.to_owned(), data.to_vec());
            }
        }
        Ok(Self {
            pipelines,
            sections,
        })
    }

    /// A speech-contract metadata section of the packet.
    pub fn metadata(&self, name: &str) -> Option<&[u8]> {
        self.sections.get(name).map(Vec::as_slice)
    }

    pub fn pipelines(&self) -> &[PacketPipeline] {
        &self.pipelines.pipelines
    }

    /// The sidecar packet a pipeline names under `key` (e.g. `encoder.packet`), else `legacy`
    /// for a packet emitted before the name was carried. A bare file name beside the packet.
    pub fn stage_file(&self, key: &str, legacy: &str) -> Result<String> {
        match self.pipelines().iter().find_map(|p| p.strings.get(key)) {
            None => Ok(legacy.to_string()),
            Some(name) => beside_packet(key, name),
        }
    }

    /// Whether a pipeline names the sidecar packet under `key`.
    pub fn names_stage(&self, key: &str) -> bool {
        self.pipelines().iter().any(|p| p.strings.contains_key(key))
    }

    /// Every sibling packet the pipelines declare, by stage: each `*.packet` stage and the
    /// clone stage's reference encoder ([`CLONE_ENCODER`]), as bare file names beside the
    /// packet. Each stage binds its own runtime, so two stages naming one file are two entries.
    pub fn stage_files(&self) -> Result<std::collections::BTreeMap<String, String>> {
        let mut out = std::collections::BTreeMap::new();
        for p in self.pipelines() {
            for (key, name) in &p.strings {
                if key.ends_with(".packet") || key == CLONE_ENCODER {
                    out.insert(key.clone(), beside_packet(key, name)?);
                }
            }
        }
        Ok(out)
    }

    /// The text-encoder sub-bundle a speech pipeline declares ([`TEXT_ENCODER`]), a bare name
    /// beside the packet.
    pub fn text_encoder(&self) -> Result<Option<String>> {
        self.pipelines()
            .iter()
            .find_map(|p| p.strings.get(TEXT_ENCODER))
            .map(|name| beside_packet(TEXT_ENCODER, name))
            .transpose()
    }

    pub fn bind(&self, name: &str, runtime: &dyn PacketRuntime) -> Result<BoundPacketPipeline> {
        let pipeline = self
            .pipelines
            .pipelines
            .iter()
            .find(|pipeline| pipeline.name == name)
            .ok_or_else(|| {
                RuntimeError::Rejected(format!("packet pipeline {name:?} is missing"))
            })?;
        self.bind_pipeline(pipeline, runtime)
    }

    pub fn bind_driver(
        &self,
        driver: &str,
        runtime: &dyn PacketRuntime,
    ) -> Result<BoundPacketPipeline> {
        let mut matches = self
            .pipelines
            .pipelines
            .iter()
            .filter(|pipeline| pipeline.driver == driver);
        let pipeline = matches.next().ok_or_else(|| {
            RuntimeError::Rejected(format!("packet pipeline driver {driver:?} is missing"))
        })?;
        if matches.next().is_some() {
            return Err(RuntimeError::Rejected(format!(
                "packet pipeline driver {driver:?} is ambiguous"
            )));
        }
        self.bind_pipeline(pipeline, runtime)
    }

    fn bind_pipeline(
        &self,
        pipeline: &PacketPipeline,
        runtime: &dyn PacketRuntime,
    ) -> Result<BoundPacketPipeline> {
        let mut tensors = BTreeMap::new();
        for (role, binding) in &pipeline.tensors {
            let tensor = runtime.tensor(&binding.name).ok_or_else(|| {
                RuntimeError::Rejected(format!("packet tensor for role {role:?} is missing"))
            })?;
            let expected = binding
                .shape
                .iter()
                .try_fold(binding.dtype.bytes(), |bytes, &dim| bytes.checked_mul(dim))
                .and_then(|bytes| usize::try_from(bytes).ok())
                .ok_or_else(|| {
                    RuntimeError::Rejected(format!("packet tensor role {role:?} overflows"))
                })?;
            if tensor.bytes != expected {
                return Err(RuntimeError::Rejected(format!(
                    "packet tensor role {role:?} has {} bytes, expected {expected}",
                    tensor.bytes
                )));
            }
            tensors.insert(role.clone(), tensor);
        }
        Ok(BoundPacketPipeline {
            name: pipeline.name.clone(),
            driver: pipeline.driver.clone(),
            programs: pipeline
                .programs
                .iter()
                .map(|(role, &program)| (role.clone(), program as usize))
                .collect(),
            tensors,
            parameters: pipeline.parameters.clone(),
            strings: pipeline.strings.clone(),
        })
    }
}

impl BoundPacketPipeline {
    pub fn string(&self, name: &str) -> Result<&str> {
        self.strings.get(name).map(String::as_str).ok_or_else(|| {
            RuntimeError::Rejected(format!("packet pipeline string {name:?} is missing"))
        })
    }

    pub fn optional_string(&self, name: &str) -> Option<&str> {
        self.strings.get(name).map(String::as_str)
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    /// Every program role and its program index.
    pub fn programs(&self) -> impl Iterator<Item = (&str, usize)> {
        self.programs
            .iter()
            .map(|(role, &program)| (role.as_str(), program))
    }

    pub fn driver(&self) -> &str {
        &self.driver
    }

    pub fn program(&self, role: &str) -> Result<usize> {
        self.programs.get(role).copied().ok_or_else(|| {
            RuntimeError::Rejected(format!("packet program role {role:?} is missing"))
        })
    }

    pub fn program_sequence(&self, role: &str) -> Result<Vec<usize>> {
        let prefix = format!("{role}.");
        let mut sequence = Vec::new();
        loop {
            let indexed = format!("{prefix}{}", sequence.len());
            let Some(&program) = self.programs.get(&indexed) else {
                break;
            };
            sequence.push(program);
        }
        if sequence.is_empty() {
            return Err(RuntimeError::Rejected(format!(
                "packet program sequence {role:?} is missing"
            )));
        }
        for name in self.programs.keys() {
            let Some(index) = name
                .strip_prefix(&prefix)
                .and_then(|suffix| suffix.parse::<usize>().ok())
            else {
                continue;
            };
            if index >= sequence.len() {
                return Err(RuntimeError::Rejected(format!(
                    "packet program sequence {role:?} has a gap before index {index}"
                )));
            }
        }
        Ok(sequence)
    }

    pub fn program_capacity_sequences(&self, role: &str) -> Result<Vec<(u32, Vec<usize>)>> {
        let prefix = format!("{role}.");
        let mut capacities = BTreeMap::<u32, BTreeMap<usize, usize>>::new();
        for (name, &program) in &self.programs {
            let Some((capacity, stage)) = name
                .strip_prefix(&prefix)
                .and_then(|suffix| suffix.split_once('.'))
                .and_then(|(capacity, stage)| {
                    Some((capacity.parse::<u32>().ok()?, stage.parse::<usize>().ok()?))
                })
            else {
                continue;
            };
            if capacities
                .entry(capacity)
                .or_default()
                .insert(stage, program)
                .is_some()
            {
                return Err(RuntimeError::Rejected(format!(
                    "packet program capacity {role}.{capacity} repeats stage {stage}"
                )));
            }
        }
        capacities
            .into_iter()
            .map(|(capacity, stages)| {
                if capacity == 0 || stages.is_empty() || stages.keys().copied().ne(0..stages.len())
                {
                    return Err(RuntimeError::Rejected(format!(
                        "packet program capacity {role}.{capacity} has a gap"
                    )));
                }
                Ok((capacity, stages.into_values().collect()))
            })
            .collect()
    }

    pub fn program_for_rows(&self, phase: &str, rows: u32) -> Result<(usize, u32)> {
        let prefix = format!("{phase}.");
        self.programs
            .iter()
            .filter_map(|(role, &program)| {
                role.strip_prefix(&prefix)
                    .and_then(|suffix| suffix.parse::<u32>().ok())
                    .filter(|&capacity| capacity >= rows)
                    .map(|capacity| (program, capacity))
            })
            .min_by_key(|&(_, capacity)| capacity)
            .ok_or_else(|| {
                RuntimeError::Rejected(format!(
                    "packet {phase} pipeline has no program covering {rows} rows"
                ))
            })
    }

    pub fn program_for_rows_at_most(&self, phase: &str, rows: u32) -> Result<(usize, u32)> {
        let prefix = format!("{phase}.");
        self.programs
            .iter()
            .filter_map(|(role, &program)| {
                role.strip_prefix(&prefix)
                    .and_then(|suffix| suffix.parse::<u32>().ok())
                    .filter(|&capacity| capacity <= rows)
                    .map(|capacity| (program, capacity))
            })
            .max_by_key(|&(_, capacity)| capacity)
            .ok_or_else(|| {
                RuntimeError::Rejected(format!(
                    "packet {phase} pipeline has no program within {rows} rows"
                ))
            })
    }

    pub fn tensor(&self, role: &str) -> Result<PacketTensor> {
        self.tensors.get(role).copied().ok_or_else(|| {
            RuntimeError::Rejected(format!("packet tensor role {role:?} is missing"))
        })
    }

    pub fn optional_tensor(&self, role: &str) -> Option<PacketTensor> {
        self.tensors.get(role).copied()
    }

    pub fn tensor_sequence(&self, role: &str) -> Result<Vec<PacketTensor>> {
        let prefix = format!("{role}.");
        let mut sequence = Vec::new();
        loop {
            let indexed = format!("{prefix}{}", sequence.len());
            let Some(&tensor) = self.tensors.get(&indexed) else {
                break;
            };
            sequence.push(tensor);
        }
        if sequence.is_empty() {
            return Err(RuntimeError::Rejected(format!(
                "packet tensor sequence {role:?} is missing"
            )));
        }
        for name in self.tensors.keys() {
            let Some(index) = name
                .strip_prefix(&prefix)
                .and_then(|suffix| suffix.parse::<usize>().ok())
            else {
                continue;
            };
            if index >= sequence.len() {
                return Err(RuntimeError::Rejected(format!(
                    "packet tensor sequence {role:?} has a gap before index {index}"
                )));
            }
        }
        Ok(sequence)
    }

    pub fn parameter(&self, name: &str) -> Result<u64> {
        self.parameters
            .get(name)
            .copied()
            .ok_or_else(|| RuntimeError::Rejected(format!("packet parameter {name:?} is missing")))
    }

    pub fn optional_parameter(&self, name: &str) -> Option<u64> {
        self.parameters.get(name).copied()
    }
}

pub(crate) fn check_transfer(
    tensor: PacketTensor,
    actual_handle: usize,
    actual_bytes: usize,
    transfer_bytes: usize,
) -> Result<()> {
    if tensor.handle != actual_handle {
        return Err(RuntimeError::Device(format!(
            "packet tensor handle {} is invalid",
            tensor.handle
        )));
    }
    if tensor.bytes != actual_bytes || transfer_bytes != actual_bytes {
        return Err(RuntimeError::Device(format!(
            "packet tensor {} has {actual_bytes} bytes, transfer has {transfer_bytes}",
            tensor.handle
        )));
    }
    Ok(())
}

pub(crate) fn check_copy(
    source: PacketTensor,
    source_bytes: usize,
    source_offset: usize,
    target: PacketTensor,
    target_bytes: usize,
    target_offset: usize,
    bytes: usize,
) -> Result<()> {
    if source.bytes != source_bytes || target.bytes != target_bytes {
        return Err(RuntimeError::Device("packet tensor size changed".into()));
    }
    let source_end = source_offset.checked_add(bytes);
    let target_end = target_offset.checked_add(bytes);
    if source_end.is_none_or(|end| end > source_bytes)
        || target_end.is_none_or(|end| end > target_bytes)
    {
        return Err(RuntimeError::Device(
            "packet tensor copy is outside its bounds".into(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "cuda")]
    #[test]
    fn pool_views_prevent_reuse_and_unretired_copies_retain_both_owners() {
        use std::sync::Arc;
        let mut pool = PacketBufferPool {
            slots: vec![vec![1; 32].into(), vec![2; 32].into()],
            next: 0,
        };
        let source = Arc::new(());
        let weak = Arc::downgrade(&source);
        let a = pool.acquire(17, source.clone()).unwrap();
        let b = pool.acquire(32, source.clone()).unwrap();
        assert!(pool.acquire(1, source.clone()).is_none());
        // SAFETY: host-only fixtures have no queued device work.
        let a = unsafe { a.retire() };
        let view = a.slice(3..11).unwrap();
        drop(a);
        assert!(pool.acquire(1, source.clone()).is_none());
        drop(view);
        assert!(pool.acquire(33, source.clone()).is_none());
        let pending = pool.acquire(32, source.clone()).unwrap();
        let destination = Arc::downgrade(&pending.buffer.as_ref().unwrap().storage);
        drop(source);
        drop(pending);
        // SAFETY: host-only fixtures have no queued device work.
        drop(unsafe { b.retire() });
        drop(pool);
        assert!(weak.upgrade().is_some(), "unretired copy freed its source");
        assert!(
            destination.upgrade().is_some(),
            "unretired copy freed its destination"
        );
    }

    #[test]
    fn packet_buffer_views_keep_storage_and_validate_before_reading() {
        let buffer: PacketBuffer = (0..16).collect::<Vec<u8>>().into();
        let owner = std::sync::Arc::downgrade(&buffer.storage);
        let view = buffer.slice(2..10).unwrap().slice(3..6).unwrap();
        assert!(buffer.slice(9..8).is_err());
        assert!(buffer.slice(0..usize::MAX).is_err());
        drop(buffer);
        assert!(owner.upgrade().is_some());
        let mut bytes = [0; 3];
        view.read(&mut bytes).unwrap();
        assert_eq!(bytes, [5, 6, 7]);
        let mut guard = [0xa5; 2];
        assert!(view.read(&mut guard).is_err());
        assert_eq!(guard, [0xa5; 2]);
        assert!(view.slice(3..3).unwrap().is_empty());
        drop(view);
        assert!(owner.upgrade().is_none());
    }

    #[test]
    fn bulk_write_fallback_preserves_order_and_validates_ranges_before_writing() {
        struct Memory([u8; 16], usize);
        impl PacketRuntime for Memory {
            fn tensor(&self, _: &str) -> Option<PacketTensor> {
                None
            }
            fn write_tensor(&mut self, t: PacketTensor, bytes: &[u8]) -> Result<()> {
                check_transfer(t, 0, 16, bytes.len())?;
                self.0.copy_from_slice(bytes);
                self.1 += 1;
                Ok(())
            }
            fn read_tensor(&self, t: PacketTensor, bytes: &mut [u8]) -> Result<()> {
                check_transfer(t, 0, 16, bytes.len())?;
                bytes.copy_from_slice(&self.0);
                Ok(())
            }
            fn copy_tensor(
                &mut self,
                _: PacketTensor,
                _: usize,
                _: PacketTensor,
                _: usize,
                _: usize,
            ) -> Result<()> {
                unreachable!()
            }
            fn run(&mut self, _: usize) -> Result<()> {
                unreachable!()
            }
            fn last_run_us(&self) -> f64 {
                0.0
            }
        }
        let t = PacketTensor {
            handle: 0,
            bytes: 16,
        };
        let mut rt = Memory([0; 16], 0);
        rt.write_tensors(&[
            PacketWrite {
                tensor: t,
                offset: 0,
                data: PacketData::Host(&[0xa5; 16]),
            },
            PacketWrite {
                tensor: t,
                offset: 3,
                data: PacketData::Host(&[1, 2, 3, 4]),
            },
            PacketWrite {
                tensor: t,
                offset: 5,
                data: PacketData::Host(&[5, 6]),
            },
        ])
        .unwrap();
        let mut expected = [0xa5; 16];
        expected[3..7].copy_from_slice(&[1, 2, 5, 6]);
        assert_eq!(rt.0, expected);
        assert_eq!(rt.1, 3);
        for offset in [16, usize::MAX] {
            assert!(rt
                .write_tensors(&[
                    PacketWrite {
                        tensor: t,
                        offset: 0,
                        data: PacketData::Host(&[0xff]),
                    },
                    PacketWrite {
                        tensor: t,
                        offset,
                        data: PacketData::Host(&[1]),
                    },
                ])
                .is_err());
            assert_eq!(rt.0, expected);
            assert_eq!(rt.1, 3);
        }
        let snapshot = rt.snapshot_tensor(t, 3, 4).unwrap();
        rt.write_tensor(t, &[0; 16]).unwrap();
        let mut expected = [0; 4];
        snapshot.read(&mut expected).unwrap();
        assert_eq!(expected, [1, 2, 5, 6]);
        let mut consumer = Memory([0xa5; 16], 0);
        consumer
            .write_tensor_from(t, 8, &snapshot.slice(1..3).unwrap())
            .unwrap();
        assert_eq!(&consumer.0[8..10], &[2, 5]);
        assert_eq!(consumer.0[..8], [0xa5; 8]);
        assert_eq!(consumer.0[10..], [0xa5; 6]);
        assert!(consumer
            .write_tensor_from(t, usize::MAX, &snapshot)
            .is_err());
        assert_eq!(consumer.1, 1);
        assert!(rt.snapshot_tensor(t, 15, 2).is_err());
        consumer
            .write_tensors(&[
                PacketWrite {
                    tensor: t,
                    offset: 2,
                    data: PacketData::Zero(12),
                },
                PacketWrite {
                    tensor: t,
                    offset: 3,
                    data: PacketData::Buffer(&snapshot),
                },
                PacketWrite {
                    tensor: t,
                    offset: 5,
                    data: PacketData::Host(&[9, 8]),
                },
            ])
            .unwrap();
        let expected = [0xa5, 0xa5, 0, 1, 2, 9, 8, 0, 0, 0, 0, 0, 0, 0, 0xa5, 0xa5];
        assert_eq!(consumer.0, expected);
        for data in [PacketData::Zero(usize::MAX), PacketData::Buffer(&snapshot)] {
            assert!(consumer
                .write_tensors(&[
                    PacketWrite {
                        tensor: t,
                        offset: 0,
                        data: PacketData::Zero(16),
                    },
                    PacketWrite {
                        tensor: t,
                        offset: 15,
                        data,
                    },
                ])
                .is_err());
            assert_eq!(consumer.0, expected);
        }
    }

    struct FakeRuntime;

    impl PacketRuntime for FakeRuntime {
        fn tensor(&self, name: &str) -> Option<PacketTensor> {
            (name == "in.values").then_some(PacketTensor {
                handle: 0,
                bytes: 16,
            })
        }
        fn write_tensor(&mut self, _: PacketTensor, _: &[u8]) -> Result<()> {
            Ok(())
        }
        fn read_tensor(&self, _: PacketTensor, _: &mut [u8]) -> Result<()> {
            Ok(())
        }
        fn copy_tensor(
            &mut self,
            _: PacketTensor,
            _: usize,
            _: PacketTensor,
            _: usize,
            _: usize,
        ) -> Result<()> {
            Ok(())
        }
        fn run(&mut self, _: usize) -> Result<()> {
            Ok(())
        }
        fn last_run_us(&self) -> f64 {
            0.0
        }
    }

    #[test]
    fn transfer_requires_the_resolved_handle_and_full_tensor() {
        let tensor = PacketTensor {
            handle: 3,
            bytes: 16,
        };
        assert!(check_transfer(tensor, 3, 16, 16).is_ok());
        assert!(check_transfer(tensor, 4, 16, 16).is_err());
        assert!(check_transfer(tensor, 3, 16, 12).is_err());
        assert!(check_copy(tensor, 16, 4, tensor, 16, 8, 8).is_ok());
        assert!(check_copy(tensor, 16, 12, tensor, 16, 0, 8).is_err());
    }

    #[test]
    fn binds_pipeline_roles_from_packet_metadata() {
        use packet::dev::DevOp;
        use packet::devbuild::{Builder, Model, SectionData, SECT_METADATA};
        use plow_asset::packet_pipeline::{PacketPipeline, PipelineDType, PipelineTensor, VERSION};

        let mut builder = Builder::new(1);
        let values = builder.tensor("in.values", 16);
        builder.emit(DevOp::SiluF32, builder.all(), &[], |inst| {
            inst.t[..2].copy_from_slice(&[values, values]);
            inst.i[0] = 4;
        });
        let tensors = builder.tensors();
        let program = builder.finish();
        let model = Model {
            n_cu: 1,
            target: 0,
            tensors,
            progs: vec![program],
            prog_t: vec![1],
            kv_row_insts: vec![],
            gen: vec![],
        };
        let metadata = PacketPipelines {
            version: VERSION,
            pipelines: vec![PacketPipeline {
                strings: Default::default(),
                name: "infer".into(),
                driver: "feedforward.v1".into(),
                programs: BTreeMap::from([
                    ("forward".into(), 0),
                    ("broken.0".into(), 0),
                    ("broken.2".into(), 0),
                    ("prefill.128".into(), 0),
                    ("prefill.512".into(), 0),
                    ("stage.0".into(), 0),
                    ("stage.1".into(), 0),
                    ("encoder.400.0".into(), 0),
                    ("encoder.400.1".into(), 0),
                    ("encoder.800.0".into(), 0),
                ]),
                tensors: BTreeMap::from([
                    (
                        "input".into(),
                        PipelineTensor {
                            name: "in.values".into(),
                            dtype: PipelineDType::F32,
                            shape: vec![4],
                        },
                    ),
                    (
                        "state.0".into(),
                        PipelineTensor {
                            name: "in.values".into(),
                            dtype: PipelineDType::F32,
                            shape: vec![4],
                        },
                    ),
                    (
                        "state.1".into(),
                        PipelineTensor {
                            name: "in.values".into(),
                            dtype: PipelineDType::F32,
                            shape: vec![4],
                        },
                    ),
                ]),
                parameters: BTreeMap::from([("batch".into(), 1)]),
            }],
        };
        let image = model.to_blob_v6(&[SectionData {
            kind: SECT_METADATA,
            name: SECTION.into(),
            data: serde_json::to_vec(&metadata).unwrap(),
        }]);
        let asset = PacketAsset::from_bytes(&image).unwrap();
        let pipeline = asset.bind("infer", &FakeRuntime).unwrap();
        assert_eq!(
            asset
                .bind_driver("feedforward.v1", &FakeRuntime)
                .unwrap()
                .name(),
            "infer"
        );
        assert_eq!(pipeline.driver(), "feedforward.v1");
        assert_eq!(pipeline.program("forward").unwrap(), 0);
        assert_eq!(pipeline.program_sequence("stage").unwrap(), [0, 0]);
        assert_eq!(
            pipeline.program_capacity_sequences("encoder").unwrap(),
            [(400, vec![0, 0]), (800, vec![0])]
        );
        assert!(pipeline.program_sequence("broken").is_err());
        assert_eq!(pipeline.program_for_rows("prefill", 129).unwrap(), (0, 512));
        assert!(pipeline.program_for_rows("prefill", 513).is_err());
        assert_eq!(
            pipeline.program_for_rows_at_most("prefill", 511).unwrap(),
            (0, 128)
        );
        assert!(pipeline.program_for_rows_at_most("prefill", 127).is_err());
        assert_eq!(pipeline.tensor("input").unwrap().bytes, 16);
        assert_eq!(pipeline.tensor_sequence("state").unwrap().len(), 2);
        assert_eq!(pipeline.parameter("batch").unwrap(), 1);
    }

    #[cfg(feature = "cpu")]
    #[test]
    fn cpu_executes_a_self_contained_packet_asset() {
        use packet::dev::DevOp;
        use packet::devbuild::{Builder, Model};

        let mut builder = Builder::new(1);
        let values = builder.tensor("act.values", 16);
        builder.emit(DevOp::SiluF32, builder.all(), &[], |inst| {
            inst.t[..2].copy_from_slice(&[values, values]);
            inst.i[0] = 4;
        });
        let tensors = builder.tensors();
        let program = builder.finish();
        let model = Model {
            n_cu: 1,
            target: 0,
            tensors,
            progs: vec![program],
            prog_t: vec![1],
            kv_row_insts: vec![],
            gen: vec![],
        };
        let path = std::env::temp_dir().join(format!(
            "plow-packet-runtime-{}-{}.plowdev",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        std::fs::write(&path, model.to_blob()).unwrap();
        let opts = crate::exec::cpu::engine::CpuEngineOpts {
            threads: 1,
            isa: crate::exec::cpu::ffi::Isa::Scalar,
            ..Default::default()
        };
        let mut runtime = crate::exec::cpu::engine::CpuEngine::load_packet(&path, &opts).unwrap();
        let tensor = runtime.tensor("act.values").unwrap();
        let input = [-2.0f32, -0.5, 0.0, 3.0];
        runtime
            .write_tensor(tensor, unsafe {
                std::slice::from_raw_parts(input.as_ptr().cast(), std::mem::size_of_val(&input))
            })
            .unwrap();
        runtime.run(0).unwrap();
        let mut bytes = [0u8; 16];
        runtime.read_tensor(tensor, &mut bytes).unwrap();
        let _ = std::fs::remove_file(path);
        for (index, expected) in input
            .iter()
            .map(|value| value / (1.0 + (-value).exp()))
            .enumerate()
        {
            let actual = f32::from_ne_bytes(bytes[index * 4..index * 4 + 4].try_into().unwrap());
            assert!((actual - expected).abs() < 1e-6, "{actual} != {expected}");
        }
    }

    #[test]
    fn a_packet_without_pipelines_is_not_a_pipeline_asset() {
        use packet::dev::DevOp;
        use packet::devbuild::{Builder, Model};

        let mut builder = Builder::new(1);
        let values = builder.tensor("act.values", 16);
        builder.emit(DevOp::SiluF32, builder.all(), &[], |inst| {
            inst.t[..2].copy_from_slice(&[values, values]);
            inst.i[0] = 4;
        });
        let tensors = builder.tensors();
        let model = Model {
            n_cu: 1,
            target: 0,
            tensors,
            progs: vec![builder.finish()],
            prog_t: vec![1],
            kv_row_insts: vec![],
            gen: vec![],
        };
        let path = std::env::temp_dir().join(format!(
            "plow-packet-no-pipelines-{}.plowdev",
            std::process::id()
        ));
        std::fs::write(&path, model.to_blob()).unwrap();
        let present = PacketAsset::load_if_present(&path);
        let strict = PacketAsset::load(&path);
        let _ = std::fs::remove_file(&path);
        assert!(matches!(present, Ok(None)));
        assert!(strict.is_err());
    }

    #[cfg(all(feature = "metal", target_os = "macos"))]
    #[test]
    fn metal_executes_embedding_overlay_packet() {
        use packet::dev::DevOp;
        use packet::devbuild::{Builder, Model};

        let mut builder = Builder::new(1);
        let output = builder.tensor("act.embeddings", 12);
        let table = builder.tensor_init(
            "embedding.weight",
            [0x3f80u16, 0x4000, 0x4040, 0x4080]
                .into_iter()
                .flat_map(u16::to_le_bytes)
                .collect(),
        );
        let tokens = builder.tensor("in.tokens", 12);
        let overlay = builder.tensor("in.overlay", 8);
        let overlay_index = builder.tensor("in.overlay_index", 12);
        builder.emit(DevOp::EmbedOverlayBf16, builder.all(), &[], |inst| {
            inst.t[..5].copy_from_slice(&[output, table, tokens, overlay, overlay_index]);
            inst.i[..4].copy_from_slice(&[3, 2, 2, 1]);
        });
        let model = Model {
            n_cu: 1,
            target: 0,
            tensors: builder.tensors(),
            progs: vec![builder.finish()],
            prog_t: vec![3],
            kv_row_insts: vec![],
            gen: vec![],
        };
        let path = std::env::temp_dir().join(format!(
            "plow-embedding-overlay-{}-{}.plowdev",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        std::fs::write(&path, model.to_blob()).unwrap();
        let mut runtime = crate::exec::apple::MetalEngine::load_packet(&path).unwrap();
        for (name, bytes) in [
            ("in.tokens", bytemuck::cast_slice(&[1u32, 0, 0])),
            ("in.overlay", bytemuck::cast_slice(&[1.001f32, -2.25])),
            (
                "in.overlay_index",
                bytemuck::cast_slice(&[u32::MAX, 0, u32::MAX]),
            ),
        ] {
            let tensor = runtime.tensor(name).unwrap();
            runtime.write_tensor(tensor, bytes).unwrap();
        }
        runtime.run(0).unwrap();
        let mut actual = [0u8; 12];
        let output = runtime.tensor("act.embeddings").unwrap();
        runtime.read_tensor(output, &mut actual).unwrap();
        let _ = std::fs::remove_file(path);
        let bits = 1.001f32.to_bits();
        let rounded = ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) as u16;
        let expected = [0x4040u16, 0x4080, rounded, 0xc010, 0x3f80, 0x4000];
        assert_eq!(
            actual.as_slice(),
            bytemuck::cast_slice::<u16, u8>(&expected)
        );
    }
}
