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
    fn read_tensor(&self, tensor: PacketTensor, bytes: &mut [u8]) -> Result<()>;
    fn copy_tensor(
        &mut self,
        source: PacketTensor,
        source_offset: usize,
        target: PacketTensor,
        target_offset: usize,
        bytes: usize,
    ) -> Result<()>;
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
}

pub struct BoundPacketPipeline {
    name: String,
    driver: String,
    programs: BTreeMap<String, usize>,
    tensors: BTreeMap<String, PacketTensor>,
    parameters: BTreeMap<String, u64>,
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
        let asset = PacketAsset::load(path)?;
        let mut loaded = load_packet_runtime(path, backend)?;
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
    let _ = path;
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
            return load_packet_runtime(path, "cpu");
        }
        #[cfg(not(any(feature = "cpu", all(feature = "metal", target_os = "macos"))))]
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

impl PacketAsset {
    pub fn load(path: &Path) -> Result<Self> {
        let image = std::fs::read(path).map_err(|source| RuntimeError::Io {
            path: path.to_owned(),
            source,
        })?;
        Self::from_bytes(&image)
    }

    pub fn from_bytes(image: &[u8]) -> Result<Self> {
        let blob = crate::asset::devblob::DevBlob::parse(image)?;
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
        Ok(Self { pipelines })
    }

    pub fn pipelines(&self) -> &[PacketPipeline] {
        &self.pipelines.pipelines
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
        })
    }
}

impl BoundPacketPipeline {
    pub fn name(&self) -> &str {
        &self.name
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
