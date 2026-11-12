//! Compiled speech inspection uses the runtime reader and ISA disassembler, not a second model builder.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use nn_graph::viz::{
    ModelModality, OpCategory, VizBlock, VizGraph, VizModelBundle, VizNetwork, VizNode, VizStage,
    VizSummary, VizTensor,
};
use nn_graph::{DType, NodeId, Origin, TensorId};
#[cfg(test)]
use plow_asset::packet_pipeline::PacketPipelines;
use plow_asset::packet_pipeline::PipelineMetadata;
use plowrt::asset::devblob::{DevBlobMetadata, DevProg};

pub fn bundle_from_path(path: &Path) -> Result<VizModelBundle, String> {
    let mut paths = if path.is_dir() {
        let mut paths = std::fs::read_dir(path)
            .map_err(|e| e.to_string())?
            .map(|e| e.map(|e| e.path()))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| e.to_string())?;
        paths.retain(|p| {
            p.extension().is_some_and(|e| e == "pkt")
                && p.file_name().is_some_and(|n| n != "extension.pkt")
        });
        paths.sort();
        paths
    } else {
        vec![path.to_owned()]
    };
    for p in &mut paths {
        *p = std::fs::canonicalize(&*p).map_err(|e| e.to_string())?;
    }
    paths.sort();
    paths.dedup();
    if paths.is_empty() {
        return Err("no model packets found".into());
    }
    let title = path
        .file_stem()
        .unwrap_or_default()
        .to_string_lossy()
        .into_owned();
    let mut bundle = VizModelBundle {
        schema: "plow.viz.v1", title: title.clone(), source: title, modality: ModelModality::Llm,
        default_network: "pipeline".into(), networks: Vec::new(), provenance: None,
        notes: vec!["Compiled execution view. Edges in program details are scheduler dependencies, not inferred tensor dataflow. Raw buffers have exact declared byte capacities; logical shapes/dtypes are shown only where the packet contract encodes them. Parameter counts are not inferred from packed bytes.".into()],
    };
    let mut overview = empty_graph();
    let mut stages = Vec::new();
    let mut contracts = Vec::new();
    let mut packet_contracts = Vec::new();
    let mut structure_only = Vec::new();
    let root = std::fs::canonicalize(if path.is_dir() {
        path
    } else {
        path.parent().unwrap_or_else(|| Path::new("."))
    })
    .map_err(|e| e.to_string())?;
    let mut links = Vec::new();
    let mut component_names = BTreeMap::new();
    let mut file_index = 0;
    while file_index < paths.len() {
        if paths.len() > 128 {
            return Err("packet dependency graph exceeds 128 files".into());
        }
        let file = paths[file_index].clone();
        let blob = DevBlobMetadata::load(&file).map_err(|e| format!("{}: {e}", file.display()))?;
        let label = file
            .strip_prefix(&root)
            .unwrap_or(&file)
            .to_string_lossy()
            .into_owned();
        if blob
            .sections
            .iter()
            .any(|(_, n, _)| n == plow_asset::structure_only::SECTION)
        {
            structure_only.push(label.to_string());
        }
        let metadata = blob
            .sections
            .iter()
            .find(|(_, n, _)| n == plow_asset::packet_pipeline::SECTION)
            .map(|(_, _, bytes)| PipelineMetadata::parse(bytes))
            .transpose()?;
        if let Some(metadata) = &metadata {
            metadata.validate(blob.progs.len(), |name| {
                blob.tensors
                    .iter()
                    .find(|t| t.name == name)
                    .map(|t| t.bytes)
            })?;
            for pipeline in metadata.pipelines() {
                if pipeline.driver.starts_with("tts.") {
                    bundle.modality = ModelModality::Tts;
                } else if (pipeline.driver.starts_with("rnnt.")
                    || pipeline.driver.starts_with("asr.")
                    || pipeline.driver.starts_with("audio_lm."))
                    && bundle.modality != ModelModality::Tts
                {
                    bundle.modality = ModelModality::Asr;
                } else if pipeline.driver.starts_with("vad.")
                    && bundle.modality == ModelModality::Llm
                {
                    bundle.modality = ModelModality::Vad;
                } else if pipeline.name.starts_with("codec")
                    && bundle.modality == ModelModality::Llm
                {
                    bundle.modality = ModelModality::AudioCodec;
                }
                contracts.push(serde_json::to_value(pipeline).map_err(|e| e.to_string())?);
                for (role, name) in pipeline.strings.iter().filter(|(k, _)| {
                    k.ends_with(".packet") || matches!(k.as_str(), "clone.encoder" | "text.encoder")
                }) {
                    if Path::new(name)
                        .components()
                        .any(|c| !matches!(c, std::path::Component::Normal(_)))
                    {
                        return Err(format!("invalid packet dependency {name:?}"));
                    }
                    let mut dependency = file.parent().unwrap_or(&root).join(name);
                    if dependency.is_dir() {
                        dependency = dependency.join("model.pkt");
                    }
                    if !dependency.is_file() {
                        return Err(format!(
                            "{label}: required {role} dependency {name:?} is missing"
                        ));
                    }
                    let dependency =
                        std::fs::canonicalize(&dependency).map_err(|e| e.to_string())?;
                    if !dependency.starts_with(&root) {
                        return Err(format!(
                            "{label}: packet dependency escapes the asset directory"
                        ));
                    }
                    let index = if let Some(index) = paths.iter().position(|p| p == &dependency) {
                        index
                    } else {
                        paths.push(dependency);
                        paths.len() - 1
                    };
                    let component_name = match role.as_str() {
                        "clone.encoder" | "reference_encoder.packet" => {
                            Some("Reference audio encoder")
                        }
                        "text.encoder" | "text_encoder.packet" => Some("Text encoder"),
                        "encoder.packet" => Some("Audio encoder"),
                        "voice_cond.packet" => Some("Voice conditioning"),
                        "vocoder.packet" => Some("Vocoder"),
                        "codec.packet" => Some("Audio decoder"),
                        _ => None,
                    };
                    if let Some(name) = component_name {
                        component_names.insert(index, name);
                    }
                    match role.as_str() {
                        "encoder.packet" | "text_encoder.packet" | "voice_cond.packet" | "reference_encoder.packet" | "clone.encoder" | "text.encoder" => links.push((index, file_index)),
                        "codec.packet" | "vocoder.packet" | "depth.packet" => links.push((file_index, index)),
                        _ => bundle.notes.push(format!("{role} dependency {name} is included; its handoff direction is not encoded by a known contract.")),
                    }
                }
            }
        }
        let network_ids: Vec<String> = (0..blob.progs.len())
            .map(|p| format!("packet{file_index}.program{p}"))
            .collect();
        let depth = depth_networks(&blob, file_index, &label)?;
        let mut program_entries: Vec<_> = network_ids
            .iter()
            .enumerate()
            .map(|(i, id)| serde_json::json!({"id":id,"label":program_label(i,metadata.as_ref())}))
            .collect();
        program_entries.extend(
            depth
                .iter()
                .map(|n| serde_json::json!({"id":n.id,"label":n.title})),
        );
        let stage_id = format!("packet{file_index}");
        let node_id = NodeId(overview.nodes.len() as u32);
        let output = TensorId(overview.tensors.len() as u32);
        overview.tensors.push(completion(output, node_id));
        overview.outputs.push(output);
        overview.nodes.push(VizNode {
            id: node_id, op: "packet", category: OpCategory::Layout,
            detail: serde_json::json!({"Packet": {"programs": program_entries, "target": blob.target, "compute_units": blob.n_cu}}),
            inputs: Vec::new(), output, block: None, stage: Some(stage_id.clone()), param_count: 0, weight_bytes: 0,
        });
        let stage_name = metadata
            .as_ref()
            .and_then(|m| m.pipelines().first())
            .map(|p| {
                if p.driver.starts_with("vad.") {
                    "Voice activity detector"
                } else if p.driver.starts_with("rnnt.") {
                    "Transducer: encoder, predictor, joint"
                } else if p.driver.starts_with("tts.") {
                    "Speech language model"
                } else if p.name.starts_with("codec") {
                    "Audio decoder"
                } else if p.name.starts_with("audio") {
                    "Audio encoder"
                } else {
                    label.as_ref()
                }
            })
            .unwrap_or(label.as_ref());
        let stage_name = if !depth.is_empty() {
            "Speech language model + depth decoder"
        } else {
            stage_name
        };
        stages.push(VizStage {
            id: stage_id,
            name: stage_name.to_string(),
            description: format!(
                "{label}: {} compiled programs; inspect each program without executing it",
                blob.progs.len() + depth.len()
            ),
            modules: Vec::new(),
            block_count: 0,
            param_count: 0,
            input_shapes: Vec::new(),
            output_shapes: Vec::new(),
        });
        overview.summary.total_weight_bytes += blob.tensors.iter().map(|t| t.bytes).sum::<u64>();
        for (program_index, program) in blob.progs.iter().enumerate() {
            bundle.networks.push(program_network(
                &blob,
                program,
                &network_ids[program_index],
                &format!(
                    "{label} · {}",
                    program_label(program_index, metadata.as_ref())
                ),
                metadata.as_ref(),
            )?);
        }
        bundle.networks.extend(depth);
        packet_contracts.push(metadata);
        file_index += 1;
    }
    for (index, name) in component_names {
        stages[index].name = name.into();
    }
    links.sort_unstable();
    links.dedup();
    for &(from, to) in &links {
        if from == to {
            return Err("packet contract references itself".into());
        }
        let dependency = overview.nodes[from].output;
        overview.nodes[to].inputs.push(dependency);
        for left in packet_contracts[from].iter().flat_map(|m| m.pipelines()) {
            for right in packet_contracts[to].iter().flat_map(|m| m.pipelines()) {
                for key in [
                    "codec.frame_codes",
                    "codec.frame_samples",
                    "codec.codebook",
                    "audio.sample_rate",
                ] {
                    if let (Some(a), Some(b)) =
                        (left.parameters.get(key), right.parameters.get(key))
                    {
                        if a != b {
                            return Err(format!(
                                "packet handoff {from} -> {to}: {key} disagrees ({a} vs {b})"
                            ));
                        }
                    }
                }
            }
        }
    }
    overview.outputs = overview
        .nodes
        .iter()
        .enumerate()
        .filter(|(i, _)| !links.iter().any(|(from, _)| from == i))
        .map(|(_, n)| n.output)
        .collect();
    overview.summary.total_nodes = overview.nodes.len();
    overview.summary.total_tensors = overview.tensors.len();
    overview
        .summary
        .category_counts
        .insert(OpCategory::Layout, overview.nodes.len());
    bundle.networks.insert(0, VizNetwork { kind: "packet", id: "pipeline".into(), title: "Speech pipeline".into(), description: "Packet handoffs come from embedded driver contracts. Detail shows every compiled program; alternative buckets and state banks are not sequential layers.".into(), stages, graph: overview, checkpoint: None, metadata: Some(serde_json::json!({"contracts": contracts, "structure_only": structure_only})) });
    if !structure_only.is_empty() {
        bundle.notes.push(format!("Structure only: {} compiled from checkpoint headers. Programs, buffer capacities and contracts are real; weights were never downloaded and these packets cannot be executed.", structure_only.join(", ")));
    }
    if paths.len() > 1 && links.is_empty() {
        bundle.notes.push("No cross-packet handoff contracts were found. Components are shown independently; execution order is not inferred from filenames.".into());
    }
    Ok(bundle)
}

fn program_label(index: usize, metadata: Option<&PipelineMetadata>) -> String {
    let roles: Vec<_> = metadata
        .into_iter()
        .flat_map(|m| m.pipelines())
        .flat_map(|p| {
            p.programs
                .iter()
                .filter(move |(_, i)| **i as usize == index)
                .map(move |(role, _)| format!("{}: {role}", p.name))
        })
        .collect();
    if roles.is_empty() {
        format!("program {index}")
    } else if roles.len() > 3 {
        format!("{} (+{} aliases)", roles[..3].join(" / "), roles.len() - 3)
    } else {
        roles.join(" / ")
    }
}

fn empty_graph() -> VizGraph {
    VizGraph {
        tensors: Vec::new(),
        nodes: Vec::new(),
        blocks: Vec::new(),
        inputs: Vec::new(),
        outputs: Vec::new(),
        summary: VizSummary {
            total_params: 0,
            total_weight_bytes: 0,
            num_blocks: 0,
            category_counts: BTreeMap::new(),
            total_nodes: 0,
            total_tensors: 0,
        },
    }
}

fn completion(id: TensorId, producer: NodeId) -> VizTensor {
    VizTensor {
        id,
        name: String::new(),
        dtype: DType::Bool,
        shape: "scheduler completion".into(),
        elements: None,
        bytes: None,
        origin: Origin::Node(producer),
        copies: 1,
    }
}

fn program_network(
    blob: &DevBlobMetadata,
    program: &DevProg,
    id: &str,
    title: &str,
    metadata: Option<&PipelineMetadata>,
) -> Result<VizNetwork, String> {
    program_graph(
        blob,
        ProgramView {
            rows: program.t,
            insts: &program.insts,
            stream: &program.stream,
            waits: &program.waits,
            succs: &program.succs,
        },
        id,
        title,
        metadata,
    )
}

struct ProgramView<'a> {
    rows: u32,
    insts: &'a [packet::dev::DevInst64],
    stream: &'a [packet::dev::StreamEnt],
    waits: &'a [packet::dev::Wait],
    succs: &'a [u32],
}

fn depth_networks(
    blob: &DevBlobMetadata,
    file: usize,
    label: &str,
) -> Result<Vec<VizNetwork>, String> {
    use plow_asset::fish_depth::{Manifest, PROGRAMS_SECTION, SECTION};
    let section = |name| {
        blob.sections
            .iter()
            .find(|(_, n, _)| n == name)
            .map(|(_, _, b)| b.as_slice())
    };
    let (manifest, programs) = match (section(SECTION), section(PROGRAMS_SECTION)) {
        (None, None) => return Ok(Vec::new()),
        (Some(m), Some(p)) => (m, p),
        _ => {
            return Err(format!(
                "{label}: depth manifest/program sections must be supplied together"
            ))
        }
    };
    let manifest: Manifest =
        serde_json::from_slice(manifest).map_err(|e| format!("{label}: {e}"))?;
    let programs = plow_asset::aux_program::parse(programs, blob.n_cu, blob.tensors.len())?;
    manifest.validate(programs.programs.len(), blob.tensors.len())?;
    programs
        .programs
        .iter()
        .enumerate()
        .map(|(i, p)| {
            if manifest.rows[i] != p.rows {
                return Err(format!(
                    "{label}: depth row count disagrees with program {i}"
                ));
            }
            let mut net = program_graph(
                blob,
                ProgramView {
                    rows: p.rows,
                    insts: &p.insts,
                    stream: &p.stream,
                    waits: &p.waits,
                    succs: &p.succs,
                },
                &format!("packet{file}.depth{i}"),
                &format!("{label} · depth: {} rows", p.rows),
                None,
            )?;
            net.metadata.as_mut().unwrap()["depth"] =
                serde_json::to_value(&manifest).map_err(|e| e.to_string())?;
            Ok(net)
        })
        .collect()
}

fn program_graph(
    blob: &DevBlobMetadata,
    program: ProgramView<'_>,
    id: &str,
    title: &str,
    metadata: Option<&PipelineMetadata>,
) -> Result<VizNetwork, String> {
    let mut graph = empty_graph();
    let names: Vec<_> = blob.tensors.iter().map(|t| t.name.as_str()).collect();
    let mut producer: BTreeMap<u32, BTreeSet<usize>> = BTreeMap::new();
    for entry in program.stream {
        for &counter in &program.succs
            [entry.succ_ofs as usize..entry.succ_ofs as usize + entry.succ_len as usize]
        {
            producer
                .entry(counter)
                .or_default()
                .insert(entry.inst as usize);
        }
    }
    let mut dependencies: Vec<BTreeSet<usize>> = vec![BTreeSet::new(); program.insts.len()];
    for entry in program.stream {
        for wait in &program.waits
            [entry.wait_ofs as usize..entry.wait_ofs as usize + entry.wait_len as usize]
        {
            if wait.threshold == 0 {
                continue;
            }
            if let Some(from) = producer.get(&wait.id) {
                dependencies[entry.inst as usize].extend(from.iter().copied());
            }
        }
    }
    let mut buffers = BTreeSet::new();
    for (index, instruction) in program.insts.iter().enumerate() {
        let decoded = packet::disasm::disasm(index, instruction, &names);
        let mut attrs = BTreeMap::new();
        for value in &decoded.ints {
            attrs.insert(
                value
                    .name
                    .map(str::to_string)
                    .unwrap_or_else(|| format!("i{}", value.slot)),
                serde_json::json!(value.value),
            );
        }
        for value in &decoded.floats {
            if value.value.is_finite() {
                attrs.insert(
                    value
                        .name
                        .map(str::to_string)
                        .unwrap_or_else(|| format!("f{}", value.slot)),
                    serde_json::json!(value.value),
                );
            }
        }
        let operands: Vec<_> = decoded.tensors.iter().filter_map(|t| {
            let handle = t.handle? as usize;
            let buffer = blob.tensors.get(handle)?;
            buffers.insert(handle);
            let shape = metadata.into_iter().flat_map(|m| m.pipelines()).flat_map(|p| p.tensors.values()).find(|t| t.name == buffer.name);
            Some(serde_json::json!({"slot":t.slot,"role":t.name,"name":buffer.name,"bytes":buffer.bytes,"dtype":shape.map(|s|s.dtype),"shape":shape.map(|s|&s.shape)}))
        }).collect();
        let block = format!("ops.{}", index / 48);
        if index % 48 == 0 {
            graph.blocks.push(VizBlock {
                id: block.clone(),
                stage: Some("program".into()),
                node_ids: Vec::new(),
                param_count: 0,
                weight_bytes: 0,
            });
        }
        let nid = NodeId(index as u32);
        let output = TensorId(index as u32);
        let inputs = dependencies[index]
            .iter()
            .filter(|&&p| p < index)
            .map(|&p| TensorId(p as u32))
            .collect();
        if dependencies[index].iter().any(|&p| p >= index) {
            return Err(format!(
                "{title}: non-topological scheduler dependencies at instruction {index}"
            ));
        }
        let op = decoded.op_name.unwrap_or("unknown");
        let category = category(decoded.op, op);
        *graph.summary.category_counts.entry(category).or_insert(0) += 1;
        graph.nodes.push(VizNode { id: nid, op, category, detail: serde_json::json!({"Packet": {"attributes":attrs,"buffers":operands,"workgroups":instruction.blocks,"raw":{"op":instruction.op,"t":instruction.t,"i":instruction.i,"fj":instruction.fj}}}), inputs, output, block: Some(block), stage: Some("program".into()), param_count: 0, weight_bytes: 0 });
        graph.tensors.push(completion(output, nid));
        graph.blocks.last_mut().unwrap().node_ids.push(nid);
    }
    let consumed: BTreeSet<_> = graph
        .nodes
        .iter()
        .flat_map(|n| n.inputs.iter().map(|t| t.0))
        .collect();
    graph.outputs = graph
        .nodes
        .iter()
        .map(|n| n.output)
        .filter(|t| !consumed.contains(&t.0))
        .collect();
    graph.summary.total_nodes = graph.nodes.len();
    graph.summary.total_tensors = graph.tensors.len();
    graph.summary.num_blocks = graph.blocks.len() as u32;
    graph.summary.total_weight_bytes = buffers.iter().map(|&i| blob.tensors[i].bytes).sum();
    let stage = VizStage {
        id: "program".into(),
        name: "Compiled operations".into(),
        description:
            "Scheduler dependency graph; details show named ISA operands and declared buffers"
                .into(),
        modules: Vec::new(),
        block_count: graph.summary.num_blocks,
        param_count: 0,
        input_shapes: Vec::new(),
        output_shapes: Vec::new(),
    };
    Ok(VizNetwork {
        kind: "packet",
        id: id.into(),
        title: title.into(),
        description: format!(
            "{} native instructions; buffer byte counts are capacities, not parameter counts",
            program.insts.len()
        ),
        stages: vec![stage],
        graph,
        checkpoint: None,
        metadata: Some(
            serde_json::json!({"rows":program.rows,"compute_units":blob.n_cu,"target":blob.target}),
        ),
    })
}

fn category(op: Option<packet::dev::DevOp>, name: &str) -> OpCategory {
    if name.contains("Conv") {
        return OpCategory::Convolution;
    }
    let classes = op.map(packet::opclass::op_classes).unwrap_or(&[]);
    if classes.contains(&"attention") {
        OpCategory::Attention
    } else if classes.contains(&"gemm") || classes.contains(&"gemv") {
        OpCategory::Linear
    } else if classes.contains(&"norm") {
        OpCategory::Normalization
    } else if classes.contains(&"moe") || classes.contains(&"indexer") {
        OpCategory::Routing
    } else if classes.contains(&"elementwise") || classes.contains(&"recurrent") {
        OpCategory::Elementwise
    } else {
        OpCategory::Layout
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Scratch(std::path::PathBuf);

    impl Scratch {
        fn new() -> Self {
            let path = std::env::temp_dir().join(format!(
                "plow-viz-{}-{}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ));
            std::fs::create_dir_all(&path).unwrap();
            Self(path)
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    // Native instructions exercise the reader; speech geometry comes from the production contract.
    fn speech_fixture(root: &Path, profile: &str) -> (packet::devbuild::Model, PacketPipelines) {
        use devgen::pipeline::{CausalPipelineSpec, CausalPipelineTensors};
        let (mut model, _) = rnnt_fixture();
        model.progs.truncate(2);
        model.prog_t = vec![2, 1];
        let tokens = model.tensors.len() as u32;
        for (name, bytes) in [
            ("fixture.tokens", 64),
            ("fixture.positions", 64),
            ("fixture.lengths", 4),
        ] {
            model.tensors.push(packet::devbuild::TensorDecl {
                name: name.into(),
                bytes,
                init: None,
            });
        }
        std::fs::write(root.join(devgen::tts::MAYA2_CONTROLLER), "{}").unwrap();
        let spec = CausalPipelineSpec {
            name: "fixture",
            max_context: 16,
            hidden: 32,
            decode_capacity: 1,
            overlay_rows: 0,
            ordered_dispatch: false,
            tensors: CausalPipelineTensors {
                tokens,
                positions: tokens + 1,
                kv_lengths: tokens + 2,
                overlay: None,
                overlay_index: None,
            },
        };
        let section = if profile.starts_with("chatterbox") {
            devgen::tts::t3_pipeline_section(
                &model,
                spec,
                tokens + 2,
                &[],
                root,
                profile == "chatterbox-mtl",
            )
        } else {
            devgen::tts::speech_pipeline_section(
                &mut model,
                spec,
                devgen::tts::profile(profile).unwrap(),
                200_000,
                root,
            )
        }
        .unwrap();
        (model, serde_json::from_slice(&section.data).unwrap())
    }

    fn write_contract(path: &Path, model: &packet::devbuild::Model, meta: &PacketPipelines) {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        let section = packet::devbuild::SectionData {
            kind: packet::devbuild::SECT_METADATA,
            name: plow_asset::packet_pipeline::SECTION.into(),
            data: serde_json::to_vec(meta).unwrap(),
        };
        std::fs::write(path, model.to_blob_v6(&[section])).unwrap();
    }

    fn write_speech(root: &Path, profile: &str) -> (packet::devbuild::Model, PacketPipelines) {
        let (model, meta) = speech_fixture(root, profile);
        for (role, name) in &meta.pipelines[0].strings {
            if role.ends_with(".packet")
                || matches!(role.as_str(), "clone.encoder" | "text.encoder")
            {
                let mut companion = meta.clone();
                companion.pipelines[0].strings.clear();
                companion.pipelines[0].name = role.clone();
                companion.pipelines[0].driver = "forward.v1".into();
                let path = root.join(name);
                let path = if path.extension().is_none() {
                    path.join("model.pkt")
                } else {
                    path
                };
                write_contract(&path, &model, &companion);
            }
        }
        write_contract(&root.join("model.pkt"), &model, &meta);
        (model, meta)
    }

    #[test]
    fn tts_profiles_discover_companions_and_export_the_same_bundle() {
        for profile in [
            "orpheus",
            "veena",
            "fish",
            "breeze",
            "maya2",
            "chatterbox",
            "chatterbox-mtl",
        ] {
            let scratch = Scratch::new();
            let (_, meta) = write_speech(&scratch.0, profile);
            let bundle = bundle_from_path(&scratch.0.join("model.pkt")).unwrap();
            assert_eq!(bundle.modality, ModelModality::Tts, "{profile}");
            let companions = meta.pipelines[0]
                .strings
                .keys()
                .filter(|k| {
                    k.ends_with(".packet") || matches!(k.as_str(), "clone.encoder" | "text.encoder")
                })
                .count();
            assert_eq!(bundle.networks[0].stages.len(), 1 + companions, "{profile}");
            assert_eq!(
                bundle.networks[0]
                    .graph
                    .nodes
                    .iter()
                    .map(|n| n.inputs.len())
                    .sum::<usize>(),
                companions
            );
            nn_graph::viz::dump_assets(&bundle, &scratch.0.join("export"), None).unwrap();
            let exported: serde_json::Value = serde_json::from_slice(
                &std::fs::read(scratch.0.join("export/model.json")).unwrap(),
            )
            .unwrap();
            assert_eq!(exported, serde_json::to_value(&bundle).unwrap());
            let html = std::fs::read_to_string(scratch.0.join("export/index.html")).unwrap();
            assert!(html.contains("Speech pipeline"));
            assert!(html.contains("Explore operations"));
            let svg = std::fs::read_to_string(scratch.0.join("export/pipeline.svg")).unwrap();
            assert!(!svg.contains("0.0%"));
            assert!(!svg.contains("<line"));
            assert_eq!(svg.matches("data-from=").count(), companions);
            for node in &bundle.networks[0].graph.nodes {
                for input in &node.inputs {
                    assert!(svg.contains(&format!(
                        "data-from=\"{}\" data-to=\"{}\"",
                        input.0, node.id.0
                    )));
                }
            }
        }
    }

    #[test]
    fn tts_rejects_missing_escaping_and_incompatible_companions() {
        let scratch = Scratch::new();
        let (model, mut meta) = write_speech(&scratch.0, "orpheus");
        for (key, value) in [
            ("audio.sample_rate", 1),
            ("codec.frame_codes", 1),
            ("codec.frame_samples", 1),
            ("codec.codebook", 1),
        ] {
            let mut codec = meta.clone();
            codec.pipelines[0].strings.clear();
            codec.pipelines[0].parameters.insert(key.into(), value);
            write_contract(&scratch.0.join("codec.pkt"), &model, &codec);
            assert!(bundle_from_path(&scratch.0.join("model.pkt"))
                .unwrap_err()
                .contains(key));
        }
        std::fs::remove_file(scratch.0.join("codec.pkt")).unwrap();
        assert!(bundle_from_path(&scratch.0.join("model.pkt"))
            .unwrap_err()
            .contains("missing"));
        for name in ["../codec.pkt", "/codec.pkt"] {
            meta.pipelines[0]
                .strings
                .insert("codec.packet".into(), name.into());
            write_contract(&scratch.0.join("model.pkt"), &model, &meta);
            assert!(bundle_from_path(&scratch.0.join("model.pkt"))
                .unwrap_err()
                .contains("invalid packet dependency"));
        }
        let outside = Scratch::new();
        write_contract(&outside.0.join("codec.pkt"), &model, &meta);
        std::os::unix::fs::symlink(outside.0.join("codec.pkt"), scratch.0.join("codec.pkt"))
            .unwrap();
        meta.pipelines[0]
            .strings
            .insert("codec.packet".into(), "codec.pkt".into());
        write_contract(&scratch.0.join("model.pkt"), &model, &meta);
        assert!(bundle_from_path(&scratch.0.join("model.pkt"))
            .unwrap_err()
            .contains("escapes"));
    }

    fn rnnt_fixture() -> (packet::devbuild::Model, packet::devbuild::SectionData) {
        use devgen::rnnt::{LinearWeights, LstmWeights, RnntSpec, RnntWeights};
        let linear = |weight, bias| LinearWeights { weight, bias };
        let lstm = [LstmWeights {
            input: linear("pred.ih.w", "pred.ih.b"),
            recurrent: linear("pred.hh.w", "pred.hh.b"),
        }];
        let weights = RnntWeights {
            prompt_in: linear("prompt.in.w", "prompt.in.b"),
            prompt_out: linear("prompt.out.w", "prompt.out.b"),
            encoder: linear("encoder.w", "encoder.b"),
            embedding: "pred.embedding",
            lstm: &lstm,
            predictor: linear("pred.out.w", "pred.out.b"),
            output: linear("joint.w", "joint.b"),
        };
        let packets = devgen::rnnt::lower(
            RnntSpec {
                frames: 2,
                encoder_width: 32,
                prompt_count: 1,
                prompt_width: 32,
                vocabulary: 16,
                durations: 0,
                predictor_width: 32,
                joint_width: 32,
                joint_batch: 1,
            },
            &weights,
            1,
        )
        .unwrap();
        let section = packets.pipeline_section(15, 10, 0).unwrap();
        (packets.model, section)
    }

    #[test]
    fn depth_sections_reuse_auxiliary_program_parser_and_common_graph() {
        use plow_asset::fish_depth::{Manifest, Tensors, PROGRAMS_SECTION, SECTION, VERSION};
        let (model, pipeline) = rnnt_fixture();
        let mut blob = DevBlobMetadata::parse(&model.to_blob()).unwrap();
        let p = &blob.progs[0];
        let bytes = plow_asset::aux_program::encode(
            blob.n_cu,
            blob.tensors.len(),
            &[plow_asset::program::Program {
                rows: 1,
                role: packet::devbuild::ProgramRole::DecodeRung { rows: 1 },
                n_counter: p.n_counter,
                insts: &p.insts,
                stream: &p.stream,
                stream_ofs: &p.stream_ofs,
                stream_len: &p.stream_len,
                waits: &p.waits,
                succs: &p.succs,
                gq_stream: &p.gq_stream,
                gq_seg_ofs: &p.gq_seg_ofs,
                l2_domains: 0,
            }],
        )
        .unwrap();
        let mut manifest = Manifest {
            version: VERSION,
            rows: vec![1],
            codebooks: 16,
            semantic_base: 0,
            semantic_tokens: 2048,
            stop_class: 2048,
            ras_window: 64,
            sampler: plow_asset::fish_depth::Sampler::Breeze,
            pair_cfg: true,
            depth_sampling: None,
            packed: None,
            planned_rows: false,
            slot_memory: Vec::new(),
            tensors: Tensors {
                params: 0,
                top_k: 0,
                seed: 0,
                step: 0,
                history: 0,
                hidden_row: 0,
                frame_codes: 0,
                prompt_codes: 0,
                force: None,
            },
        };
        let section = |name: &str, data| packet::devbuild::SectionData {
            kind: packet::devbuild::SECT_METADATA,
            name: name.into(),
            data,
        };
        let scratch = Scratch::new();
        let path = scratch.0.join("model.pkt");
        std::fs::write(
            &path,
            model.to_blob_v6(&[
                pipeline,
                section(SECTION, serde_json::to_vec(&manifest).unwrap()),
                section(PROGRAMS_SECTION, bytes.clone()),
            ]),
        )
        .unwrap();
        let bundle = bundle_from_path(&path).unwrap();
        let depth = bundle
            .networks
            .iter()
            .find(|n| n.id == "packet0.depth0")
            .unwrap();
        assert_eq!(depth.graph.nodes.len(), blob.progs[0].insts.len());
        assert_eq!(depth.metadata.as_ref().unwrap()["depth"]["pair_cfg"], true);
        assert!(
            bundle.networks[0].graph.nodes[0].detail["Packet"]["programs"]
                .as_array()
                .unwrap()
                .iter()
                .any(|p| p["id"] == depth.id)
        );
        blob.sections.push((
            packet::devbuild::SECT_METADATA,
            PROGRAMS_SECTION.into(),
            bytes,
        ));
        assert!(depth_networks(&blob, 0, "fixture")
            .unwrap_err()
            .contains("together"));
        manifest.rows[0] = 2;
        blob.sections.push((
            packet::devbuild::SECT_METADATA,
            SECTION.into(),
            serde_json::to_vec(&manifest).unwrap(),
        ));
        assert!(depth_networks(&blob, 0, "fixture")
            .unwrap_err()
            .contains("row count"));
    }

    #[test]
    fn rnnt_all_encoder_predictor_and_joint_programs_are_inspectable() {
        let (model, section) = rnnt_fixture();
        let blob = DevBlobMetadata::parse(&model.to_blob_v6(&[section])).unwrap();
        for (i, program) in blob.progs.iter().enumerate() {
            let net = program_network(&blob, program, "rnnt", "RNNT fixture", None).unwrap();
            assert_eq!(net.graph.nodes.len(), model.progs[i].insts.len());
            assert!(net.graph.blocks.iter().all(|b| b.node_ids.len() <= 48));
        }
        assert!(blob
            .progs
            .iter()
            .flat_map(|p| &p.insts)
            .any(|i| i.op == packet::dev::DevOp::LstmCellF32 as u16));
    }

    #[test]
    fn qwen_encoder_preserves_declared_program_and_buffer_geometry() {
        let p = devgen::asr::qwen::lower_audio_encoder(
            50,
            1,
            devgen::asr::qwen::AudioDims {
                width: 64,
                ffn_width: 128,
                layers: 2,
                output_width: 128,
            },
        )
        .unwrap();
        let section = p.pipeline_section(50).unwrap();
        let blob = DevBlobMetadata::parse(&p.prefix.model.to_blob_v6(&[section])).unwrap();
        for program in &blob.progs {
            program_network(&blob, program, "qwen", "Qwen encoder fixture", None).unwrap();
        }
    }

    /// Explicit operator fixtures for browser checks; these do not contain checkpoint weights.
    #[test]
    #[ignore = "writes reusable browser fixtures to temporary storage"]
    fn dump_speech_visualization_fixtures() {
        let root = std::env::temp_dir().join("plow-viz-speech-fixtures");
        std::fs::create_dir_all(&root).unwrap();
        let vad = devgen::vad::lower(1).unwrap();
        let vsection = vad.pipeline_section().unwrap();
        let (rnnt, rsection) = rnnt_fixture();
        for (name, model, section) in [
            ("vad-fixture", vad.model, vsection),
            ("rnnt-fixture", rnnt, rsection),
        ] {
            let path = root.join(format!("{name}.pkt"));
            std::fs::write(&path, model.to_blob_v6(&[section])).unwrap();
            let bundle = bundle_from_path(&path).unwrap();
            nn_graph::viz::dump_assets(&bundle, &root.join(name), None).unwrap();
        }
        println!("Speech visualization fixtures: {}", root.display());
        for profile in [
            "orpheus",
            "veena",
            "fish",
            "breeze",
            "maya2",
            "chatterbox",
            "chatterbox-mtl",
        ] {
            let dir = root.join(profile);
            std::fs::create_dir_all(&dir).unwrap();
            write_speech(&dir, profile);
            let bundle = bundle_from_path(&dir.join("model.pkt")).unwrap();
            nn_graph::viz::dump_assets(&bundle, &dir.join("export"), None).unwrap();
        }
    }

    #[test]
    fn vad_uses_existing_lowering_reader_and_disassembler() {
        let vad = devgen::vad::lower(1).unwrap();
        let section = vad.pipeline_section().unwrap();
        let blob = DevBlobMetadata::parse(&vad.model.to_blob_v6(&[section])).unwrap();
        let section = blob
            .sections
            .iter()
            .find(|(_, name, _)| name == plow_asset::packet_pipeline::SECTION)
            .unwrap();
        let metadata: PacketPipelines = serde_json::from_slice(&section.2).unwrap();
        let metadata = PipelineMetadata::V1(metadata);
        let net = program_network(&blob, &blob.progs[0], "vad", "VAD", Some(&metadata)).unwrap();
        assert_eq!(net.graph.nodes.len(), vad.model.progs[0].insts.len());
        assert!(net.graph.nodes.iter().any(|n| n.op == "LstmCellF32"));
        let conv = net
            .graph
            .nodes
            .iter()
            .find(|n| n.op == "Conv1dF32")
            .unwrap();
        assert_eq!(
            conv.detail["Packet"]["buffers"][1]["bytes"],
            devgen::vad::FRAME as u64 * 4 + devgen::vad::CONTEXT as u64 * 4
        );
        assert!(net.graph.nodes.iter().any(|n| !n.inputs.is_empty()));
    }
}
