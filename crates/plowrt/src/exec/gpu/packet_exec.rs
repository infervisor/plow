//! A self-contained packet (weights embedded, no language-model step protocol) on the CUDA
//! interpreter: the `PacketRuntime` for forward pipelines such as the Qwen3-ASR audio encoder.
//! Every program runs as one cooperative launch per global-queue segment of the speech object
//! (`interp_sm90a_speech.cubin`, `interp_sm120_speech.cubin` or `interp_sm89_speech.cubin`, the
//! interpreter with the FP32 speech arms).

use std::path::Path;
use std::sync::Arc;

use packet::dev::{DevOp, DevProgram, CTR_STRIDE};

use super::{pod_bytes, BLOCK};
use crate::asset::devblob::DevBlob;
use crate::device::cuda::{CudaBackend, CudaEvent, CudaStream, GraphExec, KernelFn, PinnedHost};
use crate::device::{Backend, DeviceMem, Module};
use crate::exec::packet_runtime::{
    check_copy, check_region, check_transfer, PacketBuffer, PacketData, PacketRuntime,
    PacketTensor, PacketWrite,
};
use crate::{Result, RuntimeError};

#[path = "handoff_packet.rs"]
mod handoff;
pub(crate) use handoff::PacketHandoffProvider;

#[cfg(test)]
#[test]
#[ignore = "CPU-only actual packet check; HANDOFF_TEST_ASSETS contains codec.pkt; use a quiet build window"]
fn handoff_native_codec_types_are_derived_from_actual_instructions() {
    let path =
        std::path::PathBuf::from(std::env::var("HANDOFF_TEST_ASSETS").unwrap()).join("codec.pkt");
    let image = crate::asset::packet_image::PacketImage::open(&path).unwrap();
    let asset = crate::exec::packet_runtime::PacketAsset::from_bytes(&image.raw).unwrap();
    let codec = asset
        .pipelines()
        .iter()
        .find(|p| p.driver == "codec.v1")
        .unwrap();
    let mut checked = 0;
    image.blob.with_packet_view(|packet| {
        for (role, &program) in &codec.programs {
            if !role.starts_with("step.") || !role.ends_with(".f1") {
                continue;
            }
            let inferred = plow_asset::packet_pipeline_native::NativePacketTypes::infer(
                packet,
                program as usize,
            )
            .unwrap();
            for (name, tensor) in &codec.tensors {
                if name != "codes" && name != "pcm" && !name.starts_with("stream.") {
                    continue;
                }
                let handle = packet
                    .tensors
                    .iter()
                    .position(|t| t.name == tensor.name)
                    .unwrap();
                if inferred.access(handle).is_some() {
                    assert_eq!(
                        inferred.dtype(handle).unwrap(),
                        tensor.dtype,
                        "{role}/{name}"
                    );
                    checked += 1;
                }
            }
        }
    });
    assert!(checked >= 6, "no real stateful codec interfaces checked");
    eprintln!("native codec types/accesses checked={checked}");
}

/// The speech object and its entry symbol for a compute capability.
pub(super) fn speech_object(cc: (u32, u32)) -> (&'static str, &'static str) {
    match cc {
        (8, 9) => (
            "interp_sm89_speech.cubin",
            "_Z18interp_sm89_speech11PlowProgram",
        ),
        (10, _) | (12, _) => (
            "interp_sm120_speech.cubin",
            "_Z19interp_sm120_speech11PlowProgram",
        ),
        _ => (
            "interp_sm90a_speech.cubin",
            "_Z19interp_sm90a_speech11PlowProgram",
        ),
    }
}
fn in_range(mem: &DeviceMem, offset: usize, bytes: usize) -> Result<()> {
    if offset as u64 + bytes as u64 > mem.len {
        return Err(RuntimeError::Device(format!(
            "packet transfer out of range: off {offset} + {bytes} > {}",
            mem.len
        )));
    }
    Ok(())
}

/// FP32 speech ops are 163..=204, bit `op - 163` of `PLOW_SPEECH_OPS` (`op_speech_f32.cuh`).
const SPEECH_OP0: u32 = 163;

fn speech_ops(blob: &DevBlob) -> u64 {
    blob.progs
        .iter()
        .flat_map(|p| &p.insts)
        .map(|d| u32::from(d.op))
        .filter(|op| (SPEECH_OP0..=204).contains(op))
        .fold(0, |mask, op| mask | 1 << (op - SPEECH_OP0))
}

fn speech_conv_bf16_variants(blob: &DevBlob) -> u64 {
    blob.progs.iter().flat_map(|p| &p.insts).fold(0, |m, d| {
        m | plow_asset::speech_flag_layout::conv_bf16_variants(d.op, &d.i, d.fj[2])
    })
}

fn speech_conv_arms(blob: &DevBlob) -> u64 {
    blob.progs.iter().flat_map(|p| &p.insts).fold(0, |m, d| {
        m | plow_asset::speech_flag_layout::conv_arms(d.op, &d.i, d.fj[1], d.fj[2])
    })
}

/// Speech-object capability bits the instructions need (`plow_speech_caps_speech`): bit 0 a
/// GemvF32Out row view (flag bit 12), bit 1 an AttentionF32 query view (flag bit 6),
/// bit 2 FP16 LayerNorm output / projection operands, bit 3 ring KV, bit 4 a chunk-canonical K
/// bf16 taps conv (an object without it would run the unchunked chain), bit 5 the fp16 short
/// AttentionF32 (`j1` bit 31).
pub(super) fn speech_caps_required<'a>(
    insts: impl IntoIterator<Item = &'a packet::dev::DevInst64>,
) -> u32 {
    use packet::dev::DevOp;
    insts.into_iter().fold(0, |caps, d| {
        caps | match DevOp::from_u16(d.op) {
            Some(DevOp::GemvF32Out) => {
                u32::from(d.i[7] & 4096 != 0) | u32::from(d.i[7] & 0x6000 != 0) << 2
            }
            Some(DevOp::AttentionF32) => {
                u32::from(d.i[6] & 64 != 0) << 1
                    | u32::from(plow_asset::speech_flag_layout::ring_kv(d.op, &d.i)) << 3
                    | u32::from(plow_asset::speech_flag_layout::half_attention(d.op, d.fj[2])) << 5
            }
            Some(DevOp::StreamTailF32) if plow_asset::speech_flag_layout::ring_kv(d.op, &d.i) => 8,
            Some(DevOp::LayerNormF32) if d.i[2] & 8 != 0 => 4,
            Some(DevOp::Conv1dF32)
                if plow_asset::speech_flag_layout::conv_bf16_chunked(d.op, d.fj[2]) =>
            {
                16
            }
            _ => 0,
        }
    })
}

/// The capability bits `required` that an object reporting `compiled` (`None` = no symbol) lacks.
pub(super) fn missing_speech_caps(required: u32, compiled: Option<u32>) -> u32 {
    required & !compiled.unwrap_or(0)
}

/// Speech-op semantics a packet needs beyond its op set, as the object's
/// `plow_speech_features_speech` bits (absent = none): bit 1 = Conv1dF32 / ConvTranspose1dF32
/// flag bit 3 (wgmma tiles spanning taps / phases at widths % 4), bit 2 = Conv1dF32 flag bit 30
/// (a split-image input), bit 3 = Conv1dF32 flag bit 17 without bit 14 (the bf16 taps conv).
/// Bit 0 is retired: it named the ragged semantics on flag bit 21, which now carries the
/// pinned-arms field.
pub(super) fn speech_features(insts: impl IntoIterator<Item = (u16, u32)>) -> u32 {
    insts.into_iter().fold(0, |m, (op, flags)| {
        let conv = op == DevOp::Conv1dF32 as u16;
        let ragged = (conv || op == DevOp::ConvTranspose1dF32 as u16) && flags >> 3 & 1 != 0;
        let split = conv && flags >> 30 & 1 != 0;
        let bf16 = conv && flags >> 17 & 1 != 0 && flags >> 14 & 1 == 0;
        m | u32::from(ragged) << 1 | u32::from(split) << 2 | u32::from(bf16) << 3
    })
}

/// Why `(op, conv flags, i2)` instructions cannot run on an object reporting features `have`,
/// when no missing feature bit says so: the conv flag bits 21 / 22 had other meanings before the
/// pinned-arms field took them.
pub(super) fn stale_speech_layout(
    insts: impl IntoIterator<Item = (u16, u32, u32)>,
    have: u32,
) -> Option<&'static str> {
    let (mut pinned, mut image_out, mut image_in) = (false, false, false);
    for (op, flags, i2) in insts {
        let conv = op == DevOp::Conv1dF32 as u16;
        pinned |= (conv || op == DevOp::ConvTranspose1dF32 as u16) && flags >> 21 & 1 != 0;
        image_in |= conv && flags >> 30 & 1 != 0;
        image_out |= op == DevOp::LayerNormF32 as u16 && i2 & 4 != 0;
    }
    if pinned && have & 1 != 0 {
        return Some(
            "the object reads conv flag bit 21 as ragged tiles, the packet as pinned arms",
        );
    }
    if image_out && !image_in {
        return Some("split-image outputs without a conv flag bit 30 reader: re-emit the packet");
    }
    None
}

/// Refuses `object` (a loaded interpreter `module`) for instructions whose first protected use
/// is `used`, from a packet declaring `declared`, when their speech-op flag layouts disagree
/// (`plow_asset::speech_flag_layout`).
pub(in crate::exec) fn check_speech_flag_layout(
    be: &CudaBackend,
    module: &Module,
    object: &Path,
    declared: Option<u32>,
    used: Option<u16>,
) -> Result<()> {
    use plow_asset::speech_flag_layout as layout;
    let have = be.module_global_u32(module, layout::OBJECT_SYMBOL)?;
    layout::check(declared, used, have)
        .map_err(|e| RuntimeError::Rejected(format!("{}: {e}", object.display())))
}

/// Rejects an object lacking a feature bit `required` has, or one whose conv flag layout
/// `stale` (see [`stale_speech_layout`]) rules out.
pub(super) fn check_speech_features(
    be: &CudaBackend,
    module: &Module,
    object: &Path,
    required: u32,
    stale: impl FnOnce(u32) -> Option<&'static str>,
) -> Result<()> {
    let have = be
        .module_global_u32(module, "plow_speech_features_speech")?
        .unwrap_or(0);
    if let Some(why) = stale(have) {
        return Err(RuntimeError::Rejected(format!(
            "{}: {why}",
            object.display()
        )));
    }
    if required & !have != 0 {
        return Err(RuntimeError::Rejected(format!(
            "{}: lacks speech features {:#x}; rebuild it for this packet",
            object.display(),
            required & !have
        )));
    }
    Ok(())
}

pub(super) fn missing_speech_ops(required: u64, compiled: u64) -> Vec<u32> {
    let missing = required & !compiled;
    (0..64)
        .filter(|b| missing >> b & 1 != 0)
        .map(|b| b + SPEECH_OP0)
        .collect()
}

#[cfg(test)]
#[test]
fn a_specialized_speech_object_must_cover_the_packet_ops() {
    assert!(missing_speech_ops(0b1011, 0b1111).is_empty());
    assert_eq!(missing_speech_ops(0b1011, 0b0011), vec![SPEECH_OP0 + 3]);
    assert_eq!(missing_speech_ops(1 << 41, 0), vec![204]);
}

#[cfg(test)]
#[test]
fn half_speech_operands_require_the_capability() {
    let mut i = packet::dev::DevInst64::default();
    i.op = DevOp::GemvF32Out as u16;
    for flags in [8192, 16384, 8192 | 16384] {
        i.i[7] = flags;
        assert_eq!(speech_caps_required([&i]), 4);
        assert_eq!(missing_speech_caps(4, Some(2)), 4);
        assert_eq!(missing_speech_caps(4, Some(6)), 0);
    }
    i.op = DevOp::LayerNormF32 as u16;
    i.i[2] = 8;
    assert_eq!(speech_caps_required([&i]), 4);
}

#[cfg(test)]
#[test]
fn ring_speech_operands_require_layout_capability() {
    let mut d = packet::dev::DevInst64::default();
    d.op = DevOp::AttentionF32 as u16;
    d.i[6] = 9 | (250 << 16);
    assert_eq!(speech_caps_required([&d]), 0);
    d.i[6] |= 0x100;
    assert_eq!(speech_caps_required([&d]), 8);
    assert_eq!(missing_speech_caps(8, Some(6)), 8);
    assert_eq!(missing_speech_caps(8, Some(14)), 0);
    d.op = DevOp::StreamTailF32 as u16;
    for mode in [2, 3] {
        d.i[7] = mode;
        assert_eq!(speech_caps_required([&d]), 8);
    }
}

#[cfg(test)]
#[test]
fn half_short_attention_requires_the_capability() {
    let mut d = packet::dev::DevInst64::default();
    d.op = DevOp::AttentionF32 as u16;
    d.fj[2] = 2048;
    assert_eq!(speech_caps_required([&d]), 0);
    d.fj[2] |= 1 << 31;
    assert_eq!(speech_caps_required([&d]), 32);
    assert_eq!(missing_speech_caps(32, Some(6 | 16)), 32);
    assert_eq!(missing_speech_caps(32, Some(38)), 0);
}

#[cfg(test)]
#[test]
fn chunked_bf16_convs_require_the_capability() {
    let mut d = packet::dev::DevInst64::default();
    d.op = DevOp::Conv1dF32 as u16;
    d.fj[2] = 0x66a000;
    assert_eq!(speech_caps_required([&d]), 0);
    d.fj[2] |= 1 << 24;
    assert_eq!(speech_caps_required([&d]), 0);
    d.fj[2] |= 10 << 24;
    assert_eq!(speech_caps_required([&d]), 16);
    assert_eq!(missing_speech_caps(16, Some(14)), 16);
    assert_eq!(missing_speech_caps(16, None), 16);
    d.fj[2] |= 1 << 14;
    assert_eq!(speech_caps_required([&d]), 0);
}

#[cfg(test)]
#[test]
fn ragged_conv_flags_require_the_speech_feature() {
    let (conv, convt) = (DevOp::Conv1dF32 as u16, DevOp::ConvTranspose1dF32 as u16);
    assert_eq!(speech_features([(conv, 0x6e000), (convt, 0xa000)]), 0);
    assert_eq!(speech_features([(conv, 0x6e000), (convt, 0xa008)]), 2);
    assert_eq!(speech_features([(conv, 0x6e000 | 1 << 21)]), 0);
    assert_eq!(speech_features([(DevOp::UnaryF32 as u16, 1 << 3)]), 0);
    assert_eq!(speech_features([(conv, 0x6e000 | 1 << 30)]), 4);
    assert_eq!(speech_features([(convt, 0xa000 | 1 << 30)]), 0);
    assert_eq!(speech_features([(conv, 0x6a000)]), 8);
    assert_eq!(speech_features([(convt, 0x6a000)]), 0);
}

#[cfg(test)]
#[test]
fn pre_integration_conv_flag_layouts_are_refused() {
    let (conv, ln) = (DevOp::Conv1dF32 as u16, DevOp::LayerNormF32 as u16);
    let pinned = [(conv, 0x6e000 | 1 << 21, 0)];
    assert!(stale_speech_layout(pinned, 1).is_some());
    assert!(stale_speech_layout(pinned, 0).is_none());
    assert!(stale_speech_layout(pinned, 2).is_none());
    assert!(stale_speech_layout([(conv, 0x6e000, 0)], 1).is_none());
    // A LayerNorm writing the image needs a bit-30 conv: bit 21 was the reader before.
    assert!(stale_speech_layout([(ln, 0, 4), (conv, 0x6e000 | 1 << 21, 0)], 6).is_some());
    assert!(stale_speech_layout([(ln, 0, 4), (conv, 0x6e000 | 1 << 30, 0)], 6).is_none());
    assert!(stale_speech_layout([(ln, 0, 3), (conv, 0x6e000, 4)], 6).is_none());
}

#[cfg(test)]
#[test]
fn test_speech_object_resolution() {
    assert_eq!(speech_object((8, 9)).0, "interp_sm89_speech.cubin");
    assert_eq!(speech_object((9, 0)).0, "interp_sm90a_speech.cubin");
    assert_eq!(speech_object((10, 0)).0, "interp_sm120_speech.cubin");
    assert_eq!(speech_object((12, 0)).0, "interp_sm120_speech.cubin");
}

#[cfg(test)]
#[test]
fn speech_object_symbols_follow_the_cmake_mangling() {
    for cc in [(8, 9), (9, 0), (10, 0), (12, 0)] {
        let (file, symbol) = speech_object(cc);
        let stem = file.strip_suffix(".cubin").unwrap();
        assert_eq!(symbol, format!("_Z{}{stem}11PlowProgram", stem.len()));
    }
}

struct Program {
    kernarg: DevProgram,
    segments: usize,
    /// Per segment: one wgmma conv alone (its device instruction, blocks), launched on the conv
    /// object's kernel entry when loaded.
    conv_segments: Vec<Option<(u64, u32)>>,
    counters: u64,
    counter_bytes: usize,
    _tables: Vec<DeviceMem>,
}

pub(crate) struct SharedPacketTensors {
    pub backend: Arc<CudaBackend>,
    pub memory: Arc<DeviceMem>,
    pub views: std::collections::BTreeMap<String, (u64, u64, bool)>,
}

pub struct CudaPacketRuntime {
    be: Arc<CudaBackend>,
    stream: CudaStream,
    events: (CudaEvent, CudaEvent),
    module: Module,
    function: KernelFn,
    grid: u32,
    smem: u32,
    conv: Option<ConvObject>,
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
    input_stage: Option<PinnedHost>,
    input_failed: bool,
    shared: Option<SharedPacketTensors>,
    /// Last field: dropped after every allocation above, so [`resident_in`] turns false only once
    /// the device memory is back.
    _resident: Arc<()>,
}

/// The conv-only speech object (`<speech stem>conv.cubin`, sm_90a): `speech_conv_node(inst, tensors)`
/// runs one wgmma conv over `blocks` CTAs, for the segments devgen isolated to it (raw segments: the
/// stream orders them, no counters cross them).
struct ConvObject {
    _module: Module,
    function: KernelFn,
    smem: u32,
}

/// Loads the conv object beside `object` when a program has a conv segment. Absent, those
/// segments run on the speech object (same bits). Both objects come from one build: an object
/// whose layout, feature or arm stamps differ from the speech object's is stale.
fn load_conv_object(
    be: &CudaBackend,
    main: &Module,
    object: &Path,
    blob: &DevBlob,
) -> Result<Option<ConvObject>> {
    use plow_asset::speech_flag_layout as layout;
    let wanted = blob.progs.iter().any(|p| {
        layout::conv_object_segments(&p.gq_seg_ofs, &p.gq_stream, |i| {
            let d = &p.insts[i as usize];
            layout::conv_object_inst(d.op, d.fj[2])
        })
        .iter()
        .any(Option::is_some)
    });
    let stem = object
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or_default();
    let path = object.with_file_name(format!("{stem}conv.cubin"));
    if !wanted {
        return Ok(None);
    }
    let image = match std::fs::read(&path) {
        Ok(image) => image,
        Err(_) => {
            tracing::warn!(object = %path.display(), "packet has conv segments but no conv object: they run on the speech object");
            return Ok(None);
        }
    };
    let module = be.module_load(&image)?;
    let function = be.get_function(&module, "speech_conv_node")?;
    for symbol in [
        layout::OBJECT_SYMBOL,
        layout::CONV_ARMS_SYMBOL,
        layout::CONV_BF16_VARIANTS_SYMBOL,
        "plow_speech_features_speech",
        "plow_speech_conv_posts_lo_speech",
        "plow_speech_conv_posts_hi_speech",
        "plow_speech_ops_lo_speech",
        "plow_speech_ops_hi_speech",
    ] {
        if be.module_global_u32(main, symbol)?
            != be.module_global_u32(&module, &format!("{symbol}conv"))?
        {
            return Err(RuntimeError::Rejected(format!(
                "{}: {symbol} differs from {}; rebuild both objects",
                path.display(),
                object.display()
            )));
        }
    }
    let smem = be
        .module_global_u32(&module, "plow_arena_bytes_speechconv")?
        .unwrap_or(49152);
    be.set_max_dynamic_smem(function, smem)?;
    tracing::info!(object = %path.display(), smem, "speech conv object loaded");
    Ok(Some(ConvObject {
        _module: module,
        function,
        smem,
    }))
}

type Residents = parking_lot::Mutex<Vec<(std::path::PathBuf, u8, std::sync::Weak<()>)>>;

#[cfg(test)]
#[test]
#[ignore = "CUDA; DEPTH_PREFIX_TEST_ASSETS contains devgen's emitted prefix blocks; run through gpuq"]
fn persistent_depth_prefix_matches_embedding_rows() {
    let dir = std::path::PathBuf::from(std::env::var("DEPTH_PREFIX_TEST_ASSETS").unwrap());
    let be = Arc::new(CudaBackend::new(0).unwrap());
    let sanitize = std::env::var_os("DEPTH_PREFIX_SANITIZE").is_some();
    let batches: &[usize] = if sanitize {
        &[33]
    } else {
        &[1, 8, 32, 64, 128, 200, 224]
    };
    let table: Vec<u16> = (0..31 * 2048 * 1024)
        .map(|i| {
            0x3c00
                | ((i / 1024 * 37 + i % 1024 * 11) % 1024) as u16
                | ((((i / 1024 + i) % 2) as u16) << 15)
        })
        .collect();
    for &batch in batches {
        for level in [1, 9, 17, 25, 31] {
            let mut runtimes = [false, true].map(|cached| {
                let path = dir.join(format!("prefix.b{batch}.l{level}.cache{cached}.pkt"));
                CudaPacketRuntime::load_on(be.clone(), &path).unwrap()
            });
            for rt in &mut runtimes {
                let t = rt.tensor("w.maya2.depth.emb.weight").unwrap();
                rt.write_tensor(t, bytemuck::cast_slice(&table)).unwrap();
            }
            let rows = if level == 31 { 1 } else { level + 1 };
            for frame in 0..3 {
                let p0: Vec<f32> = (0..batch * 1024)
                    .map(|i| (i as f32 + frame as f32) / 1024.)
                    .collect();
                let indices: Vec<Vec<u32>> = (0..31)
                    .map(|k| {
                        (0..batch)
                            .map(|r| (k * 2048 + (r * 37 + frame * 17) % 2048) as u32)
                            .collect()
                    })
                    .collect();
                let expected: Vec<f32> = (0..batch * rows * 1024)
                    .map(|i| {
                        let (b, p, d) = (i / (rows * 1024), i / 1024 % rows, i % 1024);
                        let pos = if level == 31 { level } else { p };
                        if pos == 0 {
                            p0[b * 1024 + d]
                        } else {
                            let h = table[indices[pos - 1][b] as usize * 1024 + d];
                            f32::from_bits(
                                (u32::from(h & 0x8000) << 16)
                                    | 0x3f80_0000
                                    | (u32::from(h & 1023) << 13),
                            )
                        }
                    })
                    .collect();
                for rt in &mut runtimes {
                    for (name, data) in
                        std::iter::once(("act.depth.p0".to_owned(), bytemuck::cast_slice(&p0)))
                            .chain(indices.iter().enumerate().map(|(k, v)| {
                                (format!("act.depth.eidx.{k}"), bytemuck::cast_slice(v))
                            }))
                    {
                        if let Some(t) = rt.tensor(&name) {
                            rt.write_tensor(t, data).unwrap();
                        }
                    }
                    let x = rt.tensor("act.depth.x").unwrap();
                    let poison = vec![12345f32; x.bytes / 4];
                    rt.write_tensor(x, bytemuck::cast_slice(&poison)).unwrap();
                }
                runtimes[1].run_sequence(&[0]).unwrap();
                let mut timing = Vec::new();
                for arm in [0, 1, 0, 1] {
                    let rt = &mut runtimes[arm];
                    let mut times = Vec::new();
                    for repeat in 0..if sanitize { 1 } else { 15 } {
                        rt.run_sequence(&[arm]).unwrap();
                        if repeat >= 3 || sanitize {
                            times.push(rt.last_run_us());
                        }
                    }
                    times.sort_by(f64::total_cmp);
                    timing.push((arm, times[times.len() / 2]));
                    let t = rt.tensor("act.depth.x").unwrap();
                    let mut actual = vec![0f32; t.bytes / 4];
                    rt.read_tensor_at(t, 0, bytemuck::cast_slice_mut(&mut actual))
                        .unwrap();
                    assert_eq!(
                        &actual[..expected.len()],
                        expected.as_slice(),
                        "B{batch} L{level} frame{frame} arm{arm}"
                    );
                    assert!(actual[expected.len()..].iter().all(|&v| v == 12345.));
                }
                println!("depth-prefix batch={batch} level={level} frame={frame} exact=true abab_us={timing:?}");
            }
        }
    }
}

#[cfg(test)]
#[test]
#[ignore = "CUDA; CODEC_RING_TEST_ASSETS holds the layout-3 speech object; run through gpuq"]
fn ring_attention_matches_chronological_single_block() {
    use packet::devbuild::{Builder, Model, SectionData, SECT_METADATA};
    let dir = std::path::PathBuf::from(std::env::var("CODEC_RING_TEST_ASSETS").unwrap());
    let make = |active: usize, cap: usize, ring: bool| {
        let (history, width, rows, slots) = (249usize, 512usize, 2usize, cap.max(256));
        let mut b = Builder::new(132);
        b.force_uniseg();
        let out = b.tensor("out", (cap * rows * width * 4) as u64);
        let qkv = b.tensor("qkv", (cap * rows * width * 3 * 4) as u64);
        let state = b.tensor("state", (slots * history * width * 2 * 4) as u64);
        let slot = b.tensor("slot", (cap * 4) as u64);
        let length = b.tensor("length", (cap * 4) as u64);
        let pos = b.tensor("pos", (cap * 4) as u64);
        let z = b.tensor("z", (cap * (history + rows) * width * 3 * 4) as u64);
        let tail = |b: &mut Builder, mode, deps: &[u32]| {
            let units = if mode == 2 {
                (cap * rows * width * 2).div_ceil(2048)
            } else {
                132
            };
            b.emit(
                DevOp::StreamTailF32,
                (0..units.clamp(1, 132) as u32).collect(),
                deps,
                |d| {
                    d.t[0] = if mode == 0 {
                        z
                    } else {
                        packet::dev::TENSOR_NONE
                    };
                    d.t[1..6].copy_from_slice(&[
                        if mode == 1 { z } else { qkv },
                        state,
                        slot,
                        length,
                        pos,
                    ]);
                    d.i = [
                        cap as u32,
                        history as u32,
                        rows as u32,
                        (2 * width) as u32,
                        (3 * width) as u32,
                        width as u32,
                        if mode == 1 { 0 } else { (3 * width) as u32 },
                        mode,
                    ];
                    d.j = [slots as u32, 0];
                },
            )
        };
        let dep = if ring {
            None
        } else {
            let gather = tail(&mut b, 0, &[]);
            Some(tail(&mut b, 1, &[gather]))
        };
        let attention = b.emit(
            DevOp::AttentionF32,
            b.all(),
            &dep.into_iter().collect::<Vec<_>>(),
            |d| {
                if ring {
                    d.t[..6].copy_from_slice(&[out, qkv, state, slot, length, pos]);
                } else {
                    d.t[..4].copy_from_slice(&[out, z, z, z]);
                    d.t[7] = pos;
                }
                d.i = [
                    cap as u32,
                    rows as u32,
                    (if ring { history } else { history + rows }) as u32,
                    8,
                    64,
                    (3 * width) as u32,
                    (if ring { 0x109 } else { 9 }) | (250 << 16),
                    (if ring { slots } else { history }) as u32,
                ];
                d.f[0] = 0.125;
                d.j = [width as u32, (2 * width) as u32];
            },
        );
        if ring {
            tail(&mut b, 2, &[attention]);
        }
        let tensors = b.tensors();
        let model = Model {
            n_cu: 132,
            target: 0,
            tensors,
            progs: vec![b.finish()],
            prog_t: vec![1],
            kv_row_insts: vec![],
            gen: vec![],
        };
        let abi = plow_asset::abi::Abi {
            version: 1,
            contract: 2,
            must_understand: vec![],
            target: None,
            bindings: None,
            speech_flag_layout: Some(3),
        };
        let path = dir.join(format!("attention.b{active}.c{cap}.ring{ring}.pkt"));
        std::fs::write(
            &path,
            model.to_blob_v6(&[SectionData {
                kind: SECT_METADATA,
                name: plow_asset::abi::SECTION.into(),
                data: serde_json::to_vec(&abi).unwrap(),
            }]),
        )
        .unwrap();
        path
    };
    let be = Arc::new(CudaBackend::new(0).unwrap());
    let sanitize = std::env::var_os("CODEC_RING_SANITIZE").is_some();
    let cases: &[(usize, usize)] = if sanitize {
        &[(33, 64)]
    } else {
        &[
            (32, 32),
            (64, 64),
            (128, 128),
            (200, 256),
            (224, 256),
            (300, 320),
        ]
    };
    for &(active, cap) in cases {
        let slots = cap.max(256);
        let mut runtimes = [
            CudaPacketRuntime::load_on(be.clone(), &make(active, cap, false)).unwrap(),
            CudaPacketRuntime::load_on(be.clone(), &make(active, cap, true)).unwrap(),
        ];
        let qkv: Vec<f32> = (0..cap * 2 * 1536)
            .map(|i| {
                if i / (2 * 1536) < active {
                    ((i * 37 % 101) as f32 - 50.) * 0.003
                } else {
                    0.
                }
            })
            .collect();
        let slot: Vec<u32> = (0..cap).map(|r| ((r * 73) % slots) as u32).collect();
        let length: Vec<u32> = (0..cap).map(|r| if r < active { 2 } else { 0 }).collect();
        let pos: Vec<u32> = (0..cap).map(|r| if r < active { 256 } else { 0 }).collect();
        for rt in &mut runtimes {
            for (name, data) in [
                ("qkv", bytemuck::cast_slice(&qkv)),
                ("slot", bytemuck::cast_slice(&slot)),
                ("length", bytemuck::cast_slice(&length)),
                ("pos", bytemuck::cast_slice(&pos)),
            ] {
                let tensor = rt.tensor(name).unwrap();
                rt.write_tensor(tensor, data).unwrap();
            }
        }
        let mut output = [vec![0f32; cap * 2 * 512], vec![0f32; cap * 2 * 512]];
        let mut timing = Vec::new();
        for arm in [0, 1, 0, 1] {
            let rt = &mut runtimes[arm];
            let state = rt.tensor("state").unwrap();
            let mut times = Vec::new();
            for repeat in 0..if sanitize { 1 } else { 15 } {
                rt.write_tensors(&[PacketWrite {
                    tensor: state,
                    offset: 0,
                    data: PacketData::Zero(state.bytes),
                }])
                .unwrap();
                rt.run_sequence(&[0]).unwrap();
                if repeat >= 3 || sanitize {
                    times.push(rt.last_run_us());
                }
            }
            times.sort_by(f64::total_cmp);
            timing.push((arm, times[times.len() / 2]));
            rt.read_tensor_at(
                rt.tensor("out").unwrap(),
                0,
                bytemuck::cast_slice_mut(&mut output[arm]),
            )
            .unwrap();
        }
        let error = output[0]
            .iter()
            .zip(&output[1])
            .map(|(&a, &b)| (f64::from(a) - f64::from(b)).powi(2))
            .sum::<f64>();
        let norm = output[0].iter().map(|&v| f64::from(v).powi(2)).sum::<f64>();
        let relative = (error / norm.max(1.0e-30)).sqrt();
        assert!(relative < 1.0e-5, "{active}/{cap}: {relative}");
        assert!(output[1][active * 2 * 512..].iter().all(|&v| v == 0.));
        println!("ring-interpreter active={active} capacity={cap} rel_l2={relative:e} abab_us={timing:?}");
    }
}

/// Live runtimes by the asset directory their packet sits in and their device.
fn residents() -> &'static Residents {
    static R: std::sync::OnceLock<Residents> = std::sync::OnceLock::new();
    R.get_or_init(Default::default)
}

/// Whether a packet runtime loaded from `dir` onto `device` still holds device memory.
pub fn resident_in(dir: &Path, device: u8) -> bool {
    let mut r = residents().lock();
    r.retain(|(_, _, live)| live.strong_count() > 0);
    r.iter().any(|(d, o, _)| d == dir && *o == device)
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
        Self::load_with_views(be, path, None)
    }

    pub(crate) fn load_with_views(
        be: Arc<CudaBackend>,
        path: &Path,
        shared: Option<SharedPacketTensors>,
    ) -> Result<Self> {
        let ordinal = be.device_ordinal;
        let image = crate::asset::packet_image::PacketImage::open(path)?;
        image.blob.check_l2_dispatch(false)?;
        let (blob, raw) = (&image.blob, &image.raw);
        if let Some(shared) = &shared {
            if !be.shares_context(&shared.backend)
                || shared.views.iter().any(|(name, &(offset, bytes, _))| {
                    !blob
                        .tensors
                        .iter()
                        .any(|t| t.name == *name && t.bytes == bytes)
                        || !offset
                            .checked_add(bytes)
                            .is_some_and(|end| end <= shared.memory.len)
                })
            {
                return Err(RuntimeError::Rejected(
                    "shared packet tensor binding mismatch".into(),
                ));
            }
        }
        if !blob.gen.is_empty() || blob.tp.is_some() {
            return Err(RuntimeError::Rejected(format!(
                "{}: generated tensors or TP are not supported by the CUDA packet runtime",
                path.display()
            )));
        }
        let dir = path.parent().unwrap_or(Path::new("."));
        let (file, symbol) = speech_object(be.compute_capability());
        let object = dir.join(file);
        let image = std::fs::read(&object).map_err(|source| RuntimeError::Io {
            path: object.clone(),
            source,
        })?;
        let module = be.module_load(&image)?;
        let function = be.get_function(&module, symbol)?;
        // Before the op / capability checks: rebuilding the object cannot repair a stale packet.
        let layout = blob.speech_flag_layout(raw)?;
        check_speech_flag_layout(&be, &module, &object, layout.declared, layout.any())?;
        if let (Some(lo), Some(hi)) = (
            be.module_global_u32(&module, "plow_speech_ops_lo_speech")?,
            be.module_global_u32(&module, "plow_speech_ops_hi_speech")?,
        ) {
            let ops = missing_speech_ops(speech_ops(&blob), u64::from(hi) << 32 | u64::from(lo));
            if !ops.is_empty() {
                return Err(RuntimeError::Rejected(format!(
                    "{}: {} lacks speech ops {ops:?}; rebuild it for this packet",
                    path.display(),
                    object.display()
                )));
            }
        }
        for (symbol, required, what) in [
            (
                plow_asset::speech_flag_layout::CONV_ARMS_SYMBOL,
                speech_conv_arms(&blob),
                "conv arms",
            ),
            (
                plow_asset::speech_flag_layout::CONV_BF16_VARIANTS_SYMBOL,
                speech_conv_bf16_variants(&blob),
                "bf16 conv variants",
            ),
        ] {
            if let Some(compiled) = be.module_global_u32(&module, symbol)? {
                let missing = required & !u64::from(compiled);
                if missing != 0 {
                    return Err(RuntimeError::Rejected(format!(
                        "{}: {} lacks {what} {missing:#x}; rebuild it for this packet",
                        path.display(),
                        object.display()
                    )));
                }
            }
        }
        let caps = missing_speech_caps(
            speech_caps_required(blob.progs.iter().flat_map(|p| &p.insts)),
            be.module_global_u32(&module, "plow_speech_caps_speech")?,
        );
        if caps != 0 {
            return Err(RuntimeError::Rejected(format!(
                "{}: {} lacks speech capabilities {caps:#x}; rebuild it for this packet",
                path.display(),
                object.display()
            )));
        }
        let features = speech_features(
            blob.progs
                .iter()
                .flat_map(|p| &p.insts)
                .map(|d| (d.op, d.fj[2])),
        );
        check_speech_features(&be, &module, &object, features, |have| {
            stale_speech_layout(
                blob.progs
                    .iter()
                    .flat_map(|p| &p.insts)
                    .map(|d| (d.op, d.fj[2], d.i[2])),
                have,
            )
        })?;
        let smem = be
            .module_global_u32(&module, "plow_arena_bytes_speech")?
            .unwrap_or(49152);
        if let Some(limit) = crate::config::RuntimeConfig::debug_max_inst() {
            if be.module_global_set_u32(&module, "plow_debug_max_inst_speech", limit)? {
                tracing::warn!(
                    limit,
                    "packet runtime: instructions at or past the limit are skipped"
                );
            }
        }
        be.set_max_dynamic_smem(function, smem)?;
        let capacity = be.occupancy_blocks_per_sm(function, BLOCK, smem as usize)? * be.sm_count();
        if blob.n_cu == 0 || blob.n_cu > capacity {
            return Err(RuntimeError::Rejected(format!(
                "{}: n_cu {} exceeds cooperative capacity {capacity}",
                path.display(),
                blob.n_cu
            )));
        }
        let conv = load_conv_object(&be, &module, &object, blob)?;

        let mut tensors = Vec::with_capacity(blob.tensors.len());
        for t in &blob.tensors {
            let external = shared
                .as_ref()
                .and_then(|s| s.views.get(&t.name).map(|v| (s, v)));
            let (mem, initialize) = match external {
                Some((s, &(offset, bytes, initialize))) => {
                    (s.memory.subview(offset, bytes)?, initialize)
                }
                None => (be.alloc(0, t.bytes.max(4))?, true),
            };
            if initialize {
                match &t.init {
                    Some(r) => be.upload(&mem, 0, &blob.init[r.clone()])?,
                    None => be.memset_d8(mem.base, 0, t.bytes as usize)?,
                }
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
                (
                    (p.n_counter as usize * CTR_STRIDE as usize * 4).max(4),
                    segments * CTR_STRIDE as usize * 4,
                )
            })
            .collect();
        let slab_bytes: usize = regions
            .iter()
            .map(|&(c, k)| (c + k).next_multiple_of(256))
            .sum();
        let counter_slab = be.alloc(0, slab_bytes.max(4) as u64)?;
        let mut slab_offset = 0u64;
        let mut programs = Vec::with_capacity(blob.progs.len());
        for (p, &(counter_only, cursor_bytes)) in blob.progs.iter().zip(&regions) {
            if p.gq_seg_ofs.len() < 2 || p.l2_domains != 0 {
                return Err(RuntimeError::Rejected(format!(
                    "{}: program needs an unplaced global-queue stream",
                    path.display()
                )));
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
            let conv_segments = if conv.is_some() {
                plow_asset::speech_flag_layout::conv_object_segments(
                    &p.gq_seg_ofs,
                    &p.gq_stream,
                    |i| {
                        let d = &p.insts[i as usize];
                        plow_asset::speech_flag_layout::conv_object_inst(d.op, d.fj[2])
                    },
                )
                .into_iter()
                .map(|inst| {
                    inst.map(|i| {
                        let size = std::mem::size_of::<packet::dev::DevInst64>() as u64;
                        // 0 = the interpreter's grid, as the interpreter reads it.
                        let blocks = match p.insts[i as usize].blocks {
                            0 => blob.n_cu,
                            b => u32::from(b),
                        };
                        (d_inst.base + u64::from(i) * size, blocks)
                    })
                })
                .collect()
            } else {
                vec![None; segments]
            };
            programs.push(Program {
                kernarg,
                segments,
                conv_segments,
                counters,
                counter_bytes: counter_only + cursor_bytes,
                _tables: vec![
                    d_inst,
                    d_stream,
                    d_sofs,
                    d_slen,
                    d_waits,
                    d_succs,
                    d_gq_stream,
                    d_gq_seg,
                ],
            });
        }
        // The init copies and memsets ran on the legacy stream, which the non-blocking packet
        // stream does not wait for.
        be.synchronize()?;
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
            conv,
            names: blob.tensors.iter().map(|t| t.name.clone()).collect(),
            tensors,
            _table: table,
            _counters: counter_slab,
            programs,
            graphs: std::collections::HashMap::new(),
            last_us: 0.0,
            input_stage: None,
            input_failed: false,
            shared,
            _resident: {
                let token = Arc::new(());
                residents()
                    .lock()
                    .push((dir.to_path_buf(), ordinal, Arc::downgrade(&token)));
                token
            },
        })
    }

    pub(crate) fn tick_kernels(&self) -> Result<(KernelFn, KernelFn, KernelFn)> {
        Ok((
            self.be.get_function(&self.module, "tick_codec_codes")?,
            self.be.get_function(&self.module, "tick_pack_pcm16")?,
            self.be.get_function(&self.module, "tick_codec_lengths")?,
        ))
    }

    pub(crate) fn check_sequence(&self, programs: &[usize]) -> Result<()> {
        self.check_inputs()?;
        if programs.is_empty() || programs.iter().any(|&p| p >= self.programs.len()) {
            return Err(RuntimeError::Rejected(
                "invalid chained packet sequence".into(),
            ));
        }
        Ok(())
    }

    pub(crate) fn check_chained_writes(&self, writes: &[PacketWrite<'_>]) -> Result<()> {
        self.check_inputs()?;
        for w in writes {
            let mem = self.mem(w.tensor)?;
            check_transfer(
                w.tensor,
                w.tensor.handle,
                mem.len as usize,
                mem.len as usize,
            )?;
            w.check()?;
            if matches!(w.data, PacketData::Buffer(_)) {
                return Err(RuntimeError::Rejected(
                    "chained controls require host or zero inputs".into(),
                ));
            }
        }
        let size = input_size(writes.iter().map(|w| host_input_size(&self.be, w.data)))?;
        if size > self.input_stage.as_ref().map_or(0, |s| s.len()) {
            return Err(RuntimeError::Rejected(
                "prepare packet writes before chaining".into(),
            ));
        }
        Ok(())
    }

    // Caller keeps this runtime borrowed and drains `stream` before reuse or release.
    pub(crate) unsafe fn enqueue_chained(
        &mut self,
        stream: &CudaStream,
        programs: &[usize],
        writes: &[PacketWrite<'_>],
    ) -> Result<()> {
        self.check_sequence(programs)?;
        self.check_chained_writes(writes)?;
        let mut offset = 0;
        for w in writes {
            let len = host_input_size(&self.be, w.data);
            if len != 0 {
                w.data.read(
                    &mut self.input_stage.as_mut().unwrap().as_mut_slice()[offset..offset + len],
                )?;
                offset = (offset + len + 7) & !7;
            }
        }
        offset = 0;
        for w in writes {
            let len = w.data.len();
            if len == 0 {
                continue;
            }
            let target = self.tensors[w.tensor.handle].base + w.offset as u64;
            if matches!(w.data, PacketData::Zero(_)) {
                self.be.memset_d8_async(target, 0, len, stream)?;
            } else {
                let source = &self.input_stage.as_ref().unwrap().as_slice()[offset..offset + len];
                offset = (offset + len + 7) & !7;
                // SAFETY: the chain owns pinned staging until its final drain.
                unsafe {
                    self.be.memcpy_htod_async(target, source, stream)?;
                }
            }
        }
        Ok(())
    }

    pub(crate) unsafe fn enqueue_chained_programs(
        &self,
        stream: &CudaStream,
        programs: &[usize],
    ) -> Result<()> {
        self.check_sequence(programs)?;
        match self.graphs.get(programs).and_then(Option::as_ref) {
            Some(graph) => self.be.graph_launch(graph, stream),
            None => self.enqueue_on(programs, stream),
        }
    }

    pub(crate) fn checked_device_ptr(&self, tensor: PacketTensor) -> Result<u64> {
        let mem = self.mem(tensor)?;
        check_transfer(tensor, tensor.handle, mem.len as usize, mem.len as usize)?;
        Ok(mem.base)
    }

    pub(crate) fn quarantine_chain(&mut self) {
        self.input_failed = true;
        if let Some(shared) = self.shared.take() {
            std::mem::forget(shared);
        }
        if let Some(stage) = self.input_stage.take() {
            std::mem::forget(stage);
        }
        for tensor in &mut self.tensors {
            tensor.quarantine();
        }
        self._table.quarantine();
        self._counters.quarantine();
        for p in &mut self.programs {
            for table in &mut p._tables {
                table.quarantine();
            }
        }
        std::mem::forget(self.be.clone());
        std::mem::forget(self._resident.clone());
    }

    /// Skip every instruction at or past `limit` (per-instruction cost by caps).
    pub fn set_debug_max_inst(&self, limit: u32) -> Result<bool> {
        self.be
            .module_global_set_u32(&self.module, "plow_debug_max_inst_speech", limit)
    }

    pub fn backend(&self) -> &Arc<CudaBackend> {
        &self.be
    }

    /// Device address of a tensor, for device-to-device handoff to another engine on the same GPU.
    pub fn device_ptr(&self, tensor: PacketTensor) -> Option<u64> {
        self.tensors.get(tensor.handle).map(|m| m.base)
    }

    fn enqueue(&self, programs: &[usize]) -> Result<()> {
        self.enqueue_on(programs, &self.stream)
    }

    fn enqueue_on(&self, programs: &[usize], stream: &CudaStream) -> Result<()> {
        let consecutive = programs.windows(2).all(|w| w[1] == w[0] + 1);
        if let (true, Some(&first), Some(&last)) = (consecutive, programs.first(), programs.last())
        {
            let (first, last) = (&self.programs[first], &self.programs[last]);
            let bytes = (last.counters + last.counter_bytes as u64 - first.counters) as usize;
            self.be.memset_d8_async(first.counters, 0, bytes, stream)?;
        }
        for &program in programs {
            let p = &self.programs[program];
            if !consecutive {
                self.be
                    .memset_d8_async(p.counters, 0, p.counter_bytes, stream)?;
            }
            for seg in 0..p.segments {
                if let (Some(c), Some((mut inst, blocks))) = (&self.conv, p.conv_segments[seg]) {
                    let mut tensors = p.kernarg.tensors;
                    let mut params = [
                        &mut inst as *mut u64 as *mut std::ffi::c_void,
                        &mut tensors as *mut u64 as *mut std::ffi::c_void,
                    ];
                    self.be.launch_kernel(
                        c.function,
                        blocks,
                        BLOCK,
                        c.smem,
                        &mut params,
                        Some(stream),
                    )?;
                    continue;
                }
                let mut arg = p.kernarg;
                arg.gq_seg_ofs += (seg * 4) as u64;
                arg.gq_cursor += (seg * CTR_STRIDE as usize * 4) as u64;
                let mut params = [&mut arg as *mut DevProgram as *mut std::ffi::c_void];
                self.be.launch_cooperative(
                    self.function,
                    self.grid,
                    BLOCK,
                    self.smem,
                    &mut params,
                    Some(stream),
                )?;
            }
        }
        Ok(())
    }

    /// Every transfer goes through the packet stream: it is NON_BLOCKING, so a legacy-stream copy
    /// is not ordered before the next program (a DtoD copy returns before it lands).
    fn upload(&self, mem: &DeviceMem, offset: usize, bytes: &[u8]) -> Result<()> {
        in_range(mem, offset, bytes.len())?;
        // SAFETY: `bytes` outlives the copy (synchronized below); the range is inside `mem`.
        unsafe {
            self.be
                .memcpy_htod_async(mem.base + offset as u64, bytes, &self.stream)?
        };
        self.be.stream_synchronize(&self.stream)
    }

    fn download(&self, mem: &DeviceMem, offset: usize, bytes: &mut [u8]) -> Result<()> {
        in_range(mem, offset, bytes.len())?;
        // SAFETY: as `upload`.
        unsafe {
            self.be
                .memcpy_dtoh_async(bytes, mem.base + offset as u64, &self.stream)?
        };
        self.be.stream_synchronize(&self.stream)
    }

    fn mem(&self, t: PacketTensor) -> Result<&DeviceMem> {
        self.check_inputs()?;
        self.tensors.get(t.handle).ok_or_else(|| {
            RuntimeError::Device(format!("packet tensor handle {} is missing", t.handle))
        })
    }

    fn check_inputs(&self) -> Result<()> {
        if self.input_failed {
            return Err(RuntimeError::Device(
                "packet input copies did not retire".into(),
            ));
        }
        Ok(())
    }

    fn reserve_inputs(&mut self, size: usize) -> Result<()> {
        if size > 0 && self.input_stage.as_ref().is_none_or(|s| s.len() < size) {
            self.input_stage = Some(self.be.host_alloc_pinned(size)?);
        }
        Ok(())
    }

    fn abandon_transfers(
        &mut self,
        tensors: impl IntoIterator<Item = PacketTensor>,
        error: RuntimeError,
    ) -> RuntimeError {
        self.input_failed = true;
        std::mem::forget(self.be.clone());
        std::mem::forget(self._resident.clone());
        for tensor in tensors {
            self.tensors[tensor.handle].quarantine();
        }
        #[cfg(target_os = "linux")]
        {
            self.be.quarantine_staging(&error)
        }
        #[cfg(not(target_os = "linux"))]
        {
            error
        }
    }
}

fn input_size(mut sizes: impl Iterator<Item = usize>) -> Result<usize> {
    sizes.try_fold(0usize, |total, bytes| {
        total
            .checked_add(bytes)
            .and_then(|n| n.checked_add(7))
            .map(|n| n & !7)
            .ok_or_else(|| RuntimeError::Device("packet input size overflow".into()))
    })
}

fn device_input_source(be: &CudaBackend, data: PacketData<'_>) -> Option<u64> {
    match data {
        PacketData::Buffer(source) => source
            .cuda_source()
            .filter(|(b, _)| be.shares_context(b))
            .map(|(_, address)| address),
        _ => None,
    }
}

fn host_input_size(be: &CudaBackend, data: PacketData<'_>) -> usize {
    match data {
        PacketData::Zero(_) => 0,
        _ if device_input_source(be, data).is_some() => 0,
        _ => data.len(),
    }
}

fn submit_writes(
    count: usize,
    mut enqueue: impl FnMut(usize) -> Result<()>,
    drain: impl FnOnce() -> Result<()>,
    quarantine: impl FnOnce(),
) -> Result<()> {
    let status = (0..count).try_for_each(&mut enqueue);
    if let Err(error) = drain() {
        quarantine();
        return Err(error);
    }
    status
}

#[cfg(test)]
mod write_tests {
    use super::*;
    use std::cell::{Cell, RefCell};

    #[test]
    #[ignore = "GPU correctness; run through gpuq"]
    fn packet_pool_reuses_released_slots_and_preserves_held_views() {
        use crate::exec::packet_runtime::PacketBufferPool;
        let backend = Arc::new(CudaBackend::new(0).unwrap());
        let stream = backend.stream_create().unwrap();
        let source = Arc::new(backend.alloc(0, 64).unwrap());
        let bytes: Vec<u8> = (0..64).map(|i| (i * 37) as u8).collect();
        backend.upload(&source, 0, &bytes).unwrap();
        let mut pool = PacketBufferPool::cuda(backend.clone(), 48, 2).unwrap();
        let first = pool.acquire(37, source.clone()).unwrap();
        let first_address = first.address();
        let second = pool.acquire(41, source.clone()).unwrap();
        assert_eq!(second.address() - first_address, 48);
        assert!(pool.acquire(1, source.clone()).is_none());
        backend
            .memcpy_dtod_async(first.address(), source.base + 3, 37, &stream)
            .unwrap();
        backend
            .memcpy_dtod_async(second.address(), source.base + 7, 41, &stream)
            .unwrap();
        backend.stream_synchronize(&stream).unwrap();
        // SAFETY: both pending copies were drained above.
        let (first, second) = unsafe { (first.retire(), second.retire()) };
        let view = first.slice(5..29).unwrap();
        drop(first);
        assert!(pool.acquire(1, source.clone()).is_none());
        let mut actual = [0; 24];
        view.read(&mut actual).unwrap();
        assert_eq!(actual, bytes[8..32]);
        drop(view);
        let reused = pool.acquire(48, source.clone()).unwrap();
        assert_eq!(reused.address(), first_address);
        backend.upload(&source, 0, &[0xdd; 64]).unwrap();
        backend
            .memcpy_dtod_async(reused.address(), source.base, 48, &stream)
            .unwrap();
        drop(source);
        backend.stream_synchronize(&stream).unwrap();
        // SAFETY: the reused slot and its retained producer have completed.
        let reused = unsafe { reused.retire() };
        drop(pool);
        let mut old = [0; 41];
        second.read(&mut old).unwrap();
        assert_eq!(old, bytes[7..48]);
        let mut new = [0; 48];
        reused.read(&mut new).unwrap();
        assert_eq!(new, [0xdd; 48]);
    }

    #[test]
    #[ignore = "GPU correctness; SPEECH_BATCH_TEST_ASSETS; run through gpuq"]
    fn packet_snapshot_survives_producer_reuse_and_runtime_drop() {
        let assets = std::path::PathBuf::from(std::env::var("SPEECH_BATCH_TEST_ASSETS").unwrap());
        let mut producer = CudaPacketRuntime::load(&assets.join("codec.pkt"), 0).unwrap();
        let resident = Arc::downgrade(&producer._resident);
        let input = producer.create_tensor(64).unwrap();
        let bytes: Vec<u8> = (0..64).map(|i| (i * 37) as u8).collect();
        producer.write_tensor(input, &bytes).unwrap();
        let snapshot = producer.snapshot_tensor(input, 7, 41).unwrap();
        let view = snapshot.slice(3..35).unwrap();
        producer.write_tensor(input, &[0x55; 64]).unwrap();
        drop(producer);
        drop(snapshot);
        assert!(resident.upgrade().is_some());
        let mut output = [0u8; 32];
        view.read(&mut output).unwrap();
        assert_eq!(&output, &bytes[10..42]);
        std::thread::spawn(move || {
            let mut consumer = CudaPacketRuntime::load(&assets.join("codec.pkt"), 0).unwrap();
            let target = consumer.create_tensor(64).unwrap();
            consumer.write_tensor(target, &[0xa5; 64]).unwrap();
            assert!(consumer.input_stage.is_none());
            consumer.write_tensor_from(target, 5, &view).unwrap();
            assert!(
                consumer.input_stage.is_none(),
                "same-context copy fell back to host staging"
            );
            let mut actual = [0u8; 64];
            consumer.read_tensor(target, &mut actual).unwrap();
            let mut expected = [0xa5; 64];
            expected[5..37].copy_from_slice(&bytes[10..42]);
            assert_eq!(actual, expected);
            let first = view.slice(0..11).unwrap();
            let second = view.slice(7..24).unwrap();
            consumer
                .write_tensors(&[
                    PacketWrite {
                        tensor: target,
                        offset: 2,
                        data: PacketData::Zero(46),
                    },
                    PacketWrite {
                        tensor: target,
                        offset: 3,
                        data: PacketData::Buffer(&first),
                    },
                    PacketWrite {
                        tensor: target,
                        offset: 9,
                        data: PacketData::Buffer(&second),
                    },
                    PacketWrite {
                        tensor: target,
                        offset: 64,
                        data: PacketData::Zero(0),
                    },
                ])
                .unwrap();
            assert!(
                consumer.input_stage.is_none(),
                "device-only batch staged on host"
            );
            expected[2..48].fill(0);
            expected[3..14].copy_from_slice(&bytes[10..21]);
            expected[9..26].copy_from_slice(&bytes[17..34]);
            consumer.read_tensor(target, &mut actual).unwrap();
            assert_eq!(actual, expected);
            for data in [PacketData::Zero(usize::MAX), PacketData::Buffer(&view)] {
                assert!(consumer
                    .write_tensors(&[
                        PacketWrite {
                            tensor: target,
                            offset: 0,
                            data: PacketData::Zero(64)
                        },
                        PacketWrite {
                            tensor: target,
                            offset: 63,
                            data
                        },
                    ])
                    .is_err());
                consumer.read_tensor(target, &mut actual).unwrap();
                assert_eq!(actual, expected);
            }
            for offset in [63, usize::MAX] {
                assert!(consumer.write_tensor_from(target, offset, &view).is_err());
                consumer.read_tensor(target, &mut actual).unwrap();
                assert_eq!(actual, expected);
            }
            assert!(consumer.snapshot_tensor(target, 60, 5).is_err());
            assert!(consumer
                .write_tensor_from(
                    PacketTensor {
                        bytes: 63,
                        ..target
                    },
                    0,
                    &view
                )
                .is_err());
            let host: PacketBuffer = vec![9, 8, 7].into();
            consumer
                .write_tensors(&[
                    PacketWrite {
                        tensor: target,
                        offset: 1,
                        data: PacketData::Buffer(&host),
                    },
                    PacketWrite {
                        tensor: target,
                        offset: 6,
                        data: PacketData::Buffer(&first),
                    },
                    PacketWrite {
                        tensor: target,
                        offset: 12,
                        data: PacketData::Zero(2),
                    },
                    PacketWrite {
                        tensor: target,
                        offset: 15,
                        data: PacketData::Host(&[5, 4, 3, 2, 1]),
                    },
                ])
                .unwrap();
            assert!(consumer.input_stage.is_some());
            assert_eq!(consumer.input_stage.as_ref().unwrap().len(), 16);
            expected[1..4].copy_from_slice(&[9, 8, 7]);
            expected[6..17].copy_from_slice(&bytes[10..21]);
            expected[12..14].fill(0);
            expected[15..20].copy_from_slice(&[5, 4, 3, 2, 1]);
            consumer.read_tensor(target, &mut actual).unwrap();
            assert_eq!(actual, expected);
            drop(first);
            drop(second);
            drop(view);
            assert!(resident.upgrade().is_none());
        })
        .join()
        .unwrap();
    }

    #[test]
    fn input_writes_drain_after_success_or_any_enqueue_failure() {
        for failure in [None, Some(0), Some(1), Some(2)] {
            for drain_fails in [false, true] {
                let seen = RefCell::new(Vec::new());
                let quarantined = Cell::new(false);
                let result = submit_writes(
                    3,
                    |i| {
                        seen.borrow_mut().push(i);
                        if failure == Some(i) {
                            Err(RuntimeError::Device("enqueue".into()))
                        } else {
                            Ok(())
                        }
                    },
                    || {
                        seen.borrow_mut().push(3);
                        if drain_fails {
                            Err(RuntimeError::Device("drain".into()))
                        } else {
                            Ok(())
                        }
                    },
                    || quarantined.set(true),
                );
                let mut expected: Vec<_> = (0..failure.map_or(3, |i| i + 1)).collect();
                expected.push(3);
                assert_eq!(*seen.borrow(), expected);
                assert_eq!(quarantined.get(), drain_fails);
                assert_eq!(result.is_err(), failure.is_some() || drain_fails);
            }
        }
    }

    #[test]
    #[ignore = "GPU correctness; SPEECH_BATCH_TEST_ASSETS; run through gpuq"]
    fn packet_bulk_writes_preserve_ranges_order_and_reuse() {
        let assets = std::path::PathBuf::from(std::env::var("SPEECH_BATCH_TEST_ASSETS").unwrap());
        let mut rt = CudaPacketRuntime::load(&assets.join("codec.pkt"), 0).unwrap();
        let t = rt.create_tensor(64).unwrap();
        for size in [7, 23, 3, 47, 7] {
            let initial = [0xa5; 64];
            rt.write_tensor(t, &initial).unwrap();
            let first = vec![0x37; size];
            let writes = [
                PacketWrite {
                    tensor: t,
                    offset: 5,
                    data: PacketData::Host(&first),
                },
                PacketWrite {
                    tensor: t,
                    offset: 9,
                    data: PacketData::Host(&[1, 2, 3, 4]),
                },
                PacketWrite {
                    tensor: t,
                    offset: 64,
                    data: PacketData::Host(&[]),
                },
            ];
            rt.write_tensors(&writes).unwrap();
            let mut expected = initial;
            expected[5..5 + size].copy_from_slice(&first);
            expected[9..13].copy_from_slice(&[1, 2, 3, 4]);
            let mut actual = [0; 64];
            rt.read_tensor(t, &mut actual).unwrap();
            assert_eq!(actual, expected);
            assert!(rt
                .write_tensors(&[
                    PacketWrite {
                        tensor: t,
                        offset: 0,
                        data: PacketData::Host(&[0xff]),
                    },
                    PacketWrite {
                        tensor: t,
                        offset: usize::MAX,
                        data: PacketData::Host(&[1]),
                    },
                ])
                .is_err());
            rt.read_tensor(t, &mut actual).unwrap();
            assert_eq!(actual, expected, "validation must precede all transfers");
            assert!(rt
                .write_tensors(&[PacketWrite {
                    tensor: PacketTensor { bytes: 63, ..t },
                    offset: 0,
                    data: PacketData::Host(&[1]),
                }])
                .is_err());
        }
        rt.prepare_writes(&[t, t, t]).unwrap();
        let pointer = rt.input_stage.as_ref().unwrap().as_ptr();
        rt.write_tensors(&[PacketWrite {
            tensor: t,
            offset: 0,
            data: PacketData::Host(&[0xa5; 64]),
        }])
        .unwrap();
        assert_eq!(
            pointer,
            rt.input_stage.as_ref().unwrap().as_ptr(),
            "prepared input staging grew"
        );
        assert!(input_size([usize::MAX].into_iter()).is_err());
    }
}

impl PacketRuntime for CudaPacketRuntime {
    fn cuda_runtime(&mut self) -> Option<&mut CudaPacketRuntime> {
        Some(self)
    }

    fn tensor(&self, name: &str) -> Option<PacketTensor> {
        let handle = self.names.iter().position(|n| n == name)?;
        Some(PacketTensor {
            handle,
            bytes: self.tensors[handle].len as usize,
        })
    }

    fn write_tensors(&mut self, writes: &[PacketWrite<'_>]) -> Result<()> {
        self.check_inputs()?;
        for w in writes {
            let mem = self.mem(w.tensor)?;
            check_transfer(
                w.tensor,
                w.tensor.handle,
                mem.len as usize,
                mem.len as usize,
            )?;
            w.check()?;
        }
        if writes.iter().all(|w| w.data.is_empty()) {
            return Ok(());
        }
        let size = input_size(writes.iter().map(|w| host_input_size(&self.be, w.data)))?;
        self.reserve_inputs(size)?;
        let mut offset = 0;
        for w in writes {
            let len = host_input_size(&self.be, w.data);
            if len > 0 {
                let stage = self.input_stage.as_mut().unwrap();
                w.data
                    .read(&mut stage.as_mut_slice()[offset..offset + len])?;
                offset = (offset + len + 7) & !7;
            }
        }
        let mut offset = 0;
        let mut failed = false;
        let result = submit_writes(
            writes.len(),
            |i| {
                let w = &writes[i];
                let len = w.data.len();
                if len == 0 {
                    return Ok(());
                }
                let target = self.tensors[w.tensor.handle].base + w.offset as u64;
                if let PacketData::Zero(_) = w.data {
                    return self.be.memset_d8_async(target, 0, len, &self.stream);
                }
                if let Some(source) = device_input_source(&self.be, w.data) {
                    return self
                        .be
                        .memcpy_dtod_async(target, source, len as u64, &self.stream);
                }
                let stage = self.input_stage.as_ref().unwrap();
                let source = &stage.as_slice()[offset..offset + len];
                offset = (offset + len + 7) & !7;
                // SAFETY: validated ranges, owned pinned source, and the batch drains below.
                unsafe { self.be.memcpy_htod_async(target, source, &self.stream) }
            },
            || self.be.stream_synchronize(&self.stream),
            || failed = true,
        );
        if failed {
            // Unproven completion must retain buffers and the backend's small-copy kernel.
            if let Some(stage) = self.input_stage.take() {
                std::mem::forget(stage);
            }
            for w in writes {
                if let PacketData::Buffer(source) = w.data {
                    std::mem::forget(source.clone());
                }
            }
            return Err(
                self.abandon_transfers(writes.iter().map(|w| w.tensor), result.unwrap_err())
            );
        }
        result
    }

    fn prepare_writes(&mut self, tensors: &[PacketTensor]) -> Result<()> {
        self.check_inputs()?;
        for &t in tensors {
            let mem = self.mem(t)?;
            check_transfer(t, t.handle, mem.len as usize, mem.len as usize)?;
        }
        self.reserve_inputs(input_size(tensors.iter().map(|t| t.bytes))?)
    }

    fn snapshot_tensor(
        &mut self,
        tensor: PacketTensor,
        offset: usize,
        bytes: usize,
    ) -> Result<PacketBuffer> {
        let source = self.mem(tensor)?;
        check_transfer(
            tensor,
            tensor.handle,
            source.len as usize,
            source.len as usize,
        )?;
        check_region(tensor, offset, bytes)?;
        if bytes == 0 {
            return Ok(Vec::new().into());
        }
        let address = source.base + offset as u64;
        let mut snapshot = self.be.alloc(0, bytes as u64)?;
        let mut failed = false;
        let result = submit_writes(
            1,
            |_| {
                self.be
                    .memcpy_dtod_async(snapshot.base, address, bytes as u64, &self.stream)
            },
            || self.be.stream_synchronize(&self.stream),
            || failed = true,
        );
        if failed {
            snapshot.quarantine();
            return Err(self.abandon_transfers([tensor], result.unwrap_err()));
        }
        result?;
        Ok(PacketBuffer::completed_cuda(
            snapshot,
            self.be.clone(),
            bytes,
            self._resident.clone(),
        ))
    }

    fn write_tensor(&mut self, tensor: PacketTensor, bytes: &[u8]) -> Result<()> {
        let mem = self.mem(tensor)?;
        check_transfer(tensor, tensor.handle, mem.len as usize, bytes.len())?;
        self.upload(mem, 0, bytes)
    }

    fn read_tensor(&self, tensor: PacketTensor, bytes: &mut [u8]) -> Result<()> {
        let mem = self.mem(tensor)?;
        check_transfer(tensor, tensor.handle, mem.len as usize, bytes.len())?;
        self.download(mem, 0, bytes)
    }

    fn write_tensor_at(&mut self, tensor: PacketTensor, offset: usize, bytes: &[u8]) -> Result<()> {
        let mem = self.mem(tensor)?;
        check_transfer(tensor, tensor.handle, mem.len as usize, mem.len as usize)?;
        self.upload(mem, offset, bytes)
    }

    fn read_tensor_at(&self, tensor: PacketTensor, offset: usize, bytes: &mut [u8]) -> Result<()> {
        let mem = self.mem(tensor)?;
        check_transfer(tensor, tensor.handle, mem.len as usize, mem.len as usize)?;
        self.download(mem, offset, bytes)
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
        check_copy(
            source,
            s.len as usize,
            source_offset,
            target,
            t.len as usize,
            target_offset,
            bytes,
        )?;
        self.be.memcpy_dtod_async(
            t.base + target_offset as u64,
            s.base + source_offset as u64,
            bytes as u64,
            &self.stream,
        )
    }

    fn create_tensor(&mut self, bytes: usize) -> Result<PacketTensor> {
        let mem = self.be.alloc(0, (bytes as u64).max(4))?;
        self.be
            .memset_d8_async(mem.base, 0, mem.len as usize, &self.stream)?;
        self.be.stream_synchronize(&self.stream)?;
        self.tensors.push(mem);
        // Unnamed: `tensor(name)` never resolves it; only its handle does.
        self.names.push(String::new());
        Ok(PacketTensor {
            handle: self.tensors.len() - 1,
            bytes,
        })
    }

    fn run(&mut self, program: usize) -> Result<()> {
        self.run_sequence(&[program])
    }

    /// Every program is enqueued on the one stream (stream order is the dependency), as one
    /// CUDA graph after the first run of a sequence; one sync.
    fn run_sequence(&mut self, programs: &[usize]) -> Result<()> {
        self.check_inputs()?;
        if let Some(&bad) = programs.iter().find(|&&p| p >= self.programs.len()) {
            return Err(RuntimeError::Rejected(format!(
                "packet program {bad} is missing"
            )));
        }
        if !self.graphs.contains_key(programs) {
            let graph = match self
                .be
                .graph_capture(&self.stream, || self.enqueue(programs))
            {
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

#[cfg(test)]
#[test]
#[ignore = "CUDA; single real layer native GEMM screen, run only through gpuq"]
fn native_gemm_layer_matches_packet() {
    use packet::dev::DevInst64;
    enum Step {
        Packet {
            argument: DevProgram,
            counters: usize,
            _tables: Vec<DeviceMem>,
        },
        Gemm {
            instruction: DeviceMem,
            function: KernelFn,
        },
    }
    let dir = std::path::PathBuf::from(std::env::var("DEPTH_GEMM_TEST_ASSETS").unwrap());
    let image = std::fs::read(std::env::var("DEPTH_GEMM_NODE_CUBIN").unwrap()).unwrap();
    let batches = std::env::var("DEPTH_GEMM_BATCHES").unwrap_or("32,64,128,200,224".into());
    let selections = std::env::var("DEPTH_GEMM_SELECTIONS").unwrap_or("25:0,31:0,25:3".into());
    let be = Arc::new(CudaBackend::new(0).unwrap());
    let module = be.module_load(&image).unwrap();
    assert_eq!(
        be.module_global_u32(&module, "plow_gemm_f16_node_abi")
            .unwrap(),
        Some(1)
    );
    let functions = ["gemm_f16_sm90_f32_node", "gemm_f16_sm90_f16_node"]
        .map(|s| be.get_function(&module, s).unwrap());
    for f in functions {
        be.set_max_dynamic_smem(f, 103424).unwrap();
    }
    let compare = |a: &[u8], b: &[u8]| {
        assert_eq!(a.len(), b.len());
        let (mut delta, mut norm) = (0.0f64, 0.0f64);
        for (a, b) in a.chunks_exact(4).zip(b.chunks_exact(4)) {
            let a = f32::from_le_bytes(a.try_into().unwrap()) as f64;
            let b = f32::from_le_bytes(b.try_into().unwrap()) as f64;
            assert!(a.is_finite() && b.is_finite());
            delta += (a - b) * (a - b);
            norm += a * a;
        }
        let rel = (delta / norm.max(1e-30)).sqrt();
        assert!(rel <= 2e-4, "native layer relative error {rel}");
        rel
    };
    for batch in batches.split(',').map(|s| s.parse::<u32>().unwrap()) {
        for selection in selections.split(',') {
            let (level, layer) = selection.split_once(':').unwrap();
            let (level, layer) = (level.parse::<u32>().unwrap(), layer.parse::<u32>().unwrap());
            let path = dir.join(format!("b{batch}.l{level}.layer{layer}.pkt"));
            let raw = std::fs::read(&path).unwrap();
            let blob = DevBlob::parse(&raw).unwrap();
            assert_eq!(blob.progs.len(), 1);
            let rt = CudaPacketRuntime::load_on(be.clone(), &path).unwrap();
            let g = &blob.progs[0];
            let mut steps = Vec::new();
            let mut span = Vec::new();
            let mut routed = 0;
            let emit_span = |span: &mut Vec<DevInst64>, steps: &mut Vec<Step>| {
                if span.is_empty() {
                    return;
                }
                let program = super::packed_terminal::chain(std::mem::take(span), rt.grid, g.t);
                let (argument, _, counters, tables) =
                    super::mixed_step::upload_program(&be, &program, rt._table.base, 0, 0).unwrap();
                steps.push(Step::Packet {
                    argument,
                    counters,
                    _tables: tables,
                });
            };
            for instruction in &g.insts {
                let flags = instruction.i[7];
                let measured_shape = (instruction.i[1] == 3072
                    && instruction.i[2] == 1024
                    && instruction.i[0] >= 832)
                    || (instruction.i[1] == 1024
                        && instruction.i[2] == 3072
                        && instruction.i[0] >= 1664);
                let supported = instruction.op == DevOp::GemvF32Out as u16
                    && measured_shape
                    && instruction.i[1] % 128 == 0
                    && instruction.i[2] % 64 == 0
                    && flags & (3 | 1024 | 8192) == (3 | 1024 | 8192)
                    && flags & !(3 | 4 | 8 | 16 | 512 | 1024 | 8192 | 16384) == 0;
                if !supported {
                    span.push(*instruction);
                    continue;
                }
                let (m, n, k) = (
                    u64::from(instruction.i[0]),
                    u64::from(instruction.i[1]),
                    u64::from(instruction.i[2]),
                );
                let half = flags & 16384 != 0;
                let group = if flags & 512 != 0 {
                    u64::from(instruction.i[3])
                } else {
                    0
                };
                assert!(flags & 512 == 0 || group > 0 && flags & 4 != 0);
                let mut ranges = Vec::new();
                for (index, bytes) in [
                    (0, m * n * if half { 2 } else { 4 }),
                    (1, m * k * 2),
                    (2, n * k * 2),
                    (
                        3,
                        if flags & 4 != 0 {
                            (if group > 0 { m.div_ceil(group) } else { 1 }) * n * 4
                        } else {
                            0
                        },
                    ),
                    (4, if flags & 8 != 0 { m * n * 4 } else { 0 }),
                ] {
                    if bytes == 0 {
                        continue;
                    }
                    let memory = &rt.tensors[usize::from(instruction.t[index])];
                    assert!(memory.len >= bytes && memory.base % 16 == 0);
                    ranges.push((index, memory.base, memory.base + bytes));
                }
                let (_, c, ce) = ranges[0];
                for &(index, p, pe) in &ranges[1..] {
                    assert!(ce <= p || pe <= c || index == 4 && !half && c == p);
                }
                emit_span(&mut span, &mut steps);
                let memory = be.alloc(0, 64).unwrap();
                be.upload(&memory, 0, pod_bytes(std::slice::from_ref(instruction)))
                    .unwrap();
                steps.push(Step::Gemm {
                    instruction: memory,
                    function: functions[usize::from(half)],
                });
                routed += 1;
            }
            emit_span(&mut span, &mut steps);
            let x = rt.tensor("act.depth.x").unwrap();
            let cache = rt.tensor(&format!("act.depth.cache.{layer}")).unwrap();
            let cross = rt.tensor(&format!("in.depth.cross.{layer}")).unwrap();
            let input = |bytes: usize, salt: usize, scale: f32| -> Vec<u8> {
                (0..bytes / 4)
                    .flat_map(|i| {
                        (((i * 37 + salt * 19) % 509) as f32 / 254. - 1.)
                            .mul_add(scale, 0.001)
                            .to_le_bytes()
                    })
                    .collect()
            };
            let inputs = [
                (x, input(x.bytes, 1, 0.3)),
                (cache, input(cache.bytes, 2, 0.1)),
                (cross, input(cross.bytes, 3, 0.03)),
            ];
            let reset = || {
                for (tensor, bytes) in &inputs {
                    rt.upload(&rt.tensors[tensor.handle], 0, bytes).unwrap();
                }
            };
            let enqueue = |native: bool| {
                if !native {
                    rt.enqueue_on(&[0], &rt.stream).unwrap();
                    return;
                }
                for step in &steps {
                    match step {
                        Step::Packet {
                            argument, counters, ..
                        } => {
                            be.memset_d8_async(argument.counters, 0, *counters, &rt.stream)
                                .unwrap();
                            let mut argument = *argument;
                            let mut params =
                                [&mut argument as *mut DevProgram as *mut std::ffi::c_void];
                            be.launch_cooperative(
                                rt.function,
                                rt.grid,
                                BLOCK,
                                rt.smem,
                                &mut params,
                                Some(&rt.stream),
                            )
                            .unwrap();
                        }
                        Step::Gemm {
                            instruction,
                            function,
                        } => {
                            let mut inst = instruction.base;
                            let mut table = rt._table.base;
                            let mut count = u32::try_from(rt.tensors.len()).unwrap();
                            let mut params = [
                                &mut inst as *mut u64 as *mut std::ffi::c_void,
                                &mut table as *mut u64 as *mut std::ffi::c_void,
                                &mut count as *mut u32 as *mut std::ffi::c_void,
                            ];
                            be.launch_kernel(
                                *function,
                                be.sm_count() * 2,
                                256,
                                103424,
                                &mut params,
                                Some(&rt.stream),
                            )
                            .unwrap();
                        }
                    }
                }
            };
            let read = |tensor: PacketTensor| {
                let mut out = vec![0; tensor.bytes];
                rt.download(&rt.tensors[tensor.handle], 0, &mut out)
                    .unwrap();
                out
            };
            reset();
            enqueue(false);
            be.stream_synchronize(&rt.stream).unwrap();
            let baseline = [read(x), read(cache)];
            reset();
            enqueue(true);
            be.stream_synchronize(&rt.stream).unwrap();
            let treatment = [read(x), read(cache)];
            let output_rows = if level == 25 && layer != 3 {
                batch * 26
            } else {
                batch
            };
            let end = output_rows as usize * 1024 * 4;
            let rel = compare(&baseline[0][..end], &treatment[0][..end]);
            assert_eq!(&baseline[0][end..], &inputs[0].1[end..]);
            assert_eq!(&treatment[0][end..], &inputs[0].1[end..]);
            let mut cache_rel = 0.0f64;
            let (lo, hi) = if level == 25 {
                (0, 26)
            } else {
                (level as usize, level as usize + 1)
            };
            for row in 0..batch as usize {
                let start = row * 32 * 3072 * 4;
                let (a, b, end) = (
                    start + lo * 3072 * 4,
                    start + hi * 3072 * 4,
                    start + 32 * 3072 * 4,
                );
                cache_rel = cache_rel.max(compare(&baseline[1][a..b], &treatment[1][a..b]));
                for result in &baseline[1..2] {
                    assert_eq!(&result[start..a], &inputs[1].1[start..a]);
                    assert_eq!(&result[b..end], &inputs[1].1[b..end]);
                }
                assert_eq!(&treatment[1][start..a], &inputs[1].1[start..a]);
                assert_eq!(&treatment[1][b..end], &inputs[1].1[b..end]);
            }
            let mut times = [Vec::new(), Vec::new()];
            let timing = std::env::var_os("DEPTH_GEMM_TIMING").is_some();
            for round in 0..if timing { 24 } else { 0 } {
                let native = round % 2 == 1;
                reset();
                be.event_record(&rt.events.0, &rt.stream).unwrap();
                enqueue(native);
                be.event_record(&rt.events.1, &rt.stream).unwrap();
                be.event_synchronize(&rt.events.1).unwrap();
                if round >= 4 {
                    times[usize::from(native)]
                        .push(be.event_elapsed_ms(&rt.events.0, &rt.events.1).unwrap() * 1000.);
                }
            }
            for t in &mut times {
                t.sort_by(f32::total_cmp);
            }
            println!("NATIVE_LAYER_ORACLE batch={batch} level={level} layer={layer} routed={routed} rel={rel} cache_rel={cache_rel} tails=exact weights=published");
            if timing {
                println!("NATIVE_LAYER_TIME batch={batch} level={level} layer={layer} routed={routed} baseline_us={} native_us={} policy=after_same_H2D_input_reset samples=10_each",times[0][times[0].len()/2],times[1][times[1].len()/2]);
            }
        }
    }
    be.synchronize().unwrap();
    be.module_unload(&module).unwrap();
}

/// Per-segment device time of one program (SPEECH_SEG_PKT, SPEECH_SEG_PROGRAM): every segment
/// launched alone between two events, median of 10 rounds. `conv` rows ran on the conv object.
#[cfg(test)]
#[test]
#[ignore = "CUDA; per-segment timing, run only through gpuq"]
fn speech_segment_times() {
    let path = std::path::PathBuf::from(std::env::var("SPEECH_SEG_PKT").unwrap());
    let program: usize = std::env::var("SPEECH_SEG_PROGRAM")
        .unwrap()
        .parse()
        .unwrap();
    let be = Arc::new(CudaBackend::new(0).unwrap());
    let rt = CudaPacketRuntime::load_on(be.clone(), &path).unwrap();
    let p = &rt.programs[program];
    let events: Vec<_> = (0..=p.segments)
        .map(|_| be.event_create(true).unwrap())
        .collect();
    let mut times = vec![Vec::new(); p.segments];
    for round in 0..14 {
        be.memset_d8_async(p.counters, 0, p.counter_bytes, &rt.stream)
            .unwrap();
        be.event_record(&events[0], &rt.stream).unwrap();
        for seg in 0..p.segments {
            if let (Some(c), Some((mut inst, blocks))) = (&rt.conv, p.conv_segments[seg]) {
                let mut tensors = p.kernarg.tensors;
                let mut params = [
                    &mut inst as *mut u64 as *mut std::ffi::c_void,
                    &mut tensors as *mut u64 as *mut std::ffi::c_void,
                ];
                be.launch_kernel(
                    c.function,
                    blocks,
                    BLOCK,
                    c.smem,
                    &mut params,
                    Some(&rt.stream),
                )
                .unwrap();
            } else {
                let mut arg = p.kernarg;
                arg.gq_seg_ofs += (seg * 4) as u64;
                arg.gq_cursor += (seg * CTR_STRIDE as usize * 4) as u64;
                let mut params = [&mut arg as *mut DevProgram as *mut std::ffi::c_void];
                be.launch_cooperative(
                    rt.function,
                    rt.grid,
                    BLOCK,
                    rt.smem,
                    &mut params,
                    Some(&rt.stream),
                )
                .unwrap();
            }
            be.event_record(&events[seg + 1], &rt.stream).unwrap();
        }
        be.event_synchronize(&events[p.segments]).unwrap();
        if round >= 4 {
            for (seg, t) in times.iter_mut().enumerate() {
                t.push(be.event_elapsed_ms(&events[seg], &events[seg + 1]).unwrap() * 1000.);
            }
        }
    }
    let mut total = 0.0;
    for (seg, t) in times.iter_mut().enumerate() {
        t.sort_by(f32::total_cmp);
        total += t[5];
        let kind = match p.conv_segments[seg] {
            Some((_, blocks)) => format!("conv blocks={blocks}"),
            None => "interp".into(),
        };
        println!(
            "SPEECH_SEG program={program} seg={seg} {kind} us={:.1}",
            t[5]
        );
    }
    println!(
        "SPEECH_SEG program={program} segments={} total_us={total:.1}",
        p.segments
    );
}

/// The conv object runs a packet's conv segments with the speech object's bits: every tensor after
/// one run from the same initial bytes, with and without it. Env: SPEECH_CONV_PKT (its assets hold
/// the `_speechconv` cubin), SPEECH_CONV_PROGRAMS (indices; default every program with a conv
/// segment), SPEECH_CONV_TIMING (interleaved event timing of both, median of 10 each).
#[cfg(test)]
#[test]
#[ignore = "CUDA; speech conv object equivalence, run only through gpuq"]
fn speech_conv_object_matches_speech_object() {
    let path = std::path::PathBuf::from(std::env::var("SPEECH_CONV_PKT").unwrap());
    let be = Arc::new(CudaBackend::new(0).unwrap());
    let mut rt = CudaPacketRuntime::load_on(be.clone(), &path).unwrap();
    let mut parked: Option<ConvObject> = None;
    assert!(rt.conv.is_some(), "no conv object loaded");
    let programs: Vec<usize> = match std::env::var("SPEECH_CONV_PROGRAMS") {
        Ok(list) => list.split(',').map(|s| s.parse().unwrap()).collect(),
        Err(_) => (0..rt.programs.len())
            .filter(|&p| rt.programs[p].conv_segments.iter().any(Option::is_some))
            .collect(),
    };
    let snapshot = |rt: &CudaPacketRuntime| -> Vec<Vec<u8>> {
        rt.tensors
            .iter()
            .map(|m| {
                let mut out = vec![0; m.len as usize];
                rt.download(m, 0, &mut out).unwrap();
                out
            })
            .collect()
    };
    let initial = snapshot(&rt);
    let reset = |rt: &CudaPacketRuntime| {
        for (m, bytes) in rt.tensors.iter().zip(&initial) {
            rt.upload(m, 0, bytes).unwrap();
        }
    };
    // mode 0: speech object only, 1: conv segments on the conv object.
    let mut set = |rt: &mut CudaPacketRuntime, mode: usize| {
        if rt.conv.is_some() != (mode == 1) {
            std::mem::swap(&mut rt.conv, &mut parked);
        }
    };
    for p in programs {
        let conv_segments = rt.programs[p]
            .conv_segments
            .iter()
            .filter(|c| c.is_some())
            .count();
        let mut results = Vec::new();
        for mode in 0..2 {
            set(&mut rt, mode);
            reset(&rt);
            rt.enqueue_on(&[p], &rt.stream).unwrap();
            be.stream_synchronize(&rt.stream).unwrap();
            results.push(snapshot(&rt));
        }
        let differ: Vec<usize> = (0..results[0].len())
            .filter(|&t| results[0][t] != results[1][t])
            .collect();
        println!(
            "SPEECH_CONV_BITS program={p} segments={} conv_segments={conv_segments} differing_tensors={differ:?}",
            rt.programs[p].segments
        );
        assert!(differ.is_empty());
        if std::env::var_os("SPEECH_CONV_TIMING").is_none() {
            continue;
        }
        let mut times = [Vec::new(), Vec::new()];
        for round in 0..24 {
            let mode = round % 2;
            set(&mut rt, mode);
            reset(&rt);
            be.event_record(&rt.events.0, &rt.stream).unwrap();
            rt.enqueue_on(&[p], &rt.stream).unwrap();
            be.event_record(&rt.events.1, &rt.stream).unwrap();
            be.event_synchronize(&rt.events.1).unwrap();
            if round >= 4 {
                times[mode].push(be.event_elapsed_ms(&rt.events.0, &rt.events.1).unwrap() * 1000.);
            }
        }
        for t in &mut times {
            t.sort_by(f32::total_cmp);
        }
        println!(
            "SPEECH_CONV_TIME program={p} conv_segments={conv_segments} speech_us={} conv_us={} samples=10_each",
            times[0][5], times[1][5]
        );
    }
}

/// The speech object's bf16 taps convs (Conv1dF32 flags bit 17 without 14, kernel > 1) on their own
/// kernel entry (`speech_conv_node(const PlowDevInst*, void* const*)`) between interpreter spans,
/// against the packet's program and against the same cuts with the conv as its own interpreter
/// span (the split's cost alone). Every tensor is compared bit for bit after one run from the
/// same initial bytes. Env: SPEECH_SPLIT_PKT (codec.pkt, PLOW_HSACO = its assets), SPEECH_SPLIT_ROUTE,
/// SPEECH_SPLIT_CUBIN, SPEECH_SPLIT_PROGRAMS (indices), SPEECH_SPLIT_SMEM, SPEECH_SPLIT_TIMING.
#[cfg(test)]
#[test]
#[ignore = "CUDA; speech conv split screen, run only through gpuq"]
fn speech_conv_node_split_matches_packet() {
    use packet::dev::DevInst64;
    enum Step {
        Packet {
            argument: DevProgram,
            counters: usize,
            _tables: Vec<DeviceMem>,
        },
        Conv {
            instruction: DeviceMem,
            blocks: u32,
        },
    }
    let path = std::path::PathBuf::from(std::env::var("SPEECH_SPLIT_PKT").unwrap());
    let image = std::fs::read(std::env::var("SPEECH_SPLIT_CUBIN").unwrap()).unwrap();
    let programs = std::env::var("SPEECH_SPLIT_PROGRAMS").unwrap_or("27".into());
    let smem: u32 = std::env::var("SPEECH_SPLIT_SMEM").map_or(133120, |s| s.parse().unwrap());
    let be = Arc::new(CudaBackend::new(0).unwrap());
    let module = be.module_load(&image).unwrap();
    let node = be.get_function(&module, "speech_conv_node").unwrap();
    be.set_max_dynamic_smem(node, smem).unwrap();
    let raw = std::fs::read(&path).unwrap();
    let blob = DevBlob::parse(&raw).unwrap();
    let rt = CudaPacketRuntime::load_on(be.clone(), &path).unwrap();
    // SPEECH_SPLIT_ROUTE=wgmma: every wgmma conv arm (flags bit 17, ConvTranspose1dF32 too).
    let all_wgmma = std::env::var("SPEECH_SPLIT_ROUTE").is_ok_and(|r| r == "wgmma");
    let routed = |inst: &DevInst64| {
        let conv = inst.op == DevOp::Conv1dF32 as u16;
        let wgmma =
            (conv || inst.op == DevOp::ConvTranspose1dF32 as u16) && inst.fj[2] & (1 << 17) != 0;
        wgmma && (all_wgmma || conv && inst.fj[2] & (1 << 14) == 0 && inst.i[4] > 1)
    };
    let snapshot = || -> Vec<Vec<u8>> {
        rt.tensors
            .iter()
            .map(|m| {
                let mut out = vec![0; m.len as usize];
                rt.download(m, 0, &mut out).unwrap();
                out
            })
            .collect()
    };
    let initial = snapshot();
    let reset = || {
        for (m, bytes) in rt.tensors.iter().zip(&initial) {
            rt.upload(m, 0, bytes).unwrap();
        }
    };
    for p in programs.split(',').map(|s| s.parse::<usize>().unwrap()) {
        let g = &blob.progs[p];
        // [split only, conv node]
        let mut plans: [Vec<Step>; 2] = [Vec::new(), Vec::new()];
        for (mode, steps) in plans.iter_mut().enumerate() {
            let mut span = Vec::new();
            let emit_span = |span: &mut Vec<DevInst64>, steps: &mut Vec<Step>| {
                if span.is_empty() {
                    return;
                }
                let program = super::packed_terminal::chain(std::mem::take(span), rt.grid, g.t);
                let (argument, _, counters, tables) =
                    super::mixed_step::upload_program(&be, &program, rt._table.base, 0, 0).unwrap();
                steps.push(Step::Packet {
                    argument,
                    counters,
                    _tables: tables,
                });
            };
            for inst in &g.insts {
                if !routed(inst) {
                    span.push(*inst);
                    continue;
                }
                emit_span(&mut span, steps);
                if mode == 0 {
                    span.push(*inst);
                    emit_span(&mut span, steps);
                    continue;
                }
                let memory = be.alloc(0, 64).unwrap();
                be.upload(&memory, 0, pod_bytes(std::slice::from_ref(inst)))
                    .unwrap();
                steps.push(Step::Conv {
                    instruction: memory,
                    blocks: u32::from(inst.blocks).max(1),
                });
            }
            emit_span(&mut span, steps);
        }
        let n_routed = g.insts.iter().filter(|i| routed(i)).count();
        let enqueue = |plan: Option<&Vec<Step>>| {
            let Some(steps) = plan else {
                rt.enqueue_on(&[p], &rt.stream).unwrap();
                return;
            };
            for step in steps {
                match step {
                    Step::Packet {
                        argument, counters, ..
                    } => {
                        be.memset_d8_async(argument.counters, 0, *counters, &rt.stream)
                            .unwrap();
                        let mut argument = *argument;
                        let mut params =
                            [&mut argument as *mut DevProgram as *mut std::ffi::c_void];
                        be.launch_cooperative(
                            rt.function,
                            rt.grid,
                            BLOCK,
                            rt.smem,
                            &mut params,
                            Some(&rt.stream),
                        )
                        .unwrap();
                    }
                    Step::Conv {
                        instruction,
                        blocks,
                    } => {
                        let mut inst = instruction.base;
                        let mut table = rt._table.base;
                        let mut params = [
                            &mut inst as *mut u64 as *mut std::ffi::c_void,
                            &mut table as *mut u64 as *mut std::ffi::c_void,
                        ];
                        be.launch_kernel(node, *blocks, BLOCK, smem, &mut params, Some(&rt.stream))
                            .unwrap();
                    }
                }
            }
        };
        let plan = |mode: usize| {
            if mode == 0 {
                None
            } else {
                Some(&plans[mode - 1])
            }
        };
        let mut results = Vec::new();
        for mode in 0..3 {
            reset();
            enqueue(plan(mode));
            be.stream_synchronize(&rt.stream).unwrap();
            results.push(snapshot());
        }
        for mode in 1..3 {
            let differ: Vec<usize> = (0..results[0].len())
                .filter(|&t| results[0][t] != results[mode][t])
                .collect();
            println!(
                "SPEECH_SPLIT_BITS program={p} mode={} routed={n_routed} differing_tensors={:?}",
                ["packet", "split", "node"][mode],
                differ
            );
            assert!(differ.is_empty());
        }
        if std::env::var_os("SPEECH_SPLIT_TIMING").is_none() {
            continue;
        }
        let mut times = [Vec::new(), Vec::new(), Vec::new()];
        for round in 0..36 {
            let mode = round % 3;
            reset();
            be.event_record(&rt.events.0, &rt.stream).unwrap();
            enqueue(plan(mode));
            be.event_record(&rt.events.1, &rt.stream).unwrap();
            be.event_synchronize(&rt.events.1).unwrap();
            if round >= 6 {
                times[mode].push(be.event_elapsed_ms(&rt.events.0, &rt.events.1).unwrap() * 1000.);
            }
        }
        for t in &mut times {
            t.sort_by(f32::total_cmp);
        }
        println!(
            "SPEECH_SPLIT_TIME program={p} routed={n_routed} packet_us={} split_us={} node_us={} samples=10_each",
            times[0][times[0].len() / 2],
            times[1][times[1].len() / 2],
            times[2][times[2].len() / 2]
        );
    }
    be.synchronize().unwrap();
    be.module_unload(&module).unwrap();
}
