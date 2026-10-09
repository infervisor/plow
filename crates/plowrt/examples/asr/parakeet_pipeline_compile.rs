//! Parakeet TDT (FastConformer + token-and-duration transducer) from its GGUF to an
//! `rnnt.greedy.v1` packet: full-context relative attention, centred depthwise convolutions with
//! the batch norm folded in, no language prompt, and a TDT joint.

mod nemotron_packet_support;

use std::path::Path;

use devgen::asr::frontend::LogMelSpec;
use devgen::asr_subsampling::{
    self, Conv2dStage, ConvActivation, ConvKind, ConvWeight, Projection, SubsamplingSpec,
};
use devgen::conformer::{self, ConformerSpec};
use devgen::conv2d::ConvLayout;
use devgen::rnnt::{self, LinearWeights, LstmWeights, RnntSpec, RnntWeights};
use gguf_rs_lib::format::metadata::MetadataValue;
use gguf_rs_lib::format::types::GGUFTensorType;
use nemotron_packet_support::{LayerNames, SubsamplingNames};
use plowrt::asset::gguf::GgufFile;

/// NeMo `BatchNorm1d` default epsilon (the conformer convolution module's norm).
const BATCH_NORM_EPSILON: f32 = 1e-5;
const FOLDED_WEIGHT: &str = ".bn_folded_weight";
const FOLDED_BIAS: &str = ".bn_folded_bias";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if !(4..=6).contains(&args.len()) {
        return Err("usage: asr_parakeet_pipeline_compile MODEL_GGUF FEATURE_FRAMES OUTPUT_PACKET [JOINT_BATCH] [N_CU]".into());
    }
    let mut capacities: Vec<u32> = args[2].split(',').map(str::parse).collect::<Result<_, _>>()?;
    capacities.sort_unstable();
    capacities.dedup();
    let joint_batch = args.get(4).map_or(Ok(16), |value| value.parse::<u32>())?;
    let n_cu = args.get(5).map_or(Ok(16), |value| value.parse::<u32>())?;
    let feature_frames = *capacities.last().ok_or("FEATURE_FRAMES must contain a capacity")?;
    if capacities[0] == 0 || n_cu == 0 {
        return Err("FEATURE_FRAMES and N_CU must be positive".into());
    }

    let gguf = GgufFile::open(Path::new(&args[1]))?;
    let meta = |key: &str| -> Result<u32, String> {
        gguf.metadata()
            .get_u64(key)
            .and_then(|value| u32::try_from(value).ok())
            .ok_or_else(|| format!("missing or invalid {key}"))
    };
    let text = |key: &str| gguf.metadata().get_string(key).unwrap_or_default().to_owned();
    if text("asr.head_type") != "tdt"
        || text("asr.encoder.att_context_style") != "regular"
        || text("asr.encoder.conv_norm") != "batch_norm"
        || gguf.metadata().get_bool("asr.encoder.use_bias") != Some(false)
        || gguf.metadata().get_bool("asr.encoder.xscaling") != Some(false)
    {
        return Err("not a Parakeet TDT checkpoint (full-context attention, batch-norm convolutions)".into());
    }
    let durations = u32_array(&gguf, "asr.tdt.durations")?;
    let width = meta("asr.encoder.d_model")?;
    let layers = meta("asr.encoder.n_layers")? as usize;
    let vocabulary = meta("asr.rnnt.vocab_size")?;
    let blank_id = meta("asr.rnnt.blank_id")?;
    let joint_width = meta("asr.rnnt.joint_dim")?;
    let predictor_width = meta("asr.rnnt.pred_embed_dim")?;
    let channels = meta("asr.encoder.subsampling_conv_channels")?;
    let bins = meta("asr.encoder.feat_in")?;
    let position = gguf.tensor("encoder.pos_enc.pe")?;
    let position_count = u32::try_from(position.dimensions[1])?;

    let subsampling_names: Vec<_> = [0, 2, 3, 5, 6].into_iter().map(SubsamplingNames::new).collect();
    let definitions = [
        (3, 2, 1, channels, ConvKind::Standard, true),
        (3, 2, channels, channels, ConvKind::Depthwise, false),
        (1, 1, channels, channels, ConvKind::Standard, true),
        (3, 2, channels, channels, ConvKind::Depthwise, false),
        (1, 1, channels, channels, ConvKind::Standard, true),
    ];
    // NeMo's dw_striding subsampling pads (kernel - 1) / 2 on both sides.
    let mut stages: Vec<_> = subsampling_names
        .iter()
        .zip(definitions)
        .map(|(names, (kernel, stride, input_channels, output_channels, kind, relu))| Conv2dStage {
            weight: names.weight(),
            bias: names.bias(),
            kernel,
            stride,
            pad_before: (kernel - 1) / 2,
            pad_after: (kernel - 1) / 2,
            input_channels,
            output_channels,
            kind,
            activation: if relu { ConvActivation::Relu } else { ConvActivation::None },
            output_layout: ConvLayout::ChannelsLast,
            weight_type: ConvWeight::F16,
        })
        .collect();
    stages.first_mut().unwrap().output_layout = ConvLayout::ChannelsFramesWidth;
    stages.last_mut().unwrap().output_layout = ConvLayout::FrameChannelsWidth;

    let layer_names: Vec<_> = (0..layers).map(LayerNames::new).collect();
    let folded: Vec<(String, String)> = (0..layers)
        .map(|layer| {
            let base = format!("encoder.layers.{layer}.conv.depthwise_conv");
            (format!("{base}{FOLDED_WEIGHT}"), format!("{base}{FOLDED_BIAS}"))
        })
        .collect();
    let layer_weights: Vec<_> = layer_names
        .iter()
        .zip(&folded)
        .map(|(names, (weight, bias))| {
            let mut layer = names.borrow();
            layer.convolution.depthwise = weight;
            layer.convolution.depthwise_bias = Some(bias);
            layer
        })
        .collect();
    let lstm = [lstm_weights(0), lstm_weights(1)];
    let unused = LinearWeights { weight: "unused.prompt.weight", bias: "unused.prompt.bias" };
    let rnnt_weights = RnntWeights {
        prompt_in: unused,
        prompt_out: unused,
        encoder: LinearWeights { weight: "joint.enc.weight", bias: "joint.enc.bias" },
        embedding: "decoder.prediction.embed.weight",
        lstm: &lstm,
        predictor: LinearWeights { weight: "joint.pred.weight", bias: "joint.pred.bias" },
        output: LinearWeights { weight: "joint.joint_net.2.weight", bias: "joint.joint_net.2.bias" },
    };
    let compile_capacity = |input_frames: u32| -> Result<_, Box<dyn std::error::Error>> {
        let subsampling = asr_subsampling::lower(
            SubsamplingSpec { input_frames, input_width: bins },
            &stages,
            Projection { weight: "encoder.pre_encode.out.weight", bias: "encoder.pre_encode.out.bias", output_width: width },
            n_cu,
        )?;
        let frames = subsampling.output_frames;
        let spec = ConformerSpec {
            frames,
            width,
            feed_forward_width: meta("asr.encoder.d_ff")?,
            heads: meta("asr.encoder.n_heads")?,
            convolution_kernel: meta("asr.encoder.conv_kernel_size")?,
            causal_convolution: false,
            chunk_size: frames,
            left_chunks: u32::MAX,
            position_table: "encoder.pos_enc.pe",
            position_count,
            position_center: meta("asr.encoder.pos_emb_max_len")?,
            epsilon: 1e-5,
        };
        let encoder = conformer::append(spec, &layer_weights, subsampling.into_prefix())?;
        let rnnt_spec = RnntSpec {
            frames,
            encoder_width: width,
            prompt_count: 0,
            prompt_width: 0,
            vocabulary,
            durations: u32::try_from(durations.len())?,
            predictor_width,
            joint_width,
            joint_batch,
        };
        Ok((rnnt::append(rnnt_spec, &rnnt_weights, encoder.into_prefix(), 0)?, frames))
    };
    let (mut packets, frames) = compile_capacity(feature_frames)?;
    for &capacity in &capacities[..capacities.len() - 1] {
        let (bucket, _) = compile_capacity(capacity)?;
        packets.merge_encoder_bucket(capacity, bucket)?;
    }
    packets.set_tdt_durations(&durations)?;
    packets.embed_weights(|name| resolve(&gguf, name))?;
    let (frontend, filterbank) = plowrt::asr::nemotron::NemotronFrontend::components(&gguf)?;
    packets.embed_log_mel_frontend(
        LogMelSpec {
            sample_rate: frontend.sample_rate.try_into()?,
            fft: frontend.fft.try_into()?,
            window: frontend.window.try_into()?,
            hop: frontend.hop.try_into()?,
            bins: frontend.bins.try_into()?,
            preemphasis: frontend.preemphasis,
            center_window: frontend.center_window,
            periodic_hann: frontend.periodic_hann,
            normalize_per_feature: frontend.normalize_per_feature,
            mask_invalid_frames: frontend.mask_invalid_frames,
            log_guard: frontend.log_guard,
            min_samples: (frontend.sample_rate / 2).try_into()?,
            max_samples: usize::try_from(feature_frames)?
                .checked_mul(frontend.hop)
                .and_then(|samples| samples.checked_sub(1))
                .ok_or("feature capacity overflows")?
                .try_into()?,
        },
        &filterbank,
    )?;
    let frame_transform: Vec<_> = stages
        .iter()
        .map(|stage| [stage.kernel, stage.stride, stage.pad_before, stage.pad_after])
        .collect();
    // Full context: no right-context flush; padded frames are masked, not decoded.
    let section =
        packets.pipeline_section_with_frame_transform(blank_id, 10, 0, feature_frames, &frame_transform, 0)?;
    std::fs::write(&args[3], packets.model.to_blob_v6(&[section]))?;
    println!(
        "{}",
        serde_json::json!({
            "driver": "rnnt.greedy.v1",
            "head": "tdt",
            "durations": durations,
            "frames": frames,
            "feature_frames": feature_frames,
            "feature_frame_capacities": capacities,
            "joint_batch": joint_batch,
            "n_cu": n_cu,
            "output": args[3],
        })
    );
    Ok(())
}

/// Checkpoint tensors as stored, plus each depthwise convolution with its batch norm folded in:
/// `w' = w * s`, `b' = beta - mean * s`, `s = gamma / sqrt(var + eps)` per channel (FP32).
fn resolve(gguf: &GgufFile, name: &str) -> Result<Vec<u8>, String> {
    let fold = |base: &str| -> Result<(Vec<f32>, Vec<f32>), String> {
        let layer = base.trim_end_matches(".depthwise_conv");
        let read = |name: &str| f32_values(gguf, name);
        let (gamma, beta) = (read(&format!("{layer}.batch_norm.weight"))?, read(&format!("{layer}.batch_norm.bias"))?);
        let (mean, var) = (read(&format!("{layer}.batch_norm.running_mean"))?, read(&format!("{layer}.batch_norm.running_var"))?);
        let weights = f32_values(gguf, &format!("{base}.weight"))?;
        let channels = gamma.len();
        if [&beta, &mean, &var].iter().any(|v| v.len() != channels) || weights.len() % channels != 0 {
            return Err(format!("batch norm of {base} does not match its convolution"));
        }
        let kernel = weights.len() / channels;
        let scale: Vec<f32> = (0..channels).map(|c| gamma[c] / (var[c] + BATCH_NORM_EPSILON).sqrt()).collect();
        let folded = weights.iter().enumerate().map(|(i, w)| w * scale[i / kernel]).collect();
        let bias = (0..channels).map(|c| beta[c] - mean[c] * scale[c]).collect();
        Ok((folded, bias))
    };
    let bytes = |values: Vec<f32>| values.iter().flat_map(|v| v.to_le_bytes()).collect();
    if let Some(base) = name.strip_suffix(FOLDED_WEIGHT) {
        return fold(base).map(|(weight, _)| bytes(weight));
    }
    if let Some(base) = name.strip_suffix(FOLDED_BIAS) {
        return fold(base).map(|(_, bias)| bytes(bias));
    }
    gguf.tensor(name).map(|tensor| tensor.bytes.to_vec()).map_err(|error| error.to_string())
}

/// An F32 or F16 tensor as FP32 values, in storage order.
fn f32_values(gguf: &GgufFile, name: &str) -> Result<Vec<f32>, String> {
    let tensor = gguf.tensor(name).map_err(|error| error.to_string())?;
    match tensor.dtype {
        GGUFTensorType::F32 => Ok(tensor.bytes.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect()),
        GGUFTensorType::F16 => tensor.f16_values().map_err(|error| error.to_string()),
        other => Err(format!("{name} is {other:?}, expected F32 or F16")),
    }
}

fn u32_array(gguf: &GgufFile, key: &str) -> Result<Vec<u32>, String> {
    let Some(MetadataValue::Array(values)) = gguf.metadata().data.get(key) else {
        return Err(format!("missing {key}"));
    };
    values
        .values
        .iter()
        .map(|value| match value {
            MetadataValue::U32(v) => Ok(*v),
            MetadataValue::I32(v) => u32::try_from(*v).map_err(|_| format!("{key} is negative")),
            MetadataValue::U64(v) => u32::try_from(*v).map_err(|_| format!("{key} overflows")),
            MetadataValue::I64(v) => u32::try_from(*v).map_err(|_| format!("{key} is out of range")),
            other => Err(format!("{key} holds {other:?}")),
        })
        .collect()
}

fn lstm_weights(layer: usize) -> LstmWeights<'static> {
    match layer {
        0 => LstmWeights {
            input: LinearWeights {
                weight: "decoder.prediction.dec_rnn.lstm.ih_l0.weight",
                bias: "decoder.prediction.dec_rnn.lstm.ih_l0.bias",
            },
            recurrent: LinearWeights {
                weight: "decoder.prediction.dec_rnn.lstm.hh_l0.weight",
                bias: "decoder.prediction.dec_rnn.lstm.hh_l0.bias",
            },
        },
        1 => LstmWeights {
            input: LinearWeights {
                weight: "decoder.prediction.dec_rnn.lstm.ih_l1.weight",
                bias: "decoder.prediction.dec_rnn.lstm.ih_l1.bias",
            },
            recurrent: LinearWeights {
                weight: "decoder.prediction.dec_rnn.lstm.hh_l1.weight",
                bias: "decoder.prediction.dec_rnn.lstm.hh_l1.bias",
            },
        },
        _ => unreachable!(),
    }
}
