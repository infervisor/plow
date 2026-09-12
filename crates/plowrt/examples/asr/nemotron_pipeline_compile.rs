mod nemotron_packet_support;

use std::path::Path;

use devgen::asr::frontend::LogMelSpec;
use devgen::asr_subsampling::{
    self, Conv2dStage, ConvActivation, ConvKind, ConvWeight, Projection, SubsamplingSpec,
};
use devgen::conformer::{self, ConformerSpec};
use devgen::conv2d::ConvLayout;
use devgen::pipeline::PacketPrefix;
use devgen::rnnt::{self, LinearWeights, LstmWeights, RnntSpec, RnntWeights};
use nemotron_packet_support::{LayerNames, SubsamplingNames};
use plowrt::asr::conformer::AttentionMask;
use plowrt::asr::nemotron::conformer_encoder_plan;
use plowrt::asset::gguf::GgufFile;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if !(4..=7).contains(&args.len()) {
        return Err(
            "usage: asr_nemotron_pipeline_compile MODEL_GGUF FEATURE_FRAMES OUTPUT_PACKET [JOINT_BATCH] [N_CU] [TRAILING_ENCODER_FRAMES]"
                .into(),
        );
    }
    let mut feature_frame_capacities: Vec<u32> = args[2]
        .split(',')
        .map(str::parse)
        .collect::<Result<_, _>>()?;
    feature_frame_capacities.sort_unstable();
    feature_frame_capacities.dedup();
    let joint_batch = args.get(4).map_or(Ok(16), |value| value.parse::<u32>())?;
    let n_cu = args.get(5).map_or(Ok(16), |value| value.parse::<u32>())?;
    if feature_frame_capacities.first().copied().unwrap_or(0) == 0 || n_cu == 0 {
        return Err("FEATURE_FRAMES and N_CU must be positive".into());
    }
    let feature_frames = *feature_frame_capacities
        .last()
        .ok_or("FEATURE_FRAMES must contain at least one capacity")?;

    let gguf = GgufFile::open(Path::new(&args[1]))?;
    let plan = conformer_encoder_plan(&gguf)?;
    let first = plan
        .blocks()
        .first()
        .ok_or("encoder has no Conformer layers")?;
    let AttentionMask::ChunkedLimited {
        chunk_size,
        left_chunks,
    } = first.attention.mask
    else {
        return Err("packet compiler requires chunk-limited attention".into());
    };
    let attention_right_context = u32::try_from(
        chunk_size
            .checked_sub(1)
            .ok_or("attention chunk size must be positive")?,
    )?;
    // A covering capacity program sees padded encoder rows after the exact-shape
    // frontier. Two right-context windows is the measured safe RNNT flush for
    // Nemotron 3.5; the explicit argument keeps this packet policy configurable.
    let model_trailing_encoder_frames = attention_right_context
        .checked_mul(2)
        .ok_or("trailing encoder frame count overflows")?;
    let trailing_encoder_frames = args
        .get(6)
        .map_or(Ok(model_trailing_encoder_frames), |value| {
            value.parse::<u32>()
        })?;
    if trailing_encoder_frames > 32 {
        return Err("TRAILING_ENCODER_FRAMES must be at most 32".into());
    }
    let subsampling_names: Vec<_> = [0, 2, 3, 5, 6]
        .into_iter()
        .map(SubsamplingNames::new)
        .collect();
    let definitions = [
        (3, 2, 1, 256, ConvKind::Standard, true),
        (3, 2, 256, 256, ConvKind::Depthwise, false),
        (1, 1, 256, 256, ConvKind::Standard, true),
        (3, 2, 256, 256, ConvKind::Depthwise, false),
        (1, 1, 256, 256, ConvKind::Standard, true),
    ];
    let mut stages: Vec<_> = subsampling_names
        .iter()
        .zip(definitions)
        .map(
            |(names, (kernel, stride, input_channels, output_channels, kind, relu))| Conv2dStage {
                weight: names.weight(),
                bias: names.bias(),
                kernel,
                stride,
                pad_before: if kernel == 3 { 2 } else { 0 },
                pad_after: if kernel == 3 { 1 } else { 0 },
                input_channels,
                output_channels,
                kind,
                activation: if relu {
                    ConvActivation::Relu
                } else {
                    ConvActivation::None
                },
                output_layout: ConvLayout::ChannelsLast,
                weight_type: ConvWeight::F16,
            },
        )
        .collect();
    if let Some(first) = stages.first_mut() {
        first.output_layout = ConvLayout::ChannelsFramesWidth;
    }
    if let Some(last) = stages.last_mut() {
        last.output_layout = ConvLayout::FrameChannelsWidth;
    }
    let layer_names: Vec<_> = (0..plan.blocks().len()).map(LayerNames::new).collect();
    let layers: Vec<_> = layer_names.iter().map(LayerNames::borrow).collect();
    let lstm = [lstm_weights(0), lstm_weights(1)];
    let compile_capacity = |input_frames| -> Result<_, Box<dyn std::error::Error>> {
        let subsampling = asr_subsampling::lower(
            SubsamplingSpec {
                input_frames,
                input_width: 128,
            },
            &stages,
            Projection {
                weight: "encoder.pre_encode.out.weight",
                bias: "encoder.pre_encode.out.bias",
                output_width: 1024,
            },
            n_cu,
        )?;
        let frames = subsampling.output_frames;
        let conformer = conformer::append(
            ConformerSpec {
                frames,
                width: plan.width().try_into()?,
                feed_forward_width: first.feed_forward1.expand.n().try_into()?,
                heads: first.attention.heads.try_into()?,
                convolution_kernel: first.convolution.kernel.try_into()?,
                chunk_size: chunk_size.try_into()?,
                left_chunks: left_chunks.try_into()?,
                position_table: "encoder.pos_enc.pe",
                position_count: first.attention.position_count.try_into()?,
                position_center: first.attention.position_center.try_into()?,
                epsilon: first.output_norm.epsilon(),
            },
            &layers,
            subsampling.into_prefix(),
        )?;
        let prefix: PacketPrefix = conformer.into_prefix();
        Ok((
            rnnt::append(
                RnntSpec {
                    frames,
                    encoder_width: 1024,
                    prompt_count: 128,
                    prompt_width: 2048,
                    vocabulary: 13088,
                    predictor_width: 640,
                    joint_width: 640,
                    joint_batch,
                },
                &RnntWeights {
                    prompt_in: linear("prompt_kernel.0"),
                    prompt_out: linear("prompt_kernel.2"),
                    encoder: linear("joint.enc"),
                    embedding: "decoder.prediction.embed.weight",
                    lstm: &lstm,
                    predictor: linear("joint.pred"),
                    output: linear("joint.joint_net.2"),
                },
                prefix,
                0,
            )?,
            frames,
        ))
    };
    let (mut packets, frames) = compile_capacity(feature_frames)?;
    for &capacity in &feature_frame_capacities[..feature_frame_capacities.len() - 1] {
        let (bucket, _) = compile_capacity(capacity)?;
        packets.merge_encoder_bucket(capacity, bucket)?;
    }
    packets.embed_weights(|name| {
        gguf.tensor(name)
            .map(|tensor| tensor.bytes.to_vec())
            .map_err(|error| error.to_string())
    })?;
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
        .map(|stage| {
            [
                stage.kernel,
                stage.stride,
                stage.pad_before,
                stage.pad_after,
            ]
        })
        .collect();
    let section = packets.pipeline_section_with_frame_transform(
        13087,
        10,
        0,
        feature_frames,
        &frame_transform,
        trailing_encoder_frames,
    )?;
    std::fs::write(&args[3], packets.model.to_blob_v6(&[section]))?;
    println!(
        "{}",
        serde_json::json!({
            "driver": "rnnt.greedy.v1",
            "encoder_programs": packets.encoder_programs,
            "frames": frames,
            "feature_frames": feature_frames,
            "feature_frame_capacities": feature_frame_capacities,
            "joint_batch": joint_batch,
            "n_cu": n_cu,
            "trailing_encoder_frames": trailing_encoder_frames,
            "output": args[3],
        })
    );
    Ok(())
}

fn linear(prefix: &'static str) -> LinearWeights<'static> {
    LinearWeights {
        weight: match prefix {
            "prompt_kernel.0" => "prompt_kernel.0.weight",
            "prompt_kernel.2" => "prompt_kernel.2.weight",
            "joint.enc" => "joint.enc.weight",
            "joint.pred" => "joint.pred.weight",
            "joint.joint_net.2" => "joint.joint_net.2.weight",
            _ => unreachable!(),
        },
        bias: match prefix {
            "prompt_kernel.0" => "prompt_kernel.0.bias",
            "prompt_kernel.2" => "prompt_kernel.2.bias",
            "joint.enc" => "joint.enc.bias",
            "joint.pred" => "joint.pred.bias",
            "joint.joint_net.2" => "joint.joint_net.2.bias",
            _ => unreachable!(),
        },
    }
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
