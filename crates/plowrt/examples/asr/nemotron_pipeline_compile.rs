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
    if !(4..=13).contains(&args.len()) {
        return Err(
            "usage: asr_nemotron_pipeline_compile MODEL_GGUF FEATURE_FRAMES OUTPUT_PACKET [JOINT_BATCH] [N_CU] [TRAILING_ENCODER_FRAMES] [STREAM_ROWS] [DECODE_BATCH] [STREAM_SLOTS] [STREAM_BATCH] [UPLOAD_ROWS] [UPLOAD_BATCH]"
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
                width_stride: 0,
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
    let rnnt_weights = RnntWeights {
        prompt_in: linear("prompt_kernel.0"),
        prompt_out: linear("prompt_kernel.2"),
        encoder: linear("joint.enc"),
        embedding: "decoder.prediction.embed.weight",
        lstm: &lstm,
        predictor: linear("joint.pred"),
        output: linear("joint.joint_net.2"),
    };
    let conformer_spec = |frames| -> Result<ConformerSpec<'static>, Box<dyn std::error::Error>> {
        Ok(ConformerSpec {
            frames,
            width: plan.width().try_into()?,
            feed_forward_width: first.feed_forward1.expand.n().try_into()?,
            heads: first.attention.heads.try_into()?,
            convolution_kernel: first.convolution.kernel.try_into()?,
            causal_convolution: true,
            chunk_size: chunk_size.try_into()?,
            left_chunks: left_chunks.try_into()?,
            position_table: "encoder.pos_enc.pe",
            position_count: first.attention.position_count.try_into()?,
            position_center: first.attention.position_center.try_into()?,
            epsilon: first.output_norm.epsilon(),
        })
    };
    let subsample = |input_frames, items| {
        asr_subsampling::lower_items(
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
            items,
        )
    };
    let compile_capacity = |input_frames| -> Result<_, Box<dyn std::error::Error>> {
        let subsampling = subsample(input_frames, 1)?;
        let frames = subsampling.output_frames;
        let conformer =
            conformer::append(conformer_spec(frames)?, &layers, subsampling.into_prefix())?;
        let prefix: PacketPrefix = conformer.into_prefix();
        Ok((
            rnnt::append(
                RnntSpec {
                    frames,
                    encoder_width: 1024,
                    prompt_count: 128,
                    prompt_width: 2048,
                    vocabulary: 13088,
                    durations: 0,
                    predictor_width: 640,
                    joint_width: 640,
                    joint_batch,
                },
                &rnnt_weights,
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
    // Cache-aware stream: STREAM_ROWS new encoder frames (whole attention chunks; default one
    // chunk, 0 = no stream) per step. The subsampling is causal and factor 8: a later step's mel
    // window carries 16 frames already seen, enough for its first new output row.
    let stream_rows = args
        .get(7)
        .map_or(Ok(u32::try_from(chunk_size)?), |v| v.parse::<u32>())?;
    // Batched greedy decode: up to DECODE_BATCH recordings (default 32, 0 = none) per launch,
    // each with a slot (one more for padding lanes) and a full-capacity encoder-table region.
    let decode_batch = args.get(8).map_or(Ok(32), |v| v.parse::<u32>())?;
    // Batched streams (they decode through the batched decode): STREAM_SLOTS open streams
    // (default 256, 0 = none), up to STREAM_BATCH (default 128) per launch.
    let stream_slots = args.get(9).map_or(Ok(256), |v| v.parse::<u32>())?;
    let stream_batch = args.get(10).map_or(Ok(128), |v| v.parse::<u32>())?;
    // Upload streams (whole recordings, in the streams' slots): UPLOAD_ROWS new encoder frames
    // per step (default 32, 0 = none), up to UPLOAD_BATCH (default 32) per launch.
    let upload_rows = args.get(11).map_or(Ok(32), |v| v.parse::<u32>())?;
    let upload_batch = args.get(12).map_or(Ok(32), |v| v.parse::<u32>())?;
    let chunk_items = if stream_rows != 0 && decode_batch != 0 && stream_slots != 0 {
        stream_batch
    } else {
        0
    };
    let upload_items = if chunk_items != 0 && upload_rows != 0 {
        upload_batch
    } else {
        0
    };
    let rungs = |max: u32| -> Vec<u32> {
        let mut rungs: Vec<_> = (0..)
            .map(|shift| 1u32 << shift)
            .take_while(|&rung| rung < max)
            .collect();
        rungs.push(max);
        rungs
    };
    if stream_rows != 0 {
        const FACTOR: u32 = 8;
        const HISTORY: u32 = 16;
        let stream_prefix =
            |input_frames, keep_row0| -> Result<PacketPrefix, Box<dyn std::error::Error>> {
                let subsampling = subsample(input_frames, 1)?;
                let spec = conformer_spec(subsampling.output_frames)?;
                Ok(conformer::append_stream_step(
                    spec,
                    &layers,
                    subsampling.into_prefix(),
                    keep_row0,
                    stream_rows,
                )?)
            };
        let spec = conformer_spec(stream_rows)?;
        packets.add_stream(
            rnnt::StreamGeometry {
                rows: stream_rows,
                left_rows: conformer::stream_left_rows(&spec),
                first_input_frames: FACTOR * stream_rows,
                step_input_frames: FACTOR * stream_rows + HISTORY,
                history_input_frames: HISTORY,
                step_keep_row0: HISTORY / FACTOR,
            },
            conformer::stream_init(spec, &layers, stream_rows, n_cu, conformer::STREAM_POSITION)?,
            stream_prefix(FACTOR * stream_rows, 0)?,
            stream_prefix(FACTOR * stream_rows + HISTORY, HISTORY / FACTOR)?,
            &rnnt_weights,
        )?;
        if chunk_items != 0 {
            // Cache rows: one per stream, then a scratch row for padding items. The decode's
            // predictor slots and encoder rows for streams follow the recordings'.
            let slots = stream_slots + 1;
            let mut add_chunks = |kind: rnnt::ChunkKind,
                                  rows: u32,
                                  max_items: u32|
             -> Result<(), Box<dyn std::error::Error>> {
                let spec = conformer_spec(rows)?;
                let staging_rows = 2 * max_items * rows;
                let chunk_rungs = rungs(max_items)
                    .into_iter()
                    .map(|items| -> Result<_, Box<dyn std::error::Error>> {
                        Ok(rnnt::ChunkRung {
                            items,
                            first: subsample(FACTOR * rows, items)?,
                            next: subsample(FACTOR * rows + HISTORY, items)?,
                            step: conformer::append_chunk_step(
                                spec,
                                &layers,
                                n_cu,
                                items,
                                rows,
                                slots,
                                staging_rows,
                                kind.inputs(),
                            )?,
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                packets.add_chunk_stream(
                    kind,
                    rnnt::ChunkGeometry {
                        rows,
                        left_rows: conformer::stream_left_rows(&spec),
                        slots,
                        first_input_frames: FACTOR * rows,
                        step_input_frames: FACTOR * rows + HISTORY,
                        history_input_frames: HISTORY,
                        step_keep_row0: HISTORY / FACTOR,
                        encoder_row0: decode_batch * frames,
                        predictor_slot0: decode_batch,
                    },
                    conformer::stream_init(spec, &layers, rows, n_cu, kind.inputs().position)?,
                    chunk_rungs,
                    &rnnt_weights,
                )?;
                Ok(())
            };
            add_chunks(rnnt::ChunkKind::Live, stream_rows, chunk_items)?;
            if upload_items != 0 {
                add_chunks(rnnt::ChunkKind::Upload, upload_rows, upload_items)?;
            }
        }
    }
    if decode_batch != 0 {
        let streams = if chunk_items != 0 { stream_slots } else { 0 };
        // The streams' encoder-table rows: the wider step of the two geometries.
        let step_rows = (chunk_items * stream_rows).max(upload_items * upload_rows);
        packets.add_batched_decode(
            &rnnt_weights,
            decode_batch + streams + 1,
            decode_batch * frames + step_rows,
            &rungs(decode_batch.max(chunk_items).max(upload_items)),
            &rungs((decode_batch * joint_batch).max(step_rows)),
        )?;
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
    packets.set_output(nemotron_packet_support::token_output(&gguf, Some(0))?)?;
    let section = packets.pipeline_section_with_frame_transform(
        13087,
        10,
        0,
        feature_frames,
        &frame_transform,
        trailing_encoder_frames,
    )?;
    devgen::install_sidecar_verifier(Box::new(|requests| {
        match lean_verify::call_batch_bound(requests) {
            Ok((certs, verifier)) => Ok(Some((
                certs
                    .iter()
                    .map(serde_json::to_value)
                    .collect::<Result<_, _>>()
                    .map_err(|e| e.to_string())?,
                verifier,
            ))),
            Err(e) if e.is_binary_unusable() => {
                eprintln!("lean checks skipped: {e}");
                Ok(None)
            }
            Err(e) => Err(e.to_string()),
        }
    }));
    // The packet plus `<stem>.lean-checks.json` (logical tensor effects per program).
    devgen::write_sidecar_packet(
        Path::new(&args[3]),
        &packets.model,
        &[section, packets.vocabulary_section()?],
    );
    println!(
        "{}",
        serde_json::json!({
            "driver": "rnnt.greedy.v1",
            "encoder_programs": packets.encoder_programs,
            "frames": frames,
            "feature_frames": feature_frames,
            "feature_frame_capacities": feature_frame_capacities,
            "joint_batch": joint_batch,
            "decode_batch": decode_batch,
            "stream_slots": if chunk_items != 0 { stream_slots } else { 0 },
            "stream_batch": chunk_items,
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
