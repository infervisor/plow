fn main() -> Result<(), Box<dyn std::error::Error>> {
    use std::path::Path;

    use devgen::rnnt::{self, RnntSpec, RnntWeights};
    use plowrt::asset::gguf::GgufFile;

    let args: Vec<_> = std::env::args().collect();
    if !(4..=5).contains(&args.len()) {
        return Err(
            "usage: asr_nemotron_rnnt_compile MODEL_GGUF FRAMES OUTPUT_PACKET [JOINT_BATCH]".into(),
        );
    }
    let frames = args[2].parse::<u32>()?;
    if frames == 0 {
        return Err("FRAMES must be positive".into());
    }
    let joint_batch = args.get(4).map_or(Ok(16), |value| value.parse::<u32>())?;
    let gguf = GgufFile::open(Path::new(&args[1]))?;
    let lstm = [lstm_weights(0), lstm_weights(1)];
    let weights = RnntWeights {
        prompt_in: linear("prompt_kernel.0"),
        prompt_out: linear("prompt_kernel.2"),
        encoder: linear("joint.enc"),
        embedding: "decoder.prediction.embed.weight",
        lstm: &lstm,
        predictor: linear("joint.pred"),
        output: linear("joint.joint_net.2"),
    };
    let mut packets = rnnt::lower(
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
        &weights,
        16,
    )?;
    packets.embed_weights(|name| {
        gguf.tensor(name)
            .map(|tensor| tensor.bytes.to_vec())
            .map_err(|error| error.to_string())
    })?;
    let section = packets.pipeline_section(13087, 10, 0)?;
    std::fs::write(&args[3], packets.model.to_blob_v6(&[section]))?;
    Ok(())
}

fn linear(prefix: &'static str) -> devgen::rnnt::LinearWeights<'static> {
    use devgen::rnnt::LinearWeights;
    match prefix {
        "prompt_kernel.0" => LinearWeights {
            weight: "prompt_kernel.0.weight",
            bias: "prompt_kernel.0.bias",
        },
        "prompt_kernel.2" => LinearWeights {
            weight: "prompt_kernel.2.weight",
            bias: "prompt_kernel.2.bias",
        },
        "joint.enc" => LinearWeights {
            weight: "joint.enc.weight",
            bias: "joint.enc.bias",
        },
        "joint.pred" => LinearWeights {
            weight: "joint.pred.weight",
            bias: "joint.pred.bias",
        },
        "joint.joint_net.2" => LinearWeights {
            weight: "joint.joint_net.2.weight",
            bias: "joint.joint_net.2.bias",
        },
        _ => unreachable!(),
    }
}

fn lstm_weights(layer: usize) -> devgen::rnnt::LstmWeights<'static> {
    use devgen::rnnt::{LinearWeights, LstmWeights};
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
