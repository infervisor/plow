//! Silero VAD v5 (16 kHz) lowered to a `vad.silero.v1` packet: one program per LSTM state bank,
//! each turning one window (the previous frame's last `CONTEXT` samples, then `FRAME` new ones)
//! into a speech probability. The host keeps the context samples and alternates the banks.

use packet::dev::{DevOp, ACT_RELU, ACT_SIGMOID, ACT_SQRT, TENSOR_NONE};
use packet::devbuild::{Builder, Model, SectionData, TensorDecl};

pub const DRIVER: &str = "vad.silero.v1";
pub const SAMPLE_RATE: u32 = 16_000;
pub const FRAME: u32 = 512;
pub const CONTEXT: u32 = 64;
const WINDOW: u32 = CONTEXT + FRAME;
const FFT: u32 = 256;
const HOP: u32 = 128;
const BINS: u32 = FFT / 2 + 1;
const STFT_ROWS: u32 = (WINDOW + CONTEXT - FFT) / HOP + 1;
const HIDDEN: u32 = 128;
/// `(in, out, stride)`; kernel 3, padding 1, ReLU.
const ENCODER: [(u32, u32, u32); 4] = [(BINS, 128, 1), (128, 64, 2), (64, 64, 2), (64, HIDDEN, 1)];

pub const INPUT: &str = "in.vad.window";
pub const PROBABILITY: &str = "act.vad.probability";

pub struct VadPackets {
    pub model: Model,
    steps: [usize; 2],
}

pub fn lower(n_cu: u32) -> Result<VadPackets, String> {
    if n_cu == 0 {
        return Err("VAD packet needs at least one CU".into());
    }
    let mut tensors: Vec<TensorDecl> = Vec::new();
    let mut progs = Vec::new();
    for bank in 0..2 {
        let mut b = Builder::new(n_cu);
        b.set_tensor_dedup(true);
        b.adopt_tensors(tensors);
        emit_step(&mut b, bank);
        let program = b.finish();
        tensors = program.tensors.clone();
        progs.push(program);
    }
    for t in tensors.iter_mut().filter(|t| t.name.starts_with("state.vad.")) {
        t.init = Some(vec![0; t.bytes as usize]);
    }
    Ok(VadPackets {
        model: Model { n_cu, target: 0, tensors, progs, prog_t: vec![1, 1], kv_row_insts: Vec::new(), gen: Vec::new() },
        steps: [0, 1],
    })
}

fn emit_step(b: &mut Builder, source: usize) {
    let target = source ^ 1;
    let f32s = |n: u32| u64::from(n) * 4;
    let window = b.tensor(INPUT, f32s(WINDOW));
    let basis = b.tensor("vad.stft.basis", f32s(2 * BINS * FFT));
    let stft = b.tensor("act.vad.stft", f32s(STFT_ROWS * 2 * BINS));
    let power = b.tensor("act.vad.power", f32s(STFT_ROWS * 2 * BINS));
    let real = b.tensor("act.vad.magnitude", f32s(STFT_ROWS * BINS));
    let imaginary = b.tensor("act.vad.imaginary", f32s(STFT_ROWS * BINS));

    // Silero's STFT: reflect-pad CONTEXT samples on the right, a strided conv with the
    // [real; imaginary] Fourier basis, then the magnitude of each bin.
    let mut dep = b.emit(DevOp::Conv1dF32, b.all(), &[], |d| {
        d.t[..4].copy_from_slice(&[stft, window, basis, TENSOR_NONE]);
        d.i = [1, WINDOW, 1, 2 * BINS, FFT, HOP, 1, 1];
        d.j = [CONTEXT << 16, 1];
    });
    dep = b.emit(DevOp::BinaryF32, b.all(), &[dep], |d| {
        d.t[..3].copy_from_slice(&[power, stft, stft]);
        d.i = [1, STFT_ROWS, 2 * BINS, 2, 0, 2 * BINS, 1, 0];
    });
    let split: Vec<u32> = [(real, 0), (imaginary, BINS)]
        .into_iter()
        .map(|(out, offset)| {
            b.emit(DevOp::CopyColsF32, b.all(), &[dep], |d| {
                d.t[..2].copy_from_slice(&[out, power]);
                d.i = [1, STFT_ROWS, BINS, 2 * BINS, offset, BINS, 0, 0];
            })
        })
        .collect();
    dep = b.emit(DevOp::BinaryF32, b.all(), &split, |d| {
        d.t[..3].copy_from_slice(&[real, real, imaginary]);
        d.i = [1, STFT_ROWS, BINS, 0, 0, BINS, 1, 0];
    });
    dep = b.emit(DevOp::UnaryF32, b.all(), &[dep], |d| {
        d.t[..2].copy_from_slice(&[real, real]);
        d.i[..3].copy_from_slice(&[STFT_ROWS, BINS, ACT_SQRT]);
    });

    let (mut input, mut rows) = (real, STFT_ROWS);
    for (layer, (cin, cout, stride)) in ENCODER.into_iter().enumerate() {
        let out_rows = (rows + 2 - 3) / stride + 1;
        let output = b.tensor(&format!("act.vad.encoder.{layer}"), f32s(out_rows * cout));
        let weight = b.tensor(&format!("vad.encoder.{layer}.weight"), f32s(cout * cin * 3));
        let bias = b.tensor(&format!("vad.encoder.{layer}.bias"), f32s(cout));
        dep = b.emit(DevOp::Conv1dF32, b.all(), &[dep], |d| {
            d.t[..4].copy_from_slice(&[output, input, weight, bias]);
            d.i = [1, rows, cin, cout, 3, stride, 1, 1];
            d.j = [1 | 1 << 16, ACT_RELU << 8];
        });
        (input, rows) = (output, out_rows);
    }

    let state = |b: &mut Builder, kind: &str, bank: usize| b.tensor(&format!("state.vad.{kind}.{bank}"), f32s(HIDDEN));
    let (h_source, c_source) = (state(b, "h", source), state(b, "c", source));
    let (h_target, c_target) = (state(b, "h", target), state(b, "c", target));
    let gates = b.tensor("act.vad.gates", f32s(4 * HIDDEN));
    let recurrent = b.tensor("act.vad.recurrent", f32s(4 * HIDDEN));
    let projections: Vec<u32> = [(gates, input, "ih"), (recurrent, h_source, "hh")]
        .into_iter()
        .map(|(out, x, kind)| {
            let weight = b.tensor(&format!("vad.lstm.weight_{kind}"), f32s(4 * HIDDEN * HIDDEN));
            let bias = b.tensor(&format!("vad.lstm.bias_{kind}"), f32s(4 * HIDDEN));
            b.emit(DevOp::DenseGemmF32, b.all(), &[dep], |d| {
                d.t[..4].copy_from_slice(&[out, x, weight, bias]);
                d.i[..4].copy_from_slice(&[1, 4 * HIDDEN, HIDDEN, 0]);
            })
        })
        .collect();
    dep = b.emit(DevOp::ScaledAddF32, b.all(), &projections, |d| {
        d.t[..3].copy_from_slice(&[gates, gates, recurrent]);
        d.i[0] = 4 * HIDDEN;
        d.f[0] = 1.0;
    });
    dep = b.emit(DevOp::LstmCellF32, b.all(), &[dep], |d| {
        d.t[..4].copy_from_slice(&[h_target, c_target, gates, c_source]);
        d.i[0] = HIDDEN;
    });

    let activated = b.tensor("act.vad.decoder", f32s(HIDDEN));
    let probability = b.tensor(PROBABILITY, 4);
    let weight = b.tensor("vad.out.weight", f32s(HIDDEN));
    let bias = b.tensor("vad.out.bias", 4);
    dep = b.emit(DevOp::UnaryF32, b.all(), &[dep], |d| {
        d.t[..2].copy_from_slice(&[activated, h_target]);
        d.i[..3].copy_from_slice(&[1, HIDDEN, ACT_RELU]);
    });
    dep = b.emit(DevOp::DenseGemmF32, b.all(), &[dep], |d| {
        d.t[..4].copy_from_slice(&[probability, activated, weight, bias]);
        d.i[..4].copy_from_slice(&[1, 1, HIDDEN, 0]);
    });
    b.emit(DevOp::UnaryF32, b.all(), &[dep], |d| {
        d.t[..2].copy_from_slice(&[probability, probability]);
        d.i[..3].copy_from_slice(&[1, 1, ACT_SIGMOID]);
    });
}

impl VadPackets {
    pub fn embed_weights(&mut self, mut resolve: impl FnMut(&str) -> Result<Vec<u8>, String>) -> Result<(), String> {
        for tensor in &mut self.model.tensors {
            if packet::names::is_checkpoint_weight(&tensor.name) && tensor.init.is_none() {
                let bytes = resolve(&tensor.name)?;
                if bytes.len() as u64 != tensor.bytes {
                    return Err(format!("tensor {} has {} bytes, expected {}", tensor.name, bytes.len(), tensor.bytes));
                }
                tensor.init = Some(bytes);
            }
        }
        Ok(())
    }

    pub fn pipeline_section(&self) -> Result<SectionData, String> {
        use plow_asset::packet_pipeline::{PacketPipeline, PacketPipelines, PipelineDType, PipelineTensor, SECTION, VERSION};
        use std::collections::BTreeMap;

        let f32_tensor = |name: &str, elements: u64| PipelineTensor { name: name.into(), dtype: PipelineDType::F32, shape: vec![elements] };
        let mut tensors = BTreeMap::from([
            ("input".to_string(), f32_tensor(INPUT, u64::from(WINDOW))),
            ("probability".to_string(), f32_tensor(PROBABILITY, 1)),
        ]);
        for (index, state) in self.model.tensors.iter().filter(|t| t.name.starts_with("state.vad.")).enumerate() {
            tensors.insert(format!("state.{index}"), f32_tensor(&state.name, state.bytes / 4));
        }
        let metadata = PacketPipelines {
            version: VERSION,
            pipelines: vec![PacketPipeline {
                name: "vad".into(),
                driver: DRIVER.into(),
                programs: BTreeMap::from([("step.0".into(), self.steps[0] as u32), ("step.1".into(), self.steps[1] as u32)]),
                tensors,
                parameters: BTreeMap::from([
                    ("sample_rate".into(), u64::from(SAMPLE_RATE)),
                    ("frame_samples".into(), u64::from(FRAME)),
                    ("context_samples".into(), u64::from(CONTEXT)),
                ]),
                strings: Default::default(),
            }],
        };
        metadata.validate(self.model.progs.len(), |name| {
            self.model.tensors.iter().find(|t| t.name == name).map(|t| t.bytes)
        })?;
        Ok(SectionData {
            kind: packet::devbuild::SECT_METADATA,
            name: SECTION.into(),
            data: serde_json::to_vec(&metadata).map_err(|error| error.to_string())?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn silero_lowers_to_two_bank_programs_of_generic_ops() {
        let packets = lower(1).unwrap();
        assert_eq!(packets.model.progs.len(), 2);
        let ops: Vec<u16> = packets.model.progs[0].insts.iter().map(|i| i.op).collect();
        assert_eq!(ops.len(), 17);
        assert!(ops.iter().all(|&op| [
            DevOp::Conv1dF32,
            DevOp::BinaryF32,
            DevOp::CopyColsF32,
            DevOp::UnaryF32,
            DevOp::DenseGemmF32,
            DevOp::ScaledAddF32,
            DevOp::LstmCellF32,
        ]
        .iter()
        .any(|&allowed| allowed as u16 == op)));
        let states: Vec<_> = packets.model.tensors.iter().filter(|t| t.name.starts_with("state.vad.")).collect();
        assert_eq!(states.len(), 4);
        assert!(states.iter().all(|t| t.init.as_deref() == Some(&[0u8; 512][..])));
        assert!(packets.pipeline_section().is_ok());
    }
}
