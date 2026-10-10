//! A multimodal encoder sidecar (`forward.v1` pipeline `mm.encode`): stages one launch's items
//! into the packet's inputs, runs the smallest rung that holds them, and reads the projected rows.

use super::media::{Mel, Patches};
use crate::exec::packet_runtime::{ForwardPacket, PacketTensor};
use crate::{Result, RuntimeError};

fn bad(what: impl Into<String>) -> RuntimeError {
    RuntimeError::Rejected(what.into())
}

fn f32_param(packet: &ForwardPacket, key: &str) -> Option<f32> {
    packet.optional_parameter(key).and_then(|v| u32::try_from(v).ok()).map(f32::from_bits)
}

fn round_bf16(v: f32) -> f32 {
    let bits = v.to_bits();
    let rounded = bits.wrapping_add(0x7FFF + ((bits >> 16) & 1)) & 0xFFFF_0000;
    f32::from_bits(rounded)
}

enum Inputs {
    Vision {
        item_rows: usize,
        item_tokens: usize,
        patch_values: usize,
        pool: usize,
        scale: f32,
        shift: f32,
        posx: PacketTensor,
        posy: PacketTensor,
        /// Tower inputs; an encoder-free embedder (one row per soft token) has none of them.
        rope: Option<PacketTensor>,
        valid: Option<PacketTensor>,
        pools: Vec<PacketTensor>,
    },
    Audio {
        mel_bins: usize,
        mask1: PacketTensor,
        valid: PacketTensor,
    },
    /// Encoder-free audio: rungs count tokens, each a row of `frame_samples` raw samples.
    Frames {
        frame_samples: usize,
    },
}

pub struct Encoder {
    packet: ForwardPacket,
    /// Rungs ascending: (images per launch | mel frames, program sequence).
    rungs: Vec<(u32, Vec<usize>)>,
    input: PacketTensor,
    output: PacketTensor,
    output_width: usize,
    round_input: bool,
    inputs: Inputs,
}

impl Encoder {
    pub fn load(path: &std::path::Path, device: u8) -> Result<Self> {
        let packet = ForwardPacket::load_on(path, "mm.encode", "cuda", device)?;
        let pipeline = packet.pipeline();
        let rungs = pipeline.program_capacity_sequences("forward")?;
        if rungs.is_empty() {
            return Err(bad(format!("{}: encoder has no rungs", path.display())));
        }
        let input = pipeline.tensor("input")?;
        let output = pipeline.tensor("output")?;
        let param = |k: &str| -> Result<usize> { Ok(packet.parameter(k)? as usize) };
        let output_width = param("output_width")?;
        let round_input = packet.optional_parameter("input.round_bf16") == Some(1);
        let inputs = match pipeline.optional_string("modality") {
            Some("vision") => {
                let pool = param("pool")?;
                Inputs::Vision {
                    item_rows: param("item_rows")?,
                    item_tokens: param("item_tokens")?,
                    patch_values: param("patch_values")?,
                    pool,
                    scale: f32_param(&packet, "input.scale_f32").unwrap_or(1.0),
                    shift: f32_param(&packet, "input.shift_f32").unwrap_or(0.0),
                    posx: pipeline.tensor("posx")?,
                    posy: pipeline.tensor("posy")?,
                    rope: pipeline.optional_tensor("rope"),
                    valid: pipeline.optional_tensor("valid"),
                    pools: (0..pool * pool)
                        .map_while(|j| pipeline.optional_tensor(&format!("pool.{j}")))
                        .collect(),
                }
            }
            Some("audio") => Inputs::Audio {
                mel_bins: param("mel_bins")?,
                mask1: pipeline.tensor("mask1")?,
                valid: pipeline.tensor("valid")?,
            },
            Some("frames") => Inputs::Frames { frame_samples: param("frame_samples")? },
            other => return Err(bad(format!("{}: unknown encoder modality {other:?}", path.display()))),
        };
        Ok(Self { packet, rungs, input, output, output_width, round_input, inputs })
    }

    /// Items one launch holds (vision: images; audio: always 1).
    pub fn max_items(&self) -> usize {
        match self.inputs {
            Inputs::Vision { .. } => self.rungs.last().map_or(1, |r| r.0 as usize),
            Inputs::Audio { .. } | Inputs::Frames { .. } => 1,
        }
    }

    /// Largest audio clip in log-mel frames.
    pub fn max_frames(&self) -> usize {
        self.rungs.last().map_or(0, |r| r.0 as usize)
    }

    fn run(&mut self, programs: &[usize], writes: &[(PacketTensor, Vec<u8>)], rows: usize) -> Result<Vec<f32>> {
        let mut out = vec![0f32; rows * self.output_width];
        let runtime = self.packet.runtime_mut();
        runtime.begin_execution()?;
        let result = (|| {
            for (tensor, bytes) in writes {
                runtime.write_tensor_at(*tensor, 0, bytes)?;
            }
            runtime.run_sequence(programs)?;
            runtime.read_tensor_at(self.output, 0, bytemuck::cast_slice_mut(&mut out))
        })();
        let ended = runtime.end_execution();
        result.and(ended)?;
        Ok(out)
    }

    /// Project images: one `[soft_tokens][output_width]` block per item, in order.
    pub fn encode_images(&mut self, items: &[&Patches]) -> Result<Vec<Vec<f32>>> {
        let Inputs::Vision { item_rows, item_tokens, patch_values, pool, scale, shift, posx, posy, rope, valid, ref pools } =
            self.inputs
        else {
            return Err(bad("not a vision encoder"));
        };
        let pools = pools.clone();
        let mut out = Vec::with_capacity(items.len());
        let max = self.max_items();
        for chunk in items.chunks(max) {
            let (cap, programs) = self
                .rungs
                .iter()
                .find(|(c, _)| *c as usize >= chunk.len())
                .map(|(c, p)| (*c as usize, p.clone()))
                .ok_or_else(|| bad("no vision rung holds the images"))?;
            let rows = cap * item_rows;
            let mut px = vec![0f32; rows * patch_values];
            let mut xs = vec![u32::MAX; rows];
            let mut ys = vec![u32::MAX; rows];
            let mut rp = vec![0u32; rows * 2];
            let mut counts = vec![0u32; cap];
            let k2 = pool * pool;
            let mut taps = vec![vec![u32::MAX; cap * item_tokens]; k2];
            for (i, p) in chunk.iter().enumerate() {
                let n = p.positions.len();
                if n > item_rows || p.values.len() != n * patch_values || p.soft_tokens as usize > item_tokens {
                    return Err(bad("image patches exceed the vision rung"));
                }
                counts[i] = n as u32;
                let base = i * item_rows;
                for (k, v) in p.values.iter().enumerate() {
                    let t = v * scale + shift;
                    px[base * patch_values + k] = if self.round_input { round_bf16(t) } else { t };
                }
                let pw = p.grid.1 as usize;
                let mut filled = vec![0usize; item_tokens];
                for (k, &[x, y]) in p.positions.iter().enumerate() {
                    xs[base + k] = x;
                    ys[base + k] = y;
                    rp[(base + k) * 2] = x;
                    rp[(base + k) * 2 + 1] = y;
                    let o = (x as usize / pool) + (pw / pool) * (y as usize / pool);
                    if o < item_tokens {
                        let j = filled[o];
                        if j < k2 {
                            taps[j][i * item_tokens + o] = (base + k) as u32;
                            filled[o] += 1;
                        }
                    }
                }
            }
            let mut writes = vec![
                (self.input, bytemuck::cast_slice(&px).to_vec()),
                (posx, bytemuck::cast_slice(&xs).to_vec()),
                (posy, bytemuck::cast_slice(&ys).to_vec()),
            ];
            writes.extend(rope.map(|t| (t, bytemuck::cast_slice(&rp).to_vec())));
            writes.extend(valid.map(|t| (t, bytemuck::cast_slice(&counts).to_vec())));
            for (j, t) in pools.iter().enumerate() {
                writes.push((*t, bytemuck::cast_slice(&taps[j]).to_vec()));
            }
            let rows_out = cap * item_tokens;
            let flat = self.run(&programs, &writes, rows_out)?;
            for (i, p) in chunk.iter().enumerate() {
                let start = i * item_tokens * self.output_width;
                out.push(flat[start..start + p.soft_tokens as usize * self.output_width].to_vec());
            }
        }
        Ok(out)
    }

    /// Project one audio clip: `[tokens][output_width]`.
    pub fn encode_audio(&mut self, mel: &Mel, tokens: usize) -> Result<Vec<f32>> {
        if let Inputs::Frames { frame_samples } = self.inputs {
            return self.encode_frames(mel, frame_samples);
        }
        let Inputs::Audio { mel_bins, mask1, valid } = self.inputs else {
            return Err(bad("not an audio encoder"));
        };
        let need = mel.valid_frames.next_multiple_of(4).max(4);
        let (cap, programs) = self
            .rungs
            .iter()
            .find(|(c, _)| *c as usize >= need)
            .map(|(c, p)| (*c as usize, p.clone()))
            .ok_or_else(|| bad("audio clip is longer than the largest audio rung"))?;
        let mut input = vec![0f32; cap * mel_bins];
        for f in 0..mel.valid_frames.min(cap) {
            for b in 0..mel_bins {
                let v = mel.values[f * mel_bins + b];
                input[f * mel_bins + b] = if self.round_input { round_bf16(v) } else { v };
            }
        }
        let t1 = mel.valid_frames.div_ceil(2);
        let mask: Vec<f32> = (0..cap / 2).map(|t| if t < t1 { 1.0 } else { 0.0 }).collect();
        let writes = vec![
            (self.input, bytemuck::cast_slice(&input).to_vec()),
            (mask1, bytemuck::cast_slice(&mask).to_vec()),
            (valid, (tokens as u32).to_le_bytes().to_vec()),
        ];
        let rows = cap / 4;
        let mut flat = self.run(&programs, &writes, rows)?;
        flat.truncate(tokens * self.output_width);
        Ok(flat)
    }

    fn encode_frames(&mut self, frames: &Mel, frame_samples: usize) -> Result<Vec<f32>> {
        let tokens = frames.valid_frames;
        let (cap, programs) = self
            .rungs
            .iter()
            .find(|(c, _)| *c as usize >= tokens)
            .map(|(c, p)| (*c as usize, p.clone()))
            .ok_or_else(|| bad("audio clip is longer than the largest audio rung"))?;
        let mut input = vec![0f32; cap * frame_samples];
        for (dst, &v) in input.iter_mut().zip(&frames.values[..tokens * frame_samples]) {
            *dst = if self.round_input { round_bf16(v) } else { v };
        }
        let mut flat = self.run(&programs, &[(self.input, bytemuck::cast_slice(&input).to_vec())], cap)?;
        flat.truncate(tokens * self.output_width);
        Ok(flat)
    }
}
