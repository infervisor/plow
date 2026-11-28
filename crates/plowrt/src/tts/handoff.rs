use std::collections::BTreeMap;

use plow_asset::packet_pipeline::{HandoffReadback, PipelineDType};

use crate::exec::gpu::handoff::{CudaHandoff, HandoffBind, HandoffRun, HandoffSelection, RowKey};
use crate::exec::packet_runtime::PacketBuffer;
use crate::serve::stream::{SpeechFrame, SpeechPcm};
use crate::{Result, RuntimeError};

fn reject(message: &str) -> RuntimeError {
    RuntimeError::Rejected(format!("speech handoff: {message}"))
}

#[derive(Clone, Copy)]
pub(crate) struct SemanticFrame {
    pub base: u32,
    pub stop_class: u32,
    pub stop_token: u32,
}

pub(crate) struct SpeechReadback {
    extent: String,
    stride: usize,
    stop: usize,
    cb0: usize,
    pcm: usize,
    samples: usize,
}

impl SpeechReadback {
    pub(crate) fn bind(
        readback: &HandoffReadback,
        stop: &str,
        cb0: &str,
        pcm: &str,
        samples: usize,
    ) -> Result<Self> {
        if stop == cb0
            || stop == pcm
            || cb0 == pcm
            || samples == 0
            || readback.sample_lengths.is_some()
        {
            return Err(reject(
                "unsupported speech readback fields or variable frame length",
            ));
        }
        let field = |name: &str, dtype, elements| {
            let mut matches = readback.fields.iter().filter(|f| f.name == name);
            let f = matches
                .next()
                .ok_or_else(|| reject("missing speech field"))?;
            if matches.next().is_some()
                || f.dtype != dtype
                || f.elements as usize != elements
                || f.offset % dtype.bytes() != 0
                || f.offset
                    .checked_add(elements as u64 * dtype.bytes())
                    .is_none_or(|end| end > readback.row_stride)
            {
                return Err(reject("invalid typed speech field"));
            }
            usize::try_from(f.offset).map_err(|_| reject("field offset overflow"))
        };
        let stride =
            usize::try_from(readback.row_stride).map_err(|_| reject("row stride overflow"))?;
        if stride == 0 || stride % 4 != 0 {
            return Err(reject("unaligned speech row stride"));
        }
        let stop = field(stop, PipelineDType::U32, 1)?;
        let cb0 = field(cb0, PipelineDType::U32, 1)?;
        let pcm = field(pcm, PipelineDType::I16, samples)?;
        let ranges = [(stop, stop + 4), (cb0, cb0 + 4), (pcm, pcm + samples * 2)];
        for (i, &(a, b)) in ranges.iter().enumerate() {
            if ranges[..i].iter().any(|&(c, d)| a < d && c < b) {
                return Err(reject("overlapping speech readback fields"));
            }
        }
        Ok(Self {
            extent: readback.extent.clone(),
            stride,
            stop,
            cb0,
            pcm,
            samples,
        })
    }

    pub(crate) fn frames(
        &self,
        buffer: &PacketBuffer,
        rows: &[SemanticFrame],
    ) -> Result<Vec<(u32, Option<SpeechFrame>)>> {
        let data = buffer
            .host_slice()
            .ok_or_else(|| reject("completion is not a retired host slab"))?;
        if rows.len().checked_mul(self.stride) != Some(data.len()) {
            return Err(reject("live slab length differs from row set"));
        }
        let mut frames = Vec::with_capacity(rows.len());
        for (row, semantic) in rows.iter().enumerate() {
            let start = row * self.stride;
            let word = |offset| {
                u32::from_le_bytes(data[start + offset..start + offset + 4].try_into().unwrap())
            };
            match word(self.stop) {
                1 => frames.push((semantic.stop_token, None)),
                0 => {
                    let cb0 = word(self.cb0);
                    if cb0 >= semantic.stop_class {
                        return Err(reject("completed semantic code exceeds its codebook"));
                    }
                    let token = semantic
                        .base
                        .checked_add(cb0)
                        .ok_or_else(|| reject("semantic token overflow"))?;
                    let pcm = SpeechPcm::from_buffer(
                        buffer.clone(),
                        start + self.pcm..start + self.pcm + self.samples * 2,
                    )?;
                    frames.push((token, Some(SpeechFrame::Pcm(pcm))));
                }
                _ => return Err(reject("invalid completed stop flag")),
            }
        }
        Ok(frames)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use plow_asset::packet_pipeline::{HandoffProgramRef, HandoffReadbackField};

    fn layout() -> HandoffReadback {
        HandoffReadback {
            producer: HandoffProgramRef {
                component: "a".into(),
                program: "p".into(),
            },
            view: "out".into(),
            extent: "live".into(),
            row_stride: 16,
            fields: [
                ("stop", 0, PipelineDType::U32, 1),
                ("cb0", 4, PipelineDType::U32, 1),
                ("pcm", 8, PipelineDType::I16, 4),
            ]
            .into_iter()
            .map(|(name, offset, dtype, elements)| HandoffReadbackField {
                name: name.into(),
                offset,
                dtype,
                elements,
            })
            .collect(),
            sample_lengths: None,
        }
    }
    fn semantic() -> SemanticFrame {
        SemanticFrame {
            base: 100,
            stop_class: 10,
            stop_token: 99,
        }
    }

    #[test]
    fn speech_handoff_decodes_stops_and_retains_pcm_without_copying_codes_or_samples() {
        let decoder = SpeechReadback::bind(&layout(), "stop", "cb0", "pcm", 4).unwrap();
        let mut bytes = Vec::new();
        for (stop, cb0) in [(0u32, 3u32), (1, u32::MAX)] {
            bytes.extend(stop.to_le_bytes());
            bytes.extend(cb0.to_le_bytes());
            bytes.extend([-3i16, 4, 5, -6].into_iter().flat_map(i16::to_le_bytes));
        }
        let buffer: PacketBuffer = bytes.into();
        let pointer = buffer.host_slice().unwrap().as_ptr().wrapping_add(8);
        let frames = decoder.frames(&buffer, &[semantic(), semantic()]).unwrap();
        assert_eq!(frames[0].0, 103);
        assert_eq!(frames[1].0, 99);
        assert!(frames[1].1.is_none());
        let frame = frames[0].1.as_ref().unwrap();
        assert!(matches!(frame, SpeechFrame::Pcm(_)));
        assert_eq!(frame.pcm().unwrap().as_ptr().cast::<u8>(), pointer);
        drop(buffer);
        assert_eq!(&**frame.pcm().unwrap(), &[-3, 4, 5, -6]);
    }

    #[test]
    fn speech_handoff_rejects_bad_fields_flags_tokens_and_slab_lengths() {
        for mutation in 0..4 {
            let mut layout = layout();
            match mutation {
                0 => layout.fields[1].offset = 0,
                1 => layout.fields[2].dtype = PipelineDType::F32,
                2 => layout.fields[2].elements = 3,
                _ => layout.row_stride = 15,
            }
            assert!(SpeechReadback::bind(&layout, "stop", "cb0", "pcm", 4).is_err());
        }
        let decoder = SpeechReadback::bind(&layout(), "stop", "cb0", "pcm", 4).unwrap();
        assert!(decoder.frames(&vec![0; 15].into(), &[semantic()]).is_err());
        for (stop, cb0, base) in [(2u32, 3u32, 100), (0, 10, 100), (0, 3, u32::MAX)] {
            let mut bytes = vec![0; 16];
            bytes[..4].copy_from_slice(&stop.to_le_bytes());
            bytes[4..8].copy_from_slice(&cb0.to_le_bytes());
            assert!(decoder
                .frames(&bytes.into(), &[SemanticFrame { base, ..semantic() }])
                .is_err());
        }
    }
}

pub(crate) enum SpeechHandoffRun {
    Complete(Vec<(u32, Option<SpeechFrame>)>),
    DeliveryBusy,
}

pub(crate) struct SpeechHandoff {
    executor: CudaHandoff,
    readback: SpeechReadback,
}

impl SpeechHandoff {
    pub(crate) fn new(
        executor: CudaHandoff,
        stop: &str,
        cb0: &str,
        pcm: &str,
        samples: usize,
    ) -> Result<Self> {
        let readback = SpeechReadback::bind(executor.readback(), stop, cb0, pcm, samples)?;
        Ok(Self { executor, readback })
    }

    pub(crate) fn run(
        &mut self,
        rows: &[SemanticFrame],
        live: &BTreeMap<String, u32>,
        fields: &BTreeMap<String, u64>,
        binds: &[HandoffBind<'_>],
        keys: &[&RowKey],
        maps: &[HandoffSelection<'_>],
    ) -> Result<SpeechHandoffRun> {
        if live.get(&self.readback.extent).map(|&n| n as usize) != Some(rows.len()) {
            return Err(reject("speech row set differs from declared live extent"));
        }
        match self.executor.run_mapped(live, fields, binds, keys, maps)? {
            HandoffRun::Complete(buffer) => self
                .readback
                .frames(&buffer, rows)
                .map(SpeechHandoffRun::Complete),
            HandoffRun::DeliveryBusy => Ok(SpeechHandoffRun::DeliveryBusy),
        }
    }
}
