use packet::dev::{
    DevInst64, DevOp, StreamEnt, Wait, SE_FINE, SE_KDA_INTRA_WAVE_ITEMS, SE_XCTR, SE_XR_WAVE_RS,
    TENSOR_NONE16, TENSOR_NONE_I,
};
use packet::slots::{slots_for, Provenance};

use crate::program;

type Result<T> = std::result::Result<T, String>;

pub const MAGIC: &[u8; 8] = b"PLOWPRG1";
pub const CAPABILITY: &str = "plow.dev-program";
pub const VERSION: u32 = 1;

const SECTION_HEADER_BYTES: usize = 16;
const PROGRAM_HEADER_BYTES: usize = 40;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Program {
    pub rows: u32,
    pub n_counter: u32,
    pub insts: Vec<DevInst64>,
    pub stream: Vec<StreamEnt>,
    pub stream_ofs: Vec<u32>,
    pub stream_len: Vec<u32>,
    pub waits: Vec<Wait>,
    pub succs: Vec<u32>,
    pub gq_stream: Vec<StreamEnt>,
    pub gq_seg_ofs: Vec<u32>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Section {
    pub n_cu: u32,
    pub programs: Vec<Program>,
}

#[derive(Clone, Copy)]
struct Counts {
    rows: u32,
    n_counter: u32,
    n_inst: usize,
    n_stream: usize,
    n_wait: usize,
    n_succ: usize,
    n_gq_seg: usize,
}

pub fn encode(
    n_cu: u32,
    tensor_count: usize,
    programs: &[program::Program<'_>],
) -> Result<Vec<u8>> {
    require(
        programs
            .iter()
            .all(|p| !p.packed_prefill_only && p.l2_domains == 0),
        "unsupported program placement",
    )?;
    let owned = Section {
        n_cu,
        programs: programs
            .iter()
            .map(|p| Program {
                rows: p.rows,
                n_counter: p.n_counter,
                insts: p.insts.to_vec(),
                stream: p.stream.to_vec(),
                stream_ofs: p.stream_ofs.to_vec(),
                stream_len: p.stream_len.to_vec(),
                waits: p.waits.to_vec(),
                succs: p.succs.to_vec(),
                gq_stream: p.gq_stream.to_vec(),
                gq_seg_ofs: p.gq_seg_ofs.to_vec(),
            })
            .collect(),
    };
    owned.validate(tensor_count)?;

    let mut out = Vec::new();
    out.extend_from_slice(MAGIC);
    put_u32(&mut out, n_cu);
    put_u32(&mut out, u32_len(programs.len(), "program count")?);
    for p in &owned.programs {
        put_u32(&mut out, p.rows);
        put_u32(&mut out, p.n_counter);
        put_u32(&mut out, 0);
        put_u32(&mut out, 0);
        put_u32(&mut out, u32_len(p.insts.len(), "instruction count")?);
        put_u32(&mut out, u32_len(p.stream.len(), "stream count")?);
        put_u32(&mut out, u32_len(p.waits.len(), "wait count")?);
        put_u32(&mut out, u32_len(p.succs.len(), "successor count")?);
        put_u32(
            &mut out,
            u32_len(p.gq_seg_ofs.len() - 1, "queue window count")?,
        );
        put_u32(&mut out, 0);
        for inst in &p.insts {
            put_inst(&mut out, inst);
        }
        for entry in &p.stream {
            put_stream(&mut out, entry);
        }
        put_u32s(&mut out, &p.stream_ofs);
        put_u32s(&mut out, &p.stream_len);
        for wait in &p.waits {
            put_u32(&mut out, wait.id);
            put_u32(&mut out, wait.threshold);
        }
        put_u32s(&mut out, &p.succs);
        for entry in &p.gq_stream {
            put_stream(&mut out, entry);
        }
        put_u32s(&mut out, &p.gq_seg_ofs);
    }
    Ok(out)
}

pub fn parse(bytes: &[u8], expected_n_cu: u32, tensor_count: usize) -> Result<Section> {
    let (n_cu, n_programs) = preflight(bytes, expected_n_cu)?;
    let mut reader = Reader::new(bytes);
    reader.take(MAGIC.len(), "magic")?;
    reader.u32("CU count")?;
    reader.u32("program count")?;

    let mut programs = Vec::with_capacity(n_programs);
    for index in 0..n_programs {
        let counts = read_header(&mut reader, index)?;
        let insts = (0..counts.n_inst)
            .map(|_| reader.inst(index))
            .collect::<Result<Vec<_>>>()?;
        let stream = (0..counts.n_stream)
            .map(|_| reader.stream(index, "stream"))
            .collect::<Result<Vec<_>>>()?;
        let stream_ofs = reader.u32s(n_cu as usize, index, "stream offsets")?;
        let stream_len = reader.u32s(n_cu as usize, index, "stream lengths")?;
        let waits = (0..counts.n_wait)
            .map(|_| {
                Ok(Wait {
                    id: reader.u32(&format!("program {index} wait id"))?,
                    threshold: reader.u32(&format!("program {index} wait threshold"))?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let succs = reader.u32s(counts.n_succ, index, "successors")?;
        let gq_stream = (0..counts.n_stream)
            .map(|_| reader.stream(index, "global queue"))
            .collect::<Result<Vec<_>>>()?;
        let gq_seg_ofs = reader.u32s(counts.n_gq_seg + 1, index, "queue windows")?;
        programs.push(Program {
            rows: counts.rows,
            n_counter: counts.n_counter,
            insts,
            stream,
            stream_ofs,
            stream_len,
            waits,
            succs,
            gq_stream,
            gq_seg_ofs,
        });
    }
    require(reader.offset == bytes.len(), "trailing payload bytes")?;
    let section = Section { n_cu, programs };
    section.validate(tensor_count)?;
    Ok(section)
}

fn preflight(bytes: &[u8], expected_n_cu: u32) -> Result<(u32, usize)> {
    require(
        bytes.len() >= SECTION_HEADER_BYTES,
        "truncated section header",
    )?;
    require(&bytes[..MAGIC.len()] == MAGIC, "unsupported payload magic")?;
    let n_cu = get_u32(bytes, 8, "CU count")?;
    let n_programs = usize::try_from(get_u32(bytes, 12, "program count")?)
        .map_err(|_| "aux program: program count overflow".to_string())?;
    require(n_cu > 0 && n_cu == expected_n_cu, "CU/grid mismatch")?;
    require(
        n_programs > 0
            && n_programs
                <= bytes.len().saturating_sub(SECTION_HEADER_BYTES) / PROGRAM_HEADER_BYTES,
        "program count exceeds payload",
    )?;

    let mut offset = SECTION_HEADER_BYTES;
    for index in 0..n_programs {
        let header_end = checked_end(offset, PROGRAM_HEADER_BYTES, bytes.len(), "program header")?;
        let counts = header_at(bytes, offset, index)?;
        let payload_bytes = program_payload_bytes(n_cu as usize, counts)?;
        offset = checked_end(header_end, payload_bytes, bytes.len(), "program payload")?;
    }
    require(offset == bytes.len(), "trailing payload bytes")?;
    Ok((n_cu, n_programs))
}

impl Section {
    fn validate(&self, tensor_count: usize) -> Result<()> {
        require(
            self.n_cu > 0 && tensor_count <= TENSOR_NONE16 as usize && !self.programs.is_empty(),
            "section geometry",
        )?;
        for (index, program) in self.programs.iter().enumerate() {
            program.validate(index, self.n_cu, tensor_count)?;
        }
        Ok(())
    }
}

impl Program {
    fn validate(&self, index: usize, n_cu: u32, tensor_count: usize) -> Result<()> {
        require(
            self.rows > 0
                && self.n_counter > 0
                && !self.insts.is_empty()
                && !self.stream.is_empty()
                && self.stream_ofs.len() == n_cu as usize
                && self.stream_len.len() == n_cu as usize
                && self.gq_stream.len() == self.stream.len()
                && self.gq_seg_ofs.len() >= 2,
            &format!("program {index} geometry"),
        )?;
        for inst in &self.insts {
            let Some(op) = DevOp::from_u16(inst.op) else {
                return Err(format!("aux program: program {index} opcode"));
            };
            let slots = slots_for(op);
            require(
                inst.blocks > 0
                    && inst
                        .t
                        .iter()
                        .all(|&t| t == TENSOR_NONE16 || usize::from(t) < tensor_count)
                    && integer_tensor_handles_valid(op, inst, tensor_count),
                &format!("program {index} instruction"),
            )?;
            if slots.provenance != Provenance::Undocumented {
                require(
                    slots.t.iter().enumerate().all(|(slot, spec)| {
                        spec.is_none_or(|spec| spec.optional || inst.t[slot] != TENSOR_NONE16)
                    }),
                    &format!("program {index} required tensor"),
                )?;
            }
        }
        require(
            self.waits
                .iter()
                .all(|w| w.id < self.n_counter && w.threshold > 0)
                && self.succs.iter().all(|&id| id < self.n_counter),
            &format!("program {index} counter reference"),
        )?;

        let mut cursor = 0usize;
        for (&offset, &len) in self.stream_ofs.iter().zip(&self.stream_len) {
            let offset = offset as usize;
            let len = len as usize;
            require(offset == cursor, &format!("program {index} stream ranges"))?;
            cursor = offset
                .checked_add(len)
                .filter(|&end| end <= self.stream.len())
                .ok_or_else(|| format!("aux program: program {index} stream range"))?;
        }
        require(
            cursor == self.stream.len(),
            &format!("program {index} stream coverage"),
        )?;

        for entry in self.stream.iter().chain(&self.gq_stream) {
            validate_entry(
                entry,
                &self.insts,
                self.waits.len(),
                self.succs.len(),
                self.gq_seg_ofs.len() - 1,
                index,
            )?;
        }

        let mut slices = self.stream.iter().map(stream_key).collect::<Vec<_>>();
        let mut queued = self.gq_stream.iter().map(stream_key).collect::<Vec<_>>();
        slices.sort_unstable();
        queued.sort_unstable();
        require(
            slices == queued,
            &format!("program {index} global queue permutation"),
        )?;
        for (inst_index, inst) in self.insts.iter().enumerate() {
            let mut actual = self
                .stream
                .iter()
                .filter(|entry| entry.inst == inst_index as u32)
                .map(|entry| entry.slice)
                .collect::<Vec<_>>();
            actual.sort_unstable();
            require(
                actual == (0..u32::from(inst.blocks)).collect::<Vec<_>>(),
                &format!("program {index} instruction slice coverage"),
            )?;
        }

        require(
            self.gq_seg_ofs.first() == Some(&0)
                && self.gq_seg_ofs.last() == Some(&(self.gq_stream.len() as u32))
                && self
                    .gq_seg_ofs
                    .windows(2)
                    .all(|bounds| bounds[0] <= bounds[1]),
            &format!("program {index} queue bounds"),
        )?;
        for (segment, bounds) in self.gq_seg_ofs.windows(2).enumerate() {
            require(
                self.gq_stream[bounds[0] as usize..bounds[1] as usize]
                    .iter()
                    .all(|entry| usize::from(entry.seg) == segment),
                &format!("program {index} queue segment"),
            )?;
        }
        Ok(())
    }
}

fn integer_tensor_handles_valid(op: DevOp, inst: &DevInst64, tensor_count: usize) -> bool {
    let required =
        |slot: usize| inst.i[slot] != TENSOR_NONE_I && (inst.i[slot] as usize) < tensor_count;
    let optional =
        |slot: usize| inst.i[slot] == TENSOR_NONE_I || (inst.i[slot] as usize) < tensor_count;
    match op {
        DevOp::GemvGluFp8 if inst.fj[2] != 0 => [3, 4, 6, 7].into_iter().all(required),
        DevOp::FlashMlaDecode if inst.t[7] != TENSOR_NONE16 => required(6),
        DevOp::GemvQkvg => required(6),
        DevOp::KdaConv3 => [4, 5, 6, 7].into_iter().all(required),
        DevOp::KdaStateStepG => {
            required(5) && (inst.i[4] & 4 == 0 || (inst.fj[2] as usize) < tensor_count)
        }
        DevOp::GemvQkvMxfp4 | DevOp::GemvQkvFp8 => {
            required(5)
                && required(6)
                && if inst.i[4] == 0 {
                    optional(7)
                } else {
                    required(7)
                }
        }
        _ => true,
    }
}

fn validate_entry(
    entry: &StreamEnt,
    insts: &[DevInst64],
    waits: usize,
    succs: usize,
    segments: usize,
    program: usize,
) -> Result<()> {
    let Some(inst) = insts.get(entry.inst as usize) else {
        return Err(format!(
            "aux program: program {program} instruction reference"
        ));
    };
    let flags = SE_FINE | SE_XCTR | SE_XR_WAVE_RS | SE_KDA_INTRA_WAVE_ITEMS;
    let wait_end = (entry.wait_ofs as usize).checked_add(entry.wait_len as usize);
    let succ_end = (entry.succ_ofs as usize).checked_add(entry.succ_len as usize);
    require(
        entry.slice < u32::from(inst.blocks)
            && entry.flags & !flags == 0
            && usize::from(entry.seg) < segments
            && wait_end.is_some_and(|end| end <= waits)
            && succ_end.is_some_and(|end| end <= succs),
        &format!("program {program} stream entry"),
    )
}

fn stream_key(e: &StreamEnt) -> (u32, u32, u32, u32, u16, u16, u16, u16) {
    (
        e.inst, e.slice, e.wait_ofs, e.succ_ofs, e.wait_len, e.succ_len, e.flags, e.seg,
    )
}

fn header_at(bytes: &[u8], offset: usize, index: usize) -> Result<Counts> {
    let get = |field: usize, name: &str| get_u32(bytes, offset + field * 4, name);
    let counts = Counts {
        rows: get(0, "rows")?,
        n_counter: get(1, "counter count")?,
        n_inst: get(4, "instruction count")? as usize,
        n_stream: get(5, "stream count")? as usize,
        n_wait: get(6, "wait count")? as usize,
        n_succ: get(7, "successor count")? as usize,
        n_gq_seg: get(8, "queue window count")? as usize,
    };
    require(
        get(2, "hierarchy")? == 0
            && get(3, "L2 domains")? == 0
            && get(9, "reserved")? == 0
            && counts.rows > 0
            && counts.n_counter > 0
            && counts.n_inst > 0
            && counts.n_stream > 0
            && counts.n_gq_seg > 0,
        &format!("program {index} header"),
    )?;
    Ok(counts)
}

fn read_header(reader: &mut Reader<'_>, index: usize) -> Result<Counts> {
    let offset = reader.offset;
    reader.take(PROGRAM_HEADER_BYTES, &format!("program {index} header"))?;
    header_at(reader.bytes, offset, index)
}

fn program_payload_bytes(n_cu: usize, counts: Counts) -> Result<usize> {
    let fields = [
        (counts.n_inst, 64usize),
        (counts.n_stream, 24),
        (n_cu, 4),
        (n_cu, 4),
        (counts.n_wait, 8),
        (counts.n_succ, 4),
        (counts.n_stream, 24),
        (
            counts
                .n_gq_seg
                .checked_add(1)
                .ok_or("aux program: queue window count overflow")?,
            4,
        ),
    ];
    fields.into_iter().try_fold(0usize, |sum, (count, width)| {
        sum.checked_add(
            count
                .checked_mul(width)
                .ok_or("aux program: payload length overflow")?,
        )
        .ok_or_else(|| "aux program: payload length overflow".to_string())
    })
}

fn require(ok: bool, reason: &str) -> Result<()> {
    if ok {
        Ok(())
    } else {
        Err(format!("aux program: {reason}"))
    }
}

fn checked_end(start: usize, len: usize, total: usize, what: &str) -> Result<usize> {
    start
        .checked_add(len)
        .filter(|&end| end <= total)
        .ok_or_else(|| format!("aux program: truncated {what}"))
}

fn get_u32(bytes: &[u8], offset: usize, what: &str) -> Result<u32> {
    let data = bytes
        .get(offset..offset.saturating_add(4))
        .filter(|data| data.len() == 4)
        .ok_or_else(|| format!("aux program: truncated {what}"))?;
    Ok(u32::from_le_bytes(data.try_into().unwrap()))
}

fn u32_len(len: usize, what: &str) -> Result<u32> {
    len.try_into()
        .map_err(|_| format!("aux program: {what} overflow"))
}

fn put_u16(out: &mut Vec<u8>, value: u16) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn put_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn put_u32s(out: &mut Vec<u8>, values: &[u32]) {
    for &value in values {
        put_u32(out, value);
    }
}

fn put_inst(out: &mut Vec<u8>, inst: &DevInst64) {
    put_u16(out, inst.op);
    put_u16(out, inst.blocks);
    put_u32s(out, &inst.fj);
    for &tensor in &inst.t {
        put_u16(out, tensor);
    }
    put_u32s(out, &inst.i);
}

fn put_stream(out: &mut Vec<u8>, entry: &StreamEnt) {
    put_u32(out, entry.inst);
    put_u32(out, entry.slice);
    put_u32(out, entry.wait_ofs);
    put_u32(out, entry.succ_ofs);
    put_u16(out, entry.wait_len);
    put_u16(out, entry.succ_len);
    put_u16(out, entry.flags);
    put_u16(out, entry.seg);
}

struct Reader<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn take(&mut self, len: usize, what: &str) -> Result<&'a [u8]> {
        let end = checked_end(self.offset, len, self.bytes.len(), what)?;
        let data = &self.bytes[self.offset..end];
        self.offset = end;
        Ok(data)
    }

    fn u16(&mut self, what: &str) -> Result<u16> {
        Ok(u16::from_le_bytes(self.take(2, what)?.try_into().unwrap()))
    }

    fn u32(&mut self, what: &str) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4, what)?.try_into().unwrap()))
    }

    fn u32s(&mut self, count: usize, index: usize, what: &str) -> Result<Vec<u32>> {
        (0..count)
            .map(|_| self.u32(&format!("program {index} {what}")))
            .collect()
    }

    fn inst(&mut self, index: usize) -> Result<DevInst64> {
        let op = self.u16(&format!("program {index} opcode"))?;
        let blocks = self.u16(&format!("program {index} blocks"))?;
        let mut fj = [0; 3];
        for value in &mut fj {
            *value = self.u32(&format!("program {index} scalar"))?;
        }
        let mut t = [0; 8];
        for value in &mut t {
            *value = self.u16(&format!("program {index} tensor"))?;
        }
        let mut i = [0; 8];
        for value in &mut i {
            *value = self.u32(&format!("program {index} integer"))?;
        }
        Ok(DevInst64 {
            op,
            blocks,
            fj,
            t,
            i,
        })
    }

    fn stream(&mut self, index: usize, what: &str) -> Result<StreamEnt> {
        Ok(StreamEnt {
            inst: self.u32(&format!("program {index} {what} instruction"))?,
            slice: self.u32(&format!("program {index} {what} slice"))?,
            wait_ofs: self.u32(&format!("program {index} {what} wait offset"))?,
            succ_ofs: self.u32(&format!("program {index} {what} successor offset"))?,
            wait_len: self.u16(&format!("program {index} {what} wait length"))?,
            succ_len: self.u16(&format!("program {index} {what} successor length"))?,
            flags: self.u16(&format!("program {index} {what} flags"))?,
            seg: self.u16(&format!("program {index} {what} segment"))?,
        })
    }
}

#[cfg(test)]
#[path = "aux_program_tests.rs"]
mod tests;
