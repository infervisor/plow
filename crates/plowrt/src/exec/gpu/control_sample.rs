use crate::device::cuda::{CudaBackend, CudaStream, KernelFn, PinnedHost};
use crate::device::{Backend, DeviceMem, Module};
use crate::{Result, RuntimeError};
use std::sync::Arc;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Plan {
    seed: u64,
    temperature: f32,
    top_p: f32,
    presence_penalty: f32,
    stop_p: f32,
    stop_hazard: f32,
    period_thr: f32,
    boost: f32,
    top_k: u32,
    penalty_window: u32,
    min_frames: u32,
    cap: u32,
    stall: u32,
    period_win: u32,
    pmin: u32,
    pmax: u32,
    start: u32,
    every: u32,
    backstop: u32,
    hist: u32,
    n_windows: u32,
    windows: [[u32; 2]; 16],
}
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct State {
    plan: Plan,
    ticks: u32,
    frames: u32,
    last: u32,
    boost_at: u32,
    backstop: u32,
    stopped: u32,
    hazard: f32,
    aeos_bias: f32,
    recent: [i32; 256],
    tail: [i32; 256],
}
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct Row {
    pub slot: u32,
    pub output: u32,
    pub logits: u32,
}
const _: () = assert!(std::mem::size_of::<Plan>() == 216);
const _: () = assert!(std::mem::size_of::<State>() == 2296);
const _: () = assert!(std::mem::align_of::<State>() == 8);
const _: () = assert!(std::mem::offset_of!(State, last) == 224);
const _: () = assert!(std::mem::offset_of!(State, stopped) == 236);

pub(super) fn state(p: crate::tts::maya2::Plan) -> Result<State> {
    let c = p.cb0;
    let g = p.guard;
    if c.penalty_window > 256
        || g.hist > 256
        || g.pmax > 256
        || p.n_windows > 16
        || !c.temperature.is_finite()
        || c.temperature <= 0.
        || [
            c.top_p,
            c.presence_penalty,
            c.stop_p,
            c.stop_hazard,
            g.period_thr,
            g.boost,
        ]
        .iter()
        .any(|v| !v.is_finite())
    {
        return Err(RuntimeError::Device(
            "controlled sampler plan exceeds its geometry".into(),
        ));
    }
    Ok(State {
        plan: Plan {
            seed: p.seed,
            temperature: c.temperature,
            top_p: c.top_p,
            presence_penalty: c.presence_penalty,
            stop_p: c.stop_p,
            stop_hazard: c.stop_hazard,
            period_thr: g.period_thr,
            boost: g.boost,
            top_k: c.top_k,
            penalty_window: c.penalty_window,
            min_frames: p.min_frames,
            cap: p.cap,
            stall: g.stall,
            period_win: g.period_win,
            pmin: g.pmin,
            pmax: g.pmax,
            start: g.start,
            every: g.every,
            backstop: g.backstop,
            hist: g.hist,
            n_windows: u32::from(p.n_windows),
            windows: p.windows.map(|(a, b)| [a, b]),
        },
        ticks: 0,
        frames: 0,
        last: 0,
        boost_at: u32::MAX,
        backstop: 0,
        stopped: 0,
        hazard: 0.,
        aeos_bias: 0.,
        recent: [-1; 256],
        tail: [-1; 256],
    })
}

pub(super) struct ControlledSample {
    states: Arc<DeviceMem>,
    rows: Arc<DeviceMem>,
    pub codes: Arc<DeviceMem>,
    pub readback_codes: Arc<DeviceMem>,
    codebooks: usize,
    stage: Arc<PinnedHost>,
    sample: KernelFn,
    finish: KernelFn,
    slots: usize,
    count: usize,
    slot_end: usize,
    seen_slots: Vec<bool>,
    seen_outputs: Vec<bool>,
}
impl ControlledSample {
    pub fn load(
        be: &CudaBackend,
        stream: &CudaStream,
        module: &Module,
        slots: usize,
        outputs: usize,
        codebooks: usize,
    ) -> Result<Self> {
        let frame_bytes = slots
            .checked_mul(codebooks)
            .and_then(|n| n.checked_mul(4))
            .filter(|&n| n > 0)
            .ok_or_else(|| RuntimeError::Rejected("invalid controller frame geometry".into()))?;
        let sample = be.get_function(module, "control_sample")?;
        let finish = be.get_function(module, "control_sample_finish")?;
        be.set_max_dynamic_smem(sample, 65540)?;
        let codes = Arc::new(be.alloc(0, (outputs * 4) as u64)?);
        // Load-time depth calibration consumes history before the first sampler enqueue.
        be.memset_d8_async(codes.base, 0, outputs * 4, stream)?;
        Ok(Self {
            states: Arc::new(be.alloc(0, (slots * std::mem::size_of::<State>()) as u64)?),
            rows: Arc::new(be.alloc(0, (slots * 12) as u64)?),
            codes,
            readback_codes: Arc::new(be.alloc(0, frame_bytes as u64)?),
            codebooks,
            stage: Arc::new(be.host_alloc_pinned(slots * 12)?),
            sample,
            finish,
            slots,
            count: 0,
            slot_end: 0,
            seen_slots: vec![false; slots],
            seen_outputs: vec![false; outputs],
        })
    }
    pub fn bind(
        &self,
        be: &crate::device::cuda::CudaBackend,
        slot: usize,
        plan: crate::tts::maya2::Plan,
    ) -> Result<()> {
        if slot >= self.slots {
            return Err(RuntimeError::Device(
                "controlled sampler slot out of range".into(),
            ));
        }
        let s = state(plan)?;
        let bytes = bytemuck::bytes_of(&s);
        be.upload(&self.states, (slot * bytes.len()) as u64, bytes)
    }
    pub fn enqueue(
        &mut self,
        be: &CudaBackend,
        stream: &CudaStream,
        rows: &[Row],
        logits: u64,
        stride: u32,
        classes: u32,
        stop: u32,
    ) -> Result<()> {
        if rows.is_empty()
            || rows.len() > self.slots
            || classes > 4096
            || stop.checked_add(1) != Some(classes)
            || stride < classes
            || rows.iter().any(|r| {
                r.slot as usize >= self.slots || (r.output as u64 + 1) * 4 > self.codes.len
            })
        {
            return Err(RuntimeError::Device(
                "invalid controlled sampler rows".into(),
            ));
        }
        self.seen_slots.fill(false);
        self.seen_outputs.fill(false);
        let mut slot_end = 0;
        for row in rows {
            if std::mem::replace(&mut self.seen_slots[row.slot as usize], true)
                || std::mem::replace(&mut self.seen_outputs[row.output as usize], true)
                || row.logits as usize >= self.slots
            {
                return Err(RuntimeError::Device(
                    "controlled sampler rows overlap or exceed logits".into(),
                ));
            }
            slot_end = slot_end.max(row.slot as usize + 1);
        }
        be.memset_d8_async(self.codes.base, 0, self.codes.len as usize, stream)?;
        let stage = Arc::get_mut(&mut self.stage).ok_or_else(|| {
            RuntimeError::Device("controlled sampler staging is quarantined".into())
        })?;
        let bytes = bytemuck::cast_slice(rows);
        stage.as_mut_slice()[..bytes.len()].copy_from_slice(bytes);
        // The caller drains its stream before reusing this persistent pinned staging.
        unsafe {
            be.memcpy_htod_async(self.rows.base, &stage.as_slice()[..bytes.len()], stream)?;
        }
        self.count = rows.len();
        self.slot_end = slot_end;
        let (mut states, mut map, mut l, mut codes) =
            (self.states.base, self.rows.base, logits, self.codes.base);
        let (mut n, mut stride, mut classes, mut stop) = (rows.len() as u32, stride, classes, stop);
        let mut args = [
            &mut states as *mut u64 as *mut _,
            &mut map as *mut u64 as *mut _,
            &mut l as *mut u64 as *mut _,
            &mut codes as *mut u64 as *mut _,
            &mut n as *mut u32 as *mut _,
            &mut stride as *mut u32 as *mut _,
            &mut classes as *mut u32 as *mut _,
            &mut stop as *mut u32 as *mut _,
        ];
        be.launch_kernel(self.sample, n, 256, 65540, &mut args, Some(stream))
    }
    pub fn finish(
        &self,
        be: &CudaBackend,
        stream: &CudaStream,
        codes: u64,
        end_rows: usize,
        codebooks: u32,
        stop: u32,
    ) -> Result<()> {
        if codebooks as usize != self.codebooks {
            return Err(RuntimeError::Rejected(
                "controller frame width changed".into(),
            ));
        }
        if self.count == 0 || end_rows < self.slot_end || end_rows > self.slots {
            return Err(RuntimeError::Rejected(
                "controller readback prefix does not cover its slots".into(),
            ));
        }
        be.memcpy_dtod_async(
            self.readback_codes.base,
            codes,
            (end_rows * self.codebooks * 4) as u64,
            stream,
        )?;
        let (mut states, mut map, mut codes) =
            (self.states.base, self.rows.base, self.readback_codes.base);
        let (mut n, mut codebooks, mut stop) = (self.count as u32, codebooks, stop);
        let mut args = [
            &mut states as *mut u64 as *mut _,
            &mut map as *mut u64 as *mut _,
            &mut codes as *mut u64 as *mut _,
            &mut n as *mut u32 as *mut _,
            &mut codebooks as *mut u32 as *mut _,
            &mut stop as *mut u32 as *mut _,
        ];
        be.launch_kernel(
            self.finish,
            n.div_ceil(256),
            256,
            0,
            &mut args,
            Some(stream),
        )
    }
    pub fn quarantine(&self) {
        std::mem::forget(self.stage.clone());
        std::mem::forget(self.states.clone());
        std::mem::forget(self.rows.clone());
        std::mem::forget(self.codes.clone());
        std::mem::forget(self.readback_codes.clone());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tts::maya2::{Cb0, Guard, Slot, Tick};

    fn plan(seed: u64, top_k: u32, top_p: f32) -> crate::tts::maya2::Plan {
        crate::tts::maya2::Plan {
            cb0: Cb0 {
                temperature: 0.8,
                top_k,
                top_p,
                presence_penalty: 1.1,
                penalty_window: 8,
                stop_p: 0.6,
                stop_hazard: 0.9,
                min_frames_floor: 3,
            },
            guard: Guard {
                dur_k: 1.,
                dur_margin_s: 1.,
                stall: 4,
                period_win: 16,
                period_thr: 0.8,
                pmin: 1,
                pmax: 4,
                start: 4,
                every: 1,
                boost: 5.,
                backstop: 3,
                hist: 32,
                hard_cap: 128,
            },
            min_frames: 3,
            cap: 100,
            loose: 128,
            windows: [(5, 8); 16],
            n_windows: 1,
            seed,
        }
    }
    #[test]
    fn controlled_sampler_plan_bounds() {
        let p = plan(1, 64, 0.9);
        assert!(state(p).is_ok());
        for value in [257, u32::MAX] {
            let mut q = p;
            q.cb0.penalty_window = value;
            assert!(state(q).is_err());
            let mut q = p;
            q.guard.hist = value;
            assert!(state(q).is_err());
            let mut q = p;
            q.guard.pmax = value;
            assert!(state(q).is_err());
        }
        let mut q = p;
        q.cb0.temperature = 0.;
        assert!(state(q).is_err());
        let mut q = p;
        q.cb0.top_p = f32::NAN;
        assert!(state(q).is_err());
        assert_eq!(std::mem::size_of::<Row>(), 12);
    }
    #[test]
    #[ignore = "CUDA; CONTROL_SAMPLE_CUBIN must name a compiled controlled-sampler object"]
    fn handoff_pcm16_device_matches_host() {
        let be = CudaBackend::new(0).unwrap();
        let image = std::fs::read(std::env::var("CONTROL_SAMPLE_CUBIN").unwrap()).unwrap();
        let module = be.module_load(&image).unwrap();
        assert_eq!(
            be.module_global_u32(&module, "plow_control_sample_abi")
                .unwrap(),
            Some(1)
        );
        let kernel = be.get_function(&module, "handoff_pack_pcm16").unwrap();
        let stream = be.stream_create().unwrap();
        let capacity = 320usize;
        let states = be
            .alloc(0, (capacity * std::mem::size_of::<State>()) as u64)
            .unwrap();
        let slots = be.alloc(0, (capacity * 4) as u64).unwrap();
        let map: Vec<u32> = (0..capacity)
            .map(|r| ((r * 37 + 17) % capacity) as u32)
            .collect();
        be.upload(&slots, 0, bytemuck::cast_slice(&map)).unwrap();
        for slot in 0..capacity {
            let mut value = state(plan(slot as u64, 1, 1.0)).unwrap();
            value.last = (slot * 71 % 2048) as u32;
            value.stopped = (slot % 3 == 1) as u32;
            let bytes = unsafe {
                std::slice::from_raw_parts(
                    (&value as *const State).cast::<u8>(),
                    std::mem::size_of::<State>(),
                )
            };
            be.upload(&states, (slot * bytes.len()) as u64, bytes)
                .unwrap();
        }
        let edges = [
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            -2.0,
            2.0,
            -1.0,
            1.0,
            -0.0,
            0.0,
            0.5 / 32767.0,
            -0.5 / 32767.0,
            1.5 / 32767.0,
            -1.5 / 32767.0,
            32766.5 / 32767.0,
            -32766.5 / 32767.0,
        ];
        for samples in [1usize, 1537, 1920] {
            let stride = (8 + samples * 2 + 3) & !3;
            let values: Vec<f32> = (0..capacity * samples)
                .map(|i| {
                    if i % 31 < edges.len() {
                        edges[i % 31]
                    } else {
                        (i * 7919 % 100003) as f32 / 40000.0 - 1.25
                    }
                })
                .collect();
            let pcm = be.alloc(0, (values.len() * 4) as u64).unwrap();
            let packet = be.alloc(0, (capacity * stride + 64) as u64).unwrap();
            be.upload(&pcm, 0, bytemuck::cast_slice(&values)).unwrap();
            be.synchronize().unwrap();
            for rows in [0usize, 1, 8, 32, 64, 128, 200, 224, 288, 300, 320] {
                be.memset_d8_async(packet.base, 0xa5, packet.len as usize, &stream)
                    .unwrap();
                let (mut s, mut m, mut p, mut out) =
                    (states.base, slots.base, pcm.base, packet.base);
                let (mut n, mut cap, mut ns, mut pitch) =
                    (rows as u32, capacity as u32, samples as u32, stride as u32);
                let mut args = [
                    &mut s as *mut u64 as *mut _,
                    &mut m as *mut u64 as *mut _,
                    &mut p as *mut u64 as *mut _,
                    &mut out as *mut u64 as *mut _,
                    &mut n as *mut u32 as *mut _,
                    &mut cap as *mut u32 as *mut _,
                    &mut ns as *mut u32 as *mut _,
                    &mut pitch as *mut u32 as *mut _,
                ];
                be.launch_kernel(kernel, n.max(1), 256, 0, &mut args, Some(&stream))
                    .unwrap();
                be.stream_synchronize(&stream).unwrap();
                let mut actual = vec![0u8; packet.len as usize];
                be.download(&packet, 0, &mut actual).unwrap();
                let mut expected = vec![0xa5u8; actual.len()];
                for r in 0..rows {
                    let slot = map[r] as usize;
                    let stopped = slot % 3 == 1;
                    let row = &mut expected[r * stride..(r + 1) * stride];
                    row.fill(0);
                    row[..4].copy_from_slice(&(stopped as u32).to_le_bytes());
                    row[4..8].copy_from_slice(&((slot * 71 % 2048) as u32).to_le_bytes());
                    if !stopped {
                        let mut audio = Vec::with_capacity(samples * 2);
                        crate::tts::pcm16(&values[r * samples..(r + 1) * samples], &mut audio);
                        row[8..8 + samples * 2].copy_from_slice(&audio);
                    }
                }
                assert!(
                    actual == expected,
                    "rows={rows}, samples={samples}, first mismatch {:?}",
                    actual
                        .iter()
                        .zip(&expected)
                        .enumerate()
                        .find(|(_, (a, b))| a != b)
                );
                println!("HANDOFF_PCM rows={rows} samples={samples} exact header/PCM/padding/tail");
                if samples == 1920 && [32, 64, 128, 200, 224].contains(&rows) {
                    let begin = be.event_create(true).unwrap();
                    let end = be.event_create(true).unwrap();
                    let mut times = Vec::new();
                    for _ in 0..5 {
                        be.event_record(&begin, &stream).unwrap();
                        for _ in 0..64 {
                            be.launch_kernel(kernel, n, 256, 0, &mut args, Some(&stream))
                                .unwrap();
                        }
                        be.event_record(&end, &stream).unwrap();
                        be.stream_synchronize(&stream).unwrap();
                        times.push(
                            f64::from(be.event_elapsed_ms(&begin, &end).unwrap()) * 1000. / 64.,
                        );
                    }
                    println!("HANDOFF_PCM_TIMING rows={rows} samples={samples} gpu_us={times:?}");
                }
            }
        }
    }
    #[test]
    #[ignore = "CUDA; CONTROL_SAMPLE_CUBIN must name a compiled controlled-sampler object"]
    fn handoff_codes_and_lengths_match_independent_slot_maps() {
        let be = CudaBackend::new(0).unwrap();
        let image = std::fs::read(std::env::var("CONTROL_SAMPLE_CUBIN").unwrap()).unwrap();
        let module = be.module_load(&image).unwrap();
        assert_eq!(
            be.module_global_u32(&module, "plow_control_sample_abi")
                .unwrap(),
            Some(1)
        );
        let pack = be.get_function(&module, "handoff_pack_codes").unwrap();
        let lengths_kernel = be.get_function(&module, "handoff_codec_lengths").unwrap();
        let stream = be.stream_create().unwrap();
        let (source_capacity, state_capacity) = (384u32, 353u32);
        let states = be
            .alloc(
                0,
                u64::from(state_capacity) * std::mem::size_of::<State>() as u64,
            )
            .unwrap();
        let mut expected_states = Vec::with_capacity(states.len as usize);
        for slot in 0..state_capacity {
            let mut value = state(plan(u64::from(slot), 1, 1.)).unwrap();
            value.stopped = u32::from(slot % 3 == 1);
            let bytes = unsafe {
                std::slice::from_raw_parts(
                    (&value as *const State).cast::<u8>(),
                    std::mem::size_of::<State>(),
                )
            };
            expected_states.extend_from_slice(bytes);
        }
        be.upload(&states, 0, &expected_states).unwrap();
        let maps: [Vec<u32>; 2] = [source_capacity, state_capacity].map(|cap| {
            (0..320)
                .map(|r| if r == 0 { cap - 1 } else { (r * 37 + 17) % cap })
                .collect()
        });
        let map_buffers = maps.each_ref().map(|map| {
            let buffer = be.alloc(0, (map.len() * 4) as u64).unwrap();
            be.upload(&buffer, 0, bytemuck::cast_slice(map)).unwrap();
            buffer
        });
        be.synchronize().unwrap();
        for codebooks in [1u32, 32] {
            let values: Vec<u32> = (0..source_capacity * codebooks)
                .map(|i| {
                    if i % 17 == 0 {
                        2048
                    } else {
                        i.wrapping_mul(7919)
                    }
                })
                .collect();
            let source = be.alloc(0, (values.len() * 4) as u64).unwrap();
            be.upload(&source, 0, bytemuck::cast_slice(&values))
                .unwrap();
            let output = be.alloc(0, u64::from(320 * codebooks + 16) * 4).unwrap();
            let lengths = be.alloc(0, (320 + 16) * 4).unwrap();
            be.synchronize().unwrap();
            for capacity in [1u32, 8, 32, 64, 128, 200, 224, 288, 300, 320] {
                let rows_set: std::collections::BTreeSet<_> =
                    [0, 1, capacity / 2, capacity - 1, capacity].into();
                for rows in rows_set {
                    let (mut src, mut map, mut dst) =
                        (source.base, map_buffers[0].base, output.base);
                    let (mut n, mut src_cap, mut cap, mut cb) =
                        (rows, source_capacity, capacity, codebooks);
                    let mut args = [
                        &mut src as *mut u64 as *mut _,
                        &mut map as *mut u64 as *mut _,
                        &mut dst as *mut u64 as *mut _,
                        &mut n as *mut u32 as *mut _,
                        &mut src_cap as *mut u32 as *mut _,
                        &mut cap as *mut u32 as *mut _,
                        &mut cb as *mut u32 as *mut _,
                    ];
                    be.memset_d8_async(output.base, 0xa5, output.len as usize, &stream)
                        .unwrap();
                    be.launch_kernel(
                        pack,
                        (capacity * codebooks).div_ceil(256),
                        256,
                        0,
                        &mut args,
                        Some(&stream),
                    )
                    .unwrap();
                    be.stream_synchronize(&stream).unwrap();
                    let mut actual = vec![0u32; (output.len / 4) as usize];
                    be.download(&output, 0, bytemuck::cast_slice_mut(&mut actual))
                        .unwrap();
                    let mut expected = vec![0xa5a5a5a5; actual.len()];
                    for r in 0..capacity as usize {
                        for k in 0..codebooks as usize {
                            expected[r * codebooks as usize + k] = if r < rows as usize {
                                values[maps[0][r] as usize * codebooks as usize + k]
                            } else {
                                0
                            };
                        }
                    }
                    assert!(
                        actual == expected,
                        "codes rows={rows} capacity={capacity} books={codebooks} first mismatch {:?}",
                        actual.iter().zip(&expected).position(|(a, b)| a != b)
                    );
                    for scale in [1u32, 2, 1920] {
                        let (mut src, mut map, mut dst) =
                            (states.base, map_buffers[1].base, lengths.base);
                        let (mut state_cap, mut unit) = (state_capacity, scale);
                        let mut args = [
                            &mut src as *mut u64 as *mut _,
                            &mut map as *mut u64 as *mut _,
                            &mut dst as *mut u64 as *mut _,
                            &mut n as *mut u32 as *mut _,
                            &mut state_cap as *mut u32 as *mut _,
                            &mut cap as *mut u32 as *mut _,
                            &mut unit as *mut u32 as *mut _,
                        ];
                        be.memset_d8_async(lengths.base, 0xa5, lengths.len as usize, &stream)
                            .unwrap();
                        be.launch_kernel(
                            lengths_kernel,
                            capacity.div_ceil(256),
                            256,
                            0,
                            &mut args,
                            Some(&stream),
                        )
                        .unwrap();
                        be.stream_synchronize(&stream).unwrap();
                        let mut actual = vec![0u32; (lengths.len / 4) as usize];
                        be.download(&lengths, 0, bytemuck::cast_slice_mut(&mut actual))
                            .unwrap();
                        let mut expected = vec![0xa5a5a5a5; actual.len()];
                        for r in 0..capacity as usize {
                            expected[r] = if r < rows as usize && maps[1][r] % 3 != 1 {
                                scale
                            } else {
                                0
                            };
                        }
                        assert!(
                            actual == expected,
                            "lengths rows={rows} capacity={capacity} scale={scale} first mismatch {:?}",
                            actual.iter().zip(&expected).position(|(a, b)| a != b)
                        );
                    }
                }
            }
            let mut unchanged = vec![0u32; values.len()];
            be.download(&source, 0, bytemuck::cast_slice_mut(&mut unchanged))
                .unwrap();
            assert_eq!(unchanged, values, "canonical source changed");
        }
        for (map, buffer) in maps.iter().zip(&map_buffers) {
            let mut actual = vec![0u32; map.len()];
            be.download(buffer, 0, bytemuck::cast_slice_mut(&mut actual))
                .unwrap();
            assert_eq!(&actual, map, "slot map changed");
        }
        let mut actual_states = vec![0u8; expected_states.len()];
        be.download(&states, 0, &mut actual_states).unwrap();
        assert!(actual_states == expected_states, "controller state changed");
        println!("HANDOFF_CODES exact canonical gather, independent controller map, STOP/padding and guard tails PASS");
    }
    #[test]
    #[ignore = "CUDA; CONTROL_SAMPLE_CUBIN must name a compiled controlled-sampler object"]
    fn controlled_sampler_readback_prefix() {
        let be = CudaBackend::new(0).unwrap();
        let image = std::fs::read(std::env::var("CONTROL_SAMPLE_CUBIN").unwrap()).unwrap();
        let module = be.module_load(&image).unwrap();
        let stream = be.stream_create().unwrap();
        let (slots, codebooks, stride) = (320usize, 32usize, 2057usize);
        let mut device =
            ControlledSample::load(&be, &stream, &module, slots, slots * 2, codebooks).unwrap();
        let logits = be.alloc(0, (slots * stride * 2) as u64).unwrap();
        let source = be.alloc(0, (slots * codebooks * 4) as u64).unwrap();
        let canonical: Vec<u32> = (0..slots * codebooks)
            .map(|i| ((i * 37 + 11) % 2048) as u32)
            .collect();
        let mut values = vec![((-8f32).to_bits() >> 16) as u16; slots * stride];
        for slot in 0..slots {
            values[slot * stride + 17] = (8f32.to_bits() >> 16) as u16;
        }
        be.upload(&logits, 0, bytemuck::cast_slice(&values))
            .unwrap();
        be.upload(&source, 0, bytemuck::cast_slice(&canonical))
            .unwrap();
        be.synchronize().unwrap();
        for (prefix, last) in [
            (1, 0),
            (2, 1),
            (8, 3),
            (32, 31),
            (33, 32),
            (64, 33),
            (128, 100),
            (224, 199),
            (224, 223),
            (256, 255),
            (288, 287),
            (300, 299),
            (320, 319),
        ] {
            let mut active = vec![last];
            if last != 0 {
                active.push(0);
            }
            let rows: Vec<_> = active
                .iter()
                .enumerate()
                .map(|(i, &slot)| Row {
                    slot: slot as u32,
                    output: i as u32,
                    logits: slot as u32,
                })
                .collect();
            for &slot in &active {
                let mut p = plan(slot as u64, 1, 1.0);
                p.min_frames = if slot == last { 0 } else { u32::MAX };
                p.cb0.stop_p = 0.0;
                device.bind(&be, slot, p).unwrap();
            }
            be.memset_d8_async(
                device.readback_codes.base,
                0xa5,
                device.readback_codes.len as usize,
                &stream,
            )
            .unwrap();
            be.synchronize().unwrap();
            device
                .enqueue(&be, &stream, &rows, logits.base, stride as u32, 2049, 2048)
                .unwrap();
            for invalid in [0, last, slots + 1, usize::MAX] {
                assert!(device
                    .finish(&be, &stream, source.base, invalid, codebooks as u32, 2048)
                    .is_err());
            }
            be.stream_synchronize(&stream).unwrap();
            let mut actual = vec![0u32; canonical.len()];
            be.download(
                &device.readback_codes,
                0,
                bytemuck::cast_slice_mut(&mut actual),
            )
            .unwrap();
            assert!(actual.iter().all(|&v| v == 0xa5a5a5a5));
            device
                .finish(&be, &stream, source.base, prefix, codebooks as u32, 2048)
                .unwrap();
            be.stream_synchronize(&stream).unwrap();
            be.download(
                &device.readback_codes,
                0,
                bytemuck::cast_slice_mut(&mut actual),
            )
            .unwrap();
            let mut expected = vec![0xa5a5a5a5u32; canonical.len()];
            expected[..prefix * codebooks].copy_from_slice(&canonical[..prefix * codebooks]);
            expected[last * codebooks..(last + 1) * codebooks].fill(0);
            expected[last * codebooks] = 2048;
            assert!(
                actual == expected,
                "prefix {prefix}, last slot {last}, first mismatch {:?}",
                actual
                    .iter()
                    .zip(&expected)
                    .enumerate()
                    .find(|(_, (a, b))| a != b)
            );
            be.download(&source, 0, bytemuck::cast_slice_mut(&mut actual))
                .unwrap();
            assert_eq!(actual, canonical, "canonical codes were modified");
            println!("CONTROL_PREFIX prefix={prefix} last_slot={last} exact, rejected prefixes and poisoned tail intact");
        }
    }
    #[test]
    #[ignore = "CUDA; CONTROL_SAMPLE_CUBIN must name a compiled controlled-sampler object"]
    fn checked_sampler_rejects_invalid_metadata() {
        let case = std::env::var("CONTROL_SAMPLE_REJECT").unwrap();
        let be = CudaBackend::new(0).unwrap();
        let image = std::fs::read(std::env::var("CONTROL_SAMPLE_CUBIN").unwrap()).unwrap();
        let module = be.module_load(&image).unwrap();
        let stream = be.stream_create().unwrap();
        let function = be.get_function(&module, "handoff_control_sample").unwrap();
        be.set_max_dynamic_smem(function, 65540).unwrap();
        let state_mem = be.alloc(0, std::mem::size_of::<State>() as u64).unwrap();
        let slot_mem = be.alloc(0, 4).unwrap();
        let row_mem = be.alloc(0, 4).unwrap();
        let logits_mem = be.alloc(0, 8192).unwrap();
        let codes_mem = be.alloc(0, 4).unwrap();
        let mut value = state(plan(1, 64, 1.0)).unwrap();
        let (mut slot, mut row) = (0u32, 0u32);
        let (mut states, mut slots, mut rows, mut logits, mut codes) = (
            state_mem.base,
            slot_mem.base,
            row_mem.base,
            logits_mem.base,
            codes_mem.base,
        );
        let (
            mut count,
            mut state_cap,
            mut output_cap,
            mut logits_cap,
            mut stride,
            mut classes,
            mut stop,
        ) = (1u32, 1u32, 1u32, 1u32, 4096u32, 2049u32, 2048u32);
        let mut block = 256;
        match case.as_str() {
            "state_index" => slot = 1,
            "logits_index" => row = 1,
            "count" => count = 2,
            "state_capacity" => state_cap = 0,
            "output_capacity" => output_cap = 0,
            "logits_capacity" => logits_cap = 0,
            "classes" => classes = 4097,
            "stop" => stop = 2047,
            "stride" => stride = 2048,
            "state_alignment" => states += 4,
            "map_alignment" => slots += 2,
            "null_state" => states = 0,
            "block" => block = 128,
            "history" => value.plan.hist = 257,
            "temperature" => value.plan.temperature = f32::NAN,
            "last" => value.last = stop,
            _ => panic!("unknown invalid metadata case {case}"),
        }
        be.upload(&state_mem, 0, bytemuck::bytes_of(&value))
            .unwrap();
        be.upload(&slot_mem, 0, bytemuck::bytes_of(&slot)).unwrap();
        be.upload(&row_mem, 0, bytemuck::bytes_of(&row)).unwrap();
        be.upload(&logits_mem, 0, &[0; 8192]).unwrap();
        be.upload(&codes_mem, 0, &[0xa5; 4]).unwrap();
        be.synchronize().unwrap();
        let mut args = [
            &mut states as *mut u64 as *mut _,
            &mut slots as *mut u64 as *mut _,
            &mut rows as *mut u64 as *mut _,
            &mut logits as *mut u64 as *mut _,
            &mut codes as *mut u64 as *mut _,
            &mut count as *mut u32 as *mut _,
            &mut state_cap as *mut u32 as *mut _,
            &mut output_cap as *mut u32 as *mut _,
            &mut logits_cap as *mut u32 as *mut _,
            &mut stride as *mut u32 as *mut _,
            &mut classes as *mut u32 as *mut _,
            &mut stop as *mut u32 as *mut _,
        ];
        be.launch_kernel(function, 1, block, 65540, &mut args, Some(&stream))
            .unwrap();
        let error = be
            .stream_synchronize(&stream)
            .expect_err("invalid metadata was accepted");
        assert!(
            error.to_string().contains("719"),
            "expected explicit trap719 for {case}, got {error}"
        );
        println!("CONTROL_REJECT {case} trap719");
    }

    #[test]
    #[ignore = "CUDA; CONTROL_SAMPLE_CUBIN must name a compiled controlled-sampler object"]
    fn controlled_sampler_device_matches_host() {
        let be = CudaBackend::new(0).unwrap();
        let image = std::fs::read(std::env::var("CONTROL_SAMPLE_CUBIN").unwrap()).unwrap();
        let module = be.module_load(&image).unwrap();
        let stream = be.stream_create().unwrap();
        let sanitize = std::env::var_os("CONTROL_SAMPLE_SANITIZE").is_some();
        let production = std::env::var_os("CONTROL_SAMPLE_PRODUCTION").is_some();
        let checked = std::env::var_os("CONTROL_SAMPLE_CHECKED").is_some();
        let checked_kernel = checked.then(|| {
            let function = be.get_function(&module, "handoff_control_sample").unwrap();
            be.set_max_dynamic_smem(function, 65540).unwrap();
            function
        });
        let n = std::env::var("CONTROL_SAMPLE_ROWS")
            .ok()
            .map(|s| s.parse::<usize>().unwrap())
            .unwrap_or(if sanitize { 33usize } else { 224usize });
        assert!((2..=320).contains(&n));
        let mut device = ControlledSample::load(&be, &stream, &module, n, n + 7, 32).unwrap();
        be.stream_synchronize(&stream).unwrap();
        let mut initial_codes = vec![u32::MAX; n + 7];
        be.download(
            &device.codes,
            0,
            bytemuck::cast_slice_mut(&mut initial_codes),
        )
        .unwrap();
        assert!(initial_codes.iter().all(|&code| code == 0));
        let mut host = Vec::new();
        for r in 0..n {
            let p = plan(
                193 + r as u64,
                if production {
                    64
                } else {
                    [1, 16, 64, 2049][r % 4]
                },
                if production {
                    1.
                } else {
                    [1., 0.8, 0.95][r % 3]
                },
            );
            device.bind(&be, r, p).unwrap();
            host.push(Slot::new(p));
        }
        let mut rows: Vec<_> = (0..n)
            .rev()
            .enumerate()
            .map(|(i, r)| Row {
                slot: r as u32,
                output: if checked {
                    i as u32
                } else {
                    ((r + 7) % n) as u32
                },
                logits: ((r + 13) % n) as u32,
            })
            .collect();
        let logits = be.alloc(0, (n * 2057 * 2) as u64).unwrap();
        let final_codes = be.alloc(0, (n * 32 * 4) as u64).unwrap();
        let checked_maps = checked.then(|| {
            [
                be.alloc(0, (n * 4) as u64).unwrap(),
                be.alloc(0, (n * 4) as u64).unwrap(),
            ]
        });
        let draw = |device: &mut ControlledSample, rows: &[Row]| -> Result<()> {
            let Some(kernel) = checked_kernel else {
                return device.enqueue(&be, &stream, rows, logits.base, 2057, 2049, 2048);
            };
            be.upload(&device.rows, 0, bytemuck::cast_slice(rows))?;
            let maps = checked_maps.as_ref().unwrap();
            let slots: Vec<_> = rows.iter().map(|r| r.slot).collect();
            let logits_rows: Vec<_> = rows.iter().map(|r| r.logits).collect();
            be.upload(&maps[0], 0, bytemuck::cast_slice(&slots))?;
            be.upload(&maps[1], 0, bytemuck::cast_slice(&logits_rows))?;
            device.count = rows.len();
            device.slot_end = rows.iter().map(|r| r.slot as usize + 1).max().unwrap_or(0);
            let (mut states, mut map, mut logit_map, mut source, mut output) = (
                device.states.base,
                maps[0].base,
                maps[1].base,
                logits.base,
                device.codes.base,
            );
            let (mut count, mut state_capacity, mut output_capacity, mut logits_capacity) = (
                rows.len() as u32,
                device.slots as u32,
                (device.codes.len / 4) as u32,
                n as u32,
            );
            let (mut stride, mut classes, mut stop) = (2057u32, 2049u32, 2048u32);
            let mut args: [*mut std::ffi::c_void; 12] = [
                std::ptr::addr_of_mut!(states).cast(),
                std::ptr::addr_of_mut!(map).cast(),
                std::ptr::addr_of_mut!(logit_map).cast(),
                std::ptr::addr_of_mut!(source).cast(),
                std::ptr::addr_of_mut!(output).cast(),
                std::ptr::addr_of_mut!(count).cast(),
                std::ptr::addr_of_mut!(state_capacity).cast(),
                std::ptr::addr_of_mut!(output_capacity).cast(),
                std::ptr::addr_of_mut!(logits_capacity).cast(),
                std::ptr::addr_of_mut!(stride).cast(),
                std::ptr::addr_of_mut!(classes).cast(),
                std::ptr::addr_of_mut!(stop).cast(),
            ];
            be.launch_kernel(kernel, count.max(1), 256, 65540, &mut args, Some(&stream))
        };
        if checked {
            be.memset_d8_async(device.codes.base, 0xa5, device.codes.len as usize, &stream)
                .unwrap();
            draw(&mut device, &[]).unwrap();
            be.stream_synchronize(&stream).unwrap();
            let mut canary = vec![0; device.codes.len as usize];
            be.download(&device.codes, 0, &mut canary).unwrap();
            assert!(canary.iter().all(|&b| b == 0xa5));
            be.memset_d8_async(device.codes.base, 0, device.codes.len as usize, &stream)
                .unwrap();
        }

        let mut raw = vec![0u16; n * 2057];
        let mut codes = vec![0u32; n + 7];
        let mut expected = vec![0u32; n];
        let mut scratch = Vec::new();
        let ticks = if sanitize { 12u32 } else { 48u32 };
        for tick in 0..ticks {
            for row in &rows {
                let r = row.logits as usize;
                for i in 0..2049 {
                    let value = if row.slot % 7 == 0 {
                        if i == 17 {
                            12.
                        } else {
                            -12.
                        }
                    } else if tick >= 16 && i == 2048 {
                        8.
                    } else {
                        (((r * 193 + i * 997 + tick as usize * 31) % 2001) as f32 - 1000.) / 211.
                    };
                    raw[r * 2057 + i] = (value.to_bits() >> 16) as u16;
                }
                let slot = &mut host[row.slot as usize];
                let t = slot.tick(&raw[r * 2057..r * 2057 + 2049], &mut scratch);
                expected[row.slot as usize] = match t {
                    Tick::Stop => 2048,
                    Tick::Code(c) => {
                        slot.emitted(c);
                        c
                    }
                };
            }
            be.upload(&logits, 0, bytemuck::cast_slice(&raw)).unwrap();
            draw(&mut device, &rows).unwrap();
            be.stream_synchronize(&stream).unwrap();
            be.download(&device.codes, 0, bytemuck::cast_slice_mut(&mut codes))
                .unwrap();
            for row in &rows {
                assert_eq!(
                    codes[row.output as usize], host[row.slot as usize].last,
                    "sample tick {tick} slot {}",
                    row.slot
                );
            }
            let mut frame_codes = vec![0x12345678u32; n * 32];
            for row in &rows {
                frame_codes[row.slot as usize * 32] = codes[row.output as usize];
            }
            be.upload(&final_codes, 0, bytemuck::cast_slice(&frame_codes))
                .unwrap();
            device
                .finish(&be, &stream, final_codes.base, n, 32, 2048)
                .unwrap();
            be.stream_synchronize(&stream).unwrap();
            let mut canonical = vec![0u32; n * 32];
            be.download(&final_codes, 0, bytemuck::cast_slice_mut(&mut canonical))
                .unwrap();
            assert_eq!(
                canonical, frame_codes,
                "stop headers clobbered model feedback"
            );
            be.download(
                &device.readback_codes,
                0,
                bytemuck::cast_slice_mut(&mut frame_codes),
            )
            .unwrap();
            for r in 0..n {
                assert_eq!(
                    frame_codes[r * 32],
                    expected[r],
                    "stop tick {tick} slot {r}"
                );
                assert!(frame_codes[r * 32 + 1..(r + 1) * 32]
                    .iter()
                    .all(|&c| c == if expected[r] == 2048 { 0 } else { 0x12345678 }));
            }
            assert!(codes[n..].iter().all(|&code| code == 0));
            rows.rotate_left(1);
            if checked {
                for (i, row) in rows.iter_mut().enumerate() {
                    row.output = i as u32;
                }
            }
        }
        if checked {
            for b in [1, 8, 32, 64, 128, 200, 224, 300, 320]
                .into_iter()
                .filter(|&b| b <= n)
            {
                be.download(&device.codes, 0, bytemuck::cast_slice_mut(&mut codes))
                    .unwrap();
                let before = codes.clone();
                for row in &rows[..b] {
                    let r = row.logits as usize;
                    let slot = &mut host[row.slot as usize];
                    if let Tick::Code(code) =
                        slot.tick(&raw[r * 2057..r * 2057 + 2049], &mut scratch)
                    {
                        slot.emitted(code);
                    }
                }
                draw(&mut device, &rows[..b]).unwrap();
                be.stream_synchronize(&stream).unwrap();
                be.download(&device.codes, 0, bytemuck::cast_slice_mut(&mut codes))
                    .unwrap();
                for row in &rows[..b] {
                    assert_eq!(codes[row.output as usize], host[row.slot as usize].last);
                }
                assert_eq!(&codes[b..], &before[b..]);
                println!("CHECKED_SAMPLE_PREFIX rows={b} states={n} exact cb0/padding");
            }
        }
        if !sanitize && !checked {
            let start = be.event_create(true).unwrap();
            let end = be.event_create(true).unwrap();
            for b in [1, 8, 32, 64, 128, 200, 224] {
                let mut times = Vec::new();
                for _ in 0..4 {
                    let cpu_start = std::time::Instant::now();
                    let mut draw_scratch = Vec::new();
                    for row in &rows[..b] {
                        let r = row.logits as usize;
                        let slot = &mut host[row.slot as usize];
                        if let Tick::Code(c) =
                            slot.tick(&raw[r * 2057..r * 2057 + 2049], &mut draw_scratch)
                        {
                            slot.emitted(c);
                        }
                    }
                    let cpu_us = cpu_start.elapsed().as_secs_f64() * 1e6;
                    be.event_record(&start, &stream).unwrap();
                    device
                        .enqueue(&be, &stream, &rows[..b], logits.base, 2057, 2049, 2048)
                        .unwrap();
                    be.event_record(&end, &stream).unwrap();
                    be.stream_synchronize(&stream).unwrap();
                    let gpu_us = f64::from(be.event_elapsed_ms(&start, &end).unwrap()) * 1000.;
                    times.push((cpu_us, gpu_us));
                }
                println!("CONTROL_TIMING batch={b} cpu_serial_us_gpu_sequence_us={times:?}");
            }
        }
        rows[1] = rows[0];
        assert!(device
            .enqueue(&be, &stream, &rows, logits.base, 2057, 2049, 2048)
            .is_err());
        println!("PASS {} host/device controller transitions, permutations, ties, guard, stops and tails", n * ticks as usize);
    }
}
