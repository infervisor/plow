use crate::device::cuda::{CudaBackend, CudaStream, KernelFn, PinnedHost};
use crate::device::{Backend, DeviceMem, Module};
use crate::{Result, RuntimeError};
use std::sync::Arc;

#[repr(C)]
#[derive(Clone, Copy)]
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
#[derive(Clone, Copy)]
struct State {
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

fn state(p: crate::tts::maya2::Plan) -> Result<State> {
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
        // State has no padding: the C layout sizes are checked on both sides of the ABI.
        let bytes = unsafe {
            std::slice::from_raw_parts(
                (&s as *const State).cast::<u8>(),
                std::mem::size_of::<State>(),
            )
        };
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
        for row in rows {
            if std::mem::replace(&mut self.seen_slots[row.slot as usize], true)
                || std::mem::replace(&mut self.seen_outputs[row.output as usize], true)
                || row.logits as usize >= self.slots
            {
                return Err(RuntimeError::Device(
                    "controlled sampler rows overlap or exceed logits".into(),
                ));
            }
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
        codebooks: u32,
        stop: u32,
    ) -> Result<()> {
        if codebooks as usize != self.codebooks {
            return Err(RuntimeError::Rejected(
                "controller frame width changed".into(),
            ));
        }
        be.memcpy_dtod_async(
            self.readback_codes.base,
            codes,
            self.readback_codes.len,
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
    fn controlled_sampler_device_matches_host() {
        let be = CudaBackend::new(0).unwrap();
        let image = std::fs::read(std::env::var("CONTROL_SAMPLE_CUBIN").unwrap()).unwrap();
        let module = be.module_load(&image).unwrap();
        let stream = be.stream_create().unwrap();
        let sanitize = std::env::var_os("CONTROL_SAMPLE_SANITIZE").is_some();
        let production = std::env::var_os("CONTROL_SAMPLE_PRODUCTION").is_some();
        let n = if sanitize { 33usize } else { 224usize };
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
            .map(|r| Row {
                slot: r as u32,
                output: ((r + 7) % n) as u32,
                logits: ((r + 13) % n) as u32,
            })
            .collect();
        let logits = be.alloc(0, (n * 2057 * 2) as u64).unwrap();
        let final_codes = be.alloc(0, (n * 32 * 4) as u64).unwrap();
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
            device
                .enqueue(&be, &stream, &rows, logits.base, 2057, 2049, 2048)
                .unwrap();
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
                .finish(&be, &stream, final_codes.base, 32, 2048)
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
            rows.rotate_left(1);
        }
        if !sanitize {
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
