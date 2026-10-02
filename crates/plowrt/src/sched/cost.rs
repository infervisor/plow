//! Online device-cost model for turn deadlines: an EWMA per (model, op, power-of-two size bucket),
//! fed by timings the serving paths already measure (mux ticks, vocoder renders, ASR encoder
//! launches). Lock-free on the sample and estimate paths except one uncontended read lock to
//! resolve a model name; [`id`] resolves it once for callers that keep the index.
//!
//! An estimate interpolates between the nearest warm buckets around the requested size. With one
//! warm side it scales that bucket by the cold curve's shape; with none it is the cold curve.
//! A size of 0 means "unknown size": the op's all-sizes EWMA (or the cold curve at a typical size).

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

#[derive(Clone, Copy, Debug)]
pub enum Op {
    Prefill { rows: usize },
    DecodeTick { width: usize },
    Render { streams: usize },
    Encode { audio_ms: u32 },
}

impl Op {
    fn kind(self) -> usize {
        match self {
            Op::Prefill { .. } => 0,
            Op::DecodeTick { .. } => 1,
            Op::Render { .. } => 2,
            Op::Encode { .. } => 3,
        }
    }

    fn size(self) -> f64 {
        match self {
            Op::Prefill { rows } => rows as f64,
            Op::DecodeTick { width } => width as f64,
            Op::Render { streams } => streams as f64,
            Op::Encode { audio_ms } => audio_ms as f64,
        }
    }

    /// Cold curve `floor + per_unit * size` (ms), and the size an unknown-size estimate assumes.
    /// Render: a 16-stream launch measured ~440 ms on H100 (Chatterbox S3Gen, `tts_turn_batch` 16).
    fn cold(self) -> (f64, f64, f64) {
        match self {
            Op::Prefill { .. } => (4.0, 0.04, 256.0),
            Op::DecodeTick { .. } => (8.0, 0.1, 8.0),
            Op::Render { .. } => (40.0, 25.0, 1.0),
            Op::Encode { .. } => (5.0, 0.01, 2000.0),
        }
    }

    fn cold_ms(self, size: f64) -> f64 {
        let (floor, per, _) = self.cold();
        floor + per * size
    }
}

const KINDS: usize = 4;
/// Size buckets `[2^b, 2^(b+1))`, b < BUCKETS; the last one also takes everything larger.
const BUCKETS: usize = 18;
const ANY: usize = BUCKETS;
/// EWMA weight of a new sample.
const ALPHA: f64 = 0.125;

/// One EWMA point: mean time (ms) and mean size of the samples in a bucket. 0 bits = cold.
#[derive(Default)]
struct Cell {
    ms: AtomicU64,
    size: AtomicU64,
}

impl Cell {
    fn get(&self) -> Option<(f64, f64)> {
        let ms = f64::from_bits(self.ms.load(Ordering::Relaxed));
        (ms > 0.0).then(|| (f64::from_bits(self.size.load(Ordering::Relaxed)), ms))
    }

    /// Racy read-modify-write: concurrent samples of one model may drop one, which an EWMA absorbs.
    fn update(&self, size: f64, ms: f64) {
        let (s, m) = match self.get() {
            Some((s, m)) => (s + ALPHA * (size - s), m + ALPHA * (ms - m)),
            None => (size, ms),
        };
        self.size.store(s.to_bits(), Ordering::Relaxed);
        self.ms.store(m.max(f64::MIN_POSITIVE).to_bits(), Ordering::Relaxed);
    }
}

struct Model {
    name: Box<str>,
    cells: [[Cell; BUCKETS + 1]; KINDS],
}

/// Models are few and live for the process: leaked so readers hold no lock past the lookup.
static MODELS: parking_lot::RwLock<Vec<&'static Model>> = parking_lot::RwLock::new(Vec::new());

/// The model's index in the cost table, registering it on first use.
pub fn id(model: &str) -> usize {
    if let Some(i) = MODELS.read().iter().position(|m| &*m.name == model) {
        return i;
    }
    let mut w = MODELS.write();
    if let Some(i) = w.iter().position(|m| &*m.name == model) {
        return i;
    }
    w.push(Box::leak(Box::new(Model { name: model.into(), cells: Default::default() })));
    w.len() - 1
}

fn model(id: usize) -> Option<&'static Model> {
    MODELS.read().get(id).copied()
}

fn bucket(size: f64) -> usize {
    (size.max(1.0).log2() as usize).min(BUCKETS - 1)
}

pub fn record(model: &str, op: Op, took: Duration) {
    record_id(id(model), op, took);
}

pub fn record_id(id: usize, op: Op, took: Duration) {
    let Some(m) = model(id) else { return };
    let (size, ms) = (op.size(), took.as_secs_f64() * 1e3);
    let cells = &m.cells[op.kind()];
    if size > 0.0 {
        cells[bucket(size)].update(size, ms);
    }
    cells[ANY].update(size, ms);
}

pub fn estimate(model: &str, op: Op) -> Duration {
    estimate_id(id(model), op)
}

pub fn estimate_id(id: usize, op: Op) -> Duration {
    let ms = match model(id) {
        Some(m) => estimate_ms(&m.cells[op.kind()], op),
        None => op.cold_ms(op.cold().2),
    };
    Duration::from_secs_f64(ms.max(0.0) / 1e3)
}

fn estimate_ms(cells: &[Cell; BUCKETS + 1], op: Op) -> f64 {
    let x = op.size();
    if x <= 0.0 {
        return cells[ANY].get().map_or_else(|| op.cold_ms(op.cold().2), |(_, ms)| ms);
    }
    let b = bucket(x);
    let lo = (0..=b).rev().find_map(|i| cells[i].get().filter(|&(s, _)| s <= x));
    let hi = (b..BUCKETS).find_map(|i| cells[i].get().filter(|&(s, _)| s >= x));
    let shaped = |(s, ms): (f64, f64)| ms * op.cold_ms(x) / op.cold_ms(s);
    match (lo, hi) {
        (Some(l), Some(h)) if h.0 - l.0 > f64::EPSILON => l.1 + (h.1 - l.1) * (x - l.0) / (h.0 - l.0),
        (Some(p), _) | (None, Some(p)) => shaped(p),
        (None, None) => op.cold_ms(x),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ms(d: Duration) -> f64 {
        d.as_secs_f64() * 1e3
    }

    #[test]
    fn cold_defaults_follow_the_curve() {
        let m = "cost-test-cold";
        assert!((ms(estimate(m, Op::Render { streams: 16 })) - 440.0).abs() < 1e-6);
        assert!((ms(estimate(m, Op::Prefill { rows: 1000 })) - 44.0).abs() < 1e-6);
        assert!((ms(estimate(m, Op::Prefill { rows: 0 })) - op_typical(Op::Prefill { rows: 0 })).abs() < 1e-6);
    }

    fn op_typical(op: Op) -> f64 {
        op.cold_ms(op.cold().2)
    }

    #[test]
    fn ewma_converges_and_smooths() {
        let m = "cost-test-ewma";
        let op = Op::DecodeTick { width: 8 };
        record(m, op, Duration::from_millis(20));
        assert!((ms(estimate(m, op)) - 20.0).abs() < 1e-6, "first sample seeds the mean");
        record(m, op, Duration::from_millis(28));
        assert!((ms(estimate(m, op)) - 21.0).abs() < 1e-6, "one sample moves it by ALPHA");
        for _ in 0..200 {
            record(m, op, Duration::from_millis(12));
        }
        assert!((ms(estimate(m, op)) - 12.0).abs() < 0.01);
    }

    #[test]
    fn buckets_interpolate_between_warm_points() {
        let m = "cost-test-interp";
        record(m, Op::Render { streams: 2 }, Duration::from_millis(100));
        record(m, Op::Render { streams: 16 }, Duration::from_millis(450));
        assert!((ms(estimate(m, Op::Render { streams: 2 })) - 100.0).abs() < 1e-6);
        assert!((ms(estimate(m, Op::Render { streams: 16 })) - 450.0).abs() < 1e-6);
        // 9 streams sits halfway between the warm points at 2 and 16.
        assert!((ms(estimate(m, Op::Render { streams: 9 })) - 275.0).abs() < 1e-6);
        // Outside the warm range: the nearest point, scaled by the cold curve's shape.
        let one = ms(estimate(m, Op::Render { streams: 1 }));
        assert!((one - 100.0 * 65.0 / 90.0).abs() < 1e-6, "{one}");
        let wide = ms(estimate(m, Op::Render { streams: 32 }));
        assert!((wide - 450.0 * 840.0 / 440.0).abs() < 1e-6, "{wide}");
    }

    #[test]
    fn unknown_size_reads_the_all_sizes_mean_and_models_are_separate() {
        let (a, b) = ("cost-test-a", "cost-test-b");
        record(a, Op::Prefill { rows: 100 }, Duration::from_millis(10));
        record(a, Op::Prefill { rows: 1000 }, Duration::from_millis(10));
        assert!((ms(estimate(a, Op::Prefill { rows: 0 })) - 10.0).abs() < 1e-6);
        assert_eq!(id(a), id(a));
        assert_ne!(id(a), id(b));
        assert!((ms(estimate(b, Op::Prefill { rows: 1000 })) - 44.0).abs() < 1e-6);
        // Encode samples never leak into another op's cells.
        assert!((ms(estimate(a, Op::Encode { audio_ms: 0 })) - op_typical(Op::Encode { audio_ms: 0 })).abs() < 1e-6);
    }
}
