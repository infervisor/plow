//! Bucket demand: the row counts live traffic launches with, against the capacities the packet
//! declares for each launch family. A row count with no capacity of its own runs the next larger
//! one and pays for the padding; `/metrics` and a periodic log line name those row counts so the
//! capacities can be added at the next emit (`scripts/campaign/bucket_demand.py`).
//!
//! Recording is relaxed atomic adds into arrays sized at declaration: no allocation, lock, clock
//! read or formatting on a launch path.

use std::fmt::Write as _;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use super::serving::family as metric_family;
use super::Metrics;

/// A launch kind whose capacities the packet declares.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Family {
    /// Decode token batch: rows = occupied slot extent, capacities = the decode rungs.
    Decode,
    /// Packed prefill launch: rows = prompt rows plus riders, capacities = the prefill buckets.
    Prefill,
    /// Speech depth program: rows = slot extent of the frame, capacities = the depth programs.
    Depth,
    /// Stateful codec step: rows = streams in the launch, capacities = the step batches.
    Codec,
}

impl Family {
    pub const ALL: [Self; 4] = [Self::Decode, Self::Prefill, Self::Depth, Self::Codec];

    pub fn label(self) -> &'static str {
        match self {
            Self::Decode => "decode",
            Self::Prefill => "prefill",
            Self::Depth => "depth",
            Self::Codec => "codec",
        }
    }
}

/// A ladder whose widest capacity is at most this keeps one demand bin per row count.
const EXACT_MAX: usize = 256;
/// Wider ladders keep exact bins up to this row count and [`OCTAVE_BINS`] equal bins per octave
/// above it: bin upper edges `2^e * (1 + j/8)`, `e >= 6`, `j = 1..=8` (72, 80, .. 128, 144, ..).
const FINE_MAX: usize = 64;
const OCTAVE_BINS: usize = 8;
/// Undeclared row counts exported as `plowrt_bucket_missing_score`, and named in a log line.
const MISSING_SERIES: usize = 8;
const MISSING_LOGGED: usize = 4;
/// A log line is written when at least this fraction of the interval's launched rows was padding.
pub const LOG_PADDED_FRAC: f64 = 0.05;

#[inline]
fn bin(rows: usize, exact: usize) -> usize {
    if rows <= exact {
        return rows;
    }
    let e = (rows - 1).ilog2() as usize;
    let j = (rows - (1 << e) + (1 << (e - 3)) - 1) >> (e - 3);
    FINE_MAX + (e - 6) * OCTAVE_BINS + j
}

/// The largest row count of bin `index` (its `rows` label).
fn bin_rows(index: usize, exact: usize) -> usize {
    if index <= exact {
        return index;
    }
    let k = index - FINE_MAX - 1;
    let e = 6 + k / OCTAVE_BINS;
    (1 << e) + (k % OCTAVE_BINS + 1) * (1 << (e - 3))
}

#[derive(Default)]
struct CapStat {
    launches: AtomicU64,
    padded: AtomicU64,
    timed: AtomicU64,
    us: AtomicU64,
}

#[derive(Clone, Default)]
struct Counts {
    live: Vec<u64>,
    launches: Vec<u64>,
    padded: Vec<u64>,
    overflow: u64,
}

impl Counts {
    fn since(&self, prev: &Self) -> Self {
        let sub = |a: &[u64], b: &[u64]| {
            a.iter()
                .enumerate()
                .map(|(i, &v)| v.saturating_sub(b.get(i).copied().unwrap_or(0)))
                .collect()
        };
        Self {
            live: sub(&self.live, &prev.live),
            launches: sub(&self.launches, &prev.launches),
            padded: sub(&self.padded, &prev.padded),
            overflow: self.overflow.saturating_sub(prev.overflow),
        }
    }
}

struct Ladder {
    /// Ascending, distinct, non-zero.
    caps: Box<[usize]>,
    widest: usize,
    exact: usize,
    live: Box<[AtomicU64]>,
    stats: Box<[CapStat]>,
    overflow: AtomicU64,
    max: AtomicU64,
    /// The counts at the last log line; only the logger takes it.
    logged: parking_lot::Mutex<Counts>,
}

impl Ladder {
    fn new(mut caps: Vec<usize>) -> Option<Self> {
        caps.retain(|&c| c > 0);
        caps.sort_unstable();
        caps.dedup();
        let widest = *caps.last()?;
        let exact = if widest <= EXACT_MAX {
            widest
        } else {
            FINE_MAX
        };
        Some(Self {
            live: (0..=bin(widest, exact))
                .map(|_| AtomicU64::new(0))
                .collect(),
            stats: caps.iter().map(|_| CapStat::default()).collect(),
            caps: caps.into(),
            widest,
            exact,
            overflow: AtomicU64::new(0),
            max: AtomicU64::new(0),
            logged: Default::default(),
        })
    }

    #[inline]
    fn demand(&self, live: usize, n: u64) {
        if live as u64 > self.max.load(Relaxed) {
            self.max.fetch_max(live as u64, Relaxed);
        }
        if live > self.widest {
            self.overflow.fetch_add(n, Relaxed);
        } else {
            self.live[bin(live, self.exact)].fetch_add(n, Relaxed);
        }
    }

    #[inline]
    fn launch(&self, cap: usize, live: usize, n: u64, us: Option<u64>) {
        let Some(at) = self.caps.iter().position(|&c| c == cap) else {
            return;
        };
        let s = &self.stats[at];
        s.launches.fetch_add(n, Relaxed);
        s.padded
            .fetch_add(cap.saturating_sub(live) as u64 * n, Relaxed);
        if let Some(us) = us {
            s.timed.fetch_add(n, Relaxed);
            s.us.fetch_add(us, Relaxed);
        }
    }

    fn counts(&self) -> Counts {
        let load = |a: &AtomicU64| a.load(Relaxed);
        Counts {
            live: self.live.iter().map(load).collect(),
            launches: self.stats.iter().map(|s| load(&s.launches)).collect(),
            padded: self.stats.iter().map(|s| load(&s.padded)).collect(),
            overflow: load(&self.overflow),
        }
    }

    fn rows(&self, index: usize) -> usize {
        bin_rows(index, self.exact).min(self.widest)
    }

    fn cover(&self, rows: usize) -> usize {
        self.caps
            .iter()
            .copied()
            .find(|&c| c >= rows)
            .unwrap_or(self.widest)
    }

    /// Undeclared row counts by the share of all launched rows that was their padding, had each
    /// frame run the smallest capacity holding it: `n(rows) * (cover(rows) - rows) / sum(n * cover)`.
    fn missing(&self, live: &[u64], top: usize) -> Vec<(usize, f64)> {
        let demand = || {
            live.iter()
                .enumerate()
                .skip(1)
                .filter(|(_, &n)| n > 0)
                .map(|(i, &n)| (self.rows(i), n))
        };
        let launched: u64 = demand().map(|(rows, n)| n * self.cover(rows) as u64).sum();
        let mut waste: Vec<(usize, u64)> = demand()
            .map(|(rows, n)| (rows, n * (self.cover(rows) - rows) as u64))
            .filter(|&(_, w)| w > 0)
            .collect();
        waste.sort_unstable_by_key(|&(rows, w)| (std::cmp::Reverse(w), rows));
        waste.truncate(top);
        waste
            .into_iter()
            .map(|(rows, w)| (rows, w as f64 / launched as f64))
            .collect()
    }

    /// The interval's line, when it launched anything and padded past the threshold or
    /// overflowed the ladder.
    fn log_line(&self, family: Family) -> Option<String> {
        let now = self.counts();
        let d = now.since(&std::mem::replace(&mut *self.logged.lock(), now.clone()));
        let launches: u64 = d.launches.iter().sum();
        let launched: u64 = d
            .launches
            .iter()
            .zip(&self.caps[..])
            .map(|(&n, &c)| n * c as u64)
            .sum();
        let padded: u64 = d.padded.iter().sum();
        let frac = if launched == 0 {
            0.0
        } else {
            padded as f64 / launched as f64
        };
        if launches == 0 || (frac < LOG_PADDED_FRAC && d.overflow == 0) {
            return None;
        }
        let list = |items: Vec<String>| items.join(",");
        Some(format!(
            "bucket demand: family={} declared=[{}] launches={launches} padded_frac={frac:.2} overflow={} missing=[{}]",
            family.label(),
            list(self.caps.iter().map(|c| c.to_string()).collect()),
            d.overflow,
            list(
                self.missing(&d.live, MISSING_LOGGED)
                    .into_iter()
                    .map(|(rows, score)| format!("{rows}:{score:.2}"))
                    .collect()
            ),
        ))
    }
}

/// One model's demand counters, one ladder per declared family.
#[derive(Default)]
pub struct BucketDemand {
    families: [OnceLock<Ladder>; Family::ALL.len()],
}

impl BucketDemand {
    /// Declare `family`'s capacities (at load). The first non-empty declaration stands: a later
    /// one is ignored, and launches at a capacity it lacks are not counted.
    pub fn declare(&self, family: Family, caps: impl IntoIterator<Item = usize>) {
        let slot = &self.families[family as usize];
        if slot.get().is_none() {
            if let Some(ladder) = Ladder::new(caps.into_iter().collect()) {
                let _ = slot.set(ladder);
            }
        }
    }

    /// A frame of `live` rows asked for a launch.
    #[inline]
    pub fn record_demand(&self, family: Family, live: usize) {
        if let Some(l) = self.families[family as usize].get() {
            l.demand(live, 1);
        }
    }

    /// One launch of capacity `cap` ran `live` rows, in `us` when the caller measured it.
    #[inline]
    pub fn record_launch(&self, family: Family, cap: usize, live: usize, us: Option<u64>) {
        if let Some(l) = self.families[family as usize].get() {
            l.launch(cap, live, 1, us);
        }
    }

    /// A frame served by one launch.
    #[inline]
    pub fn record(&self, family: Family, live: usize, cap: usize, us: Option<u64>) {
        self.record_n(family, live, cap, 1, us);
    }

    /// `n` consecutive frames of `live` rows, each one launch of `cap`; `us` is their total.
    #[inline]
    pub fn record_n(&self, family: Family, live: usize, cap: usize, n: u64, us: Option<u64>) {
        if let Some(l) = self.families[family as usize].get() {
            l.demand(live, n);
            l.launch(cap, live, n, us);
        }
    }

    fn ladders(&self) -> impl Iterator<Item = (Family, &Ladder)> {
        Family::ALL
            .into_iter()
            .filter_map(|f| Some((f, self.families[f as usize].get()?)))
    }

    /// The lines of the interval since the last call (see [`Ladder::log_line`]).
    pub fn log_lines(&self) -> Vec<String> {
        self.ladders().filter_map(|(f, l)| l.log_line(f)).collect()
    }

    pub fn log(&self, slug: &str) {
        for line in self.log_lines() {
            tracing::info!(%slug, "{line}");
        }
    }
}

/// When the next interval line is due (`PLOW_BUCKET_LOG_S`; 0 = never). Fed the dispatcher's own
/// timestamps, so it reads no clock.
pub struct LogTicker {
    every: Option<Duration>,
    next: Instant,
}

impl LogTicker {
    pub fn new(every_s: u32, now: Instant) -> Self {
        let every = (every_s > 0).then(|| Duration::from_secs(u64::from(every_s)));
        Self {
            every,
            next: now + every.unwrap_or_default(),
        }
    }

    pub fn enabled(&self) -> bool {
        self.every.is_some()
    }

    #[inline]
    pub fn due(&mut self, now: Instant) -> bool {
        match self.every {
            Some(every) if now >= self.next => {
                self.next = now + every;
                true
            }
            _ => false,
        }
    }
}

/// The `plowrt_bucket_*` series of every model; `labels` are the models' label sets in order.
pub fn write(out: &mut String, models: &[(String, Arc<Metrics>, bool)], labels: &[String]) {
    let ladders: Vec<(String, &Ladder, Counts)> = models
        .iter()
        .zip(labels)
        .flat_map(|((_, m, _), labels)| {
            m.buckets
                .ladders()
                .map(move |(f, l)| (format!("{labels},family=\"{}\"", f.label()), l, l.counts()))
        })
        .collect();
    let mut series =
        |name: &str, kind: &str, help: &str, emit: &dyn Fn(&mut String, &str, &Ladder, &Counts)| {
            metric_family(out, name, kind, help);
            for (labels, ladder, counts) in &ladders {
                emit(out, labels, ladder, counts);
            }
        };
    series(
        "plowrt_bucket_capacity_info",
        "gauge",
        "Launch capacities the loaded packet declares, per family.",
        &|out, labels, l, _| {
            for c in &l.caps[..] {
                let _ = writeln!(
                    out,
                    "plowrt_bucket_capacity_info{{{labels},capacity=\"{c}\"}} 1"
                );
            }
        },
    );
    series(
        "plowrt_bucket_demand_total",
        "counter",
        "Frames whose live row count was `rows` (exact up to 256-row ladders; wider ladders bin rows above 64 to 8 bins per octave, labelled by the bin's largest row count).",
        &|out, labels, l, c| {
            for (i, &n) in c.live.iter().enumerate().filter(|(_, &n)| n > 0) {
                let _ = writeln!(
                    out,
                    "plowrt_bucket_demand_total{{{labels},rows=\"{}\"}} {n}",
                    l.rows(i)
                );
            }
        },
    );
    series(
        "plowrt_bucket_launches_total",
        "counter",
        "Launches run at each declared capacity.",
        &|out, labels, l, c| {
            for (cap, n) in l.caps.iter().zip(&c.launches) {
                let _ = writeln!(
                    out,
                    "plowrt_bucket_launches_total{{{labels},capacity=\"{cap}\"}} {n}"
                );
            }
        },
    );
    series(
        "plowrt_bucket_padded_rows_total",
        "counter",
        "Rows launched beyond the live rows (capacity - live), per declared capacity.",
        &|out, labels, l, c| {
            for (cap, n) in l.caps.iter().zip(&c.padded) {
                let _ = writeln!(
                    out,
                    "plowrt_bucket_padded_rows_total{{{labels},capacity=\"{cap}\"}} {n}"
                );
            }
        },
    );
    for (name, help, seconds) in [
        (
            "plowrt_bucket_launch_seconds_sum",
            "Measured time of the timed launches at each capacity (only where the launch path already times itself).",
            true,
        ),
        (
            "plowrt_bucket_launch_seconds_count",
            "Launches behind plowrt_bucket_launch_seconds_sum.",
            false,
        ),
    ] {
        series(name, "counter", help, &|out, labels, l, _| {
            for (cap, s) in l.caps.iter().zip(&l.stats[..]) {
                let timed = s.timed.load(Relaxed);
                if timed == 0 {
                    continue;
                }
                let _ = if seconds {
                    writeln!(
                        out,
                        "{name}{{{labels},capacity=\"{cap}\"}} {}",
                        s.us.load(Relaxed) as f64 / 1e6
                    )
                } else {
                    writeln!(out, "{name}{{{labels},capacity=\"{cap}\"}} {timed}")
                };
            }
        });
    }
    series(
        "plowrt_bucket_overflow_total",
        "counter",
        "Frames with more live rows than the widest declared capacity.",
        &|out, labels, _, c| {
            let _ = writeln!(
                out,
                "plowrt_bucket_overflow_total{{{labels}}} {}",
                c.overflow
            );
        },
    );
    series(
        "plowrt_bucket_demand_max",
        "gauge",
        "Largest live row count seen.",
        &|out, labels, l, _| {
            let _ = writeln!(
                out,
                "plowrt_bucket_demand_max{{{labels}}} {}",
                l.max.load(Relaxed)
            );
        },
    );
    series(
        "plowrt_bucket_missing_score",
        "gauge",
        "Top undeclared row counts: share of all launched rows that was their padding since start (smallest covering capacity assumed).",
        &|out, labels, l, c| {
            for (rows, score) in l.missing(&c.live, MISSING_SERIES) {
                let _ = writeln!(
                    out,
                    "plowrt_bucket_missing_score{{{labels},rows=\"{rows}\"}} {score:.4}"
                );
            }
        },
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    const SPEECH: [usize; 7] = [1, 2, 4, 8, 16, 32, 64];

    fn speech_at_20() -> BucketDemand {
        let b = BucketDemand::default();
        b.declare(Family::Depth, SPEECH);
        for _ in 0..900 {
            b.record(Family::Depth, 20, 32, None);
        }
        for _ in 0..100 {
            b.record(Family::Depth, 12, 16, None);
        }
        b
    }

    fn text(b: BucketDemand) -> String {
        let m = Arc::new(Metrics {
            buckets: Arc::new(b),
            ..Default::default()
        });
        let mut out = String::new();
        write(
            &mut out,
            &[("tts".into(), m, true)],
            &[super::super::serving::model_labels("tts")],
        );
        out
    }

    #[test]
    fn twenty_rows_on_a_power_of_two_ladder_name_the_missing_bucket() {
        let b = speech_at_20();
        assert_eq!(
            b.log_lines(),
            ["bucket demand: family=depth declared=[1,2,4,8,16,32,64] launches=1000 padded_frac=0.37 overflow=0 missing=[20:0.36,12:0.01]"]
        );
        assert!(b.log_lines().is_empty(), "an idle interval logs nothing");
        let out = text(b);
        let l = r#"model_name="tts",engine="0",family="depth""#;
        for line in [
            format!("plowrt_bucket_demand_total{{{l},rows=\"20\"}} 900"),
            format!("plowrt_bucket_demand_total{{{l},rows=\"12\"}} 100"),
            format!("plowrt_bucket_capacity_info{{{l},capacity=\"64\"}} 1"),
            format!("plowrt_bucket_launches_total{{{l},capacity=\"32\"}} 900"),
            format!("plowrt_bucket_launches_total{{{l},capacity=\"64\"}} 0"),
            format!("plowrt_bucket_padded_rows_total{{{l},capacity=\"32\"}} 10800"),
            format!("plowrt_bucket_padded_rows_total{{{l},capacity=\"16\"}} 400"),
            format!("plowrt_bucket_overflow_total{{{l}}} 0"),
            format!("plowrt_bucket_demand_max{{{l}}} 20"),
            format!("plowrt_bucket_missing_score{{{l},rows=\"20\"}} 0.3553"),
            format!("plowrt_bucket_missing_score{{{l},rows=\"12\"}} 0.0132"),
        ] {
            assert!(out.lines().any(|x| x == line), "missing {line}\n{out}");
        }
        assert_eq!(out.matches("plowrt_bucket_demand_total{").count(), 2);
        assert!(!out.contains("launch_seconds_sum{"), "nothing was timed");
        assert_eq!(out.matches("# TYPE plowrt_bucket_").count(), 9);
    }

    #[test]
    fn declared_capacities_are_never_missing_and_only_the_top_eight_export() {
        let b = BucketDemand::default();
        b.declare(Family::Decode, [64, 0, 16, 32, 16]);
        for rows in 1..=64 {
            for _ in 0..rows {
                b.record(
                    Family::Decode,
                    rows,
                    [16, 32, 64][(rows > 16) as usize + (rows > 32) as usize],
                    None,
                );
            }
        }
        let l = b.families[Family::Decode as usize].get().unwrap();
        assert_eq!(&l.caps[..], [16, 32, 64]);
        let missing = l.missing(&l.counts().live, MISSING_SERIES);
        assert_eq!(missing.len(), 8);
        assert!(missing.iter().all(|(rows, _)| ![16, 32, 64].contains(rows)));
        // n * (cover - n) peaks around half the 64 rung's span.
        assert_eq!(missing[0].0, 33);
        assert!(missing.windows(2).all(|w| w[0].1 >= w[1].1));
        let all = l.missing(&l.counts().live, usize::MAX);
        let padded: u64 = l.counts().padded.iter().sum();
        let launched: u64 = (1..=64u64).map(|r| r * l.cover(r as usize) as u64).sum();
        let total: f64 = all.iter().map(|x| x.1).sum();
        assert!((total - padded as f64 / launched as f64).abs() < 1e-9);
        assert_eq!(text(b).matches("plowrt_bucket_missing_score{").count(), 8);
    }

    #[test]
    fn wide_ladders_bin_rows_above_64_and_stay_under_the_series_bound() {
        for rows in 1..=70_000 {
            let i = bin(rows, FINE_MAX);
            let upper = bin_rows(i, FINE_MAX);
            assert!(rows <= upper && (i == 0 || bin_rows(i - 1, FINE_MAX) < rows));
            assert!(
                rows <= 64 || (upper - rows) * 8 < upper,
                "{rows} in {upper}"
            );
        }
        assert_eq!(
            (65..=73).map(|i| bin_rows(i, FINE_MAX)).collect::<Vec<_>>(),
            [72, 80, 88, 96, 104, 112, 120, 128, 144]
        );
        let b = BucketDemand::default();
        b.declare(Family::Prefill, [512, 1024, 4160, 8192]);
        let l = b.families[Family::Prefill as usize].get().unwrap();
        assert_eq!(l.live.len(), 65 + 7 * 8);
        b.record(Family::Prefill, 4100, 4160, None);
        b.record(Family::Prefill, 8192, 8192, None);
        b.record(Family::Prefill, 300, 512, None);
        b.record_demand(Family::Prefill, 9000);
        let out = text(b);
        for rows in [320, 4608, 8192] {
            assert!(
                out.contains(&format!("family=\"prefill\",rows=\"{rows}\"}} 1")),
                "{out}"
            );
        }
        assert!(out.contains(
            "plowrt_bucket_overflow_total{model_name=\"tts\",engine=\"0\",family=\"prefill\"} 1"
        ));
        assert!(out.contains(
            "plowrt_bucket_demand_max{model_name=\"tts\",engine=\"0\",family=\"prefill\"} 9000"
        ));

        let narrow = BucketDemand::default();
        narrow.declare(Family::Decode, [256]);
        let l = narrow.families[Family::Decode as usize].get().unwrap();
        assert_eq!((l.exact, l.live.len()), (256, 257));
    }

    #[test]
    fn a_composed_frame_counts_demand_once_and_each_launch_its_own_padding() {
        let b = BucketDemand::default();
        b.declare(Family::Depth, SPEECH);
        b.record_demand(Family::Depth, 20);
        b.record_launch(Family::Depth, 16, 16, Some(3_000));
        b.record_launch(Family::Depth, 4, 4, Some(1_000));
        b.record_n(Family::Depth, 3, 4, 5, Some(10_000));
        b.record_launch(Family::Depth, 24, 20, None);
        let out = text(b);
        let l = r#"model_name="tts",engine="0",family="depth""#;
        for line in [
            format!("plowrt_bucket_demand_total{{{l},rows=\"20\"}} 1"),
            format!("plowrt_bucket_demand_total{{{l},rows=\"3\"}} 5"),
            format!("plowrt_bucket_launches_total{{{l},capacity=\"16\"}} 1"),
            format!("plowrt_bucket_launches_total{{{l},capacity=\"4\"}} 6"),
            format!("plowrt_bucket_padded_rows_total{{{l},capacity=\"4\"}} 5"),
            format!("plowrt_bucket_launch_seconds_sum{{{l},capacity=\"4\"}} 0.011"),
            format!("plowrt_bucket_launch_seconds_count{{{l},capacity=\"4\"}} 6"),
            format!("plowrt_bucket_launch_seconds_sum{{{l},capacity=\"16\"}} 0.003"),
        ] {
            assert!(out.lines().any(|x| x == line), "missing {line}\n{out}");
        }
        assert_eq!(out.matches("launch_seconds_sum{").count(), 2);
    }

    #[test]
    fn the_log_covers_one_interval_and_stays_quiet_below_the_threshold() {
        let b = BucketDemand::default();
        b.declare(Family::Codec, [1, 2, 4, 8]);
        b.declare(Family::Codec, [3]);
        for _ in 0..100 {
            b.record(Family::Codec, 8, 8, None);
        }
        b.record(Family::Codec, 7, 8, None);
        assert!(b.log_lines().is_empty(), "1/808 padded");
        for _ in 0..10 {
            b.record(Family::Codec, 5, 8, None);
        }
        assert_eq!(
            b.log_lines(),
            ["bucket demand: family=codec declared=[1,2,4,8] launches=10 padded_frac=0.38 overflow=0 missing=[5:0.38]"]
        );
        b.record(Family::Codec, 8, 8, None);
        b.record_demand(Family::Codec, 11);
        assert_eq!(
            b.log_lines(),
            ["bucket demand: family=codec declared=[1,2,4,8] launches=1 padded_frac=0.00 overflow=1 missing=[]"]
        );
        b.record(Family::Prefill, 5, 8, None);
        assert!(
            b.log_lines().is_empty(),
            "an undeclared family records nothing"
        );
    }

    #[test]
    fn the_ticker_fires_once_per_interval_and_never_when_off() {
        let t0 = Instant::now();
        let mut t = LogTicker::new(300, t0);
        assert!(t.enabled() && !t.due(t0 + Duration::from_secs(299)));
        assert!(t.due(t0 + Duration::from_secs(300)));
        assert!(!t.due(t0 + Duration::from_secs(301)));
        assert!(t.due(t0 + Duration::from_secs(600)));
        let mut off = LogTicker::new(0, t0);
        assert!(!off.enabled() && !off.due(t0 + Duration::from_secs(86_400)));
    }

    #[test]
    fn engines_of_one_model_keep_their_own_series() {
        let rank = |rows| {
            let b = BucketDemand::default();
            b.declare(Family::Decode, [8, 16]);
            b.record(Family::Decode, rows, 16, None);
            Arc::new(Metrics {
                buckets: Arc::new(b),
                ..Default::default()
            })
        };
        let models = [
            ("g#0".to_string(), rank(9), true),
            ("g#1".to_string(), rank(10), true),
        ];
        let labels: Vec<String> = models
            .iter()
            .map(|(k, _, _)| super::super::serving::model_labels(k))
            .collect();
        let mut out = String::new();
        write(&mut out, &models, &labels);
        assert!(out.contains(
            r#"plowrt_bucket_demand_total{model_name="g",engine="0",family="decode",rows="9"} 1"#
        ));
        assert!(out.contains(
            r#"plowrt_bucket_demand_total{model_name="g",engine="1",family="decode",rows="10"} 1"#
        ));
        assert_eq!(out.matches("# TYPE plowrt_bucket_demand_total").count(), 1);
    }
}
