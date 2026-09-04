//! Append-only storage with transactional publication.
//!
//! Two properties matter more than convenience here.
//!
//! **A campaign that dies leaves nothing half-qualified.** Measurements are
//! staged and promoted as one unit; the file on disk either has the whole
//! campaign or none of it. A partially-written winner is worse than no winner,
//! because it is selectable.
//!
//! **Nothing is overwritten.** Records append, and a superseded record becomes
//! stale rather than disappearing. Negative results are kept with their reason,
//! so the next campaign does not spend GPU time rediscovering that a tile does
//! not fit.

use std::collections::BTreeMap;
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

use crate::attention::AttentionMeasurement;
use crate::decode::{rank_by_cell, CellRanking, DecodeMeasurement};
use crate::moe_decode::MoeDecodeMeasurement;
use crate::record::{Correctness, Digests, KernelMeasurement, RecordState};

#[derive(Debug)]
pub enum StoreError {
    Io(std::io::Error),
    Json(serde_json::Error),
    /// A record was staged that cannot legally be promoted.
    NotQualifiable {
        kernel: String,
        blockers: Vec<String>,
    },
}

impl std::fmt::Display for StoreError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StoreError::Io(e) => write!(f, "tuning store io: {e}"),
            StoreError::Json(e) => write!(f, "tuning store json: {e}"),
            StoreError::NotQualifiable { kernel, blockers } => {
                write!(f, "{kernel} cannot be qualified: {}", blockers.join("; "))
            }
        }
    }
}

impl std::error::Error for StoreError {}

impl From<std::io::Error> for StoreError {
    fn from(e: std::io::Error) -> Self {
        StoreError::Io(e)
    }
}

impl From<serde_json::Error> for StoreError {
    fn from(e: serde_json::Error) -> Self {
        StoreError::Json(e)
    }
}

/// One hardware cell's measurement file.
///
/// The path is derived from the fingerprint (`nvidia/sm_90a/h100-nvl`), so
/// records for different hardware cannot land in the same file and be confused
/// for one population.
pub struct TuneStore {
    root: PathBuf,
}

impl TuneStore {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        TuneStore { root: root.into() }
    }

    fn kernel_path(&self, hardware: &str) -> PathBuf {
        self.root.join(hardware).join("kernel_measurement.jsonl")
    }

    fn attention_path(&self, hardware: &str) -> PathBuf {
        self.root.join(hardware).join("attention_measurement.jsonl")
    }

    /// Every hardware cell that has a `kernel_measurement.jsonl`, as `vendor/isa/sku`.
    ///
    /// Exists so an empty cell can be reported as a MISMATCH rather than a cold start: the cell is
    /// a path (`amd/gfx950/mi350x`) while the caller usually holds a SKU (`MI355X`), and the two
    /// need not spell the same. A reader that cannot see the neighbours has no way to tell "nothing
    /// measured yet" from "measured under a different name", and those want opposite responses.
    pub fn cells(&self) -> Result<Vec<String>, StoreError> {
        fn walk(dir: &Path, prefix: &str, out: &mut Vec<String>) -> Result<(), StoreError> {
            let rd = match std::fs::read_dir(dir) {
                Ok(rd) => rd,
                Err(_) => return Ok(()),
            };
            for e in rd {
                let e = e?;
                if !e.file_type()?.is_dir() {
                    continue;
                }
                let name = e.file_name().to_string_lossy().into_owned();
                let cell = if prefix.is_empty() {
                    name
                } else {
                    format!("{prefix}/{name}")
                };
                if e.path().join("kernel_measurement.jsonl").exists() {
                    out.push(cell);
                } else {
                    walk(&e.path(), &cell, out)?;
                }
            }
            Ok(())
        }
        let mut out = Vec::new();
        walk(&self.root, "", &mut out)?;
        out.sort();
        Ok(out)
    }

    /// Every kernel measurement stored for one hardware cell.
    pub fn load_kernels(&self, hardware: &str) -> Result<Vec<KernelMeasurement>, StoreError> {
        let path = self.kernel_path(hardware);
        if !path.exists() {
            return Ok(Vec::new());
        }
        let mut out = Vec::new();
        for line in BufReader::new(File::open(&path)?).lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            out.push(serde_json::from_str(&line)?);
        }
        Ok(out)
    }

    /// Attention policy records are separate from kernel measurements because
    /// their value changes packet scratch/merge operands rather than an opcode.
    pub fn load_attention(&self, hardware: &str) -> Result<Vec<AttentionMeasurement>, StoreError> {
        let path = self.attention_path(hardware);
        if !path.exists() {
            return Ok(Vec::new());
        }
        let mut out = Vec::new();
        for line in BufReader::new(File::open(&path)?).lines() {
            let line = line?;
            if !line.trim().is_empty() {
                out.push(serde_json::from_str(&line)?);
            }
        }
        Ok(out)
    }

    /// Publish attention choices only after the same correctness/sample gates
    /// as every other selectable tuning record.
    pub fn publish_attention(
        &self,
        hardware: &str,
        mut records: Vec<AttentionMeasurement>,
    ) -> Result<usize, StoreError> {
        for r in &records {
            let blockers = r.qualification_blockers();
            if !blockers.is_empty() {
                return Err(StoreError::NotQualifiable {
                    kernel: format!("{}:{:?}:ns{}", r.cell.key(), r.algorithm, r.nsplit),
                    blockers,
                });
            }
        }
        for r in &mut records {
            r.state = RecordState::Qualified;
        }
        self.append_jsonl(&self.attention_path(hardware), &records)?;
        Ok(records.len())
    }

    /// Publish a campaign as one unit.
    ///
    /// Every record is checked before anything is written; if any cannot be
    /// qualified, nothing is. The append itself goes to a temp file that is
    /// renamed over the target, so a crash mid-write leaves the previous
    /// contents intact rather than a truncated final line.
    pub fn publish(
        &self,
        hardware: &str,
        mut records: Vec<KernelMeasurement>,
    ) -> Result<usize, StoreError> {
        for r in &records {
            let blockers = r.qualification_blockers();
            if !blockers.is_empty() {
                return Err(StoreError::NotQualifiable {
                    kernel: r.kernel_name.clone(),
                    blockers,
                });
            }
        }
        for r in &mut records {
            r.state = RecordState::Qualified;
        }
        self.append_atomic(hardware, &records)?;
        Ok(records.len())
    }

    /// Store records without qualifying them — measurements that failed a gate,
    /// and the reason. Kept so the negative result is not relearned.
    pub fn record_rejected(
        &self,
        hardware: &str,
        mut records: Vec<KernelMeasurement>,
        reason: &str,
    ) -> Result<usize, StoreError> {
        for r in &mut records {
            r.state = RecordState::Rejected {
                reason: reason.to_string(),
            };
        }
        self.append_atomic(hardware, &records)?;
        Ok(records.len())
    }

    fn moe_decode_path(&self, hardware: &str) -> PathBuf {
        self.root
            .join(hardware)
            .join("moe_decode_measurement.jsonl")
    }

    /// Grouped-MoE decode route records for one hardware key; empty when the
    /// campaign has never run here.
    pub fn load_moe_decode(&self, hardware: &str) -> Result<Vec<MoeDecodeMeasurement>, StoreError> {
        let path = self.moe_decode_path(hardware);
        if !path.exists() {
            return Ok(Vec::new());
        }
        let mut out = Vec::new();
        for line in BufReader::new(File::open(&path)?).lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            out.push(serde_json::from_str(&line)?);
        }
        Ok(out)
    }

    /// Publish grouped-MoE decode route records under the shared gates: an
    /// unchecked or under-sampled route aborts the whole publication.
    pub fn publish_moe_decode(
        &self,
        hardware: &str,
        mut records: Vec<MoeDecodeMeasurement>,
    ) -> Result<usize, StoreError> {
        for r in &records {
            let blockers = r.qualification_blockers();
            if !blockers.is_empty() {
                return Err(StoreError::NotQualifiable {
                    kernel: format!("{}/{:?}", r.cell.key(), r.route),
                    blockers,
                });
            }
        }
        for r in &mut records {
            r.state = RecordState::Qualified;
        }
        self.append_jsonl(&self.moe_decode_path(hardware), &records)?;
        Ok(records.len())
    }

    fn decode_path(&self, hardware: &str) -> PathBuf {
        self.root.join(hardware).join("decode_measurement.jsonl")
    }

    /// Every decode-knob measurement stored for one hardware cell.
    pub fn load_decode(&self, hardware: &str) -> Result<Vec<DecodeMeasurement>, StoreError> {
        let path = self.decode_path(hardware);
        if !path.exists() {
            return Ok(Vec::new());
        }
        let mut out = Vec::new();
        for line in BufReader::new(File::open(&path)?).lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            out.push(serde_json::from_str(&line)?);
        }
        Ok(out)
    }

    /// Publish decode-knob records as one unit, under the same gates as kernel
    /// measurements: an unchecked or under-sampled configuration aborts the
    /// whole publication rather than landing a selectable half-winner.
    pub fn publish_decode(
        &self,
        hardware: &str,
        mut records: Vec<DecodeMeasurement>,
    ) -> Result<usize, StoreError> {
        for r in &records {
            let blockers = r.qualification_blockers();
            if !blockers.is_empty() {
                return Err(StoreError::NotQualifiable {
                    kernel: r.knobs.label(),
                    blockers,
                });
            }
        }
        for r in &mut records {
            r.state = RecordState::Qualified;
        }
        self.append_jsonl(&self.decode_path(hardware), &records)?;
        Ok(records.len())
    }

    /// Store decode records that did not pass a gate — a screening sweep whose
    /// reps are below the publishable minimum, or a configuration that lost.
    /// Kept so the next campaign does not re-measure the same dead grid point.
    pub fn record_decode_unqualified(
        &self,
        hardware: &str,
        mut records: Vec<DecodeMeasurement>,
        state: RecordState,
    ) -> Result<usize, StoreError> {
        for r in &mut records {
            r.state = state.clone();
        }
        self.append_jsonl(&self.decode_path(hardware), &records)?;
        Ok(records.len())
    }

    /// Qualified decode records whose digests still match, ranked per cell.
    ///
    /// Returns the rankings *and* the stale records, so a caller can say "there
    /// is data, but it was measured under another compiler" rather than
    /// silently reporting no winner.
    pub fn best_decode_for(
        &self,
        hardware: &str,
        want: &Digests,
    ) -> Result<(Vec<CellRanking>, Vec<StaleNote>), StoreError> {
        let mut usable = Vec::new();
        let mut stale = Vec::new();
        for rec in self.load_decode(hardware)? {
            if !rec.state.is_selectable() {
                continue;
            }
            let changed = rec.digests.stale_against(want);
            if changed.is_empty() {
                usable.push(rec);
            } else {
                stale.push(StaleNote {
                    op_case: rec.cell.key(),
                    kernel_name: rec.knobs.label(),
                    changed: changed.iter().map(|s| s.to_string()).collect(),
                });
            }
        }
        Ok((rank_by_cell(usable), stale))
    }

    /// Read-modify-rename. Appending in place would leave a half-line behind if
    /// the process died mid-write, and a half-line makes the whole file
    /// unparseable.
    fn append_atomic(
        &self,
        hardware: &str,
        records: &[KernelMeasurement],
    ) -> Result<(), StoreError> {
        self.append_jsonl(&self.kernel_path(hardware), records)
    }

    fn append_jsonl<T: serde::Serialize>(
        &self,
        path: &PathBuf,
        records: &[T],
    ) -> Result<(), StoreError> {
        let dir = path.parent().expect("record path has a parent");
        fs::create_dir_all(dir)?;

        let tmp = path.with_extension("jsonl.tmp");
        // Start from whatever is already committed, so the rename is a superset
        // and never loses history.
        if path.exists() {
            fs::copy(&path, &tmp)?;
        } else if tmp.exists() {
            fs::remove_file(&tmp)?;
        }
        {
            let mut f = OpenOptions::new().create(true).append(true).open(&tmp)?;
            for r in records {
                writeln!(f, "{}", serde_json::to_string(r)?)?;
            }
            f.sync_all()?;
        }
        fs::rename(&tmp, &path)?;
        Ok(())
    }

    /// Best qualified measurement per op case, after discarding anything whose
    /// digests have moved.
    ///
    /// Returns the record *and* whatever made stale records unusable, so a
    /// caller can report "there was data, but it is for a different compiler"
    /// rather than silently falling back to the analytical model.
    pub fn best_for(
        &self,
        hardware: &str,
        want: &Digests,
    ) -> Result<(BTreeMap<String, KernelMeasurement>, Vec<StaleNote>), StoreError> {
        let mut best: BTreeMap<String, KernelMeasurement> = BTreeMap::new();
        let mut stale = Vec::new();

        for rec in self.load_kernels(hardware)? {
            if !rec.state.is_selectable() {
                continue;
            }
            let changed = rec.digests.stale_against(want);
            if !changed.is_empty() {
                stale.push(StaleNote {
                    op_case: rec.op_case.clone(),
                    kernel_name: rec.kernel_name.clone(),
                    changed: changed.iter().map(|s| s.to_string()).collect(),
                });
                continue;
            }
            match best.get(&rec.op_case) {
                Some(cur) if !rec.stats.beats(&cur.stats) => {}
                _ => {
                    best.insert(rec.op_case.clone(), rec);
                }
            }
        }
        Ok((best, stale))
    }
}

/// A record that exists but cannot be used, and why.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StaleNote {
    pub op_case: String,
    pub kernel_name: String,
    pub changed: Vec<String>,
}

/// Convenience for building a measurement that has passed its oracle.
pub fn passed(mut m: KernelMeasurement) -> KernelMeasurement {
    m.correctness = Correctness::Pass;
    m
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::record::Correctness;
    use crate::sample::Stats;

    fn tmpdir(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("tunedb_{}_{}", tag, std::process::id()));
        let _ = fs::remove_dir_all(&d);
        d
    }

    fn digests() -> Digests {
        Digests {
            implementation: "impl-a".into(),
            interpreter: "interp-a".into(),
            toolchain: "cuda-13.0".into(),
            oracle: "oracle-v1".into(),
        }
    }

    fn meas(case: &str, id: u16, median: f64) -> KernelMeasurement {
        // Tight samples around the median so `beats` is decided by the median,
        // not by manufactured jitter.
        let s = vec![median - 1.0, median, median, median, median + 1.0];
        KernelMeasurement {
            op_case: case.into(),
            kernel_id: id,
            kernel_name: format!("PLOW_DOP_{id}"),
            profile: "prefill_dense".into(),
            hardware: "nvidia/sm_90a/h100-nvl".into(),
            sku: "H100 NVL".into(),
            digests: digests(),
            stats: Stats::from_samples(s).unwrap(),
            correctness: Correctness::Pass,
            registers: Some(208),
            state: RecordState::Provisional,
            campaign: "c1".into(),
        }
    }

    const HW: &str = "nvidia/sm_90a/h100-nvl";

    #[test]
    fn publish_then_read_back() {
        let dir = tmpdir("publish");
        let s = TuneStore::new(&dir);
        assert_eq!(s.publish(HW, vec![meas("case-a", 8, 100.0)]).unwrap(), 1);

        let all = s.load_kernels(HW).unwrap();
        assert_eq!(all.len(), 1);
        assert_eq!(all[0].state, RecordState::Qualified, "publish qualifies");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn attention_publish_qualifies_and_uses_a_separate_store() {
        let dir = tmpdir("attention_publish");
        let s = TuneStore::new(&dir);
        let record = crate::attention::AttentionMeasurement {
            cell: crate::attention::AttentionCell {
                hardware: HW.into(),
                n_cu: 256,
                decode_rung: 8,
                kv_bucket: crate::attention::KvBucket::K8,
                shape: "mla/dk512/dr64/h12/gf4".into(),
            },
            algorithm: crate::attention::AttentionAlgorithm::SplitReduce,
            nsplit: 32,
            digests: digests(),
            stats: Stats::from_samples(vec![100.0; 5]).unwrap(),
            correctness: Correctness::Pass,
            state: RecordState::Provisional,
            campaign: "attention-c1".into(),
        };

        assert_eq!(s.publish_attention(HW, vec![record]).unwrap(), 1);
        let all = s.load_attention(HW).unwrap();
        assert_eq!(all.len(), 1);
        assert_eq!(all[0].state, RecordState::Qualified);
        assert!(s.load_kernels(HW).unwrap().is_empty());
        let _ = fs::remove_dir_all(&dir);
    }

    /// The transactional property: one unqualifiable record aborts the whole
    /// campaign, leaving the store untouched rather than partly updated.
    #[test]
    fn a_failing_record_aborts_the_entire_publication() {
        let dir = tmpdir("atomic");
        let s = TuneStore::new(&dir);
        s.publish(HW, vec![meas("case-a", 8, 100.0)]).unwrap();

        let good = meas("case-b", 14, 50.0);
        let mut bad = meas("case-c", 15, 40.0);
        bad.correctness = Correctness::Fail {
            detail: "nan".into(),
        };

        let err = s.publish(HW, vec![good, bad]).unwrap_err();
        assert!(matches!(err, StoreError::NotQualifiable { .. }));

        let all = s.load_kernels(HW).unwrap();
        assert_eq!(all.len(), 1, "the good record must not have landed either");
        assert_eq!(all[0].op_case, "case-a");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn appending_preserves_history() {
        let dir = tmpdir("append");
        let s = TuneStore::new(&dir);
        s.publish(HW, vec![meas("case-a", 8, 100.0)]).unwrap();
        s.publish(HW, vec![meas("case-a", 14, 60.0)]).unwrap();

        let all = s.load_kernels(HW).unwrap();
        assert_eq!(
            all.len(),
            2,
            "the superseded record is retained, not overwritten"
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn best_for_picks_the_fastest_qualified_record() {
        let dir = tmpdir("best");
        let s = TuneStore::new(&dir);
        s.publish(HW, vec![meas("case-a", 8, 100.0), meas("case-a", 14, 60.0)])
            .unwrap();

        let (best, stale) = s.best_for(HW, &digests()).unwrap();
        assert!(stale.is_empty());
        assert_eq!(best["case-a"].kernel_id, 14);
        let _ = fs::remove_dir_all(&dir);
    }

    /// Data measured under another compiler must not answer a query for this
    /// one, and the caller must be able to say so.
    #[test]
    fn a_moved_digest_makes_a_record_unusable_and_reportable() {
        let dir = tmpdir("stale");
        let s = TuneStore::new(&dir);
        s.publish(HW, vec![meas("case-a", 8, 100.0)]).unwrap();

        let mut want = digests();
        want.toolchain = "cuda-12.4".into();

        let (best, stale) = s.best_for(HW, &want).unwrap();
        assert!(
            best.is_empty(),
            "must not serve a record from another toolchain"
        );
        assert_eq!(stale.len(), 1);
        assert_eq!(stale[0].changed, vec!["toolchain"]);
        let _ = fs::remove_dir_all(&dir);
    }

    /// Rejections are stored and are never selectable.
    #[test]
    fn rejected_records_are_kept_but_not_served() {
        let dir = tmpdir("reject");
        let s = TuneStore::new(&dir);
        s.record_rejected(HW, vec![meas("case-a", 8, 100.0)], "alias of PLOW_DOP_GEMM")
            .unwrap();

        let all = s.load_kernels(HW).unwrap();
        assert_eq!(all.len(), 1);
        match &all[0].state {
            RecordState::Rejected { reason } => assert!(reason.contains("alias")),
            other => panic!("expected rejected, got {other:?}"),
        }

        let (best, _) = s.best_for(HW, &digests()).unwrap();
        assert!(best.is_empty(), "a rejected record is never selectable");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn hardware_cells_do_not_share_a_file() {
        let dir = tmpdir("cells");
        let s = TuneStore::new(&dir);
        s.publish("nvidia/sm_90a/h100-nvl", vec![meas("case-a", 8, 100.0)])
            .unwrap();
        s.publish("amd/gfx950/mi350x", vec![meas("case-a", 8, 10.0)])
            .unwrap();

        assert_eq!(s.load_kernels("nvidia/sm_90a/h100-nvl").unwrap().len(), 1);
        assert_eq!(s.load_kernels("amd/gfx950/mi350x").unwrap().len(), 1);
        assert_eq!(
            s.load_kernels("nvidia/sm_90a/h100-nvl").unwrap()[0]
                .stats
                .median_ns,
            100.0,
            "the AMD record must not leak into the NVIDIA cell"
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn an_absent_cell_reads_as_empty_not_an_error() {
        let dir = tmpdir("absent");
        let s = TuneStore::new(&dir);
        assert!(s.load_kernels(HW).unwrap().is_empty());
        assert!(s.load_decode(HW).unwrap().is_empty());
        let _ = fs::remove_dir_all(&dir);
    }

    fn decode_meas(n_cu: u32, ctx: u32, unroll: u32, ms: f64, samples: usize) -> DecodeMeasurement {
        let ns = ms * 1.0e6;
        DecodeMeasurement {
            cell: crate::decode::DecodeCell {
                hardware: HW.into(),
                dtype: "bf16".into(),
                n_cu,
                ctx_bucket: crate::decode::CtxBucket::of(ctx),
                model: "gemma-4-26B-A4B-it".into(),
                batch: 1,
            },
            knobs: crate::decode::DecodeKnobs {
                extra_defines: Default::default(),
                extra_emit: Default::default(),
                minblk: n_cu / 132,
                n_cu,
                gv_unroll: unroll,
                gv_unroll_glu: 0,
                gv_moe_un: 2,
                moe_down_sg: 4,
                gv_mm_max: None,
                ns_abs: 16,
                fa_wpr: None,
                fa_gf: None,
                fa_gf_full: None,
                fa_kun: None,
                ns_full_abs: 0,
            },
            ctx,
            digests: digests(),
            stats: Stats::from_samples(vec![ns; samples]).unwrap(),
            registers: Some(180),
            correctness: Correctness::Pass,
            state: RecordState::Provisional,
            campaign: "c1".into(),
        }
    }

    /// Decode records live in their own file. A kernel-selection query and a
    /// knob-set query are different populations and must not be one another's
    /// noise.
    #[test]
    fn decode_records_do_not_share_a_file_with_kernel_records() {
        let dir = tmpdir("decfile");
        let s = TuneStore::new(&dir);
        s.publish(HW, vec![meas("case-a", 8, 100.0)]).unwrap();
        s.publish_decode(HW, vec![decode_meas(132, 1024, 8, 6.04, 5)])
            .unwrap();

        assert_eq!(s.load_kernels(HW).unwrap().len(), 1);
        assert_eq!(s.load_decode(HW).unwrap().len(), 1);
        assert_eq!(s.load_decode(HW).unwrap()[0].state, RecordState::Qualified);
        let _ = fs::remove_dir_all(&dir);
    }

    /// The sample floor binds *before* a record exists: three reps cannot even
    /// be summarised, so a screening sweep never reaches the store to be
    /// argued about. This is why `tune_decode_sweep.sh` takes 5 reps and not
    /// the 3 the campaign cards quote by hand.
    #[test]
    fn three_reps_cannot_become_a_record_at_all() {
        assert_eq!(
            Stats::from_samples(vec![6.04e6; 3]),
            Err(crate::sample::SampleError::TooFew { got: 3, need: 5 })
        );
    }

    /// A configuration whose correctness was never checked must not become
    /// selectable — but the GPU time must not be thrown away either.
    #[test]
    fn an_unchecked_configuration_is_kept_but_never_selectable() {
        let dir = tmpdir("decscreen");
        let s = TuneStore::new(&dir);
        let mut m = decode_meas(132, 1024, 8, 6.04, 5);
        m.correctness = Correctness::Unchecked;

        let err = s.publish_decode(HW, vec![m.clone()]).unwrap_err();
        assert!(matches!(err, StoreError::NotQualifiable { .. }));
        assert!(s.load_decode(HW).unwrap().is_empty(), "nothing landed");

        s.record_decode_unqualified(HW, vec![m], RecordState::Provisional)
            .unwrap();
        let held = s.load_decode(HW).unwrap();
        assert_eq!(held.len(), 1);
        assert!(!held[0].state.is_selectable());
        let (best, _) = s.best_decode_for(HW, &digests()).unwrap();
        assert!(best.is_empty(), "a provisional record is never a winner");
        let _ = fs::remove_dir_all(&dir);
    }

    /// The whole point of the ctx and occupancy axes: one query returns one
    /// winner per cell, not one global winner.
    #[test]
    fn best_decode_returns_a_winner_per_occupancy_and_context_cell() {
        let dir = tmpdir("decbest");
        let s = TuneStore::new(&dir);
        s.publish_decode(
            HW,
            vec![
                decode_meas(132, 1024, 8, 6.04, 5),
                decode_meas(132, 1024, 4, 6.33, 5),
                decode_meas(264, 1024, 4, 5.62, 5),
                decode_meas(264, 1024, 8, 5.81, 5),
                decode_meas(132, 32768, 8, 9.21, 5),
            ],
        )
        .unwrap();

        let (best, stale) = s.best_decode_for(HW, &digests()).unwrap();
        assert!(stale.is_empty());
        assert_eq!(best.len(), 3, "occ-1/1k, occ-2/1k, occ-1/32k");
        for cell in &best {
            let w = cell.winner().unwrap();
            match (cell.cell.n_cu, cell.cell.ctx_bucket) {
                (132, crate::decode::CtxBucket::K1) => assert_eq!(w.knobs.gv_unroll, 8),
                (264, crate::decode::CtxBucket::K1) => assert_eq!(w.knobs.gv_unroll, 4),
                _ => {}
            }
        }
        let _ = fs::remove_dir_all(&dir);
    }

    /// A knob set measured against a different object is not evidence about
    /// this one — the interpreter digest is the cubin's own hash, which is
    /// exactly what this sweep varies.
    #[test]
    fn a_decode_record_from_another_object_is_stale_and_reportable() {
        let dir = tmpdir("decstale");
        let s = TuneStore::new(&dir);
        s.publish_decode(HW, vec![decode_meas(132, 1024, 8, 6.04, 5)])
            .unwrap();

        let mut want = digests();
        want.interpreter = "some-other-cubin".into();
        let (best, stale) = s.best_decode_for(HW, &want).unwrap();
        assert!(best.is_empty());
        assert_eq!(stale.len(), 1);
        assert_eq!(stale[0].changed, vec!["interpreter"]);
        let _ = fs::remove_dir_all(&dir);
    }
}
