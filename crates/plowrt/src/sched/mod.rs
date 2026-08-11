//! §E Scheduler — per-iteration bucket selection, dispatch, completion.

pub mod admission;
pub mod batching;
pub mod mdq;
pub mod multistep;
pub mod rungs;

use crate::asset::{Bucket, ModelBundle, Phase};
use crate::exec::counters::CounterPool;
use crate::exec::ExecutorSet;
use crate::{Result, RuntimeError};

/// Drives one inference iteration onto the executor set.
pub struct Scheduler<'a> {
    execset: &'a ExecutorSet,
}

/// Result of running one iteration.
#[derive(Clone, Copy, Debug)]
pub struct IterationOutcome {
    pub executed: usize,
    pub completed: bool,
}

impl<'a> Scheduler<'a> {
    pub fn new(execset: &'a ExecutorSet) -> Self {
        Scheduler { execset }
    }

    /// Pick the bucket covering `(phase, batch, seq)` and run it.
    pub fn run(
        &self,
        bundle: &ModelBundle,
        phase: Phase,
        batch: i64,
        seq: i64,
    ) -> Result<IterationOutcome> {
        let key = batching::select_bucket(bundle, phase, batch, seq)
            .ok_or_else(|| RuntimeError::Rejected("no compiled bucket covers request".into()))?;
        let bucket = bundle
            .bucket(key)
            .ok_or_else(|| RuntimeError::Msg("selected bucket missing".into()))?;
        self.run_bucket(bucket)
    }

    /// Run a specific bucket to completion, returning the outcome. On the CPU
    /// backend this cooperatively interprets the stream; on a real device it
    /// enqueues the packet stream and polls the milestone counter.
    pub fn run_bucket(&self, bucket: &Bucket) -> Result<IterationOutcome> {
        let pool = self.execset.counter_pool(&bucket.program);
        let stats = self.execset.run_reference(&bucket.program, &pool);
        if !stats.completed {
            return Err(RuntimeError::Deadlock(format!(
                "bucket did not complete: {}/{} instructions fired",
                stats.executed, stats.total
            )));
        }
        Ok(IterationOutcome {
            executed: stats.executed,
            completed: stats.completed,
        })
    }

    /// Build a fresh counter pool for a bucket (exposed for the monitor/tests).
    pub fn pool_for(&self, bucket: &Bucket) -> CounterPool {
        self.execset.counter_pool(&bucket.program)
    }
}
