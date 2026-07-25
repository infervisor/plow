//! §D Executor set + counters + queues + OOB.
//!
//! Owns the counter pool, per-executor packet queues, and the bidirectional
//! out-of-band channel, and brings the persistent kernels up once. The hot path
//! (enqueue packet, poll counter) touches only lock-free structures here.

pub mod counters;
pub mod engine_thread;
#[cfg(feature = "cuda")]
pub mod gpu;
pub mod health;
pub mod host;
pub mod indirection;
pub mod oob;
pub mod queue;

use std::sync::Arc;

use packet::Program;

use crate::device::{Backend, ExecutorTarget, LaunchCfg};
use crate::exec::counters::CounterPool;
use crate::Result;

/// Device-local executor id (SM/CU/thread index).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ExecId(pub u32);

/// The live set of executors on one backend, plus their coordination memory.
pub struct ExecutorSet {
    backend: Arc<dyn Backend>,
    targets: Vec<ExecutorTarget>,
}

impl ExecutorSet {
    /// Enumerate executors and launch the persistent kernel(s) once.
    pub fn bringup(backend: Arc<dyn Backend>) -> Result<Self> {
        let targets = backend.enumerate();
        let cfg = LaunchCfg {
            executors: targets.len() as u32,
            workers: targets.first().map(|t| t.worker_count).unwrap_or(1),
        };
        // A real backend loads the prebuilt module first; the CPU backend's
        // launch is a no-op (cooperative interpret drives it per iteration).
        let module = backend.module_load(&[])?;
        backend.launch_persistent(&module, cfg)?;
        Ok(ExecutorSet { backend, targets })
    }

    pub fn targets(&self) -> &[ExecutorTarget] {
        &self.targets
    }

    pub fn backend(&self) -> &Arc<dyn Backend> {
        &self.backend
    }

    /// Build a counter pool sized for a program's counter table.
    pub fn counter_pool(&self, program: &Program) -> CounterPool {
        CounterPool::from_counters(&program.counters)
    }

    /// Run one program to completion on the CPU reference backend, returning
    /// whether every instruction fired (i.e. no deadlock). On a real device this
    /// is instead "enqueue the packet stream and poll the milestone counter".
    pub fn run_reference(&self, program: &Program, pool: &CounterPool) -> crate::device::cpu::InterpretStats {
        pool.reset_all();
        crate::device::cpu::interpret(program, pool)
    }

    /// Same as [`run_reference`], but records each fired packet into `observer`
    /// so a live run can emit the §K/§O timeline (e.g. `GET /trace`).
    pub fn run_reference_traced<O: crate::device::cpu::StepObserver>(
        &self,
        program: &Program,
        pool: &CounterPool,
        observer: &mut O,
    ) -> crate::device::cpu::InterpretStats {
        pool.reset_all();
        crate::device::cpu::run_streams(program, pool, observer)
    }

    /// [`run_reference_traced`] over a caller-owned [`StreamSet`] — the
    /// allocation-free per-token path. Counters are reset before the walk.
    pub fn run_reference_traced_reuse<O: crate::device::cpu::StepObserver>(
        &self,
        program: &Program,
        pool: &CounterPool,
        observer: &mut O,
        streams: &mut crate::device::cpu::StreamSet,
    ) -> crate::device::cpu::InterpretStats {
        pool.reset_all();
        crate::device::cpu::run_streams_reuse(program, pool, observer, streams)
    }
}
