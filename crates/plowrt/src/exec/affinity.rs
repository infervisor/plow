//! Thread placement for the two pools that must not contend: the GPU tick's
//! engine thread and the CPU prefill-head pool.
//!
//! The head pool exists only to spend CPU the serving path is not using, so it
//! must not be able to take any. Two mechanisms, and they bound different
//! things:
//!
//! * **Affinity** keeps the pools on disjoint physical cores. Whole cores, not
//!   logical cpus — an SMT sibling running a head shares the core's execution
//!   resources with the engine thread, so reserving one thread of a core and
//!   handing the other to the head pool reserves nothing.
//! * **`SCHED_IDLE`** means a head thread cannot preempt a normally-scheduled
//!   serving thread even where the masks do overlap. Its precondition is that
//!   the head pool shares no lock with the serving path: an idle-priority
//!   thread holding one inverts priority. It shares no engine state by
//!   construction; the allocator is the real hazard, which is why head buffers
//!   are allocated when the pool arms rather than inside a head.
//!
//! Neither bounds MEMORY BANDWIDTH. An idle-priority thread that is running
//! still saturates the memory controllers, and the serving path's pinned-staging
//! fills and H2D DMA reads slow down accordingly. That is bounded by the WIDTH
//! of the head pool, which is why [`CorePlan::head`] is a budget and not
//! "whatever is left over".

use super::cpu::topology::Core;

/// Which logical cpus each pool may run on.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CorePlan {
    /// The serving path: engine threads, H2D staging, the async runtime.
    pub serving: Vec<u32>,
    /// The prefill-head pool. Empty means "no head may run here".
    pub head: Vec<u32>,
}

impl CorePlan {
    pub fn is_empty(&self) -> bool {
        self.head.is_empty()
    }
}

/// Split the machine's physical cores between the serving path and the head
/// pool.
///
/// `reserve` whole cores go to serving, taken from the low end; the head pool
/// gets the rest. An `explicit` cpu list overrides the split entirely — that is
/// `--het-cores`, for an operator who knows their box better than this does —
/// and is intersected with the cores actually present so a typo cannot pin a
/// thread to a cpu that does not exist.
///
/// A `reserve` at or past the core count yields an empty head pool rather than
/// an error. "There is no room for a head on this box" is a configuration, not
/// a failure: the caller declines to arm and serves exactly as before.
pub fn plan_cores(cores: &[Core], explicit: Option<&[u32]>, reserve: usize) -> CorePlan {
    let all: Vec<u32> = cores.iter().flat_map(|c| c.siblings.iter().copied()).collect();
    if let Some(want) = explicit {
        let head: Vec<u32> = all.iter().copied().filter(|c| want.contains(c)).collect();
        let serving: Vec<u32> = all.iter().copied().filter(|c| !head.contains(c)).collect();
        return CorePlan { serving, head };
    }
    let split = reserve.min(cores.len());
    CorePlan {
        serving: cores[..split]
            .iter()
            .flat_map(|c| c.siblings.iter().copied())
            .collect(),
        head: cores[split..]
            .iter()
            .flat_map(|c| c.siblings.iter().copied())
            .collect(),
    }
}

/// The process-wide plan, resolved once from `--het-cores` /
/// `--het-reserve-cores` against the live topology.
///
/// Empty (both pools) when neither knob is set: heterogeneous prefill is off and
/// no thread is placed that was not placed before.
pub fn plan() -> &'static CorePlan {
    static PLAN: std::sync::OnceLock<CorePlan> = std::sync::OnceLock::new();
    PLAN.get_or_init(|| {
        let cfg = &crate::config::RuntimeConfig::get().het;
        let explicit = cfg
            .cores
            .as_deref()
            .map(super::cpu::topology::parse_cpulist);
        if explicit.is_none() && cfg.reserve_cores == 0 {
            return CorePlan::default();
        }
        let topo = super::cpu::topology::Topology::detect();
        let plan = plan_cores(
            &topo.cores,
            explicit.as_deref(),
            cfg.reserve_cores as usize,
        );
        tracing::info!(
            serving = plan.serving.len(),
            head = plan.head.len(),
            cores = topo.cores.len(),
            "heterogeneous prefill core plan"
        );
        plan
    })
}

#[cfg(target_os = "linux")]
mod sys {
    /// Pin the calling thread to `cpus`. Returns false and warns on failure —
    /// a box that refuses the mask should serve, not abort.
    pub fn pin(cpus: &[u32]) -> bool {
        if cpus.is_empty() {
            return false;
        }
        // SAFETY: cpu_set_t is POD; sched_setaffinity acts on the calling thread.
        unsafe {
            let mut set: libc::cpu_set_t = std::mem::zeroed();
            for &cpu in cpus {
                if cpu as usize >= libc::CPU_SETSIZE as usize {
                    tracing::warn!(cpu, "CPU id exceeds affinity mask capacity");
                    continue;
                }
                libc::CPU_SET(cpu as usize, &mut set);
            }
            if libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set) != 0 {
                tracing::warn!(
                    ?cpus,
                    error = %std::io::Error::last_os_error(),
                    "CPU affinity failed"
                );
                return false;
            }
        }
        true
    }

    /// Put the calling thread on `SCHED_IDLE`, so it runs only when nothing
    /// else wants the cpu.
    pub fn set_idle_priority() -> bool {
        // SAFETY: sched_param is POD; SCHED_IDLE takes priority 0 and the call
        // acts on the calling thread.
        unsafe {
            let param = libc::sched_param { sched_priority: 0 };
            if libc::sched_setscheduler(0, libc::SCHED_IDLE, &param) != 0 {
                tracing::warn!(
                    error = %std::io::Error::last_os_error(),
                    "SCHED_IDLE failed; head threads can preempt the serving path"
                );
                return false;
            }
        }
        true
    }

    /// The cpus this process is actually allowed to use, honouring cpusets and
    /// `taskset`. Empty when the file is unreadable.
    pub fn allowed_cpus() -> Vec<u32> {
        std::fs::read_to_string("/proc/self/status")
            .ok()
            .and_then(|s| {
                s.lines()
                    .find_map(|l| l.strip_prefix("Cpus_allowed_list:"))
                    .map(crate::exec::cpu::topology::parse_cpulist)
            })
            .unwrap_or_default()
    }
}

#[cfg(not(target_os = "linux"))]
mod sys {
    pub fn pin(_cpus: &[u32]) -> bool {
        false
    }
    pub fn set_idle_priority() -> bool {
        false
    }
    pub fn allowed_cpus() -> Vec<u32> {
        Vec::new()
    }
}

pub use sys::{allowed_cpus, pin, set_idle_priority};

#[cfg(test)]
mod tests {
    use super::*;

    fn core(cpu: u32, node: u32, siblings: &[u32]) -> Core {
        Core {
            cpu,
            node,
            siblings: siblings.to_vec(),
        }
    }

    /// Two nodes, four physical cores each with one SMT sibling — the EPYC
    /// numbering where sibling(i) = i + n_cores.
    fn smt_box() -> Vec<Core> {
        (0..4)
            .map(|i| core(i, i / 2, &[i, i + 4]))
            .collect()
    }

    #[test]
    fn a_reservation_takes_whole_cores_with_their_siblings() {
        let plan = plan_cores(&smt_box(), None, 2);
        // Cores 0 and 1 reserved: both threads of each, not just the low one.
        assert_eq!(plan.serving, vec![0, 4, 1, 5]);
        assert_eq!(plan.head, vec![2, 6, 3, 7]);
    }

    #[test]
    fn the_pools_never_share_a_logical_cpu() {
        for reserve in 0..=6 {
            let plan = plan_cores(&smt_box(), None, reserve);
            assert!(
                !plan.serving.iter().any(|c| plan.head.contains(c)),
                "overlap at reserve {reserve}"
            );
        }
    }

    #[test]
    fn reserving_the_whole_box_leaves_no_head_pool() {
        // Not an error: the caller declines to arm and serves as before.
        let plan = plan_cores(&smt_box(), None, 4);
        assert!(plan.is_empty());
        assert_eq!(plan.serving.len(), 8);
        // Past the core count is the same answer, not a panic.
        assert!(plan_cores(&smt_box(), None, 99).is_empty());
    }

    #[test]
    fn an_explicit_list_overrides_the_split() {
        let plan = plan_cores(&smt_box(), Some(&[6, 7]), 2);
        assert_eq!(plan.head, vec![6, 7]);
        assert!(!plan.serving.contains(&6));
        assert_eq!(plan.serving.len(), 6);
    }

    #[test]
    fn an_explicit_list_cannot_name_a_cpu_the_box_does_not_have() {
        let plan = plan_cores(&smt_box(), Some(&[6, 999]), 2);
        assert_eq!(plan.head, vec![6]);
    }

    #[test]
    fn no_reservation_is_a_plan_that_places_nothing() {
        // The off state has to be an EMPTY serving set too: a caller that pins
        // to `serving` must find nothing to pin to, not the whole box.
        let plan = plan_cores(&smt_box(), None, 0);
        assert!(plan.serving.is_empty());
        assert_eq!(plan.head.len(), 8);
        assert!(CorePlan::default().serving.is_empty());
        assert!(CorePlan::default().is_empty());
    }
}
