//! Intel Xeon CPU descriptors, for the plowrt CPU engine. One "SM" is one physical core: the
//! packet's `n_cu` virtual executors map onto cores, `shared_mem` is the per-core L2 (the
//! pseudo-lock SRAM budget), `l2` is the shared LLC, and the matrix engine is AMX.
pub mod xeon6;
