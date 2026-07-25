//! Per-checkpoint typed wrappers. Each mirrors §5.10 of the plan.

pub mod schedule;
pub mod memory;
pub mod sram;
pub mod wire;
pub mod rewrite;
pub mod tile_partition;

pub use memory::check_address_map;
pub use schedule::check_schedule;
pub use sram::check_sram_fit;
pub use wire::check_wire_roundtrip;
pub use rewrite::check_rewrite_rules;
pub use tile_partition::check_tile_partition;
