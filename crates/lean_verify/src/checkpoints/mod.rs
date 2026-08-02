//! Per-checkpoint typed wrappers. Each mirrors §5.10 of the plan.

pub mod memory;
pub mod rewrite;
pub mod schedule;
pub mod sram;
pub mod tile_partition;
pub mod wire;

pub use memory::check_address_map;
pub use rewrite::check_rewrite_rules;
pub use schedule::check_schedule;
pub use sram::check_sram_fit;
pub use tile_partition::check_tile_partition;
pub use wire::check_wire_roundtrip;
