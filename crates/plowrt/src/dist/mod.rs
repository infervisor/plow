//! Asset distribution: resolve a model reference against a live machine, fetch
//! what is missing, and hand `serve` a directory it already knows how to load.
//!
//! The split is deliberate. This module resolves and materializes; it never
//! serves. `serve` reads only from the local store and performs **no network
//! I/O** — a missing blob is an error naming the `pull` that would fix it — so
//! the serving process keeps no egress and CI can pin that property.
//!
//! Selection lives in `plow_asset::dist` rather than here, because the packer
//! needs the same rule: a variant the rule could never pick must not be
//! publishable. Two implementations could disagree, which is the hazard
//! `exec/gpu.rs`'s pairing check already refuses on principle.

pub mod reference;
pub mod store;

pub use reference::Reference;
pub use store::{Digest, Store};

use plow_asset::dist::{Constraints, LiveTarget, ModelIndex, Variant};

/// Describe the machine `plowrt` is running on, for selection.
///
/// `gpus` is how many devices of the *same* kind are visible, which bounds the
/// parallelism a variant may ask for. The CPU reference backend reports no
/// silicon and therefore no fingerprint; it is described as `cpu` so a GPU
/// variant is refused at selection rather than at first token.
pub fn live_target(backends: &[std::sync::Arc<dyn crate::device::Backend>]) -> LiveTarget {
    let gpus = backends.len() as u32;
    match backends.first().and_then(|b| b.fingerprint()) {
        Some(fp) => LiveTarget {
            vendor: match fp.isa.vendor() {
                hwspec::Vendor::Nvidia => "nvidia",
                hwspec::Vendor::Amd => "amd",
                hwspec::Vendor::Apple => "apple",
            }
            .to_string(),
            isa: fp.isa.arch_flag().to_string(),
            sku: Some(fp.sku.clone()),
            units: fp.units,
            mem_bytes: fp.mem_bytes,
            gpus,
            toolchain: fp.toolchain.clone(),
        },
        None => LiveTarget {
            vendor: "cpu".into(),
            isa: "cpu".into(),
            sku: None,
            units: backends
                .first()
                .map(|b| b.enumerate().len() as u32)
                .unwrap_or(0),
            mem_bytes: 0,
            gpus,
            toolchain: None,
        },
    }
}

/// Turn a reference's explicit `:label@gN` into selection constraints.
pub fn constraints_for(r: &Reference) -> Constraints {
    Constraints {
        label: r.label.clone(),
        generation: r.generation,
        oversub: crate::config::RuntimeConfig::get().amd.oversub,
        ..Default::default()
    }
}

/// Pick the variant this machine should run, or explain why none fits.
pub fn resolve<'a>(
    index: &'a ModelIndex,
    r: &Reference,
    live: &LiveTarget,
    extra: Constraints,
) -> Result<&'a Variant, String> {
    index.validate()?;
    let c = Constraints {
        label: r.label.clone().or(extra.label),
        generation: r.generation.or(extra.generation),
        ..extra
    };
    plow_asset::dist::select(&index.variants, live, &c)
}
