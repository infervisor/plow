//! Lookup of known device specs by name. The runtime resolves a topology's
//! device-model string to a static [`GpuSpec`] through here.
//!
//! Short aliases are supported so CLI users don't need to type the full
//! canonical name (e.g. `--gpu rtx6000pro` instead of
//! `--gpu "RTX 6000 Pro Blackwell"`). Lookup is always case-insensitive.

use crate::amd::{mi300, mi350};
use crate::nvidia::{ada, blackwell, h100};
use crate::spec::GpuSpec;

/// Every GPU model this crate describes.
pub const ALL: &[&GpuSpec] = &[
    &ada::RTX_4090,
    &h100::H100_SXM5,
    &h100::H100_PCIE,
    &h100::H100_NVL,
    &h100::H200_SXM,
    &blackwell::RTX_5090,
    &blackwell::RTX_6000_PRO,
    &blackwell::B200,
    &mi300::MI300X,
    &mi300::MI325X,
    &mi350::MI350X,
    &mi350::MI355X,
];

/// Short aliases mapping to canonical spec names. Each entry is
/// `(alias, canonical_name)`. Lookup falls through here when the input
/// doesn't match any canonical name directly.
pub const ALIASES: &[(&str, &str)] = &[
    // NVIDIA Ada
    ("rtx4090", "RTX 4090"),
    ("4090", "RTX 4090"),
    // NVIDIA Hopper
    ("h100", "H100 SXM5"),
    ("h100sxm5", "H100 SXM5"),
    ("h100sxm", "H100 SXM5"),
    ("h100pcie", "H100 PCIe"),
    ("h100nvl", "H100 NVL"),
    ("h200", "H200 SXM"),
    ("h200sxm", "H200 SXM"),
    // NVIDIA Blackwell
    ("rtx5090", "RTX 5090"),
    ("5090", "RTX 5090"),
    ("rtx6000pro", "RTX 6000 Pro Blackwell"),
    ("6000pro", "RTX 6000 Pro Blackwell"),
    ("rtx6000", "RTX 6000 Pro Blackwell"),
    ("b200", "B200"),
    // AMD CDNA 3
    ("mi300", "MI300X"),
    ("mi300x", "MI300X"),
    ("mi325", "MI325X"),
    ("mi325x", "MI325X"),
    // AMD CDNA 4
    ("mi350", "MI350X"),
    ("mi350x", "MI350X"),
    ("mi355", "MI355X"),
    ("mi355x", "MI355X"),
];

/// Resolve a model name (case-insensitive) to its spec.
///
/// Tries the canonical name first, then falls back to the short-alias table.
/// Returns `None` only if neither matches.
pub fn lookup(name: &str) -> Option<&'static GpuSpec> {
    // 1. Direct canonical match.
    if let Some(spec) = ALL
        .iter()
        .copied()
        .find(|s| s.name.eq_ignore_ascii_case(name))
    {
        return Some(spec);
    }
    // 2. Alias table → resolve to canonical, then look up.
    let canonical = ALIASES
        .iter()
        .find(|(alias, _)| alias.eq_ignore_ascii_case(name))
        .map(|(_, canon)| *canon)?;
    ALL.iter()
        .copied()
        .find(|s| s.name.eq_ignore_ascii_case(canonical))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spec::InterconnectKind;

    #[test]
    fn every_spec_has_copy_engines() {
        for s in ALL {
            assert!(s.copy_engines >= 1, "{} has no copy engines", s.name);
        }
    }

    #[test]
    fn datacenter_parts_have_fast_fabric() {
        let h100 = lookup("H100 SXM5").unwrap();
        let ic = h100.interconnect.expect("H100 SXM5 has NVLink");
        assert_eq!(ic.kind, InterconnectKind::NvLink);
        assert_eq!(ic.per_gpu_bandwidth.0, 900.0);
        assert_eq!(ic.domain_size, 8);

        let mi300 = lookup("MI300X").unwrap();
        assert_eq!(
            mi300.interconnect.unwrap().kind,
            InterconnectKind::InfinityFabric
        );
    }

    /// A REPORTED bandwidth bound must divide by the measured figure, never the datasheet peak.
    ///
    /// The `plowc --lean-oracle` decode floor divided 61.4 GB of Gemma-4-31B weights by MI350X's
    /// 8000 GB/s datasheet number and printed 7719.3 µs. The measured denominator (6200 GB/s
    /// whole-GPU streaming read, `runtime/amd/op_gemm.h:38`) gives 9.96 ms — the bound was 22.5%
    /// OPTIMISTIC, which on a lower bound means reporting headroom that is not there. Isolated
    /// decode GEMV measures at 95–103% of the 6200 ceiling, so 6200 is where the part is.
    #[test]
    fn measured_bandwidth_governs_a_reported_bound_on_mi350() {
        for name in ["MI350X", "MI355X"] {
            let s = lookup(name).unwrap();
            assert_eq!(s.mem.bandwidth.0, 8000.0, "{name} datasheet peak");
            assert_eq!(
                s.mem.bandwidth_measured.map(|b| b.0),
                Some(6200.0),
                "{name}: MI355X inherits this from MI350X via `..MI350X`; if that broke, a bound \
                 silently reverts to the datasheet number"
            );
            assert_eq!(
                s.mem.bandwidth_for_bound().0,
                6200.0,
                "{name} bound denominator"
            );
            // The number the oracle actually prints, at this part's clock.
            let floor_ms = 61.4e9 / (s.mem.bandwidth_for_bound().0 * 1e9) * 1e3;
            assert!(
                (9.8..10.0).contains(&floor_ms),
                "{name}: Gemma-4-31B bf16 weight-stream floor is 9.90 ms, got {floor_ms:.2}"
            );
        }
    }

    #[test]
    fn measured_bandwidth_governs_a_reported_bound_on_mi325() {
        let s = lookup("MI325X").unwrap();
        assert_eq!(s.mem.bandwidth.0, 6000.0);
        assert_eq!(s.mem.bandwidth_measured.map(|b| b.0), Some(4164.0));
        assert_eq!(s.mem.bandwidth_for_bound().0, 4164.0);
    }

    /// Parts with no measurement fall back to the datasheet peak rather than to zero — a `None`
    /// that divided as 0 would make every bound infinite.
    #[test]
    fn unmeasured_parts_fall_back_to_the_datasheet_peak() {
        let h100 = lookup("H100 SXM5").unwrap();
        assert!(h100.mem.bandwidth_measured.is_none());
        assert_eq!(h100.mem.bandwidth_for_bound().0, h100.mem.bandwidth.0);
    }

    #[test]
    fn h200_is_native_hopper() {
        let h200 = lookup("H200 SXM").unwrap();
        assert_eq!(h200.arch, crate::spec::Arch::Hopper);
        assert_eq!(h200.compute_cap, (9, 0));
        assert_eq!(h200.mem.capacity.as_gib(), 141);
    }

    #[test]
    fn consumer_parts_have_no_fast_fabric() {
        // PCIe-only ⇒ the scheduler takes the slow-link path.
        assert!(lookup("RTX 4090").unwrap().interconnect.is_none());
        assert!(lookup("RTX 5090").unwrap().interconnect.is_none());
    }

    #[test]
    fn short_aliases_resolve() {
        // Every alias in the table must resolve to a valid spec.
        for (alias, canonical) in super::ALIASES {
            let spec = lookup(alias).unwrap_or_else(|| panic!("alias {:?} did not resolve", alias));
            assert_eq!(
                spec.name.to_ascii_lowercase(),
                canonical.to_ascii_lowercase(),
                "alias {:?} resolved to {:?}, expected {:?}",
                alias,
                spec.name,
                canonical,
            );
        }
    }

    #[test]
    fn aliases_are_case_insensitive() {
        assert_eq!(lookup("RTX6000PRO").unwrap().name, "RTX 6000 Pro Blackwell");
        assert_eq!(lookup("H100").unwrap().name, "H100 SXM5");
        assert_eq!(lookup("Mi350").unwrap().name, "MI350X");
    }

    #[test]
    fn canonical_names_still_work() {
        assert!(lookup("H100 SXM5").is_some());
        assert!(lookup("RTX 6000 Pro Blackwell").is_some());
        assert!(lookup("MI300X").is_some());
        assert!(lookup("RTX 5090").is_some());
    }

    #[test]
    fn unknown_name_returns_none() {
        assert!(lookup("totally fake gpu").is_none());
        assert!(lookup("").is_none());
    }
}
