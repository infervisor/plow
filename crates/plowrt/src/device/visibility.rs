//! §B Visible-device masks — which physical GPUs plowrt is allowed to touch.
//!
//! Three variables are in play and they do NOT behave the same way, which is
//! the whole reason this module exists:
//!
//! * **`CUDA_VISIBLE_DEVICES`** is applied by libcuda itself, at `cuInit`.
//!   Every ordinal plowrt passes to `cuDeviceGet` is already an index into the
//!   masked set, so there is nothing to apply here — only to report, and to
//!   validate `--devices` against.
//!
//! * **`ROCR_VISIBLE_DEVICES`** is applied by ROCr at `hsa_init`, so
//!   `hsa_iterate_agents` returns the masked set. Same deal: honoured for free.
//!
//! * **`HIP_VISIBLE_DEVICES` is applied by the HIP runtime, and plowrt never
//!   loads the HIP runtime.** It dlopens ROCr directly. So the variable had no
//!   effect whatsoever — measured, `HIP_VISIBLE_DEVICES=4,5,6,7` still
//!   enumerated all 8 agents. An operator who leased one GPU that way got a
//!   process quietly using every GPU on the box, including other people's.
//!   That is the case this module fixes.
//!
//! **Composition.** HIP indexes into the set ROCr already made visible; the two
//! do not name the same numbering. With `ROCR_VISIBLE_DEVICES=4` there is
//! exactly one visible agent, at index 0, and `HIP_VISIBLE_DEVICES=4` then
//! refers to nothing. Tooling that exports both to the same absolute id (as
//! this repo's own scripts note) is relying on HIP being ignored. So the rule
//! is deliberately asymmetric, and safe in both directions:
//!
//! | `ROCR_VISIBLE_DEVICES` | `HIP_VISIBLE_DEVICES` | what plowrt does |
//! |---|---|---|
//! | set | anything | ROCr already narrowed the set; HIP is ignored, with a warning saying so |
//! | unset | set | plowrt applies HIP itself as an index mask over the enumerated agents |
//! | unset | unset | every agent |
//!
//! The first row keeps the existing lease workflows working — plowrt can only
//! ever reach GPUs ROCr granted it. The second row is the fix: an operator who
//! restricted the process with HIP alone now gets what they asked for instead
//! of the whole node.

use std::fmt;

/// One entry of a visible-device list.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Selector {
    /// A plain ordinal into the enumeration below this variable.
    Index(u32),
    /// `GPU-<uuid>` / `MIG-<uuid>`. Accepted by the vendor runtimes; plowrt
    /// cannot resolve one without querying agent identifiers, and guessing
    /// would select the wrong GPU silently.
    Uuid(String),
}

/// A parsed visible-device list.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Mask {
    /// What the variable actually said, for the log line.
    pub raw: String,
    pub entries: Vec<Selector>,
    /// Index of the first entry that failed to parse, if any.
    ///
    /// Both vendor runtimes TRUNCATE at the first bad entry rather than
    /// skipping it, so `0,1,x,3` makes exactly devices 0 and 1 visible and
    /// silently drops 3. Recorded so it can be reported instead of discovered.
    pub truncated_at: Option<usize>,
}

impl Mask {
    /// Parse a visible-device list with the vendor runtimes' own rules:
    /// comma-separated, whitespace tolerated, and truncation at the first
    /// entry that is neither an ordinal nor a UUID handle.
    pub fn parse(raw: &str) -> Mask {
        let mut entries = Vec::new();
        let mut truncated_at = None;
        for (i, field) in raw.split(',').enumerate() {
            let field = field.trim();
            // An empty list ("") means NO devices, not all of them; an empty
            // field mid-list is a parse failure like any other.
            if field.is_empty() {
                truncated_at = Some(i);
                break;
            }
            if let Ok(n) = field.parse::<u32>() {
                entries.push(Selector::Index(n));
            } else if field.starts_with("GPU-") || field.starts_with("MIG-") {
                entries.push(Selector::Uuid(field.to_string()));
            } else {
                truncated_at = Some(i);
                break;
            }
        }
        Mask {
            raw: raw.to_string(),
            entries,
            truncated_at,
        }
    }

    /// Read the mask from the environment, `None` when the variable is unset.
    /// An empty value is a real (empty) mask, not an absent one.
    pub fn from_env(var: &str) -> Option<Mask> {
        std::env::var(var).ok().map(|v| Mask::parse(&v))
    }

    /// The mask as plain indices, or `None` if it names any UUID it would take
    /// a device query to resolve.
    pub fn indices(&self) -> Option<Vec<u32>> {
        self.entries
            .iter()
            .map(|s| match s {
                Selector::Index(i) => Some(*i),
                Selector::Uuid(_) => None,
            })
            .collect()
    }

    /// Devices this mask makes visible, given `n` enumerated below it.
    ///
    /// Out-of-range entries are dropped the way the vendor runtimes drop them.
    /// `None` when the mask names UUIDs, since the caller must not pretend.
    pub fn apply(&self, n: usize) -> Option<Vec<u32>> {
        Some(
            self.indices()?
                .into_iter()
                .filter(|&i| (i as usize) < n)
                .collect(),
        )
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

impl fmt::Display for Mask {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.raw)?;
        if let Some(at) = self.truncated_at {
            write!(f, " (truncated at entry {at} — the rest is ignored)")?;
        }
        Ok(())
    }
}

/// What plowrt should do about the HSA visible-device variables.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HsaVisibility {
    /// Use every agent ROCr enumerated.
    All,
    /// ROCr already applied a mask; nothing more to do. Carries the raw value
    /// for the startup log.
    RocrApplied(String),
    /// `HIP_VISIBLE_DEVICES` was set and `ROCR_VISIBLE_DEVICES` was not, so
    /// plowrt applies it: keep only these indices of the enumerated agents.
    ApplyHip { mask: Mask },
    /// Both were set. ROCr already narrowed the set and HIP's indices refer to
    /// a numbering that no longer exists, so HIP is ignored — loudly.
    HipIgnored { rocr: String, hip: String },
    /// `HIP_VISIBLE_DEVICES` names UUIDs, which plowrt cannot resolve. Refused
    /// rather than silently ignored: ignoring it is how a lease gets violated.
    HipUnresolvable { hip: String },
}

/// Decide what to do with `ROCR_VISIBLE_DEVICES` / `HIP_VISIBLE_DEVICES`.
///
/// Pure, so the decision table above is testable without a GPU.
pub fn hsa_visibility(rocr: Option<Mask>, hip: Option<Mask>) -> HsaVisibility {
    match (rocr, hip) {
        (Some(r), Some(h)) => HsaVisibility::HipIgnored {
            rocr: r.raw,
            hip: h.raw,
        },
        (Some(r), None) => HsaVisibility::RocrApplied(r.raw),
        (None, Some(h)) => {
            if h.indices().is_none() {
                HsaVisibility::HipUnresolvable { hip: h.raw }
            } else {
                HsaVisibility::ApplyHip { mask: h }
            }
        }
        (None, None) => HsaVisibility::All,
    }
}

/// Read the HSA decision from the environment.
pub fn hsa_visibility_from_env() -> HsaVisibility {
    hsa_visibility(
        Mask::from_env("ROCR_VISIBLE_DEVICES"),
        Mask::from_env("HIP_VISIBLE_DEVICES"),
    )
}

/// One line naming every visible-device variable in force, for startup.
/// Empty when none is set.
pub fn describe_env() -> Vec<(&'static str, String)> {
    [
        "CUDA_VISIBLE_DEVICES",
        "ROCR_VISIBLE_DEVICES",
        "HIP_VISIBLE_DEVICES",
    ]
    .iter()
    .filter_map(|v| std::env::var(v).ok().map(|val| (*v, val)))
    .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn m(s: &str) -> Mask {
        Mask::parse(s)
    }

    #[test]
    fn parses_an_ordinary_list() {
        assert_eq!(
            m("0,1,2").entries,
            vec![
                Selector::Index(0),
                Selector::Index(1),
                Selector::Index(2)
            ]
        );
        assert_eq!(m("0,1,2").truncated_at, None);
    }

    #[test]
    fn tolerates_whitespace() {
        assert_eq!(m(" 4 , 5 ").indices().unwrap(), vec![4, 5]);
    }

    /// Both runtimes TRUNCATE at the first bad entry rather than skipping it,
    /// so `0,1,x,3` is devices 0 and 1 — device 3 is silently gone. plowrt
    /// records where it happened so the log can say so.
    #[test]
    fn a_bad_entry_truncates_the_rest() {
        let mask = m("0,1,x,3");
        assert_eq!(mask.indices().unwrap(), vec![0, 1]);
        assert_eq!(mask.truncated_at, Some(2));
        assert!(format!("{mask}").contains("truncated at entry 2"));
    }

    /// An empty value means no devices at all, which is a legitimate (if
    /// hostile) thing to ask for — it must not read as "unset".
    #[test]
    fn an_empty_value_is_an_empty_mask_not_an_absent_one() {
        let mask = m("");
        assert!(mask.is_empty());
        assert_eq!(mask.apply(8).unwrap(), Vec::<u32>::new());
    }

    #[test]
    fn uuid_entries_parse_but_cannot_be_resolved_to_indices() {
        let mask = m("GPU-8f1b2c3d,1");
        assert_eq!(mask.entries.len(), 2);
        assert_eq!(mask.indices(), None);
        assert_eq!(mask.apply(8), None);
    }

    #[test]
    fn apply_drops_out_of_range_entries() {
        assert_eq!(m("0,9,2").apply(4).unwrap(), vec![0, 2]);
    }

    /// The mask's ORDER is the new numbering: `HIP_VISIBLE_DEVICES=2,0` makes
    /// physical 2 into device 0.
    #[test]
    fn the_mask_order_is_the_new_ordinal_order() {
        assert_eq!(m("2,0").apply(4).unwrap(), vec![2, 0]);
    }

    // ── the decision table ──────────────────────────────────────────────────

    #[test]
    fn nothing_set_means_every_agent() {
        assert_eq!(hsa_visibility(None, None), HsaVisibility::All);
    }

    #[test]
    fn rocr_alone_is_already_applied_by_rocr() {
        assert_eq!(
            hsa_visibility(Some(m("4,5")), None),
            HsaVisibility::RocrApplied("4,5".into())
        );
    }

    /// The fix: HIP alone used to be silently ignored, which handed the process
    /// every GPU on the box.
    #[test]
    fn hip_alone_is_applied_by_plowrt() {
        match hsa_visibility(None, Some(m("4,5"))) {
            HsaVisibility::ApplyHip { mask } => {
                assert_eq!(mask.indices().unwrap(), vec![4, 5])
            }
            other => panic!("expected ApplyHip, got {other:?}"),
        }
    }

    /// With both set, ROCr has already narrowed the set and HIP's absolute ids
    /// no longer address it. Applying HIP on top would select the wrong agents
    /// or none; ignoring it can only ever leave us inside what ROCr granted.
    #[test]
    fn both_set_ignores_hip_because_rocr_already_narrowed_the_set() {
        assert_eq!(
            hsa_visibility(Some(m("4")), Some(m("4"))),
            HsaVisibility::HipIgnored {
                rocr: "4".into(),
                hip: "4".into()
            }
        );
    }

    /// A UUID-only HIP mask is refused rather than ignored — ignoring it is
    /// exactly how a lease gets violated.
    #[test]
    fn a_uuid_hip_mask_is_refused_not_ignored() {
        assert_eq!(
            hsa_visibility(None, Some(m("GPU-abc"))),
            HsaVisibility::HipUnresolvable {
                hip: "GPU-abc".into()
            }
        );
    }
}
