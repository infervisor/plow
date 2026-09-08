//! §I.5 Placement — which device group each model lives on.
//!
//! A **group** is the set of device ordinals one model instance runs on:
//! degree 1 is a single GPU, degree *n* a tensor-parallel run over *n*
//! contiguous ordinals (the shape [`crate::exec::tp::TpGroup::split_replicas`]
//! already builds, and for the same reason — every pair on a node is 1-hop, so
//! contiguous runs are as good as any other partition).
//!
//! This module is deliberately pure: it takes ordinals, TP degrees and planned
//! byte requirements and returns an assignment. No device is opened, nothing is
//! loaded. `BlobPlan` can be read from a blob header without a GPU, so a whole
//! node's layout is decidable — and loggable, and testable — before the first
//! H2D.

use std::fmt;

/// How models are spread over the available groups.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Place {
    /// One model per group where possible: round-robin onto the least-loaded
    /// group, so co-residency happens only when models outnumber groups. Best
    /// latency isolation, and the default.
    #[default]
    Spread,
    /// Fill a group while the planner says the next model fits, then move on.
    /// Leaves whole GPUs free at the cost of co-tenants sharing SMs.
    Pack,
    /// Every model must name its device. Ambiguity is an error, not a guess.
    Explicit,
}

impl std::str::FromStr for Place {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "spread" => Ok(Place::Spread),
            "pack" => Ok(Place::Pack),
            "explicit" => Ok(Place::Explicit),
            other => Err(format!(
                "unknown placement {other:?} (expected spread, pack or explicit)"
            )),
        }
    }
}

/// One model, as the planner sees it.
#[derive(Clone, Debug)]
pub struct ModelSpec {
    pub slug: String,
    /// Tensor-parallel fan-out from the blob (`DevBlob.tp.n_gpu`), at least 1.
    pub tp: u32,
    /// Planner requirement in bytes: tensors + overhead + reserve.
    pub required: u64,
    /// Operator override: the first ordinal of the group this model must use.
    pub device: Option<u32>,
}

/// A contiguous run of device ordinals that one model instance can occupy.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Group {
    pub ordinals: Vec<u32>,
    /// Usable bytes on this group — the per-device capacity (groups are
    /// homogeneous; a TP model's weights are sharded, so its requirement is
    /// compared against one device's capacity).
    pub capacity: u64,
}

impl Group {
    pub fn degree(&self) -> u32 {
        self.ordinals.len() as u32
    }
    pub fn first(&self) -> u32 {
        self.ordinals[0]
    }
}

/// The computed layout: `assignment[i]` is the group index for `models[i]`.
#[derive(Clone, Debug)]
pub struct Layout {
    pub groups: Vec<Group>,
    pub assignment: Vec<usize>,
}

impl Layout {
    /// Slugs assigned to a group, in registration order.
    pub fn members<'a>(&'a self, models: &'a [ModelSpec], group: usize) -> Vec<&'a str> {
        self.assignment
            .iter()
            .enumerate()
            .filter(|(_, &g)| g == group)
            .map(|(i, _)| models[i].slug.as_str())
            .collect()
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum PlacementError {
    NoDevices,
    /// A model's TP degree does not divide the visible device count, so no
    /// contiguous run of that width exists.
    NoRunOfWidth { slug: String, tp: u32, visible: usize },
    /// An explicit `--device` named an ordinal with no group starting there.
    UnknownDevice { slug: String, device: u32 },
    /// `--place explicit` and a model did not name a device.
    MissingDevice { slug: String },
    /// A TP model was pinned to a group of the wrong width.
    DegreeMismatch { slug: String, tp: u32, degree: u32 },
    /// Nothing on the node has room for this model, even empty.
    WontFitAnywhere { slug: String, required: u64, capacity: u64 },
}

impl fmt::Display for PlacementError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PlacementError::NoDevices => write!(f, "no visible devices to place models on"),
            PlacementError::NoRunOfWidth { slug, tp, visible } => write!(
                f,
                "{slug} needs a tensor-parallel run of {tp} device(s), but only {visible} are visible"
            ),
            PlacementError::UnknownDevice { slug, device } => write!(
                f,
                "{slug} is pinned to device {device}, which is not the first ordinal of any group"
            ),
            PlacementError::MissingDevice { slug } => write!(
                f,
                "--place explicit: {slug} does not name a device (use slug@ordinal or --device)"
            ),
            PlacementError::DegreeMismatch { slug, tp, degree } => write!(
                f,
                "{slug} has tensor-parallel degree {tp} but is pinned to a group of {degree} device(s)"
            ),
            PlacementError::WontFitAnywhere {
                slug,
                required,
                capacity,
            } => write!(
                f,
                "{slug} needs {} MiB but a whole group holds only {} MiB",
                required >> 20,
                capacity >> 20
            ),
        }
    }
}

/// Partition `devices` into groups of width `tp`, over contiguous runs.
///
/// Every model on a node shares one grouping, so the width is the maximum TP
/// degree in play: a TP4 model and a TP1 model on an 8-GPU node give two groups
/// of four, and the TP1 model occupies one of them alone. Mixing widths would
/// mean overlapping groups, and two models sharing *some* of their devices is
/// the one shape that has no honest capacity answer.
pub fn plan_groups(devices: &[u32], tp: u32, capacity: u64) -> Result<Vec<Group>, PlacementError> {
    if devices.is_empty() {
        return Err(PlacementError::NoDevices);
    }
    let tp = tp.max(1) as usize;
    Ok(devices
        .chunks_exact(tp)
        .map(|run| Group {
            ordinals: run.to_vec(),
            capacity,
        })
        .collect())
}

/// Assign every model to a group.
///
/// Explicit pins are honoured first and unconditionally — an operator who named
/// a device gets it, and a pin that cannot be satisfied is an error rather than
/// a silent relocation. The policy then places whatever is left.
pub fn assign(
    models: &[ModelSpec],
    groups: &[Group],
    policy: Place,
) -> Result<Layout, PlacementError> {
    if groups.is_empty() {
        return Err(PlacementError::NoDevices);
    }
    let mut assignment = vec![usize::MAX; models.len()];
    // Bytes already promised to each group, so `spread` can pick the emptiest
    // and `pack` can tell when one is full.
    let mut used = vec![0u64; groups.len()];

    // Pass 1: explicit pins.
    for (i, m) in models.iter().enumerate() {
        let Some(dev) = m.device else { continue };
        let g = groups
            .iter()
            .position(|g| g.first() == dev)
            .ok_or_else(|| PlacementError::UnknownDevice {
                slug: m.slug.clone(),
                device: dev,
            })?;
        if groups[g].degree() != m.tp.max(1) {
            return Err(PlacementError::DegreeMismatch {
                slug: m.slug.clone(),
                tp: m.tp.max(1),
                degree: groups[g].degree(),
            });
        }
        assignment[i] = g;
        used[g] += m.required;
    }

    // Pass 2: the policy.
    for (i, m) in models.iter().enumerate() {
        if assignment[i] != usize::MAX {
            continue;
        }
        if m.required > groups[0].capacity {
            return Err(PlacementError::WontFitAnywhere {
                slug: m.slug.clone(),
                required: m.required,
                capacity: groups[0].capacity,
            });
        }
        let g = match policy {
            Place::Explicit => {
                return Err(PlacementError::MissingDevice {
                    slug: m.slug.clone(),
                })
            }
            // Emptiest group wins, ties to the lowest ordinal: with as many
            // groups as models this puts each on its own device, and past that
            // it degrades to balanced co-residency rather than piling onto one.
            Place::Spread => (0..groups.len())
                .min_by_key(|&g| (used[g], g))
                .expect("non-empty"),
            // First group with room, else the emptiest — packing must not fail
            // outright just because no group has room to spare, since the
            // manager can still switch models in and out of an oversubscribed
            // group.
            Place::Pack => (0..groups.len())
                .find(|&g| used[g] + m.required <= groups[g].capacity)
                .unwrap_or_else(|| {
                    (0..groups.len())
                        .min_by_key(|&g| (used[g], g))
                        .expect("non-empty")
                }),
        };
        assignment[i] = g;
        used[g] += m.required;
    }

    Ok(Layout {
        groups: groups.to_vec(),
        assignment,
    })
}

/// Parse `--pin slug@ordinal` entries into a slug -> ordinal map.
///
/// Rejected rather than ignored: a pin that does not parse is an operator
/// saying where a model goes, and silently placing it somewhere else is the
/// one outcome that must not happen.
pub fn parse_pins(pins: &[String]) -> Result<Vec<(String, u32)>, String> {
    let mut out = Vec::with_capacity(pins.len());
    for pin in pins {
        let (slug, ord) = pin
            .rsplit_once('@')
            .ok_or_else(|| format!("--pin {pin:?} is not slug@ordinal"))?;
        if slug.is_empty() {
            return Err(format!("--pin {pin:?} names no model"));
        }
        let ord: u32 = ord
            .parse()
            .map_err(|_| format!("--pin {pin:?}: {ord:?} is not a device ordinal"))?;
        out.push((slug.to_string(), ord));
    }
    Ok(out)
}

/// The maximum TP degree across models — the grouping width (see [`plan_groups`]).
pub fn grouping_width(models: &[ModelSpec], visible: usize) -> Result<u32, PlacementError> {
    let width = models.iter().map(|m| m.tp.max(1)).max().unwrap_or(1);
    if width as usize > visible {
        let m = models
            .iter()
            .find(|m| m.tp.max(1) == width)
            .expect("width came from a model");
        return Err(PlacementError::NoRunOfWidth {
            slug: m.slug.clone(),
            tp: width,
            visible,
        });
    }
    Ok(width)
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: u64 = 1 << 30;

    fn spec(slug: &str, required: u64) -> ModelSpec {
        ModelSpec {
            slug: slug.into(),
            tp: 1,
            required,
            device: None,
        }
    }

    fn groups(n: u32, capacity: u64) -> Vec<Group> {
        plan_groups(&(0..n).collect::<Vec<_>>(), 1, capacity).unwrap()
    }

    #[test]
    fn one_group_per_device_at_tp1() {
        let g = plan_groups(&[0, 1, 2, 3], 1, 80 * GIB).unwrap();
        assert_eq!(g.len(), 4);
        assert_eq!(g[2].ordinals, vec![2]);
    }

    /// The deployment shape the TP code already targets: 8 GPUs as two TP4
    /// replicas, over contiguous runs.
    #[test]
    fn tp4_on_eight_devices_gives_two_contiguous_replicas() {
        let g = plan_groups(&(0..8).collect::<Vec<_>>(), 4, 80 * GIB).unwrap();
        assert_eq!(g.len(), 2);
        assert_eq!(g[0].ordinals, vec![0, 1, 2, 3]);
        assert_eq!(g[1].ordinals, vec![4, 5, 6, 7]);
    }

    /// A run that does not divide the device count leaves the remainder unused
    /// rather than forming a short, unusable group.
    #[test]
    fn a_partial_run_is_not_a_group() {
        let g = plan_groups(&[0, 1, 2], 2, 80 * GIB).unwrap();
        assert_eq!(g.len(), 1);
        assert_eq!(g[0].ordinals, vec![0, 1]);
    }

    #[test]
    fn spread_puts_each_model_on_its_own_device() {
        let models = vec![spec("a", 20 * GIB), spec("b", 20 * GIB), spec("c", 20 * GIB)];
        let l = assign(&models, &groups(4, 80 * GIB), Place::Spread).unwrap();
        assert_eq!(l.assignment, vec![0, 1, 2]);
    }

    /// Past one-per-device, spread balances rather than piling onto device 0.
    #[test]
    fn spread_balances_when_models_outnumber_devices() {
        let models = vec![
            spec("a", 40 * GIB),
            spec("b", 10 * GIB),
            spec("c", 10 * GIB),
            spec("d", 10 * GIB),
        ];
        let l = assign(&models, &groups(2, 80 * GIB), Place::Spread).unwrap();
        // a→0, b→1, then the emptiest is 1 (10) vs 0 (40), so c→1, then d→1
        // (20) vs 0 (40).
        assert_eq!(l.assignment, vec![0, 1, 1, 1]);
        assert_eq!(l.members(&models, 0), vec!["a"]);
        assert_eq!(l.members(&models, 1), vec!["b", "c", "d"]);
    }

    #[test]
    fn pack_fills_a_device_before_moving_on() {
        let models = vec![spec("a", 30 * GIB), spec("b", 30 * GIB), spec("c", 30 * GIB)];
        let l = assign(&models, &groups(2, 80 * GIB), Place::Pack).unwrap();
        // a+b = 60 ≤ 80 so both land on 0; c would make 90, so it moves to 1.
        assert_eq!(l.assignment, vec![0, 0, 1]);
    }

    /// The distinction between the two policies is the whole point of the flag.
    #[test]
    fn spread_and_pack_disagree_on_the_same_input() {
        let models = vec![spec("a", 10 * GIB), spec("b", 10 * GIB)];
        let g = groups(2, 80 * GIB);
        assert_eq!(assign(&models, &g, Place::Spread).unwrap().assignment, [0, 1]);
        assert_eq!(assign(&models, &g, Place::Pack).unwrap().assignment, [0, 0]);
    }

    #[test]
    fn an_explicit_pin_wins_over_the_policy() {
        let mut models = vec![spec("a", 10 * GIB), spec("b", 10 * GIB)];
        models[0].device = Some(3);
        let l = assign(&models, &groups(4, 80 * GIB), Place::Spread).unwrap();
        assert_eq!(l.assignment[0], 3);
        // b takes an empty group, not the one now holding a.
        assert_ne!(l.assignment[1], 3);
    }

    /// A pin that cannot be honoured is an error. Silently relocating a model
    /// an operator placed by hand is the one outcome that is never wanted.
    #[test]
    fn an_unsatisfiable_pin_is_an_error() {
        let mut models = vec![spec("a", 10 * GIB)];
        models[0].device = Some(9);
        assert_eq!(
            assign(&models, &groups(2, 80 * GIB), Place::Spread).unwrap_err(),
            PlacementError::UnknownDevice {
                slug: "a".into(),
                device: 9
            }
        );
    }

    #[test]
    fn a_tp_model_pinned_to_a_narrow_group_is_an_error() {
        let mut models = vec![spec("a", 10 * GIB)];
        models[0].tp = 4;
        models[0].device = Some(0);
        assert!(matches!(
            assign(&models, &groups(2, 80 * GIB), Place::Spread),
            Err(PlacementError::DegreeMismatch { .. })
        ));
    }

    #[test]
    fn explicit_policy_refuses_an_unplaced_model() {
        let models = vec![spec("a", 10 * GIB)];
        assert_eq!(
            assign(&models, &groups(2, 80 * GIB), Place::Explicit).unwrap_err(),
            PlacementError::MissingDevice { slug: "a".into() }
        );
    }

    /// A model larger than a whole GPU is refused at plan time, before any
    /// load, rather than after a multi-second checkpoint upload.
    #[test]
    fn a_model_too_big_for_any_group_is_refused_up_front() {
        let models = vec![spec("huge", 200 * GIB)];
        assert!(matches!(
            assign(&models, &groups(4, 80 * GIB), Place::Spread),
            Err(PlacementError::WontFitAnywhere { .. })
        ));
    }

    /// Packing an oversubscribed node still produces a layout — the manager
    /// switches models within a group, so "does not all fit at once" is normal.
    #[test]
    fn pack_still_places_everything_when_nothing_has_room_to_spare() {
        let models = vec![spec("a", 70 * GIB), spec("b", 70 * GIB), spec("c", 70 * GIB)];
        let l = assign(&models, &groups(2, 80 * GIB), Place::Pack).unwrap();
        assert!(l.assignment.iter().all(|&g| g < 2));
    }

    #[test]
    fn grouping_width_is_the_widest_model() {
        let mut models = vec![spec("a", GIB), spec("b", GIB)];
        models[1].tp = 4;
        assert_eq!(grouping_width(&models, 8).unwrap(), 4);
        // ...and a node too small for it is refused by name.
        assert_eq!(
            grouping_width(&models, 2),
            Err(PlacementError::NoRunOfWidth {
                slug: "b".into(),
                tp: 4,
                visible: 2
            })
        );
    }

    #[test]
    fn pins_parse_and_bad_ones_are_refused() {
        let pins = vec!["a@0".to_string(), "b@3".to_string()];
        assert_eq!(
            parse_pins(&pins).unwrap(),
            vec![("a".to_string(), 0), ("b".to_string(), 3)]
        );
        // A slug may itself contain '@' (an org-qualified name), so the SPLIT
        // is from the right.
        assert_eq!(
            parse_pins(&["org@v1/model@2".to_string()]).unwrap(),
            vec![("org@v1/model".to_string(), 2)]
        );
        for bad in ["a", "a@", "@2", "a@x", ""] {
            assert!(
                parse_pins(&[bad.to_string()]).is_err(),
                "{bad:?} should be refused"
            );
        }
    }

    #[test]
    fn place_parses_its_three_policies_and_rejects_others() {
        assert_eq!("spread".parse::<Place>().unwrap(), Place::Spread);
        assert_eq!("pack".parse::<Place>().unwrap(), Place::Pack);
        assert_eq!("explicit".parse::<Place>().unwrap(), Place::Explicit);
        assert!("packed".parse::<Place>().is_err());
    }
}
