use std::collections::BTreeMap;
use std::sync::Arc;

use plow_asset::packet_pipeline::{HandoffBindPhase, HandoffPlan, HandoffRowMap};

use super::reject;
use crate::Result;

#[derive(Clone, Debug)]
pub(crate) struct RowKey {
    owner: Arc<()>,
    domain: usize,
    slot: u32,
    epoch: u64,
}

pub(crate) struct Admission {
    key: RowKey,
}

impl Admission {
    pub(crate) fn slot(&self) -> u32 {
        self.key.slot
    }
    pub(crate) fn epoch(&self) -> u64 {
        self.key.epoch
    }
}

impl RowKey {
    pub(crate) fn slot(&self) -> u32 {
        self.slot
    }
    pub(crate) fn epoch(&self) -> u64 {
        self.epoch
    }
}

struct Domain {
    name: String,
    epochs: Vec<u64>,
    seen: Vec<u8>,
}

pub(super) struct RowDomains {
    owner: Arc<()>,
    names: BTreeMap<String, usize>,
    domains: Vec<Domain>,
    next_epoch: u64,
}

impl RowDomains {
    pub(super) fn new(plan: &HandoffPlan) -> Result<Self> {
        let mut capacities = BTreeMap::<String, u32>::new();
        for view in plan.views.values() {
            if let HandoffRowMap::Gather { domain, .. } = &view.row_map {
                let cap = capacities.entry(domain.clone()).or_default();
                *cap = (*cap).max(view.rows);
            }
        }
        for bind in &plan.bind_writes {
            if bind.phase == HandoffBindPhase::Admission {
                let view = plan.view(&bind.view).map_err(reject)?;
                let domain = match &view.row_map {
                    HandoffRowMap::Contiguous { domain } | HandoffRowMap::Gather { domain, .. } => {
                        domain
                    }
                };
                let cap = capacities.entry(domain.clone()).or_default();
                *cap = (*cap).max(view.rows);
            }
        }
        Ok(Self::from_capacities(capacities))
    }

    fn from_capacities(capacities: BTreeMap<String, u32>) -> Self {
        let mut names = BTreeMap::new();
        let domains = capacities
            .into_iter()
            .enumerate()
            .map(|(index, (name, capacity))| {
                names.insert(name.clone(), index);
                Domain {
                    name,
                    epochs: vec![0; capacity as usize],
                    seen: vec![0; capacity as usize],
                }
            })
            .collect();
        Self {
            owner: Arc::new(()),
            names,
            domains,
            next_epoch: 1,
        }
    }

    pub(super) fn prepare(&mut self, domain: &str, slot: u32) -> Result<Admission> {
        let domain = *self
            .names
            .get(domain)
            .ok_or_else(|| reject("unknown admission domain"))?;
        let current = self.domains[domain]
            .epochs
            .get(slot as usize)
            .ok_or_else(|| reject("admission slot exceeds the packet capacity"))?;
        if *current != 0 {
            return Err(reject("admission slot is still live"));
        }
        let epoch = self.next_epoch;
        self.next_epoch = epoch
            .checked_add(1)
            .ok_or_else(|| reject("handoff generation exhausted"))?;
        Ok(Admission {
            key: RowKey {
                owner: self.owner.clone(),
                domain,
                slot,
                epoch,
            },
        })
    }

    pub(super) fn check_admission(&self, admission: &Admission) -> Result<()> {
        let key = &admission.key;
        if !Arc::ptr_eq(&self.owner, &key.owner)
            || self
                .domains
                .get(key.domain)
                .and_then(|d| d.epochs.get(key.slot as usize))
                != Some(&0)
        {
            return Err(reject(
                "admission belongs to another executor or a live slot",
            ));
        }
        Ok(())
    }

    pub(super) fn domain(&self, admission: &Admission) -> Result<&str> {
        self.check_admission(admission)?;
        Ok(&self.domains[admission.key.domain].name)
    }

    // Publish only after every declared bind/reset write for this row has retired.
    pub(super) fn commit(&mut self, admission: Admission) -> Result<RowKey> {
        self.check_admission(&admission)?;
        let key = admission.key;
        self.domains[key.domain].epochs[key.slot as usize] = key.epoch;
        Ok(key)
    }

    fn check_key(&self, key: &RowKey) -> Result<()> {
        if !Arc::ptr_eq(&self.owner, &key.owner)
            || self
                .domains
                .get(key.domain)
                .and_then(|d| d.epochs.get(key.slot as usize))
                != Some(&key.epoch)
        {
            return Err(reject("stale or foreign handoff row generation"));
        }
        Ok(())
    }

    pub(super) fn release(&mut self, key: &RowKey) -> Result<()> {
        self.check_key(key)?;
        self.domains[key.domain].epochs[key.slot as usize] = 0;
        self.domains[key.domain].seen[key.slot as usize] = 0;
        Ok(())
    }

    pub(super) fn live(&mut self, keys: &[&RowKey]) -> Result<()> {
        for domain in &mut self.domains {
            domain.seen.fill(0);
        }
        let result = (|| {
            for key in keys {
                self.check_key(key)?;
                let seen = &mut self.domains[key.domain].seen[key.slot as usize];
                if *seen != 0 {
                    return Err(reject("duplicate live handoff row"));
                }
                *seen = 1;
            }
            Ok(())
        })();
        if result.is_err() {
            for domain in &mut self.domains {
                domain.seen.fill(0);
            }
        }
        result
    }

    pub(super) fn contains(&self, domain: &str, slot: u32) -> bool {
        self.names
            .get(domain)
            .and_then(|&i| self.domains[i].seen.get(slot as usize))
            == Some(&1)
    }

    pub(super) fn selection(
        &self,
        domain: &str,
        keys: &[&RowKey],
        bytes: &[u8],
        capacity: u32,
        unique: bool,
    ) -> Result<()> {
        let domain_id = self
            .names
            .get(domain)
            .ok_or_else(|| reject("unknown map domain"))?;
        if keys.len().checked_mul(4) != Some(bytes.len()) {
            return Err(reject("map and generation selection lengths differ"));
        }
        for (index, (key, word)) in keys.iter().zip(bytes.chunks_exact(4)).enumerate() {
            self.check_key(key)?;
            let slot = u32::from_le_bytes(word.try_into().unwrap());
            if key.domain != *domain_id
                || key.slot != slot
                || slot >= capacity
                || !self.contains(domain, slot)
            {
                return Err(reject("map row differs from its admitted live generation"));
            }
            if unique && keys[..index].iter().any(|old| old.slot == slot) {
                return Err(reject("writable map selects a physical row twice"));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn domains() -> RowDomains {
        RowDomains::from_capacities(BTreeMap::from([
            ("physical".into(), 128),
            ("codec".into(), 128),
        ]))
    }

    #[test]
    fn map_selection_proves_order_domain_generation_and_writable_uniqueness() {
        let mut rows = domains();
        let a = rows.prepare("physical", 97).unwrap();
        let a = rows.commit(a).unwrap();
        let b = rows.prepare("physical", 5).unwrap();
        let b = rows.commit(b).unwrap();
        let codec = rows.prepare("codec", 5).unwrap();
        let codec = rows.commit(codec).unwrap();
        rows.live(&[&a, &b, &codec]).unwrap();
        let bytes: Vec<_> = [97u32, 5].into_iter().flat_map(u32::to_le_bytes).collect();
        rows.selection("physical", &[&a, &b], &bytes, 128, true)
            .unwrap();
        assert!(rows
            .selection("physical", &[&a, &b], &bytes, 64, true)
            .is_err());
        rows.selection("codec", &[&codec], &5u32.to_le_bytes(), 128, true)
            .unwrap();
        assert!(rows
            .selection("physical", &[&b, &a], &bytes, 128, true)
            .is_err());
        assert!(rows
            .selection("physical", &[&codec], &5u32.to_le_bytes(), 128, true)
            .is_err());
        assert!(rows
            .selection("physical", &[&a], &bytes, 128, true)
            .is_err());
        let repeated: Vec<_> = [97u32, 97].into_iter().flat_map(u32::to_le_bytes).collect();
        rows.selection("physical", &[&a, &a], &repeated, 128, false)
            .unwrap();
        assert!(rows
            .selection("physical", &[&a, &a], &repeated, 128, true)
            .is_err());
        rows.live(&[&b]).unwrap();
        assert!(rows
            .selection("physical", &[&a], &97u32.to_le_bytes(), 128, true)
            .is_err());
        rows.release(&a).unwrap();
        let next = rows.prepare("physical", 97).unwrap();
        let next = rows.commit(next).unwrap();
        rows.live(&[&next]).unwrap();
        assert!(rows
            .selection("physical", &[&a], &97u32.to_le_bytes(), 128, true)
            .is_err());
        rows.selection("physical", &[&next], &97u32.to_le_bytes(), 128, true)
            .unwrap();
    }

    #[test]
    fn prepared_rows_are_not_live_and_domains_have_independent_sparse_slots() {
        let mut rows = domains();
        let physical = rows.prepare("physical", 97).unwrap();
        let codec = rows.prepare("codec", 5).unwrap();
        assert_eq!(physical.slot(), 97);
        assert_ne!(physical.epoch(), codec.epoch());
        assert!(!rows.contains("physical", 97));
        let physical = rows.commit(physical).unwrap();
        let codec = rows.commit(codec).unwrap();
        rows.live(&[&physical, &codec]).unwrap();
        assert!(rows.contains("physical", 97));
        assert!(rows.contains("codec", 5));
        assert!(!rows.contains("physical", 5));
        assert!(!rows.contains("codec", 97));
        assert!(rows.prepare("physical", 97).is_err());
        assert!(rows.prepare("physical", 128).is_err());
    }

    #[test]
    fn stopped_reused_and_foreign_rows_cannot_qualify_a_new_tick() {
        let mut rows = domains();
        let first = rows.prepare("physical", 97).unwrap();
        let first = rows.commit(first).unwrap();
        rows.release(&first).unwrap();
        assert!(rows.live(&[&first]).is_err());
        let next = rows.prepare("physical", 97).unwrap();
        let next = rows.commit(next).unwrap();
        assert_ne!(first.epoch, next.epoch);
        assert!(rows.release(&first).is_err());
        assert!(rows.live(&[&next, &next]).is_err());
        assert!(!rows.contains("physical", 97));
        rows.live(&[&next]).unwrap();
        let mut other = domains();
        assert!(other.live(&[&next]).is_err());
        assert!(other.release(&next).is_err());
        assert!(rows.contains("physical", 97));
    }

    #[test]
    fn duplicate_preparations_and_epoch_exhaustion_do_not_overwrite_live_state() {
        let mut rows = domains();
        let a = rows.prepare("physical", 7).unwrap();
        let b = rows.prepare("physical", 7).unwrap();
        assert_eq!(rows.domain(&a).unwrap(), "physical");
        let a = rows.commit(a).unwrap();
        assert!(rows.commit(b).is_err());
        let mut other = domains();
        let foreign = other.prepare("physical", 8).unwrap();
        assert!(rows.commit(foreign).is_err());
        rows.next_epoch = u64::MAX;
        assert!(rows.prepare("physical", 8).is_err());
        rows.live(&[&a]).unwrap();
        assert!(rows.contains("physical", 7));
    }
}
