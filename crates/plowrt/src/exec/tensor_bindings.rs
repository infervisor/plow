use crate::device::{provenance::MemoryRegion, DeviceMem};
use crate::{Result, RuntimeError};

pub(crate) fn slot_bindings(
    memories: &[DeviceMem],
    strides: &[(usize, u64)],
    slot: usize,
) -> Result<Vec<(usize, TensorBinding)>> {
    strides
        .iter()
        .map(|&(i, stride)| {
            let reject = || RuntimeError::Rejected("invalid KV tensor binding extent".into());
            let mem = memories.get(i).ok_or_else(reject)?;
            let base = stride
                .checked_mul(slot as u64)
                .and_then(|offset| mem.base.checked_add(offset))
                .ok_or_else(reject)?;
            let bytes = if slot == 0 { mem.len } else { stride };
            Ok((i, TensorBinding::capture(mem, base, bytes)?))
        })
        .collect()
}

pub(crate) struct TensorBinding {
    address: u64,
    bytes: u64,
    region: Option<MemoryRegion>,
}

pub(crate) fn band_bindings(
    memories: &[DeviceMem],
    names: &[String],
    views: &[(usize, u64)],
) -> Result<Vec<(usize, TensorBinding)>> {
    views
        .iter()
        .map(|&(i, address)| {
            let reject = || RuntimeError::Rejected("invalid band binding parent".into());
            let (parent, _) = names
                .get(i)
                .ok_or_else(reject)?
                .split_once("@band")
                .ok_or_else(reject)?;
            let parent = names
                .iter()
                .position(|name| name == parent)
                .ok_or_else(reject)?;
            let bytes = memories.get(i).ok_or_else(reject)?.len;
            Ok((
                i,
                TensorBinding::capture(memories.get(parent).ok_or_else(reject)?, address, bytes)?,
            ))
        })
        .collect()
}

impl TensorBinding {
    pub(crate) fn address(&self) -> u64 {
        self.address
    }

    pub(crate) fn capture(owner: &DeviceMem, address: u64, bytes: u64) -> Result<Self> {
        Ok(Self {
            address,
            bytes,
            region: owner.binding_region(address, bytes)?,
        })
    }

    pub(crate) fn whole(owner: &DeviceMem) -> Result<Self> {
        Self::capture(owner, owner.base, owner.len)
    }

    fn validate(&self) -> Result<()> {
        if self.region.as_ref().is_some_and(|r| !r.is_live()) {
            return Err(RuntimeError::Rejected(
                "stale tensor allocation or mapping generation".into(),
            ));
        }
        Ok(())
    }
}

/// Unknown raw-pointer views remain usable for bringup, never physically qualified.
/// This ledger observes rebinding boundaries, not GPU retirement/completion.
pub(crate) struct TensorBindings {
    slots: Vec<TensorBinding>,
    generation: u64,
    valid: bool,
}

impl TensorBindings {
    pub(crate) fn new(slots: Vec<TensorBinding>) -> Result<Self> {
        let table = Self {
            slots,
            generation: 1,
            valid: true,
        };
        table.validate()?;
        Ok(table)
    }

    fn validate(&self) -> Result<()> {
        if !self.valid {
            return Err(RuntimeError::Rejected(
                "tensor table upload did not complete".into(),
            ));
        }
        for slot in &self.slots {
            slot.validate()?;
        }
        Ok(())
    }

    pub(crate) fn upload(&mut self, upload: impl FnOnce(&[u8]) -> Result<()>) -> Result<Vec<u8>> {
        self.rebind(Vec::new(), upload)
    }

    pub(crate) fn rebind(
        &mut self,
        mut updates: Vec<(usize, TensorBinding)>,
        upload: impl FnOnce(&[u8]) -> Result<()>,
    ) -> Result<Vec<u8>> {
        self.validate()?;
        let generation = self
            .generation
            .checked_add(1)
            .ok_or_else(|| RuntimeError::Rejected("tensor binding generation exhausted".into()))?;
        updates.sort_unstable_by_key(|(index, _)| *index);
        let mut previous = None;
        for (index, binding) in &updates {
            if *index >= self.slots.len() || previous == Some(*index) {
                return Err(RuntimeError::Rejected(
                    "duplicate or missing tensor binding slot".into(),
                ));
            }
            previous = Some(*index);
            binding.validate()?;
        }
        let mut bytes: Vec<u8> = self
            .slots
            .iter()
            .flat_map(|slot| slot.address.to_le_bytes())
            .collect();
        for (index, binding) in &updates {
            bytes[index * 8..index * 8 + 8].copy_from_slice(&binding.address.to_le_bytes());
        }
        self.valid = false;
        upload(&bytes)?;
        for (index, binding) in updates {
            self.slots[index] = binding;
        }
        self.generation = generation;
        self.valid = true;
        Ok(bytes)
    }

    pub(crate) fn generation(&self) -> u64 {
        self.generation
    }

    pub(crate) fn check_pointers(&self, pointers: &[u64]) -> Result<()> {
        self.validate()?;
        if pointers.len() != self.slots.len()
            || pointers
                .iter()
                .zip(&self.slots)
                .any(|(pointer, slot)| *pointer != slot.address)
        {
            return Err(RuntimeError::Rejected(
                "tensor table differs from allocation bindings".into(),
            ));
        }
        Ok(())
    }

    pub(crate) fn fully_tracked(&self) -> bool {
        self.validate().is_ok() && self.slots.iter().all(|s| s.region.is_some())
    }

    pub(crate) fn require_disjoint(&self, a: usize, b: usize) -> Result<()> {
        self.validate()?;
        let evidence = |i: usize| {
            self.slots
                .get(i)
                .and_then(|s| s.region.as_ref()?.evidence(s.address, s.bytes))
        };
        let (Some(a), Some(b)) = (evidence(a), evidence(b)) else {
            return Err(RuntimeError::Rejected(
                "untracked tensor disjointness".into(),
            ));
        };
        if !crate::device::provenance::ranges_disjoint(&a, &b) {
            return Err(RuntimeError::Rejected("physical tensor alias".into()));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::{cpu::CpuBackend, Backend};

    #[test]
    fn tensor_bindings_load_rebind_and_owner_drop() {
        let be = CpuBackend::new(1);
        let owner = be.alloc(0, 64).unwrap();
        let mut table = TensorBindings::new(vec![TensorBinding::whole(&owner).unwrap()]).unwrap();
        assert!(table.fully_tracked());
        let initial = table
            .upload(|bytes| {
                assert_eq!(bytes, owner.base.to_le_bytes());
                Ok(())
            })
            .unwrap();
        let generation = table.generation();
        assert!(table.check_pointers(&[owner.base + 1]).is_err());
        assert!(table.check_pointers(&[]).is_err());
        let changed = table
            .rebind(
                vec![(
                    0,
                    TensorBinding::capture(&owner, owner.base + 32, 32).unwrap(),
                )],
                |_| Ok(()),
            )
            .unwrap();
        assert_ne!(initial, changed);
        assert_eq!(table.generation(), generation + 1);
        assert!(TensorBinding::capture(&owner, owner.base + 32, 33).is_err());
        let mut mutated = owner.subview(0, 32).unwrap();
        mutated.base += 1;
        assert!(TensorBinding::whole(&mutated).is_err());
        drop(owner);
        assert!(table
            .upload(|_| panic!("must reject before upload"))
            .is_err());
    }

    #[test]
    fn tensor_bindings_actual_slot_producer_checks_geometry_and_restore() {
        let memories = vec![CpuBackend::new(1).alloc(0, 128).unwrap()];
        let mut table =
            TensorBindings::new(vec![TensorBinding::whole(&memories[0]).unwrap()]).unwrap();
        for slot in [0, 1, 3, 0] {
            let bytes = table
                .rebind(slot_bindings(&memories, &[(0, 32)], slot).unwrap(), |_| {
                    Ok(())
                })
                .unwrap();
            assert_eq!(bytes, (memories[0].base + 32 * slot as u64).to_le_bytes());
        }
        for (index, stride, slot) in [(0, 32, 4), (1, 32, 1), (0, u64::MAX, 2)] {
            assert!(slot_bindings(&memories, &[(index, stride)], slot).is_err());
        }
    }

    #[test]
    fn tensor_bindings_actual_band_producer_uses_parent_not_old_view() {
        let owner = CpuBackend::new(1).alloc(0, 128).unwrap();
        let old = owner.subview(64, 64).unwrap();
        let base = owner.base;
        let memories = vec![owner, old];
        let names = vec!["act.x".into(), "act.x@band16".into()];
        let mut table = TensorBindings::new(
            memories
                .iter()
                .map(TensorBinding::whole)
                .collect::<Result<Vec<_>>>()
                .unwrap(),
        )
        .unwrap();
        for offset in [24, 64] {
            let changed = table
                .rebind(
                    band_bindings(&memories, &names, &[(1, base + offset)]).unwrap(),
                    |_| Ok(()),
                )
                .unwrap();
            assert_eq!(&changed[8..16], &(base + offset).to_le_bytes());
            assert!(table.require_disjoint(0, 1).is_err());
        }
        assert!(band_bindings(&memories, &names, &[(1, base + 65)]).is_err());
        assert!(band_bindings(&memories, &names, &[(0, base)]).is_err());
        assert!(band_bindings(&memories, &names, &[(2, base)]).is_err());
    }

    #[test]
    fn tensor_bindings_preserve_unknowns_and_reject_alias_disjointness() {
        let be = CpuBackend::new(1);
        let owner = be.alloc(0, 64).unwrap();
        let alias = owner.subview(16, 32).unwrap();
        let table = TensorBindings::new(vec![
            TensorBinding::whole(&owner).unwrap(),
            TensorBinding::whole(&alias).unwrap(),
        ])
        .unwrap();
        assert!(table.require_disjoint(0, 1).is_err());
        let raw = DeviceMem::view(owner.base, owner.len);
        let mut unknown = TensorBindings::new(vec![TensorBinding::whole(&raw).unwrap()]).unwrap();
        assert!(!unknown.fully_tracked());
        assert!(unknown.require_disjoint(0, 0).is_err());
        unknown.upload(|_| Ok(())).unwrap();
    }

    #[test]
    fn tensor_bindings_upload_failure_and_bad_updates_fail_closed() {
        let owner = CpuBackend::new(1).alloc(0, 64).unwrap();
        let mut table = TensorBindings::new(vec![TensorBinding::whole(&owner).unwrap()]).unwrap();
        assert!(table
            .rebind(
                vec![(1, TensorBinding::whole(&owner).unwrap())],
                |_| panic!()
            )
            .is_err());
        assert!(table
            .rebind(
                vec![
                    (0, TensorBinding::whole(&owner).unwrap()),
                    (0, TensorBinding::whole(&owner).unwrap())
                ],
                |_| panic!()
            )
            .is_err());
        let generation = table.generation();
        assert!(table
            .upload(|_| Err(RuntimeError::Rejected("injected upload failure".into())))
            .is_err());
        assert_eq!(table.generation(), generation);
        assert!(!table.fully_tracked());
        assert!(table.upload(|_| panic!("invalid table reused")).is_err());
    }

    #[test]
    fn tensor_bindings_recycled_handle_and_replaced_owner_reject() {
        use crate::device::provenance::VmmProvenance;
        let mut p = VmmProvenance::default();
        p.created(1, 64);
        p.mapped(1024, 64, 1);
        p.mapped(2048, 64, 1);
        let old = DeviceMem::mapped_view(1024, 64, p.region(1024, 64));
        let alias = DeviceMem::mapped_view(2048, 64, p.region(2048, 64));
        let mut table = TensorBindings::new(vec![
            TensorBinding::whole(&old).unwrap(),
            TensorBinding::whole(&alias).unwrap(),
        ])
        .unwrap();
        assert!(table.require_disjoint(0, 1).is_err());
        p.unmapped(1024, 64);
        p.released(1);
        p.created(1, 64);
        p.mapped(1024, 64, 1);
        let new = DeviceMem::mapped_view(1024, 64, p.region(1024, 64));
        assert!(TensorBinding::whole(&old).is_err());
        assert!(table
            .rebind(vec![(0, TensorBinding::whole(&new).unwrap())], |_| panic!(
                "stale binding cannot be silently replaced"
            ))
            .is_err());
        assert!(TensorBindings::new(vec![TensorBinding::whole(&new).unwrap()]).is_ok());
    }

    #[test]
    fn tensor_bindings_unmap_remap_without_release_still_invalidates() {
        use crate::device::provenance::VmmProvenance;
        let mut p = VmmProvenance::default();
        p.created(1, 64);
        p.mapped(1024, 64, 1);
        let mem = DeviceMem::mapped_view(1024, 64, p.region(1024, 64));
        let mut table = TensorBindings::new(vec![TensorBinding::whole(&mem).unwrap()]).unwrap();
        p.unmapped(1024, 64);
        p.mapped(1024, 64, 1);
        assert!(table
            .upload(|_| panic!("same physical ID does not restore an old map generation"))
            .is_err());
        assert!(!table.fully_tracked());
    }
}
