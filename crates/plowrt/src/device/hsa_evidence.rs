use std::collections::BTreeMap;

use serde::Serialize;

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub struct ResolvedKernelEvidence {
    pub entry: String,
    pub kernarg_bytes: u32,
    pub static_lds_bytes: u32,
    pub private_bytes_per_workitem: u32,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub struct LoadedObjectEvidence {
    pub sha256: String,
    pub bytes: usize,
    pub resolved_kernels: Vec<ResolvedKernelEvidence>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct SelectedKernelEvidence {
    pub object_sha256: String,
    pub object_bytes: usize,
    pub symbol: ResolvedKernelEvidence,
}

#[derive(Default)]
pub(super) struct LoadedObjects {
    objects: BTreeMap<u64, LoadedObjectEvidence>,
    kernels: BTreeMap<u64, Vec<(u64, String)>>,
}

impl LoadedObjects {
    pub fn loaded(&mut self, handle: u64, image: &[u8]) {
        self.unloaded(handle);
        self.objects.insert(
            handle,
            LoadedObjectEvidence {
                sha256: plow_asset::decode_objects::image_sha256(image),
                bytes: image.len(),
                resolved_kernels: Vec::new(),
            },
        );
    }

    pub fn resolved(
        &mut self,
        handle: u64,
        kernel_object: u64,
        kernel: ResolvedKernelEvidence,
    ) -> Result<(), String> {
        let object = self
            .objects
            .get_mut(&handle)
            .ok_or("untracked HSA executable")?;
        let key = (handle, kernel.entry.clone());
        if let Some(previous) = object
            .resolved_kernels
            .iter()
            .find(|k| k.entry == kernel.entry)
        {
            if previous != &kernel {
                return Err("HSA symbol resources changed in a frozen executable".into());
            }
        } else {
            object.resolved_kernels.push(kernel);
            object.resolved_kernels.sort();
        }
        let entries = self.kernels.entry(kernel_object).or_default();
        if !entries.contains(&key) {
            entries.push(key);
        }
        Ok(())
    }

    pub fn selected(
        &self,
        kernel_object: u64,
        kernarg: u32,
        lds: u32,
        private: u32,
    ) -> Result<SelectedKernelEvidence, String> {
        let entries = self
            .kernels
            .get(&kernel_object)
            .ok_or("untracked selected HSA kernel")?;
        let [(handle, entry)] = entries.as_slice() else {
            return Err("selected HSA kernel has ambiguous symbol identity".into());
        };
        let object = self
            .objects
            .get(handle)
            .ok_or("unloaded selected HSA executable")?;
        let symbol = object
            .resolved_kernels
            .iter()
            .find(|k| &k.entry == entry)
            .ok_or("untracked selected HSA symbol")?;
        if symbol.kernarg_bytes != kernarg
            || symbol.static_lds_bytes != lds
            || symbol.private_bytes_per_workitem != private
        {
            return Err("selected HSA kernel resources differ from resolved symbol".into());
        }
        Ok(SelectedKernelEvidence {
            object_sha256: object.sha256.clone(),
            object_bytes: object.bytes,
            symbol: symbol.clone(),
        })
    }

    pub fn unloaded(&mut self, handle: u64) {
        self.objects.remove(&handle);
        self.kernels.retain(|_, entries| {
            entries.retain(|(module, _)| *module != handle);
            !entries.is_empty()
        });
    }

    pub fn snapshot(&self) -> Vec<LoadedObjectEvidence> {
        let mut objects: Vec<_> = self.objects.values().cloned().collect();
        objects.sort();
        objects
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kernel(entry: &str) -> ResolvedKernelEvidence {
        ResolvedKernelEvidence {
            entry: entry.into(),
            kernarg_bytes: 320,
            static_lds_bytes: 8192,
            private_bytes_per_workitem: 16,
        }
    }

    #[test]
    fn inventory_binds_bytes_and_resources_without_process_handles() {
        let mut a = LoadedObjects::default();
        let mut b = LoadedObjects::default();
        a.loaded(5, b"object a");
        a.loaded(6, b"object b");
        b.loaded(500, b"object b");
        b.loaded(600, b"object a");
        for entry in ["z", "a", "z"] {
            a.resolved(5, 55, kernel(entry)).unwrap();
        }
        for entry in ["a", "z"] {
            b.resolved(600, 6000, kernel(entry)).unwrap();
        }
        assert_eq!(a.snapshot(), b.snapshot());
        let mut changed = kernel("a");
        changed.static_lds_bytes += 1;
        assert!(a.resolved(5, 55, changed).is_err());
        assert_eq!(a.snapshot(), b.snapshot());
        assert!(a.resolved(999, 55, kernel("a")).is_err());
        b.loaded(500, b"changed object b");
        assert_ne!(a.snapshot(), b.snapshot());
    }

    #[test]
    fn unloaded_and_recycled_handles_do_not_retain_old_bindings() {
        let mut objects = LoadedObjects::default();
        objects.loaded(5, b"old");
        objects.resolved(5, 55, kernel("old_entry")).unwrap();
        objects.unloaded(5);
        assert!(objects.snapshot().is_empty());
        assert!(objects.resolved(5, 55, kernel("old_entry")).is_err());
        objects.loaded(5, b"new");
        let snapshot = objects.snapshot();
        assert!(snapshot[0].resolved_kernels.is_empty());
        assert_eq!(
            snapshot[0].sha256,
            plow_asset::decode_objects::image_sha256(b"new")
        );
    }

    #[test]
    fn selected_kernel_binds_actual_handle_resources_and_rejects_alias_or_recycle() {
        let mut objects = LoadedObjects::default();
        objects.loaded(5, b"object");
        objects.resolved(5, 55, kernel("a")).unwrap();
        let selected = objects.selected(55, 320, 8192, 16).unwrap();
        assert_eq!(
            selected.object_sha256,
            plow_asset::decode_objects::image_sha256(b"object")
        );
        assert_eq!(selected.symbol.entry, "a");
        assert!(objects.selected(99, 320, 8192, 16).is_err());
        assert!(objects.selected(55, 64, 8192, 16).is_err());
        assert!(objects.selected(55, 320, 0, 16).is_err());
        assert!(objects.selected(55, 320, 8192, 0).is_err());
        objects.resolved(5, 55, kernel("alias")).unwrap();
        assert!(objects.selected(55, 320, 8192, 16).is_err());
        objects.loaded(5, b"replacement");
        assert!(objects.selected(55, 320, 8192, 16).is_err());
        objects.resolved(5, 55, kernel("new")).unwrap();
        assert_ne!(objects.selected(55, 320, 8192, 16).unwrap(), selected);
        objects.unloaded(5);
        assert!(objects.selected(55, 320, 8192, 16).is_err());
    }
}
