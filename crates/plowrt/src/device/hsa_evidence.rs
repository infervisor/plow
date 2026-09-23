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

#[derive(Default)]
pub(super) struct LoadedObjects(BTreeMap<u64, LoadedObjectEvidence>);

impl LoadedObjects {
    pub fn loaded(&mut self, handle: u64, image: &[u8]) {
        self.0.insert(
            handle,
            LoadedObjectEvidence {
                sha256: plow_asset::decode_objects::image_sha256(image),
                bytes: image.len(),
                resolved_kernels: Vec::new(),
            },
        );
    }

    pub fn resolved(&mut self, handle: u64, kernel: ResolvedKernelEvidence) -> Result<(), String> {
        let object = self.0.get_mut(&handle).ok_or("untracked HSA executable")?;
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
        Ok(())
    }

    pub fn unloaded(&mut self, handle: u64) {
        self.0.remove(&handle);
    }

    pub fn snapshot(&self) -> Vec<LoadedObjectEvidence> {
        let mut objects: Vec<_> = self.0.values().cloned().collect();
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
            a.resolved(5, kernel(entry)).unwrap();
        }
        for entry in ["a", "z"] {
            b.resolved(600, kernel(entry)).unwrap();
        }
        assert_eq!(a.snapshot(), b.snapshot());
        let mut changed = kernel("a");
        changed.static_lds_bytes += 1;
        assert!(a.resolved(5, changed).is_err());
        assert_eq!(a.snapshot(), b.snapshot());
        assert!(a.resolved(999, kernel("a")).is_err());
        b.loaded(500, b"changed object b");
        assert_ne!(a.snapshot(), b.snapshot());
    }

    #[test]
    fn unloaded_and_recycled_handles_do_not_retain_old_bindings() {
        let mut objects = LoadedObjects::default();
        objects.loaded(5, b"old");
        objects.resolved(5, kernel("old_entry")).unwrap();
        objects.unloaded(5);
        assert!(objects.snapshot().is_empty());
        assert!(objects.resolved(5, kernel("old_entry")).is_err());
        objects.loaded(5, b"new");
        let snapshot = objects.snapshot();
        assert!(snapshot[0].resolved_kernels.is_empty());
        assert_eq!(
            snapshot[0].sha256,
            plow_asset::decode_objects::image_sha256(b"new")
        );
    }
}
