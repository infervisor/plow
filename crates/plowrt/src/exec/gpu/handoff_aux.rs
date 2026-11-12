use std::collections::BTreeMap;
use std::sync::Arc;

use plow_asset::packet_pipeline::{HandoffExecutable, HandoffProgramRef, HandoffProofCatalog};

use super::handoff_provider::{HandoffProvider, ProviderTensor, SharedHandoffArena};
use crate::device::cuda::{CudaBackend, CudaStream, KernelFn};
use crate::device::{Backend, Module};
use crate::{Result, RuntimeError};

fn reject(message: &str) -> RuntimeError {
    RuntimeError::Rejected(format!("auxiliary handoff: {message}"))
}

pub(crate) struct AuxHandoffProvider {
    arena: Option<Arc<SharedHandoffArena>>,
    module: Option<Module>,
    facts: HandoffProofCatalog,
    kernels: BTreeMap<HandoffProgramRef, KernelFn>,
    active: bool,
    failed: bool,
}

impl AuxHandoffProvider {
    // Safety: facts certify the actual typed kernel ABI, accesses, initialization and
    // state lifetimes independently of parsed plan declarations. Arena contents are quiescent.
    pub(crate) unsafe fn from_native(
        arena: Arc<SharedHandoffArena>,
        image: &[u8],
        component: &str,
        object_role: &str,
        facts: HandoffProofCatalog,
    ) -> Result<Self> {
        if facts.arena.as_ref() != Some(arena.layout()) || facts.programs.is_empty() {
            return Err(reject("native facts lack the allocated arena or kernels"));
        }
        for view in facts.views.values() {
            if view.component != component
                || view
                    .offset
                    .checked_add(view.bytes)
                    .is_none_or(|end| end > arena.memory().len)
            {
                return Err(reject(
                    "native view has a foreign owner or exceeds the arena",
                ));
            }
        }
        if facts
            .readbacks
            .keys()
            .any(|reference| reference.component != component)
        {
            return Err(reject("foreign native readback"));
        }
        let mut provider = Self {
            arena: Some(arena),
            module: None,
            facts,
            kernels: BTreeMap::new(),
            active: false,
            failed: false,
        };
        provider.module = Some(provider.backend().module_load(image)?);
        for (reference, proof) in &provider.facts.programs {
            let HandoffExecutable::Kernel {
                object_role: role,
                symbol,
                abi_version,
                capability,
                capability_value,
                shared_bytes,
                ..
            } = &proof.entry
            else {
                return Err(reject("auxiliary owner cannot execute packet programs"));
            };
            if reference.component != component
                || role != object_role
                || *abi_version != 1
                || provider
                    .backend()
                    .module_global_u32(provider.module.as_ref().unwrap(), capability)?
                    != Some(*capability_value)
            {
                return Err(reject(
                    "native program differs from the loaded auxiliary ABI",
                ));
            }
            let function = provider
                .backend()
                .get_function(provider.module.as_ref().unwrap(), symbol)?;
            if *shared_bytes != 0 {
                provider
                    .backend()
                    .set_max_dynamic_smem(function, *shared_bytes)?;
            }
            provider.kernels.insert(reference.clone(), function);
        }
        Ok(provider)
    }
}

// The owner retains the module and shared allocation throughout every execution lease.
unsafe impl HandoffProvider for AuxHandoffProvider {
    fn backend(&self) -> &Arc<CudaBackend> {
        self.arena.as_ref().unwrap().backend()
    }
    fn facts(&self) -> &HandoffProofCatalog {
        &self.facts
    }
    fn tensor(&mut self, name: &str) -> Result<ProviderTensor> {
        let view = self
            .facts
            .views
            .get(name)
            .ok_or_else(|| reject("unknown native view"))?;
        Ok(ProviderTensor {
            address: self.arena.as_ref().unwrap().memory().base + view.offset,
            bytes: view.bytes,
        })
    }
    fn resolve_packet(&mut self, _: &HandoffProgramRef) -> Result<usize> {
        Err(reject("auxiliary owner has no packet program"))
    }
    fn resolve_kernel(&mut self, reference: &HandoffProgramRef) -> Result<KernelFn> {
        self.kernels
            .get(reference)
            .copied()
            .ok_or_else(|| reject("unknown native kernel"))
    }
    fn ready(&mut self) -> Result<()> {
        if self.backend().is_poisoned() {
            self.backend().bind()?;
        }
        if self.active || self.failed {
            return Err(reject("owner is active or quarantined"));
        }
        self.active = true;
        Ok(())
    }
    fn quiescent(&self) -> bool {
        !self.active && !self.failed && !self.backend().is_poisoned()
    }
    fn retired(&mut self, succeeded: bool) {
        self.active = false;
        self.failed |= !succeeded;
    }
    unsafe fn enqueue_packet(&mut self, _: usize, _: &CudaStream) -> Result<()> {
        Err(reject("auxiliary owner cannot enqueue packet programs"))
    }
}

impl Drop for AuxHandoffProvider {
    fn drop(&mut self) {
        if let Some(arena) = self.arena.take() {
            if let Err(error) = arena.backend().synchronize() {
                tracing::warn!(%error, "auxiliary provider drain failed; retaining module and arena");
                std::mem::forget(arena);
                if let Some(module) = self.module.take() {
                    std::mem::forget(module);
                }
            } else if let Some(module) = self.module.take() {
                if let Err(error) = arena.backend().module_unload(&module) {
                    tracing::warn!(%error, "auxiliary provider module unload failed");
                }
            }
        }
    }
}
