use std::collections::BTreeMap;
use std::sync::Arc;

use plow_asset::packet_pipeline::{
    HandoffArena, HandoffPlan, HandoffProgramRef, HandoffProofCatalog, ProvenHandoffPlan,
};

use crate::device::cuda::{CudaBackend, CudaStream, KernelFn};
use crate::device::{Backend, DeviceMem};
use crate::{Result, RuntimeError};

fn reject(message: &str) -> RuntimeError {
    RuntimeError::Rejected(format!("handoff provider: {message}"))
}

pub(crate) struct SharedHandoffArena {
    backend: Arc<CudaBackend>,
    memory: Arc<DeviceMem>,
    layout: HandoffArena,
}

impl SharedHandoffArena {
    pub(crate) fn allocate(
        backend: Arc<CudaBackend>,
        proven: &ProvenHandoffPlan<'_>,
    ) -> Result<Arc<Self>> {
        let layout = proven.plan().arena.clone();
        let memory = backend.alloc(0, layout.bytes)?;
        if memory.base % u64::from(layout.alignment) != 0 {
            return Err(reject("allocation does not satisfy shared-arena alignment"));
        }
        Ok(Arc::new(Self {
            backend,
            memory: Arc::new(memory),
            layout,
        }))
    }

    pub(crate) fn backend(&self) -> &Arc<CudaBackend> {
        &self.backend
    }
    pub(crate) fn memory(&self) -> &Arc<DeviceMem> {
        &self.memory
    }
    pub(crate) fn layout(&self) -> &HandoffArena {
        &self.layout
    }
}

pub(crate) struct ProviderTensor {
    pub address: u64,
    pub bytes: u64,
}

// Implementations are native causal/packet resource owners, never callbacks named by metadata.
// Safety: Self exclusively retains all referenced allocations, modules, pins and mapping leases.
// Providers are quiescent at binding; ready claims exclusive mutation leases through retired.
// Readiness, resolution and retired enqueue nothing; retired cannot fail or panic.
// Failed readiness acquires no new lease; successful retirement with no enqueue cancels a claim.
// Enqueue uses only the supplied stream, never waits/readbacks, and cannot invalidate its bindings.
// Bank overrides resolve actual bank storage, packet counters/tables and kernel globals before bind.
pub(crate) unsafe trait HandoffProvider: std::any::Any + Send + 'static {
    fn backend(&self) -> &Arc<CudaBackend>;
    fn facts(&self) -> &HandoffProofCatalog;
    fn tensor(&mut self, view: &str) -> Result<ProviderTensor>;
    fn resolve_packet(&mut self, program: &HandoffProgramRef) -> Result<usize>;
    fn resolve_kernel(&mut self, program: &HandoffProgramRef) -> Result<KernelFn>;
    fn tensor_bank(&mut self, view: &str, bank: u32) -> Result<ProviderTensor> {
        if bank != 0 {
            return Err(reject("native owner has no banked tensor bindings"));
        }
        self.tensor(view)
    }
    fn resolve_packet_bank(&mut self, program: &HandoffProgramRef, bank: u32) -> Result<usize> {
        if bank != 0 {
            return Err(reject("native owner has no banked packet handles"));
        }
        self.resolve_packet(program)
    }
    fn resolve_kernel_bank(&mut self, program: &HandoffProgramRef, bank: u32) -> Result<KernelFn> {
        if bank != 0 {
            return Err(reject("native owner has no banked kernel handles"));
        }
        self.resolve_kernel(program)
    }
    fn ready(&mut self) -> Result<()>;
    fn quiescent(&self) -> bool;
    fn retired(&mut self, succeeded: bool);
    unsafe fn enqueue_packet(&mut self, handle: usize, stream: &CudaStream) -> Result<()>;
}

pub(crate) type OwnedHandoffProviders = BTreeMap<String, Box<dyn HandoffProvider>>;

pub(crate) fn prove_providers(
    plan: &HandoffPlan,
    arena: &SharedHandoffArena,
    providers: &mut OwnedHandoffProviders,
) -> Result<()> {
    if providers.len() != plan.components.len()
        || providers.keys().ne(plan.components.keys())
        || arena.layout != plan.arena
    {
        return Err(reject("provider set or allocated layout differs from plan"));
    }
    let mut actual = HandoffProofCatalog {
        arena: Some(arena.layout.clone()),
        ..Default::default()
    };
    for (component, provider) in providers.iter_mut() {
        if !arena.backend.shares_context(provider.backend()) {
            return Err(reject("providers do not share the arena context"));
        }
        if provider.backend().is_poisoned() {
            provider.backend().bind()?;
        }
        if !provider.quiescent() {
            return Err(reject("native proof requires a quiescent provider"));
        }
        let facts = provider.facts();
        if facts.arena.as_ref() != Some(&arena.layout) {
            return Err(reject(
                "native provider facts describe another arena layout",
            ));
        }
        for (name, &capacity) in &facts.extents {
            if actual
                .extents
                .insert(name.clone(), capacity)
                .is_some_and(|old| old != capacity)
            {
                return Err(reject("native live capacities disagree"));
            }
        }
        for (name, view) in &facts.views {
            if &view.component != component
                || actual.views.insert(name.clone(), view.clone()).is_some()
            {
                return Err(reject("duplicate or foreign native tensor fact"));
            }
        }
        for (reference, program) in &facts.programs {
            if &reference.component != component
                || actual
                    .programs
                    .insert(reference.clone(), program.clone())
                    .is_some()
            {
                return Err(reject("duplicate or foreign native program fact"));
            }
        }
        for (reference, readback) in &facts.readbacks {
            if &reference.component != component
                || actual
                    .readbacks
                    .insert(reference.clone(), readback.clone())
                    .is_some()
            {
                return Err(reject("duplicate or foreign native readback fact"));
            }
        }
    }
    plan.prove(&actual).map_err(RuntimeError::Rejected)?;
    for (name, expected) in &plan.views {
        let actual = providers
            .get_mut(&expected.component)
            .unwrap()
            .tensor(name)?;
        if actual.address != arena.memory.base + expected.offset || actual.bytes != expected.bytes {
            return Err(reject(
                "actual bound tensor differs from the proven shared view",
            ));
        }
    }
    Ok(())
}
