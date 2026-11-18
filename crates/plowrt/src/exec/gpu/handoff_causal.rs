use std::collections::BTreeMap;
use std::sync::Arc;

use plow_asset::packet_pipeline::{
    HandoffExecutable, HandoffProgramRef, HandoffProofCatalog, ProvenHandoffPlan,
};

use super::handoff_provider::SharedHandoffArena;
use crate::asset::devblob::DevBlob;
use crate::device::cuda::{CudaBackend, CudaStream, KernelFn};
use crate::device::{Backend, DeviceMem};
use crate::{Result, RuntimeError};

fn reject(message: &str) -> RuntimeError {
    RuntimeError::Rejected(format!("causal handoff: {message}"))
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct Binding {
    offset: u64,
    bytes: u64,
    initialize: bool,
}

pub(crate) struct SharedCausalTensors {
    arena: Arc<SharedHandoffArena>,
    bindings: BTreeMap<String, Binding>,
}

impl SharedCausalTensors {
    // The native proof must cover this component's actual packet programs and initialization.
    pub(crate) unsafe fn new(
        arena: Arc<SharedHandoffArena>,
        proven: &ProvenHandoffPlan<'_>,
        component: &str,
    ) -> Result<Self> {
        if arena.layout() != &proven.plan().arena
            || !proven.plan().components.contains_key(component)
        {
            return Err(reject("component or allocated arena differs from proof"));
        }
        let mut bindings = BTreeMap::new();
        for view in proven
            .plan()
            .views
            .values()
            .filter(|v| v.component == component)
        {
            let binding = Binding {
                offset: view.offset,
                bytes: view.bytes,
                initialize: view.packet_initialize,
            };
            if bindings
                .insert(view.tensor.clone(), binding)
                .is_some_and(|old| old != binding)
            {
                return Err(reject("one causal tensor has incompatible shared bindings"));
            }
        }
        Ok(Self { arena, bindings })
    }

    pub(crate) fn validate(
        &self,
        backend: &CudaBackend,
        blob: &DevBlob,
        eligible: impl Fn(usize, &str) -> bool,
    ) -> Result<()> {
        if !self.arena.backend().shares_context(backend) {
            return Err(reject("shared bindings use another CUDA context"));
        }
        for (name, binding) in &self.bindings {
            let (id, tensor) = blob
                .tensors
                .iter()
                .enumerate()
                .find(|(_, t)| &t.name == name)
                .ok_or_else(|| reject("shared tensor is absent from packet"))?;
            if tensor.bytes != binding.bytes
                || !eligible(id, name)
                || binding
                    .offset
                    .checked_add(binding.bytes)
                    .is_none_or(|end| end > self.arena.memory().len)
            {
                return Err(reject(
                    "shared tensor shape, ownership or range differs from packet",
                ));
            }
        }
        Ok(())
    }

    pub(crate) fn contains(&self, name: &str) -> bool {
        self.bindings.contains_key(name)
    }

    pub(crate) fn memory(&self, name: &str) -> Result<Option<DeviceMem>> {
        self.bindings
            .get(name)
            .map(|b| self.arena.memory().subview(b.offset, b.bytes))
            .transpose()
    }

    pub(crate) fn initialize(&self, name: &str) -> bool {
        self.bindings.get(name).is_none_or(|b| b.initialize)
    }
}

impl Drop for SharedCausalTensors {
    fn drop(&mut self) {
        if let Err(error) = self.arena.backend().synchronize() {
            tracing::warn!(%error, "causal handoff drain failed; retaining shared arena");
            std::mem::forget(Arc::clone(&self.arena));
        }
    }
}

#[derive(Clone, Copy)]
pub(super) struct PreparedCausalProgram {
    pub argument: super::DevProgram,
    pub function: KernelFn,
    pub grid: u32,
    pub block: u32,
    pub shared_bytes: u32,
    pub counter_bytes: usize,
    pub rows: usize,
}

pub(crate) struct CausalHandoffProvider {
    engine: Option<super::GpuEngine>,
    facts: HandoffProofCatalog,
    programs: Vec<PreparedCausalProgram>,
    handles: BTreeMap<HandoffProgramRef, usize>,
    kernels: BTreeMap<HandoffProgramRef, KernelFn>,
    active: bool,
    failed: bool,
    mappings_ready: bool,
}

impl CausalHandoffProvider {
    // Safety: facts certify the actual typed IR/ABI accesses and lifetimes, independently of
    // parsed declarations. The exclusively transferred engine has no outstanding external work.
    pub(crate) unsafe fn from_native(
        engine: super::GpuEngine,
        packet: (&DevBlob, &[u8]),
        component: &str,
        object_role: &str,
        facts: HandoffProofCatalog,
    ) -> Result<Self> {
        use super::handoff_provider::HandoffProvider;
        let shared = engine
            .handoff_shared
            .as_ref()
            .ok_or_else(|| reject("engine has no pre-initialization shared bindings"))?;
        if facts.arena.as_ref() != Some(shared.arena.layout())
            || engine.pipe_busy()
            || engine.pipe.is_some()
            || engine.rank_bind.is_some()
            || engine.span_launch
            || engine.spec.is_some()
            || engine.multistep.is_some()
            || engine.decode_contexts.is_some()
            || (engine.batch == 1 && !engine.kvrow.is_empty())
            || engine.nv_nsplit.is_some()
        {
            return Err(reject("unsupported ownership, mapping or execution state"));
        }
        if let Err(error) = engine.be.stream_synchronize(&engine.stream) {
            std::mem::forget(engine);
            return Err(error);
        }
        let mut programs = Vec::new();
        let mut handles = BTreeMap::new();
        let mut kernels = BTreeMap::new();
        for (reference, proof) in &facts.programs {
            if reference.component != component {
                return Err(reject("foreign native program"));
            }
            match &proof.entry {
                HandoffExecutable::Packet { section, program } => {
                    let prepared = match section.as_str() {
                        "packet" => prepare_decode(&engine, packet.0, *program as usize)?,
                        "fish_depth" => engine.prepare_handoff_depth(packet, *program as usize)?,
                        _ => return Err(reject("unsupported causal program section")),
                    };
                    handles.insert(reference.clone(), programs.len());
                    programs.push(prepared);
                }
                HandoffExecutable::Kernel {
                    object_role: role,
                    symbol,
                    abi_version,
                    capability,
                    capability_value,
                    ..
                } => {
                    if role != object_role || *abi_version != 1 {
                        return Err(reject("unsupported native auxiliary object ABI"));
                    }
                    kernels.insert(
                        reference.clone(),
                        engine.resolve_handoff_depth_kernel(
                            symbol,
                            capability,
                            *capability_value,
                        )?,
                    );
                }
            }
        }
        if facts.readbacks.keys().any(|r| r.component != component)
            || facts.views.values().any(|v| v.component != component)
        {
            return Err(reject("foreign native tensor or readback"));
        }
        let mappings_ready = engine.vmm.is_none();
        let mut provider = Self {
            engine: Some(engine),
            facts,
            programs,
            handles,
            kernels,
            active: false,
            failed: false,
            mappings_ready,
        };
        for name in provider.facts.views.keys().cloned().collect::<Vec<_>>() {
            provider.tensor(&name)?;
        }
        Ok(provider)
    }

    pub(crate) fn prepare_mappings(&mut self, slots: &[usize]) -> Result<()> {
        if self.active || self.failed {
            return Err(reject(
                "mapping preparation requires a healthy quiescent owner",
            ));
        }
        self.mappings_ready = false;
        let engine = self.engine.as_mut().expect("owned causal engine");
        let rows = self.programs.iter().map(|p| p.rows).max().unwrap_or(0);
        let prefix = mapping_prefix(slots, &engine.pos, rows, engine.max_ctx)?;
        if let Some(vmm) = engine.vmm.as_mut() {
            if let Some(rings) = vmm.rings.as_mut() {
                rings.ensure_prefix(prefix)?;
            }
            for &slot in slots {
                vmm.ensure_rows(slot, engine.pos[slot] + 1)?;
            }
        }
        self.mappings_ready = true;
        Ok(())
    }

    pub(crate) fn engine_mut(&mut self) -> Result<CausalEngineGuard<'_>> {
        if self.active || self.failed {
            return Err(reject("provider is active or quarantined"));
        }
        self.mappings_ready = false;
        Ok(CausalEngineGuard { provider: self })
    }
}

fn mapping_prefix(
    slots: &[usize],
    positions: &[u32],
    rows: usize,
    max_context: usize,
) -> Result<usize> {
    if rows == 0 || rows > positions.len() || slots.is_empty() || slots.len() > rows {
        return Err(reject("live mapping slots exceed the prepared program"));
    }
    let mut prefix = rows;
    for &slot in slots {
        let next = positions.get(slot).and_then(|pos| pos.checked_add(1));
        if next.is_none_or(|pos| pos as usize > max_context) {
            return Err(reject(
                "native slot or position exceeds the compiled context",
            ));
        }
        prefix = prefix.max(slot + 1);
    }
    Ok(prefix)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sparse_mapping_covers_physical_slots_beyond_launch_rows() {
        let mut positions = vec![0u32; 353];
        positions[352] = 2047;
        assert_eq!(
            mapping_prefix(&[352, 200, 16], &positions, 32, 2048).unwrap(),
            353
        );
        assert_eq!(
            mapping_prefix(&[31, 1, 0], &positions, 32, 2048).unwrap(),
            32
        );
        assert!(mapping_prefix(&[353], &positions, 32, 2048).is_err());
        assert!(mapping_prefix(&[], &positions, 32, 2048).is_err());
        assert!(mapping_prefix(&[1, 2], &positions, 1, 2048).is_err());
        assert!(mapping_prefix(&[1], &positions, 354, 2048).is_err());
        positions[352] = 2048;
        assert!(mapping_prefix(&[352], &positions, 32, 2048).is_err());
        positions[352] = u32::MAX;
        assert!(mapping_prefix(&[352], &positions, 32, 2048).is_err());
    }
}

pub(crate) struct CausalEngineGuard<'a> {
    provider: &'a mut CausalHandoffProvider,
}

impl std::ops::Deref for CausalEngineGuard<'_> {
    type Target = super::GpuEngine;
    fn deref(&self) -> &Self::Target {
        self.provider.engine.as_ref().expect("owned causal engine")
    }
}

impl std::ops::DerefMut for CausalEngineGuard<'_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.provider.engine.as_mut().expect("owned causal engine")
    }
}

impl Drop for CausalEngineGuard<'_> {
    fn drop(&mut self) {
        let engine = self.provider.engine.as_ref().expect("owned causal engine");
        if let Err(error) = engine.be.stream_synchronize(&engine.stream) {
            self.provider.failed = true;
            tracing::warn!(%error, "quiescent causal owner access failed to drain");
        }
    }
}

fn prepare_decode(
    e: &super::GpuEngine,
    blob: &DevBlob,
    index: usize,
) -> Result<PreparedCausalProgram> {
    let start = blob.progs.len() - blob.decode_progs().len();
    let program = blob
        .progs
        .get(index)
        .filter(|_| index >= start)
        .ok_or_else(|| reject("handoff requires a decode packet program"))?;
    program.check_coarse_single_segment()?;
    let (argument, instructions, function, grid, block, shared_bytes, counter_bytes) =
        if index + 1 == blob.progs.len() {
            if !e.cublaslt_decode.is_empty() || !e.decode_packet_roles.is_empty() {
                return Err(reject("library-routed widest decode is unsupported"));
            }
            (
                e.kernarg,
                &e.h_inst,
                e.f,
                e.grid,
                super::BLOCK,
                e.smem,
                e.ctr_bytes.max(4) + e.cursor_bytes,
            )
        } else {
            let rung = e
                .decode_rungs
                .get(index - start)
                .filter(|r| r.rows == program.t as usize && r.library.is_none())
                .ok_or_else(|| reject("plain decode rung differs from packet"))?;
            let object = rung.object.as_deref();
            (
                rung.kernarg,
                &rung.host_insts,
                object.map_or(e.f, |o| o.function),
                object.map_or(e.grid, |o| o.grid),
                object.map_or(super::BLOCK, |o| o.block),
                object.map_or(
                    if rung.group_arena {
                        e.smem
                    } else {
                        e.smem_narrow
                    },
                    |o| o.smem,
                ),
                rung.counter_bytes,
            )
        };
    if super::pod_bytes(instructions) != super::pod_bytes(&program.insts) {
        return Err(reject(
            "prepared decode instructions differ from native packet",
        ));
    }
    Ok(PreparedCausalProgram {
        argument,
        function,
        grid,
        block,
        shared_bytes,
        counter_bytes,
        rows: program.t as usize,
    })
}

// The sole native engine retains every module/table/allocation. The executor owns its lease.
unsafe impl super::handoff_provider::HandoffProvider for CausalHandoffProvider {
    fn backend(&self) -> &Arc<CudaBackend> {
        &self.engine.as_ref().expect("owned causal engine").be
    }
    fn facts(&self) -> &HandoffProofCatalog {
        &self.facts
    }
    fn tensor(&mut self, name: &str) -> Result<super::handoff_provider::ProviderTensor> {
        let view = self
            .facts
            .views
            .get(name)
            .ok_or_else(|| reject("unknown native view"))?;
        let engine = self.engine.as_ref().expect("owned causal engine");
        let memory = engine
            .handle_of(&view.tensor)
            .and_then(|h| engine.devp.get(h))
            .ok_or_else(|| reject("native view tensor is absent"))?;
        let shared = engine
            .handoff_shared
            .as_ref()
            .expect("shared causal engine");
        if memory.len != view.bytes || memory.base != shared.arena.memory().base + view.offset {
            return Err(reject("native view is not the actual shared tensor"));
        }
        Ok(super::handoff_provider::ProviderTensor {
            address: memory.base,
            bytes: memory.len,
        })
    }
    fn resolve_packet(&mut self, reference: &HandoffProgramRef) -> Result<usize> {
        self.handles
            .get(reference)
            .copied()
            .ok_or_else(|| reject("unknown native packet program"))
    }
    fn resolve_kernel(&mut self, reference: &HandoffProgramRef) -> Result<KernelFn> {
        self.kernels
            .get(reference)
            .copied()
            .ok_or_else(|| reject("unknown native auxiliary kernel"))
    }
    fn ready(&mut self) -> Result<()> {
        if self.backend().is_poisoned() {
            self.backend().bind()?;
        }
        if self.active || self.failed || !self.mappings_ready {
            return Err(reject("provider is active or quarantined"));
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
        self.mappings_ready = false;
    }
    unsafe fn enqueue_packet(&mut self, handle: usize, stream: &CudaStream) -> Result<()> {
        if !self.active || self.failed {
            return Err(reject("native enqueue requires a healthy lease"));
        }
        let program = self
            .programs
            .get(handle)
            .ok_or_else(|| reject("invalid native program handle"))?;
        let be = self.backend();
        be.memset_d8_async(program.argument.counters, 0, program.counter_bytes, stream)?;
        let mut argument = program.argument;
        let mut arguments = [&mut argument as *mut super::DevProgram as *mut std::ffi::c_void];
        be.launch_cooperative(
            program.function,
            program.grid,
            program.block,
            program.shared_bytes,
            &mut arguments,
            Some(stream),
        )
    }
}

impl Drop for CausalHandoffProvider {
    fn drop(&mut self) {
        if let Some(engine) = self.engine.take() {
            if let Err(error) = engine.be.synchronize() {
                tracing::warn!(%error, "causal provider drain failed; retaining entire owner");
                std::mem::forget(engine);
            }
        }
    }
}
