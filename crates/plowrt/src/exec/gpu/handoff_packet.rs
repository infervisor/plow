use std::sync::Arc;

use plow_asset::packet_pipeline::{HandoffExecutable, HandoffProgramRef, HandoffProofCatalog};

use super::CudaPacketRuntime;
use crate::device::cuda::{CudaBackend, CudaStream, KernelFn};
use crate::exec::gpu::handoff_provider::{HandoffProvider, ProviderTensor};
use crate::exec::packet_runtime::PacketRuntime;
use crate::{Result, RuntimeError};

fn reject(message: &str) -> RuntimeError {
    RuntimeError::Rejected(format!("packet provider: {message}"))
}

pub(crate) struct PacketHandoffProvider {
    runtime: CudaPacketRuntime,
    facts: HandoffProofCatalog,
    active: bool,
}

impl PacketHandoffProvider {
    // Safety: facts come from native typed IR/ABI checks, not parsed plan declarations, and
    // certify the actual accesses, dtypes, maps, initialization and private-state lifetimes.
    // The existing runtime is quiescent and exclusively transferred here.
    pub(crate) unsafe fn from_native(
        runtime: CudaPacketRuntime,
        component: &str,
        object_role: &str,
        facts: HandoffProofCatalog,
    ) -> Result<Self> {
        for view in facts.views.values() {
            let tensor = runtime
                .tensor(&view.tensor)
                .ok_or_else(|| reject("native tensor is absent from the existing runtime"))?;
            if view.component != component || tensor.bytes as u64 != view.bytes {
                return Err(reject("native component or tensor byte extent differs"));
            }
            runtime.checked_device_ptr(tensor)?;
        }
        for (reference, program) in &facts.programs {
            if reference.component != component {
                return Err(reject("foreign native program fact"));
            }
            match &program.entry {
                HandoffExecutable::Packet { section, program } => {
                    if section != "packet" {
                        return Err(reject("unsupported native packet section"));
                    }
                    runtime.check_sequence(&[*program as usize])?;
                }
                HandoffExecutable::Kernel {
                    object_role: role,
                    abi_version,
                    capability,
                    capability_value,
                    symbol,
                    ..
                } => {
                    if role != object_role
                        || *abi_version != 1
                        || runtime.be.module_global_u32(&runtime.module, capability)?
                            != Some(*capability_value)
                    {
                        return Err(reject("loaded module lacks the native auxiliary ABI"));
                    }
                    runtime.be.get_function(&runtime.module, symbol)?;
                }
            }
        }
        if facts.readbacks.keys().any(|r| r.component != component) {
            return Err(reject("foreign native readback fact"));
        }
        Ok(Self {
            runtime,
            facts,
            active: false,
        })
    }

    pub(crate) fn runtime_mut(&mut self) -> Result<&mut CudaPacketRuntime> {
        if self.active {
            return Err(reject("provider execution lease is active"));
        }
        Ok(&mut self.runtime)
    }
}

// The runtime owns all referenced allocations, modules, tables, graphs and pinned inputs.
unsafe impl HandoffProvider for PacketHandoffProvider {
    fn backend(&self) -> &Arc<CudaBackend> {
        self.runtime.backend()
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
        let tensor = self
            .runtime
            .tensor(&view.tensor)
            .ok_or_else(|| reject("native tensor disappeared"))?;
        Ok(ProviderTensor {
            address: self.runtime.checked_device_ptr(tensor)?,
            bytes: tensor.bytes as u64,
        })
    }

    fn resolve_packet(&mut self, reference: &HandoffProgramRef) -> Result<usize> {
        match &self
            .facts
            .programs
            .get(reference)
            .ok_or_else(|| reject("unknown native program"))?
            .entry
        {
            HandoffExecutable::Packet { program, .. } => Ok(*program as usize),
            _ => Err(reject("native entry is not a packet program")),
        }
    }

    fn resolve_kernel(&mut self, reference: &HandoffProgramRef) -> Result<KernelFn> {
        match &self
            .facts
            .programs
            .get(reference)
            .ok_or_else(|| reject("unknown native kernel"))?
            .entry
        {
            HandoffExecutable::Kernel { symbol, .. } => {
                self.runtime.be.get_function(&self.runtime.module, symbol)
            }
            _ => Err(reject("native entry is not an auxiliary kernel")),
        }
    }

    fn ready(&mut self) -> Result<()> {
        if self.active {
            return Err(reject("provider already has an active execution lease"));
        }
        self.runtime.check_inputs()?;
        self.active = true;
        Ok(())
    }

    fn quiescent(&self) -> bool {
        !self.active && !self.runtime.input_failed && !self.runtime.backend().is_poisoned()
    }

    fn retired(&mut self, _succeeded: bool) {
        self.active = false;
    }

    unsafe fn enqueue_packet(&mut self, handle: usize, stream: &CudaStream) -> Result<()> {
        if !self.active {
            return Err(reject("native enqueue requires an execution lease"));
        }
        // SAFETY: the owner/executor retains this runtime and the supplied stream through retirement.
        unsafe { self.runtime.enqueue_chained_programs(stream, &[handle]) }
    }
}
