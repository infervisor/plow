use std::collections::BTreeMap;
#[cfg(test)]
use std::path::Path;
use std::sync::Arc;

use plow_asset::packet_pipeline::{
    HandoffBindPhase, HandoffConsumer, HandoffExecutable, HandoffExecution, HandoffExpr,
    HandoffOrder, HandoffParameter, HandoffPlan, HandoffRowMap, HandoffRows, ProvenHandoffPlan,
};

use super::handoff_provider::{
    prove_providers, HandoffProvider, OwnedHandoffProviders, SharedHandoffArena,
};
#[cfg(test)]
use super::packet_exec::{CudaPacketRuntime, SharedPacketTensors};
use crate::device::cuda::{CudaBackend, CudaEvent, CudaStream, KernelFn, PinnedHost};
#[cfg(test)]
use crate::device::Backend;
use crate::device::DeviceMem;
#[cfg(test)]
use crate::device::Module;
#[cfg(test)]
use crate::exec::packet_runtime::PacketRuntime;
use crate::exec::packet_runtime::{PacketBuffer, PacketHostBufferPool};
use crate::{Result, RuntimeError};

#[path = "handoff_rows.rs"]
mod rows;
pub(crate) use rows::{Admission, RowKey};

fn reject(message: impl Into<String>) -> RuntimeError {
    RuntimeError::Rejected(format!("packet handoff: {}", message.into()))
}

fn retire_failed<R>(resources: R, error: RuntimeError, drain: Result<()>) -> RuntimeError {
    match drain {
        Ok(()) => error,
        Err(drain) => {
            std::mem::forget(resources);
            if (drain.is_fatal() && !error.is_fatal())
                || (error.device_fault().is_none() && drain.device_fault().is_some())
            {
                drain
            } else {
                error
            }
        }
    }
}

struct Kernel {
    function: KernelFn,
    parameters: Vec<HandoffParameter>,
    values: Vec<u64>,
    grid: [HandoffExpr; 3],
    evaluated_grid: [u32; 3],
    block: u32,
    shared_bytes: u32,
}

enum Node {
    #[cfg(test)]
    Packet {
        component: usize,
        program: [usize; 1],
    },
    OwnedPacket {
        provider: usize,
        handle: usize,
    },
    Kernel(Kernel),
}

struct Dependency {
    event: CudaEvent,
    producer: usize,
    consumer: usize,
}

struct Gather {
    map: String,
    domain: String,
    unique: bool,
    capacity: u32,
    uses: Vec<HandoffRows>,
}

struct Resources {
    backend: Arc<CudaBackend>,
    arena: Arc<DeviceMem>,
    stream: CudaStream,
    complete: CudaEvent,
    programs: ProgramResources,
    nodes: Vec<Node>,
    dependencies: Vec<Dependency>,
    input: PinnedHost,
    output: PacketHostBufferPool,
    inflight: bool,
    rows: rows::RowDomains,
    gathers: Vec<Gather>,
}

impl Resources {
    fn ensure_quiescent(&self) -> Result<()> {
        if self.backend.is_poisoned() {
            self.backend.bind()?;
        }
        for provider in &self.programs.providers {
            if provider.backend().is_poisoned() {
                provider.backend().bind()?;
            }
        }
        if self.inflight || self.programs.providers.iter().any(|p| !p.quiescent()) {
            return Err(reject("native owner is not quiescent"));
        }
        Ok(())
    }
}

#[derive(Default)]
struct ProgramResources {
    #[cfg(test)]
    components: Vec<CudaPacketRuntime>,
    #[cfg(test)]
    _modules: Vec<Module>,
    providers: Vec<Box<dyn HandoffProvider>>,
    _arena_owner: Option<Arc<SharedHandoffArena>>,
}

impl ProgramResources {
    fn ready(&mut self) -> Result<()> {
        for index in 0..self.providers.len() {
            if let Err(error) = self.providers[index].ready() {
                for provider in &mut self.providers[..index] {
                    provider.retired(true);
                }
                return Err(error);
            }
        }
        Ok(())
    }

    fn retired(&mut self, succeeded: bool) {
        for provider in &mut self.providers {
            provider.retired(succeeded);
        }
    }
}

pub(crate) struct HandoffBind<'a> {
    pub name: &'a str,
    pub bytes: &'a [u8],
}

pub(crate) struct HandoffSelection<'a> {
    pub map: &'a str,
    pub rows: &'a [&'a RowKey],
}

pub(crate) enum HandoffRun {
    Complete(PacketBuffer),
    DeliveryBusy,
}

pub(crate) struct CudaHandoff {
    plan: HandoffPlan,
    resources: Option<Resources>,
}

impl CudaHandoff {
    pub(crate) fn provider_mut<T: HandoffProvider>(&mut self, component: &str) -> Result<&mut T> {
        let index = self
            .plan
            .components
            .keys()
            .position(|name| name == component)
            .ok_or_else(|| reject("unknown native component"))?;
        let resources = self
            .resources
            .as_mut()
            .ok_or_else(|| reject("executor quarantined"))?;
        resources.ensure_quiescent()?;
        let provider = resources
            .programs
            .providers
            .get_mut(index)
            .ok_or_else(|| reject("native owner is unavailable"))?;
        let owner: &mut dyn std::any::Any = provider.as_mut();
        owner
            .downcast_mut::<T>()
            .ok_or_else(|| reject("native owner type differs from caller"))
    }

    pub(crate) fn readback(&self) -> &plow_asset::packet_pipeline::HandoffReadback {
        &self.plan.readback
    }

    pub(crate) fn bind_owned(
        proven: &ProvenHandoffPlan<'_>,
        arena: Arc<SharedHandoffArena>,
        mut providers: OwnedHandoffProviders,
        completed_slabs: usize,
    ) -> Result<Self> {
        let plan = proven.plan();
        Self::supported(plan, completed_slabs)?;
        prove_providers(plan, &arena, &mut providers)?;
        let backend = arena.backend().clone();
        let ids: BTreeMap<_, _> = providers
            .keys()
            .enumerate()
            .map(|(i, id)| (id.clone(), i))
            .collect();
        let mut nodes = Vec::new();
        for reference in &plan.schedule {
            let provider = providers.get_mut(&reference.component).unwrap();
            let entry = &plan.program(reference).map_err(reject)?.entry;
            match entry {
                HandoffExecutable::Packet { .. } => nodes.push(Node::OwnedPacket {
                    provider: ids[&reference.component],
                    handle: provider.resolve_packet(reference)?,
                }),
                HandoffExecutable::Kernel { .. } => {
                    let function = provider.resolve_kernel(reference)?;
                    nodes.push(Self::kernel_node(&backend, entry, function)?);
                }
            }
        }
        Self::finish(
            plan,
            backend,
            arena.memory().clone(),
            ProgramResources {
                providers: providers.into_values().collect(),
                _arena_owner: Some(arena),
                ..Default::default()
            },
            nodes,
            completed_slabs,
        )
    }

    fn kernel_node(
        backend: &CudaBackend,
        entry: &HandoffExecutable,
        function: KernelFn,
    ) -> Result<Node> {
        let HandoffExecutable::Kernel {
            parameters,
            grid,
            block,
            shared_bytes,
            abi_version,
            ..
        } = entry
        else {
            return Err(reject("expected a typed auxiliary entry"));
        };
        if *abi_version != 1 || block[1..] != [1, 1] {
            return Err(reject("unsupported auxiliary ABI version or block shape"));
        }
        backend.set_max_dynamic_smem(function, *shared_bytes)?;
        Ok(Node::Kernel(Kernel {
            function,
            parameters: parameters.clone(),
            values: vec![0; parameters.len()],
            grid: grid.clone(),
            evaluated_grid: [0; 3],
            block: block[0],
            shared_bytes: *shared_bytes,
        }))
    }

    #[cfg(test)]
    pub(crate) fn load(
        backend: Arc<CudaBackend>,
        root: &Path,
        proven: &ProvenHandoffPlan<'_>,
        completed_slabs: usize,
    ) -> Result<Self> {
        let owner = backend.clone();
        owner.unload_modules_on_err(|| Self::load_inner(backend, root, proven, completed_slabs))
    }

    #[cfg(test)]
    fn load_inner(
        backend: Arc<CudaBackend>,
        root: &Path,
        proven: &ProvenHandoffPlan<'_>,
        completed_slabs: usize,
    ) -> Result<Self> {
        let plan = proven.plan();
        Self::supported(plan, completed_slabs)?;
        let arena = Arc::new(backend.alloc(0, plan.arena.bytes)?);
        if arena.base % u64::from(plan.arena.alignment) != 0 {
            return Err(reject("device allocation does not satisfy arena alignment"));
        }
        let mut components = Vec::new();
        let mut component_ids = BTreeMap::new();
        let mut modules = Vec::new();
        let mut functions = BTreeMap::new();
        for (id, component) in &plan.components {
            let mut views = BTreeMap::new();
            for view in plan.views.values().filter(|v| &v.component == id) {
                if views
                    .insert(
                        view.tensor.clone(),
                        (view.offset, view.bytes, view.packet_initialize),
                    )
                    .is_some()
                {
                    return Err(reject(
                        "multiple views of one packet tensor are unsupported",
                    ));
                }
            }
            let path = root.join(&component.packet);
            let asset = crate::exec::packet_runtime::PacketAsset::load(&path)?;
            if !asset
                .pipelines()
                .iter()
                .any(|p| p.driver == component.driver)
            {
                return Err(reject("component driver is not declared by its packet"));
            }
            let runtime = CudaPacketRuntime::load_with_views(
                backend.clone(),
                &path,
                Some(SharedPacketTensors {
                    backend: backend.clone(),
                    memory: arena.clone(),
                    views,
                }),
            )?;
            for view in plan.views.values().filter(|v| &v.component == id) {
                let tensor = runtime
                    .tensor(&view.tensor)
                    .ok_or_else(|| reject("missing bound tensor"))?;
                if runtime.checked_device_ptr(tensor)? != arena.base + view.offset {
                    return Err(reject(
                        "actual tensor address differs from shared-arena plan",
                    ));
                }
            }
            component_ids.insert(id.clone(), components.len());
            components.push(runtime);
            for (role, file) in &component.objects {
                let file = path.parent().unwrap_or(root).join(file);
                let bytes = std::fs::read(&file).map_err(|source| RuntimeError::Io {
                    path: file.clone(),
                    source,
                })?;
                let module = backend.module_load(&bytes)?;
                for (program_id, program) in &component.programs {
                    if let HandoffExecutable::Kernel {
                        object_role,
                        symbol,
                        capability,
                        capability_value,
                        ..
                    } = &program.entry
                    {
                        if object_role != role {
                            continue;
                        }
                        if backend.module_global_u32(&module, capability)?
                            != Some(*capability_value)
                        {
                            return Err(reject("auxiliary object ABI/capability stamp mismatch"));
                        }
                        functions.insert(
                            (id.clone(), program_id.clone()),
                            backend.get_function(&module, symbol)?,
                        );
                    }
                }
                modules.push(module);
            }
        }
        let mut nodes = Vec::new();
        for reference in &plan.schedule {
            match &plan.program(reference).map_err(reject)?.entry {
                HandoffExecutable::Packet { section, program } => {
                    if section != "packet" {
                        return Err(reject("unsupported packet program section"));
                    }
                    let component = component_ids[&reference.component];
                    let program = [*program as usize];
                    components[component].check_sequence(&program)?;
                    nodes.push(Node::Packet { component, program });
                }
                entry @ HandoffExecutable::Kernel { .. } => {
                    let function =
                        functions[&(reference.component.clone(), reference.program.clone())];
                    nodes.push(Self::kernel_node(&backend, entry, function)?);
                }
            }
        }
        Self::finish(
            plan,
            backend,
            arena,
            ProgramResources {
                components,
                _modules: modules,
                ..Default::default()
            },
            nodes,
            completed_slabs,
        )
    }

    fn supported(plan: &HandoffPlan, completed_slabs: usize) -> Result<()> {
        plan.validate().map_err(reject)?;
        if completed_slabs == 0 {
            return Err(reject("empty completed readback pool"));
        }
        if !matches!(plan.execution, HandoffExecution::Synchronous)
            || plan.arena.banks != 1
            || cfg!(target_endian = "big")
        {
            return Err(reject(
                "only synchronous single-bank execution is supported",
            ));
        }
        for view in plan.views.values() {
            if let HandoffRowMap::Gather { map, .. } = &view.row_map {
                if !plan
                    .bind_writes
                    .iter()
                    .any(|b| b.view == *map && b.phase == HandoffBindPhase::Tick)
                {
                    return Err(reject(
                        "gather maps require a generation-qualified tick bind",
                    ));
                }
            }
        }
        for (index, bind) in plan.bind_writes.iter().enumerate() {
            let view = &plan.views[&bind.view];
            if plan.bind_writes[..index].iter().any(|prior| {
                let prior = &plan.views[&prior.view];
                prior.offset < view.offset + view.bytes && view.offset < prior.offset + prior.bytes
            }) {
                return Err(reject("overlapping host bind targets are unsupported"));
            }
        }
        for bind in plan
            .bind_writes
            .iter()
            .filter(|b| b.phase == HandoffBindPhase::Admission)
        {
            let view = &plan.views[&bind.view];
            if !view.packet_initialize
                || view.lifetime != plow_asset::packet_pipeline::HandoffLifetime::Persistent
            {
                return Err(reject(
                    "admission rows require initialized persistent storage",
                ));
            }
        }
        Ok(())
    }

    fn finish(
        plan: &HandoffPlan,
        backend: Arc<CudaBackend>,
        arena: Arc<DeviceMem>,
        programs: ProgramResources,
        nodes: Vec<Node>,
        completed_slabs: usize,
    ) -> Result<Self> {
        let mut dependencies = Vec::new();
        for edge in &plan.edges {
            if matches!(edge.order, HandoffOrder::Event { .. }) {
                let producer = plan
                    .schedule
                    .iter()
                    .position(|p| p == &edge.producer.program)
                    .unwrap();
                let consumer = match &edge.consumer {
                    HandoffConsumer::Program { endpoint } => plan
                        .schedule
                        .iter()
                        .position(|p| p == &endpoint.program)
                        .unwrap(),
                    HandoffConsumer::Host => nodes.len(),
                };
                dependencies.push(Dependency {
                    event: backend.event_create(false)?,
                    producer,
                    consumer,
                });
            }
        }
        let mut tick_bytes = 0usize;
        let mut admission_bytes = BTreeMap::<&str, usize>::new();
        for bind in &plan.bind_writes {
            let view = plan.view(&bind.view).map_err(reject)?;
            if view.row_stride != u64::from(view.row_width) * view.dtype.bytes()
                || !matches!(bind.order, HandoffOrder::SameStream)
                || (bind.phase == HandoffBindPhase::Tick
                    && !matches!(view.row_map, HandoffRowMap::Contiguous { .. }))
            {
                return Err(reject(
                    "bind requires contiguous rows on the execution stream",
                ));
            }
            let target = if bind.phase == HandoffBindPhase::Admission {
                let domain = match &view.row_map {
                    HandoffRowMap::Contiguous { domain } | HandoffRowMap::Gather { domain, .. } => {
                        domain
                    }
                };
                admission_bytes.entry(domain).or_default()
            } else {
                &mut tick_bytes
            };
            let rows = if bind.phase == HandoffBindPhase::Admission {
                1
            } else {
                plan.rows_capacity(&bind.rows).map_err(reject)?
            };
            let bytes = u64::from(rows) * view.row_stride;
            *target = target
                .checked_add(usize::try_from(bytes).map_err(|_| reject("bind size overflow"))?)
                .ok_or_else(|| reject("staging size overflow"))?;
        }
        let input_bytes = tick_bytes.max(admission_bytes.into_values().max().unwrap_or(0));
        let mut gathers = Vec::<Gather>::new();
        for (name, view) in &plan.views {
            if let HandoffRowMap::Gather {
                map,
                domain,
                unique,
            } = &view.row_map
            {
                let mut uses = Vec::new();
                for reference in &plan.schedule {
                    for access in &plan.program(reference).map_err(reject)?.accesses {
                        if access.view == *name && !uses.contains(&access.rows) {
                            uses.push(access.rows.clone());
                        }
                    }
                }
                if uses.is_empty() {
                    continue;
                }
                if let Some(prior) = gathers.iter_mut().find(|g| g.map == *map) {
                    if prior.domain != *domain {
                        return Err(reject("one map cannot select different physical domains"));
                    }
                    prior.unique |= *unique;
                    prior.capacity = prior.capacity.min(view.rows);
                    for rows in uses {
                        if !prior.uses.contains(&rows) {
                            prior.uses.push(rows);
                        }
                    }
                } else {
                    gathers.push(Gather {
                        map: map.clone(),
                        domain: domain.clone(),
                        unique: *unique,
                        capacity: view.rows,
                        uses,
                    });
                }
            }
        }
        let output_bytes =
            u64::from(plan.extents[&plan.readback.extent]) * plan.readback.row_stride;
        // Pageable cold-load uploads must retire before the nonblocking execution stream reads them.
        backend.synchronize()?;
        let resources = Resources {
            arena,
            stream: backend.stream_create()?,
            complete: backend.event_create(false)?,
            programs,
            nodes,
            dependencies,
            input: backend.host_alloc_pinned(input_bytes.max(1))?,
            output: PacketHostBufferPool::pinned(
                &backend,
                usize::try_from(output_bytes).map_err(|_| reject("readback size overflow"))?,
                completed_slabs,
            )?,
            inflight: false,
            rows: rows::RowDomains::new(plan)?,
            gathers,
            backend,
        };
        Ok(Self {
            plan: plan.clone(),
            resources: Some(resources),
        })
    }

    pub(crate) fn prepare_admission(&mut self, domain: &str, slot: u32) -> Result<Admission> {
        let r = self
            .resources
            .as_mut()
            .ok_or_else(|| reject("executor quarantined"))?;
        if r.inflight {
            return Err(reject("unretired submission prevents admission"));
        }
        r.rows.prepare(domain, slot)
    }

    pub(crate) fn release(&mut self, key: &RowKey) -> Result<()> {
        let r = self
            .resources
            .as_mut()
            .ok_or_else(|| reject("executor quarantined"))?;
        if r.inflight {
            return Err(reject("unretired submission prevents row release"));
        }
        r.rows.release(key)
    }

    pub(crate) fn admit(
        &mut self,
        admission: Admission,
        binds: &[HandoffBind<'_>],
    ) -> Result<RowKey> {
        let r = self
            .resources
            .as_mut()
            .ok_or_else(|| reject("executor quarantined"))?;
        if r.inflight {
            return Err(reject("unretired submission prevents admission"));
        }
        let domain = r.rows.domain(&admission)?.to_owned();
        let expected: Vec<_> = self
            .plan
            .bind_writes
            .iter()
            .filter(|b| {
                b.phase == HandoffBindPhase::Admission
                    && match &self.plan.views[&b.view].row_map {
                        HandoffRowMap::Contiguous { domain: d }
                        | HandoffRowMap::Gather { domain: d, .. } => d == &domain,
                    }
            })
            .collect();
        if binds.len() != expected.len() {
            return Err(reject("incomplete admission bind set"));
        }
        let mut offset = 0;
        for (index, bind) in binds.iter().enumerate() {
            let declared = expected
                .iter()
                .find(|b| b.name == bind.name)
                .ok_or_else(|| reject("undeclared admission bind"))?;
            let view = &self.plan.views[&declared.view];
            if binds[..index].iter().any(|b| b.name == bind.name)
                || bind.bytes.len() as u64 != view.row_stride
                || admission.slot() >= self.plan.rows_capacity(&declared.rows).map_err(reject)?
            {
                return Err(reject("invalid admission row or bind size"));
            }
            r.input.as_mut_slice()[offset..offset + bind.bytes.len()].copy_from_slice(bind.bytes);
            offset += bind.bytes.len();
        }
        if binds.is_empty() {
            r.ensure_quiescent()?;
            return r.rows.commit(admission);
        }
        r.inflight = true;
        if let Err(error) = r.programs.ready() {
            r.inflight = false;
            return Err(error);
        }
        let result = (|| {
            let mut offset = 0;
            for bind in binds {
                let declared = expected.iter().find(|b| b.name == bind.name).unwrap();
                let view = &self.plan.views[&declared.view];
                let target =
                    r.arena.base + view.offset + u64::from(admission.slot()) * view.row_stride;
                // SAFETY: this exclusively borrowed executor retains staging through retirement.
                unsafe {
                    r.backend.memcpy_htod_async(
                        target,
                        &r.input.as_slice()[offset..offset + bind.bytes.len()],
                        &r.stream,
                    )?;
                }
                offset += bind.bytes.len();
            }
            r.backend.event_record(&r.complete, &r.stream)?;
            r.backend.event_synchronize(&r.complete)
        })();
        if let Err(error) = result {
            return Err(self.failed(error));
        }
        r.programs.retired(true);
        r.inflight = false;
        r.rows.commit(admission)
    }

    fn failed(&mut self, error: RuntimeError) -> RuntimeError {
        let r = self.resources.as_mut().unwrap();
        let drained = r.backend.stream_synchronize(&r.stream);
        if drained.is_ok() {
            r.programs.retired(false);
        }
        let r = self.resources.take().unwrap();
        retire_failed(r, error, drained)
    }

    pub(crate) fn run(
        &mut self,
        live: &BTreeMap<String, u32>,
        fields: &BTreeMap<String, u64>,
        binds: &[HandoffBind<'_>],
        keys: &[&RowKey],
    ) -> Result<HandoffRun> {
        self.run_mapped(live, fields, binds, keys, &[])
    }

    pub(crate) fn run_mapped(
        &mut self,
        live: &BTreeMap<String, u32>,
        fields: &BTreeMap<String, u64>,
        binds: &[HandoffBind<'_>],
        keys: &[&RowKey],
        selections: &[HandoffSelection<'_>],
    ) -> Result<HandoffRun> {
        let r = self
            .resources
            .as_mut()
            .ok_or_else(|| reject("executor quarantined"))?;
        if r.inflight {
            return Err(reject("unretired submission prevents executor reuse"));
        }
        if live.len() != self.plan.extents.len()
            || live
                .iter()
                .any(|(name, count)| !self.plan.extents.get(name).is_some_and(|cap| count <= cap))
            || binds.len()
                != self
                    .plan
                    .bind_writes
                    .iter()
                    .filter(|b| b.phase == HandoffBindPhase::Tick)
                    .count()
        {
            return Err(reject("live extents or bind set differ from the plan"));
        }
        let count = |rows: &HandoffRows| match rows {
            HandoffRows::Live { extent } => live[extent],
            HandoffRows::Fixed { rows } => *rows,
        };
        r.rows.live(keys)?;
        if selections.len() != r.gathers.len() {
            return Err(reject("incomplete gather generation selections"));
        }
        for (index, selection) in selections.iter().enumerate() {
            let gather = r
                .gathers
                .iter()
                .find(|g| g.map == selection.map)
                .ok_or_else(|| reject("undeclared gather map"))?;
            if selections[..index].iter().any(|s| s.map == selection.map) {
                return Err(reject("duplicate gather selection"));
            }
            let declared = self
                .plan
                .bind_writes
                .iter()
                .find(|b| b.phase == HandoffBindPhase::Tick && b.view == gather.map)
                .unwrap();
            let bind = binds
                .iter()
                .find(|b| b.name == declared.name)
                .ok_or_else(|| reject("missing gather map bind"))?;
            let referenced = gather.uses.iter().map(&count).max().unwrap_or(0) as usize;
            if selection.rows.len() != referenced {
                return Err(reject("gather selection differs from live row extent"));
            }
            let bytes = referenced
                .checked_mul(4)
                .and_then(|end| bind.bytes.get(..end))
                .ok_or_else(|| reject("map bind omits referenced rows"))?;
            r.rows.selection(
                &gather.domain,
                selection.rows,
                bytes,
                gather.capacity,
                gather.unique,
            )?;
        }
        for reference in &self.plan.schedule {
            for access in &self.plan.program(reference).map_err(reject)?.accesses {
                if !self
                    .plan
                    .bind_writes
                    .iter()
                    .any(|b| b.phase == HandoffBindPhase::Admission && b.view == access.view)
                {
                    continue;
                }
                let HandoffRowMap::Contiguous { domain } = &self.plan.views[&access.view].row_map
                else {
                    continue;
                };
                if (0..count(&access.rows)).any(|slot| !r.rows.contains(domain, slot)) {
                    return Err(reject("program references an unqualified admission row"));
                }
            }
        }
        let mut stage_offset = 0;
        for (i, bind) in binds.iter().enumerate() {
            let declared = self
                .plan
                .bind_writes
                .iter()
                .find(|b| b.name == bind.name && b.phase == HandoffBindPhase::Tick)
                .ok_or_else(|| reject("undeclared bind"))?;
            let view = &self.plan.views[&declared.view];
            let expected = u64::from(count(&declared.rows)) * view.row_stride;
            if bind.bytes.len() as u64 != expected || binds[..i].iter().any(|b| b.name == bind.name)
            {
                return Err(reject("duplicate or incorrectly sized bind"));
            }
            r.input.as_mut_slice()[stage_offset..stage_offset + bind.bytes.len()]
                .copy_from_slice(bind.bytes);
            stage_offset += bind.bytes.len();
        }
        for node in &mut r.nodes {
            if let Node::Kernel(kernel) = node {
                for (value, parameter) in kernel.values.iter_mut().zip(&kernel.parameters) {
                    *value = match parameter {
                        HandoffParameter::Tensor { view, .. } => {
                            r.arena.base + self.plan.views[view].offset
                        }
                        HandoffParameter::Scalar { value, .. } => {
                            value.evaluate(live, fields).map_err(reject)?
                        }
                    };
                }
                for (actual, expr) in kernel.evaluated_grid.iter_mut().zip(&kernel.grid) {
                    *actual = u32::try_from(expr.evaluate(live, fields).map_err(reject)?)
                        .map_err(|_| reject("grid overflow"))?;
                }
            }
        }
        let output_bytes = usize::try_from(
            u64::from(live[&self.plan.readback.extent]) * self.plan.readback.row_stride,
        )
        .map_err(|_| reject("readback size overflow"))?;
        if live.values().all(|count| *count == 0) {
            r.ensure_quiescent()?;
            return Ok(HandoffRun::Complete(Vec::new().into()));
        }
        if !r.output.begin(output_bytes) {
            r.ensure_quiescent()?;
            return Ok(HandoffRun::DeliveryBusy);
        }
        r.inflight = true;
        if let Err(error) = r.programs.ready() {
            // SAFETY: readiness cannot enqueue, so the reserved slab has no pending DMA.
            unsafe { r.output.cancel_unsubmitted() };
            r.inflight = false;
            return Err(error);
        }
        let result = (|| {
            let mut offset = 0;
            for bind in binds {
                let declared = self
                    .plan
                    .bind_writes
                    .iter()
                    .find(|b| b.name == bind.name)
                    .unwrap();
                let target = r.arena.base + self.plan.views[&declared.view].offset;
                if !bind.bytes.is_empty() {
                    // SAFETY: owned pinned staging remains untouched through event retirement.
                    unsafe {
                        r.backend.memcpy_htod_async(
                            target,
                            &r.input.as_slice()[offset..offset + bind.bytes.len()],
                            &r.stream,
                        )?;
                    }
                }
                offset += bind.bytes.len();
            }
            for (index, node) in r.nodes.iter().enumerate() {
                for dependency in r.dependencies.iter().filter(|d| d.consumer == index) {
                    r.backend.stream_wait_event(&r.stream, &dependency.event)?;
                }
                match node {
                    #[cfg(test)]
                    Node::Packet { component, program } => {
                        // SAFETY: all component resources and shared views are owned through retirement.
                        unsafe {
                            r.programs.components[*component]
                                .enqueue_chained_programs(&r.stream, program)?;
                        }
                    }
                    Node::OwnedPacket { provider, handle } => {
                        // SAFETY: the resource set owns the provider and its exclusive execution lease.
                        unsafe {
                            r.programs.providers[*provider].enqueue_packet(*handle, &r.stream)?;
                        }
                    }
                    Node::Kernel(kernel) if kernel.evaluated_grid.iter().all(|n| *n > 0) => {
                        let mut params = [std::ptr::null_mut(); 128];
                        for (pointer, value) in params.iter_mut().zip(&kernel.values) {
                            *pointer = (value as *const u64).cast_mut().cast();
                        }
                        r.backend.launch_kernel_grid(
                            kernel.function,
                            kernel.evaluated_grid,
                            kernel.block,
                            kernel.shared_bytes,
                            &mut params[..kernel.values.len()],
                            Some(&r.stream),
                        )?;
                    }
                    Node::Kernel(_) => {}
                }
                for dependency in r.dependencies.iter().filter(|d| d.producer == index) {
                    r.backend.event_record(&dependency.event, &r.stream)?;
                }
            }
            for dependency in r
                .dependencies
                .iter()
                .filter(|d| d.consumer == r.nodes.len())
            {
                r.backend.stream_wait_event(&r.stream, &dependency.event)?;
            }
            if output_bytes > 0 {
                let source = r.arena.base + self.plan.views[&self.plan.readback.view].offset;
                // SAFETY: output is exclusively owned until the completion event retires.
                unsafe {
                    r.backend
                        .memcpy_dtoh_async(r.output.writable(), source, &r.stream)?;
                }
            }
            r.backend.event_record(&r.complete, &r.stream)?;
            r.backend.event_synchronize(&r.complete)
        })();
        if let Err(error) = result {
            return Err(self.failed(error));
        }
        r.programs.retired(true);
        r.inflight = false;
        // SAFETY: the completion event retired every write into the reserved host buffer.
        Ok(HandoffRun::Complete(unsafe { r.output.complete() }))
    }
}

impl Drop for CudaHandoff {
    fn drop(&mut self) {
        if let Some(resources) = self.resources.take() {
            if resources.inflight
                && resources
                    .backend
                    .stream_synchronize(&resources.stream)
                    .is_err()
            {
                std::mem::forget(resources);
            }
        }
    }
}

#[cfg(test)]
#[path = "handoff_tests.rs"]
mod tests;
