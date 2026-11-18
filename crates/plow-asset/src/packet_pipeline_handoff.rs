use super::{identifier, PacketPipeline, PacketPipelines, PipelineDType, VERSION};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const HANDOFF_VERSION: u32 = 2;
const MAX_ITEMS: usize = 4096;
type Result<T> = std::result::Result<T, String>;

fn require(ok: bool, message: &str) -> Result<()> {
    if ok {
        Ok(())
    } else {
        Err(format!("pipeline handoff: {message}"))
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PacketPipelinesV2 {
    pub version: u32,
    pub pipelines: Vec<PacketPipeline>,
    pub handoff_plans: BTreeMap<String, HandoffPlan>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PipelineMetadata {
    V1(PacketPipelines),
    V2(PacketPipelinesV2),
}

impl From<PacketPipelines> for PipelineMetadata {
    fn from(value: PacketPipelines) -> Self {
        Self::V1(value)
    }
}

impl PipelineMetadata {
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        #[derive(Deserialize)]
        struct Header {
            version: u32,
        }
        match serde_json::from_slice::<Header>(bytes)
            .map_err(|e| e.to_string())?
            .version
        {
            VERSION => serde_json::from_slice(bytes)
                .map(Self::V1)
                .map_err(|e| e.to_string()),
            HANDOFF_VERSION => serde_json::from_slice(bytes)
                .map(Self::V2)
                .map_err(|e| e.to_string()),
            _ => Err("unsupported packet pipeline version".into()),
        }
    }

    pub fn pipelines(&self) -> &[PacketPipeline] {
        match self {
            Self::V1(p) => &p.pipelines,
            Self::V2(p) => &p.pipelines,
        }
    }

    pub fn handoff_plans(&self) -> Option<&BTreeMap<String, HandoffPlan>> {
        match self {
            Self::V1(_) => None,
            Self::V2(p) => Some(&p.handoff_plans),
        }
    }

    pub fn validate(&self, programs: usize, tensors: impl Fn(&str) -> Option<u64>) -> Result<()> {
        match self {
            Self::V1(p) => p.validate(programs, tensors),
            Self::V2(p) => {
                require(
                    p.version == HANDOFF_VERSION
                        && !p.handoff_plans.is_empty()
                        && p.handoff_plans.len() <= MAX_ITEMS,
                    "invalid version or plan count",
                )?;
                PacketPipelines {
                    version: VERSION,
                    pipelines: p.pipelines.clone(),
                }
                .validate(programs, tensors)?;
                for (name, plan) in &p.handoff_plans {
                    require(identifier(name, 128), "invalid plan name")?;
                    plan.validate()?;
                }
                Ok(())
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HandoffProgramRef {
    pub component: String,
    pub program: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HandoffComponent {
    pub packet: String,
    pub driver: String,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub objects: BTreeMap<String, String>,
    pub programs: BTreeMap<String, HandoffProgram>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HandoffProgram {
    pub entry: HandoffExecutable,
    pub accesses: Vec<HandoffAccess>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum HandoffExecutable {
    Packet {
        section: String,
        program: u32,
    },
    Kernel {
        object_role: String,
        symbol: String,
        abi_version: u32,
        capability: String,
        capability_value: u32,
        parameters: Vec<HandoffParameter>,
        grid: [HandoffExpr; 3],
        block: [u32; 3],
        shared_bytes: u32,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HandoffAccessMode {
    Read,
    Write,
    ReadWrite,
}

impl HandoffAccessMode {
    pub fn reads(self) -> bool {
        self != Self::Write
    }
    pub fn writes(self) -> bool {
        self != Self::Read
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HandoffAccess {
    pub view: String,
    pub mode: HandoffAccessMode,
    pub rows: HandoffRows,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mask: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum HandoffRows {
    Live { extent: String },
    Fixed { rows: u32 },
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum HandoffParameter {
    Tensor {
        view: String,
        dtype: PipelineDType,
        access: HandoffAccessMode,
        alignment: u32,
    },
    Scalar {
        dtype: HandoffScalarType,
        value: HandoffExpr,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HandoffScalarType {
    U32,
    I32,
    U64,
    F32,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum HandoffExpr {
    Constant {
        value: u64,
    },
    LiveRows {
        extent: String,
    },
    CeilDiv {
        value: Box<HandoffExpr>,
        divisor: u32,
    },
    Field {
        name: String,
        min: u64,
        max: u64,
    },
}

impl HandoffExpr {
    fn bound(&self, extents: &BTreeMap<String, u32>, depth: usize) -> Result<u64> {
        require(
            depth <= 4,
            "launch expression nesting exceeds supported depth",
        )?;
        match self {
            Self::Constant { value } => Ok(*value),
            Self::LiveRows { extent } => extents
                .get(extent)
                .map(|&v| u64::from(v))
                .ok_or_else(|| "pipeline handoff: unknown live extent".into()),
            Self::CeilDiv { value, divisor } => {
                require(*divisor != 0, "zero launch divisor")?;
                Ok(value
                    .bound(extents, depth + 1)?
                    .div_ceil(u64::from(*divisor)))
            }
            Self::Field { name, min, max } => {
                require(
                    identifier(name, 128) && min <= max,
                    "invalid scalar field bounds",
                )?;
                Ok(*max)
            }
        }
    }

    pub fn evaluate(
        &self,
        extents: &BTreeMap<String, u32>,
        fields: &BTreeMap<String, u64>,
    ) -> Result<u64> {
        self.bound(extents, 0)?;
        self.evaluate_inner(extents, fields)
    }

    fn evaluate_inner(
        &self,
        extents: &BTreeMap<String, u32>,
        fields: &BTreeMap<String, u64>,
    ) -> Result<u64> {
        match self {
            Self::Constant { value } => Ok(*value),
            Self::LiveRows { extent } => extents
                .get(extent)
                .map(|&v| u64::from(v))
                .ok_or_else(|| "pipeline handoff: missing live extent".into()),
            Self::CeilDiv { value, divisor } => {
                require(*divisor > 0, "zero launch divisor")?;
                Ok(value
                    .evaluate_inner(extents, fields)?
                    .div_ceil(u64::from(*divisor)))
            }
            Self::Field { name, min, max } => fields
                .get(name)
                .copied()
                .filter(|v| *min <= *v && *v <= *max)
                .ok_or_else(|| "pipeline handoff: missing or out-of-range scalar field".into()),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HandoffArena {
    pub bytes: u64,
    pub alignment: u32,
    pub banks: u32,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum HandoffLifetime {
    Persistent,
    Transient { bank_stride: u64 },
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum HandoffRowMap {
    Contiguous {
        domain: String,
    },
    Gather {
        map: String,
        domain: String,
        unique: bool,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HandoffView {
    pub component: String,
    pub tensor: String,
    pub offset: u64,
    pub bytes: u64,
    pub dtype: PipelineDType,
    pub rows: u32,
    pub row_width: u32,
    pub row_stride: u64,
    pub row_map: HandoffRowMap,
    pub lifetime: HandoffLifetime,
    pub packet_initialize: bool,
    pub immutable: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HandoffEndpoint {
    pub program: HandoffProgramRef,
    pub view: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum HandoffConsumer {
    Program { endpoint: HandoffEndpoint },
    Host,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum HandoffClass {
    Alias,
    DeviceCopy { program: HandoffProgramRef },
    Host,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum HandoffOrder {
    SameStream,
    Event { name: String },
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HandoffEdge {
    pub producer: HandoffEndpoint,
    pub consumer: HandoffConsumer,
    pub class: HandoffClass,
    pub dtype: PipelineDType,
    pub rows: HandoffRows,
    pub order: HandoffOrder,
    pub generation_delta: u32,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HandoffBindWrite {
    pub name: String,
    pub view: String,
    pub rows: HandoffRows,
    pub order: HandoffOrder,
    pub phase: HandoffBindPhase,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HandoffBindPhase {
    Admission,
    Tick,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HandoffReadbackField {
    pub name: String,
    pub offset: u64,
    pub dtype: PipelineDType,
    pub elements: u32,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HandoffReadback {
    pub producer: HandoffProgramRef,
    pub view: String,
    pub extent: String,
    pub row_stride: u64,
    pub fields: Vec<HandoffReadbackField>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sample_lengths: Option<HandoffSampleLengths>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HandoffSampleLengths {
    pub view: String,
    pub field: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum HandoffExecution {
    Synchronous,
    Overlap { feedback: String },
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HandoffPlan {
    pub components: BTreeMap<String, HandoffComponent>,
    pub arena: HandoffArena,
    pub extents: BTreeMap<String, u32>,
    pub views: BTreeMap<String, HandoffView>,
    pub schedule: Vec<HandoffProgramRef>,
    pub edges: Vec<HandoffEdge>,
    pub bind_writes: Vec<HandoffBindWrite>,
    pub readback: HandoffReadback,
    pub execution: HandoffExecution,
}

impl HandoffPlan {
    pub fn program(&self, p: &HandoffProgramRef) -> Result<&HandoffProgram> {
        self.components
            .get(&p.component)
            .and_then(|c| c.programs.get(&p.program))
            .ok_or_else(|| "pipeline handoff: unresolved program reference".into())
    }

    pub fn view(&self, name: &str) -> Result<&HandoffView> {
        self.views
            .get(name)
            .ok_or_else(|| "pipeline handoff: unresolved tensor view".into())
    }

    pub fn rows_capacity(&self, rows: &HandoffRows) -> Result<u32> {
        match rows {
            HandoffRows::Fixed { rows } => {
                require(*rows > 0, "empty fixed extent")?;
                Ok(*rows)
            }
            HandoffRows::Live { extent } => self
                .extents
                .get(extent)
                .copied()
                .ok_or_else(|| "pipeline handoff: unresolved live extent".into()),
        }
    }

    fn logical_capacity(&self, view: &HandoffView) -> Result<u32> {
        match &view.row_map {
            HandoffRowMap::Contiguous { .. } => Ok(view.rows),
            HandoffRowMap::Gather { map, .. } => Ok(self.view(map)?.rows),
        }
    }

    fn position(&self, program: &HandoffProgramRef) -> Result<usize> {
        self.schedule
            .iter()
            .position(|p| p == program)
            .ok_or_else(|| "pipeline handoff: unscheduled program".into())
    }

    fn access(&self, endpoint: &HandoffEndpoint, write: bool) -> Result<&HandoffAccess> {
        require(
            self.view(&endpoint.view)?.component == endpoint.program.component,
            "program and endpoint component differ",
        )?;
        self.program(&endpoint.program)?
            .accesses
            .iter()
            .find(|a| {
                a.view == endpoint.view
                    && if write {
                        a.mode.writes()
                    } else {
                        a.mode.reads()
                    }
            })
            .ok_or_else(|| "pipeline handoff: endpoint has no matching program access".into())
    }

    fn validate_order(order: &HandoffOrder) -> Result<()> {
        if let HandoffOrder::Event { name } = order {
            require(identifier(name, 128), "invalid event name")?;
        }
        Ok(())
    }

    pub fn validate(&self) -> Result<()> {
        require(
            !self.components.is_empty()
                && !self.views.is_empty()
                && !self.schedule.is_empty()
                && self.components.len() <= MAX_ITEMS
                && self.views.len() <= MAX_ITEMS
                && self.schedule.len() <= MAX_ITEMS
                && self.edges.len() <= MAX_ITEMS
                && self.bind_writes.len() <= MAX_ITEMS,
            "invalid plan collection size",
        )?;
        require(
            self.arena.bytes > 0
                && self.arena.alignment.is_power_of_two()
                && self.arena.alignment >= 8
                && matches!(self.arena.banks, 1 | 2),
            "invalid shared arena",
        )?;
        require(
            !self.extents.is_empty()
                && self.extents.len() <= MAX_ITEMS
                && self
                    .extents
                    .iter()
                    .all(|(n, &cap)| identifier(n, 128) && cap > 0),
            "invalid live extents",
        )?;
        for (name, component) in &self.components {
            require(
                identifier(name, 128)
                    && identifier(&component.driver, 128)
                    && !component.packet.is_empty()
                    && component.packet.len() <= 256
                    && !component.packet.starts_with('/')
                    && !component.packet.contains('\\')
                    && component
                        .packet
                        .split('/')
                        .all(|p| identifier(p, 128) && !matches!(p, "." | "..")),
                "invalid component",
            )?;
            require(
                component.objects.len() <= MAX_ITEMS
                    && component.objects.iter().all(|(role, file)| {
                        identifier(role, 128)
                            && !file.is_empty()
                            && file.len() <= 256
                            && file
                                .split('/')
                                .all(|p| identifier(p, 128) && !matches!(p, "." | ".."))
                    }),
                "invalid component object roles",
            )?;
            require(
                !component.programs.is_empty() && component.programs.len() <= MAX_ITEMS,
                "invalid program collection size",
            )?;
            for id in component.programs.keys() {
                require(identifier(id, 128), "invalid program id")?;
            }
        }
        let mut initialized = Vec::new();
        for (name, view) in &self.views {
            require(
                identifier(name, 128)
                    && self.components.contains_key(&view.component)
                    && identifier(&view.tensor, 127),
                "invalid tensor view id",
            )?;
            let width = u64::from(view.row_width).checked_mul(view.dtype.bytes());
            let span = width.and_then(|w| {
                u64::from(view.rows)
                    .checked_sub(1)?
                    .checked_mul(view.row_stride)?
                    .checked_add(w)
            });
            let last_bank = match view.lifetime {
                HandoffLifetime::Persistent => Some(0),
                HandoffLifetime::Transient { bank_stride } => {
                    require(
                        self.arena.banks == 1 || bank_stride >= view.bytes,
                        "overlap requires independent transient banks",
                    )?;
                    require(
                        bank_stride % view.dtype.bytes() == 0,
                        "unaligned bank stride",
                    )?;
                    bank_stride.checked_mul(u64::from(self.arena.banks - 1))
                }
            };
            require(
                view.rows > 0
                    && view.row_width > 0
                    && width.is_some_and(|w| view.row_stride >= w)
                    && view.row_stride % view.dtype.bytes() == 0
                    && view.offset % view.dtype.bytes() == 0
                    && span.is_some_and(|n| n <= view.bytes)
                    && last_bank
                        .and_then(|b| view.offset.checked_add(b)?.checked_add(view.bytes))
                        .is_some_and(|end| end <= self.arena.bytes),
                "view exceeds its arena or row geometry",
            )?;
            let domain = match &view.row_map {
                HandoffRowMap::Contiguous { domain } => domain,
                HandoffRowMap::Gather { map, domain, .. } => {
                    let map = self.view(map)?;
                    require(
                        map.dtype == PipelineDType::U32
                            && map.row_width == 1
                            && map.immutable
                            && matches!(map.row_map, HandoffRowMap::Contiguous { .. }),
                        "invalid generation-owned row map",
                    )?;
                    domain
                }
            };
            require(identifier(domain, 128), "invalid row domain")?;
            if view.packet_initialize {
                for (begin, end) in self.view_ranges(view) {
                    require(
                        initialized.iter().all(|&(a, b)| end <= a || begin >= b),
                        "overlapping packet initialization",
                    )?;
                    initialized.push((begin, end));
                }
            }
        }
        for component in self.components.values() {
            for program in component.programs.values() {
                self.validate_program(program)?;
                if let HandoffExecutable::Kernel { object_role, .. } = &program.entry {
                    require(
                        component.objects.contains_key(object_role),
                        "kernel object role is unresolved",
                    )?;
                }
            }
        }
        self.validate_bank_separation()?;
        let mut scheduled = BTreeSet::new();
        for p in &self.schedule {
            self.program(p)?;
            require(scheduled.insert(p), "duplicate scheduled program")?;
        }
        for (component, c) in &self.components {
            for program in c.programs.keys() {
                self.position(&HandoffProgramRef {
                    component: component.clone(),
                    program: program.clone(),
                })?;
            }
        }
        for edge in &self.edges {
            self.validate_edge(edge)?;
        }
        let mut names = BTreeSet::new();
        for bind in &self.bind_writes {
            let view = self.view(&bind.view)?;
            require(
                identifier(&bind.name, 128)
                    && names.insert(&bind.name)
                    && self.rows_capacity(&bind.rows)? <= self.logical_capacity(view)?,
                "invalid bind write",
            )?;
            Self::validate_order(&bind.order)?;
        }
        self.validate_readback()?;
        if let HandoffExecution::Overlap { feedback } = &self.execution {
            require(
                self.arena.banks == 2
                    && self.view(feedback)?.lifetime == HandoffLifetime::Persistent,
                "overlap requires two banks and persistent feedback",
            )?;
            require(
                self.edges.iter().any(|e| e.generation_delta == 1),
                "overlap has no declared feedback edge",
            )?;
        } else {
            require(
                self.edges.iter().all(|e| e.generation_delta == 0),
                "synchronous plan has next-generation edges",
            )?;
        }
        self.prove_dataflow()
    }

    fn validate_program(&self, program: &HandoffProgram) -> Result<()> {
        require(
            program.accesses.len() <= MAX_ITEMS,
            "too many program accesses",
        )?;
        let mut views = BTreeSet::new();
        for access in &program.accesses {
            let view = self.view(&access.view)?;
            require(
                views.insert((&access.view, access.mode))
                    && !program.accesses.iter().any(|a| {
                        a.view == access.view
                            && a.mode != access.mode
                            && (a.mode == HandoffAccessMode::ReadWrite
                                || access.mode == HandoffAccessMode::ReadWrite)
                    })
                    && self.rows_capacity(&access.rows)? <= self.logical_capacity(view)?,
                "duplicate or out-of-range program access",
            )?;
            if access.mode.writes() {
                require(
                    !view.immutable,
                    "program writes a generation-owned immutable view",
                )?;
                if let HandoffRowMap::Gather { unique, .. } = view.row_map {
                    require(unique, "writable gather map is not unique")?;
                }
            }
            if let Some(mask) = &access.mask {
                let mask = self.view(mask)?;
                require(
                    mask.dtype == PipelineDType::U32
                        && mask.row_width == 1
                        && mask.rows >= self.rows_capacity(&access.rows)?,
                    "invalid access validity mask",
                )?;
            }
        }
        for (i, left) in program.accesses.iter().enumerate() {
            for right in &program.accesses[i + 1..] {
                if left.view == right.view || !(left.mode.writes() || right.mode.writes()) {
                    continue;
                }
                let a = self.view(&left.view)?;
                let b = self.view(&right.view)?;
                require(
                    a.offset + a.bytes <= b.offset || b.offset + b.bytes <= a.offset,
                    "distinct program operands overlap without a typed in-place contract",
                )?;
            }
        }
        match &program.entry {
            HandoffExecutable::Packet { section, .. } => {
                require(identifier(section, 128), "invalid packet program section")
            }
            HandoffExecutable::Kernel {
                object_role,
                symbol,
                abi_version,
                capability,
                capability_value,
                parameters,
                grid,
                block,
                ..
            } => {
                require(
                    identifier(object_role, 128)
                        && identifier(symbol, 256)
                        && *abi_version > 0
                        && identifier(capability, 256)
                        && *capability_value > 0
                        && parameters.len() <= 128,
                    "kernel lacks a versioned typed ABI or capability",
                )?;
                require(
                    block.iter().all(|&n| n > 0)
                        && block
                            .iter()
                            .try_fold(1u32, |n, &v| n.checked_mul(v))
                            .is_some_and(|n| n <= 1024),
                    "invalid kernel block geometry",
                )?;
                for dim in grid {
                    let max = dim.bound(&self.extents, 0)?;
                    require(max > 0 && max <= u32::MAX as u64, "kernel grid overflow")?;
                }
                for param in parameters {
                    match param {
                        HandoffParameter::Tensor {
                            view,
                            dtype,
                            access,
                            alignment,
                        } => {
                            let reads = program
                                .accesses
                                .iter()
                                .any(|a| a.view == *view && a.mode.reads());
                            let writes = program
                                .accesses
                                .iter()
                                .any(|a| a.view == *view && a.mode.writes());
                            require(
                                self.view(view)?.dtype == *dtype
                                    && reads == access.reads()
                                    && writes == access.writes(),
                                "kernel pointer lacks matching typed access",
                            )?;
                            let tensor = self.view(view)?;
                            let bank_stride = match tensor.lifetime {
                                HandoffLifetime::Persistent => 0,
                                HandoffLifetime::Transient { bank_stride } => bank_stride,
                            };
                            require(
                                alignment.is_power_of_two()
                                    && u64::from(*alignment) >= dtype.bytes()
                                    && *alignment <= self.arena.alignment
                                    && tensor.offset % u64::from(*alignment) == 0
                                    && tensor.row_stride % u64::from(*alignment) == 0
                                    && bank_stride % u64::from(*alignment) == 0,
                                "kernel pointer alignment is not proven",
                            )?;
                        }
                        HandoffParameter::Scalar { dtype, value } => {
                            let max = value.bound(&self.extents, 0)?;
                            let fits = match dtype {
                                HandoffScalarType::U64 => true,
                                HandoffScalarType::U32 | HandoffScalarType::F32 => {
                                    max <= u32::MAX as u64
                                }
                                HandoffScalarType::I32 => max <= i32::MAX as u64,
                            };
                            require(fits, "scalar expression exceeds its declared type")?;
                            if *dtype == HandoffScalarType::F32 {
                                require(
                                    matches!(value, HandoffExpr::Constant { .. })
                                        && f32::from_bits(max as u32).is_finite(),
                                    "floating scalar must be a finite constant bit pattern",
                                )?;
                            }
                        }
                    }
                }
                Ok(())
            }
        }
    }

    fn validate_edge(&self, edge: &HandoffEdge) -> Result<()> {
        let source = self.view(&edge.producer.view)?;
        let produced = self.access(&edge.producer, true)?;
        let p = self.position(&edge.producer.program)?;
        require(
            edge.dtype == source.dtype
                && edge.generation_delta <= 1
                && rows_cover(&produced.rows, &edge.rows, &self.extents),
            "producer does not cover edge rows/type",
        )?;
        if edge.generation_delta == 1 {
            require(
                matches!(&self.execution, HandoffExecution::Overlap { feedback }
                    if feedback == &edge.producer.view)
                    && source.lifetime == HandoffLifetime::Persistent,
                "next-generation edge requires persistent feedback storage",
            )?;
        }
        Self::validate_order(&edge.order)?;
        match (&edge.class, &edge.consumer) {
            (HandoffClass::Host, HandoffConsumer::Host) => {
                require(
                    edge.generation_delta == 0
                        && edge.producer.program == self.readback.producer
                        && edge.producer.view == self.readback.view
                        && produced.mask.is_none(),
                    "host edge is not the declared readback",
                )?;
            }
            (class, HandoffConsumer::Program { endpoint }) => {
                let target = self.view(&endpoint.view)?;
                let consumed = self.access(endpoint, false)?;
                let q = self.position(&endpoint.program)?;
                require(
                    edge.dtype == target.dtype
                        && source.row_width == target.row_width
                        && rows_cover(&edge.rows, &consumed.rows, &self.extents)
                        && (produced.mask.is_none() || produced.mask == consumed.mask),
                    "consumer rows/type/validity exceed producer",
                )?;
                require(
                    edge.generation_delta == 1 || p < q,
                    "edge violates enqueue order",
                )?;
                if edge.generation_delta == 1 {
                    require(
                        source.packet_initialize
                            || target.packet_initialize
                            || self.bind_writes.iter().any(|bind| {
                                bind.phase == HandoffBindPhase::Admission
                                    && (bind.view == edge.producer.view
                                        || bind.view == endpoint.view)
                                    && rows_cover(&bind.rows, &consumed.rows, &self.extents)
                            }),
                        "next-generation edge has no first-tick initialization",
                    )?;
                }
                match class {
                    HandoffClass::Alias => {
                        let full_physical_source = edge.generation_delta == 0
                            && produced.mask.is_none()
                            && matches!(
                                (&source.row_map, &target.row_map, &produced.rows),
                                (
                                    HandoffRowMap::Contiguous { domain: a },
                                    HandoffRowMap::Gather { domain: b, .. },
                                    HandoffRows::Fixed { rows },
                                ) if a == b && *rows >= target.rows
                            );
                        require(
                            source.offset == target.offset
                                && source.row_stride == target.row_stride
                                && (source.row_map == target.row_map || full_physical_source)
                                && source.lifetime == target.lifetime,
                            "alias address functions differ",
                        )?;
                    }
                    HandoffClass::DeviceCopy { program } => {
                        let copy = self.program(program)?;
                        let c = self.position(program)?;
                        require(
                            edge.generation_delta == 0
                                && p < c
                                && c < q
                                && copy
                                    .accesses
                                    .iter()
                                    .any(|a| a.view == edge.producer.view && a.mode.reads())
                                && copy
                                    .accesses
                                    .iter()
                                    .any(|a| a.view == endpoint.view && a.mode.writes()),
                            "device copy has no ordered typed pack program",
                        )?;
                        let read = copy
                            .accesses
                            .iter()
                            .find(|a| a.view == edge.producer.view && a.mode.reads())
                            .unwrap();
                        let write = copy
                            .accesses
                            .iter()
                            .find(|a| a.view == endpoint.view && a.mode.writes())
                            .unwrap();
                        require(
                            rows_cover(&produced.rows, &read.rows, &self.extents)
                                && (produced.mask.is_none() || produced.mask == read.mask)
                                && rows_cover(&write.rows, &consumed.rows, &self.extents)
                                && (write.mask.is_none() || write.mask == consumed.mask)
                                && (source.offset + source.bytes <= target.offset
                                    || target.offset + target.bytes <= source.offset),
                            "pack reads unwritten rows or has overlapping destinations",
                        )?;
                    }
                    HandoffClass::Host => {
                        return Err("pipeline handoff: host edge has a device consumer".into())
                    }
                }
            }
            _ => return Err("pipeline handoff: device edge has a host consumer".into()),
        }
        Ok(())
    }

    fn validate_readback(&self) -> Result<()> {
        let r = &self.readback;
        let view = self.view(&r.view)?;
        self.position(&r.producer)?;
        let cap = self
            .extents
            .get(&r.extent)
            .ok_or("pipeline handoff: missing readback live extent")?;
        require(
            view.dtype == PipelineDType::U8
                && view.row_stride == r.row_stride
                && u64::from(view.row_width) == r.row_stride
                && *cap <= view.rows
                && matches!(view.row_map, HandoffRowMap::Contiguous { .. })
                && !r.fields.is_empty()
                && r.fields.len() <= 128,
            "invalid live-prefix readback geometry",
        )?;
        let host: Vec<_> = self
            .edges
            .iter()
            .filter(|e| matches!(e.class, HandoffClass::Host))
            .collect();
        require(
            host.len() == 1
                && host[0].rows
                    == HandoffRows::Live {
                        extent: r.extent.clone(),
                    },
            "readback must copy exactly one declared live prefix",
        )?;
        let mut names = BTreeSet::new();
        let mut ranges = Vec::new();
        for field in &r.fields {
            let end = u64::from(field.elements)
                .checked_mul(field.dtype.bytes())
                .and_then(|bytes| field.offset.checked_add(bytes));
            require(
                identifier(&field.name, 128)
                    && names.insert(&field.name)
                    && field.elements > 0
                    && field.offset % field.dtype.bytes() == 0
                    && r.row_stride % field.dtype.bytes() == 0
                    && end.is_some_and(|n| n <= r.row_stride),
                "invalid typed readback field",
            )?;
            let end = end.unwrap();
            require(
                ranges.iter().all(|&(a, b)| end <= a || field.offset >= b),
                "overlapping readback fields",
            )?;
            ranges.push((field.offset, end));
        }
        if let Some(lengths) = &r.sample_lengths {
            let field = r.fields.iter().find(|f| f.name == lengths.field);
            let lengths = self.view(&lengths.view)?;
            require(
                lengths.dtype == PipelineDType::U32
                    && lengths.row_width == 1
                    && lengths.rows >= *cap
                    && field.is_some_and(|f| f.dtype == PipelineDType::U32 && f.elements == 1),
                "invalid sample length tensor",
            )?;
        }
        Ok(())
    }

    fn view_ranges(&self, view: &HandoffView) -> impl Iterator<Item = (u64, u64)> {
        let count = if matches!(view.lifetime, HandoffLifetime::Persistent) {
            1
        } else {
            self.arena.banks
        };
        let stride = match view.lifetime {
            HandoffLifetime::Persistent => 0,
            HandoffLifetime::Transient { bank_stride } => bank_stride,
        };
        let (offset, bytes) = (view.offset, view.bytes);
        (0..count).map(move |bank| {
            let begin = offset + u64::from(bank) * stride;
            (begin, begin + bytes)
        })
    }

    fn validate_bank_separation(&self) -> Result<()> {
        let views: Vec<_> = self.views.values().collect();
        for immutable in views.iter().filter(|view| view.immutable) {
            for program in self.components.values().flat_map(|c| c.programs.values()) {
                for access in program.accesses.iter().filter(|a| a.mode.writes()) {
                    let writable = self.view(&access.view)?;
                    require(
                        !self
                            .view_ranges(immutable)
                            .any(|(a, b)| self.view_ranges(writable).any(|(c, d)| a < d && c < b)),
                        "program write aliases immutable storage",
                    )?;
                }
            }
        }
        for (i, left) in views.iter().enumerate() {
            for right in &views[i + 1..] {
                for (a_bank, (a, b)) in self.view_ranges(left).into_iter().enumerate() {
                    for (c_bank, (c, d)) in self.view_ranges(right).into_iter().enumerate() {
                        if b <= c || d <= a {
                            continue;
                        }
                        require(
                            matches!(
                                (&left.lifetime, &right.lifetime),
                                (HandoffLifetime::Persistent, HandoffLifetime::Persistent)
                            ) || matches!(
                                (&left.lifetime, &right.lifetime),
                                (
                                    HandoffLifetime::Transient { .. },
                                    HandoffLifetime::Transient { .. }
                                )
                            ) && a_bank == c_bank,
                            "execution banks or persistent/transient lifetimes overlap",
                        )?;
                    }
                }
            }
        }
        Ok(())
    }
}

fn rows_cover(
    producer: &HandoffRows,
    consumer: &HandoffRows,
    extents: &BTreeMap<String, u32>,
) -> bool {
    match (producer, consumer) {
        (HandoffRows::Live { extent: a }, HandoffRows::Live { extent: b }) => a == b,
        (HandoffRows::Fixed { rows: a }, HandoffRows::Fixed { rows: b }) => a >= b,
        (HandoffRows::Fixed { rows }, HandoffRows::Live { extent }) => {
            extents.get(extent).is_some_and(|n| rows >= n)
        }
        _ => false,
    }
}

// These records come from typed lowering/kernel signatures, never from deserializing a plan.
#[derive(Clone, Debug)]
pub struct HandoffProgramProof {
    pub entry: HandoffExecutable,
    pub accesses: Vec<HandoffAccess>,
    pub copies: Vec<HandoffCopyProof>,
    pub private_transients_banked: bool,
    pub persistent_stop_safe: bool,
}

#[derive(Clone, Debug)]
pub struct HandoffCopyProof {
    pub source: String,
    pub destination: String,
    pub rows: HandoffRows,
    pub mask: Option<String>,
}

#[derive(Clone, Debug, Default)]
pub struct HandoffProofCatalog {
    pub arena: Option<HandoffArena>,
    pub extents: BTreeMap<String, u32>,
    pub views: BTreeMap<String, HandoffView>,
    pub programs: BTreeMap<HandoffProgramRef, HandoffProgramProof>,
    pub readbacks: BTreeMap<HandoffProgramRef, HandoffReadback>,
}

pub struct ProvenHandoffPlan<'a> {
    plan: &'a HandoffPlan,
}

impl ProvenHandoffPlan<'_> {
    pub fn plan(&self) -> &HandoffPlan {
        self.plan
    }
}

impl HandoffPlan {
    pub fn prove<'a>(&'a self, catalog: &HandoffProofCatalog) -> Result<ProvenHandoffPlan<'a>> {
        self.validate()?;
        require(
            catalog.arena.as_ref() == Some(&self.arena)
                && catalog.extents == self.extents
                && catalog.views == self.views,
            "arena layout differs from typed lowering",
        )?;
        for reference in &self.schedule {
            let declared = self.program(reference)?;
            let actual = catalog
                .programs
                .get(reference)
                .ok_or("pipeline handoff: no typed program proof")?;
            require(
                actual.entry == declared.entry && actual.accesses == declared.accesses,
                "declared program differs from typed lowering/ABI",
            )?;
            if matches!(self.execution, HandoffExecution::Overlap { .. }) {
                require(
                    actual.private_transients_banked,
                    "unbanked private program resources",
                )?;
                for access in &actual.accesses {
                    if access.mode.writes()
                        && self.view(&access.view)?.lifetime == HandoffLifetime::Persistent
                    {
                        require(
                            actual.persistent_stop_safe,
                            "speculative persistent state is not stop-safe",
                        )?;
                    }
                }
            }
        }
        for edge in &self.edges {
            if let (HandoffClass::DeviceCopy { program }, HandoffConsumer::Program { endpoint }) =
                (&edge.class, &edge.consumer)
            {
                let consumed = self.access(endpoint, false)?;
                require(
                    catalog.programs.get(program).is_some_and(|p| {
                        p.copies.iter().any(|copy| {
                            copy.source == edge.producer.view
                                && copy.destination == endpoint.view
                                && rows_cover(&copy.rows, &consumed.rows, &self.extents)
                                && (copy.mask.is_none() || copy.mask == consumed.mask)
                        })
                    }),
                    "pack program has no typed copy/permutation proof",
                )?;
            }
        }
        require(
            catalog.readbacks.get(&self.readback.producer) == Some(&self.readback),
            "readback layout differs from typed pack signature",
        )?;
        Ok(ProvenHandoffPlan { plan: self })
    }

    fn prove_dataflow(&self) -> Result<()> {
        self.prove_read_coverage()?;
        for edge in &self.edges {
            if let HandoffConsumer::Program { endpoint } = &edge.consumer {
                let p = self.position(&edge.producer.program)?;
                let q = self.position(&endpoint.program)?;
                match &edge.class {
                    HandoffClass::Alias => {
                        self.prove_live_span(&edge.producer.view, p, q, edge.generation_delta)?
                    }
                    HandoffClass::DeviceCopy { program } => {
                        let c = self.position(program)?;
                        self.prove_live_span(&edge.producer.view, p, c, 0)?;
                        self.prove_live_span(&endpoint.view, c, q, 0)?;
                    }
                    HandoffClass::Host => unreachable!(),
                }
            } else {
                self.prove_live_span(
                    &edge.producer.view,
                    self.position(&edge.producer.program)?,
                    self.schedule.len(),
                    0,
                )?;
            }
        }
        Ok(())
    }

    fn prove_read_coverage(&self) -> Result<()> {
        for program in &self.schedule {
            let accesses = &self.program(program)?.accesses;
            for access in accesses {
                let mut implicit = Vec::new();
                if let HandoffRowMap::Gather { map, .. } = &self.view(&access.view)?.row_map {
                    implicit.push(map);
                }
                if let Some(mask) = &access.mask {
                    implicit.push(mask);
                }
                for view in implicit {
                    require(
                        accesses.iter().any(|a| {
                            &a.view == view
                                && a.mode.reads()
                                && a.mask.is_none()
                                && rows_cover(&a.rows, &access.rows, &self.extents)
                        }),
                        "implicit row map/mask lacks an unconditional typed read",
                    )?;
                }
            }
            for read in self
                .program(program)?
                .accesses
                .iter()
                .filter(|a| a.mode.reads())
            {
                let initialized = self.view(&read.view)?.packet_initialize
                    || self.bind_writes.iter().any(|b| {
                        b.view == read.view && rows_cover(&b.rows, &read.rows, &self.extents)
                    });
                let handed = self.edges.iter().any(|e| {
                    let consumes = matches!(&e.consumer, HandoffConsumer::Program { endpoint }
                        if endpoint.program == *program && endpoint.view == read.view);
                    let packs = matches!(&e.class, HandoffClass::DeviceCopy { program: p }
                        if p == program && e.producer.view == read.view);
                    (consumes || packs) && rows_cover(&e.rows, &read.rows, &self.extents)
                });
                require(
                    initialized || handed,
                    "read has no initialization or ordered producer edge",
                )?;
            }
        }
        Ok(())
    }

    fn prove_live_span(
        &self,
        view: &str,
        producer: usize,
        consumer: usize,
        delta: u32,
    ) -> Result<()> {
        let source = self.view(view)?;
        if delta == 1 {
            for bind in self
                .bind_writes
                .iter()
                .filter(|b| b.phase == HandoffBindPhase::Tick)
            {
                let written = self.view(&bind.view)?;
                require(
                    source.offset + source.bytes <= written.offset
                        || written.offset + written.bytes <= source.offset,
                    "tick bind destroys next-generation feedback",
                )?;
            }
        }
        for (i, reference) in self.schedule.iter().enumerate() {
            let between = if delta == 0 {
                producer < i && i < consumer
            } else {
                i > producer || i < consumer
            };
            if !between {
                continue;
            }
            for access in self
                .program(reference)?
                .accesses
                .iter()
                .filter(|a| a.mode.writes())
            {
                let written = self.view(&access.view)?;
                let disjoint = source.offset + source.bytes <= written.offset
                    || written.offset + written.bytes <= source.offset;
                require(disjoint, "intervening writer destroys a live handoff span")?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "packet_pipeline_handoff_tests.rs"]
mod tests;
