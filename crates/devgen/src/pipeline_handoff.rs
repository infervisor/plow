use packet::devbuild::Model;
use plow_asset::packet_pipeline::{
    HandoffArena, HandoffComponent, HandoffLifetime, HandoffProofCatalog, HandoffRowMap,
    HandoffView, PipelineDType,
};
use std::collections::BTreeMap;

pub struct LoweredComponent {
    pub declaration: HandoffComponent,
    pub facts: HandoffProofCatalog,
}

pub fn section_with_plan(
    model: &Model,
    section: packet::devbuild::SectionData,
    name: &str,
    plan: plow_asset::packet_pipeline::HandoffPlan,
    facts: &HandoffProofCatalog,
) -> Result<packet::devbuild::SectionData, String> {
    use plow_asset::packet_pipeline::*;
    if section.kind != packet::devbuild::SECT_METADATA || section.name != SECTION {
        return Err("handoff requires the canonical packet pipeline section".into());
    }
    plan.prove(facts)?;
    let PipelineMetadata::V1(legacy) = PipelineMetadata::parse(&section.data)? else {
        return Err("handoff section already contains a versioned plan".into());
    };
    let metadata = PipelineMetadata::V2(PacketPipelinesV2 {
        version: HANDOFF_VERSION,
        pipelines: legacy.pipelines,
        handoff_plans: BTreeMap::from([(name.into(), plan)]),
    });
    metadata.validate(model.progs.len(), |name| {
        model
            .tensors
            .iter()
            .find(|t| t.name == name)
            .map(|t| t.bytes)
    })?;
    let PipelineMetadata::V2(metadata) = metadata else {
        unreachable!()
    };
    Ok(packet::devbuild::SectionData {
        data: serde_json::to_vec(&metadata).map_err(|e| e.to_string())?,
        ..section
    })
}

/// Physical layout constructed alongside typed lowering, before packet initialization.
#[derive(Default)]
pub struct SharedLayout {
    bytes: u64,
    views: BTreeMap<String, HandoffView>,
}

impl SharedLayout {
    pub fn arena(&self) -> HandoffArena {
        HandoffArena {
            bytes: self.bytes,
            alignment: 256,
            banks: 1,
        }
    }

    pub fn views(&self) -> &BTreeMap<String, HandoffView> {
        &self.views
    }

    #[allow(clippy::too_many_arguments)]
    pub fn allocate(
        &mut self,
        name: &str,
        component: &str,
        tensor: &str,
        dtype: PipelineDType,
        rows: u32,
        width: u32,
        domain: &str,
        persistent: bool,
        initialized: bool,
        immutable: bool,
    ) -> Result<HandoffView, String> {
        if rows == 0 || width == 0 || self.views.contains_key(name) {
            return Err("handoff layout: empty or duplicate view".into());
        }
        let stride = u64::from(width)
            .checked_mul(dtype.bytes())
            .ok_or("handoff layout: stride overflow")?;
        let bytes = u64::from(rows)
            .checked_mul(stride)
            .ok_or("handoff layout: size overflow")?;
        let offset = self
            .bytes
            .checked_add(255)
            .map(|n| n & !255)
            .ok_or("handoff layout: alignment overflow")?;
        let end = offset
            .checked_add(bytes)
            .ok_or("handoff layout: arena overflow")?;
        let view = HandoffView {
            component: component.into(),
            tensor: tensor.into(),
            offset,
            bytes,
            dtype,
            rows,
            row_width: width,
            row_stride: stride,
            row_map: HandoffRowMap::Contiguous {
                domain: domain.into(),
            },
            lifetime: if persistent {
                HandoffLifetime::Persistent
            } else {
                HandoffLifetime::Transient { bank_stride: 0 }
            },
            packet_initialize: initialized,
            immutable,
        };
        self.bytes = end;
        self.views.insert(name.into(), view.clone());
        Ok(view)
    }

    /// `dtype` and geometry must come from the operation which lowered this tensor.
    #[allow(clippy::too_many_arguments)]
    pub fn packet_tensor(
        &mut self,
        name: &str,
        component: &str,
        model: &Model,
        handle: u32,
        dtype: PipelineDType,
        rows: u32,
        width: u32,
        domain: &str,
        persistent: bool,
        initialized: bool,
    ) -> Result<HandoffView, String> {
        let tensor = model
            .tensors
            .get(handle as usize)
            .ok_or("handoff layout: missing tensor")?;
        let bytes = u64::from(rows)
            .checked_mul(u64::from(width))
            .and_then(|n| n.checked_mul(dtype.bytes()))
            .ok_or("handoff layout: tensor size overflow")?;
        let stride = u64::from(width) * dtype.bytes();
        if rows == 0
            || stride == 0
            || bytes > tensor.bytes
            || tensor.bytes % stride != 0
            || initialized != tensor.init.is_some()
            || model.gen.iter().any(|g| g.tensor == handle)
            || model
                .tensors
                .iter()
                .filter(|t| t.name == tensor.name)
                .count()
                != 1
            || self.views.contains_key(name)
        {
            return Err("handoff layout: typed tensor size differs from lowered allocation".into());
        }
        if let Some(source) = self
            .views
            .values()
            .find(|view| view.component == component && view.tensor == tensor.name)
        {
            let lifetime = if persistent {
                HandoffLifetime::Persistent
            } else {
                HandoffLifetime::Transient { bank_stride: 0 }
            };
            if source.bytes != tensor.bytes
                || source.dtype != dtype
                || source.row_width != width
                || source.row_stride != stride
                || source.lifetime != lifetime
                || source.packet_initialize != initialized
            {
                return Err(
                    "handoff layout: repeated native tensor has incompatible storage".into(),
                );
            }
            let mut view = source.clone();
            view.rows = rows;
            view.row_map = HandoffRowMap::Contiguous {
                domain: domain.into(),
            };
            self.views.insert(name.into(), view.clone());
            return Ok(view);
        }
        let storage_rows = u32::try_from(tensor.bytes / stride)
            .map_err(|_| "handoff layout: storage row count overflow")?;
        let mut view = self.allocate(
            name,
            component,
            &tensor.name,
            dtype,
            storage_rows,
            width,
            domain,
            persistent,
            initialized,
            false,
        )?;
        view.rows = rows;
        self.views.insert(name.into(), view.clone());
        Ok(view)
    }

    pub fn alias(
        &mut self,
        name: &str,
        component: &str,
        tensor: &str,
        source: &str,
    ) -> Result<HandoffView, String> {
        if self.views.contains_key(name) {
            return Err("handoff layout: duplicate alias view".into());
        }
        let mut view = self
            .views
            .get(source)
            .cloned()
            .ok_or("handoff layout: alias source is missing")?;
        view.component = component.into();
        view.tensor = tensor.into();
        view.packet_initialize = false;
        self.views.insert(name.into(), view.clone());
        Ok(view)
    }

    pub fn packet_alias(
        &mut self,
        name: &str,
        component: &str,
        model: &Model,
        handle: u32,
        source: &str,
    ) -> Result<HandoffView, String> {
        let tensor = model
            .tensors
            .get(handle as usize)
            .ok_or("handoff layout: missing alias packet tensor")?;
        let view = self
            .views
            .get(source)
            .ok_or("handoff layout: missing alias source")?;
        let span = u64::from(view.rows - 1)
            .checked_mul(view.row_stride)
            .and_then(|n| n.checked_add(u64::from(view.row_width) * view.dtype.bytes()));
        if tensor.init.is_some()
            || model.gen.iter().any(|g| g.tensor == handle)
            || tensor.bytes > view.bytes
            || span.is_none_or(|n| n > tensor.bytes)
        {
            return Err("handoff layout: alias initialization or native extent differs".into());
        }
        let mut view = self.alias(name, component, &tensor.name, source)?;
        view.bytes = tensor.bytes;
        self.views.insert(name.into(), view.clone());
        Ok(view)
    }

    pub fn gather(
        &mut self,
        name: &str,
        map: &str,
        domain: &str,
        unique: bool,
    ) -> Result<HandoffView, String> {
        let index = self
            .views
            .get(map)
            .ok_or("handoff layout: missing gather map")?;
        if index.dtype != PipelineDType::U32 || index.row_width != 1 || !index.immutable {
            return Err("handoff layout: gather map is not immutable U32 rows".into());
        }
        let view = self
            .views
            .get_mut(name)
            .ok_or("handoff layout: missing gather view")?;
        view.row_map = HandoffRowMap::Gather {
            map: map.into(),
            domain: domain.into(),
            unique,
        };
        Ok(view.clone())
    }

    pub fn mark_immutable(&mut self, name: &str) -> Result<(), String> {
        self.views
            .get_mut(name)
            .ok_or("handoff layout: missing immutable view")?
            .immutable = true;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn repeated_native_tensor_views_share_storage_and_keep_independent_row_maps() {
        let mut builder = packet::devbuild::Builder::new(1);
        let hidden = builder.tensor("act.hidden", 512);
        let other = builder.tensor("act.other", 512);
        let model = Model {
            n_cu: 1,
            target: 0,
            tensors: builder.tensors(),
            progs: vec![],
            kv_row_insts: vec![],
            prog_t: vec![],
            gen: vec![],
        };
        let mut layout = SharedLayout::default();
        let producer = layout
            .packet_tensor(
                "backbone.hidden",
                "model",
                &model,
                hidden,
                PipelineDType::Bf16,
                8,
                16,
                "physical",
                false,
                false,
            )
            .unwrap();
        layout
            .allocate(
                "depth.map",
                "model",
                "row_map",
                PipelineDType::U32,
                4,
                1,
                "packed",
                false,
                false,
                true,
            )
            .unwrap();
        let before = layout.arena();
        let target = layout
            .packet_tensor(
                "depth.hidden",
                "model",
                &model,
                hidden,
                PipelineDType::Bf16,
                8,
                16,
                "physical",
                false,
                false,
            )
            .unwrap();
        let gathered = layout
            .gather("depth.hidden", "depth.map", "physical", false)
            .unwrap();
        assert_eq!(layout.arena(), before);
        assert_eq!(
            (producer.offset, producer.bytes),
            (target.offset, target.bytes)
        );
        assert_ne!(producer.row_map, gathered.row_map);
        assert_eq!(layout.views()["backbone.hidden"], producer);
        let smaller = layout
            .packet_tensor(
                "small.hidden",
                "model",
                &model,
                hidden,
                PipelineDType::Bf16,
                4,
                16,
                "physical",
                false,
                false,
            )
            .unwrap();
        assert_eq!(
            (smaller.offset, smaller.bytes, smaller.rows),
            (producer.offset, 512, 4)
        );
        assert_eq!(layout.arena(), before);
        for (dtype, rows, width, persistent, initialized) in [
            (PipelineDType::F32, 8, 8, false, false),
            (PipelineDType::Bf16, 8, 8, false, false),
            (PipelineDType::Bf16, 17, 16, false, false),
            (PipelineDType::Bf16, 8, 16, true, false),
            (PipelineDType::Bf16, 8, 16, false, true),
        ] {
            assert!(layout
                .packet_tensor(
                    "bad",
                    "model",
                    &model,
                    hidden,
                    dtype,
                    rows,
                    width,
                    "physical",
                    persistent,
                    initialized
                )
                .is_err());
            assert_eq!(layout.arena(), before);
        }
        assert!(layout
            .packet_tensor(
                "depth.hidden",
                "model",
                &model,
                hidden,
                PipelineDType::Bf16,
                8,
                16,
                "physical",
                false,
                false
            )
            .is_err());
        let distinct = layout
            .packet_tensor(
                "other.hidden",
                "model",
                &model,
                other,
                PipelineDType::Bf16,
                8,
                16,
                "physical",
                false,
                false,
            )
            .unwrap();
        assert_ne!(distinct.offset, producer.offset);
        let foreign = layout
            .packet_tensor(
                "codec.hidden",
                "codec",
                &model,
                hidden,
                PipelineDType::Bf16,
                8,
                16,
                "physical",
                false,
                false,
            )
            .unwrap();
        assert_ne!(foreign.offset, producer.offset);
    }

    #[test]
    fn repeated_native_views_preserve_the_actual_initializer_and_reject_ambiguous_names() {
        let mut builder = packet::devbuild::Builder::new(1);
        let handle = builder.tensor_init("state", vec![0; 32]);
        let mut model = Model {
            n_cu: 1,
            target: 0,
            tensors: builder.tensors(),
            progs: vec![],
            kv_row_insts: vec![],
            prog_t: vec![],
            gen: vec![],
        };
        let mut layout = SharedLayout::default();
        let a = layout
            .packet_tensor(
                "a",
                "model",
                &model,
                handle,
                PipelineDType::U32,
                8,
                1,
                "physical",
                true,
                true,
            )
            .unwrap();
        let arena = layout.arena();
        let b = layout
            .packet_tensor(
                "b",
                "model",
                &model,
                handle,
                PipelineDType::U32,
                4,
                1,
                "physical",
                true,
                true,
            )
            .unwrap();
        assert_eq!((a.offset, a.bytes), (b.offset, b.bytes));
        assert!(a.packet_initialize && b.packet_initialize);
        assert_eq!(layout.arena(), arena);
        model.tensors.push(model.tensors[handle as usize].clone());
        assert!(layout
            .packet_tensor(
                "ambiguous",
                "model",
                &model,
                handle,
                PipelineDType::U32,
                8,
                1,
                "physical",
                true,
                true
            )
            .is_err());
        assert_eq!(layout.arena(), arena);
    }

    #[test]
    fn aliases_keep_one_allocation_and_reject_initializer_or_extent_changes() {
        let mut layout = SharedLayout::default();
        let source = layout
            .allocate(
                "codec.codes",
                "codec",
                "input",
                PipelineDType::U32,
                224,
                32,
                "packed",
                false,
                true,
                false,
            )
            .unwrap();
        let before = layout.arena();
        let alias = layout
            .alias("pack.codes", "pack", "packed", "codec.codes")
            .unwrap();
        assert_eq!(layout.arena(), before);
        assert_eq!(
            (alias.offset, alias.bytes, alias.row_map, alias.lifetime),
            (
                source.offset,
                source.bytes,
                source.row_map.clone(),
                source.lifetime.clone()
            )
        );
        assert!(!alias.packet_initialize);
        assert!(layout
            .alias("pack.codes", "pack", "other", "codec.codes")
            .is_err());
        let mut builder = packet::devbuild::Builder::new(1);
        let h = builder.tensor("runtime.codes", source.bytes);
        let mut model = Model {
            n_cu: 1,
            target: 0,
            tensors: builder.tensors(),
            progs: vec![],
            kv_row_insts: vec![],
            prog_t: vec![],
            gen: vec![],
        };
        assert!(layout
            .packet_alias("model.codes", "model", &model, h, "codec.codes")
            .is_ok());
        model.tensors[h as usize].bytes += 4;
        assert!(layout
            .packet_alias("bad.big", "model", &model, h, "codec.codes")
            .is_err());
        model.tensors[h as usize].bytes = source.bytes - 4;
        assert!(layout
            .packet_alias("bad.short", "model", &model, h, "codec.codes")
            .is_err());
        model.tensors[h as usize].bytes = source.bytes;
        model.tensors[h as usize].init = Some(vec![0; source.bytes as usize]);
        assert!(layout
            .packet_alias("bad.init", "model", &model, h, "codec.codes")
            .is_err());
        assert_eq!(layout.arena(), before);
    }
}
