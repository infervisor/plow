use serde::Serialize;

use super::{AmdEngine, DecodeSegmentRoute, WG_THREADS_8};
use crate::device::hsa::SelectedKernelEvidence;

#[derive(Clone, Debug, Serialize)]
pub struct SelectedLaunch {
    pub program: usize,
    pub segment: usize,
    pub dispatch: usize,
    pub rows: u32,
    pub object_sha256: String,
    pub entry: String,
    pub grid_workgroups: [u32; 3],
    pub threads: u32,
    pub dynamic_lds_bytes: u32,
    pub explicit_argument_bytes: usize,
    pub variant: &'static str,
    pub caller_abi_sha256: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
pub struct NativeSiteContract {
    pub program: usize,
    pub segment: usize,
    pub variant: &'static str,
    pub contract: serde_json::Value,
    pub stride_wv_instruction_sha256: Option<String>,
    pub stride_wv_segments: Vec<usize>,
}

#[derive(Clone, Debug, Serialize)]
pub struct RouteGap {
    pub program: usize,
    pub segment: Option<usize>,
    pub reason: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct SelectedRouteManifest {
    pub schema: u32,
    pub scope: &'static str,
    pub packet_sha256: String,
    pub rank: u32,
    pub objects: Vec<SelectedKernelEvidence>,
    pub launches: Vec<SelectedLaunch>,
    pub native_contracts: Vec<NativeSiteContract>,
    pub gaps: Vec<RouteGap>,
    pub missing_identity_fields: Vec<&'static str>,
}

impl SelectedRouteManifest {
    pub(super) fn new(packet_sha256: String, rank: u32) -> Self {
        Self {
            schema: 2,
            scope: "post-load decode routes and conditional native variants, not executed dispatch or certification",
            packet_sha256,
            rank,
            objects: Vec::new(),
            launches: Vec::new(),
            native_contracts: Vec::new(),
            gaps: Vec::new(),
            missing_identity_fields: vec![
                "ABI field layout/precision contract",
                "register counts/wave size/occupancy",
                "full N/K/head/tail/inactive/live domain",
                "model/revision/checkpoint and hardware/topology/runtime/config/compiler/oracle",
                "segment dependency DAG and post-load instruction/counter patches",
                "dynamic dispatch/rebinding, explicit kernel overrides and auxiliary launches",
            ],
        }
    }

    fn record(
        &mut self,
        object: SelectedKernelEvidence,
        launch: SelectedLaunch,
    ) -> Result<(), String> {
        if launch.object_sha256 != object.object_sha256
            || launch.entry != object.symbol.entry
            || !plow_asset::certificates::is_sha256(&object.object_sha256)
            || object.object_bytes == 0
            || object.symbol.entry.is_empty()
            || launch.rows == 0
            || launch.grid_workgroups.contains(&0)
            || launch.threads == 0
            || launch.threads > 1024
            || launch.explicit_argument_bytes > object.symbol.kernarg_bytes as usize
            || self.launches.iter().any(|l| {
                (l.program, l.segment, l.dispatch, l.variant)
                    == (
                        launch.program,
                        launch.segment,
                        launch.dispatch,
                        launch.variant,
                    )
            })
        {
            return Err("invalid or duplicated selected launch/object binding".into());
        }
        if let Some(existing) = self.objects.iter().find(|o| {
            o.object_sha256 == object.object_sha256 && o.symbol.entry == object.symbol.entry
        }) {
            if existing != &object {
                return Err("immutable selected object facts changed".into());
            }
        } else {
            self.objects.push(object);
        }
        self.launches.push(launch);
        Ok(())
    }

    fn gap(&mut self, program: usize, segment: Option<usize>, reason: impl Into<String>) {
        self.gaps.push(RouteGap {
            program,
            segment,
            reason: reason.into(),
        });
    }

    fn decode_program(
        &mut self,
        program: usize,
        rows: u32,
        n_cu: u32,
        routes: &[DecodeSegmentRoute],
        launches: usize,
        mut resolve: impl FnMut(bool) -> Result<SelectedKernelEvidence, String>,
    ) {
        for segment in 0..launches {
            let attention = match routes.get(segment) {
                Some(DecodeSegmentRoute::Interpreter) => false,
                Some(DecodeSegmentRoute::MlaAttention) => true,
                Some(DecodeSegmentRoute::SparseMlaDecode(_)) => {
                    self.gap(
                        program,
                        Some(segment),
                        "sparse native/interpreter choice depends on live KV lengths",
                    );
                    continue;
                }
                Some(_) => {
                    self.gap(
                        program,
                        Some(segment),
                        "native adapter launch producer not covered",
                    );
                    continue;
                }
                None => {
                    self.gap(
                        program,
                        Some(segment),
                        "selected segment has no route record",
                    );
                    continue;
                }
            };
            let recorded = resolve(attention).and_then(|object| {
                let launch = SelectedLaunch {
                    program,
                    segment,
                    dispatch: 0,
                    rows,
                    object_sha256: object.object_sha256.clone(),
                    entry: object.symbol.entry.clone(),
                    grid_workgroups: [n_cu, 1, 1],
                    threads: WG_THREADS_8,
                    dynamic_lds_bytes: 0,
                    explicit_argument_bytes: std::mem::size_of::<super::DevProgram>(),
                    variant: "fixed",
                    caller_abi_sha256: None,
                };
                self.record(object, launch)
            });
            if let Err(error) = recorded {
                self.gap(program, Some(segment), error);
            }
        }
    }

    fn record_native(
        &mut self,
        program: usize,
        segment: usize,
        variants: Vec<crate::exec::amd_mla_bf16::NativeVariantEvidence>,
        wv: Option<(String, Vec<usize>)>,
    ) -> Result<(), String> {
        if variants.is_empty() {
            return Err("native route has no variants".into());
        }
        for native in variants {
            let variant = if native.contract.reuse_metadata {
                "metadata_reuser"
            } else {
                "metadata_producer"
            };
            if native.contract.output_head_stride == 1024
                && wv.as_ref().is_none_or(|w| w.1.is_empty())
            {
                return Err("padded native route has no actual WV instruction binding".into());
            }
            for (dispatch, launch) in native.launches.into_iter().enumerate() {
                let spec = launch.dispatch;
                let abi = serde_json::json!({
                    "scope": "caller source and loaded size/resource ABI, not field-layout or FP proof",
                    "caller_source_sha256": native.contract.caller_source_sha256,
                    "phase": spec.phase, "explicit_bytes": spec.explicit_bytes,
                    "kernarg_bytes": spec.kernarg_bytes, "static_lds": spec.static_lds,
                    "private_bytes_per_workitem": spec.private_bytes,
                });
                let binding = SelectedLaunch {
                    program,
                    segment,
                    dispatch,
                    rows: native.contract.rows,
                    object_sha256: launch.object.object_sha256.clone(),
                    entry: launch.object.symbol.entry.clone(),
                    grid_workgroups: spec.grid,
                    threads: u32::from(spec.threads),
                    dynamic_lds_bytes: spec.dynamic_lds,
                    explicit_argument_bytes: spec.explicit_bytes as usize,
                    variant,
                    caller_abi_sha256: Some(plow_asset::decode_objects::image_sha256(
                        &serde_json::to_vec(&abi).map_err(|e| e.to_string())?,
                    )),
                };
                self.record(launch.object, binding)?;
            }
            self.native_contracts.push(NativeSiteContract {
                program,
                segment,
                variant,
                contract: serde_json::to_value(&native.contract).map_err(|e| e.to_string())?,
                stride_wv_instruction_sha256: wv.as_ref().map(|w| w.0.clone()),
                stride_wv_segments: wv.as_ref().map_or_else(Vec::new, |w| w.1.clone()),
            });
        }
        Ok(())
    }

    pub fn scoped_digest(&self) -> Result<String, serde_json::Error> {
        Ok(plow_asset::decode_objects::image_sha256(
            &serde_json::to_vec(self)?,
        ))
    }
}

impl AmdEngine {
    pub fn selected_route_evidence(&self) -> &SelectedRouteManifest {
        &self.selected_routes
    }

    pub(super) fn capture_selected_routes(&self) -> SelectedRouteManifest {
        let mut out = SelectedRouteManifest::new(
            self.selected_routes.packet_sha256.clone(),
            self.tp.map_or(0, |tp| tp.rank),
        );
        for (program, prog) in self.progs.iter().enumerate() {
            if !prog.role.is_decode_rung() {
                out.gap(
                    program,
                    None,
                    "prefill/packed/body route producer not covered",
                );
                continue;
            }
            let segmented =
                self.tp.is_some() || super::requires_segmented_decode(&prog.decode_routes);
            let count = if segmented {
                self.decode_launches(program)
            } else {
                1
            };
            out.decode_program(
                program,
                prog.t,
                self.n_cu,
                &prog.decode_routes,
                count,
                |attention| {
                    let kernel = if attention {
                        self.k_decode_mla
                    } else {
                        Some(self.decode_kernel_for(program))
                    }
                    .ok_or("selected route has no resolved kernel")?;
                    self.be
                        .selected_kernel_evidence(kernel)
                        .map_err(|e| e.to_string())
                },
            );
            for (segment, route) in prog.decode_routes.iter().take(count).enumerate() {
                let DecodeSegmentRoute::MlaBf16(route) = route else {
                    continue;
                };
                let recorded = self
                    .mla_bf16
                    .as_ref()
                    .ok_or_else(|| "native BF16 route has no loaded adapter".to_string())
                    .and_then(|kernel| {
                        kernel
                            .selected_evidence(
                                &self.be,
                                *route,
                                crate::config::RuntimeConfig::get()
                                    .amd
                                    .mla_bf16_metadata_hoist,
                            )
                            .map_err(|e| e.to_string())
                    })
                    .and_then(|variants| {
                        let output = variants.first().map(|v| v.contract.output_handle);
                        let wv = self.pf_src[program]
                            .iter()
                            .enumerate()
                            .find(|(_, inst)| {
                                inst.op == packet::dev::DevOp::MlaBmmFp8 as u16
                                    && Some(inst.t[1]) == output
                                    && inst.i[5] == 1024
                            })
                            .map(|(instruction, inst)| {
                                let mut segments: Vec<_> = self.pf_stream[program]
                                    .iter()
                                    .filter(|e| e.inst as usize == instruction)
                                    .map(|e| e.seg as usize)
                                    .collect();
                                segments.sort_unstable();
                                segments.dedup();
                                (
                                    plow_asset::decode_objects::image_sha256(super::as_bytes(
                                        std::slice::from_ref(inst),
                                    )),
                                    segments,
                                )
                            });
                        out.record_native(program, segment, variants, wv)
                    });
                match recorded {
                    Ok(()) => out
                        .gaps
                        .retain(|gap| gap.program != program || gap.segment != Some(segment)),
                    Err(error) => out.gap(program, Some(segment), error),
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn object() -> SelectedKernelEvidence {
        SelectedKernelEvidence {
            object_sha256: "b".repeat(64),
            object_bytes: 4096,
            symbol: crate::device::hsa::ResolvedKernelEvidence {
                entry: "kernel".into(),
                kernarg_bytes: 320,
                static_lds_bytes: 8192,
                private_bytes_per_workitem: 0,
            },
        }
    }

    fn launch(object: &SelectedKernelEvidence, program: usize) -> SelectedLaunch {
        SelectedLaunch {
            program,
            segment: 0,
            dispatch: 0,
            rows: 16 << program,
            object_sha256: object.object_sha256.clone(),
            entry: object.symbol.entry.clone(),
            grid_workgroups: [256, 1, 1],
            threads: WG_THREADS_8,
            dynamic_lds_bytes: 0,
            explicit_argument_bytes: 64,
            variant: "fixed",
            caller_abi_sha256: None,
        }
    }

    #[test]
    fn selected_entry_is_deduplicated_across_actual_sites_without_resource_defaults() {
        let object = object();
        let mut manifest = SelectedRouteManifest::new("a".repeat(64), 0);
        manifest.record(object.clone(), launch(&object, 0)).unwrap();
        let mut second = launch(&object, 1);
        second.grid_workgroups[0] = 128;
        manifest.record(object.clone(), second).unwrap();
        assert_eq!(manifest.objects.len(), 1);
        assert_eq!(manifest.launches.len(), 2);
        assert!(!manifest.missing_identity_fields.is_empty());
        let json = serde_json::to_value(&manifest).unwrap();
        assert!(json["objects"][0]["symbol"].get("vgpr").is_none());
        assert!(
            serde_json::from_value::<plow_asset::certificates::ExecutionIdentity>(json).is_err()
        );
        assert!(manifest.record(object.clone(), launch(&object, 0)).is_err());
        let mut bad = launch(&object, 2);
        bad.entry = "unknown".into();
        assert!(manifest.record(object.clone(), bad).is_err());
        let mut bad = launch(&object, 2);
        bad.explicit_argument_bytes = 4096;
        assert!(manifest.record(object.clone(), bad).is_err());
        let mut changed = object.clone();
        changed.symbol.static_lds_bytes += 1;
        assert!(manifest.record(changed, launch(&object, 2)).is_err());
    }

    #[test]
    fn producer_consumes_selected_route_order_and_records_unresolved_sites_as_gaps() {
        let routes = [
            DecodeSegmentRoute::Interpreter,
            DecodeSegmentRoute::MlaAttention,
            DecodeSegmentRoute::Interpreter,
        ];
        let mut manifest = SelectedRouteManifest::new("a".repeat(64), 3);
        let mut selected = Vec::new();
        manifest.decode_program(2, 32, 128, &routes, 4, |attention| {
            selected.push(attention);
            if attention {
                Err("missing MLA object".into())
            } else {
                Ok(object())
            }
        });
        assert_eq!(selected, [false, true, false]);
        assert_eq!(manifest.objects.len(), 1);
        assert_eq!(
            manifest
                .launches
                .iter()
                .map(|l| l.segment)
                .collect::<Vec<_>>(),
            [0, 2]
        );
        assert_eq!(
            manifest.gaps.iter().map(|g| g.segment).collect::<Vec<_>>(),
            [Some(1), Some(3)]
        );
        for l in &manifest.launches {
            assert_eq!(
                (l.program, l.rows, l.grid_workgroups, l.threads),
                (2, 32, [128, 1, 1], WG_THREADS_8)
            );
            assert_eq!(
                l.explicit_argument_bytes,
                std::mem::size_of::<super::super::DevProgram>()
            );
        }
        manifest.decode_program(3, 16, 256, &routes[..1], 1, |_| Ok(object()));
        assert_eq!(manifest.objects.len(), 1);
        assert_eq!(manifest.launches.len(), 3);
    }

    #[test]
    fn native_scoped_identity_changes_with_same_packet_but_flags_objects_or_consumer_change() {
        let packet = "a".repeat(64);
        let mut digests = std::collections::BTreeSet::new();
        for padded in [false, true] {
            for hoist in [false, true] {
                let variants = crate::exec::amd_mla_bf16::test_native_evidence(padded, hoist);
                let mut manifest = SelectedRouteManifest::new(packet.clone(), 0);
                let wv = padded.then(|| ("b".repeat(64), vec![2]));
                manifest
                    .record_native(0, 1, variants.clone(), wv.clone())
                    .unwrap();
                assert!(digests.insert(manifest.scoped_digest().unwrap()));
                assert_eq!(manifest.packet_sha256, packet);
                assert!(
                    manifest
                        .launches
                        .iter()
                        .all(|l| l.caller_abi_sha256.is_some())
                );
                let mut changed = variants.clone();
                for v in &mut changed {
                    for l in &mut v.launches {
                        l.object.object_sha256 = "c".repeat(64);
                    }
                }
                let mut other = SelectedRouteManifest::new(packet.clone(), 0);
                other.record_native(0, 1, changed, wv).unwrap();
                assert_ne!(
                    manifest.scoped_digest().unwrap(),
                    other.scoped_digest().unwrap()
                );
                if padded {
                    let mut missing = SelectedRouteManifest::new(packet.clone(), 0);
                    assert!(missing.record_native(0, 1, variants.clone(), None).is_err());
                    let mut other = SelectedRouteManifest::new(packet.clone(), 0);
                    other
                        .record_native(0, 1, variants, Some(("d".repeat(64), vec![2])))
                        .unwrap();
                    assert_ne!(
                        manifest.scoped_digest().unwrap(),
                        other.scoped_digest().unwrap()
                    );
                }
            }
        }
        assert_eq!(digests.len(), 4);
    }
}
