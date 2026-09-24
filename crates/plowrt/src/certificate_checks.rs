//! Load-time replay of compiler obligations. This is not execution-identity,
//! floating-point implementation, runtime-rewrite, or empirical qualification.

use std::collections::BTreeSet;
use std::path::Path;
use std::sync::Mutex;

use plow_asset::certificates::{PacketCheckReceipts, PACKET_CHECKS_FILE};
use plow_asset::decode_objects::image_sha256;

use crate::asset::devblob::DevBlob;
use crate::{Result, RuntimeError};

pub(crate) fn check_packet(blob_path: &Path, raw: &[u8], blob: &DevBlob) -> Result<()> {
    let path = blob_path.with_file_name(PACKET_CHECKS_FILE);
    let bytes = match std::fs::read(&path) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(source) => return Err(RuntimeError::Io { path, source }),
    };
    let rejected = |reason: String| RuntimeError::Rejected(format!("{}: {reason}", path.display()));
    let receipts: PacketCheckReceipts =
        serde_json::from_slice(&bytes).map_err(|error| rejected(error.to_string()))?;
    receipts.validate_packet(raw).map_err(rejected)?;
    for check in &receipts.checks {
        let Some(index) = check.program else {
            continue;
        };
        let program = blob
            .progs
            .get(index)
            .ok_or_else(|| rejected("receipt program is absent from packet".into()))?;
        if check.scope == plow_asset::certificates::SemanticScope::SelectedGemmPolicy {
            let expected = blob.with_packet_view(|packet|
                plow_asset::gemm_policy::binding(packet, index, &check.request)).map_err(rejected)?;
            if check.request.get("wire_binding") != Some(&expected) {
                return Err(rejected("GEMM policy differs from loaded instruction/placement".into()));
            }
            continue;
        }
        if check.scope == plow_asset::certificates::SemanticScope::CoarseDependencyPreservation {
            let expected = blob.with_packet_view(|packet|
                plow_asset::logical_effects::coarse_protocol(&packet.programs[index]))
                .map_err(rejected)?;
            if check.request.get("protocol") != Some(&expected) {
                return Err(rejected("dependency receipt differs from loaded wire counters".into()));
            }
        }
        if check.scope == plow_asset::certificates::SemanticScope::LogicalTensorEffects {
            let expected = blob.with_packet_view(|packet|
                plow_asset::logical_effects::obligation(packet, index)).map_err(rejected)?;
            if check.request != expected {
                return Err(rejected("logical effects differ from loaded operands/counters".into()));
            }
            continue;
        }
        if check.scope == plow_asset::certificates::SemanticScope::LayoutMapping {
            let producer_index = check
                .request
                .get("producer_instruction")
                .and_then(serde_json::Value::as_u64)
                .and_then(|n| usize::try_from(n).ok())
                .ok_or_else(|| rejected("layout producer index missing".into()))?;
            let producer = program
                .insts
                .get(producer_index)
                .ok_or_else(|| rejected("layout producer absent".into()))?;
            let consumer = producer_index
                .checked_add(1)
                .and_then(|index| program.insts.get(index))
                .ok_or_else(|| rejected("layout consumer absent".into()))?;
            let capacity = blob
                .tensors
                .get(producer.t[0] as usize)
                .ok_or_else(|| rejected("layout output absent".into()))?
                .bytes;
            let expected = plow_asset::certificates::mla_layout_obligation(
                producer_index,
                producer,
                consumer,
                capacity,
            )
            .map_err(rejected)?;
            if check.request != expected {
                return Err(rejected(
                    "layout obligation differs from loaded instructions".into(),
                ));
            }
            continue;
        }
        if check
            .request
            .pointer("/task_graph/n")
            .and_then(serde_json::Value::as_u64)
            != Some(program.insts.len() as u64)
        {
            return Err(rejected(
                "receipt instruction domain differs from packet".into(),
            ));
        }
    }
    if receipts.checks.is_empty() {
        return Ok(());
    }
    let verifier = match lean_verify::verifier_sha256() {
        Ok(digest) => digest,
        Err(error) => {
            tracing::warn!(%error, "compiler check receipts not rechecked: no verifier identity; loading unverified");
            return Ok(());
        }
    };
    if receipts
        .checks
        .iter()
        .any(|check| check.verifier_sha256 != verifier)
    {
        tracing::warn!(
            "compiler check receipts not rechecked: verifier identity changed; loading unverified"
        );
        return Ok(());
    }
    // Shared TP ranks validate identical obligations once at load; no token-loop work.
    static CHECKED: Mutex<BTreeSet<(String, String)>> = Mutex::new(BTreeSet::new());
    let key = (image_sha256(&bytes), verifier.clone());
    let mut checked = CHECKED.lock().unwrap();
    if checked.contains(&key) {
        return Ok(());
    }
    let requests: Vec<_> = receipts
        .checks
        .iter()
        .map(|check| (check.checkpoint.as_str(), check.request.clone()))
        .collect();
    let (certs, executed_verifier) = match lean_verify::call_batch_bound(&requests) {
        Ok(certs) => certs,
        Err(error) if error.is_binary_unusable() => {
            tracing::warn!(%error, "compiler check receipts not rechecked: verifier unusable; loading unverified");
            return Ok(());
        }
        Err(error) => return Err(rejected(error.to_string())),
    };
    if executed_verifier != verifier {
        return Err(rejected(
            "verifier changed before replaying compiler obligations".into(),
        ));
    }
    for (receipt, cert) in receipts.checks.iter().zip(certs) {
        if !cert.ok
            || serde_json::to_value(&cert).map_err(|error| rejected(error.to_string()))?
                != receipt.response
        {
            return Err(rejected(format!(
                "program {:?}: compiler obligation rejected or envelope changed: {:?}",
                receipt.program, cert.reason
            )));
        }
    }
    checked.insert(key);
    tracing::info!(checks = receipts.checks.len(),
        "packet-bound compiler obligations rechecked; supplied coarse-graph/measured-policy scope only, no performance qualification");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use packet::dev::DevOp;
    use packet::devbuild::{Builder, Model};
    use plow_asset::certificates::{CompileCheckReceipt, SemanticScope};
    use serde_json::json;

    #[test]
    #[ignore = "requires built plow_verify; CPU-only"]
    fn selected_gemm_receipt_reconstructs_wire_after_packet_hash_changes() {
        let mut b = Builder::new(2);
        let out = b.tensor("output", 128 * 256 * 2);
        let input = b.tensor("input", 128 * 512 * 2);
        let weight = b.tensor("weight", 256 * 512 * 2);
        b.emit(DevOp::GemmMed, vec![0,1], &[], |d| {
            d.t[..3].copy_from_slice(&[out,input,weight]);
            d.i[..3].copy_from_slice(&[128,256,512]);
        });
        let p = b.finish();
        let mut decode = Builder::new(2);
        decode.emit(DevOp::Nop,vec![0],&[],|_| {});
        let mut model = Model { n_cu:2,target:0,tensors:p.tensors.clone(),
            progs:vec![p,decode.finish()],prog_t:vec![128,1],gen:vec![],kv_row_insts:vec![] };
        let geometry = json!({"m":128,"n":256,"k":512,"n_cu":2,"quant":"None"});
        let domain = image_sha256(&serde_json::to_vec(&geometry).unwrap());
        let key = format!("{}:implementation",DevOp::GemmMed as u16);
        let request = json!({"policy_kind":plow_asset::gemm_policy::KIND,"geometry":geometry,
            "selected_opcode":DevOp::GemmMed as u16,"required":[domain],
            "candidates":[{"domain":domain,"key":key,"cost":1,"qualified":true}],
            "choices":[{"domain":domain,"key":key}]});
        let request = plow_asset::program::with_model(&model,|packet|
            plow_asset::gemm_policy::bind(packet,0,&request).unwrap());
        let (mut certs,verifier) = lean_verify::call_batch_bound(&[("R",request.clone())]).unwrap();
        let cert = certs.remove(0);
        assert!(cert.ok);
        let raw = model.to_blob();
        let mut receipts = PacketCheckReceipts { schema:1,packet_sha256:image_sha256(&raw),
            compiler_sha256:"a".repeat(64),checks:vec![CompileCheckReceipt {
                program:Some(0),scope:SemanticScope::SelectedGemmPolicy,checkpoint:"R".into(),
                request_sha256:plow_asset::certificates::request_sha256(&request).unwrap(),
                verifier_sha256:verifier,request,response:serde_json::to_value(cert).unwrap(),
            }] };
        let directory = std::env::temp_dir().join(format!("plow-gemm-receipts-{}",std::process::id()));
        std::fs::create_dir_all(&directory).unwrap();
        let path = directory.join("model.pkt");
        let sidecar = directory.join(PACKET_CHECKS_FILE);
        let write = |value:&PacketCheckReceipts| std::fs::write(&sidecar,serde_json::to_vec(value).unwrap()).unwrap();
        write(&receipts);
        let blob = DevBlob::parse(&raw).unwrap();
        check_packet(&path,&raw,&blob).unwrap();
        check_packet(&path,&raw,&blob).unwrap();
        for change in 0..5 {
            let original = model.progs[0].insts[0].clone();
            match change {
                0 => model.progs[0].insts[0].op = DevOp::GemmSmall as u16,
                1 => model.progs[0].insts[0].i[1] += 1,
                2 => model.progs[0].insts[0].t[7] = out,
                3 => model.progs[0].insts[0].f[0] = 0.125,
                _ => model.progs[0].insts[0].t.swap(1,2),
            }
            let raw = model.to_blob();
            receipts.packet_sha256 = image_sha256(&raw);
            write(&receipts);
            let blob = DevBlob::parse(&raw).unwrap();
            assert!(check_packet(&path,&raw,&blob).is_err(),"wire mutation {change}");
            model.progs[0].insts[0] = original;
        }
        std::fs::remove_file(sidecar).unwrap();
        std::fs::remove_dir(directory).unwrap();
    }

    #[test]
    #[ignore = "requires built plow_verify; CPU-only"]
    fn logical_effect_receipt_is_reconstructed_from_loaded_packet() {
        let mut b = Builder::new(1);
        let x = b.tensor("x", 16);
        let y = b.tensor("y", 16);
        let first = b.emit(DevOp::Residual, vec![0], &[], |d| {
            d.t[..3].copy_from_slice(&[y,x,x]); d.i[0] = 8;
        });
        b.emit(DevOp::Residual, vec![0], &[first], |d| {
            d.t[..3].copy_from_slice(&[x,y,y]); d.i[0] = 8;
        });
        let p = b.finish();
        let mut model = Model { n_cu:1,target:0,tensors:p.tensors.clone(),progs:vec![p],
            kv_row_insts:vec![],prog_t:vec![1],gen:vec![] };
        let request = plow_asset::program::with_model(&model, |packet|
            plow_asset::logical_effects::obligation(packet,0).unwrap());
        let (mut certs, verifier) = lean_verify::call_batch_bound(&[("D",request.clone())]).unwrap();
        let cert = certs.remove(0);
        assert!(cert.ok, "{:?}",cert.reason);
        let raw = model.to_blob();
        let mut receipts = PacketCheckReceipts { schema:1,packet_sha256:image_sha256(&raw),
            compiler_sha256:"a".repeat(64), checks:vec![CompileCheckReceipt {
                program:Some(0),scope:SemanticScope::LogicalTensorEffects,checkpoint:"D".into(),
                request_sha256:plow_asset::certificates::request_sha256(&request).unwrap(),
                verifier_sha256:verifier,request,response:serde_json::to_value(cert).unwrap(),
            }] };
        let directory = std::env::temp_dir().join(format!("plow-effects-receipts-{}",std::process::id()));
        std::fs::create_dir_all(&directory).unwrap();
        let packet_path = directory.join("model.pkt");
        let sidecar = directory.join(PACKET_CHECKS_FILE);
        std::fs::write(&sidecar,serde_json::to_vec(&receipts).unwrap()).unwrap();
        check_packet(&packet_path,&raw,&DevBlob::parse(&raw).unwrap()).unwrap();
        for mutation in 0..3 {
            match mutation {
                0 => model.progs[0].insts[1].t[0] = y,
                1 => model.tensors[x as usize].bytes += 16,
                _ => model.progs[0].waits[0].threshold = 0,
            }
            let raw = model.to_blob();
            receipts.packet_sha256 = image_sha256(&raw);
            std::fs::write(&sidecar,serde_json::to_vec(&receipts).unwrap()).unwrap();
            assert!(check_packet(&packet_path,&raw,&DevBlob::parse(&raw).unwrap()).is_err(),
                "mutation {mutation}: even a matching new packet hash cannot reuse effects");
        }
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    #[ignore = "requires built plow_verify; CPU-only"]
    fn packet_receipts_replay_at_load_and_reject_tampering() {
        let mut builder = Builder::new(1);
        let first = builder.emit(DevOp::Residual, vec![0], &[], |_| {});
        builder.emit(DevOp::Residual, vec![0], &[first], |_| {});
        let program = builder.finish();
        let mut model = Model {
            n_cu: 1,
            target: 0,
            tensors: program.tensors.clone(),
            progs: vec![program],
            kv_row_insts: vec![],
            prog_t: vec![1],
            gen: vec![],
        };
        let raw = model.to_blob();
        let blob = DevBlob::parse(&raw).unwrap();
        let request = json!({"task_graph":{"n":2,"edges":[[0,1]]},
            "protocol":{"waits":[[],[0]],"succs":[[0],[1]],"resource":[0,1],
                "stream_idx":[0,0],"threshold":{}},
            "dependency_paths":[[]],"address_map":[]});
        let cert = lean_verify::call("D", request.clone()).unwrap();
        assert!(cert.ok);
        let receipt = CompileCheckReceipt {
            program: Some(0),
            scope: SemanticScope::CoarseDependencyPreservation,
            checkpoint: "D".into(),
            request_sha256: plow_asset::certificates::request_sha256(&request).unwrap(),
            verifier_sha256: lean_verify::verifier_sha256().unwrap(),
            request,
            response: serde_json::to_value(cert).unwrap(),
        };
        let mut receipts = PacketCheckReceipts {
            schema: 1,
            packet_sha256: image_sha256(&raw),
            compiler_sha256: "a".repeat(64),
            checks: vec![receipt],
        };
        let policy = json!({"required":["rung16"],"candidates":[
            {"domain":"rung16","key":"baseline","cost":20,"qualified":true},
            {"domain":"rung16","key":"selected","cost":10,"qualified":true}],
            "choices":[{"domain":"rung16","key":"selected"}]});
        let cert = lean_verify::call("R", policy.clone()).unwrap();
        assert!(cert.ok);
        receipts.checks.push(CompileCheckReceipt {
            program: None,
            scope: SemanticScope::MeasuredPolicy,
            checkpoint: "R".into(),
            request_sha256: plow_asset::certificates::request_sha256(&policy).unwrap(),
            verifier_sha256: lean_verify::verifier_sha256().unwrap(),
            request: policy,
            response: serde_json::to_value(cert).unwrap(),
        });
        let var = |name: &str| json!(["variable",name,[]]);
        let node = |head: &str,args: Vec<serde_json::Value>| json!(["call",head,args]);
        let lhs = node("Linear",vec![node("RmsNorm",vec![var("x"),var("w"),var("eps")]),var("wl"),var("out")]);
        let rhs = node("FusedNormLinear",vec![var("x"),var("w"),var("wl"),var("eps"),var("out")]);
        let rewrite = json!({"rules":["rmsnorm-linear-fuse"],"source_sha256":"c".repeat(64),
            "bodies":[{"name":"rmsnorm-linear-fuse","lhs":lhs,"rhs":rhs}]});
        let cert = lean_verify::call("A",rewrite.clone()).unwrap();
        assert!(cert.ok);
        receipts.checks.push(CompileCheckReceipt {
            program:None,scope:SemanticScope::RewriteBodyExpansion,checkpoint:"A".into(),
            request_sha256:plow_asset::certificates::request_sha256(&rewrite).unwrap(),
            verifier_sha256:lean_verify::verifier_sha256().unwrap(),request:rewrite,
            response:serde_json::to_value(cert).unwrap(),
        });
        let directory = std::env::temp_dir().join(format!("plow-receipts-{}", std::process::id()));
        std::fs::create_dir_all(&directory).unwrap();
        let packet_path = directory.join("model.pkt");
        let sidecar = directory.join(PACKET_CHECKS_FILE);
        let write = |value: &PacketCheckReceipts| {
            std::fs::write(&sidecar, serde_json::to_vec(value).unwrap()).unwrap()
        };
        // Missing receipts retain legacy unverified bringup.
        check_packet(&packet_path, &raw, &blob).unwrap();
        write(&receipts);
        check_packet(&packet_path, &raw, &blob).unwrap();
        check_packet(&packet_path, &raw, &blob).unwrap(); // cached, still packet-bound
        for mutation in 0..7 {
            let mut bad = receipts.clone();
            match mutation {
                0 => bad.packet_sha256 = "b".repeat(64),
                1 => bad.checks[0].request["dependency_paths"] = json!([]),
                2 => bad.checks[0].program = Some(1),
                3 => bad.checks[0].response["notes"] = json!("invented full-kernel proof"),
                4 => {
                    bad.checks[0].request["protocol"]["waits"] = json!([[], []]);
                    bad.checks[0].request_sha256 =
                        plow_asset::certificates::request_sha256(&bad.checks[0].request).unwrap();
                }
                5 => {
                    bad.checks[1].request["choices"][0]["key"] = json!("baseline");
                    bad.checks[1].request_sha256 =
                        plow_asset::certificates::request_sha256(&bad.checks[1].request).unwrap();
                }
                _ => {
                    bad.checks[2].request["bodies"][0]["rhs"][2][4][1] = json!("changed_out");
                    bad.checks[2].request_sha256 = plow_asset::certificates::request_sha256(&bad.checks[2].request).unwrap();
                }
            }
            write(&bad);
            assert!(
                check_packet(&packet_path, &raw, &blob).is_err(),
                "mutation {mutation}"
            );
        }
        let program = &mut model.progs[0];
        for entry in program.stream.iter_mut().chain(program.gq_stream.iter_mut()) {
            entry.wait_len = 0;
        }
        let altered = model.to_blob();
        receipts.packet_sha256 = image_sha256(&altered);
        write(&receipts);
        assert!(check_packet(&packet_path, &altered, &DevBlob::parse(&altered).unwrap()).is_err(),
            "updating the packet hash cannot reuse a receipt for removed wire waits");
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    #[ignore = "requires built plow_verify; CPU-only"]
    fn layout_receipt_is_reconstructed_from_loaded_packet() {
        let mut builder = Builder::new(1);
        let padded = builder.tensor("padded", 16 * 8192 * 2);
        let output = builder.tensor("output", 16 * 8 * 256 * 2);
        let producer = builder.emit(DevOp::FlashMerge, vec![0], &[], |d| {
            d.t[0] = padded;
            d.i[..5].copy_from_slice(&[16, 8, 1, 512, 1024]);
        });
        builder.emit(DevOp::MlaBmmFp8, vec![0], &[producer], |d| {
            d.t[..2].copy_from_slice(&[output, padded]);
            d.i[..6].copy_from_slice(&[16, 8, 256, 512, 0, 1024]);
        });
        let program = builder.finish();
        let mut model = Model {
            n_cu: 1,
            target: 0,
            tensors: program.tensors.clone(),
            progs: vec![program],
            kv_row_insts: vec![],
            prog_t: vec![16],
            gen: vec![],
        };
        let raw = model.to_blob();
        let blob = DevBlob::parse(&raw).unwrap();
        let request = plow_asset::certificates::mla_layout_obligation(
            0,
            &blob.progs[0].insts[0],
            &blob.progs[0].insts[1],
            blob.tensors[padded as usize].bytes,
        )
        .unwrap();
        let response = lean_verify::call("L", request.clone()).unwrap();
        assert!(response.ok);
        let receipts = PacketCheckReceipts {
            schema: 1,
            packet_sha256: image_sha256(&raw),
            compiler_sha256: "a".repeat(64),
            checks: vec![CompileCheckReceipt {
                program: Some(0),
                scope: SemanticScope::LayoutMapping,
                checkpoint: "L".into(),
                request_sha256: plow_asset::certificates::request_sha256(&request).unwrap(),
                verifier_sha256: lean_verify::verifier_sha256().unwrap(),
                request,
                response: serde_json::to_value(response).unwrap(),
            }],
        };
        let directory =
            std::env::temp_dir().join(format!("plow-layout-receipts-{}", std::process::id()));
        std::fs::create_dir_all(&directory).unwrap();
        let packet_path = directory.join("model.pkt");
        let sidecar = directory.join(PACKET_CHECKS_FILE);
        std::fs::write(&sidecar, serde_json::to_vec(&receipts).unwrap()).unwrap();
        check_packet(&packet_path, &raw, &blob).unwrap();
        for mutation in 0..4 {
            model.progs[0].insts[1].i[5] = 1024;
            model.progs[0].insts[0].i[4] = 1024;
            model.tensors[padded as usize].bytes = 16 * 8192 * 2;
            model.progs[0].insts[1].t[1] = padded;
            match mutation {
                0 => model.progs[0].insts[1].i[5] = 0,
                1 => model.progs[0].insts[0].i[4] = 0,
                2 => model.tensors[padded as usize].bytes /= 2,
                _ => model.progs[0].insts[1].t[1] = output,
            }
            let raw = model.to_blob();
            let blob = DevBlob::parse(&raw).unwrap();
            let mut changed_receipts = receipts.clone();
            changed_receipts.packet_sha256 = image_sha256(&raw);
            std::fs::write(&sidecar, serde_json::to_vec(&changed_receipts).unwrap()).unwrap();
            assert!(
                check_packet(&packet_path, &raw, &blob).is_err(),
                "mutation {mutation}"
            );
        }
        std::fs::remove_dir_all(directory).unwrap();
    }
}
