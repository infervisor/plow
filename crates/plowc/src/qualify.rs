//! `plowc qualify` and the emit-time qualification: derives every required Lean obligation from
//! the packets, matches the receipts, checks verifier approval and replays them. This is not
//! execution-identity, floating-point implementation, runtime-rewrite, or empirical qualification.

use std::collections::BTreeSet;
use std::path::Path;
use std::sync::Mutex;

use plow_asset::certificates::{
    is_approved_verifier, qualify_packet, qualify_sidecar, sidecar_checks_file,
    CompileCheckReceipt, PacketCheckReceipts, Qualification, SemanticScope, SidecarCheckReceipts,
    PACKET_CHECKS_FILE,
};
use plow_asset::decode_objects::image_sha256;
use plowrt::asset::devblob::DevBlob;
use plowrt::asset::packet_image::PacketImage;

type Result<T> = std::result::Result<T, String>;

/// A packet as the checks read it. The digest is computed at most once, and not at all when no
/// receipt needs it.
struct Pkt<'a> {
    path: &'a Path,
    raw: &'a [u8],
    blob: &'a DevBlob,
    digest: std::sync::OnceLock<String>,
}

impl<'a> Pkt<'a> {
    fn of(image: &'a PacketImage) -> Self {
        Pkt {
            path: &image.path,
            raw: &image.raw,
            blob: &image.blob,
            digest: std::sync::OnceLock::new(),
        }
    }

    fn sha256(&self) -> String {
        self.digest.get_or_init(|| image_sha256(self.raw)).clone()
    }
}

fn read_optional(path: &Path) -> Result<Option<Vec<u8>>> {
    match std::fs::read(path) {
        Ok(bytes) => Ok(Some(bytes)),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(format!("{}: {error}", path.display())),
    }
}

/// `model.pkt`, a language-model packet under any name (`DevBlob::find_in_dir` loads one; it is
/// the only packet with `in.ids`), or any packet whose `lean-checks.json` binds its bytes,
/// carries compiler receipts; every other packet (`encoder.pkt`, `codec.pkt`, ...) is a sidecar
/// with its own file.
fn is_compiler_packet(pkt: &Pkt, receipts: Option<&[u8]>) -> bool {
    pkt.path.file_stem().is_some_and(|stem| stem == "model")
        || pkt.blob.tensors.iter().any(|t| t.name == "in.ids")
        || receipts
            .and_then(|bytes| serde_json::from_slice::<PacketCheckReceipts>(bytes).ok())
            .is_some_and(|r| r.packet_sha256 == pkt.sha256())
}

fn qualify_compiler(
    receipts: Option<&PacketCheckReceipts>,
    bytes: Option<&[u8]>,
    pkt: &Pkt,
    replay_checks: bool,
) -> Qualification {
    let sha = pkt.sha256();
    let mut q = pkt
        .blob
        .with_packet_view(|packet| qualify_packet(receipts, &sha, packet));
    if let (true, Some(receipts), Some(bytes)) = (replay_checks, receipts, bytes) {
        if let Err(gap) = replay(&receipts.checks, bytes, &q.scope_set_sha256()) {
            q.gaps.push(gap);
        }
    }
    q
}

fn qualify_sidecar_blob(
    receipts: Option<&SidecarCheckReceipts>,
    pkt: &Pkt,
    replay_checks: bool,
) -> Qualification {
    let sha = pkt.sha256();
    let mut q = pkt
        .blob
        .with_packet_view(|packet| qualify_sidecar(receipts, &sha, packet));
    if let (true, Some(receipts)) = (replay_checks, receipts) {
        let key = serde_json::to_vec(receipts).unwrap_or_default();
        if let Err(gap) = replay(&receipts.checks, &key, &q.scope_set_sha256()) {
            q.gaps.push(gap);
        }
    }
    q
}

/// Re-run every receipt's request on the current verifier, which must be approved, and require
/// the identical accepted envelope. Cached per (receipts, verifier, required scope set).
fn replay(
    checks: &[CompileCheckReceipt],
    key: &[u8],
    scope_set: &str,
) -> std::result::Result<(), String> {
    replay_approved(checks, key, scope_set, is_approved_verifier)
}

fn replay_approved(
    checks: &[CompileCheckReceipt],
    key: &[u8],
    scope_set: &str,
    approved: impl Fn(&str) -> bool,
) -> std::result::Result<(), String> {
    if checks.is_empty() {
        return Ok(());
    }
    let verifier =
        lean_verify::verifier_sha256().map_err(|e| format!("no verifier identity: {e}"))?;
    if !approved(&verifier) {
        return Err(format!("current verifier {verifier} is not approved"));
    }
    static CHECKED: Mutex<BTreeSet<(String, String, String)>> = Mutex::new(BTreeSet::new());
    let cache = (image_sha256(key), verifier.clone(), scope_set.to_string());
    if CHECKED.lock().unwrap().contains(&cache) {
        return Ok(());
    }
    let requests: Vec<_> = checks
        .iter()
        .map(|c| (c.checkpoint.as_str(), c.request.clone()))
        .collect();
    let (certs, executed) =
        lean_verify::call_batch_bound(&requests).map_err(|e| format!("verifier: {e}"))?;
    if executed != verifier {
        return Err("verifier changed while replaying obligations".into());
    }
    for (receipt, cert) in checks.iter().zip(certs) {
        let envelope = serde_json::to_value(&cert).map_err(|e| e.to_string())?;
        if !cert.ok || envelope != receipt.response {
            return Err(format!(
                "program {:?} {:?}: obligation rejected or envelope changed: {:?}",
                receipt.program, receipt.scope, cert.reason
            ));
        }
    }
    CHECKED.lock().unwrap().insert(cache);
    Ok(())
}

/// `plowc qualify`: every `*.pkt` in `dir`, in name order, with its qualification. A packet
/// with a speech or multimodal pipeline also owes the bundle's `media_geometry.v1` obligation, and one
/// with packed prefill over a sliding ring owes `kv_ring.v1`, and one with fused speech sites
/// owes `speech_fusion.v1`.
pub fn qualify_dir(
    dir: &Path,
    replay_checks: bool,
) -> Result<Vec<(std::path::PathBuf, Qualification)>> {
    let entries = std::fs::read_dir(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let mut packets: Vec<_> = entries
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "pkt"))
        .collect();
    packets.sort();
    if packets.is_empty() {
        return Err(format!("{}: no packets", dir.display()));
    }
    let mut out = Vec::new();
    let mut media = Vec::new();
    for path in packets {
        let image = PacketImage::open(&path).map_err(|e| e.to_string())?;
        image
            .blob
            .check_l2_dispatch(false)
            .map_err(|e| e.to_string())?;
        let (raw, blob) = (&image.raw, &image.blob);
        let mut q = qualify_loaded(&Pkt::of(&image), replay_checks)?;
        kv_ring_obligation(raw, blob, replay_checks, &mut q);
        speech_fusion_obligation(blob, replay_checks, &mut q);
        media.push(MediaInfo::read(&path, raw, blob));
        out.push((path, q));
    }
    let packets: Vec<_> = media
        .iter()
        .filter_map(|m| m.as_ref().ok().and_then(Option::as_ref))
        .map(MediaInfo::packet)
        .collect();
    for (index, (path, q)) in out.iter_mut().enumerate() {
        let main = match &media[index] {
            Ok(Some(main)) => main,
            Ok(None) => continue,
            Err(error) => {
                q.required.insert(media_obligation());
                q.gaps
                    .push(format!("media geometry: {}: {error}", path.display()));
                continue;
            }
        };
        let request = match plow_asset::media_geometry::request(&main.packet(), &packets) {
            Ok(Some(request)) => request,
            Ok(None) => continue,
            Err(error) => {
                q.required.insert(media_obligation());
                q.gaps.push(format!("media geometry: {error}"));
                continue;
            }
        };
        q.required.insert(media_obligation());
        match check_endpoint(plow_asset::media_geometry::ENDPOINT, request, replay_checks) {
            Ok(()) => {
                q.satisfied.insert(media_obligation());
            }
            Err(gap) => q.gaps.push(format!("media geometry: {gap}")),
        }
    }
    Ok(out)
}

/// [`qualify_dir`] over a bundle directory, or over a packet file's directory keeping only that
/// packet (the media obligation still reads its siblings).
pub fn qualify_path(
    path: &Path,
    replay_checks: bool,
) -> Result<Vec<(std::path::PathBuf, Qualification)>> {
    if path.is_dir() {
        return qualify_dir(path, replay_checks);
    }
    let dir = path
        .parent()
        .filter(|d| !d.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let name = path.file_name();
    let found: Vec<_> = qualify_dir(dir, replay_checks)?
        .into_iter()
        .filter(|(p, _)| p.file_name() == name)
        .collect();
    if found.is_empty() {
        return Err(format!("{}: not a packet", path.display()));
    }
    Ok(found)
}

fn media_obligation() -> plow_asset::certificates::Obligation {
    plow_asset::certificates::Obligation {
        scope: plow_asset::certificates::SemanticScope::MediaGeometry,
        checkpoint: plow_asset::media_geometry::ENDPOINT,
        program: None,
        site: None,
    }
}

/// `kv_ring.v1` for a packet with a packed prefill section and a sliding ring.
fn kv_ring_obligation(raw: &[u8], blob: &DevBlob, replay_checks: bool, q: &mut Qualification) {
    let request = (|| -> std::result::Result<Option<serde_json::Value>, String> {
        let Some(pf) = blob
            .reserved_metadata(raw, plow_asset::packed_prefill::SECTION)
            .map_err(|e| e.to_string())?
        else {
            return Ok(None);
        };
        let pf: plow_asset::packed_prefill::Manifest =
            serde_json::from_slice(pf).map_err(|e| format!("packed prefill metadata: {e}"))?;
        let live = plowrt::memory::vmm::LiveKvLayout::manifest(blob, raw)
            .map_err(|e| e.to_string())?
            .ok_or("packed prefill without a live KV manifest")?;
        blob.with_packet_view(|p| {
            pf.validate(p, &live)?;
            let rungs: Vec<u32> = p.programs[..p.prefill_count]
                .iter()
                .map(|g| g.rows)
                .collect();
            plow_asset::kv_ring::request(&pf, &live, &rungs)
        })
    })();
    packet_obligation(
        SemanticScope::KvRing,
        plow_asset::kv_ring::ENDPOINT,
        "kv ring",
        request,
        replay_checks,
        q,
    );
}

/// `speech_fusion.v1` for a packet with a fused speech site.
fn speech_fusion_obligation(blob: &DevBlob, replay_checks: bool, q: &mut Qualification) {
    let request = blob.with_packet_view(plow_asset::speech_fusion::request);
    packet_obligation(
        SemanticScope::SpeechFusion,
        plow_asset::speech_fusion::ENDPOINT,
        "speech fusion",
        request,
        replay_checks,
        q,
    );
}

/// Adds a packet-level qualification obligation: none for `Ok(None)`, a gap for a derivation
/// error or a rejection.
fn packet_obligation(
    scope: SemanticScope,
    endpoint: &'static str,
    what: &str,
    request: std::result::Result<Option<serde_json::Value>, String>,
    replay_checks: bool,
    q: &mut Qualification,
) {
    let obligation = plow_asset::certificates::Obligation {
        scope,
        checkpoint: endpoint,
        program: None,
        site: None,
    };
    let request = match request {
        Ok(None) => return,
        Ok(Some(request)) => request,
        Err(error) => {
            q.required.insert(obligation);
            q.gaps.push(format!("{what}: {error}"));
            return;
        }
    };
    q.required.insert(obligation.clone());
    match check_endpoint(endpoint, request, replay_checks) {
        Ok(()) => {
            q.satisfied.insert(obligation);
        }
        Err(gap) => q.gaps.push(format!("{what}: {gap}")),
    }
}

fn check_endpoint(
    endpoint: &str,
    request: serde_json::Value,
    replay_checks: bool,
) -> std::result::Result<(), String> {
    if !replay_checks {
        return Err("not checked (--no-replay)".into());
    }
    let verifier =
        lean_verify::verifier_sha256().map_err(|e| format!("no verifier identity: {e}"))?;
    if !is_approved_verifier(&verifier) {
        return Err(format!("current verifier {verifier} is not approved"));
    }
    let (certs, executed) = lean_verify::call_batch_bound(&[(endpoint, request)])
        .map_err(|e| format!("verifier: {e}"))?;
    let cert = certs
        .into_iter()
        .next()
        .ok_or("verifier returned no certificate")?;
    if executed != verifier {
        return Err("verifier changed while checking".into());
    }
    if !cert.ok {
        return Err(cert.reason.unwrap_or_default());
    }
    Ok(())
}

/// A packet's pipeline section, vocabulary size and multimodal contract, for the media obligation.
struct MediaInfo {
    file: String,
    pipelines: Vec<plow_asset::packet_pipeline::PacketPipeline>,
    vocabulary: Option<usize>,
    multimodal: Option<plow_asset::multimodal::MmContract>,
    mm_tensors: Option<(u64, u64)>,
}

impl MediaInfo {
    /// `Ok(None)` for a packet without a pipeline section or multimodal contract.
    fn read(path: &Path, raw: &[u8], blob: &DevBlob) -> std::result::Result<Option<Self>, String> {
        use plow_asset::multimodal::{MmContract, SECTION as MM, SLAB_TENSOR, TABLE_TENSOR};
        let file = path
            .file_name()
            .and_then(|f| f.to_str())
            .ok_or("packet file name")?
            .to_owned();
        let has_pipelines = blob
            .reserved_metadata(raw, plow_asset::packet_pipeline::SECTION)
            .map_err(|e| e.to_string())?
            .is_some();
        let multimodal = blob
            .reserved_metadata(raw, MM)
            .map_err(|e| e.to_string())?
            .map(|bytes| {
                serde_json::from_slice::<MmContract>(bytes).map_err(|e| format!("{MM}: {e}"))
            })
            .transpose()?;
        if let Some(mm) = &multimodal {
            mm.validate()?;
        }
        if !has_pipelines && multimodal.is_none() {
            return Ok(None);
        }
        let (pipelines, vocabulary) = if has_pipelines {
            let asset = plowrt::exec::packet_runtime::PacketAsset::from_bytes(raw)
                .map_err(|e| e.to_string())?;
            let vocabulary = asset
                .metadata(plow_asset::speech_contract::VOCABULARY_SECTION)
                .map(|bytes| {
                    serde_json::from_slice::<plow_asset::speech_contract::Vocabulary>(bytes)
                        .map(|v| v.pieces.len())
                        .map_err(|e| e.to_string())
                })
                .transpose()?;
            (asset.pipelines().to_vec(), vocabulary)
        } else {
            (Vec::new(), None)
        };
        let bytes = |name: &str| {
            blob.tensors
                .iter()
                .find(|t| t.name == name)
                .map(|t| t.bytes)
        };
        let mm_tensors = bytes(SLAB_TENSOR).zip(bytes(TABLE_TENSOR));
        Ok(Some(Self {
            file,
            pipelines,
            vocabulary,
            multimodal,
            mm_tensors,
        }))
    }

    fn packet(&self) -> plow_asset::media_geometry::MediaPacket<'_> {
        plow_asset::media_geometry::MediaPacket {
            file: &self.file,
            pipelines: &self.pipelines,
            vocabulary: self.vocabulary,
            multimodal: self.multimodal.as_ref(),
            mm_tensors: self.mm_tensors,
        }
    }
}

/// One loaded packet's receipt qualification.
fn qualify_loaded(pkt: &Pkt, replay_checks: bool) -> Result<Qualification> {
    let blob_path = pkt.path;
    let path = blob_path.with_file_name(PACKET_CHECKS_FILE);
    let bytes = read_optional(&path)?;
    if is_compiler_packet(pkt, bytes.as_deref()) {
        let receipts = bytes
            .as_deref()
            .map(serde_json::from_slice::<PacketCheckReceipts>)
            .transpose()
            .map_err(|e| format!("{}: {e}", path.display()))?;
        return Ok(qualify_compiler(
            receipts.as_ref(),
            bytes.as_deref(),
            pkt,
            replay_checks,
        ));
    }
    let path = sidecar_checks_file(blob_path);
    let receipts = read_optional(&path)?
        .map(|bytes| serde_json::from_slice::<SidecarCheckReceipts>(&bytes))
        .transpose()
        .map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(qualify_sidecar_blob(receipts.as_ref(), pkt, replay_checks))
}

#[cfg(test)]
mod tests {
    use super::*;
    use packet::dev::DevOp;
    use packet::devbuild::{Builder, Model};
    use plow_asset::certificates::{CompileCheckReceipt, SemanticScope};
    use serde_json::json;

    fn qualification(path: &Path, raw: &[u8]) -> Qualification {
        std::fs::write(path, raw).unwrap();
        qualify_path(path, true).unwrap().remove(0).1
    }

    /// Receipts present bind the packet and replay; only absent obligations are gaps.
    fn bound(path: &Path, raw: &[u8]) -> std::result::Result<(), String> {
        let q = qualification(path, raw);
        let unbound: Vec<_> = q
            .gaps
            .iter()
            .filter(|g| {
                !g.contains(" missing for ")
                    && !matches!(
                        g.as_str(),
                        "no compiler check receipts" | "no sidecar check receipts"
                    )
            })
            .collect();
        match unbound.is_empty() {
            true => Ok(()),
            false => Err(format!("{unbound:?}")),
        }
    }

    fn strict(path: &Path, raw: &[u8]) -> std::result::Result<(), String> {
        let q = qualification(path, raw);
        match q.qualified() {
            true => Ok(()),
            false => Err(q.gaps.join("; ")),
        }
    }

    #[test]
    #[ignore = "requires built plow_verify listed in lean-plow/approved-verifiers.json; CPU-only"]
    fn selected_gemm_receipt_reconstructs_wire_after_packet_hash_changes() {
        let mut b = Builder::new(2);
        let out = b.tensor("output", 128 * 256 * 2);
        let input = b.tensor("input", 128 * 512 * 2);
        let weight = b.tensor("weight", 256 * 512 * 2);
        b.emit(DevOp::GemmMed, vec![0, 1], &[], |d| {
            d.t[..3].copy_from_slice(&[out, input, weight]);
            d.i[..3].copy_from_slice(&[128, 256, 512]);
        });
        let p = b.finish();
        let mut decode = Builder::new(2);
        decode.emit(DevOp::Nop, vec![0], &[], |_| {});
        let mut model = Model {
            n_cu: 2,
            target: 0,
            tensors: p.tensors.clone(),
            progs: vec![p, decode.finish()],
            prog_t: vec![128, 1],
            gen: vec![],
            kv_row_insts: vec![],
        };
        let geometry = json!({"m":128,"n":256,"k":512,"n_cu":2,"quant":"None"});
        let domain = image_sha256(&serde_json::to_vec(&geometry).unwrap());
        let key = format!("{}:implementation", DevOp::GemmMed as u16);
        let request = json!({"policy_kind":plow_asset::gemm_policy::KIND,"geometry":geometry,
            "selected_opcode":DevOp::GemmMed as u16,"required":[domain],
            "candidates":[{"domain":domain,"key":key,"cost":1,"qualified":true}],
            "choices":[{"domain":domain,"key":key}]});
        let request = plow_asset::program::with_model(&model, |packet| {
            plow_asset::gemm_policy::bind(packet, 0, &request).unwrap()
        });
        let (mut certs, verifier) =
            lean_verify::call_batch_bound(&[("R", request.clone())]).unwrap();
        let cert = certs.remove(0);
        assert!(cert.ok);
        let raw = model.to_blob();
        let mut receipts = PacketCheckReceipts {
            schema: 1,
            packet_sha256: image_sha256(&raw),
            compiler_sha256: "a".repeat(64),
            checks: vec![CompileCheckReceipt {
                program: Some(0),
                scope: SemanticScope::SelectedGemmPolicy,
                checkpoint: "R".into(),
                request_sha256: plow_asset::certificates::request_sha256(&request).unwrap(),
                verifier_sha256: verifier,
                request,
                response: serde_json::to_value(cert).unwrap(),
            }],
        };
        let directory =
            std::env::temp_dir().join(format!("plow-gemm-receipts-{}", std::process::id()));
        std::fs::create_dir_all(&directory).unwrap();
        let path = directory.join("model.pkt");
        let sidecar = directory.join(PACKET_CHECKS_FILE);
        let write = |value: &PacketCheckReceipts| {
            std::fs::write(&sidecar, serde_json::to_vec(value).unwrap()).unwrap()
        };
        write(&receipts);
        bound(&path, &raw).unwrap();
        bound(&path, &raw).unwrap();
        for change in 0..5 {
            let original = model.progs[0].insts[0].clone();
            match change {
                0 => model.progs[0].insts[0].op = DevOp::GemmSmall as u16,
                1 => model.progs[0].insts[0].i[1] += 1,
                2 => model.progs[0].insts[0].t[7] = out,
                3 => model.progs[0].insts[0].f[0] = 0.125,
                _ => model.progs[0].insts[0].t.swap(1, 2),
            }
            let raw = model.to_blob();
            receipts.packet_sha256 = image_sha256(&raw);
            write(&receipts);
            assert!(bound(&path, &raw).is_err(), "wire mutation {change}");
            model.progs[0].insts[0] = original;
        }
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    #[ignore = "requires built plow_verify listed in lean-plow/approved-verifiers.json; CPU-only"]
    fn logical_effect_receipt_is_reconstructed_from_loaded_packet() {
        let mut b = Builder::new(1);
        let x = b.tensor("x", 16);
        let y = b.tensor("y", 16);
        let first = b.emit(DevOp::Residual, vec![0], &[], |d| {
            d.t[..3].copy_from_slice(&[y, x, x]);
            d.i[0] = 8;
        });
        b.emit(DevOp::Residual, vec![0], &[first], |d| {
            d.t[..3].copy_from_slice(&[x, y, y]);
            d.i[0] = 8;
        });
        let p = b.finish();
        let mut model = Model {
            n_cu: 1,
            target: 0,
            tensors: p.tensors.clone(),
            progs: vec![p],
            kv_row_insts: vec![],
            prog_t: vec![1],
            gen: vec![],
        };
        let request = plow_asset::program::with_model(&model, |packet| {
            plow_asset::logical_effects::obligation(packet, 0).unwrap()
        });
        let (mut certs, verifier) =
            lean_verify::call_batch_bound(&[("D", request.clone())]).unwrap();
        let cert = certs.remove(0);
        assert!(cert.ok, "{:?}", cert.reason);
        let raw = model.to_blob();
        let mut receipts = PacketCheckReceipts {
            schema: 1,
            packet_sha256: image_sha256(&raw),
            compiler_sha256: "a".repeat(64),
            checks: vec![CompileCheckReceipt {
                program: Some(0),
                scope: SemanticScope::LogicalTensorEffects,
                checkpoint: "D".into(),
                request_sha256: plow_asset::certificates::request_sha256(&request).unwrap(),
                verifier_sha256: verifier,
                request,
                response: serde_json::to_value(cert).unwrap(),
            }],
        };
        let directory =
            std::env::temp_dir().join(format!("plow-effects-receipts-{}", std::process::id()));
        std::fs::create_dir_all(&directory).unwrap();
        let packet_path = directory.join("model.pkt");
        let sidecar = directory.join(PACKET_CHECKS_FILE);
        std::fs::write(&sidecar, serde_json::to_vec(&receipts).unwrap()).unwrap();
        bound(&packet_path, &raw).unwrap();
        for mutation in 0..3 {
            match mutation {
                0 => model.progs[0].insts[1].t[0] = y,
                1 => model.tensors[x as usize].bytes += 16,
                _ => model.progs[0].waits[0].threshold = 0,
            }
            let raw = model.to_blob();
            receipts.packet_sha256 = image_sha256(&raw);
            std::fs::write(&sidecar, serde_json::to_vec(&receipts).unwrap()).unwrap();
            assert!(
                bound(&packet_path, &raw).is_err(),
                "mutation {mutation}: even a matching new packet hash cannot reuse effects"
            );
        }
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    #[ignore = "requires built plow_verify listed in lean-plow/approved-verifiers.json; CPU-only"]
    fn packet_receipts_replay_and_reject_tampering() {
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
        let var = |name: &str| json!(["variable", name, []]);
        let node = |head: &str, args: Vec<serde_json::Value>| json!(["call", head, args]);
        let lhs = node(
            "Linear",
            vec![
                node("RmsNorm", vec![var("x"), var("w"), var("eps")]),
                var("wl"),
                var("out"),
            ],
        );
        let rhs = node(
            "FusedNormLinear",
            vec![var("x"), var("w"), var("wl"), var("eps"), var("out")],
        );
        let rewrite = json!({"rules":["rmsnorm-linear-fuse"],"source_sha256":"c".repeat(64),
            "bodies":[{"name":"rmsnorm-linear-fuse","lhs":lhs,"rhs":rhs}]});
        let cert = lean_verify::call("A", rewrite.clone()).unwrap();
        assert!(cert.ok);
        receipts.checks.push(CompileCheckReceipt {
            program: None,
            scope: SemanticScope::RewriteBodyExpansion,
            checkpoint: "A".into(),
            request_sha256: plow_asset::certificates::request_sha256(&rewrite).unwrap(),
            verifier_sha256: lean_verify::verifier_sha256().unwrap(),
            request: rewrite,
            response: serde_json::to_value(cert).unwrap(),
        });
        let directory = std::env::temp_dir().join(format!("plow-receipts-{}", std::process::id()));
        std::fs::create_dir_all(&directory).unwrap();
        let packet_path = directory.join("model.pkt");
        let sidecar = directory.join(PACKET_CHECKS_FILE);
        let write = |value: &PacketCheckReceipts| {
            std::fs::write(&sidecar, serde_json::to_vec(value).unwrap()).unwrap()
        };
        // Missing receipts are gaps, not binding failures.
        bound(&packet_path, &raw).unwrap();
        write(&receipts);
        bound(&packet_path, &raw).unwrap();
        bound(&packet_path, &raw).unwrap();
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
                    bad.checks[2].request_sha256 =
                        plow_asset::certificates::request_sha256(&bad.checks[2].request).unwrap();
                }
            }
            write(&bad);
            assert!(bound(&packet_path, &raw).is_err(), "mutation {mutation}");
        }
        let program = &mut model.progs[0];
        for entry in program
            .stream
            .iter_mut()
            .chain(program.gq_stream.iter_mut())
        {
            entry.wait_len = 0;
        }
        let altered = model.to_blob();
        receipts.packet_sha256 = image_sha256(&altered);
        write(&receipts);
        assert!(
            bound(&packet_path, &altered).is_err(),
            "updating the packet hash cannot reuse a receipt for removed wire waits"
        );
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    #[ignore = "requires built plow_verify listed in lean-plow/approved-verifiers.json; CPU-only"]
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
        bound(&packet_path, &raw).unwrap();
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
            let mut changed_receipts = receipts.clone();
            changed_receipts.packet_sha256 = image_sha256(&raw);
            std::fs::write(&sidecar, serde_json::to_vec(&changed_receipts).unwrap()).unwrap();
            assert!(bound(&packet_path, &raw).is_err(), "mutation {mutation}");
        }
        std::fs::remove_dir_all(directory).unwrap();
    }

    fn residual_model(extra: bool) -> Model {
        let mut b = Builder::new(1);
        let x = b.tensor("x", 16);
        let y = b.tensor("y", 16);
        let first = b.emit(DevOp::Residual, vec![0], &[], |d| {
            d.t[..3].copy_from_slice(&[y, x, x]);
            d.i[0] = 8;
        });
        let second = b.emit(DevOp::Residual, vec![0], &[first], |d| {
            d.t[..3].copy_from_slice(&[x, y, y]);
            d.i[0] = 8;
        });
        if extra {
            b.emit(DevOp::Residual, vec![0], &[second], |d| {
                d.t[..3].copy_from_slice(&[y, x, x]);
                d.i[0] = 8;
            });
        }
        let p = b.finish();
        Model {
            n_cu: 1,
            target: 0,
            tensors: p.tensors.clone(),
            progs: vec![p],
            kv_row_insts: vec![],
            prog_t: vec![1],
            gen: vec![],
        }
    }

    fn scratch(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("plow-{name}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn absent_receipts_are_gaps_and_fail_only_strict() {
        let raw = residual_model(false).to_blob();
        let dir = scratch("strict-absent");
        let packet = dir.join("model.pkt");
        bound(&packet, &raw).unwrap();
        bound(&packet, &raw).unwrap();
        let error = strict(&packet, &raw).unwrap_err();
        assert!(
            error.to_string().contains("no compiler check receipts"),
            "{error}"
        );
        let sidecar = dir.join("codec.pkt");
        bound(&sidecar, &raw).unwrap();
        bound(&sidecar, &raw).unwrap();
        assert!(strict(&sidecar, &raw).is_err());
        std::fs::remove_dir_all(dir).unwrap();
    }

    /// `encoder.pkt` beside `model.pkt` must read `encoder.lean-checks.json`, never the compiler
    /// packet's `lean-checks.json` (whose hash binding would reject the encoder).
    #[test]
    fn a_sidecar_beside_compiler_receipts_reads_its_own_receipts() {
        let model = residual_model(false).to_blob();
        let encoder = residual_model(true).to_blob();
        let dir = scratch("sidecar-file");
        let receipts = PacketCheckReceipts {
            schema: 1,
            packet_sha256: image_sha256(&model),
            compiler_sha256: "a".repeat(64),
            checks: vec![],
        };
        std::fs::write(
            dir.join(PACKET_CHECKS_FILE),
            serde_json::to_vec(&receipts).unwrap(),
        )
        .unwrap();
        let path = dir.join("encoder.pkt");
        bound(&path, &encoder).unwrap();
        let error = strict(&path, &encoder).unwrap_err();
        assert!(
            error.to_string().contains("no sidecar check receipts"),
            "{error}"
        );
        bound(&dir.join("model.pkt"), &model).unwrap();
        std::fs::remove_dir_all(dir).unwrap();
    }

    /// A language-model packet under another name (`find_in_dir` loads it) is still the compiler
    /// packet: receipts that do not bind its bytes reject it instead of demoting it to a sidecar.
    #[test]
    fn a_renamed_language_model_packet_keeps_its_receipt_binding() {
        let mut model = residual_model(false);
        model.tensors.push(packet::devbuild::TensorDecl {
            name: "in.ids".into(),
            bytes: 16,
            init: None,
        });
        let raw = model.to_blob();
        let dir = scratch("renamed-main");
        let receipts = PacketCheckReceipts {
            schema: 1,
            packet_sha256: "b".repeat(64),
            compiler_sha256: "a".repeat(64),
            checks: vec![],
        };
        std::fs::write(
            dir.join(PACKET_CHECKS_FILE),
            serde_json::to_vec(&receipts).unwrap(),
        )
        .unwrap();
        let path = dir.join("gemma.pkt");
        let error = bound(&path, &raw).unwrap_err();
        assert!(
            error.to_string().contains("does not match loaded packet"),
            "{error}"
        );
        assert!(strict(&path, &raw).is_err());
        std::fs::remove_dir_all(dir).unwrap();
    }

    fn lean_receipts(raw: &[u8], model: &Model) -> PacketCheckReceipts {
        let var = |name: &str| json!(["variable", name, []]);
        let node = |head: &str, args: Vec<serde_json::Value>| json!(["call", head, args]);
        let lhs = node(
            "Linear",
            vec![
                node("RmsNorm", vec![var("x"), var("w"), var("eps")]),
                var("wl"),
                var("out"),
            ],
        );
        let rhs = node(
            "FusedNormLinear",
            vec![var("x"), var("w"), var("wl"), var("eps"), var("out")],
        );
        let mut requests = vec![(
            None,
            SemanticScope::RewriteBodyExpansion,
            "A",
            json!({"rules": ["rmsnorm-linear-fuse"], "source_sha256": "c".repeat(64),
                "bodies": [{"name": "rmsnorm-linear-fuse", "lhs": lhs, "rhs": rhs}]}),
        )];
        plow_asset::program::with_model(model, |packet| {
            for (p, program) in packet.programs.iter().enumerate() {
                requests.push((
                    Some(p),
                    SemanticScope::CoarseDependencyPreservation,
                    "D",
                    json!({"task_graph": {"n": program.insts.len(), "edges": []},
                        "protocol": plow_asset::logical_effects::coarse_protocol(program).unwrap(),
                        "dependency_paths": [], "address_map": []}),
                ));
                requests.push((
                    Some(p),
                    SemanticScope::LogicalTensorEffects,
                    "D",
                    plow_asset::logical_effects::obligation(packet, p).unwrap(),
                ));
            }
        });
        let batch: Vec<_> = requests
            .iter()
            .map(|(_, _, cp, r)| (*cp, r.clone()))
            .collect();
        let (certs, verifier) = lean_verify::call_batch_bound(&batch).unwrap();
        let checks = requests
            .into_iter()
            .zip(certs)
            .map(|((program, scope, cp, request), cert)| {
                assert!(cert.ok, "{scope:?}: {:?}", cert.reason);
                CompileCheckReceipt {
                    program,
                    scope,
                    checkpoint: cp.into(),
                    request_sha256: plow_asset::certificates::request_sha256(&request).unwrap(),
                    verifier_sha256: verifier.clone(),
                    request,
                    response: serde_json::to_value(cert).unwrap(),
                }
            })
            .collect();
        PacketCheckReceipts {
            schema: 1,
            packet_sha256: image_sha256(raw),
            compiler_sha256: "a".repeat(64),
            checks,
        }
    }

    #[test]
    #[ignore = "requires built plow_verify listed in lean-plow/approved-verifiers.json; CPU-only"]
    fn complete_receipts_qualify_and_substituted_evidence_does_not() {
        let model = residual_model(false);
        let raw = model.to_blob();
        let receipts = lean_receipts(&raw, &model);
        let dir = scratch("strict-replay");
        let packet = dir.join("model.pkt");
        let write = |r: &PacketCheckReceipts| {
            std::fs::write(dir.join(PACKET_CHECKS_FILE), serde_json::to_vec(r).unwrap()).unwrap()
        };
        write(&receipts);
        strict(&packet, &raw).unwrap();
        let bytes = serde_json::to_vec(&receipts).unwrap();
        let denied =
            replay_approved(&receipts.checks, &bytes, "fresh-scope-set", |_| false).unwrap_err();
        assert!(denied.contains("not approved"), "{denied}");
        for mutation in 0..4 {
            let mut bad = receipts.clone();
            match mutation {
                0 => bad.checks[1].response["notes"] = json!("invented proof"),
                1 => {
                    let response = bad.checks[1].response.clone();
                    bad.checks[1].response = bad.checks[2].response.clone();
                    bad.checks[2].response = response;
                }
                2 => {
                    bad.checks.remove(2);
                }
                _ => bad.checks[0].verifier_sha256 = "b".repeat(64),
            }
            write(&bad);
            assert!(strict(&packet, &raw).is_err(), "mutation {mutation}");
        }
        std::fs::remove_dir_all(dir).unwrap();
    }

    fn empty_prefix(tensors: &[(&str, u64)]) -> (devgen::pipeline::PacketPrefix, Vec<u32>) {
        let mut builder = Builder::new(4);
        let handles: Vec<u32> = tensors
            .iter()
            .map(|&(name, bytes)| builder.tensor(name, bytes))
            .collect();
        let model = Model {
            n_cu: 4,
            target: 0,
            tensors: builder.tensors(),
            progs: Vec::new(),
            kv_row_insts: Vec::new(),
            prog_t: Vec::new(),
            gen: Vec::new(),
        };
        let prefix = devgen::pipeline::PacketPrefix {
            model,
            programs: Vec::new(),
            input: handles[0],
            output: handles[0],
            input_shape: vec![],
        };
        (prefix, handles)
    }

    /// `speech_fusion.v1` on the sites devgen actually emits: the fused Qwen audio LayerNorm
    /// prologue (two layers) and a row-scaled Conv1dF32 epilogue are accepted; mutating any bound
    /// precondition rejects, and a GEMM whose stats tensor has no earlier writer cannot be derived.
    #[test]
    #[ignore = "requires built plow_verify listed in lean-plow/approved-verifiers.json; CPU-only"]
    fn emitted_speech_fusion_sites_meet_the_lean_preconditions() {
        use devgen::asr::qwen::{append_audio_transformer_layers, AudioTransformerSpec};
        use devgen::pipeline::{Activation, Conv1dF32Stage, DenseSplit, PadMode, TensorRef};
        let (prefix, _) = empty_prefix(&[("positioned", 2 * 4 * 4)]);
        let ln = append_audio_transformer_layers(
            prefix,
            AudioTransformerSpec {
                rows: 2,
                width: 4,
                ffn_width: 8,
                head_width: 2,
                group_rows: 2,
                valid_rows: None,
                group_table: false,
                split: DenseSplit::Parallel,
                first_layer: 0,
                layers: 2,
                weight_prefix: "tower",
                activation_prefix: "act.audio",
                fuse_layer_norm: true,
            },
        )
        .unwrap();
        let (prefix, h) = empty_prefix(&[("x", 2 * 10 * 8 * 4), ("scale", 2 * 10 * 4)]);
        let mut p = prefix.program();
        p.conv1d_f32_row_scaled(
            h[0],
            &[],
            Conv1dF32Stage {
                input_split: false,
                output_split: false,
                output: TensorRef::Named("y"),
                weight: TensorRef::Named("w"),
                bias: Some(TensorRef::Named("b")),
                alpha: None,
                residual: Some(h[0]),
                lengths: None,
                batch: 2,
                in_rows: 10,
                in_channels: 8,
                out_channels: 8,
                kernel: 3,
                stride: 1,
                dilation_or_output_padding: 1,
                groups: 1,
                pad_before: 1,
                pad_after: 1,
                pad_mode: PadMode::Zero,
                input_activation: Activation::None,
                output_activation: Activation::None,
                slope: 0.0,
                weight_f16: false,
                split_bf16: false,
                weight_tap_major: false,
                wgmma: false,
                weight_split: false,
                dual_alpha: false,
            },
            h[1],
        )
        .unwrap();
        let conv = p.finish(10);
        let derive =
            |m: &Model| plow_asset::program::with_model(m, plow_asset::speech_fusion::request);
        let ln_req = derive(&ln.model).unwrap().expect("fused LayerNorm sites");
        let conv_req = derive(&conv.model).unwrap().expect("row-scaled conv site");
        let ln_sites = ln_req["sites"].as_array().unwrap().len();
        assert!(
            ln_sites >= 4,
            "two layers, attention and FFN prologues: {ln_sites}"
        );
        assert_eq!(conv_req["sites"].as_array().unwrap().len(), 1);

        let mutate = |req: &serde_json::Value, field: &str, value: serde_json::Value| {
            let mut req = req.clone();
            req["sites"][0][field] = value;
            req
        };
        let ln0 = &ln_req["sites"][0];
        let conv0 = &conv_req["sites"][0];
        let cases = vec![
            ("ln", ln_req.clone(), true),
            ("conv", conv_req.clone(), true),
            (
                "A written between",
                mutate(&ln_req, "a_writes_between", json!(1)),
                false,
            ),
            (
                "stats rewritten",
                mutate(&ln_req, "stats_writes_between", json!(1)),
                false,
            ),
            (
                "stats of another tensor",
                mutate(
                    &ln_req,
                    "writer_x",
                    json!(ln0["gemm_a"].as_u64().unwrap() + 1),
                ),
                false,
            ),
            (
                "writer not RowStats",
                mutate(&ln_req, "writer_row_stats", json!(false)),
                false,
            ),
            (
                "rows not covered",
                mutate(
                    &ln_req,
                    "m",
                    json!(ln0["writer_rows"].as_u64().unwrap() + 1),
                ),
                false,
            ),
            (
                "width mismatch",
                mutate(
                    &ln_req,
                    "writer_feat",
                    json!(ln0["gemm_k"].as_u64().unwrap() + 1),
                ),
                false,
            ),
            (
                "same program",
                mutate(&ln_req, "writer_program", ln0["gemm_program"].clone()),
                false,
            ),
            (
                "scale short",
                mutate(
                    &conv_req,
                    "row_scale_bytes",
                    json!(conv0["row_scale_bytes"].as_u64().unwrap() - 4),
                ),
                false,
            ),
            (
                "no residual",
                mutate(&conv_req, "residual_bytes", json!(0)),
                false,
            ),
            // Tensors only need to cover the footprint (slab-backed sites), so a mutation that
            // shrinks out_rows (stride 2) still verifies; one that grows it past the tensors must not.
            (
                "in_rows grows out_rows past the footprint",
                mutate(
                    &conv_req,
                    "in_rows",
                    json!(conv0["in_rows"].as_u64().unwrap() * 2),
                ),
                false,
            ),
            (
                "scale aliases out",
                mutate(&conv_req, "row_scale_is_out", json!(true)),
                false,
            ),
        ];
        let requests: Vec<_> = cases
            .iter()
            .map(|(_, r, _)| (plow_asset::speech_fusion::ENDPOINT, r.clone()))
            .collect();
        let certs = lean_verify::call_batch(&requests).unwrap();
        for ((what, _, ok), cert) in cases.iter().zip(certs) {
            assert_eq!(cert.ok, *ok, "{what}: {:?}", cert.reason);
        }

        let mut orphan = ln.model;
        orphan.progs.remove(0);
        orphan.prog_t.remove(0);
        assert!(derive(&orphan).unwrap_err().contains("no earlier writer"));
    }
}
