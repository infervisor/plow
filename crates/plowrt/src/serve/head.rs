//! The CPU prefill-head pool: the twin packet, the cores it is allowed, and a
//! head that can be abandoned the moment the device is ready for the request.
//!
//! ## Placement is inherited, not re-implemented
//!
//! The pool's workers are placed by giving [`CpuEngine::load`] a **restricted
//! topology** — only the cores [`crate::exec::affinity::plan`] reserved for
//! heads — so `WorkerPool` pins each worker inside the reservation using the
//! machinery it already has, and nothing in `exec::cpu::workers` changes.
//!
//! `SCHED_IDLE` comes the same way. Linux `pthread_create` defaults to
//! `PTHREAD_INHERIT_SCHED`, so a thread inherits its creator's scheduling
//! policy; loading the engine from a bootstrap thread that has already put
//! itself on `SCHED_IDLE` puts every worker there too. **That inheritance is
//! load-bearing**: without it a head thread can preempt the engine thread
//! wherever the two masks overlap, which is the one thing this design must not
//! do. It is asserted after the pool is up rather than assumed.
//!
//! Neither mechanism bounds MEMORY BANDWIDTH — an idle-priority thread that is
//! running still saturates the memory controllers — which is why the pool's
//! width is a budget and the contention guard watches service time.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};

use crate::exec::cpu::engine::{next_chunk, CpuEngine, CpuEngineOpts};
use crate::exec::cpu::ffi::Isa;
use crate::exec::cpu::topology::{Core, Topology};
use crate::exec::kv_handoff::{self, CopySpan};
use crate::exec::kvrow::{self, KvSlotTensor};
use crate::{Result, RuntimeError};

/// Why a head stopped short of the rows it was asked for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HeadEnd {
    /// Every requested row is prefilled and its KV is ready to hand over.
    Complete,
    /// The device wanted the request before the head finished. Rows already
    /// covered are still valid — the frontier is where it stopped.
    Abandoned,
}

/// A finished (or abandoned) head.
#[derive(Clone, Copy, Debug)]
pub struct Head {
    pub rows: u32,
    pub end: HeadEnd,
}

/// Where a head's twin packet comes from, and whether this host can run one.
///
/// A twin is another VARIANT of the same model — same context, a target the CPU
/// interpreter accepts — so selecting one is variant selection and belongs to
/// whatever pulled the bundle. This resolves the packet inside the bundle that
/// selection materialised: `--het-twin` names it outright, otherwise the
/// conventional `cpu-twin/model.pkt` beside the device packet.
///
/// `None` is the ordinary answer on a bundle that carries no twin, and it means
/// heads are off for that model. Not an error: a device packet without a twin
/// is a complete, servable bundle.
pub fn resolve_twin(assets: &Path, configured: Option<&str>) -> Option<PathBuf> {
    if let Some(p) = configured {
        let p = PathBuf::from(p);
        // A directory names a bundle; a file names the packet in one.
        let pkt = if p.is_dir() { p.join("model.pkt") } else { p };
        return pkt.is_file().then_some(pkt);
    }
    let conventional = assets.join("cpu-twin").join("model.pkt");
    conventional.is_file().then_some(conventional)
}

/// Whether this host can run prefill heads at all, independent of load.
///
/// Two gates, and both are about not making serving worse:
///
/// * **A vector tier.** A scalar CPU prefill of a serving-sized model is slow
///   enough that no queue hides it, so the head would be abandoned every time
///   and the pool would only ever cost cores. AVX-512 or AMX is the floor.
/// * **A reservation.** With no cores reserved a head runs on the serving
///   path's, which is the one thing this must not do.
///
/// Returns the reason rather than a bool: a host that cannot run heads should
/// say why once at load, not silently serve as though the feature were off.
pub fn host_supports_heads(isa: Isa, reserved_cores: usize) -> std::result::Result<(), String> {
    if matches!(isa, Isa::Scalar) {
        return Err(
            "CPU prefill heads need the AVX-512 or AMX tier; this host probed scalar, where a \
             head is slower than any queue it could hide in"
                .into(),
        );
    }
    if reserved_cores == 0 {
        return Err(
            "no cores reserved for prefill heads; set --het-reserve-cores or --het-cores, or a \
             head would run on the serving path's cores"
                .into(),
        );
    }
    Ok(())
}

pub struct HeadPool {
    eng: CpuEngine,
    /// The twin's transferable caches, in the order the digest hashes them.
    contract: Vec<KvSlotTensor>,
    digest: String,
    pools: Vec<(String, u32)>,
    rows_per_slot: u32,
    /// Compiled prefill buckets `(program, rows)` — the abandon granularity.
    buckets: Vec<(usize, u32)>,
    slots: usize,
}

impl HeadPool {
    /// Load the twin onto `cores`.
    ///
    /// Call from a thread that has already set its own affinity and
    /// `SCHED_IDLE` (see the module note): the worker threads inherit both.
    pub fn load(
        twin: &Path,
        checkpoint: &Path,
        opts: &CpuEngineOpts,
        cores: &[u32],
    ) -> Result<HeadPool> {
        if cores.is_empty() {
            return Err(RuntimeError::Rejected(
                "no cores reserved for prefill heads; set --het-cores or --het-reserve-cores"
                    .into(),
            ));
        }
        // REFUSE ON THE PACKET BEFORE PAYING FOR THE WEIGHTS. Whether a model's
        // caches are addressable as row ranges is a property of the packet
        // alone, and a twin that cannot hand over should say so in
        // milliseconds — not after a 22 GiB checkpoint load, which is what the
        // first cut of this did and what a Gemma-4-12B bundle actually costs.
        //
        // Every prefill program is checked, not just one: a packet can reach
        // the caches through more than one program.
        let bytes = std::fs::read(twin).map_err(|e| {
            RuntimeError::Device(format!("CPU twin {}: {e}", twin.display()))
        })?;
        let blob = crate::asset::devblob::DevBlob::parse(&bytes)?;
        let names: Vec<String> = blob.tensors.iter().map(|t| t.name.clone()).collect();
        let mut pools = Vec::new();
        for p in &blob.progs {
            kv_handoff::check_seq_major(&p.insts, &names)?;
            for entry in kv_handoff::pooled_caches(&p.insts, &names) {
                if !pools.contains(&entry) {
                    pools.push(entry);
                }
            }
        }
        drop(blob);
        drop(bytes);

        let mut opts = opts.clone();
        opts.topology = Some(restrict(&Topology::detect(), cores));
        let eng = CpuEngine::load(twin, checkpoint, &opts)?;

        let blob = &eng.model().blob;
        let batch = eng.batch() as u32;
        let contract = kvrow::kv_slot_tensors(blob, batch)?;
        if contract.is_empty() {
            return Err(RuntimeError::Device(
                "CPU twin declares no transferable KV cache; nothing to hand over".into(),
            ));
        }
        let digest = kvrow::kv_contract_digest(&contract);

        let buckets = eng.prefill_buckets();
        if buckets.is_empty() {
            return Err(RuntimeError::Device(
                "CPU twin has no prefill program; it cannot run a head".into(),
            ));
        }
        let rows_per_slot = eng.max_ctx() as u32;
        let slots = eng.batch();
        tracing::info!(
            cores = cores.len(),
            slots,
            rows_per_slot,
            caches = contract.len(),
            pooled = pools.len(),
            kv_contract = %digest,
            "CPU prefill head pool ready"
        );
        Ok(HeadPool {
            eng,
            contract,
            digest,
            pools,
            rows_per_slot,
            buckets,
            slots,
        })
    }

    /// The twin's KV contract; the device engine's must equal it.
    pub fn kv_contract(&self) -> &str {
        &self.digest
    }

    pub fn slots(&self) -> usize {
        self.slots
    }

    /// Prefill `prompt[..rows]` into head slot `slot`, checking `cancel`
    /// between chunks.
    ///
    /// Returns how many rows are actually covered. **A cancelled head is not a
    /// failed one**: the rows it did cover are complete and their KV is
    /// transferable, so the caller may hand over the shorter prefix or drop it.
    /// Chunk granularity is the twin's own bucket ladder, which is why the twin
    /// is emitted with small rungs — it sets how promptly the device can take
    /// the request back.
    pub fn run(
        &mut self,
        slot: usize,
        prompt: &[u32],
        rows: u32,
        cancel: &AtomicBool,
    ) -> Result<Head> {
        if slot >= self.slots {
            return Err(RuntimeError::Rejected(format!(
                "head slot {slot} past the twin's {} slots",
                self.slots
            )));
        }
        let rows = rows.min(prompt.len() as u32).min(self.rows_per_slot);
        if rows == 0 {
            return Ok(Head {
                rows: 0,
                end: HeadEnd::Complete,
            });
        }
        let mut done = 0u32;
        while done < rows {
            if cancel.load(Ordering::Relaxed) {
                return Ok(Head {
                    rows: done,
                    end: HeadEnd::Abandoned,
                });
            }
            let ch = next_chunk(&self.buckets, rows, done, u32::MAX);
            self.eng.prefill_slot_chunk(slot, &prompt[..rows as usize], ch)?;
            done += ch.clen;
        }
        Ok(Head {
            rows: done,
            end: HeadEnd::Complete,
        })
    }

    /// Plan the copies handing `rows` of head slot `src` to device slot `dst`.
    pub fn plan(&self, src: u32, dst: u32, rows: u32) -> Result<Vec<CopySpan>> {
        kv_handoff::plan(
            &self.contract,
            self.rows_per_slot,
            &self.pools,
            src,
            dst,
            rows,
        )
    }

    /// The source bytes for one planned copy, read straight out of the twin's
    /// host tensor.
    pub fn source(&self, span: &CopySpan) -> Result<&[u8]> {
        self.eng
            .model()
            .tensor_range(span.handle, span.src_off, span.bytes)
    }
}

/// A topology restricted to `cores` — the reservation the head pool may use.
///
/// Whole physical cores with their SMT siblings, because that is what the
/// reservation is cut in: keeping a core whose sibling is outside the set would
/// put a head on the same execution resources as a serving thread.
fn restrict(topo: &Topology, cores: &[u32]) -> Topology {
    let keep: Vec<Core> = topo
        .cores
        .iter()
        .filter(|c| c.siblings.iter().all(|s| cores.contains(s)))
        .cloned()
        .collect();
    let mut nodes: Vec<u32> = keep.iter().map(|c| c.node).collect();
    nodes.sort_unstable();
    nodes.dedup();
    Topology {
        cores: keep,
        nodes,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn core(cpu: u32, node: u32, siblings: &[u32]) -> Core {
        Core {
            cpu,
            node,
            siblings: siblings.to_vec(),
        }
    }

    fn smt_box() -> Topology {
        Topology {
            cores: (0..4).map(|i| core(i, i / 2, &[i, i + 4])).collect(),
            nodes: vec![0, 1],
        }
    }

    #[test]
    fn a_bundle_without_a_twin_simply_has_no_heads() {
        // Not an error: a device packet with no twin is a complete bundle.
        let dir = std::env::temp_dir().join("plow_head_none");
        let _ = std::fs::create_dir_all(&dir);
        assert_eq!(resolve_twin(&dir, None), None);
    }

    #[test]
    fn the_conventional_location_is_beside_the_device_packet() {
        let dir = std::env::temp_dir().join("plow_head_conv");
        let twin = dir.join("cpu-twin");
        std::fs::create_dir_all(&twin).unwrap();
        let pkt = twin.join("model.pkt");
        std::fs::write(&pkt, b"x").unwrap();
        assert_eq!(resolve_twin(&dir, None), Some(pkt));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_configured_directory_names_a_bundle_and_a_file_names_a_packet() {
        let dir = std::env::temp_dir().join("plow_head_cfg");
        std::fs::create_dir_all(&dir).unwrap();
        let pkt = dir.join("model.pkt");
        std::fs::write(&pkt, b"x").unwrap();
        assert_eq!(
            resolve_twin(Path::new("/nonexistent"), Some(dir.to_str().unwrap())),
            Some(pkt.clone())
        );
        assert_eq!(
            resolve_twin(Path::new("/nonexistent"), Some(pkt.to_str().unwrap())),
            Some(pkt)
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_configured_twin_that_is_not_there_resolves_to_nothing() {
        // Reported as "no heads" by the caller rather than papered over with
        // the conventional path, which would serve a twin the operator did not
        // ask for.
        assert_eq!(
            resolve_twin(Path::new("/nonexistent"), Some("/nonexistent/model.pkt")),
            None
        );
    }

    #[test]
    fn a_scalar_host_cannot_run_heads() {
        let err = host_supports_heads(Isa::Scalar, 8).unwrap_err();
        assert!(err.contains("AVX-512 or AMX"), "{err}");
    }

    #[test]
    fn no_reservation_means_no_heads() {
        let err = host_supports_heads(Isa::Avx512, 0).unwrap_err();
        assert!(err.contains("no cores reserved"), "{err}");
    }

    #[test]
    fn a_vector_host_with_a_reservation_can() {
        assert!(host_supports_heads(Isa::Avx512, 4).is_ok());
        assert!(host_supports_heads(Isa::Amx, 4).is_ok());
    }

    #[test]
    fn a_restricted_topology_keeps_only_whole_reserved_cores() {
        let t = restrict(&smt_box(), &[2, 6, 3, 7]);
        assert_eq!(t.cores.len(), 2);
        assert_eq!(t.cores[0].cpu, 2);
        assert_eq!(t.nodes, vec![1]);
    }

    #[test]
    fn a_core_with_a_sibling_outside_the_reservation_is_dropped() {
        // Half a core is not a reservation: the sibling would share execution
        // resources with whatever holds the other thread.
        let t = restrict(&smt_box(), &[2, 3, 7]);
        assert_eq!(t.cores.len(), 1, "only core 3 has both siblings reserved");
        assert_eq!(t.cores[0].cpu, 3);
    }

    #[test]
    fn an_empty_reservation_restricts_to_nothing() {
        let t = restrict(&smt_box(), &[]);
        assert!(t.cores.is_empty());
        assert!(t.nodes.is_empty());
    }
}
