//! Hardware test for the multi-GPU (tensor-parallel) host layer.
//!
//! Covers what the host owes the device before a single collective packet can
//! run: N backends with ranks, one
//! peer-mapped reduction region per rank visible to every peer, a
//! cache-line-isolated cross-GPU counter region inside it, the `[n_gpu]`
//! peer-pointer table, and the zero-every-rank-before-any-launch obligation.
//!
//! What it CANNOT cover: plowrt has no AMD interpreter engine, so nothing here
//! dispatches a kernel. The device half — a system-scope atomic on peer memory
//! being coherent across GPUs — is proven separately by
//! `runtime/tests/tp_p2p_bench` (re-measured on ROCm 7.2.4: works, ~0.06 µs
//! one-way handshake, 58.6 GB/s peer store). These tests use the copy engine,
//! which proves the mapping and the addresses, not the memory model.
//!
//! Needs at least 2 real gfx9xx GPUs + ROCr:
//!
//!   PLOW_GPU_TEST=1 ROCR_VISIBLE_DEVICES=4,5,6,7 \
//!     cargo test -p plowrt --features hsa --test multi_gpu
//!
//! NOTE `ROCR_VISIBLE_DEVICES`, not `HIP_VISIBLE_DEVICES`: plowrt dlopens ROCr
//! directly and never loads the HIP runtime, so the HIP variable is ignored
//! (measured — `HIP_VISIBLE_DEVICES=4,5,6,7` still enumerated all 8 agents).

#![cfg(feature = "hsa")]

use plowrt::device;
use plowrt::exec::tp::{PeerLayout, TpGroup, XctrReset, XCTR_STRIDE};

fn gpu_enabled() -> bool {
    std::env::var("PLOW_GPU_TEST").as_deref() == Ok("1")
}

/// Gemma-4 12B decode shape: H=3840, one token per dispatch, 48 layers × 2
/// one-shot collectives = 96 cross-GPU gates.
const HIDDEN: u32 = 3840;
const LAYERS: u32 = 48;

fn decode_layout() -> PeerLayout {
    PeerLayout::new(HIDDEN, 1, PeerLayout::counters_for(LAYERS, false))
        .expect("12B decode layout is line-aligned")
}

/// ONE group for the whole process, not one per test.
///
/// Two reasons, and both were learned the hard way in `hsa_primitives.rs`:
/// HSA is a process-global runtime, and — more to the point here — a second
/// group would allocate a second set of peer regions and a second pointer
/// table, which is not how plowrt holds them.
///
/// PLOW_GPU_TEST=1 is an ASSERTION that GPUs are present, so a failed probe is
/// a FAILURE, not a skip. An earlier version of the sibling suite returned
/// `None` here and turned a missing `libelf.so.1` (ROCr never loaded, CPU
/// fallback taken) into five green "ok"s — a run that reported success while
/// testing nothing at all.
fn backends() -> &'static Vec<std::sync::Arc<dyn device::Backend>> {
    static B: std::sync::OnceLock<Vec<std::sync::Arc<dyn device::Backend>>> =
        std::sync::OnceLock::new();
    B.get_or_init(|| {
        let all = device::select_all(1);
        assert!(
            all.len() >= 2,
            "PLOW_GPU_TEST=1 but only {} device(s) visible — this suite needs at \
             least 2 GPUs. Set ROCR_VISIBLE_DEVICES (HIP_VISIBLE_DEVICES has no \
             effect on ROCr), and run inside `nix develop` under `sg render`.",
            all.len()
        );
        for (i, be) in all.iter().enumerate() {
            assert!(
                be.peer().is_some(),
                "device {i} ({:?}) has no peer memory — the CPU fallback backend \
                 was selected, which means ROCr did not load",
                be.class()
            );
        }
        all
    })
}

fn group() -> &'static TpGroup {
    static G: std::sync::OnceLock<TpGroup> = std::sync::OnceLock::new();
    G.get_or_init(|| TpGroup::bringup(backends().clone(), decode_layout()).expect("TP bring-up"))
}

/// The deployment shape: the node split into independent replicas of half its
/// devices each (2 × TP4 on an 8-GPU node; 2 × TP2 on the 4-GPU test subset).
/// A separate bring-up from [`group`] on purpose — a device hosting several
/// peer regions at once is exactly what two co-tenant replicas do.
fn replicas() -> &'static Vec<TpGroup> {
    static R: std::sync::OnceLock<Vec<TpGroup>> = std::sync::OnceLock::new();
    R.get_or_init(|| {
        let all = backends().clone();
        let tp = (all.len() / 2) as u32;
        TpGroup::split_replicas(all, tp, decode_layout()).expect("replica bring-up")
    })
}

/// Every rank must come up with its own peer region, its own counter region
/// inside it, and a grid size that is a real CU count.
#[test]
fn every_rank_brings_up_with_a_distinct_peer_region() {
    if !gpu_enabled() {
        eprintln!("skipped: set PLOW_GPU_TEST=1 (needs >=2 GPUs)");
        return;
    }
    let g = group();
    assert!(g.n_gpu() >= 2);

    let mut bases: Vec<u64> = Vec::new();
    for (i, r) in g.ranks().iter().enumerate() {
        assert_eq!(r.rank(), i as u32, "rank must equal position in the group");
        // A zero grid is a zero-block dispatch, and the CU-count agent-info
        // enum was wrong once already (30115, the PCI chip id). No shipping GPU
        // has more than 1024 CUs.
        assert!(
            (1..=1024).contains(&r.executors()),
            "rank {i} grid {} is not a plausible CU count",
            r.executors()
        );
        assert_eq!(
            r.xctr() - r.scratch_base(),
            g.layout().xctr_off(),
            "xctr must sit at the layout offset inside this rank's own peer region — \
             a producer signals peer r at peer_scratch[r] + that offset"
        );
        assert_ne!(r.peer_scratch_table(), 0);
        bases.push(r.scratch_base());
    }

    // Distinct allocations, not N aliases of one buffer: a group whose ranks
    // shared a region would pass a naive reduce-and-compare while summing one
    // rank's partial N times.
    let mut sorted = bases.clone();
    sorted.sort_unstable();
    sorted.dedup();
    assert_eq!(sorted.len(), bases.len(), "peer regions alias: {bases:x?}");
}

/// The one that matters: rank A stores into rank B's peer region and B observes
/// it — over XGMI, at the address B published in the peer-pointer table.
///
/// `hsa_amd_agents_allow_access` REPLACES a buffer's allow-list, so naming
/// agents one at a time leaves only the LAST one mapped and every other peer
/// faults on first touch. All `N·(N-1)` directed pairs are checked because that
/// bug is direction-specific and would otherwise hide behind whichever pair the
/// test happened to pick.
#[test]
fn peer_regions_are_reachable_from_every_rank() {
    if !gpu_enabled() {
        eprintln!("skipped: set PLOW_GPU_TEST=1 (needs >=2 GPUs)");
        return;
    }
    group().verify_peer_visibility().expect("peer visibility");
}

/// A peer must be able to reach the OWNER's counter region at
/// `peer_scratch[owner] + xctr_off`, and each counter must sit on its own
/// 128 B line.
///
/// The stride matters more across GPUs than within one: two counters sharing a
/// line means two ranks' independent system-scope release RMWs contend for the
/// same line over the fabric, serialising signals that the design counts on
/// being concurrent.
#[test]
fn a_peer_can_write_the_owners_xctr_cells() {
    if !gpu_enabled() {
        eprintln!("skipped: set PLOW_GPU_TEST=1 (needs >=2 GPUs)");
        return;
    }
    let g = group();
    let n = g.n_gpu();
    let owner = g.rank(0).unwrap();
    let xbytes = g.layout().xctr_bytes() as usize;

    g.zero_xctr().expect("zero");

    // Every non-owner rank signals a DIFFERENT counter of rank 0, staged
    // through its own peer region and pushed over the fabric — the shape of
    // the coarse publish in §6b, minus the atomicity the copy engine cannot
    // provide.
    for src in 1..n {
        let s = g.rank(src).unwrap();
        let cell = src as u64 * XCTR_STRIDE as u64;
        let value = (0x1000u32 + src).to_le_bytes();
        s.write_scratch(0, &value).expect("stage");
        s.publish_to(owner, 0, g.layout().xctr_off() + cell, value.len() as u64)
            .expect("peer store into owner xctr");
    }

    let mut back = vec![0u8; xbytes];
    owner
        .read_scratch(g.layout().xctr_off(), &mut back)
        .expect("read xctr");

    for src in 1..n {
        let cell = src as usize * XCTR_STRIDE;
        let got = u32::from_le_bytes(back[cell..cell + 4].try_into().unwrap());
        assert_eq!(
            got,
            0x1000 + src,
            "rank {src}'s signal did not land in rank 0's counter {src}"
        );
        // Neighbouring cells must be untouched: a stride collapse would show
        // up here as one write clobbering another's counter.
        assert!(
            back[cell + 4..cell + XCTR_STRIDE].iter().all(|&b| b == 0),
            "counter {src} bled past 4 bytes into its 128 B line"
        );
    }
    // Cell 0 belongs to no signaller in this test and must still read zero.
    assert!(back[..XCTR_STRIDE].iter().all(|&b| b == 0));
}

/// Two co-tenant replicas on one node must each come up whole, with ranks
/// numbered from 0 inside the replica and NOT equal to the device ordinal.
///
/// Conflating rank with ordinal is the bug this guards: replica 1's rank 0
/// lives on device `tp`, and a peer store that used the rank as the target
/// ordinal would address replica 0's device instead — corrupting the other
/// tenant's partials rather than failing.
#[test]
fn a_node_hosts_independent_replicas() {
    if !gpu_enabled() {
        eprintln!("skipped: set PLOW_GPU_TEST=1 (needs >=2 GPUs)");
        return;
    }
    let reps = replicas();
    assert_eq!(reps.len(), 2, "expected the node split in two");

    let tp = reps[0].n_gpu();
    let mut all_ordinals = Vec::new();
    for (i, rep) in reps.iter().enumerate() {
        assert_eq!(rep.n_gpu(), tp, "replicas must be the same degree");
        for (r, rank) in rep.ranks().iter().enumerate() {
            assert_eq!(rank.rank(), r as u32);
            assert_eq!(
                rank.ordinal() as usize,
                i * tp as usize + r,
                "replica {i} rank {r} is on the wrong device"
            );
            all_ordinals.push(rank.ordinal());
        }
        // Each replica must be internally sound on its own.
        rep.verify_peer_visibility()
            .unwrap_or_else(|e| panic!("replica {i} peer visibility: {e}"));
    }

    // No device serves two replicas, and no peer region is shared.
    all_ordinals.sort_unstable();
    let n = all_ordinals.len();
    all_ordinals.dedup();
    assert_eq!(all_ordinals.len(), n, "a device serves two replicas");

    let mut bases: Vec<u64> = reps
        .iter()
        .flat_map(|r| r.ranks().iter().map(|k| k.scratch_base()))
        .collect();
    let n = bases.len();
    bases.sort_unstable();
    bases.dedup();
    assert_eq!(bases.len(), n, "replicas share a peer region");
}

/// The allow-list is the isolation mechanism, so the bring-up must refuse a
/// list that does not name the owner.
///
/// Coarse-grained VRAM is not even self-accessible until its owner appears on
/// the list, so an omitted owner is a buffer nothing can touch — and the
/// failure would surface as a fault inside the first collective, not at the
/// allocation that caused it.
#[test]
fn alloc_peer_rejects_an_allow_list_without_the_owner() {
    if !gpu_enabled() {
        eprintln!("skipped: set PLOW_GPU_TEST=1 (needs >=2 GPUs)");
        return;
    }
    let all = backends();
    let peer = all[0].peer().expect("peer");
    assert!(
        peer.alloc_peer(4096, &[1]).is_err(),
        "dev 0 allocated a peer buffer it is not itself allowed to access"
    );
    assert!(
        peer.alloc_peer(4096, &[0, 200]).is_err(),
        "a nonexistent peer ordinal was accepted into the allow-list"
    );
}

/// MEASURED, and it is why there is no "the other replica's SDMA copy fails"
/// test here.
///
/// `hsa_amd_memory_async_copy` names its source and destination agents
/// explicitly, and the driver programs the copy engine from that pair — so a
/// D2D copy between two replicas' buffers **succeeds** even though neither
/// buffer is on the other's `agents_allow_access` list. The isolation the
/// design relies on is the *kernel-visible mapping*: a shader on rank A
/// dereferencing an address that was never mapped into A's address space
/// faults. That cannot be tested from here, because plowrt has no AMD engine
/// to run a shader with.
///
/// So this test pins the property that IS host-observable and load-bearing:
/// two replicas share no peer region, no counter region, and no device. It is
/// deliberately not named "isolated" — the mapping-level guarantee is
/// asserted by `alloc_peer`'s allow-list and still owes a device-side check.
#[test]
fn replicas_share_no_peer_state() {
    if !gpu_enabled() {
        eprintln!("skipped: set PLOW_GPU_TEST=1 (needs >=2 GPUs)");
        return;
    }
    let reps = replicas();
    let mut regions: Vec<(u64, u64)> = Vec::new();
    for rep in reps {
        for r in rep.ranks() {
            regions.push((r.scratch_base(), r.xctr()));
        }
    }
    let n = regions.len();
    regions.sort_unstable();
    regions.dedup();
    assert_eq!(n, regions.len(), "two ranks share a peer or counter region");

    // And the counter regions do not overlap: rank A's 12 KB of counters must
    // not start inside rank B's.
    let xbytes = reps[0].layout().xctr_bytes();
    let mut xs: Vec<u64> = regions.iter().map(|(_, x)| *x).collect();
    xs.sort_unstable();
    for w in xs.windows(2) {
        assert!(
            w[1] - w[0] >= xbytes,
            "counter regions at {:#x} and {:#x} overlap ({xbytes} B each)",
            w[0],
            w[1]
        );
    }
}

/// §6d's single host obligation: EVERY rank's counters are zero before ANY rank
/// is launched.
///
/// Zeroing rank-by-rank as each is launched breaks the deadlock argument — an
/// early rank can signal a late rank's counter and have the late rank's own
/// zeroing wipe the signal, after which it waits forever. So the test dirties
/// every rank first and requires one call to clean all of them.
#[test]
fn zero_xctr_clears_every_rank_before_any_launch() {
    if !gpu_enabled() {
        eprintln!("skipped: set PLOW_GPU_TEST=1 (needs >=2 GPUs)");
        return;
    }
    let g = group();
    let xbytes = g.layout().xctr_bytes() as usize;
    let dirty = vec![0xABu8; xbytes];

    for r in g.ranks() {
        r.write_scratch(g.layout().xctr_off(), &dirty)
            .expect("dirty");
    }

    let mut launched: Vec<u32> = Vec::new();
    g.launch_token(XctrReset::Host, |r| {
        // Stand-in for the persistent dispatch: assert the precondition that
        // dispatch would depend on. By the time ANY rank is "launched", EVERY
        // rank's counters must already read zero — including ranks later in the
        // loop, which is what rank-by-rank zeroing would get wrong.
        for peer in g.ranks() {
            let mut back = vec![0u8; xbytes];
            peer.read_scratch(g.layout().xctr_off(), &mut back)
                .expect("read");
            assert!(
                back.iter().all(|&b| b == 0),
                "rank {} launched while rank {}'s xctr still held stale counts",
                r.rank(),
                peer.rank()
            );
        }
        launched.push(r.rank());
        Ok(())
    })
    .expect("launch_token");

    assert_eq!(
        launched,
        (0..g.n_gpu()).collect::<Vec<_>>(),
        "every rank must be dispatched exactly once, in rank order"
    );
}

/// `HostDirect` must clear the same bytes as the copy-engine reset, and the
/// device must read back what the host stored.
///
/// This is the 16× cut on the token's only host cost (31.5 → 1.97 µs at TP=4,
/// measured), so it is worth an explicit equivalence test rather than trusting
/// that two paths spelled differently do the same thing. What it does NOT
/// establish is the cache ordering a real dispatch would need — the readback
/// here is through the copy engine, not a shader.
#[test]
fn host_direct_reset_clears_the_same_bytes() {
    if !gpu_enabled() {
        eprintln!("skipped: set PLOW_GPU_TEST=1 (needs >=2 GPUs)");
        return;
    }
    let g = group();
    let xbytes = g.layout().xctr_bytes() as usize;
    let dirty = vec![0xC3u8; xbytes];
    for r in g.ranks() {
        r.write_scratch(g.layout().xctr_off(), &dirty)
            .expect("dirty");
    }

    g.zero_xctr_direct().expect("host-direct reset");

    for r in g.ranks() {
        let mut back = vec![0u8; xbytes];
        r.read_scratch(g.layout().xctr_off(), &mut back)
            .expect("read");
        assert!(
            back.iter().all(|&b| b == 0),
            "rank {}'s counters survived the host-direct reset",
            r.rank()
        );
    }
}

/// The hostless mode must touch nothing: `XctrReset::Program` means the token's
/// entire host cost is the dispatches.
///
/// Asserted by leaving the counters dirty and requiring them to STAY dirty. A
/// "reset policy" that quietly zeroed anyway would look identical in every
/// correctness test and only show up as host time in a latency trace — which is
/// exactly the kind of cost this design refuses to pay invisibly.
#[test]
fn program_reset_leaves_the_counters_to_the_device() {
    if !gpu_enabled() {
        eprintln!("skipped: set PLOW_GPU_TEST=1 (needs >=2 GPUs)");
        return;
    }
    let g = group();
    let xbytes = g.layout().xctr_bytes() as usize;
    let dirty = vec![0x5Au8; xbytes];
    for r in g.ranks() {
        r.write_scratch(g.layout().xctr_off(), &dirty)
            .expect("dirty");
    }

    let mut n = 0;
    g.launch_token(XctrReset::Program, |_| {
        n += 1;
        Ok(())
    })
    .expect("launch_token");
    assert_eq!(n, g.n_gpu());

    for r in g.ranks() {
        let mut back = vec![0u8; xbytes];
        r.read_scratch(g.layout().xctr_off(), &mut back)
            .expect("read");
        assert_eq!(
            back,
            dirty,
            "rank {} had its counters cleared by the host under XctrReset::Program",
            r.rank()
        );
    }
    // Leave the group in the state every other test expects.
    g.zero_xctr().expect("zero");
}
