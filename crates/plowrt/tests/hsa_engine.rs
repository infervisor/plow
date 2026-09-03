//! Hardware test for `impl EngineDevice for HsaBackend` — the surface the
//! serving engine drives, exercised against the REAL production code objects.
//!
//! `hsa_primitives.rs` covers the memory/event plumbing. What is under test
//! here is the half that cannot be checked with a synthetic buffer: loading a
//! `build-amd/hsaco/*.elf` code object, resolving the interpreter's kernel
//! symbol out of it, packing a `DevProgram` kernarg block, and getting an AQL
//! dispatch of that kernel to run and retire.
//!
//! That chain is where the port's real risks live, and each one fails in a way
//! that looks like something else:
//!
//!   * a bundled (not unbundled) code object → `load_agent_code_object` errors;
//!   * a wrong symbol name → resolve fails (the loader wants `<name>.kd`);
//!   * a register-overcommitted object → the DISPATCH fails with
//!     `HSA_STATUS_ERROR_INVALID_ISA`, which reads as "bad ISA" but means
//!     "too many registers for this workgroup size";
//!   * grid passed in workgroups where AQL wants threads → runs 1/256th or
//!     256× the work, silently.
//!
//!   PLOW_GPU_TEST=1 ROCR_VISIBLE_DEVICES=4,5,6,7 \
//!     cargo test -p plowrt --features hsa --test hsa_engine

#![cfg(feature = "hsa")]

use packet::dev::DevProgram;
use plowrt::device::hsa::HsaBackend;
use plowrt::exec::device_api::EngineDevice;

fn gpu_enabled() -> bool {
    std::env::var("PLOW_GPU_TEST").as_deref() == Ok("1")
}

/// ONE backend per process — HSA is a process-global runtime, and a failed
/// probe under `PLOW_GPU_TEST=1` is a FAILURE, not a skip. Returning `None`
/// here once turned a missing `libelf.so.1` into five green "ok"s.
fn backend() -> &'static HsaBackend {
    static BE: std::sync::OnceLock<HsaBackend> = std::sync::OnceLock::new();
    BE.get_or_init(|| {
        HsaBackend::new(0).unwrap_or_else(|e| {
            panic!(
                "PLOW_GPU_TEST=1 but no HSA device: {e}\n\
                 Run inside `nix develop` under `sg render`."
            )
        })
    })
}

/// The prebuilt gfx950 objects. Absent means the kernel side has not been
/// built, which is a real failure of this test's premise rather than a reason
/// to report success.
fn hsaco_dir() -> std::path::PathBuf {
    let d = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../build-amd/hsaco")
        .canonicalize()
        .unwrap_or_else(|e| {
            panic!("build-amd/hsaco is missing ({e}) — run scripts/build_gfx950.sh")
        });
    assert!(
        d.join("interp_decode.elf").exists(),
        "no interp_decode.elf in {}",
        d.display()
    );
    d
}

/// The arch key must be the ISA name, because it is what selects both the code
/// object and the kernel symbol inside it (`plow_interp_dec_gfx950`).
#[test]
fn arch_is_the_isa_name_not_a_version_pair() {
    if !gpu_enabled() {
        eprintln!("skipped: set PLOW_GPU_TEST=1 (needs GPU)");
        return;
    }
    let arch = EngineDevice::arch(backend());
    assert!(
        arch.starts_with("gfx"),
        "arch {arch:?} is not an ISA name — the profile table keys on this string"
    );
    eprintln!("arch = {arch}");
}

/// Load every production object and resolve the symbol it is supposed to
/// export. A rename on either side breaks serving with a resolve error far
/// from its cause, so the whole matrix is checked at once.
#[test]
fn every_production_object_exports_its_interpreter_symbol() {
    if !gpu_enabled() {
        eprintln!("skipped: set PLOW_GPU_TEST=1 (needs GPU)");
        return;
    }
    let be = backend();
    let dir = hsaco_dir();
    let arch = EngineDevice::arch(be);

    // The static/GQ pairing from the AMD reference driver: prefill and decode
    // are separate objects because register allocation is per-kernel, and the
    // `_gq` twins differ only in the scheduling loop.
    let cases: &[(&str, &str)] = &[
        ("interp_prefill.elf", "plow_interp"),
        ("interp_prefill_gq.elf", "plow_interp"),
        ("interp_decode.elf", "plow_interp_dec"),
        ("interp_decode_gq.elf", "plow_interp_dec"),
        ("interp_flash.elf", "plow_interp_flash"),
        ("interp_flash_gq.elf", "plow_interp_flash"),
    ];

    for (file, base) in cases {
        let path = dir.join(file);
        if !path.exists() {
            panic!("{} missing — the object set is incomplete", path.display());
        }
        let image = std::fs::read(&path).expect("read code object");
        let m = EngineDevice::module_load(be, &image).unwrap_or_else(|e| {
            panic!(
                "{file}: module_load failed: {e}\n\
                 A bundled object gives exactly this — was it run through \
                 clang-offload-bundler --unbundle?"
            )
        });
        let gq = file.ends_with("_gq.elf");
        let sym = format!("{base}_{arch}{}", if gq { "_gq" } else { "" });
        EngineDevice::get_function(be, &m, &sym)
            .unwrap_or_else(|e| panic!("{file}: no symbol {sym}: {e}"));
        EngineDevice::module_unload(be, &m).expect("unload");
        eprintln!("{file}: {sym} ok");
    }
}

/// THE ONE THAT MATTERS: dispatch the real persistent interpreter and have it
/// retire.
///
/// The program is deliberately empty — `stream_len[cu] == 0` for every CU — so
/// each workgroup finds nothing to run and exits. That is not a weaker test
/// than a real token: everything this port newly owns is on the path to the
/// first instruction fetch (module load, symbol resolve, the `PlowProgram` kernarg
/// block, the COv5 implicit-arg tail, dispatch geometry, the completion
/// signal), and none of it depends on the stream being non-empty. What it does
/// NOT cover is the numerics, which need the weights.
#[test]
fn the_production_interpreter_dispatches_and_retires() {
    if !gpu_enabled() {
        eprintln!("skipped: set PLOW_GPU_TEST=1 (needs GPU)");
        return;
    }
    let be = backend();
    let arch = EngineDevice::arch(be);
    let image = std::fs::read(hsaco_dir().join("interp_decode.elf")).expect("read");
    let m = EngineDevice::module_load(be, &image).expect("module_load");
    let f = EngineDevice::get_function(be, &m, &format!("plow_interp_dec_{arch}"))
        .expect("get_function");

    // One workgroup per CU, 8 waves of 64 — the geometry `dev_isa.h` fixes as
    // PLOW_WG_THREADS. A wrong wave width here is a launch that either
    // under-fills the machine or is rejected outright.
    let n_cu = EngineDevice::sm_count(be);
    assert!((1..=1024).contains(&n_cu), "implausible CU count {n_cu}");
    const WG_THREADS: u32 = 8 * 64;

    // Empty per-CU streams. `stream_ofs`/`stream_len` are [n_cu] and must be
    // real device memory even when they describe no work — the interpreter
    // indexes them before it can discover there is nothing to do.
    let zeros = vec![0u8; (n_cu as usize) * 4];
    let d_sofs = EngineDevice::alloc(be, zeros.len() as u64).expect("alloc sofs");
    let d_slen = EngineDevice::alloc(be, zeros.len() as u64).expect("alloc slen");
    EngineDevice::upload(be, &d_sofs, 0, &zeros).expect("upload sofs");
    EngineDevice::upload(be, &d_slen, 0, &zeros).expect("upload slen");

    // Every other table gets one live page rather than a null: a null here is
    // an unmapped dereference on a GPU with no fault handler to blame it on.
    let scratch = EngineDevice::alloc(be, 4096).expect("alloc scratch");
    EngineDevice::memset_d8(be, scratch.base, 0, 4096).expect("memset");

    let prog = DevProgram {
        insts: scratch.base,
        stream: scratch.base,
        stream_ofs: d_sofs.base,
        stream_len: d_slen.base,
        waits: scratch.base,
        succs: scratch.base,
        counters: scratch.base,
        tensors: scratch.base,
        trace: 0,
        cur_seg: 0,
        l2_domains: 0,
        // Hierarchy off, as in both engines: nothing here sets `l2_domains`,
        // and the two-level maintenance scratch is meaningless without it.
        hier_base: 0,
        n_seg: 1,
        gq_stream: 0,
        gq_seg_ofs: 0,
        gq_cursor: 0,
        xctr: 0,
        peer_scratch: 0,
        rank: 0,
        n_gpu: 0,
        seg_ofs: 0,
        prefill_spans: 0,
        prefill_parked: 0,
        n_prefill_spans: 0,
        n_prefill_rows: 0,
    };
    // The kernarg block is the current struct's bytes. `dev_isa.h` static-asserts its size and
    // `packet::dev_abi` pins the Rust mirror against the C header.
    assert_eq!(
        std::mem::size_of::<DevProgram>(),
        168,
        "PlowProgram ABI drifted"
    );
    // SAFETY: `DevProgram` is `repr(C)` and POD (all u64/u32); reading it as
    // its own bytes is exactly what the kernarg memcpy does.
    let args: &[u8] = unsafe {
        std::slice::from_raw_parts(
            &prog as *const DevProgram as *const u8,
            std::mem::size_of::<DevProgram>(),
        )
    };

    EngineDevice::launch_cooperative(be, f, n_cu, WG_THREADS, 0, args, None).unwrap_or_else(|e| {
        panic!(
            "dispatch failed: {e}\n\
             HSA_STATUS_ERROR_INVALID_ISA (1017) here means the object's \
             register count is too high for {WG_THREADS} threads, not that the \
             ISA is wrong — check __launch_bounds__/amdgpu_waves_per_eu."
        )
    });
    EngineDevice::synchronize(be).expect("the interpreter did not retire");

    EngineDevice::module_unload(be, &m).expect("unload");
    eprintln!("dispatched plow_interp_dec_{arch}: grid={n_cu} wg={WG_THREADS} — retired");
}

/// SETTLES A DISPUTED PREMISE: does gfx950 allow a workgroup past 64 KiB of
/// LDS, and does the AMD interpreter need a re-tiled prefill arena?
///
/// The claim under test came from reading the CUDA engine's
/// `SMEM_PF = 21312 * 4 = 85,248 B` next to "CDNA's 64 KiB per-workgroup limit"
/// and concluding the AMD prefill path was over budget. That conflates two
/// different quantities. `SMEM_PF` is a **dynamic** shared-memory budget opted
/// into with `cuFuncSetAttribute`; the AMD interpreter declares
/// `__shared__ plow_smem sm` **statically**, so its arena is baked into the code
/// object's group-segment size and every dispatch passes `dynamic_lds = 0`.
///
/// So the question is empirical: what group segment do the real objects carry,
/// and does the hardware run them? This test reads the number off each object
/// and DISPATCHES the largest one. If gfx950 refused >64 KiB, the dispatch
/// fails here rather than in a serving step.
#[test]
fn gfx950_runs_the_prefill_arena_the_objects_actually_declare() {
    if !gpu_enabled() {
        eprintln!("skipped: set PLOW_GPU_TEST=1 (needs GPU)");
        return;
    }
    let be = backend();
    let dir = hsaco_dir();
    let arch = EngineDevice::arch(be);

    let mut worst: Option<(String, u32)> = None;
    for (file, base) in [
        ("interp_prefill.elf", "plow_interp"),
        ("interp_prefill_mla.elf", "plow_interp"),
        ("interp_decode.elf", "plow_interp_dec"),
        ("interp_flash.elf", "plow_interp_flash"),
    ] {
        let path = dir.join(file);
        if !path.exists() {
            eprintln!("{file}: absent, skipping");
            continue;
        }
        let image = std::fs::read(&path).expect("read");
        let m = EngineDevice::module_load(be, &image).expect("module_load");
        let f = EngineDevice::get_function(be, &m, &format!("{base}_{arch}")).expect("symbol");
        let lds = HsaBackend::kernel_lds_bytes(&f);
        eprintln!(
            "{file}: static LDS = {lds} B ({:.1} KiB)",
            lds as f64 / 1024.0
        );
        if worst.as_ref().is_none_or(|(_, w)| lds > *w) {
            worst = Some((file.to_string(), lds));
        }
        EngineDevice::module_unload(be, &m).expect("unload");
    }
    let (file, lds) = worst.expect("no objects found at all");

    // Now run the biggest one. The flash object is built 4-wave (256 threads);
    // prefill/decode are 8-wave (512). Using the wrong width is an INVALID_ISA,
    // so pick by the file, not by hope.
    let base = if file.starts_with("interp_decode") {
        "plow_interp_dec"
    } else if file.starts_with("interp_flash") {
        "plow_interp_flash"
    } else {
        "plow_interp"
    };
    let threads: u32 = if file.starts_with("interp_flash") {
        4 * 64
    } else {
        8 * 64
    };

    let image = std::fs::read(dir.join(&file)).expect("read");
    let m = EngineDevice::module_load(be, &image).expect("module_load");
    let f = EngineDevice::get_function(be, &m, &format!("{base}_{arch}")).expect("symbol");

    let n_cu = EngineDevice::sm_count(be);
    let zeros = vec![0u8; (n_cu as usize) * 4];
    let d_sofs = EngineDevice::alloc(be, zeros.len() as u64).expect("alloc");
    let d_slen = EngineDevice::alloc(be, zeros.len() as u64).expect("alloc");
    EngineDevice::upload(be, &d_sofs, 0, &zeros).expect("upload");
    EngineDevice::upload(be, &d_slen, 0, &zeros).expect("upload");
    let scratch = EngineDevice::alloc(be, 4096).expect("alloc");
    EngineDevice::memset_d8(be, scratch.base, 0, 4096).expect("memset");

    let prog = DevProgram {
        insts: scratch.base,
        stream: scratch.base,
        stream_ofs: d_sofs.base,
        stream_len: d_slen.base,
        waits: scratch.base,
        succs: scratch.base,
        counters: scratch.base,
        tensors: scratch.base,
        trace: 0,
        cur_seg: 0,
        l2_domains: 0,
        // Hierarchy off, as in both engines: nothing here sets `l2_domains`,
        // and the two-level maintenance scratch is meaningless without it.
        hier_base: 0,
        n_seg: 1,
        gq_stream: 0,
        gq_seg_ofs: 0,
        gq_cursor: 0,
        xctr: 0,
        peer_scratch: 0,
        rank: 0,
        n_gpu: 0,
        seg_ofs: 0,
        prefill_spans: 0,
        prefill_parked: 0,
        n_prefill_spans: 0,
        n_prefill_rows: 0,
    };
    // SAFETY: `DevProgram` is `repr(C)` POD; this is the kernarg memcpy's view.
    let args: &[u8] = unsafe {
        std::slice::from_raw_parts(
            &prog as *const DevProgram as *const u8,
            std::mem::size_of::<DevProgram>(),
        )
    };

    // dynamic_lds = 0: the arena is static, exactly as the reference driver does
    // it (every `plow_hsa_launch` in gemma4_chat.c passes literal 0).
    EngineDevice::launch_cooperative(be, f, n_cu, threads, 0, args, None).unwrap_or_else(|e| {
        panic!(
            "{file} declares {lds} B of static LDS and would not dispatch: {e}\n\
             If this is the LDS and not the register budget, gfx950 does NOT \
             permit a workgroup this large and the arena must be re-tiled."
        )
    });
    EngineDevice::synchronize(be).expect("did not retire");
    EngineDevice::module_unload(be, &m).expect("unload");

    eprintln!(
        "{file} dispatched and retired at {lds} B static LDS, {threads} threads — \
         gfx950 permits a workgroup arena of at least this size"
    );
    assert!(
        lds > 0,
        "a zero group segment means the arena was not compiled in — wrong object?"
    );
}

/// `memset_d8` is the counter re-arm, so it must actually write the bytes —
/// and on HSA it is a copy from a cached fill buffer, which is the kind of
/// thing that silently reuses a stale value when the value changes.
#[test]
fn memset_writes_the_requested_value_and_notices_when_it_changes() {
    if !gpu_enabled() {
        eprintln!("skipped: set PLOW_GPU_TEST=1 (needs GPU)");
        return;
    }
    let be = backend();
    const N: usize = 8192;
    let mem = EngineDevice::alloc(be, N as u64).expect("alloc");
    let mut back = vec![0u8; N];

    for value in [0u8, 0xA5, 0x00, 0xFF] {
        EngineDevice::memset_d8(be, mem.base, value, N).expect("memset");
        EngineDevice::download(be, &mem, 0, &mut back).expect("download");
        assert!(
            back.iter().all(|&b| b == value),
            "memset {value:#x} left {:#x} — is the fill buffer cached past a value change?",
            back.iter().find(|&&b| b != value).copied().unwrap_or(value)
        );
    }
}
