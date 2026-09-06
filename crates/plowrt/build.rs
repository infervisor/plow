//! Builds the CPU kernel library (`runtime/cpu/dev/`) into plowrt under the `cpu`
//! feature. Sources are globbed at build time so a new kernel file is picked up
//! without touching this script; per-directory `-m` flags give the AVX-512 and
//! AMX tiers their ISA while the rest of the library stays baseline x86-64
//! (tier selection happens at runtime via cpuid, not at link time).
//!
//! Also emits `abi_probe.c`, whose `plow_cpu_abi_*` functions report the C
//! compiler's `sizeof`/`offsetof` for the structs `exec::cpu::ffi` mirrors;
//! `tests/cpu_abi.rs` asserts them against Rust's.

fn main() {
    #[cfg(feature = "cpu")]
    cpu::build();
}

#[cfg(feature = "cpu")]
mod cpu {
    use std::path::{Path, PathBuf};

    /// Stands in for the library until the first real kernel source lands, so
    /// the crate always links. Every entry reports "nothing available".
    const STUB: &str = r#"
#include "cpu_dev.h"
int plow_cpu_init(int isa_cap) { (void)isa_cap; return PLOW_CPU_ISA_SCALAR; }
int plow_cpu_isa(void) { return PLOW_CPU_ISA_SCALAR; }
int plow_cpu_thread_init(PlowCpuCtx* ctx) { if (ctx) ctx->isa = PLOW_CPU_ISA_SCALAR; return 0; }
uint32_t plow_cpu_scratch_bytes(void) { return 0; }
int plow_cpu_has(uint16_t op) { (void)op; return 0; }
plow_cpu_kernel_fn plow_cpu_kernel(uint16_t op) { (void)op; return (plow_cpu_kernel_fn)0; }
int plow_cpu_exec(const PlowDevInst* in, uint32_t slice, uint32_t nblk, void* const* tensors, PlowCpuCtx* ctx) {
    (void)in; (void)slice; (void)nblk; (void)tensors; (void)ctx; return -1;
}
size_t plow_cpu_prepack_bf16_b_bytes(uint32_t n, uint32_t k) { (void)n; (void)k; return 0; }
int plow_cpu_prepack_bf16_b(void* dst, const void* src, uint32_t n, uint32_t k) {
    (void)dst; (void)src; (void)n; (void)k; return -1;
}
"#;

    const ABI_PROBE: &str = r#"
#include <stddef.h>
#include "cpu_dev.h"
size_t plow_cpu_abi_sizeof_ctx(void) { return sizeof(PlowCpuCtx); }
size_t plow_cpu_abi_sizeof_inst(void) { return sizeof(PlowDevInst); }
size_t plow_cpu_abi_offsetof_ctx_scratch(void) { return offsetof(PlowCpuCtx, scratch); }
size_t plow_cpu_abi_offsetof_ctx_scratch_bytes(void) { return offsetof(PlowCpuCtx, scratch_bytes); }
size_t plow_cpu_abi_offsetof_ctx_worker(void) { return offsetof(PlowCpuCtx, worker); }
size_t plow_cpu_abi_offsetof_ctx_node(void) { return offsetof(PlowCpuCtx, node); }
size_t plow_cpu_abi_offsetof_ctx_isa(void) { return offsetof(PlowCpuCtx, isa); }
size_t plow_cpu_abi_offsetof_ctx_reserved(void) { return offsetof(PlowCpuCtx, reserved); }
size_t plow_cpu_abi_offsetof_inst_op(void) { return offsetof(PlowDevInst, op); }
size_t plow_cpu_abi_offsetof_inst_blocks(void) { return offsetof(PlowDevInst, blocks); }
size_t plow_cpu_abi_offsetof_inst_fj(void) { return offsetof(PlowDevInst, fj); }
size_t plow_cpu_abi_offsetof_inst_t(void) { return offsetof(PlowDevInst, t); }
size_t plow_cpu_abi_offsetof_inst_i(void) { return offsetof(PlowDevInst, i); }
int plow_cpu_abi_isa_scalar(void) { return PLOW_CPU_ISA_SCALAR; }
int plow_cpu_abi_isa_avx512(void) { return PLOW_CPU_ISA_AVX512; }
int plow_cpu_abi_isa_amx(void) { return PLOW_CPU_ISA_AMX; }
int plow_cpu_abi_dop_table(void) { return PLOW_CPU_DOP_TABLE; }
"#;

    fn c_files(dir: &Path) -> Vec<PathBuf> {
        let mut v: Vec<PathBuf> = match std::fs::read_dir(dir) {
            Ok(rd) => rd
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| p.extension().is_some_and(|x| x == "c"))
                .collect(),
            Err(_) => Vec::new(),
        };
        v.sort();
        v
    }

    /// Whether the C compiler accepts `flag` (probed with an empty TU).
    fn accepts(base: &cc::Build, flag: &str) -> bool {
        base.is_flag_supported(flag).unwrap_or(false)
    }

    fn base(root: &Path) -> cc::Build {
        let mut b = cc::Build::new();
        b.include(root.join("runtime/common"))
            .include(root.join("runtime/cpu/dev"))
            .include(root.join("include"))
            .flag("-std=c11")
            .opt_level(2)
            .warnings(true)
            .cargo_metadata(false);
        b
    }

    pub fn build() {
        let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
        let root = manifest.join("../..").canonicalize().unwrap();
        let dev = root.join("runtime/cpu/dev");
        let out = PathBuf::from(std::env::var("OUT_DIR").unwrap());

        for d in ["", "golden", "avx512", "amx"] {
            println!("cargo:rerun-if-changed={}", dev.join(d).display());
        }
        println!("cargo:rerun-if-changed={}", dev.join("cpu_dev.h").display());
        println!(
            "cargo:rerun-if-changed={}",
            root.join("runtime/common/dev_isa.h").display()
        );

        let plain: Vec<PathBuf> = c_files(&dev)
            .into_iter()
            .chain(c_files(&dev.join("golden")))
            .collect();
        let avx = c_files(&dev.join("avx512"));
        let amx = c_files(&dev.join("amx"));

        // One object set per flag group, all archived into a single library.
        // `cargo_metadata(false)` on the parts so only the final archive is
        // announced to cargo.
        let mut objects: Vec<PathBuf> = Vec::new();

        let mut plain_srcs = plain.clone();
        let stub = out.join("cpu_dev_stub.c");
        if plain.is_empty() && avx.is_empty() && amx.is_empty() {
            std::fs::write(&stub, STUB).unwrap();
            plain_srcs.push(stub);
            println!("cargo:warning=runtime/cpu/dev has no kernel sources yet; linking the no-op stub");
        }
        let probe = out.join("cpu_dev_abi_probe.c");
        std::fs::write(&probe, ABI_PROBE).unwrap();
        plain_srcs.push(probe);

        let mut b = base(&root);
        b.files(&plain_srcs);
        objects.extend(b.compile_intermediates());

        if !avx.is_empty() {
            let mut b = base(&root);
            for f in [
                "-mavx512f",
                "-mavx512bw",
                "-mavx512vl",
                "-mavx512bf16",
                "-mavx512vnni",
            ] {
                b.flag(f);
            }
            if accepts(&b, "-mavx512fp16") {
                b.flag("-mavx512fp16");
            }
            b.files(&avx);
            objects.extend(b.compile_intermediates());
        }
        if !amx.is_empty() {
            let mut b = base(&root);
            for f in [
                "-mavx512f",
                "-mavx512bw",
                "-mavx512vl",
                "-mavx512bf16",
                "-mavx512vnni",
                "-mamx-tile",
                "-mamx-bf16",
                "-mamx-int8",
            ] {
                b.flag(f);
            }
            if accepts(&b, "-mavx512fp16") {
                b.flag("-mavx512fp16");
            }
            b.files(&amx);
            objects.extend(b.compile_intermediates());
        }

        // Archive + link. `cc::Build::compile` would re-run every source with one
        // flag set, so the archive is assembled from the intermediates instead.
        let lib = out.join("libplow_cpu_dev.a");
        let _ = std::fs::remove_file(&lib);
        let ar = cc::Build::new().get_archiver();
        let mut cmd = ar;
        cmd.arg("crs").arg(&lib);
        cmd.args(&objects);
        let st = cmd.status().expect("run archiver for libplow_cpu_dev.a");
        assert!(st.success(), "archiving libplow_cpu_dev.a failed");

        println!("cargo:rustc-link-search=native={}", out.display());
        println!("cargo:rustc-link-lib=static=plow_cpu_dev");
        // libm for the golden kernels' expf/tanhf/etc.
        println!("cargo:rustc-link-lib=m");

    }
}
