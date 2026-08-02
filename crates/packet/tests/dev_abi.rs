//! Cross-language ABI lock for the device ISA.
//!
//! `crates/packet/src/dev.rs` and `runtime/common/dev_isa.h` describe the same
//! bytes: the Rust side builds the instruction stream, the HIP interpreter
//! executes it. If they drift, the GPU reads garbage — and it reads it silently,
//! because there is no tag to check against.
//!
//! Rather than hand-copy the numbers into two places and hope, this test
//! *compiles the C header* and asks the C compiler for its sizes and offsets,
//! then compares them to Rust's. A field reordered on either side fails here.

use std::mem::{offset_of, size_of};
use std::process::Command;

use packet::dev::{DevInst64, DevProgram, StreamEnt, TraceRec, Wait, CTR_STRIDE, TENSOR_NONE16};
use packet::devbuild::{
    BlobHeader, BlobProgHeader, BlobSectionEntry, BlobTensor, BLOB_MAGIC, BLOB_MAGIC_V7, INIT_NONE,
    NAME_LEN, SECT_NAME_LEN,
};
use packet::rope::GenTensor;

/// Ask C for `sizeof`/`offsetof` of everything we mirror.
fn c_layout() -> Option<Vec<(String, usize)>> {
    let dir = std::env::temp_dir().join(format!("plow_dev_abi_{}", std::process::id()));
    std::fs::create_dir_all(&dir).ok()?;
    let src = dir.join("probe.c");
    let bin = dir.join("probe");

    let hdr = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../runtime/common/dev_blob.h"
    );
    if !std::path::Path::new(hdr).exists() {
        return None;
    }

    let probe = format!(
        r#"
#include <stddef.h>
#include <stdio.h>
#include "{hdr}"
int main(void) {{
    printf("Wait.size %zu\n", sizeof(PlowWait));
    printf("StreamEnt.size %zu\n", sizeof(PlowStreamEnt));
    printf("DevProgram.size %zu\n", sizeof(PlowProgram));
    printf("DevProgram.trace %zu\n", offsetof(PlowProgram, trace));
    printf("TraceRec.size %zu\n", sizeof(PlowTraceRec));
    printf("TraceRec.cu %zu\n", offsetof(PlowTraceRec, cu));
    printf("TraceRec.pc %zu\n", offsetof(PlowTraceRec, pc));
    printf("TraceRec.inst %zu\n", offsetof(PlowTraceRec, inst));
    printf("TraceRec.op %zu\n", offsetof(PlowTraceRec, op));
    printf("TraceRec.slice %zu\n", offsetof(PlowTraceRec, slice));
    printf("TraceRec.t_arrive %zu\n", offsetof(PlowTraceRec, t_arrive));
    printf("TraceRec.t_ready %zu\n", offsetof(PlowTraceRec, t_ready));
    printf("TraceRec.t_end %zu\n", offsetof(PlowTraceRec, t_end));
    printf("CTR_STRIDE %u\n", (unsigned)PLOW_CTR_STRIDE);
    /* The container format. It was duplicated by hand and broke twice; now it is locked. */
    printf("BlobTensor.size %zu\n", sizeof(PlowTensorDecl));
    printf("BlobTensor.name %zu\n", offsetof(PlowTensorDecl, name));
    printf("BlobTensor.bytes %zu\n", offsetof(PlowTensorDecl, bytes));
    printf("BlobTensor.init_off %zu\n", offsetof(PlowTensorDecl, init_off));
    printf("NAME_LEN %u\n", (unsigned)PLOW_NAME_LEN);
    printf("BlobHeader.size %zu\n", sizeof(PlowBlobHeader));
    printf("BlobHeader.n_cu %zu\n", offsetof(PlowBlobHeader, n_cu));
    printf("BlobHeader.n_tensor %zu\n", offsetof(PlowBlobHeader, n_tensor));
    printf("BlobHeader.n_prog %zu\n", offsetof(PlowBlobHeader, n_prog));
    printf("BlobHeader.n_kvrow %zu\n", offsetof(PlowBlobHeader, n_kvrow));
    printf("BlobHeader.init_bytes %zu\n", offsetof(PlowBlobHeader, init_bytes));
    printf("ProgHeader.size %zu\n", sizeof(PlowProgHeader));
    printf("ProgHeader.n_counter %zu\n", offsetof(PlowProgHeader, n_counter));
    printf("ProgHeader.t %zu\n", offsetof(PlowProgHeader, t));
    printf("DevInst.size %zu\n", sizeof(PlowDevInst));
    printf("DevInst.op %zu\n", offsetof(PlowDevInst, op));
    printf("DevInst.blocks %zu\n", offsetof(PlowDevInst, blocks));
    printf("DevInst.t %zu\n", offsetof(PlowDevInst, t));
    printf("DevInst.i %zu\n", offsetof(PlowDevInst, i));
    printf("DevInst.fj %zu\n", offsetof(PlowDevInst, fj));
    printf("TENSOR_NONE %u\n", (unsigned)PLOW_TENSOR_NONE);
    /* StreamEnt gates are the ONLY gates on the wire now — lock their offsets too. */
    printf("StreamEnt.wait_ofs %zu\n", offsetof(PlowStreamEnt, wait_ofs));
    printf("StreamEnt.succ_ofs %zu\n", offsetof(PlowStreamEnt, succ_ofs));
    printf("StreamEnt.wait_len %zu\n", offsetof(PlowStreamEnt, wait_len));
    printf("StreamEnt.succ_len %zu\n", offsetof(PlowStreamEnt, succ_len));
    /* v6 section directory */
    printf("SectionEntry.size %zu\n", sizeof(PlowSectionEntry));
    printf("SectionEntry.kind %zu\n", offsetof(PlowSectionEntry, kind));
    printf("SectionEntry.offset %zu\n", offsetof(PlowSectionEntry, offset));
    printf("SectionEntry.size_f %zu\n", offsetof(PlowSectionEntry, size));
    printf("SectionEntry.name %zu\n", offsetof(PlowSectionEntry, name));
    printf("SECT_NAME_LEN %u\n", (unsigned)PLOW_SECT_NAME_LEN);
    /* v7 generated-tensor recipe */
    printf("GenTensor.size %zu\n", sizeof(PlowGenTensor));
    printf("GenTensor.tensor %zu\n", offsetof(PlowGenTensor, tensor));
    printf("GenTensor.kind %zu\n", offsetof(PlowGenTensor, kind));
    printf("GenTensor.ctx %zu\n", offsetof(PlowGenTensor, ctx));
    printf("GenTensor.hd %zu\n", offsetof(PlowGenTensor, hd));
    printf("GenTensor.aux %zu\n", offsetof(PlowGenTensor, aux));
    printf("GenTensor.scale %zu\n", offsetof(PlowGenTensor, scale));
    printf("GenTensor.theta %zu\n", offsetof(PlowGenTensor, theta));
    printf("GenTensor.frac %zu\n", offsetof(PlowGenTensor, frac));
    printf("GenTensor.factor %zu\n", offsetof(PlowGenTensor, factor));
    printf("GenTensor.orig %zu\n", offsetof(PlowGenTensor, orig));
    return 0;
}}
"#
    );
    std::fs::write(&src, probe).ok()?;

    let cc = std::env::var("CC").unwrap_or_else(|_| "cc".into());
    let out = Command::new(cc)
        .arg("-o")
        .arg(&bin)
        .arg(&src)
        .output()
        .ok()?;
    if !out.status.success() {
        panic!(
            "dev_isa.h failed to compile — its own _Static_asserts may have fired:\n{}",
            String::from_utf8_lossy(&out.stderr)
        );
    }
    let run = Command::new(&bin).output().ok()?;
    let text = String::from_utf8_lossy(&run.stdout).into_owned();
    let _ = std::fs::remove_dir_all(&dir);

    Some(
        text.lines()
            .filter_map(|l| {
                let (k, v) = l.split_once(' ')?;
                Some((k.to_string(), v.trim().parse().ok()?))
            })
            .collect(),
    )
}

#[test]
fn rust_and_c_agree_on_the_device_isa() {
    let Some(c) = c_layout() else {
        // No C compiler / header: the const asserts in dev.rs still hold the line.
        eprintln!("skipping: no C toolchain or dev_isa.h not found");
        return;
    };
    let get = |k: &str| -> usize {
        c.iter()
            .find(|(n, _)| n == k)
            .unwrap_or_else(|| panic!("C probe did not report {k}"))
            .1
    };

    assert_eq!(size_of::<Wait>(), get("Wait.size"), "PlowWait size");
    assert_eq!(
        size_of::<StreamEnt>(),
        get("StreamEnt.size"),
        "PlowStreamEnt size"
    );
    assert_eq!(
        size_of::<DevProgram>(),
        get("DevProgram.size"),
        "PlowProgram size"
    );
    assert_eq!(
        offset_of!(DevProgram, trace),
        get("DevProgram.trace"),
        "DevProgram.trace"
    );
    assert_eq!(
        size_of::<TraceRec>(),
        get("TraceRec.size"),
        "PlowTraceRec size"
    );
    assert_eq!(offset_of!(TraceRec, cu), get("TraceRec.cu"), "TraceRec.cu");
    assert_eq!(offset_of!(TraceRec, pc), get("TraceRec.pc"), "TraceRec.pc");
    assert_eq!(
        offset_of!(TraceRec, inst),
        get("TraceRec.inst"),
        "TraceRec.inst"
    );
    assert_eq!(offset_of!(TraceRec, op), get("TraceRec.op"), "TraceRec.op");
    assert_eq!(
        offset_of!(TraceRec, slice),
        get("TraceRec.slice"),
        "TraceRec.slice"
    );
    assert_eq!(
        offset_of!(TraceRec, t_arrive),
        get("TraceRec.t_arrive"),
        "TraceRec.t_arrive"
    );
    assert_eq!(
        offset_of!(TraceRec, t_ready),
        get("TraceRec.t_ready"),
        "TraceRec.t_ready"
    );
    assert_eq!(
        offset_of!(TraceRec, t_end),
        get("TraceRec.t_end"),
        "TraceRec.t_end"
    );
    assert_eq!(
        CTR_STRIDE as usize,
        get("CTR_STRIDE"),
        "counter cache-line stride"
    );

    // The container format. Hand-duplicating it cost a segfault and a silent misparse that
    // ran a stale program against a fresh interpreter; the model just spoke nonsense.
    assert_eq!(
        size_of::<BlobTensor>(),
        get("BlobTensor.size"),
        "PlowTensorDecl size"
    );
    assert_eq!(
        offset_of!(BlobTensor, name),
        get("BlobTensor.name"),
        "BlobTensor.name"
    );
    assert_eq!(
        offset_of!(BlobTensor, bytes),
        get("BlobTensor.bytes"),
        "BlobTensor.bytes"
    );
    assert_eq!(
        offset_of!(BlobTensor, init_off),
        get("BlobTensor.init_off"),
        "BlobTensor.init_off"
    );
    assert_eq!(NAME_LEN, get("NAME_LEN"), "PLOW_NAME_LEN");
    assert_eq!(
        size_of::<BlobHeader>(),
        get("BlobHeader.size"),
        "PlowBlobHeader size"
    );
    assert_eq!(
        offset_of!(BlobHeader, n_cu),
        get("BlobHeader.n_cu"),
        "BlobHeader.n_cu"
    );
    assert_eq!(
        offset_of!(BlobHeader, n_tensor),
        get("BlobHeader.n_tensor"),
        "BlobHeader.n_tensor"
    );
    assert_eq!(
        offset_of!(BlobHeader, n_prog),
        get("BlobHeader.n_prog"),
        "BlobHeader.n_prog"
    );
    assert_eq!(
        offset_of!(BlobHeader, n_kvrow),
        get("BlobHeader.n_kvrow"),
        "BlobHeader.n_kvrow"
    );
    assert_eq!(
        offset_of!(BlobHeader, init_bytes),
        get("BlobHeader.init_bytes"),
        "BlobHeader.init_bytes"
    );
    assert_eq!(
        size_of::<BlobProgHeader>(),
        get("ProgHeader.size"),
        "PlowProgHeader size"
    );
    assert_eq!(
        offset_of!(BlobProgHeader, n_counter),
        get("ProgHeader.n_counter"),
        "ProgHeader.n_counter"
    );
    assert_eq!(
        offset_of!(BlobProgHeader, t),
        get("ProgHeader.t"),
        "ProgHeader.t"
    );
    // the magic must agree too, or the runtime silently accepts an old blob
    assert_eq!(
        BLOB_MAGIC, b"PLOWDEV\x07",
        "blob magic (64-byte DevInst64 format)"
    );
    assert_eq!(INIT_NONE, u64::MAX);
    assert_eq!(
        size_of::<DevInst64>(),
        get("DevInst.size"),
        "PlowDevInst size"
    );

    // Sizes matching is not enough — a swapped pair of same-width fields keeps the
    // size and silently reinterprets the operands. Check every offset.
    assert_eq!(offset_of!(DevInst64, op), get("DevInst.op"));
    assert_eq!(offset_of!(DevInst64, blocks), get("DevInst.blocks"));
    assert_eq!(offset_of!(DevInst64, t), get("DevInst.t"));
    assert_eq!(offset_of!(DevInst64, i), get("DevInst.i"));
    assert_eq!(offset_of!(DevInst64, fj), get("DevInst.fj"));
    assert_eq!(
        TENSOR_NONE16 as usize,
        get("TENSOR_NONE"),
        "PLOW_TENSOR_NONE (u16 wire sentinel)"
    );
    // StreamEnt gates are the only wire gates — a drifted offset deadlocks silently.
    assert_eq!(offset_of!(StreamEnt, wait_ofs), get("StreamEnt.wait_ofs"));
    assert_eq!(offset_of!(StreamEnt, succ_ofs), get("StreamEnt.succ_ofs"));
    assert_eq!(offset_of!(StreamEnt, wait_len), get("StreamEnt.wait_len"));
    assert_eq!(offset_of!(StreamEnt, succ_len), get("StreamEnt.succ_len"));

    // v6 section directory entry
    assert_eq!(
        size_of::<BlobSectionEntry>(),
        get("SectionEntry.size"),
        "PlowSectionEntry size"
    );
    assert_eq!(
        offset_of!(BlobSectionEntry, kind),
        get("SectionEntry.kind"),
        "SectionEntry.kind"
    );
    assert_eq!(
        offset_of!(BlobSectionEntry, offset),
        get("SectionEntry.offset"),
        "SectionEntry.offset"
    );
    assert_eq!(
        offset_of!(BlobSectionEntry, size),
        get("SectionEntry.size_f"),
        "SectionEntry.size"
    );
    assert_eq!(
        offset_of!(BlobSectionEntry, name),
        get("SectionEntry.name"),
        "SectionEntry.name"
    );
    assert_eq!(SECT_NAME_LEN, get("SECT_NAME_LEN"), "PLOW_SECT_NAME_LEN");

    // v7 generated-tensor recipe. A drifted offset here means the runtime builds
    // a RoPE table from the wrong scalars — fluent, wrong output, no error.
    assert_eq!(
        size_of::<GenTensor>(),
        get("GenTensor.size"),
        "PlowGenTensor size"
    );
    assert_eq!(offset_of!(GenTensor, tensor), get("GenTensor.tensor"));
    assert_eq!(offset_of!(GenTensor, kind), get("GenTensor.kind"));
    assert_eq!(offset_of!(GenTensor, ctx), get("GenTensor.ctx"));
    assert_eq!(offset_of!(GenTensor, hd), get("GenTensor.hd"));
    assert_eq!(offset_of!(GenTensor, aux), get("GenTensor.aux"));
    assert_eq!(offset_of!(GenTensor, scale), get("GenTensor.scale"));
    assert_eq!(offset_of!(GenTensor, theta), get("GenTensor.theta"));
    assert_eq!(offset_of!(GenTensor, frac), get("GenTensor.frac"));
    assert_eq!(offset_of!(GenTensor, factor), get("GenTensor.factor"));
    assert_eq!(offset_of!(GenTensor, orig), get("GenTensor.orig"));
    assert_eq!(
        BLOB_MAGIC_V7, b"PLOWDEV\x09",
        "v7 blob magic (generated tensors)"
    );
}
