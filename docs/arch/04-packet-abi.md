# 04 — Packet ABI

> A packet (`model.pkt`, and sidecars such as `encoder.pkt` or `codec.pkt`) is the binary
> contract between the compiler and the runtime: a tensor table, compiled device programs and
> named metadata sections. One generic runtime loads every packet; everything model-specific is
> in the packet.

---

## Where the format is defined

| Piece | Definition |
|-------|------------|
| Container (header, tensor declarations, program headers, section directory) | `runtime/common/dev_blob.h`, mirrored by `#[repr(C)]` structs in `crates/packet/src/devbuild.rs` |
| Instruction and stream records (`PlowDevInst` 64 B, `PlowStreamEnt` 24 B, `PlowWait` 8 B) | `runtime/common/dev_isa.h`, mirrored in `crates/packet/src/dev.rs` |
| Generated-tensor recipes (`PlowGenTensor` 72 B) | `runtime/common/dev_blob.h`, `crates/packet/src/rope.rs` |
| Metadata section payloads | `crates/plow-asset` (one module per section) |
| Writer | `packet::devbuild::Model::to_blob*` (compiler) |
| Reader | `plowrt::asset::devblob::DevBlob::parse` (runtime) |

Record sizes and field offsets are locked against the C headers by `crates/packet/tests/dev_abi.rs`.
Everything is little-endian. `include/packet.h` and `crates/packet/src/lib.rs` describe the older
variable-length instruction stream; the runtime does not load that format.

---

## Container layout

```
BlobHeader (64 B)
TensorDecl[n_tensor] (96 B each: name[80], bytes u64, init_off u64)
init bytes[init_bytes]
kvrow u32[n_kvrow]
per program:
    ProgHeader (24 B: n_inst, n_stream, n_wait, n_succ, n_counter, t)
    DevInst64[n_inst]
    StreamEnt[n_stream]
    stream_ofs u32[n_cu], stream_len u32[n_cu]
    Wait[n_wait]
    succ u32[n_succ]
optional "GQ01" appendix: per program { n_seg u32, gq_stream StreamEnt[n_stream], gq_seg_ofs u32[n_seg+1] }
section data (contiguous)
section directory: "SECT", count u32, SectionEntry[count] (48 B each)
```

### Header

| Field | Meaning |
|-------|---------|
| `magic[8]` | `PLOWDEV` + version byte. `\x07` plain, `\x08` sectioned, `\x09` sectioned with generated tensors, `\x0a`/`\x0b` the same with combined L2 segment placement. An extension container has its own magic (`packet::ext`). |
| `n_cu` | Workgroups each program's streams are laid out for (one stream per CU). |
| `n_tensor`, `n_prog`, `n_kvrow`, `init_bytes` | Table sizes. |
| `flags` | `1` global-queue appendix present, `2` L2-domain placement, `4` L2 placement keeps segments in `seg`. A reader refuses any other bit. |
| `target` | Fingerprint of the compile target; informational. |
| `reserved[0]` | Byte offset of the section directory (0 = none). |
| `reserved[1]`, `reserved[2]` | SMs per partition and L2 domain count for placed packets. |

### Programs

A program is a compiled schedule for one row count (`t`): a prefill bucket, a decode rung, or a
role-tagged variant (packed sibling, token-batch body, dense-exact rung, row-split sibling, modular
block). Role tags live in the high bits of `t`; the reader derives `ProgramRole` once at parse.
`DevInst64` carries the opcode, block count, eight 16-bit tensor handles (`0xFFFF` = absent), eight
32-bit integers and three float/integer words. `StreamEnt` names an instruction and its slice, its
wait and successor lists, flags and segment. Opcode numbers are stable; a packet using an opcode the
runtime does not know is refused.

### Sections

`SectionEntry { kind u32, flags u32, offset u64, size u64, name[24] }`. Kinds: `0` programs (unused),
`1` CUDA object, `2` HSA object, `3` weight map, `4` named metadata, `5` static tensors, `6`
generated-tensor recipes, `7` extension parent reference. `flags` bit 0 is MUST_UNDERSTAND: a reader
that does not know the section's kind refuses the packet instead of skipping it; every other bit is
reserved and refused.

Named metadata sections (kind 4) carry the runtime contract of features that do not fit the fixed
records, for example: `live_kv` (KV cache layout), `packed_prefill`, `token_batch`,
`object_requirements` (object symbols each program needs), `recurrent_state`, `kv_shared_tail`,
`decode_objects.json`, `decode_context.json`, `segment_roles.json`, `derived_weights.json`,
`weight_shards.json`, `packet_pipeline.json` (speech and multimodal pipelines), `serve.json`
(chat, tool and serving defaults), `spec.json`, `spec.context.json` and `spec.draft.programs`
(speculative drafting). Each is defined and validated by its `plow-asset` module.

`abi.json` (`plow_asset::abi`, appended after every other section) states what the packet needs
from the runtime, so a missing `build.json` cannot drop it:

| Block | Reader |
|-------|--------|
| `contract` | refuses a contract above its own. |
| `must_understand` | metadata sections the packet carries whose semantics a runtime must implement (`plow_asset::abi::MUST_UNDERSTAND` at the writer's commit); refuses a name it does not know or one the packet lacks. |
| `target` | the instruction set (`sm_90a`, `gfx942`, ...) and `n_cu`; refuses only an instruction-set mismatch (an `a` target needs its exact compute capability, a plain or `f` one the same major and an equal or higher minor). The product it was tuned on is logged, never refused. |
| `bindings` | the tensor of each fixed runtime role and every KV-cache write site; must equal what the tensor names imply, or the packet is refused. |

A packet without `abi.json` loads as before.

### Generated tensors

Tables that are a pure function of a few scalars (RoPE cos/sin, tensor-map descriptors) are carried
as `GenTensor` recipes instead of bytes. The runtime materialises them at bind time with the same
host code the compiler uses, so the bytes are identical. A recipe whose kind or scaling scheme this
runtime cannot produce refuses the packet.

---

## Validation at load

`DevBlob::parse` checks, before any device work:

- every table lies inside the file (checked arithmetic on every count and offset);
- every stream entry names an instruction of its program, every wait/successor list lies inside the
  program's tables and names a counter the program declares (local counters), every GQ window is
  ordered and inside the stream;
- every tensor handle is declared (an extension's handles are checked against its parent at merge);
- the kv-row sites lie inside the decode program they patch;
- sections lie between the program area and the directory, do not overlap, and a (kind, name) pair
  appears once;
- each generated-tensor recipe targets a distinct, runtime-filled tensor of exactly its size;
- header flag bits and section flags are known;
- `abi.json`, when present, as described above.

A packet is read, parsed and hashed once per load (`plowrt::asset::packet_image`); every consumer of
that load shares the result.

---

## Requirements outside the packet

`build.json` beside the packet records how it was compiled: the knob certificate, the object
pairing hash (`pairing.hash`, matched against specialised interpreter objects), and
`runtime_requires` (`plowrt_contract`, minimum library versions). A missing `build.json` requires
nothing; an unreadable or malformed one refuses the load.

Packets cross machines through the distribution store (`docs/DISTRIBUTION.md`). A packet compiled
for one GPU loads on any GPU of a compatible instruction set whose CU count matches the packet's
`n_cu` (an H100 SXM packet runs on H200 SXM); memory size, L2 size and product name never refuse it.
