#!/usr/bin/env python3
"""gfx950_objects.py — which gfx950 code objects a packet needs, and the -D's that build them.

THE GAP THIS CLOSES. build.json's `backends` map has one renderer, `nvcc`, so a manifest can
tell a builder which -D's an sm_120 cubin needs and cannot say anything at all about gfx950.
With 34 objects, and combinations like PLOW_MLA_PREFILL + PLOW_MOE_PREFILL + PLOW_MOE_PF_A4W4 +
PLOW_MXFP4 now having to land in ONE object for an all-MXFP4 Kimi packet, that is exactly the
drift the cmake consolidation was built to prevent — reintroduced one level up.

SINGLE SOURCE OF TRUTH. This does NOT restate the define sets: it PARSES the `_hs_ax_*` axis
variables and the `_hs_rows` table out of runtime/CMakeLists.txt, which is the same table the
build actually uses. A row added there is visible here with no edit, and a define changed there
cannot drift from what this reports. Restating them in a second place is the failure mode, not
the fix.

  ./scripts/gfx950_objects.py                       # every object and its defines
  ./scripts/gfx950_objects.py build-amd/g31b-fp8/build.json   # just what THIS packet needs
  ./scripts/gfx950_objects.py --cover build-amd/g31b-w8a8/build.json   # ...and check every
                                                    # opcode it emits has an arm in them

--cover IS THE POINT OF THIS FILE NOW. Four times on this branch an arm existed, was correct,
was register-gated, and had NOTHING ROUTING TO IT: an unreachable GEMM_MXFP4 behind a false
`#if`; fp8 flash segments misclassified by wave class; the fp8 prefill object never selected;
and — the one that cost the most — the bf16 `GemmSmall` lm_head every w8a8 packet still emits,
dropped out of interp_prefill_fp8 by a `#if PLOW_FP8` that SWAPPED instead of adding, so the
logits were all zero and every prompt sampled token id 0. Each was invisible to reading either
side alone. --cover preprocesses interp.hip under each selected object's REAL define set,
collects the `case PLOW_DOP_*` labels that survive, and reports any packet opcode with no arm.
It is the mechanical form of "for each arm, what selects it, and is that selector complete over
precisions?" — and it needs hipcc, so it is a check you run, not an import.

SELECTION COMES FROM THE MANIFEST, NOT FROM A MAP HERE. devgen's `backend_gfx950` renderer
(crates/devgen/src/manifest.rs) writes `backends.gfx950.requires` — the packet's own flag set —
and this file matches cmake rows against it. It used to compute the same answer from `features`
with a hand-written ladder, and the two drifted exactly as predicted: the ladder keyed the MLA
prefill object off the `mla` feature, so a decode-only GLM block was told it needed
`interp_prefill_mla` when it has no prefill program at all. The renderer keys off
`has("FlashMlaPrefill")` and was right. The ladder survives only as a fallback for manifests
emitted before the renderer existed.
"""
import json, re, sys, os, subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CMAKE = os.path.join(ROOT, "runtime", "CMakeLists.txt")
HIPCC = os.environ.get("PLOW_HSACO_HIPCC", "/opt/rocm/bin/hipcc")


def parse_cmake(path):
    """-> (axes: name -> [defines], rows: [(stem, symbol, axis1, axis2)])"""
    src = open(path).read()
    axes, rows = {}, []
    for m in re.finditer(r"set\((_hs_ax_\w+)\s+([^)]*)\)", src):
        toks = m.group(2).split()
        out = []
        for t in toks:
            t = t.strip()
            if t.startswith("${") and t.endswith("}"):      # axis composed of other axes
                out += axes.get(t[2:-1], [])
            elif t.startswith("-D"):
                out.append(t)
        axes[m.group(1)] = out
    for m in re.finditer(r'"(\w+)\|\$\{(\w+)\}\|(\w+)\|(\w+)\|', src):
        rows.append((m.group(1), m.group(2), m.group(3), m.group(4)))
    return axes, rows


def defines_for(axes, a1, a2):
    return axes.get(a1, []) + axes.get(a2, [])


def arms_in(defines):
    """The PLOW_DOP_* arms that SURVIVE the preprocessor under `defines`.

    Preprocessed, never grepped from the source: every guard in interp.hip is an `#if` over the
    precision/bucket flags, so the only honest answer to "does this object carry op X" comes from
    running the preprocessor with that object's real define set. The BUCKET_FLASH object has no
    switch at all — it is an if-chain on `in->op` — hence the second pattern."""
    cmd = [HIPCC, "--offload-arch=gfx950", "-E", "-w",
           "-I" + os.path.join(ROOT, "runtime", "amd"),
           "-I" + os.path.join(ROOT, "runtime", "common")] + defines + \
          [os.path.join(ROOT, "runtime", "amd", "interp.hip")]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise SystemExit(f"hipcc -E failed for {' '.join(defines)}:\n{p.stderr[-2000:]}")
    a = set(re.findall(r"case (PLOW_DOP_\w+):", p.stdout))
    a |= set(re.findall(r"in->op == (PLOW_DOP_\w+)", p.stdout))
    return {s[len("PLOW_DOP_"):] for s in a}


def key(name):
    """Fold a name to a comparison key. The emitter spells opcodes CamelCase (`GemvGluFp8`) and
    the ISA header SNAKE_CASE (`PLOW_DOP_GEMV_GLU_FP8`), and the word breaks do NOT agree
    (`RmsNorm`/`RMSNORM`, `SoftCap`/`SOFTCAP`, `HeadNormRope`/`HEADNORM_ROPE`). Dropping the
    separators entirely is the only mapping that needs no hand-written table — and a table is
    exactly the second source of truth this file exists to avoid."""
    return name.replace("_", "").upper()


def stems_from_requires(axes, rows, req, opcodes):
    """Select objects from the manifest's OWN `backends.gfx950.requires`.

    This replaces a hand-maintained FEATURE->AXIS map that had already drifted: it keyed the MLA
    prefill object off the `mla` feature, so a decode-only GLM block — which has no prefill
    program at all — was told it needed `interp_prefill_mla`. devgen's `backend_gfx950` renderer
    keys the same decision off `has("FlashMlaPrefill")` and got it right. Two maps for one
    decision is the drift this file exists to prevent, so it now consumes the renderer's answer
    instead of computing a second one.

    MATCHING IS BEST-SUBSET PER BUCKET, not equality. `requires` is a packet-level flag set; a
    row's axis defines are that packet's flags restricted to what THAT object compiles. The
    decode object never takes PLOW_MLA_PREFILL or PLOW_MOE_PREFILL (prefill-only axes) and
    PLOW_W8A8 is an EMIT flag with no kernel define at all, so equality would match nothing.
    A row is a candidate if every define it carries is required; the most specific candidate
    wins. That is the same "most specific wins" the old ladder encoded, derived rather than
    written down."""
    want = {r.lstrip("-D") for r in req}
    def pick(bucket):
        best, best_n = None, -1
        for stem, sym, a1, a2 in rows:
            d = {x.lstrip("-D") for x in defines_for(axes, a1, a2)}
            if f"PLOW_BUCKET_DECODE={bucket}" not in d or any("BUCKET_FLASH" in x for x in d):
                continue
            rest = d - {f"PLOW_BUCKET_DECODE={bucket}"}
            if not rest <= want:          # carries an arm this packet did not ask for
                continue
            if len(rest) > best_n:
                best, best_n = stem, len(rest)
        return [best] if best else []
    # A packet with no prefill buckets emits no `PLOW_BUCKET_DECODE=0` and needs no prefill object.
    out = (pick(0) if "PLOW_BUCKET_DECODE=0" in want else []) + pick(1)
    # CLASS-4 FLASH object, keyed on the OPCODE — never on a precision flag. That rule is not a
    # style preference: the classifier that tested a flag instead of the opcode found 0 of 60
    # flash segments on an fp8-KV packet.
    if "FlashPrefillFp8" in opcodes:
        out += ["interp_flash_fp8kv"]
    elif "FlashPrefill" in opcodes:
        out += ["interp_flash"]
    return out


# FEATURE -> AXIS. LEGACY: used only for manifests with no `backends.gfx950` block (emitted
# before the renderer landed). New manifests take `stems_from_requires` above.
def needed_stems(features, opcodes):
    op = set(opcodes)
    fp8 = features.get("fp8_weights") or features.get("w8a8")
    fp8kv = features.get("fp8_kv")
    mla = features.get("mla")
    moe_pf = features.get("moe_prefill")
    a4w4 = features.get("a4w4")
    # EVERY mxfp4 tile rung. The prefill fp4 GEMM used to be one opcode pinned to 256x256; it is
    # tile-selected now, so a packet's only fp4 GEMM may be GemmSmallMxfp4 (Kimi kv_a_proj) and
    # naming just GemmMxfp4 here would classify that packet bf16 and hand it interp_prefill,
    # whose `default:` writes NOTHING rather than trapping. Keep in sync with
    # `devgen::manifest`'s classifier, which decides the same thing for new manifests.
    mxfp4 = bool(op & {"GemvMxfp4", "GemvGluMxfp4", "GemmMxfp4", "GemmMedMxfp4",
                       "GemmSmallMxfp4", "GemmWideMxfp4", "GemmC5Mxfp4"})
    pf, dec = [], []
    # PREFILL object: one row, most specific wins.
    if mla and moe_pf and a4w4 and mxfp4: pf = ["interp_prefill_mla_moe_a4w4_full"]
    elif mla and moe_pf and a4w4:         pf = ["interp_prefill_mla_moe_a4w4"]
    elif mla and moe_pf and fp8:          pf = ["interp_prefill_mla_moe_fp8"]
    elif mla and moe_pf:                  pf = ["interp_prefill_mla_moe"]
    elif mla and fp8:                     pf = ["interp_prefill_mla_fp8"]
    elif mla:                             pf = ["interp_prefill_mla"]
    elif fp8kv:                           pf = ["interp_prefill_fp8kv"]
    elif fp8:                             pf = ["interp_prefill_fp8"]
    elif mxfp4:                           pf = ["interp_prefill_mxfp4"]
    else:                                 pf = ["interp_prefill"]
    # DECODE object.
    if fp8kv:    dec = ["interp_decode_fp8kv"]
    elif mxfp4:  dec = ["interp_decode_mxfp4"]
    elif fp8:    dec = ["interp_decode_fp8"]
    else:        dec = ["interp_decode"]
    # CLASS-4 FLASH object, iff the packet emits a flash_prefill in EITHER precision. The fp8
    # twin is a distinct object: an fp8-KV packet emits FLASH_PREFILL_FP8 and never the bf16 one,
    # and interp_flash carries only the arm it was built with.
    fl = []
    if "FlashPrefillFp8" in op: fl = ["interp_flash_fp8kv"]
    elif "FlashPrefill" in op:  fl = ["interp_flash"]
    return pf + dec + fl


def stems_for(features, opcodes, req):
    """The renderer's answer when the manifest has one, the legacy map when it does not."""
    if req is None:
        return needed_stems(features, opcodes)
    axes, rows = parse_cmake(CMAKE)
    return stems_from_requires(axes, rows, req, opcodes)


def gfx950_requires(man):
    b = man.get("backends", {}).get("gfx950")
    return None if b is None else b.get("requires", [])


def stems_by_phase(features, opcodes, req=None):
    """The same selection, split by which PHASE's programs each object serves.

    --cover needs this because the union over all three objects hides the failure it most needs
    to catch. `Gemv` (op 10) lives in the DECODE object and, until this was found, in no prefill
    object at all — so when devgen started emitting the prefill lm_head as `Gemv`, a union check
    saw the opcode covered and passed while the prefill program's lm_head fell to `default:` and
    wrote zeros. An arm is only covered if it is in an object that runs THAT program."""
    s = stems_for(features, opcodes, req)
    pf = [x for x in s if x.startswith("interp_prefill")]
    dec = [x for x in s if x.startswith("interp_decode")]
    fl = [x for x in s if x.startswith("interp_flash")]
    # The class-4 flash object runs PREFILL segments, so it counts toward prefill coverage.
    return {"prefill": pf + fl, "decode": dec}


def main():
    args = sys.argv[1:]
    cover = "--cover" in args
    args = [a for a in args if a != "--cover"]
    axes, rows = parse_cmake(CMAKE)
    by_stem = {r[0]: r for r in rows}
    man = None
    if args:
        man = json.load(open(args[0]))
        req = gfx950_requires(man)
        stems = stems_for(man.get("features", {}), man.get("opcodes", []), req)
        src = "backends.gfx950.requires" if req is not None else "legacy FEATURE->AXIS map"
        print(f"# {args[0]} -> {len(stems)} gfx950 object(s)   [selected from {src}]")
    else:
        stems = [r[0] for r in rows]
    miss = [s for s in stems if s not in by_stem]
    for s in stems:
        if s not in by_stem:
            print(f"{s:36s}  ** NO CMAKE ROW — object cannot be built **")
            continue
        stem, sym, a1, a2 = by_stem[s]
        print(f"{stem:36s} {' '.join(defines_for(axes, a1, a2))}")
        print(f"{'  (+ global-queue twin ' + stem + '_gq)':36s} "
              f"{' '.join(defines_for(axes, a1, a2) + axes.get('_hs_ax_gq', []))}")
    if not cover:
        return 1 if miss else 0
    if man is None:
        raise SystemExit("--cover needs a build.json (it checks a PACKET against its objects)")
    # PER PHASE, not the union over every selected object.
    #
    # The union version of this check passed the very packet it was written to catch. `Gemv`
    # (op 10) was in the DECODE object and in no prefill object at all; when devgen moved the
    # prefill lm_head from `GemmSmall` to `Gemv`, the union saw the opcode covered while the
    # prefill program's lm_head fell to `default:` and wrote zeros — the same silent-zeros bug,
    # one opcode over. An arm is covered only if it is in an object that runs THAT program.
    #
    # The phase split comes from the manifest's own `programs[].kind`/`arms`, so a packet whose
    # buckets shift ops between phases is followed automatically. `arms` carry a shape suffix
    # (`FlashPrefill/hd256`) — that is a template parameter, not an opcode, so it is dropped.
    phases = stems_by_phase(man.get("features", {}), man.get("opcodes", []), gfx950_requires(man))
    progs = man.get("programs", [])
    if not progs:
        raise SystemExit("build.json has no `programs`; cannot check coverage per phase")
    cache, fails, checked = {}, [], 0
    for phase, ss in phases.items():
        want = set()
        for p in progs:
            if p.get("kind") == phase:
                want |= {a.split("/")[0] for a in p.get("arms", [])}
        if not want:
            continue
        have = set()
        for s in ss:
            if s not in by_stem:
                continue
            _, _, a1, a2 = by_stem[s]
            d = tuple(defines_for(axes, a1, a2))
            if d not in cache:
                cache[d] = arms_in(list(d))
            have |= {key(c) for c in cache[d]}
        gap = sorted(a for a in want if key(a) not in have)
        checked += len(want)
        print(f"\n{phase:8s} programs emit {len(want):2d} arm(s); objects {', '.join(ss)}")
        if gap:
            fails.append((phase, gap))
            print(f"  ** NO ARM IN ANY {phase.upper()} OBJECT: {', '.join(gap)}")
            print(f"     These hit the interpreter's `default:` and write NOTHING.")
        else:
            print(f"  all covered")
    if fails:
        return 2
    print(f"\ncover: OK — every arm of every program has a dispatch in its phase's object "
          f"({checked} arm-slots checked)")
    return 1 if miss else 0


if __name__ == "__main__":
    sys.exit(main())
