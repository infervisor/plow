// Interior-range RoPE indexing, checked on the host because it does not need a GPU to be settled.
//
// `d_qwen_headnorm_rope_t` holds a head row lane-strided: element `lane + j*PLOW_WAVE` lives in
// register `x[j]`. The rotary section used to be hardcoded to `x[0]`, which is the PREFIX form --
// the only one Qwen emits. DeepSeek-V4.1 rotates the SUFFIX: `apply_rotary_emb(q[..., -rd:], ...)`
// at `inference/model.py:772`, i.e. elements [448, 512) of a 512-wide head.
//
// The whole change is that the register index becomes `rot_offset / PLOW_WAVE`. What has to be
// true for that to be correct is pure integer arithmetic, so it is checked here rather than behind
// a GPU lease:
//
//   REGISTER:  every element of the rotary range lands in ONE register, and it is `rreg`.
//   PARTNER:   the rope partner of element `e` is `e + rotary/2` WITHIN the range, and the
//              `__shfl_xor(x[rreg], half)` exchange reaches exactly that element -- same register,
//              lane `lane ^ half`.
//   DISJOINT:  no element OUTSIDE [rot_offset, rot_offset + rotary) is touched. A suffix rope that
//              quietly rotated register 0 as well would leave the nope half rotated, and attention
//              over a wrongly-rotated nope half still produces fluent output.
//   PREFIX:    at rot_offset = 0 the mapping is identical to the old hardcoded `x[0]`, so every
//              Qwen packet is byte-identical.
//
// WHAT THIS DOES NOT PROVE. It checks the FORMULA `rreg = rot_offset / PLOW_WAVE` and the range
// arithmetic around it; it re-derives that formula rather than reading the kernel, so it would not
// catch someone hardcoding `x[0]` again. That is the compiler's job (the parameter is threaded and
// unused-arg warnings are on) and the gfx942 numerics run's. What it DOES settle, without a lease,
// is that 448 is the right register, that the shuffle partner is the reference's partner, and that
// nothing outside [448, 512) is reachable -- the last being the one that produces fluent garbage.
//
// Build and run (no GPU, no ROCm):
//     c++ -O2 -std=c++17 -o /tmp/ropeoff runtime/tests/dsv41_rope_offset_test.cpp && /tmp/ropeoff
#include <cstdio>
#include <cstdlib>
#include <vector>

static const unsigned WAVE = 64;

static int failures = 0;
static long checks = 0;
#define CHECK(cond, ...)                                                                           \
    do {                                                                                           \
        ++checks;                                                                                  \
        if (!(cond)) {                                                                             \
            if (failures < 20) { printf("FAIL: "); printf(__VA_ARGS__); printf("\n"); }            \
            ++failures;                                                                            \
        }                                                                                          \
    } while (0)

// The kernel's own mapping, transliterated: which (register, lane) holds element `e` of a row.
static inline unsigned reg_of(unsigned e) { return e / WAVE; }
static inline unsigned lane_of(unsigned e) { return e % WAVE; }

// One case: a `dim`-wide head rotating `rotary` elements starting at `rot_offset`.
static void one(unsigned dim, unsigned rotary, unsigned rot_offset) {
    const unsigned rreg = rot_offset / WAVE;
    const unsigned half = rotary >> 1;

    // REGISTER: the whole rotary range is in `rreg`, and the lanes it uses are [0, rotary).
    for (unsigned k = 0; k < rotary; k++) {
        const unsigned e = rot_offset + k;
        CHECK(reg_of(e) == rreg, "dim=%u off=%u: element %u is in reg %u, not %u", dim, rot_offset,
              e, reg_of(e), rreg);
        CHECK(lane_of(e) == k, "dim=%u off=%u: element %u is lane %u, expected %u", dim, rot_offset,
              e, lane_of(e), k);
        // The kernel guards with `lane < rotary`, so the range must start at lane 0.
        CHECK(lane_of(rot_offset) == 0, "dim=%u off=%u does not start at lane 0", dim, rot_offset);
    }

    // PARTNER: `__shfl_xor(x[rreg], half)` gives lane `l ^ half` of the SAME register, which must
    // be the reference's partner element `e +/- half`.
    for (unsigned k = 0; k < rotary; k++) {
        const unsigned e = rot_offset + k;
        const unsigned lane = lane_of(e);
        const unsigned partner_lane = lane ^ half;
        const unsigned partner_e = rreg * WAVE + partner_lane;
        const unsigned want = (k < half) ? e + half : e - half;
        CHECK(partner_e == want, "dim=%u off=%u: element %u partners %u, reference wants %u", dim,
              rot_offset, e, partner_e, want);
        // And the partner is inside the rotary range, never in the nope half.
        CHECK(partner_e >= rot_offset && partner_e < rot_offset + rotary,
              "dim=%u off=%u: element %u partners %u, OUTSIDE the rotary range", dim, rot_offset, e,
              partner_e);
    }

    // DISJOINT: every element outside the range is in a different register, or in `rreg` at a lane
    // the `lane < rotary` guard excludes. Either way the kernel leaves it alone.
    for (unsigned e = 0; e < dim; e++) {
        const bool in_range = e >= rot_offset && e < rot_offset + rotary;
        if (in_range) continue;
        const bool touched = (reg_of(e) == rreg) && (lane_of(e) < rotary);
        CHECK(!touched, "dim=%u off=%u rotary=%u: element %u is OUTSIDE the range but the kernel "
                        "would rotate it",
              dim, rot_offset, rotary, e);
    }
}

int main() {
    // Qwen: the prefix form, which must keep selecting register 0.
    for (unsigned dim : {64u, 128u, 192u, 256u}) {
        one(dim, 64, 0);
        CHECK(0 / WAVE == 0, "prefix must select register 0");
    }
    // DeepSeek-V4.1: a 512-wide head rotating its LAST 64 -- register 7.
    one(512, 64, 448);
    CHECK(448 / WAVE == 7, "V4.1's rope range is register 7");
    CHECK(448 % WAVE == 0, "and it starts on a register boundary");
    CHECK(512 - 64 == 448, "nope = head_dim - qk_rope");

    // Every legal offset on a 512-wide head, at both legal rotary widths.
    for (unsigned off = 0; off + 64 <= 512; off += WAVE) one(512, 64, off);
    for (unsigned off = 0; off + 32 <= 512; off += WAVE) one(512, 32, off);

    // A suffix rope that had kept the hardcoded `x[0]` would rotate the NOPE half instead. Pin the
    // difference so the bug cannot come back looking like a refactor.
    CHECK(reg_of(448) != reg_of(0), "the V4.1 range and register 0 must be different registers");

    printf("%s: %ld checks, %d failures\n", failures ? "FAILED" : "PASSED", checks, failures);
    return failures ? 1 : 0;
}
