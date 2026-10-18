/* moe_route_key.h: one-pass rank select vs the two serial k-pass references the GPU routers
 * replaced — Gemma's (score kill sentinel, 64-bit packed key) and AMD's (key zeroed on pick,
 * group mask on the packed key) — over ties, signed zeros, domain extremes and group masks. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "moe_route_key.h"

#define NMAX 1024u
#define KMAX 16u

static uint64_t rs = 0x9E3779B97F4A7C15ull;
static uint32_t ru(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (uint32_t)(rs >> 32); }
static float uf(float lo, float hi) { return lo + (hi - lo) * (float)(ru() >> 8) * (1.0f / 16777216.0f); }
static int fails = 0, cases = 0;
#define CHECK(c, ...) do { if (!(c)) { fails++; printf("FAIL: " __VA_ARGS__); printf("\n"); } } while (0)

typedef struct {
    uint32_t n_exp, k, n_group, topk_group;
    int norm;
    float scale;
} Cfg;

/* kept[e]: expert survives the group mask (all 1 without groups). */
static void group_mask(const Cfg* c, const float* sel, unsigned char* kept) {
    memset(kept, 1, c->n_exp);
    if (!plow_route_group_active(c->n_group, c->topk_group)) return;
    const uint32_t gsz = c->n_exp / c->n_group;
    uint32_t gord[PLOW_ROUTE_MAX_GROUPS];
    uint64_t gk[PLOW_ROUTE_MAX_GROUPS];
    for (uint32_t g = 0; g < c->n_group; g++) {
        gord[g] = plow_route_ord(plow_route_group_top2(sel + g * gsz, gsz));
        gk[g] = ((uint64_t)gord[g] << 8) | (uint64_t)((c->n_group - 1u - g) & 0xFFu);
    }
    for (uint32_t g = 0; g < c->n_group; g++) {
        uint32_t r = 0, r64 = 0;
        for (uint32_t f = 0; f < c->n_group; f++) {
            r += (uint32_t)plow_route_beats(gord[f], f, gord[g], g);
            r64 += gk[f] > gk[g];
        }
        CHECK(r == r64, "group rank (ord,id) %u != packed %u", r, r64);
        if (r >= c->topk_group) memset(kept + g * gsz, 0, gsz);
    }
}

/* AMD/Gemma serial tail, written out independently of the header. */
static void ref_gates(const Cfg* c, const float* score, const uint32_t* win, float* out) {
    float sum = 0.0f;
    for (uint32_t j = 0; j < c->k; j++) sum += score[win[j]];
    for (uint32_t j = 0; j < c->k; j++) {
        float g = score[win[j]];
        if (c->norm && sum != 0.0f) g /= sum;
        out[j] = g * c->scale;
    }
}

static void gates(const Cfg* c, const float* score, const uint32_t* win, float* out) {
    float g[KMAX];
    for (uint32_t j = 0; j < c->k; j++) g[j] = score[win[j]];
    const float s = plow_route_gate_sum(g, c->k);
    for (uint32_t j = 0; j < c->k; j++) out[j] = plow_route_gate(g[j], s, c->norm, c->scale);
}

/* Gemma serial body: rescan packed keys of the live scores, kill the winner's score. */
static void ref_sentinel(const Cfg* c, const float* sel, const unsigned char* kept, uint32_t* win) {
    float sc[NMAX];
    memcpy(sc, sel, c->n_exp * sizeof(float));
    for (uint32_t j = 0; j < c->k; j++) {
        uint64_t best = 0;
        uint32_t bid = 0;
        for (uint32_t e = 0; e < c->n_exp; e++) {
            const uint64_t key = plow_route_key64(kept[e] ? plow_route_ord(sc[e]) : 0u, e, c->n_exp);
            if (key > best) { best = key; bid = e; }
        }
        win[j] = bid;
        sc[bid] = PLOW_ROUTE_KILL;
    }
}

/* AMD block_max rounds: max packed key, zero the winner's key. */
static void ref_zero(const Cfg* c, const float* sel, const unsigned char* kept, uint32_t* win) {
    uint64_t keys[NMAX];
    for (uint32_t e = 0; e < c->n_exp; e++)
        keys[e] = plow_route_key64(kept[e] ? plow_route_ord(sel[e]) : 0u, e, c->n_exp);
    for (uint32_t j = 0; j < c->k; j++) {
        uint64_t best = 0;
        for (uint32_t e = 0; e < c->n_exp; e++) best = keys[e] > best ? keys[e] : best;
        const uint32_t eid = c->n_exp - 1u - (uint32_t)(best & 0xFFFFFu);
        win[j] = eid;
        keys[eid] = 0;
    }
}

static void rank_select(const Cfg* c, const float* sel, const unsigned char* kept, uint32_t* win) {
    uint32_t ord[NMAX];
    for (uint32_t e = 0; e < c->n_exp; e++) ord[e] = kept[e] ? plow_route_ord(sel[e]) : 0u;
    for (uint32_t j = 0; j < c->k; j++) win[j] = ~0u;
    for (uint32_t e = 0; e < c->n_exp; e++) {
        const uint32_t r = plow_route_rank(ord, c->n_exp, e);
        if (r < c->k) {
            CHECK(win[r] == ~0u, "rank %u assigned twice", r);
            win[r] = e;
        }
    }
}

/* mode: 0 random, 1 integer ties, 2 all equal, 3 signed zeros, 4 domain extremes, 5 sigmoid. */
static void fill(float* score, float* bias, uint32_t n, int mode, int with_bias) {
    for (uint32_t e = 0; e < n; e++) {
        float v = uf(-4.0f, 4.0f);
        if (mode == 1) v = (float)(int)v;
        if (mode == 2) v = 0.25f;
        if (mode == 3) v = (ru() & 1u) ? -0.0f : 0.0f;
        if (mode == 4) {
            static const float ext[] = {-1e29f, 1e29f, -0.0f, 0.0f, 1e-45f, -1e-45f, 3e38f, -9e29f};
            v = ext[ru() & 7u];
        }
        if (mode == 5) v = 1.0f / (1.0f + __builtin_expf(-v));
        score[e] = v;
        bias[e] = (with_bias && mode != 3 && mode != 4) ? uf(-0.05f, 0.05f) : 0.0f;
    }
}

static void run(const Cfg* c, int mode, int with_bias) {
    float score[NMAX], bias[NMAX], sel[NMAX];
    unsigned char kept[NMAX];
    uint32_t wr[KMAX], ws[KMAX], wz[KMAX];
    float gr[KMAX], gs[KMAX];
    fill(score, bias, c->n_exp, mode, with_bias);
    uint32_t nneg0 = 0, npos0 = 0;
    for (uint32_t e = 0; e < c->n_exp; e++) {
        sel[e] = with_bias ? score[e] + bias[e] : score[e];
        nneg0 += plow_route_ord(sel[e]) == plow_route_ord(-0.0f);
        npos0 += plow_route_ord(sel[e]) == plow_route_ord(0.0f);
        CHECK(plow_route_in_domain(sel[e]), "generated sel %g outside the domain", (double)sel[e]);
    }
    if (mode == 3 && !with_bias) CHECK(nneg0 && npos0, "signed zeros did not reach selection");
    group_mask(c, sel, kept);
    rank_select(c, sel, kept, wr);
    ref_sentinel(c, sel, kept, ws);
    ref_zero(c, sel, kept, wz);
    gates(c, score, wr, gr);
    ref_gates(c, score, ws, gs);
    cases++;
    CHECK(!memcmp(wr, ws, c->k * 4) && !memcmp(wr, wz, c->k * 4),
          "n_exp %u k %u groups %u/%u mode %d bias %d: rank != k-pass", c->n_exp, c->k, c->n_group,
          c->topk_group, mode, with_bias);
    CHECK(!memcmp(gr, gs, c->k * 4), "n_exp %u k %u mode %d: gates differ", c->n_exp, c->k, mode);
}

int main(void) {
    CHECK(plow_route_ord(-0.0f) < plow_route_ord(0.0f), "-0 must order below +0");
    CHECK(plow_route_ord(PLOW_ROUTE_KILL) < plow_route_ord(-1e29f), "kill sentinel above a live key");
    CHECK(plow_route_ord(-1.0f) < plow_route_ord(-0.5f) && plow_route_ord(0.5f) < plow_route_ord(1.0f),
          "ord not monotone");
    const float nan = __builtin_nanf(""), inf = __builtin_inff();
    CHECK(!plow_route_in_domain(nan) && !plow_route_in_domain(-nan) && !plow_route_in_domain(inf) &&
              !plow_route_in_domain(-inf) && !plow_route_in_domain(PLOW_ROUTE_KILL) &&
              !plow_route_in_domain(-3e38f),
          "domain admits NaN/Inf/sentinel");
    CHECK(plow_route_in_domain(-1e29f) && plow_route_in_domain(3e38f) && plow_route_in_domain(-0.0f),
          "domain rejects a live key");

    static const uint32_t ns[] = {128, 256, 288, 896}, ks[] = {8, 16};
    for (unsigned a = 0; a < 4; a++)
        for (unsigned b = 0; b < 2; b++) {
            const uint32_t n = ns[a], k = ks[b];
            const uint32_t groups[][2] = {{1, 1}, {8, 4}, {8, 1}, {n / 16u, 2}};
            for (unsigned g = 0; g < 4; g++) {
                const Cfg c = {n, k, groups[g][0], groups[g][1], (int)(g & 1u), 2.5f};
                if (!plow_route_group_ok(n, k, c.n_group, c.topk_group)) continue;
                for (int mode = 0; mode < 6; mode++)
                    for (int rep = 0; rep < 20; rep++) run(&c, mode, rep & 1);
            }
        }

    /* 4 of 64 groups of 2 keep 8 experts at k=16: refused, and the sentinel reference diverges. */
    const Cfg bad = {128, 16, 64, 4, 1, 1.0f};
    CHECK(!plow_route_group_ok(bad.n_exp, bad.k, bad.n_group, bad.topk_group), "ineligible groups accepted");
    CHECK(!plow_route_group_ok(130, 8, 8, 4) && !plow_route_group_ok(128, 8, 128, 64) &&
              !plow_route_group_ok(256, 8, 128, 4) && plow_route_group_ok(256, 8, 8, 4) &&
              plow_route_group_ok(256, 8, 1, 1) && plow_route_group_ok(256, 16, 8, 8),
          "group_ok thresholds");
    {
        float score[NMAX], bias[NMAX];
        unsigned char kept[NMAX];
        uint32_t wr[KMAX], ws[KMAX];
        fill(score, bias, bad.n_exp, 0, 0);
        group_mask(&bad, score, kept);
        rank_select(&bad, score, kept, wr);
        ref_sentinel(&bad, score, kept, ws);
        CHECK(memcmp(wr, ws, bad.k * 4) != 0, "ineligible groups: expected the sentinel reference to diverge");
    }

    printf("route_key: %d cases, %d failures\n", cases, fails);
    return fails ? 1 : 0;
}
