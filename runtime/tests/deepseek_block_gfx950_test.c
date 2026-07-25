/* deepseek_block_gfx950_test.c — a FULL DeepSeek decoder block on the real interpreter. [DEEPSEEK-MLA]
 *
 * Assembles the two novel DeepSeek pieces — MLA (Multi-head Latent Attention) + the shared
 * MoE FFN core — into one decode block, stacked N times, driven by ONE persistent-interpreter
 * launch. Per block (decode, bs=1, one new token, a real ~4k latent KV context):
 *
 *   RMSNorm(x)                                             -> xn
 *   GEMV W_ckv  · xn  -> c_kv[qpos]   (writes the new token's latent cache row)
 *   GEMV W_krope· xn  -> k_rope[qpos] (the shared rope key row)
 *   GEMV W_qabs · xn  -> Q_abs[nh][DK]   (the absorbed query, W_uk folded in)
 *   GEMV W_qrope· xn  -> Q_rope[nh][DR]
 *   FLASH_MLA_DECODE  -> (O_latent, m, l) partials     (q_abs·C_kv + q_rope·K_rope, PV on latent)
 *   FLASH_MERGE<512>  -> O_latent[nh][DK]              (the split-KV LSE merge, latent-wide)
 *   O_UV_FOLD         -> o[nh][V]                       (per-query W_uv fold to v_head_dim)
 *   GEMV W_oproj· o   -> attn_out
 *   RESIDUAL x + attn_out -> x_mid
 *   RMSNorm(x_mid)    -> xn2
 *   MoeRouter -> K×(MoeExpertGlu -> MoeExpertDown) -> MoeCombine(x_mid + Σ part) -> x_next
 *
 * The MLA WRITE path is entirely existing ops (GEMV to a strided cache row); only the flash
 * READ path is the newly-wired FLASH_MLA_DECODE / O_UV_FOLD (op_attention.h). RoPE's rotary
 * twist is out of scope here (synthetic weights) — the flash still dots Q_rope·K_rope over DR,
 * which is what exercises the shared-rope term; the rotary reshuffle is orthogonal to the loop.
 *
 * VALIDATION (truth = independent CPU reference over the SAME op reduction boundaries):
 *   - MoE dispatch is validated BIT-EXACT: the oracle MoE is fed the DEVICE's normed input
 *     (xn2) + residual (x_mid) and must reproduce x_next byte-for-byte — the router top-k
 *     (lowest-id tie-break), the weight-base indirection, and the fixed-order combine.
 *   - MLA is validated to the established output-scaled tolerance (~1e-3): it keeps P in f32
 *     but latent/query/W_uv are bf16 and the online-softmax reassociates vs the reference's
 *     single-pass softmax (sparse-attn-design.md §2.6) — the SAME metric mla_gfx950_test.c uses.
 *   The oracle is fed the DEVICE's RMSNorm output (xn) for the MLA leg so the block reductions
 *   (block_sum + rsqrtf, not bit-reproducible by a scalar reference) don't pollute the compare.
 *
 * Scaled DeepSeek config: MLA DK=512/DR=64/V=128 (the kernel's compile-time latent geometry),
 * nh=8, ctx=4096; MoE E=8/K=2 (cardinality-independent dispatch), sigmoid+norm_topk+scale=2.5.
 */
#include "../amd/hsa_backend.h"
#include "../common/dev_isa.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- geometry (DK/DR/V fixed by the MLA kernel template) ---- */
#define H        512
#define DK       512     /* kv_lora_rank (latent width)      */
#define DR       64      /* qk_rope_dim (shared rope key)    */
#define VD       128     /* v_head_dim                       */
#define NH       8       /* attention heads (multiple of GF=2) */
#define CTX      4096    /* KV context length (real long-ctx flash loop) */
#define NSPLIT   8       /* split-KV (exercises the merge)   */
#define SCALE    0.08838835f  /* 1/sqrt(128) */
/* MoE (scaled GLM/DeepSeek RouterCfg) */
#define I_MOE    32
#define N_EXP    8
#define K        2
#define ACT      1u      /* SwiGLU (silu) */
#define ROUTE_SCALE 2.5f
#define FLAGS    (1u | 2u)  /* sigmoid + norm_topk */
#define N_BLK    3
#define EPS      1e-6f

typedef uint16_t bf16;
static float b2f(bf16 v){ union{uint32_t u;float f;}c; c.u=(uint32_t)v<<16; return c.f; }
static bf16 f2b(float f){ union{float f;uint32_t u;}c; c.f=f; uint32_t r=c.u+0x7fff+((c.u>>16)&1); return (bf16)(r>>16); }

static uint64_t rs;
static void seed(uint64_t s){ rs = s*6364136223846793005ULL + 1442695040888963407ULL; }
static float frand(void){ rs = rs*6364136223846793005ULL + 1442695040888963407ULL; return (float)((int32_t)(rs>>33)%2001-1000)/4000.0f; }

/* ---- reference op bodies (sequential f32 dot, bf16-rounded — mirror the kernels) ---- */
static float mdot(const bf16* x, const bf16* w, int Kk){ float a=0; for(int i=0;i<Kk;i++) a+=b2f(x[i])*b2f(w[i]); return b2f(f2b(a)); }
static float silu(float x){ return x/(1.0f+expf(-x)); }

/* absorbed-MLA reference (mirrors mla_ref.rs): given per-head absorbed queries and the latent
 * cache, produce o[nh][V]. Single-pass softmax; oacc rounded to bf16 (the merge writes bf16). */
static void ref_mla(const bf16* Qabs, const bf16* Qrope, const bf16* Ckv, const bf16* Krope,
                    const bf16* Wuv, bf16* o_out) {
    for (int h = 0; h < NH; h++) {
        float sc[CTX], mx = -1e30f;
        for (int t = 0; t < CTX; t++) {
            float d = 0;
            for (int l = 0; l < DK; l++) d += b2f(Qabs[h*DK+l]) * b2f(Ckv[(size_t)t*DK+l]);
            for (int r = 0; r < DR; r++) d += b2f(Qrope[h*DR+r]) * b2f(Krope[(size_t)t*DR+r]);
            sc[t] = d * SCALE; if (sc[t] > mx) mx = sc[t];
        }
        float sum = 0, p[CTX];
        for (int t = 0; t < CTX; t++) { p[t] = expf(sc[t]-mx); sum += p[t]; }
        float inv = sum > 0 ? 1.0f/sum : 0.0f;
        bf16 oacc[DK];
        for (int l = 0; l < DK; l++) {
            float a = 0;
            for (int t = 0; t < CTX; t++) a += p[t] * b2f(Ckv[(size_t)t*DK+l]);
            oacc[l] = f2b(a * inv);
        }
        for (int v = 0; v < VD; v++) {
            float a = 0;
            for (int l = 0; l < DK; l++) a += b2f(oacc[l]) * b2f(Wuv[(size_t)(h*DK+l)*VD+v]);
            o_out[h*VD+v] = f2b(a);
        }
    }
}

/* reference router (mirrors d_moe_router bit-for-bit). */
static void ref_router(const bf16* x, const bf16* Wr, unsigned* id, float* g) {
    float s[N_EXP];
    for (int e = 0; e < N_EXP; e++) s[e] = 1.0f/(1.0f+expf(-mdot(x, Wr+(size_t)e*H, H)));
    float live[N_EXP]; memcpy(live, s, sizeof(s));
    for (int j = 0; j < K; j++) {
        unsigned long long best = 0; int bi = 0;
        for (int e = 0; e < N_EXP; e++) {
            unsigned sb; float sc = live[e]; memcpy(&sb,&sc,4);
            sb = (sb & 0x80000000u) ? ~sb : (sb | 0x80000000u);
            unsigned long long key = ((unsigned long long)sb<<20) | (unsigned long long)((N_EXP-1-e)&0xFFFFF);
            if (key > best) { best = key; bi = e; }
        }
        id[j] = bi; g[j] = live[bi]; live[bi] = -1e30f;
    }
    float sum = 0; for (int j=0;j<K;j++) sum += g[j];
    for (int j = 0; j < K; j++) { if (FLAGS&2u) g[j]/=sum; g[j]*=ROUTE_SCALE; }
}

/* reference MoE FFN given the (device) normed input xn2 and residual x_mid → x_next. BIT-EXACT. */
static void ref_moe(const bf16* xn2, const bf16* xmid, const bf16* Wr,
                    const bf16* const gw[N_EXP], const bf16* const uw[N_EXP],
                    const bf16* const dw[N_EXP], bf16* xnext) {
    unsigned id[K]; float g[K]; ref_router(xn2, Wr, id, g);
    float part[K][H];
    for (int slot = 0; slot < K; slot++) {
        unsigned e = id[slot]; bf16 fu[I_MOE];
        for (int n = 0; n < I_MOE; n++) {
            float gg = mdot(xn2, gw[e]+(size_t)n*H, H);
            float uu = mdot(xn2, uw[e]+(size_t)n*H, H);
            fu[n] = f2b(silu(gg)*uu);
        }
        for (int hh = 0; hh < H; hh++) part[slot][hh] = g[slot]*mdot(fu, dw[e]+(size_t)hh*I_MOE, I_MOE);
    }
    for (int hh = 0; hh < H; hh++) {
        float a = b2f(xmid[hh]);
        for (int slot = 0; slot < K; slot++) a += part[slot][hh];
        xnext[hh] = f2b(a);
    }
}

/* ---- program builder helpers ---- */
static PlowDevInst g_inst[4096];
static PlowWait    g_wait[8192];
static uint32_t    g_succ[4096];
typedef struct { uint32_t wait_ofs, succ_ofs; uint16_t wait_len, succ_len; } PlowGate;
static PlowGate g_gate[4096]; /* per-inst gates, copied onto every StreamEnt (64B DevInst carries none) */
static int g_nops = 0, g_nw = 0;

static int emitop(uint16_t op, uint16_t blocks) {
    int i = g_nops++;
    g_inst[i].op = op; g_inst[i].blocks = blocks;
    g_gate[i].succ_ofs = i; g_gate[i].succ_len = 1; g_succ[i] = i;
    return i;
}
static void addwait(int op, int producer, uint32_t thr) {
    if (g_gate[op].wait_len == 0) g_gate[op].wait_ofs = g_nw;
    g_wait[g_nw].id = producer; g_wait[g_nw].threshold = thr; g_nw++;
    g_gate[op].wait_len++;
}

/* tensor registry: a handle can point mid-buffer (row offset). */
static void* g_tens[1024];
static int g_nt = 0;
static int reg(void* p){ g_tens[g_nt] = p; return g_nt++; }

static float relerr_report(const char* what, const bf16* got, const bf16* want, int n, double* worst) {
    double mw=0,md=0,se=0,sw=0;
    for (int i=0;i<n;i++){ double d=fabs(b2f(got[i])-b2f(want[i])); double w=fabs(b2f(want[i]));
        mw=fmax(mw,w); md=fmax(md,d); se+=d*d; sw+=(double)b2f(want[i])*b2f(want[i]); }
    double rmax=md/(mw+1e-12), rrms=sqrt(se/n)/(sqrt(sw/n)+1e-12);
    *worst = rmax;
    (void)what;
    return (float)rrms;
}

int main(int argc, char** argv) {
    const char* elf = argc > 1 ? argv[1] : "interp_decode.elf";
    plow_hsa* h = plow_hsa_init();
    if (!h) { printf("hsa init failed\n"); return 1; }
    char gfx[64]; uint32_t cus=0, lds=0; plow_hsa_device_info(h,0,gfx,&cus,&lds);
    const unsigned NCU = cus;
    printf("dev0: %s  CUs=%u\n", gfx, cus);

    FILE* f = fopen(elf,"rb"); if(!f){ printf("%s missing\n", elf); return 1; }
    fseek(f,0,SEEK_END); long co_n=ftell(f); fseek(f,0,SEEK_SET);
    void* co=malloc(co_n); if(fread(co,1,co_n,f)!=(size_t)co_n) return 1; fclose(f);
    if (plow_hsa_load_code_object(h,0,co,co_n)) { printf("load failed\n"); return 1; }
    plow_hsa_kernel kern;
    if (plow_hsa_get_kernel(h,0,"plow_interp_dec_gfx950",&kern)) { printf("no kernel\n"); return 1; }

    /* ---- host weights (seeded) ---- */
    static bf16 h_g1[N_BLK][H], h_g2[N_BLK][H];
    static bf16 h_Wckv[N_BLK][DK*H], h_Wkr[N_BLK][DR*H];
    static bf16 h_Wqa[N_BLK][NH*DK*H], h_Wqr[N_BLK][NH*DR*H];
    static bf16 h_Wuv[N_BLK][NH*DK*VD], h_Wo[N_BLK][H*NH*VD];
    static bf16 h_Wr[N_BLK][N_EXP*H];
    static bf16 h_gate[N_BLK][N_EXP][I_MOE*H], h_up[N_BLK][N_EXP][I_MOE*H], h_down[N_BLK][N_EXP][H*I_MOE];
    static bf16 h_ckv[N_BLK][(size_t)CTX*DK], h_kr[N_BLK][(size_t)CTX*DR];
    static bf16 h_x0[H];

    for (int b=0;b<N_BLK;b++){
        seed(0x51A0 ^ (uint64_t)b);
        for (int i=0;i<H;i++){ h_g1[b][i]=f2b(1.0f+frand()*0.1f); h_g2[b][i]=f2b(1.0f+frand()*0.1f); }
        for (size_t i=0;i<DK*H;i++)   h_Wckv[b][i]=f2b(frand());
        for (size_t i=0;i<DR*H;i++)   h_Wkr[b][i]=f2b(frand());
        for (size_t i=0;i<(size_t)NH*DK*H;i++) h_Wqa[b][i]=f2b(frand()*0.2f);
        for (size_t i=0;i<(size_t)NH*DR*H;i++) h_Wqr[b][i]=f2b(frand()*0.2f);
        for (size_t i=0;i<(size_t)NH*DK*VD;i++) h_Wuv[b][i]=f2b(frand()*0.1f);
        for (size_t i=0;i<(size_t)H*NH*VD;i++)  h_Wo[b][i]=f2b(frand()*0.1f);
        for (int i=0;i<N_EXP*H;i++) h_Wr[b][i]=f2b(frand());
        for (int e=0;e<N_EXP;e++){
            for (size_t i=0;i<I_MOE*H;i++){ h_gate[b][e][i]=f2b(frand()); h_up[b][e][i]=f2b(frand()); }
            for (size_t i=0;i<H*I_MOE;i++) h_down[b][e][i]=f2b(frand());
        }
        /* prefilled latent context (small so softmax is well-conditioned) */
        for (size_t i=0;i<(size_t)CTX*DK;i++) h_ckv[b][i]=f2b(frand()*0.4f);
        for (size_t i=0;i<(size_t)CTX*DR;i++) h_kr[b][i]=f2b(frand()*0.4f);
    }
    seed(999); for (int i=0;i<H;i++) h_x0[i]=f2b(frand());

    /* ---- device allocations + upload ---- */
#define UP(dp, src, bytes) do{ dp = plow_hsa_alloc(h,0,bytes); plow_hsa_upload(h,0,dp,src,bytes);}while(0)
    void *d_x[N_BLK+1], *d_xn[N_BLK], *d_attn[N_BLK], *d_xmid[N_BLK], *d_xn2[N_BLK];
    for (int b=0;b<=N_BLK;b++) d_x[b]=plow_hsa_alloc(h,0,H*2);
    plow_hsa_upload(h,0,d_x[0],h_x0,H*2);
    for (int b=0;b<N_BLK;b++){ d_xn[b]=plow_hsa_alloc(h,0,H*2); d_attn[b]=plow_hsa_alloc(h,0,H*2);
                               d_xmid[b]=plow_hsa_alloc(h,0,H*2); d_xn2[b]=plow_hsa_alloc(h,0,H*2); }
    void *d_g1[N_BLK],*d_g2[N_BLK],*d_Wckv[N_BLK],*d_Wkr[N_BLK],*d_Wqa[N_BLK],*d_Wqr[N_BLK],*d_Wuv[N_BLK],*d_Wo[N_BLK];
    void *d_Wr[N_BLK],*d_wtab[N_BLK],*d_ckv[N_BLK],*d_kr[N_BLK];
    void *d_gate[N_BLK][N_EXP],*d_up[N_BLK][N_EXP],*d_down[N_BLK][N_EXP];
    for (int b=0;b<N_BLK;b++){
        UP(d_g1[b],h_g1[b],H*2); UP(d_g2[b],h_g2[b],H*2);
        UP(d_Wckv[b],h_Wckv[b],DK*H*2); UP(d_Wkr[b],h_Wkr[b],DR*H*2);
        UP(d_Wqa[b],h_Wqa[b],(size_t)NH*DK*H*2); UP(d_Wqr[b],h_Wqr[b],(size_t)NH*DR*H*2);
        UP(d_Wuv[b],h_Wuv[b],(size_t)NH*DK*VD*2); UP(d_Wo[b],h_Wo[b],(size_t)H*NH*VD*2);
        UP(d_Wr[b],h_Wr[b],N_EXP*H*2);
        UP(d_ckv[b],h_ckv[b],(size_t)CTX*DK*2); UP(d_kr[b],h_kr[b],(size_t)CTX*DR*2);
        uint64_t wt[N_EXP*3];
        for (int e=0;e<N_EXP;e++){ UP(d_gate[b][e],h_gate[b][e],I_MOE*H*2); UP(d_up[b][e],h_up[b][e],I_MOE*H*2);
            UP(d_down[b][e],h_down[b][e],H*I_MOE*2);
            wt[e*3+0]=(uint64_t)(uintptr_t)d_gate[b][e]; wt[e*3+1]=(uint64_t)(uintptr_t)d_up[b][e]; wt[e*3+2]=(uint64_t)(uintptr_t)d_down[b][e]; }
        UP(d_wtab[b],wt,sizeof(wt));
    }
    /* shared scratch (blocks run sequentially → safe to reuse) */
    void* d_Qa   = plow_hsa_alloc(h,0,(size_t)NH*DK*2);
    void* d_Qr   = plow_hsa_alloc(h,0,(size_t)NH*DR*2);
    void* d_Opart= plow_hsa_alloc(h,0,(size_t)NH*NSPLIT*DK*4);
    void* d_ml   = plow_hsa_alloc(h,0,(size_t)NH*NSPLIT*2*4);
    void* d_Olat = plow_hsa_alloc(h,0,(size_t)NH*DK*2);
    void* d_oat  = plow_hsa_alloc(h,0,(size_t)NH*VD*2);
    void* d_tab  = plow_hsa_alloc(h,0,K*8);
    void* d_fu   = plow_hsa_alloc(h,0,(size_t)K*I_MOE*2);
    void* d_part = plow_hsa_alloc(h,0,(size_t)K*H*4);
    int32_t klen = CTX; void* d_klen; UP(d_klen,&klen,4);

    /* ---- tensor handles ---- */
    int TQA=reg(d_Qa), TQR=reg(d_Qr), TOP=reg(d_Opart), TML=reg(d_ml), TOLAT=reg(d_Olat),
        TOAT=reg(d_oat), TTAB=reg(d_tab), TFU=reg(d_fu), TPART=reg(d_part), TKLEN=reg(d_klen);
    int TX[N_BLK+1]; for(int b=0;b<=N_BLK;b++) TX[b]=reg(d_x[b]);
    int TXN[N_BLK],TATT[N_BLK],TXMID[N_BLK],TXN2[N_BLK];
    int TG1[N_BLK],TG2[N_BLK],TWCKV[N_BLK],TWKR[N_BLK],TWQA[N_BLK],TWQR[N_BLK],TWUV[N_BLK],TWO[N_BLK];
    int TWR[N_BLK],TWTAB[N_BLK],TCKV[N_BLK],TKR[N_BLK],TCKVROW[N_BLK],TKRROW[N_BLK];
    const size_t qpos = CTX-1;
    for (int b=0;b<N_BLK;b++){
        TXN[b]=reg(d_xn[b]); TATT[b]=reg(d_attn[b]); TXMID[b]=reg(d_xmid[b]); TXN2[b]=reg(d_xn2[b]);
        TG1[b]=reg(d_g1[b]); TG2[b]=reg(d_g2[b]); TWCKV[b]=reg(d_Wckv[b]); TWKR[b]=reg(d_Wkr[b]);
        TWQA[b]=reg(d_Wqa[b]); TWQR[b]=reg(d_Wqr[b]); TWUV[b]=reg(d_Wuv[b]); TWO[b]=reg(d_Wo[b]);
        TWR[b]=reg(d_Wr[b]); TWTAB[b]=reg(d_wtab[b]); TCKV[b]=reg(d_ckv[b]); TKR[b]=reg(d_kr[b]);
        TCKVROW[b]=reg((char*)d_ckv[b]+qpos*DK*2); TKRROW[b]=reg((char*)d_kr[b]+qpos*DR*2);
    }

    /* ---- build the program ---- */
#define GEMV(o_h, x_h, w_h, N_, K_) do{ int _i=emitop(PLOW_DOP_GEMV,NCU); PlowDevInst*_d=&g_inst[_i]; \
        _d->t[0]=o_h; _d->t[1]=x_h; _d->t[2]=w_h; _d->t[3]=PLOW_TENSOR_NONE; _d->t[4]=PLOW_TENSOR_NONE; \
        _d->i[0]=1; _d->i[1]=(N_); _d->i[2]=(K_); _d->i[3]=0; _d->i[4]=0; _d->fj[0].f=1.0f; _last=_i; }while(0)
    int prev_combine = -1, _last = -1;
    for (int b=0;b<N_BLK;b++){
        /* rmsnorm1: x[b] -> xn[b] */
        int i_rn1 = emitop(PLOW_DOP_RMSNORM, 1);
        { PlowDevInst* d=&g_inst[i_rn1]; d->t[0]=TXN[b]; d->t[1]=TX[b]; d->t[2]=TG1[b]; d->i[0]=1; d->i[1]=H; d->fj[0].f=EPS; }
        if (prev_combine>=0) addwait(i_rn1, prev_combine, NCU);

        GEMV(TCKVROW[b], TXN[b], TWCKV[b], DK, H);   int i_ckv=_last;  addwait(i_ckv, i_rn1, 1);
        GEMV(TKRROW[b],  TXN[b], TWKR[b],  DR, H);    int i_kr=_last;   addwait(i_kr, i_rn1, 1);
        GEMV(TQA,        TXN[b], TWQA[b],  NH*DK, H);  int i_qa=_last;  addwait(i_qa, i_rn1, 1);
        GEMV(TQR,        TXN[b], TWQR[b],  NH*DR, H);  int i_qr=_last;  addwait(i_qr, i_rn1, 1);

        /* flash MLA decode */
        int i_fl = emitop(PLOW_DOP_FLASH_MLA_DECODE, NCU);
        { PlowDevInst* d=&g_inst[i_fl]; d->t[0]=TOP; d->t[1]=TML; d->t[2]=TQA; d->t[3]=TQR;
          d->t[4]=TCKV[b]; d->t[5]=TKR[b]; d->t[6]=TKLEN;
          d->i[0]=1; d->i[1]=NH; d->i[2]=CTX; d->i[3]=0; d->i[4]=NSPLIT; d->i[5]=PLOW_KV_MASK_NONE; d->fj[0].f=SCALE; }
        addwait(i_fl, i_ckv, NCU); addwait(i_fl, i_kr, NCU); addwait(i_fl, i_qa, NCU); addwait(i_fl, i_qr, NCU);

        /* merge<512> */
        int i_mg = emitop(PLOW_DOP_FLASH_MERGE, NCU);
        { PlowDevInst* d=&g_inst[i_mg]; d->t[0]=TOLAT; d->t[1]=TOP; d->t[2]=TML; d->i[0]=1; d->i[1]=NH; d->i[2]=NSPLIT; d->i[3]=512; }
        addwait(i_mg, i_fl, NCU);

        /* o_uv fold */
        int i_uv = emitop(PLOW_DOP_O_UV_FOLD, NCU);
        { PlowDevInst* d=&g_inst[i_uv]; d->t[0]=TOAT; d->t[1]=TOLAT; d->t[2]=TWUV[b]; d->i[0]=1; d->i[1]=NH; d->i[2]=VD; }
        addwait(i_uv, i_mg, NCU);

        /* o_proj: o[nh*V] -> attn_out[b] */
        GEMV(TATT[b], TOAT, TWO[b], H, NH*VD); int i_op=_last; addwait(i_op, i_uv, NCU);

        /* residual: x[b] + attn_out -> xmid[b] */
        int i_rs = emitop(PLOW_DOP_RESIDUAL, 1);
        { PlowDevInst* d=&g_inst[i_rs]; d->t[0]=TXMID[b]; d->t[1]=TX[b]; d->t[2]=TATT[b]; d->i[0]=H; d->fj[0].f=1.0f; }
        addwait(i_rs, i_op, NCU);

        /* rmsnorm2: xmid -> xn2 */
        int i_rn2 = emitop(PLOW_DOP_RMSNORM, 1);
        { PlowDevInst* d=&g_inst[i_rn2]; d->t[0]=TXN2[b]; d->t[1]=TXMID[b]; d->t[2]=TG2[b]; d->i[0]=1; d->i[1]=H; d->fj[0].f=EPS; }
        addwait(i_rn2, i_rs, 1);

        /* MoE FFN: router on xn2, experts on xn2, combine residual = xmid -> x[b+1] */
        int i_router = emitop(PLOW_DOP_MOE_ROUTER, 1);
        { PlowDevInst* d=&g_inst[i_router]; d->t[0]=TTAB; d->t[1]=TXN2[b]; d->t[2]=TWR[b];
          d->i[0]=H; d->i[1]=N_EXP; d->i[2]=K; d->i[3]=FLAGS; d->fj[0].f=ROUTE_SCALE; }
        addwait(i_router, i_rn2, 1);

        int downs[K];
        for (int slot=0;slot<K;slot++){
            int i_g = emitop(PLOW_DOP_MOE_EXPERT_GLU, NCU);
            { PlowDevInst* d=&g_inst[i_g]; d->t[0]=TFU; d->t[1]=TXN2[b]; d->t[2]=TTAB; d->t[3]=TWTAB[b];
              d->i[0]=slot; d->i[1]=I_MOE; d->i[2]=H; d->i[3]=N_EXP; d->i[5]=ACT; }
            addwait(i_g, i_router, 1);
            int i_d = emitop(PLOW_DOP_MOE_EXPERT_DOWN, NCU);
            { PlowDevInst* d=&g_inst[i_d]; d->t[0]=TPART; d->t[1]=TFU; d->t[2]=TTAB; d->t[3]=TWTAB[b];
              d->i[0]=slot; d->i[1]=H; d->i[2]=I_MOE; d->i[3]=N_EXP; }
            addwait(i_d, i_g, NCU);
            downs[slot]=i_d;
        }
        int i_cmb = emitop(PLOW_DOP_MOE_COMBINE, NCU);
        { PlowDevInst* d=&g_inst[i_cmb]; d->t[0]=TX[b+1]; d->t[1]=TXMID[b]; d->t[2]=PLOW_TENSOR_NONE; d->t[3]=TPART;
          d->i[0]=H; d->i[1]=K; }
        for (int slot=0;slot<K;slot++) addwait(i_cmb, downs[slot], NCU);
        prev_combine = i_cmb;
    }
    const int n_ops = g_nops;

    /* ---- upload tensor table + per-CU streams (blocks==1 ops only on CU0) ---- */
    void* d_tens = plow_hsa_alloc(h,0,(size_t)g_nt*sizeof(void*));
    plow_hsa_upload(h,0,d_tens,g_tens,(size_t)g_nt*sizeof(void*));

    uint32_t* sofs = malloc(4u*NCU); uint32_t* slen = malloc(4u*NCU);
    size_t total = 0;
    for (unsigned cu=0;cu<NCU;cu++) for (int op=0;op<n_ops;op++) if (!(g_inst[op].blocks==1 && cu!=0)) total++;
    PlowStreamEnt* stream = calloc(total, sizeof(PlowStreamEnt));
    size_t si=0;
    for (unsigned cu=0;cu<NCU;cu++){ sofs[cu]=(uint32_t)si;
        for (int op=0;op<n_ops;op++){ if (g_inst[op].blocks==1 && cu!=0) continue;
            stream[si].inst=(uint32_t)op; stream[si].slice=(g_inst[op].blocks==1)?0u:cu;
            stream[si].wait_ofs=g_gate[op].wait_ofs; stream[si].wait_len=g_gate[op].wait_len;
            stream[si].succ_ofs=g_gate[op].succ_ofs; stream[si].succ_len=g_gate[op].succ_len; si++; }
        slen[cu]=(uint32_t)si - sofs[cu]; }

    void* d_inst=plow_hsa_alloc(h,0,(size_t)n_ops*sizeof(PlowDevInst));
    void* d_stream=plow_hsa_alloc(h,0,total*sizeof(PlowStreamEnt));
    void* d_sofs=plow_hsa_alloc(h,0,4u*NCU); void* d_slen=plow_hsa_alloc(h,0,4u*NCU);
    void* d_waits=plow_hsa_alloc(h,0,(size_t)(g_nw?g_nw:1)*sizeof(PlowWait));
    void* d_succs=plow_hsa_alloc(h,0,(size_t)n_ops*4);
    void* d_ctr=plow_hsa_alloc(h,0,(size_t)n_ops*PLOW_CTR_STRIDE*4);
    plow_hsa_upload(h,0,d_inst,g_inst,(size_t)n_ops*sizeof(PlowDevInst));
    plow_hsa_upload(h,0,d_stream,stream,total*sizeof(PlowStreamEnt));
    plow_hsa_upload(h,0,d_sofs,sofs,4u*NCU); plow_hsa_upload(h,0,d_slen,slen,4u*NCU);
    if (g_nw) plow_hsa_upload(h,0,d_waits,g_wait,(size_t)g_nw*sizeof(PlowWait));
    plow_hsa_upload(h,0,d_succs,g_succ,(size_t)n_ops*4);
    uint32_t* zc=calloc((size_t)n_ops*PLOW_CTR_STRIDE,4);
    plow_hsa_upload(h,0,d_ctr,zc,(size_t)n_ops*PLOW_CTR_STRIDE*4);

    PlowProgram prog; memset(&prog,0,sizeof(prog));
    prog.insts=d_inst; prog.stream=d_stream; prog.stream_ofs=d_sofs; prog.stream_len=d_slen;
    prog.waits=d_waits; prog.succs=d_succs; prog.counters=d_ctr; prog.tensors=(void* const*)d_tens;

    printf("program: %d ops, %zu workgroup-packets, ctx=%d nsplit=%d\n\n", n_ops, total, CTX, NSPLIT);
    if (plow_hsa_launch(h,0,&kern,NCU*PLOW_WG_THREADS,1,1,PLOW_WG_THREADS,1,1,0,&prog,sizeof(prog))) {
        printf("LAUNCH FAILED\n"); return 1; }
    plow_hsa_wait(h,0);

    /* ---- validate per block ---- */
    int ok = 1;
    printf("  %-8s %-30s %-30s\n", "block", "MLA attn (rel rms / max)", "MoE dispatch (bit-exact)");
    for (int b=0;b<N_BLK;b++){
        bf16 xn[H], att[H], xmid[H], xn2[H], xnext[H];
        plow_hsa_download(h,0,xn,d_xn[b],H*2);
        plow_hsa_download(h,0,att,d_attn[b],H*2);
        plow_hsa_download(h,0,xmid,d_xmid[b],H*2);
        plow_hsa_download(h,0,xn2,d_xn2[b],H*2);
        plow_hsa_download(h,0,xnext,d_x[b+1],H*2);

        /* MLA leg: oracle fed device xn -> attn_out, tolerance compare */
        bf16 o_mla[NH*VD], att_ref[H];
        /* compute Qabs/Qrope from device xn (mirror the projection GEMV sequentially) */
        static bf16 Qa[NH*DK], Qr[NH*DR];
        for (int n=0;n<NH*DK;n++) Qa[n]=f2b(mdot(xn, h_Wqa[b]+(size_t)n*H, H));
        for (int n=0;n<NH*DR;n++) Qr[n]=f2b(mdot(xn, h_Wqr[b]+(size_t)n*H, H));
        /* new-token latent row from device xn, overwrite the prefilled copy at qpos */
        static bf16 ckv_ref[(size_t)CTX*DK], kr_ref[(size_t)CTX*DR];
        memcpy(ckv_ref, h_ckv[b], sizeof(ckv_ref)); memcpy(kr_ref, h_kr[b], sizeof(kr_ref));
        for (int l=0;l<DK;l++) ckv_ref[qpos*DK+l]=f2b(mdot(xn, h_Wckv[b]+(size_t)l*H, H));
        for (int r=0;r<DR;r++) kr_ref[qpos*DR+r]=f2b(mdot(xn, h_Wkr[b]+(size_t)r*H, H));
        ref_mla(Qa, Qr, ckv_ref, kr_ref, h_Wuv[b], o_mla);
        for (int hh=0;hh<H;hh++) att_ref[hh]=f2b(mdot(o_mla, h_Wo[b]+(size_t)hh*NH*VD, NH*VD));
        double mla_max; float mla_rms = relerr_report("mla", att, att_ref, H, &mla_max);
        int mla_ok = (mla_max < 3e-2 && mla_rms < 1e-2);

        /* MoE leg: oracle fed device xn2 + xmid -> x_next, BIT-EXACT compare */
        const bf16 *gw[N_EXP],*uw[N_EXP],*dw[N_EXP];
        for (int e=0;e<N_EXP;e++){ gw[e]=h_gate[b][e]; uw[e]=h_up[b][e]; dw[e]=h_down[b][e]; }
        ref_moe(xn2, xmid, h_Wr[b], gw, uw, dw, xnext);  /* xnext overwritten with ref */
        bf16 dev_next[H]; plow_hsa_download(h,0,dev_next,d_x[b+1],H*2);
        int moe_exact = 1; for (int i=0;i<H;i++) if (dev_next[i]!=xnext[i]) moe_exact=0;

        ok &= (mla_ok && moe_exact);
        char c1[64]; snprintf(c1,sizeof(c1),"%s rms=%.5f max=%.4f", mla_ok?"PASS":"FAIL", mla_rms, mla_max);
        printf("  %-8d %-30s %-30s\n", b, c1, moe_exact?"PASS (byte-identical)":"*** FAIL ***");

        unsigned id[K]; float g[K]; ref_router(xn2, h_Wr[b], id, g);
        printf("           routing: "); for (int j=0;j<K;j++) printf("e%u(g=%.4f) ",id[j],g[j]); printf("\n");
    }

    plow_hsa_download(h,0,zc,d_ctr,(size_t)n_ops*PLOW_CTR_STRIDE*4);
    int ctr_ok=1; for (int op=0;op<n_ops;op++){ if (zc[(size_t)op*PLOW_CTR_STRIDE]!=g_inst[op].blocks){ ctr_ok=0; } }
    printf("\n  executed==total (every packet fired): %s\n", ctr_ok?"YES":"NO");
    printf("\n%s\n", (ok && ctr_ok) ? "DEEPSEEK BLOCK OK — MLA within tolerance, MoE dispatch BIT-EXACT"
                                    : "*** DEEPSEEK BLOCK MISMATCH ***");
    plow_hsa_shutdown(h);
    return (ok && ctr_ok) ? 0 : 1;
}
