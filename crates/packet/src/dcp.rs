//! Decode context parallelism: the KV row -> rank map, its inverse, the per-rank byte
//! accounting the admission budget reads, and the reference log-sum-exp merge.
//!
//! The planner deliberately knows no model names, no architecture and no tensor names.
//! Ownership comes from `(world_size, degree, page_rows)` and an ABSOLUTE row index; nothing
//! here is per-sequence, so a shared prefix's pages land on the same shards for every sequence
//! that shares it and the content-addressed prefix cache key stays rank-independent.
//!
//! Rows are block-cyclic at a DCP page. The alternative — a contiguous range per shard — moves
//! its split points as the sequence grows, so it needs either KV migration on the decode hot
//! path or a static split at `max_ctx` that idles `degree - 1` shards for every sequence shorter
//! than `max_ctx / degree`. Block-cyclic growth is free: the next page lands on the next shard
//! by the same closed form, and no row ever moves.
//!
//! `degree == 1` is the replicated layout and every map here is the identity, so a caller can
//! thread a [`DcpLayout`] through unconditionally without changing a single byte of the
//! non-DCP path.

pub const DCP_ABI_VERSION: u32 = 1;

/// Default rows per DCP page.
pub const DCP_PAGE_ROWS_DEFAULT: u32 = 64;

/// Smallest admissible page: the flash kernel's inner KV tile (`FA_BKV`, `glm_nsplit`'s
/// constant). A page below it would let one KV tile straddle two shards.
pub const DCP_MIN_PAGE_ROWS: u32 = 32;

/// The finite sentinel the attention kernels write for a dead split's running max
/// (`FA_NEG_INF`, `runtime/amd/op_attention_common.h`). It is finite on purpose: the merge
/// subtracts it, and `-inf - -inf` is NaN.
pub const DCP_NEG_INF: f32 = -3.0e38;

/// One-resident KV row layout. `degree` shards partition a DCP group's KV rows; the
/// `world_size / degree` groups each hold a replica of the whole cache, so every cached row is
/// resident `world_size / degree` times independent of the factorization.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DcpLayout {
    pub world_size: u32,
    /// DCP degree `d`. 1 = replicated (today's behaviour).
    pub degree: u32,
    /// Rows per DCP page `P`.
    pub page_rows: u32,
}

impl DcpLayout {
    /// The replicated layout: every rank holds every row. Every map below is the identity.
    pub const fn replicated(world_size: u32) -> Self {
        DcpLayout {
            world_size,
            degree: 1,
            page_rows: DCP_PAGE_ROWS_DEFAULT,
        }
    }

    pub const fn new(world_size: u32, degree: u32, page_rows: u32) -> Self {
        DcpLayout {
            world_size,
            degree,
            page_rows,
        }
    }

    /// Structural admissibility. `block_rows` is the VMM pool's rows per physical block when
    /// the caller has one; passing `None` checks only what does not depend on the pool.
    pub fn validate(self, block_rows: Option<u32>) -> Result<(), &'static str> {
        if self.world_size == 0 || self.degree == 0 {
            return Err("world size and DCP degree must be non-zero");
        }
        if !self.world_size.is_multiple_of(self.degree) {
            return Err("DCP degree must divide world size");
        }
        if self.page_rows < DCP_MIN_PAGE_ROWS {
            return Err("DCP page must cover at least one flash KV tile");
        }
        if !self.page_rows.is_power_of_two() {
            return Err("DCP page rows must be a power of two");
        }
        if let Some(block_rows) = block_rows {
            if block_rows == 0 || !block_rows.is_multiple_of(self.page_rows) {
                return Err("DCP page must divide the KV pool block rows");
            }
        }
        Ok(())
    }

    /// The pool-side half of [`Self::validate`]: a shard's local window must be a whole number
    /// of physical blocks, or `bph_local` is not exact and local slot `k` stops being global
    /// block `k * degree + shard` — which is the relation that keeps the map closed-form
    /// instead of a table.
    pub fn validate_pool(self, block_rows: u32, max_ctx: u64) -> Result<(), &'static str> {
        self.validate(Some(block_rows))?;
        if !self
            .local_capacity(max_ctx)
            .is_multiple_of(u64::from(block_rows))
        {
            return Err("DCP local capacity must be a whole number of KV pool blocks");
        }
        Ok(())
    }

    /// `true` when this layout actually shards anything.
    pub const fn enabled(self) -> bool {
        self.degree > 1
    }

    /// Number of DCP groups; each holds one whole replica of the cache.
    pub const fn groups(self) -> u32 {
        self.world_size / self.degree
    }

    /// The group a rank belongs to. Groups are `degree` CONSECUTIVE ranks, so the shards of one
    /// group share the shortest interconnect path.
    pub fn group_of(self, rank: u32) -> u32 {
        assert!(rank < self.world_size, "rank out of range");
        rank / self.degree
    }

    /// This rank's shard index within its DCP group.
    pub fn shard_of(self, rank: u32) -> u32 {
        assert!(rank < self.world_size, "rank out of range");
        rank % self.degree
    }

    /// The shard that owns an absolute KV row.
    pub fn shard_of_row(self, row: u64) -> u32 {
        ((row / u64::from(self.page_rows)) % u64::from(self.degree)) as u32
    }

    pub fn owns(self, rank: u32, row: u64) -> bool {
        self.shard_of(rank) == self.shard_of_row(row)
    }

    /// Where an absolute row sits in its owning shard's local cache.
    pub fn local_row(self, row: u64) -> u64 {
        let p = u64::from(self.page_rows);
        let d = u64::from(self.degree);
        (row / (p * d)) * p + row % p
    }

    /// Inverse of [`Self::local_row`] paired with [`Self::shard_of_row`].
    pub fn global_row(self, shard: u32, local: u64) -> u64 {
        assert!(shard < self.degree, "shard out of range");
        let p = u64::from(self.page_rows);
        let d = u64::from(self.degree);
        (local / p) * (p * d) + u64::from(shard) * p + local % p
    }

    /// Rows of a `len`-row sequence that live on `shard`.
    pub fn local_rows(self, len: u64, shard: u32) -> u64 {
        assert!(shard < self.degree, "shard out of range");
        if self.degree == 1 {
            return len;
        }
        let p = u64::from(self.page_rows);
        let d = u64::from(self.degree);
        let (full, tail) = (len / p, len % p);
        let (q, r) = (full / d, full % d);
        let s = u64::from(shard);
        q * p + if s < r { p } else if s == r { tail } else { 0 }
    }

    /// The most any one shard holds of a `len`-row sequence. THIS, not `len / degree`, is the
    /// admission figure: ownership is quantised to pages, so the binding constraint is the
    /// worst shard and the mean under-counts it by up to one page.
    pub fn max_local_rows(self, len: u64) -> u64 {
        if self.degree == 1 {
            return len;
        }
        let p = u64::from(self.page_rows);
        let d = u64::from(self.degree);
        let (full, tail) = (len / p, len % p);
        let (q, r) = (full / d, full % d);
        q * p + if r > 0 { p } else { tail }
    }

    /// The least any one shard holds. `max_local_rows - min_local_rows <= page_rows` at every
    /// length; that bound is the load-balance guarantee decode leans on.
    pub fn min_local_rows(self, len: u64) -> u64 {
        if self.degree == 1 {
            return len;
        }
        let p = u64::from(self.page_rows);
        let d = u64::from(self.degree);
        let (full, tail) = (len / p, len % p);
        let (q, r) = (full / d, full % d);
        q * p + if r + 1 < d { 0 } else { tail }
    }

    /// Rows one rank must provision to serve any sequence up to `max_ctx`, rounded to whole
    /// pages so a page never straddles the end of the local cache.
    ///
    /// EXACTLY `max_ctx` when `degree == 1`, so a caller that always routes the compiled
    /// context through here emits an unchanged packet on the non-DCP path.
    pub fn local_capacity(self, max_ctx: u64) -> u64 {
        if self.degree == 1 {
            return max_ctx;
        }
        let p = u64::from(self.page_rows);
        let d = u64::from(self.degree);
        max_ctx.div_ceil(p).div_ceil(d) * p
    }

    /// Physical pool blocks a shard needs for the first `rows` global rows.
    pub fn local_blocks(self, rows: u64, block_rows: u32, shard: u32) -> u64 {
        assert!(block_rows > 0, "block rows must be non-zero");
        self.local_rows(rows, shard).div_ceil(u64::from(block_rows))
    }

    /// KV bytes ONE rank holds for a `len`-row sequence, given the replicated per-token figure.
    ///
    /// This is where `bytes_per_token` changes meaning: it stops being "bytes to hold one token
    /// of this sequence" and becomes "bytes THIS DEVICE holds for one token of this sequence".
    pub fn local_kv_bytes(self, per_token_replicated: u64, len: u64) -> Option<u64> {
        per_token_replicated.checked_mul(self.max_local_rows(len))
    }

    /// Sequences of `len` rows a `budget` of device bytes can seat on one rank.
    pub fn seats(self, budget: u64, per_token_replicated: u64, len: u64) -> u64 {
        match self.local_kv_bytes(per_token_replicated, len) {
            Some(0) | None => 0,
            Some(bytes) => budget / bytes,
        }
    }

    /// The merge's split-axis index for `(shard, local split)`. Rank-major and ascending, which
    /// is what makes the fold order fixed and therefore reproducible run to run.
    pub fn split_index(self, shard: u32, split: u32, ns_local: u32) -> u32 {
        assert!(shard < self.degree && split < ns_local, "split out of range");
        shard * ns_local + split
    }

    /// Splits the merge sees after the cross-rank gather.
    pub const fn merged_splits(self, ns_local: u32) -> u32 {
        self.degree * ns_local
    }
}

/// Recover the DCP degree from declared sizes rather than a flag, the way weight sharding and
/// `DevTp` do: a flag can disagree with the packet, and a wrong KV stride is a silently wrong
/// token with no host-side signal.
///
/// `global_ctx` is the compiled context (`in.pos` bytes / 4) and `kv_stride` the flash op's
/// `i[2]`. Returns `None` when the two are not a clean shard of one another.
pub fn recover_degree(global_ctx: u64, kv_stride: u64, page_rows: u32) -> Option<u32> {
    if global_ctx == 0 || kv_stride == 0 || kv_stride > global_ctx {
        return None;
    }
    if kv_stride == global_ctx {
        return Some(1);
    }
    let p = u64::from(page_rows);
    if p == 0 || kv_stride % p != 0 {
        return None;
    }
    let pages = global_ctx.div_ceil(p);
    let local_pages = kv_stride / p;
    let degree = pages.div_ceil(local_pages);
    let candidate = u32::try_from(degree).ok()?;
    // The stride must be exactly what this degree would have produced.
    (DcpLayout::new(candidate.max(1), candidate, page_rows).local_capacity(global_ctx) == kv_stride)
        .then_some(candidate)
}

/// One attention partial over a contiguous piece of the KV axis: the running max, the sum of
/// exponentials, and the unnormalised accumulator over the latent.
#[derive(Clone, Copy, Debug)]
pub struct Partial<'a> {
    pub m: f32,
    pub l: f32,
    pub acc: &'a [f32],
}

impl Partial<'_> {
    /// A shard with no rows below the query position. Routine under DCP — any sequence shorter
    /// than `page_rows * degree` leaves shards idle — where it was a corner case for `nsplit`.
    pub fn is_dead(&self) -> bool {
        self.m == DCP_NEG_INF || self.l == 0.0
    }
}

/// The reference log-sum-exp merge, in the order and the base the kernels use.
///
/// `plow`'s flash writes BASE-2 logits (`FA_EXP(x) = exp2(x)`), so the rescale is `exp2`, not
/// `exp`. `parts` must be in the fold's canonical order — ascending `split_index`, i.e. ascending
/// shard then ascending local split — because floating-point addition is not associative and the
/// order is what makes the result reproducible.
///
/// Returns `(M, L)`: the merged max and the merged sum of exponentials, so a caller can fold a
/// merge of merges.
///
/// Stability: `M` is the running max, so every weight is in `(0, 1]` and the exponent cannot
/// overflow; it underflows to zero only when a shard's best score is ~127 binades below the
/// global max, where its true contribution is already below f32 resolution. `L > 0` whenever one
/// shard is live, and one always is, because page 0 belongs to shard 0 and every sequence has at
/// least one KV row. The all-dead case returns a zero output rather than a NaN.
pub fn merge_partials(parts: &[Partial<'_>], out: &mut [f32]) -> (f32, f32) {
    out.fill(0.0);
    let mut gm = DCP_NEG_INF;
    for p in parts {
        if !p.is_dead() && p.m > gm {
            gm = p.m;
        }
    }
    if gm == DCP_NEG_INF {
        return (DCP_NEG_INF, 0.0);
    }
    let mut gl = 0.0f32;
    for p in parts {
        let w = if p.is_dead() {
            0.0
        } else {
            exp2f(p.m - gm)
        };
        if w == 0.0 {
            continue;
        }
        gl += w * p.l;
        for (o, &v) in out.iter_mut().zip(p.acc.iter()) {
            *o += w * v;
        }
    }
    let inv = if gl > 0.0 { 1.0 / gl } else { 0.0 };
    for o in out.iter_mut() {
        *o *= inv;
    }
    (gm, gl)
}

#[inline]
fn exp2f(x: f32) -> f32 {
    // `f32::exp2` on the host; the device body is `__builtin_amdgcn_exp2f`. Both are the same
    // function to within one ulp, which is why this is a reference and not a bit-exact oracle.
    x.exp2()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn layouts() -> Vec<DcpLayout> {
        let mut v = Vec::new();
        for &w in &[1u32, 2, 4, 8] {
            for &d in &[1u32, 2, 4, 8] {
                if !w.is_multiple_of(d) {
                    continue;
                }
                for &p in &[32u32, 64, 128] {
                    v.push(DcpLayout::new(w, d, p));
                }
            }
        }
        v
    }

    #[test]
    fn degree_one_is_the_identity_everywhere() {
        let l = DcpLayout::replicated(8);
        assert!(!l.enabled());
        for row in 0..4096u64 {
            assert_eq!(l.shard_of_row(row), 0);
            assert_eq!(l.local_row(row), row);
            assert_eq!(l.global_row(0, row), row);
            assert_eq!(l.local_rows(row, 0), row);
            assert_eq!(l.max_local_rows(row), row);
            assert_eq!(l.min_local_rows(row), row);
            assert_eq!(l.local_capacity(row), row);
        }
        assert_eq!(l.local_capacity(81920), 81920);
        assert_eq!(l.merged_splits(16), 16);
    }

    #[test]
    fn the_row_map_is_a_bijection() {
        for l in layouts() {
            let mut seen = vec![None; 4096];
            for row in 0..4096u64 {
                let s = l.shard_of_row(row);
                assert!(s < l.degree);
                let local = l.local_row(row);
                assert_eq!(l.global_row(s, local), row, "{l:?} row {row}");
                let key = (s as usize, local as usize);
                assert!(
                    seen[local as usize].replace(s).map_or(true, |p| p != s),
                    "{l:?} collision at {key:?}"
                );
            }
        }
    }

    #[test]
    fn local_rows_counts_exactly_the_rows_the_map_assigns() {
        for l in layouts() {
            for len in 0..1024u64 {
                for shard in 0..l.degree {
                    let brute = (0..len).filter(|&r| l.shard_of_row(r) == shard).count() as u64;
                    assert_eq!(l.local_rows(len, shard), brute, "{l:?} len {len} shard {shard}");
                }
            }
        }
    }

    #[test]
    fn local_rows_are_dense_from_zero() {
        // The rows a shard owns must map onto `0..local_rows` with no holes, or the local cache
        // is not a compact array and `local_row` needs a table.
        for l in layouts() {
            for len in [0u64, 1, 63, 64, 65, 511, 512, 1023] {
                for shard in 0..l.degree {
                    let mut locals: Vec<u64> = (0..len)
                        .filter(|&r| l.shard_of_row(r) == shard)
                        .map(|r| l.local_row(r))
                        .collect();
                    locals.sort_unstable();
                    let want: Vec<u64> = (0..l.local_rows(len, shard)).collect();
                    assert_eq!(locals, want, "{l:?} len {len} shard {shard}");
                }
            }
        }
    }

    #[test]
    fn shards_are_balanced_to_within_one_page() {
        for l in layouts() {
            for len in 0..4096u64 {
                let max = l.max_local_rows(len);
                let min = l.min_local_rows(len);
                let brute: Vec<u64> = (0..l.degree).map(|s| l.local_rows(len, s)).collect();
                assert_eq!(max, *brute.iter().max().unwrap(), "{l:?} len {len}");
                assert_eq!(min, *brute.iter().min().unwrap(), "{l:?} len {len}");
                assert!(max - min <= u64::from(l.page_rows), "{l:?} len {len}");
                assert_eq!(brute.iter().sum::<u64>(), len, "{l:?} len {len}");
            }
        }
    }

    #[test]
    fn capacity_covers_the_worst_shard_at_every_length() {
        for l in layouts() {
            for max_ctx in [1u64, 64, 1000, 4096, 8192, 70_000, 81_920, 135_168] {
                let cap = l.local_capacity(max_ctx);
                assert!(cap.is_multiple_of(u64::from(l.page_rows)) || l.degree == 1);
                for len in [1u64, 63, 64, 65, 1000, max_ctx] {
                    if len > max_ctx {
                        continue;
                    }
                    assert!(
                        cap >= l.max_local_rows(len),
                        "{l:?} max_ctx {max_ctx} len {len}: cap {cap} < {}",
                        l.max_local_rows(len)
                    );
                }
            }
        }
    }

    #[test]
    fn capacity_and_seats_scale_with_the_degree() {
        // The measured GLM-5.3 figure: 55_608 B/token per rank, 81920 max ctx.
        const PER_TOKEN: u64 = 55_608;
        let rep = DcpLayout::replicated(8);
        let dcp = DcpLayout::new(8, 8, 64);
        assert_eq!(rep.local_capacity(81_920), 81_920);
        assert_eq!(dcp.local_capacity(81_920), 10_240);

        let budget = 50 * 1024 * 1024 * 1024u64;
        let len = 70_016; // a whole number of 64-row pages
        let seats_rep = rep.seats(budget, PER_TOKEN, len);
        let seats_dcp = dcp.seats(budget, PER_TOKEN, len);
        assert_eq!(seats_rep, 13);
        // Eight-way sharding buys close to 8x, short of it only by the page quantisation.
        assert!(seats_dcp >= 8 * seats_rep, "{seats_dcp} vs {seats_rep}");
    }

    #[test]
    fn seats_use_the_worst_shard_not_the_mean() {
        let l = DcpLayout::new(8, 8, 64);
        // 65 rows: shard 0 holds 64, shard 1 holds 1, shards 2..8 hold nothing.
        assert_eq!(l.max_local_rows(65), 64);
        assert_eq!(l.local_kv_bytes(1, 65), Some(64));
        // The mean would say 65/8 = 8 rows and seat 8x too many sequences on shard 0.
        assert!(l.max_local_rows(65) > 65 / 8);
    }

    #[test]
    fn local_blocks_never_exceed_the_replicated_count() {
        let l = DcpLayout::new(8, 8, 64);
        for rows in [1u64, 64, 1024, 16_384, 70_000] {
            let global = rows.div_ceil(4096);
            let sum: u64 = (0..l.degree).map(|s| l.local_blocks(rows, 4096, s)).sum();
            assert!(sum >= global, "rows {rows}: {sum} < {global}");
            // Cyclic pages spread across more physical blocks than a contiguous split would;
            // that is the price of never moving a row, and it is bounded by the shard count.
            assert!(sum <= global + u64::from(l.degree), "rows {rows}: {sum}");
        }
    }

    #[test]
    fn validate_rejects_the_inadmissible() {
        assert!(DcpLayout::new(8, 8, 64).validate(Some(4096)).is_ok());
        assert!(DcpLayout::replicated(1).validate(None).is_ok());
        assert!(DcpLayout::new(8, 3, 64).validate(None).is_err(), "3 ∤ 8");
        assert!(DcpLayout::new(8, 0, 64).validate(None).is_err());
        assert!(DcpLayout::new(8, 8, 16).validate(None).is_err(), "below FA_BKV");
        assert!(DcpLayout::new(8, 8, 96).validate(None).is_err(), "not pow2");
        assert!(
            DcpLayout::new(8, 8, 64).validate(Some(96)).is_err(),
            "page must divide block rows"
        );
        // 81920 / 8 = 10240 rows, a whole number of 4096-row blocks? No: 10240 % 4096 = 2048.
        assert!(DcpLayout::new(8, 8, 64).validate_pool(4096, 81_920).is_err());
        assert!(DcpLayout::new(8, 8, 64).validate_pool(2048, 81_920).is_ok());
        assert!(DcpLayout::replicated(8).validate_pool(4096, 81_920).is_ok());
    }

    #[test]
    fn groups_partition_the_world() {
        let l = DcpLayout::new(8, 4, 64);
        assert_eq!(l.groups(), 2);
        for rank in 0..8 {
            assert_eq!(l.group_of(rank), rank / 4);
            assert_eq!(l.shard_of(rank), rank % 4);
        }
    }

    #[test]
    fn split_index_is_rank_major_and_dense() {
        let l = DcpLayout::new(8, 8, 64);
        let ns = 2;
        let mut seen = Vec::new();
        for shard in 0..l.degree {
            for split in 0..ns {
                seen.push(l.split_index(shard, split, ns));
            }
        }
        assert_eq!(seen, (0..l.merged_splits(ns)).collect::<Vec<_>>());
    }

    #[test]
    fn degree_is_recoverable_from_the_declared_stride() {
        for &ctx in &[4096u64, 8192, 70_000, 81_920, 135_168] {
            for &d in &[1u32, 2, 4, 8] {
                let l = DcpLayout::new(8, d, 64);
                let stride = l.local_capacity(ctx);
                assert_eq!(
                    recover_degree(ctx, stride, 64),
                    Some(d),
                    "ctx {ctx} degree {d} stride {stride}"
                );
            }
        }
        assert_eq!(recover_degree(0, 0, 64), None);
        assert_eq!(recover_degree(4096, 8192, 64), None, "stride above ctx");
        assert_eq!(recover_degree(4096, 100, 64), None, "not a page multiple");
    }

    // ---- merge math ----

    /// A monolithic softmax over `scores`, in base 2 to match the kernels.
    fn reference(scores: &[f32], values: &[Vec<f32>]) -> Vec<f32> {
        let dim = values[0].len();
        let m = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut num = vec![0.0f64; dim];
        let mut den = 0.0f64;
        for (s, v) in scores.iter().zip(values) {
            let w = f64::from((s - m).exp2());
            den += w;
            for (n, &x) in num.iter_mut().zip(v) {
                *n += w * f64::from(x);
            }
        }
        num.iter().map(|&n| (n / den) as f32).collect()
    }

    /// Fold `scores`/`values` into `n` contiguous pieces, mimicking one shard per piece.
    fn partials(scores: &[f32], values: &[Vec<f32>], n: usize, dim: usize) -> Vec<(f32, f32, Vec<f32>)> {
        let mut out = Vec::new();
        for i in 0..n {
            let idx: Vec<usize> = (0..scores.len()).filter(|k| k % n == i).collect();
            if idx.is_empty() {
                out.push((DCP_NEG_INF, 0.0, vec![0.0; dim]));
                continue;
            }
            let m = idx.iter().map(|&k| scores[k]).fold(f32::NEG_INFINITY, f32::max);
            let mut l = 0.0f32;
            let mut acc = vec![0.0f32; dim];
            for &k in &idx {
                let w = (scores[k] - m).exp2();
                l += w;
                for (a, &x) in acc.iter_mut().zip(&values[k]) {
                    *a += w * x;
                }
            }
            // The kernels hand the merge the UNNORMALISED accumulator alongside (m, l).
            out.push((m, l, acc));
        }
        out
    }

    fn synth(rows: usize, dim: usize, spread: f32) -> (Vec<f32>, Vec<Vec<f32>>) {
        let mut scores = Vec::with_capacity(rows);
        let mut values = Vec::with_capacity(rows);
        let mut s: u64 = 0x2545_F491_4F6C_DD1D;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 11) as f32 / (1u64 << 53) as f32) * 2.0 - 1.0
        };
        for _ in 0..rows {
            scores.push(next() * spread);
            values.push((0..dim).map(|_| next()).collect());
        }
        (scores, values)
    }

    fn merged(parts: &[(f32, f32, Vec<f32>)], dim: usize) -> Vec<f32> {
        let refs: Vec<Partial<'_>> = parts
            .iter()
            .map(|(m, l, a)| Partial {
                m: *m,
                l: *l,
                acc: a,
            })
            .collect();
        let mut out = vec![0.0f32; dim];
        merge_partials(&refs, &mut out);
        out
    }

    #[test]
    fn merge_matches_a_monolithic_softmax() {
        let dim = 64;
        for &spread in &[1.0f32, 20.0, 120.0] {
            for &rows in &[1usize, 7, 64, 513] {
                let (scores, values) = synth(rows, dim, spread);
                let want = reference(&scores, &values);
                for &n in &[1usize, 2, 4, 8, 16] {
                    let got = merged(&partials(&scores, &values, n, dim), dim);
                    for (a, b) in got.iter().zip(&want) {
                        assert!(
                            (a - b).abs() <= 2e-5 * (1.0 + b.abs()),
                            "spread {spread} rows {rows} n {n}: {a} vs {b}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn the_result_does_not_depend_on_the_shard_count() {
        let dim = 32;
        let (scores, values) = synth(257, dim, 40.0);
        let base = merged(&partials(&scores, &values, 1, dim), dim);
        for &n in &[2usize, 3, 5, 8, 16, 32] {
            let got = merged(&partials(&scores, &values, n, dim), dim);
            for (a, b) in got.iter().zip(&base) {
                assert!((a - b).abs() <= 2e-5 * (1.0 + b.abs()), "n {n}: {a} vs {b}");
            }
        }
    }

    #[test]
    fn dead_shards_contribute_nothing_and_never_nan() {
        let dim = 8;
        let live_acc = vec![1.0f32; dim];
        let dead_acc = vec![7.0f32; dim]; // poison: must never be read
        let mut out = vec![0.0f32; dim];
        let parts = [
            Partial { m: DCP_NEG_INF, l: 0.0, acc: &dead_acc },
            Partial { m: 3.0, l: 2.0, acc: &live_acc },
            Partial { m: DCP_NEG_INF, l: 0.0, acc: &dead_acc },
        ];
        let (m, l) = merge_partials(&parts, &mut out);
        assert_eq!(m, 3.0);
        assert_eq!(l, 2.0);
        for &v in &out {
            assert_eq!(v, 0.5, "dead shard leaked");
        }
    }

    #[test]
    fn an_all_dead_merge_is_zero_not_nan() {
        let dim = 4;
        let acc = vec![5.0f32; dim];
        let mut out = vec![9.0f32; dim];
        let parts = [Partial { m: DCP_NEG_INF, l: 0.0, acc: &acc }];
        let (m, l) = merge_partials(&parts, &mut out);
        assert_eq!((m, l), (DCP_NEG_INF, 0.0));
        assert!(out.iter().all(|v| *v == 0.0 && v.is_finite()));
    }

    #[test]
    fn a_far_below_max_shard_underflows_instead_of_overflowing() {
        let dim = 4;
        let hot = vec![1.0f32; dim];
        let cold = vec![1e30f32; dim];
        let mut out = vec![0.0f32; dim];
        let parts = [
            Partial { m: 1000.0, l: 1.0, acc: &hot },
            Partial { m: -1000.0, l: 1.0, acc: &cold },
        ];
        let (_, l) = merge_partials(&parts, &mut out);
        assert!(l.is_finite() && l > 0.0);
        assert!(out.iter().all(|v| v.is_finite()), "{out:?}");
        for &v in &out {
            assert!((v - 1.0).abs() < 1e-6, "{v}");
        }
    }

    #[test]
    fn the_merge_is_reproducible_for_a_fixed_order() {
        let dim = 48;
        let (scores, values) = synth(300, dim, 30.0);
        let parts = partials(&scores, &values, 8, dim);
        let first = merged(&parts, dim);
        for _ in 0..16 {
            assert_eq!(merged(&parts, dim), first, "merge is not deterministic");
        }
    }
}
