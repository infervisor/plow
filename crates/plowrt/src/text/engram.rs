//! §H Engram stage 1: the n-gram hash, which is host work.
//!
//! DeepSeek-V4.1-Flash hashes each position against the `max_ngram_size - 1` tokens before it,
//! once per (n-gram size, head) pair, into that pair's own prime-sized bucket range. The result
//! is `(max_ngram_size - 1) * n_heads` = 24 row ids per token, which the device side then gathers
//! (op 183) and mixes (op 182).
//!
//! **This is integer work over TOKEN IDS ALONE.** It reads no activation, so it is not a kernel —
//! it is a tensor the host builds, like `pos` or `row_token`. It cannot be folded into the embed
//! either: the lookback stops at a dead token, so a position's ids depend on the MASK, not only on
//! its own id.
//!
//! The tables it hashes with come from [`nn_graph::models::config::DeepSeekV41Config::
//! engram_hash_tables`]. Only the compressed-token map is built here, because only it needs the
//! tokenizer.

use std::collections::HashMap;

/// Maps every token id onto a smaller id space where tokens that normalize alike collapse
/// together, so `" The"`, `"the"` and `"THE"` hash the same way.
///
/// A PORT of the reference's `build_compressed_token_map`, not a reimplementation of it: Python's
/// `tokenizers` **is** this Rust crate, and every normalizer in the reference's sequence exists
/// here under the same name. Transcribing it is therefore exact, where re-deriving the
/// normalization by hand would not be.
///
/// Returns `(lookup, compressed_vocab_size)`. The size is not merely a bound —
/// **every hash multiplier is derived from it**, so a config whose
/// `engram_compressed_vocab_size` disagrees would silently rehash the entire table rather than
/// fail. [`EngramHasher::new`] checks it for that reason.
#[cfg(feature = "hf-tokenizer")]
pub fn build_compressed_token_map(tok: &tokenizers::Tokenizer) -> (Vec<u32>, usize) {
    use tokenizers::normalizers::{
        replace::ReplacePattern, Lowercase, Replace, Sequence, Strip, StripAccents, NFD, NFKC,
    };
    use tokenizers::{NormalizedString, Normalizer};

    // A private-use char. A token that is exactly one space would otherwise collapse to the empty
    // string under Strip and merge with unrelated tokens, so it is parked here across the Strip
    // and restored after.
    const SENTINEL: &str = "\u{e000}";
    let seq = Sequence::new(vec![
        NFKC.into(),
        NFD.into(),
        StripAccents.into(),
        Lowercase.into(),
        Replace::new(ReplacePattern::Regex(r"[ \t\r\n]+".into()), " ")
            .expect("whitespace-collapse regex")
            .into(),
        Replace::new(ReplacePattern::Regex(r"^ $".into()), SENTINEL)
            .expect("lone-space regex")
            .into(),
        Strip::new(true, true).into(),
        Replace::new(ReplacePattern::String(SENTINEL.into()), " ")
            .expect("sentinel restore")
            .into(),
    ]);

    let n = tok.get_vocab_size(true);
    let mut key_to_new: HashMap<String, u32> = HashMap::with_capacity(n);
    let mut lookup = vec![0u32; n];
    for id in 0..n as u32 {
        // `skip_special_tokens = false`: the reference decodes the raw backend tokenizer, matching
        // what training decoded with.
        let text = tok.decode(&[id], false).unwrap_or_default();
        let key = if text.contains('\u{fffd}') {
            // A partial UTF-8 byte token. There is nothing to normalize, so it is keyed by its raw
            // form — normalizing the replacement character would merge every such token into one.
            tok.id_to_token(id).unwrap_or_default()
        } else {
            let mut ns = NormalizedString::from(text.as_str());
            match seq.normalize(&mut ns) {
                Ok(()) if !ns.get().is_empty() => ns.get().to_string(),
                // An empty normalization keeps the original, so distinct whitespace-only tokens do
                // not all collapse onto one another.
                _ => text.clone(),
            }
        };
        let next = key_to_new.len() as u32;
        lookup[id as usize] = *key_to_new.entry(key).or_insert(next);
    }
    let size = key_to_new.len();
    (lookup, size)
}

/// A dead token: one that takes no part in any n-gram (an image span). An n-gram never spans one,
/// and the lookback stops there exactly as it stops at the start of the sequence.
const DEAD: i64 = -1;

/// Engram's per-sequence hash state, carrying the compressed ids across the prefill/decode split.
///
/// The cache is why this is a struct rather than a function. A decode step hashes position
/// `start_pos` against the three tokens before it, which were consumed in an earlier call — so the
/// compressed ids have to outlive the step that produced them. Storing the COMPRESSED ids (not the
/// raw ones) also means the map lookup happens once per token, not once per lookback.
pub struct EngramHasher {
    /// `[engram layer][lookback]`.
    multipliers: Vec<Vec<i64>>,
    /// `[engram layer][flat (n-gram size, head) column]`.
    primes: Vec<Vec<i64>>,
    offsets: Vec<Vec<i64>>,
    max_ngram: usize,
    n_heads: usize,
    token_map: Vec<u32>,
    /// The compressed id of the pad token, substituted wherever lookback is blocked.
    pad: i64,
    /// Compressed ids by absolute position, `DEAD` where masked.
    cache: Vec<i64>,
}

impl EngramHasher {
    /// `token_map` / `compressed_vocab` come from [`build_compressed_token_map`]; the tables from
    /// `DeepSeekV41Config::engram_hash_tables`.
    ///
    /// Refuses a compressed vocab that disagrees with the config. That check is not defensive
    /// tidiness: the multipliers were drawn against `engram_compressed_vocab_size`, so a different
    /// map produces a valid-looking hash of the wrong table for every token, and the only symptom
    /// is that the model gets worse.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        multipliers: Vec<Vec<i64>>,
        primes: Vec<Vec<i64>>,
        offsets: Vec<Vec<i64>>,
        max_ngram: usize,
        n_heads: usize,
        token_map: Vec<u32>,
        compressed_vocab: usize,
        cfg_compressed_vocab: usize,
        pad_token_id: usize,
        max_seq_len: usize,
    ) -> Result<Self, String> {
        if compressed_vocab != cfg_compressed_vocab {
            return Err(format!(
                "engram: the tokenizer's compressed vocab is {compressed_vocab} but the config \
                 says {cfg_compressed_vocab}. Every hash multiplier is derived from that number, \
                 so continuing would rehash the whole table rather than fail."
            ));
        }
        let cols = (max_ngram - 1) * n_heads;
        for (l, p) in primes.iter().enumerate() {
            if p.len() != cols {
                return Err(format!(
                    "engram layer {l}: {} bucket ranges, expected (max_ngram-1)*n_heads = {cols}",
                    p.len()
                ));
            }
        }
        let pad = *token_map
            .get(pad_token_id)
            .ok_or_else(|| format!("engram: pad token {pad_token_id} is outside the vocab"))?
            as i64;
        Ok(Self {
            multipliers,
            primes,
            offsets,
            max_ngram,
            n_heads,
            token_map,
            pad,
            cache: vec![DEAD; max_seq_len],
        })
    }

    /// `(max_ngram - 1) * n_heads` — the hash ids each token produces, per engram layer.
    pub fn cols(&self) -> usize {
        (self.max_ngram - 1) * self.n_heads
    }

    pub fn layers(&self) -> usize {
        self.primes.len()
    }

    /// Hash `ids` placed at absolute position `start_pos`, appending `[L][n_engram_layers][cols]`
    /// row ids to `out`.
    ///
    /// `mask[i] == false` marks a token that takes no part in an n-gram. It is written to the
    /// cache as `DEAD`, which blocks lookback THROUGH it for later positions as well — that
    /// persistence is the whole reason the mask is applied here rather than at the gate.
    pub fn hash(&mut self, ids: &[u32], start_pos: usize, mask: Option<&[bool]>, out: &mut Vec<i64>) {
        let end = start_pos + ids.len();
        if end > self.cache.len() {
            self.cache.resize(end, DEAD);
        }
        for (i, &id) in ids.iter().enumerate() {
            let alive = mask.map(|m| m[i]).unwrap_or(true);
            self.cache[start_pos + i] = if alive {
                self.token_map.get(id as usize).copied().unwrap_or(0) as i64
            } else {
                DEAD
            };
        }

        out.reserve(ids.len() * self.layers() * self.cols());
        let mut window = vec![0i64; self.max_ngram];
        for i in 0..ids.len() {
            let pos = start_pos + i;
            // Walk back `max_ngram` tokens. Once blocked — by the start of the sequence or by a
            // dead token — every FURTHER lookback is blocked too, so the flag latches.
            let mut blocked = false;
            for (shift, w) in window.iter_mut().enumerate() {
                let src = if shift > pos {
                    DEAD
                } else {
                    self.cache[pos - shift]
                };
                blocked |= shift > pos || src == DEAD;
                *w = if blocked { self.pad } else { src };
            }
            for l in 0..self.layers() {
                let (mult, primes, offs) =
                    (&self.multipliers[l], &self.primes[l], &self.offsets[l]);
                // XOR the multiplied ids together one lookback at a time, so the running value
                // after step k is the hash of the (k+1)-gram. `wrapping_mul` is the i64 overflow
                // the multiplier bound exists to prevent, spelled out rather than left to a
                // debug-build panic.
                let mut rolling = window[0].wrapping_mul(mult[0]);
                for k in 1..self.max_ngram {
                    rolling ^= window[k].wrapping_mul(mult[k]);
                    for h in 0..self.n_heads {
                        let c = (k - 1) * self.n_heads + h;
                        out.push(rolling.rem_euclid(primes[c]) + offs[c]);
                    }
                }
            }
        }
    }

    /// Forget everything at or after `pos` — a rollback (a rejected speculative block, a
    /// restarted request) must not leave stale compressed ids that a later lookback would read.
    pub fn truncate(&mut self, pos: usize) {
        for slot in self.cache.iter_mut().skip(pos) {
            *slot = DEAD;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The released checkpoint's constants. The multipliers come from numpy's PCG64 and cannot be
    /// derived; the primes can, and are, by the same walk `nn-graph` uses.
    const MULTS: [[i64; 4]; 2] = [
        [76_632_096_046_245, 4_839_876_093_313, 35_959_672_319_349, 73_987_337_458_391],
        [67_716_810_739_261, 51_510_806_800_915, 30_921_347_202_721, 82_619_226_485_591],
    ];
    const ENGRAM_VOCAB: i64 = 16_000_000;
    const MAX_NGRAM: usize = 4;
    const N_HEADS: usize = 8;

    fn is_prime(n: i64) -> bool {
        if n < 2 {
            return false;
        }
        if n % 2 == 0 {
            return n == 2;
        }
        let mut d = 3i64;
        while d * d <= n {
            if n % d == 0 {
                return false;
            }
            d += 2;
        }
        true
    }

    /// The reference's walk: one GLOBAL `seen` set across layers, restarting at
    /// `engram_vocab_size - 1` for every n-gram size. That globality is what keeps the bucket
    /// ranges disjoint.
    fn tables() -> (Vec<Vec<i64>>, Vec<Vec<i64>>) {
        let mut seen: Vec<i64> = Vec::new();
        let (mut primes, mut offsets) = (Vec::new(), Vec::new());
        for _ in 0..2 {
            let mut flat = Vec::new();
            for _ in 0..MAX_NGRAM - 1 {
                let mut cur = ENGRAM_VOCAB - 1;
                for _ in 0..N_HEADS {
                    loop {
                        cur += 1;
                        if is_prime(cur) && !seen.contains(&cur) {
                            break;
                        }
                    }
                    seen.push(cur);
                    flat.push(cur);
                }
            }
            let mut offs = Vec::with_capacity(flat.len());
            let mut acc = 0i64;
            for p in &flat {
                offs.push(acc);
                acc += *p;
            }
            primes.push(flat);
            offsets.push(offs);
        }
        (primes, offsets)
    }

    /// An identity token map, so a hash test exercises the hash and the lookback ALONE. The pad
    /// token id (2) then maps to compressed 2, which is what the pinned values below assume.
    fn hasher(max_seq_len: usize) -> EngramHasher {
        let (primes, offsets) = tables();
        EngramHasher::new(
            MULTS.iter().map(|r| r.to_vec()).collect(),
            primes,
            offsets,
            MAX_NGRAM,
            N_HEADS,
            (0..1024u32).collect(),
            99_092,
            99_092,
            2,
            max_seq_len,
        )
        .unwrap()
    }

    /// `[position][layer][column]` from one `hash` call.
    fn run(h: &mut EngramHasher, ids: &[u32], at: usize, mask: Option<&[bool]>) -> Vec<Vec<Vec<i64>>> {
        let mut out = Vec::new();
        h.hash(ids, at, mask, &mut out);
        let (nl, nc) = (h.layers(), h.cols());
        out.chunks(nl * nc)
            .map(|tok| tok.chunks(nc).map(<[i64]>::to_vec).collect())
            .collect()
    }

    const SEQ: [u32; 6] = [100, 200, 300, 400, 500, 600];
    const MASK: [bool; 6] = [true, true, false, true, true, true];

    /// Pinned against an independent Python transcription of `inference/engram.py` run on the
    /// released config -- two implementations of one spec, not one checked against itself.
    #[test]
    fn the_hash_matches_the_reference() {
        let mut h = hasher(64);
        let r = run(&mut h, &SEQ, 0, Some(&MASK));
        assert_eq!(r.len(), 6);
        assert_eq!(r[0].len(), 2, "two engram layers");
        assert_eq!(r[0][0].len(), 24, "(max_ngram-1) * n_heads columns");

        // column 0 is the 2-gram; 8 the 3-gram; 16 the 4-gram.
        let col = |r: &Vec<Vec<Vec<i64>>>, l: usize, c: usize| -> Vec<i64> {
            r.iter().map(|p| p[l][c]).collect()
        };
        assert_eq!(col(&r, 0, 0), vec![4052413, 11457621, 4299726, 15507829, 1611640, 15990408]);
        assert_eq!(
            col(&r, 0, 8),
            vec![138893193, 135125969, 129401291, 136244361, 140764033, 128381820]
        );
        assert_eq!(
            col(&r, 0, 16),
            vec![270913409, 257763495, 264341430, 262364454, 264586790, 263519714]
        );
        // The second layer hashes differently: its own multipliers, its own bucket ranges.
        assert_eq!(col(&r, 1, 0), vec![12021899, 1475491, 6788701, 11232881, 6182201, 10797156]);

        assert_eq!(
            r[0][0],
            vec![
                4052413, 23667694, 42906971, 52829082, 75741025, 88719377, 111210122, 112422857,
                138893193, 154754281, 160059424, 191821447, 207733256, 223702343, 239672137,
                250328800, 270913409, 282584305, 299653391, 308191258, 324551334, 349093302,
                360573537, 378065828
            ]
        );
    }

    /// The mask does not merely blank its own position -- it BLOCKS LOOKBACK THROUGH itself for
    /// later positions, and it blocks the longer n-grams further along than the short ones.
    ///
    /// Position 4 is the case that pins the rule. Its 2-gram (column 0) never reaches the dead
    /// token at position 2, so masking leaves it untouched; its 3-gram and 4-gram do reach it, and
    /// both change. A lookback that stopped at the masked position itself, or one that did not
    /// latch, would pass the first assertion and fail these.
    #[test]
    fn a_dead_token_blocks_lookback_through_it() {
        let mut a = hasher(64);
        let masked = run(&mut a, &SEQ, 0, Some(&MASK));
        let mut b = hasher(64);
        let plain = run(&mut b, &SEQ, 0, None);

        assert_eq!(masked[4][0][0], plain[4][0][0], "the 2-gram at pos 4 never sees pos 2");
        assert_ne!(masked[4][0][8], plain[4][0][8], "the 3-gram at pos 4 does");
        assert_ne!(masked[4][0][16], plain[4][0][16], "and so does the 4-gram");

        // Position 5 reaches back to 2 through shift 3 ONLY -- so its 4-gram changes and its
        // shorter n-grams do not. This is the edge of the window, and the arithmetic is worth
        // spelling out because it is easy to be off by one: `max_ngram = 4` means shifts 0..=3,
        // so position 5 sees positions 5, 4, 3 and 2, and the dead token is the last of them.
        assert_eq!(masked[5][0][..16], plain[5][0][..16], "2- and 3-grams are clear of pos 2");
        assert_ne!(masked[5][0][16], plain[5][0][16], "the 4-gram at pos 5 still reaches it");

        // And the dead position itself hashes pad against pad, whatever the token was.
        assert_eq!(masked[2][0][0], 4299726);
    }

    /// The cache exists for exactly this: a decode step hashes against tokens consumed in an
    /// earlier call, so hashing in two calls must equal hashing in one.
    #[test]
    fn the_cache_carries_lookback_across_the_prefill_decode_split() {
        let mut one = hasher(64);
        let whole = run(&mut one, &SEQ, 0, Some(&MASK));

        let mut split = hasher(64);
        let pre = run(&mut split, &SEQ[..4], 0, Some(&MASK[..4]));
        let dec = run(&mut split, &SEQ[4..], 4, Some(&MASK[4..]));

        assert_eq!(pre.len(), 4);
        assert_eq!(dec.len(), 2);
        assert_eq!(whole[..4], pre[..]);
        assert_eq!(whole[4..], dec[..], "the decode step must see the prefill's tokens");
    }

    /// Every id must land inside its own column's bucket range, and the ranges must tile the
    /// table. An id that escaped its range would read another column's rows -- a valid-looking
    /// gather of the wrong embedding.
    #[test]
    fn every_id_lands_in_its_own_bucket_range() {
        let (primes, offsets) = tables();
        let mut h = hasher(64);
        let r = run(&mut h, &SEQ, 0, Some(&MASK));
        for tok in &r {
            for (l, layer) in tok.iter().enumerate() {
                for (c, &id) in layer.iter().enumerate() {
                    let lo = offsets[l][c];
                    assert!(
                        id >= lo && id < lo + primes[l][c],
                        "layer {l} col {c}: {id} outside [{lo}, {})",
                        lo + primes[l][c]
                    );
                }
            }
        }
        // The released checkpoint's table heights, which the ranges must sum to exactly.
        assert_eq!(primes[0].iter().sum::<i64>(), 384_006_168);
        assert_eq!(primes[1].iter().sum::<i64>(), 384_016_682);
    }

    /// A compressed vocab that disagrees with the config is refused, LOUDLY.
    ///
    /// Every hash multiplier is derived from that number. A mismatch does not produce out-of-range
    /// ids or a panic -- it produces a valid hash of the wrong table for every token, and the only
    /// symptom is that the model gets worse.
    #[test]
    fn a_mismatched_compressed_vocab_is_refused() {
        let (primes, offsets) = tables();
        // `unwrap_err` would demand `Debug` on the hasher, which carries a 129k-entry map.
        let err = match EngramHasher::new(
            MULTS.iter().map(|r| r.to_vec()).collect(),
            primes,
            offsets,
            MAX_NGRAM,
            N_HEADS,
            (0..1024u32).collect(),
            99_091,
            99_092,
            2,
            64,
        ) {
            Ok(_) => panic!("a compressed vocab of 99091 against a config saying 99092 must refuse"),
            Err(e) => e,
        };
        assert!(err.contains("rehash the whole table"), "{err}");
    }

    /// The token map against the released tokenizer: the size that every multiplier is derived
    /// from, and the collapse that is the whole point of the map.
    ///
    /// Skips when the checkpoint is not on this machine.
    #[cfg(feature = "hf-tokenizer")]
    #[test]
    fn the_compressed_token_map_reproduces_the_configs_vocab_size() {
        let p = std::path::Path::new("/workspace/models/DeepSeek-V4.1-Flash/tokenizer.json");
        if !p.exists() {
            eprintln!("skipping: DeepSeek-V4.1-Flash tokenizer not present");
            return;
        }
        let tok = tokenizers::Tokenizer::from_file(p).unwrap();
        let (map, size) = build_compressed_token_map(&tok);

        // THE load-bearing assertion. `engram_compressed_vocab_size` is 99092 in the released
        // config, and it is the bound every hash multiplier was drawn against.
        assert_eq!(size, 99_092, "the normalizer sequence does not reproduce the reference's map");
        assert_eq!(map.len(), tok.get_vocab_size(true));

        // Case and leading whitespace collapse -- " The", "the" and "THE" must hash alike.
        let id = |s: &str| {
            let e = tok.encode(s, false).unwrap();
            assert_eq!(e.get_ids().len(), 1, "{s:?} is not a single token");
            map[e.get_ids()[0] as usize]
        };
        let the = id("The");
        for v in [" The", "the", "THE", " the"] {
            assert_eq!(id(v), the, "{v:?} should collapse onto \"The\"");
        }
        // Runs of whitespace collapse onto one space.
        assert_eq!(id("  "), id(" "));
        assert_eq!(id("\n"), id(" "));
        // ...but distinct words do not.
        assert_ne!(id("hello"), the);
    }
}
