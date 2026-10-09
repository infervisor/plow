//! Table decode: the exact text `tokenizers::Tokenizer::decode` returns, from per-id byte pieces
//! built once at load, without the library's per-call `Vec<String>` and per-token clones.
//!
//! Only two decoder chains are taken, the ones the served checkpoints use, and each is emulated
//! step for step:
//!
//! * `ByteLevel` (GPT-2 / Qwen / GLM): every token maps char-by-char through the byte-level
//!   alphabet (raw token bytes when a char is outside it), all tokens' bytes are concatenated and
//!   decoded lossily as ONE string.
//! * `Sequence[Replace("▁" -> " "), ByteFallback, Fuse]` (Gemma / SentencePiece BPE): `<0xNN>`
//!   tokens accumulate into a byte run that is emitted as text when valid UTF-8 and as one U+FFFD
//!   per byte otherwise; every other token is its replaced text.
//!
//! In both, special tokens are filtered BEFORE the decoder, so a skipped special never breaks a
//! byte run. Anything else (no decoder, Strip, Metaspace, WordPiece, ...) keeps the library path,
//! and [`DecodeTable::build`] refuses a table that disagrees with the library on a randomized
//! self-check.

use std::cell::RefCell;

const TEXT: u8 = 1;
const BYTE: u8 = 2;
const SPECIAL: u8 = 4;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Chain {
    ByteLevel,
    ByteFallback,
}

pub(crate) struct DecodeTable {
    chain: Chain,
    /// Piece bytes of id `i` are `bytes[start[i]..start[i + 1]]`; a `BYTE` piece is its one byte.
    bytes: Vec<u8>,
    start: Vec<u32>,
    /// `TEXT` / `BYTE` (0 = no token: the library skips the id), `| SPECIAL`.
    kind: Vec<u8>,
}

thread_local! {
    static SCRATCH: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
}

/// GPT-2 `bytes_to_unicode`, inverted: char -> byte. The alphabet ends at U+0143.
fn byte_level_inverse() -> [Option<u8>; 324] {
    let direct = |b: u32| (0x21..=0x7E).contains(&b) || (0xA1..=0xAC).contains(&b) || (0xAE..=0xFF).contains(&b);
    let mut inv = [None; 324];
    let mut shifted = 256;
    for b in 0..256u32 {
        if direct(b) {
            inv[b as usize] = Some(b as u8);
        } else {
            inv[shifted] = Some(b as u8);
            shifted += 1;
        }
    }
    inv
}

fn chain_of(t: &tokenizers::Tokenizer) -> Option<Chain> {
    let d = serde_json::to_value(t.get_decoder()?).ok()?;
    if d["type"] == "ByteLevel" {
        return Some(Chain::ByteLevel);
    }
    let steps = d["decoders"].as_array()?;
    (d["type"] == "Sequence"
        && matches!(steps.as_slice(), [r, b, f]
            if r["type"] == "Replace"
                && r["pattern"]["String"] == "\u{2581}"
                && r["content"] == " "
                && b["type"] == "ByteFallback"
                && f["type"] == "Fuse"))
        .then_some(Chain::ByteFallback)
}

impl DecodeTable {
    /// A table for `t`, or `None` when its decoder chain is not one of the two emulated, or the
    /// table disagrees with `t.decode` anywhere on the self-check.
    pub(crate) fn build(t: &tokenizers::Tokenizer) -> Option<Self> {
        let chain = chain_of(t)?;
        let n = t.get_vocab(true).values().copied().max().map_or(0, |m| m as usize + 1);
        let special: rustc_hash::FxHashSet<u32> =
            t.get_added_tokens_decoder().into_iter().filter(|(_, a)| a.special).map(|(id, _)| id).collect();
        let inv = byte_level_inverse();
        let mut table = DecodeTable {
            chain,
            bytes: Vec::new(),
            start: Vec::with_capacity(n + 1),
            kind: Vec::with_capacity(n),
        };
        for id in 0..n as u32 {
            table.start.push(u32::try_from(table.bytes.len()).ok()?);
            let Some(tok) = t.id_to_token(id) else {
                table.kind.push(0);
                continue;
            };
            let mut kind = TEXT;
            match chain {
                Chain::ByteLevel => {
                    let mapped: Option<Vec<u8>> =
                        tok.chars().map(|c| inv.get(c as usize).copied().flatten()).collect();
                    table.bytes.extend_from_slice(mapped.as_deref().unwrap_or(tok.as_bytes()));
                }
                Chain::ByteFallback => {
                    let piece = tok.replace('\u{2581}', " ");
                    let byte = (piece.len() == 6 && piece.starts_with("<0x") && piece.ends_with('>'))
                        .then(|| u8::from_str_radix(&piece[3..5], 16).ok())
                        .flatten();
                    match byte {
                        Some(b) => {
                            kind = BYTE;
                            table.bytes.push(b);
                        }
                        None => table.bytes.extend_from_slice(piece.as_bytes()),
                    }
                }
            }
            if special.contains(&id) {
                kind |= SPECIAL;
            }
            table.kind.push(kind);
        }
        table.start.push(u32::try_from(table.bytes.len()).ok()?);
        match table.self_check(t, n as u32, &special) {
            Ok(()) => Some(table),
            Err(ids) => {
                tracing::warn!(?ids, "table detokenize disagrees with the tokenizer; using the library decode");
                None
            }
        }
    }

    fn piece(&self, id: usize) -> &[u8] {
        &self.bytes[self.start[id] as usize..self.start[id + 1] as usize]
    }

    /// Append the decode of `ids` to `out`: `t.decode(ids, skip_special)`, byte for byte.
    pub(crate) fn decode_into(&self, ids: &[u32], skip_special: bool, out: &mut String) {
        let live = |id: u32| {
            let k = self.kind.get(id as usize).copied().unwrap_or(0);
            (k != 0 && !(skip_special && k & SPECIAL != 0)).then_some((id as usize, k))
        };
        SCRATCH.with_borrow_mut(|run| {
            run.clear();
            match self.chain {
                Chain::ByteLevel => {
                    for (id, _) in ids.iter().filter_map(|&id| live(id)) {
                        run.extend_from_slice(self.piece(id));
                    }
                    push_lossy(run, out);
                }
                Chain::ByteFallback => {
                    for (id, k) in ids.iter().filter_map(|&id| live(id)) {
                        if k & BYTE != 0 {
                            run.extend_from_slice(self.piece(id));
                            continue;
                        }
                        flush_bytes(run, out);
                        // Pieces of TEXT tokens are the token strings themselves (replaced).
                        out.push_str(std::str::from_utf8(self.piece(id)).expect("token text is UTF-8"));
                    }
                    flush_bytes(run, out);
                }
            }
        });
    }

    /// Randomized windows over the whole vocabulary, runs of byte tokens and special tokens,
    /// against the library, both skip modes. `Err` carries the first disagreeing window.
    fn self_check(
        &self,
        t: &tokenizers::Tokenizer,
        n: u32,
        special: &rustc_hash::FxHashSet<u32>,
    ) -> Result<(), Vec<u32>> {
        let bytes: Vec<u32> = (0..n).filter(|&i| self.kind[i as usize] & BYTE != 0).collect();
        let special: Vec<u32> = special.iter().copied().collect();
        let mut seed = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        let mut out = String::new();
        for w in 0..1500 {
            let len = 1 + (next() % 12) as usize;
            let ids: Vec<u32> = (0..len)
                .map(|_| match (w % 3, next() % 4) {
                    (1, 0..=2) if !bytes.is_empty() => bytes[(next() % bytes.len() as u64) as usize],
                    (2, 0) if !special.is_empty() => special[(next() % special.len() as u64) as usize],
                    _ => (next() % u64::from(n.max(1) + 2)) as u32,
                })
                .collect();
            for skip in [true, false] {
                out.clear();
                self.decode_into(&ids, skip, &mut out);
                if t.decode(&ids, skip).map_or(true, |lib| lib != out) {
                    return Err(ids);
                }
            }
        }
        Ok(())
    }
}

fn push_lossy(run: &[u8], out: &mut String) {
    match std::str::from_utf8(run) {
        Ok(s) => out.push_str(s),
        Err(_) => out.push_str(&String::from_utf8_lossy(run)),
    }
}

fn flush_bytes(run: &mut Vec<u8>, out: &mut String) {
    if run.is_empty() {
        return;
    }
    match std::str::from_utf8(run) {
        Ok(s) => out.push_str(s),
        Err(_) => (0..run.len()).for_each(|_| out.push('\u{FFFD}')),
    }
    run.clear();
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;

    fn tokenizer(decoder: serde_json::Value, vocab: &[&str]) -> tokenizers::Tokenizer {
        let vocab: serde_json::Map<String, serde_json::Value> =
            vocab.iter().enumerate().map(|(i, t)| (t.to_string(), i.into())).collect();
        let n = vocab.len();
        let json = serde_json::json!({
            "version": "1.0", "truncation": null, "padding": null, "normalizer": null,
            "added_tokens": [
                {"id": n, "content": "<start>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
                {"id": n + 1, "content": "<tool>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": false}],
            "pre_tokenizer": null, "post_processor": null, "decoder": decoder,
            "model": {"type": "BPE", "dropout": null, "unk_token": null, "continuing_subword_prefix": null,
                "end_of_word_suffix": null, "fuse_unk": false, "byte_fallback": true, "ignore_merges": false,
                "vocab": vocab, "merges": []}
        });
        tokenizers::Tokenizer::from_str(&json.to_string()).unwrap()
    }

    fn sp_vocab() -> Vec<String> {
        let mut v: Vec<String> = (0..=255u32).map(|b| format!("<0x{b:02X}>")).collect();
        v.extend(["▁the", "▁quick", "fox", "▁", "▁▁", "é", "日本", "a▁b", "<0x4>", "<0x+5>", "<0xZZ>"].map(String::from));
        v
    }

    fn assert_matches_library(t: &tokenizers::Tokenizer) {
        let table = DecodeTable::build(t).expect("table");
        let n = t.get_vocab_size(true) as u32;
        let mut seed = 7u64;
        for _ in 0..4000 {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let len = 1 + (seed >> 60) as usize;
            let ids: Vec<u32> = (0..len)
                .map(|k| ((seed >> (k * 5 % 50)) % u64::from(n + 3)) as u32)
                .collect();
            for skip in [true, false] {
                let mut out = String::new();
                table.decode_into(&ids, skip, &mut out);
                assert_eq!(out, t.decode(&ids, skip).unwrap(), "{ids:?} skip={skip}");
            }
        }
    }

    #[test]
    fn byte_fallback_chain_matches_the_library() {
        let v = sp_vocab();
        let refs: Vec<&str> = v.iter().map(String::as_str).collect();
        let t = tokenizer(
            serde_json::json!({"type": "Sequence", "decoders": [
                {"type": "Replace", "pattern": {"String": "\u{2581}"}, "content": " "},
                {"type": "ByteFallback"}, {"type": "Fuse"}]}),
            &refs,
        );
        assert_eq!(chain_of(&t), Some(Chain::ByteFallback));
        assert_matches_library(&t);
        // A split multi-byte run across a skipped special, and an invalid run.
        let (e3, a6) = (0xE6u32, 0x97u32);
        let special = t.token_to_id("<start>").unwrap();
        let table = DecodeTable::build(&t).unwrap();
        for ids in [vec![e3, special, a6, 0xA5], vec![0xFF, 0xFE, 256], vec![e3]] {
            let mut out = String::new();
            table.decode_into(&ids, true, &mut out);
            assert_eq!(out, t.decode(&ids, true).unwrap());
        }
    }

    #[test]
    fn byte_level_chain_matches_the_library() {
        let mut v: Vec<String> = tokenizers::pre_tokenizers::byte_level::ByteLevel::alphabet()
            .into_iter()
            .map(String::from)
            .collect();
        v.sort();
        v.extend(["Ġthe", "Ġquick", "ĊĊ", "Ã©", "plain€text", "æĹ¥"].map(String::from));
        let refs: Vec<&str> = v.iter().map(String::as_str).collect();
        let t = tokenizer(serde_json::json!({"type": "ByteLevel", "add_prefix_space": true, "trim_offsets": true, "use_regex": true}), &refs);
        assert_eq!(chain_of(&t), Some(Chain::ByteLevel));
        assert_matches_library(&t);
    }

    #[test]
    fn other_chains_keep_the_library_decode() {
        let v = sp_vocab();
        let refs: Vec<&str> = v.iter().map(String::as_str).collect();
        for d in [
            serde_json::Value::Null,
            serde_json::json!({"type": "Sequence", "decoders": [
                {"type": "Replace", "pattern": {"String": "\u{2581}"}, "content": " "},
                {"type": "ByteFallback"}, {"type": "Fuse"},
                {"type": "Strip", "content": " ", "start": 1, "stop": 0}]}),
            serde_json::json!({"type": "Metaspace", "replacement": "\u{2581}", "prepend_scheme": "always", "split": true}),
        ] {
            assert!(DecodeTable::build(&tokenizer(d, &refs)).is_none());
        }
    }

    #[test]
    fn byte_level_alphabet_inverts() {
        let inv = byte_level_inverse();
        let alphabet = tokenizers::pre_tokenizers::byte_level::ByteLevel::alphabet();
        assert_eq!(alphabet.len(), 256);
        let mut seen = [false; 256];
        for c in alphabet {
            seen[inv[c as usize].expect("alphabet char") as usize] = true;
        }
        assert!(seen.iter().all(|&s| s));
        assert_eq!(inv['Ġ' as usize], Some(b' '));
        assert_eq!(inv.iter().flatten().count(), 256);
    }

    /// `PLOW_TEST_TOKENIZERS=<tokenizer.json>[:...]`: each real checkpoint tokenizer takes the
    /// table and agrees with the library on random windows and on every token window of the
    /// encoded `docs/runtime` text.
    #[test]
    #[ignore = "set PLOW_TEST_TOKENIZERS"]
    fn real_tokenizers_match_the_library() {
        let paths = std::env::var("PLOW_TEST_TOKENIZERS").unwrap();
        let docs = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../docs/runtime");
        let text: String = std::fs::read_dir(docs)
            .unwrap()
            .flatten()
            .filter_map(|e| std::fs::read_to_string(e.path()).ok())
            .collect();
        for path in paths.split(':') {
            let t = tokenizers::Tokenizer::from_file(path).unwrap();
            let table = DecodeTable::build(&t).unwrap_or_else(|| panic!("{path}: no table"));
            let ids = t.encode(text.as_str(), false).unwrap().get_ids().to_vec();
            let mut out = String::new();
            for w in 1..=8 {
                for win in ids.windows(w).step_by(w.max(3) - 2) {
                    for skip in [true, false] {
                        out.clear();
                        table.decode_into(win, skip, &mut out);
                        assert_eq!(out, t.decode(win, skip).unwrap(), "{path}: {win:?}");
                    }
                }
            }
            println!("{path}: {} ids, {} entries", ids.len(), table.kind.len());
        }
    }
}
