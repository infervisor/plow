//! Word segmentation by a linear-chain CRF over character n-gram and dictionary-word features
//! (the pkuseg feature templates), decoded by Viterbi in f64, then a dictionary merge pass. Model
//! data are packet text tables `PREFIX.{features,weights,unigram,tags,merge,norm,meta}`:
//! - `features`: feature strings, one per line in weight-row order (an empty line never fires)
//! - `weights`: f64 LE, `n_feature * n_tag` node weights then `n_tag * n_tag` transitions
//! - `unigram`: dictionary words the word features test; `merge`: words the merge pass joins
//! - `tags`: tag names by index (a tag containing `B` starts a word)
//! - `norm`: `char\tnode` replacements applied before feature extraction
//! - `meta`: `{"word_min", "word_max", "word_feature"}`
//!
//! Scores and ties follow the reference decoder exactly (sums in feature order, `>=` on ties),
//! so the segmentation is identical, not approximately equal.

use std::collections::{HashMap, HashSet};

use super::rules::Tables;
use crate::{Result, RuntimeError};

pub struct CrfSegmenter {
    features: HashMap<Box<str>, u32>,
    w: Vec<f64>,
    n_tag: usize,
    starts: Vec<bool>,
    unigram: HashSet<Box<str>>,
    merge: HashSet<Box<str>>,
    norm: HashMap<char, Box<str>>,
    word_min: usize,
    word_max: usize,
    word_feature: bool,
}

const NO_WORD: &str = "**noWord";

impl CrfSegmenter {
    pub fn load(t: &Tables, prefix: &str) -> Result<Self> {
        let name = |s: &str| format!("{prefix}.{s}");
        let lines = |s: &str| -> Result<Vec<&str>> { Ok(t.text(&name(s))?.split('\n').collect()) };
        let bad = |m: &str| RuntimeError::Rejected(format!("{prefix}: {m}"));
        let feats = lines("features")?;
        let features: HashMap<Box<str>, u32> =
            feats.iter().enumerate().filter(|(_, f)| !f.is_empty()).map(|(i, f)| ((*f).into(), i as u32)).collect();
        let tags = lines("tags")?;
        let n_tag = tags.len();
        let wb = t.bytes(&name("weights"))?;
        let w: Vec<f64> = wb.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect();
        if w.len() != feats.len() * n_tag + n_tag * n_tag || wb.len() % 8 != 0 {
            return Err(bad("weights disagree with features x tags"));
        }
        let set = |s: &str| -> Result<HashSet<Box<str>>> { Ok(lines(s)?.into_iter().filter(|l| !l.is_empty()).map(Into::into).collect()) };
        let mut norm = HashMap::new();
        for l in lines("norm")?.into_iter().filter(|l| !l.is_empty()) {
            let (k, v) = l.split_once('\t').ok_or_else(|| bad("norm line"))?;
            let mut c = k.chars();
            let (Some(ch), None) = (c.next(), c.next()) else { return Err(bad("norm key")) };
            norm.insert(ch, v.into());
        }
        let meta: serde_json::Value = serde_json::from_str(t.text(&name("meta"))?).map_err(|e| bad(&e.to_string()))?;
        let num = |k: &str| meta[k].as_u64().map(|v| v as usize).ok_or_else(|| bad(k));
        Ok(CrfSegmenter {
            features,
            w,
            n_tag,
            starts: tags.iter().map(|t| t.contains('B')).collect(),
            unigram: set("unigram")?,
            merge: set("merge")?,
            norm,
            word_min: num("word_min")?,
            word_max: num("word_max")?,
            word_feature: meta["word_feature"].as_bool().ok_or_else(|| bad("word_feature"))?,
        })
    }

    /// Whitespace-separated fragments, each segmented; words joined by one space.
    pub fn cut(&self, text: &str) -> String {
        let mut words: Vec<String> = Vec::new();
        for frag in text.split(py_space).filter(|f| !f.is_empty()) {
            words.extend(self.merge_words(self.crf_words(frag)));
        }
        words.join(" ")
    }

    fn crf_words(&self, frag: &str) -> Vec<String> {
        let chars: Vec<char> = frag.chars().collect();
        let nodes: Vec<String> = chars.iter().map(|c| self.norm.get(c).map_or_else(|| c.to_string(), |s| s.to_string())).collect();
        let (n, nt) = (nodes.len(), self.n_tag);
        let mut node = vec![0f64; n * nt];
        let mut key = String::new();
        let mut flist = Vec::new();
        for i in 0..n {
            flist.clear();
            self.features_at(i, &nodes, &mut flist, &mut key);
            for &f in &flist {
                for s in 0..nt {
                    node[i * nt + s] += self.w[f as usize * nt + s];
                }
            }
        }
        let back = self.w.len() - nt * nt;
        // edge[a][b]: tag a at i followed by tag b at i + 1.
        let edge: Vec<f64> = (0..nt * nt).map(|k| 1.0 + self.w[back + (k % nt) * nt + k / nt]).collect();
        let mut best = vec![0f64; n * nt];
        let mut prev = vec![0usize; n * nt];
        best[(n - 1) * nt..].copy_from_slice(&node[(n - 1) * nt..]);
        for i in (0..n.saturating_sub(1)).rev() {
            for y in 0..nt {
                for yp in 0..nt {
                    let sc = best[(i + 1) * nt + yp] + node[i * nt + y] + edge[y * nt + yp];
                    if yp == 0 || sc >= best[i * nt + y] {
                        best[i * nt + y] = sc;
                        prev[i * nt + y] = yp;
                    }
                }
            }
        }
        let mut tag = 0;
        for y in 1..nt {
            if best[tag] < best[y] {
                tag = y;
            }
        }
        let mut words: Vec<String> = Vec::new();
        for (i, &c) in chars.iter().enumerate() {
            if i > 0 {
                tag = prev[(i - 1) * nt + tag];
            }
            if i == 0 || self.starts[tag] {
                words.push(String::new());
            }
            words.last_mut().unwrap().push(c);
        }
        words
    }

    fn features_at(&self, i: usize, nodes: &[String], flist: &mut Vec<u32>, key: &mut String) {
        let n = nodes.len();
        let mut push = |parts: &[&str], flist: &mut Vec<u32>| {
            key.clear();
            parts.iter().for_each(|p| key.push_str(p));
            if let Some(&f) = self.features.get(key.as_str()) {
                flist.push(f);
            }
        };
        flist.push(0);
        let c = nodes[i].as_str();
        push(&["c.", c], flist);
        if i > 0 {
            push(&["c-1.", &nodes[i - 1]], flist);
            push(&["c-1c.", &nodes[i - 1], ".", c], flist);
        }
        if i + 1 < n {
            push(&["c1.", &nodes[i + 1]], flist);
            push(&["cc1.", c, ".", &nodes[i + 1]], flist);
        }
        if i > 1 {
            push(&["c-2.", &nodes[i - 2]], flist);
            push(&["c-2c-1.", &nodes[i - 2], ".", &nodes[i - 1]], flist);
        }
        if i + 2 < n {
            push(&["c2.", &nodes[i + 2]], flist);
        }
        if !self.word_feature {
            return;
        }
        // get_slice_str: "" unless nodes[start..start + len] lies inside the fragment.
        let slice = |start: isize, len: usize| -> Option<String> {
            (start >= 0 && (start as usize) < n && start as usize + len <= n).then(|| nodes[start as usize..start as usize + len].concat())
        };
        let word = |s: Option<String>| s.filter(|w| self.unigram.contains(w.as_str()));
        let lens: Vec<usize> = (self.word_min..=self.word_max).rev().collect();
        let ii = i as isize;
        let mut pre_in = Vec::with_capacity(lens.len());
        for &l in &lens {
            let w = word(slice(ii - l as isize + 1, l));
            if let Some(w) = &w {
                push(&["w-1.", w], flist);
            }
            pre_in.push(w);
        }
        let mut post_in = Vec::with_capacity(lens.len());
        for &l in &lens {
            let w = word(slice(ii, l));
            if let Some(w) = &w {
                push(&["w1.", w], flist);
            }
            post_in.push(w);
        }
        let pre_ex: Vec<_> = lens.iter().map(|&l| word(slice(ii - l as isize, l))).collect();
        let post_ex: Vec<_> = lens.iter().map(|&l| word(slice(ii + 1, l))).collect();
        let s = |w: &Option<String>| w.as_deref().unwrap_or(NO_WORD).to_string();
        for pre in &pre_ex {
            for post in &post_in {
                push(&["ww.l.", &s(pre), "*", &s(post)], flist);
            }
        }
        for pre in &pre_in {
            for post in &post_ex {
                push(&["ww.r.", &s(pre), "*", &s(post)], flist);
            }
        }
    }

    /// Join runs of 7..2 words whose concatenation is a merge word unless every part is one.
    fn merge_words(&self, mut sent: Vec<String>) -> Vec<String> {
        if self.merge.is_empty() {
            return sent;
        }
        for m in (2..8).rev() {
            if sent.len() < m {
                continue;
            }
            let mut end = sent.len() - m;
            let mut i = 0;
            while i < end + 1 {
                let merged = sent[i..i + m].concat();
                if self.merge.contains(merged.as_str()) && !sent[i..i + m].iter().all(|w| self.merge.contains(w.as_str())) {
                    sent.splice(i..i + m, std::iter::once(merged));
                    i += 1;
                    if sent.len() < m {
                        break;
                    }
                    end = sent.len() - m;
                } else {
                    i += 1;
                }
            }
        }
        sent
    }
}

/// Python `str.split()` separators (`str.isspace`).
fn py_space(c: char) -> bool {
    c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c)
}
