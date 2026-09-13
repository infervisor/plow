//! Checkpoint S inputs: a packet's per-program facts, base/variant pairing, and the route trace.
//!
//! A program is keyed by its role, rows, topology and sparse flag. Its instructions carry
//! `(op, blocks, fj, t, i)`, their segment and the byte size of each operand; operands are named by
//! tensor, so a renumbered tensor table moves nothing. Its object facts are the arms `build.json`
//! lists for it. One `global` program carries the tensor table and the packet-wide object facts
//! (`union`, `plow_config.h` defines).
//!
//! Pairing is by key. A pair whose digests agree is sent without a body; otherwise the two
//! instruction lists are aligned by opcode, so an inserted or removed instruction is one
//! difference and not a shift of every instruction after it.

use crate::asset::devblob::DevBlob;
use packet::dev::{DevInst64, DevOp, TENSOR_NONE16};
use packet::devbuild::ProgramRole;
use plow_asset::knob::{Allow, Target};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::path::Path;

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Key {
    pub kind: &'static str,
    pub rows: u32,
    pub topology: &'static str,
    pub sparse: bool,
}

impl Key {
    pub fn to_json(&self) -> Value {
        json!({"kind": self.kind, "rows": self.rows, "topology": self.topology, "sparse": self.sparse})
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Inst {
    pub op: u16,
    pub blocks: u16,
    pub fj: [u32; 3],
    /// Operand tensor names, `None` for an absent operand.
    pub t: [Option<String>; 8],
    pub i: [u32; 8],
    pub seg: u32,
    pub tb: [u64; 8],
}

pub struct Program {
    pub key: Key,
    pub insts: Vec<Inst>,
    pub facts: Vec<String>,
    /// `name=bytes` for every tensor; the `global` program's only.
    pub tensors: Vec<String>,
    pub digest: String,
}

pub struct Packet {
    pub model: Option<String>,
    pub tp: bool,
    pub programs: Vec<Program>,
    /// The parsed blob, for the route trace's per-program analyses.
    pub blob: Option<DevBlob>,
}

fn sparse_prefill(insts: &[DevInst64]) -> bool {
    insts.iter().any(|d| {
        d.op == DevOp::IndexTpPf as u16
            || d.op == DevOp::FlashGatherPrefill as u16
            || (d.op == DevOp::FlashMlaPrefillFp8 as u16 && d.fj[1] != 0)
    })
}

fn role_key(role: ProgramRole, insts: &[DevInst64]) -> Key {
    let (kind, topology) = match role {
        ProgramRole::PrefillBucket { .. } => ("prefill", "ordinary"),
        ProgramRole::DecodeRung { .. } => ("decode", "ordinary"),
        ProgramRole::PackedSibling { .. } => ("packed", "packed"),
        ProgramRole::TokenBatchBody { .. } => ("token_batch", "ordinary"),
    };
    Key {
        kind,
        rows: role.rows(),
        topology,
        sparse: role.is_prefill_side() && sparse_prefill(insts),
    }
}

/// The per-program arm lists of `build.json` `programs`, keyed like [`role_key`].
fn manifest_arms(man: &Value) -> BTreeMap<(String, String, u64), Vec<String>> {
    let mut out: BTreeMap<(String, String, u64), Vec<String>> = BTreeMap::new();
    for p in man["programs"].as_array().into_iter().flatten() {
        let kind = p["kind"].as_str().unwrap_or_default().to_string();
        let topology = p["topology"].as_str().unwrap_or_default().to_string();
        let rows = p["bucket"].as_u64().or(p["batch"].as_u64()).unwrap_or(0);
        let arms = out.entry((kind, topology, rows)).or_default();
        for a in p["arms"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(Value::as_str)
        {
            if !arms.iter().any(|x| x == a) {
                arms.push(a.to_string());
            }
        }
    }
    for arms in out.values_mut() {
        arms.sort();
    }
    out
}

fn program(key: Key, insts: Vec<Inst>, facts: Vec<String>, tensors: Vec<String>) -> Program {
    let named: Vec<Value> = insts
        .iter()
        .map(|d| json!([d.op, d.blocks, d.fj, d.t, d.i, d.seg, d.tb]))
        .collect();
    let digest = plow_asset::knob::sha256_hex(
        &serde_json::to_vec(&json!([key.to_json(), named, facts, tensors])).unwrap_or_default(),
    );
    Program {
        key,
        insts,
        facts,
        tensors,
        digest,
    }
}

/// Facts of the packet in `assets` (`model.pkt`, `build.json`, `plow_config.h`).
pub fn extract(assets: &Path) -> Result<Packet, String> {
    let blob_path = assets.join("model.pkt");
    let buf = std::fs::read(&blob_path).map_err(|e| format!("{}: {e}", blob_path.display()))?;
    let blob =
        DevBlob::parse_l2(&buf, true).map_err(|e| format!("{}: {e}", blob_path.display()))?;
    let man: Value = std::fs::read(assets.join("build.json"))
        .ok()
        .and_then(|b| serde_json::from_slice(&b).ok())
        .unwrap_or(Value::Null);
    let arms = manifest_arms(&man);
    let mut programs = Vec::with_capacity(blob.progs.len() + 1);
    for p in &blob.progs {
        let key = role_key(p.role, &p.insts);
        let mut segs = vec![u32::MAX; p.insts.len()];
        for e in &p.stream {
            if let Some(s) = segs.get_mut(e.inst as usize) {
                *s = e.seg as u32;
            }
        }
        let tensor = |h: u16| blob.tensors.get(h as usize).filter(|_| h != TENSOR_NONE16);
        let insts = p
            .insts
            .iter()
            .zip(&segs)
            .map(|(d, &seg)| Inst {
                op: d.op,
                blocks: d.blocks,
                fj: d.fj,
                t: d.t.map(|h| tensor(h).map(|x| x.name.clone())),
                i: d.i,
                seg,
                tb: d.t.map(|h| tensor(h).map_or(0, |x| x.bytes)),
            })
            .collect();
        let manifest_kind = if key.kind == "packed" {
            "prefill"
        } else {
            key.kind
        };
        let facts = arms
            .get(&(
                manifest_kind.to_string(),
                key.topology.to_string(),
                key.rows as u64,
            ))
            .cloned()
            .unwrap_or_default();
        programs.push(program(key, insts, facts, Vec::new()));
    }
    let mut facts: Vec<String> = man["union"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .map(|a| format!("union:{a}"))
        .collect();
    if let Ok(h) = std::fs::read_to_string(assets.join("plow_config.h")) {
        facts.extend(
            h.lines()
                .filter(|l| l.trim_start().starts_with("#define "))
                .map(|l| l.trim().to_string()),
        );
    }
    let mut table: Vec<String> = blob
        .tensors
        .iter()
        .map(|t| format!("{}={}", t.name, t.bytes))
        .collect();
    table.sort();
    programs.push(program(
        Key {
            kind: "global",
            rows: 0,
            topology: "global",
            sparse: false,
        },
        Vec::new(),
        facts,
        table,
    ));
    let model = man
        .pointer("/knobs/target/model")
        .and_then(Value::as_str)
        .map(String::from);
    Ok(Packet {
        model,
        tp: blob.tp.is_some(),
        programs,
        blob: Some(blob),
    })
}

/// Aligns `a` and `b` by equal elements with the fewest insertions and deletions (Myers). Past
/// `max_d` edits the rest pairs positionally.
pub fn align<T: PartialEq>(a: &[T], b: &[T], max_d: usize) -> Vec<(Option<usize>, Option<usize>)> {
    let (n, m) = (a.len() as isize, b.len() as isize);
    let limit = (max_d as isize).min(n + m);
    let mut trace: Vec<Vec<isize>> = Vec::new();
    let mut end = None;
    'outer: for d in 0..=limit {
        let mut v = vec![0isize; (2 * d + 1) as usize];
        for k in (-d..=d).step_by(2) {
            let prev = |k: isize| trace[(d - 1) as usize][(k + d - 1) as usize];
            let mut x = if d == 0 {
                0
            } else if k == -d || (k != d && prev(k - 1) < prev(k + 1)) {
                prev(k + 1)
            } else {
                prev(k - 1) + 1
            };
            let mut y = x - k;
            while x < n && y < m && a[x as usize] == b[y as usize] {
                x += 1;
                y += 1;
            }
            v[(k + d) as usize] = x;
            if x >= n && y >= m {
                trace.push(v);
                end = Some(d);
                break 'outer;
            }
        }
        trace.push(v);
    }
    let Some(dmax) = end else {
        let len = a.len().max(b.len());
        return (0..len)
            .map(|i| ((i < a.len()).then_some(i), (i < b.len()).then_some(i)))
            .collect();
    };
    let (mut x, mut y) = (n, m);
    let mut out = Vec::new();
    for d in (1..=dmax).rev() {
        let v = &trace[(d - 1) as usize];
        let at = |k: isize| v[(k + d - 1) as usize];
        let k = x - y;
        let pk = if k == -d || (k != d && at(k - 1) < at(k + 1)) {
            k + 1
        } else {
            k - 1
        };
        let px = at(pk);
        let py = px - pk;
        while x > px && y > py {
            x -= 1;
            y -= 1;
            out.push((Some(x as usize), Some(y as usize)));
        }
        if x == px {
            y -= 1;
            out.push((None, Some(y as usize)));
        } else {
            x -= 1;
            out.push((Some(x as usize), None));
        }
    }
    while x > 0 && y > 0 {
        x -= 1;
        y -= 1;
        out.push((Some(x as usize), Some(y as usize)));
    }
    out.reverse();
    out
}

const MAX_EDITS: usize = 4096;
const NO_TENSOR: u64 = u32::MAX as u64;

/// One pair per key: `{"a", "b", "body"}`, the body absent where the digests agree.
pub fn pairs(base: &Packet, variant: &Packet) -> Result<Vec<Value>, String> {
    fn by_key(p: &Packet) -> Result<BTreeMap<&Key, &Program>, String> {
        let mut m = BTreeMap::new();
        for prog in &p.programs {
            if m.insert(&prog.key, prog).is_some() {
                return Err(format!("two programs share the key {:?}", prog.key));
            }
        }
        Ok(m)
    }
    let (a, b) = (by_key(base)?, by_key(variant)?);
    let mut ids: BTreeMap<&str, u64> = BTreeMap::new();
    for prog in base.programs.iter().chain(&variant.programs) {
        for name in prog.insts.iter().flat_map(|d| d.t.iter().flatten()) {
            ids.entry(name.as_str()).or_insert(0);
        }
    }
    for (n, id) in ids.values_mut().enumerate() {
        *id = n as u64;
    }
    let inst = |d: Option<&Inst>| match d {
        None => Value::Null,
        Some(d) => json!([
            d.op,
            d.blocks,
            d.fj,
            d.t.iter()
                .map(|t| t.as_deref().map_or(NO_TENSOR, |n| ids[n]))
                .collect::<Vec<_>>(),
            d.i,
            d.seg,
            d.tb
        ]),
    };
    let empty = Program {
        key: Key {
            kind: "",
            rows: 0,
            topology: "",
            sparse: false,
        },
        insts: Vec::new(),
        facts: Vec::new(),
        tensors: Vec::new(),
        digest: String::new(),
    };
    let mut keys: Vec<&Key> = a.keys().chain(b.keys()).copied().collect();
    keys.sort();
    keys.dedup();
    Ok(keys
        .into_iter()
        .map(|k| {
            let (pa, pb) = (a.get(k).copied(), b.get(k).copied());
            let key = |p: Option<&Program>| p.map_or(Value::Null, |p| p.key.to_json());
            let body = match (pa, pb) {
                (Some(x), Some(y)) if x.digest == y.digest => Value::Null,
                _ => {
                    let (x, y) = (pa.unwrap_or(&empty), pb.unwrap_or(&empty));
                    let ops = |p: &Program| p.insts.iter().map(|d| d.op).collect::<Vec<_>>();
                    let insts: Vec<Value> = align(&ops(x), &ops(y), MAX_EDITS)
                        .into_iter()
                        .map(|(i, j)| {
                            json!([inst(i.map(|i| &x.insts[i])), inst(j.map(|j| &y.insts[j]))])
                        })
                        .collect();
                    json!({
                        "insts": insts,
                        "tensors": [x.tensors, y.tensors],
                        "facts": [x.facts, y.facts],
                    })
                }
            };
            json!({"a": key(pa), "b": key(pb), "body": body})
        })
        .collect())
}

/// Runtime knobs the route trace models.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RouteKnobs {
    pub tail_sparse_ctx: Option<u32>,
    pub decode_min_rung: Option<u32>,
    pub union_skip: bool,
}

impl RouteKnobs {
    pub fn set(&mut self, id: &str, value: &str) -> Result<(), String> {
        let num = || -> Result<Option<u32>, String> {
            if value.is_empty() {
                return Ok(None);
            }
            value
                .parse::<u32>()
                .map(Some)
                .map_err(|e| format!("{id}={value}: {e}"))
        };
        match id {
            "rt.tail_sparse_ctx" => self.tail_sparse_ctx = num()?,
            "rt.decode_min_rung" => self.decode_min_rung = num()?,
            "rt.union_skip" => self.union_skip = matches!(value, "1" | "true"),
            _ => return Err(format!("no route model for {id}")),
        }
        Ok(())
    }
}

/// `{"prefill": [{"id", "from", "to"}], "decode": [rows]}`: prefill spans and decode slot counts.
#[derive(serde::Deserialize, Default)]
pub struct Workload {
    #[serde(default)]
    pub prefill: Vec<Span>,
    #[serde(default)]
    pub decode: Vec<u32>,
}

#[derive(serde::Deserialize)]
pub struct Span {
    pub id: String,
    pub from: u32,
    pub to: u32,
}

/// One planned step: its label, the program that serves it, and the segments the host skips.
pub type Step = (String, Key, Vec<usize>);

/// Which program serves each step of `w`, planned the way the AMD serve plans it, with the
/// sparse routes assumed active (`PLOW_MLA_PF_AITER=1`).
#[cfg(feature = "hsa")]
pub fn route(p: &Packet, w: &Workload, knobs: RouteKnobs) -> Result<Vec<Step>, String> {
    let cfg = &crate::config::RuntimeConfig::get().amd;
    let prefill: BTreeMap<u32, &Key> = p
        .programs
        .iter()
        .filter(|x| x.key.kind == "prefill")
        .map(|x| (x.key.rows, &x.key))
        .collect();
    let skippable = |key: &Key| -> Result<Vec<usize>, String> {
        let Some(blob) = &p.blob else {
            return Ok(Vec::new());
        };
        let Some(prog) = blob
            .progs
            .iter()
            .find(|q| role_key(q.role, &q.insts) == *key)
        else {
            return Ok(Vec::new());
        };
        crate::exec::skippable_unions(prog, &blob.tensors).map_err(|e| e.to_string())
    };
    let buckets: Vec<u32> = prefill.keys().copied().collect();
    let mut out = Vec::new();
    for s in &w.prefill {
        let mut chunks = crate::exec::amd::plan_chunks_cfg(
            &buckets,
            s.to.saturating_sub(s.from),
            cfg.launch_rows.unwrap_or(crate::exec::amd::LAUNCH_ROWS),
            cfg.ragged_chunk,
        )
        .map_err(|e| e.to_string())?;
        if let Some(min_ctx) = knobs.tail_sparse_ctx {
            let sparse = prefill.values().filter(|k| k.sparse).map(|k| k.rows).max();
            let dense = |w: u32| prefill.get(&w).is_some_and(|k| !k.sparse);
            crate::serve::engine::retarget_dense_tail(&mut chunks, s.from, min_ctx, sparse, dense);
        }
        let mut c0 = s.from;
        for (i, ch) in chunks.iter().enumerate() {
            let key = *prefill
                .get(ch)
                .ok_or_else(|| format!("no prefill bucket of {ch} rows"))?;
            let skip = if knobs.union_skip && key.sparse && c0 >= crate::exec::SPAN_MIN_PRIOR {
                skippable(key)?
            } else {
                Vec::new()
            };
            let rows = s.to.saturating_sub(c0).min(*ch);
            out.push((
                format!("{}:{i} rows={rows} prior={c0}", s.id),
                key.clone(),
                skip,
            ));
            c0 += ch;
        }
    }
    let mut rungs: Vec<&Key> = p
        .programs
        .iter()
        .filter(|x| x.key.kind == "decode")
        .map(|x| &x.key)
        .collect();
    rungs.sort();
    for &rows in &w.decode {
        let want = if p.tp {
            rows.max(knobs.decode_min_rung.unwrap_or(8).max(1))
        } else {
            rows
        };
        let key = rungs
            .iter()
            .find(|k| k.rows >= want)
            .or(rungs.last())
            .ok_or("the packet has no decode rung")?;
        out.push((format!("decode:{rows}"), (*key).clone(), Vec::new()));
    }
    Ok(out)
}

pub fn route_steps(off: &[Step], on: &[Step]) -> Vec<Value> {
    let mut labels: Vec<&String> = off.iter().chain(on).map(|(l, _, _)| l).collect();
    labels.sort();
    labels.dedup();
    let find = |side: &[Step], l: &String| side.iter().find(|(x, _, _)| x == l).cloned();
    labels
        .into_iter()
        .map(|l| {
            let (a, b) = (find(off, l), find(on, l));
            json!({
                "label": l,
                "off": a.as_ref().map_or(Value::Null, |s| s.1.to_json()),
                "on": b.as_ref().map_or(Value::Null, |s| s.1.to_json()),
                "off_skip": a.map_or(Vec::new(), |s| s.2),
                "on_skip": b.map_or(Vec::new(), |s| s.2),
            })
        })
        .collect()
}

pub fn class_table() -> Value {
    Value::Array(
        packet::opclass::class_table()
            .into_iter()
            .map(|(op, cs)| json!([op, cs]))
            .collect(),
    )
}

/// The declared scope of `knob`, from the generated registry; empty when it declares none.
pub fn declared_scope(knob: &str) -> &'static [Allow] {
    plow_asset::knob_gen::KNOBS
        .iter()
        .find(|k| k.id == knob)
        .map_or(&[], |k| k.scope)
}

/// The checkpoint S payload. `delta` is empty for an off→off comparison, which S checks against
/// the empty scope.
pub fn payload(
    model: &str,
    delta: &[String],
    scope: &[Allow],
    pairs: Vec<Value>,
    routes: Vec<Value>,
) -> Value {
    json!({
        "model": model,
        "classes": class_table(),
        "delta": delta,
        "scope": scope.iter().map(Allow::to_json).collect::<Vec<_>>(),
        "pairs": pairs,
        "routes": routes,
    })
}

/// The target a packet names, for callers that want it.
pub fn target_of(assets: &Path) -> Option<Target> {
    let man: Value =
        serde_json::from_slice(&std::fs::read(assets.join("build.json")).ok()?).ok()?;
    Target::from_json(man.pointer("/knobs/target")?)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn inst(op: u16, blocks: u16, tensor: &str) -> Inst {
        let mut t: [Option<String>; 8] = Default::default();
        t[0] = Some(tensor.to_string());
        Inst {
            op,
            blocks,
            fj: [0; 3],
            t,
            i: [0; 8],
            seg: 0,
            tb: [0; 8],
        }
    }

    fn packet(progs: Vec<(&'static str, u32, Vec<Inst>)>) -> Packet {
        Packet {
            model: Some("glm_moe_dsa".into()),
            tp: true,
            blob: None,
            programs: progs
                .into_iter()
                .map(|(kind, rows, insts)| {
                    let key = Key {
                        kind,
                        rows,
                        topology: "ordinary",
                        sparse: false,
                    };
                    program(key, insts, Vec::new(), Vec::new())
                })
                .collect(),
        }
    }

    #[test]
    fn alignment_pairs_an_inserted_instruction_with_nothing() {
        let a = [8u16, 87, 29, 8];
        let b = [8u16, 29, 8];
        assert_eq!(
            align(&a, &b, 16),
            vec![
                (Some(0), Some(0)),
                (Some(1), None),
                (Some(2), Some(1)),
                (Some(3), Some(2))
            ]
        );
        assert_eq!(align(&b, &a, 16)[1], (None, Some(1)));
        let (x, y) = ([1u16, 2, 3], [4u16, 5]);
        let past = align(&x, &y, 1);
        assert_eq!(
            past.len(),
            3,
            "over the edit cap the rest pairs positionally"
        );
    }

    #[test]
    fn pairs_send_bodies_only_where_the_digests_differ() {
        let a = packet(vec![
            ("prefill", 128, vec![inst(8, 64, "x")]),
            ("decode", 8, vec![inst(8, 8, "y")]),
        ]);
        let b = packet(vec![
            ("prefill", 128, vec![inst(8, 16, "x")]),
            ("decode", 8, vec![inst(8, 8, "y")]),
        ]);
        let p = pairs(&a, &b).unwrap();
        let decode = p.iter().find(|x| x["a"]["kind"] == "decode").unwrap();
        assert!(decode["body"].is_null());
        let prefill = p.iter().find(|x| x["a"]["kind"] == "prefill").unwrap();
        assert_eq!(prefill["body"]["insts"][0][1][1], 16);

        let c = packet(vec![("prefill", 128, vec![inst(8, 64, "x")])]);
        let missing = pairs(&a, &c).unwrap();
        assert!(
            missing.iter().any(|x| x["b"].is_null()),
            "a dropped program pairs with null"
        );
    }

    #[test]
    fn operands_are_named_so_a_renumbered_table_moves_nothing() {
        let a = packet(vec![("prefill", 128, vec![inst(8, 64, "w.q")])]);
        let b = packet(vec![("prefill", 128, vec![inst(8, 64, "w.q")])]);
        assert_eq!(a.programs[0].digest, b.programs[0].digest);
        assert!(pairs(&a, &b).unwrap()[0]["body"].is_null());
    }

    #[test]
    fn two_programs_with_one_key_are_refused() {
        let a = packet(vec![
            ("prefill", 128, vec![]),
            ("prefill", 128, vec![inst(8, 1, "z")]),
        ]);
        assert!(pairs(&a, &a).is_err());
    }

    /// The incident fixtures of the plan's §4.6, and the G1/G4 candidates, on real packets.
    /// `KNOB_SCOPE_FIXTURES` holds asset directories: `small-cus-off`, `small-cus-off-again` (same
    /// tree), `small-cus-narrow` (6deb4025), `small-cus-wide`, `glm53` (the production emit),
    /// `rowsplit-off`, `gemma-role-on`, and `seed-default`, `seed-off`, `seed-on` (one tree).
    #[test]
    #[ignore = "requires plow_verify and the fixture packets in KNOB_SCOPE_FIXTURES"]
    fn incident_fixtures() {
        let dir = std::path::PathBuf::from(
            std::env::var("KNOB_SCOPE_FIXTURES").expect("set KNOB_SCOPE_FIXTURES"),
        );
        let load = |name: &str| extract(&dir.join(name)).expect(name);
        let check = |knob: &str, delta: bool, a: &Packet, b: &Packet, routes: Vec<Value>| {
            let delta = if delta {
                vec![knob.to_string()]
            } else {
                Vec::new()
            };
            let p = payload(
                "glm_moe_dsa",
                &delta,
                declared_scope(knob),
                pairs(a, b).unwrap(),
                routes,
            );
            lean_verify::checkpoints::scope::check_scope(&p).expect("plow_verify answered")
        };
        let small = "emit.glm_pf_small_cus";
        let off = load("small-cus-off");
        let cert = check(small, false, &off, &load("small-cus-off-again"), vec![]);
        assert!(cert.ok, "off→off: {cert:?}");
        let cert = check(small, true, &off, &load("small-cus-narrow"), vec![]);
        assert!(
            !cert.ok && cert.reason.unwrap().contains("ops [29]"),
            "6deb4025 narrows XReduceTwoShot"
        );
        let cert = check(small, true, &off, &load("small-cus-wide"), vec![]);
        assert!(cert.ok, "wide collectives: {cert:?}");

        let glm = load("glm53");
        let cert = check(
            "emit.glm_rowsplit_attn",
            false,
            &glm,
            &load("rowsplit-off"),
            vec![],
        );
        assert!(cert.ok, "row split off: {cert:?}");
        let role = load("gemma-role-on");
        let cert = check(
            "emit.gemma4_sm90_hd256_gqa2_role",
            true,
            &glm,
            &role,
            vec![],
        );
        assert!(cert.ok && cert.notes.unwrap().contains("empty_effect"));

        let seed = "emit.glm_moe_shared_seed";
        let (seed_default, seed_off) = (load("seed-default"), load("seed-off"));
        let cert = check(seed, false, &seed_default, &seed_off, vec![]);
        assert!(cert.ok, "shared seed off→off: {cert:?}");
        let cert = check(seed, true, &seed_off, &load("seed-on"), vec![]);
        assert!(cert.ok, "shared seed on: {cert:?}");

        #[cfg(feature = "hsa")]
        {
            let span = |id: &str, from, to| Span {
                id: id.into(),
                from,
                to,
            };
            let w = Workload {
                prefill: vec![
                    span("a", 0, 70_000),
                    span("b", 12_000, 21_000),
                    span("c", 20_000, 23_000),
                    span("between-floors", 9_000, 9_600),
                ],
                decode: vec![1, 4, 8, 20],
            };
            let at = |ctx| RouteKnobs {
                tail_sparse_ctx: Some(ctx),
                ..RouteKnobs::default()
            };
            let a = route(&glm, &w, at(16384)).unwrap();
            let b = route(&glm, &w, at(8192)).unwrap();
            let moved: Vec<(&Key, &Key)> = a
                .iter()
                .zip(&b)
                .filter(|(x, y)| x.1 != y.1)
                .map(|(x, y)| (&x.1, &y.1))
                .collect();
            assert!(!moved.is_empty());
            for (from, to) in &moved {
                assert!(
                    from.kind == "prefill" && !from.sparse,
                    "moved from {from:?}"
                );
                assert!(
                    to.kind == "prefill" && to.sparse && to.rows == 8192,
                    "moved to {to:?}"
                );
            }
            let cert = check("rt.tail_sparse_ctx", true, &glm, &glm, route_steps(&a, &b));
            assert!(cert.ok, "{cert:?}");

            let skip = |on| RouteKnobs {
                union_skip: on,
                ..at(16384)
            };
            let (off, on) = (
                route(&glm, &w, skip(false)).unwrap(),
                route(&glm, &w, skip(true)).unwrap(),
            );
            let skipped: Vec<&Step> = on.iter().filter(|s| !s.2.is_empty()).collect();
            assert!(!skipped.is_empty(), "union skip fires on the sparse chunks");
            assert!(skipped.iter().all(|s| s.1.sparse && s.1.kind == "prefill"));
            let cert = check("rt.union_skip", false, &glm, &glm, route_steps(&off, &off));
            assert!(cert.ok, "union skip off→off: {cert:?}");
            let cert = check("rt.union_skip", true, &glm, &glm, route_steps(&off, &on));
            assert!(cert.ok, "union skip on: {cert:?}");
        }
    }
}
