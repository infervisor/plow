//! Offline prefill measurements, not automatic placement policy.

#[cfg(all(feature = "ane", target_os = "macos"))]
mod bench {
    use clap::Parser;
    use plow_asset::decode_objects::image_sha256;
    use plowrt::exec::apple::MetalEngine;
    use plowrt::text::tokenizer::load_tokenizer;
    use serde::{Deserialize, Serialize};
    use serde_json::{json, Value};
    use std::collections::BTreeMap;
    use std::path::{Path, PathBuf};
    use std::process::Command;
    use std::time::Instant;

    #[derive(Parser)]
    pub struct Args {
        gpu_blob: PathBuf,
        hetero_blob: PathBuf,
        checkpoint: PathBuf,
        #[arg(
            long,
            value_delimiter = ',',
            default_value = "31,32,33,63,64,65,91,127,128"
        )]
        rows: Vec<usize>,
        #[arg(long, value_delimiter = ',', default_value = "25,50,75")]
        shares: Vec<u32>,
        #[arg(long, default_value_t = 20)]
        reps: usize,
        #[arg(long, default_value_t = 3)]
        warmup: usize,
        /// Both engines resident. Default = fresh processes in GPU/split/split/GPU order.
        #[arg(long)]
        interleaved: bool,
        /// Diagnostic timings, not promotion samples.
        #[arg(long)]
        profile: bool,
        #[arg(long)]
        output: PathBuf,
        #[arg(long, hide = true)]
        worker: Option<String>,
    }

    #[derive(Clone, Debug, Serialize, Deserialize)]
    struct Sample {
        wall_ms: f64,
        first: u32,
        profile: Value,
    }

    #[derive(Serialize, Deserialize)]
    struct Cell {
        rows: usize,
        ane_pct: Option<u32>,
        bucket: u32,
        gpu_rows: usize,
        ane_rows: usize,
        ane_padded_rows: usize,
        first_request_ms: f64,
        logits: Vec<f32>,
        samples: Vec<Sample>,
    }

    #[derive(Serialize, Deserialize)]
    struct Run {
        identity: Value,
        load_ms: f64,
        memory_before: Value,
        memory_after: Value,
        warm_memory: Vec<Value>,
        cells: Vec<Cell>,
    }

    fn command(program: &str, args: &[&str]) -> String {
        let out = Command::new(program).args(args).output().expect(program);
        assert!(
            out.status.success(),
            "{program}: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        String::from_utf8(out.stdout).unwrap().trim().to_owned()
    }

    fn memory() -> Value {
        let mut usage = std::mem::MaybeUninit::<libc::rusage>::uninit();
        // SAFETY: getrusage initializes this structure on success; macOS reports bytes.
        let peak = unsafe {
            assert_eq!(libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()), 0);
            usage.assume_init().ru_maxrss
        };
        json!({
            "process_peak_rss_bytes": peak,
            "swap": command("/usr/sbin/sysctl", &["-n", "vm.swapusage"]),
            "vm_stat": command("/usr/bin/vm_stat", &[]),
            "pressure_level": command("/usr/sbin/sysctl", &["-n", "kern.memorystatus_vm_pressure_level"]),
        })
    }

    fn identity(eng: &MetalEngine, args: &Args) -> Value {
        let mut weights = BTreeMap::new();
        let mut kv = BTreeMap::new();
        for (h, name) in eng.model.names.iter().enumerate() {
            if packet::names::is_checkpoint_weight(name) {
                let bytes = eng.tensor_bytes(h);
                weights.insert(
                    name.clone(),
                    json!({"bytes": bytes.len(), "sha256": image_sha256(bytes)}),
                );
            } else if name.starts_with("kv.") {
                kv.insert(name.clone(), eng.tensor_bytes(h).len());
            }
        }
        assert!(!weights.is_empty(), "no weights identified");
        let decode: Vec<_> = eng.model.blob.progs[eng.model.dec_ix..]
            .iter()
            .map(|p| {
                let insts: Vec<_> = p
                    .insts
                    .iter()
                    .map(|d| {
                        let names: Vec<_> =
                            d.t.iter()
                                .map(|&h| eng.model.names.get(h as usize))
                                .collect();
                        json!({"op": d.op, "fj": d.fj, "i": d.i, "t": names})
                    })
                    .collect();
                json!({"rows": p.t, "sha256": image_sha256(&serde_json::to_vec(&insts).unwrap())})
            })
            .collect();
        json!({
            "gpu": eng.gpu_name,
            "os": command("/usr/bin/sw_vers", &["-buildVersion"]),
            "backend": "metal-coreml-row",
            "metal_sha256": image_sha256(include_bytes!("../../../runtime/apple/interp.metal")),
            "ane_graph_sha256": image_sha256(include_bytes!("../src/exec/apple/hetero.rs")),
            "coreml_wrapper_sha256": image_sha256(include_bytes!("../src/exec/ane.rs")),
            "metal_runtime_sha256": image_sha256(include_bytes!("../src/exec/apple/mod.rs")),
            "gpu_packet_sha256": image_sha256(&std::fs::read(&args.gpu_blob).unwrap()),
            "split_packet_sha256": image_sha256(&std::fs::read(&args.hetero_blob).unwrap()),
            "weights": weights, "kv_tensors": kv, "decode_programs": decode,
            "prefill_buckets": eng.prefill_buckets(), "batch": eng.model.batch,
            "prefill_opcodes": eng.model.blob.progs[..eng.model.dec_ix].iter().map(|p| {
                p.insts.iter().map(|d| d.op).collect::<std::collections::BTreeSet<_>>()
            }).collect::<Vec<_>>(),
            "environment": std::env::vars().filter(|(k, _)| k.starts_with("PLOW_")).collect::<BTreeMap<_, _>>(),
        })
    }

    fn vm_counter(memory: &Value, name: &str) -> Option<u64> {
        memory["vm_stat"].as_str()?.lines().find_map(|line| {
            let (key, value) = line.split_once(':')?;
            (key == name)
                .then(|| value.trim().trim_end_matches('.').parse().ok())
                .flatten()
        })
    }

    fn memory_quiet(before: &Value, after: &Value) -> bool {
        before["pressure_level"] == "1"
            && after["pressure_level"] == "1"
            && ["Swapins", "Swapouts"].iter().all(|name| {
                let a = vm_counter(before, name);
                a.is_some() && a == vm_counter(after, name)
            })
    }

    fn logits(eng: &MetalEngine) -> Vec<f32> {
        let bytes = eng.tensor_bytes(eng.model.wk.logits.unwrap());
        bytes[..bytes.len() / eng.model.batch]
            .chunks_exact(2)
            .map(|b| f32::from_bits((u16::from_le_bytes([b[0], b[1]]) as u32) << 16))
            .collect()
    }

    fn sample(eng: &mut MetalEngine, ids: &[u32], share: Option<u32>, profile: bool) -> Sample {
        if let Some(share) = share {
            eng.hetero.as_mut().unwrap().plan.ane_pct = share;
        }
        eng.set_profiling(profile);
        let start = Instant::now();
        let first = eng.prefill(ids).expect("prefill");
        let wall_ms = start.elapsed().as_secs_f64() * 1e3;
        Sample {
            wall_ms,
            first,
            profile: if profile {
                json!({"gpu": eng.profile, "lanes": eng.hetero.as_ref().map(|h| h.stats)})
            } else {
                Value::Null
            },
        }
    }

    fn load(path: &Path, args: &Args, split: bool) -> (MetalEngine, f64) {
        let start = Instant::now();
        let eng = MetalEngine::load(path, &args.checkpoint).expect("Metal load");
        let load_ms = start.elapsed().as_secs_f64() * 1e3;
        assert_eq!(
            eng.hetero.is_some(),
            split,
            "baseline must be an unsplit GPU asset"
        );
        if let Some(h) = &eng.hetero {
            assert_eq!(h.plan.cpu_pct, 0, "CPU participation is deferred");
        }
        for &n in &args.rows {
            assert!(
                eng.prefill_buckets().iter().any(|&(_, t)| t as usize >= n),
                "rows={n}: needs a larger bucket"
            );
        }
        (eng, load_ms)
    }

    fn run(args: &Args, gpu: bool, split: bool) -> Run {
        let memory_before = memory();
        assert_eq!(
            memory_before["pressure_level"], "1",
            "memory pressure is not normal; do not load another engine"
        );
        let mut engines = Vec::new();
        let mut load_ms = 0.0;
        for (enabled, path, split) in [
            (gpu, &args.gpu_blob, false),
            (split, &args.hetero_blob, true),
        ] {
            if enabled {
                let (eng, ms) = load(path, args, split);
                engines.push(eng);
                load_ms += ms;
            }
        }
        let id = identity(&engines[0], args);
        for eng in &engines[1..] {
            assert_eq!(id, identity(eng, args), "precision/model/bucket mismatch");
        }
        let tok = load_tokenizer(&args.checkpoint);
        let text = "The history of computing spans gears, relays, transistors, and integrated circuits. Each generation makes different tradeoffs between memory, arithmetic, and communication. ";
        let mut prompt = text.to_owned();
        while tok.encode_with_special_tokens(&prompt, true).len() < *args.rows.iter().max().unwrap()
        {
            prompt.push_str(text);
        }
        let ids = tok.encode_with_special_tokens(&prompt, true);
        let mut policies = Vec::new();
        if gpu {
            policies.push((0, None));
        }
        if split {
            let ix = usize::from(gpu);
            policies.push((ix, Some(0)));
            policies.extend(
                args.shares
                    .iter()
                    .filter(|&&s| s > 0)
                    .map(|&s| (ix, Some(s))),
            );
        }
        let mut cells = Vec::new();
        let mut warm_memory = Vec::new();
        for &rows in &args.rows {
            let mut warm_before = Value::Null;
            let base = cells.len();
            for &(e, ane_pct) in &policies {
                let eng = &mut engines[e];
                let (program, bucket) = *eng
                    .prefill_buckets()
                    .iter()
                    .filter(|&&(_, t)| t as usize >= rows)
                    .min_by_key(|&&(_, t)| t)
                    .unwrap();
                let ane_rows = if let Some(pct) = ane_pct {
                    let h = eng.hetero.as_mut().unwrap();
                    h.plan.ane_pct = pct;
                    let counts = h
                        .rows_for_chunk(program as usize, rows as u32)
                        .expect("split bucket missing plan");
                    assert_eq!(counts.2, 0);
                    counts.1 as usize
                } else {
                    0
                };
                cells.push(Cell {
                    rows,
                    ane_pct,
                    bucket,
                    gpu_rows: rows - ane_rows,
                    ane_rows,
                    ane_padded_rows: ane_rows.div_ceil(64) * 64,
                    first_request_ms: 0.0,
                    logits: Vec::new(),
                    samples: Vec::new(),
                });
            }
            for round in 0..args.warmup + args.reps {
                if round == args.warmup {
                    warm_before = memory();
                    assert_eq!(
                        warm_before["pressure_level"], "1",
                        "memory pressure after warmup; stop the sweep"
                    );
                }
                for j in 0..policies.len() {
                    let ix = (j + round) % policies.len();
                    let (e, share) = policies[ix];
                    let s = sample(&mut engines[e], &ids[..rows], share, args.profile);
                    let cell = &mut cells[base + ix];
                    if round == 0 {
                        cell.first_request_ms = s.wall_ms;
                    }
                    if round >= args.warmup {
                        assert!(
                            cell.samples.first().is_none_or(|old| old.first == s.first),
                            "unstable first token"
                        );
                        let actual = logits(&engines[e]);
                        assert!(actual.iter().all(|x| x.is_finite()), "nonfinite logits");
                        if !cell.logits.is_empty() {
                            assert_eq!(
                                cell.logits, actual,
                                "repeated identical input changed logits"
                            );
                        }
                        cell.logits = actual;
                        cell.samples.push(s);
                    }
                }
            }
            let after = memory();
            warm_memory.push(
                json!({"rows": rows, "quiet": memory_quiet(&warm_before, &after),
                "before": warm_before, "after": after}),
            );
        }
        Run {
            identity: id,
            load_ms,
            memory_before,
            memory_after: memory(),
            warm_memory,
            cells,
        }
    }

    fn write_new(path: &Path, value: &impl Serialize) {
        let out = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
            .expect("create new report (will not overwrite)");
        serde_json::to_writer_pretty(out, value).expect("write report");
    }

    fn ordered(samples: &[Sample]) -> Vec<&Sample> {
        assert!(!samples.is_empty());
        let mut order: Vec<_> = samples.iter().collect();
        order.sort_by(|a, b| a.wall_ms.total_cmp(&b.wall_ms));
        order
    }

    fn percentile(n: usize, pct: usize) -> usize {
        (n * pct).div_ceil(100).saturating_sub(1)
    }

    fn summary(samples: &[Sample]) -> Value {
        let order = ordered(samples);
        let median = order[order.len() / 2];
        let mut deviations: Vec<_> = order
            .iter()
            .map(|s| (s.wall_ms - median.wall_ms).abs())
            .collect();
        deviations.sort_by(f64::total_cmp);
        json!({"samples": order.len(), "min_ms": order[0].wall_ms,
            "median_sample": median, "p95_ms": order[percentile(order.len(), 95)].wall_ms,
            "mad_ms": deviations[deviations.len() / 2]})
    }

    fn quality(actual: &[f32], expected: &[f32]) -> Value {
        assert_eq!(actual.len(), expected.len());
        assert!(!actual.is_empty());
        assert!(actual.iter().chain(expected).all(|v| v.is_finite()));
        let mut diff = 0.0f64;
        let mut norm = 0.0f64;
        let mut max_abs = 0.0f32;
        for (&a, &b) in actual.iter().zip(expected) {
            diff += (f64::from(a) - f64::from(b)).powi(2);
            norm += f64::from(b).powi(2);
            max_abs = max_abs.max((a - b).abs());
        }
        let top = |v: &[f32]| {
            let mut indices: Vec<_> = (0..v.len()).collect();
            indices.sort_by(|&a, &b| v[b].total_cmp(&v[a]).then(a.cmp(&b)));
            indices.truncate(10);
            indices
        };
        let a = top(actual);
        let b = top(expected);
        let rel_l2 = (diff / norm.max(1e-30)).sqrt();
        json!({"rel_l2": rel_l2, "max_abs": max_abs,
            "top10_overlap": a.iter().filter(|i| b.contains(i)).count(), "logits_pass": rel_l2 < 0.03})
    }

    pub fn main() {
        let args = Args::parse();
        assert!(args.reps > 0 && args.warmup > 0 && !args.rows.is_empty());
        assert!(args.rows.iter().all(|&n| n > 0));
        assert!(args.shares.iter().all(|&s| s < 100));
        for name in ["PLOW_CPU_SHARE", "PLOW_ANE", "PLOW_METAL_SERIAL"] {
            assert!(
                std::env::var_os(name).is_none(),
                "remove diagnostic override {name}"
            );
        }
        assert!(
            std::env::var("PLOW_ANE_UNITS").is_err(),
            "use default CPUAndNeuralEngine; placement remains unverified"
        );
        if let Some(worker) = &args.worker {
            assert!(worker == "gpu" || worker == "split");
            write_new(
                &args.output,
                &run(&args, worker == "gpu", worker == "split"),
            );
            return;
        }
        assert!(!args.output.exists(), "report already exists");
        let runs = if args.interleaved {
            eprintln!("interleaved protocol: both engines resident; caller must ensure they co-fit without swap");
            vec![run(&args, true, true)]
        } else {
            let mut runs = Vec::new();
            for (session, mode) in ["gpu", "split", "split", "gpu"].iter().enumerate() {
                let path = args.output.with_extension(format!("session{session}.json"));
                let mut cmd = Command::new(std::env::current_exe().unwrap());
                cmd.arg(&args.gpu_blob)
                    .arg(&args.hetero_blob)
                    .arg(&args.checkpoint)
                    .arg("--rows")
                    .arg(
                        args.rows
                            .iter()
                            .map(usize::to_string)
                            .collect::<Vec<_>>()
                            .join(","),
                    )
                    .arg("--shares")
                    .arg(
                        args.shares
                            .iter()
                            .map(u32::to_string)
                            .collect::<Vec<_>>()
                            .join(","),
                    )
                    .arg("--reps")
                    .arg(args.reps.to_string())
                    .arg("--warmup")
                    .arg(args.warmup.to_string())
                    .arg("--worker")
                    .arg(mode)
                    .arg("--output")
                    .arg(&path);
                if args.profile {
                    cmd.arg("--profile");
                }
                eprintln!("session {session}: {mode}");
                assert!(
                    cmd.status().expect("fresh benchmark process").success(),
                    "session {session} failed; raw reports retained"
                );
                runs.push(serde_json::from_slice::<Run>(&std::fs::read(path).unwrap()).unwrap());
            }
            runs
        };
        for run in &runs[1..] {
            assert_eq!(
                runs[0].identity, run.identity,
                "precision/model/bucket mismatch"
            );
        }
        let mut cells = Vec::new();
        for (session, run) in runs.iter().enumerate() {
            for cell in &run.cells {
                let reference_run = if args.interleaved || session < 2 {
                    &runs[0]
                } else {
                    runs.last().unwrap()
                };
                let reference = reference_run
                    .cells
                    .iter()
                    .find(|c| c.rows == cell.rows && c.ane_pct.is_none())
                    .unwrap();
                let baseline = ordered(&reference.samples);
                let samples = ordered(&cell.samples);
                let speedup =
                    baseline[baseline.len() / 2].wall_ms / samples[samples.len() / 2].wall_ms;
                let q = quality(&cell.logits, &reference.logits);
                let same_first = cell.samples[0].first == reference.samples[0].first;
                let memory_gate = run
                    .warm_memory
                    .iter()
                    .find(|m| m["rows"] == cell.rows)
                    .unwrap()["quiet"]
                    == true
                    && reference_run
                        .warm_memory
                        .iter()
                        .find(|m| m["rows"] == cell.rows)
                        .unwrap()["quiet"]
                        == true;
                let timing_gate = args.reps >= 20
                    && !args.profile
                    && speedup > 1.0 / 0.95
                    && samples[percentile(samples.len(), 95)].wall_ms
                        <= baseline[percentile(baseline.len(), 95)].wall_ms * 1.03;
                println!("session={session} rows={} ane={:?} median={:.3} ms speedup={speedup:.3} rel_l2={} same_first={same_first}",
                    cell.rows, cell.ane_pct, samples[samples.len() / 2].wall_ms, q["rel_l2"]);
                cells.push(json!({"session": session, "rows": cell.rows, "ane_pct": cell.ane_pct,
                    "bucket": cell.bucket, "gpu_rows": cell.gpu_rows, "ane_rows": cell.ane_rows,
                    "ane_padded_rows": cell.ane_padded_rows, "first_request_ms": cell.first_request_ms,
                    "summary": summary(&cell.samples), "speedup": speedup, "quality": q,
                    "same_first_token": same_first, "timing_gate_only": timing_gate,
                    "memory_gate_only": memory_gate, "samples": cell.samples}));
            }
        }
        write_new(
            &args.output,
            &json!({
                "schema": "apple-prefill-measurements-v2", "eligible_for_policy": false,
                "protocol": if args.interleaved { "co-resident-rotating" } else { "fresh-process-ABBA" },
                "profile_enabled": args.profile, "warmup": args.warmup, "reps": args.reps,
                "identity": runs[0].identity, "cells": cells,
            "sessions": runs.iter().map(|r| json!({"load_ms": r.load_ms, "memory_before": r.memory_before,
                "memory_after": r.memory_after, "warm_memory": r.warm_memory})).collect::<Vec<_>>(),
                "missing_gates": ["placement evidence", "20-prompt/128-token sequence validation", "memory-pressure admission", "row-range policy selection"],
            }),
        );
        println!(
            "wrote {} (measurement only; no tuning record changed)",
            args.output.display()
        );
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn median_keeps_its_own_profile() {
            let samples: Vec<_> = [9.0, 1.0, 5.0]
                .into_iter()
                .map(|wall_ms| Sample {
                    wall_ms,
                    first: 1,
                    profile: json!({"marker": wall_ms}),
                })
                .collect();
            let got = summary(&samples);
            assert_eq!(got["median_sample"]["wall_ms"], 5.0);
            assert_eq!(got["median_sample"]["profile"]["marker"], 5.0);
            assert_eq!(got["p95_ms"], 9.0);
            assert_eq!(percentile(20, 95), 18);
        }

        #[test]
        fn quality_detects_error_and_top_changes() {
            assert_eq!(quality(&[1.0, 2.0], &[1.0, 2.0])["rel_l2"], 0.0);
            assert_eq!(quality(&[2.0, 1.0], &[1.0, 2.0])["logits_pass"], false);
            let a: Vec<f32> = (0..20).map(|i| i as f32).collect();
            let b: Vec<f32> = a.iter().rev().copied().collect();
            assert_eq!(quality(&a, &b)["top10_overlap"], 0);
        }

        #[test]
        fn memory_gate_rejects_swap_or_missing_evidence() {
            let before = json!({"pressure_level": "1", "vm_stat": "Swapins: 12.\nSwapouts: 34."});
            assert!(memory_quiet(&before, &before));
            let after = json!({"pressure_level": "1", "vm_stat": "Swapins: 13.\nSwapouts: 34."});
            assert!(!memory_quiet(&before, &after));
            assert!(!memory_quiet(&Value::Null, &Value::Null));
        }

        #[test]
        #[should_panic]
        fn quality_rejects_nonfinite_logits() {
            quality(&[f32::NAN], &[1.0]);
        }
    }
}

#[cfg(all(feature = "ane", target_os = "macos"))]
fn main() {
    bench::main();
}

#[cfg(not(all(feature = "ane", target_os = "macos")))]
fn main() {
    eprintln!("build with --features ane on macOS");
}
