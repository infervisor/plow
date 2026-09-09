//! Fresh-process GPU/hetero sequence qualification. Reports divergences; never writes policy.

#[cfg(all(feature = "ane", target_os = "macos"))]
mod check {
    use clap::Parser;
    use plow_asset::decode_objects::image_sha256;
    use plowrt::{exec::apple::MetalEngine, text::tokenizer::load_tokenizer};
    use serde_json::{json, Value};
    use std::{collections::BTreeMap, path::PathBuf, process::Command, time::Instant};

    #[derive(Parser)]
    struct Args {
        gpu_blob: PathBuf,
        hetero_blob: PathBuf,
        checkpoint: PathBuf,
        /// JSON array of prompt strings. Default = 20 varied tasks and context lengths.
        #[arg(long)]
        prompts: Option<PathBuf>,
        #[arg(long, default_value_t = 128)]
        tokens: usize,
        #[arg(long, default_value_t = 50)]
        share: u32,
        #[arg(long)]
        output: PathBuf,
        #[arg(long, hide = true)]
        worker: Option<String>,
    }

    fn prompts(args: &Args) -> Vec<String> {
        if let Some(path) = &args.prompts {
            return serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        }
        [
            "Explain how a compiler differs from an interpreter.",
            "Prove that the sum of two even integers is even.",
            "Write a Rust function that reverses a slice in place.",
            "Describe the water cycle to a primary-school student.",
            "Translate 'The train leaves at noon' into French and Spanish.",
            "A box has three red balls and two blue balls. What is the probability of drawing blue?",
            "List three differences between TCP and UDP.",
            "Write a short story about a lighthouse keeper finding a letter.",
            "Explain why the seasons change throughout the year.",
            "Convert the decimal number 173 to binary, showing your work.",
            "Describe the role of mitochondria in a cell.",
            "Write a SQL query to find customers with more than three orders.",
            "Compare a linked list with a dynamically sized array.",
            "Explain the difference between correlation and causation using an example.",
            "Write a polite email asking to reschedule a meeting.",
            "Derive the derivative of x cubed plus two x.",
            "Explain how a rainbow forms.",
            "Give an example of a deadlock and describe how to prevent it.",
            "Summarize the process of photosynthesis in three sentences.",
            "Continue the sequence 2, 3, 5, 8, 13 and explain its rule.",
        ].iter().enumerate().map(|(i,task)| {
            let context = "Reference context: a library stores books by subject. Readers borrow books, return them, and discuss what they learn. ";
            format!("{}\nTask: {task}\nAnswer:",context.repeat([0,2,4,12][i%4]))
        }).collect()
    }

    fn write(path: &PathBuf, value: &Value) {
        let file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
            .unwrap();
        serde_json::to_writer(file, value).unwrap();
    }

    fn run(args: &Args, split: bool) -> Value {
        let tok = load_tokenizer(&args.checkpoint);
        let prompts = prompts(args);
        let start = Instant::now();
        let mut eng = MetalEngine::load(
            if split {
                &args.hetero_blob
            } else {
                &args.gpu_blob
            },
            &args.checkpoint,
        )
        .unwrap();
        let load_ms = start.elapsed().as_secs_f64() * 1e3;
        assert_eq!(eng.model.batch, 1, "sequence check requires one slot");
        if split {
            let h = eng.hetero.as_mut().expect("heterogeneous asset");
            assert_eq!(h.plan.cpu_pct, 0, "CPU lane is out of scope");
            h.plan.ane_pct = args.share;
        } else {
            assert!(eng.hetero.is_none(), "baseline must be an unsplit asset");
        }
        let mut weights = BTreeMap::new();
        for (h, name) in eng.model.names.iter().enumerate() {
            if packet::names::is_checkpoint_weight(name) {
                weights.insert(name.clone(), image_sha256(eng.tensor_bytes(h)));
            }
        }
        assert!(!weights.is_empty());
        let config: Value =
            serde_json::from_slice(&std::fs::read(args.checkpoint.join("config.json")).unwrap())
                .unwrap();
        let vocab = config["vocab_size"].as_u64().unwrap() as usize;
        let decode: Vec<_> = eng.model.blob.progs[eng.model.dec_ix..]
            .iter()
            .map(|p| {
                p.insts
                    .iter()
                    .map(|d| {
                        json!({"op":d.op,"fj":d.fj,"i":d.i,
                "t":d.t.iter().map(|&h|eng.model.names.get(h as usize)).collect::<Vec<_>>()})
                    })
                    .collect::<Vec<_>>()
            })
            .collect();
        let kv: BTreeMap<_, _> = eng
            .model
            .names
            .iter()
            .enumerate()
            .filter(|(_, n)| n.starts_with("kv."))
            .map(|(h, n)| (n.clone(), eng.tensor_bytes(h).len()))
            .collect();
        let mut cases = Vec::new();
        for (index, prompt) in prompts.iter().enumerate() {
            let ids = tok.encode_with_special_tokens(prompt, true);
            assert!(
                !ids.is_empty()
                    && ids
                        .len()
                        .checked_add(args.tokens)
                        .is_some_and(|n| n <= eng.max_ctx())
            );
            eng.set_profiling(false);
            let start = Instant::now();
            let first = eng.prefill(&ids).unwrap();
            let prefill_ms = start.elapsed().as_secs_f64() * 1e3;
            let ane_runs = eng.hetero.as_ref().map_or(0, |h| h.stats.ane_runs);
            let bytes = eng.tensor_bytes(eng.model.wk.logits.unwrap());
            assert_eq!(bytes.len(), vocab * 2, "expected BF16 logits");
            let logits: Vec<_> = bytes
                .chunks_exact(2)
                .map(|c| f32::from_bits(u32::from(u16::from_le_bytes([c[0], c[1]])) << 16))
                .collect();
            assert!(
                logits.iter().all(|x| x.is_finite()),
                "nonfinite logits at prompt {index}"
            );
            let mut tokens = vec![first];
            for step in 1..args.tokens {
                let pos = (ids.len() + step - 1) as u32;
                tokens.push(eng.decode_step(pos, pos + 1).unwrap());
            }
            eprintln!(
                "{} prompt {index}: rows={} tokens={} ANE_calls={ane_runs}",
                if split { "split" } else { "gpu" },
                ids.len(),
                tokens.len()
            );
            cases.push(json!({"prompt":prompt,"ids":ids,"tokens":tokens,"logits":logits,
                "prefill_ms":prefill_ms,"total_ms":start.elapsed().as_secs_f64()*1e3,"ane_calls":ane_runs}));
        }
        json!({"weights":weights,"decode":decode,"kv":kv,"vocab":vocab,"load_ms":load_ms,"cases":cases})
    }

    fn compare(a: &[f32], b: &[f32]) -> Value {
        assert!(!a.is_empty() && a.len() == b.len());
        if !a.iter().chain(b).all(|x| x.is_finite()) {
            return json!({"pass":false,"reason":"nonfinite"});
        }
        let (mut diff, mut norm, mut max) = (0.0f64, 0.0f64, 0.0f32);
        for (&a, &b) in a.iter().zip(b) {
            diff += (f64::from(a) - f64::from(b)).powi(2);
            norm += f64::from(b).powi(2);
            max = max.max((a - b).abs());
        }
        let top = |v: &[f32]| {
            let mut ix: Vec<_> = (0..v.len()).collect();
            ix.sort_by(|&a, &b| v[b].total_cmp(&v[a]));
            ix.truncate(10);
            ix
        };
        let (ta, tb) = (top(a), top(b));
        let rel = (diff / norm.max(1e-30)).sqrt();
        json!({"pass":rel<0.03,"rel_l2":rel,"max_abs":max,"top10_overlap":ta.iter().filter(|i|tb.contains(i)).count()})
    }

    pub fn main() {
        let args = Args::parse();
        assert!(args.tokens > 0 && args.share > 0 && args.share < 100);
        for name in [
            "PLOW_CPU_SHARE",
            "PLOW_ANE",
            "PLOW_ANE_UNITS",
            "PLOW_METAL_SERIAL",
        ] {
            assert!(std::env::var_os(name).is_none(), "remove override {name}");
        }
        assert!(!args.output.exists());
        if let Some(mode) = &args.worker {
            assert!(mode == "gpu" || mode == "split");
            write(&args.output, &run(&args, mode == "split"));
            return;
        }
        let mut runs = Vec::new();
        for mode in ["gpu", "split"] {
            let path = args.output.with_extension(format!("{mode}.json"));
            let mut cmd = Command::new(std::env::current_exe().unwrap());
            cmd.arg(&args.gpu_blob)
                .arg(&args.hetero_blob)
                .arg(&args.checkpoint)
                .arg("--tokens")
                .arg(args.tokens.to_string())
                .arg("--share")
                .arg(args.share.to_string())
                .arg("--worker")
                .arg(mode)
                .arg("--output")
                .arg(&path);
            if let Some(p) = &args.prompts {
                cmd.arg("--prompts").arg(p);
            }
            assert!(
                cmd.status().unwrap().success(),
                "{mode} worker failed; raw reports retained"
            );
            runs.push(serde_json::from_slice::<Value>(&std::fs::read(&path).unwrap()).unwrap());
        }
        assert_eq!(
            runs[0]["weights"], runs[1]["weights"],
            "precision-matched loaded weights required"
        );
        assert_eq!(
            runs[0]["decode"], runs[1]["decode"],
            "decode semantics mismatch"
        );
        assert_eq!(runs[0]["kv"], runs[1]["kv"], "KV layout mismatch");
        let (gpu, split) = (
            runs[0]["cases"].as_array().unwrap(),
            runs[1]["cases"].as_array().unwrap(),
        );
        assert_eq!(gpu.len(), split.len());
        let active_prompts = split
            .iter()
            .filter(|s| s["ane_calls"].as_u64().unwrap() > 0)
            .count();
        let mut pass = gpu.len() >= 20 && args.tokens >= 128 && active_prompts > 0;
        let mut cases = Vec::new();
        for (index, (g, s)) in gpu.iter().zip(split).enumerate() {
            assert_eq!(g["ids"], s["ids"]);
            let (gt, st) = (
                g["tokens"].as_array().unwrap(),
                s["tokens"].as_array().unwrap(),
            );
            assert_eq!(gt.len(), args.tokens);
            assert_eq!(st.len(), args.tokens);
            let first_divergence = gt.iter().zip(st).position(|(a, b)| a != b);
            let gl: Vec<f32> = serde_json::from_value(g["logits"].clone()).unwrap();
            let sl: Vec<f32> = serde_json::from_value(s["logits"].clone()).unwrap();
            let quality = compare(&sl, &gl);
            pass &= first_divergence.is_none() && quality["pass"] == true;
            eprintln!("prompt {index}: first_divergence={first_divergence:?} quality={quality}");
            cases.push(json!({"index":index,"rows":g["ids"].as_array().unwrap().len(),"quality":quality,
                "same_first":gt[0]==st[0],"first_divergence":first_divergence,"gpu_tokens":gt,"split_tokens":st}));
        }
        write(
            &args.output,
            &json!({"schema":"apple-sequence-check-v1","share":args.share,"generated_tokens":args.tokens,
            "prompts":gpu.len(),"active_prompts":active_prompts,"sequence_gate_pass":pass,"eligible_for_policy":false,"cases":cases}),
        );
        println!("sequence_gate_pass={pass}; wrote {}", args.output.display());
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        #[test]
        fn logits_gate() {
            assert_eq!(compare(&[1.0, 2.0], &[1.0, 2.0])["pass"], true);
            assert_eq!(compare(&[1.0, 2.0], &[2.0, 1.0])["pass"], false);
            assert_eq!(compare(&[f32::NAN], &[1.0])["pass"], false);
        }
    }
}

#[cfg(all(feature = "ane", target_os = "macos"))]
fn main() {
    check::main();
}
#[cfg(not(all(feature = "ane", target_os = "macos")))]
fn main() {
    eprintln!("requires macOS --features ane");
}
