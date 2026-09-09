#[cfg(all(feature = "ane", target_os = "macos"))]
mod check {
    use clap::Parser;
    use plowrt::{exec::apple::MetalEngine, text::tokenizer::load_tokenizer};
    use serde_json::{json, Value};
    use std::{path::PathBuf, process::Command};

    #[derive(Parser)]
    struct Args {
        blob: PathBuf,
        checkpoint: PathBuf,
        #[arg(long)]
        output: PathBuf,
        #[arg(long, default_value_t = 8)]
        tokens: usize,
        #[arg(long, hide = true)]
        worker: Option<String>,
    }

    fn write(path: &PathBuf, value: &Value) {
        use std::io::Write;
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
            .unwrap();
        file.write_all(&serde_json::to_vec(value).unwrap()).unwrap();
    }

    fn worker(a: &Args, mode: &str) -> Value {
        let mut engine = MetalEngine::load(&a.blob, &a.checkpoint).unwrap();
        if mode != "gpu" {
            assert!(engine.channel.is_some(), "channel load/placement failed");
        }
        let tok = load_tokenizer(&a.checkpoint);
        let ids = tok.encode_with_special_tokens(
            &"Explain how computer memory and processors cooperate to execute a program. "
                .repeat(64),
            true,
        );
        let mut cases = Vec::new();
        let (prog, bucket) = engine
            .prefill_buckets()
            .into_iter()
            .find(|&(_, t)| t == 128)
            .unwrap();
        assert!(engine
            .prepare_prefill_chunk(
                &ids,
                plowrt::exec::cpu::engine::Chunk {
                    prog,
                    c0: 0,
                    clen: bucket + 1,
                }
            )
            .is_err());
        let oversized = vec![0; engine.max_ctx() + 1];
        assert!(engine
            .prepare_prefill_chunk(
                &oversized,
                plowrt::exec::cpu::engine::Chunk {
                    prog,
                    c0: engine.max_ctx() as u32,
                    clen: 1,
                }
            )
            .is_err());
        for rows in [91usize, 128, 31, 255] {
            assert!(rows + a.tokens <= engine.max_ctx());
            let before = engine.channel.as_ref().map_or(0, |c| c.stats.mlps);
            let first = engine.prefill(&ids[..rows]).unwrap();
            let logits: Vec<u16> = engine
                .tensor_bytes(engine.model.wk.logits.unwrap())
                .chunks_exact(2)
                .map(|b| u16::from_le_bytes([b[0], b[1]]))
                .collect();
            let mut tokens = vec![first];
            for i in 1..a.tokens {
                let p = (rows + i - 1) as u32;
                tokens.push(engine.decode_step(p, p + 1).unwrap());
            }
            let after = engine.channel.as_ref().map_or(0, |c| c.stats.mlps);
            if mode == "active" {
                assert_eq!(after > before, rows >= 64);
            }
            if rows == 31 {
                assert_eq!(after, before);
            }
            cases.push(json!({"rows":rows,"logits":logits,"tokens":tokens,"mlps":after-before}));
        }
        let stats = engine.channel.as_ref().map(|c| &c.stats);
        if mode != "active" && mode != "gpu" {
            let stats = stats.unwrap();
            assert!(stats.disabled);
            assert_eq!(stats.fallbacks, 1);
            assert_eq!(stats.mlps, 0);
        }
        json!({"mode":mode,"cases":cases,"stats":stats,"eligible_for_policy":false})
    }

    pub fn main() {
        tracing_subscriber::fmt()
            .with_env_filter("info")
            .with_writer(std::io::stderr)
            .init();
        let a = Args::parse();
        assert!(a.tokens > 0);
        if let Some(mode) = &a.worker {
            write(&a.output, &worker(&a, mode));
            return;
        }
        assert!(!a.output.exists());
        let mut reports = Vec::new();
        for mode in [
            "gpu",
            "active",
            "before_submit",
            "after_submit",
            "after_join",
        ] {
            let out = a.output.with_extension(format!("{mode}.json"));
            assert!(!out.exists());
            let mut cmd = Command::new(std::env::current_exe().unwrap());
            cmd.arg(&a.blob)
                .arg(&a.checkpoint)
                .args([
                    "--worker",
                    mode,
                    "--tokens",
                    &a.tokens.to_string(),
                    "--output",
                ])
                .arg(&out)
                .env("PLOW_ANE_MLP", if mode == "gpu" { "0" } else { "1" })
                .env_remove("PLOW_ANE_MLP_FAIL");
            if mode != "gpu" && mode != "active" {
                cmd.env("PLOW_ANE_MLP_FAIL", mode);
            }
            assert!(cmd.status().unwrap().success(), "worker {mode}");
            reports.push(serde_json::from_slice::<Value>(&std::fs::read(out).unwrap()).unwrap());
        }
        let mut quality = Vec::new();
        let bf = |v: &Value| f32::from_bits((v.as_u64().unwrap() as u32) << 16) as f64;
        for report in &reports[1..] {
            let mode = report["mode"].as_str().unwrap();
            for (base, case) in reports[0]["cases"]
                .as_array()
                .unwrap()
                .iter()
                .zip(report["cases"].as_array().unwrap())
            {
                let mut diff = 0.0;
                let mut norm = 0.0;
                let mut max_abs = 0.0f64;
                for (x, y) in base["logits"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .zip(case["logits"].as_array().unwrap())
                {
                    let (x, y) = (bf(x), bf(y));
                    assert!(x.is_finite() && y.is_finite());
                    diff += (x - y).powi(2);
                    norm += x * x;
                    max_abs = max_abs.max((x - y).abs());
                }
                let rel_l2 = (diff / norm.max(1e-30)).sqrt();
                assert!(rel_l2 < 0.03);
                assert_eq!(base["tokens"][0], case["tokens"][0]);
                if mode != "active" || case["rows"] == 31 {
                    assert_eq!(base["logits"], case["logits"]);
                    assert_eq!(base["tokens"], case["tokens"]);
                }
                quality.push(json!({"mode":mode,"rows":case["rows"],"rel_l2":rel_l2,"max_abs":max_abs,"tokens_equal":base["tokens"]==case["tokens"]}));
            }
        }
        write(
            &a.output,
            &json!({"eligible_for_policy":false,"quality":quality,"reports":reports}),
        );
        println!("channel active/tail/multi-chunk and three failure stages passed");
    }
}

#[cfg(all(feature = "ane", target_os = "macos"))]
fn main() {
    check::main();
}
#[cfg(not(all(feature = "ane", target_os = "macos")))]
fn main() {
    panic!("requires --features ane on macOS");
}
