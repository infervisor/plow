//! Serving host-path load harness, CPU only (ignored; run in release with `--nocapture`).
//!
//! The production router, hyper connection builder and muxer serve a reference bundle whose
//! device work is a two-instruction CPU walk, so what is measured is the host path: HTTP, the
//! handler, tokenize, admission, the tick's token emit (detokenize + channel send), SSE framing and
//! the write path. A raw HTTP/1.1 client on its own runtime drives it. Reported per cell: server
//! CPU (threads not owned by the client) per request and per token, heap allocations per request
//! and per token (counting allocator, client threads excluded), and client-side latency.
//!
//! `HOSTBENCH_TOKENIZER` (a `tokenizer.json`) swaps the byte tokenizer for a real one;
//! `HOSTBENCH_SCALE` multiplies the request counts; `HOSTBENCH_CELLS=a,b` runs only those cells.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::io::{AsyncReadExt, AsyncWriteExt};

mod common;

struct Counting;
/// Per-thread shards: one shared counter bounced its cache line across every server thread and
/// showed up as 7% of the request handler's profile.
const SHARDS: usize = 64;
static ALLOCS: [crossbeam_utils::CachePadded<AtomicU64>; SHARDS] =
    [const { crossbeam_utils::CachePadded::new(AtomicU64::new(0)) }; SHARDS];
static NEXT_SHARD: AtomicU64 = AtomicU64::new(0);
thread_local! {
    static CLIENT: Cell<bool> = const { Cell::new(false) };
    static SHARD: Cell<usize> = const { Cell::new(usize::MAX) };
}

fn count(_n: usize) {
    if CLIENT.try_with(Cell::get).unwrap_or(true) {
        return;
    }
    let shard = SHARD
        .try_with(|s| {
            if s.get() == usize::MAX {
                s.set(NEXT_SHARD.fetch_add(1, Relaxed) as usize % SHARDS);
            }
            s.get()
        })
        .unwrap_or(0);
    ALLOCS[shard].fetch_add(1, Relaxed);
}

fn allocs() -> u64 {
    ALLOCS.iter().map(|a| a.load(Relaxed)).sum()
}

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        count(l.size());
        System.alloc(l)
    }
    unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
        count(l.size());
        System.alloc_zeroed(l)
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, n: usize) -> *mut u8 {
        count(n);
        System.realloc(p, l, n)
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        System.dealloc(p, l)
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

/// CPU ms of every thread of this process whose name does not start with `cli`.
fn server_cpu_ms() -> f64 {
    let tick = 100.0; // USER_HZ
    let mut ms = 0.0;
    for e in std::fs::read_dir("/proc/self/task").unwrap().flatten() {
        let Ok(raw) = std::fs::read_to_string(e.path().join("stat")) else { continue };
        let (Some(a), Some(b)) = (raw.find('('), raw.rfind(')')) else { continue };
        if raw[a + 1..b].starts_with("cli") {
            continue;
        }
        let f: Vec<&str> = raw[b + 2..].split_whitespace().collect();
        ms += (f[11].parse::<f64>().unwrap() + f[12].parse::<f64>().unwrap()) * 1000.0 / tick;
    }
    ms
}

/// Server CPU is summed from /proc at 10 ms resolution; cells run long enough for that to vanish.
struct Cell_ {
    name: &'static str,
    conc: usize,
    requests: usize,
    max_tokens: usize,
    stream: bool,
    long: bool,
}

#[derive(Default)]
struct Stats {
    ttft_us: Vec<f64>,
    itl_us: Vec<f64>,
    e2e_us: Vec<f64>,
    tokens: usize,
}

fn pct(v: &mut [f64], p: f64) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[((v.len() - 1) as f64 * p).round() as usize]
}

async fn read_head(s: &mut tokio::net::TcpStream, buf: &mut Vec<u8>) -> usize {
    loop {
        if let Some(i) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
            return i + 4;
        }
        let mut tmp = [0u8; 8192];
        let n = s.read(&mut tmp).await.unwrap();
        assert!(n > 0, "server closed the connection");
        buf.extend_from_slice(&tmp[..n]);
    }
}

/// One request over a kept-alive connection: returns (ttft, inter-token gaps, e2e, tokens).
async fn one(s: &mut tokio::net::TcpStream, req: &[u8], stream: bool, st: &mut Stats) {
    let t0 = Instant::now();
    s.write_all(req).await.unwrap();
    let mut buf = Vec::with_capacity(16 * 1024);
    let head = read_head(s, &mut buf).await;
    let head_s = std::str::from_utf8(&buf[..head]).unwrap().to_ascii_lowercase();
    assert!(head_s.starts_with("http/1.1 200"), "{head_s}");
    let mut body = buf.split_off(head);
    let mut tmp = vec![0u8; 64 * 1024];
    if let Some(i) = head_s.find("content-length:") {
        let len: usize = head_s[i + 15..].lines().next().unwrap().trim().parse().unwrap();
        while body.len() < len {
            let n = s.read(&mut tmp).await.unwrap();
            body.extend_from_slice(&tmp[..n]);
        }
        st.e2e_us.push(t0.elapsed().as_secs_f64() * 1e6);
        st.ttft_us.push(t0.elapsed().as_secs_f64() * 1e6);
        let v: serde_json::Value = serde_json::from_slice(&body[..len]).unwrap();
        st.tokens += v["usage"]["completion_tokens"].as_u64().unwrap() as usize;
        return;
    }
    // Chunked: count `data:` frames carrying a token, timestamp each read that delivers some.
    let (mut pos, mut last, mut first, mut n_tok) = (0usize, t0, None, 0usize);
    loop {
        let mut progressed = true;
        while progressed {
            progressed = false;
            let Some(eol) = body[pos..].windows(2).position(|w| w == b"\r\n") else { break };
            let size = usize::from_str_radix(std::str::from_utf8(&body[pos..pos + eol]).unwrap().trim(), 16).unwrap();
            if body.len() < pos + eol + 2 + size + 2 {
                break;
            }
            if size == 0 {
                st.e2e_us.push(t0.elapsed().as_secs_f64() * 1e6);
                st.tokens += n_tok;
                return;
            }
            let chunk = &body[pos + eol + 2..pos + eol + 2 + size];
            let toks = chunk.windows(20).filter(|w| w == b"\"finish_reason\":null").count();
            if toks > 0 && stream {
                let now = Instant::now();
                match first {
                    None => {
                        first = Some(now);
                        st.ttft_us.push((now - t0).as_secs_f64() * 1e6);
                    }
                    Some(_) => st.itl_us.push((now - last).as_secs_f64() * 1e6 / toks as f64),
                }
                last = now;
                n_tok += toks;
            }
            pos += eol + 2 + size + 2;
            progressed = true;
        }
        let n = s.read(&mut tmp).await.unwrap();
        assert!(n > 0, "server closed mid-response");
        body.extend_from_slice(&tmp[..n]);
    }
}

fn request(prompt: &str, max_tokens: usize, stream: bool) -> Vec<u8> {
    let body = serde_json::json!({
        "model": "hb", "prompt": prompt, "max_tokens": max_tokens, "stream": stream,
        "ignore_eos": true, "temperature": 0.0, "stream_options": {"include_usage": true},
    })
    .to_string();
    format!(
        "POST /v1/completions HTTP/1.1\r\nhost: 127.0.0.1\r\ncontent-type: application/json\r\ncontent-length: {}\r\n\r\n{body}",
        body.len()
    )
    .into_bytes()
}

#[test]
#[ignore]
fn host_serve_bench() {
    let dir = std::env::temp_dir().join(format!("plowrt_hostserve_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    common::write_bundle_with_sample_batch(&dir, "hb", 64);
    if let Ok(tok) = std::env::var("HOSTBENCH_TOKENIZER") {
        std::fs::copy(tok, dir.join("tokenizer.json")).unwrap();
    }
    let scale: f64 = std::env::var("HOSTBENCH_SCALE").ok().and_then(|s| s.parse().ok()).unwrap_or(1.0);

    let srv = tokio::runtime::Builder::new_multi_thread().enable_all().thread_name("srv").build().unwrap();
    let addr = srv.block_on(async {
        let backend: Arc<dyn plowrt::device::Backend> = Arc::new(plowrt::device::cpu::CpuBackend::new(4));
        let execset = Arc::new(plowrt::exec::ExecutorSet::bringup(backend).unwrap());
        let registry = plowrt::orch::Registry::new();
        registry.load(&dir, None).unwrap();
        let state = Arc::new(plowrt::serve::AppState::new(registry, execset));
        for slug in state.registry.slugs() {
            let bundle = state.registry.get(&slug).unwrap();
            let m = plowrt::serve::mux::spawn(slug.clone(), bundle, Arc::clone(&state), plowrt::serve::mux::MuxConfig::default());
            state.install_mux(slug, m);
        }
        let router = plowrt::serve::app(state);
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let mut builder = hyper_util::server::conn::auto::Builder::new(hyper_util::rt::TokioExecutor::new());
        builder
            .http1()
            .timer(hyper_util::rt::TokioTimer::new())
            .header_read_timeout(Some(Duration::from_secs(30)));
        let builder = Arc::new(builder);
        let svc = hyper_util::service::TowerToHyperService::new(router);
        tokio::spawn(async move {
            loop {
                let (stream, _) = listener.accept().await.unwrap();
                let _ = stream.set_nodelay(true);
                let (builder, svc) = (Arc::clone(&builder), svc.clone());
                tokio::spawn(async move {
                    let _ = builder.serve_connection_with_upgrades(hyper_util::rt::TokioIo::new(stream), svc).await;
                });
            }
        });
        addr
    });

    let cli = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .enable_all()
        .thread_name("cli")
        .on_thread_start(|| CLIENT.with(|c| c.set(true)))
        .build()
        .unwrap();
    // ~1000 tokens of English-like text for a real tokenizer; the byte tokenizer sees 4-5x that.
    let words = ["the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog", "serving", "token"];
    let prompt: String = (0..1000).map(|i| words[(i * 7 + i / 3) % words.len()]).collect::<Vec<_>>().join(" ");
    let short = "Say hello.";
    // Streams use a short prompt: the reference walk hashes the whole prompt per token, which at
    // ~1000 tokens would dominate the tick. The long prompt is measured on its own (request path).
    let cells = [
        Cell_ { name: "warm", conc: 8, requests: 64, max_tokens: 16, stream: true, long: false },
        Cell_ { name: "req_c16_t1", conc: 16, requests: 4000, max_tokens: 1, stream: false, long: false },
        Cell_ { name: "req_c16_isl1k_t1", conc: 16, requests: 2000, max_tokens: 1, stream: false, long: true },
        Cell_ { name: "stream_c1_t128", conc: 1, requests: 80, max_tokens: 128, stream: true, long: false },
        Cell_ { name: "stream_c64_t128", conc: 64, requests: 1280, max_tokens: 128, stream: true, long: false },
    ];
    println!("HOSTSERVE cell req tok wall_s req/s tok/s srv_cpu_us/req srv_cpu_us/tok allocs/req allocs/tok ttft_p50_us itl_p50_us e2e_p50_us");
    let only = std::env::var("HOSTBENCH_CELLS").ok();
    for cell in cells.iter().filter(|c| c.name == "warm" || only.as_deref().map_or(true, |o| o.split(',').any(|n| n == c.name))) {
        let requests = ((cell.requests as f64 * scale) as usize).max(cell.conc);
        let req = Arc::new(request(if cell.long { &prompt } else { short }, cell.max_tokens, cell.stream));
        let (cpu0, a0) = (server_cpu_ms(), allocs());
        let t0 = Instant::now();
        let left = Arc::new(AtomicU64::new(requests as u64));
        let mut stats = cli.block_on(async {
            let mut tasks = Vec::new();
            for _ in 0..cell.conc {
                let (req, left, stream) = (Arc::clone(&req), Arc::clone(&left), cell.stream);
                tasks.push(tokio::spawn(async move {
                    let mut s = tokio::net::TcpStream::connect(addr).await.unwrap();
                    s.set_nodelay(true).unwrap();
                    let mut st = Stats::default();
                    while left.fetch_update(Relaxed, Relaxed, |n| n.checked_sub(1)).is_ok() {
                        one(&mut s, &req, stream, &mut st).await;
                    }
                    st
                }));
            }
            let mut all = Stats::default();
            for t in tasks {
                let st = t.await.unwrap();
                all.ttft_us.extend(st.ttft_us);
                all.itl_us.extend(st.itl_us);
                all.e2e_us.extend(st.e2e_us);
                all.tokens += st.tokens;
            }
            all
        });
        let wall = t0.elapsed().as_secs_f64();
        std::thread::sleep(Duration::from_millis(50));
        let (cpu, n_allocs) = (server_cpu_ms() - cpu0, allocs() - a0);
        let tok = stats.tokens.max(1) as f64;
        println!(
            "HOSTSERVE {} {} {} {:.2} {:.0} {:.0} {:.1} {:.2} {:.0} {:.1} {:.0} {:.1} {:.0}",
            cell.name,
            requests,
            stats.tokens,
            wall,
            requests as f64 / wall,
            stats.tokens as f64 / wall,
            cpu * 1e3 / requests as f64,
            cpu * 1e3 / tok,
            n_allocs as f64 / requests as f64,
            n_allocs as f64 / tok,
            pct(&mut stats.ttft_us, 0.5),
            pct(&mut stats.itl_us, 0.5),
            pct(&mut stats.e2e_us, 0.5),
        );
    }
    drop(cli);
    srv.shutdown_timeout(Duration::from_secs(1));
    let _ = std::fs::remove_dir_all(&dir);
}
