//! Host (CPU) port of speechbrain's ECAPA-TDNN speaker embedding (spkrec-ecapa-voxceleb) exactly as Maya 2 Native's
//! `Ecapa.__call__` (maya3/api.py) runs it: scipy `resample_poly(x, 2, 3)` 24 -> 16 kHz, speechbrain
//! `Fbank(n_mels=80)`, sentence mean normalisation, `ECAPA_TDNN`, L2 normalisation. Weights come from
//! `scripts/tts/ecapa_export.py`. Exactness target: fp32 rounding only (cosine >= 0.9999 and max abs <= 2e-3 vs the
//! fp32 torch reference; `tests::parity`).

use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::error::{Result, RuntimeError};

pub const EMBED_DIM: usize = 192;
const MIN_SAMPLES: usize = 2400;
const N_MELS: usize = 80;
const N_FFT: usize = 400;
const HOP: usize = 160;
const N_BINS: usize = N_FFT / 2 + 1;
const C: usize = 1024;
const M: usize = 3072;
const SCALE: usize = 8;
const ATT: usize = 128;

const MR: usize = 4;
const NR: usize = 24;
const KC: usize = 256;
const GROUP: usize = 8;

/// Conv1d (stride 1, reflect "same" padding) with weights packed in MR-row panels: `w[p][j][r]`, `j = k * cin + ci`.
struct Conv {
    cout: usize,
    cin: usize,
    k: usize,
    dil: usize,
    w: Vec<f32>,
    b: Vec<f32>,
}

struct Tdnn {
    conv: Conv,
    scale: Vec<f32>,
    shift: Vec<f32>,
}

struct Dense {
    cout: usize,
    cin: usize,
    w: Vec<f32>,
    b: Vec<f32>,
}

struct SeRes2 {
    tdnn1: Tdnn,
    res2: Vec<Tdnn>,
    tdnn2: Tdnn,
    se1: Dense,
    se2: Dense,
}

pub struct Ecapa {
    block0: Tdnn,
    blocks: Vec<SeRes2>,
    mfa: Tdnn,
    asp_x: Tdnn,
    asp_mean: Dense,
    asp_std: Dense,
    asp_conv: Conv,
    fc: Dense,
    fir: Vec<f32>,
    dft: Vec<f32>,
    mel: Vec<f32>,
    threads: usize,
    fma: bool,
}

/// Rows of `s` floats each (`s` is a multiple of NR >= t); columns `t..s` are finite scratch.
struct Act {
    t: usize,
    s: usize,
    d: Vec<f32>,
}

impl Act {
    fn zeros(c: usize, t: usize) -> Self {
        let s = t.div_ceil(NR) * NR;
        Self {
            t,
            s,
            d: vec![0.0; c * s],
        }
    }
    fn row(&self, i: usize) -> &[f32] {
        &self.d[i * self.s..][..self.t]
    }
}

#[derive(Clone, Copy)]
struct SyncPtr(*mut f32);
unsafe impl Send for SyncPtr {}
unsafe impl Sync for SyncPtr {}

fn par_for(threads: usize, n: usize, f: impl Fn(usize) + Sync) {
    let nt = threads.min(n);
    if nt <= 1 {
        (0..n).for_each(f);
        return;
    }
    let next = AtomicUsize::new(0);
    let work = || loop {
        let i = next.fetch_add(1, Ordering::Relaxed);
        if i >= n {
            break;
        }
        f(i);
    };
    std::thread::scope(|s| {
        for _ in 1..nt {
            s.spawn(work);
        }
        work();
    });
}

fn pack(w: &[f32], cout: usize, kdim: usize) -> Vec<f32> {
    let mut p = vec![0.0; cout * kdim];
    for co in 0..cout {
        let (pi, r) = (co / MR, co % MR);
        for j in 0..kdim {
            p[(pi * kdim + j) * MR + r] = w[co * kdim + j];
        }
    }
    p
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn micro_avx2(
    wp: *const f32,
    kn: usize,
    x: *const f32,
    xoff: *const usize,
    y: *mut f32,
    ys: usize,
    first: bool,
) {
    use std::arch::x86_64::*;
    unsafe {
        let mut c = [[_mm256_setzero_ps(); 3]; MR];
        if !first {
            for (r, cr) in c.iter_mut().enumerate() {
                for (q, v) in cr.iter_mut().enumerate() {
                    *v = _mm256_loadu_ps(y.add(r * ys + q * 8));
                }
            }
        }
        for j in 0..kn {
            let xp = x.add(*xoff.add(j));
            let x0 = _mm256_loadu_ps(xp);
            let x1 = _mm256_loadu_ps(xp.add(8));
            let x2 = _mm256_loadu_ps(xp.add(16));
            let w = wp.add(j * MR);
            for (r, cr) in c.iter_mut().enumerate() {
                let b = _mm256_broadcast_ss(&*w.add(r));
                cr[0] = _mm256_fmadd_ps(b, x0, cr[0]);
                cr[1] = _mm256_fmadd_ps(b, x1, cr[1]);
                cr[2] = _mm256_fmadd_ps(b, x2, cr[2]);
            }
        }
        for (r, cr) in c.iter().enumerate() {
            for (q, v) in cr.iter().enumerate() {
                _mm256_storeu_ps(y.add(r * ys + q * 8), *v);
            }
        }
    }
}

unsafe fn micro_scalar(
    wp: *const f32,
    kn: usize,
    x: *const f32,
    xoff: *const usize,
    y: *mut f32,
    ys: usize,
    first: bool,
) {
    unsafe {
        let mut c = [[0.0f32; NR]; MR];
        if !first {
            for (r, cr) in c.iter_mut().enumerate() {
                for (q, v) in cr.iter_mut().enumerate() {
                    *v = *y.add(r * ys + q);
                }
            }
        }
        for j in 0..kn {
            let xr = &*(x.add(*xoff.add(j)) as *const [f32; NR]);
            let w = &*(wp.add(j * MR) as *const [f32; MR]);
            for (cr, &wr) in c.iter_mut().zip(w) {
                for (v, &xv) in cr.iter_mut().zip(xr) {
                    *v += wr * xv;
                }
            }
        }
        for (r, cr) in c.iter().enumerate() {
            for (q, v) in cr.iter().enumerate() {
                *y.add(r * ys + q) = *v;
            }
        }
    }
}

/// `y[co][t] = epi(co, sum_j W[co][j] * x[xoff[j] + t])` for `t < ncols` (a multiple of NR).
#[allow(clippy::too_many_arguments)]
fn gemm(
    threads: usize,
    fma: bool,
    w: &[f32],
    cout: usize,
    x: &[f32],
    xoff: &[usize],
    y: &mut [f32],
    ys: usize,
    ncols: usize,
    epi: impl Fn(usize, &mut [f32]) + Sync,
) {
    let kdim = xoff.len();
    assert!(
        cout % MR == 0
            && ncols % NR == 0
            && ncols <= ys
            && w.len() == cout * kdim
            && y.len() >= cout * ys
    );
    assert!(xoff.iter().all(|&o| o + ncols <= x.len()));
    let npanels = cout / MR;
    let ngroups = npanels.div_ceil(GROUP);
    let ntiles = ncols / NR;
    let chunks = (2 * threads).div_ceil(ngroups).clamp(1, ntiles.max(1));
    let per = ntiles.div_ceil(chunks);
    let yp = SyncPtr(y.as_mut_ptr());
    #[cfg(target_arch = "x86_64")]
    let micro = if fma { micro_avx2 } else { micro_scalar };
    #[cfg(not(target_arch = "x86_64"))]
    let micro = {
        let _ = fma;
        micro_scalar
    };
    par_for(threads, ngroups * chunks, |item| {
        let yp = yp;
        let (g, ch) = (item / chunks, item % chunks);
        let (p0, p1) = (g * GROUP, ((g + 1) * GROUP).min(npanels));
        let (t0, t1) = (ch * per, ((ch + 1) * per).min(ntiles));
        if t0 >= t1 {
            return;
        }
        for j0 in (0..kdim).step_by(KC) {
            let kn = KC.min(kdim - j0);
            for tile in t0..t1 {
                let col = tile * NR;
                for p in p0..p1 {
                    // SAFETY: bounds asserted above; items write disjoint (panel, tile) blocks of y.
                    unsafe {
                        micro(
                            w.as_ptr().add((p * kdim + j0) * MR),
                            kn,
                            x.as_ptr().add(col),
                            xoff.as_ptr().add(j0),
                            yp.0.add(p * MR * ys + col),
                            ys,
                            j0 == 0,
                        )
                    };
                }
            }
        }
        let (c0, c1) = (t0 * NR, t1 * NR);
        for co in p0 * MR..p1 * MR {
            // SAFETY: row co, columns c0..c1 belong to this item only.
            let row = unsafe { std::slice::from_raw_parts_mut(yp.0.add(co * ys + c0), c1 - c0) };
            epi(co, row);
        }
    });
}

impl Conv {
    fn forward(
        &self,
        e: &Ecapa,
        x: &[f32],
        t: usize,
        s: usize,
        y: &mut [f32],
        epi: impl Fn(usize, &mut [f32]) + Sync,
    ) {
        if self.k == 1 {
            let xoff: Vec<usize> = (0..self.cin).map(|ci| ci * s).collect();
            gemm(e.threads, e.fma, &self.w, self.cout, x, &xoff, y, s, s, epi);
            return;
        }
        let p = self.dil * (self.k - 1) / 2;
        let sp = s + 2 * p;
        let mut xp = vec![0.0f32; self.cin * sp];
        for ci in 0..self.cin {
            let src = &x[ci * s..][..t];
            let dst = &mut xp[ci * sp..][..t + 2 * p];
            dst[p..p + t].copy_from_slice(src);
            for i in 1..=p {
                dst[p - i] = src[i];
                dst[p + t - 1 + i] = src[t - 1 - i];
            }
        }
        let xoff: Vec<usize> = (0..self.k)
            .flat_map(|k| (0..self.cin).map(move |ci| ci * sp + k * self.dil))
            .collect();
        gemm(
            e.threads, e.fma, &self.w, self.cout, &xp, &xoff, y, s, s, epi,
        );
    }
}

impl Tdnn {
    fn forward_into(&self, e: &Ecapa, x: &[f32], t: usize, s: usize, y: &mut [f32]) {
        self.forward_bias(e, x, t, s, y, &self.conv.b);
    }
    fn forward_bias(&self, e: &Ecapa, x: &[f32], t: usize, s: usize, y: &mut [f32], bias: &[f32]) {
        self.conv.forward(e, x, t, s, y, |co, row| {
            let (b, sc, sh) = (bias[co], self.scale[co], self.shift[co]);
            for v in row {
                *v = (*v + b).max(0.0) * sc + sh;
            }
        });
    }
    fn forward(&self, e: &Ecapa, x: &Act) -> Act {
        let mut y = Act::zeros(self.conv.cout, x.t);
        self.forward_into(e, &x.d, x.t, x.s, &mut y.d);
        y
    }
}

impl Dense {
    fn apply(&self, x: &[f32]) -> Vec<f32> {
        (0..self.cout)
            .map(|o| {
                let w = &self.w[o * self.cin..][..self.cin];
                self.b[o] + w.iter().zip(x).map(|(a, b)| (a * b) as f64).sum::<f64>() as f32
            })
            .collect()
    }
}

impl SeRes2 {
    fn forward(&self, e: &Ecapa, x: &Act) -> Act {
        let (t, s) = (x.t, x.s);
        let a = self.tdnn1.forward(e, x);
        let h = C / SCALE;
        let mut r = Act::zeros(C, t);
        r.d[..h * s].copy_from_slice(&a.d[..h * s]);
        let mut tmp = vec![0.0f32; h * s];
        for (i, blk) in self.res2.iter().enumerate() {
            let ci = i + 1;
            let (prev, cur) = r.d.split_at_mut(ci * h * s);
            let xi = &a.d[ci * h * s..][..h * s];
            if i == 0 {
                tmp.copy_from_slice(xi);
            } else {
                let prev = &prev[(ci - 1) * h * s..];
                for ((d, &u), &v) in tmp.iter_mut().zip(xi).zip(prev) {
                    *d = u + v;
                }
            }
            blk.forward_into(e, &tmp, t, s, &mut cur[..h * s]);
        }
        let mut y = self.tdnn2.forward(e, &r);
        let mean: Vec<f32> = (0..C)
            .map(|c| (y.row(c).iter().map(|&v| v as f64).sum::<f64>() / t as f64) as f32)
            .collect();
        let g: Vec<f32> = self
            .se2
            .apply(
                &self
                    .se1
                    .apply(&mean)
                    .iter()
                    .map(|v| v.max(0.0))
                    .collect::<Vec<_>>(),
            )
            .iter()
            .map(|v| 1.0 / (1.0 + (-v).exp()))
            .collect();
        for c in 0..C {
            let (yr, xr) = (&mut y.d[c * s..][..s], &x.d[c * s..][..s]);
            for (v, &res) in yr.iter_mut().zip(xr) {
                *v = *v * g[c] + res;
            }
        }
        y
    }
}

fn stats(x: &[f32], w: Option<&[f32]>) -> (f32, f32) {
    let t = x.len() as f64;
    let mean = match w {
        Some(w) => x
            .iter()
            .zip(w)
            .map(|(&a, &b)| a as f64 * b as f64)
            .sum::<f64>(),
        None => x.iter().map(|&a| a as f64).sum::<f64>() / t,
    };
    let var = match w {
        Some(w) => x
            .iter()
            .zip(w)
            .map(|(&a, &b)| b as f64 * (a as f64 - mean).powi(2))
            .sum::<f64>(),
        None => x.iter().map(|&a| (a as f64 - mean).powi(2)).sum::<f64>() / t,
    };
    (mean as f32, (var.max(1e-12) as f32).sqrt())
}

fn read_safetensors(path: &Path) -> Result<HashMap<String, (Vec<usize>, Vec<f32>)>> {
    let bad = |reason: String| RuntimeError::Packet {
        path: path.to_path_buf(),
        reason,
    };
    let bytes = std::fs::read(path).map_err(|source| RuntimeError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let n = u64::from_le_bytes(
        bytes
            .get(..8)
            .ok_or_else(|| bad("short file".into()))?
            .try_into()
            .unwrap(),
    ) as usize;
    let header = bytes
        .get(8..8 + n)
        .ok_or_else(|| bad("short header".into()))?;
    let header: serde_json::Map<String, serde_json::Value> = serde_json::from_slice(header)
        .map_err(|source| RuntimeError::Json {
            path: path.to_path_buf(),
            source,
        })?;
    let data = &bytes[8 + n..];
    let mut out = HashMap::new();
    for (k, v) in header {
        if k == "__metadata__" {
            continue;
        }
        if v["dtype"] != "F32" {
            return Err(bad(format!("{k}: dtype {} (want F32)", v["dtype"])));
        }
        let shape: Vec<usize> = v["shape"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|d| d.as_u64())
            .map(|d| d as usize)
            .collect();
        let off: Vec<usize> = v["data_offsets"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|d| d.as_u64())
            .map(|d| d as usize)
            .collect();
        let raw = (off.len() == 2)
            .then(|| data.get(off[0]..off[1]))
            .flatten()
            .ok_or_else(|| bad(format!("{k}: bad offsets")))?;
        let vals: Vec<f32> = raw
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        if vals.len() != shape.iter().product::<usize>() {
            return Err(bad(format!("{k}: size mismatch")));
        }
        out.insert(k, (shape, vals));
    }
    Ok(out)
}

struct Weights {
    path: std::path::PathBuf,
    map: HashMap<String, (Vec<usize>, Vec<f32>)>,
}

impl Weights {
    fn open(path: std::path::PathBuf) -> Result<Self> {
        Ok(Self {
            map: read_safetensors(&path)?,
            path,
        })
    }
    fn take(&mut self, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
        let reason = match self.map.remove(name) {
            Some((s, v)) if s == shape => return Ok(v),
            Some((s, _)) => format!("{name}: shape {s:?}, want {shape:?}"),
            None => format!("missing tensor {name}"),
        };
        Err(RuntimeError::Packet {
            path: self.path.clone(),
            reason,
        })
    }
    fn conv(&mut self, p: &str, cout: usize, k: usize, cin: usize, dil: usize) -> Result<Conv> {
        let w = self.take(&format!("{p}.weight"), &[cout, k, cin])?;
        Ok(Conv {
            cout,
            cin,
            k,
            dil,
            w: pack(&w, cout, k * cin),
            b: self.take(&format!("{p}.bias"), &[cout])?,
        })
    }
    fn tdnn(&mut self, p: &str, cout: usize, k: usize, cin: usize, dil: usize) -> Result<Tdnn> {
        let conv = self.conv(&format!("{p}.conv"), cout, k, cin, dil)?;
        Ok(Tdnn {
            conv,
            scale: self.take(&format!("{p}.bn.scale"), &[cout])?,
            shift: self.take(&format!("{p}.bn.shift"), &[cout])?,
        })
    }
    /// `bias = false`: `p` names the weight tensor itself and the bias is zero.
    fn dense(&mut self, p: &str, cout: usize, cin: usize, bias: bool) -> Result<Dense> {
        if !bias {
            return Ok(Dense {
                cout,
                cin,
                w: self.take(p, &[cout, 1, cin])?,
                b: vec![0.0; cout],
            });
        }
        Ok(Dense {
            cout,
            cin,
            w: self.take(&format!("{p}.weight"), &[cout, 1, cin])?,
            b: self.take(&format!("{p}.bias"), &[cout])?,
        })
    }
}

/// scipy `firwin(61, 1/3, window=('kaiser', 5.0))`, cast to f32 and scaled by `up = 2` as `resample_poly` does.
fn resample_taps() -> Vec<f32> {
    fn i0(x: f64) -> f64 {
        let (mut sum, mut term, mut k) = (1.0, 1.0, 1.0);
        while term > 1e-17 * sum {
            term *= (x / (2.0 * k)).powi(2);
            sum += term;
            k += 1.0;
        }
        sum
    }
    let (n, fc, beta) = (61usize, 1.0 / 3.0, 5.0);
    let alpha = (n - 1) as f64 / 2.0;
    let mut h: Vec<f64> = (0..n)
        .map(|i| {
            let m = i as f64 - alpha;
            let sinc = if m == 0.0 {
                1.0
            } else {
                (std::f64::consts::PI * fc * m).sin() / (std::f64::consts::PI * fc * m)
            };
            let win = i0(beta * (1.0 - (m / alpha).powi(2)).max(0.0).sqrt()) / i0(beta);
            fc * sinc * win
        })
        .collect();
    let s: f64 = h.iter().sum();
    h.iter_mut().for_each(|v| *v /= s);
    h.iter().map(|&v| v as f32 * 2.0).collect()
}

/// speechbrain `Filterbank` triangular matrix `[N_MELS][N_BINS]`, computed in f32 like torch.
fn mel_matrix() -> Vec<f32> {
    let n = N_MELS + 2;
    let (start, end) = (0.0f32, (2595.0 * (1.0 + 8000.0f64 / 700.0).log10()) as f32);
    let step = (end - start) / (n - 1) as f32;
    let hz: Vec<f32> = (0..n)
        .map(|i| {
            if i < n / 2 {
                start + step * i as f32
            } else {
                end - step * (n - 1 - i) as f32
            }
        })
        .map(|m| 700.0 * (10f32.powf(m / 2595.0) - 1.0))
        .collect();
    let mut w = vec![0.0f32; N_MELS * N_BINS];
    for m in 0..N_MELS {
        let (fc, band) = (hz[m + 1], hz[m + 1] - hz[m]);
        for k in 0..N_BINS {
            let slope = (k as f32 * 40.0 - fc) / band;
            w[m * N_BINS + k] = (slope + 1.0).min(1.0 - slope).max(0.0);
        }
    }
    w
}

/// Periodic-Hamming-windowed real DFT basis `[2 * N_BINS rounded up to MR][N_FFT]`: cos rows then sin rows.
fn dft_matrix() -> Vec<f32> {
    let rows = (2 * N_BINS).div_ceil(MR) * MR;
    let mut w = vec![0.0f32; rows * N_FFT];
    let tau = 2.0 * std::f64::consts::PI;
    for i in 0..N_FFT {
        let win = (0.54 - 0.46 * (tau * i as f64 / N_FFT as f64).cos()) as f32 as f64;
        for k in 0..N_BINS {
            let ph = tau * ((k * i) % N_FFT) as f64 / N_FFT as f64;
            w[k * N_FFT + i] = (win * ph.cos()) as f32;
            w[(N_BINS + k) * N_FFT + i] = (win * ph.sin()) as f32;
        }
    }
    w
}

impl Ecapa {
    pub fn load(dir: &Path) -> Result<Self> {
        let mut ws = Weights::open(dir.join("model.safetensors"))?;
        let block0 = ws.tdnn("blocks.0", C, 5, N_MELS, 1)?;
        let mut blocks = Vec::new();
        for b in 1..=3 {
            blocks.push(SeRes2 {
                tdnn1: ws.tdnn(&format!("blocks.{b}.tdnn1"), C, 1, C, 1)?,
                res2: (0..SCALE - 1)
                    .map(|i| {
                        ws.tdnn(
                            &format!("blocks.{b}.res2net.{i}"),
                            C / SCALE,
                            3,
                            C / SCALE,
                            b + 1,
                        )
                    })
                    .collect::<Result<Vec<_>>>()?,
                tdnn2: ws.tdnn(&format!("blocks.{b}.tdnn2"), C, 1, C, 1)?,
                se1: ws.dense(&format!("blocks.{b}.se.conv1"), ATT, C, true)?,
                se2: ws.dense(&format!("blocks.{b}.se.conv2"), C, ATT, true)?,
            });
        }
        let mfa = ws.tdnn("mfa", M, 1, 3 * C, 1)?;
        let asp_x = Tdnn {
            conv: Conv {
                cout: ATT,
                cin: M,
                k: 1,
                dil: 1,
                w: pack(&ws.take("asp.tdnn.conv.weight_x", &[ATT, 1, M])?, ATT, M),
                b: ws.take("asp.tdnn.conv.bias", &[ATT])?,
            },
            scale: ws.take("asp.tdnn.bn.scale", &[ATT])?,
            shift: ws.take("asp.tdnn.bn.shift", &[ATT])?,
        };
        let asp_mean = ws.dense("asp.tdnn.conv.weight_mean", ATT, M, false)?;
        let asp_std = ws.dense("asp.tdnn.conv.weight_std", ATT, M, false)?;
        let asp_conv = ws.conv("asp.conv", M, 1, ATT, 1)?;
        let fc = ws.dense("fc", EMBED_DIM, 2 * M, true)?;
        let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
        #[cfg(target_arch = "x86_64")]
        let fma = std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma");
        #[cfg(not(target_arch = "x86_64"))]
        let fma = false;
        let dft = dft_matrix();
        let mel = pack(&mel_matrix(), N_MELS, N_BINS);
        Ok(Self {
            block0,
            blocks,
            mfa,
            asp_x,
            asp_mean,
            asp_std,
            asp_conv,
            fc,
            fir: resample_taps(),
            dft: pack(&dft, dft.len() / N_FFT, N_FFT),
            mel,
            threads,
            fma,
        })
    }

    /// `Ecapa.__call__`: 24 kHz mono in, L2-normalised speaker embedding out.
    pub fn embed(&self, pcm24: &[f32]) -> std::result::Result<[f32; EMBED_DIM], String> {
        if pcm24.len() < MIN_SAMPLES {
            return Err(format!(
                "ecapa: reference audio too short ({} samples < {MIN_SAMPLES})",
                pcm24.len()
            ));
        }
        let mut x = self.fbank(&self.resample(pcm24));
        for c in 0..N_MELS {
            let t = x.t;
            let row = &mut x.d[c * x.s..][..t];
            let mean = (row.iter().map(|&v| v as f64).sum::<f64>() / t as f64) as f32;
            row.iter_mut().for_each(|v| *v -= mean);
        }
        Ok(self.network(&x))
    }

    /// `scipy.signal.resample_poly(x, 2, 3)` (zero padding): `y[m] = sum_j x[j] h[3m + 30 - 2j]`.
    fn resample(&self, x: &[f32]) -> Vec<f32> {
        let n = x.len();
        let n_out = (2 * n).div_ceil(3);
        let mut y = vec![0.0f32; n_out];
        const CH: usize = 8192;
        let yp = SyncPtr(y.as_mut_ptr());
        par_for(self.threads, n_out.div_ceil(CH), |c| {
            let yp = yp;
            for m in c * CH..((c + 1) * CH).min(n_out) {
                let center = 3 * m + 30;
                let j0 = center.saturating_sub(60).div_ceil(2);
                let j1 = (center / 2).min(n - 1);
                let mut acc = 0.0f32;
                for j in j0..=j1 {
                    acc += x[j] * self.fir[center - 2 * j];
                }
                // SAFETY: each chunk writes its own indices.
                unsafe { *yp.0.add(m) = acc };
            }
        });
        y
    }

    /// speechbrain `Fbank(n_mels=80)` (centered zero-padded STFT, power spectrum, 10*log10 with top_db = 80) as
    /// `[N_MELS][T]`.
    fn fbank(&self, w: &[f32]) -> Act {
        let t = 1 + w.len() / HOP;
        let mut frames = Act::zeros(N_FFT, t);
        let s = frames.s;
        for i in 0..N_FFT {
            let row = &mut frames.d[i * s..][..t];
            for (f, v) in row.iter_mut().enumerate() {
                let idx = (f * HOP + i) as isize - (N_FFT / 2) as isize;
                *v = if idx >= 0 && (idx as usize) < w.len() {
                    w[idx as usize]
                } else {
                    0.0
                };
            }
        }
        let rows = self.dft.len() / N_FFT;
        let mut spec = vec![0.0f32; rows * s];
        let xoff: Vec<usize> = (0..N_FFT).map(|i| i * s).collect();
        gemm(
            self.threads,
            self.fma,
            &self.dft,
            rows,
            &frames.d,
            &xoff,
            &mut spec,
            s,
            s,
            |_, _| {},
        );
        let mut pow = vec![0.0f32; N_BINS * s];
        for k in 0..N_BINS {
            for f in 0..s {
                let (re, im) = (spec[k * s + f], spec[(N_BINS + k) * s + f]);
                pow[k * s + f] = re * re + im * im;
            }
        }
        let mut out = Act::zeros(N_MELS, t);
        let xoff: Vec<usize> = (0..N_BINS).map(|k| k * s).collect();
        gemm(
            self.threads,
            self.fma,
            &self.mel,
            N_MELS,
            &pow,
            &xoff,
            &mut out.d,
            s,
            s,
            |_, row| {
                row.iter_mut()
                    .for_each(|v| *v = 10.0 * v.max(1e-10).log10());
            },
        );
        let max = (0..N_MELS)
            .flat_map(|c| out.row(c).iter().copied())
            .fold(f32::NEG_INFINITY, f32::max);
        out.d.iter_mut().for_each(|v| *v = v.max(max - 80.0));
        out
    }

    fn network(&self, x: &Act) -> [f32; EMBED_DIM] {
        let (t, s) = (x.t, x.s);
        let mut h = self.block0.forward(self, x);
        let mut cat = Act::zeros(3 * C, t);
        for (b, blk) in self.blocks.iter().enumerate() {
            h = blk.forward(self, &h);
            cat.d[b * C * s..][..C * s].copy_from_slice(&h.d);
        }
        drop(h);
        let y = self.mfa.forward(self, &cat);
        drop(cat);

        let (mean, std): (Vec<f32>, Vec<f32>) = (0..M).map(|c| stats(y.row(c), None)).unzip();
        let bias: Vec<f32> = self
            .asp_x
            .conv
            .b
            .iter()
            .zip(self.asp_mean.apply(&mean))
            .zip(self.asp_std.apply(&std))
            .map(|((b, m), s)| b + m + s)
            .collect();
        let mut a = Act::zeros(ATT, t);
        self.asp_x.forward_bias(self, &y.d, t, s, &mut a.d, &bias);
        a.d.iter_mut().for_each(|v| *v = v.tanh());
        let mut logits = Act::zeros(M, t);
        self.asp_conv
            .forward(self, &a.d, t, s, &mut logits.d, |co, row| {
                let b = self.asp_conv.b[co];
                row.iter_mut().for_each(|v| *v += b);
            });

        let mut pooled = vec![0.0f32; 2 * M];
        let pp = SyncPtr(pooled.as_mut_ptr());
        let lp = SyncPtr(logits.d.as_mut_ptr());
        let yv = &y;
        par_for(self.threads, M.div_ceil(64), |g| {
            let (pp, lp) = (pp, lp);
            for c in g * 64..((g + 1) * 64).min(M) {
                // SAFETY: each group owns rows c of logits and entries c, M + c of pooled.
                let l = unsafe { std::slice::from_raw_parts_mut(lp.0.add(c * s), t) };
                let mx = l.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0f64;
                for v in l.iter_mut() {
                    *v = (*v - mx).exp();
                    sum += *v as f64;
                }
                let inv = (1.0 / sum) as f32;
                l.iter_mut().for_each(|v| *v *= inv);
                let (m, sd) = stats(yv.row(c), Some(l));
                unsafe {
                    *pp.0.add(c) = m;
                    *pp.0.add(M + c) = sd;
                }
            }
        });
        let e = self.fc.apply(&pooled);
        let norm = (e.iter().map(|&v| v as f64 * v as f64).sum::<f64>().sqrt() as f32).max(1e-12);
        let mut out = [0.0f32; EMBED_DIM];
        for (o, v) in out.iter_mut().zip(e) {
            *o = v / norm;
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::time::Instant;

    fn s16(path: &Path) -> Vec<f32> {
        std::fs::read(path)
            .unwrap()
            .chunks_exact(2)
            .map(|b| i16::from_le_bytes([b[0], b[1]]) as f32 / 32768.0)
            .collect()
    }

    #[test]
    fn parity() {
        let dir = std::env::var_os("PLOW_ECAPA_DIR").map_or_else(
            || PathBuf::from("/home/ssm-user/asr-work/maya2/ecapa"),
            PathBuf::from,
        );
        let data = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/ecapa");
        if !dir.join("model.safetensors").exists() || !data.join("cases.json").exists() {
            eprintln!("skip ecapa parity: no export at {} (scripts/tts/ecapa_export.py) or fixtures at {}", dir.display(), data.display());
            return;
        }
        let e = Ecapa::load(&dir).unwrap();
        let spec: serde_json::Value =
            serde_json::from_slice(&std::fs::read(data.join("cases.json")).unwrap()).unwrap();
        let mut clips: HashMap<String, Vec<f32>> = HashMap::new();
        let mut clip = |n: &str| {
            clips
                .entry(n.to_string())
                .or_insert_with(|| s16(&data.join(format!("{n}.s16"))))
                .clone()
        };

        let fb = &spec["fbank"];
        let want: Vec<f32> = std::fs::read(data.join(fb["file"].as_str().unwrap()))
            .unwrap()
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        let got = e.fbank(&e.resample(&clip(fb["case"].as_str().unwrap())));
        let t = fb["frames"].as_u64().unwrap() as usize;
        assert_eq!(got.t, t);
        let fdiff = (0..t)
            .flat_map(|f| (0..N_MELS).map(move |c| (f, c)))
            .map(|(f, c)| (got.d[c * got.s + f] - want[f * N_MELS + c]).abs())
            .fold(0.0f32, f32::max);
        eprintln!("fbank {}: max abs {fdiff:.2e}", fb["case"]);

        let mut fails = Vec::new();
        for case in spec["cases"].as_array().unwrap() {
            let name = case["name"].as_str().unwrap();
            let mut pcm: Vec<f32> = case["parts"]
                .as_array()
                .unwrap()
                .iter()
                .flat_map(|p| clip(p.as_str().unwrap()))
                .collect();
            pcm.truncate(case["samples"].as_u64().unwrap() as usize);
            let want: Vec<f32> = case["embedding"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap() as f32)
                .collect();
            let t0 = Instant::now();
            let got = e.embed(&pcm).unwrap();
            let ms = t0.elapsed().as_secs_f64() * 1e3;
            let cos: f64 = got
                .iter()
                .zip(&want)
                .map(|(&a, &b)| a as f64 * b as f64)
                .sum();
            let maxd = got
                .iter()
                .zip(&want)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            eprintln!(
                "{name}: {:.2}s audio, cos {cos:.7}, max abs {maxd:.2e}, {ms:.1} ms",
                pcm.len() as f64 / 24000.0
            );
            if cos < 0.9999 || maxd > 2e-3 {
                fails.push(name.to_string());
            }
        }
        assert!(e.embed(&[0.0; MIN_SAMPLES - 1]).is_err());
        assert!(fdiff <= 1e-3, "fbank max abs {fdiff}");
        assert!(fails.is_empty(), "parity failures: {fails:?}");
    }
}
