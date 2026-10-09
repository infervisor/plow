//! Host preprocessing of chat media: data-URL decoding, images to patches (aspect-preserving
//! resize to a patch budget), audio to log-mel frames. Each algorithm is parameterized by the
//! packet's multimodal contract (`plow_asset::multimodal`), never by a model name.

use base64::Engine as _;

/// A request-facing failure: becomes a 400 with this message.
pub type MediaResult<T> = std::result::Result<T, String>;

/// `data:<mime>;base64,<payload>` -> (mime, bytes).
pub fn data_url(url: &str) -> MediaResult<(String, Vec<u8>)> {
    let rest = url.strip_prefix("data:").ok_or("only data: URLs are accepted for media")?;
    let (meta, payload) = rest.split_once(',').ok_or("malformed data URL")?;
    let mime = meta.split(';').next().unwrap_or("").to_ascii_lowercase();
    if !meta.split(';').any(|p| p.eq_ignore_ascii_case("base64")) {
        return Err("data URLs must be base64-encoded".into());
    }
    Ok((mime, base64_decode(payload)?))
}

pub fn base64_decode(payload: &str) -> MediaResult<Vec<u8>> {
    let clean: String = payload.chars().filter(|c| !c.is_ascii_whitespace()).collect();
    base64::engine::general_purpose::STANDARD
        .decode(clean.as_bytes())
        .or_else(|_| base64::engine::general_purpose::STANDARD_NO_PAD.decode(clean.as_bytes()))
        .map_err(|e| format!("invalid base64 media: {e}"))
}

/// An 8-bit RGB image, row-major `[height][width][3]`.
pub struct Rgb {
    pub width: u32,
    pub height: u32,
    pub data: Vec<u8>,
}

/// Decode PNG / JPEG / WebP / GIF bytes to RGB (alpha dropped, as `convert("RGB")`).
pub fn decode_image(bytes: &[u8], max_pixels: u64) -> MediaResult<Rgb> {
    let reader = image::ImageReader::new(std::io::Cursor::new(bytes))
        .with_guessed_format()
        .map_err(|e| format!("unreadable image: {e}"))?;
    let (w, h) = reader.into_dimensions().map_err(|e| format!("unreadable image: {e}"))?;
    if w == 0 || h == 0 || u64::from(w) * u64::from(h) > max_pixels {
        return Err(format!("image is {w}x{h}; at most {max_pixels} pixels are accepted"));
    }
    let img = image::ImageReader::new(std::io::Cursor::new(bytes))
        .with_guessed_format()
        .map_err(|e| format!("unreadable image: {e}"))?
        .decode()
        .map_err(|e| format!("unreadable image: {e}"))?;
    let rgb = img.to_rgb8();
    Ok(Rgb { width: rgb.width(), height: rgb.height(), data: rgb.into_raw() })
}

/// Parameters of the `aspect_patches` processor (contract `image` modality).
#[derive(Clone, Debug)]
pub struct PatchParams {
    pub patch: u32,
    pub pool: u32,
    pub max_soft_tokens: u32,
    pub rescale: f32,
    pub normalize: bool,
    pub mean: [f32; 3],
    pub std: [f32; 3],
}

/// One image as encoder input: `[patches][patch*patch*3]` values in `(py, px, channel)` order,
/// row-major over the patch grid, and each patch's `(x, y)` grid position.
pub struct Patches {
    pub values: Vec<f32>,
    pub positions: Vec<[u32; 2]>,
    pub grid: (u32, u32),
    pub soft_tokens: u32,
}

/// The largest `pool*patch`-multiple size with at most `max_soft_tokens * pool^2` patches that
/// keeps the aspect ratio (Gemma-4's `get_aspect_ratio_preserving_size`, verbatim arithmetic).
pub fn aspect_size(height: u32, width: u32, p: &PatchParams) -> MediaResult<(u32, u32)> {
    let max_patches = f64::from(p.max_soft_tokens * p.pool * p.pool);
    let target_px = max_patches * f64::from(p.patch * p.patch);
    let factor = (target_px / (f64::from(height) * f64::from(width))).sqrt();
    let side = f64::from(p.pool * p.patch);
    let mut th = ((factor * f64::from(height)) / side).floor() * side;
    let mut tw = ((factor * f64::from(width)) / side).floor() * side;
    if th == 0.0 && tw == 0.0 {
        return Err("image resizes to 0x0".into());
    }
    let max_side = f64::from(p.max_soft_tokens) * side;
    if th == 0.0 {
        th = side;
        tw = ((f64::from(width) / f64::from(height)).floor() * side).min(max_side);
    } else if tw == 0.0 {
        tw = side;
        th = ((f64::from(height) / f64::from(width)).floor() * side).min(max_side);
    }
    if th * tw > target_px {
        return Err(format!("image of {height}x{width} does not fit the patch budget"));
    }
    Ok((th as u32, tw as u32))
}

pub fn image_patches(img: &Rgb, p: &PatchParams) -> MediaResult<Patches> {
    let (th, tw) = aspect_size(img.height, img.width, p)?;
    let resized = if (th, tw) == (img.height, img.width) {
        img.data.clone()
    } else {
        resize_bicubic(&img.data, img.width, img.height, tw, th)
    };
    let (ph, pw) = (th / p.patch, tw / p.patch);
    let ps = p.patch as usize;
    let n = (ph * pw) as usize;
    let per = ps * ps * 3;
    let mut values = vec![0f32; n * per];
    let mut positions = Vec::with_capacity(n);
    for r in 0..ph as usize {
        for c in 0..pw as usize {
            let k = r * pw as usize + c;
            positions.push([c as u32, r as u32]);
            let out = &mut values[k * per..(k + 1) * per];
            for py in 0..ps {
                for px in 0..ps {
                    let src = ((r * ps + py) * tw as usize + c * ps + px) * 3;
                    for ch in 0..3 {
                        let mut v = f32::from(resized[src + ch]) * p.rescale;
                        if p.normalize {
                            v = (v - p.mean[ch]) / p.std[ch];
                        }
                        out[(py * ps + px) * 3 + ch] = v;
                    }
                }
            }
        }
    }
    Ok(Patches { values, positions, grid: (ph, pw), soft_tokens: ph * pw / (p.pool * p.pool) })
}

/// Pillow's `Image.resize(BICUBIC)` on 8-bit RGB: separable, support scaled by the downscale
/// factor (antialias), 22-bit fixed-point coefficients, horizontal pass first.
pub fn resize_bicubic(src: &[u8], w: u32, h: u32, tw: u32, th: u32) -> Vec<u8> {
    const PRECISION: u32 = 22;
    fn cubic(x: f64) -> f64 {
        let a = -0.5;
        let x = x.abs();
        if x < 1.0 {
            ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0
        } else if x < 2.0 {
            (((x - 5.0) * x + 8.0) * x - 4.0) * a
        } else {
            0.0
        }
    }
    // (first input index, coefficients) per output index.
    fn coeffs(in_size: u32, out_size: u32) -> Vec<(usize, Vec<i32>)> {
        let scale = f64::from(in_size) / f64::from(out_size);
        let filterscale = scale.max(1.0);
        let support = 2.0 * filterscale;
        (0..out_size)
            .map(|xx| {
                let center = (f64::from(xx) + 0.5) * scale;
                let ss = 1.0 / filterscale;
                let xmin = ((center - support + 0.5) as i64).max(0) as usize;
                let xmax = ((center + support + 0.5) as i64).min(i64::from(in_size)) as usize;
                let mut k: Vec<f64> = (xmin..xmax).map(|x| cubic((x as f64 - center + 0.5) * ss)).collect();
                let ww: f64 = k.iter().sum();
                if ww != 0.0 {
                    k.iter_mut().for_each(|v| *v /= ww);
                }
                let fixed = k
                    .iter()
                    .map(|&v| {
                        let s = v * f64::from(1u32 << PRECISION);
                        if v < 0.0 {
                            (-0.5 + s) as i32
                        } else {
                            (0.5 + s) as i32
                        }
                    })
                    .collect();
                (xmin, fixed)
            })
            .collect()
    }
    fn clip8(v: i64) -> u8 {
        (v >> PRECISION).clamp(0, 255) as u8
    }
    let round = 1i64 << (PRECISION - 1);
    let (w, h, tw, th) = (w as usize, h as usize, tw as usize, th as usize);
    let horizontal = |src: &[u8], rows: usize, cw: &[(usize, Vec<i32>)]| -> Vec<u8> {
        let ow = cw.len();
        let mut out = vec![0u8; rows * ow * 3];
        for y in 0..rows {
            for (x, (x0, k)) in cw.iter().enumerate() {
                for ch in 0..3 {
                    let mut s = round;
                    for (i, &kv) in k.iter().enumerate() {
                        s += i64::from(src[(y * w + x0 + i) * 3 + ch]) * i64::from(kv);
                    }
                    out[(y * ow + x) * 3 + ch] = clip8(s);
                }
            }
        }
        out
    };
    let mid = if tw != w { horizontal(src, h, &coeffs(w as u32, tw as u32)) } else { src.to_vec() };
    if th == h {
        return mid;
    }
    let ch_ = coeffs(h as u32, th as u32);
    let mut out = vec![0u8; th * tw * 3];
    for (y, (y0, k)) in ch_.iter().enumerate() {
        for x in 0..tw {
            for c in 0..3 {
                let mut s = round;
                for (i, &kv) in k.iter().enumerate() {
                    s += i64::from(mid[((y0 + i) * tw + x) * 3 + c]) * i64::from(kv);
                }
                out[(y * tw + x) * 3 + c] = clip8(s);
            }
        }
    }
    out
}

/// Decode a WAV (PCM 8/16/24/32-bit or float) to mono f32 in [-1, 1] and its sample rate.
pub fn decode_wav(bytes: &[u8]) -> MediaResult<(Vec<f32>, u32)> {
    let mut reader = hound::WavReader::new(std::io::Cursor::new(bytes)).map_err(|e| format!("unreadable WAV: {e}"))?;
    let spec = reader.spec();
    let channels = usize::from(spec.channels.max(1));
    let interleaved: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => {
            reader.samples::<f32>().collect::<Result<_, _>>().map_err(|e| format!("unreadable WAV: {e}"))?
        }
        hound::SampleFormat::Int => {
            let scale = 1.0 / (1u64 << (spec.bits_per_sample.clamp(1, 32) - 1)) as f32;
            reader
                .samples::<i32>()
                .map(|s| s.map(|v| v as f32 * scale))
                .collect::<Result<_, _>>()
                .map_err(|e| format!("unreadable WAV: {e}"))?
        }
    };
    let mono = interleaved.chunks(channels).map(|c| c.iter().sum::<f32>() / c.len() as f32).collect();
    Ok((mono, spec.sample_rate))
}

/// Band-limited resampling (Kaiser-windowed sinc, 32 zero crossings).
pub fn resample(x: &[f32], from: u32, to: u32) -> Vec<f32> {
    if from == to || x.is_empty() {
        return x.to_vec();
    }
    let ratio = f64::from(to) / f64::from(from);
    let cutoff = ratio.min(1.0);
    let zeros = 32.0;
    let half = zeros / cutoff;
    let beta = 8.6f64;
    let bessel_i0 = |v: f64| {
        let (mut sum, mut term, mut k) = (1.0f64, 1.0f64, 1.0f64);
        while term > 1e-12 * sum {
            term *= (v / (2.0 * k)).powi(2);
            sum += term;
            k += 1.0;
        }
        sum
    };
    let norm = bessel_i0(beta);
    let n_out = ((x.len() as f64) * ratio).round() as usize;
    (0..n_out)
        .map(|i| {
            let t = i as f64 / ratio;
            let lo = ((t - half).ceil() as i64).max(0);
            let hi = ((t + half).floor() as i64).min(x.len() as i64 - 1);
            let mut acc = 0.0f64;
            for j in lo..=hi {
                let d = t - j as f64;
                let arg = d * cutoff;
                let sinc = if arg.abs() < 1e-12 { 1.0 } else { (std::f64::consts::PI * arg).sin() / (std::f64::consts::PI * arg) };
                let r = d / half;
                let win = if r.abs() >= 1.0 { 0.0 } else { bessel_i0(beta * (1.0 - r * r).sqrt()) / norm };
                acc += f64::from(x[j as usize]) * sinc * win * cutoff;
            }
            acc as f32
        })
        .collect()
}

/// Parameters of the `semicausal_log_mel` processor (contract `audio` modality).
#[derive(Clone, Debug)]
pub struct MelParams {
    pub sample_rate: u32,
    pub frame_length: usize,
    pub hop_length: usize,
    pub fft_length: usize,
    pub mel_bins: usize,
    pub min_frequency: f64,
    pub max_frequency: f64,
    pub mel_floor: f64,
    pub pad_multiple: usize,
}

/// Log-mel frames `[frames][bins]` of one clip and how many frames carry audio.
pub struct Mel {
    pub values: Vec<f32>,
    pub frames: usize,
    pub valid_frames: usize,
}

fn hz_to_mel(f: f64) -> f64 {
    2595.0 * (1.0 + f / 700.0).log10()
}
fn mel_to_hz(m: f64) -> f64 {
    700.0 * (10f64.powf(m / 2595.0) - 1.0)
}

/// HTK triangular filters `[fft_bins][mel_bins]`, unnormalized (transformers `mel_filter_bank`).
pub fn mel_filters(p: &MelParams) -> Vec<f64> {
    let bins = p.fft_length / 2 + 1;
    let (lo, hi) = (hz_to_mel(p.min_frequency), hz_to_mel(p.max_frequency));
    let n = p.mel_bins;
    let filter_freqs: Vec<f64> = (0..n + 2).map(|i| mel_to_hz(lo + (hi - lo) * i as f64 / (n + 1) as f64)).collect();
    let nyquist = f64::from(p.sample_rate / 2);
    let fft_freqs: Vec<f64> = (0..bins).map(|i| nyquist * i as f64 / (bins - 1) as f64).collect();
    let mut out = vec![0f64; bins * n];
    for (b, &f) in fft_freqs.iter().enumerate() {
        for m in 0..n {
            let down = -(filter_freqs[m] - f) / (filter_freqs[m + 1] - filter_freqs[m]);
            let up = (filter_freqs[m + 2] - f) / (filter_freqs[m + 2] - filter_freqs[m + 1]);
            out[b * n + m] = down.min(up).max(0.0);
        }
    }
    out
}

/// Semicausal framing (`frame_length / 2` zeros ahead), periodic Hann window, |rfft|, HTK mel,
/// `ln(mel + floor)`; the clip is first padded to a multiple of `pad_multiple` samples, and a
/// frame is valid when its last sample is real audio.
pub fn log_mel(samples: &[f32], p: &MelParams) -> Mel {
    use rustfft::num_complex::Complex;
    let padded = samples.len().next_multiple_of(p.pad_multiple.max(1));
    let pad_left = p.frame_length / 2;
    let total = pad_left + padded;
    let size = p.frame_length + 1;
    let frames = if total >= size { (total - size) / p.hop_length + 1 } else { 0 };
    let window: Vec<f64> = (0..p.frame_length)
        .map(|i| {
            // np.hanning(frame_length + 1)[:-1]
            let m = p.frame_length as f64;
            0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / m).cos()
        })
        .map(|w| f64::from(w as f32))
        .collect();
    let filters = mel_filters(p);
    let bins = p.fft_length / 2 + 1;
    let fft = rustfft::FftPlanner::<f64>::new().plan_fft_forward(p.fft_length);
    let sample = |i: usize| -> f64 {
        if i < pad_left || i - pad_left >= samples.len() {
            0.0
        } else {
            f64::from(samples[i - pad_left])
        }
    };
    let mut values = vec![0f32; frames * p.mel_bins];
    let mut buf = vec![Complex::new(0f64, 0f64); p.fft_length];
    let mut mag = vec![0f64; bins];
    let mut valid_frames = 0;
    for f in 0..frames {
        let start = f * p.hop_length;
        buf.iter_mut().for_each(|c| *c = Complex::new(0.0, 0.0));
        for i in 0..p.frame_length {
            // float32 frames * float32 window, as numpy does before the float64 FFT.
            buf[i] = Complex::new(f64::from((sample(start + i) as f32) * (window[i] as f32)), 0.0);
        }
        fft.process(&mut buf);
        for b in 0..bins {
            mag[b] = buf[b].norm();
        }
        for m in 0..p.mel_bins {
            let mut s = 0f64;
            for b in 0..bins {
                s += mag[b] * filters[b * p.mel_bins + m];
            }
            values[f * p.mel_bins + m] = (s + p.mel_floor).ln() as f32;
        }
        let end = start + size - 1;
        if end >= pad_left && end - pad_left < samples.len() {
            valid_frames += 1;
        }
    }
    Mel { values, frames, valid_frames }
}

/// Raw-sample frames (encoder-free audio): `ceil(len / frame)` rows of `frame` samples, the last
/// zero-padded, every row valid. Carried as a `Mel` whose bins are the frame's samples.
pub fn waveform_frames(samples: &[f32], frame: usize) -> Mel {
    let frames = samples.len().div_ceil(frame);
    let mut values = samples.to_vec();
    values.resize(frames * frame, 0.0);
    Mel { values, frames, valid_frames: frames }
}

/// Soft tokens of `valid_frames` after `subsample` (a power of two) halvings of stride-2 convs.
pub fn audio_tokens(valid_frames: usize, subsample: usize) -> usize {
    let mut t = valid_frames;
    let mut s = subsample;
    while s > 1 {
        t = t.div_ceil(2);
        s /= 2;
    }
    t
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params() -> PatchParams {
        PatchParams { patch: 16, pool: 3, max_soft_tokens: 280, rescale: 1.0 / 255.0, normalize: false, mean: [0.0; 3], std: [1.0; 3] }
    }

    #[test]
    fn aspect_sizes_match_the_processor() {
        // Measured with transformers' Gemma4ImageProcessor.
        assert_eq!(aspect_size(480, 640, &params()).unwrap(), (672, 912));
        assert_eq!(aspect_size(1024, 768, &params()).unwrap(), (912, 672));
        assert_eq!(aspect_size(100, 333, &params()).unwrap(), (432, 1440));
        assert_eq!(aspect_size(64, 64, &params()).unwrap(), (768, 768));
    }

    #[test]
    fn identity_resize_and_patch_order() {
        let (w, h) = (96u32, 48u32);
        let data: Vec<u8> = (0..w * h * 3).map(|i| (i % 251) as u8).collect();
        assert_eq!(resize_bicubic(&data, w, h, w, h), data);
        let mut p = params();
        p.max_soft_tokens = 2; // 18 patches budget: 48x96 -> 3x6 patches of 16
        let pt = image_patches(&Rgb { width: w, height: h, data: data.clone() }, &p).unwrap();
        assert_eq!(pt.grid, (3, 6));
        assert_eq!(pt.soft_tokens, 2);
        // Patch (row 1, col 2), pixel (py 3, px 4), channel 1.
        let k = 6 + 2;
        let v = pt.values[k * 768 + (3 * 16 + 4) * 3 + 1];
        let src = ((16 + 3) * 96 + 32 + 4) * 3 + 1;
        assert!((v - f32::from(data[src]) / 255.0).abs() < 1e-7);
        assert_eq!(pt.positions[k], [2, 1]);
    }

    #[test]
    fn audio_token_counts_follow_the_conv_lengths() {
        let p = MelParams {
            sample_rate: 16_000,
            frame_length: 320,
            hop_length: 160,
            fft_length: 512,
            mel_bins: 128,
            min_frequency: 0.0,
            max_frequency: 8000.0,
            mel_floor: 1e-3,
            pad_multiple: 128,
        };
        for n in [16_000usize, 23_456, 480_000] {
            let mel = log_mel(&vec![0.1f32; n], &p);
            // Processor: (n + 160 - 321) // 160 + 1 mel frames, two ceil-halvings.
            let frames = (n + 160 - 321) / 160 + 1;
            assert_eq!(mel.valid_frames, frames, "{n}");
            assert_eq!(audio_tokens(frames, 4), frames.div_ceil(2).div_ceil(2));
        }
    }
}
