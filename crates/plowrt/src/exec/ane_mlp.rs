use crate::asset::Checkpoint;
use crate::exec::ane::{f32_to_f16, half_to_f32, Layer, NetSpec};

#[derive(Clone)]
pub struct Matrix {
    pub n: usize,
    pub k: usize,
    pub encoding: u32,
    pub data: Vec<u8>,
    pub scales: Vec<u8>,
}

impl Matrix {
    pub fn load(ck: &Checkpoint, name: &str, n: usize, k: usize, encoding: u32) -> Self {
        let prefix = match encoding {
            0 => "",
            1 => "fp8/",
            2 => "mxfp4/",
            _ => panic!("encoding"),
        };
        let name = format!("{prefix}{name}");
        let data = ck.tensor(&name).expect(&name).to_vec();
        let scales = if encoding == 0 {
            vec![]
        } else {
            ck.tensor(&format!("{name}_scale"))
                .expect("weight scale")
                .to_vec()
        };
        let bytes = match encoding {
            0 => n * k * 2,
            1 => n * k,
            _ => n * k / 2,
        };
        assert_eq!(data.len(), bytes);
        assert_eq!(
            scales.len(),
            match encoding {
                0 => 0,
                1 => n * 4,
                _ => n * k.div_ceil(32),
            }
        );
        Self {
            n,
            k,
            encoding,
            data,
            scales,
        }
    }

    pub fn slice(&self, rows: std::ops::Range<usize>, cols: std::ops::Range<usize>) -> Self {
        assert!(rows.start <= rows.end && rows.end <= self.n);
        assert!(cols.start <= cols.end && cols.end <= self.k);
        if self.encoding == 2 {
            assert_eq!(cols.start % 32, 0);
            assert_eq!(cols.len() % 32, 0);
        }
        let byte = |k: usize| match self.encoding {
            0 => k * 2,
            1 => k,
            _ => k / 2,
        };
        let mut data = Vec::with_capacity(rows.len() * byte(cols.len()));
        let mut scales = Vec::new();
        for r in rows.clone() {
            data.extend_from_slice(
                &self.data[r * byte(self.k) + byte(cols.start)..r * byte(self.k) + byte(cols.end)],
            );
            match self.encoding {
                1 => scales.extend_from_slice(&self.scales[r * 4..r * 4 + 4]),
                2 => scales.extend_from_slice(
                    &self.scales[r * self.k.div_ceil(32) + cols.start / 32
                        ..r * self.k.div_ceil(32) + cols.end / 32],
                ),
                _ => {}
            }
        }
        Self {
            n: rows.len(),
            k: cols.len(),
            encoding: self.encoding,
            data,
            scales,
        }
    }

    pub fn value(&self, n: usize, k: usize) -> f32 {
        let ix = n * self.k + k;
        match self.encoding {
            0 => bf(u16::from_le_bytes(
                self.data[2 * ix..2 * ix + 2].try_into().unwrap(),
            )),
            1 => {
                let b = self.data[ix];
                let sign = if b & 128 == 0 { 1.0 } else { -1.0 };
                let exponent = (b >> 3) & 15;
                let mantissa = (b & 7) as f32;
                assert!(exponent != 15 || mantissa != 7.0, "FP8 NaN weight");
                sign * if exponent == 0 {
                    mantissa / 512.0
                } else {
                    (1.0 + mantissa / 8.0) * 2f32.powi(exponent as i32 - 7)
                } * f32::from_le_bytes(self.scales[4 * n..4 * n + 4].try_into().unwrap())
            }
            _ => {
                const LUT: [f32; 16] = [
                    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0,
                    -4.0, -6.0,
                ];
                let b = self.data[ix / 2];
                LUT[(if k % 2 == 0 { b & 15 } else { b >> 4 }) as usize]
                    * f32::from_bits((self.scales[n * self.k.div_ceil(32) + k / 32] as u32) << 23)
            }
        }
    }

    pub fn half(&self) -> Vec<u16> {
        (0..self.n * self.k)
            .map(|i| {
                let x = self.value(i / self.k, i % self.k);
                let h = f32_to_f16(x);
                assert!(
                    x.is_finite() && half_to_f32(h).is_finite(),
                    "FP16 weight range"
                );
                h
            })
            .collect()
    }
}

pub fn bf(x: u16) -> f32 {
    f32::from_bits((x as u32) << 16)
}
pub fn to_bf(x: f32) -> u16 {
    let b = x.to_bits();
    (b.wrapping_add(0x7fff + ((b >> 16) & 1)) >> 16) as u16
}

pub fn mlp_spec(rows: usize, gate: &Matrix, up: &Matrix, down: &Matrix, deep_k: usize) -> NetSpec {
    let (h, c) = (gate.k, gate.n);
    assert_eq!((up.n, up.k, down.n, down.k), (c, h, h, c));
    let mut spec = NetSpec {
        inputs: vec![("x".into(), h)],
        outputs: vec![("y".into(), h)],
        t_enum: vec![rows],
        flex_outputs: false,
        range: false,
        out_range: false,
        w8: false,
        layers: vec![
            Layer::InnerProduct {
                input: "x".into(),
                output: "g".into(),
                k: h,
                n: c,
                w_f16: gate.half(),
            },
            Layer::InnerProduct {
                input: "x".into(),
                output: "u".into(),
                k: h,
                n: c,
                w_f16: up.half(),
            },
            Layer::Sigmoid {
                input: "g".into(),
                output: "s".into(),
            },
            Layer::Mul {
                a: "g".into(),
                b: "s".into(),
                output: "a".into(),
            },
            Layer::Mul {
                a: "a".into(),
                b: "u".into(),
                output: "z".into(),
            },
            Layer::InnerProduct {
                input: "z".into(),
                output: "y".into(),
                k: c,
                n: h,
                w_f16: down.half(),
            },
        ],
    };
    if deep_k > 0 && deep_k < c {
        assert_eq!(deep_k % 128, 0);
        spec.layers.pop();
        let mut sum = String::new();
        for (part, start) in (0..c).step_by(deep_k).enumerate() {
            let end = (start + deep_k).min(c);
            let slice = format!("z{part}");
            let product = format!("p{part}");
            spec.layers.push(Layer::SliceCols {
                input: "z".into(),
                output: slice.clone(),
                start,
                end,
            });
            spec.layers.push(Layer::InnerProduct {
                input: slice,
                output: product.clone(),
                k: end - start,
                n: h,
                w_f16: down.slice(0..h, start..end).half(),
            });
            if part == 0 {
                sum = product;
            } else {
                let output = if end == c {
                    "y".into()
                } else {
                    format!("sum{part}")
                };
                spec.layers.push(Layer::Add {
                    a: sum,
                    b: product,
                    output: output.clone(),
                });
                sum = output;
            }
        }
    }
    spec
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn packed_column_and_row_coverage() {
        for encoding in 0..=2 {
            let (n, k, cut) = (3, 384, 128);
            let bytes = match encoding {
                0 => n * k * 2,
                1 => n * k,
                _ => n * k / 2,
            };
            let data = (0..bytes).map(|i| (i % 119) as u8).collect();
            let scales = match encoding {
                0 => vec![],
                1 => [0.25f32, 0.5, 0.75]
                    .iter()
                    .flat_map(|x| x.to_le_bytes())
                    .collect(),
                _ => (0..n * k / 32).map(|i| 120 + (i % 8) as u8).collect(),
            };
            let w = Matrix {
                n,
                k,
                encoding,
                data,
                scales,
            };
            let a = w.slice(0..n, 0..cut);
            let b = w.slice(0..n, cut..k);
            for r in 0..n {
                for c in 0..k {
                    assert_eq!(
                        w.value(r, c).to_bits(),
                        if c < cut {
                            a.value(r, c)
                        } else {
                            b.value(r, c - cut)
                        }
                        .to_bits()
                    );
                }
            }
            let rows = w.slice(1..3, 0..k);
            for r in 0..2 {
                for c in 0..k {
                    assert_eq!(rows.value(r, c).to_bits(), w.value(r + 1, c).to_bits());
                }
            }
            assert!(w.slice(0..0, 0..k).data.is_empty());
            assert!(w.slice(0..n, k..k).data.is_empty());
        }
    }
    #[test]
    #[should_panic]
    fn rejects_split_inside_mxfp4_scale_group() {
        Matrix {
            n: 1,
            k: 64,
            encoding: 2,
            data: vec![0; 32],
            scales: vec![127; 2],
        }
        .slice(0..1, 1..33);
    }

    #[test]
    fn deep_k_graph_covers_down_columns_and_reduces_in_graph() {
        let matrix = |n, k| Matrix {
            n,
            k,
            encoding: 0,
            data: (0..n * k)
                .flat_map(|i| to_bf((i % 17) as f32 / 16.0).to_le_bytes())
                .collect(),
            scales: vec![],
        };
        let gate = matrix(640, 128);
        let down = matrix(128, 640);
        for chunk in [0, 128, 256, 512, 1024] {
            let spec = mlp_spec(91, &gate, &gate, &down, chunk);
            let mut slices = Vec::new();
            let mut products = Vec::new();
            let mut sums = 0;
            for layer in &spec.layers[5..] {
                match layer {
                    Layer::SliceCols { start, end, .. } => slices.push((*start, *end)),
                    Layer::InnerProduct { k, n, w_f16, .. } => {
                        assert_eq!(*n, 128);
                        products.push((*k, w_f16));
                    }
                    Layer::Add { .. } => sums += 1,
                    _ => panic!("unexpected down layer"),
                }
            }
            let mut offset = 0;
            for (j, (k, w)) in products.iter().enumerate() {
                if !slices.is_empty() {
                    assert_eq!(slices[j], (offset, offset + k));
                }
                assert_eq!(**w, down.slice(0..128, offset..offset + k).half());
                offset += k;
            }
            assert_eq!(offset, 640);
            assert_eq!(sums, products.len() - 1);
            assert_eq!(spec.outputs, vec![("y".into(), 128)]);
        }
    }
}
