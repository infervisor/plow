use packet::dev::{DevInst64, DevOp, ROPE_PAIR_HALF, TENSOR_NONE16};

pub const SYMBOLS: [&str; 6] = [
    "plow_decode_bf16_abi",
    "plow_decode_gf256",
    "plow_decode_gf512",
    "plow_decode_staging_bytes",
    "plow_gemv_mm_cap",
    "plow_arena_bytes",
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DenseBf16(pub [u32; 6]);

impl DenseBf16 {
    pub fn from_image(image: &[u8]) -> Result<Self, String> {
        let mut fields = [0; 6];
        for (field, name) in fields.iter_mut().zip(SYMBOLS) {
            *field = crate::cubin::global_u32(image, name)
                .ok_or_else(|| format!("missing initialized decode capability {name}"))?;
        }
        let value = Self(fields);
        value.validate()?;
        Ok(value)
    }
    pub fn program(
        &self,
        packet: &crate::program::Packet<'_>,
        index: usize,
        splitk: Option<u32>,
    ) -> Result<(), String> {
        self.validate()?;
        let p = packet
            .programs
            .get(index)
            .ok_or("bound program out of range")?;
        let needs_splitk = p.insts.iter().any(|d| {
            matches!(
                DevOp::from_u16(d.op),
                Some(DevOp::ZeroF32 | DevOp::GemmSplitK | DevOp::CastF32Bf16)
            )
        });
        if needs_splitk {
            crate::splitk::capability(splitk, Some(256), self.0[5])?;
            crate::splitk::validate(packet)?.ok_or("missing splitK packet proof")?;
        }
        for d in p.insts {
            if matches!(
                DevOp::from_u16(d.op),
                Some(DevOp::ZeroF32 | DevOp::GemmSplitK | DevOp::CastF32Bf16)
            ) {
                continue;
            }
            self.instruction(d, p.rows)?;
        }
        Ok(())
    }
    pub fn validate(&self) -> Result<(), String> {
        let [abi, gf256, gf512, staging, mm, arena] = self.0;
        if abi != 1
            || !matches!(gf256, 2 | 6)
            || !matches!(gf512, 1 | 2 | 4 | 8 | 16)
            || !matches!(mm, 8 | 16 | 32)
            || staging > arena
            || arena < 64
        {
            return Err("unsupported compiled dense BF16 decode contract".into());
        }
        Ok(())
    }
    // This contract admits only the audited dense BF16 arms. Tensor precision is encoded
    // by the opcode; raw tensor declarations have byte extents, not dtype tags.
    pub fn instruction(&self, d: &DevInst64, rows: u32) -> Result<(), String> {
        self.validate()?;
        let op = DevOp::from_u16(d.op).ok_or("unknown bound opcode")?;
        let i = d.i;
        if rows == 0 || d.blocks == 0 {
            return Err("empty bound work".into());
        }
        let batch = |n: u32| n > 0 && n <= rows;
        let gemv = || batch(i[0]) && i[1] > 0 && i[2] > 0 && i[2] % 8 == 0;
        let staged = || i[0] != 1 || u64::from(i[2]) * 2 <= u64::from(self.0[5]);
        let accepted = match op {
            DevOp::Nop => true,
            DevOp::Gemv => gemv() && i[3] == 0,
            DevOp::GemvQkv => {
                gemv()
                    && staged()
                    && i[1]
                        .checked_add(i[3])
                        .and_then(|n| n.checked_add(i[4]))
                        .is_some()
            }
            DevOp::GemvGlu => gemv() && staged() && i[5] <= 1,
            DevOp::RmsNorm => {
                batch(i[0]) && i[1] > 0 && d.t[3] == TENSOR_NONE16 && d.t[4] == TENSOR_NONE16
            }
            DevOp::AddNorm | DevOp::NormResidual | DevOp::NormResidualNorm => {
                batch(i[0]) && i[1] > 0
            }
            DevOp::Embed => batch(i[0]) && i[1] > 0,
            DevOp::Residual => i[0] > 0,
            DevOp::SoftCap => {
                i[0] > 0 && f32::from_bits(d.fj[0]).is_finite() && f32::from_bits(d.fj[0]) > 0.0
            }
            DevOp::Glu => i[0] > 0 && i[1] <= 1,
            DevOp::HeadNormRope => {
                batch(i[0])
                    && i[1] > 0
                    && matches!((i[2], i[5]), (64, ROPE_PAIR_HALF) | (256 | 512, 0))
            }
            DevOp::FlashDecode => {
                let gqa = i[1].checked_div(i[2]).unwrap_or(0);
                let gf = match i[6] {
                    // HD64 is emitted only in a packet-paired object. Its manifest-derived
                    // specialization instantiates the packet's exact GQA group.
                    64 => gqa,
                    256 if gqa % self.0[1] == 0 => self.0[1],
                    256 => 2,
                    512 => self.0[2],
                    _ => return Err("head dimension absent from dense BF16 contract".into()),
                };
                batch(i[0])
                    && i[1] > 0
                    && i[2] > 0
                    && i[1] % i[2] == 0
                    && gqa > 0
                    && gqa % gf == 0
                    && i[3] > 0
                    && i[5] > 0
            }
            DevOp::FlashMerge => {
                batch(i[0]) && i[1] > 0 && i[2] > 0 && matches!(i[3], 64 | 256 | 512)
            }
            DevOp::Argmax | DevOp::ArgmaxFin => i[0] > 0 && batch(i[1].max(1)),
            _ => false,
        };
        if accepted {
            Ok(())
        } else {
            Err(format!("unsupported bound dense BF16 arm {op:?}: {i:?}"))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn capability_elf() -> Vec<u8> {
        crate::cubin::synthetic_elf(
            "_Z12interp_sm90a11PlowProgram",
            &[
                ("plow_decode_bf16_abi", 1),
                ("plow_decode_gf256", 2),
                ("plow_decode_gf512", 8),
                ("plow_decode_staging_bytes", 16384),
                ("plow_gemv_mm_cap", 16),
                ("plow_arena_bytes", 16384),
            ],
            90,
        )
    }
    fn caps() -> DenseBf16 {
        DenseBf16([1, 6, 8, 16384, 16, 16384])
    }
    fn flash() -> DevInst64 {
        DevInst64 {
            op: DevOp::FlashDecode as u16,
            blocks: 1,
            i: [4, 16, 2, 1024, 0, 2, 512, u32::MAX],
            ..Default::default()
        }
    }
    #[test]
    fn reads_actual_initialized_globals_and_rejects_malformed_images() {
        let image = capability_elf();
        assert_eq!(
            DenseBf16::from_image(&image).unwrap().0,
            [1, 2, 8, 16384, 16, 16384]
        );
        for len in 0..image.len() {
            assert!(DenseBf16::from_image(&image[..len]).is_err());
        }
        let mut wrong = image.clone();
        let end = wrong.len();
        wrong[end - 24..end - 20].fill(0);
        assert!(DenseBf16::from_image(&wrong).is_err());
        for (offset, value) in [(64 + 3 * 64 + 4, 8u32), (320 + 24 + 4, 0x12u32)] {
            let mut wrong = image.clone();
            wrong[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
            assert!(DenseBf16::from_image(&wrong).is_err());
        }
        let mut duplicate = image.clone();
        duplicate[368..372].copy_from_slice(&image[344..348]);
        assert!(DenseBf16::from_image(&duplicate).is_err());
        let mut overflow = image.clone();
        overflow[352..360].copy_from_slice(&u64::MAX.to_le_bytes());
        assert!(DenseBf16::from_image(&overflow).is_err());
        let mut unknown = image.clone();
        unknown[0] = 0;
        assert!(DenseBf16::from_image(&unknown).is_err());
    }
    #[test]
    fn requires_live_compiled_contract() {
        for field in [0, 1, 2, 4, 5] {
            let mut c = caps();
            c.0[field] = 0;
            assert!(c.validate().is_err());
        }
        let mut c = caps();
        c.0[3] = 16385;
        assert!(c.validate().is_err());
    }
    #[test]
    fn rejects_absent_precision_heads_and_gqa() {
        let c = caps();
        let d = flash();
        assert!(c.instruction(&d, 4).is_ok());
        for op in [
            DevOp::FlashDecodeFp8,
            DevOp::GemvFp8,
            DevOp::GemmFp8,
            DevOp::QwenGdnStep,
        ] {
            let mut d = d;
            d.op = op as u16;
            assert!(c.instruction(&d, 4).is_err());
        }
        for hd in [128, 1024] {
            let mut d = d;
            d.i[6] = hd;
            assert!(c.instruction(&d, 4).is_err());
        }
        for (nh, nkv) in [(0, 0), (16, 0), (15, 2), (12, 2)] {
            let mut d = d;
            d.i[1] = nh;
            d.i[2] = nkv;
            assert!(c.instruction(&d, 4).is_err());
        }
        let mut d = d;
        d.i[6] = 256;
        d.i[1] = 12;
        assert!(c.instruction(&d, 4).is_ok());
        d.i[1] = 16;
        assert!(c.instruction(&d, 4).is_ok()); // Compiled GF2 fallback.
    }
    #[test]
    fn accepts_packet_paired_hd64_attention_and_half_split_rope() {
        let c = caps();
        let mut d = flash();
        d.i = [4, 64, 8, 1024, 128, 2, 64, 1023];
        assert!(c.instruction(&d, 4).is_ok());
        d.op = DevOp::FlashMerge as u16;
        d.i = [4, 64, 2, 64, 0, 0, 0, 0];
        assert!(c.instruction(&d, 4).is_ok());
        d.op = DevOp::HeadNormRope as u16;
        d.i = [4, 64, 64, 0, 1, ROPE_PAIR_HALF, 4, 0];
        assert!(c.instruction(&d, 4).is_ok());
        d.i[5] = 0;
        assert!(c.instruction(&d, 4).is_err());
    }
    #[test]
    fn rejects_fused_quant_and_undersized_m1_staging() {
        let c = caps();
        let mut d = DevInst64 {
            op: DevOp::RmsNorm as u16,
            blocks: 1,
            t: [TENSOR_NONE16; 8],
            i: [1, 3840, 0, 0, 0, 0, 0, 0],
            ..Default::default()
        };
        assert!(c.instruction(&d, 1).is_ok());
        d.t[3] = 8;
        assert!(c.instruction(&d, 1).is_err());
        d.op = DevOp::GemvGlu as u16;
        d.i = [1, 3840, 16384, 0, 0, 1, 0, 0];
        assert!(c.instruction(&d, 1).is_err());
        d.i[0] = 4;
        assert!(c.instruction(&d, 4).is_ok());
        d.i[2] = 16383;
        assert!(c.instruction(&d, 4).is_err());
    }
}
