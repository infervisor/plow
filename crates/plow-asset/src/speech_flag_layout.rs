//! SPEECH FLAG LAYOUT: which meaning a packet's speech-op flag bits carry.
//!
//! The speech ops' flag words gained fields after the first released layout (layout 0), and one
//! bit changed meaning between branches before it settled. A packet whose instructions set any
//! bit assigned since layout 0 ([`protected`]) declares the layout they were emitted for in
//! `abi.json` (`speech_flag_layout`); the speech interpreter object exports the layout it was
//! compiled for ([`OBJECT_SYMBOL`]). A loader runs the instructions only when the two agree.
//! A packet that sets none of those bits means the same under every layout: it stays undeclared
//! (and byte-identical to an emit from before this module) and loads on any object.
//!
//! Reassigning a speech-op flag bit bumps [`LAYOUT`] and extends the masks.
use packet::dev::{DevInst64, DevOp, TENSOR_NONE16};

/// The layout this tree emits and compiles: conv flags bit 2 SnakeBeta output, bit 3 ragged
/// wgmma tiles, 21..29 pinned arms, 30 / 31 split-image input / output; AttentionF32 `i6` bit 6
/// query view, bit 7 split-image output; LayerNormF32 `i2` bit 2 split-image output; GemvF32Out
/// `i7` bit 12 row view. Layout 2 adds LayerNormF32 bit 3 (f16 output) and GemvF32Out
/// bits 13 / 14 (f16 input / output). Layout 3 adds slot-indexed ring attention and tail modes.
/// Layout 4 adds single-product bf16 wgmma convs (Conv1dF32 / ConvTranspose1dF32 `i6` bit 31,
/// above the dilation / output padding). Layout 5 adds the fp16 short attention (AttentionF32 `j1`
/// bit 31, above `v_col0`). Every Hopper speech object exports 5; ring KV stays a capability.
pub const LAYOUT: u32 = 5;
pub const DEFAULT_LAYOUT: u32 = 2;
/// `extern "C" __device__ unsigned` exported by every speech interpreter object.
pub const OBJECT_SYMBOL: &str = "plow_speech_flag_layout_speech";

/// `extern "C" __device__ unsigned` of a speech object compiled with a conv arm subset.
pub const CONV_ARMS_SYMBOL: &str = "plow_speech_conv_arms_speech";
/// `extern "C" __device__ unsigned` of a speech object compiled with a bf16 conv variant subset.
pub const CONV_BF16_VARIANTS_SYMBOL: &str = "plow_speech_conv_bf16_variants_speech";

/// Conv arms of `PLOW_SPEECH_CONV_ARMS` (`op_speech_f32.cuh` `SPA_*`). A speech object compiled
/// with a subset exports it as [`CONV_ARMS_SYMBOL`]; the loader refuses a packet whose convs
/// reach an arm outside it (the object would trap).
pub mod conv_arm {
    pub const ROWS: u64 = 1 << 0;
    pub const DEPTHWISE: u64 = 1 << 1;
    pub const STAGED: u64 = 1 << 2;
    pub const DIRECT: u64 = 1 << 3;
    pub const WIDE: u64 = 1 << 4;
    pub const TC: u64 = 1 << 5;
    pub const FFMA: u64 = 1 << 6;
    pub const WG: u64 = 1 << 7;
    pub const WG_CT: u64 = 1 << 8;
    pub const ALL: u64 = (1 << 9) - 1;
}

/// The conv arms one Conv1dF32 / ConvTranspose1dF32 instruction can reach, mirroring the static
/// half of `sp_wg_conv1d` and `d_conv1d_f32` (`op_speech_f32.cuh`): where the device choice also depends on operand
/// alignment (the emitter aligns every tensor to 16 bytes) or the launch width, every outcome
/// is included. The split-image and bf16 taps convs have their own defines and fail closed.
pub fn conv_arms(op: u16, i: &[u32; 8], pads: u32, f: u32) -> u64 {
    use conv_arm::*;
    let transpose = op == DevOp::ConvTranspose1dF32 as u16;
    if !transpose && op != DevOp::Conv1dF32 as u16 {
        return 0;
    }
    let bit = |n: u32| (f >> n) & 1 == 1;
    if !transpose && (bit(30) || (bit(17) && !bit(14))) {
        return 0;
    }
    let (cin, cout, kernel, stride, groups) = (i[2], i[3], i[4], i[5], i[7].max(1));
    let (cg, ng) = (cin / groups, cout / groups);
    let (mode, pre, post) = (
        if transpose { 0 } else { f & 3 },
        (f >> 4) & 15,
        (f >> 8) & 15,
    );
    let (wf16, tapmajor, ragged) = (bit(12), !transpose && bit(15), bit(3));
    let pin = (f >> 21) & 0x1FF;
    let pinned = pin & 1 != 0;
    let pre_ok = pre == 0 || pre == 9;
    let mut m = 0;
    if transpose {
        if bit(17) && bit(18) {
            m |= WG_CT;
            let ct = bit(15)
                && kernel == 2 * stride
                && groups == 1
                && i[6] & 0x7FFF_FFFF == 0
                && pre == 0
                && post == 0
                && !wf16
                && cin % 64 == 0
                && (cout % 64 == 0 || (ragged && cout % 4 == 0 && cout >= 64));
            if ct {
                return m;
            }
        }
    } else if bit(17) && !bit(16) {
        m |= WG;
        let pw = kernel == 1 && stride == 1 && pads == 0 && cg % 4 == 0 && pre == 0;
        let taps = |channel_tiled: bool| {
            kernel > 1
                && (tapmajor || channel_tiled)
                && stride == 1
                && mode == 0
                && groups == 1
                && (cin % 64 == 0 || (ragged && !channel_tiled && cin % 4 == 0))
                && pre_ok
        };
        let eligible = |channel_tiled: bool| {
            (pw || taps(channel_tiled)) && (!bit(2) || taps(channel_tiled)) && cout % 4 == 0
        };
        let wide_static =
            bit(14) && groups == 1 && cin % 4 == 0 && !wf16 && pre_ok && (kernel == 1 || tapmajor);
        let takes = if !tapmajor && kernel > 1 {
            eligible(true) && pre == 0 && post == 0 && !wf16 && bit(14) && bit(18) && !bit(20)
        } else {
            let grouped = groups > 1
                && kernel == 1
                && post == 0
                && ng % 4 == 0
                && cin % 4 == 0
                && !wf16
                && bit(14);
            eligible(false) && (grouped || (bit(18) && wide_static))
        };
        if takes {
            return m;
        }
    }
    if !transpose && bit(16) && groups == 1 {
        m |= ROWS;
        if cout <= 4 {
            return m;
        }
    }
    if !transpose && cg == 1 && ng == 1 {
        m |= DEPTHWISE;
    }
    if ng < 16 || cg * kernel < 16 {
        if !transpose && pinned && groups == 1 && cout <= 4 {
            m |= STAGED;
        }
        return m | DIRECT;
    }
    if !transpose
        && bit(14)
        && groups == 1
        && cin % 4 == 0
        && !wf16
        && pre_ok
        && (kernel == 1 || tapmajor)
    {
        m |= WIDE;
        if mode == 0 {
            return m;
        }
    }
    m | if !bit(13) {
        FFMA
    } else if pinned {
        if pin & 2 != 0 {
            FFMA
        } else {
            TC
        }
    } else if !transpose && pre == 0 {
        TC
    } else {
        TC | FFMA
    }
}

/// The bf16 taps conv's plan kind (`spb_plan`, `op_speech_f32.cuh`): 1 = cout 96 (128 rows x 96
/// columns), 2 = 128 rows x 64 columns, 3 = 64 rows x 128 columns over plane groups; `None` = the
/// shape does not fit the arm's arena.
pub fn conv_bf16_kind(cin: u32, cout: u32, kernel: u32, dilation: u32) -> Option<u32> {
    // SP_LAUNCH_ARENA_FLOATS under PLOW_SPEECH_CONV_BF16, less the 1024-byte alignment slack.
    const ARENA: u64 = 33280 * 4 - 1024;
    let (cin, cout, k, dil) = (
        u64::from(cin),
        u64::from(cout),
        u64::from(kernel),
        u64::from(dilation),
    );
    if kernel < 2 || dil == 0 || cin % 16 != 0 || cin > 1024 || cout % 8 != 0 {
        return None;
    }
    let rpad = |bm: u64| (bm + (k - 1) * dil + 7) & !7;
    let fits = |bm: u64, nu: u64, pg: u64| pg * rpad(bm) * 128 + 2 * nu * 128 + 3 * nu * 4 <= ARENA;
    let planes = cin.div_ceil(64);
    if cout == 96 {
        return fits(128, 96, planes).then_some(1);
    }
    if cin % 64 != 0 || cout % 64 != 0 {
        return None;
    }
    if fits(128, 64, planes) {
        return Some(2);
    }
    (1..=planes)
        .any(|pg| planes % pg == 0 && fits(64, 128, pg))
        .then_some(3)
}

/// The `PLOW_SPEECH_CONV_BF16_VARIANTS` bit of a bf16 taps conv (Conv1dF32 flags bit 17 without
/// bit 14): `4 * (kind - 1) + pre + 2 * stream`, pre = an input SnakeBeta (bits 19 and 2), stream
/// = bit 22, plus 12 for chunk-canonical K (bits 24..29 above 1). 0 for any other instruction or a
/// shape the arm does not take (the object traps).
pub fn conv_bf16_variants(op: u16, i: &[u32; 8], f: u32) -> u64 {
    if op != DevOp::Conv1dF32 as u16 || (f >> 17) & 1 == 0 || (f >> 14) & 1 == 1 {
        return 0;
    }
    let pre = (f >> 19) & 1 == 1 && (f >> 2) & 1 == 1;
    let stream = (f >> 22) & 1 == 1;
    let chunked = conv_bf16_chunked(op, f);
    conv_bf16_kind(i[2], i[3], i[4], i[6]).map_or(0, |kind| {
        1 << (4 * (kind - 1) + u32::from(pre) + 2 * u32::from(stream) + 12 * u32::from(chunked))
    })
}

/// A bf16 taps conv with chunk-canonical K (Conv1dF32 flags bit 17 without bit 14, bits 24..29
/// above 1): its scratch handle is `fj[0] & 0x7FFF_FFFF`, and it needs speech capability bit 4.
pub fn conv_bf16_chunked(op: u16, f: u32) -> bool {
    op == DevOp::Conv1dF32 as u16 && (f >> 17) & 1 == 1 && (f >> 14) & 1 == 0 && (f >> 24) & 63 > 1
}

/// A conv on a wgmma arm (Conv1dF32 / ConvTranspose1dF32 flags bit 17): what the conv-only speech
/// object (`_speechconv`) runs.
pub fn conv_object_inst(op: u16, f: u32) -> bool {
    (op == DevOp::Conv1dF32 as u16 || op == DevOp::ConvTranspose1dF32 as u16) && (f >> 17) & 1 == 1
}

/// Per global-queue segment (`seg_ofs` windows over `stream`): the instruction when every entry is
/// the same [`conv_object_inst`] (`conv(inst)`) with no counter obligation (a raw segment: no
/// waits, successors or cross-segment counter), which the conv object's kernel entry then runs
/// alone. A conv the interpreter must signal stays on the interpreter.
pub fn conv_object_segments(
    seg_ofs: &[u32],
    stream: &[packet::dev::StreamEnt],
    conv: impl Fn(u32) -> bool,
) -> Vec<Option<u32>> {
    seg_ofs
        .windows(2)
        .map(|w| {
            let entries = &stream[w[0] as usize..w[1] as usize];
            let first = entries.first()?.inst;
            (entries.iter().all(|e| {
                e.inst == first
                    && e.wait_len == 0
                    && e.succ_len == 0
                    && e.flags & packet::dev::SE_XCTR == 0
            }) && conv(first))
            .then_some(first)
        })
        .collect()
}

/// Conv1dF32 / ConvTranspose1dF32 flags (`j1`, wire `fj[2]`) assigned since layout 0.
pub const CONV_FLAGS: u32 = 0xffe0_000c;
/// AttentionF32 `i6` bits assigned since layout 0.
pub const ATTENTION_I6: u32 = 0xc0;
/// LayerNormF32 `i2` bits assigned since layout 0.
pub const LAYER_NORM_I2: u32 = 0xc;
/// GemvF32Out `i7` bits assigned since layout 0.
pub const GEMV_F32_OUT_I7: u32 = 0x7000;

pub fn is_ring_attention(flags: u32) -> bool {
    flags & 0xff08 == 0x108
}

/// A Conv1dF32 / ConvTranspose1dF32 whose wgmma products are one bf16 product per k16.
pub fn single_product(op: u16, i: &[u32; 8]) -> bool {
    matches!(
        DevOp::from_u16(op),
        Some(DevOp::Conv1dF32 | DevOp::ConvTranspose1dF32)
    ) && i[6] >> 31 != 0
}

/// An AttentionF32 on fp16 Q / K / V and output rows (the short body only).
pub fn half_attention(op: u16, j1: u32) -> bool {
    op == DevOp::AttentionF32 as u16 && j1 >> 31 != 0
}

pub fn ring_kv(op: u16, i: &[u32; 8]) -> bool {
    match DevOp::from_u16(op) {
        Some(DevOp::AttentionF32) => is_ring_attention(i[6]),
        Some(DevOp::StreamTailF32) => matches!(i[7], 2 | 3),
        _ => false,
    }
}

/// Whether an instruction (`j1` = `DevInst::j[1]` = `DevInst64::fj[2]`) sets a flag bit that
/// layout 0 left clear.
pub fn protected(op: u16, j1: u32, i: &[u32; 8]) -> bool {
    match DevOp::from_u16(op) {
        Some(DevOp::Conv1dF32 | DevOp::ConvTranspose1dF32) => {
            j1 & CONV_FLAGS != 0 || single_product(op, i)
        }
        Some(DevOp::AttentionF32) => {
            i[6] & ATTENTION_I6 != 0 || ring_kv(op, i) || half_attention(op, j1)
        }
        Some(DevOp::StreamTailF32) => ring_kv(op, i),
        Some(DevOp::LayerNormF32) => i[2] & LAYER_NORM_I2 != 0,
        Some(DevOp::GemvF32Out) => i[7] & GEMV_F32_OUT_I7 != 0,
        _ => false,
    }
}

/// The opcode of the first instruction of `insts` (`(op, j1, i)`) that sets a protected bit.
pub fn first_protected(insts: impl IntoIterator<Item = (u16, u32, [u32; 8])>) -> Option<u16> {
    insts
        .into_iter()
        .find(|(op, j1, i)| protected(*op, *j1, i))
        .map(|(op, ..)| op)
}

pub fn required_layout(insts: impl IntoIterator<Item = (u16, u32, [u32; 8])>) -> Option<u32> {
    insts
        .into_iter()
        .filter_map(|(op, j1, i)| {
            protected(op, j1, &i).then(|| {
                if half_attention(op, j1) {
                    5
                } else if single_product(op, &i) {
                    4
                } else if ring_kv(op, &i) {
                    3
                } else if (op == DevOp::LayerNormF32 as u16 && i[2] & 8 != 0)
                    || (op == DevOp::GemvF32Out as u16 && i[7] & 0x6000 != 0)
                {
                    2
                } else {
                    1
                }
            })
        })
        .max()
}

/// Whether a metadata section holds auxiliary programs (`aux_program` wire form).
pub fn is_aux_programs(section: &[u8]) -> bool {
    section.starts_with(crate::aux_program::MAGIC)
}

/// The instructions of an auxiliary-program section, for the scan only: tensor handles are not
/// checked against the packet (each consumer parses its own section with its tensor count).
pub fn aux_insts(section: &[u8], n_cu: u32) -> Result<Vec<DevInst64>, String> {
    let parsed = crate::aux_program::parse(section, n_cu, TENSOR_NONE16 as usize)?;
    Ok(parsed.programs.into_iter().flat_map(|p| p.insts).collect())
}

/// Refuses auxiliary programs bound for an object that is not a speech object (the decode
/// objects read no flag bit assigned since layout 0) when one of them sets a protected bit.
pub fn refuse_on_decode_objects<'a>(
    insts: impl IntoIterator<Item = &'a DevInst64>,
) -> Result<(), String> {
    match first_protected(insts.into_iter().map(|d| (d.op, d.fj[2], d.i))) {
        None => Ok(()),
        Some(op) => Err(format!(
            "opcode {op} sets a speech flag bit the decode objects do not read"
        )),
    }
}

/// The baseline object layout; ring programs use [`required_layout`].
pub fn declare(used: Option<u16>) -> Option<u32> {
    used.map(|_| DEFAULT_LAYOUT)
}

/// Whether the instructions an object runs, whose first protected use is `used`, from a packet
/// declaring `declared`, may run on that object, which exports `object` (`None` = no symbol).
/// Instructions with no protected bit read the same under every layout and run on any object.
pub fn check(declared: Option<u32>, used: Option<u16>, object: Option<u32>) -> Result<(), String> {
    if let Some(layout) = declared.filter(|&l| l == 0 || l > LAYOUT) {
        return Err(format!(
            "the packet declares speech flag layout {layout}, this runtime reads {LAYOUT}"
        ));
    }
    let Some(op) = used else {
        return Ok(());
    };
    let Some(layout) = declared else {
        return Err(format!(
            "opcode {op} sets a speech flag bit assigned after the packet's undeclared flag \
             layout: re-emit the packet"
        ));
    };
    match object {
        Some(have) if have >= layout && have <= LAYOUT => Ok(()),
        Some(have) => Err(format!(
            "opcode {op} needs speech flag layout {layout}, the object is compiled for {have}: \
             rebuild the object for this packet"
        )),
        None => Err(format!(
            "opcode {op} needs speech flag layout {layout}, the object exports none: it is not a \
             speech object of this layout"
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conv_object_segments_hold_only_wgmma_convs() {
        let wg = 1 << 17;
        assert!(conv_object_inst(DevOp::Conv1dF32 as u16, wg));
        assert!(conv_object_inst(
            DevOp::ConvTranspose1dF32 as u16,
            wg | 1 << 14
        ));
        assert!(!conv_object_inst(DevOp::Conv1dF32 as u16, 1 << 14));
        assert!(!conv_object_inst(DevOp::UnaryF32 as u16, wg));
        // Segments [0, 2) mixed, [2, 3) one conv, [3, 3) empty, [3, 5) one conv's slices, [5, 7)
        // two convs, [7, 8) a conv that signals a successor.
        let conv = [false, true, true, true, true, true];
        let entry = |inst: u32| packet::dev::StreamEnt {
            inst,
            ..Default::default()
        };
        let mut stream: Vec<_> = [0, 1, 2, 3, 3, 3, 4, 5].into_iter().map(entry).collect();
        stream[7].succ_len = 1;
        let segments = conv_object_segments(&[0, 2, 3, 3, 5, 7, 8], &stream, |i| conv[i as usize]);
        assert_eq!(segments, [None, Some(2), None, Some(3), None, None]);
    }

    fn inst(op: DevOp, j1: u32, slot: usize, v: u32) -> (u16, u32, [u32; 8]) {
        let mut i = [0; 8];
        i[slot] = v;
        (op as u16, j1, i)
    }

    #[test]
    fn ring_modes_need_layout_three_without_reassigning_old_fields() {
        assert_eq!(
            required_layout([inst(DevOp::AttentionF32, 0, 6, 9 | (250 << 16))]),
            None
        );
        assert_eq!(
            required_layout([inst(DevOp::AttentionF32, 0, 6, 128)]),
            Some(1)
        );
        assert_eq!(
            required_layout([inst(DevOp::AttentionF32, 0, 6, 0x109 | (250 << 16))]),
            Some(3)
        );
        for mode in [0, 1] {
            assert_eq!(
                required_layout([inst(DevOp::StreamTailF32, 0, 7, mode)]),
                None
            );
        }
        for mode in [2, 3] {
            let i = inst(DevOp::StreamTailF32, 0, 7, mode);
            assert_eq!(required_layout([i]), Some(3));
            assert!(check(Some(3), first_protected([i]), Some(2)).is_err());
            assert!(check(Some(3), first_protected([i]), Some(3)).is_ok());
        }
        assert_eq!(declare(Some(DevOp::AttentionF32 as u16)), Some(2));
    }

    #[test]
    fn single_product_convs_need_layout_four_and_keep_their_geometry() {
        for op in [DevOp::Conv1dF32, DevOp::ConvTranspose1dF32] {
            let i = inst(op, 0, 6, 1 << 31);
            assert_eq!(required_layout([i]), Some(4));
            assert!(check(Some(4), first_protected([i]), Some(3)).is_err());
            assert!(check(Some(4), first_protected([i]), Some(4)).is_ok());
            assert_eq!(required_layout([inst(op, 0, 6, 2)]), None);
        }
        // The merged-phase ConvTranspose arm reads the output padding below bit 31.
        let f = (1 << 15) | (1 << 17) | (1 << 18);
        let mut i = [1, 64, 128, 128, 8, 4, 0, 1];
        let base = conv_arms(DevOp::ConvTranspose1dF32 as u16, &i, 0, f);
        assert_eq!(base, conv_arm::WG_CT);
        i[6] = 1 << 31;
        assert_eq!(conv_arms(DevOp::ConvTranspose1dF32 as u16, &i, 0, f), base);
    }

    #[test]
    fn half_attention_needs_layout_five() {
        let (op, i) = (DevOp::AttentionF32 as u16, [16, 3, 4, 16, 64, 3072, 1, 0]);
        assert_eq!(required_layout([(op, 2048, i)]), None);
        let half = (op, 2048 | 1 << 31, i);
        assert_eq!(required_layout([half]), Some(5));
        assert!(check(Some(5), first_protected([half]), Some(4)).is_err());
        assert!(check(Some(5), first_protected([half]), Some(5)).is_ok());
    }

    /// Every bit the layout-0 (RC9, 45e1da259) emitters could set stays unprotected. The values
    /// are those emitters' fields, not the masks' complements: conv pad 0..1, activations 4..11,
    /// routes 12..20; attention options 0..5, key-length groups 8..15, window 16..31; LayerNorm
    /// rounding / statistics 0..1 and activation from bit 4; GemvF32Out modes through bit 11.
    #[test]
    fn layout_zero_bits_are_not_protected() {
        for op in [DevOp::Conv1dF32, DevOp::ConvTranspose1dF32] {
            assert_eq!(first_protected([inst(op, 0x001f_fff3, 0, 0)]), None);
        }
        assert_eq!(
            first_protected([inst(DevOp::AttentionF32, 0, 6, 0xffff_ff3f)]),
            None
        );
        assert_eq!(
            first_protected([inst(DevOp::LayerNormF32, 0, 2, 0xffff_fff3)]),
            None
        );
        assert_eq!(
            first_protected([inst(DevOp::GemvF32Out, 0, 7, 0x0fff)]),
            None
        );
        // The same bit patterns on other ops are not flags.
        assert_eq!(
            first_protected([inst(DevOp::Gemv, u32::MAX, 7, u32::MAX)]),
            None
        );
        assert_eq!(
            first_protected([inst(DevOp::UnaryF32, 1 << 3, 6, 0xc0)]),
            None
        );
    }

    #[test]
    fn each_reassigned_bit_is_protected() {
        for bit in (2..=3).chain(21..=31) {
            for op in [DevOp::Conv1dF32, DevOp::ConvTranspose1dF32] {
                assert_eq!(
                    first_protected([inst(op, 1 << bit, 0, 0)]),
                    Some(op as u16),
                    "{bit}"
                );
            }
        }
        for bit in [6, 7] {
            let op = DevOp::AttentionF32;
            assert_eq!(first_protected([inst(op, 0, 6, 1 << bit)]), Some(op as u16));
        }
        let ln = DevOp::LayerNormF32;
        for bit in [2, 3] {
            assert_eq!(first_protected([inst(ln, 0, 2, 1 << bit)]), Some(ln as u16));
        }
        let head = DevOp::GemvF32Out;
        for bit in [12, 13, 14] {
            assert_eq!(
                first_protected([inst(head, 0, 7, 1 << bit)]),
                Some(head as u16)
            );
        }
        // Found behind unprotected instructions.
        let conv = DevOp::Conv1dF32;
        assert_eq!(
            first_protected([inst(conv, 0x6e000, 0, 0), inst(head, 0, 7, 4096 | 2048)]),
            Some(head as u16)
        );
    }

    #[test]
    fn auxiliary_program_sections_are_scanned() {
        use packet::devbuild::{Builder, Model, TensorDecl};
        let section = |view: bool| {
            let mut b = Builder::new(4);
            b.force_uniseg();
            b.emit(DevOp::Nop, b.all(), &[], |_| {});
            b.emit(DevOp::GemvF32Out, b.all(), &[], |i| {
                i.t = [0; 8];
                i.i[7] = 2048 | if view { 4096 } else { 0 };
            });
            let model = Model {
                n_cu: 4,
                target: 0,
                tensors: vec![TensorDecl {
                    name: "x".into(),
                    bytes: 8,
                    init: None,
                }],
                progs: vec![b.finish()],
                kv_row_insts: vec![],
                prog_t: vec![1],
                gen: vec![],
            };
            crate::program::with_model(&model, |p| {
                crate::aux_program::encode(4, 1, &[p.programs[0]]).unwrap()
            })
        };
        let scan = |bytes: &[u8]| {
            assert!(is_aux_programs(bytes));
            first_protected(
                aux_insts(bytes, 4)
                    .unwrap()
                    .iter()
                    .map(|d| (d.op, d.fj[2], d.i)),
            )
        };
        assert_eq!(scan(&section(false)), None);
        assert_eq!(scan(&section(true)), Some(DevOp::GemvF32Out as u16));
        assert!(!is_aux_programs(b"{\"version\":1}"));
        assert!(aux_insts(&section(true), 5).is_err());
    }

    #[test]
    fn decode_objects_refuse_protected_auxiliary_instructions() {
        let wire = |i7: u32| {
            let mut d = DevInst64 {
                op: DevOp::GemvF32Out as u16,
                ..Default::default()
            };
            d.i[7] = i7;
            d
        };
        assert!(refuse_on_decode_objects(&[wire(0), wire(2048)]).is_ok());
        let e = refuse_on_decode_objects(&[wire(2048), wire(2048 | 4096)]).unwrap_err();
        assert!(e.contains("do not read"), "{e}");
    }

    #[test]
    fn undeclared_packets_load_only_without_protected_bits() {
        let conv = Some(DevOp::Conv1dF32 as u16);
        for object in [None, Some(0), Some(LAYOUT), Some(LAYOUT + 1)] {
            assert!(check(None, None, object).is_ok());
            let e = check(None, conv, object).unwrap_err();
            assert!(e.contains("re-emit"), "{e}");
        }
        assert_eq!(declare(None), None);
        assert_eq!(declare(conv), Some(DEFAULT_LAYOUT));
    }

    #[test]
    fn declared_packets_need_the_same_object_layout() {
        let conv = Some(DevOp::Conv1dF32 as u16);
        assert!(check(Some(LAYOUT), conv, Some(LAYOUT)).is_ok());
        assert!(check(Some(LAYOUT), conv, None)
            .unwrap_err()
            .contains("exports none"));
        for have in [0, 1, LAYOUT + 1] {
            let e = check(Some(LAYOUT), conv, Some(have)).unwrap_err();
            assert!(e.contains("compiled for"), "{e}");
        }
        // An object that runs none of the protected instructions (a Maya 2 packet's backbone
        // decode object, its depth programs being the speech object's) is not constrained.
        for object in [None, Some(0), Some(LAYOUT), Some(LAYOUT + 1)] {
            assert!(check(Some(LAYOUT), None, object).is_ok());
        }
        // A layout this runtime does not read is refused on every object.
        for used in [None, conv] {
            let e = check(Some(LAYOUT + 1), used, Some(LAYOUT + 1)).unwrap_err();
            assert!(e.contains("this runtime reads"), "{e}");
        }
    }

    #[test]
    fn layout_one_packets_load_on_layout_two_objects() {
        let op = Some(DevOp::GemvF32Out as u16);
        assert!(check(Some(1), op, Some(1)).is_ok());
        assert!(check(Some(1), op, Some(2)).is_ok());
        assert!(check(Some(2), op, Some(1)).is_err());
        assert_eq!(
            required_layout([inst(DevOp::GemvF32Out, 0, 7, 4096)]),
            Some(1)
        );
        assert_eq!(
            required_layout([inst(DevOp::GemvF32Out, 0, 7, 8192)]),
            Some(2)
        );
        assert_eq!(
            required_layout([inst(DevOp::LayerNormF32, 0, 2, 8)]),
            Some(2)
        );
    }

    #[test]
    fn conv_arms_of_the_qwen3_tts_codec() {
        use conv_arm::*;
        let (conv, convt) = (DevOp::Conv1dF32 as u16, DevOp::ConvTranspose1dF32 as u16);
        // [batch, in_rows, cin, cout, kernel, stride, dil, groups], wire flags as emitted.
        let arms = |op, i: [u32; 8], f: u32| conv_arms(op, &i, 0, f);
        assert_eq!(arms(conv, [8, 17, 512, 1024, 1, 1, 1, 1], 0x266000), WG);
        assert_eq!(arms(conv, [8, 19, 512, 1024, 3, 1, 1, 1], 0x26e000), WG);
        assert_eq!(
            arms(conv, [8, 23, 1024, 1024, 7, 1, 1, 1024], 0xa26000),
            WG | DEPTHWISE | DIRECT
        );
        assert_eq!(
            arms(conv, [8, 23, 96, 1, 7, 1, 1, 1], 0x226000),
            WG | STAGED | DIRECT
        );
        assert_eq!(arms(convt, [8, 17, 192, 96, 6, 3, 0, 1], 0x26e008), WG_CT);
        // The fused codec unit c1 (stream + input SnakeBeta), the stream-only pre conv, a plain one.
        let var = |i: [u32; 8], f: u32| conv_bf16_variants(conv, &i, f);
        assert_eq!(var([8, 32694, 96, 96, 7, 1, 9, 1], 0x6eaccc), 1 << 3);
        assert_eq!(var([8, 2774, 384, 384, 7, 1, 9, 1], 0x6eaccc), 1 << (8 + 3));
        assert_eq!(var([8, 19, 512, 1024, 3, 1, 1, 1], 0x66a000), 1 << (8 + 2));
        assert_eq!(
            var([8, 10934, 192, 192, 7, 1, 9, 1], 0x26a00c | 12 << 8),
            1 << 4
        );
        assert_eq!(var([8, 17, 512, 1024, 1, 1, 1, 1], 0x266000), 0);
        assert_eq!(
            var([8, 19, 512, 1024, 3, 1, 1, 1], 0x66a000 | 10 << 24),
            1 << (12 + 8 + 2)
        );
        assert_eq!(conv_bf16_kind(192, 192, 7, 9), Some(2));
        assert_eq!(conv_bf16_kind(768, 768, 7, 9), Some(3));
        assert_eq!(conv_bf16_kind(96, 1, 7, 1), None);
        // bf16 taps and split-image convs: their own defines.
        assert_eq!(
            arms(conv, [8, 71, 96, 96, 7, 1, 9, 1], 0x26a00c | 12 << 8),
            0
        );
        assert_eq!(
            arms(conv, [8, 17, 512, 1024, 1, 1, 1, 1], 0x266000 | 1 << 30),
            0
        );
        // Not pinned: the launch width picks tensor cores or FFMA for a transposed conv.
        assert_eq!(arms(convt, [1, 9, 64, 64, 4, 2, 0, 1], 1 << 13), TC | FFMA);
        assert_eq!(arms(conv, [1, 9, 64, 2, 3, 1, 1, 1], 1 << 16), ROWS);
        assert_eq!(arms(conv, [1, 9, 64, 64, 3, 1, 1, 1], 0), FFMA);
        assert_eq!(
            arms(conv, [1, 9, 64, 64, 1, 1, 1, 1], 1 << 14 | 1 << 13),
            WIDE
        );
    }
}
