//! Minimal ELF reader for NVIDIA cubins — enough to answer the two questions
//! the loader must not guess at: **what SM is this image for**, and **which
//! kernel entry points does it actually contain**.
//!
//! A cubin is a plain ELF64-LE object. Its `e_flags` carry the SM number and
//! its `.symtab` names every entry point, so an image describes itself; the
//! file it happens to be stored under does not. Trusting the name is how a
//! swapped decode/prefill pair (`build_sm90a_cubin.sh` derives the prefill path
//! from the decode one, so an extensionless argument produces exactly that)
//! turns into `cuModuleGetFunction(...): CUDA_ERROR_NOT_FOUND` at model load,
//! and how a wrong-arch image turns into an opaque driver error. Reading 24
//! bytes per symbol at startup costs nothing next to the weight upload.
//!
//! Parsing is total: every field is bounds-checked and a malformed image
//! returns `None` rather than panicking. Nothing here is on the hot path.

/// What a cubin says about itself.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CubinInfo {
    /// SM number from `e_flags` — 90 for `sm_90`/`sm_90a`, 120 for `sm_120a`.
    /// The `a` (architecture-accelerated) suffix is NOT encoded here: `sm_90`
    /// and `sm_90a` produce identical flags.
    pub sm: u32,
    /// Global `FUNC` symbols: the image's kernel entry points.
    pub entries: Vec<String>,
}

/// Which interpreter object a caller wants.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    Decode,
    Prefill,
}

impl Role {
    pub fn as_str(self) -> &'static str {
        match self {
            Role::Decode => "decode",
            Role::Prefill => "prefill",
        }
    }
}

impl CubinInfo {
    /// This image's persistent-interpreter entry for `role`, if it has one.
    ///
    /// The entries are the Itanium-mangled `interp_<arch>[_pf](PlowProgram)`
    /// kernels — `_Z12interp_sm90a11PlowProgram`,
    /// `_Z15interp_sm120_pf11PlowProgram`. Matching on the shape rather than on
    /// a hard-coded per-arch constant is what makes a new arch a table entry in
    /// `interpreter_profile` instead of a new pair of mangled-name literals.
    pub fn interp_entry(&self, role: Role) -> Option<&str> {
        self.entries
            .iter()
            .map(String::as_str)
            .find(|n| interp_role(n) == Some(role))
    }
}

/// The role of a mangled interpreter entry name, or `None` if it is not one.
fn interp_role(sym: &str) -> Option<Role> {
    if !sym.starts_with("_Z") || !sym.ends_with("11PlowProgram") || !sym.contains("interp_") {
        return None;
    }
    // `interp_sm120_pf` vs `interp_sm120` — the `_pf` object is the only one
    // carrying the tiled-GEMM / flash-prefill arms.
    Some(if sym.contains("_pf") { Role::Prefill } else { Role::Decode })
}

const EI_NIDENT: usize = 16;
const SHT_SYMTAB: u32 = 2;
const STB_GLOBAL: u8 = 1;
const STT_FUNC: u8 = 2;
/// `Elf64_Sym` size.
const SYM_SZ: usize = 24;

fn u16_at(b: &[u8], o: usize) -> Option<u16> {
    b.get(o..o + 2).map(|s| u16::from_le_bytes([s[0], s[1]]))
}
fn u32_at(b: &[u8], o: usize) -> Option<u32> {
    b.get(o..o + 4).map(|s| u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
}
fn u64_at(b: &[u8], o: usize) -> Option<u64> {
    b.get(o..o + 8)
        .map(|s| u64::from_le_bytes([s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7]]))
}

/// True if `image` starts with an ELF64-LE header — the cheap pre-filter that
/// keeps the loader from reading a multi-GiB checkpoint shard as a candidate.
pub fn is_elf64_le(image: &[u8]) -> bool {
    image.len() >= EI_NIDENT && image[..4] == *b"\x7fELF" && image[4] == 2 && image[5] == 1
}

/// Parse a cubin's SM number and global entry points. `None` if `image` is not
/// a well-formed ELF64-LE object.
pub fn inspect(image: &[u8]) -> Option<CubinInfo> {
    if !is_elf64_le(image) {
        return None;
    }
    // Elf64_Ehdr: e_flags at 0x30, e_shoff at 0x28, e_shentsize/num at 0x3a/0x3c.
    let e_flags = u32_at(image, 0x30)?;
    let e_shoff = u64_at(image, 0x28)? as usize;
    let e_shentsize = u16_at(image, 0x3a)? as usize;
    let e_shnum = u16_at(image, 0x3c)? as usize;

    // NVIDIA packs the SM number into byte 1 of e_flags: 0x…5a04 → sm_90,
    // 0x…7802 → sm_120 (byte 0 is the cubin ABI version). Verified against
    // nvcc -arch={sm_80,sm_90,sm_90a,sm_120}.
    let sm = (e_flags >> 8) & 0xff;

    // Elf64_Shdr: sh_type at 4, sh_offset at 0x18, sh_size at 0x20, sh_link at
    // 0x28 (the symtab's string table).
    let shdr = |i: usize| -> Option<&[u8]> {
        let o = e_shoff.checked_add(i.checked_mul(e_shentsize)?)?;
        image.get(o..o + e_shentsize)
    };
    let mut entries = Vec::new();
    for i in 0..e_shnum {
        let sh = shdr(i)?;
        if u32_at(sh, 4)? != SHT_SYMTAB {
            continue;
        }
        let off = u64_at(sh, 0x18)? as usize;
        let size = u64_at(sh, 0x20)? as usize;
        let syms = image.get(off..off.checked_add(size)?)?;
        let strtab = shdr(u32_at(sh, 0x28)? as usize)?;
        let str_off = u64_at(strtab, 0x18)? as usize;
        let str_size = u64_at(strtab, 0x20)? as usize;
        let strs = image.get(str_off..str_off.checked_add(str_size)?)?;

        for s in syms.chunks_exact(SYM_SZ) {
            let info = s[4];
            if info >> 4 != STB_GLOBAL || info & 0xf != STT_FUNC {
                continue;
            }
            let n = u32_at(s, 0)? as usize;
            let name = strs.get(n..)?;
            let end = name.iter().position(|&c| c == 0).unwrap_or(name.len());
            if let Ok(name) = std::str::from_utf8(&name[..end]) {
                entries.push(name.to_string());
            }
        }
    }
    Some(CubinInfo { sm, entries })
}

/// One-line summary for an operator-facing error: what this image really is.
pub fn describe(image: &[u8]) -> String {
    match inspect(image) {
        None => format!("not an ELF cubin ({} B)", image.len()),
        Some(i) => {
            let role = |r: Role| i.interp_entry(r).map(|_| r.as_str());
            let roles: Vec<&str> = [Role::Decode, Role::Prefill].into_iter().filter_map(role).collect();
            let what = if roles.is_empty() {
                format!("no interpreter entry; {} global kernel(s)", i.entries.len())
            } else {
                roles.join("+")
            };
            format!("sm_{} {what} ({} B)", i.sm, image.len())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifies_interpreter_entries() {
        assert_eq!(interp_role("_Z12interp_sm90a11PlowProgram"), Some(Role::Decode));
        assert_eq!(interp_role("_Z15interp_sm90a_pf11PlowProgram"), Some(Role::Prefill));
        assert_eq!(interp_role("_Z12interp_sm12011PlowProgram"), Some(Role::Decode));
        assert_eq!(interp_role("_Z15interp_sm120_pf11PlowProgram"), Some(Role::Prefill));
        // Not the megakernel: the device sampler and the MoE block helpers.
        assert_eq!(interp_role("plow_sample"), None);
        assert_eq!(interp_role("_Z25plow_moe_slot_glu_fp8_blkP13__nv_bfloat16"), None);
    }

    #[test]
    fn entry_lookup_is_role_exact() {
        let pf = CubinInfo {
            sm: 90,
            entries: vec!["_Z15interp_sm90a_pf11PlowProgram".into()],
        };
        assert_eq!(pf.interp_entry(Role::Decode), None);
        assert_eq!(pf.interp_entry(Role::Prefill), Some("_Z15interp_sm90a_pf11PlowProgram"));
    }

    #[test]
    fn rejects_garbage() {
        assert!(inspect(b"").is_none());
        assert!(inspect(b"not an elf at all, really").is_none());
        // ELF64-LE magic with a truncated header must not panic.
        let mut trunc = vec![0u8; 40];
        trunc[..4].copy_from_slice(b"\x7fELF");
        trunc[4] = 2;
        trunc[5] = 1;
        assert!(inspect(&trunc).is_none());
    }

    /// The SM encoding is the one load-bearing magic constant here, so pin it
    /// against a synthetic header rather than only against real cubins (which
    /// this crate's tests cannot build).
    #[test]
    fn reads_sm_from_e_flags() {
        for (flags, sm) in [(0x0600_5a04u32, 90), (0x0600_7802, 120), (0x0600_5004, 80)] {
            let mut e = vec![0u8; 0x40];
            e[..4].copy_from_slice(b"\x7fELF");
            e[4] = 2;
            e[5] = 1;
            e[0x30..0x34].copy_from_slice(&flags.to_le_bytes());
            // e_shoff = 0, e_shnum = 0: header-only, no sections to walk.
            let info = inspect(&e).expect("header-only ELF64 parses");
            assert_eq!(info.sm, sm);
            assert!(info.entries.is_empty());
        }
    }
}
