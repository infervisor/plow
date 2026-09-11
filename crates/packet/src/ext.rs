//! The extension container (`extension.pkt`) — docs/arch/19, phase 2.
//!
//! An extension is the SAME container as `model.pkt` with the tensor table replaced by a
//! **reference** to the parent's: the parent packet hash, the parent's tensor count, and a
//! digest over the parent's tensor tuples. It carries programs only — no checkpoint, no
//! weights, no tensor data, no init section.
//!
//! ## Why a distinct magic
//!
//! The container could ride [`crate::devbuild::BLOB_MAGIC_V6`] with `n_tensor == 0`, and that
//! is exactly the silent failure this tree bumps magics to avoid (see
//! [`crate::devbuild::BLOB_MAGIC_V7`]'s note). A reader that mistook an extension for a model
//! would declare zero tensors, bind nothing, and run programs whose every tensor handle is out
//! of range. It would also be picked up by `DevBlob::find_in_dir`, whose only test is the
//! `PLOWDEV` magic, and reported as a second model. So an extension gets its own 8-byte magic
//! and [`crate::devbuild::is_blob_magic`] stays false for it.
//!
//! ## Layout
//!
//! Byte-for-byte the v6/v7 container, with two differences:
//!
//! * `magic` is [`EXT_MAGIC`]; `n_tensor` and `init_bytes` are `0` (there is no tensor table
//!   and no init section to hold it).
//! * A [`SECT_PARENT_REF`] section, written FIRST, carries [`BlobParentRef`].
//!
//! Everything after the (empty) tensor table — the kvrow table, the program records, the
//! `GQ01` appendix, the section data and the section directory — is the parent format
//! unchanged, so one reader walks both.

/// `extension.pkt`'s container magic. `\x01` is the extension container version; it is
/// independent of the `PLOWDEV` sequence because the two are read by different entry points.
pub const EXT_MAGIC: &[u8; 8] = b"PLOWEXT\x01";

/// Every extension container version this build can read.
pub const EXT_MAGICS: [&[u8; 8]; 1] = [EXT_MAGIC];

pub fn is_ext_magic(m: &[u8; 8]) -> bool {
    EXT_MAGICS.contains(&m)
}

/// Section kind carrying [`BlobParentRef`]. Present exactly once in an extension container
/// and never in a parent packet.
pub const SECT_PARENT_REF: u32 = 7;

/// `BlobParentRef::magic`.
pub const PARENT_REF_MAGIC: [u8; 4] = *b"PEXT";
/// `BlobParentRef::version`.
pub const PARENT_REF_VERSION: u32 = 1;

/// The reference an extension carries INSTEAD of a tensor table.
///
/// `tensor_digest` is what makes an extension safe at all: the extension's instructions carry
/// tensor INDICES, and an index only means something against the table it was emitted
/// against. The digest pins that table by content, so a parent re-emitted with one extra
/// tensor — same file name, same everything else — cannot silently re-point an extension's
/// `t[0]` at a different buffer.
///
/// `parent_hash` is the coarser check and is reported first because its message is the one a
/// human can act on ("this extension is for a different packet"); the digest catches the case
/// where somebody rebuilt the parent and kept the name.
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct BlobParentRef {
    pub magic: [u8; 4],
    pub version: u32,
    /// SHA-256 of the parent `model.pkt` image.
    pub parent_hash: [u8; 32],
    /// SHA-256 over the parent's tensor tuples — see
    /// `plow_asset::extension::tensor_table_digest`.
    pub tensor_digest: [u8; 32],
    /// The parent's tensor count, restated so a rewritten parent is named by count before the
    /// opaque digest mismatch is reported.
    pub n_tensor: u32,
    pub _pad: u32,
}

const _: () = assert!(std::mem::size_of::<BlobParentRef>() == 80);

impl BlobParentRef {
    pub fn new(parent_hash: [u8; 32], tensor_digest: [u8; 32], n_tensor: u32) -> BlobParentRef {
        BlobParentRef {
            magic: PARENT_REF_MAGIC,
            version: PARENT_REF_VERSION,
            parent_hash,
            tensor_digest,
            n_tensor,
            _pad: 0,
        }
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut b = Vec::with_capacity(std::mem::size_of::<BlobParentRef>());
        b.extend_from_slice(&self.magic);
        b.extend_from_slice(&self.version.to_le_bytes());
        b.extend_from_slice(&self.parent_hash);
        b.extend_from_slice(&self.tensor_digest);
        b.extend_from_slice(&self.n_tensor.to_le_bytes());
        b.extend_from_slice(&self._pad.to_le_bytes());
        b
    }

    /// Decode a [`SECT_PARENT_REF`] section body. Every failure names what is wrong rather
    /// than returning `None`: this section is the extension's entire claim about its parent,
    /// and guessing past a broken one is how the check would stop checking.
    pub fn from_bytes(b: &[u8]) -> Result<BlobParentRef, String> {
        let want = std::mem::size_of::<BlobParentRef>();
        if b.len() != want {
            return Err(format!("parent-ref section is {} B, want {want}", b.len()));
        }
        let magic: [u8; 4] = b[0..4].try_into().unwrap();
        if magic != PARENT_REF_MAGIC {
            return Err("parent-ref section has bad magic".into());
        }
        let version = u32::from_le_bytes(b[4..8].try_into().unwrap());
        if version != PARENT_REF_VERSION {
            return Err(format!(
                "parent-ref version {version} — this plowrt reads {PARENT_REF_VERSION}"
            ));
        }
        Ok(BlobParentRef {
            magic,
            version,
            parent_hash: b[8..40].try_into().unwrap(),
            tensor_digest: b[40..72].try_into().unwrap(),
            n_tensor: u32::from_le_bytes(b[72..76].try_into().unwrap()),
            _pad: u32::from_le_bytes(b[76..80].try_into().unwrap()),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ext_magic_is_not_a_model_magic() {
        assert!(!crate::devbuild::is_blob_magic(EXT_MAGIC));
        assert!(is_ext_magic(EXT_MAGIC));
        for m in crate::devbuild::BLOB_MAGICS {
            assert!(!is_ext_magic(m), "model magic read as an extension");
        }
    }

    #[test]
    fn parent_ref_round_trips_and_refuses_junk() {
        let r = BlobParentRef::new([7u8; 32], [9u8; 32], 1234);
        assert_eq!(BlobParentRef::from_bytes(&r.to_bytes()).unwrap(), r);

        let mut bad = r.to_bytes();
        bad[0] = b'X';
        assert!(BlobParentRef::from_bytes(&bad).unwrap_err().contains("magic"));

        let mut ver = r.to_bytes();
        ver[4] = 9;
        assert!(BlobParentRef::from_bytes(&ver)
            .unwrap_err()
            .contains("version"));

        assert!(BlobParentRef::from_bytes(&r.to_bytes()[..40])
            .unwrap_err()
            .contains("want 80"));
    }
}
