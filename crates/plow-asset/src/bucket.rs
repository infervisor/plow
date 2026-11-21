//! Modular packet bucket container and manifest specification.
//!
//! Docs: `docs/arch/31-modular-packet-buckets.md`.
//! Supports modular, sharded bucket packets (`buckets/*.pkt`) governed by a lightweight
//! manifest (`buckets.json`), allowing new buckets to be emitted independently and
//! hot-plugged with zero downtime.

use serde::{Deserialize, Serialize};

/// Container magic for modular bucket packets.
pub const BUCKET_MAGIC: &[u8; 8] = b"PLOWBKT\x01";

pub fn is_bucket_magic(m: &[u8; 8]) -> bool {
    m == BUCKET_MAGIC
}

/// Binary header at the start of every modular bucket container (`buckets/*.pkt`).
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct PlowBucketHeader {
    pub magic: [u8; 8],
    /// SHA-256 of the parent `base.pkt`.
    pub base_packet_hash: [u8; 32],
    /// SHA-256 over the parent's tensor table tuples.
    pub tensor_table_digest: [u8; 32],
    /// Target hardware architecture (e.g. sm_90a, gfx942).
    pub target_arch: u32,
    /// Number of compute units / SMs the bucket was compiled for.
    pub n_cu: u32,
    /// Role tag discriminant: 0 = PrefillBucket, 1 = DecodeRung, 2 = TokenBatch, 3 = SpecDraft.
    pub role_tag: u32,
    /// Bucket width: token count M for prefill, batch size B for decode.
    pub role_rows: u32,
}

pub const BUCKET_HEADER_SIZE: usize = std::mem::size_of::<PlowBucketHeader>();
const _: () = assert!(BUCKET_HEADER_SIZE == 88);

impl PlowBucketHeader {
    pub fn new(
        base_packet_hash: [u8; 32],
        tensor_table_digest: [u8; 32],
        target_arch: u32,
        n_cu: u32,
        role: BucketRole,
    ) -> Self {
        let (role_tag, role_rows) = role.to_tag_and_rows();
        Self {
            magic: *BUCKET_MAGIC,
            base_packet_hash,
            tensor_table_digest,
            target_arch,
            n_cu,
            role_tag,
            role_rows,
        }
    }

    pub fn role(&self) -> Option<BucketRole> {
        BucketRole::from_tag_and_rows(self.role_tag, self.role_rows)
    }

    pub fn to_bytes(&self) -> [u8; BUCKET_HEADER_SIZE] {
        let mut b = [0u8; BUCKET_HEADER_SIZE];
        b[0..8].copy_from_slice(&self.magic);
        b[8..40].copy_from_slice(&self.base_packet_hash);
        b[40..72].copy_from_slice(&self.tensor_table_digest);
        b[72..76].copy_from_slice(&self.target_arch.to_le_bytes());
        b[76..80].copy_from_slice(&self.n_cu.to_le_bytes());
        b[80..84].copy_from_slice(&self.role_tag.to_le_bytes());
        b[84..88].copy_from_slice(&self.role_rows.to_le_bytes());
        b
    }

    pub fn from_bytes(b: &[u8]) -> Result<Self, String> {
        if b.len() < BUCKET_HEADER_SIZE {
            return Err(format!(
                "bucket header truncated: got {} B, want {}",
                b.len(),
                BUCKET_HEADER_SIZE
            ));
        }
        let magic: [u8; 8] = b[0..8].try_into().unwrap();
        if !is_bucket_magic(&magic) {
            return Err("invalid bucket magic".into());
        }
        let mut base_packet_hash = [0u8; 32];
        base_packet_hash.copy_from_slice(&b[8..40]);
        let mut tensor_table_digest = [0u8; 32];
        tensor_table_digest.copy_from_slice(&b[40..72]);
        let target_arch = u32::from_le_bytes(b[72..76].try_into().unwrap());
        let n_cu = u32::from_le_bytes(b[76..80].try_into().unwrap());
        let role_tag = u32::from_le_bytes(b[80..84].try_into().unwrap());
        let role_rows = u32::from_le_bytes(b[84..88].try_into().unwrap());

        Ok(Self {
            magic,
            base_packet_hash,
            tensor_table_digest,
            target_arch,
            n_cu,
            role_tag,
            role_rows,
        })
    }
}

/// The role and dimensions of a modular bucket.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "rows")]
pub enum BucketRole {
    PrefillBucket(u32),
    DecodeRung(u32),
    TokenBatch(u32),
    SpecDraft(u32),
}

impl BucketRole {
    pub fn to_tag_and_rows(self) -> (u32, u32) {
        match self {
            BucketRole::PrefillBucket(m) => (0, m),
            BucketRole::DecodeRung(b) => (1, b),
            BucketRole::TokenBatch(b) => (2, b),
            BucketRole::SpecDraft(k) => (3, k),
        }
    }

    pub fn from_tag_and_rows(tag: u32, rows: u32) -> Option<Self> {
        match tag {
            0 => Some(BucketRole::PrefillBucket(rows)),
            1 => Some(BucketRole::DecodeRung(rows)),
            2 => Some(BucketRole::TokenBatch(rows)),
            3 => Some(BucketRole::SpecDraft(rows)),
            _ => None,
        }
    }

    pub fn rows(self) -> u32 {
        match self {
            BucketRole::PrefillBucket(r)
            | BucketRole::DecodeRung(r)
            | BucketRole::TokenBatch(r)
            | BucketRole::SpecDraft(r) => r,
        }
    }
}

/// Single bucket descriptor inside `buckets.json`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BucketEntry {
    pub id: String,
    pub path: String,
    pub sha256: String,
    pub role: BucketRole,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub required_objects: Vec<String>,
    #[serde(default)]
    pub verified_lean: bool,
}

/// Manifest indexing modular bucket packets (`buckets.json`).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BucketManifest {
    pub schema_version: String,
    pub base_hash: String,
    pub tensor_table_digest: String,
    pub target: String,
    pub n_cu: u32,
    pub max_decode_batch: u32,
    pub max_chunk_rows: u32,
    pub buckets: Vec<BucketEntry>,
}

pub const BUCKET_MANIFEST_SCHEMA: &str = "plow.bucket_manifest.v1";

impl BucketManifest {
    pub fn new(
        base_hash: impl Into<String>,
        tensor_table_digest: impl Into<String>,
        target: impl Into<String>,
        n_cu: u32,
        max_decode_batch: u32,
        max_chunk_rows: u32,
    ) -> Self {
        Self {
            schema_version: BUCKET_MANIFEST_SCHEMA.to_string(),
            base_hash: base_hash.into(),
            tensor_table_digest: tensor_table_digest.into(),
            target: target.into(),
            n_cu,
            max_decode_batch,
            max_chunk_rows,
            buckets: Vec::new(),
        }
    }

    pub fn add_bucket(&mut self, entry: BucketEntry) {
        self.buckets.push(entry);
    }

    /// Validate all fail-closed load invariants against parent base packet facts.
    pub fn validate_against_base(
        &self,
        base_hash_hex: &str,
        tensor_digest_hex: &str,
        n_cu: u32,
    ) -> Result<(), String> {
        if self.base_hash.to_lowercase() != base_hash_hex.to_lowercase() {
            return Err(format!(
                "base packet hash mismatch: manifest declares {}, parent base is {}",
                self.base_hash, base_hash_hex
            ));
        }
        if self.tensor_table_digest.to_lowercase() != tensor_digest_hex.to_lowercase() {
            return Err(format!(
                "tensor table digest mismatch: manifest declares {}, parent base is {}",
                self.tensor_table_digest, tensor_digest_hex
            ));
        }
        if self.n_cu != n_cu {
            return Err(format!(
                "CU count mismatch: manifest declares {} CUs, target device has {}",
                self.n_cu, n_cu
            ));
        }

        // Validate envelope bounds
        for b in &self.buckets {
            match b.role {
                BucketRole::PrefillBucket(rows) => {
                    if rows > self.max_chunk_rows {
                        return Err(format!(
                            "bucket {} rows ({}) exceeds max_chunk_rows ({})",
                            b.id, rows, self.max_chunk_rows
                        ));
                    }
                }
                BucketRole::DecodeRung(rows) => {
                    if rows > self.max_decode_batch {
                        return Err(format!(
                            "bucket {} rows ({}) exceeds max_decode_batch ({})",
                            b.id, rows, self.max_decode_batch
                        ));
                    }
                }
                _ => {}
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bucket_header_round_trip() {
        let parent_hash = [0xabu8; 32];
        let tensor_digest = [0xcdu8; 32];
        let hdr = PlowBucketHeader::new(
            parent_hash,
            tensor_digest,
            0x90a,
            132,
            BucketRole::PrefillBucket(4096),
        );
        let bytes = hdr.to_bytes();
        let decoded = PlowBucketHeader::from_bytes(&bytes).unwrap();
        assert_eq!(decoded, hdr);
        assert_eq!(decoded.role(), Some(BucketRole::PrefillBucket(4096)));
    }

    #[test]
    fn bucket_manifest_validation() {
        let mut manifest = BucketManifest::new(
            "deadbeef0123456789abcdef0123456789abcdef0123456789abcdef01234567",
            "11223344556677889900aabbccddeeff11223344556677889900aabbccddeeff",
            "sm_90a",
            132,
            64,
            8192,
        );
        manifest.add_bucket(BucketEntry {
            id: "prefill_4096".into(),
            path: "buckets/prefill_4096.pkt".into(),
            sha256: "aabbcc".into(),
            role: BucketRole::PrefillBucket(4096),
            required_objects: vec!["d_gemm".into()],
            verified_lean: true,
        });

        assert!(manifest
            .validate_against_base(
                "deadbeef0123456789abcdef0123456789abcdef0123456789abcdef01234567",
                "11223344556677889900aabbccddeeff11223344556677889900aabbccddeeff",
                132,
            )
            .is_ok());

        // Base hash mismatch
        assert!(manifest
            .validate_against_base(
                "0000000000000000000000000000000000000000000000000000000000000000",
                "11223344556677889900aabbccddeeff11223344556677889900aabbccddeeff",
                132,
            )
            .is_err());

        // Envelope violation
        manifest.add_bucket(BucketEntry {
            id: "overflow".into(),
            path: "buckets/overflow.pkt".into(),
            sha256: "aabbcc".into(),
            role: BucketRole::PrefillBucket(16384),
            required_objects: vec![],
            verified_lean: false,
        });
        assert!(manifest
            .validate_against_base(
                "deadbeef0123456789abcdef0123456789abcdef0123456789abcdef01234567",
                "11223344556677889900aabbccddeeff11223344556677889900aabbccddeeff",
                132,
            )
            .is_err());
    }
}
