//! Drafter context the TARGET programs maintain (`PLOW_EMIT_SPEC` with a block drafter).
//!
//! A drafter that keeps its own KV (DSpark) gets it from the target: every target program ends
//! with a context tail that projects the post-layer residuals of the `taps` layers into the
//! drafter's `kv_layers`, at each row's own position and slot, in the same launch that writes the
//! target KV. The drafter layers are ordinary full-attention layers of the packet's KV layout
//! (`kv.{l}.k`, listed in `serve.json`'s full layers), so paging and prefix snapshots carry them;
//! `layout_sha256` names what wrote them, so a prefix is only ever restored into a packet whose
//! tail would have written the same rows.
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const SECTION: &str = "spec.context.json";
pub const VERSION: u32 = 1;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    pub version: u32,
    /// Drafter family, e.g. `gemma4_dspark`.
    pub family: String,
    /// Target layers whose post-layer residual the tail reads, ascending.
    pub taps: Vec<u32>,
    /// Packet KV layers the tail writes (`kv.{l}.k` / `.v`), after the target's own.
    pub kv_layers: Vec<u32>,
    /// KV heads and head dim of every drafter layer.
    pub kv_heads: u32,
    pub head_dim: u32,
    /// `e4m3` (per-row f32 scales) or `bf16`.
    pub kv_elem: String,
    /// The tail's projection route: `bf16`, `w8a16` or `w8a8` (FP8 twins, activations e4m3).
    pub projection: String,
    /// The drafter weight shard the tail binds, and its sha256.
    pub shard: String,
    pub shard_sha256: String,
    /// sha256 of the FP8 twin shard the tail binds; empty on the bf16 route.
    pub twins_sha256: String,
    pub layout_sha256: String,
}

impl Manifest {
    /// Digest over everything that decides the bytes the tail writes into the drafter layers.
    pub fn layout_digest(&self) -> String {
        let mut h = Sha256::new();
        h.update(b"plow-spec-context-v1");
        for s in [
            &self.family,
            &self.kv_elem,
            &self.projection,
            &self.shard,
            &self.shard_sha256,
            &self.twins_sha256,
        ] {
            h.update((s.len() as u64).to_le_bytes());
            h.update(s.as_bytes());
        }
        for list in [&self.taps, &self.kv_layers] {
            h.update((list.len() as u64).to_le_bytes());
            for v in list {
                h.update(v.to_le_bytes());
            }
        }
        h.update(self.kv_heads.to_le_bytes());
        h.update(self.head_dim.to_le_bytes());
        format!("{:x}", h.finalize())
    }

    /// `target_layers` is the target's own layer count; drafter layers follow it.
    pub fn validate(&self, target_layers: u32) -> Result<(), String> {
        let ascending = |v: &[u32]| v.windows(2).all(|w| w[0] < w[1]);
        if self.version != VERSION
            || self.family.is_empty()
            || self.taps.is_empty()
            || !ascending(&self.taps)
            || self.taps.iter().any(|&l| l >= target_layers)
            || self.kv_layers.is_empty()
            || !ascending(&self.kv_layers)
            || self.kv_layers[0] < target_layers
            || self.kv_heads == 0
            || self.head_dim == 0
            || !matches!(self.kv_elem.as_str(), "e4m3" | "bf16")
            || !matches!(self.projection.as_str(), "bf16" | "w8a16" | "w8a8")
            || (self.projection == "bf16") != self.twins_sha256.is_empty()
            || self.shard.is_empty()
        {
            return Err("spec context: manifest geometry".into());
        }
        if self.layout_sha256 != self.layout_digest() {
            return Err("spec context: layout_sha256 does not match the manifest".into());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn manifest() -> Manifest {
        let mut m = Manifest {
            version: VERSION,
            family: "gemma4_dspark".into(),
            taps: vec![5, 17, 29, 41, 46],
            kv_layers: (48..53).collect(),
            kv_heads: 1,
            head_dim: 512,
            kv_elem: "e4m3".into(),
            projection: "w8a8".into(),
            shard: "model-dspark.safetensors".into(),
            shard_sha256: "00".into(),
            twins_sha256: "22".into(),
            layout_sha256: String::new(),
        };
        m.layout_sha256 = m.layout_digest();
        m
    }

    #[test]
    fn validates_and_round_trips() {
        let m = manifest();
        m.validate(48).unwrap();
        let back: Manifest = serde_json::from_slice(&serde_json::to_vec(&m).unwrap()).unwrap();
        assert_eq!(back, m);
    }

    #[test]
    fn refuses_bad_geometry_and_a_stale_digest() {
        assert!(
            manifest().validate(46).is_err(),
            "tap past the target layers"
        );
        assert!(
            manifest().validate(49).is_err(),
            "drafter layer inside the target's"
        );
        let mut m = manifest();
        m.taps = vec![17, 5];
        m.layout_sha256 = m.layout_digest();
        assert!(m.validate(48).is_err());
        let mut m = manifest();
        m.shard_sha256 = "11".into();
        assert!(m.validate(48).is_err(), "digest no longer covers the shard");
        let mut m = manifest();
        m.projection = "bf16".into();
        m.layout_sha256 = m.layout_digest();
        assert!(m.validate(48).is_err(), "bf16 route binds no twins");
    }

    #[test]
    fn the_digest_covers_taps_layers_and_precision() {
        let base = manifest().layout_digest();
        let mut m = manifest();
        m.taps[4] = 47;
        assert_ne!(m.layout_digest(), base);
        let mut m = manifest();
        m.kv_elem = "bf16".into();
        assert_ne!(m.layout_digest(), base);
        let mut m = manifest();
        m.projection = "w8a16".into();
        assert_ne!(m.layout_digest(), base);
        let mut m = manifest();
        m.twins_sha256 = "33".into();
        assert_ne!(m.layout_digest(), base);
        let mut m = manifest();
        m.kv_layers.pop();
        assert_ne!(m.layout_digest(), base);
    }
}
