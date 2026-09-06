use crate::decode_objects::DecodeObject;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub const SECTION: &str = "decode_context.json";

#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DecodeContexts {
    pub version: u32,
    pub kernarg_bytes: usize,
    pub bands: Vec<ContextBand>,
}

#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ContextBand {
    pub base_program: usize,
    pub rows: u32,
    pub kv_min: u32,
    pub kv_max: u32,
    pub program: ContextProgram,
    pub object: DecodeObject,
    pub capabilities: Vec<Capability>,
    pub qualification_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ContextProgram {
    pub file: String,
    pub sha256: String,
    pub index: usize,
}

#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Capability {
    pub symbol: String,
    pub value: u32,
}

// Structural validation is separate from materializing and qualifying the bound program.
#[derive(Debug)]
pub struct ContextTable {
    metadata: DecodeContexts,
    base_programs: Box<[(usize, u32)]>,
    max_kv: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ContextSelection {
    pub base_program: usize,
    pub band: Option<usize>,
}

fn digest(s: &str) -> bool {
    s.len() == 64
        && s.bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

fn filename(s: &str) -> bool {
    let mut parts = std::path::Path::new(s).components();
    !s.as_bytes().contains(&0)
        && matches!(parts.next(), Some(std::path::Component::Normal(_)))
        && parts.next().is_none()
}

pub fn parse_sections(
    sections: &[&[u8]],
    base_programs: &[(usize, u32)],
    grid: u32,
    kernarg_bytes: usize,
    max_kv: u32,
) -> Result<Option<ContextTable>, String> {
    let raw = match sections {
        [] => return Ok(None),
        [raw] if raw.len() <= 1024 * 1024 => raw,
        _ => return Err("duplicate or oversized decode context metadata".into()),
    };
    serde_json::from_slice::<DecodeContexts>(raw)
        .map_err(|e| e.to_string())?
        .validate(base_programs, grid, kernarg_bytes, max_kv)
        .map(Some)
}

impl DecodeContexts {
    pub fn validate(
        self,
        base_programs: &[(usize, u32)],
        grid: u32,
        kernarg_bytes: usize,
        max_kv: u32,
    ) -> Result<ContextTable, String> {
        if self.version != 1
            || self.kernarg_bytes != kernarg_bytes
            || kernarg_bytes == 0
            || max_kv == 0
            || grid == 0
            || self.bands.is_empty()
            || self.bands.len() > 1024
            || base_programs.is_empty()
            || base_programs
                .iter()
                .any(|&(_, rows)| rows == 0 || rows > 128)
            || base_programs
                .windows(2)
                .any(|w| w[0].0 >= w[1].0 || w[0].1 >= w[1].1)
        {
            return Err("unsupported decode context metadata or base ladder".into());
        }
        let mut files = BTreeMap::new();
        let mut objects = BTreeMap::new();
        let mut previous: Option<&ContextBand> = None;
        for band in &self.bands {
            if !base_programs.contains(&(band.base_program, band.rows))
                || band.kv_min == 0
                || band.kv_min > band.kv_max
                || band.kv_max > max_kv
                || !filename(&band.program.file)
                || !digest(&band.program.sha256)
                || !digest(&band.qualification_sha256)
                || band.capabilities.is_empty()
                || band.capabilities.len() > 32
            {
                return Err("invalid context band, program identity or qualification".into());
            }
            if previous.is_some_and(|p| {
                p.base_program > band.base_program
                    || (p.base_program == band.base_program && p.kv_max >= band.kv_min)
            }) {
                return Err("context bands must be ordered and non-overlapping per rung".into());
            }
            previous = Some(band);
            band.object.validate(grid)?;
            if !filename(&band.object.file)
                || band.object.profile != "sm90a"
                || band.object.entry != "_Z12interp_sm90a11PlowProgram"
                || band.object.threads != 256
            {
                return Err("context object requires the Hopper cooperative decode ABI".into());
            }
            let mut caps = BTreeMap::new();
            for cap in &band.capabilities {
                if !cap.symbol.starts_with("plow_")
                    || cap.symbol.len() > 96
                    || !cap
                        .symbol
                        .bytes()
                        .all(|b| b.is_ascii_alphanumeric() || b == b'_')
                    || caps.insert(cap.symbol.as_str(), cap.value).is_some()
                {
                    return Err("invalid or duplicate initialized capability".into());
                }
            }
            // QK: 1=FP32 CUDA, 2=BF16 MMA with K64 FP32 partial promotion.
            // PV: 1=FP32 CUDA, 2=two TF32 hi/lo contributions; BF16-P rounding is absent.
            if caps.get("plow_attention_decode_abi") != Some(&1)
                || !matches!(
                    (
                        caps.get("plow_attention_qk_abi"),
                        caps.get("plow_attention_pv_abi")
                    ),
                    (Some(1), Some(1)) | (Some(2), Some(1)) | (Some(2), Some(2))
                )
            {
                return Err("missing or unsupported arithmetic capability".into());
            }
            for (file, hash, kind) in [
                (&band.program.file, &band.program.sha256, 0),
                (&band.object.file, &band.object.sha256, 1),
            ] {
                if files
                    .insert(file.as_str(), (hash.as_str(), kind))
                    .is_some_and(|old| old != (hash.as_str(), kind))
                {
                    return Err("conflicting context file identity".into());
                }
            }
            if objects
                .insert(band.object.file.as_str(), (&band.object, caps.clone()))
                .is_some_and(|old| old != (&band.object, caps))
            {
                return Err("conflicting context object contract".into());
            }
        }
        Ok(ContextTable {
            metadata: self,
            base_programs: base_programs.into(),
            max_kv,
        })
    }
}

impl ContextTable {
    pub fn bands(&self) -> &[ContextBand] {
        &self.metadata.bands
    }

    pub fn check_object_image(&self, band: usize, image: &[u8]) -> Result<(), String> {
        let band = self
            .metadata
            .bands
            .get(band)
            .ok_or("context band out of range")?;
        if !band.object.matches_image(image) {
            return Err("context object SHA256 mismatch".into());
        }
        let info = crate::cubin::inspect(image).ok_or("invalid context object ELF")?;
        if info.sm != 90 || !info.entries.contains(&band.object.entry) {
            return Err("context object ISA/entry mismatch".into());
        }
        for (symbol, value) in [
            ("plow_block", band.object.threads),
            ("plow_arena_bytes", band.object.arena_bytes),
            ("plow_dyn_kvrow", 1),
            ("plow_segment_gq_abi", 1),
        ] {
            if crate::cubin::global_u32(image, symbol) != Some(value) {
                return Err(format!("context object resource mismatch: {symbol}"));
            }
        }
        for cap in &band.capabilities {
            if crate::cubin::global_u32(image, &cap.symbol) != Some(cap.value) {
                return Err(format!(
                    "context arithmetic capability mismatch: {}",
                    cap.symbol
                ));
            }
        }
        Ok(())
    }

    pub fn check_program_image(&self, band: usize, image: &[u8]) -> Result<(), String> {
        let band = self
            .metadata
            .bands
            .get(band)
            .ok_or("context band out of range")?;
        if crate::decode_objects::image_sha256(image) != band.program.sha256 {
            return Err("context auxiliary packet SHA256 mismatch".into());
        }
        Ok(())
    }

    pub fn select(
        &self,
        positions: &[u32],
        slots: impl IntoIterator<Item = usize>,
    ) -> Result<ContextSelection, &'static str> {
        let mut seen = 0u128;
        let mut highest = 0usize;
        let mut min_kv = u32::MAX;
        let mut max_kv = 0;
        let widest = self
            .base_programs
            .last()
            .expect("validated nonempty ladder")
            .1 as usize;
        for slot in slots {
            if slot >= widest || slot >= positions.len() || seen & (1u128 << slot) != 0 {
                return Err("invalid or duplicate live decode slot");
            }
            seen |= 1u128 << slot;
            let kv = positions[slot]
                .checked_add(1)
                .ok_or("live KV requirement overflow")?;
            if kv > self.max_kv {
                return Err("live KV requirement exceeds compiled capacity");
            }
            highest = highest.max(slot);
            min_kv = min_kv.min(kv);
            max_kv = max_kv.max(kv);
        }
        if seen == 0 {
            return Err("empty live decode selection");
        }
        let &(base_program, rows) = self
            .base_programs
            .iter()
            .find(|&&(_, rows)| rows as usize > highest)
            .expect("validated physical slot range");
        let dense_mask = u128::MAX >> (128 - rows);
        let band = (seen == dense_mask)
            .then(|| {
                self.metadata.bands.iter().position(|b| {
                    b.base_program == base_program && b.kv_min <= min_kv && max_kv <= b.kv_max
                })
            })
            .flatten();
        Ok(ContextSelection { base_program, band })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decode_objects::DecodeObject;

    fn band(rows: u32, lo: u32, hi: u32) -> ContextBand {
        ContextBand {
            base_program: match rows {
                1 => 2,
                4 => 3,
                _ => 4,
            },
            rows,
            kv_min: lo,
            kv_max: hi,
            program: ContextProgram {
                file: "attention.pkt".into(),
                sha256: "b".repeat(64),
                index: 3,
            },
            object: DecodeObject {
                file: "attention.cubin".into(),
                sha256: "c".repeat(64),
                profile: "sm90a".into(),
                entry: "_Z12interp_sm90a11PlowProgram".into(),
                threads: 256,
                arena_bytes: 126592,
                grid: 132,
            },
            capabilities: vec![
                Capability {
                    symbol: "plow_attention_decode_abi".into(),
                    value: 1,
                },
                Capability {
                    symbol: "plow_attention_qk_abi".into(),
                    value: 2,
                },
                Capability {
                    symbol: "plow_attention_pv_abi".into(),
                    value: 2,
                },
            ],
            qualification_sha256: "d".repeat(64),
        }
    }
    fn metadata() -> DecodeContexts {
        DecodeContexts {
            version: 1,
            kernarg_bytes: 192,
            bands: vec![
                band(1, 16000, 16002),
                band(1, 32000, 32002),
                band(4, 8000, 8002),
                band(16, 8000, 8002),
            ],
        }
    }
    fn table() -> ContextTable {
        metadata()
            .validate(&[(2, 1), (3, 4), (4, 16)], 132, 192, 65536)
            .unwrap()
    }
    fn valid(m: DecodeContexts) -> bool {
        m.validate(&[(2, 1), (3, 4), (4, 16)], 132, 192, 65536)
            .is_ok()
    }

    fn object_image(qk: u32) -> Vec<u8> {
        let names = [
            "_Z12interp_sm90a11PlowProgram",
            "plow_block",
            "plow_arena_bytes",
            "plow_dyn_kvrow",
            "plow_segment_gq_abi",
            "plow_attention_decode_abi",
            "plow_attention_qk_abi",
            "plow_attention_pv_abi",
        ];
        let mut strings = vec![0u8];
        let mut symbols = vec![];
        for (i, name) in names.iter().enumerate() {
            let mut sym = [0u8; 24];
            sym[..4].copy_from_slice(&(strings.len() as u32).to_le_bytes());
            sym[4] = if i == 0 { 0x12 } else { 0x11 };
            sym[6..8].copy_from_slice(&3u16.to_le_bytes());
            sym[8..16].copy_from_slice(&((i.saturating_sub(1) * 4) as u64).to_le_bytes());
            sym[16..24].copy_from_slice(&4u64.to_le_bytes());
            symbols.extend_from_slice(&sym);
            strings.extend_from_slice(name.as_bytes());
            strings.push(0);
        }
        let mut image = vec![0u8; 320];
        image[..6].copy_from_slice(b"\x7fELF\x02\x01");
        image[40..48].copy_from_slice(&64u64.to_le_bytes());
        image[48..52].copy_from_slice(&(90u32 << 8).to_le_bytes());
        image[58..60].copy_from_slice(&64u16.to_le_bytes());
        image[60..62].copy_from_slice(&4u16.to_le_bytes());
        let values = [256u32, 126592, 1, 1, 1, qk, 2]
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect();
        for (i, kind, data) in [(1, 2u32, symbols), (2, 3, strings), (3, 1, values)] {
            let h = 64 + i * 64;
            image[h + 4..h + 8].copy_from_slice(&kind.to_le_bytes());
            let ofs = image.len() as u64;
            image[h + 24..h + 32].copy_from_slice(&ofs.to_le_bytes());
            image[h + 32..h + 40].copy_from_slice(&(data.len() as u64).to_le_bytes());
            image.extend_from_slice(&data);
        }
        image[168..172].copy_from_slice(&2u32.to_le_bytes());
        image[184..192].copy_from_slice(&24u64.to_le_bytes());
        image
    }
    #[test]
    fn actual_images_not_only_metadata_are_bound() {
        let image = object_image(2);
        let packet = b"immutable auxiliary packet";
        let mut m = metadata();
        for b in &mut m.bands {
            b.object.sha256 = crate::decode_objects::image_sha256(&image);
            b.program.sha256 = crate::decode_objects::image_sha256(packet);
        }
        let t = m
            .clone()
            .validate(&[(2, 1), (3, 4), (4, 16)], 132, 192, 65536)
            .unwrap();
        assert!(t.check_object_image(0, &image).is_ok());
        assert!(t.check_program_image(0, packet).is_ok());
        assert!(t.check_object_image(4, &image).is_err());
        assert!(t.check_program_image(0, b"changed").is_err());
        let wrong = object_image(1);
        assert!(t.check_object_image(0, &wrong).is_err());
        for b in &mut m.bands {
            b.object.sha256 = crate::decode_objects::image_sha256(&wrong);
        }
        let t = m
            .validate(&[(2, 1), (3, 4), (4, 16)], 132, 192, 65536)
            .unwrap();
        assert!(t
            .check_object_image(0, &wrong)
            .unwrap_err()
            .contains("arithmetic"));
    }

    #[test]
    fn absent_and_duplicate_sections_are_distinct() {
        assert!(parse_sections(&[], &[(2, 1)], 132, 192, 65536)
            .unwrap()
            .is_none());
        let raw = serde_json::to_vec(&metadata()).unwrap();
        assert!(
            parse_sections(&[&raw], &[(2, 1), (3, 4), (4, 16)], 132, 192, 65536)
                .unwrap()
                .is_some()
        );
        assert!(
            parse_sections(&[&raw, &raw], &[(2, 1), (3, 4), (4, 16)], 132, 192, 65536).is_err()
        );
        assert!(parse_sections(&[b"{}"], &[(2, 1)], 132, 192, 65536).is_err());
    }
    #[test]
    fn actual_positions_sparse_slots_reset_and_interval_crossing() {
        let t = table();
        let mut pos = [0u32; 16];
        pos[0] = 15999;
        assert_eq!(
            t.select(&pos, [0]).unwrap(),
            ContextSelection {
                base_program: 2,
                band: Some(0)
            }
        );
        pos[0] = 16001;
        assert_eq!(t.select(&pos, [0]).unwrap().band, Some(0));
        pos[0] = 16002;
        assert_eq!(t.select(&pos, [0]).unwrap().band, None);
        pos[0] = 0;
        assert_eq!(t.select(&pos, [0]).unwrap().band, None);
        pos[3] = 7999;
        assert_eq!(
            t.select(&pos, [3]).unwrap(),
            ContextSelection {
                base_program: 3,
                band: None
            }
        );
        pos[15] = 7999;
        assert_eq!(
            t.select(&pos, [15]).unwrap(),
            ContextSelection {
                base_program: 4,
                band: None
            }
        );
        pos[0] = 15999;
        assert_eq!(t.select(&pos, [0]).unwrap().band, Some(0));
        assert_eq!(t.select(&pos, [0, 3]).unwrap().band, None);
        pos[..4].fill(8001);
        assert_eq!(t.select(&pos, 0..4).unwrap().band, Some(2));
        assert_eq!(t.select(&pos, [0, 3]).unwrap().band, None);
        pos[1] = 15999;
        assert_eq!(t.select(&pos, 0..4).unwrap().band, None);
        pos[0] = 15999;
        pos[15] = u32::MAX;
        assert_eq!(t.select(&pos, [0]).unwrap().band, Some(0));
    }
    #[test]
    fn adjacent_intervals_and_widest_mask_have_exact_edges() {
        let mut m = metadata();
        m.bands[1].kv_min = 16003;
        let t = m
            .validate(&[(2, 1), (3, 4), (4, 16)], 132, 192, 65536)
            .unwrap();
        let mut pos = [0u32; 16];
        pos[0] = 16001;
        assert_eq!(t.select(&pos, [0]).unwrap().band, Some(0));
        pos[0] = 16002;
        assert_eq!(t.select(&pos, [0]).unwrap().band, Some(1));
        let m = DecodeContexts {
            version: 1,
            kernarg_bytes: 192,
            bands: vec![band(128, 8000, 8002)],
        };
        let t = m.validate(&[(4, 128)], 132, 192, 65536).unwrap();
        let pos = [7999u32; 128];
        assert_eq!(t.select(&pos, 0..128).unwrap().band, Some(0));
        assert_eq!(t.select(&pos, [127]).unwrap().band, None);
        assert!(t.select(&pos, [128]).is_err());
    }

    #[test]
    fn invalid_live_feeds_reject() {
        let t = table();
        let mut pos = [0u32; 16];
        assert!(t.select(&pos, []).is_err());
        assert!(t.select(&pos, [0, 0]).is_err());
        assert!(t.select(&pos, [16]).is_err());
        assert!(t.select(&pos, [usize::MAX]).is_err());
        assert!(t.select(&pos[..2], [3]).is_err());
        pos[0] = u32::MAX;
        assert!(t.select(&pos, [0]).is_err());
        pos[0] = 65536;
        assert!(t.select(&pos, [0]).is_err());
    }
    #[test]
    fn malformed_band_geometry_and_base_ladder_reject() {
        let mut cases = vec![];
        let mut m = metadata();
        m.version = 2;
        cases.push(m);
        let mut m = metadata();
        m.kernarg_bytes = 184;
        cases.push(m);
        let mut m = metadata();
        m.bands.clear();
        cases.push(m);
        let mut m = metadata();
        m.bands[0].rows = 2;
        cases.push(m);
        let mut m = metadata();
        m.bands[0].base_program = 0;
        cases.push(m);
        let mut m = metadata();
        m.bands[0].kv_min = 0;
        cases.push(m);
        let mut m = metadata();
        m.bands[0].kv_max = 15999;
        cases.push(m);
        let mut m = metadata();
        m.bands[0].kv_max = 65537;
        cases.push(m);
        let mut m = metadata();
        m.bands[1].kv_min = 16002;
        cases.push(m);
        let mut m = metadata();
        m.bands.swap(0, 1);
        cases.push(m);
        for m in cases {
            assert!(!valid(m));
        }
        assert!(metadata()
            .validate(&[(2, 1), (3, 4), (4, 4)], 132, 192, 65536)
            .is_err());
        assert!(metadata()
            .validate(&[(3, 1), (2, 4), (4, 16)], 132, 192, 65536)
            .is_err());
        assert!(metadata()
            .validate(&[(2, 1), (3, 4), (4, 256)], 132, 192, 65536)
            .is_err());
    }
    #[test]
    fn identities_capabilities_and_conflicting_filenames_reject() {
        for field in [
            "program_path",
            "program_hash",
            "object_hash",
            "object_grid",
            "object_arena",
            "evidence",
            "cap_missing",
            "cap_duplicate",
            "cap_name",
            "file_collision",
            "same_file_changed_hash",
            "same_object_changed_resource",
        ] {
            let mut m = metadata();
            match field {
                "program_path" => m.bands[0].program.file = "../x.pkt".into(),
                "program_hash" => m.bands[0].program.sha256 = "X".repeat(64),
                "object_hash" => m.bands[0].object.sha256 = "bad".into(),
                "object_grid" => m.bands[0].object.grid = 264,
                "object_arena" => m.bands[0].object.arena_bytes = 0,
                "evidence" => m.bands[0].qualification_sha256.clear(),
                "cap_missing" => m.bands[0].capabilities.clear(),
                "cap_duplicate" => {
                    let cap = m.bands[0].capabilities[0].clone();
                    m.bands[0].capabilities.push(cap)
                }
                "cap_name" => m.bands[0].capabilities[0].symbol = "not an ABI".into(),
                "file_collision" => m.bands[0].program.file = m.bands[0].object.file.clone(),
                "same_file_changed_hash" => m.bands[0].program.sha256 = "a".repeat(64),
                "same_object_changed_resource" => m.bands[0].object.arena_bytes += 64,
                _ => unreachable!(),
            }
            assert!(!valid(m), "{field}");
        }
    }
    #[test]
    fn json_rejects_unknown_and_duplicate_fields() {
        let raw = serde_json::to_string(&metadata()).unwrap();
        let duplicate = raw.replacen("\"version\":1", "\"version\":1,\"version\":1", 1);
        assert!(serde_json::from_str::<DecodeContexts>(&duplicate).is_err());
        let mut v = serde_json::to_value(metadata()).unwrap();
        v["bands"][0]["runtime_model"] = serde_json::json!("x");
        assert!(serde_json::from_value::<DecodeContexts>(v).is_err());
    }
}
