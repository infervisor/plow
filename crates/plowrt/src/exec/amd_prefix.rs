use crate::asset::devblob::DevTensor;
use crate::exec::kv_layout::RingWindow;
use packet::dev::{DevInst64, DevOp};

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Region {
    pub tensor: usize,
    pub slot_bytes: u64,
    heads: u32,
    ring: u32,
    window: u32,
    row_bytes: u64,
}
impl Region {
    pub fn max_copies(&self) -> usize {
        if self.ring == 0 {
            1
        } else {
            self.heads as usize * 2
        }
    }
    pub fn bytes(&self) -> u64 {
        if self.ring == 0 {
            self.slot_bytes
        } else {
            self.heads as u64 * self.window as u64 * self.row_bytes
        }
    }
    pub fn copy_spans(&self, rows: u32, mut copy: impl FnMut(u64, u64, u64)) {
        if self.ring == 0 {
            copy(0, 0, self.slot_bytes);
            return;
        }
        let span = RingWindow::new(rows.into(), self.window.into(), self.ring.into());
        for head in 0..self.heads as u64 {
            let dst = head * self.window as u64 * self.row_bytes;
            let src = (head * self.ring as u64 + span.start) * self.row_bytes;
            if span.first > 0 {
                copy(dst, src, span.first * self.row_bytes);
            }
            if span.rows > span.first {
                copy(
                    dst + span.first * self.row_bytes,
                    head * self.ring as u64 * self.row_bytes,
                    (span.rows - span.first) * self.row_bytes,
                );
            }
        }
    }
}
pub(super) fn snapshot_regions<'a>(
    tensors: &[DevTensor],
    instructions: impl Iterator<Item = &'a DevInst64> + Clone,
    batch: usize,
    max_ctx: usize,
) -> Option<Vec<Region>> {
    if batch == 0 || max_ctx == 0 {
        return None;
    }
    let mut known = vec![false; tensors.len()];
    let mut needed = vec![false; tensors.len()];
    let mut regions = Vec::new();
    for (i, t) in tensors.iter().enumerate() {
        if super::is_carried_state(&t.name) && !t.name.contains("blkres") {
            if t.bytes == 0 || t.bytes % batch as u64 != 0 {
                return None;
            }
            known[i] = true;
            regions.push(Region {
                tensor: i,
                slot_bytes: t.bytes / batch as u64,
                heads: 1,
                ring: 0,
                window: 0,
                row_bytes: 0,
            });
        }
    }
    for d in instructions.clone() {
        let dst = d.t[0] as usize;
        if !tensors.get(dst).is_some_and(|t| t.name.starts_with("kv.")) {
            continue;
        }
        if d.op == DevOp::HeadNormRope as u16 || d.op == DevOp::HeadNormRopeFp8 as u16 {
            known[dst] = true;
            needed[dst] |= d.fj[2] != u32::MAX && (d.fj[1] as usize) < max_ctx;
            if d.op == DevOp::HeadNormRopeFp8 as u16 {
                let scale = d.t[6] as usize;
                if !tensors
                    .get(scale)
                    .is_some_and(|t| t.name.starts_with("kv."))
                {
                    return None;
                }
                known[scale] = true;
                needed[scale] |= needed[dst];
            }
        } else if d.op == DevOp::RmsNorm as u16 {
            known[dst] = true;
        }
    }
    for d in instructions {
        let fp8 = d.op == DevOp::FlashDecodeFp8 as u16;
        if !fp8 && d.op != DevOp::FlashDecode as u16 {
            continue;
        }
        let nrf = !fp8 && d.i[1] & (1 << 16) != 0;
        let ring = if nrf {
            d.i[3] & ((1 << 20) - 1)
        } else {
            d.i[3]
        };
        let window = if nrf { d.i[0] >> 8 } else { d.i[4] };
        let heads = d.i[2];
        let hd = if nrf { d.i[6] & 0xffff } else { d.i[6] };
        for operand in if fp8 { &[3, 4, 6, 7][..] } else { &[3, 4][..] } {
            let index = d.t[*operand] as usize;
            let tensor = tensors.get(index)?;
            if !tensor.name.starts_with("kv.") {
                return None;
            }
            known[index] = true;
            if d.i[7] == u32::MAX || ring as usize >= max_ctx {
                continue;
            }
            if window == 0
                || window > ring
                || !ring.is_power_of_two()
                || d.i[7] != ring - 1
                || heads == 0
                || hd == 0
            {
                return None;
            }
            let row_bytes = if *operand >= 6 {
                4
            } else {
                hd as u64 * if fp8 { 1 } else { 2 }
            };
            let slot_bytes = (heads as u64)
                .checked_mul(ring as u64)?
                .checked_mul(row_bytes)?;
            if slot_bytes.checked_mul(batch as u64)? != tensor.bytes {
                return None;
            }
            let region = Region {
                tensor: index,
                slot_bytes,
                heads,
                ring,
                window,
                row_bytes,
            };
            if let Some(prior) = regions.iter().find(|r| r.tensor == index) {
                if prior != &region {
                    return None;
                }
            } else {
                regions.push(region);
            }
        }
    }
    for (i, t) in tensors.iter().enumerate() {
        if t.name.starts_with("kv.") && !t.name.contains("blkres") && !known[i] {
            return None;
        }
        if needed[i] && !regions.iter().any(|r| r.tensor == i) {
            return None;
        }
    }
    Some(regions)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn emitted_gemma_bf16_and_fp8_kv_keep_window_sized_snapshots_and_all_rungs() {
        if std::env::var_os("PLOW_PREFIX_EMIT_CHILD").is_none() {
            let out = std::process::Command::new(std::env::current_exe().unwrap())
                .args(["exec::amd::amd_prefix::tests::emitted_gemma_bf16_and_fp8_kv_keep_window_sized_snapshots_and_all_rungs",
                    "--exact", "--nocapture"])
                .env("PLOW_PREFIX_EMIT_CHILD", "1").output().unwrap();
            assert!(
                out.status.success(),
                "{}\n{}",
                String::from_utf8_lossy(&out.stdout),
                String::from_utf8_lossy(&out.stderr)
            );
            return;
        }
        let dir = std::env::temp_dir().join(format!("plow-prefix-emit-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("config.json"), r#"{
            "model_type":"gemma4_text", "hidden_size":512, "intermediate_size":1024,
            "num_hidden_layers":2, "num_attention_heads":8, "head_dim":64,
            "global_head_dim":64, "num_key_value_heads":2, "num_global_key_value_heads":2,
            "sliding_window":512, "rms_norm_eps":1e-6, "vocab_size":4096,
            "final_logit_softcapping":30.0, "tie_word_embeddings":true,
            "layer_types":["sliding_attention","full_attention"],
            "rope_parameters":{"sliding_attention":{"rope_theta":10000.0,"partial_rotary_factor":1.0},
                "full_attention":{"rope_theta":1000000.0,"partial_rotary_factor":1.0}}
        }"#).unwrap();
        for fp8 in [false, true] {
            let mut cfg = devgen::emit_config::EmitConfig::from_env();
            cfg.fp8_kv = fp8;
            cfg.decode_ladder = Some("1,2,4".into());
            cfg.max_chunk = Some(1024);
            let path = dir.join(format!("model-{fp8}.pkt"));
            devgen::run(devgen::EmitArgs {
                dir: dir.clone(),
                ctx: 4096,
                out: path.display().to_string(),
                n_cu: 128,
                tp: 1,
                block_spec: None,
                embed_cubin: None,
                embed_hsaco: None,
                rope_gen: true,
                l2_layout: None,
                gpu: String::new(),
                arch: "gfx942".into(),
                emit_cfg: Some(cfg),
                whole_graph_fusions: devgen::WholeGraphFusionDecisions::default(),
            });
            let blob =
                crate::asset::devblob::DevBlob::parse_l2(&std::fs::read(path).unwrap(), true)
                    .unwrap();
            assert_eq!(
                blob.decode_progs().iter().map(|p| p.t).collect::<Vec<_>>(),
                [1, 2, 4]
            );
            assert_eq!(
                blob.prefill_progs().iter().map(|p| p.t).collect::<Vec<_>>(),
                [128, 512, 1024]
            );
            let regions = snapshot_regions(
                &blob.tensors,
                blob.progs.iter().flat_map(|p| &p.insts),
                4,
                4096,
            )
            .expect("emitted Gemma prefix geometry");
            assert_eq!(
                regions.iter().map(Region::bytes).sum::<u64>(),
                if fp8 { 139264 } else { 262144 }
            );
        }
        std::fs::remove_dir_all(dir).unwrap();
    }
    fn tensor(name: &str, bytes: u64) -> DevTensor {
        DevTensor {
            name: name.into(),
            bytes,
            init: None,
        }
    }
    fn flash(fp8: bool) -> DevInst64 {
        let mut d = DevInst64::default();
        d.op = if fp8 {
            DevOp::FlashDecodeFp8
        } else {
            DevOp::FlashDecode
        } as u16;
        d.t[3] = 0;
        d.t[4] = 0;
        d.t[6] = 1;
        d.t[7] = 1;
        d.i[2] = 2;
        d.i[3] = 16;
        d.i[4] = 4;
        d.i[6] = 8;
        d.i[7] = 15;
        d
    }
    #[test]
    fn sliding_snapshot_includes_fp8_scales_and_deduplicates_aliases() {
        let tensors = [tensor("kv.0.k", 1024), tensor("kv.0.ks", 512)];
        let insts = [flash(true), flash(true)];
        let regions = snapshot_regions(&tensors, insts.iter(), 4, 4096).unwrap();
        assert_eq!(
            regions.iter().map(Region::bytes).collect::<Vec<_>>(),
            [64, 32]
        );
        assert_eq!(
            regions.iter().map(|r| r.slot_bytes).collect::<Vec<_>>(),
            [256, 128]
        );
        assert!(snapshot_regions(&tensors, insts.iter(), 4, 16)
            .unwrap()
            .is_empty());
    }
    #[test]
    fn folded_decode_fields_keep_the_same_snapshot_geometry() {
        let tensors = [tensor("kv.0.k", 2048)];
        let plain = flash(false);
        let mut folded = plain;
        folded.i[0] = 4 | (plain.i[4] << 8);
        folded.i[1] |= 1 << 16;
        folded.i[3] |= 8 << 20;
        folded.i[4] = 91;
        folded.i[6] |= 93 << 16;
        assert_eq!(
            snapshot_regions(&tensors, [plain].iter(), 4, 4096),
            snapshot_regions(&tensors, [folded].iter(), 4, 4096)
        );
    }

    #[test]
    fn unknown_kv_fractional_state_and_inconsistent_ring_refuse_reuse() {
        let mut tensors = vec![tensor("kv.0.k", 2048), tensor("kv.0.unknown_ring", 128)];
        let mut insts = [flash(false)];
        assert!(snapshot_regions(&tensors, insts.iter(), 4, 4096).is_none());
        tensors[1] = tensor("kv.0.state", 129);
        assert!(snapshot_regions(&tensors, insts.iter(), 4, 4096).is_none());
        tensors[1].bytes = 128;
        assert_eq!(
            snapshot_regions(&tensors, insts.iter(), 4, 4096)
                .unwrap()
                .len(),
            2
        );
        insts[0].i[3] = 8;
        assert!(snapshot_regions(&tensors, insts.iter(), 4, 4096).is_none());
    }
    #[test]
    fn window_copy_restores_exact_history_across_ring_wrap_without_other_rows() {
        let region = Region {
            tensor: 0,
            slot_bytes: 32,
            heads: 2,
            ring: 16,
            window: 4,
            row_bytes: 1,
        };
        for rows in [1_u32, 4, 15, 16, 17, 33] {
            let original: Vec<u8> = (0..32).collect();
            let mut snapshot = vec![0; region.bytes() as usize];
            region.copy_spans(rows, |dst, src, bytes| {
                snapshot[dst as usize..(dst + bytes) as usize]
                    .copy_from_slice(&original[src as usize..(src + bytes) as usize])
            });
            let mut restored = vec![255; 32];
            region.copy_spans(rows, |src, dst, bytes| {
                restored[dst as usize..(dst + bytes) as usize]
                    .copy_from_slice(&snapshot[src as usize..(src + bytes) as usize])
            });
            for head in 0..2 {
                for index in 0..16 {
                    let needed = (rows.saturating_sub(4)..rows).any(|row| row % 16 == index);
                    assert_eq!(
                        restored[(head * 16 + index) as usize],
                        if needed {
                            original[(head * 16 + index) as usize]
                        } else {
                            255
                        }
                    );
                }
            }
        }
    }
}
