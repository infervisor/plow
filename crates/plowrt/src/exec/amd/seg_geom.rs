//! Per-segment launch geometry (`PLOW_SEG_GEOM`): grid and object overrides per prefill segment.
//!
//! Spec: comma-separated `[T<bucket>:]<selector>=<value>[+<value>]`. A selector is a segment
//! family label (the `family=` of `PLOW_PREFILL_SEG_TIMING`, e.g. `interpreter`,
//! `flash_interpreter`, `kda_intra_wave_items`, `kda_wu_lean_keys`) or `s<index>`. Values:
//! `g<grid>` (workgroups) and `alt` (the same-named object from `PLOW_SEG_ALT_DIR`). Later entries
//! win. Unset = the shipped geometry, bit-identical.
//!
//! Grid overrides are legal only where the kernel strides its work over `gridDim`: the global-queue
//! interpreter (workgroups claim packets; the per-CU static streams need exactly `n_cu`) and the
//! grid-strided KDA intra / lean-Wu objects. Anything else is refused at load.

use crate::device::hsa::HsaKernel;
use crate::{Result, RuntimeError};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct SegGeom {
    /// 0 = shipped grid.
    pub grid: u32,
    pub alt: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct SegGeomRule {
    bucket: Option<u32>,
    family: Option<String>,
    seg: Option<usize>,
    geom: SegGeom,
}

pub(super) fn parse(spec: &str) -> Result<Vec<SegGeomRule>> {
    let bad =
        |e: &str, why: &str| RuntimeError::Device(format!("PLOW_SEG_GEOM entry `{e}`: {why}"));
    let mut rules = Vec::new();
    for ent in spec.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        let (lhs, rhs) = ent
            .split_once('=')
            .ok_or_else(|| bad(ent, "expected sel=value"))?;
        let (bucket, sel) = match lhs.split_once(':') {
            Some((b, s)) => {
                let b = b
                    .strip_prefix('T')
                    .ok_or_else(|| bad(ent, "bucket must be T<rows>"))?;
                (
                    Some(b.parse().map_err(|_| bad(ent, "bucket is not a u32"))?),
                    s,
                )
            }
            None => (None, lhs),
        };
        let seg = sel
            .strip_prefix('s')
            .filter(|n| !n.is_empty() && n.bytes().all(|c| c.is_ascii_digit()))
            .map(|n| n.parse::<usize>())
            .transpose()
            .map_err(|_| bad(ent, "segment index"))?;
        let family = seg.is_none().then(|| sel.to_string());
        let mut geom = SegGeom::default();
        for v in rhs.split('+') {
            if v == "alt" {
                geom.alt = true;
            } else if let Some(g) = v.strip_prefix('g') {
                geom.grid = g.parse().map_err(|_| bad(ent, "grid is not a u32"))?;
                if geom.grid == 0 {
                    return Err(bad(ent, "grid must be > 0"));
                }
            } else {
                return Err(bad(ent, "value must be g<grid> or alt"));
            }
        }
        rules.push(SegGeomRule {
            bucket,
            family,
            seg,
            geom,
        });
    }
    Ok(rules)
}

/// The merged override for one segment (`family` is the label of the path that launches it).
pub(super) fn resolve(rules: &[SegGeomRule], bucket: u32, seg: usize, family: &str) -> SegGeom {
    let mut out = SegGeom::default();
    for r in rules {
        if r.bucket.is_some_and(|b| b != bucket) {
            continue;
        }
        let hit = match (&r.family, r.seg) {
            (Some(f), _) => f == family,
            (None, Some(s)) => s == seg,
            (None, None) => false,
        };
        if hit {
            if r.geom.grid != 0 {
                out.grid = r.geom.grid;
            }
            out.alt |= r.geom.alt;
        }
    }
    out
}

/// Families whose kernels stride over `gridDim` (safe to relaunch at any grid).
pub(super) fn route_grid_free(family: &str) -> bool {
    matches!(
        family,
        "kda_intra_wave_items" | "kda_wu_lean" | "kda_wu_lean_keys"
    )
}

impl super::AmdEngine {
    #[inline]
    pub(super) fn seg_route_grid(&self, p: usize, seg: usize, grid: u32) -> u32 {
        match self.progs[p].seg_geom_route.get(seg) {
            Some(&g) if g != 0 => g,
            _ => grid,
        }
    }

    /// The interpreter object and grid for one prefill segment launch.
    #[inline]
    pub(super) fn seg_interp_geom(
        &self,
        p: usize,
        seg: usize,
        k: HsaKernel,
        family: &str,
    ) -> (HsaKernel, u32) {
        let g = self.progs[p]
            .seg_geom_interp
            .get(seg)
            .copied()
            .unwrap_or_default();
        let k = match (g.alt, family) {
            (true, "interpreter") => self.k_prefill_alt.unwrap_or(k),
            (true, "flash_interpreter") => self.k_flash_alt.unwrap_or(k),
            _ => k,
        };
        (k, if g.grid != 0 { g.grid } else { self.n_cu })
    }

    /// Resolve `PLOW_SEG_GEOM` into per-segment tables on every prefill-side program.
    pub(super) fn apply_seg_geom(&mut self, rules: &[SegGeomRule]) -> Result<()> {
        if rules.is_empty() {
            return Ok(());
        }
        let mut n_interp = 0usize;
        let mut n_route = 0usize;
        for p in 0..self.progs.len() {
            if !self.progs[p].role.is_prefill_side() {
                continue;
            }
            let bucket = self.prog_t(p);
            let n_seg = self.progs[p].seg_class.len();
            let mut interp = vec![SegGeom::default(); n_seg];
            let mut route = vec![0u32; n_seg];
            for seg in 0..n_seg {
                if let Ok((_, _, fam)) = self.prefill_interpreter_kernel(p, seg) {
                    let g = resolve(rules, bucket, seg, fam);
                    if g.grid != 0 && self.sched_prefill != super::Sched::GlobalQueue {
                        return Err(RuntimeError::Device(format!(
                            "PLOW_SEG_GEOM grid on T{bucket} segment {seg} ({fam}): static \
                             per-CU streams need exactly n_cu workgroups; use the GQ objects"
                        )));
                    }
                    // A placed program drains one window per physical XCD; a resized grid could
                    // leave a domain without workers.
                    if g.grid != 0 && self.progs[p].l2_domains != 0 {
                        return Err(RuntimeError::Device(format!(
                            "PLOW_SEG_GEOM grid on T{bucket} segment {seg} ({fam}): the program \
                             is L2-domain placed"
                        )));
                    }
                    if g.alt {
                        let have = match fam {
                            "interpreter" => self.k_prefill_alt.is_some(),
                            "flash_interpreter" => self.k_flash_alt.is_some(),
                            _ => false,
                        };
                        if !have {
                            return Err(RuntimeError::Device(format!(
                                "PLOW_SEG_GEOM alt on T{bucket} segment {seg} ({fam}): no \
                                 PLOW_SEG_ALT_DIR object for that family"
                            )));
                        }
                    }
                    n_interp += usize::from(g != SegGeom::default());
                    interp[seg] = g;
                }
                let fam = self.prefill_segment_family(p, seg);
                if !matches!(fam, "interpreter" | "flash_interpreter") {
                    let g = resolve(rules, bucket, seg, fam);
                    if g.alt {
                        return Err(RuntimeError::Device(format!(
                            "PLOW_SEG_GEOM alt on T{bucket} segment {seg}: {fam} has no alt object"
                        )));
                    }
                    if g.grid != 0 {
                        if !route_grid_free(fam) {
                            return Err(RuntimeError::Device(format!(
                                "PLOW_SEG_GEOM grid on T{bucket} segment {seg}: {fam} does not \
                                 stride over gridDim"
                            )));
                        }
                        route[seg] = g.grid;
                        n_route += 1;
                    }
                }
            }
            self.progs[p].seg_geom_interp = interp;
            self.progs[p].seg_geom_route = route;
        }
        tracing::info!(n_interp, n_route, "PLOW_SEG_GEOM per-segment overrides");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_and_resolve() {
        let r = parse("interpreter=g192, T8192:s14=g64+alt,flash_interpreter=alt").unwrap();
        assert_eq!(
            resolve(&r, 4096, 3, "interpreter"),
            SegGeom {
                grid: 192,
                alt: false
            }
        );
        assert_eq!(
            resolve(&r, 8192, 14, "interpreter"),
            SegGeom {
                grid: 64,
                alt: true
            }
        );
        assert_eq!(
            resolve(&r, 4096, 14, "interpreter"),
            SegGeom {
                grid: 192,
                alt: false
            }
        );
        assert_eq!(
            resolve(&r, 8192, 13, "flash_interpreter"),
            SegGeom { grid: 0, alt: true }
        );
        assert_eq!(resolve(&r, 8192, 1, "kda_wu_lean_keys"), SegGeom::default());
        assert!(parse("interpreter=g0").is_err());
        assert!(parse("interpreter=x2").is_err());
        assert!(parse("8192:s1=g4").is_err());
        assert!(parse("").unwrap().is_empty());
    }
}
