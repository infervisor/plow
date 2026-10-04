use super::*;

pub(super) struct Native {
    be: Arc<CudaBackend>,
    _module: Arc<DecodeModule>,
    kernels: Vec<KernelFn>,
    reduce: KernelFn,
    workspace: DeviceMem,
    decode_plan: Vec<[u32; 5]>,
}

pub(super) struct Plan {
    owner: Arc<Native>,
    shape: [u32; 3],
    kernel: usize,
    splits: u32,
    smem: u32,
}

fn selection(plan: &[[u32; 5]], m: u32, n: u32, k: u32) -> Result<(usize, u32, u32)> {
    let &[_, _, _, bk, splits] = plan
        .iter()
        .find(|shape| shape[..3] == [m, n, k])
        .ok_or_else(|| {
            RuntimeError::Rejected(format!(
                "native decode has no measured BF16 shape M{m}/N{n}/K{k}"
            ))
        })?;
    let capacity = if m <= 8 {
        8
    } else if m <= 16 {
        16
    } else {
        32
    };
    let stages = if bk == 128 { 3 } else { 2 };
    let kernel = (usize::from(m > 8) + usize::from(m > 16)) * 2 + usize::from(bk == 256);
    Ok((kernel, splits, stages * (64 + capacity) * (bk + 8) * 2))
}

impl Native {
    pub(super) fn load(
        be: &Arc<CudaBackend>,
        assets: &Path,
        object: &SegmentObject,
        profile: &str,
        segments: &[Option<cublaslt::DecodeSegment>],
    ) -> Result<Arc<Self>> {
        let image = std::fs::read(assets.join(&object.file))
            .map_err(|e| RuntimeError::Rejected(format!("native decode object: {e}")))?;
        if profile != "sm90a"
            || plow_asset::cubin::inspect(&image).is_none_or(|i| i.sm != 90)
            || object.sha256.as_deref()
                != Some(plow_asset::decode_objects::image_sha256(&image).as_str())
            || plow_asset::cubin::global_u32(&image, "plow_gemv_transposed_abi") != Some(1)
            || plow_asset::cubin::global_u32(&image, "plow_gemv_transposed_block") != Some(128)
        {
            return Err(RuntimeError::Rejected(
                "native decode object hash or ABI mismatch".into(),
            ));
        }
        let wide = segments.iter().flatten().any(|s| s.m > 16);
        if wide
            && plow_asset::cubin::global_u32(&image, "plow_gemv_transposed_max_rows") != Some(32)
        {
            return Err(RuntimeError::Rejected(
                "native decode object has no B32 capability".into(),
            ));
        }
        let bytes = segments
            .iter()
            .flatten()
            .map(|s| {
                selection(object.decode_plan_or_legacy(), s.m, s.n, s.k)?;
                Ok(u64::from(s.m) * u64::from(s.n) * 8 * 4)
            })
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .max()
            .unwrap_or(0);
        if bytes == 0 {
            return Err(RuntimeError::Rejected(
                "native decode has no projections".into(),
            ));
        }
        let module = DecodeModule::load(be, &image)?;
        let mut kernels = Vec::new();
        for capacity in [8, 16, 32]
            .into_iter()
            .filter(|&capacity| capacity <= 16 || wide)
        {
            for (bk, stages) in [(128, 3), (256, 2)] {
                let f = be.get_function(
                    &module,
                    &format!("plow_gemv_bf16_m{capacity}_bk{bk}_s{stages}"),
                )?;
                be.set_max_dynamic_smem(f, stages * (64 + capacity) * (bk + 8) * 2)?;
                kernels.push(f);
            }
        }
        let reduce = be.get_function(&module, "plow_gemv_bf16_reduce")?;
        let workspace = be.alloc(0, bytes)?;
        tracing::info!(bytes, "native BF16 decode object loaded");
        Ok(Arc::new(Self {
            be: Arc::clone(be),
            _module: module,
            kernels,
            reduce,
            workspace,
            decode_plan: object.decode_plan_or_legacy().to_vec(),
        }))
    }

    pub(super) fn plan(
        self: &Arc<Self>,
        m: u32,
        n: u32,
        k: u32,
        template: Option<&Plan>,
    ) -> Result<Plan> {
        let (mut kernel, mut splits, mut smem) = selection(&self.decode_plan, m, n, k)?;
        if let Some(template) = template {
            if !Arc::ptr_eq(self, &template.owner)
                || template.shape[0] < m
                || template.shape[1..] != [n, k]
            {
                return Err(RuntimeError::Rejected(
                    "native decode rung template differs".into(),
                ));
            }
            // Rung changes must preserve the FP32 summation order.
            let bk = if template.kernel % 2 == 0 { 128 } else { 256 };
            let capacity = if m <= 8 {
                8
            } else if m <= 16 {
                16
            } else {
                32
            };
            let stages = if bk == 128 { 3 } else { 2 };
            kernel = (usize::from(m > 8) + usize::from(m > 16)) * 2 + template.kernel % 2;
            splits = template.splits;
            smem = stages * (64 + capacity) * (bk + 8) * 2;
        }
        if u64::from(m) * u64::from(n) * u64::from(splits) * 4 > self.workspace.len {
            return Err(RuntimeError::Rejected(
                "native decode scratch exceeds owner".into(),
            ));
        }
        Ok(Plan {
            owner: Arc::clone(self),
            shape: [m, n, k],
            kernel,
            splits,
            smem,
        })
    }
}

impl Plan {
    pub(super) fn run(
        &self,
        mut input: u64,
        mut weight: u64,
        mut output: u64,
        stream: &CudaStream,
    ) -> Result<()> {
        let [mut m, mut n, mut k] = self.shape;
        let mut splits = self.splits;
        let mut partial = self.owner.workspace.base;
        let mut params = [
            (&mut output as *mut u64).cast(),
            (&mut partial as *mut u64).cast(),
            (&mut input as *mut u64).cast(),
            (&mut weight as *mut u64).cast(),
            (&mut m as *mut u32).cast(),
            (&mut n as *mut u32).cast(),
            (&mut k as *mut u32).cast(),
            (&mut splits as *mut u32).cast(),
        ];
        self.owner.be.launch_kernel_grid(
            self.owner.kernels[self.kernel],
            [n.div_ceil(64), splits, 1],
            128,
            self.smem,
            &mut params,
            Some(stream),
        )?;
        if splits > 1 {
            let mut count = m * n;
            let mut params = [
                (&mut output as *mut u64).cast(),
                (&mut partial as *mut u64).cast(),
                (&mut count as *mut u32).cast(),
                (&mut splits as *mut u32).cast(),
            ];
            self.owner.be.launch_kernel(
                self.owner.reduce,
                count.div_ceil(256),
                256,
                0,
                &mut params,
                Some(stream),
            )?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn measured_shapes_have_bounded_launch_and_scratch() {
        let shapes = &plow_asset::segment_roles::NATIVE_DECODE_BF16_SHAPES;
        for &[m, n, k, _, _] in shapes {
            let (kernel, splits, smem) = selection(shapes, m, n, k).unwrap();
            assert!(kernel < 6 && [1, 4, 8].contains(&splits));
            assert!(smem <= 101376 && u64::from(m) * u64::from(n) <= i32::MAX as u64);
        }
        assert!(selection(shapes, 64, 15360, 3840).is_err());
        assert!(selection(shapes, 16, 15360, 3848).is_err());
        assert!(selection(shapes, 16, 262145, 3840).is_err());
    }

    #[test]
    fn descriptor_plan_replaces_the_measured_table() {
        let plan = [[8, 2048, 2048, 128, 1], [32, 2816, 4096, 256, 4]];
        assert_eq!(selection(&plan, 8, 2048, 2048).unwrap(), (0, 1, 3 * 72 * 136 * 2));
        assert_eq!(selection(&plan, 32, 2816, 4096).unwrap(), (5, 4, 2 * 96 * 264 * 2));
        assert!(selection(&plan, 8, 512, 3840).is_err());
    }
}

