//! Qwen-Image VAE encoder (`AutoencoderKLQwenImage`) → symbolic operator graph.
//!
//! A Wan-style 3D causal autoencoder. The encoder is a Conv3d stem, a stack of
//! resnet down-blocks (GroupNorm + SiLU + Conv3d) that halve spatial resolution
//! `len(dim_mult)-1` times, a middle block, and a Conv3d head producing
//! `2 * z_dim` channels (the mean/logvar of the latent distribution).
//!
//! Spatial dims are compiled per shape bucket (static), as is standard for
//! diffusion; the batch axis stays symbolic. We encode a single image frame
//! (temporal length 1).

use super::config::{parse_dtype, QwenImageVaeConfig};
use super::ShapeBucket;
use crate::op::ActKind;
use crate::Nn;
use crate::{Dim, Graph, TensorId};

/// GroupNorm group count (Wan VAE uses 32; shape-preserving regardless).
const GROUPS: u32 = 32;

pub fn build(cfg: &QwenImageVaeConfig, bucket: &ShapeBucket) -> Graph {
    let dt = parse_dtype(cfg.torch_dtype.as_deref());
    let mut nn = Nn::new(dt, dt);

    let in_c = 3;
    let stage_dims = cfg.stage_dims();
    let eps = 1e-6f32;

    let b = nn.sym("B");
    // [B, C, T=1, H, W] at the bucket's resolution; the down-block conv strides
    // derive the latent resolution exactly.
    let x = nn.input(
        "pixel_values",
        nn.shape([
            b.clone(),
            Dim::stat(in_c),
            Dim::stat(1),
            Dim::stat(bucket.image_height),
            Dim::stat(bucket.image_width),
        ]),
        dt,
    );

    // Conv stem.
    let mut h = nn.conv3d(
        "encoder.conv_in",
        x,
        in_c,
        cfg.base_dim,
        (3, 3, 3),
        (1, 1, 1),
        (1, 1, 1),
        true,
    );
    let mut cur = cfg.base_dim;

    // Down-blocks.
    let n_stages = stage_dims.len();
    for (i, &out_dim) in stage_dims.iter().enumerate() {
        nn.begin_block(&format!("encoder.down.{i}"));
        for r in 0..cfg.num_res_blocks {
            h = resnet_block(
                &mut nn,
                &format!("encoder.down.{i}.res.{r}"),
                h,
                cur,
                out_dim,
                eps,
            );
            cur = out_dim;
        }
        // Downsample between stages (not after the last).
        if i < n_stages - 1 {
            let t_stride = if *cfg.temperal_downsample.get(i).unwrap_or(&false) {
                2
            } else {
                1
            };
            h = nn.conv3d(
                &format!("encoder.down.{i}.downsample"),
                h,
                cur,
                cur,
                (3, 3, 3),
                (t_stride, 2, 2),
                (1, 1, 1),
                true,
            );
        }
    }
    nn.end_block();

    // Middle block (attn_scales empty ⇒ no attention).
    h = resnet_block(&mut nn, "encoder.mid.0", h, cur, cur, eps);
    h = resnet_block(&mut nn, "encoder.mid.1", h, cur, cur, eps);

    // Output head: GroupNorm + SiLU + Conv3d → 2*z_dim (mean | logvar).
    h = nn.groupnorm("encoder.norm_out", h, cur, GROUPS, eps);
    h = nn.act(ActKind::Silu, h);
    let moments = nn.conv3d(
        "encoder.conv_out",
        h,
        cur,
        2 * cfg.z_dim,
        (3, 3, 3),
        (1, 1, 1),
        (1, 1, 1),
        true,
    );

    nn.mark_output(moments);
    nn.finish()
}

/// GroupNorm→SiLU→Conv3d ×2 with a (projected) residual.
fn resnet_block(nn: &mut Nn, p: &str, x: TensorId, in_c: i64, out_c: i64, eps: f32) -> TensorId {
    let mut h = nn.groupnorm(&format!("{p}.norm1"), x, in_c, GROUPS, eps);
    h = nn.act(ActKind::Silu, h);
    h = nn.conv3d(
        &format!("{p}.conv1"),
        h,
        in_c,
        out_c,
        (3, 3, 3),
        (1, 1, 1),
        (1, 1, 1),
        true,
    );
    h = nn.groupnorm(&format!("{p}.norm2"), h, out_c, GROUPS, eps);
    h = nn.act(ActKind::Silu, h);
    h = nn.conv3d(
        &format!("{p}.conv2"),
        h,
        out_c,
        out_c,
        (3, 3, 3),
        (1, 1, 1),
        (1, 1, 1),
        true,
    );

    let shortcut = if in_c != out_c {
        nn.conv3d(
            &format!("{p}.conv_shortcut"),
            x,
            in_c,
            out_c,
            (1, 1, 1),
            (1, 1, 1),
            (0, 0, 0),
            true,
        )
    } else {
        x
    };
    nn.add(h, shortcut)
}
