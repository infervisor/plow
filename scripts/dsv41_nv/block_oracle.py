"""DeepSeek-V4.1 one-block oracle: the reference `model.Block(layer)` (inference/model.py, kernel.py) on a
prefill chunk, for parity against a plowc `--block` packet run by `block_run check`.

  gpulease -n 1 dsv41-oracle <venv-python> scripts/dsv41_nv/block_oracle.py ref  <layer> <T> <outdir>
  <venv-python> scripts/dsv41_nv/block_oracle.py cmp <outdir>/ref.npy <plow out.npy>
  gpulease ... block_oracle.py ref_decode <layer> <len,len,..> <steps> <outdir> [attn]

`ref` writes <outdir>/x.npy (the block input) and <outdir>/ref.npy (its output), both
[T * hc_mult, dim] f32 in the packet's token-major `act.hc_residual_a` layout ([T][hc][dim], what
HyperConnPre reads; block.json's [hc, T, dim] shape lists the same element count). The input is real
token embeddings expanded to hc_mult copies with the identity pre-mix, i.e. what the block sees as
the first layer of a model -- the packet's attention HyperConnPre runs in SEED mode for the same
reason.
"""
import json
import os
import sys

import numpy as np
import torch

REF = os.environ.get(
    "DSV41_REF",
    "/root/dsv41/hf/hub/models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots/dba1be0a40aa45a94ad051997016db3960a90277/inference",
)
CKPT = os.path.dirname(REF)


def npy_write(path, a):
    np.save(path, np.ascontiguousarray(a, dtype=np.float32))


def ref(layer, t, outdir):
    sys.path.insert(0, REF)
    import model as mr
    from safetensors import safe_open

    os.makedirs(outdir, exist_ok=True)
    cfg = json.load(open(os.path.join(REF, "config.json")))
    cfg["max_batch_size"] = 1
    cfg["max_seq_len"] = max(t, 8192)
    args = mr.ModelArgs(**{k: v for k, v in cfg.items() if k in mr.ModelArgs.__dataclass_fields__})
    mr.world_size, mr.rank = 1, 0
    mr.default_dtype = torch.float8_e4m3fn if args.dtype == "fp8" else torch.bfloat16
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("cuda")
    torch.manual_seed(0)
    blk = mr.Block(layer, args)
    index = json.load(open(os.path.join(CKPT, "model.safetensors.index.json")))["weight_map"]
    prefix = f"layers.{layer}."
    by_file = {}
    for name, f in index.items():
        if name.startswith(prefix):
            by_file.setdefault(f, []).append(name)
    params = dict(blk.named_parameters())
    loaded = 0
    dtypes = {}
    for f, names in by_file.items():
        with safe_open(os.path.join(CKPT, f), framework="pt", device="cuda") as sf:
            for name in names:
                key = name[len(prefix):]
                if key not in params:
                    continue
                p, src = params[key], sf.get_tensor(name)
                dtypes[key] = (str(src.dtype), str(p.dtype))
                if src.dtype == torch.float8_e4m3fn and p.dtype in (torch.bfloat16, torch.float32):
                    # convert.py: a block-fp8 weight the model keeps in bf16 (wo_a) is dequantized
                    # with its [32, 32] ue8m0 grid
                    sc = sf.get_tensor(name.replace(".weight", ".scale")).float()
                    bo, bi = src.shape[0] // sc.shape[0], src.shape[1] // sc.shape[1]
                    src = (src.float().unflatten(0, (-1, bo)).unflatten(-1, (-1, bi)) * sc[:, None, :, None]).flatten(2, 3).flatten(0, 1)
                    src = src.to(p.dtype)
                elif src.dtype != p.dtype:
                    src = src.view(p.dtype) if src.element_size() == p.element_size() else src.to(p.dtype)
                p.data.copy_(src.view(p.shape))
                loaded += 1
    missing = [k for k in params if not any(n[len(prefix):] == k for ns in by_file.values() for n in ns)]
    print({k: v for k, v in dtypes.items() if "experts." not in k or "experts.0." in k})
    print(f"loaded {loaded} tensors; unloaded params: {missing[:8]}{'...' if len(missing) > 8 else ''}")
    with safe_open(os.path.join(CKPT, index["embed.weight"]), framework="pt", device="cuda") as sf:
        emb = sf.get_tensor("embed.weight")
    ids = torch.randint(1000, 100000, (t,), device="cuda")
    h = emb[ids].to(torch.bfloat16).view(1, t, 1, -1).repeat(1, 1, args.hc_mult, 1)
    stages = {}
    orig_sparse = mr.sparse_attn

    def sparse_rec(q, kv, sink, idx, scale):
        o = orig_sparse(q, kv, sink, idx, scale)
        stages["sparse_q"], stages["sparse_kv"], stages["sparse_idx"] = q.detach().clone(), kv.detach().clone(), idx.detach().clone()
        stages["sparse_o"] = o.detach().clone()
        return o

    mr.sparse_attn = sparse_rec
    blk.attn.wo_b.register_forward_hook(lambda m, i, o: stages.__setitem__("wo_b_in", i[0].detach().clone()))
    for name in ("attn_norm", "attn", "ffn_norm", "ffn", "attn.wq_a", "attn.q_norm", "attn.wq_b", "attn.wkv", "attn.wo_a", "attn.wo_b"):
        blk.get_submodule(name).register_forward_hook(lambda m, i, o, name=name: stages.__setitem__(name, o.detach().clone()))
    with torch.inference_mode():
        out, _ = blk(h, 0, mr.make_identity_pre_mix(h, args.hc_mult), None)
    for name, o in stages.items():
        a = o.reshape(-1, o.shape[-1])
        np.save(os.path.join(outdir, f"stage_{name}.npy"), a.cpu().numpy() if a.dtype == torch.int32 else a.float().cpu().numpy())
    rows = lambda a: a[0].reshape(-1, a.shape[-1]).float().cpu().numpy()  # [T][hc][dim], token-major
    ffn_in = os.environ.get("ORACLE_FFN_IN")  # the packet's ffn_xn (raw bf16 [T][dim]): reference MoE on it
    if ffn_in:
        xin = torch.frombuffer(bytearray(open(ffn_in, "rb").read()), dtype=torch.bfloat16)[: t * args.dim]
        with torch.inference_mode():
            y = blk.ffn(xin.view(1, t, args.dim).cuda(), None)
        npy_write(os.path.join(outdir, "moe_on_plow_in.npy"), y.reshape(t, -1).float().cpu().numpy())
    npy_write(os.path.join(outdir, "x.npy"), rows(h))
    npy_write(os.path.join(outdir, "ref.npy"), rows(out))
    print(f"wrote {outdir}/x.npy and ref.npy: T={t} hc={args.hc_mult} dim={args.dim}")


def load_block(layer, max_seq):
    sys.path.insert(0, REF)
    import model as mr
    from safetensors import safe_open

    cfg = json.load(open(os.path.join(REF, "config.json")))
    cfg["max_batch_size"] = 1
    cfg["max_seq_len"] = max_seq
    args = mr.ModelArgs(**{k: v for k, v in cfg.items() if k in mr.ModelArgs.__dataclass_fields__})
    mr.world_size, mr.rank = 1, 0
    mr.default_dtype = torch.float8_e4m3fn if args.dtype == "fp8" else torch.bfloat16
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("cuda")
    blk = mr.Block(layer, args)
    index = json.load(open(os.path.join(CKPT, "model.safetensors.index.json")))["weight_map"]
    prefix = f"layers.{layer}."
    params = dict(blk.named_parameters())
    for f in sorted({f for n, f in index.items() if n.startswith(prefix)}):
        with safe_open(os.path.join(CKPT, f), framework="pt", device="cuda") as sf:
            for name in sf.keys():
                key = name[len(prefix):]
                if not name.startswith(prefix) or key not in params:
                    continue
                p, src = params[key], sf.get_tensor(name)
                if src.dtype == torch.float8_e4m3fn and p.dtype in (torch.bfloat16, torch.float32):
                    sc = sf.get_tensor(name.replace(".weight", ".scale")).float()
                    bo, bi = src.shape[0] // sc.shape[0], src.shape[1] // sc.shape[1]
                    src = (src.float().unflatten(0, (-1, bo)).unflatten(-1, (-1, bi)) * sc[:, None, :, None]).flatten(2, 3).flatten(0, 1)
                    src = src.to(p.dtype)
                elif src.dtype != p.dtype:
                    src = src.view(p.dtype) if src.element_size() == p.element_size() else src.to(p.dtype)
                p.data.copy_(src.view(p.shape))
    with safe_open(os.path.join(CKPT, index["embed.weight"]), framework="pt", device="cuda") as sf:
        emb = sf.get_tensor("embed.weight")
    return mr, args, blk, emb


def ref_decode(layer, lens, steps, outdir, attn_only):
    """Prefill each slot b to lens[b] tokens (start_pos 0), then `steps` one-token decode steps for
    every slot. Slots are independent sequences: the block's per-sequence buffers (window ring,
    compressor state, caches, index keys) are snapshotted per slot and restored around each step.
    Writes pre_{b}.npy [lens[b]*hc, dim], dec_x_{s}.npy / dec_ref_{s}.npy [B*hc, dim] (slot-major)."""
    os.makedirs(outdir, exist_ok=True)
    mr, args, blk, emb = load_block(layer, max(8192, max(lens) + steps + 8))
    torch.manual_seed(0)
    hc = args.hc_mult
    cap = {}
    if attn_only:
        orig_post = blk.hc_post

        def post_first(*a, **k):
            y = orig_post(*a, **k)
            if "attn" not in cap:
                cap["attn"], cap["post"], cap["comb"] = y, a[2], a[3]
            return y

        blk.hc_post = post_first
    bufs = lambda: {n: b.detach().clone() for n, b in blk.named_buffers() if "freqs" not in n}
    state = []
    with torch.inference_mode():
        for bi, t in enumerate(lens):
            for n, b in blk.named_buffers():
                if "freqs" not in n:
                    b.zero_() if "score_state" not in n else b.fill_(-float("inf"))
            ids = torch.randint(1000, 100000, (t,), device="cuda")
            h = emb[ids].to(torch.bfloat16).view(1, t, 1, -1).repeat(1, 1, hc, 1)
            blk(h, 0, mr.make_identity_pre_mix(h, hc), None)
            cap.clear()
            npy_write(os.path.join(outdir, f"pre_{bi}.npy"), h[0].reshape(-1, h.shape[-1]).float().cpu().numpy())
            state.append(bufs())
        orig_sa, seen = mr.sparse_attn, []

        def sa_first(q, kv, sink, idx, scale):
            o = orig_sa(q, kv, sink, idx, scale)
            seen.append((q.detach().clone(), idx.detach().clone(), o.detach().clone()))
            return o

        orig_attn, attn_out = blk.attn.forward, []

        def attn_first(*a, **k):
            y = orig_attn(*a, **k)
            if mr.sparse_attn is sa_first:
                attn_out.append(y.detach().clone())
            return y

        blk.attn.forward = attn_first
        ffn_cap = {k: [] for k in ("ffn_in", "ffn_out", "ffn_idx", "ffn_w", "ffn_sh")}
        orig_ffn, orig_gate, orig_sh = blk.ffn.forward, blk.ffn.gate.forward, blk.ffn.shared_experts.forward

        def on(f, keys):
            def g(*a, **k):
                y = f(*a, **k)
                if mr.sparse_attn is sa_first:
                    for key, v in zip(keys, y if isinstance(y, tuple) else (y,)):
                        ffn_cap[key].append(v.detach().reshape(1 if key != "ffn_in" else -1, -1).float().clone())
                    if keys == ("ffn_out",):
                        ffn_cap["ffn_in"].append(a[0].detach().reshape(1, -1).float().clone())
                return y
            return g

        blk.ffn.forward = on(orig_ffn, ("ffn_out",))
        blk.ffn.gate.forward = on(orig_gate, ("ffn_w", "ffn_idx"))
        blk.ffn.shared_experts.forward = on(orig_sh, ("ffn_sh",))

        mixes = []
        for s in range(steps):
            mr.sparse_attn = sa_first if s == 0 else orig_sa
            xs, ys = [], []
            for bi, t in enumerate(lens):
                for n, b in blk.named_buffers():
                    if n in state[bi]:
                        b.copy_(state[bi][n])
                ids = torch.randint(1000, 100000, (1,), device="cuda")
                h = emb[ids].to(torch.bfloat16).view(1, 1, 1, -1).repeat(1, 1, hc, 1)
                cap.clear()
                out, _ = blk(h, t + s, mr.make_identity_pre_mix(h, hc), None)
                if attn_only:
                    out = cap["attn"]
                    if s == 0:
                        mixes.append((cap["post"].reshape(1, -1).float(), cap["comb"].reshape(1, -1).float()))
                state[bi] = bufs()
                xs.append(h[0].reshape(-1, h.shape[-1]))
                ys.append(out[0].reshape(-1, out.shape[-1]))
            npy_write(os.path.join(outdir, f"dec_x_{s}.npy"), torch.cat(xs).float().cpu().numpy())
            npy_write(os.path.join(outdir, f"dec_ref_{s}.npy"), torch.cat(ys).float().cpu().numpy())
            if s == 0:  # step-0 attention internals, slot-major: q (roped), o (before inverse rope), topk rows
                for name, k in (("q", 0), ("o", 2)):
                    npy_write(os.path.join(outdir, f"ref0_{name}.npy"),
                              torch.cat([t[k].reshape(-1, t[k].shape[-1]) for t in seen]).float().cpu().numpy())
                for k, name in ((0, "post"), (1, "comb")):
                    if mixes:
                        npy_write(os.path.join(outdir, f"ref0_{name}.npy"), torch.cat([m[k] for m in mixes]).cpu().numpy())
                for key, v in ffn_cap.items():
                    if v:
                        npy_write(os.path.join(outdir, f"ref0_{key}.npy"), torch.cat(v).cpu().numpy())
                npy_write(os.path.join(outdir, "ref0_attn.npy"),
                          torch.cat([t.reshape(-1, t.shape[-1]) for t in attn_out]).float().cpu().numpy())
                wid = max(t[1].numel() for t in seen)
                npy_write(os.path.join(outdir, "ref0_idx.npy"),
                          torch.cat([torch.nn.functional.pad(t[1].reshape(1, -1), (0, wid - t[1].numel()), value=-1)
                                     for t in seen]).float().cpu().numpy())
        mr.sparse_attn = orig_sa
    json.dump({"layer": layer, "lens": lens, "steps": steps, "attn_only": attn_only}, open(os.path.join(outdir, "decode.json"), "w"))
    print(f"wrote {outdir}: lens={lens} steps={steps} attn_only={attn_only}")


def cmp(ref_path, got_path):
    r, g = np.load(ref_path).astype(np.float64), np.load(got_path).astype(np.float64)
    assert r.shape == g.shape, (r.shape, g.shape)
    d = g - r
    rel = np.linalg.norm(d) / np.linalg.norm(r)
    cos = (r * g).sum() / (np.linalg.norm(r) * np.linalg.norm(g))
    rows = np.linalg.norm(d, axis=1) / np.maximum(np.linalg.norm(r, axis=1), 1e-30)
    print(f"rel_l2={rel:.3e} cos={cos:.6f} max_abs={np.abs(d).max():.3e} worst_row_rel={rows.max():.3e} "
          f"p99_row_rel={np.percentile(rows, 99):.3e} nonfinite={int((~np.isfinite(g)).sum())}")
    return rel


if __name__ == "__main__":
    if sys.argv[1] == "ref":
        ref(int(sys.argv[2]), int(sys.argv[3]), sys.argv[4])
    elif sys.argv[1] == "ref_decode":  # ref_decode <layer> <len,len,...> <steps> <outdir> [attn]
        ref_decode(int(sys.argv[2]), [int(v) for v in sys.argv[3].split(",")], int(sys.argv[4]), sys.argv[5], len(sys.argv) > 6 and sys.argv[6] == "attn")
    else:
        cmp(sys.argv[2], sys.argv[3])
