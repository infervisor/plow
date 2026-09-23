import struct
import json
import tempfile
import unittest
import itertools
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace

import torch

from block_fp8_aiter_compare import expected_shapes, load_case, load_quant_case, load_tensor, packet_oproj_cases, packet_shared_cases, ordered_bf16_sum, shared_qualification, write_case, export_shared, load_routes, routed_weights, routed_stage_snapshots, export_routed, tensor_digest, isolated_route_weights, unsort_routed_hidden, native_routed_boundaries, bf16_order_bounds, check_routed_ab, qkva_weights


class CaptureTests(unittest.TestCase):
    def test_persistent_attention_export_ragged_and_direct_outputs(self):
        from block_fp8_aiter_compare import export_attention_ps_case

        query = torch.zeros((2, 16, 576), dtype=torch.bfloat16)
        kv = torch.zeros((2, 128, 576), dtype=torch.bfloat16)
        reference = torch.full((2, 16, 512), 3, dtype=torch.bfloat16)
        md = SimpleNamespace(work_info_set=torch.tensor([[0, -1, 0, 1, 0, 1, 0, 0],
            [1, 0, 1, 2, 1, 65, 64, 0], [1, 1, 1, 2, 65, 129, 0, 0]], dtype=torch.int32),
            work_indptr=torch.tensor([0] + [3] * 256, dtype=torch.int32),
            max_seq_len=128, topk_tokens=2048, reduce_partial_map=torch.tensor([0, 1, -1], dtype=torch.int32),
            qo_indptr=torch.tensor([0, 1, 2], dtype=torch.int32),
            paged_kv_indptr=torch.tensor([0, 1, 129], dtype=torch.int32),
            paged_kv_indices=torch.cat((torch.zeros(1, dtype=torch.int32), torch.arange(128, 256, dtype=torch.int32))),
            paged_kv_last_page_len=torch.ones(2, dtype=torch.int32), work_meta_data=None,
            reduce_indptr=torch.tensor([0, 0, 2], dtype=torch.int32),
            reduce_final_map=torch.tensor([[-1, -1], [1, 2]], dtype=torch.int32))
        state = SimpleNamespace(write_direct=True)
        def stage(*args):
            self.assertTrue(bool((args[14].view(torch.int32) == -1).all()))
            args[14][:2].fill_(1)
            args[15][:2].fill_(2)
            if state.write_direct:
                args[16][0].fill_(3)
        def reduce(*args):
            args[7][1].fill_(3)
        modules = {"aiter": SimpleNamespace(mla_decode_stage1_asm_fwd=stage, mla_reduce_v1=reduce),
            "aiter.mla": SimpleNamespace(_use_persistent_mla_decode=lambda *a: True),
            "aiter.ops.attention": SimpleNamespace(get_mla_decode_fwd_max_splits=lambda *a: 4)}
        original_empty = torch.empty
        def cpu_empty(*args, **kwargs):
            kwargs["device"] = "cpu"
            return original_empty(*args, **kwargs)
        with tempfile.TemporaryDirectory() as directory, patch.dict("sys.modules", modules), patch("torch.empty", cpu_empty):
            path = Path(directory) / "ragged.bin"
            result = export_attention_ps_case(path, query, kv, md, reference, 0.0625, kv_lengths=[1, 128])
            self.assertEqual(result["partial_slots"], 2)
            self.assertEqual(result["kv_lengths"], [1, 128])
            self.assertEqual(struct.unpack("<9I", path.read_bytes()[:36]),
                             (0x41505332, 2, 128, 2048, 3, 256, 3, 1, 128))
            for lengths in ([0, 128], [1, 129], [1], [2, 128], None):
                with self.assertRaises(ValueError):
                    export_attention_ps_case(Path(directory) / "bad.bin", query, kv, md, reference, 0.0625, kv_lengths=lengths)
            state.write_direct = False
            with self.assertRaises(ValueError):
                export_attention_ps_case(Path(directory) / "missing.bin", query, kv, md, reference, 0.0625, kv_lengths=[1, 128])
            self.assertFalse((Path(directory) / "missing.bin").exists())

    def test_persistent_attention_export_requires_exact_repeat_and_serving_output(self):
        from block_fp8_aiter_compare import export_attention_ps_case

        query = torch.zeros((1, 16, 576), dtype=torch.bfloat16)
        kv = torch.zeros((1, 64, 576), dtype=torch.bfloat16)
        reference = torch.full((1, 16, 512), 3, dtype=torch.bfloat16)
        md = SimpleNamespace(work_info_set=torch.zeros((2, 8), dtype=torch.int32),
            work_indptr=torch.full((257,), 2, dtype=torch.int32), max_seq_len=64,
            topk_tokens=2048, reduce_partial_map=torch.arange(2, dtype=torch.int32),
            qo_indptr=torch.tensor([0, 1], dtype=torch.int32),
            paged_kv_indptr=torch.tensor([0, 64], dtype=torch.int32),
            paged_kv_indices=torch.arange(64, dtype=torch.int32),
            paged_kv_last_page_len=torch.ones(1, dtype=torch.int32),
            work_meta_data=None, reduce_indptr=torch.tensor([0, 2], dtype=torch.int32),
            reduce_final_map=torch.tensor([[0, 1]], dtype=torch.int32))
        md.work_info_set[:, 1] = torch.arange(2)
        md.work_indptr[0] = 0
        state = SimpleNamespace(route=True, calls=0, unstable=False, nonfinite=False)
        def stage(*args):
            state.calls += 1
            self.assertIsNone(args[6])
            self.assertEqual(args[10:14], (1, 1, 1, 0.0625))
            args[14].fill_(float("nan") if state.nonfinite else 1 + state.unstable * state.calls)
            args[15].fill_(2)
        def reduce(*args):
            self.assertEqual(args[5:7], (1, 4))
            args[7].fill_(3)
        modules = {"aiter": SimpleNamespace(mla_decode_stage1_asm_fwd=stage, mla_reduce_v1=reduce),
            "aiter.mla": SimpleNamespace(_use_persistent_mla_decode=lambda *a: state.route),
            "aiter.ops.attention": SimpleNamespace(get_mla_decode_fwd_max_splits=lambda *a: 4)}
        original_empty = torch.empty
        def cpu_empty(*args, **kwargs):
            kwargs["device"] = "cpu"
            return original_empty(*args, **kwargs)
        with tempfile.TemporaryDirectory() as directory, patch.dict("sys.modules", modules), patch("torch.empty", cpu_empty):
            root = Path(directory)
            result = export_attention_ps_case(root / "case.bin", query, kv, md, reference, 0.0625)
            self.assertTrue(result["serving_reference_bitwise"] and result["repeat_bitwise"])
            self.assertEqual(struct.unpack("<7I", (root / "case.bin").read_bytes()[:28]),
                             (0x41505331, 1, 64, 2048, 2, 256, 2))
            self.assertEqual(struct.unpack("<6I", (root / result["reduce_file"]).read_bytes()[:24]),
                             (0x41505231, 1, 2, 2, 1, 4))
            self.assertEqual((root / result["reduce_file"]).stat().st_size,
                             24 + 2 * 4 + 2 * 4 + 2 * 4 + reference.numel() * 2)
            for failure in ("route", "unstable", "nonfinite", "reference"):
                state.route = failure != "route"
                state.unstable = failure == "unstable"
                state.nonfinite = failure == "nonfinite"
                with self.assertRaises(ValueError):
                    export_attention_ps_case(root / "bad.bin", query, kv, md,
                        reference + (failure == "reference"), 0.0625)
                self.assertFalse((root / "bad.bin").exists())

    def test_attention_rounding_model_preserves_unrounded_denominator(self):
        from block_fp8_aiter_compare import split_attention_rounding_model

        query = torch.zeros((1, 1, 65), dtype=torch.bfloat16)
        query[..., 0] = 1
        kv = torch.zeros((1, 2, 65), dtype=torch.bfloat16)
        kv[:, 1, 0] = 1
        indices = torch.tensor([[0, 1]], dtype=torch.int32)
        p = torch.exp(torch.tensor(-0.0625, dtype=torch.float64))
        expected = (1 / (1 + p)).bfloat16().reshape(1, 1, 1)
        for splits in (1, 2):
            self.assertTrue(torch.equal(split_attention_rounding_model(query, kv, indices, splits, 0.0625, True), expected))
        kv[:, 0, 0] = 1
        kv[:, 1, 0] = 2
        expected = ((p.bfloat16().double() + 2) / (1 + p)).bfloat16().reshape(1, 1, 1)
        self.assertTrue(torch.equal(split_attention_rounding_model(query, kv, indices, 1, 0.0625, True), expected))
        with self.assertRaises(ValueError):
            split_attention_rounding_model(query, kv, indices, 0, 0.0625, True)
        with self.assertRaises(ValueError):
            split_attention_rounding_model(query, kv, indices + 1, 1, 0.0625, True)
        with self.assertRaises(ValueError):
            split_attention_rounding_model(query, kv, indices, 1, 0.0625, True, split_ends=[2, 1])
        expected = ((p + 2) / (1 + p)).bfloat16().reshape(1, 1, 1)
        self.assertTrue(torch.equal(split_attention_rounding_model(query, kv, indices, 1, 0.0625, True,
            split_ends=[2], tile=1), expected))

    def test_sparse_decode_selected_key_mapping(self):
        from block_fp8_aiter_compare import sparse_decode_indices

        indices = torch.full((2, 2048), -1, dtype=torch.int32)
        indices[:, :3] = torch.tensor([[2, 0, 1], [1, 2, 0]])
        self.assertEqual(sparse_decode_indices(indices, 3).tolist(), [2, 0, 1, 4, 5, 3])
        for where, value in (((0, 0), -1), ((0, 0), 3), ((0, 0), 0), ((0, 3), 0)):
            bad = indices.clone()
            bad[where] = value
            with self.assertRaisesRegex(ValueError, "invalid selected-key"):
                sparse_decode_indices(bad, 3)
        with self.assertRaises(ValueError):
            sparse_decode_indices(indices.long(), 3)
        full = torch.arange(2048, dtype=torch.int32).reshape(1, -1)
        self.assertTrue(torch.equal(sparse_decode_indices(full, 8192), full.flatten()))

    def test_boundary_difference_no_implicit_tolerance(self):
        from block_fp8_aiter_compare import boundary_difference

        original = torch.ones((2, 4), dtype=torch.bfloat16)
        self.assertTrue(boundary_difference(original, original)["bitwise"])
        changed = original.clone()
        changed[0, 0] = 1.0078125
        difference = boundary_difference(changed, original)
        self.assertFalse(difference["bitwise"])
        self.assertEqual(difference["mismatch_count"], 1)
        self.assertEqual(difference["max_row_rel_l2"], 0.00390625)
        changed[0, 0] = float("nan")
        difference = boundary_difference(changed, original)
        self.assertFalse(difference["finite"])
        self.assertIsNone(difference["max_abs"])
        with self.assertRaises(ValueError):
            boundary_difference(original.float(), original)

    def test_qb_checker_rejects_changed_parts_atomic_values_and_partition(self):
        from block_fp8_aiter_compare import check_qb

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "outputs").mkdir()
            cases = []
            def raw(path, tensor):
                path.write_bytes(tensor.contiguous().view(torch.uint8).numpy().tobytes())
            for m in (1, 8, 16, 32, 64, 128):
                n, k = 16, 1024
                splits = 8 if m <= 16 else 4 if m <= 64 else 1
                a = torch.ones((m, k), dtype=torch.uint8).view(torch.float8_e4m3fn)
                w = torch.ones((n, k), dtype=torch.float8_e4m3fn)
                asc, ws = torch.ones((k // 128, m)), torch.ones((1, k // 128))
                out = torch.full((m, n), 1024, dtype=torch.bfloat16)
                stem = f"rank0.m{m}"
                path = root / f"{stem}.qb.bin"
                write_case(path, a.view(torch.uint8), w, asc, ws, out)
                raw(root / f"{stem}.repeat.bf16", out)
                parts, records = [], []
                for p in range(splits):
                    lo, hi = p * (k // splits), (p + 1) * (k // splits)
                    value = out / splits
                    part = root / f"{stem}.part{p}.qb.bin"
                    write_case(part, a[:, lo:hi].contiguous().view(torch.uint8), w[:, lo:hi], asc[lo // 128:hi // 128],
                               ws[:, lo // 128:hi // 128], value)
                    records.append(dict(part=p, k_start=lo, k_end=hi, file=part.name, sha256=load_case(part)[3],
                                        reference_sha256=tensor_digest(value)))
                    parts.append(value)
                raw(root / "outputs" / f"{stem}.parts.bf16", torch.stack(parts))
                for run in range(4):
                    raw(root / "outputs" / f"{stem}.atomic{run}.bf16", out)
                cases.append(dict(rank=0, shape=[m, n, k], file=path.name, sha256=load_case(path)[3],
                    reference_sha256=tensor_digest(out), repeat_sha256=tensor_digest(out), split_count=splits,
                    finite=True, norm_quant_scales_repeat_bitwise=True, isolated_parts_repeat_bitwise=True, parts=records))
            audit = dict(audit_complete=True, vllm_version="0.29.0", tp=1, cases=cases)
            reference = root / "reference.json"
            reference.write_text(json.dumps(audit))
            args = SimpleNamespace(capture=root, reference_json=reference, output=root / "pass.json")
            self.assertEqual(check_qb(args), 0)
            raw(root / "outputs/rank0.m1.atomic0.bf16", torch.zeros((1, 16), dtype=torch.bfloat16))
            args.output = root / "bad-atomic.json"
            self.assertEqual(check_qb(args), 1)
            raw(root / "outputs/rank0.m1.atomic0.bf16", torch.full((1, 16), 1024, dtype=torch.bfloat16))
            raw(root / "outputs/rank0.m1.parts.bf16", torch.zeros((8, 1, 16), dtype=torch.bfloat16))
            args.output = root / "bad-parts.json"
            self.assertEqual(check_qb(args), 1)
            cases[0]["parts"][0]["k_start"] = 128
            reference.write_text(json.dumps(audit))
            with self.assertRaisesRegex(ValueError, "original full operands or partition"):
                check_qb(args)

    def test_qb_original_tp_slice_and_norm(self):
        from safetensors.torch import save_file
        from block_fp8_aiter_compare import qb_weights

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg = dict(num_attention_heads=4, q_lora_rank=128, qk_nope_head_dim=192, qk_rope_head_dim=64)
            (root / "config.json").write_text(json.dumps(cfg))
            name = "model.layers.3.self_attn.q_b_proj.weight"
            w = torch.full((1024, 128), 128, dtype=torch.uint8)
            ws = torch.arange(1, 9, dtype=torch.float32).reshape(8, 1)
            gamma = torch.arange(128).to(torch.bfloat16)
            tensors = {name: w.view(torch.float8_e4m3fn), name + "_scale_inv": ws,
                       "model.layers.3.self_attn.q_a_layernorm.weight": gamma}
            save_file(tensors, root / "weights.safetensors")
            (root / "model.safetensors.index.json").write_text(json.dumps(dict(
                weight_map={key: "weights.safetensors" for key in tensors})))
            actual, scales, norm = qb_weights(root, 3, 1, 2)
            self.assertTrue(torch.equal(actual.view(torch.uint8), w[512:]))
            self.assertTrue(torch.equal(scales, ws[4:]))
            self.assertTrue(torch.equal(norm, gamma))
            for rank, tp in ((0, 0), (2, 2), (0, 3)):
                with self.assertRaisesRegex(ValueError, "geometry"):
                    qb_weights(root, 3, rank, tp)

    def test_mla_checker_requires_complete_hash_bound_bitwise_replays(self):
        from block_fp8_aiter_compare import write_mla_case, load_mla_case, check_mla

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "outputs").mkdir()
            cases, weights = [], []
            w = torch.ones((1, 16, 16), dtype=torch.float8_e4m3fn)
            scale = torch.tensor(0.25)
            for p in ("W_K", "W_V"):
                weights.append(dict(rank=0, projection=p, weight_sha256=tensor_digest(w), scalar_sha256=tensor_digest(scale)))
                for m in (1, 8, 16, 32, 64, 128):
                    x = torch.ones((m, 1, 16), dtype=torch.bfloat16)
                    out = x * 4
                    path = root / f"rank0.m{m}.{p}.mla.bin"
                    write_mla_case(path, x, w, scale, out)
                    (root / "outputs" / f"rank0.m{m}.{p}.bf16").write_bytes(out.view(torch.uint8).numpy().tobytes())
                    cases.append(dict(rank=0, projection=p, shape=[m, 1, 16, 16], file=path.name,
                        sha256=load_mla_case(path)[2], input_sha256=tensor_digest(x), reference_sha256=tensor_digest(out),
                        repeat_sha256=tensor_digest(out), finite=True, output_repeat_bitwise=True))
            audit = dict(audit_complete=True, vllm_version="0.29.0", tp=1, weights=weights, cases=cases)
            reference = root / "reference.json"
            reference.write_text(json.dumps(audit))
            args = SimpleNamespace(capture=root, reference_json=reference, output=root / "pass.json")
            self.assertEqual(check_mla(args), 0)
            qb_cases = []
            for m in (1, 8, 16, 32, 64, 128):
                q = torch.ones((m, 80), dtype=torch.bfloat16)
                qb_file = root / f"qb.m{m}.bin"
                write_case(qb_file, torch.zeros((m, 128), dtype=torch.uint8),
                           torch.zeros((80, 128), dtype=torch.float8_e4m3fn),
                           torch.ones((1, m)), torch.ones((1, 1)), q)
                qb_cases.append(dict(rank=0, shape=[m, 80, 128], file=qb_file.name,
                                     sha256=load_case(qb_file)[3], reference_sha256=tensor_digest(q)))
                (root / "outputs" / f"rank0.m{m}.W_K.bf16.rope").write_bytes(
                    q[:, 16:].contiguous().view(torch.uint8).numpy().tobytes())
            qb_reference = root / "qb.json"
            qb_reference.write_text(json.dumps(dict(audit_complete=True, tp=1, cases=qb_cases)))
            import hashlib
            audit["qb_reference_sha256"] = hashlib.sha256(qb_reference.read_bytes()).hexdigest()
            reference.write_text(json.dumps(audit))
            args.qb_reference = qb_reference
            args.output = root / "rope-pass.json"
            self.assertEqual(check_mla(args), 0)
            (root / "outputs/rank0.m1.W_K.bf16.rope").write_bytes(bytes(128))
            args.output = root / "rope-fail.json"
            self.assertEqual(check_mla(args), 1)
            qb_reference.write_text("{}")
            with self.assertRaisesRegex(ValueError, "input provenance"):
                check_mla(args)
            args.qb_reference = None
            (root / "outputs/rank0.m1.W_K.bf16").write_bytes(torch.zeros((1, 1, 16), dtype=torch.bfloat16).view(torch.uint8).numpy().tobytes())
            args.output = root / "fail.json"
            self.assertEqual(check_mla(args), 1)
            cases[0]["sha256"] = "stale"
            reference.write_text(json.dumps(audit))
            with self.assertRaisesRegex(ValueError, "hash/geometry/stability"):
                check_mla(args)
            cases.pop()
            reference.write_text(json.dumps(audit))
            with self.assertRaisesRegex(ValueError, "exactly once"):
                check_mla(args)

    def test_mla_replay_roundtrip_tail_and_rejections(self):
        from block_fp8_aiter_compare import write_mla_case, load_mla_case

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mla.bin"
            x = torch.arange(2 * 3 * 192).reshape(2, 3, 192).to(torch.bfloat16)
            w = torch.full((3, 16, 192), 128, dtype=torch.uint8).view(torch.float8_e4m3fn)
            scale = torch.tensor(0.25)
            out = torch.zeros((2, 3, 16), dtype=torch.bfloat16)
            write_mla_case(path, x, w, scale, out)
            shape, tensors, digest = load_mla_case(path)
            self.assertEqual(shape, (2, 3, 16, 192))
            self.assertEqual(len(digest), 64)
            for a, b in zip((x, w, scale, out), tensors):
                self.assertEqual(tensor_digest(a), tensor_digest(b))
            with self.assertRaises(FileExistsError):
                write_mla_case(path, x, w, scale, out)
            for bad_scale in (scale.reshape(1), -scale, scale * float("nan")):
                with self.assertRaises(ValueError):
                    write_mla_case(path, x, w, bad_scale, out)
            original = path.read_bytes()
            for raw in (b"", original[:15], original[:-1], original + b"x",
                        struct.pack("<I", 0) + original[4:]):
                path.write_bytes(raw)
                with self.assertRaises(ValueError):
                    load_mla_case(path)

    def test_mla_original_tp_slice_preserves_bytes_and_scales(self):
        from safetensors.torch import save_file
        from block_fp8_aiter_compare import mla_kvb_weights

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg = dict(num_attention_heads=4, kv_lora_rank=128, qk_nope_head_dim=192, v_head_dim=64)
            (root / "config.json").write_text(json.dumps(cfg))
            name = "model.layers.3.self_attn.kv_b_proj.weight"
            w = torch.arange(1024, dtype=torch.int32).remainder(127).to(torch.uint8)[:, None].expand(1024, 128).contiguous()
            w[512] = 128
            ws = torch.arange(1, 9, dtype=torch.float32).reshape(8, 1)
            tensors = {name: w.view(torch.float8_e4m3fn), name + "_scale_inv": ws}
            save_file(tensors, root / "weights.safetensors")
            (root / "model.safetensors.index.json").write_text(json.dumps(dict(
                weight_map={key: "weights.safetensors" for key in tensors})))
            for rank in (0, 1):
                actual, scales = mla_kvb_weights(root, 3, rank, 2)
                self.assertTrue(torch.equal(actual.view(torch.uint8), w[rank * 512:(rank + 1) * 512]))
                self.assertTrue(torch.equal(scales, ws[rank * 4:(rank + 1) * 4]))
            for rank, tp in ((0, 0), (2, 2), (0, 3)):
                with self.assertRaisesRegex(ValueError, "geometry"):
                    mla_kvb_weights(root, 3, rank, tp)

    def test_qkva_connected_checker_is_hash_bound_and_bitwise(self):
        from block_fp8_aiter_compare import check_qkva

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "inputs").mkdir()
            (root / "outputs").mkdir()
            (root / "config.json").write_text(json.dumps(dict(hidden_size=128, q_lora_rank=128,
                kv_lora_rank=128, qk_rope_head_dim=64)))
            (root / "inputs/reference.json").write_text(json.dumps(dict(batch=2, layer=3)))
            norm = torch.ones((2, 128), dtype=torch.bfloat16)
            quant = norm.to(torch.float8_e4m3fn)
            weight = torch.ones((320, 128), dtype=torch.float8_e4m3fn)
            xs, ws = torch.ones((1, 2)), torch.ones((3, 1))
            out = torch.ones((2, 320), dtype=torch.bfloat16)
            def write(path, value):
                path.write_bytes(value.contiguous().view(torch.uint8).numpy().tobytes())
            write(root / "inputs/act.x.bin", norm)
            write(root / "m2.norm.bf16", norm)
            replay = root / "replay.bin"
            write_case(replay, quant.view(torch.uint8), weight, xs, ws, out)
            case = dict(shape=[2, 320, 128], file="replay.bin", sha256=load_case(replay)[3],
                normalized_input_sha256=tensor_digest(norm), reference_sha256=tensor_digest(out),
                finite=True, norm_quant_scales_repeat_bitwise=True, output_repeat_bitwise=True)
            audit = dict(audit_complete=True, vllm_version="0.29.0", layer=3, source_batch=2,
                input_sha256=tensor_digest(norm), weight_sha256=tensor_digest(weight),
                scale_sha256=tensor_digest(ws), cases=[case])
            (root / "audit.json").write_text(json.dumps(audit))
            prefix = "model.layers.3.self_attn.fused_qkv_a_proj"
            for name, value in (("act.xn", norm), ("act.qkva_xq", quant), ("act.qkva_xs", xs),
                (prefix + ".weight_fp8", weight), (prefix + ".weight_scale_inv", ws),
                ("act.qlr", out[:, :128]), ("act.ckvraw", out[:, 128:256]), ("act.krr", out[:, 256:])):
                write(root / "outputs" / f"rank0.{name}.bin", value)
            args = SimpleNamespace(capture=root, checkpoint=root, reference_json=root / "audit.json",
                                   output=root / "pass.json", tp=1)
            self.assertEqual(check_qkva(args), 0)
            write(root / "outputs/rank0.act.krr.bin", out[:, 256:] * 2)
            args.output = root / "fail.json"
            self.assertEqual(check_qkva(args), 1)
            write(root / "inputs/act.x.bin", norm * 2)
            with self.assertRaisesRegex(ValueError, "input/geometry"):
                check_qkva(args)
            write(root / "inputs/act.x.bin", norm)
            write(root / "m2.norm.bf16", norm * 2)
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                check_qkva(args)

    def test_qkva_original_bytes_scales_and_tail(self):
        from safetensors.torch import save_file

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = dict(hidden_size=128, q_lora_rank=128, kv_lora_rank=128, qk_rope_head_dim=64)
            (root / "config.json").write_text(json.dumps(config))
            p = "model.layers.3."
            q = torch.full((128, 128), 128, dtype=torch.uint8).view(torch.float8_e4m3fn)
            kv = torch.ones((192, 128), dtype=torch.uint8).view(torch.float8_e4m3fn)
            tensors = {p + "self_attn.q_a_proj.weight": q,
                       p + "self_attn.q_a_proj.weight_scale_inv": torch.tensor([[1.]]),
                       p + "self_attn.kv_a_proj_with_mqa.weight": kv,
                       p + "self_attn.kv_a_proj_with_mqa.weight_scale_inv": torch.tensor([[2.], [3.]]),
                       p + "input_layernorm.weight": torch.ones(128, dtype=torch.bfloat16)}
            save_file(tensors, root / "weights.safetensors")
            (root / "model.safetensors.index.json").write_text(json.dumps(dict(
                weight_map={key: "weights.safetensors" for key in tensors})))
            weight, scale, gamma = qkva_weights(root, 3)
            self.assertEqual(tuple(weight.shape), (320, 128))
            self.assertTrue(torch.equal(weight[:128].view(torch.uint8), q.view(torch.uint8)))
            self.assertTrue(torch.equal(weight[128:].view(torch.uint8), kv.view(torch.uint8)))
            self.assertEqual(scale.flatten().tolist(), [1., 2., 3.])
            self.assertEqual(gamma.dtype, torch.bfloat16)
            config["q_lora_rank"] = 64
            (root / "config.json").write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "concatenation boundary"):
                qkva_weights(root, 3)

    def test_routed_ab_checks_boundaries_reduction_and_final_rounding(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pinned = root / "pinned"
            pinned.mkdir()
            x = torch.ones((1, 128), dtype=torch.bfloat16)
            q = torch.zeros((1, 128), dtype=torch.float8_e4m3fn)
            scale = torch.ones((1, 1))
            table = struct.pack("<If", 0, 0.5)
            boundaries = {}
            for key, value in (("stage1.input", q), ("stage1.scale", scale),
                               ("stage1.output", x.reshape(1, 1, 128)),
                               ("stage2.input", q.reshape(1, 1, 128)),
                               ("stage2.scale", scale.reshape(1, 1, 1))):
                (pinned / f"audit.rank0.{key}.bin").write_bytes(value.view(torch.uint8).numpy().tobytes())
                boundaries[key] = dict(sha256=tensor_digest(value), repeat_bitwise=True, plow_bitwise=True, finite=True)
            case = pinned / "down.bin"
            write_case(case, q.view(torch.uint8), torch.zeros((128, 128), dtype=torch.float8_e4m3fn),
                       scale, scale, x * 0.5, row_weights=torch.tensor([0.5]))
            maps = torch.full((64,), -1, dtype=torch.int32); maps[0] = 0
            gates = torch.zeros(64); gates[0] = 0.5
            hidden = torch.zeros((64, 128), dtype=torch.bfloat16); hidden[0] = x[0]
            for arm in ("ctl", "treat", "ctl2", "treat2"):
                out = root / arm / "outputs"; out.mkdir(parents=True)
                (out / "rank0.act.tab.bin").write_bytes(table)
                for name, value in (("xn2", x), ("routed_xq", q), ("routed_xs", scale),
                                   ("routed_hq", q), ("routed_hs", scale),
                                   ("moe_meta", torch.tensor([0, 1, 0, 1], dtype=torch.int32)),
                                   ("moe_rowtok", maps), ("moe_rowpart", maps), ("moe_rowgate", gates),
                                   ("moe_fug", hidden), ("part", x * 0.5), ("shared", x),
                                   ("attn", x * 1.5), ("xmid", x * 2), ("xnext", x * 3.5)):
                    (out / f"rank0.act.{name}.bin").write_bytes(value.view(torch.uint8).numpy().tobytes())
            _, _, routehash = load_routes(root / "ctl/outputs/rank0.act.tab.bin", 1, 1, 1)
            row = dict(rank=0, shape=[1, 128, 128, 1, 1], boundaries=boundaries,
                       input_sha256=tensor_digest(x), routes_sha256=routehash,
                       isolated_weighted_down=[dict(file=case.name, sha256=load_case(case, weighted=True)[-1],
                           tokens=[0], slots=[0], finite=True, repeat_bitwise=True)])
            audit = dict(passed=True, audit_complete=True, vllm_version="0.29.0", cases=[row])
            reference = pinned / "audit.json"; reference.write_text(json.dumps(audit))
            args = SimpleNamespace(capture=root, reference_json=reference, tp=1, output=root / "checked.json")
            self.assertEqual(check_routed_ab(args), 0)
            self.assertEqual(len(json.loads(args.output.read_text())["checks"]), 4)
            path = root / "treat/outputs/rank0.act.xnext.bin"
            path.write_bytes(x.view(torch.uint8).numpy().tobytes())
            with self.assertRaisesRegex(ValueError, "rounding mismatch"):
                check_routed_ab(args)
            path = root / "treat/outputs/rank0.act.part.bin"
            path.write_bytes(x.view(torch.uint8).numpy().tobytes())
            with self.assertRaisesRegex(ValueError, "outside addition-order"):
                check_routed_ab(args)
            path = root / "treat/outputs/rank0.act.routed_xq.bin"
            path.write_bytes(bytes([1]) * 128)
            with self.assertRaisesRegex(ValueError, "nonbitwise stable boundary"):
                check_routed_ab(args)

    def test_native_routed_boundaries_preserve_layout_and_validate_maps(self):
        ids = torch.tensor([[1, 0], [0, 1]], dtype=torch.int32)
        weights = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
        capacity = 4 + 2 * 127
        meta = torch.tensor([0, 64, 2, 2, 0, 1, 2], dtype=torch.int32)
        tokens = torch.full((capacity,), -1, dtype=torch.int32)
        slots = tokens.clone()
        gates = torch.zeros(capacity)
        hidden = torch.zeros((capacity, 128), dtype=torch.bfloat16)
        rows, parts = [0, 1, 64, 65], [1, 2, 0, 3]
        slots[rows] = torch.tensor(parts, dtype=torch.int32)
        tokens[rows] = slots[rows] // 2
        gates[rows] = weights.flatten()[parts]
        hidden[rows] = torch.tensor(parts).reshape(4, 1).expand(4, 128).bfloat16()
        expected = torch.arange(4).reshape(2, 2, 1).expand(2, 2, 128).bfloat16()
        self.assertTrue(torch.equal(unsort_routed_hidden(meta, tokens, slots, gates, hidden, ids, weights), expected))
        for tensor, at, value in ((meta, 2, 1), (slots, 1, 1), (tokens, 0, 1),
                                  (gates, 0, 0.5), (slots, 2, 0), (hidden, (2, 0), 1)):
            old = tensor[at].clone()
            tensor[at] = value
            with self.assertRaises(ValueError):
                unsort_routed_hidden(meta, tokens, slots, gates, hidden, ids, weights)
            tensor[at] = old
        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory) / "rank0"
            q = torch.arange(256).to(torch.uint8).reshape(2, 128)
            hq = torch.arange(512).to(torch.uint8).reshape(2, 2, 128)
            hs = torch.arange(1., 5.).reshape(1, 4)
            for name, value in (("routed_xq", q), ("routed_xs", hs[:, :2].contiguous()),
                               ("routed_hq", hq), ("routed_hs", hs), ("moe_meta", meta),
                               ("moe_rowtok", tokens), ("moe_rowpart", slots), ("moe_rowgate", gates),
                               ("moe_fug", hidden), ("part", torch.ones((16, 128), dtype=torch.bfloat16))):
                Path(str(prefix) + f".act.{name}.bin").write_bytes(value.view(torch.uint8).numpy().tobytes())
            boundaries, hashes = native_routed_boundaries(prefix, 128, 128, 2, ids, weights)
            self.assertTrue(torch.equal(boundaries["stage1.input"].view(torch.uint8), q))
            self.assertTrue(torch.equal(boundaries["stage1.output"], expected))
            self.assertTrue(torch.equal(boundaries["stage2.input"].view(torch.uint8), hq))
            self.assertEqual(boundaries["stage2.scale"].flatten().tolist(), [1., 2., 3., 4.])
            self.assertEqual(tuple(boundaries["stage2.output"].shape), (2, 128))
            self.assertEqual(len(hashes), 10)

    def test_bf16_order_extrema_match_exhaustive_permutations(self):
        parts = torch.tensor([256., 1., -256., 0.5, -0.5, 0.00390625], dtype=torch.bfloat16).reshape(1, 6, 1)
        for topk in range(1, 7):
            values = []
            for order in itertools.permutations(range(topk)):
                result = torch.zeros((1, 1), dtype=torch.bfloat16)
                for slot in order:
                    result = result + parts[:, slot]
                values.append(result)
            lo, hi = bf16_order_bounds(parts[:, :topk])
            self.assertTrue(torch.equal(lo, torch.stack(values).amin(0)))
            self.assertTrue(torch.equal(hi, torch.stack(values).amax(0)))
        zeros = torch.zeros((2, 8, 3), dtype=torch.bfloat16)
        self.assertTrue(all(bool((v == 0).all()) for v in bf16_order_bounds(zeros)))
        for bad in (parts.float(), parts[:, :0], torch.zeros((1, 9, 1), dtype=torch.bfloat16), parts * float('nan')):
            with self.assertRaises(ValueError):
                bf16_order_bounds(bad)

    def test_weighted_replay_keeps_final_fp32_route_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weighted.bin"
            a = torch.zeros((2, 128), dtype=torch.uint8)
            w = torch.zeros((128, 128), dtype=torch.float8_e4m3fn)
            asc, ws = torch.ones((1, 2)), torch.ones((1, 1))
            expected = torch.zeros((2, 128), dtype=torch.bfloat16)
            gates = torch.tensor([0., 0.12345679], dtype=torch.float32)
            write_case(path, a, w, asc, ws, expected, row_weights=gates)
            shape, operands, actual, _ = load_case(path, weighted=True)
            self.assertEqual(shape, (2, 128, 128))
            self.assertTrue(torch.equal(operands[-1].view(torch.int32), gates.view(torch.int32)))
            self.assertTrue(torch.equal(actual, expected))
            with self.assertRaises(ValueError):
                load_case(path)
            with self.assertRaises(ValueError):
                write_case(Path(directory) / "bad.bin", a, w, asc, ws, expected, row_weights=gates.bfloat16())

    def test_isolated_route_weights_mask_padding_and_other_experts(self):
        ids = torch.tensor([[1, 0], [0, 1]], dtype=torch.int32)
        sorted_ids = torch.tensor([1 << 24, 1, 0, (1 << 24) | 1, 2, -1, 0], dtype=torch.int32)
        weights = torch.arange(1., 8.)
        self.assertEqual(isolated_route_weights(ids, sorted_ids, weights, 6, 0).tolist(),
                         [1., 2., 0., 0., 0., 0., 0.])
        self.assertEqual(isolated_route_weights(ids, sorted_ids, weights, 6, 1).tolist(),
                         [0., 0., 3., 4., 0., 0., 0.])

    def test_routed_export_checks_audit_and_groups_token_slots(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("inputs", "outputs", "cases"):
                (root / name).mkdir()
            (root / "inputs/reference.json").write_text(json.dumps(dict(batch=2, layer=3)))
            route = root / "outputs/rank0.act.tab.bin"
            route.write_bytes(struct.pack("<IfIfIfIf", 1, 0.5, 0, 0.5, 0, 0.5, 1, 0.5))
            _, _, route_digest = load_routes(route, 2, 2, 2)
            weights = (torch.zeros((2, 256, 128), dtype=torch.float8_e4m3fn),
                       torch.zeros((2, 128, 128), dtype=torch.float8_e4m3fn),
                       torch.ones((2, 2, 1)), torch.ones((2, 1, 1)))
            output = torch.arange(4).reshape(2, 2, 1).expand(2, 2, 128).bfloat16().contiguous()
            boundaries = {}
            for key, value in (("stage1.input", torch.zeros((2, 128), dtype=torch.uint8)),
                               ("stage1.scale", torch.ones((2, 1))), ("stage1.output", output)):
                digest = tensor_digest(value)
                boundaries[key] = dict(finite=True, repeat_bitwise=True, sha256=digest, repeat_sha256=digest)
                for suffix in ("", ".repeat"):
                    (root / f"audit.rank0.{key}{suffix}.bin").write_bytes(value.view(torch.uint8).numpy().tobytes())
            row = dict(rank=0, shape=[2, 128, 128, 2, 2], selected=dict(run_1stage=False),
                finite=True, routes_sha256=route_digest, boundaries=boundaries,
                checkpoint_shard_sha256={key: tensor_digest(value) for key, value in
                    zip(("gate_up", "down", "gate_up_scale", "down_scale"), weights)})
            audit = dict(audit_complete=True, vllm_version="0.29.0", cases=[row])
            reference = root / "audit.json"
            reference.write_text(json.dumps(audit))
            args = SimpleNamespace(checkpoint=root, reference_json=reference, capture=root,
                                   output=root / "cases/replay.json", tp=1)
            with patch("block_fp8_aiter_compare.routed_weights", return_value=weights):
                self.assertEqual(export_routed(args), 0)
                records = json.loads(args.output.read_text())["cases"]
                self.assertEqual([r["slots"] for r in records], [[1, 0], [0, 1]])
                _, _, actual, _ = load_case(root / "cases/rank0.expert0.glu.bin", glu=True)
                self.assertEqual(actual[:, 0].tolist(), [1.0, 2.0])
                (root / "grouped").mkdir()
                args.export_routed_grouped = True
                args.grouped_repeat = 6
                args.output = root / "grouped/replay.json"
                self.assertEqual(export_routed(args), 0)
                raw = (root / "grouped/rank0.grouped.bin").read_bytes()
                self.assertEqual(struct.unpack_from("<6I", raw), (12, 128, 128, 2, 2, 2))
                offset = 24 + 12 * 128 + 12 * 4 + 12 * 2 * 8
                repeated_output = torch.frombuffer(bytearray(raw), dtype=torch.bfloat16,
                                                  count=12 * 2 * 128, offset=offset).reshape(12, 2, 128)
                self.assertTrue(torch.equal(repeated_output, output.repeat(6, 1, 1)))
                self.assertEqual(len(raw), offset + 12 * 2 * 128 * 2 + 2 * (4 + 256 * 128 + 2 * 4))
                for key, value in (("stage2.input", torch.zeros((2, 2, 128), dtype=torch.uint8)),
                                   ("stage2.scale", torch.ones((2, 2, 1)))):
                    digest = tensor_digest(value)
                    boundaries[key] = dict(finite=True, repeat_bitwise=True, sha256=digest, repeat_sha256=digest)
                    for suffix in ("", ".repeat"):
                        (root / f"audit.rank0.{key}{suffix}.bin").write_bytes(value.view(torch.uint8).numpy().tobytes())
                ids, gates, _ = load_routes(route, 2, 2, 2)
                row["isolated_weighted_down"] = []
                for expert in range(2):
                    token, slot = (ids == expert).nonzero(as_tuple=True)
                    isolated = root / f"expert{expert}.bin"
                    write_case(isolated, torch.zeros((2, 128), dtype=torch.uint8), weights[1][expert],
                        torch.ones((1, 2)), weights[3][expert], output[token, slot], row_weights=gates[token, slot])
                    _, _, _, digest = load_case(isolated, weighted=True)
                    row["isolated_weighted_down"].append(dict(expert=expert, tokens=token.tolist(), slots=slot.tolist(),
                        finite=True, repeat_bitwise=True, file=isolated.name, sha256=digest))
                reference.write_text(json.dumps(audit))
                (root / "down").mkdir()
                args.export_routed_grouped = False
                args.export_routed_down_grouped = True
                args.output = root / "down/replay.json"
                self.assertEqual(export_routed(args), 0)
                raw = (root / "down/rank0.grouped.bin").read_bytes()
                self.assertEqual(struct.unpack_from("<6I", raw), (12, 128, 128, 2, 2, 2))
                offset = 24 + 24 * 128 + 24 * 4 + 12 * 2 * 8
                self.assertEqual(len(raw), offset + 24 * 128 * 2 + 2 * (4 + 128 * 128 + 4))
                actual = torch.frombuffer(bytearray(raw), dtype=torch.bfloat16, count=24 * 128, offset=offset)
                self.assertTrue(torch.equal(actual.reshape(12, 2, 128), output.repeat(6, 1, 1)))
                hidden_path = root / "down/rank0.grouped.hidden.bf16"
                expected_hidden = output.repeat(6, 1, 1)
                self.assertEqual(hidden_path.read_bytes(), expected_hidden.view(torch.uint8).numpy().tobytes())
                record = json.loads(args.output.read_text())["cases"][0]
                self.assertEqual(record["hidden_sha256"], tensor_digest(expected_hidden))
                row["isolated_weighted_down"].pop()
                reference.write_text(json.dumps(audit))
                with self.assertRaisesRegex(ValueError, "missing isolated"):
                    export_routed(args)
                args.export_routed_down_grouped = False
                row["boundaries"]["stage1.output"]["sha256"] = "invalid"
                reference.write_text(json.dumps(audit))
                with self.assertRaisesRegex(ValueError, "hash or repeat"):
                    export_routed(args)

    def test_glu_replay_writer_round_trip(self):
        a = torch.arange(256, dtype=torch.int32).to(torch.uint8).reshape(2, 128)
        w = torch.zeros((256, 128), dtype=torch.float8_e4m3fn)
        w.view(torch.uint8)[128:] = 128
        asc = torch.tensor([[1., 2.]])
        ws = torch.tensor([[3.], [4.]])
        expected = torch.ones((2, 128), dtype=torch.bfloat16)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "glu.bin"
            write_case(path, a, w, asc, ws, expected, glu=True)
            shape, (aa, ww, ss, wss), out, _ = load_case(path, glu=True)
            self.assertEqual(shape, (2, 128, 128))
            self.assertTrue(torch.equal(aa.view(torch.uint8), a))
            self.assertTrue(torch.equal(ww.view(torch.uint8), w.view(torch.uint8)))
            self.assertTrue(torch.equal(ss, asc.T))
            self.assertTrue(torch.equal(wss, ws))
            self.assertTrue(torch.equal(out, expected))
            with self.assertRaises(ValueError):
                load_case(path)
            with self.assertRaises(ValueError):
                write_case(Path(directory) / "bad.bin", a, w[:128], asc, ws, expected, glu=True)
            singleton = torch.tensor([[1., 2.]]).T
            one = Path(directory) / "one.bin"
            write_case(one, torch.zeros((1, 256), dtype=torch.uint8),
                torch.zeros((256, 256), dtype=torch.float8_e4m3fn), singleton,
                torch.ones((2, 2)), torch.zeros((1, 128), dtype=torch.bfloat16), glu=True)
            self.assertEqual(load_case(one, glu=True)[1][2].tolist(), [[1., 2.]])

    def test_routed_boundary_snapshots_are_owned_and_complete(self):
        x = torch.zeros((2, 128), dtype=torch.float8_e4m3fn)
        y = torch.ones((2, 128), dtype=torch.bfloat16)
        scale = torch.ones((2, 1))
        calls = [(name, SimpleNamespace(args=(x, None, None, None, None, None, y),
                                       keywords={key: scale})) for name, key in
                 (("stage1", "a1_scale"), ("stage2", "a2_scale"))]
        values = routed_stage_snapshots(calls, False)
        self.assertEqual(len(values), 6)
        y.fill_(4)
        scale.fill_(8)
        self.assertTrue(bool((values["stage1.output"] == 1).all()))
        self.assertTrue(bool((values["stage2.scale"] == 1).all()))
        self.assertEqual(routed_stage_snapshots(calls[:1], True), {})
        with self.assertRaises(ValueError):
            routed_stage_snapshots(calls[:1], False)

    def test_routed_table_preserves_gate_bits_and_rejects_invalid_routes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "routes.bin"
            path.write_bytes(struct.pack("<IfIfIfIf", 2, 0.25, 0, 0.75, 1, 0.5, 2, 0.5))
            ids, gates, digest = load_routes(path, 2, 2, 3)
            self.assertEqual(ids.tolist(), [[2, 0], [1, 2]])
            self.assertEqual(gates.tolist(), [[0.25, 0.75], [0.5, 0.5]])
            self.assertEqual(len(digest), 64)
            for invalid in ((0, 0.5, 0, 0.5), (3, 0.5, 1, 0.5),
                            (0, float("nan"), 1, 0.5), (0, -1.0, 1, 0.5)):
                path.write_bytes(struct.pack("<IfIf", *invalid))
                with self.assertRaises(ValueError):
                    load_routes(path, 1, 2, 3)

    def test_routed_original_fp8_tp_shards(self):
        from safetensors.torch import save_file

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text(json.dumps(dict(
                hidden_size=128, moe_intermediate_size=256, n_routed_experts=2)))
            tensors = {}
            for expert in range(2):
                for proj, tag in (("gate", 0), ("up", 16), ("down", 32)):
                    name = f"model.layers.3.mlp.experts.{expert}.{proj}_proj.weight"
                    raw = torch.empty((256, 128), dtype=torch.uint8)
                    raw[:128] = 128 if proj == "gate" and expert == 0 else tag + expert
                    raw[128:] = tag + expert + 1
                    scale = torch.tensor([[1.0 + tag + expert], [2.0 + tag + expert]])
                    if proj == "down":
                        raw, scale = raw.T.contiguous(), scale.T.contiguous()
                    tensors[name] = raw.view(torch.float8_e4m3fn)
                    tensors[name + "_scale_inv"] = scale
            save_file(tensors, root / "weights.safetensors")
            (root / "model.safetensors.index.json").write_text(json.dumps(dict(
                weight_map={key: "weights.safetensors" for key in tensors})))
            for rank in range(2):
                gu, down, gus, downs = routed_weights(root, 3, rank, 2)
                for expert in range(2):
                    prefix = f"model.layers.3.mlp.experts.{expert}."
                    for slot, proj in enumerate(("gate", "up")):
                        name = prefix + proj + "_proj.weight"
                        self.assertTrue(torch.equal(gu[expert, slot * 128:(slot + 1) * 128].view(torch.uint8),
                            tensors[name][rank * 128:(rank + 1) * 128].view(torch.uint8)))
                        self.assertTrue(torch.equal(gus[expert, slot], tensors[name + "_scale_inv"][rank]))
                    name = prefix + "down_proj.weight"
                    self.assertTrue(torch.equal(down[expert].view(torch.uint8),
                        tensors[name][:, rank * 128:(rank + 1) * 128].view(torch.uint8)))
                    self.assertTrue(torch.equal(downs[expert], tensors[name + "_scale_inv"][:, rank:rank + 1]))
            for rank, tp in ((2, 2), (0, 0), (0, 4)):
                with self.assertRaises(ValueError):
                    routed_weights(root, 3, rank, tp)

    def test_replay_writer_round_trip(self):
        a = torch.arange(512, dtype=torch.int32).to(torch.uint8).reshape(2, 256)
        w = torch.zeros((128, 256), dtype=torch.float8_e4m3fn)
        asc = torch.tensor([[1., 2.], [3., 4.]])
        wsc = torch.tensor([[5., 6.]])
        expected = torch.ones((2, 128), dtype=torch.bfloat16)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "case.bin"
            write_case(path, a, w, asc, wsc, expected)
            shape, operands, output, _ = load_case(path)
            self.assertEqual(shape, (2, 128, 256))
            self.assertTrue(torch.equal(operands[0].view(torch.uint8), a))
            self.assertTrue(torch.equal(operands[2], asc.T))
            self.assertTrue(torch.equal(output, expected))
            with self.assertRaises(FileExistsError):
                write_case(path, a, w, asc, wsc, expected)
            with self.assertRaises(ValueError):
                write_case(path, a, w, asc, wsc, expected.float())

    def test_shared_diagnostics_never_qualify_a_failed_block(self):
        rows = [dict(rank=r, passed=True, block_oracle_verified=True) for r in range(2)]
        self.assertEqual(shared_qualification(rows, 2), (True, True))
        rows[1]["block_oracle_verified"] = False
        self.assertEqual(shared_qualification(rows, 2), (True, False))
        rows[1]["passed"] = False
        self.assertEqual(shared_qualification(rows, 2), (False, False))
        self.assertEqual(shared_qualification(rows[:1], 2), (False, False))
        self.assertEqual(shared_qualification([rows[0], rows[0]], 2), (False, False))
        self.assertEqual(shared_qualification([], 0), (False, False))

    def test_ordered_sum_keeps_each_bf16_round(self):
        partials = [torch.tensor([v], dtype=torch.bfloat16) for v in (256.0, 1.0, -256.0, 0.0)]
        self.assertEqual(ordered_bf16_sum(partials).item(), 0.0)
        self.assertEqual(sum(p.float() for p in partials).bfloat16().item(), 1.0)
        for wrong in (partials[:3], [p.float() for p in partials], partials[:3] + [torch.zeros(2)]):
            with self.assertRaises(ValueError):
                ordered_bf16_sum(wrong)

    def test_layout(self):
        m, n, k = 2, 128, 256
        data = struct.pack("<III", m, n, k) + bytes(m * k + n * k)
        data += struct.pack("<4f", 1, 2, 3, 4) + struct.pack("<2f", 5, 6)
        data += bytes(m * n * 2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "case.bin"
            path.write_bytes(data)
            shape, operands, output, digest = load_case(path)
            self.assertEqual(shape, (m, n, k))
            self.assertEqual(operands[0].dtype, torch.float8_e4m3fn)
            self.assertEqual(operands[2].tolist(), [[1, 3], [2, 4]])
            self.assertEqual(output.dtype, torch.bfloat16)
            self.assertEqual(len(digest), 64)
            for malformed in (data[:-1], data + b"\0", bytes(12)):
                path.write_bytes(malformed)
                with self.assertRaises(ValueError):
                    load_case(path)

    def test_matrix(self):
        shapes = expected_shapes()
        self.assertEqual(len(shapes), 14)
        self.assertEqual(sum(n % 128 == 0 and k % 128 == 0 for m, n, k in shapes), 12)

    def test_quant_layout(self):
        m, k = 2, 256
        raw = struct.pack("<II", m, k) + bytes(m * k * 3) + struct.pack("<4f", 1, 2, 3, 4)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "case.quant"
            path.write_bytes(raw)
            shape, x, q, scales, _ = load_quant_case(path)
            self.assertEqual(shape, (m, k))
            self.assertEqual(x.dtype, torch.bfloat16)
            self.assertEqual(q.dtype, torch.uint8)
            self.assertEqual(scales.tolist(), [[1, 3], [2, 4]])
            path.write_bytes(raw[:-1])
            with self.assertRaises(ValueError):
                load_quant_case(path)

    def test_tensor_size_is_exact(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tensor.bin"
            path.write_bytes(struct.pack("<4f", 1, 2, 3, 4))
            value, digest = load_tensor(path, torch.float32, (2, 2))
            self.assertEqual(value.tolist(), [[1, 2], [3, 4]])
            self.assertEqual(len(digest), 64)
            for shape in ((1, 2), (0, 4), (4, 4)):
                with self.assertRaises(ValueError):
                    load_tensor(path, torch.float32, shape)

    def test_packet_shards_and_missing_rank(self):
        from safetensors.torch import save_file

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint"
            checkpoint.mkdir()
            (root / "outputs").mkdir()
            (root / "inputs").mkdir()
            (root / "measurement.json").write_text(json.dumps(dict(
                tp=2, batch=2, scope="single-block-decode", oracle_verified=True)))
            (root / "inputs/reference.json").write_text(json.dumps(dict(batch=2, layer=3)))
            name = "model.layers.3.self_attn.o_proj.weight"
            w = torch.zeros((128, 256), dtype=torch.float8_e4m3fn)
            w.view(torch.uint8)[:, 128:] = 17
            scales = torch.tensor([[1.0, 2.0]])
            save_file({name: w, name + "_scale_inv": scales}, checkpoint / "model.safetensors")
            (checkpoint / "model.safetensors.index.json").write_text(json.dumps(dict(weight_map={
                name: "model.safetensors", name + "_scale_inv": "model.safetensors"})))
            for rank in range(2):
                for tensor_name, value in (
                    ("act.oat", torch.zeros((2, 128), dtype=torch.bfloat16)),
                    ("act.blk_xq", torch.zeros((2, 128), dtype=torch.uint8)),
                    ("act.blk_xs", torch.ones((1, 2))),
                    ("act.og_tp", torch.zeros((2, 128), dtype=torch.bfloat16)),
                    (name + "_fp8", w[:, rank * 128:(rank + 1) * 128].view(torch.uint8)),
                    (name + "_scale_inv", scales[:, rank:rank + 1]),
                ):
                    data = value.contiguous().view(torch.uint8).numpy().tobytes()
                    (root / "outputs" / f"rank{rank}.{tensor_name}.bin").write_bytes(data)
            cases = list(packet_oproj_cases(root, checkpoint, 2))
            self.assertEqual(len(cases), 2)
            self.assertTrue(all(row["checkpoint_weight_bitwise"] and row["checkpoint_scale_bitwise"]
                                for row, _ in cases))
            self.assertEqual(cases[1][1][4].tolist(), [[2.0]])
            wrong_scale = root / "outputs" / f"rank1.{name}_scale_inv.bin"
            wrong_scale.write_bytes(struct.pack("<f", 1.0))
            self.assertFalse(list(packet_oproj_cases(root, checkpoint, 2))[1][0]["checkpoint_scale_bitwise"])
            (root / "outputs/rank1.act.og_tp.bin").unlink()
            with self.assertRaises(FileNotFoundError):
                list(packet_oproj_cases(root, checkpoint, 2))
            with self.assertRaises(ValueError):
                list(packet_oproj_cases(root, checkpoint, 8))

    def test_shared_column_and_row_shards(self):
        from safetensors.torch import save_file

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint"
            checkpoint.mkdir()
            (root / "inputs").mkdir()
            (root / "outputs").mkdir()
            measurement = dict(tp=2, batch=2, scope="single-block-decode", oracle_verified=True)
            (root / "measurement.json").write_text(json.dumps(measurement))
            (root / "inputs/reference.json").write_text(json.dumps(dict(batch=2, layer=3)))
            prefix = "model.layers.3.mlp.shared_experts."
            originals = {}
            for p in ("gate", "up", "down"):
                w = torch.zeros((128, 256) if p == "down" else (256, 128), dtype=torch.float8_e4m3fn)
                s = torch.ones((1, 2) if p == "down" else (2, 1))
                if p == "down":
                    w.view(torch.uint8)[:, 128:] = 17
                    s[:, 1:] = 2
                else:
                    w.view(torch.uint8)[128:] = 17
                    s[1:] = 2
                originals[prefix + p + "_proj.weight"] = w
                originals[prefix + p + "_proj.weight_scale_inv"] = s
            save_file(originals, checkpoint / "model.safetensors")
            (checkpoint / "model.safetensors.index.json").write_text(json.dumps(dict(
                weight_map={name: "model.safetensors" for name in originals})))
            for rank in range(2):
                tensors = {"act." + name: torch.zeros((2, 128), dtype=dtype) for name, dtype in (
                    ("xn2", torch.bfloat16), ("sh_gate", torch.bfloat16), ("shfu_up", torch.bfloat16),
                    ("shfu", torch.bfloat16), ("shared", torch.bfloat16),
                    ("sh_xq", torch.uint8), ("sh_hq", torch.uint8))}
                tensors.update({"act." + name: torch.ones((1, 2)) for name in ("sh_xs", "sh_hs")})
                for name, whole in originals.items():
                    width = 1 if name.endswith("_scale_inv") else 128
                    part = (whole[:, rank * width:(rank + 1) * width] if "down_proj" in name
                            else whole[rank * width:(rank + 1) * width])
                    tensors[name + ("_fp8" if name.endswith(".weight") else "")] = part
                for name, value in tensors.items():
                    (root / "outputs" / f"rank{rank}.{name}.bin").write_bytes(
                        value.contiguous().view(torch.uint8).numpy().tobytes())
            cases = list(packet_shared_cases(root, checkpoint, 2))
            self.assertEqual(len(cases), 2)
            self.assertTrue(all(all(row["checkpoint_bitwise"].values()) for row, _ in cases))
            self.assertEqual(cases[1][1]["down.weight_scale_inv"].item(), 2)
            reference_rows = []
            for row, _ in cases:
                reference_rows.append(dict(**row, quantization={stage: {
                    check: True for check in ("fp8_bitwise", "scales_bitwise", "repeat_bitwise")}
                    for stage in ("input", "hidden")},
                    stages={stage: {"reference_repeat_bitwise": True} for stage in ("gate_up", "down")}))
                for stage, size in (("gate_up", 2 * 256 * 2), ("down", 2 * 128 * 2)):
                    for tag in ("reference", "repeat"):
                        (root / f"ref.rank{row['rank']}.{stage}.{tag}.bf16").write_bytes(bytes(size))
            reference_json = root / "ref.json"
            reference_json.write_text(json.dumps(dict(cases=reference_rows)))
            export_dir = root / "replay"
            export_dir.mkdir()
            args = SimpleNamespace(capture=root, checkpoint=checkpoint, tp=2,
                                   reference_json=reference_json, output=export_dir / "replay.json")
            self.assertEqual(export_shared(args), 0)
            self.assertEqual(len(list(export_dir.glob("*.bin"))), 6)
            self.assertFalse(json.loads(args.output.read_text())["precision_qualified"])
            self.assertTrue((load_case(export_dir / "rank1.gate.bin")[1][1].view(torch.uint8) == 17).all())
            reference_rows[0]["hashes"] = {}
            reference_json.write_text(json.dumps(dict(cases=reference_rows)))
            with self.assertRaises(ValueError):
                export_shared(args)
            bad = root / "outputs" / f"rank1.{prefix}up_proj.weight_scale_inv.bin"
            bad.write_bytes(struct.pack("<f", 1))
            self.assertFalse(list(packet_shared_cases(root, checkpoint, 2))[1][0]["checkpoint_bitwise"]["up.weight_scale_inv"])
            measurement["oracle_verified"] = False
            (root / "measurement.json").write_text(json.dumps(measurement))
            self.assertTrue(all(not row["block_oracle_verified"]
                                for row, _ in packet_shared_cases(root, checkpoint, 2)))
            bad.unlink()
            with self.assertRaises(FileNotFoundError):
                list(packet_shared_cases(root, checkpoint, 2))
            with self.assertRaises(ValueError):
                list(packet_shared_cases(root, checkpoint, 3))
            measurement.pop("oracle_verified")
            (root / "measurement.json").write_text(json.dumps(measurement))
            with self.assertRaises(ValueError):
                list(packet_shared_cases(root, checkpoint, 2))


if __name__ == "__main__":
    unittest.main()
