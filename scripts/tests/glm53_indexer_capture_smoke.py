import json
from pathlib import Path
import tempfile
from types import SimpleNamespace

import torch

from glm53_indexer_capture import capture_config
from vllm_forward_capture import install


_context = SimpleNamespace(attn_metadata=None)


def forward_context():
    return _context


class Indexer:
    def __init__(self):
        self.k_cache = SimpleNamespace(
            prefix="model.layers.6.self_attn.indexer.k_cache",
            kv_cache=torch.zeros((2, 16, 132), dtype=torch.uint8),
        )

    def forward_hip(self, hidden, q, k, weights):
        self.k_cache.kv_cache.add_(1)
        return torch.zeros((hidden.shape[0], 2048), dtype=torch.int32)

    def forward(self, hidden, qr, positions):
        rows = hidden.shape[0]
        return self.forward_hip(hidden,
                                torch.zeros((rows, 32, 128), dtype=torch.float8_e4m3fn),
                                torch.zeros((rows, 128), dtype=torch.bfloat16),
                                torch.zeros((rows, 32), dtype=torch.float32))


class Attention:
    scale = 0.0625
    topk_indices_buffer = torch.zeros((1, 2048), dtype=torch.int32)

    def _forward_mla(self, layer, query, cache, metadata):
        return query[:, :8, :512]


def main():
    with tempfile.TemporaryDirectory() as tmp:
        config = capture_config(tmp, "a" * 64, attention=True)
        config = json.loads(json.dumps(config).replace(
            "vllm.forward_context.get_forward_context", "__main__.forward_context"
        ).replace(
            "vllm.v1.attention.backends.mla.rocm_aiter_mla_sparse.ROCMAiterMLASparseImpl._forward_mla",
            "__main__.Attention._forward_mla",
        ).replace(
            "vllm.model_executor.models.deepseek_v2.Indexer.forward", "__main__.Indexer.forward"
        ).replace(
            "vllm.model_executor.layers.sparse_attn_indexer.SparseAttnIndexer.forward_hip",
            "__main__.Indexer.forward_hip",
        ))
        install(config)
        indexer = Indexer()
        for invocation, rows in enumerate((8, 8, 1)):
            if invocation:
                decode = None if rows == 8 else SimpleNamespace(
                    block_table=torch.tensor([[1, 0]], dtype=torch.int32),
                    seq_lens=torch.tensor([9], dtype=torch.int32),
                    decode_lens=torch.tensor([1], dtype=torch.int32),
                    schedule_metadata=None,
                )
                _context.attn_metadata = {indexer.k_cache.prefix: SimpleNamespace(
                    num_decodes=int(rows == 1), num_decode_tokens=int(rows == 1),
                    num_prefills=int(rows == 8), num_prefill_tokens=8 if rows == 8 else 0,
                    max_seq_len=8 if rows == 8 else 9,
                    slot_mapping=torch.arange(rows, dtype=torch.int64),
                    seq_lens=torch.tensor([8 if rows == 8 else 9], dtype=torch.int32),
                    decode=decode,
                )}
            indexer.forward(torch.zeros((rows, 6144), dtype=torch.bfloat16),
                            torch.ones((rows, 2048), dtype=torch.bfloat16),
                            torch.arange(rows, dtype=torch.int64))
            if invocation == 0:
                assert not list(Path(tmp).glob("*.json"))
        records = [json.loads(p.read_text()) for p in Path(tmp).glob("*.json")]
        assert len(records) == 27
        for invocation, count in ((1, 12), (2, 15)):
            group = [r for r in records if r["invocation_index"] == invocation]
            assert len(group) == count
            assert len({r["context_sha256"] for r in group}) == 1
            hidden = next(r for r in group if r["semantic"] == "indexer.hidden")
            original = next(r for r in group if r["semantic"] == "indexer.input.hidden")
            assert hidden["sha256"] == original["sha256"]
            for phase, byte in (("before", invocation), ("after", invocation + 1)):
                record = next(r for r in group if r["semantic"] == f"indexer.cache.{phase}")
                assert (Path(tmp) / record["file"]).read_bytes() == bytes([byte]) * (2 * 16 * 132)
        assert not any(r["semantic"] == "indexer.decode.schedule_metadata" for r in records)
        metadata = SimpleNamespace(num_actual_tokens=1, max_seq_len=9, block_size=1, topk_tokens=2048,
            block_table=torch.zeros((1, 16), dtype=torch.int32), req_id_per_token=torch.zeros(1, dtype=torch.int32),
            qo_indptr=torch.tensor([0, 1], dtype=torch.int32), paged_kv_indptr=torch.tensor([0, 9], dtype=torch.int32),
            paged_kv_indices=torch.arange(9, dtype=torch.int32), paged_kv_last_page_len=torch.ones(1, dtype=torch.int32),
            work_meta_data=None, work_indptr=None, work_info_set=None, reduce_indptr=None,
            reduce_final_map=None, reduce_partial_map=None)
        query = torch.ones((1, 16, 576), dtype=torch.bfloat16)
        layer = SimpleNamespace(layer_name="model.layers.5.self_attn.attn")
        attention = Attention()
        attention._forward_mla(layer, query, query, metadata)
        assert not list(Path(tmp).glob("attention.*.json"))
        layer.layer_name = "model.layers.6.self_attn.attn"
        attention._forward_mla(layer, query, query, metadata)
        attention_records = [json.loads(p.read_text()) for p in Path(tmp).glob("attention.*.json")]
        assert len(attention_records) == 10
        assert {r["invocation_index"] for r in attention_records} == {1}
        assert len({r["context_sha256"] for r in attention_records}) == 1
        output = next(r for r in attention_records if r["semantic"] == "attention.output")
        assert output["source_shape"] == [1, 8, 512]
        assert (Path(tmp) / output["file"]).read_bytes() == query[:, :8, :512].contiguous().view(torch.uint8).numpy().tobytes()


if __name__ == "__main__":
    main()
