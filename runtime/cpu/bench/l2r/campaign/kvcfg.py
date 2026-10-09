import json
for m in ("E2B", "E4B"):
    c = json.load(open(f"/tmp/models/google/gemma-4-{m}-it/config.json")); t = c.get("text_config", c)
    lt = t["layer_types"]
    print(m, "layers", len(lt), "full", [i for i, x in enumerate(lt) if x == "full_attention"], "shared", t.get("num_kv_shared_layers"),
          "kvh", t["num_key_value_heads"], "gkvh", t.get("num_global_key_value_heads"), "hd", t["head_dim"], "ghd", t.get("global_head_dim"),
          "win", t["sliding_window"], "maxpos", t["max_position_embeddings"], "k_eq_v", t.get("attention_k_eq_v"))
