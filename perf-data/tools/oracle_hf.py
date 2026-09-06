"""Reference oracle: HF transformers (CPU, bf16) forward of Gemma-4-12B on a token-id
prompt. Prints top-5 next tokens for the prompt and for each greedy step, plus per-layer
hidden-state stats, so the plow CPU engine can be compared layer by layer.

usage: python oracle.py <ckpt-dir> --ids 2,818,5279,529,7001,563 [--steps 4] [--layers]
"""
import argparse, sys, time
import torch

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--ids")
    ap.add_argument("--repeat-prompt", type=int, default=0, help="<bos> + repeated fox sentence, truncated to N (matches plow cpu_bench/cpu_probe)")
    ap.add_argument("--dump", help="write final-position logits as bf16 to this file")
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--layers", action="store_true")
    ap.add_argument("--dump-hidden", help="dir: write hidden_states[i] ([T,H] bf16 raw) as h<i>.bin for i <= --max-hidden")
    ap.add_argument("--max-hidden", type=int, default=1)
    a = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.ckpt)
    if a.repeat_prompt:
        base = tok.encode("The quick brown fox jumps over the lazy dog while the river flows quietly past the old mill. ", add_special_tokens=False)
        ids = [2]
        while len(ids) < a.repeat_prompt:
            ids += base
        ids = ids[:a.repeat_prompt]
    else:
        ids = [int(x) for x in a.ids.split(",")]
    print("ids:", ids, repr(tok.decode(ids)))
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(a.ckpt, torch_dtype=torch.bfloat16,
                                                 low_cpu_mem_usage=True).to("cpu")
    model.eval()
    print(f"loaded in {time.time()-t0:.1f}s; type={type(model).__name__}")
    x = torch.tensor([ids])
    with torch.no_grad():
        out = model(x, output_hidden_states=(a.layers or a.dump_hidden is not None), use_cache=True)
        if a.dump_hidden:
            import os
            os.makedirs(a.dump_hidden, exist_ok=True)
            for li, h in enumerate(out.hidden_states):
                if li > a.max_hidden:
                    break
                open(os.path.join(a.dump_hidden, f"h{li}.bin"), "wb").write(
                    h[0].to(torch.bfloat16).contiguous().view(torch.int16).numpy().tobytes())
                hf = h[0].float()
                print(f"  dumped h{li}: shape {tuple(h[0].shape)} mean|.| {hf.abs().mean():.4f} max {hf.abs().max():.3f}")
        logits = out.logits[0, -1].float()
        if a.dump:
            torch.save(logits, a.dump + ".pt")
            open(a.dump, "wb").write(logits.to(torch.bfloat16).view(torch.int16).numpy().tobytes())
        top = torch.topk(logits, 8)
        print("prefill top8:", [(int(i), float(v), tok.decode([int(i)])) for v, i in zip(top.values, top.indices)])
        if a.layers:
            for li, h in enumerate(out.hidden_states):
                hf = h[0].float()
                print(f"  layer {li:2d} hidden: min {hf.min():.4f} max {hf.max():.4f} mean|.| {hf.abs().mean():.4f} last-row[:4] {hf[-1,:4].tolist()}")
        past = out.past_key_values
        nxt = int(torch.argmax(logits))
        gen = [nxt]
        for s in range(a.steps):
            out = model(torch.tensor([[nxt]]), past_key_values=past, use_cache=True)
            past = out.past_key_values
            logits = out.logits[0, -1].float()
            top = torch.topk(logits, 5)
            print(f"step {s} top5:", [(int(i), float(v), tok.decode([int(i)])) for v, i in zip(top.values, top.indices)])
            nxt = int(torch.argmax(logits))
            gen.append(nxt)
        print("greedy:", gen, repr(tok.decode(gen)))

if __name__ == "__main__":
    main()
