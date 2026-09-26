p = '/root/plow/.claude/worktrees/tts-veena-chatterbox/scripts/tts/sample_kernel_bench.py'
s = open(p).read()
old = '''    for p, h in zip(args.cubins[1:], hists[1:]):
        tvd = 0.5 * (h - hists[0]).abs().sum().item()
        # Same-size resample of the control gives the sampling-noise floor for this D.
        print(f"TVD {p.split('/')[-1]} vs {args.cubins[0].split('/')[-1]}: {tvd:.4f} (support {(hists[0] > 0).sum().item()} tokens, D={D})")'''
new = '''    # Exact kept distribution (the sampler contract): e = exp((l - max)/t), floor = largest v with
    # kept mass > top_p * total, p = e[e >= floor] / sum.
    l = logits[0].float()
    e = torch.exp((l - l.max()) / args.temp)
    srt = torch.sort(e, descending=True).values
    csum = torch.cumsum(srt, 0)
    cut = int(torch.searchsorted(csum, args.top_p * e.sum()).item())
    floor = srt[min(cut, len(srt) - 1)]
    exact = torch.where(e >= floor, e, torch.zeros_like(e))
    exact /= exact.sum()
    ref = torch.multinomial(exact, D, replacement=True, generator=torch.Generator(device="cuda").manual_seed(11))
    noise = 0.5 * (torch.bincount(ref, minlength=V).float() / D - exact).abs().sum().item()
    print(f"kept support {(exact > 0).sum().item()} tokens, D={D}; multinomial noise floor TVD {noise:.4f}")
    for p, h in zip(args.cubins, hists):
        print(f"TVD {p.split('/')[-1]} vs exact: {0.5 * (h - exact).abs().sum().item():.4f}")'''
assert s.count(old) == 1
s = s.replace(old, new)
open(p, 'w').write(s)
print("ok")
