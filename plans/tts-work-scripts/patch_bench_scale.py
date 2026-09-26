p = '/root/plow/.claude/worktrees/tts-veena-chatterbox/scripts/tts/sample_kernel_bench.py'
s = open(p).read()
s = s.replace('''    ap.add_argument("--iters", type=int, default=200)''', '''    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--scale", type=float, default=2.0, help="logit noise scale (smaller = broader)")''')
s = s.replace('''* 2.0)''', '''* args.scale)''')
open(p, 'w').write(s)
print(s.count("args.scale"))
