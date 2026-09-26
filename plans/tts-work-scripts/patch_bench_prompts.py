p = '/root/plow/.claude/worktrees/tts-veena-chatterbox/scripts/tts/tts_bench.py'
s = open(p).read()


def rep(a, b):
    global s
    assert s.count(a) == 1, a[:60]
    s = s.replace(a, b)


rep("""from veena_ref import PROMPTS
""", """from veena_ref import PROMPTS as VEENA_PROMPTS
from chatterbox_ref import PROMPTS as CBX_TEXTS
""")
rep("""    ap.add_argument("--warmup", type=int, default=2)""", """    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--prompt-set", choices=["veena", "chatterbox"], default="veena")
    ap.add_argument("--voice", default=None, help="override every prompt's voice")""")
rep("""    jobs = [(PROMPTS[i % len(PROMPTS)], i) for i in range(args.n)]""",
    """    prompts = VEENA_PROMPTS if args.prompt_set == "veena" else [("default", t) for t in CBX_TEXTS]
    if args.voice:
        prompts = [(args.voice, t) for _, t in prompts]
    jobs = [(prompts[i % len(prompts)], i) for i in range(args.n)]""")
open(p, 'w').write(s)
print("ok")
