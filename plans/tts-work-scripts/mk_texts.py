import json, os, sys, glob
sys.path.insert(0, "/root/plow/.claude/worktrees/tts-veena-chatterbox/scripts/tts")
from veena_ref import PROMPTS
d = sys.argv[1]
out = {}
for w in glob.glob(f"{d}/*.wav"):
    i = int(os.path.basename(w).split("_")[2])
    out[os.path.basename(w)] = PROMPTS[i % len(PROMPTS)][1]
json.dump(out, open(f"{d}/texts.json", "w"), ensure_ascii=False, indent=0)
print(len(out))
