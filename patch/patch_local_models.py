import re, glob
LOCAL = "ollama_chat/qwen2.5:7b"
pat = re.compile(r'(["\'])(?:openai/|anthropic/)?(gpt-[\w.\-]+|o[134][\w.\-]*|claude-[\w.\-]+)\1')
for p in glob.glob("src/tau2/**/*.py", recursive=True):
    lines = open(p).read().split("\n"); hit = False
    for i, l in enumerate(lines):
        if pat.search(l) and re.search(r'(DEFAULT|default|_LLM|_MODEL|model\s*[:=])', l):
            lines[i] = pat.sub(rf'\1{LOCAL}\1', l); hit = True
    if hit: open(p, "w").write("\n".join(lines))
