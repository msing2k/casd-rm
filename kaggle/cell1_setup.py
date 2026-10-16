import os, time, json, subprocess, re, glob, random
def sh(c):
    r = subprocess.run(c, shell=True, capture_output=True, text=True); print((r.stdout + r.stderr)[-800:]); return r.returncode == 0
# --- Ollama + models ---
if not sh("which ollama"):
    sh("apt-get -qq update && apt-get -qq install -y zstd pciutils > /dev/null")
    sh("curl -fsSL https://ollama.com/install.sh | sh")
os.environ.update(OLLAMA_NUM_PARALLEL="2", OLLAMA_CONTEXT_LENGTH="32768", OLLAMA_KEEP_ALIVE="-1", OPENAI_API_KEY="unused-local")
if subprocess.run("curl -s localhost:11434/api/tags", shell=True, capture_output=True).returncode != 0:
    subprocess.Popen("nohup ollama serve > /kaggle/working/ollama.log 2>&1 &", shell=True); time.sleep(15)
sh("ollama pull qwen2.5:14b > /dev/null; ollama pull qwen2.5:7b > /dev/null; ollama list")
# --- tau2-bench ---
os.chdir("/kaggle/working")
if not os.path.exists("tau2-bench"):
    sh("git clone -q https://github.com/sierra-research/tau2-bench")
sh("pip install -q -e tau2-bench sentence-transformers litellm")
os.chdir("/kaggle/working/tau2-bench")
# --- patch hidden OpenAI defaults -> local model ---
LOCAL = "ollama_chat/qwen2.5:7b"
pat = re.compile(r'(["\'])(?:openai/|anthropic/)?(gpt-[\w.\-]+|o[134][\w.\-]*|claude-[\w.\-]+)\1')
changed = []
for p in glob.glob("src/tau2/**/*.py", recursive=True):
    lines = open(p).read().split("\n"); hit = False
    for i, l in enumerate(lines):
        if pat.search(l) and re.search(r'(DEFAULT|default|_LLM|_MODEL|model\s*[:=])', l):
            lines[i] = pat.sub(rf'\1{LOCAL}\1', l); changed.append(f"{p}:{i+1}"); hit = True
    if hit: open(p, "w").write("\n".join(lines))
print("PATCHED:", changed)
print("REMAINING:", subprocess.run("grep -rn '\"gpt-\\|openai/gpt' src/tau2 | grep -iv 'cost\\|price' | head -10",
      shell=True, capture_output=True, text=True).stdout)
# --- helper scripts ---
open("convert_tau2.py", "w").write(r'''
import json, argparse, hashlib, os
ap = argparse.ArgumentParser()
ap.add_argument("--sim", required=True); ap.add_argument("--domain", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args()
p = a.sim if a.sim.endswith(".json") else os.path.join(a.sim, "results.json")
d = json.load(open(p))
tasks = {str(t["id"]): t for t in (d.get("tasks") or json.load(open(f"data/tau2/domains/{a.domain}/tasks.json")))}
with open(a.out, "w") as f:
    for s in d["simulations"]:
        steps, by_id, queue, first_user, ptok = [], {}, [], None, 0
        for i, m in enumerate(s["messages"]):
            role, content = m.get("role"), m.get("content")
            if role == "user" and content and first_user is None: first_user = str(content)
            if role == "assistant": ptok += ((m.get("usage") or {}).get("prompt_tokens") or 0)
            if m.get("tool_calls"):
                for tc in m["tool_calls"]:
                    st = {"t": i, "role": role, "tool": tc["name"], "args": tc.get("arguments", {}), "error": False}
                    steps.append(st); by_id[tc.get("id")] = st; queue.append(st)
                if content: steps.append({"t": i, "role": role, "text": str(content)[:1000]})
            elif role == "tool":
                c = str(content or ""); st = by_id.pop(m.get("id"), None) or (queue[0] if queue else None)
                if st is not None:
                    queue = [q for q in queue if q is not st]
                    st["error"] = bool(m.get("error")) or c.lower().startswith("error"); st["obs"] = c[:300]
            elif content:
                steps.append({"t": i, "role": role, "text": str(content)[:1000]})
        tid = str(s["task_id"]); crit = tasks.get(tid, {}).get("evaluation_criteria") or {}
        f.write(json.dumps({"traj_id": f"{a.domain}_{tid}_{s.get('trial', 0)}", "domain": a.domain, "task_id": tid,
            "trial": s.get("trial", 0), "reward": float((s.get("reward_info") or {}).get("reward") or 0.0),
            "termination": str(s.get("termination_reason")), "duration": s.get("duration"), "agent_prompt_tokens": ptok,
            "gold_tools": [x["name"] for x in (crit.get("actions") or [])],
            "ep_hash": hashlib.md5((first_user or "").encode()).hexdigest(), "steps": steps}) + "\n")
print("converted", len(d["simulations"]), "->", a.out)
'''.lstrip())
open("corpus_stats.py", "w").write(r'''
import json, sys, collections, pandas as pd
recs = [json.loads(l) for p in sys.argv[1:] for l in open(p)]
rows, bigr, bigr_fail = [], collections.Counter(), collections.Counter()
for r in recs:
    calls = [s for s in r["steps"] if s.get("tool") and s["role"] == "assistant"]
    for c in calls: rows.append({"domain": r["domain"], "tool": c["tool"], "error": c["error"], "reward": r["reward"]})
    names = [c["tool"] for c in calls]
    for b in zip(names, names[1:]): bigr[b] += 1; bigr_fail[b] += r["reward"] < 1
df = pd.DataFrame(rows)
dom = pd.DataFrame([{"domain": r["domain"], "reward": r["reward"], "n_steps": len(r["steps"]), "term": r.get("termination"),
       "n_calls": sum(1 for s in r["steps"] if s.get("tool") and s["role"] == "assistant")} for r in recs])
out = {"n_traj": len(recs),
  "per_domain": dom.groupby("domain").agg(n=("reward","size"), success=("reward","mean"), mean_steps=("n_steps","mean"),
                 tool_calls=("n_calls","sum")).round(3).to_dict("index"),
  "termination": dom.groupby(["domain","term"]).size().to_dict().__repr__(),
  "error_rate": df.groupby("domain")["error"].mean().round(3).to_dict() if len(df) else {},
  "per_tool": df.groupby(["domain","tool"]).agg(calls=("error","size"), err_rate=("error","mean"),
               traj_success=("reward","mean")).round(3).reset_index().to_dict("records") if len(df) else [],
  "bigram_fail_rate": sorted([{"a": a, "b": b, "n": n, "fail_rate": round(bigr_fail[(a,b)]/n, 3)}
                       for (a,b), n in bigr.items() if n >= 3], key=lambda x: -x["fail_rate"])[:40],
  "missing_gold_in_failures": collections.Counter(g for r in recs if r["reward"] < 1
       for g in set(r["gold_tools"]) - {s.get("tool") for s in r["steps"]}).most_common(30)}
json.dump(out, open("corpus_stats.json", "w"), indent=1)
print(json.dumps(out["per_domain"], indent=1), out["error_rate"], out["termination"])
'''.lstrip())
# --- splits ---
os.makedirs("splits", exist_ok=True)
for D in ["retail", "airline"]:
    sp = json.load(open(f"data/tau2/domains/{D}/split_tasks.json"))
    if "train" in sp and "test" in sp:
        tr, te = [str(x) for x in sp["train"]], [str(x) for x in sp["test"]]
    else:
        ids = [str(t["id"]) for t in json.load(open(f"data/tau2/domains/{D}/tasks.json"))]
        random.Random(42).shuffle(ids); tr, te = ids[:len(ids)//2], ids[len(ids)//2:]
    json.dump({"train": tr, "test": te}, open(f"splits/{D}.json", "w"))
    print(D, "split keys:", list(sp.keys()), "| train", len(tr), "test", len(te))
print("SETUP COMPLETE")
