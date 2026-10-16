import os, json, time, hashlib, re
import numpy as np
_RET = None; _STATE = {}
CONFIRM_USER = re.compile(r"\b(yes|yeah|confirm|confirmed|proceed|go ahead|please do|that's correct|correct)\b", re.I)
ASK_CONFIRM = re.compile(r"(would you like (me )?to proceed|shall i|do you (want|confirm)|please confirm|confirm (that|if|whether)|can you confirm)", re.I)
def _g(o, k, d=None): return o.get(k, d) if isinstance(o, dict) else getattr(o, k, d)
def _ret():
    global _RET
    if _RET is None:
        from sentence_transformers import SentenceTransformer
        bank = json.load(open(os.environ["CASD_BANK"]))
        m = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
        _RET = (bank, m, m.encode([r["t"] for r in bank], normalize_embeddings=True))
    return _RET
def retrieve(query, tools, k, phase_commit=False, prev=(), kg=3, delta=.9, v2=False):
    bank, m, E = _ret()
    if k < 0 or k >= len(bank): return bank                      # CASD: all rules
    q = m.encode([query or "start"], normalize_embeddings=True)[0]
    sim, ql, ts = E @ q, (query or "").lower(), set(tools)
    sc = []
    for i, r in enumerate(bank):
        rt = set(r["tools"])
        mu = 0.5 + 0.5 * float(bool(rt & ts) or any(t.lower() in ql for t in rt))
        if v2: sc.append(.5 * sim[i] + .2 * mu + .1 * r["c"] + .2 * r.get("imp", 0))   # + outcome importance
        else:  sc.append(.6 * sim[i] + .3 * mu + .1 * r["c"])
    chosen = [i for i, r in enumerate(bank) if r["g"]][:kg]
    if v2 and phase_commit:                                                  # phase trigger: commit-step rules
        chosen += [i for i, r in enumerate(bank) if r.get("commit") and i not in chosen]
    order = [int(i) for i in np.argsort(sc)[::-1]]
    if v2: order = [i for i in order if i in prev] + [i for i in order if i not in prev]   # stickiness
    added = 0
    for i in order:
        i = int(i)
        if added >= k: break
        if i in chosen or any(float(E[i] @ E[j]) > delta for j in chosen): continue
        chosen.append(i); added += 1
    return [bank[i] for i in chosen]
def install(cls):
    orig = cls.generate_next_message
    K = int(os.environ.get("CASD_K", "5")); LOG = os.environ.get("CASD_LOG", "casd_log.jsonl")
    V2 = os.environ.get("CASD_V2") == "1"
    def patched(self, message, state):
        try:
            msgs = list(_g(state, "messages", []) or []) + ([message] if message is not None else [])
            first_user = next((str(_g(x, "content")) for x in msgs if _g(x, "role") == "user" and _g(x, "content")), "")
            ep = hashlib.md5(first_user.encode()).hexdigest()
            texts, tools, calls = [], [], []
            for x in msgs[-6:]:
                if _g(x, "content"): texts.append(str(_g(x, "content"))[:400])
                for tc in (_g(x, "tool_calls") or []): tools.append(_g(tc, "name"))
            for x in msgs[-8:]:
                for tc in (_g(x, "tool_calls") or []): calls.append((_g(tc, "name"), json.dumps(_g(tc, "arguments"), sort_keys=True, default=str)))
            q = " ".join(texts[-3:])
            last_user = next((str(_g(x, "content")) for x in reversed(msgs) if _g(x, "role") == "user" and _g(x, "content")), "")
            last_asst = next((str(_g(x, "content")) for x in reversed(msgs) if _g(x, "role") == "assistant" and _g(x, "content")), "")
            commit = V2 and bool(CONFIRM_USER.search(last_user) or ASK_CONFIRM.search(last_asst))
            st = _STATE.setdefault(ep, {"hist": []})
            prev = st["hist"][-1] if (V2 and st["hist"]) else []
            t0 = time.perf_counter(); rules = retrieve(q, tools, K, commit, prev, v2=V2); ms = (time.perf_counter() - t0) * 1000
            bank = _ret()[0]
            st["hist"].append([bank.index(r) for r in rules if not r["g"]])
            lines = ["- " + r["t"] for r in rules]
            loop = V2 and len(calls) >= 2 and calls[-1] == calls[-2]
            if loop: lines.append("- WARNING: you just repeated the same tool call with the same arguments. Do not repeat it again; use a different argument from earlier results or ask the user.")
            block = "\n\n<learned_rules>\nFollow these rules learned from past mistakes:\n" + "\n".join(lines) + "\n</learned_rules>"
            SM = type(state.system_messages[0])
            state.system_messages = [SM(role="system", content=self.system_prompt + block)]
            with open(LOG, "a") as f:
                f.write(json.dumps({"ep_hash": ep, "rules": [r["id"] for r in rules], "commit": commit, "loop_warn": loop,
                                    "tokens": len(block) // 4, "ms": round(ms, 2)}) + "\n")
        except Exception as e:
            with open(LOG, "a") as f: f.write(json.dumps({"hook_error": str(e)[:200]}) + "\n")
        return orig(self, message, state)
    patched._casd = True
    cls.generate_next_message = patched
