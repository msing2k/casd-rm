import os, json, subprocess, time, signal
def say(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)
def ok(c): return subprocess.run(c, shell=True, capture_output=True, text=True)
def ram(): return ok("free -g | awk '/Mem/{print $3\"/\"$2\" GB\"}'").stdout.strip()
say("EVAL v2 CELL STARTED")
os.chdir("/kaggle/working/tau2-bench")
assert os.path.exists("convert_tau2.py"), "convert_tau2.py missing -> run Cell 1 first"

# ---------- SETTINGS (change only these between commits) ----------
DOMAIN    = "retail"            # "retail" or "airline"
SPLIT     = "train"             # "train" = dev check (commit 0); "test" = final runs (commits 1-6)
SEED      = 301                 # one commit per seed: 301, 302, 303
N_TASKS   = 10                  # 10 for dev check; 999 for final runs
METHODS   = {"b0": (None, 0), "casd": (-1, 0), "casdrm": (5, 0), "casdrm2": (5, 1)}   # name: (k, v2 flag)
BATCH, MAX_STEPS = 5, 40
BATCH_TIMEOUT_MIN = 25          # watchdog per method per batch (same for all methods)
TIME_BUDGET_H     = 10.5
START_BATCH       = 1
MA, MU    = "ollama_chat/qwen2.5:14b", "ollama_chat/qwen2.5:7b"

# ---------- RULE BANKS (frozen; distilled from the training corpus only) ----------
BANKS = json.loads(r'''{"retail":[{"id":"R01","t":"Never invent or guess an order ID, item ID or product ID. Only use identifiers that appeared earlier in the conversation or in a previous tool result; if an ID is unknown, look it up or ask the user.","tools":["get_order_details","get_item_details","cancel_pending_order","return_delivered_order_items","exchange_delivered_order_items","modify_pending_order_items"],"c":0.4807,"g":true},{"id":"R02","t":"Order IDs have the exact form '#W' followed by 7 digits (e.g. '#W1234567'). Never pass bare digits, placeholder numbers or IDs without the '#W' prefix.","tools":["get_order_details","cancel_pending_order","modify_pending_order_items","modify_pending_order_address","return_delivered_order_items","exchange_delivered_order_items"],"c":0.5597,"g":false},{"id":"R03","t":"After the user is authenticated, call get_user_details to obtain the user's list of order IDs before calling get_order_details; select the order from that list instead of guessing.","tools":["get_user_details","get_order_details"],"c":0.512,"g":false},{"id":"R04","t":"get_product_details requires a numeric product_id taken from the 'product_id' field of an order item (from get_order_details). Never pass a product name such as 'keyboard'.","tools":["get_product_details"],"c":0.5533,"g":false},{"id":"R05","t":"If a tool call returns an error, do not repeat the same call with the same arguments. Change the arguments using information from the conversation or tool results, ask the user for the missing detail, or try another tool.","tools":[],"c":0.5683,"g":true},{"id":"R06","t":"Before any order lookup or account action, authenticate the user with find_user_id_by_email or find_user_id_by_name_zip, even if the user already gave an order ID.","tools":["find_user_id_by_email","find_user_id_by_name_zip"],"c":0.5048,"g":true},{"id":"R07","t":"If find_user_id_by_name_zip fails, do not retry it with the same details; ask the user for their email and use find_user_id_by_email instead.","tools":["find_user_id_by_name_zip","find_user_id_by_email"],"c":0.5235,"g":false},{"id":"R08","t":"Before any state-changing action (cancel, modify, return, exchange, address change), list the exact action details to the user and wait for an explicit 'yes' before calling the tool.","tools":["cancel_pending_order","modify_pending_order_items","modify_pending_order_address","modify_pending_order_payment","modify_user_address","return_delivered_order_items","exchange_delivered_order_items"],"c":0.4888,"g":false},{"id":"R09","t":"Check the order status from get_order_details before acting: only 'pending' orders can be cancelled or modified; only 'delivered' orders can be returned or exchanged.","tools":["get_order_details","cancel_pending_order","modify_pending_order_items","return_delivered_order_items","exchange_delivered_order_items"],"c":0.5235,"g":false},{"id":"R10","t":"For returns, exchanges and item modifications, take item_ids from the order's items and new_item_ids from available variants in get_product_details of the same product; the new item must differ from the old one and be available.","tools":["return_delivered_order_items","exchange_delivered_order_items","modify_pending_order_items","get_product_details"],"c":0.5299,"g":false},{"id":"R11","t":"Once the user has confirmed the requested change and all checks pass, actually execute the corresponding action tool; do not end the conversation after only describing what would be done.","tools":["cancel_pending_order","modify_pending_order_items","modify_pending_order_address","return_delivered_order_items","exchange_delivered_order_items"],"c":0.6113,"g":false},{"id":"R12","t":"modify_pending_order_items can be called only once per order: collect all items the user wants to change first, confirm them together, then make a single call.","tools":["modify_pending_order_items"],"c":0.5214,"g":false},{"id":"R13","t":"Do not call get_item_details with guessed IDs; item details and variant item_ids are available from get_order_details and get_product_details.","tools":["get_item_details","get_product_details"],"c":0.536,"g":false}],"airline":[{"id":"A01","t":"Never guess a reservation ID. Reservation IDs are 6-character codes listed in the user's profile (get_user_details) or given by the user; never pass a user ID, flight number or placeholder like 'AA1234'.","tools":["get_reservation_details","cancel_reservation","update_reservation_flights","update_reservation_passengers","update_reservation_baggages"],"c":0.5159,"g":true},{"id":"A02","t":"After identifying the user, call get_user_details first to obtain their reservation list and payment methods, then open the relevant reservation with get_reservation_details.","tools":["get_user_details","get_reservation_details"],"c":0.546,"g":false},{"id":"A03","t":"get_flight_status needs a flight number (e.g. 'HAT170') and a date from the reservation's flight segments, not a reservation ID.","tools":["get_flight_status"],"c":0.4953,"g":false},{"id":"A04","t":"Do not repeat the same flight search. Search each origin-destination-date once; if no suitable flight is found, tell the user and ask whether to try another date or a one-stop search instead of looping.","tools":["search_direct_flight","search_onestop_flight"],"c":0.5286,"g":false},{"id":"A05","t":"Before any booking, cancellation or reservation update, state the exact change and cost to the user and wait for an explicit 'yes'.","tools":["book_reservation","cancel_reservation","update_reservation_flights","update_reservation_passengers","update_reservation_baggages"],"c":0.526,"g":false},{"id":"A06","t":"update_reservation_passengers must keep the same number of passengers as the reservation; only change passenger details, never add or remove passengers.","tools":["update_reservation_passengers"],"c":0.5476,"g":false},{"id":"A07","t":"For book_reservation, compute the total price with calculate, use only payment IDs from the user's profile, and make the payment amounts add up exactly to the total.","tools":["book_reservation","calculate"],"c":0.5476,"g":false},{"id":"A08","t":"Check the policy before any state-changing action: verify cancellation/change eligibility (cabin class, booking time, insurance, flight status). If the request is not allowed, explain why and do not perform the action.","tools":["cancel_reservation","update_reservation_flights","book_reservation","update_reservation_baggages"],"c":0.5286,"g":true},{"id":"A09","t":"If a tool call returns an error, do not repeat the same call with the same arguments; correct the arguments from earlier results or ask the user.","tools":[],"c":0.5476,"g":true},{"id":"A10","t":"Once the user has confirmed and the change is allowed, actually execute the corresponding reservation tool before ending the conversation.","tools":["book_reservation","cancel_reservation","update_reservation_flights","update_reservation_passengers","update_reservation_baggages"],"c":0.5816,"g":false},{"id":"A11","t":"Perform each reservation update once: collect all requested changes, confirm them together, then make one call per update type instead of repeating updates.","tools":["update_reservation_flights","update_reservation_baggages","update_reservation_passengers"],"c":0.5816,"g":false}]}''')
# outcome importance = share of failed TRAINING episodes showing the targeted pattern (Table 4); commit-step rules
IMP = {"R01":29,"R02":44,"R03":45,"R04":15,"R05":42,"R06":19,"R07":7,"R08":12,"R09":7,"R10":11,"R11":43,"R12":6,"R13":17,
       "A01":6,"A02":5,"A03":1,"A04":4,"A05":1,"A06":2,"A07":2,"A08":4,"A09":2,"A10":4,"A11":4}
NFAIL = {"retail": 66, "airline": 10}
COMMIT = {"R08", "R11", "R12", "A05", "A07", "A10", "A11"}
for d in BANKS:
    for r in BANKS[d]: r["imp"] = round(IMP[r["id"]] / NFAIL[d], 3); r["commit"] = r["id"] in COMMIT
open("rule_bank.json", "w").write(json.dumps(BANKS[DOMAIN]))
say(f"rule bank: {DOMAIN} = {len(BANKS[DOMAIN])} rules | split={SPLIT} seed={SEED}")

# ---------- HOOK (v1 = similarity retrieval; v2 = + outcome weight, commit-phase trigger, stickiness, loop warning) ----------
HOOK = r'''import os, json, time, hashlib, re
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
    if v2: order = [i for i in order if i in prev] + [i for i in order if i not in prev]   # stickiness: keep last turn's rules first
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
'''
open("src/tau2/casd_rm_hook.py", "w").write(HOOK)
p = "src/tau2/agent/llm_agent.py"; s = open(p).read()
if "CASD-RM PATCH" not in s:
    open(p, "w").write(s + "\n# === CASD-RM PATCH ===\nimport os as _cos\nif _cos.environ.get('CASD_BANK'):\n"
                           "    from tau2.casd_rm_hook import install as _cinstall\n    _cinstall(LLMAgent)\n")
ok("python -c \"from sentence_transformers import SentenceTransformer as S; S('sentence-transformers/all-MiniLM-L6-v2')\"")
t = ok("CASD_BANK=rule_bank.json CASD_K=5 python -c \"import tau2.agent.llm_agent as m; print('PATCHED' if getattr(m.LLMAgent.generate_next_message,'_casd',False) else 'NOT PATCHED')\"")
say("HOOK SELF-TEST:", (t.stdout.strip() or "FAILED").splitlines()[-1])
t = ok("CASD_BANK=rule_bank.json python -c \"from tau2.casd_rm_hook import retrieve; print([r['id'] for r in retrieve('yes please proceed', [], 5, True, [], v2=True)])\"")
say("RETRIEVER v2 SELF-TEST:", (t.stdout.strip() or ("FAILED " + t.stderr[-200:])).splitlines()[-1])

# ---------- OLLAMA ----------
ok("pkill -f 'ollama serve'; sleep 3")
os.environ.update(OLLAMA_NUM_PARALLEL="2", OLLAMA_CONTEXT_LENGTH="16384", OLLAMA_KEEP_ALIVE="-1",
                  OLLAMA_MAX_LOADED_MODELS="2", OLLAMA_SCHED_SPREAD="1")
subprocess.Popen("nohup ollama serve > /kaggle/working/ollama.log 2>&1 &", shell=True); time.sleep(15)
for m in ["qwen2.5:14b", "qwen2.5:7b"]: ok(f"ollama run {m} 'Reply with OK'")
ps = ok("ollama ps").stdout
say("GPU CHECK:", "OK" if ps.count("100% GPU") >= 2 else "WARNING - model partly on CPU", "| RAM", ram())

# ---------- EVALUATION ON THE TEST SPLIT (with watchdog + time budget) ----------
env0 = dict(os.environ, PYTHONUNBUFFERED="1", OPENAI_API_KEY="unused-local")
for k in ["CASD_BANK", "CASD_K", "CASD_DOMAIN", "CASD_LOG", "CASD_V2"]: env0.pop(k, None)
ids = json.load(open(f"splits/{DOMAIN}.json"))[SPLIT][:N_TASKS]
batches = [ids[i:i + BATCH] for i in range(0, len(ids), BATCH)]
os.makedirs("eval/res", exist_ok=True); os.makedirs("eval/logs", exist_ok=True)
say(f"START {DOMAIN}: batches {START_BATCH}..{len(batches)} of {len(batches)} | timeout {BATCH_TIMEOUT_MIN} min/batch | budget {TIME_BUDGET_H} h")
T0 = time.time()
def save_zip():
    ok(f"cp rule_bank.json eval/ && rm -f /kaggle/working/eval_{DOMAIN}_{SPLIT}_s{SEED}.zip && zip -rq /kaggle/working/eval_{DOMAIN}_{SPLIT}_s{SEED}.zip eval")
for b, chunk in enumerate(batches, 1):
    if b < START_BATCH: continue
    if (time.time() - T0) / 3600 > TIME_BUDGET_H:
        say(f"TIME BUDGET reached -> stopping before batch {b}"); break
    for meth, (k, v2) in METHODS.items():
        name = f"ev_{DOMAIN}_{SPLIT}_s{SEED}_{meth}_b{b}_{time.strftime('%H%M%S')}"
        env = dict(env0)
        if k is not None:
            env.update(CASD_BANK=os.path.abspath("rule_bank.json"), CASD_K=str(k), CASD_DOMAIN=DOMAIN,
                       CASD_LOG=os.path.abspath(f"eval/logs/inject_{DOMAIN}_{SPLIT}_s{SEED}_{meth}.jsonl"), CASD_V2=str(v2))
        log, t = f"eval/logs/{name}.log", time.time()
        p = subprocess.Popen(
            f"tau2 run --domain {DOMAIN} --agent-llm {MA} --user-llm {MU} --num-trials 1 --seed {SEED} --max-steps {MAX_STEPS} "
            f"--task-ids {' '.join(chunk)} --max-concurrency 2 --save-to {name}",
            shell=True, stdout=open(log, "w"), stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=env,
            start_new_session=True)
        timed_out = False
        while p.poll() is None:
            time.sleep(60)
            mins = (time.time() - t) / 60
            if mins > BATCH_TIMEOUT_MIN:                      # watchdog: stop a stuck batch
                os.killpg(p.pid, signal.SIGKILL); p.wait(); timed_out = True; break
            if int(mins) % 4 == 0:
                say(f"   batch {b} {meth}: {open(log, errors='ignore').read().count('Simulation Overview')}/{len(chunk)} "
                    f"| {mins:.0f} min | RAM {ram()}")
        out = f"eval/res/{DOMAIN}_{SPLIT}_s{SEED}_{meth}.jsonl"
        ok(f"rm -f /tmp/x.jsonl; python convert_tau2.py --sim data/simulations/{name} --domain {DOMAIN} --out /tmp/x.jsonl")
        got = [json.loads(l) for l in open("/tmp/x.jsonl")] if os.path.exists("/tmp/x.jsonl") else []
        done_ids = {r["task_id"] for r in got}
        for tid in chunk:                                     # unfinished episodes count as failures (same rule for all methods)
            if tid not in done_ids:
                got.append({"traj_id": f"{DOMAIN}_{tid}_0", "domain": DOMAIN, "task_id": tid, "trial": 0, "seed": SEED, "reward": 0.0,
                            "termination": "timeout", "duration": None, "agent_prompt_tokens": 0,
                            "gold_tools": [], "ep_hash": "", "steps": []})
        with open(out, "a") as f:
            for r in got: f.write(json.dumps(r) + "\n")
        R = [json.loads(l)["reward"] for l in open(out) if l.strip()]
        say(f"batch {b}/{len(batches)} {meth:7s}: {'TIMEOUT-KILLED ' if timed_out else ''}{len(chunk) - len(done_ids & set(chunk))} unfinished "
            f"| {meth} so far {sum(r >= 1 for r in R)}/{len(R)} solved | total {(time.time() - T0) / 3600:.2f} h")
    save_zip()

# ---------- SUMMARY + CLEANUP ----------
for meth in METHODS:
    out = f"eval/res/{DOMAIN}_{SPLIT}_s{SEED}_{meth}.jsonl"
    R = [json.loads(l)["reward"] for l in open(out) if l.strip()] if os.path.exists(out) else []
    say(f"FINAL {DOMAIN} {meth:7s}: {sum(r >= 1 for r in R)}/{len(R)} solved ({100 * sum(r >= 1 for r in R) / max(1, len(R)):.1f}%)")
save_zip()
say(f"READY: eval_{DOMAIN}_{SPLIT}_s{SEED}.zip")
ok("pkill -f 'ollama serve'; pkill -f 'tau2 run'")
os.chdir("/kaggle/working"); ok("rm -rf /kaggle/working/tau2-bench /kaggle/working/*.log")
say("Output folder:", ok("ls /kaggle/working").stdout.replace(chr(10), " "))
