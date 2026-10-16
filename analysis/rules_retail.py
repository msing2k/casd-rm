import json,sys,re,collections
R=[json.loads(l) for l in open(sys.argv[1])]
R=[x for x in R if x["termination"]!="infrastructure_error"]          # exclude infra failures
WRITE={"cancel_pending_order","modify_pending_order_items","modify_pending_order_address","modify_pending_order_payment",
       "modify_user_address","return_delivered_order_items","exchange_delivered_order_items"}
AUTH={"find_user_id_by_email","find_user_id_by_name_zip"}
OID=re.compile(r"^#W\d{7}$")
def C(x): return [s for s in x["steps"] if s.get("tool") and s["role"]=="assistant"]
def ctx_before(x,s):
    out=""
    for p in x["steps"]:
        if p is s: break
        out+=" "+p.get("text","")+" "+p.get("obs","")
    return out
def ids_in_args(s,keys): 
    v=[]
    for k in keys:
        a=s["args"].get(k)
        if isinstance(a,list): v+= [str(i) for i in a]
        elif a is not None: v.append(str(a))
    return v
D={}
def det(name):
    def w(f): D[name]=f; return f
    return w
@det("R01")
def _(x): # ungrounded order/item ids
    for s in C(x):
        for i in ids_in_args(s,["order_id","item_id","item_ids","new_item_ids"]):
            if i.lstrip("#") not in ctx_before(x,s): return True
    return False
@det("R02")
def _(x): return any(("order_id" in s["args"]) and not OID.match(str(s["args"]["order_id"])) for s in C(x))
@det("R03")
def _(x):
    seen=False
    for s in C(x):
        if s["tool"]=="get_user_details" and not s["error"]: seen=True
        if s["tool"]=="get_order_details" and not seen: return True
    return False
@det("R04")
def _(x): return any(s["tool"]=="get_product_details" and not str(s["args"].get("product_id","")).isdigit() for s in C(x))
@det("R05")
def _(x):
    failed=set()
    for s in C(x):
        k=(s["tool"],json.dumps(s["args"],sort_keys=True))
        if k in failed: return True
        if s["error"]: failed.add(k)
    return False
@det("R06")
def _(x):
    for s in C(x):
        if s["tool"] in AUTH and not s["error"]: return False
        if s["tool"] not in AUTH: return True
    return False
@det("R07")
def _(x):
    c=C(x)
    return any(a["tool"]=="find_user_id_by_name_zip" and a["error"] and b["tool"]=="find_user_id_by_name_zip" for a,b in zip(c,c[1:]))
@det("R08")
def _(x):
    st=x["steps"]
    for i,s in enumerate(st):
        if s.get("tool") in WRITE and s["role"]=="assistant":
            u=[p for p in st[:i] if p["role"]=="user" and p.get("text")]
            if not u or not re.search(r"\b(yes|confirm|proceed|go ahead|sure)\b",u[-1]["text"].lower()): return True
    return False
@det("R09")
def _(x): return any(s["error"] and re.search(r"Non-(pending|delivered)",s.get("obs","")) for s in C(x))
@det("R10")
def _(x): return any(s["tool"] in WRITE and s["error"] and re.search(r"Variant not found|Some item not found|New item|Number of|not found or available|should be different",s.get("obs","")) for s in C(x))
@det("R11")
def _(x):
    gw={g for g in x["gold_tools"] if g in WRITE}
    aw={s["tool"] for s in C(x) if s["tool"] in WRITE and not s["error"]}
    return bool(gw) and not aw
@det("R12")
def _(x): return sum(1 for s in C(x) if s["tool"]=="modify_pending_order_items")>1
@det("R13")
def _(x): return any(s["tool"]=="get_item_details" and s["error"] for s in C(x))

TEXT={
"R01":("Never invent or guess an order ID, item ID or product ID. Only use identifiers that appeared earlier in the conversation or in a previous tool result; if an ID is unknown, look it up or ask the user.",["get_order_details","get_item_details","cancel_pending_order","return_delivered_order_items","exchange_delivered_order_items","modify_pending_order_items"],"pre-call",True),
"R02":("Order IDs have the exact form '#W' followed by 7 digits (e.g. '#W1234567'). Never pass bare digits, placeholder numbers or IDs without the '#W' prefix.",["get_order_details","cancel_pending_order","modify_pending_order_items","modify_pending_order_address","return_delivered_order_items","exchange_delivered_order_items"],"pre-call",False),
"R03":("After the user is authenticated, call get_user_details to obtain the user's list of order IDs before calling get_order_details; select the order from that list instead of guessing.",["get_user_details","get_order_details"],"planning",False),
"R04":("get_product_details requires a numeric product_id taken from the 'product_id' field of an order item (from get_order_details). Never pass a product name such as 'keyboard'.",["get_product_details"],"pre-call",False),
"R05":("If a tool call returns an error, do not repeat the same call with the same arguments. Change the arguments using information from the conversation or tool results, ask the user for the missing detail, or try another tool.",[],"post-call",True),
"R06":("Before any order lookup or account action, authenticate the user with find_user_id_by_email or find_user_id_by_name_zip, even if the user already gave an order ID.",["find_user_id_by_email","find_user_id_by_name_zip"],"planning",True),
"R07":("If find_user_id_by_name_zip fails, do not retry it with the same details; ask the user for their email and use find_user_id_by_email instead.",["find_user_id_by_name_zip","find_user_id_by_email"],"post-call",False),
"R08":("Before any state-changing action (cancel, modify, return, exchange, address change), list the exact action details to the user and wait for an explicit 'yes' before calling the tool.",["cancel_pending_order","modify_pending_order_items","modify_pending_order_address","modify_pending_order_payment","modify_user_address","return_delivered_order_items","exchange_delivered_order_items"],"pre-call",False),
"R09":("Check the order status from get_order_details before acting: only 'pending' orders can be cancelled or modified; only 'delivered' orders can be returned or exchanged.",["get_order_details","cancel_pending_order","modify_pending_order_items","return_delivered_order_items","exchange_delivered_order_items"],"pre-call",False),
"R10":("For returns, exchanges and item modifications, take item_ids from the order's items and new_item_ids from available variants in get_product_details of the same product; the new item must differ from the old one and be available.",["return_delivered_order_items","exchange_delivered_order_items","modify_pending_order_items","get_product_details"],"pre-call",False),
"R11":("Once the user has confirmed the requested change and all checks pass, actually execute the corresponding action tool; do not end the conversation after only describing what would be done.",["cancel_pending_order","modify_pending_order_items","modify_pending_order_address","return_delivered_order_items","exchange_delivered_order_items"],"response",False),
"R12":("modify_pending_order_items can be called only once per order: collect all items the user wants to change first, confirm them together, then make a single call.",["modify_pending_order_items"],"planning",False),
"R13":("Do not call get_item_details with guessed IDs; item details and variant item_ids are available from get_order_details and get_product_details.",["get_item_details","get_product_details"],"pre-call",False),
}
N=len(R); base_fail=sum(x["reward"]<1 for x in R)/N
# call-level error base rate (for tool-call rules)
allc=[s for x in R for s in C(x)]; base_err=sum(s["error"] for s in allc)/len(allc)
M,CMIN,CMAX=10,0.05,0.95
bank=[]
for rid,f in D.items():
    hit=[x for x in R if f(x)]; s=len(hit); fl=sum(x["reward"]<1 for x in hit)
    # call-level: error rate of calls in episodes exhibiting the pattern vs base
    hc=[c for x in hit for c in C(x)]; perr=(sum(c["error"] for c in hc)/len(hc)) if hc else base_err
    lift_ep=s/(s+M)*((fl/s if s else 0)-base_fail)
    lift_call=s/(s+M)*(perr-base_err)
    lift=max(lift_ep,lift_call)
    txt,tools,phase,glob=TEXT[rid]
    bank.append({"rule_id":rid,"text":txt,"scope":{"domain":"retail","tools":tools,"phase":phase},
      "support":s,"failures":fl,"episode_fail_rate":round(fl/s,3) if s else None,"call_error_rate":round(perr,3),
      "lift_episode":round(lift_ep,4),"lift_call":round(lift_call,4),
      "confidence":round(min(CMAX,max(CMIN,0.5+lift)),4),"global":glob,"conflicts_with":[],"n_applied":0,"archived":False})
print(f"N={N} base_fail={base_fail:.3f} base_call_err={base_err:.3f}")
for b in bank: print(b["rule_id"],"support",b["support"],"fail",b["failures"],"callErr",b["call_error_rate"],"c0",b["confidence"],"G" if b["global"] else "")
json.dump(bank,open(sys.argv[2],"w"),indent=1)
