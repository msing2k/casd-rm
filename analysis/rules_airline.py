import json,sys,re,collections
R=[json.loads(l) for l in open(sys.argv[1])]
WRITE={"book_reservation","cancel_reservation","update_reservation_flights","update_reservation_passengers","update_reservation_baggages","send_certificate"}
SEARCH={"search_direct_flight","search_onestop_flight"}
def C(x): return [s for s in x["steps"] if s.get("tool") and s["role"]=="assistant"]
def before(x,s):
    o=""
    for p in x["steps"]:
        if p is s: break
        o+=" "+p.get("text","")+" "+p.get("obs","")
    return o
D={}
def det(n):
    def w(f): D[n]=f; return f
    return w
@det("A01")
def _(x): return any(s["tool"]=="get_reservation_details" and str(s["args"].get("reservation_id","")) not in before(x,s) or (s["tool"]=="get_reservation_details" and s["error"]) for s in C(x))
@det("A02")
def _(x):
    for s in C(x):
        if s["tool"]=="get_user_details": return False
        if s["tool"]=="get_reservation_details": return True
    return False
@det("A03")
def _(x): return any(s["tool"]=="get_flight_status" and s["error"] for s in C(x))
@det("A04")
def _(x):
    sig=collections.Counter((s["tool"],json.dumps(s["args"],sort_keys=True)) for s in C(x) if s["tool"] in SEARCH)
    return any(v>=2 for v in sig.values()) or sum(sig.values())>6
@det("A05")
def _(x):
    st=x["steps"]
    for i,s in enumerate(st):
        if s.get("tool") in WRITE and s["role"]=="assistant":
            u=[p for p in st[:i] if p["role"]=="user" and p.get("text")]
            if not u or not re.search(r"\b(yes|confirm|proceed|go ahead|sure)\b",u[-1]["text"].lower()): return True
    return False
@det("A06")
def _(x): return any(s["tool"]=="update_reservation_passengers" and s["error"] for s in C(x))
@det("A07")
def _(x): return any(s["tool"]=="book_reservation" and s["error"] for s in C(x))
@det("A08")
def _(x): return any(s["tool"] in WRITE and not s["error"] and s["tool"] not in x["gold_tools"] for s in C(x))
@det("A09")
def _(x):
    f=set()
    for s in C(x):
        k=(s["tool"],json.dumps(s["args"],sort_keys=True))
        if k in f: return True
        if s["error"]: f.add(k)
    return False
@det("A10")
def _(x):
    gw={g for g in x["gold_tools"] if g in WRITE}; aw={s["tool"] for s in C(x) if s["tool"] in WRITE and not s["error"]}
    return bool(gw-aw)
@det("A11")
def _(x):
    c=collections.Counter(s["tool"] for s in C(x) if s["tool"] in WRITE and not s["error"])
    g=collections.Counter(t for t in x["gold_tools"] if t in WRITE)
    return any(v>g.get(k,0) and v>=2 for k,v in c.items())
TEXT={
"A01":("Never guess a reservation ID. Reservation IDs are 6-character codes listed in the user's profile (get_user_details) or given by the user; never pass a user ID, flight number or placeholder like 'AA1234'.",["get_reservation_details","cancel_reservation","update_reservation_flights","update_reservation_passengers","update_reservation_baggages"],"pre-call",True),
"A02":("After identifying the user, call get_user_details first to obtain their reservation list and payment methods, then open the relevant reservation with get_reservation_details.",["get_user_details","get_reservation_details"],"planning",False),
"A03":("get_flight_status needs a flight number (e.g. 'HAT170') and a date from the reservation's flight segments, not a reservation ID.",["get_flight_status"],"pre-call",False),
"A04":("Do not repeat the same flight search. Search each origin-destination-date once; if no suitable flight is found, tell the user and ask whether to try another date or a one-stop search instead of looping.",["search_direct_flight","search_onestop_flight"],"post-call",False),
"A05":("Before any booking, cancellation or reservation update, state the exact change and cost to the user and wait for an explicit 'yes'.",["book_reservation","cancel_reservation","update_reservation_flights","update_reservation_passengers","update_reservation_baggages"],"pre-call",False),
"A06":("update_reservation_passengers must keep the same number of passengers as the reservation; only change passenger details, never add or remove passengers.",["update_reservation_passengers"],"pre-call",False),
"A07":("For book_reservation, compute the total price with calculate, use only payment IDs from the user's profile, and make the payment amounts add up exactly to the total.",["book_reservation","calculate"],"pre-call",False),
"A08":("Check the policy before any state-changing action: verify cancellation/change eligibility (cabin class, booking time, insurance, flight status). If the request is not allowed, explain why and do not perform the action.",["cancel_reservation","update_reservation_flights","book_reservation","update_reservation_baggages"],"planning",True),
"A09":("If a tool call returns an error, do not repeat the same call with the same arguments; correct the arguments from earlier results or ask the user.",[],"post-call",True),
"A10":("Once the user has confirmed and the change is allowed, actually execute the corresponding reservation tool before ending the conversation.",["book_reservation","cancel_reservation","update_reservation_flights","update_reservation_passengers","update_reservation_baggages"],"response",False),
"A11":("Perform each reservation update once: collect all requested changes, confirm them together, then make one call per update type instead of repeating updates.",["update_reservation_flights","update_reservation_baggages","update_reservation_passengers"],"planning",False),
}
N=len(R); bf=sum(x["reward"]<1 for x in R)/N
allc=[s for x in R for s in C(x)]; be=sum(s["error"] for s in allc)/len(allc)
M=10; bank=[]
for rid,f in D.items():
    hit=[x for x in R if f(x)]; s=len(hit); fl=sum(x["reward"]<1 for x in hit)
    hc=[c for x in hit for c in C(x)]; pe=(sum(c["error"] for c in hc)/len(hc)) if hc else be
    le=s/(s+M)*((fl/s if s else 0)-bf); lc=s/(s+M)*(pe-be)
    t,tools,ph,g=TEXT[rid]
    bank.append({"rule_id":rid,"text":t,"scope":{"domain":"airline","tools":tools,"phase":ph},"support":s,"failures":fl,
      "episode_fail_rate":round(fl/s,3) if s else None,"call_error_rate":round(pe,3),"lift_episode":round(le,4),"lift_call":round(lc,4),
      "confidence":round(min(.95,max(.05,.5+max(le,lc))),4),"global":g,"conflicts_with":[],"n_applied":0,"archived":False})
print(f"N={N} base_fail={bf:.3f} base_call_err={be:.3f}")
for b in bank: print(b["rule_id"],"sup",b["support"],"fail",b["failures"],"c0",b["confidence"],"G" if b["global"] else "")
json.dump(bank,open(sys.argv[2],"w"),indent=1)
F=[x for x in R if x["reward"]<1]; print("failed",len(F))
cats={"Fabricated/ungrounded reservation IDs":["A01","A03"],"Missing prerequisite lookup":["A02"],"Search loops / repeated calls":["A04","A09"],
"Write without explicit confirmation":["A05"],"Invalid write arguments (passengers/payment)":["A06","A07"],"Policy-violating or unrequested write":["A08","A11"],"Required action never executed":["A10"]}
for k,v in cats.items():
    n=sum(any(D[r](x) for r in v) for x in F); print(f"{n:2d} {100*n/len(F):5.1f}% {k}")
