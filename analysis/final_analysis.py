import json,glob,sys,collections,statistics,random
import numpy as np
from scipy.stats import wilcoxon
F=sys.argv[1]; M=["b0","casd","casdrm","casdrm2"]; DOMS=["retail","airline"]
def C(x): return [s for s in x["steps"] if s.get("tool") and s["role"]=="assistant"]
def f1(c,g):
    c,g=collections.Counter(c),collections.Counter(g); tp=sum((c&g).values()); return 2*tp/(sum(c.values())+sum(g.values()))
E=collections.defaultdict(list)   # (dom,m) -> episodes
for f in glob.glob(f"{F}/*/eval/res/*.jsonl"):
    b=f.split("/")[-1][:-6]; dom,_,seed,m=b.split("_",3)
    for l in open(f): r=json.loads(l); r["_seed"]=seed; E[(dom,m)].append(r)
INJ=collections.defaultdict(list)
for f in glob.glob(f"{F}/*/eval/logs/inject_*.jsonl"):
    b=f.split("/")[-1][7:-6]; dom,_,seed,m=b.split("_",3)
    INJ[(dom,m)]+= [json.loads(l) for l in open(f)]
def task_p1(doms,m,excl_infra=False):
    d=collections.defaultdict(list)
    for dom in doms:
        for r in E[(dom,m)]:
            if excl_infra and r["termination"]=="infrastructure_error": continue
            d[dom+":"+r["task_id"]].append(1.0 if r["reward"]>=1 else 0.0)
    return {k:sum(v)/len(v) for k,v in d.items()}
def boot(a,b,B=10000):
    ks=sorted(set(a)&set(b)); d=np.array([a[k]-b[k] for k in ks]); rng=np.random.default_rng(0)
    m=rng.choice(d,(B,len(d))).mean(1); return d.mean(),np.percentile(m,2.5),np.percentile(m,97.5)
def test(doms,x,y,excl=False):
    a,b=task_p1(doms,x,excl),task_p1(doms,y,excl); ks=sorted(set(a)&set(b))
    d=[a[k]-b[k] for k in ks]
    p=wilcoxon(d,zero_method="wilcox").pvalue if any(d) else 1.0
    mean,lo,hi=boot(a,b); return mean,lo,hi,p,sum(v>0 for v in d),sum(v<0 for v in d),len(ks)
def holm(ps):
    o=sorted(range(len(ps)),key=lambda i:ps[i]); adj=[0]*len(ps); run=0
    for r,i in enumerate(o): run=max(run,min(1,(len(ps)-r)*ps[i])); adj[i]=run
    return adj
print("=== PRIMARY (pooled 60 tasks, pass1 = mean over 2 seeds) ===")
for m in M:
    t=task_p1(DOMS,m); print(f"{m:8s} pass1 {100*np.mean(list(t.values())):.1f}%  n_tasks {len(t)}")
H=[("H1","casdrm2","b0"),("H2","casdrm2","casd"),("H3","casdrm2","casdrm")]
res=[test(DOMS,x,y) for _,x,y in H]; adj=holm([r[3] for r in res])
for (h,x,y),r,pa in zip(H,res,adj):
    print(f"{h} {x} vs {y}: diff {100*r[0]:+.1f}pp CI[{100*r[1]:+.1f},{100*r[2]:+.1f}] wilcoxon p={r[3]:.4f} holm p={pa:.4f} | tasks better {r[4]} worse {r[5]} of {r[6]}")
print("\n=== SENSITIVITY: excluding infrastructure errors ===")
res2=[test(DOMS,x,y,True) for _,x,y in H]; adj2=holm([r[3] for r in res2])
for (h,x,y),r,pa in zip(H,res2,adj2): print(f"{h}: diff {100*r[0]:+.1f}pp CI[{100*r[1]:+.1f},{100*r[2]:+.1f}] p={r[3]:.4f} holm={pa:.4f}")
print("\n=== OTHER PAIRS (exploratory, pooled) ===")
for x,y in [("casd","b0"),("casdrm","b0"),("casdrm","casd")]:
    r=test(DOMS,x,y); print(f"{x} vs {y}: diff {100*r[0]:+.1f}pp CI[{100*r[1]:+.1f},{100*r[2]:+.1f}] p={r[3]:.4f}")
print("\n=== PER DOMAIN (secondary) ===")
for dom in DOMS:
    print(dom, {m: round(100*np.mean(list(task_p1([dom],m).values())),1) for m in M})
    for h,x,y in H:
        r=test([dom],x,y); print(f"   {h} {x} vs {y}: {100*r[0]:+.1f}pp CI[{100*r[1]:+.1f},{100*r[2]:+.1f}] p={r[3]:.4f}")
    for s in ["s301","s302"]:
        print("   seed",s,{m: sum(r["reward"]>=1 for r in E[(dom,m)] if r["_seed"]==s) for m in M})
print("\n=== SECONDARY PROCESS METRICS ===")
for dom in DOMS:
    for m in M:
        R=E[(dom,m)]; R2=[x for x in R if len(C(x))<100]; calls=[s for x in R2 for s in C(x)]
        tsa=[f1([s["tool"] for s in C(x)],x["gold_tools"]) for x in R if x["gold_tools"] and x["steps"]]
        I=[i for i in INJ[(dom,m)] if "tokens" in i]; he=sum("hook_error" in i for i in INJ[(dom,m)])
        term=collections.Counter(x["termination"] for x in R)
        print(f"{dom:7s} {m:8s} n={len(R)} solved={sum(x['reward']>=1 for x in R)} ICR={100*sum(s['error'] for s in calls)/max(1,len(calls)):.1f} TSA={100*np.mean(tsa):.1f} calls/ep={len(calls)/max(1,len(R2)):.1f} runaway={len(R)-len(R2)} "
              f"tok={statistics.median([i['tokens'] for i in I]) if I else 0} ms={statistics.median([i['ms'] for i in I]) if I else 0:.0f} hookerr={he} timeout={term.get('timeout',0)} infra={term.get('infrastructure_error',0)} commit={sum(i.get('commit',0) for i in I)} loop={sum(i.get('loop_warn',0) for i in I)}")
