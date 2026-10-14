"""Read-only independent empirical audit; no project-module imports.

All fitted policies below use optimization rows; hindsight local search is only
a mathematical counterexample to calling a test-fitted greedy set a bound.
The expanded-k calculation is an exploratory sensitivity, not fresh confirmation.
"""
from pathlib import Path
import csv, json
from collections import Counter
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / 'experiments/results'
Q = json.loads((RES / 'beam_power_quality.json').read_text())
J = json.loads((RES / 'static_shortlist.json').read_text())
KS = [1,2,3,5,8,16]

def greedy(D, steps=16):
    order=[]; residual=np.full(len(D),np.inf)
    for j in range(steps):
        losses=np.minimum(residual[:,None],D).mean(0)
        losses[order]=np.inf
        b=int(np.argmin(losses)); order.append(b)
        residual=np.minimum(residual,D[:,b])
    return order

def summary(D, shortlists):
    # For static indices or seeds x frames x k radar indices.
    if shortlists.ndim == 1:
        L=D[:,shortlists].min(1)[None]
    else:
        L=np.take_along_axis(D[None], shortlists, axis=2).min(2)
    return {"mean_regret_db":float(np.median(L.mean(1))),
            "exceed_3dB":float(np.median((L>3).mean(1))),
            "inclusion":float(np.median((L<=1e-12).mean(1)))}

out={}
for sc in (31,33):
    base=ROOT/f'datasets/deepsense6g/Scenario{sc}'
    rows=list(csv.DictReader((base/f'scenario{sc}_dev.csv').open()))
    P=np.array([np.loadtxt(base/r['unit1_pwr_60ghz'].removeprefix('./')) for r in rows])
    y=np.array([int(float(r['unit1_beam']))-1 for r in rows])
    z=np.load(RES/f'beam_shortlists_sc{sc}.npz')
    n=len(rows); ntr=int(z['n_train']); nval=ntr//10; nte=int(z['n_test'])
    ft=np.arange(n-nte,n); fv=np.arange(ntr-nval,ntr); ff=np.arange(ntr-nval)
    finite=np.isfinite(P).all(1)
    ff=ff[finite[ff]]; fv=fv[finite[fv]]; ft=ft[finite[ft]]
    assert (P[finite]>0).all()
    assert np.array_equal(P[finite].argmax(1),y[finite])
    D=10*np.log10(P.max(1)[:,None]/P)
    train_order=greedy(D[ff]); hindsight=greedy(D[ft])
    rep=J['scenarios'][str(sc)]; assert train_order==rep['order']; assert hindsight==rep['hindsight_order']
    record={'positive_finite_power':True, 'n_fit':len(ff),'n_val':len(fv),'n_test':len(ft),
            'test_tied_maxima':int(((P[ft]==P[ft].max(1)[:,None]).sum(1)>1).sum())}
    errors=[]; points=[]
    for p in Q['scenarios'][str(sc)]['frontier']['points']:
        if p['k']!=1: continue
        label=p['setting']
        t=z[f'spatial|{label}|shortlist'][:,finite[n-nte:],:].astype(int)
        v=z[f'spatial|{label}|val_shortlist'][:,finite[ntr-nval:ntr],:].astype(int)
        for k in range(1,17):
            test=summary(D[ft],t[:,:,:k]); val=summary(D[fv],v[:,:,:k])
            q={'name':f'radar, {label}', 'k':k, 'val':val,'test':test,
               'reuse':5*k,'dedicated':p['n_chirps']*p['n_samples']/100+5*k}
            points.append(q)
            if k in KS:
                r=next(r for r in Q['scenarios'][str(sc)]['frontier']['points'] if r['k']==k and r['setting']==label)
                errors += [abs(q[split][f]-r[split][f]) for split in ['test','val'] for f in test]
    prior=[b for b,_ in Counter(y[ff].tolist()).most_common()]
    prior += [b for b in range(64) if b not in prior]
    for name,order in [('power-aware static',train_order),('fixed training prior',prior)]:
        for k in range(1,17):
            p={'name':name,'k':k,'val':summary(D[fv],np.array(order[:k])),
               'test':summary(D[ft],np.array(order[:k])),'reuse':5*k,'dedicated':5*k}
            points.append(p)
            if k in KS:
                field='' if name=='power-aware static' else 'prior_'
                errors += [abs(p[split][f]-rep[field+split][str(k)][f]) for split in ['test','val'] for f in p[split]]
    target=next(p['val']['mean_regret_db'] for p in points if p['name']=='radar, chirps=250' and p['k']==5)
    record['validation_target']=target
    record['scalar_checks']={'count':len(errors),'max_error':max(errors)}
    record['selection']={}
    for grid in ['published','every_k_1_to_16']:
        feasible=[p for p in points if p['val']['mean_regret_db']<=target+1e-12 and (grid!='published' or p['k'] in KS)]
        record['selection'][grid]={regime:min(feasible,key=lambda p:(p[regime],p['val']['mean_regret_db'],p['k'],p['name'])) for regime in ['dedicated','reuse']}
    # Show a valid same-k set beating the test-fitted greedy objective by one-swap descent.
    examples=[]
    for k in KS:
        S=list(hindsight[:k]); old=float(D[ft][:,S].min(1).mean())
        while True:
            cur=float(D[ft][:,S].min(1).mean()); best=cur; nxt=None
            for j in range(k):
                remaining=S[:j]+S[j+1:]
                residual=D[ft][:,remaining].min(1) if remaining else np.full(len(ft),np.inf)
                vals=np.minimum(residual[:,None],D[ft]).mean(0)
                vals[S]=np.inf; b=int(vals.argmin())
                if vals[b]<best-1e-12:
                    best=float(vals[b]); nxt=(j,b)
            if nxt is None: break
            S[nxt[0]]=nxt[1]
        examples.append({'k':k,'greedy':hindsight[:k],'greedy_mean_db':old,
                         'better_set':S,'better_mean_db':cur,'improvement_db':old-cur,
                         'train_set_mean_db':summary(D[ft],np.array(train_order[:k]))['mean_regret_db']})
    record['hindsight_not_a_bound']=examples
    record['static_no_catastrophic_claim']={str(k):summary(D[ft],np.array(train_order[:k])) for k in KS}
    record['best_radar_vs_static_tail_from5']={str(k):{'static':rep['test'][str(k)]['exceed_3dB'],
          'minimum_radar_median':min(p['test']['exceed_3dB'] for p in points if p['name'].startswith('radar') and p['k']==k)} for k in [5,8,16]}
    out[str(sc)]=record
print(json.dumps(out,indent=2))
