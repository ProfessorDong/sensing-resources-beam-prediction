"""Read-only independent audit of the current six-figure Paper B.

No imports from project code. Raw CSV/power data and saved rankings are inputs.
The additional static baseline is an exploratory audit, not a fresh confirmatory
experiment. Fits use optimization rows only, excluding validation and test.
Output is JSON on stdout. Research files are never modified.
"""
from pathlib import Path
from collections import Counter
import csv
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / 'experiments/results'
REPORTED = json.loads((RES / 'beam_power_quality.json').read_text())
KS = (1, 2, 3, 5, 8, 16)
CFG = {**{f'chirps={c}': (4, 256, c) for c in (8,16,32,64,125,250)},
       **{f'samples={s}': (4, s, 250) for s in (64,128)},
       **{f'rx={m}': (m, 256, 250) for m in (1,2)}}

def losses(P, S):
    # S is seeds x frames x k; P already has the required frame mask applied.
    got = np.take_along_axis(P[None], S.astype(int), axis=2).max(axis=2)
    return 10*np.log10(P.max(axis=1)[None]/got)

def aggregate(L):
    return {'mean_regret_db': float(np.median(L.mean(axis=1))),
            'exceed_3dB': float(np.median((L>3).mean(axis=1))),
            'inclusion': float(np.median((L<=1e-12).mean(axis=1)))}

def static_metric(P, beams, k):
    S = np.broadcast_to(np.array(beams[:k])[None,None], (1,len(P),k))
    return aggregate(losses(P,S))

def scenario(sc):
    base = ROOT / f'datasets/deepsense6g/Scenario{sc}'
    rows = list(csv.DictReader((base / f'scenario{sc}_dev.csv').open()))
    P = np.array([np.loadtxt(base / r['unit1_pwr_60ghz'].removeprefix('./')) for r in rows])
    y = np.array([int(float(r['unit1_beam']))-1 for r in rows])
    seq = np.array([int(r['seq_index']) for r in rows])
    raw_index = np.array([int(r['index']) for r in rows])
    def stamp(r):
        hms, us = r['time_stamp'].split('-')
        h,m,s = map(int,hms.split(':'))
        return (3600*h+60*m+s)*1000000+int(us)
    times = np.array([stamp(r) for r in rows])
    n = len(rows); ntrain = int(.7*n); nval = ntrain//10; nfit = ntrain-nval
    start = ntrain+int(.02*n)
    z = np.load(RES / f'beam_shortlists_sc{sc}.npz')
    assert np.array_equal(z['val_row_index'],np.arange(nfit,ntrain))
    assert np.array_equal(z['test_row_index'],np.arange(start,n))
    finite = np.isfinite(P).all(1)
    assert (P[finite]>0).all()
    assert np.array_equal(P[finite].argmax(1),y[finite])
    out = {'n_fit':nfit, 'n_val':nval, 'n_test':n-start,
           'n_finite_fit':int(finite[:nfit].sum()),
           'n_finite_val':int(finite[nfit:ntrain].sum()),
           'n_finite_test':int(finite[start:].sum()),
           'validation_sequences':dict(Counter(seq[nfit:ntrain].astype(str))),
           'fit_validation_common_sequence_ids':sorted(set(seq[:nfit].tolist()) & set(seq[nfit:ntrain].tolist()))}
    rec = REPORTED['scenarios'][str(sc)]['frontier']
    reported = {(r['setting'],r['k']):r for r in rec['points']}
    points=[]; seedmetrics={}; checks=0; worst=0.0
    for label,(m,ns,nc) in CFG.items():
        for k in KS:
            p = {'setting':label,'k':k,'cost_dedicated_us':nc*ns/100+5*k,'cost_reuse_us':5*k}
            for split,inds in [('val',np.arange(nfit,ntrain)),('test',np.arange(start,n))]:
                mask=finite[inds]; suff='val_shortlist' if split=='val' else 'shortlist'
                L=losses(P[inds][mask], z[f'spatial|{label}|{suff}'][:,mask,:k])
                p[split]=aggregate(L)
                seedmetrics[(label,k,split)]=L.mean(1)
                for field,value in p[split].items():
                    err=abs(value-reported[(label,k)][split][field]); worst=max(worst,err)
                    assert err<1e-11,(sc,label,k,split,field,err)
                    checks+=1
            points.append(p)
    default = next(p for p in points if p['setting']=='chirps=250' and p['k']==5)
    target=default['val']['mean_regret_db']
    def select(threshold,regime='dedicated',seed=None):
        def value(p):
            return p['val']['mean_regret_db'] if seed is None else seedmetrics[(p['setting'],p['k'],'val')][seed]
        feasible=[p for p in points if value(p)<=threshold+1e-12]
        if not feasible:return None,[]
        pick=min(feasible,key=lambda p:(p[f'cost_{regime}_us'],p['k']))
        ties=[p['setting'] for p in feasible if p[f'cost_{regime}_us']==pick[f'cost_{regime}_us'] and p['k']==pick['k']]
        return pick,ties
    pick,ties=select(target)
    assert (pick['setting'],pick['k'])==(rec['selection']['dedicated']['selected']['setting'],rec['selection']['dedicated']['selected']['k'])
    out['validation_and_test_rederivation']={'scalar_checks':checks,'max_abs_difference':worst,'target_db':target,'selected':pick}
    reuse,reuse_ties=select(target,'reuse')
    out['reuse_selection']={'pick':reuse,'equal_cost_equal_k_settings':reuse_ties}
    out['threshold_sensitivity']={str(f):select(target*f)[0] for f in (.8,.9,1,1.1,1.2)}
    seedchoices=Counter()
    for s in range(15):
        sp,_=select(seedmetrics[('chirps=250',5,'val')][s],seed=s)
        seedchoices[f"{sp['setting']},k={sp['k']}"]+=1
    out['per_seed_rule_choices_own_default_target']=dict(seedchoices)
    delta=seedmetrics[(pick['setting'],pick['k'],'test')]-seedmetrics[('chirps=250',5,'test')]
    out['selected_vs_default_per_seed_test_mean_regret_delta']={'min':float(delta.min()),'median':float(np.median(delta)),'max':float(delta.max()),'n_better':int((delta<0).sum())}
    # Fit a fixed set to the same loss now used to select the radar operating point.
    # At each step select the largest mean log-power after adding that beam.
    # This is deterministic greedy set selection; no claim of global optimality.
    logp=10*np.log10(P[:nfit][finite[:nfit]])
    current=np.full(len(logp),-np.inf); selected=[]
    for k in range(1,65):
        values=np.maximum(current[:,None],logp).mean(0)
        values[selected]=-np.inf
        beam=int(np.argmax(values)); selected.append(beam)
        current=np.maximum(current,logp[:,beam])
    Pv=P[nfit:ntrain][finite[nfit:ntrain]]; Pt=P[start:][finite[start:]]
    priorfit=[b for b,c in Counter(y[:nfit]).most_common()]
    priorfit+=sorted(set(range(64))-set(priorfit))
    priorpaper=[b for b,c in Counter(y[:ntrain]).most_common()]
    priorpaper+=sorted(set(range(64))-set(priorpaper))
    out['static_baselines']={}
    for name,beams in [('power_greedy_fit_only',selected),('frequency_fit_only',priorfit),('frequency_paper_train_region',priorpaper)]:
        curve=[{'k':k,'cost_us':5*k,'val':static_metric(Pv,beams,k),'test':static_metric(Pt,beams,k)} for k in KS]
        feasible=[p for p in curve if p['val']['mean_regret_db']<=target+1e-12]
        out['static_baselines'][name]={'beam_order_zero_based_first16':beams[:16],'curve':curve,'selected_on_same_validation_target_and_k_grid':min(feasible,key=lambda p:p['cost_us']) if feasible else None}
    # Verify the first 16 greedy choices by minimizing regret directly, with a
    # separate expression from the log-power maximization used above.
    fitP=P[:nfit][finite[:nfit]]
    all_regret=10*np.log10(fitP.max(1)[:,None]/fitP)
    residual=np.full(len(fitP),np.inf); verified=[]
    for k in range(16):
        means=np.array([np.minimum(residual,all_regret[:,b]).mean() for b in range(64)])
        means[verified]=np.inf
        b=int(np.argmin(means)); verified.append(b)
        residual=np.minimum(residual,all_regret[:,b])
    assert verified==selected[:16]
    out['static_greedy_direct_regret_crosscheck']=True
    # Conditional recording sensitivity at the paper's fixed selected radar point.
    # Selection is not repeated on resamples; these are not external-validity CIs.
    k=pick['k']; mask=finite[start:]
    Lrad=losses(Pt,z[f"spatial|{pick['setting']}|shortlist"][:,mask,:k])
    Lstatic=losses(Pt,np.broadcast_to(np.array(selected[:k])[None,None],(1,len(Pt),k)))
    sq=seq[start:][mask]; seqs=np.unique(sq)
    sizes=np.array([(sq==s).sum() for s in seqs])
    sums=np.stack([(Lstatic-Lrad)[:,sq==s].sum(1) for s in seqs],axis=1)
    rng=np.random.default_rng(20261009+sc)
    weights=rng.multinomial(len(seqs),np.full(len(seqs),1/len(seqs)),size=20000)
    boot=np.median((sums@weights.T)/(sizes@weights.T)[None],axis=0)
    out['static_vs_selected_radar_same_k_recording_sensitivity']={
        'method':'20,000 whole-recording paired bootstrap draws; frame weighted; fixed selected setting and fixed 15 seeds; exploratory conditional sensitivity',
        'n_seeds_static_mean_better':int((Lstatic.mean(1)<Lrad.mean(1)).sum()),
        'median_seed_mean_delta_db':float(np.median((Lstatic-Lrad).mean(1))),
        'bootstrap_95_delta_db':np.percentile(boot,[2.5,97.5]).tolist(),
        'by_sequence':[{'sequence':int(s),'n':int(sizes[j]),'median_seed_mean_delta_db':float(np.median(sums[:,j]/sizes[j]))} for j,s in enumerate(seqs)]}
    # Correct the manuscript's ostensibly matched persistence comparison.
    step=(np.diff(seq)==0)&(np.diff(raw_index)==1)&(np.diff(times)>0)&(np.diff(times)<=200000)
    eligible=np.r_[False,step][start:]; matched=eligible&finite[start:]
    yp=np.r_[y[0],y[:-1]][start:]; yt=y[start:]
    S=z['spatial|chirps=250|shortlist']
    out['persistence_masks']={'n_temporally_eligible':int(eligible.sum()),'n_finite_eligible':int(matched.sum()),
        'persistence_top1_all_eligible':float((yp[eligible]==yt[eligible]).mean()),
        'persistence_top1_finite_eligible':float((yp[matched]==yt[matched]).mean()),
        'radar_top1_finite_eligible':float(np.median((S[:,matched,0]==yt[None,matched]).mean(1)))}
    top5=np.sort(S[:,:,:5],axis=2)
    out['figure1_top5_index_geometry']={'fraction_contiguous_sets':float((np.diff(top5,axis=2)==1).all(2).mean()),
        'index_span_min_median_p95_max':np.percentile(top5[:,:,-1].astype(int)-top5[:,:,0].astype(int),[0,50,95,100]).tolist(),
        'note':'Descriptive index geometry only; no physical codebook angle data were used.'}
    # Figure 3's actual projection excludes aperture/bandwidth settings.
    shown=[p for p in points if p['setting'].startswith('chirps=')]
    def pareto(points,field='cost_dedicated_us'):
        return [p for p in points if not any(q[field]<=p[field] and q['test']['mean_regret_db']<=p['test']['mean_regret_db'] and (q[field]<p[field] or q['test']['mean_regret_db']<p['test']['mean_regret_db']) for q in points)]
    front=pareto(points)
    out['figure3_projection']={'n_scored_points':len(points),'n_displayed_points':len(shown),'radar_only_test_nondominated_points':front,'omitted_test_nondominated_points':[p for p in front if p not in shown]}
    return out

print(json.dumps({'method':'Independent read-only calculations; static baseline is exploratory and uses fit rows only.',
                  'scenarios':{str(sc):scenario(sc) for sc in (31,33)}},indent=2,
                 default=lambda x:x.item() if isinstance(x,np.generic) else x))
