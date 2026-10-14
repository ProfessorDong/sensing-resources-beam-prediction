"""Independent current-grid and matched-recording audit; no project imports.

Read-only research inputs. Outputs stdout JSON; all new evidence belongs in
review_artifacts/2026-10-14. No model is retrained or selected using test values.
"""
from pathlib import Path
from collections import Counter
import csv, json, hashlib
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
RES=ROOT/'experiments/results'
J=json.loads((RES/'static_shortlist.json').read_text())
Q=json.loads((RES/'beam_power_quality.json').read_text())
old=json.loads((ROOT/'review_artifacts/2026-10-13/source_hashes.json').read_text())
out={'unchanged_input_hashes':{}, 'scenarios':{}}
for name in ['beam_shortlists_sc31.npz','beam_shortlists_sc33.npz','beam_power_quality.json']:
    rel=f'experiments/results/{name}'; h=hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()
    out['unchanged_input_hashes'][name]={'sha256':h,'matches_2026_10_13':h==old[rel]}

for sc in (31,33):
    base=ROOT/f'datasets/deepsense6g/Scenario{sc}'
    rows=list(csv.DictReader((base/f'scenario{sc}_dev.csv').open()))
    P=np.array([np.loadtxt(base/r['unit1_pwr_60ghz'].removeprefix('./')) for r in rows])
    seq=np.array([int(r['seq_index']) for r in rows])
    y=np.array([int(float(r['unit1_beam']))-1 for r in rows])
    z=np.load(RES/f'beam_shortlists_sc{sc}.npz')
    n=len(y); ntr=int(z['n_train']); nval=ntr//10; nte=int(z['n_test'])
    fit=np.arange(ntr-nval); val=np.arange(ntr-nval,ntr); test=np.arange(n-nte,n)
    finite=np.isfinite(P).all(1)
    ff=fit[finite[fit]]; fv=val[finite[val]]; ft=test[finite[test]]
    assert (P[finite]>0).all() and np.array_equal(P[finite].argmax(1),y[finite])
    D=10*np.log10(P.max(1)[:,None]/P)
    order=[]; residual=np.full(len(ff),np.inf)
    for k in range(16):
        obj=np.minimum(residual[:,None],D[ff]).mean(0);obj[order]=np.inf
        b=int(obj.argmin());order.append(b);residual=np.minimum(residual,D[ff,b])
    prior=[b for b,_ in Counter(y[ff].tolist()).most_common()]
    assert order==J['scenarios'][str(sc)]['order']
    def static_loss(ids, beams,k):return D[ids][:,beams[:k]].min(1)[None]
    def radar_loss(ids,label,k):
        key='val_shortlist' if ids is fv else 'shortlist'
        mask=finite[val] if ids is fv else finite[test]
        S=z[f'spatial|{label}|{key}'][:,mask,:k].astype(int)
        return np.take_along_axis(D[ids][None],S,axis=2).min(2)
    def stats(L):return {'mean':float(np.median(L.mean(1))),
                         'tail':float(np.median((L>3).mean(1))),
                         'inclusion':float(np.median((L<=1e-12).mean(1)))}
    target=stats(radar_loss(fv,'chirps=250',5))['mean']
    points=[]; errors=[]
    for name,beams in [('power-aware static',order),('fixed training prior',prior)]:
        for k in range(1,17):
            point={'name':name,'k':k,'dedicated':5*k,'reuse':5*k,
                   'val':stats(static_loss(fv,beams,k)), 'test':stats(static_loss(ft,beams,k))}
            points.append(point)
            field='' if name=='power-aware static' else 'prior_'
            for spl in ['val','test']:
                rep=J['scenarios'][str(sc)][field+spl][str(k)]
                errors.extend(abs(point[spl][a]-rep[b]) for a,b in [('mean','mean_regret_db'),('tail','exceed_3dB'),('inclusion','inclusion')])
    labels={p['setting']:(p['n_chirps'],p['n_samples']) for p in Q['scenarios'][str(sc)]['frontier']['points']}
    radar_errors=[]
    for label,(nc,ns) in labels.items():
        for k in range(1,17):
            p={'name':'radar, '+label,'k':k,'dedicated':nc*ns/100+5*k,'reuse':5*k,
               'val':stats(radar_loss(fv,label,k)),'test':stats(radar_loss(ft,label,k))}
            points.append(p)
            if k in [1,2,3,5,8,16]:
                reported=next(q for q in Q['scenarios'][str(sc)]['frontier']['points'] if q['setting']==label and q['k']==k)
                for spl in ['val','test']:
                    radar_errors.extend(abs(p[spl][a]-reported[spl][b]) for a,b in [('mean','mean_regret_db'),('tail','exceed_3dB'),('inclusion','inclusion')])
    choices={}; cheapest_radar={}
    for regime in ['dedicated','reuse']:
        eligible=[p for p in points if p['val']['mean']<=target+1e-12]
        key=lambda p:(p[regime],p['val']['mean'],p['k'],p['name'])
        best=min(eligible,key=key);choices[regime]=best
        cheapest_radar[regime]=min([p for p in eligible if p['name'].startswith('radar')],key=key)
        rep=J['scenarios'][str(sc)]['selection_'+regime]['selected']
        assert (best['name'],best['k'],best[regime])==(rep['policy'],rep['k'],rep['cost_us'])
    # Heterogeneity/paired interval at the CURRENT cheapest eligible radar setting/k.
    r=cheapest_radar['dedicated'];k=r['k'];label=r['name'].removeprefix('radar, ')
    Ls=static_loss(ft,order,k);Lr=radar_loss(ft,label,k)
    gs=np.unique(seq[ft]);idx=[np.flatnonzero(seq[ft]==g) for g in gs]
    ngrp=np.array([len(v) for v in idx]);sstat=np.array([Ls[:,v].sum() for v in idx])
    srad=np.stack([Lr[:,v].sum(1) for v in idx],axis=1)
    rng=np.random.default_rng(20261011)
    pick=rng.integers(0,len(gs),(20000,len(gs)))
    w=np.array([np.bincount(p,minlength=len(gs)) for p in pick])
    bs=(sstat@w.T)/(ngrp@w.T)-np.median((srad@w.T)/(ngrp@w.T)[None],axis=0)
    paired={'k':k,'radar_setting':label,'static_mean':float(Ls.mean()),
            'radar_seedmedian_mean':float(np.median(Lr.mean(1))),
            'difference':float(Ls.mean()-np.median(Lr.mean(1))),
            'whole_recording_ci95':np.percentile(bs,[2.5,97.5]).tolist(),
            'static_better_seeds':int((Ls.mean()<Lr.mean(1)).sum()),
            'static_better_recordings':int(sum(Ls[:,v].mean()<np.median(Lr[:,v].mean(1)) for v in idx)),
            'recordings':len(gs),'stored_paired_analysis':J['scenarios'][str(sc)]['paired_vs_selected_radar']}
    full=[p for p in points if p['name']=='radar, chirps=250']
    comparisons=[]
    for p in full:
        s=next(q for q in points if q['name']=='power-aware static' and q['k']==p['k'])
        comparisons.append({'k':p['k'],'radar_mean':p['test']['mean'],'static_mean':s['test']['mean'],
                            'radar_tail':p['test']['tail'],'static_tail':s['test']['tail'],
                            'min_radar_tail':min(q['test']['tail'] for q in points if q['name'].startswith('radar') and q['k']==p['k'])})
    record={'static_metric_checks':{'count':len(errors),'max_error':max(errors)},
            'radar_metric_checks':{'count':len(radar_errors),'max_error':max(radar_errors)},
            'target':target,'selections':choices,'cheapest_radar':cheapest_radar,
            'current_selected_radar_paired':paired,'all_k_comparisons':comparisons,
            'validation_recordings':dict(Counter(seq[fv].astype(str))),
            'fit_validation_shared_recordings':sorted(set(seq[ff].tolist())&set(seq[fv].tolist())),
            'static_all_label':{str(k):float(np.isin(y[test],order[:k]).mean()) for k in [1,5]},
            'all_label_bound':float(np.bincount(y[test]).max()/len(test))}
    out['scenarios'][str(sc)]=record
print(json.dumps(out,indent=2))
