"""Read-only audit from CSV power vectors and saved rankings; no project imports.

Run with the w26 Python environment. Results go to stdout, never research files.
Bootstrap summaries are conditional sensitivity analyses, not route-generalization CIs.
"""
from pathlib import Path
from collections import Counter
import csv
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / 'experiments/results'
rng = np.random.default_rng(20261009)


def metric(P, S, y):
    # S: seeds x frames x k. Power is evaluated only on externally supplied masks.
    got = np.take_along_axis(P[None, :, :], S.astype(int), axis=2).max(axis=2)
    loss = 10*np.log10(P.max(axis=1)[None, :]/got)
    inc = np.any(S == y[None, :, None], axis=2)
    return loss, inc


def summary(loss, inc):
    return dict(inclusion=float(np.median(inc.mean(axis=1))),
                mean_loss_db=float(np.median(loss.mean(axis=1))),
                over_3db=float(np.median((loss > 3).mean(axis=1))))


def scenario(sc):
    base = ROOT / f'datasets/deepsense6g/Scenario{sc}'
    rows = list(csv.DictReader((base/f'scenario{sc}_dev.csv').open()))
    y = np.array([int(float(r['unit1_beam']))-1 for r in rows])
    seq = np.array([int(r['seq_index']) for r in rows])
    ix = np.array([int(r['index']) for r in rows])
    def time(r):
        hms, us = r['time_stamp'].split('-')
        h,m,s = map(int,hms.split(':'))
        return (h*3600+m*60+s)*1000000+int(us)
    times = np.array([time(r) for r in rows])
    P = np.array([np.loadtxt(base/r['unit1_pwr_60ghz'].removeprefix('./')) for r in rows])
    good = np.isfinite(P).all(axis=1)
    assert (P[good] > 0).all()
    assert np.array_equal(P[good].argmax(axis=1), y[good])
    z = np.load(RES/f'beam_shortlists_sc{sc}.npz')
    n = len(rows); nt = int(.7*n); start = nt + int(.02*n)
    assert int(z['n_train']) == nt
    assert np.array_equal(z['test_row_index'],np.arange(start,n))
    assert np.array_equal(z['beams_all'],y)
    prior = np.array([v for v,_ in Counter(y[:nt]).most_common(16)])
    yp, pp, gp, sq = y[start:], P[start:], good[start:], seq[start:]
    S = z['spatial|chirps=250|shortlist']
    out = dict(n_test=len(yp), n_finite=int(gp.sum()),
               nonfinite_all=int((~good).sum()), nonfinite_test=int((~gp).sum()),
               positive_finite_powers=True, minimum_finite_power=float(P[good].min()),
               argmax_label_agreement_all_finite=True,
               tied_power_maxima_all_finite=int(((P[good] == P[good].max(1)[:,None]).sum(1)>1).sum()),
               finite_test_missing_by_sequence={str(q):int(((sq==q)&~gp).sum()) for q in np.unique(sq)})
    priorS = np.broadcast_to(prior[:5], (1,len(yp),5))
    lp, ip = metric(pp[gp],priorS[:,gp],yp[gp])
    l, inc = metric(pp[gp], S[:,gp,:5],yp[gp])
    assert np.array_equal(inc,l<=1e-12)
    out['full_frame'] = summary(l,inc)
    out['prior'] = summary(lp,ip)
    out['cost_comparisons'] = {}
    for maskname,mask in [('all_test',np.ones(len(yp),bool)),('finite_power',gp)]:
        pr = np.isin(yp[mask],prior[:5]).mean()
        costs = {}
        for key in ['spatial|chirps=8','wide|chirps=8','spatial|chirps=250']:
            ss = z[key+'|shortlist'][:,mask,:5]
            p = (ss == yp[mask][None,:,None]).any(2).mean(1)
            nc = int(key.split('=')[1])
            delta = nc*256/100-295*(p-pr)
            boots = delta[rng.integers(0,len(delta),(20000,len(delta)))].mean(1)
            costs[key] = dict(n=int(mask.sum()), prior_top5=float(pr),
                top5_mean=float(p.mean()), top5_median=float(np.median(p)),
                prior_cost_us=float(320-295*pr), dedicated_mean_us=float(320-295*p.mean()+nc*256/100),
                delta_mean_us=float(delta.mean()), delta_median_us=float(np.median(delta)),
                seeds_beating_prior=int((delta<0).sum()),
                seed_only_mean_delta_percentile_bootstrap_95_us=np.percentile(boots,[2.5,97.5]).tolist(),
                break_even_Msps_mean=float(nc*256/(295*(p.mean()-pr))),
                break_even_Msps_median=float(nc*256/(295*(np.median(p)-pr))),
                break_even_Tsec_us_at_100Msps=float(nc*256/100/(59*(p.mean()-pr))))
        out['cost_comparisons'][maskname] = costs
    # Pair recording resampling between the radar and prior. Seeds remain fixed.
    seqs = np.unique(sq[gp]); nq = len(seqs)
    sizes = np.array([((sq[gp])==q).sum() for q in seqs])
    meanloss = np.stack([(l-lp)[:,sq[gp]==q].sum(1) for q in seqs],axis=1)
    tail = np.stack([((l>3)-(lp>3).astype(int))[:,sq[gp]==q].sum(1) for q in seqs],axis=1)
    weights = rng.multinomial(nq,np.full(nq,1/nq),size=20000)
    def estimates(sums):
        vals = (sums@weights.T)/(sizes@weights.T)[None,:]
        return np.percentile(np.median(vals,axis=0),[2.5,97.5]).tolist()
    out['sequence_sensitivity'] = dict(
        method='20,000 paired recording bootstrap samples, whole sequences, frame-weighted; median across fixed 15 training seeds; conditional exploratory sensitivity only',
        mean_loss_delta_db_95=estimates(meanloss),
        tail_probability_delta_95=estimates(tail),
        equal_sequence_weight_tail_delta=float(np.median((tail/sizes).mean(1))),
        equal_sequence_weight_loss_delta=float(np.median((meanloss/sizes).mean(1))),
        by_sequence=[])
    for qi,q in enumerate(seqs):
        m = sq[gp] == q
        leave = np.arange(nq)!=qi
        out['sequence_sensitivity']['by_sequence'].append(dict(sequence=int(q),n=int(m.sum()),
            radar=summary(l[:,m],inc[:,m]), prior=summary(lp[:,m],ip[:,m]),
            mean_loss_delta_db=float(np.median(meanloss[:,qi]/sizes[qi])),
            tail_probability_delta=float(np.median(tail[:,qi]/sizes[qi])),
            leave_this_sequence_out_tail_delta=float(np.median(tail[:,leave].sum(1)/sizes[leave].sum()))))
    # Paired delay: use identical future frames for stale and fresh rankings.
    step = (np.diff(seq)==0)&(np.diff(ix)==1)&(np.diff(times)>0)&(np.diff(times)<=200000)
    out['delay_matched'] = {}
    for lag in [1,2,3,5]:
        a = np.arange(start,n-lag)
        ok = good[a] & good[a+lag]
        for j in range(lag):
            ok &= step[a+j]
        a = a[ok]; b = a+lag
        stale, si = metric(P[b],S[:,a-start,:5],y[b])
        fresh, fi = metric(P[b],S[:,b-start,:5],y[b])
        src, src_i = metric(P[a],S[:,a-start,:5],y[a])
        ages = (times[b]-times[a])/1000
        out['delay_matched'][str(lag)] = dict(n=len(a),
            age_ms_min_median_p95_max=np.percentile(ages,[0,50,95,100]).tolist(),
            stale_future=summary(stale,si), fresh_same_future=summary(fresh,fi), fresh_source=summary(src,src_i),
            median_seed_loss_increase_db=float(np.median((stale-fresh).mean(1))),
            median_seed_inclusion_change=float(np.median((si.astype(int)-fi).mean(1))))
    # All saved settings on both metrics make a fixed-budget frontier possible without retraining.
    out['fixed_budget_k5_resource_sweep'] = {}
    for key in z.files:
        if key.endswith('|shortlist') and key.startswith('spatial|'):
            ll,ii=metric(pp[gp],z[key][:,gp,:5],yp[gp])
            out['fixed_budget_k5_resource_sweep'][key] = summary(ll,ii)
    out['fixed_budget_illustrations'] = {}
    for nc,k in [(8,16),(250,5)]:
        ll,ii = metric(pp[gp],z[f'spatial|chirps={nc}|shortlist'][:,gp,:k],yp[gp])
        out['fixed_budget_illustrations'][f'chirps={nc},k={k}'] = dict(
            **summary(ll,ii), reuse_probe_us=k*5,
            dedicated_active_window_plus_probe_us_at_100Msps=nc*256/100+k*5)
    return out


files = ['paper_b/PaperB.tex','paper_b/PaperB.pdf','paper_b/tables.tex','paper_b/cover_letter.txt',
         'experiments/audit_beam_power_regret.py','experiments/exp_beam_shortlists.py',
         'experiments/results/beam_power_quality.json','experiments/results/beam_shortlists_sc31.npz',
         'experiments/results/beam_shortlists_sc33.npz']
hashes = {s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in files}
out = dict(source_sha256=hashes, scenarios={str(sc):scenario(sc) for sc in [31,33]})
b = json.loads((RES/'sc31_bearing_mediation_max.json').read_text())['settings']
out['full_aperture_resource_order'] = [dict(setting=k,R=v['resource_product_R'],RMSE=v['rmse_deg_median'])
    for k,v in sorted(b.items(),key=lambda x:x[1]['resource_product_R']) if v['n_rx']==4]
assert hashes == {s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in files}
out['source_files_unchanged_during_audit'] = True
print(json.dumps(out,indent=2))
