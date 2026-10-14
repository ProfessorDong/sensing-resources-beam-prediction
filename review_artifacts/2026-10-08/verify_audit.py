"""Independent, read-only checks for the October 8 Paper B audit.

Run with the w26 Python environment. Prints JSON and never edits research results.
"""
import collections
import csv
import hashlib
import itertools
import json
import re
from pathlib import Path

import numpy as np
from scipy import optimize, stats

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / 'experiments/results'
CH = [8, 16, 32, 64, 125, 250]


def read(name):
    return json.loads((RES / name).read_text())


def exact_spearman(y):
    r = stats.spearmanr(CH, y)
    null = [stats.spearmanr(CH, p).statistic for p in itertools.permutations(y)]
    return {'rho': float(r.statistic), 'asymptotic_p': float(r.pvalue),
            'exact_two_sided_p': float(np.mean(np.abs(null) >= abs(r.statistic)-1e-12))}


def scenario(sc):
    rows = list(csv.DictReader((ROOT / f'datasets/deepsense6g/Scenario{sc}/scenario{sc}_dev.csv').open()))
    n = len(rows); nt = int(.7*n); start = nt+int(.02*n)
    y = np.array([int(r['unit1_beam'])-1 for r in rows])
    seq = np.array([int(r['seq_index']) for r in rows])
    idx = np.array([int(r['index']) for r in rows])
    times = []
    for row in rows:
        hms, us = row['time_stamp'].split('-')
        h, m, s = map(int, hms.split(':'))
        times.append((3600*h+60*m+s)+int(us)/1e6)
    dt = np.diff(times)
    same = (seq[1:] == seq[:-1]) & (idx[1:] == idx[:-1]+1) & (dt > 0)
    period = float(np.median(dt[same]))
    elig = same & (dt <= .2)
    elig_claimed = same & (dt <= 2*period)
    ct = np.bincount(y[:nt], minlength=64); ce = np.bincount(y[start:], minlength=64)
    p5 = float(np.isin(y[start:], np.argsort(-ct)[:5]).mean())
    A = read(f'sc{sc}_resource_ablation_G_max.json')
    B = read(f'sc{sc}_bearing_mediation_max.json')
    bc = [A['settings'][f'chirps={c}']['n_collapsed'] for c in CH]
    rc = [B['settings'][f'chirps={c}']['n_collapsed'] for c in CH]
    brg = np.load(RES / f'sc{sc}_bearing_labels.npz')['bearing_deg'].astype(float)
    tb = np.radians(read('beam_coherence.json')['tolerated_pointing_error_deg'])
    rate = np.abs(np.diff(np.radians(brg)))/np.where(dt > 0, dt, np.nan)
    pos = elig & (rate > 0)
    coh = tb/rate[pos]
    ablation = []
    for c in CH:
        v = A['settings'][f'chirps={c}']; pk = v['top5_mean']
        saving = 295*(pk-p5)
        ablation.append({'chirps':c, 'top1_median':v['top1_median'],
                         'top5_mean':pk, 'reuse_us':320-295*pk,
                         'saving_vs_prior_us':saving,
                         'break_even_adc_Msps':256*c/saving if saving>0 else None,
                         'dedicated_at_25Msps_us':320-295*pk+c*10.24})
    def overlap(a):
        return len(set(a[:nt]) & set(a[start:]))
    return {'n':n, 'test_n':n-start, 'fit_n':nt-nt//10, 'validation_n':nt//10,
            'train_test_sequence_overlap':overlap(seq),
            'train_test_file_overlap':overlap([r['unit1_radar'] for r in rows]),
            'test_sequences':sorted(map(int,set(seq[start:]))),
            'prior_top5':p5, 'oracle_top1':float(ce.max()/ce.sum()),
            'beam_degeneracy':{'counts':bc, **exact_spearman(bc)},
            'bearing_degeneracy':{'counts':rc, **exact_spearman(rc)},
            'chirp_top1_trend':exact_spearman([a['top1_median'] for a in ablation]),
            'persistence_n_ineligible_200ms':int((~elig[start-1:]).sum()),
            'persistence_n_ineligible_twice_median':int((~elig_claimed[start-1:]).sum()),
            'persistence_rule_disagreements_test':int((elig[start-1:]!=elig_claimed[start-1:]).sum()),
            'median_interval_ms':1000*period,
            'persistence_top1_eligible':float((y[1:]==y[:-1])[start-1:][elig[start-1:]].mean()),
            'coherence_median_ms_reproduced':float(1000*np.median(coh)),
            'coherence_stale_vs_median_period_fraction':float(np.mean(coh<period)),
            'coherence_stale_vs_actual_pair_dt_fraction':float(np.mean(coh<dt[pos])),
            'train_mean_bearing_RMSE':float(np.sqrt(np.mean((brg[start:]-brg[:nt].mean())**2))),
            'test_optimal_constant_bearing_RMSE':float(brg[start:].std()),
            'airtime_algebra_not_hardware_measurement':ablation,
            'top5_available_fields':sorted(k for k in A['settings']['chirps=250'] if 'top5' in k)}


def fim():
    out=[]
    for m, ns, nc in [(4,16,8),(2,32,16),(8,16,8)]:
        n,k,l=np.meshgrid(np.arange(m),np.arange(ns),np.arange(nc),indexing='ij')
        z=np.exp(2j*np.pi*(.11*n+.23*k+.07*l))
        D=[(2j*np.pi*i*z).ravel() for i in (n,k,l)]+[z.ravel(),(1j*z).ravel()]
        J=np.array([[2*np.vdot(a,b).real for b in D] for a in D])
        sch=J[:3,:3]-J[:3,3:]@np.linalg.inv(J[3:,3:])@J[3:,:3]
        target=8*np.pi**2*ns*nc*m*(m*m-1)/12
        out.append({'shape':[m,ns,nc], 'formula_ratio':float(sch[0,0]/target),
                    'relative_off_diagonal':float(np.max(np.abs(sch-np.diag(np.diag(sch))))/target)})
    return out


def resource_fit():
    B=read('sc31_bearing_mediation_max.json')['settings']
    A=read('sc31_resource_ablation_G_max.json')['settings']
    valid=[(k,v) for k,v in B.items() if v['resource_product_R']>0]
    x=np.array([np.log10(v['resource_product_R']) for k,v in valid])
    y=np.array([v['rmse_deg_median'] for k,v in valid])
    z=np.array([A[k]['top1_median'] for k,v in valid])
    ry=y-np.polyval(np.polyfit(x,y,1),x)
    rz=z-np.polyval(np.polyfit(x,z,1),x)
    r=float(np.corrcoef(ry,rz)[0,1]); df=len(valid)-3
    beta_p=float(2*stats.t.sf(abs(r)*np.sqrt(df/(1-r*r)),df))
    v=[v for v in B.values() if v['n_rx']==4]
    R=np.array([a['resource_product_R'] for a in v],float); R=R/R.max()
    msq=np.array([a['rmse_deg_median'] for a in v])**2
    f=lambda R,a,b,c:a*R**(-b)+c
    fit=optimize.curve_fit(f,R,msq,p0=[msq.max(),1,msq.min()/2],
                           bounds=([0,.05,0],[np.inf,6,msq.max()]),maxfev=200000)[0]
    return {'alpha_pearson':list(map(float,stats.pearsonr(x,y))),
            'beta_partial_r':r,'beta_df':df,'beta_p':beta_p,
            'fit_b':float(fit[1]), 'fitted_floor_deg':float(np.sqrt(fit[2])),
            'settings_by_R':[{'setting':k,'R':v['resource_product_R'],
                             'bearing_RMSE':v['rmse_deg_median'],
                             'top1':A[k]['top1_median']} for k,v in sorted(B.items(),key=lambda kv:kv[1]['resource_product_R'])]}


def main():
    src=(ROOT/'paper_b/PaperB.tex').read_text()
    ab=src.split(r'\begin{abstract}')[1].split(r'\end{abstract}')[0]
    out={'abstract_whitespace_word_count':len(ab.split()),
         'hashes':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in
                   ['paper_b/PaperB.tex','paper_b/PaperB.pdf','paper_b/numbers.tex','paper_b/numbers_audit.tex','paper_b/tables.tex']},
         'scenarios':{sc:scenario(sc) for sc in ['31','33']},
         'five_parameter_FIM':fim(), 'resource_fit':resource_fit(),
         'fallback_p0_us':{'retained':25+59*5,'restart':25+64*5},
         'reference_ratios':{'Sc31_raw_accuracy_ratio':.23574337363243103/.5,
                             'Sc33_raw_accuracy_ratio':.1827957034111023/.4543010752688172}}
    print(json.dumps(out,indent=2))


if __name__=='__main__':
    main()
