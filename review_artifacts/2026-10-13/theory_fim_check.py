"""Independent numerical check of Paper B's full five-parameter FIM.

No project modules or result JSONs are imported. This checks all frequency
nuisances, which the repository's three-parameter check does not include.
"""
import json
from pathlib import Path
import numpy as np

out = []
for m, ns, nc in [(4,16,8),(4,32,16),(2,16,8),(8,16,8),(4,8,8),(12,16,8),(3,24,12),(6,16,16)]:
    n, k, ell = np.meshgrid(np.arange(m), np.arange(ns), np.arange(nc), indexing='ij')
    phase = np.exp(2j*np.pi*(0.11*n + 0.23*k + 0.07*ell))
    alpha = 0.8 + 0.6j
    mu = alpha*phase
    deriv = np.stack([(2j*np.pi*v*mu).ravel() for v in (n,k,ell)] + [phase.ravel(), (1j*phase).ravel()])
    information = 2*(deriv.conj() @ deriv.T).real
    schur = information[:3,:3] - information[:3,3:] @ np.linalg.solve(information[3:,3:],information[3:,:3])
    numeric = np.sqrt(np.linalg.inv(information)[0,0])
    closed = np.sqrt(3/(2*np.pi**2*ns*nc*m*(m*m-1)))
    cross = schur - np.diag(np.diag(schur))
    row = dict(m=m,ns=ns,nc=nc,full_fim_sigma_nu=float(numeric),closed_sigma_nu=float(closed),relative_error=float(abs(numeric/closed-1)),normalized_schur_cross_max=float(np.abs(cross).max()/np.diag(schur).max()))
    assert row['relative_error'] < 1e-12
    assert row['normalized_schur_cross_max'] < 1e-12
    out.append(row)
target = Path(__file__).with_name('theory_fim_check.json')
target.write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({'cases':len(out),'max_relative_error':max(v['relative_error'] for v in out),'max_normalized_cross_term':max(v['normalized_schur_cross_max'] for v in out)},indent=2))
