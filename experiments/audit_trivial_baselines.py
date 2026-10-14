"""Honest trivial baselines: every hyperparameter selected on TRAIN, reported on TEST."""
import os, sys, json, glob
import numpy as np
import os as _os
sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from config import RESULTS_DIR
DS=_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "datasets")

def iou(p,g):
    i=np.logical_and(p,g).sum(); u=np.logical_or(p,g).sum(); return float(i/u) if u else 0.0

def metrics(p,g,nb=15):
    pb,gb=p>0.5,g>0.5; tp=np.logical_and(pb,gb).sum()
    pc=np.clip(p,1e-7,1-1e-7); pf,gf=p.ravel(),g.ravel(); e=np.linspace(0,1,nb+1); ece=0.
    for k in range(nb):
        m=(pf>=e[k])&(pf<e[k+1])
        if m.sum(): ece+=m.sum()/len(pf)*abs(pf[m].mean()-gf[m].mean())
    return {"IoU":iou(pb,gb),
            "F1":float(2*tp/(pb.sum()+gb.sum())) if (pb.sum()+gb.sum()) else 0.,
            "NLL":float(-(g*np.log(pc)+(1-g)*np.log(1-pc)).mean()),
            "Brier":float(((p-g)**2).mean()),"ECE":float(ece)}

def run(name, tr_s,tr_m, te_s,te_m, pub):
    otr,ote=tr_s[:,0],te_s[:,0]
    # constant: train mean, evaluated on test
    const=np.repeat(otr.mean(0,keepdims=True),len(ote),0)
    mc=metrics(const,ote)
    # threshold: tau AND channel selected on TRAIN, applied to TEST
    best=(-1,None,None)
    for c in range(tr_m.shape[1]):
        for t in np.arange(0.02,0.99,0.01):
            v=iou(tr_m[:,c]>t, otr>0.5)
            if v>best[0]: best=(v,c,round(float(t),2))
    _,c,t=best
    thr_test=iou(te_m[:,c]>t, ote>0.5)
    thr_train=best[0]
    # soft threshold as a probability map (for calibration metrics): identity on Y
    mt=metrics(np.clip(te_m[:,c],0,1),ote)
    print(f"\n{name}")
    print(f"  constant (train mean)         IoU={mc['IoU']:.4f}  ECE={mc['ECE']:.4f}  NLL={mc['NLL']:.4f}  Brier={mc['Brier']:.6f}")
    print(f"  threshold  ch{c} tau={t:.2f} (train-selected: train IoU={thr_train:.4f})")
    print(f"                                IoU={thr_test:.4f}")
    print(f"  raw Y as probability map      IoU={mt['IoU']:.4f}  ECE={mt['ECE']:.4f}  NLL={mt['NLL']:.4f}  Brier={mt['Brier']:.6f}")
    print(f"  published S2S-Diff            IoU={pub['IoU']:.4f}  ECE={pub['ECE']:.4f}  NLL={pub['NLL']:.4f}  Brier={pub['Brier']:.6f}")
    beat=[]
    if mc['IoU']>=pub['IoU']: beat.append("constant")
    if thr_test>=pub['IoU']: beat.append("threshold")
    print(f"  --> beats S2S-Diff on IoU: {beat if beat else 'neither'}")
    return {"constant":mc,"threshold":{"channel":c,"tau":t,"train_IoU":thr_train,"test_IoU":thr_test},
            "rawY_as_prob":mt,"published":pub,"trivial_beats_S2S_IoU":beat}

PUB={"DeepSense_ISAC":{"IoU":0.8533,"ECE":0.0061,"NLL":0.0766,"Brier":0.017912},
     "DeepSense_V2I":{"IoU":0.1849,"ECE":0.2636,"NLL":1.4901,"Brier":0.266422},
     "RADIal":{"IoU":0.3866,"ECE":0.0525,"NLL":0.0736,"Brier":0.011883}}
from train_all_real import preprocess_isac_scenario, preprocess_radial
from preprocess_deepsense import preprocess_scenario as pv2i
out={}
ps,pm=[],[]
for sc in (42,43):
    a,b=preprocess_isac_scenario(os.path.join(DS,"deepsense6g",f"Scenario{sc}"),sc,max_frames=1500)
    ps.append(a); pm.append(b)
ts,tm=preprocess_isac_scenario(os.path.join(DS,"deepsense6g","Scenario44"),44,max_frames=1500)
out["DeepSense_ISAC"]=run("DeepSense_ISAC",np.concatenate(ps),np.concatenate(pm),ts,tm,PUB["DeepSense_ISAC"])
a_s,a_m,_=pv2i(os.path.join(DS,"deepsense6g","Scenario31")); b_s,b_m,_=pv2i(os.path.join(DS,"deepsense6g","Scenario33"))
out["DeepSense_V2I"]=run("DeepSense_V2I",a_s,a_m,b_s,b_m,PUB["DeepSense_V2I"])
seq=sorted(glob.glob(os.path.join(DS,"radial","extracted","RECORD*")))
s,m=preprocess_radial(seq[0],os.path.join(DS,"radial","labels_CVPR.csv"),
                      os.path.join(DS,"radial","miscellaneous","RADAR_CONFIG.m"),max_frames=500)
n=int(0.8*len(s))
out["RADIal"]=run("RADIal",s[:n],m[:n],s[n:],m[n:],PUB["RADIal"])
p=os.path.join(RESULTS_DIR,"trivial_baselines.json"); json.dump(out,open(p,"w"),indent=2)
print(f"\nSaved: {p}")
