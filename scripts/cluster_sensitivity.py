#!/usr/bin/env python3
"""Reproduce descriptor-deletion sensitivity from frozen 5 ms case errors.
Uses the included case CSV and Python standard library.
"""
from pathlib import Path
import argparse,csv,hashlib,json,statistics
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/cluster_sensitivity'
METHODS=['projected_gradient','fista','restarted_fista']
mean=statistics.mean
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=ROOT/'outputs/cluster_sensitivity.json');args=ap.parse_args()
    provenance=json.loads((OUT/'metadata.json').read_text())
    assert sha(OUT/'cases.csv')==provenance['case_csv_sha256']
    rows=list(csv.DictReader((OUT/'cases.csv').open()))
    assert len(rows)==96 and len({r['case'] for r in rows})==96
    for r in rows:
        r['replicate']=int(r['replicate']);r['neural']=mean(float(r['neural_s'+str(s)]) for s in range(3))
        for m in METHODS:r[m]=float(r[m])
    clusters={i:[r for r in rows if r['replicate']==i] for i in range(16)}
    conditions={(r['contrast'],r['target_type']) for r in rows};assert len(conditions)==6
    for i,rr in clusters.items():
        assert len(rr)==6 and {(r['contrast'],r['target_type']) for r in rr}==conditions
        assert {r['family'] for r in rr}=={'layered' if i%2==0 else 'inclusions'}
    def avg(method,omit=None):
        return mean(mean(mean(r[method] for r in clusters[i]) for i in range(p,16,2) if i!=omit) for p in [0,1])
    result={'scope':'Descriptive leave-one-cluster-out sensitivity of existing 5 ms errors. Fixed solver iteration choices and three-seed predictions; six conditions removed together; material-family weights remain 1/2 each. No new independent cases, confidence interval or latency reselection.','case_csv_sha256':sha(OUT/'cases.csv'),'comparisons':{}}
    for m in METHODS:
        full=100*(1-avg('neural')/avg(m));assert abs(full-provenance['published_reductions'][m])<1e-10
        omitted=[]
        for i in range(16):
            rel=100*(1-avg('neural',i)/avg(m,i))
            # Independent row-weighted calculation checks family weighting after deletion.
            kept=[r for r in rows if r['replicate']!=i]
            weight=lambda r:1/(2*6*(7 if r['replicate']%2==i%2 else 8))
            assert abs(sum(weight(r) for r in kept)-1)<1e-12
            direct=100*(1-sum(weight(r)*r['neural'] for r in kept)/sum(weight(r)*r[m] for r in kept))
            assert abs(rel-direct)<1e-10
            omitted.append({'omitted_descriptor':i,'relative_reduction_percent':rel})
        result['comparisons'][m]={'full_relative_reduction_percent':full,'loco_range_percent':[min(r['relative_reduction_percent'] for r in omitted),max(r['relative_reduction_percent'] for r in omitted)],'all_deletions_positive':all(r['relative_reduction_percent']>0 for r in omitted),'deletions':omitted}
    result['checks_passed']=True
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({m:{k:v for k,v in d.items() if k!='deletions'} for m,d in result['comparisons'].items()},indent=2))
if __name__=='__main__':main()
