"""Paired benchmark statistics with descriptor-cluster bootstrap intervals."""
from pathlib import Path
import hashlib,json,csv
import numpy as np
ROOT=Path(__file__).resolve().parents[1]/'results/budget_benchmark';OUT=ROOT;REPORT=ROOT.parents[1]/'outputs/budget_reanalysis'
METHODS=['projected_gradient','fista','restarted_fista'];SEEDS=[0,1,2]
BUDGETS=[5.,10.,20.]

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,obj):Path(p).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')

def main():
 cfg=read(ROOT/'protocol.json');protocol=cfg
 s=read(OUT/'measurements.json');descriptors=read(OUT/'cases.json');REPORT.mkdir(parents=True,exist_ok=True)
 cell_ids=[];cells=[]
 for contrast in protocol['contrast']:
  for target in protocol['target_types']:
   ids=[d['case'] for d in descriptors if d['contrast']==contrast and d['target_type']==target]
   assert len(ids)==16 and [descriptors[i]['replicate'] for i in ids]==list(range(16))
   assert [descriptors[i]['family'] for i in ids]==['layered' if i%2==0 else 'inclusions' for i in range(16)]
   cell_ids.append(ids);cells.append({'contrast':contrast,'target_type':target})
 errors={};latency={};raw_times={}
 for method in cfg['methods']:
  if method=='reference_fista':continue
  rows={}
  for group in range(36):
   entry=s[method+'/group_%02d'%group]
   for t in entry['timing']:
    step=t['steps'];rows.setdefault(step,{})
    for r in t['cases']:
     assert r['case'] not in rows[step];rows[step][r['case']]=r
  for step,values in rows.items():
   assert sorted(values)==list(range(144));key=(method,step)
   errors[key]=np.array([values[i]['error'] for i in range(144)])
   latency[key]=np.array([values[i]['median_ms'] for i in range(144)])
   raw_times[key]=np.array([values[i]['repeat_ms'] for i in range(144)])
 neural=np.array([[errors[('neural_s%d'%seed,0)][ids] for ids in cell_ids] for seed in SEEDS])
 neural_latency=np.array([[latency[('neural_s%d'%seed,0)][ids] for ids in cell_ids] for seed in SEEDS])
 assert neural.shape==(3,9,16)
 # Repeated material/target descriptor IDs are the resampling units, across all factors.
 rng=np.random.default_rng(2026114001);NBOOT=20000
 boot=np.concatenate([rng.choice(np.arange(0,16,2),size=(NBOOT,8),replace=True),rng.choice(np.arange(1,16,2),size=(NBOOT,8),replace=True)],axis=1)
 def stats(num,nn):
  # Both arrays are cells x16. Average cells before resampling paired replicate IDs.
  n=num.mean(axis=0);a=nn.mean(axis=0);nb=n[boot].mean(axis=1);ab=a[boot].mean(axis=1)
  diff=nb-ab;rel=100*(1-ab/nb);actual=n-a
  return {'numerical_error_percent':float(100*n.mean()),'neural_error_percent':float(100*a.mean()),
   'absolute_reduction_percentage_points':float(100*actual.mean()),'absolute_reduction_ci95_pp':(100*np.percentile(diff,[2.5,97.5])).tolist(),
   'relative_reduction_percent':float(100*(1-a.mean()/n.mean())),
   'relative_reduction_ci95_percent':np.percentile(rel,[2.5,97.5]).tolist(),
   'paired_case_win_fraction':float((nn<num).mean())}
 primary=[i for i,c in enumerate(cells) if c['contrast']>1]
 groups={'primary_heterogeneous':primary,'geometry_heterogeneous':[i for i in primary if cells[i]['target_type']=='geometry_broad'],
         'reachable_heterogeneous':[i for i in primary if cells[i]['target_type'].startswith('reachable')],
         'homogeneous_control':[0,1,2]}
 results=[];selected=[];tables=[];selection_audit=[]
 for budget in BUDGETS:
  for method in METHODS:
   chosen=[];num=[];num_latency=[]
   for cell,ids in zip(cells,cell_ids):
    options=[]
    for step in cfg['milestones'][1:]:
     ms=float(np.median(latency[(method,step)][ids]));options.append({'steps':step,'median_ms':ms})
    eligible=[o for o in options if o['median_ms']<=budget]
    choice=max(eligible,key=lambda o:o['steps']) if eligible else None
    selection_audit.append({'budget_ms':budget,'method':method,**cell,'all_latency_candidates':options,'chosen':choice})
    chosen.append(choice)
    if choice is None:num.append(np.full(16,np.nan));num_latency.append(np.full(16,np.nan))
    else:num.append(errors[(method,choice['steps'])][ids]);num_latency.append(latency[(method,choice['steps'])][ids])
   num=np.array(num);num_latency=np.array(num_latency)
   for label,idx in groups.items():
    supported=all(chosen[i] is not None for i in idx)
    feasible=bool(np.all(np.median(neural_latency[:,idx,:],axis=-1)<=budget))
    entry={'budget_ms':budget,'baseline':method,'stratum':label,'case_count':len(idx)*16,
           'numeric_budget_supported':supported,'all_neural_seeds_within_regime_median_budget':feasible}
    if supported:
     entry.update(stats(num[idx],neural.mean(axis=0)[idx]))
     entry['per_seed']=[{'seed':seed,**stats(num[idx],neural[seed,idx])} for seed in SEEDS]
     entry['neural_latency_median_ms']=float(np.median(neural_latency[:,idx,:]))
     entry['numeric_latency_median_ms']=float(np.median(num_latency[idx]))
    results.append(entry)
   for i,cell in enumerate(cells):
    if chosen[i] is None:continue
    row={'budget_ms':budget,'baseline':method,**cell,'numerical_steps':chosen[i]['steps'],
         'numerical_latency_median_ms':chosen[i]['median_ms'],'neural_latency_median_ms':float(np.median(neural_latency[:,i,:])),
         **stats(num[i:i+1],neural.mean(axis=0)[i:i+1])}
    tables.append(row)
 primary_rows=[r for r in results if r['budget_ms']==5 and r['stratum']=='primary_heterogeneous']
 reference=[]
 for group in range(36):
  h=s['reference_fista/group_%02d'%group]['history'][-1]
  reference.append({'group':group,'error_percent':100*h['validation_terminal_field_relative_l2'],'half_objective_gap_bound':h['max_normalized_objective_gap_bound']})
 report={'bootstrap':{'replicates':NBOOT,'seed':2026114001,'unit':'paired descriptor replicate ID across all factors;8resamples per material family'},'primary_case_count':96,'total_case_count':144,'model_seeds':SEEDS,'comparisons':results,'per_regime':tables,'reference':reference}
 save(REPORT/'statistics.json',report);save(REPORT/'latency_selection.json',selection_audit)
 with (REPORT/'regime_results.csv').open('w') as f:
  keys=[k for k in tables[0] if not isinstance(tables[0][k],list)];w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows([{k:r[k] for k in keys} for r in tables])
 print('Computed paired statistics for 144 cases and three model seeds.',flush=True)
 # Cross-check that the selected iteration count was determined by latency alone.
 for a in selection_audit:
  eligible=[v for v in a['all_latency_candidates'] if v['median_ms']<=a['budget_ms']]
  assert a['chosen']==(max(eligible,key=lambda v:v['steps']) if eligible else None)
 assert all(r['case_count']==96 for r in primary_rows)

if __name__=='__main__':main()
