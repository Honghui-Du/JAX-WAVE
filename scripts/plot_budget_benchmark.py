"""Publication figures from the frozen independent benchmark; no training needed."""
from pathlib import Path
import argparse, hashlib, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'results/budget_benchmark'
parser = argparse.ArgumentParser()
parser.add_argument('--output', type=Path, default=ROOT / 'outputs/figures')
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.family':'STIXGeneral', 'mathtext.fontset':'stix',
    'font.size':9, 'axes.labelsize':9, 'legend.fontsize':8,
    'axes.spines.top':False, 'axes.spines.right':False, 'pdf.fonttype':42})
read = lambda p: json.loads((DATA/p).read_text())
results = read('statistics.json')
summary = read('measurements.json')
desc = read('cases.json')
colors = {'projected_gradient':'#666666','fista':'#326C8C','restarted_fista':'#397A66'}
styles = {'projected_gradient':'--','fista':'-','restarted_fista':'-.'}
labels = {'projected_gradient':'Projected gradient','fista':'FISTA','restarted_fista':'Restarted FISTA'}
outputs = []
def save(fig,name):
    for ext in ['pdf','png']:
        path=args.output/(name+'.'+ext)
        opts={'metadata':{'CreationDate':None}} if ext=='pdf' else {'dpi':220}
        fig.savefig(path,bbox_inches='tight',pad_inches=.05,**opts)
        outputs.append(path)
    plt.close(fig)

fig,axes=plt.subplots(1,2,figsize=(7.0,2.8),sharey=True)
for ax,kind,label in zip(axes,['geometry','reachable'],['(a) Smooth geometry','(b) Reachable targets']):
    ids={d['case'] for d in desc if d['contrast']>1 and
         ((d['target_type']=='geometry_broad') if kind=='geometry' else d['target_type'].startswith('reachable'))}
    for method in colors:
        rows={}
        for group in range(36):
            for record in summary[method+'/group_%02d'%group]['timing']:
                rows.setdefault(record['steps'],[]).extend(v for v in record['cases'] if v['case'] in ids)
        xs=[];ys=[]
        for step in sorted(rows):
            q=rows[step];xs.append(np.median([v['median_ms'] for v in q]));ys.append(100*np.mean([v['error'] for v in q]))
        ax.plot(xs,ys,marker='o',linestyle=styles[method],color=colors[method],
                label=labels[method],markersize=2.8,linewidth=1.1)
    for seed in range(3):
        q=[v for group in range(36) for v in summary['neural_s%d/group_%02d'%(seed,group)]['timing'][0]['cases'] if v['case'] in ids]
        ax.scatter(np.median([v['median_ms'] for v in q]),100*np.mean([v['error'] for v in q]),
                   marker='*',s=65,color='#C35635',zorder=5,label='JAX-WAVE (3 seeds)' if seed==0 else None)
    ax.axvline(5,color='#23333D',linestyle='--',linewidth=.7)
    ax.set_title(label,loc='left',fontsize=9,pad=7)
    ax.set_xscale('log');ax.set_xlim(3,330);ax.set_ylim(0,57)
    ax.set_xlabel('Median query time (ms)');ax.grid(alpha=.17,linewidth=.5)
axes[0].set_ylabel('Mean relative field error (%)')
axes[1].legend(loc='upper right',bbox_to_anchor=(1,.86),frameon=False,fontsize=7)
fig.tight_layout(w_pad=1.3);save(fig,'budget_accuracy_latency')

fig,ax=plt.subplots(figsize=(6.4,2.8));x=np.arange(3);width=.23
for off,method in zip([-1,0,1],colors):
    q=[next(v for v in results['comparisons'] if v['budget_ms']==b and v['baseline']==method and v['stratum']=='primary_heterogeneous') for b in [5,10,20]]
    y=np.array([v['relative_reduction_percent'] for v in q]);bounds=np.array([v['relative_reduction_ci95_percent'] for v in q])
    ax.bar(x+off*width,y,width,label=labels[method],color=colors[method],hatch=['','//','..'][off+1],edgecolor='white',linewidth=.5)
    ax.errorbar(x+off*width,y,yerr=np.stack([y-bounds[:,0],bounds[:,1]-y]),fmt='none',ecolor='#23333D',capsize=3,linewidth=.8)
ax.axhline(0,color='black',linewidth=.7);ax.set_xticks(x);ax.set_xticklabels(['5 ms (primary)','10 ms','20 ms'])
ax.set_ylabel('Relative error reduction (%)');ax.legend(fontsize=7,frameon=False,loc='lower left');fig.tight_layout();save(fig,'budget_advantage')

fields=np.load(DATA/'representative_fields.npz')['fields']
metadata=read('field_examples.json')
fig,axs=plt.subplots(3,4,figsize=(6.8,4.4))
for row in range(3):
    lim=float(abs(fields[row]).max())
    for col in range(4):
        ax=axs[row,col];im=ax.imshow(fields[row,col].T,origin='lower',cmap='RdBu_r',vmin=-lim,vmax=lim,extent=[0,1,0,1])
        ax.set_xticks([]);ax.set_yticks([])
        for spine in ax.spines.values(): spine.set_visible(False)
        if row==0:ax.set_title(['Target','JAX-WAVE','FISTA (5 ms)','Reference'][col],fontsize=9,pad=5)
        if col:ax.set_xlabel('%.2f%%'%metadata[row]['errors_percent'][col-1],fontsize=8,labelpad=2)
    axs[row,0].set_ylabel(['Smooth\ngeometry','Reachable\n$K=2$','Reachable\n$K=8$'][row],fontsize=9)
    fig.colorbar(im,ax=axs[row,3],fraction=.046,pad=.035,ticks=[-lim,0,lim],format='%.2f').ax.tick_params(labelsize=6)
fig.subplots_adjust(left=.10,right=.96,bottom=.05,top=.92,hspace=.25,wspace=.10)
save(fig,'budget_fields')
manifest={'data_manifest_sha256':hashlib.sha256((DATA/'manifest.json').read_bytes()).hexdigest(),
          'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'outputs':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in outputs},
          'examples':'contrast 1.5, descriptor 0, all three target families, neural seed 0',
          'curves':'uniform iteration counts; table uses timing-only selection per condition'}
(args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('Wrote',len(outputs),'figure artifacts')
