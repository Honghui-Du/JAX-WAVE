"""Recompute the frozen paired analysis from recorded single-query results."""
from pathlib import Path
import argparse, hashlib, importlib.util, json
ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser()
parser.add_argument('--output',type=Path,default=ROOT/'outputs/budget_reanalysis')
args=parser.parse_args()
data=ROOT/'results/budget_benchmark'
for name,digest in json.loads((data/'manifest.json').read_text()).items():
    assert hashlib.sha256((data/name).read_bytes()).hexdigest()==digest,name
spec=importlib.util.spec_from_file_location('frozen_confirmation',ROOT/'jaxwave/benchmark_analysis.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
module.ROOT=data;module.OUT=data;module.REPORT=args.output
args.output.mkdir(parents=True,exist_ok=True)
module.main()
for name in ['statistics.json','latency_selection.json','regime_results.csv']:
    if name.endswith('.json'):
        assert json.loads((args.output/name).read_text())==json.loads((data/name).read_text()),name
    else:
        assert (args.output/name).read_bytes()==(data/name).read_bytes(),name
print('Paired analysis matches frozen published results exactly.')
