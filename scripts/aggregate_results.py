"""Recompute paper metrics from portable evaluation records.

All sequence positions are included in reported normalized forward transfer.
FT is omitted unless an explicit fixed single-task reference is provided.
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
def interval(values):
    x = np.asarray(values, dtype=float)
    samples = np.random.RandomState(0).choice(len(x), (4000, len(x)), replace=True)
    low, high = np.quantile(x[samples].mean(axis=1), [.05,.95], method='midpoint')
    return {'mean':float(x.mean()),'low':float(low),'high':float(high),'n_seeds':len(x)}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--output', type=Path, default=ROOT/'results/recomputed_metrics.json')
    args = parser.parse_args()
    reference = None
    if args.reference:
        b = json.loads(args.reference.read_text())
        reference = {t:float(np.mean(c)) for t,c in zip(b['tasks'],b['stochastic_success_curves'],strict=True)}
    groups = defaultdict(lambda:defaultdict(lambda:defaultdict(list)))
    run_results = []
    for item in json.loads((ROOT/'results/run_index.json').read_text()):
        rows = list(csv.DictReader((ROOT/item['evaluations']).open()))
        active = defaultdict(dict)
        tasks = {}
        for row in rows:
            i,j,t = int(row['active_task_index']),int(row['evaluation_task_index']),int(row['active_task_step'])
            tasks[j] = row['evaluation_task_name']
            if i==j and 0<t<=500000:
                active[j].setdefault(t,float(row['stochastic_success_rate']))
        # Final evaluation rounds share the terminal training step.
        terminal = [r for r in rows if int(r['global_step'])==500000*len(tasks)]
        final = defaultdict(list)
        for row in terminal:
            final[int(row['evaluation_task_index'])].append(float(row['stochastic_success_rate']))
        end = [np.mean([v for _,v in sorted(active[j].items())][-5:]) for j in sorted(tasks)]
        last = [np.mean(final[j][-5:]) for j in sorted(tasks)]
        values = {'average_performance':float(np.mean(last)), 'average_forgetting':float(np.mean(np.array(end)-last))}
        if reference is not None:
            ft=[]
            for j in sorted(tasks):
                base=reference[tasks[j]]
                if base>=1: raise ValueError('FT undefined for a unit-AUC reference')
                ft.append((float(np.mean(list(active[j].values())))-base)/(1-base))
            values['forward_transfer']=float(np.mean(ft))
        for metric,value in values.items():groups[item['group']][metric][item['seed']].append(value)
        run_results.append({'name':item['name'],**values})
    result = {'reference':str(args.reference) if args.reference else None,'groups':{},'runs':run_results}
    for group,metrics in groups.items():
        result['groups'][group] = {m:interval([np.mean(v) for _,v in sorted(seeds.items())]) for m,seeds in metrics.items()}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(args.output)
if __name__ == '__main__':main()
