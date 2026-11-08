"""Plot active-task curves from the archived evaluations, without retraining."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'results/figures')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    index=json.loads((ROOT/'results/run_index.json').read_text())
    labels={'cable_cw10':('CABLE','#DE8EAF'),'clonex_sac_cw10':('ClonEx-SAC','#83B8DA'),'recall_cw10':('RECALL','#E8C500')}
    fig,axes=plt.subplots(2,5,figsize=(14,5.5),sharex=True,sharey=True)
    for group,(label,color) in labels.items():
        curves=defaultdict(list);names={}
        for item in sorted((i for i in index if i['group']==group),key=lambda i:i['seed']):
            points=defaultdict(dict)
            for r in csv.DictReader((ROOT/item['evaluations']).open()):
                j=int(r['active_task_index']);t=int(r['active_task_step'])
                if j==int(r['evaluation_task_index']) and 0<t<=500000:
                    points[j].setdefault(t,float(r['stochastic_success_rate']));names[j]=r['active_task_name']
            for j,values in points.items():curves[j].append([v for _,v in sorted(values.items())])
        for j,values in curves.items():
            a=np.asarray(values);x=np.arange(1,a.shape[1]+1)*20000
            indices=np.random.RandomState(0).choice(len(a),(4000,len(a)),replace=True)
            low,high=np.quantile(a[indices].mean(axis=1),[.05,.95],axis=0,method='midpoint')
            ax=axes.flat[j];ax.plot(x,a.mean(axis=0),color=color,label=label);ax.fill_between(x,low,high,color=color,alpha=.16)
            ax.set_title(names[j].replace('-v3','').replace('-',' ').title());ax.set_ylim(0,1.04)
            ax.spines[['top','right']].set_visible(False)
    for ax in axes[-1]:ax.set_xlabel('Training steps')
    for ax in axes[:,0]:ax.set_ylabel('Active-task success')
    handles,labels=axes.flat[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=3,frameon=False)
    fig.tight_layout(rect=(0,.06,1,1))
    for ext in ['pdf','png']:fig.savefig(args.output/('cw10_learning_curves.'+ext),dpi=200)
    plt.close(fig)
if __name__=='__main__':main()
