from pathlib import Path
import csv,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
parser=argparse.ArgumentParser(description='Plot saved scalar metrics only; never infer a model or decode raster data.')
parser.add_argument('--audit', type=Path, default=Path('reports/scientific_improvements_v1'))
parser.add_argument('--output', type=Path, required=True)
args=parser.parse_args()
root=args.audit
out=args.output
if out.exists(): raise FileExistsError(out)
d=json.loads((root/'saved_metrics_audit.json').read_text(encoding='utf-8'))
rows=d['pooled']; names=['CNN historique\nseuil 0,95','U-Net historique\nseuil 0,95','U-Net poids10\nseuil 0,78']
fig,axs=plt.subplots(1,3,figsize=(15.6,4.8),layout='constrained')
colors=['#b45309','#0f766e','#7c3aed']
ax=axs[0]
for offset,key,label,color in [(-.18,'dice','Dice micro','#0f766e'),(.18,'average_precision','AP pooled','#b45309')]:
 bars=ax.bar([i+offset for i in range(3)],[r[key] for r in rows],width=.34,label=label,color=color)
 ax.bar_label(bars,fmt='%.3f',fontsize=9,padding=3)
ax.set_xticks(range(3),names,fontsize=9);ax.set_ylim(0,.65);ax.set_title('Des classements différents');ax.legend(fontsize=9,loc='upper right');ax.set_ylabel('Score sur les mêmes pixels-jours')
ax=axs[1]
groups=d['new_fire_partition']; r=[groups[k]['recall'] for k in ['new_fire_candidates','observed_fire_candidates']]
bars=ax.bar(['Nouveaux pixels\nabsence à J','Déjà actifs à J'],[v*100 for v in r],color=['#b91c1c','#0f766e'],width=.5)
ax.bar_label(bars,fmt='%.2f %%',padding=4);ax.set_ylim(0,105);ax.set_ylabel('Rappel (%)');ax.set_title('Poids10 : les nouveaux pixels manqués')
ax=axs[2]
with (root/'per_event_saved.csv').open(encoding='utf-8',newline='') as f: events=list(csv.DictReader(f))
for model,label,color in zip(['B1','B2','B2_w10'],['CNN historique','U-Net historique','U-Net poids10'],colors):
 values=sorted(float(r['dice']) for r in events if r['model']==model)
 ax.step(values,[(i+1)/len(values)*100 for i in range(len(values))],where='post',label=label,color=color)
ax.set(xlim=(0,1),ylim=(0,100),xlabel='Dice par événement',ylabel='Événements cumulés (%)',title='169 événements : dispersion du Dice');ax.legend(fontsize=8,loc='lower right')
for ax in axs:
 ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
fig.suptitle('GeoFire-X — sorties enregistrées, confirmation historique 169',fontsize=15)
fig.supxlabel('Seuils déjà fixés sur développement32 • Agrégations de résultats sauvegardés • Aucune nouvelle inférence • Aucun pixel2021',fontsize=9)
with out.open('xb') as stream:
 fig.savefig(stream,format='png',dpi=160)
plt.close(fig)
