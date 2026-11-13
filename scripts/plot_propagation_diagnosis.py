"""Static demo figure from saved probabilities only, no re-inference."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/propagation_20261112_v1'
report=json.loads((OUT/'diagnosis.json').read_text())
case=report['cases'][1]  # Same metadata-selected middle case, not a new selection.
stem=next(iter(case['layers'].values())).removesuffix('_observed.png')
path=ROOT/'reports/unet_confirmation_20261112_v1/confirmation_cpu/B2_w10_holdout169'/(stem+'_raw.npz')
with np.load(path) as z:p,y,v,o=[z[k] for k in ('probability','target','valid','observed')]
n=v & (o==0);target=n & (y==1);prediction=n & (p>=np.float32(.78))
error=np.where(prediction,np.where(target,1,2),np.where(target,3,0))
union=(v & (o==1))|target|prediction
rr,cc=np.where(union);roi=np.s_[max(0,rr.min()-16):min(len(v),rr.max()+17),max(0,cc.min()-16):min(v.shape[1],cc.max()+17)]
fig,axes=plt.subplots(2,3,figsize=(14,8.8),layout='constrained')
panels=[(o,v,'Feu observe J','inferno',1),(target,v,'Nouveaux feux cibles J+1','inferno',1),
    (prediction,v,'Prediction nouveaux feux | seuil 0,78','inferno',1),(p,n,'Probabilite | absence de feu a J','inferno',1),
    (error,n,'Erreurs nouvelles : TP vert / FP rouge / FN bleu',ListedColormap(['#16212b','#30bb80','#f25252','#3f8efc']),3)]
for ax,(a,mask,title,cmap,vmax) in zip(axes.ravel(),panels):
    layer=ax.imshow(np.ma.array(a[roi],mask=~mask[roi]),vmin=0,vmax=vmax,cmap=cmap,interpolation='nearest')
    if a is p:fig.colorbar(layer,ax=ax,fraction=.046,pad=.03,label='Probabilite [0,1]')
    ax.set_title(title,fontsize=10);ax.set_xticks([]);ax.set_yticks([])
ax=axes[1,2]
values=sorted(r['recall'] for r in report['per_event'] if r['recall'] is not None)
ax.plot(values,np.arange(1,len(values)+1)/len(values),color='#397d6c')
ax.set(xlabel='Rappel nouveaux pixels par evenement',ylabel='Fraction cumulee des evenements',xlim=(0,1),ylim=(0,1),title='169 evenements | 165 rappels definis')
ax.axvline(report['summary']['new_recall_event']['median'],color='#974f7b',linestyle='--',label='mediane 0,0877');ax.legend(fontsize=9);ax.grid(alpha=.2)
fig.suptitle('U-Net poids 10 | '+case['event_id']+' | '+case['target_date']+'\nROI autour des detections et erreurs, grille de pixels | 228 nouveaux feux, 27 TP, 201 FN | pas une estimation du corpus\nBlanc dans probabilites/erreurs : deja observe a J ou invalide, exclu du domaine nouveau',fontsize=11)
fig.savefig(OUT/'propagation_diagnostic.png',dpi=170);plt.close(fig)
print(OUT/'propagation_diagnostic.png')
