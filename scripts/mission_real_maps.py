"""Readable full-scene cases from executed B0 and trained B1 checkpoint."""
import json
from pathlib import Path
import numpy as np
import rasterio
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap,BoundaryNorm
from geofirex.mission_streaming import load,save,sha
from geofirex.runner_data import TemporalStore
from geofirex.benchmark_models import make_model
from geofirex.unet_data import tile_coordinates
from geofirex.windows import Window

def main():
    base=Path('reports/mission_20261111');out=base/'scene_maps';out.mkdir(exist_ok=False)
    _,groups,domains,n=load('reports/full_corpus_v1/prepared_spark_v1','D:/GeoFire-X/WildfireSpreadTS/rasters')
    b0=json.loads((base/'b0_v2/results.json').read_text());config=json.loads((base/'b1_cpu_pilot/config.json').read_text())
    candidates=sorted(b0['per_window'],key=lambda r:(-r['fn'],r['event_id'],r['target_date']))
    cases=[candidates[0],next(r for r in candidates if r['event_id'] in config['val_events'])]
    checkpoint=base/'b1_cpu_pilot/checkpoint_epoch_02.pt';state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    model=make_model('B1',8,3,architecture='spatial_cnn_v2');model.load_state_dict(state['model']);model.eval();torch.set_num_threads(2)
    lookup={(w.event_id,w.target_date):w for w in groups['val']};index=[]
    for case_id,row in enumerate(cases):
        w=lookup[(row['event_id'],row['target_date'])];dd=domains[(w.event_id,w.target_date)]
        domain=Window(event_id=w.event_id,target_date=w.target_date,target_path=w.target_path,context_dates=tuple(Path(p).stem for p in dd),context_paths=tuple(dd))
        store=TemporalStore([w],n,[domain],cache_bytes=64*1024**2)
        height,width=store.shapes[0];context=np.zeros((3,height,width),np.float32);target=np.zeros((height,width));observed=np.zeros_like(target);probability=np.zeros_like(target);valid=np.zeros_like(target,dtype=bool)
        for r,c in tile_coordinates(height,width,64):
            x,y,v,p=store.tile(0,r,c,64);h=min(64,height-r);ww=min(64,width-c)
            context[:,r:r+h,c:c+ww]=x[:,7,:h,:ww];target[r:r+h,c:c+ww]=y[0,:h,:ww];observed[r:r+h,c:c+ww]=p[0,:h,:ww];valid[r:r+h,c:c+ww]=v[0,:h,:ww]
            if case_id==1:
                with torch.no_grad():probability[r:r+h,c:c+ww]=model.predict_proba(torch.from_numpy(x).unsqueeze(0))[0,0,:h,:ww].numpy()
        error=np.where((target==1)&(observed==0),1,np.where((target==0)&(observed==1),-1,0))
        fig,axes=plt.subplots(2,3,figsize=(12,8));cmap=ListedColormap(['#2878b5','#e4e8e8','#db3c35']);norm=BoundaryNorm([-1.5,-.5,.5,1.5],3)
        panels=[(context[i],f'Observation {w.context_dates[i]}') for i in range(3)]+[(target,f'Verite terrain {w.target_date}'),(observed,'B0 Persistence'),(error,'Erreurs B0 : rouge FN / bleu FP')]
        for ax,(a,title) in zip(axes.flat,panels):
            if title.startswith('Erreurs'):ax.imshow(np.ma.array(a,mask=~valid),cmap=cmap,norm=norm,interpolation='nearest')
            else:ax.imshow(np.ma.array(a,mask=~valid),cmap=ListedColormap(['#15292f','#ff943d']),vmin=0,vmax=1,interpolation='nearest')
            ax.set_title(title,fontsize=10);ax.set_xlabel('Colonne pixel');ax.set_ylabel('Ligne pixel')
        fig.suptitle(f'{w.event_id} | scene complete {height} x {width}\nCas a FN eleves choisi pour illustration, pas un echantillon representatif',fontsize=11);fig.tight_layout();fig.savefig(out/f'case_{case_id:02d}_B0.png',dpi=150);plt.close(fig)
        with rasterio.open(w.target_path) as ds:meta=dict(event_id=w.event_id,target_date=w.target_date,context_dates=list(w.context_dates),crs=str(ds.crs),transform=list(ds.transform),shape=[height,width],case_selection='highest B0 FN overall / highest B0 FN among six B1 validation events',b0_metrics=row)
        np.savez_compressed(out/f'case_{case_id:02d}.npz',context=context,target=target,observed=observed,valid=valid,error_b0=error,**({'b1_probability':probability} if case_id==1 else {}))
        if case_id==1:
            fig,axes=plt.subplots(1,4,figsize=(15,4))
            for ax,a,title in zip(axes,[target,observed,probability,probability>=.5],['Verite J+1','B0 Persistence','B1 CPU probabilite','B1 CPU seuil 0.5']):
                im=ax.imshow(np.ma.array(a,mask=~valid),vmin=0,vmax=1,cmap='inferno',interpolation='nearest');ax.set_title(title);ax.axis('off')
            fig.colorbar(im,ax=list(axes),fraction=.02,pad=.02);fig.suptitle(f'{w.event_id} | {w.target_date} | B1 deux epoques, validation historique');fig.savefig(out/'B1_actual_prediction.png',dpi=150,bbox_inches='tight');plt.close(fig)
            meta['b1_checkpoint_sha256']=sha(checkpoint);meta['b1_optimizer_updates']=state['optimizer_updates']
        save(out/f'case_{case_id:02d}.json',meta);index.append(meta)
    save(out/'index.json',index);print('Full-scene B0 cases and actual trained B1 prediction saved')

if __name__=='__main__':
    with rasterio.Env(GDAL_CACHEMAX=32*1024**2):main()
