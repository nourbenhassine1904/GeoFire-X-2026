"""Offline Studio assets from saved V2 tiles; never changes the serving model."""
from pathlib import Path
import numpy as np
from geofirex.protocol_v2 import save,sha256


def error_map(a,threshold):
    admissible=(a['v']==1)&(a['o']==0);truth=a['y']==1
    pred=np.zeros(truth.shape,bool) if threshold is None else a['p']>=np.float32(threshold)
    error=np.where(pred,np.where(truth,1,2),np.where(truth,3,0)).astype(np.uint8)
    error[~admissible]=255
    return error,admissible,pred&admissible,truth&admissible


def export_cases(best,out,source,checkpoint):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    out=Path(out);cases=[]
    palette=np.array([[22,33,43,255],[48,187,128,255],[242,82,82,255],[63,142,252,255]],np.uint8)
    for label,row in best.items():
        a=row['data'];key=row['key'];stem=label+'_'+key[0].replace('/','__')+'_'+key[1]+f'_r{key[2]}_c{key[3]}'
        error,mask,pred,truth=error_map(a,.78)
        raw=out/(stem+'.npz')
        with raw.open('xb') as f:np.savez_compressed(f,probability=a['p'],target=a['y'],observed=a['o'],valid=a['v'],
            new_admissible=mask,new_target=truth,new_prediction=pred,error=error)
        layers={}
        for name,data,alpha in [('probability',a['p'],a['v']==1),('observed',a['o'],a['v']==1),
            ('new_target',truth,mask),('new_prediction',pred,mask)]:
            rgba=(plt.get_cmap('inferno')(data.astype(float))*255).astype(np.uint8);rgba[...,3]=alpha.astype(np.uint8)*255
            p=out/(stem+'_'+name+'.png');Image.fromarray(rgba).save(p);layers[name]=p.name
        rgba=palette[np.minimum(error,3)].copy();rgba[~mask,3]=0
        p=out/(stem+'_new_errors.png');Image.fromarray(rgba).save(p);layers['new_errors']=p.name
        fig,axes=plt.subplots(1,5,figsize=(14,3.5))
        for ax,name in zip(axes,('observed','new_target','probability','new_prediction','new_errors')):
            ax.imshow(Image.open(out/layers[name]),interpolation='nearest');ax.set_title(name,fontsize=10);ax.axis('off')
        fig.suptitle(f'NEWPOS4 | validation {key[0]} | cible {key[1]} | seuil 0.78',fontsize=11)
        fig.text(.5,.01,'Diagnostic sélectionné par maximum de '+label+' ; non représentatif. Grille de pixels, sans CRS.',ha='center',fontsize=9)
        fig.tight_layout(rect=(0,.06,1,.95));preview=out/(stem+'_preview.png');fig.savefig(preview,dpi=130);plt.close(fig)
        cases.append(dict(selection=label+'_maximum_at_078_outcome_selected_not_representative',key=list(key),
            shape=list(a.shape),threshold=.78,npz=raw.name,npz_sha256=sha256(raw),layers=layers,
            preview=preview.name,new_tp=int((error==1).sum()),new_fp=int((error==2).sum()),new_fn=int((error==3).sum())))
    save(out/'map_manifest.json',dict(schema='geofirex-v2-newfire-studio-bundle-1',run_id='B2_C8_V2_NEWPOS4_seed42',
        checkpoint_sha256=checkpoint,source_archive=source,validation_only=True,promotion_authorized=False,
        serving_model_changed=False,spatial_reference='64x64 pixel tiles; row/col from frozen index; no CRS/affine certified here',
        categories={'0':'new-domain TN','1':'new TP green','2':'new FP red','3':'new FN blue','255':'invalid or observed J'},
        all_tiles_index='tile_index.csv',all_tiles_index_sha256=sha256(out/'tile_index.csv'),
        all_predictions='Reference existing COMPLETE/evaluation/predictions; not duplicated in this bundle',cases=cases))
