"""Import completed aggregate results; never train or rerun evaluation."""
import ast
import base64
import hashlib
import io
import json
from pathlib import Path
import shutil
import zipfile

import torch
from geofirex.calibration_io import sha
from geofirex.compact_calibration import metrics
from geofirex.propagation_experiment import PropagationUNet

ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'artifacts/propagation-comparison'


def integrate():
    source=Path('C:/Users/nourb/Downloads/GeoFireX_UNET_PROPAGATION_COMPLETE.zip')
    DEST.mkdir(parents=True,exist_ok=True)
    target=DEST/source.name
    if target.exists():
        if sha(target)!=sha(source):raise ValueError('Existing import differs; do not overwrite')
    else:shutil.copyfile(source,target)
    with zipfile.ZipFile(target) as z:
        selection=json.loads(z.read('selection32/selection32.json'));config=json.loads(z.read('training/config.json'))
        complete=json.loads(z.read('training/training_complete.json'));lock=json.loads(z.read('selection32/selection_scope_locked.json'))
        cp=z.read('training/checkpoint_latest.pt')
    digest=hashlib.sha256(cp).hexdigest()
    if not digest==complete['checkpoint_sha256']==selection['checkpoint_sha256']==lock['checkpoint_sha256']:raise ValueError('Checkpoint SHA differs')
    state=torch.load(io.BytesIO(cp),map_location='cpu',weights_only=False)
    if state['config']!=config or (state['epoch'],state['cursor'],state['updates'])!=(5,0,23404):raise ValueError('Incomplete checkpoint')
    if (config['seed'],config['train_events'],config['train_tiles'],config['model_context'])!=(42,250,93603,3):raise ValueError('Protocol differs')
    baseline=json.loads((ROOT/'reports/propagation_20261112_v1/baseline32.json').read_text())
    if lock['events']!=baseline['events'] or config['calibration_events']!=baseline['events']:raise ValueError('32-event IDs differ')
    if selection['holdout169_pixels_opened'] or selection['test_2021_pixels_opened']:raise ValueError('Unexpected evaluation scope')
    for field in ('full','new_fire'):
        row=selection[field];recomputed=metrics([row[k] for k in ('tp','fp','fn','tn')])
        for k,v in recomputed.items():
            if v is not None and abs(v-row[k])>1e-12:raise ValueError('Saved counts/metrics differ: '+field+'/'+k)
        saved=next(r for r in baseline['curves' if field=='full' else 'new_fire_candidates'] if r['threshold']==.78)
        if saved!=selection['baseline32'][field]:raise ValueError('Historical reference differs')
    if selection['threshold']!=.845 or selection['accepted_on_development32'] is not False:raise ValueError('Unexpected frozen decision')
    # Retrieve the actual embedded source from the downloaded executed notebook.
    matched=None
    for path in Path('C:/Users/nourb/Downloads').glob('notebookb385a8b739*.ipynb'):
        notebook=json.loads(path.read_text(encoding='utf-8'))
        for cell in notebook['cells']:
            text=''.join(cell.get('source',[]))
            if cell['cell_type']!='code' or 'base64.b64decode' not in text:continue
            tree=ast.parse(text)
            for node in ast.walk(tree):
                if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='b64decode' and node.args and isinstance(node.args[0],ast.Constant):
                    payload=base64.b64decode(node.args[0].value)
                    with zipfile.ZipFile(io.BytesIO(payload)) as bundle:
                        contents={n:bundle.read(n) for n in bundle.namelist()}
                    name='geofirex/propagation_experiment.py'
                    # The executed notebook records this one literal correction.
                    # Reconstruct bytes only; never execute a notebook or patch its history.
                    correction='config=dict(**plan,training_executed=True,'
                    replacement='config=dict(plan,training_executed=True,'
                    patch_recorded=any(correction in ''.join(c.get('source',[])) and replacement in ''.join(c.get('source',[])) for c in notebook['cells'])
                    if patch_recorded:
                        source=contents[name].decode()
                        if source.count(correction)!=1:raise ValueError('Unexpected patch source')
                        contents[name]=source.replace(correction,replacement,1).encode()
                    if all(hashlib.sha256(contents['geofirex/'+n]).hexdigest()==h for n,h in config['source_sha256'].items()):
                        buffer=io.BytesIO()
                        with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED) as output:
                            for n,data in contents.items():output.writestr(n,data)
                        matched=buffer.getvalue()
    if matched is None:raise ValueError('Exact training source not found in downloaded notebooks')
    code=DEST/'executed-training-code.zip';code.write_bytes(matched)
    with zipfile.ZipFile(code) as z:executed=z.read('geofirex/propagation_experiment.py').decode()
    def classes(source):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(source).body if isinstance(n,ast.ClassDef) and n.name in ('PropagationUNet','PropagationLoss')}
    if classes(executed)!=classes((ROOT/'src/geofirex/propagation_experiment.py').read_text(encoding='utf-8')):
        raise ValueError('Training architecture/loss source differs from local implementation')
    model=PropagationUNet();model.load_state_dict(state['model'],strict=True)
    if any(not torch.isfinite(v).all() for v in model.state_dict().values()):raise ValueError('Nonfinite model weights')
    reference=selection['baseline32'];new=selection
    delta={field:{k:new[field][k]-reference[field][k] for k in ('dice','iou','precision','recall','tp','fp','fn')} for field in ('full','new_fire')}
    delta['full']['average_precision']=new['full']['average_precision']-reference['full']['average_precision']
    result=dict(status='VERIFIED_COMPARATIVE_ONLY',promoted=False,principal_model='B2 pos_weight10',principal_threshold=.78,
        selection=selection,delta=delta,training=dict(epochs=4,updates=23404,gpu=config['gpu_name'],seconds=complete['gpu_wall_seconds']),
        provenance=dict(archive=target.relative_to(ROOT).as_posix(),archive_sha256=sha(target),checkpoint_sha256=digest,
            executed_source_bundle=code.relative_to(ROOT).as_posix(),executed_source_sha256=sha(code),
            architecture_and_loss_ast_match=True,initial_main_sha256=config['initial_main_state_sha256']),
        limitations=['ZIP contains aggregate metrics and checkpoint only; per-event outputs, threshold curves and new-model maps not supplied. No evaluation rerun.',
            'Threshold is the recorded selection; optimum cannot be reverified without the saved full curve.'],
        evaluations_rerun=False,test_2021_pixels_opened=0)
    (DEST/'comparison.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(dict(status=result['status'],delta=delta),indent=2))


if __name__=='__main__':integrate()
