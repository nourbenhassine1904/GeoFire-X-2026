"""Serve verified U-Net ablation/confirmation artifacts without importing torch."""
import json
from pathlib import Path,PurePosixPath
from geofirex.calibration_io import sha

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'reports/unet_confirmation_20261112_v1'


class UnetResults:
    def __init__(self,historical,root=BASE):
        self.historical=historical;self.root=Path(root);self._stamp=None;self._confirmed=None
        self.audit_path=self.root/'ablation_verified.json'
        self.audited=None
        if self.audit_path.is_file():
            self.audited=json.loads(self.audit_path.read_text(encoding='utf-8'))
            archive=self.root/'inputs/GeoFireX_UNet_ABLATION_3_MODELS.zip'
            if self.audited['status']!='VERIFIED' or sha(archive)!=self.audited['archive_sha256']:
                raise ValueError('Ablation archive/audit mismatch')
            if self.audited['winner']['pos_weight']!=10 or self.audited['winner']['threshold']!=.78:
                raise ValueError('Ablation winner differs from frozen decision')

    def confirmed(self):
        path=self.root/'confirmation_cpu/status_complete.json'
        if not path.is_file():return None
        stamp=path.stat().st_mtime_ns
        if self._stamp==stamp:return self._confirmed
        if self.audited is None:raise ValueError('Confirmation requires verified ablation')
        folder=path.parent
        read=lambda name:json.loads((folder/name).read_text(encoding='utf-8'))
        complete=read('status_complete.json');lock=read('confirmation_locked.json');manifest=read('app_manifest.json')
        if (complete['status'],complete['events'],complete['tiles'],complete['threshold'],complete['test_2021_pixels_opened'])!=('COMPLETE',169,65012,.78,0):
            raise ValueError('Incomplete confirmation')
        if (sha(folder/'confirmation_locked.json')!=complete['confirmation_lock_sha256'] or
            complete['confirmation_lock_sha256']!=manifest['confirmation_lock_sha256'] or
            sha(folder/'app_manifest.json')!=complete['manifest_sha256'] or
            sha(folder/'B2_w10_holdout169/evaluation.json')!=complete['evaluation_sha256']):
            raise ValueError('Confirmation artifact integrity mismatch')
        if lock['checkpoint_sha256']!=self.audited['winner']['checkpoint_sha256'] or lock['threshold']!=.78:
            raise ValueError('Checkpoint/threshold lock differs')
        if manifest['historical_archive_sha256']!=self.historical.zip_sha256 or manifest['ablation_archive_sha256']!=self.audited['archive_sha256']:
            raise ValueError('Source archive differs')
        evaluation=read('B2_w10_holdout169/evaluation.json')
        if evaluation['events']!=self.historical.protocol['holdout_events'] or len(evaluation['curves'])!=1 or evaluation['curves'][0]['threshold']!=.78:
            raise ValueError('Unexpected event/threshold domain')
        if lock['event_ids']!=evaluation['events']:raise ValueError('Frozen event IDs differ')
        row=evaluation['curves'][0]
        for k in ('tp','fp','fn','tn','valid_pixels','positive_pixels'):
            if sum(r[k] for r in evaluation['per_event'])!=row[k]:raise ValueError('Per-event counts differ')
        for model in ('B1','B2'):
            baseline=[r for r in self.historical.per_event['holdout169',model] if r['threshold']==.95]
            lookup={r['event_id']:r for r in baseline}
            for r in evaluation['per_event']:
                if any(r[k]!=lookup[r['event_id']][k] for k in ('valid_pixels','positive_pixels','tiles','invalid_pixels')):
                    raise ValueError('Historical confirmation domain differs')
        historical_rows=[r for r in self.historical.rows if r['phase']=='holdout169' and r['mode']=='calibrated']
        if manifest['comparison'][:2]!=historical_rows or manifest['comparison'][2]['dice']!=row['dice']:
            raise ValueError('Historical comparison differs')
        assets=set()
        for case in manifest['maps']:
            if case['event_id'] not in evaluation['events'] or case['threshold']!=.78:raise ValueError('Map outside confirmation')
            for name in [case['preview'],case['npz'],*case['layers'].values()]:
                logical=PurePosixPath(case['directory'])/name
                if logical.is_absolute() or '..' in logical.parts:raise ValueError('Unsafe map asset')
                if not (folder/logical).is_file():raise ValueError('Missing map asset')
                assets.add(logical.as_posix())
        self._confirmed=dict(status='COMPLETE',manifest=manifest,evaluation=evaluation,new_fire=read('new_fire.json'),
                             execution=complete,lock=lock,assets=assets,folder=folder)
        self._stamp=stamp
        return self._confirmed

    def payload(self):
        if self.audited is None:return dict(status='NOT_AVAILABLE',confirmation=dict(status='NOT_STARTED'))
        a=self.audited;c=self.confirmed()
        variants=[{k:r[k] for k in ('pos_weight','threshold','metrics','checkpoint_sha256','initial_state_sha256')} for r in a['variants']]
        result=dict(status='VERIFIED',variants=variants,selected=dict(model='B2',pos_weight=10,threshold=.78,
                    checkpoint_sha256=a['winner']['checkpoint_sha256'],criterion='micro Dice on 32 only'),
                    delta_dice_10_vs_30=a['delta_dice_10_vs_30'],delta_ap_10_vs_30=a['delta_ap_10_vs_30'],
                    caveat=a['caveat'],provenance=dict(archive_sha256=a['archive_sha256'],protocol_sha256=a['protocol_sha256'],
                    initial_state_sha256=a['initial_state_sha256'],audit_sha256=sha(self.audit_path)),
                    per_event32=a['per_event32'])
        if c:
            evaluation=c['evaluation']
            result['confirmation']=dict(status='COMPLETE',comparison=c['manifest']['comparison'],
                macro_event=c['manifest']['macro_event'],per_event=evaluation['per_event'],
                probabilities=evaluation['probabilities'],new_fire=c['new_fire'],maps=c['manifest']['maps'],
                execution=c['execution'],confirmation_lock_sha256=c['manifest']['confirmation_lock_sha256'],
                historical_new_fire=[r for r in self.historical.diagnosis['rows'] if r['phase']=='holdout169'],
                per_event_new_fire=evaluation['per_event_new_fire'],
                historical_per_event={m:[r for r in self.historical.per_event['holdout169',m] if r['threshold']==.95] for m in ('B1','B2')},
                checkpoint_sha256=c['manifest']['checkpoint_sha256'])
        else:
            started=self.root/'confirmation_cpu/status_started.json'
            progress=list((self.root/'confirmation_cpu/B2_w10_holdout169').glob('event_*_probabilities.json'))
            result['confirmation']=dict(status='RUNNING' if started.is_file() else 'NOT_STARTED',events_completed=len(progress),events_total=169)
        return result


def attach_unet_routes(router,historical):
    from fastapi import HTTPException,Response
    store=UnetResults(historical)
    @router.get('/unet-ablation')
    def unet_ablation():return store.payload()
    @router.get('/retained-model')
    def retained_model():
        data=store.payload()
        return dict(**data.get('selected',{}),confirmation_status=data['confirmation']['status'],
                    live_inference=False,test_2021_locked=True)
    @router.get('/unet-assets/{name:path}')
    def asset(name:str):
        c=store.confirmed()
        if c is None or name not in c['assets']:raise HTTPException(404,'Carte confirmée indisponible')
        path=c['folder']/name
        return Response(path.read_bytes(),media_type='image/png' if name.endswith('.png') else 'application/octet-stream')
    return store
