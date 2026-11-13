"""Focused no-network tests for create-new admission and pagination safety."""
import copy
from pathlib import Path
import sys
from types import SimpleNamespace as N
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from kaggle_dataset_preflight import classify,collect,Immediate403Stop

OWNER='nnnnbhss';TARGET=OWNER+'/geofirex-wsts-fold0-tv-001-v1';CONTROL=OWNER+'/existing-private'


def evidence():
    return dict(owner=OWNER,handle=TARGET,identity=dict(active=True,username=OWNER),inventories={
        'mine':dict(handles=[CONTROL],complete=True),'private':dict(handles=[CONTROL],complete=True),
        'public':dict(handles=[],complete=True)},read_control=dict(ok=True,handle=CONTROL,version=1),
        target_status=dict(ok=False,http_status=403))


class PreflightSafetyTests(unittest.TestCase):
    def test_403_alone_never_proves_absence(self):
        e=evidence();e['inventories']={}
        r=classify(e)
        self.assertFalse(r['allow_create_new']);self.assertEqual(r['direct_lookup_state'],'ACCESS_FORBIDDEN')

    def test_404_alone_also_requires_owner_catalogue(self):
        e=evidence();e['inventories']={};e['target_status']['http_status']=404
        self.assertFalse(classify(e)['allow_create_new'])

    def test_complete_independent_catalogue_proof_keeps_403_separate(self):
        r=classify(evidence())
        self.assertEqual(r['state'],'ABSENT_VERIFIED');self.assertTrue(r['allow_create_new'])
        self.assertEqual(r['direct_lookup_state'],'ACCESS_FORBIDDEN')
        self.assertFalse(r['create_permission_verified'])

    def test_existing_target_on_partial_page_blocks(self):
        e=evidence();e['inventories']['mine']=dict(handles=[TARGET],complete=False)
        r=classify(e);self.assertEqual(r['state'],'DATASET_EXISTS');self.assertFalse(r['allow_create_new'])

    def test_existing_status_blocks_despite_catalogue_omission(self):
        e=evidence();e['target_status']=dict(ok=True,value='READY')
        self.assertEqual(classify(e)['state'],'DATASET_EXISTS')

    def test_incomplete_pagination_is_unknown(self):
        e=evidence();e['inventories']['private']['complete']=False
        self.assertEqual(classify(e)['state'],'UNKNOWN');self.assertFalse(classify(e)['allow_create_new'])

    def test_catalogue_permission_denied_blocks(self):
        e=evidence();e['inventories']['mine']=dict(complete=False,http_status=403)
        self.assertEqual(classify(e)['state'],'ACCESS_FORBIDDEN')

    def test_read_control_permission_denied_blocks(self):
        e=evidence();e['read_control']=dict(ok=False,http_status=403)
        self.assertEqual(classify(e)['state'],'ACCESS_FORBIDDEN')

    def test_wrong_owner_blocks(self):
        e=evidence();e['identity']['username']='someone-else'
        self.assertEqual(classify(e)['state'],'ACCESS_FORBIDDEN')

    def test_disagreeing_views_and_duplicates_block(self):
        e=evidence();e['inventories']['mine']['handles']=[]
        self.assertEqual(classify(e)['state'],'UNKNOWN')
        e=evidence();e['inventories']['private']['handles'].append(CONTROL)
        self.assertFalse(classify(e)['allow_create_new'])

    def test_server_failure_is_not_missing(self):
        e=evidence();e['target_status']['http_status']=500
        self.assertEqual(classify(e)['state'],'UNKNOWN')

    def test_empty_owner_catalogue_requires_healthy_read_control(self):
        e=evidence()
        for row in e['inventories'].values():row['handles']=[]
        e.pop('read_control')
        self.assertFalse(classify(e)['allow_create_new'])

    def test_collector_follows_next_page_and_blocks_existing(self):
        calls=[]
        def list_datasets(req):
            calls.append(req.page_token)
            return N(datasets=[N(ref=TARGET if req.page_token else CONTROL)],next_page_token='' if req.page_token else 'page-2')
        client=N(datasets=N(dataset_api_client=N(list_datasets=list_datasets)))
        r=collect(client,OWNER,TARGET.split('/')[1],dict(active=True,username=OWNER))
        self.assertEqual(len(calls),2);self.assertEqual(r['decision']['state'],'DATASET_EXISTS')

    def test_collector_does_not_query_on_wrong_identity(self):
        r=collect(None,OWNER,TARGET.split('/')[1],dict(active=True,username='wrong'))
        self.assertFalse(r['decision']['allow_create_new'])

    def test_owner_catalogue_admission_needs_no_forbidden_target_probe(self):
        calls=[]
        def list_datasets(req):
            calls.append('list')
            handles=[] if req.group.name.endswith('MY_PUBLIC') else [CONTROL]
            return N(datasets=[N(ref=h) for h in handles],next_page_token='')
        def get_dataset(req):
            calls.append('control');return N(current_version_number=1,is_private=True)
        client=N(datasets=N(dataset_api_client=N(list_datasets=list_datasets,get_dataset=get_dataset)))
        r=collect(client,OWNER,TARGET.split('/')[1],dict(active=True,username=OWNER),probe_target=False)
        self.assertTrue(r['decision']['allow_create_new']);self.assertEqual(r['decision']['direct_lookup_state'],'NOT_QUERIED')
        self.assertEqual(calls,['list','list','list','control'])

    def test_fatal_403_stops_before_any_other_probe(self):
        calls=[]
        def forbidden(req):
            calls.append('one');raise Immediate403Stop(N(status_code=403))
        client=N(datasets=N(dataset_api_client=N(list_datasets=forbidden)))
        with self.assertRaises(Immediate403Stop):
            collect(client,OWNER,TARGET.split('/')[1],dict(active=True,username=OWNER),probe_target=False)
        self.assertEqual(calls,['one'])


if __name__=='__main__':unittest.main()
