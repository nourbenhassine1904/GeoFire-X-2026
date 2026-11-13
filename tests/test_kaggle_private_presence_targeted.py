"""Three tiny catalogue fixtures; no network, rasters or models."""
from pathlib import Path
import sys
from types import SimpleNamespace as N
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from kaggle_private_presence import confirm_private_presence,wait_private_presence
from kaggle_dataset_preflight import collect


class PrivatePresenceTests(unittest.TestCase):
    def client(self,private=True,inconsistent=False):
        def list_datasets(req):
            view=req.group.name
            included=not view.endswith('MY_PUBLIC') if private else not view.endswith('MY_PRIVATE')
            if inconsistent and view.endswith('MY'):included=False
            resources=[N(ref='nnnnbhss/lot002',is_private=private,current_version_number=1)] if included else []
            if inconsistent and view.endswith('MY_PRIVATE'):
                resources.append(N(ref='nnnnbhss/unrelated',is_private=True,current_version_number=1))
            return N(datasets=resources,next_page_token='')
        return N(datasets=N(dataset_api_client=N(list_datasets=list_datasets)))

    def test_private_presence_without_getdataset(self):
        self.assertTrue(confirm_private_presence(self.client(),'nnnnbhss','lot002')['private'])

    def test_public_target_is_refused(self):
        with self.assertRaises(ValueError):confirm_private_presence(self.client(False),'nnnnbhss','lot002')

    def test_inconsistent_catalogues_are_refused(self):
        with self.assertRaises(ValueError):confirm_private_presence(self.client(inconsistent=True),'nnnnbhss','lot002')

    def test_catalogue_control_avoids_getdataset_before_create(self):
        outcome=collect(self.client(),'nnnnbhss','new-lot003',dict(active=True,username='nnnnbhss'),
                        probe_target=False,catalogue_read_control=True)
        self.assertTrue(outcome['decision']['allow_create_new'])
        self.assertEqual(outcome['decision']['direct_lookup_state'],'NOT_QUERIED')

    def test_just_created_unindexed_target_is_pending(self):
        outcome=confirm_private_presence(self.client(),'nnnnbhss','not-yet-indexed')
        self.assertEqual(outcome['status'],'PRIVATE_INDEXING_PENDING')

    def test_pending_indexing_polls_reads_without_retrying_errors(self):
        with patch('kaggle_private_presence.confirm_private_presence',side_effect=[
                dict(status='PRIVATE_INDEXING_PENDING'),dict(status='PRIVATE_REMOTE_PRESENCE_CONFIRMED')]) as poll:
            result=wait_private_presence(None,'nnnnbhss','lot002',sleep=lambda _:None)
            self.assertEqual(result['status'],'PRIVATE_REMOTE_PRESENCE_CONFIRMED')
            self.assertEqual(poll.call_count,2)
        with patch('kaggle_private_presence.confirm_private_presence',side_effect=ValueError('public')) as poll:
            with self.assertRaises(ValueError):wait_private_presence(None,'nnnnbhss','lot002',sleep=lambda _:None)
            self.assertEqual(poll.call_count,1)


if __name__=='__main__':unittest.main()
