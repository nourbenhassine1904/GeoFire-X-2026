"""Mock control-flow checks only; these tests produce no Kaggle result claims."""
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from geofirex.cloud_corpus import seal
from geofirex.corpus import json_new, sha256_file
from scripts.kaggle_chain import completed, ensure_submitted, run, recoverable_publication


def test_archived_evidence_tamper_stops_followup(tmp_path):
    evidence = tmp_path/'bootstrap.json'
    evidence.write_text('{}')
    json_new(tmp_path/'events/1.json', {'kind': 'EVIDENCE_VERIFIED', 'path': str(evidence),
                                      'sha256': sha256_file(evidence)})
    runner = SimpleNamespace(load=lambda job_id:(tmp_path, {}), refresh=Mock())
    assert completed(runner, 'job') == str(evidence)
    evidence.write_text('{"modified":true}')
    with pytest.raises(ValueError, match='evidence changed'):
        completed(runner, 'job')
    runner.refresh.assert_not_called()


def test_resume_never_blindly_resubmits_attempted_job(tmp_path):
    json_new(tmp_path/'events/1.json', {'kind': 'SUBMISSION_ATTEMPT'})
    runner = SimpleNamespace(load=lambda job_id:(tmp_path, {}), submit=Mock())
    ensure_submitted(runner, 'job')
    runner.submit.assert_not_called()


def test_scope_mismatch_prevents_cloud_actions(tmp_path):
    scope = tmp_path/'scope.json'
    json_new(scope, seal({'parent_job_identities': ['different'],
        'purpose': 'discarded Tesla T4 throughput only', 'safety_cap_seconds': 900,
        'long_training_authorized': False}))
    runner = SimpleNamespace(load=lambda job_id:(tmp_path, {'kind':'acquire', 'identity':job_id}),
                             refresh=Mock(), submit=Mock())
    with pytest.raises(ValueError, match='Scope does not authorize'):
        run(tmp_path/'chain', runner, ['a', 'b'], scope, minutes=1)
    runner.submit.assert_not_called()
    runner.refresh.assert_not_called()


def test_finished_chain_does_not_restart_jobs(tmp_path):
    json_new(tmp_path/'finished.json', {'phase': 'COMPLETE_AWAITING_LONG_GPU_BUDGET'})
    runner = SimpleNamespace(load=Mock(), submit=Mock())
    run(tmp_path, runner, [], tmp_path/'missing-scope.json', minutes=1)
    runner.load.assert_not_called()
    runner.submit.assert_not_called()


@pytest.mark.parametrize('complete,charged,allowed', [(False, 50, False), (True, 101, False), (True, 50, True)])
def test_incomplete_or_over_budget_acquisition_cannot_enter_recovery(tmp_path, complete, charged, allowed):
    folder = tmp_path/'retrievals/attempt/geofirex-result'
    json_new(folder/'result.json', {'job_id': 'parent', 'success': False})
    json_new(folder/'publication.json', seal({'upload_called': True, 'status': 'FAILED_FILES_PRESERVED',
        'handle': 'owner/corpus', 'inventory_identity': 'inventory'}))
    json_new(folder/'acquisition.json', {'plan_identity': 'plan', 'inventory_identity': 'inventory',
        'integrity': {'complete': complete}, 'transfer': {'reserved_bytes_including_retries': charged}})
    parent = {'job_id': 'parent', 'dataset_handle': 'owner/corpus',
              'plan': {'identity': 'plan', 'limits': {'transfer_bytes': 100}}}
    runner = SimpleNamespace(load=lambda job_id:(tmp_path, parent))
    assert (recoverable_publication(runner, parent) is not None) is allowed
