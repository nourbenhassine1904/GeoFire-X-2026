"""Essential contracts, one pytest node per capped process. No corpus scan/cloud."""
import argparse
import json
import hashlib
from pathlib import Path
import sys

from run_memory_guarded import run

ROOT = Path(__file__).resolve().parents[1]


def essential_nodes():
    specs = {
        'test_memory_guard.py': ['test_native_job_rejects_excess_commit', 'test_monitor_records_owned_memory',
            'test_monitor_stops_owned_child_on_rss_limit', 'test_monitor_refuses_without_starting_child',
            'test_sampling_handles_normal_process_exit'],
        'test_recovery_tools.py::RecoveryTools': ['test_deadline_owns_venv_descendants',
            'test_resource_error_stops_child', 'test_stored_zip_prediction_matches_streamed_archive',
            'test_freezer_preserves_inputs_and_refuses_old_configs'],
        'test_gpu_operational.py': ['test_training_scope_does_not_require_test_files',
            'test_scope_rejects_test_rows_and_aliases_before_raster_resolution',
            'test_cached_normalizer_is_train_only_pinned_and_relocatable',
            'test_lots_are_complete_exclusive_bounded_and_never_test', 'test_profiling_windows_cover_distinct_events'],
        'test_gpu_resources.py': ['test_exact_ap_scratch_cleanup_preserves_scores_and_unowned_files',
            'test_parent_deadline_stops_real_child_and_records_limit', 'test_parent_resource_failure_stops_child',
            'test_http_budget_durable_cumulative_includes_retries',
            'test_requests_guard_blocks_before_send_and_preserves_partial_journal',
            'test_staging_stream_verifies_sha_crc_and_preserves_invalid_part'],
        'test_runner.py': ['test_resume_across_changed_readonly_mount_paths',
            'test_checkpoint_resume_is_identical_and_test_requires_lock'],
        'test_benchmark_contracts.py': ['test_all_twelve_folds_match_authors_order_and_disjoint_years',
            'test_contexts_share_exact_target_dates_and_consecutive_inputs',
            'test_alignment_handles_gaps_without_short_context_advantage',
            'test_alignment_rejects_reused_paths_and_unassigned_events',
            'test_external_ap_matches_sklearn_across_chunks_ties_and_merge_levels[False]',
            'test_external_ap_matches_sklearn_across_chunks_ties_and_merge_levels[True]',
            'test_persistence_is_exact_last_observation_and_temporal_model_uses_history',
            'test_model_common_contract_finite_backward_and_odd_shapes[3-B1]',
            'test_model_common_contract_finite_backward_and_odd_shapes[3-B2]',
            'test_model_common_contract_finite_backward_and_odd_shapes[3-B3]'],
    }
    return [f'tests/{module}::{node}' for module, nodes in specs.items() for node in nodes]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    out = Path(a.out).resolve()
    if not out.is_relative_to(ROOT):
        p.error('Output must be inside the checkout')
    out.mkdir(parents=True, exist_ok=False)
    nodes = essential_nodes()
    files = [ROOT/'pyproject.toml']+list((ROOT/'src/geofirex').glob('*.py'))+list((ROOT/'tests').glob('*.py'))
    files += [ROOT/'scripts'/name for name in ('run_benchmark.py', 'profile_expanded_gpu.py',
        'run_expanded_benchmark.py', 'mount_expanded_corpus.py', 'prepare_expanded_code.py',
        'audit_gpu_recovery.py', 'run_memory_guarded.py', 'run_bounded_local.py')]
    bindings = {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    (out/'source-manifest.json').open('x').write(json.dumps(bindings, indent=2))
    (out/'plan.json').open('x').write(json.dumps(dict(nodes=nodes, per_test_seconds=120, retries=0), indent=2))
    results = []
    status = 'ESSENTIAL_TESTS_PASSED'
    for index, node in enumerate(nodes, 1):
        command = [sys.executable, '-m', 'pytest', node, '-vv', '-s', '-p', 'no:cacheprovider',
                   '--basetemp', str(out/f'tmp-{index:02d}'), '--junitxml', str(out/f'test-{index:02d}.xml'),
                   '-o', 'faulthandler_timeout=60']
        result = run(command, out/f'test-{index:02d}', seconds=120, verbose=False)
        result['node'] = node
        results.append(result)
        print(f"{index}/{len(nodes)} {result['status']} {result.get('elapsed_seconds', 0):.1f}s {node}", flush=True)
        if result['status'] != 'FINISHED':
            status = 'STOPPED_WITHOUT_RETRY'
            break
    with (out/'summary.json').open('x') as f:
        json.dump(dict(status=status, requested=len(nodes), completed=len(results),
                       results=results, remaining=nodes[len(results):]), f, indent=2)
    return 0 if status == 'ESSENTIAL_TESTS_PASSED' else 1


if __name__ == '__main__':
    raise SystemExit(main())
