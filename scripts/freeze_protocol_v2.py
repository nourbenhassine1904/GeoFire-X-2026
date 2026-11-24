"""Freeze the V2 comparison BEFORE any fit; no executable training path."""
from copy import deepcopy
import math
from pathlib import Path
import json

from geofirex.experiment_readiness import profile_channels
from geofirex.protocol_v2 import save, sha256


def freeze(root=Path('reports/protocol_v2_20261118'), destination=Path('configs/protocol_v2')):
    summary = json.loads((root/'split_summary.json').read_text())
    checks = json.loads((root/'cpu_loader_checks.json').read_text())
    distributions = json.loads((root/'distributions.json').read_text())
    if checks['status'] != 'PASS' or distributions['test_pixels_opened']:
        raise ValueError('V2 CPU verification not complete')
    destination.mkdir(parents=True, exist_ok=False)
    train_tiles = summary['distributions']['train']['tiles']
    source_files = ['src/geofirex/protocol_v2.py', 'src/geofirex/v2_evaluation.py',
                    'src/geofirex/fast_compact.py', 'src/geofirex/benchmark_models.py',
                    'src/geofirex/unet.py', 'src/geofirex/feature_profiles.py']
    base = dict(schema='geofirex-v2-experiment-preregistration-1',
                run_id='B2_C8_V2_seed42', training_authorized=False, launch_enabled=False,
                status='CPU_DATA_READY_GPU_RUNNER_NOT_PACKAGED',
                event_manifest=(root/'events.csv').as_posix(), event_manifest_sha256=sha256(root/'events.csv'),
                index=(root/'index').as_posix(), index_sha256=sha256(root/'index/index.json'),
                normalization_sha256=sha256(root/'index/normalization.json'),
                normalization_policy='frozen_2018_2019_subset_of_v2_train',
                train_tiles=train_tiles, validation_tiles=summary['distributions']['val']['tiles'],
                model='B2/StackedUNet/LightUNet', base_channels=8, depth=3,
                feature_policy='conservative_fire_terrain', channels_per_day=8,
                channel_names=profile_channels('conservative_fire_terrain'), fire_channel=7,
                context_days=3, model_input_channels=24, tile_size=64,
                initialization='fresh_seed42_no_checkpoint_resume', seed=42,
                epochs=4, fixed_checkpoint_epoch=4, early_stopping=False,
                batch_size=16, drop_last=False, updates_per_epoch=math.ceil(train_tiles/16),
                total_updates=4*math.ceil(train_tiles/16),
                optimizer=dict(name='AdamW', lr=.001, weight_decay=.0001, clip_grad_norm=1.),
                loss=dict(name='SegmentationLoss', dice_weight=.5, pos_weight=10.,
                          bce_reduction='mean_valid_pixels', dice_reduction='mean_valid_images'),
                precision='float32', amp=False, augmentation=False,
                sampler=dict(name='LocalitySampler', seed=42, block_size=256,
                             epochs=[0,1,2,3], policy='each_train_tile_once_per_epoch'),
                validation='all_80_events_all_30882_tiles_once_after_fixed_epoch4',
                threshold=.78, threshold_tuning=False,
                exclusions=['purged_events', '2021', 'historical_checkpoints_as_initialization'],
                studio_promotion=False, source_sha256={p:sha256(p) for p in source_files})
    save(destination/'baseline_c8.json', base)
    candidate = deepcopy(base)
    candidate.update(run_id='B2_C36_V2_seed42', status='NO_GO_CAUSAL_EVIDENCE_AND_OVERLAY_MISSING',
                     feature_policy='terrain_vegetation_v1', channels_per_day=36,
                     channel_names=profile_channels('terrain_vegetation_v1'), fire_channel=17,
                     model_input_channels=108,
                     normalization_policy='frozen_base_slope_elevation_plus_new_train_only_extra_features',
                     base_normalization_sha256=base['normalization_sha256'], normalization_sha256=None,
                     overlay=None, availability_evidence=None,
                     prerequisites=['reviewed_exhaustive_asset_evidence_by_J',
                                    'immutable_exact_key_geometry_overlay',
                                    'new_statistics_on_unique_V2_train_input_paths_only',
                                    'CPU_parity_of_y_valid_observed_J_for_every_key',
                                    'V2_GPU_runner_packaged_and_explicit_run_authorization'])
    save(destination/'candidate_c36.json', candidate)
    evaluation = dict(schema='geofirex-v2-evaluation-preregistration-1',
        launch_enabled=False, test_year=2021, final_test_evaluations_allowed=1,
        event_manifest_sha256=base['event_manifest_sha256'],
        reference='fresh_B2_C8_V2_seed42_not_historical_169', candidate='B2_C36_V2_seed42',
        selection_scope='V2_80_validation_events_already_historically_observed_development_only',
        checkpoint='fixed_epoch4_no_checkpoint_search', primary='exact_pooled_average_precision',
        threshold=.78, threshold_tuning=False,
        ap='noninterpolated_global_ties_unmodified_float32_probabilities_external_sort',
        validity='same_base_target_and_five_prior_day_masks_padding_excluded',
        paired_comparison_requires='identical_evaluation_domain_sha256_and_tile_event_coverage',
        new_domain='valid & (observed_J == 0)',
        required_metrics=['AP_pooled', 'Dice_micro', 'IoU_micro', 'precision_micro', 'recall_micro',
                          'new_fire_recall', 'new_fire_Dice', 'new_fire_FP_count', 'new_fire_FPR',
                          'Dice_macro_event_equal_weight', 'Brier_pooled',
                          'GPU_seconds_fit_and_inference_separately', 'GPU_card_hours',
                          'peak_VRAM', 'peak_host_RSS', 'scratch_bytes', 'loader_IO_seconds'],
        acceptance_all_required=dict(delta_AP_min=.005, delta_Dice_micro_min=-.005,
            delta_new_fire_recall_min=.02, delta_new_fire_Dice_min=.01,
            new_fire_FP_ratio_max=1.15, delta_Dice_macro_event_min=-.01,
            delta_Brier_max=0., runtime_within_explicit_authorized_budget=True),
        zero_baseline_FP='candidate_FP_must_also_be_zero',
        empty_policy='no_valid:null; empty_empty_Dice_IoU:1; precision_no_pred:null; recall_AP_no_positives:null; macro_excludes_null_reports_defined_events',
        incomplete_evaluation='NO_GO_no_selection_no_approximate_AP_fallback',
        inference='explicit_observed_J_argument_never_assume_channel7_for_C36_or_C50',
        interpretation='one_seed_exploratory_no_significance_claim_no_direct_comparison_to_14.68_percent_on_different_events',
        final_test_rule='lock_selected_model_code_data_normalizer_threshold_first; one_complete_2021_evaluation_of_the_selected_model_only; no_return_to_tuning',
        final_test_not_prepared='separate_guarded_final_reader_and_one_use_lock_required_after_development',
        fallback='retain_C8_if_C36_unavailable_or_fails_any_guard; do_not_launch_extra_search',
        metrics_implementation_sha256=sha256('src/geofirex/v2_evaluation.py'))
    save(destination/'evaluation.json', evaluation)
    save(root/'decision.json', dict(data_decision='GO', first_model='B2_C8_V2_seed42',
        scientific_decision='GO_BASELINE_C8_ON_V2', launch_decision='NO_GO_NOW',
        launch_blockers=['V2_specific_GPU_runner_and_disabled_notebook_not_yet_packaged',
                         'run_specific_GPU_budget_and_verified_backup_destinations_not_authorized'],
        next_model='B2_C36_V2_seed42', multimodal_decision='NO_GO_PENDING_CAUSAL_EVIDENCE_AND_OVERLAY',
        historical_focal_authorization_carried_over=False,
        training_executed=False, model_forwards=0, optimizer_steps=0, test_pixels_opened=0,
        preregistration={p.as_posix():sha256(p) for p in sorted(destination.glob('*.json'))}))
    print('V2 preregistered:', train_tiles, 'tiles,', base['total_updates'], 'future updates, training disabled')


if __name__ == '__main__':
    freeze()
