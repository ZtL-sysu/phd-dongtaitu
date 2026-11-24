"""Root acceptance of actual training-only descriptive node-group fits."""
import json
from root_accept_completed_preparations_v1 import ROOT, digest, write_record


def main():
    review_rel = 'reviews/activity-strata-training-fit-independent-v1.json'
    review = json.loads((ROOT / review_rel).read_text(encoding='utf-8-sig'))
    assert digest(ROOT / review_rel) == 'a446977879f3460f6a607fd38d7cdcbf63d53fc660ec00aabe95ad8ce457034c'
    assert review['overall'] == 'pass_with_materialization_disclosure'
    assert {r['dataset'] for r in review['datasets']} == {'synthetic', 'email', 'mooc', 'email-episode'}
    expected_nodes = {'synthetic': 64, 'email': 986, 'mooc': 78, 'email-episode': 893}
    files = [review_rel, 'reviews/activity-strata-training-fit-independent-v1.md',
             'edsgn/scripts/prepare_training_activity_strata_root_v1.py',
             'edsgn/staging/activity_stratified_metrics_v1.py',
             'protocols/activity-strata-reporting-root-rule-v1.md']
    for row in review['datasets']:
        rel = row['artifact_path']
        assert digest(ROOT / rel) == row['artifact_sha256']
        fit = json.loads((ROOT / rel).read_text(encoding='utf-8-sig'))
        assert fit['dataset_id'] == row['dataset'] and len(fit['fit']['node_ids']) == expected_nodes[row['dataset']]
        assert fit['node_order_int64_le_sha256'] == row['node_order_int64_le_sha256']
        assert fit['training_sources'] == row['training_sources']
        assert fit['scope'] == 'descriptive_training_only_node_group_fit'
        assert fit['pretest_execution_binding_accepted'] is False
        assert fit['validation_or_test_values_used_in_fit'] is False and fit['test_feedback_used'] is False
        assert fit['targets_loaded'] is False and fit['oracle_loaded'] is False and fit['model_executed'] is False
        assert digest(ROOT / 'edsgn/scripts/prepare_training_activity_strata_root_v1.py') == fit['preparation_source_sha256']
        for source, expected in fit['input_sha256'].items():
            assert digest(ROOT / source) == expected
            files.append(source)
        files.append(rel)
    write_record('activity-strata-training-fit-root-acceptance-v1', '''# Root acceptance: actual training-derived descriptive node groups

Root accepts the four actual training-only fits after independent recomputation
of node order, valid cell counts, raw training score, positive-node linear1/3 and
2/3 cutpoints, ties and full groups, with all source and training-slice hashes.
Synthetic concatenates only eight train trajectories,16384 time rows and64
nodes; its mean absolute signal groups contain22/21/21 nodes. Email's own562
training rows yield330/327/329 groups across986 nodes. MOOC's499 rows yield
26/26/26 across78 nodes. The separately fitted episode's368 rows yield299/296/
298 across893 nodes. All four have zero nodes in zero_train/unobserved_train;
these groups remain explicit and will carry null metrics when support is empty.

Chronological raw_values and input_mask NPZ members are materialized in full,
then sliced immediately to the training interval. No targets or oracle were
opened, and no validation/test observations entered the fits. No predictions,
model, checkpoint, GPU or optimizer were involved. These descriptive groups
apply identically to every method and cannot select models or alter49 primary
endpoints. Actual pretest bindings and guarded final metrics remain pending.
This record accepts the actual fits, not a test freeze, experiment result,
primary hypothesis or stage advancement.
''', sorted(set(files)), 'actual_training_only_descriptive_node_group_fits',
                 regimes=list(expected_nodes), actual_training_only_fits=4,
                 actual_test_metrics_computed=False, pretest_execution_binding_accepted=False)
    print(json.dumps({'status': 'pass_training_only', 'regimes': 4, 'pipeline_advanced': False}))


if __name__ == '__main__':
    main()
