"""Shape/mask-only graph diagnostic capacity preparation; no model replay."""
from pathlib import Path
import json
import math
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
for rel in ('edsgn/src', 'edsgn/scripts', 'edsgn/staging'):
    sys.path.insert(0, str(ROOT / rel))
import c06_domain_manifest_preflight_v4 as metadata
import frozen_graph_state_diagnostic_v7 as graph
from root_accept_completed_preparations_v1 import digest


def main():
    output = ROOT / 'protocols/graph-artifact-capacity-root-preparation-v1.json'
    if output.exists():
        raise FileExistsError(output)
    source_sha = digest(Path(graph.__file__).resolve())
    metadata_sha = digest(Path(metadata.__file__).resolve())
    free = shutil.disk_usage(ROOT).free
    rows = []
    roots = sorted(p for p in (ROOT / 'data/processed').iterdir()
                   if p.is_dir() and p.name.startswith('synthetic'))
    for collection in roots:
        paths, datasets, audit = metadata.load_metadata_collection(collection, 'test')
        origins = [list(ds.origins('test')) for ds in datasets]
        histories = {int(ds.history) for ds in datasets}
        nodes = {int(ds.values.shape[1]) for ds in datasets}
        if len(histories) != 1 or len(nodes) != 1:
            raise RuntimeError('Heterogeneous shape/history needs explicit per-domain planning')
        history = next(iter(histories)); n = next(iter(nodes))
        count = sum(map(len, origins))
        detailed_count = sum(len(graph.uniform_detailed_origins(o)) for o in origins)
        elements = sum(len(o) * int(ds.hmax) * int(ds.values.shape[1]) * int(ds.values.shape[2])
                       for o, ds in zip(origins, datasets))
        # This large temporary arithmetic cap only obtains the conservative
        # estimator's byte count; it is not an authorized artifact allowance.
        estimate = graph.estimate_artifact_bytes(count, history, n, detailed_count,
                                               elements, max_artifact_gib=1024, d_total=64)
        proposed_cap = math.ceil(estimate['estimated_bytes'] * 1.10 / (1024 ** 3))
        bounded = graph.estimate_artifact_bytes(count, history, n, detailed_count,
                                              elements, max_artifact_gib=proposed_cap, d_total=64)
        required_free = graph.check_disk_capacity(free, bounded)
        rows.append({
            'collection_path': collection.relative_to(ROOT).as_posix(),
            'ordered_test_archives': [{'path': p.relative_to(ROOT).as_posix(), 'sha256': digest(p)}
                                      for p in paths],
            'trajectory_count': len(datasets), 'origin_count': count,
            'history': history, 'node_count': n, 'detailed_origin_count': detailed_count,
            'prediction_elements': elements, 'conservative_estimate_bytes': estimate['estimated_bytes'],
            'default_8gib_cap_sufficient': estimate['estimated_bytes'] <= 8 * 1024 ** 3,
            'proposed_per_output_cap_gib': proposed_cap,
            'required_free_bytes_at_launch': required_free,
            'decoded_npz_members': audit,
        })
    if digest(Path(graph.__file__).resolve()) != source_sha or digest(Path(metadata.__file__).resolve()) != metadata_sha:
        raise RuntimeError('Capacity-preparation source changed')
    report = {
        'status': 'root_shape_based_preparation_not_launch_authorization',
        'scope': 'conservative uncompressed graph diagnostic capacity per complete test collection',
        'source_sha256': {'edsgn/staging/frozen_graph_state_diagnostic_v7.py': source_sha,
                          'edsgn/staging/c06_domain_manifest_preflight_v4.py': metadata_sha},
        'free_disk_bytes_observed': free, 'rows': rows,
        'cap_rule': 'ceil(1.10*conservative_estimated_bytes/GiB); freeze exact cap before replay',
        'disk_rule': 'recheck >=1.25*estimated_bytes before each actual replay; no total-matrix storage claim',
        'compression_assumed_for_admission': False,
        'test_values_or_targets_or_oracle_decoded': False,
        'actual_model_or_graph_replay': False, 'pipeline_advanced': False,
        'scientific_configuration_or_hypothesis_changed': False,
        'note': 'The default toy-size8GiB cap cannot admit these full synthetic collections under the accepted conservative estimator. These prospective resource limits do not change origin coverage or reduce traces, and are not actual measured output sizes.'}
    with output.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'collections': len(rows), 'free_disk_gib': free/(1024**3),
                      'proposed_caps_gib': sorted({r['proposed_per_output_cap_gib'] for r in rows}),
                      'actual_replay': False}))


if __name__ == '__main__':
    main()
