import json
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from bind_training_inputs import (
    BindingMismatch,
    _dataset_source_files,
    prepare_binding,
    verify_binding,
)


def _config(data_root, output):
    return {"variant":"GRU","seed":0,"dataset_path":str(data_root),"output_dir":str(output),"epochs":100,"patience":10}


def test_hashes_every_npz_json_without_name_based_exclusions_and_detects_tamper(tmp_path):
    data=tmp_path/'data'; data.mkdir()
    # Deliberately not valid archives: prepare/verify must hash, not load them.
    for name,content in {
        'train_seed0.npz':b'train bytes',
        'validation_seed100.npz':b'val bytes',
        'test_seed200.npz':b'test-labelled bytes',
        'oracle_test_seed200.npz':b'oracle-labelled bytes',
        'manifest.json':b'{"partition":"validation"}',
        'test_notes.json':b'{"label":"test"}',
    }.items():
        (data/name).write_bytes(content)
    output=tmp_path/'run'
    binding_path=prepare_binding(config=_config(data,output))
    binding=json.loads(binding_path.read_text(encoding='utf-8'))
    recorded={Path(e['path']).name for e in binding['dataset_sources']}
    assert recorded=={'train_seed0.npz','validation_seed100.npz','test_seed200.npz','oracle_test_seed200.npz','manifest.json','test_notes.json'}
    assert verify_binding(binding_path)['status']=='verified'
    (data/'test_notes.json').write_text('{"label":"tampered"}',encoding='utf-8')
    with pytest.raises(BindingMismatch):
        verify_binding(binding_path)


def test_existing_at_launch_binding_is_never_overwritten(tmp_path):
    data=tmp_path/'data';data.mkdir();(data/'x.npz').write_bytes(b'npz bytes')
    output=tmp_path/'run';cfg=_config(data,output)
    binding=prepare_binding(config=cfg)
    original=binding.read_bytes()
    with pytest.raises(FileExistsError):
        prepare_binding(config=cfg)
    assert binding.read_bytes()==original


def test_binding_detects_added_same_extension_file(tmp_path):
    data=tmp_path/'data';data.mkdir();(data/'bundle.npz').write_bytes(b'first')
    output=tmp_path/'run';binding=prepare_binding(config=_config(data,output))
    (data/'new_test.json').write_text('{}',encoding='utf-8')
    with pytest.raises(BindingMismatch):
        verify_binding(binding)


def test_single_npz_binds_and_verifies_both_real_sidecar_names(tmp_path):
    data=tmp_path/'data';data.mkdir()
    bundle=data/'episode.npz';bundle.write_bytes(b'npz bytes')
    (data/'manifest.json').write_text('{"kind":"manifest"}',encoding='utf-8')
    (data/'data-manifest.json').write_text('{"kind":"data-manifest"}',encoding='utf-8')
    binding=prepare_binding(config=_config(bundle,tmp_path/'run'))
    source_names={Path(x['path']).name for x in json.loads(binding.read_text(encoding='utf-8'))['dataset_sources']}
    assert source_names=={'episode.npz','manifest.json','data-manifest.json'}
    assert verify_binding(binding)['status']=='verified'
    (data/'data-manifest.json').write_text('{"kind":"changed"}',encoding='utf-8')
    with pytest.raises(BindingMismatch):
        verify_binding(binding)


@pytest.mark.parametrize('operation', ['add','remove'])
def test_single_npz_rejects_sidecar_set_changes(tmp_path,operation):
    data=tmp_path/operation;data.mkdir()
    bundle=data/'episode.npz';bundle.write_bytes(b'npz bytes')
    manifest=data/'manifest.json';manifest.write_text('{}',encoding='utf-8')
    binding=prepare_binding(config=_config(bundle,tmp_path/f'run-{operation}'))
    data_manifest=data/'data-manifest.json'
    if operation=='add':
        data_manifest.write_text('{}',encoding='utf-8')
    else:
        manifest.unlink()
    with pytest.raises(BindingMismatch):
        verify_binding(binding)


def test_real_email_episode_single_npz_discovers_data_manifest_sidecar():
    root=Path(__file__).resolve().parents[2]
    data_path=root/'data'/'processed'/'email-episode-v1'/'email-episode.npz'
    if not data_path.is_file():
        pytest.skip('local email episode data not present')
    _,files=_dataset_source_files(data_path)
    assert {p.name for p in files}=={'email-episode.npz','data-manifest.json'}
