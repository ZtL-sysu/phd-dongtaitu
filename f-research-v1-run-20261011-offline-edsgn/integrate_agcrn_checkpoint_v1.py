"""Root C05 source-only integration after reviewed CPU parity, no model execution."""
from pathlib import Path
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'edsgn/scripts'))
from candidate_gpu_fullpath_probe_v2 import _owned_model_processes

stages = {row['id']: row['status'] for row in json.loads((ROOT / 'pipeline-state.json').read_text(encoding='utf-8'))['stages']}
if stages.get('P1B') != 'passed' or stages.get('P2') != 'pending' or stages.get('P3') != 'pending':
    raise RuntimeError('Integration requires explicit root P2 C05 back-edge')
if any((ROOT / rel).exists() for rel in ('logs/model-resource.lock', 'logs/offline-phase-wall/campaign.lock')) or _owned_model_processes(ROOT):
    raise RuntimeError('Model/resource work is active')
target = ROOT / 'edsgn/src/edsgn/rp_baselines.py'
helper = ROOT / 'edsgn/src/edsgn/agcrn_checkpoint.py'
staged = ROOT / 'edsgn/staging/agcrn_checkpoint_adapter_v1.py'
raw = target.read_bytes()
source = raw.decode('utf-8')
old = 'num_layers: int = 2, cheb_k: int = 2, **kwargs):'
new = 'num_layers: int = 2, cheb_k: int = 2, training_checkpoint_cells: bool = False, **kwargs):'
if source.count(old) != 1:
    raise RuntimeError('Constructor anchor changed')
source = source.replace(old, new, 1)
old = '        _init_agcrn_author(self.network)\n'
new = old + '        if type(training_checkpoint_cells) is not bool:\n            raise TypeError("training_checkpoint_cells must be bool")\n        self.training_checkpoint_cells = training_checkpoint_cells\n'
if source.count(old) != 1:
    raise RuntimeError('Initialization anchor changed')
source = source.replace(old, new, 1)
old = '        logits = self.network(values, targets=None, teacher_forcing_ratio=0.0)\n'
new = '''        if self.training_checkpoint_cells and self.training and torch.is_grad_enabled():
            from .agcrn_checkpoint import agcrn_cell_checkpoint
            with agcrn_cell_checkpoint(self, enabled=True):
                logits = self.network(values, targets=None, teacher_forcing_ratio=0.0)
        else:
            logits = self.network(values, targets=None, teacher_forcing_ratio=0.0)
'''
if source.count(old) != 1:
    raise RuntimeError('Forward anchor changed')
source = source.replace(old, new, 1)
helper_source = staged.read_text(encoding='utf-8').replace('It is staging-only and deliberately not wired into package training.', 'The RP adapter calls this only in opt-in gradient-enabled training.')
compile(source, str(target), 'exec')
compile(helper_source, str(helper), 'exec')
snapshot = ROOT / 'snapshots/rp-baselines-before-agcrn-checkpoint-v1.py'
with snapshot.open('xb') as stream:
    stream.write(raw)
with helper.open('x', encoding='utf-8', newline='\n') as stream:
    stream.write(helper_source)
if target.read_bytes() != raw:
    raise RuntimeError('RP source changed during source-only integration')
target.write_text(source, encoding='utf-8', newline='\n')
receipt = {'scope': 'C05 source-only exact activation checkpoint integration',
           'source_before_sha256': hashlib.sha256(raw).hexdigest(),
           'source_after_sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
           'staged_helper_sha256': hashlib.sha256(staged.read_bytes()).hexdigest(),
           'runtime_helper_sha256': hashlib.sha256(helper.read_bytes()).hexdigest(),
           'optimizer_steps': 0, 'real_data_accessed': False, 'gpu_used': False,
           'default_enabled': False, 'eval_no_grad_reference_path': True}
with (ROOT / 'snapshots/agcrn-checkpoint-integration-receipt-v1.json').open('x', encoding='utf-8') as stream:
    stream.write(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt, indent=2))
