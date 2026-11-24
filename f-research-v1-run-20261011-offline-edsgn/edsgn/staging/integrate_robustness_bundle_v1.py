"""Generate, but do not apply, the bounded robustness-freeze bundle patch."""
from __future__ import annotations
import argparse, ast, hashlib, json, os, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
TARGET_REL=Path('edsgn/src/edsgn/robustness.py')
SNAPSHOT_REL=Path('snapshots/robustness-before-bundle-integration-v1.py')
HELPER=Path(__file__).with_name('extend_robustness_freeze_bundle_v1.py')


class PreflightError(RuntimeError):
    def __init__(self, errors, state=None, locks=None, active=None):
        self.errors=errors;self.state=state;self.locks=locks;self.active=active
        super().__init__('Integration preflight refused: '+'; '.join(errors))


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def owned_model_processes(root, psutil_module=None):
    if psutil_module is None:
        import psutil as psutil_module
    root=Path(root).resolve();root_norm=str(root).replace('\\','/').casefold();found=[]
    modules={'edsgn.train','edsgn.calibrate','edsgn.evaluate','edsgn.profile'}
    try:processes=list(psutil_module.process_iter(['pid','cmdline','cwd','exe']))
    except (psutil_module.AccessDenied,OSError) as exc:raise RuntimeError('process-list inspection failed closed') from exc
    for proc in processes:
        try:
            pid=int(proc.info.get('pid',proc.pid))
            if pid==os.getpid():continue
            argv=[str(a).replace('\\','/').casefold() for a in (proc.info.get('cmdline') or [])]
            exe=str(proc.info.get('exe') or (argv[0] if argv else '')).replace('\\','/').casefold()
            if not Path(exe).name.startswith('python'):continue
            cwd=str(proc.info.get('cwd') or '').replace('\\','/').casefold()
            owned=root_norm in ' '.join(argv) or cwd==root_norm or any(a.startswith(root_norm+'/') for a in argv)
            if not owned:continue
            module=any(argv[i]=='-m' and i+1<len(argv) and argv[i+1] in modules for i in range(len(argv)))
            script=any(Path(a).name in {'baseline_campaign.py','bound_phase_campaign.py','frozen_robustness_phase.py','candidate_gpu_fullpath_probe_v1.py','candidate_gpu_fullpath_probe_v2.py'} or Path(a).name.endswith('runner.py') or ('campaign' in Path(a).name and Path(a).name.endswith('.py')) or ('resource' in Path(a).name and ('probe' in Path(a).name or 'diagnostic' in Path(a).name)) for a in argv)
            if module or script:found.append({'pid':pid,'argv':argv})
        except psutil_module.NoSuchProcess:continue
        except (psutil_module.AccessDenied,OSError,TypeError,ValueError) as exc:raise RuntimeError('process-list inspection failed closed') from exc
    return found


def preflight(root=ROOT, psutil_module=None):
    root=Path(root).resolve();errors=[];state=None;locks=[];active=[]
    state_path=root/'pipeline-state.json'
    try:
        state=json.loads(state_path.read_text(encoding='utf-8-sig'))
        statuses={row['id']:row['status'] for row in state['stages']}
        expected={'P1B':'passed','P2':'pending','P3':'pending'}
        for stage,value in expected.items():
            if statuses.get(stage)!=value:errors.append(f'{stage} must be {value}, found {statuses.get(stage)!r}')
    except (OSError,ValueError,KeyError,TypeError) as exc:errors.append(f'pipeline state unreadable: {exc}')
    lock_paths=[root/'logs'/'model-resource.lock',root/'logs'/'offline-phase-wall'/'campaign.lock']
    locks=[str(p) for p in lock_paths if p.exists()]
    if locks:errors.append(f'active resource/campaign lock(s): {locks}')
    try:active=owned_model_processes(root,psutil_module)
    except Exception as exc:errors.append(str(exc))
    if active:errors.append(f'owned model/runner process(es) active: {active}')
    target=root/TARGET_REL;snapshot=root/SNAPSHOT_REL
    if not target.is_file():errors.append(f'target source missing: {target}')
    if not snapshot.is_file():errors.append(f'exclusive source snapshot missing: {snapshot}')
    if target.is_file() and snapshot.is_file() and sha256(target)!=sha256(snapshot):errors.append('target robustness.py differs from the saved pre-integration snapshot')
    if errors:raise PreflightError(errors,state,locks,active)
    return {'statuses':{row['id']:row['status'] for row in state['stages']},'locks':locks,'active_processes':active,
      'target_sha256':sha256(target),'snapshot_sha256':sha256(snapshot)}


def _helper_defs(helper_text):
    tree=ast.parse(helper_text);wanted={'_SHA256_RE','_digest','robustness_freeze_bundle_extras','validate_robustness_bundle_extra','robustness_manifest_authorization_fields'}
    nodes=[]
    for node in tree.body:
        name=node.name if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) else (node.targets[0].id if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name) else None)
        if name in wanted:nodes.append(node)
    if {node.name if isinstance(node,ast.FunctionDef) else node.targets[0].id for node in nodes}!=wanted:
        raise RuntimeError('Staging helper does not contain the exact expected integration surface')
    lines=helper_text.splitlines(keepends=True)
    return '\n'.join(''.join(lines[n.lineno-1:n.end_lineno]).rstrip() for n in nodes)+'\n'


def _replace_function(source,name,replacement):
    tree=ast.parse(source);nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name]
    if len(nodes)!=1:raise RuntimeError(f'Expected exactly one top-level {name}')
    node=nodes[0];lines=source.splitlines(keepends=True)
    return ''.join(lines[:node.lineno-1])+replacement.rstrip()+'\n\n'+''.join(lines[node.end_lineno:])


def generate_source(source, helper_text):
    tree=ast.parse(source)
    if not any(isinstance(n,ast.Import) and any(a.name=='re' for a in n.names) for n in tree.body):
        anchor='import json\n'
        if source.count(anchor)!=1:raise RuntimeError('Could not add regex import at stable import anchor')
        source=source.replace(anchor,anchor+'import re\n',1)
    # Insert helpers immediately before the freeze validator; canonical_recipe,
    # recipe_digest and robustness_implementation_sha256 are already defined above.
    marker='def validate_robustness_freeze('
    if source.count(marker)!=1:raise RuntimeError('Could not find unique robustness freeze validator')
    source=source.replace(marker,_helper_defs(helper_text)+'\n'+marker,1)
    validator='''def validate_robustness_freeze(checkpoint, model, ckpt, policy, recipe: dict[str, Any]) -> dict[str, Any]:
    """Validate ordinary model freeze plus the exact authorized recipe bundle."""
    from . import engine

    freeze = engine.validate_freeze(checkpoint, model, ckpt, policy)
    validate_robustness_bundle_extra(freeze.get("extra"), recipe)
    return freeze
'''
    source=_replace_function(source,'validate_robustness_freeze',validator)
    old='''        "allowed_robustness_recipe_digest": freeze["extra"]["allowed_robustness_recipe_digest"],
        "allowed_robustness_implementation_sha256": freeze["extra"]["allowed_robustness_implementation_sha256"],'''
    new='''        "allowed_robustness_recipe_digest": recipe_digest(recipe),
        **robustness_manifest_authorization_fields(freeze["extra"], recipe),'''
    if source.count(old)!=1:raise RuntimeError('Could not locate the exact evaluator manifest authorization fields')
    source=source.replace(old,new,1)
    compile(source,'edsgn/src/edsgn/robustness.py (planned)','exec')
    return source


def build_plan(root=ROOT, psutil_module=None):
    root=Path(root).resolve();checks=preflight(root,psutil_module)
    target=root/TARGET_REL;helper_hash_before=sha256(HELPER);integration_hash=sha256(Path(__file__))
    source=target.read_text(encoding='utf-8')
    helper=HELPER.read_text(encoding='utf-8')
    planned=generate_source(source,helper)
    target_after=sha256(target);helper_after=sha256(HELPER)
    if target_after!=checks['target_sha256'] or helper_after!=helper_hash_before:
        raise RuntimeError('Target/helper changed while generating the integration plan')
    return {'scope':'source-only staged integration plan; no target write','target':str(target),
      'snapshot':str(root/SNAPSHOT_REL),'original_sha256':checks['target_sha256'],
      'snapshot_sha256':checks['snapshot_sha256'],'planned_sha256':hashlib.sha256(planned.encode('utf-8')).hexdigest(),
      'helper_sha256':helper_hash_before,'integration_script_sha256':integration_hash,
      'preflight':checks,'planned_source':planned}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--emit-source',action='store_true');args=parser.parse_args()
    plan=build_plan()
    if args.emit_source:sys.stdout.write(plan['planned_source'])
    else:print(json.dumps({k:v for k,v in plan.items() if k!='planned_source'},indent=2))

if __name__=='__main__':main()
