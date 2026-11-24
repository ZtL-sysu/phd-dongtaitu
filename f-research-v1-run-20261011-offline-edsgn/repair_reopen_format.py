"""Correct the documented reopen event to the installed validator vocabulary."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
path=ROOT/'pipeline-state.json';state=json.loads(path.read_text(encoding='utf-8'))
event=state['history'][-1]
if event['action']!='reopen_suffix':raise ValueError('Only the known rejected event may be corrected')
(ROOT/'snapshots/pipeline-rejected-history-format-v2.json').write_text(json.dumps(state,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
event['action']='reopen';event['reason']=event.pop('trigger')
path.write_text(json.dumps(state,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
