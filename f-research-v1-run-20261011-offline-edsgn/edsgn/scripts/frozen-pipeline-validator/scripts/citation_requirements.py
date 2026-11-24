#!/usr/bin/env python3
"""Deterministic project bibliography gate; inventory claims remain human assertions."""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
import re
import sys
from pathlib import Path

CITE_RE = re.compile(r"\\(?:citep|citet|cite)(?:\s*\[[^\]]*\]){0,2}\s*\{([^{}]*)\}")
ENTRY_START = re.compile(r"@([A-Za-z]+)\s*([({])")

class GateError(Exception):
    pass

def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()

def read_input(path: Path) -> str:
    if not path.is_file(): raise GateError(f"input does not exist or is not a file: {path}")
    data=path.read_bytes()
    if not data.strip(): raise GateError(f"input is empty: {path}")
    try: return data.decode('utf-8-sig')
    except UnicodeDecodeError as e: raise GateError(f"input is not UTF-8: {path}: {e}")

def strip_tex_comments(text: str) -> str:
    out=[]
    for line in text.splitlines():
        keep=[]
        for i,ch in enumerate(line):
            if ch=='%' and (i==0 or line[i-1] != '\\'): break
            keep.append(ch)
        out.append(''.join(keep))
    return '\n'.join(out)

def parse_bib(text: str) -> list[dict]:
    entries=[]; pos=0
    while True:
        m=ENTRY_START.search(text,pos)
        if not m: break
        typ=m.group(1).lower(); opener=m.group(2); closer='}' if opener=='{' else ')'
        i=m.end(); start=i; depth=1; quoted=False; escaped=False
        while i<len(text):
            ch=text[i]
            if escaped: escaped=False
            elif ch=='\\': escaped=True
            elif ch=='"' and depth>=1: quoted=not quoted
            elif not quoted and opener=='{' and ch=='{': depth+=1
            elif not quoted and ch==closer:
                depth-=1
                if depth==0: break
            i+=1
        if depth: raise GateError(f"unclosed BibTeX entry beginning near byte {m.start()}")
        body=text[start:i].strip()
        comma=body.find(',')
        if comma<1: raise GateError(f"entry without key/field separator near byte {m.start()}")
        key=body[:comma].strip(); fields=parse_fields(body[comma+1:])
        entries.append({'type':typ,'key':key,'fields':fields})
        pos=i+1
    if not entries: raise GateError('no BibTeX entries parsed')
    return entries

def parse_fields(text: str) -> dict:
    fields={}; i=0; n=len(text)
    while i<n:
        while i<n and (text[i].isspace() or text[i]==','): i+=1
        if i>=n: break
        m=re.match(r"([A-Za-z][A-Za-z0-9_-]*)\s*=",text[i:])
        if not m: raise GateError(f"cannot parse BibTeX field near: {text[i:i+45]!r}")
        name=m.group(1).lower(); i+=m.end()
        while i<n and text[i].isspace(): i+=1
        if i>=n: raise GateError(f"missing value for BibTeX field {name}")
        if text[i]=='{':
            start=i+1; i+=1; depth=1; escaped=False
            while i<n:
                ch=text[i]
                if escaped: escaped=False
                elif ch=='\\': escaped=True
                elif ch=='{': depth+=1
                elif ch=='}':
                    depth-=1
                    if depth==0: break
                i+=1
            if depth: raise GateError(f"unclosed braced BibTeX value for {name}")
            value=text[start:i]; i+=1
        elif text[i]=='"':
            i+=1; start=i; escaped=False
            while i<n:
                ch=text[i]
                if escaped: escaped=False
                elif ch=='\\': escaped=True
                elif ch=='"': break
                i+=1
            if i>=n: raise GateError(f"unclosed quoted BibTeX value for {name}")
            value=text[start:i]; i+=1
        else:
            start=i
            while i<n and text[i]!=',': i+=1
            value=text[start:i].strip()
        if name in fields: raise GateError(f"duplicate BibTeX field {name}")
        fields[name]=value
        while i<n and text[i].isspace(): i+=1
        if i<n and text[i]!=',': raise GateError(f"expected comma after BibTeX field {name}")
    return fields

def norm(value: str) -> str:
    value=re.sub(r"\\[a-zA-Z]+\s*",'',str(value))
    return re.sub(r"[{}\s]",'',value).casefold()

def norm_doi(value: str) -> str:
    value=re.sub(r"^\s*(?:https?://(?:dx\.)?doi\.org/|doi\s*:\s*)",'',str(value),flags=re.I)
    return value.strip().rstrip(' .;,').casefold()

def norm_arxiv(value: str) -> str:
    value=str(value).strip()
    value=re.sub(r"^https?://arxiv\.org/(?:abs|pdf)/",'',value,flags=re.I)
    value=re.sub(r"^arxiv\s*:\s*",'',value,flags=re.I)
    value=re.sub(r"\.pdf$",'',value,flags=re.I)
    value=re.sub(r"v\d+$",'',value,flags=re.I)
    return value.strip().casefold()

def policy_validate(p: dict) -> tuple[int,dt.date,int,bool]:
    if p.get('schema_version')!=1: raise GateError('policy schema_version must be 1')
    minimum=p.get('min_unique',36)
    if type(minimum) is not int or minimum<36: raise GateError('min_unique must be an integer >= 36')
    quota=p.get('projectquota',minimum)
    if type(quota) is not int or quota<36: raise GateError('projectquota, when supplied, must be an integer >= 36')
    minimum=max(minimum,quota)
    try: asof=dt.date.fromisoformat(p['as_of'])
    except Exception: raise GateError('policy as_of must be YYYY-MM-DD')
    if p.get('as_of')!=asof.isoformat(): raise GateError('policy as_of must use YYYY-MM-DD')
    years=p.get('recent_window_years')
    if type(years) is not int or years not in (3,4,5): raise GateError('recent_window_years must be 3, 4, or 5')
    if not isinstance(p.get('window_rationale'),str) or not p['window_rationale'].strip(): raise GateError('window_rationale is required')
    majority=p.get('require_recent_majority')
    if type(majority) is not bool or not majority: raise GateError('require_recent_majority must be true')
    return minimum,asof,years,majority

def check(bib_path: Path | dict, tex_path: Path | None = None,
          inv_path: Path | None = None, pol_path: Path | None = None) -> dict:
    roles = ("bib", "manuscript", "inventory", "policy")
    if isinstance(bib_path, dict) and tex_path is None and inv_path is None and pol_path is None:
        missing = [role for role in roles if role not in bib_path]
        extra = sorted(set(bib_path) - set(roles))
        if missing or extra:
            raise GateError(f"input role mapping mismatch; missing={missing}, extra={extra}")
        input_paths = {role: Path(bib_path[role]) for role in roles}
    else:
        positional = (bib_path, tex_path, inv_path, pol_path)
        if any(path is None for path in positional):
            raise GateError("check requires bib, manuscript, inventory, and policy inputs")
        input_paths = dict(zip(roles, (Path(path) for path in positional)))
    bib_path = input_paths["bib"]
    tex_path = input_paths["manuscript"]
    inv_path = input_paths["inventory"]
    pol_path = input_paths["policy"]
    raw={p:read_input(p) for p in (bib_path,tex_path,inv_path,pol_path)}
    try: policy=json.loads(raw[pol_path])
    except Exception as e: raise GateError(f"invalid policy JSON: {e}")
    try: inventory=json.loads(raw[inv_path])
    except Exception as e: raise GateError(f"invalid inventory JSON: {e}")
    if not isinstance(policy,dict): raise GateError('policy must be a JSON object')
    if not isinstance(inventory,list): raise GateError('inventory must be a JSON array')
    minimum,asof,window,_=policy_validate(policy)
    entries=parse_bib(raw[bib_path]); tex=strip_tex_comments(raw[tex_path])
    bykey={}; dupkeys=[]
    for e in entries:
        if not e['key']: raise GateError('empty BibTeX key')
        if e['key'] in bykey: dupkeys.append(e['key'])
        bykey.setdefault(e['key'],e)
    cites=set()
    for m in CITE_RE.finditer(tex):
        cites.update(k.strip() for k in m.group(1).split(',') if k.strip())
    errors=[]
    if dupkeys: errors.append('duplicate BibTeX keys: '+', '.join(sorted(set(dupkeys))))
    unknown=sorted(cites-set(bykey))
    if unknown: errors.append('cited keys absent from BibTeX: '+', '.join(unknown))
    invkeys=[]; invmap={}
    for i,r in enumerate(inventory):
        if not isinstance(r,dict): errors.append(f'inventory[{i}] must be an object'); continue
        key=r.get('key')
        if not isinstance(key,str) or not key.strip(): errors.append(f'inventory[{i}].key is required'); continue
        key=key.strip(); invkeys.append(key)
        if key in invmap: errors.append(f'duplicate inventory key: {key}')
        invmap.setdefault(key,r)
    missing=sorted(set(bykey)-set(invmap)); extra=sorted(set(invmap)-set(bykey))
    if missing: errors.append('BibTeX keys missing from inventory: '+', '.join(missing))
    if extra: errors.append('inventory keys absent from BibTeX: '+', '.join(extra))
    if set(bykey)-cites: errors.append('uncited BibTeX entries: '+', '.join(sorted(set(bykey)-cites)))
    if cites-set(bykey): pass
    candidates=[]; recent=0; undated=0; future=[]; idowners={}; dupids=[]; field_mismatches=[]
    for key,r in invmap.items():
        if key not in bykey: continue
        f=bykey[key]['fields']
        item_error_count=len(errors); item_mismatch_count=len(field_mismatches)
        for field in ('title','source_url','verification_status','relevance_status','support_locations'):
            if field not in r or r[field] is None or r[field]=='' or r[field]==[]: errors.append(f'{key}: inventory {field} is required')
        if r.get('verification_status')!='verified': errors.append(f'{key}: verification_status must be verified')
        if r.get('relevance_status')!='relevant': errors.append(f'{key}: relevance_status must be relevant')
        loc=r.get('support_locations')
        if not isinstance(loc,list) or not any(isinstance(x,str) and x.strip() for x in loc): errors.append(f'{key}: support_locations must contain a nonempty location')
        if not isinstance(r.get('source_url'),str) or not re.match(r'^https?://\S+$',r.get('source_url','')): errors.append(f'{key}: source_url must be an http(s) URL')
        if not f.get('title'): errors.append(f'{key}: BibTeX title is missing')
        elif norm(r.get('title',''))!=norm(f.get('title','')): field_mismatches.append(f'{key}: inventory/BibTeX title mismatch')
        if 'year' not in r: errors.append(f'{key}: inventory year is required; use null for undated')
        year=r.get('year')
        bibyear=f.get('year','').strip()
        if year is not None:
            if type(year) is not int: errors.append(f'{key}: inventory year must be an integer or null'); year=None
            elif not re.fullmatch(r'\d{4}',bibyear) or int(bibyear)!=year: field_mismatches.append(f'{key}: inventory/BibTeX year mismatch')
        elif bibyear:
            if norm(bibyear) in ('nd','n.d.','nodate'):
                if not isinstance(r.get('year_basis'),str) or not r['year_basis'].strip(): errors.append(f'{key}: undated year requires a nonempty year_basis with the independent official source explanation')
                else: undated+=1
            elif re.fullmatch(r'\d{4}',bibyear): field_mismatches.append(f'{key}: inventory year is undated but BibTeX year is {bibyear}')
            else: errors.append(f'{key}: undated inventory year requires blank or BibTeX n.d.')
        else:
            if not isinstance(r.get('year_basis'),str) or not r['year_basis'].strip(): errors.append(f'{key}: undated year requires a nonempty year_basis with the independent official source explanation')
            else: undated+=1
        if year is not None and year>asof.year: future.append(key); errors.append(f'{key}: future publication year {year}')
        doi=r.get('doi'); bibdoi=f.get('doi','')
        if doi:
            if not bibdoi or norm_doi(doi)!=norm_doi(bibdoi): field_mismatches.append(f'{key}: inventory/BibTeX DOI mismatch')
        elif bibdoi: doi=bibdoi
        arxiv=r.get('arxiv'); bibarxiv=f.get('eprint','')
        if arxiv:
            if not bibarxiv or norm_arxiv(arxiv)!=norm_arxiv(bibarxiv): field_mismatches.append(f'{key}: inventory/BibTeX arXiv mismatch')
        elif bibarxiv: arxiv=bibarxiv
        titleid=norm(f.get('title',''))
        for kind,val,normfn in [('doi',doi,norm_doi),('arxiv',arxiv,norm_arxiv),('title',titleid,lambda x:x)]:
            ident=normfn(val) if val else ''
            if ident:
                token=(kind,ident)
                if token in idowners and idowners[token]!=key: dupids.append((kind,ident,idowners[token],key))
                else: idowners[token]=key
        if r.get('verification_status')=='verified' and r.get('relevance_status')=='relevant' and key in cites and (year is None or year<=asof.year) and len(errors)==item_error_count and len(field_mismatches)==item_mismatch_count:
            candidates.append(key)
            if year is not None and year>=asof.year-window+1: recent+=1
    if field_mismatches: errors.extend(field_mismatches)
    if dupids: errors.append('duplicate DOI/arXiv/title identities: '+'; '.join(f'{k} {v} ({a}, {b})' for k,v,a,b in dupids))
    if len(candidates)<minimum: errors.append(f'distinct cited verified relevant nonfuture sources {len(candidates)} < min_unique {minimum}')
    total=len(candidates)
    if policy['require_recent_majority'] and recent*2<=total: errors.append(f'recent cited verified relevant sources {recent} are not a strict majority of {total}')
    return {
      'schema_version':1,'verdict':'pass' if not errors else 'fail','errors':errors,
      'policy':{'min_unique':minimum,'projectquota':policy.get('projectquota'),'as_of':asof.isoformat(),'recent_window_years':window,'recent_years_inclusive':[asof.year-window+1,asof.year],'require_recent_majority':True,'window_rationale':policy['window_rationale']},
      'checks':{'bib_entry_records':len(entries),'bib_unique_keys':len(bykey),'duplicate_bib_keys':len(set(dupkeys)),'inventory_records':len(inventory),'inventory_unique_keys':len(invmap),'cited_keys':len(cites),'unknown_cited_keys':len(unknown),'uncited_bib_keys':len(set(bykey)-cites),'bib_keys_missing_inventory':len(missing),'inventory_keys_missing_bib':len(extra),'eligible_distinct_sources':total,'recent_eligible_sources':recent,'undated_inventory_records':undated,'future_year_records':len(future),'duplicate_identity_conflicts':len(dupids),'field_mismatches':len(field_mismatches)},
      'inputs':{role:{'path':str(path.resolve()),'sha256':sha256(path)} for role,path in input_paths.items()},
      'binding_inputs':{str(p):{'sha256':sha256(p),'bytes':p.stat().st_size} for p in (bib_path,tex_path,inv_path,pol_path)},
      'assertions_are_human_supplied':True,
      'limitation':'The inventory verification, relevance, support locations, and source URLs are human-supplied assertions. This deterministic gate checks their presence and internal consistency; it does not independently verify that a source exists, supports a claim, or is relevant.'
    }

def main(argv=None) -> int:
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('bib','manuscript','inventory','policy','report'): ap.add_argument('--'+name,required=True,type=Path)
    args=ap.parse_args(argv)
    if args.report.exists(): print(f'ERROR: report already exists; refusing to overwrite: {args.report}',file=sys.stderr); return 1
    try:
        result=check(args.bib,args.manuscript,args.inventory,args.policy)
        args.report.parent.mkdir(parents=True,exist_ok=True)
        with args.report.open('x',encoding='utf-8',newline='\n') as f: json.dump(result,f,ensure_ascii=False,indent=2); f.write('\n')
    except (GateError,OSError) as e:
        print(f'ERROR: {e}',file=sys.stderr); return 1
    print(f"{result['verdict'].upper()}: eligible={result['checks']['eligible_distinct_sources']}, recent={result['checks']['recent_eligible_sources']}, report={args.report}")
    if result['errors']:
        for e in result['errors']: print(' - '+e,file=sys.stderr)
    return 0 if result['verdict']=='pass' else 1
if __name__=='__main__': raise SystemExit(main())





