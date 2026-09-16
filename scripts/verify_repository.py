#!/usr/bin/env python3
"""Verify the clean public repository membership and content hashes."""
from __future__ import annotations
import gzip, hashlib, json, re
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
MANIFEST=ROOT/'MANIFEST.json'
FORBIDDEN_SUFFIXES={'.joblib','.pt','.pth','.safetensors','.bin','.zip','.log'}
FORBIDDEN_PATH=re.compile(
    r'(?i)(?:(?<![a-z0-9])[a-z]:[\\/]|/us' + r'ers/|/ho' + r'me/|\\\\)'
)
SECRET_PATTERNS=(re.compile(r'gh[pousr]_[A-Za-z0-9_]{20,}'),re.compile(r'(?i)(?:api[_-]?key|token|password)\s*[:=]\s*["\'][^"\']+["\']'))

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def included(path):
    rel=path.relative_to(ROOT)
    return (path.is_file() and '.git' not in rel.parts and '__pycache__' not in rel.parts
            and 'reproduced' not in rel.parts
            and not any(part.endswith('.egg-info') for part in rel.parts)
            and path.suffix.lower() not in {'.pyc','.pyo'} and rel.as_posix()!='MANIFEST.json')
def main():
    manifest=json.loads(MANIFEST.read_text(encoding='utf-8')); checks=0
    declared={r['path']:r for r in manifest['files']}
    actual={p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*') if included(p)}
    if actual!=set(declared): raise AssertionError(f'membership mismatch: missing={sorted(set(declared)-actual)} extra={sorted(actual-set(declared))}')
    checks+=1
    for rel,record in declared.items():
        path=ROOT/rel
        if path.stat().st_size!=record['size_bytes'] or sha(path)!=record['sha256']: raise AssertionError('hash mismatch: '+rel)
        if path.suffix.lower() in FORBIDDEN_SUFFIXES: raise AssertionError('forbidden payload: '+rel)
        checks+=3
        try: text=path.read_text(encoding='utf-8')
        except UnicodeDecodeError: continue
        if rel!='scripts/verify_repository.py' and FORBIDDEN_PATH.search(text): raise AssertionError('absolute path: '+rel)
        if any(p.search(text) for p in SECRET_PATTERNS): raise AssertionError('secret-like value: '+rel)
        checks+=2
    hgb=ROOT/'src/mars/state_symmetric.py'
    if sha(hgb)!='3724b5ac77722b70cabdf2589379d5584942f34ec81f3f5a04f88e0665f8f717': raise AssertionError('historical HGB source hash')
    checks+=1
    numeric=ROOT/'outputs/reproduction_v1/TRACE_NUMERIC.jsonl.gz'
    with gzip.open(numeric,'rt',encoding='utf-8',newline='') as stream:
        numeric_rows=[json.loads(line) for line in stream]
    expected={'dataset','retriever','group_id','eligible','forced_keep_reason','scores','actions','a0_em','a1_em','a0_f1','a1_f1'}
    if len(numeric_rows)!=18000 or any(set(row)!=expected for row in numeric_rows): raise AssertionError('numeric reproduction schema/count')
    if len({(row['dataset'],row['group_id']) for row in numeric_rows})!=6000: raise AssertionError('numeric reproduction group count')
    if any(not re.fullmatch(r'q[0-9]{4}',row['group_id']) for row in numeric_rows): raise AssertionError('non-opaque public group ID')
    numeric_text=json.dumps(numeric_rows,separators=(',',':'))
    if 'sample_id' in numeric_text or re.search(r'(?<![0-9a-f])[0-9a-f]{24,32}(?![0-9a-f])',numeric_text): raise AssertionError('benchmark-like identifier in numeric release')
    checks+=4
    development=ROOT/'outputs/reproduction_v1/DEVELOPMENT_NUMERIC.jsonl.gz'
    with gzip.open(development,'rt',encoding='utf-8',newline='') as stream:
        development_rows=[json.loads(line) for line in stream]
    development_schema={'dataset','retriever','group_id','role','eligible','numeric','a0_em','a1_em'}
    if len(development_rows)!=13500 or any(set(row)!=development_schema for row in development_rows): raise AssertionError('development reproduction schema/count')
    if len({(row['dataset'],row['group_id']) for row in development_rows})!=4500: raise AssertionError('development group count')
    if any(not re.fullmatch(r'q[0-9]{4}',row['group_id']) for row in development_rows): raise AssertionError('non-opaque development group ID')
    if any(row['role'] not in {'fit','cal'} or len(row['numeric'])!=11 for row in development_rows): raise AssertionError('development role/width')
    fit=[row for row in development_rows if row['role']=='fit' and row['eligible']]
    cal=[row for row in development_rows if row['role']=='cal' and row['eligible']]
    if (len(fit),sum(row['a0_em']==0 and row['a1_em']==1 for row in fit))!=(2572,557): raise AssertionError('fit target counts')
    if (len(cal),sum(row['a0_em']==0 and row['a1_em']==1 for row in cal))!=(630,132): raise AssertionError('calibration target counts')
    development_text=json.dumps(development_rows,separators=(',',':'))
    if 'sample_id' in development_text or re.search(r'(?<![0-9a-f])[0-9a-f]{24,32}(?![0-9a-f])',development_text): raise AssertionError('benchmark-like identifier in development release')
    heads=json.loads((ROOT/'outputs/reproduction_v1/CURRENT_HEADS.json').read_text(encoding='utf-8'))['heads']
    if set(heads)!={'ROA-FULL','ROA-NOGBV','HGB_GBV_R','HGB_ONLY_R','GBV_ONLY_R'}: raise AssertionError('current-head membership')
    if any(head['target_counts']!={'fit':[2015,557],'cal':[498,132]} for head in heads.values()): raise AssertionError('current-head target counts')
    checks+=8
    print(json.dumps({'status':'PASS_CLEAN_PUBLIC_REPOSITORY','checks':checks,'files':len(declared),'scientific_fits':0,'model_forwards':0},indent=2))
if __name__=='__main__': main()
