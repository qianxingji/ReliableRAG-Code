#!/usr/bin/env python3
"""Verify the clean public repository membership and content hashes."""
from __future__ import annotations
import hashlib, json, re
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
            and not any(part.endswith('.egg-info') for part in rel.parts)
            and path.suffix.lower() not in {'.pyc','.pyo'} and path.name!='MANIFEST.json')
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
    print(json.dumps({'status':'PASS_CLEAN_PUBLIC_REPOSITORY','checks':checks,'files':len(declared),'scientific_fits':0,'model_forwards':0},indent=2))
if __name__=='__main__': main()
